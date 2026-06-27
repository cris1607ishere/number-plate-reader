import re
import sqlite3
import cv2
import numpy as np
import base64
import requests
import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import onnxruntime as ort
from ultralytics import YOLO

load_dotenv()

app = FastAPI(title="Gate Entry System")

# ── Config ─────────────────────────────────────────────────────────────────────
DETECTION_CONF_MIN    = 0.30
OCR_API_KEY           = os.getenv("OCR_API_KEY", "helloworld")
DB_PATH               = "gate_entry.db"
IMAGES_DIR            = Path("static/images")
MODEL_PATH            = Path("models/yolov8n_plate.onnx")
ODOMETER_MODEL_PATH   = Path("models/best.pt")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Load plate ONNX model once at startup ──────────────────────────────────────
_session = None

def get_session():
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"ONNX model not found at {MODEL_PATH}. "
                "Run: python download_model.py"
            )
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        print(f"[ANPR] Model loaded: {MODEL_PATH}")
    return _session

# ── Load odometer model once at startup ───────────────────────────────────────
_odometer_model = None

def get_odometer_model():
    global _odometer_model
    if _odometer_model is None:
        if not ODOMETER_MODEL_PATH.exists():
            raise RuntimeError(
                f"Odometer model not found at {ODOMETER_MODEL_PATH}. "
                "Place best.pt inside the models/ folder."
            )
        _odometer_model = YOLO(str(ODOMETER_MODEL_PATH))
        print(f"[ODOMETER] Model loaded: {ODOMETER_MODEL_PATH}")
    return _odometer_model

# ── Database ───────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vehicles (
            plate_number TEXT PRIMARY KEY,
            company_name TEXT,
            registered_at TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            driver_name  TEXT,
            company_name TEXT,
            load_status  TEXT CHECK(load_status IN ('Loaded','Empty')),
            odometer_km  TEXT,
            timestamp    TEXT DEFAULT (datetime('now','localtime'))
        );
    """)
    # Add odometer_km column if upgrading an existing DB that lacks it
    try:
        conn.execute("ALTER TABLE vehicle_logs ADD COLUMN odometer_km TEXT")
        conn.commit()
        print("[DB] Added odometer_km column to vehicle_logs")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

init_db()

# ── YOLO ONNX inference (plate) ────────────────────────────────────────────────
INPUT_SIZE = 640

def _letterbox(img: np.ndarray, size: int = INPUT_SIZE):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top  = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    return canvas, scale, pad_top, pad_left

def detect_plate(image_bytes: bytes):
    session   = get_session()
    nparr     = np.frombuffer(image_bytes, np.uint8)
    img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0
    orig_h, orig_w = img.shape[:2]
    canvas, scale, pad_top, pad_left = _letterbox(img)
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: blob})
    preds      = outputs[0][0].T
    scores     = preds[:, 4]
    best_idx   = int(np.argmax(scores))
    best_conf  = float(scores[best_idx])
    if best_conf < DETECTION_CONF_MIN:
        return None, 0.0
    cx, cy, bw, bh = preds[best_idx, :4]
    x1 = (cx - bw / 2 - pad_left) / scale
    y1 = (cy - bh / 2 - pad_top)  / scale
    x2 = (cx + bw / 2 - pad_left) / scale
    y2 = (cy + bh / 2 - pad_top)  / scale
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    crop = img[y1:y2, x1:x2]
    return crop, best_conf

# ── Odometer detection (best.pt via ultralytics) ─────────────────
def detect_odometer(image_bytes: bytes):
    """
    Returns ALL odometer-class detections as a list of crops.
    Caller picks the right one by comparing OCR values.
    """
    model = get_odometer_model()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [], img

    results      = model(img, verbose=False)[0]
    class_names  = results.names
    odometer_cls = next(
        (k for k, v in class_names.items() if v.lower() == "odometer"), None
    )

    crops = []
    if odometer_cls is not None:
        orig_h, orig_w = img.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            if cls_id == odometer_cls and conf >= DETECTION_CONF_MIN:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if x2 > x1 and y2 > y1:
                    crops.append((img[y1:y2, x1:x2], conf))
                    print(f"[ODOMETER] Box found  conf={conf:.2f}  box=[{x1},{y1},{x2},{y2}]")

    if not crops:
        # Fallback — no detections, send full image
        print("[ODOMETER] No detections — falling back to full image")
        return [(img, 0.5)], img

    return crops, img

def _preprocess_odometer_crop(crop: np.ndarray) -> np.ndarray:
    """
    Adaptive preprocessing — handles both bright-on-dark (digital)
    and dark-on-light (analog) odometer displays.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # If the image is predominantly dark (digital backlit display),
    # invert it so digits become dark-on-white — OCR handles that better
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    # Mild contrast boost
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    # Upscale small crops — OCR.Space struggles below ~100px height
    h, w = enhanced.shape
    if h < 100:
        enhanced = cv2.resize(enhanced, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ── JPEG compression (conditional — only fires if crop > 1 MB) ────────────────
def _compress_jpeg(img: np.ndarray, max_bytes: int = 1_000_000) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("JPEG encoding failed")

    if buf.nbytes < max_bytes:
        return buf.tobytes()

    print(f"[ANPR] Crop {buf.nbytes/1024:.0f}KB exceeds limit, compressing...")

    quality = 90
    while True:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        if buf.nbytes < max_bytes or quality <= 20:
            break
        quality -= 10

    while buf.nbytes >= max_bytes:
        h, w = img.shape[:2]
        if min(h, w) <= 50:
            break
        img = cv2.resize(img, (int(w * 0.85), int(h * 0.85)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(quality, 50)])

    return buf.tobytes()

# ── OCR.Space — shared by plate and odometer ───────────────────────────────────
def ocr_crop(crop: np.ndarray) -> list:
    """
    Send a cropped image to OCR.Space.
    Returns list of 3 identical readings so analyse_ocr_outputs stays unchanged.
    """
    jpeg_bytes = _compress_jpeg(crop, max_bytes=1_000_000)
    b64 = base64.b64encode(jpeg_bytes).decode()

    try:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            data={
                "apikey":            OCR_API_KEY,
                "base64Image":       f"data:image/jpeg;base64,{b64}",
                "language":          "eng",
                "isOverlayRequired": False,
                "detectOrientation": True,
                "scale":             True,
                "OCREngine":         2,
            },
            timeout=20,
        )
        resp.raise_for_status()
        result   = resp.json()
        parsed   = result.get("ParsedResults", [])
        raw_text = parsed[0]["ParsedText"].strip() if parsed else ""
        raw_text = re.sub(r'\s+', '', raw_text).upper()
        print(f"[OCR.Space] Raw: {repr(raw_text)}")
    except Exception as e:
        print(f"[OCR.Space] Error: {e}")
        raw_text = ""

    return [raw_text, raw_text, raw_text]

def ocr_odometer(crop: np.ndarray) -> str:
    """
    Sends the preprocessed odometer crop to OCR.Space.
    Tries engine 2 first, falls back to engine 1 if result is empty.
    Does NOT use isNumeric — it's unreliable on the free tier and
    can cause empty returns. Digit extraction is done locally instead.
    """
    enhanced   = _preprocess_odometer_crop(crop)
    jpeg_bytes = _compress_jpeg(enhanced, max_bytes=1_000_000)
    b64        = base64.b64encode(jpeg_bytes).decode()
    print(f"[ODOMETER OCR] Sending to OCR.Space  "
          f"size={len(jpeg_bytes)//1024}KB  dims={enhanced.shape[1]}x{enhanced.shape[0]}px")

    def _call_ocr(engine: int) -> str:
        try:
            resp = requests.post(
                "https://api.ocr.space/parse/image",
                data={
                    "apikey":            OCR_API_KEY,
                    "base64Image":       f"data:image/jpeg;base64,{b64}",
                    "language":          "eng",
                    "isOverlayRequired": False,
                    "detectOrientation": True,
                    "scale":             True,
                    "OCREngine":         engine,
                },
                timeout=20,
            )
            resp.raise_for_status()
            result   = resp.json()
            parsed   = result.get("ParsedResults", [])
            raw_text = parsed[0]["ParsedText"].strip() if parsed else ""
            print(f"[ODOMETER OCR] Engine {engine} raw: {repr(raw_text)}")
            return raw_text
        except Exception as e:
            print(f"[ODOMETER OCR] Engine {engine} error: {e}")
            return ""

    # Try engine 2 first (neural), fall back to engine 1 (Tesseract)
    raw = _call_ocr(2)
    if not raw.strip():
        print("[ODOMETER OCR] Engine 2 empty — retrying with engine 1")
        raw = _call_ocr(1)

    # Extract the longest digit sequence (the odometer reading)
    # e.g. "36880 km" → "36880", "Total: 1,23,456" → "123456"
    digits_only = re.sub(r'[^\d]', '', raw)
    numbers     = re.findall(r'\d+', raw)

    if numbers:
        reading = max(numbers, key=len)
    else:
        reading = digits_only

    print(f"[ODOMETER OCR] Final reading: {repr(reading)}")
    return reading

# ── OCR ambiguity map ──────────────────────────────────────────────────────────
CHAR_ALTERNATIVES = {
    "0": ["O"],       "O": ["0"],
    "1": ["I"],       "I": ["1"],
    "5": ["S"],       "S": ["5"],
    "8": ["B"],       "B": ["8"],
    "Z": ["2"],       "2": ["Z"],
    "G": ["6"],       "6": ["G"],
}

# ── Indian plate helpers ───────────────────────────────────────────────────────
def _normalize(raw: str) -> str:
    text = re.sub(r'[^A-Z0-9]', '', raw.upper())
    text = re.sub(r'^IND?', '', text)
    if len(text) > 10:
        match = re.search(r'[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}', text)
        if match:
            text = match.group()
        else:
            text = text[-10:]
    return text

def _is_valid(plate: str) -> bool:
    return bool(
        re.fullmatch(r'[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}', plate) or
        re.fullmatch(r'\d{2}BH\d{4}[A-Z]{2}', plate)
    )

def _try_align(short: str, base: str) -> str:
    result = []
    si = 0
    for bc in base:
        if si < len(short) and (short[si] == bc or short[si] in CHAR_ALTERNATIVES.get(bc, [])):
            result.append(short[si])
            si += 1
        else:
            result.append('?')
    return "".join(result)

def _position_correct(s: str) -> str:
    r = list(s)
    n = len(r)
    if n < 8:
        return s
    TO_DIGIT  = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
    TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B"}
    r[0] = TO_LETTER.get(r[0], r[0])
    r[1] = TO_LETTER.get(r[1], r[1])
    r[2] = TO_DIGIT.get(r[2], r[2])
    r[3] = TO_DIGIT.get(r[3], r[3])
    for i in range(n - 4, n):
        r[i] = TO_DIGIT.get(r[i], r[i])
    TO_LETTER_SERIES = {"0": "O", "1": "I", "5": "S", "8": "B"}
    for i in range(4, n - 4):
        r[i] = TO_LETTER_SERIES.get(r[i], r[i])
    return "".join(r)

# ── Core: character-level analysis ────────────────────────────────────────────
def analyse_ocr_outputs(psm_results: list, det_conf: float) -> dict:
    normalized = [_normalize(r) for r in psm_results if r]
    if not normalized:
        return _empty()

    valid = [n for n in normalized if _is_valid(n)]
    base  = max(valid, key=len) if valid else max(normalized, key=len)
    if not base:
        return _empty()
    base     = _position_correct(base)
    base_len = len(base)

    aligned = []
    for n in normalized:
        if len(n) == base_len:
            aligned.append(n)
        elif abs(len(n) - base_len) == 1:
            aligned.append(_try_align(n, base))
    if not aligned:
        aligned = [base]

    characters  = []
    n_uncertain = 0

    for i, base_char in enumerate(base):
        readings = set()
        for a in aligned:
            if i < len(a) and a[i] != '?':
                readings.add(a[i])

        if len(readings) <= 1:
            characters.append({
                "char":      base_char,
                "uncertain": False,
                "options":   [base_char],
            })
        else:
            n_uncertain += 1
            is_digit_zone  = (i in {2, 3}) or (i >= len(base) - 4)
            is_letter_zone = i in {0, 1}
            if is_digit_zone:
                filtered = {CHAR_ALTERNATIVES.get(c, [c])[0] if not c.isdigit() else c for c in readings}
                options  = sorted(filtered)
            elif is_letter_zone:
                filtered = {CHAR_ALTERNATIVES.get(c, [c])[0] if not c.isalpha() else c for c in readings}
                options  = sorted(filtered)
            else:
                options = sorted(readings)
            characters.append({
                "char":      base_char,
                "uncertain": True,
                "options":   options,
            })

    uncertainty_penalty = (n_uncertain / max(len(characters), 1)) * 0.4
    confidence = round(det_conf * (1.0 - uncertainty_penalty), 2)
    print(f"[ANPR] Base: {base}  Uncertain: {n_uncertain}  Conf: {confidence}")

    SERIES_POSITIONS = {4, 5}
    for i, ch in enumerate(characters):
        if ch["uncertain"]:
            continue
        if i in SERIES_POSITIONS and ch["char"].isdigit() and len(base) >= 9:
            ch["uncertain"] = True
            alternatives    = CHAR_ALTERNATIVES.get(ch["char"], [])
            ch["options"]   = sorted(set([ch["char"]] + alternatives))
            n_uncertain    += 1

    return {
        "base_plate": base,
        "characters": characters,
        "confidence": confidence,
        "is_valid":   _is_valid(base),
    }

def _empty() -> dict:
    return {"base_plate": None, "characters": [], "confidence": 0.0, "is_valid": False}

# ── Pydantic models ────────────────────────────────────────────────────────────
class SaveEntryRequest(BaseModel):
    plate_number: str
    driver_name:  str
    company_name: str
    load_status:  str
    odometer_km:  str = ""   # blank if odometer wasn't detected

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return Path("templates/index.html").read_text(encoding="utf-8")

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        plate_crop, plate_conf = detect_plate(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plate detection error: {e}")

    if plate_crop is None:
        return {"plate": None, "characters": [], "confidence": 0,
                "is_valid": False, "known_vehicle": None}

    psm_results  = ocr_crop(plate_crop)
    plate_result = analyse_ocr_outputs(psm_results, plate_conf)

    conn    = get_db()
    vehicle = None
    if plate_result["base_plate"]:
        row     = conn.execute(
            "SELECT * FROM vehicles WHERE plate_number = ?",
            (plate_result["base_plate"],)
        ).fetchone()
        vehicle = dict(row) if row else None
    conn.close()

    return {
        "plate":         plate_result["base_plate"],
        "characters":    plate_result["characters"],
        "confidence":    plate_result["confidence"],
        "is_valid":      plate_result["is_valid"],
        "known_vehicle": vehicle,
    }

@app.post("/ocr_odometer")
async def ocr_odometer_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        crops, _ = detect_odometer(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Odometer detection error: {e}")

    if not crops:
        return {"odometer": None, "odometer_conf": 0.0}

    best_reading  = None
    best_value    = -1
    best_conf     = 0.0

    for i, (crop, conf) in enumerate(crops):
        raw = ocr_odometer(crop)
        if not raw:
            continue

        # Strip decimal part — compare integer portion only
        # e.g. "5176.7" → 5176, "65866" → 65866
        try:
            integer_part = int(float(raw.split('.')[0]))
        except ValueError:
            continue

        print(f"[ODOMETER] Crop {i}: raw={repr(raw)}  integer={integer_part}  conf={conf:.2f}")

        if integer_part > best_value:
            best_value   = integer_part
            best_reading = raw
            best_conf    = conf

    # Save the winning crop for debugging
    if best_reading:
        winning_idx = next(
            i for i, (_, c) in enumerate(crops)
            if abs(c - best_conf) < 0.001
        )

    print(f"[ODOMETER] Winner: {best_reading}  (integer={best_value})")

    return {
        "odometer":      best_reading if best_reading else None,
        "odometer_conf": round(best_conf, 2),
    }

@app.post("/save")
async def save_entry(entry: SaveEntryRequest):
    conn = get_db()
    conn.execute("""
        INSERT INTO vehicles (plate_number, company_name)
        VALUES (?, ?)
        ON CONFLICT(plate_number) DO UPDATE SET company_name = excluded.company_name
    """, (entry.plate_number, entry.company_name))
    conn.execute("""
        INSERT INTO vehicle_logs (plate_number, driver_name, company_name, load_status, odometer_km)
        VALUES (?, ?, ?, ?, ?)
    """, (entry.plate_number, entry.driver_name, entry.company_name,
          entry.load_status, entry.odometer_km))
    conn.commit()
    log_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"success": True, "log_id": log_id, "plate": entry.plate_number}

@app.get("/logs")
async def get_logs(search: str = "", limit: int = 50):
    conn = get_db()
    if search:
        rows = conn.execute("""
            SELECT * FROM vehicle_logs
            WHERE plate_number LIKE ? OR driver_name LIKE ? OR company_name LIKE ?
            ORDER BY id DESC LIMIT ?
        """, (f"%{search}%", f"%{search}%", f"%{search}%", limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM vehicle_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "model_ready":      MODEL_PATH.exists(),
        "odometer_ready":   ODOMETER_MODEL_PATH.exists(),
        "ocr_key":          OCR_API_KEY[:6] + "...",
        "time":             datetime.now().isoformat(),
    }

@app.delete("/logs")
async def clear_logs():
    conn = get_db()
    conn.execute("DELETE FROM vehicle_logs")
    conn.commit()
    conn.close()
    return {"success": True}