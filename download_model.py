"""
download_model.py
-----------------
Run this ONCE before starting the server:

    python download_model.py

Downloads a YOLOv8n model trained for Indian/generic number-plate detection
and exports it to models/yolov8n_plate.onnx.

Two strategies are attempted in order:
  1. Direct ONNX download from a public HuggingFace repo (fastest).
  2. Download a .pt checkpoint and export to ONNX locally using Ultralytics
     (requires: pip install ultralytics).

If you have your own trained best.pt, place it at models/best.pt and run:
    python download_model.py --from-pt models/best.pt
"""

import argparse
import sys
from pathlib import Path

MODEL_DIR  = Path("models")
ONNX_PATH  = MODEL_DIR / "yolov8n_plate.onnx"
PT_PATH    = MODEL_DIR / "best.pt"

MODEL_DIR.mkdir(exist_ok=True)

# ── Public model sources ──────────────────────────────────────────────────────
# These are community-trained YOLOv8 plate detection models on HuggingFace.
# Priority order: ONNX first (no Ultralytics needed), then .pt fallback.
ONNX_URLS = [
    # ml-debi/yolov8-license-plate-detection — MIT license, public, no login required
    "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx",
]

PT_URLS = []  # no .pt fallback needed — ONNX is directly available


def download(url: str, dest: Path, desc: str = ""):
    import urllib.request
    print(f"Downloading {desc or url} → {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        dest.unlink(missing_ok=True)
        return False


def export_pt_to_onnx(pt_path: Path, out_path: Path):
    print(f"\nExporting {pt_path} → {out_path} using Ultralytics …")
    try:
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        model.export(format="onnx", imgsz=640, simplify=True, opset=12)
        # Ultralytics writes the ONNX next to the .pt file
        auto_path = pt_path.with_suffix(".onnx")
        if auto_path.exists():
            auto_path.rename(out_path)
            print(f"  ✓ Exported to {out_path}")
            return True
        print("  ✗ Export succeeded but output file not found.")
        return False
    except ImportError:
        print("  ✗ Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-pt", metavar="PATH", help="Export a local .pt to ONNX")
    args = parser.parse_args()

    if ONNX_PATH.exists():
        print(f"✓ Model already present at {ONNX_PATH}. Nothing to do.")
        print("  Delete it and re-run if you want a fresh download.")
        return

    # ── User supplied a .pt path ──────────────────────────────────────────────
    if args.from_pt:
        pt = Path(args.from_pt)
        if not pt.exists():
            sys.exit(f"File not found: {pt}")
        if export_pt_to_onnx(pt, ONNX_PATH):
            print(f"\n✅ Ready. Run: uvicorn main:app --reload")
        else:
            sys.exit("Export failed.")
        return

    # ── Strategy 1: download ONNX directly ───────────────────────────────────
    print("=== Strategy 1: Direct ONNX download ===")
    for url in ONNX_URLS:
        if download(url, ONNX_PATH, "YOLOv8n plate ONNX"):
            print(f"\n✅ Model ready at {ONNX_PATH}")
            print("Run: uvicorn main:app --reload")
            return

    # ── Strategy 2: download .pt, export to ONNX ─────────────────────────────
    print("\n=== Strategy 2: Download .pt → export to ONNX ===")
    for url in PT_URLS:
        if download(url, PT_PATH, "YOLOv8n plate .pt"):
            if export_pt_to_onnx(PT_PATH, ONNX_PATH):
                print(f"\n✅ Model ready at {ONNX_PATH}")
                print("Run: uvicorn main:app --reload")
                PT_PATH.unlink(missing_ok=True)   # keep only the ONNX
                return

    print("""
❌ All strategies failed.

Manual options:
  A) Train your own model with Ultralytics, then run:
         python download_model.py --from-pt your_best.pt

  B) Place any YOLOv8-ONNX plate detection model at:
         models/yolov8n_plate.onnx
     (single output tensor, shape [1, 5, 8400] — class 0 = plate)
""")
    sys.exit(1)


if __name__ == "__main__":
    main()