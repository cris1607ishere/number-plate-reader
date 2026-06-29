# 🚗 AI Gate Entry System

An AI-powered Gate Entry System that automates vehicle entry by recognizing **vehicle number plates** and **odometer readings** from uploaded images.

The system combines **YOLOv8**, **ONNX Runtime**, **OCR.Space API**, and **FastAPI** to provide fast and accurate vehicle information extraction through a simple web interface.

---

# Features

* 🚘 Automatic Number Plate Detection
* 🔤 Number Plate Recognition using OCR.Space
* 📊 Odometer Detection using a custom-trained YOLOv8 model
* 🔢 Automatic Odometer Reading Extraction
* ⚡ Fast ONNX inference for number plate detection
* 🌐 FastAPI backend
* 📷 Image Upload Interface
* 📱 Responsive Web UI

---

# Tech Stack

## Backend

* Python 3.11
* FastAPI
* Uvicorn

## AI / Computer Vision

* YOLOv8
* ONNX Runtime
* OpenCV
* OCR.Space API

## Frontend

* HTML
* CSS
* JavaScript

---

# Project Structure

```text
project-root/
│
├── models/
│   ├── best.pt                 # Odometer detection model
│   └── yolov8n_plate.onnx      # Number plate detection model
│
├── static/
│   ├── images/
│   └── styles/
│
├── templates/
│   └── index.html
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/cris1607ishere/number-plate-reader.git
```

Navigate to the project directory

```bash
cd path/to/number-plate-reader
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configure OCR.Space

Create a file named **`.env`** in the project root.

Example:

```env
OCR_SPACE_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your OCR.Space API key.

---

# Running the Application

Start the FastAPI server

```bash
uvicorn main:app --reload
```

Open your browser and visit

```
http://127.0.0.1:8000
```

---

# How It Works

## Number Plate Pipeline

1. Upload a vehicle image.
2. YOLO ONNX detects the number plate.
3. The detected plate is cropped.
4. OCR.Space extracts the plate text.
5. The detected plate is validated and displayed.

---

## Odometer Pipeline

1. Upload a dashboard image.
2. Custom-trained YOLOv8 detects the odometer display.
3. The detected region is cropped.
4. OCR.Space extracts the mileage.
5. Post-processing returns the final odometer reading.

---

# AI Models

## Number Plate Detection

* YOLOv8 ONNX
* ONNX Runtime inference

## Odometer Detection

* Custom-trained YOLOv8 model
* Fine-tuned for dashboard odometer localization

---

# Requirements

* Python 3.11+
* OCR.Space API Key

---

# Dependencies

```
FastAPI
Uvicorn
Ultralytics
ONNX Runtime
OpenCV
NumPy
Pillow
Requests
Pydantic
python-dotenv
```

---

# Future Improvements

* Live camera support
* Video stream inference
* Database integration
* Entry history
* Multi-vehicle support
* Improved dashboard support
* Automatic trip meter filtering

---

# Author

**Cris Thomas**

GitHub: https://github.com/cris1607ishere
