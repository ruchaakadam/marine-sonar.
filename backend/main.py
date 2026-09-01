from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO


app = FastAPI(
    title="Marine Sonar API",
    version="0.3.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\sih\marine-sonar\runs\detect\runs\detect"
    r"\groupval_onlineaug_1024\weights\best.pt"
)

CONFIDENCE = 0.05
NMS_IOU = 0.40


print("=" * 70)
print("LOADING MARINE SONAR MODEL")
print("=" * 70)
print(f"Model path : {MODEL_PATH}")
print(f"Confidence : {CONFIDENCE}")
print(f"NMS IoU    : {NMS_IOU}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("MODEL LOADED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "marine-sonar-api",
        "status": "running",
        "model_loaded": True,
        "docs": "/docs",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "marine-sonar-api",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
    }


# ============================================================
# OBJECT DETECTION
# ============================================================

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG, JPEG, or WebP image."
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    try:
        image = Image.open(BytesIO(contents))
        image.load()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    # --------------------------------------------------------
    # Run YOLO inference
    # --------------------------------------------------------

    try:
        results = model.predict(
            source=image,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            verbose=False,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Convert YOLO results to JSON
    # --------------------------------------------------------

    detections = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            xyxy = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                },
            })

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "model": MODEL_PATH.stem,
        "confidence_threshold": CONFIDENCE,
        "nms_iou_threshold": NMS_IOU,

        "image": {
            "width": image.width,
            "height": image.height,
        },

        "detection_count": len(detections),

        "detections": detections,
    }