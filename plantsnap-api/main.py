from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
import base64
import os
import boto3
from botocore.exceptions import ClientError

import models
import schemas
from database import engine, get_db

from dotenv import load_dotenv
load_dotenv()

# Create tables
models.Base.metadata.create_all(bind=engine)

# Local fallback folder
IMAGES_DIR = Path("feedback_images")
IMAGES_DIR.mkdir(exist_ok=True)

# S3 config from environment variables
S3_BUCKET    = os.getenv("S3_BUCKET_NAME")
AWS_REGION   = os.getenv("AWS_REGION", "us-west-2")
USE_S3       = bool(S3_BUCKET)  # auto-detect: use S3 if bucket configured

# S3 client (only created if S3 configured)
s3_client = None
if USE_S3:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

app = FastAPI(
    title="PlantSnap API",
    description="Feedback collection for PlantSnap herb classifier",
    version="1.0.0"
)

def save_image(image_id: str, image_base64: str):
    """
    Save image locally OR to S3 depending on environment.
    Returns (local_path, s3_key) — one will be None.
    """
    try:
        image_data = base64.b64decode(image_base64)
        filename = f"{image_id}.jpg"

        if USE_S3 and s3_client:
            # Production: save to S3
            s3_key = f"feedback/{filename}"
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=image_data,
                ContentType="image/jpeg"
            )
            print(f"✅ Saved to S3: {s3_key}")
            return None, s3_key
        else:
            # Development: save locally
            local_path = str(IMAGES_DIR / filename)
            with open(local_path, "wb") as f:
                f.write(image_data)
            print(f"✅ Saved locally: {local_path}")
            return local_path, None

    except Exception as e:
        print(f"❌ Image save failed: {e}")
        return None, None

# ── Health check ────────────────────────────────────
@app.get("/")
def root():
    storage = "S3" if USE_S3 else "local"
    return {
        "status": "PlantSnap API is running! 🌿",
        "storage": storage,
        "version": "1.0.0"
    }

# ── POST /feedback ───────────────────────────────────
@app.post("/feedback", response_model=schemas.FeedbackResponse)
def submit_feedback(
    feedback: schemas.FeedbackCreate,
    db: Session = Depends(get_db)
):
    local_path = None
    s3_key     = None

    # Only save image if confidence is LOW (most valuable for retraining!)
    if feedback.image_base64 and feedback.confidence < 0.7:
        local_path, s3_key = save_image(
            feedback.image_id,
            feedback.image_base64
        )

    db_feedback = models.Feedback(
        image_id       = feedback.image_id,
        predicted_herb = feedback.predicted_herb,
        correct_herb   = feedback.correct_herb,
        confidence     = feedback.confidence,
        device_id      = feedback.device_id,
        app_version    = feedback.app_version,
        image_path     = local_path,
        s3_key         = s3_key
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

# ── POST /metrics ────────────────────────────────────
@app.post("/metrics")
def submit_metric(
    metric: schemas.MetricCreate,
    db: Session = Depends(get_db)
):
    db_metric = models.Metric(
        herb_name   = metric.herb_name,
        confidence  = metric.confidence,
        was_correct = metric.was_correct,
        device_id   = metric.device_id
    )
    db.add(db_metric)
    db.commit()
    return {"status": "metric recorded ✅"}

# ── GET /version ─────────────────────────────────────
@app.get("/version", response_model=schemas.VersionResponse)
def get_version():
    return {
        "model_version":    "1.0.0",
        "min_app_version":  "1.0.0",
        "update_available": False
    }

# ── GET /feedback/stats ──────────────────────────────
@app.get("/feedback/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(models.Feedback).count()
    with_images = db.query(models.Feedback).filter(
        (models.Feedback.image_path != None) |
        (models.Feedback.s3_key != None)
    ).count()

    return {
        "total_feedback":   total,
        "images_collected": with_images,
        "storage_type":     "S3" if USE_S3 else "local"
    }