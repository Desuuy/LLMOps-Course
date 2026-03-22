#!/usr/bin/env python3
"""
Upload the finetuned Gemma-1B model files to MinIO.

Usage:
    python upload_model_to_minio.py

Environment variables (all have defaults matching .env):
    MINIO_ENDPOINT      host:port of MinIO S3 API (default: localhost:9100)
    MINIO_ACCESS_KEY    (default: minioadmin)
    MINIO_SECRET_KEY    (default: minioadmin123)
    MINIO_BUCKET        bucket name (default: models)
    MODEL_SRC_DIR       local path to model dir (default: ../gemma-1b-finetune)
    MODEL_PREFIX        object prefix inside bucket (default: gemma-1b-finetune)
"""

import os
import sys
from pathlib import Path

from minio import Minio
from minio.error import S3Error

# ── Config ──────────────────────────────────────────────────────────────────
MINIO_ENDPOINT  = os.getenv("MINIO_ENDPOINT",  "localhost:9100")
MINIO_ACCESS    = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET    = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET          = os.getenv("MINIO_BUCKET",     "models")
MODEL_SRC_DIR   = Path(os.getenv("MODEL_SRC_DIR", str(Path(__file__).parent.parent.parent / "gemma-1b-finetune")))
MODEL_PREFIX    = os.getenv("MODEL_PREFIX",     "gemma-1b-finetune")

# Detect if running inside Docker (endpoint has no scheme)
secure = not (MINIO_ENDPOINT.startswith("localhost") or MINIO_ENDPOINT.startswith("minio:"))

# ── Client ──────────────────────────────────────────────────────────────────
client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=secure)

def ensure_bucket(bucket: str):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"[minio] Created bucket: {bucket}")
    else:
        print(f"[minio] Bucket exists:  {bucket}")

def upload_model(src_dir: Path, bucket: str, prefix: str):
    files = sorted(src_dir.rglob("*"))
    model_files = [f for f in files if f.is_file()]
    total = len(model_files)
    print(f"\n[upload] Uploading {total} files from {src_dir}  →  s3://{bucket}/{prefix}/\n")

    for i, fpath in enumerate(model_files, 1):
        rel = fpath.relative_to(src_dir)
        object_name = f"{prefix}/{rel}"
        size = fpath.stat().st_size
        print(f"  [{i}/{total}]  {rel}  ({size / 1e6:.1f} MB)")
        try:
            client.fput_object(bucket, object_name, str(fpath))
        except S3Error as exc:
            print(f"  ERROR uploading {rel}: {exc}", file=sys.stderr)
            raise

    print(f"\n[upload] ✓ Done. All {total} files uploaded to s3://{bucket}/{prefix}/")

if __name__ == "__main__":
    if not MODEL_SRC_DIR.is_dir():
        print(f"ERROR: Model source directory not found: {MODEL_SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    ensure_bucket(BUCKET)
    upload_model(MODEL_SRC_DIR, BUCKET, MODEL_PREFIX)
