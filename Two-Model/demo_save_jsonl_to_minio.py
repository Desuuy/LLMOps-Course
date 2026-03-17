import io
import json
import os
from typing import Iterable

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from minio import Minio
from minio.error import S3Error


MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "xlam-data")
MINIO_OBJECT_NAME = os.environ.get("MINIO_OBJECT_NAME", "xlam-function-calling-60k-sample.jsonl")

HF_DATASET_ID = os.environ.get("HF_DATASET_ID", "Salesforce/xlam-function-calling-60k")
HF_SPLIT = os.environ.get("HF_SPLIT", "train[:100]")


def iter_to_jsonl_bytes(examples: Iterable[dict]) -> bytes:
    buffer = io.StringIO()
    for row in examples:
        buffer.write(json.dumps(row, ensure_ascii=False) + "\n")
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    print(f"Loading dataset '{HF_DATASET_ID}' (split: {HF_SPLIT}) ...")
    try:
        ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    except DatasetNotFoundError as exc:
        print(f"Failed to load '{HF_DATASET_ID}': {exc}")
        print("This is often caused by a gated/private dataset that requires Hugging Face authentication + access approval.")
        print("Falling back to a public dataset so the MinIO upload demo can still run.")
        HF_DATASET_ID_FALLBACK = "ag_news"
        HF_SPLIT_FALLBACK = "train[:100]"
        print(f"Loading fallback dataset '{HF_DATASET_ID_FALLBACK}' (split: {HF_SPLIT_FALLBACK}) ...")
        ds = load_dataset(HF_DATASET_ID_FALLBACK, split=HF_SPLIT_FALLBACK)

    print("Converting to JSONL in memory...")
    data_bytes = iter_to_jsonl_bytes(ds)

    print(f"Connecting to MinIO at {MINIO_ENDPOINT} ...")
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Ensure bucket exists
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            print(f"Bucket '{MINIO_BUCKET}' does not exist, creating...")
            client.make_bucket(MINIO_BUCKET)
        else:
            print(f"Bucket '{MINIO_BUCKET}' already exists.")
    except S3Error as exc:
        print(f"Error ensuring bucket exists: {exc}")
        return

    print(f"Uploading JSONL data as object '{MINIO_OBJECT_NAME}' into bucket '{MINIO_BUCKET}' ...")
    data_stream = io.BytesIO(data_bytes)
    data_stream.seek(0)

    try:
        client.put_object(
            MINIO_BUCKET,
            MINIO_OBJECT_NAME,
            data_stream,
            length=len(data_bytes),
            # Use text/plain so MinIO console and browsers treat it as plain text
            content_type="text/plain; charset=utf-8",
        )
    except S3Error as exc:
        print(f"Error uploading object: {exc}")
        return

    print("Upload complete.")
    print(f"You can access it via MinIO console at http://localhost:9001/")
    print(f"Bucket: {MINIO_BUCKET}, Object: {MINIO_OBJECT_NAME}")


if __name__ == "__main__":
    main()

