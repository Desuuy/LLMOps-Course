import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

import mlflow
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def _load_env():
    # Prefer llm_serving/.env (one directory up from this file).
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback to whatever is in the environment.
        load_dotenv()

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9100")
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID",
        os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY",
        os.environ.get("MINIO_SECRET_KEY", "minioadmin123"),
    )
    os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def main():
    _load_env()

    parser = argparse.ArgumentParser(description="Download LoRA from MLflow and merge with base model into a HF folder.")
    parser.add_argument("--run_id", required=True, help="MLflow Run ID containing the adapter artifacts.")
    parser.add_argument(
        "--artifact_path",
        default="model",
        help="Artifact path inside the MLflow run where the adapter was logged.",
    )
    parser.add_argument(
        "--output_dir",
        default="../models/gemma-3-1b-it-merged",
        help="Output directory for merged model (HF format).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Max sequence length used for loading/merging.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    print(f"=== 1) Downloading adapter from MLflow: run_id={args.run_id} artifact_path={args.artifact_path} ===")
    adapter_dir = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=args.artifact_path)
    print(f"Downloaded adapter to: {adapter_dir}")

    print("=== 2) Loading base + adapter and merging ===")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    merge_model, merge_tokenizer = FastModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=dtype,
    )
    merge_tokenizer = get_chat_template(merge_tokenizer, chat_template="gemma-3")

    out_dir = Path(args.output_dir).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== 3) Saving merged model to: {out_dir} ===")
    merge_model.save_pretrained_merged(str(out_dir), merge_tokenizer, save_method="merged_16bit")

    print(f"=== Done. Merged model saved at: {out_dir} ===")


if __name__ == "__main__":
    main()

