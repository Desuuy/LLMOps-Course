import os
import argparse
from dotenv import load_dotenv

import mlflow
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Load .env from parent directory where docker-compose resides
load_dotenv("../.env")

# Set up MLflow environment variables for S3 MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9100")
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", "minioadmin123"))
os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

def main():
    parser = argparse.ArgumentParser(description="Download LoRA from MLflow and merge with local base model.")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow Run ID containing the adapter.")
    parser.add_argument("--artifact_path", type=str, default="model", help="Path inside MLflow run where adapter is stored.")
    parser.add_argument("--output_dir", type=str, default="../models/gemma-3-1b-it-merged", help="Output directory for merged model.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    print(f"=== 1. Downloading LoRA adapter '{args.artifact_path}' from MLflow Run ID {args.run_id} ===")
    try:
        adapter_dir = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=args.artifact_path)
        print(f"Downloaded adapter successfully to temporary location: {adapter_dir}")
    except Exception as e:
        print(f"Error downloading adapter. Ensure your MinIO is running and Run ID is correct. Details: {e}")
        return

    print(f"=== 2. Loading base model and applying LoRA adapter ===")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # FastModel will figure out the base model path from the adapter_config.json
    merge_model, merge_tokenizer = FastModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=dtype,
    )
    merge_tokenizer = get_chat_template(merge_tokenizer, chat_template="gemma-3")

    print(f"=== 3. Saving merged 16-bit model to {args.output_dir} ===")
    # Unsloth merges the weights and saves a single standard HuggingFace model
    merge_model.save_pretrained_merged(args.output_dir, merge_tokenizer, save_method="merged_16bit")
    
    print(f"=== Success! Merged model saved to {os.path.abspath(args.output_dir)} ===")
    print("You can now serve this merged model using vLLM or SGLang without loading the adapter separately.")

if __name__ == "__main__":
    main()
