import os
from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "google/gemma-3-1b-it"
LOCAL_DIR = Path("models/gemma-3-1b-it")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

kwargs = {}
token = os.environ.get("HF_TOKEN")
if token:
    kwargs["token"] = token

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,
    **kwargs
)
print("Downloaded to:", LOCAL_DIR.resolve())
