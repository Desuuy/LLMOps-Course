# Training (LLMOps) — `llm_serving/train/`

This folder contains a **host-run training workflow** that plugs into the **same** MinIO + MLflow services defined in `llm_serving/docker-compose.yml`.

## The lifecycle (HF → train → MLflow/MinIO → merge → serve)

1. **Load base model** (from Hugging Face or a local cache).
2. **Fine-tune** with Unsloth/TRL (LoRA adapter).
3. **Track experiments** in MLflow (`http://localhost:5000`), with artifacts stored in MinIO (`http://localhost:9101`).
4. **Merge** the best adapter into a single HF model folder (recommended for SGLang/vLLM).
5. **Deploy** by uploading merged weights to MinIO and syncing into the serving volume (`model_weights`) via `scripts/update_model.sh`.

## Prereqs

- NVIDIA GPU + drivers
- Python 3.10+ (Conda recommended)
- `llm_serving` stack running (MinIO + MLflow at minimum)

Start infra:

```bash
cd llm_serving
docker compose up -d minio mlflow
```

## Environment

Training scripts read:

- `llm_serving/.env` (preferred)
- your shell environment (fallback)

Key variables:

- `HF_TOKEN` (optional, for gated/fast HF downloads)
- `MLFLOW_TRACKING_URI` (default `http://localhost:5000`)
- `MLFLOW_S3_ENDPOINT_URL` (default `http://localhost:9100`)
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`

## Install deps (host)

```bash
cd llm_serving/train
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training

Edit `config.json` if needed (especially `model_name` path), then:

```bash
cd llm_serving/train
python train.py --config config.json
```

Outputs:

- LoRA adapter saved to `save_dir` (default: `gemma_1b_tool_call_lora/`)
- MLflow run created (copy the **Run ID** from the MLflow UI)

## Merge adapter to a serving-ready model

```bash
cd llm_serving/train
python merge_model.py --run_id YOUR_RUN_ID --output_dir ../models/gemma-3-1b-it-merged
```

## Deploy to the serving stack (MinIO → volume → restart)

From `llm_serving/`:

```bash
bash scripts/update_model.sh \
  --model-dir ./models/gemma-3-1b-it-merged \
  --model-name gemma-3-1b-it-merged \
  --engine sglang
```

Then call via LiteLLM:

```bash
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-llmserving-master-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-1b-finetune","messages":[{"role":"user","content":"Hello!"}]}'
```

If you changed the served model name, update `litellm/config.yaml` to match and restart `litellm`.

