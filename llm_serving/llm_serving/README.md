# 🚀 LLMOps: Train → Track → Register → Serve (Gemma-3 1B)

This folder is a single **end-to-end LLMOps stack**:

- **Train (host-run)** with Unsloth/TRL LoRA in `llm_serving/train/`
- **Track** runs in **MLflow** (`:5000`)
- **Store artifacts** in **MinIO** (S3-compatible, `:9100`/`:9101`)
- **Deploy** to inference by syncing a “production” model into a shared Docker volume
- **Serve** via **SGLang** and **vLLM** (both enabled by default), fronted by **LiteLLM**
- **Observe** with **Prometheus + Grafana**

---

## Table of Contents

1. [Architecture](#architecture)
2. [Port Reference](#port-reference)
3. [LLMOps Flow (HF → Train → MLflow/MinIO → Serve)](#llmops-flow-hf--train--mlflowminio--serve)
4. [Step-by-Step: New VM Setup](#step-by-step-new-vm-setup)
5. [Quick Start (Existing Machine)](#quick-start-existing-machine)
6. [Training: Run + Merge + Deploy](#training-run--merge--deploy)
7. [Model Flow: MinIO → Volume → Inference](#model-flow-minio--volume--inference)
8. [Hot-Swap: Replace Model Without Downtime](#hot-swap-replace-model-without-downtime)
9. [API Usage](#api-usage)
10. [vLLM (Enabled by Default)](#vllm-enabled-by-default)
11. [Monitoring: Grafana & Prometheus](#monitoring-grafana--prometheus)
12. [WandB Integration (Post-Training)](#wandb-integration-post-training)
13. [File Structure](#file-structure)
14. [Troubleshooting](#troubleshooting)
15. [Stop / Clean Up](#stop--clean-up)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Phase                               │
│  HF model → Unsloth/TRL LoRA → MLflow metrics → MinIO artifacts       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ merge + deploy
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│           Model registry (S3-compatible store: MinIO)                 │
│     - MLflow artifacts: s3://models/mlflow/...                        │
│     - Serving weights:  s3://models/<model-name>/...                  │
│              http://localhost:9100  |  console :9101                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ model-sync (mc mirror)
                                ▼
                    ┌───────────────────────┐
                    │  Docker volume        │
                    │  model_weights        │
                    └────────┬──────────────┘
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌──────────────────┐
    │ SGLang :30002   │           │ vLLM :30003       │
    │ OpenAI /v1      │           │ OpenAI /v1        │
    │ (default)       │           │ (opt-in profile)  │
    └────────┬────────┘           └────────┬──────────┘
             └──────────────┬──────────────┘
                            ▼
               ┌────────────────────────┐
               │  LiteLLM Proxy :4001   │
               │  OpenAI-compatible     │
               │  Redis cache + PG logs │
               └────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌──────────────────┐       ┌───────────────────────┐
   │ Prometheus :9091 │◄──────│ DCGM + Redis + Node   │
   └────────┬─────────┘       │ exporters             │
            ▼                 └───────────────────────┘
   ┌──────────────────┐
   │  Grafana :3000   │
   │  4 dashboards    │
   └──────────────────┘
```

---

## Port Reference

| Service | Host Port | URL | Notes |
|---------|-----------|-----|-------|
| MinIO S3 API | `9100` | `http://localhost:9100` | S3 endpoint for mc/boto3 |
| MinIO Console | `9101` | `http://localhost:9101` | Web UI (minioadmin) |
| **MLflow** | `5000` | `http://localhost:5000` | Tracking UI + artifact browsing |
| **SGLang** | `30002` | `http://localhost:30002/v1` | OpenAI-compatible |
| **vLLM** | `30003` | `http://localhost:30003/v1` | Enabled by default |
| **LiteLLM Proxy** | `4001` | `http://localhost:4001/v1` | Main gateway |
| LiteLLM UI | `4001` | `http://localhost:4001/ui` | Admin dashboard |
| Prometheus | `9091` | `http://localhost:9091` | Metrics scraper |
| **Grafana** | `3000` | `http://localhost:3000` | Dashboards |
| Redis | `6380` | — | LiteLLM cache |
| PostgreSQL | `5433` | — | LiteLLM logs |
| Node Exporter | `9102` | — | Host metrics |
| DCGM Exporter | `9400` | — | GPU metrics |
| Redis Exporter | `9121` | — | Redis metrics |

> All ports are intentionally different from the existing `cole_labs/docker-compose.yml` (LiteLLM:4000, Prometheus:9090, Postgres:5432).

---

## LLMOps Flow (HF → Train → MLflow/MinIO → Serve)

This stack is designed so that **training produces tracked artifacts**, and **serving only ever reads from MinIO** (never from a local “mystery path”).

### Data + artifact contracts

- **Training output (LoRA adapter)**:
  - Logged to MLflow as artifacts at `artifact_path="model"` inside an MLflow run.
  - Stored in MinIO automatically via MLflow’s `--default-artifact-root s3://models/mlflow`.
- **Serving input (merged model folder)**:
  - A standard Hugging Face model directory uploaded to MinIO at:
    - `s3://models/<MODEL_PATH_IN_MINIO>/`
  - Synced into Docker volume `model_weights` at `/model/`

### Why merge?

Inference engines like **SGLang** and **vLLM** work best when pointed at a single HF model directory. That’s why the recommended path is:

1) fine-tune LoRA → 2) log adapter in MLflow → 3) merge adapter → 4) upload merged → 5) serve merged.

---

## Step-by-Step: New VM Setup

> Complete walkthrough for a **fresh Ubuntu/Debian VM** with an NVIDIA GPU.

### Prerequisites

- Ubuntu 20.04+ or Debian 11+ (or Fedora/Arch — auto-detected)
- NVIDIA GPU with driver installed (see step 1 below)
- Root / sudo access
- Git (to clone this repo)
- The finetuned model directory at `../gemma-1b-finetune/` relative to `llm_serving/`

---

### Step 1 — Install NVIDIA GPU Driver

> Skip if `nvidia-smi` already works.

```bash
# Ubuntu — auto-detect and install best driver
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot, verify:
nvidia-smi
# Expected: table showing GPU name, driver version, memory
```

---

### Step 2 — Clone / Copy Project

```bash
# If cloning:
git clone <your-repo-url> /opt/cole_labs
cd /opt/cole_labs/llm_serving

# If copying from existing machine:
scp -r user@oldhost:/mnt/.../cole_labs/llm_serving /opt/cole_labs/
scp -r user@oldhost:/mnt/.../gemma-1b-finetune    /opt/cole_labs/
```

Make sure directory layout is:
```
/opt/cole_labs/
├── gemma-1b-finetune/       ← model weights
└── llm_serving/             ← this folder
    ├── docker-compose.yml
    ├── .env
    └── scripts/
```

---

### Step 3 — Configure Environment

```bash
cd /opt/cole_labs/llm_serving

# Review and edit credentials if needed
nano .env
```

Key variables to change for a new machine:

```bash
# Minimum you should change:
LITELLM_MASTER_KEY=sk-your-secret-key-here   # API key for LiteLLM
GF_ADMIN_PASSWORD=your-grafana-password
MINIO_ROOT_PASSWORD=your-minio-password
REDIS_PASSWORD=your-redis-password
```

---

### Step 4 — Run the Bootstrap Script

This single command handles everything else automatically:

```bash
sudo bash scripts/setup.sh
```

What it does internally:

| # | Action |
|---|--------|
| 1 | Installs Docker Engine (if missing) |
| 2 | Installs Docker Compose plugin (if missing) |
| 3 | Checks NVIDIA driver |
| 4 | Installs `nvidia-container-toolkit` (if missing) |
| 5 | Configures Docker GPU runtime (`nvidia-ctk configure`) |
| 6 | Restarts Docker daemon |
| 7 | Downloads Grafana community dashboards |
| 8 | Starts MinIO & uploads model weights |
| 9 | Syncs model from MinIO → Docker volume |
| 10 | Starts all services (SGLang + vLLM) |
| 11 | Waits for SGLang to finish loading |

**Expected total time**: ~10–20 min (depending on model sync speed)

---

### Step 5 — Verify Everything is Up

```bash
# Check all services
docker compose ps

# Test inference
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer $(grep LITELLM_MASTER_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-1b-finetune","messages":[{"role":"user","content":"Hello!"}]}'

# Check Grafana (in browser)
echo "Open: http://$(hostname -I | awk '{print $1}'):3000"
```

---

### GPU-Only Fix (if setup.sh already ran but GPU fails)

```bash
sudo bash scripts/install_gpu_docker.sh
# Then retry:
sudo docker compose up -d sglang
```

---

## Quick Start (Existing Machine)

If Docker + NVIDIA driver + toolkit are already installed:

```bash
cd llm_serving

# Download community Grafana dashboards
bash scripts/download_dashboards.sh

# Start everything
docker compose up -d

# Watch logs
docker compose logs -f sglang
```

---

## Training: Run + Merge + Deploy

Training is run on the **host** (not inside Docker), while MinIO + MLflow run in Docker.

### 1) Start tracking infrastructure

```bash
cd llm_serving
docker compose up -d minio mlflow
```

### 2) Train (LoRA) and log to MLflow

See `llm_serving/train/README.md`.

```bash
cd llm_serving/train
python train.py --config config.json
```

Grab the **Run ID** from the MLflow UI (`http://localhost:5000`).

### 3) Merge adapter into a single HF model folder

```bash
cd llm_serving/train
python merge_model.py --run_id YOUR_RUN_ID --output_dir ../models/gemma-3-1b-it-merged
```

### 4) Deploy to serving (upload → sync → restart)

```bash
cd llm_serving
bash scripts/update_model.sh \
  --model-dir ./models/gemma-3-1b-it-merged \
  --model-name gemma-3-1b-it-merged \
  --engine sglang
```

If you change the served model name, update `litellm/config.yaml` accordingly and restart:

```bash
docker compose restart litellm
```

---

## Model Flow: MinIO → Volume → Inference

Understanding how the model gets to SGLang/vLLM:

```
[1] model-uploader
    Reads:  ../gemma-1b-finetune/ (host filesystem)
    Writes: s3://models/gemma-1b-finetune/ (MinIO)

[2] model-sync
    Reads:  s3://models/gemma-1b-finetune/ (MinIO, via mc mirror)
    Writes: Docker volume "model_weights" at /model

[3] SGLang / vLLM
    Reads:  /model (from Docker volume, read-only mount)
    No local filesystem dependency — portable across any machine.
```

To re-run just the sync (if the volume is lost):
```bash
docker compose run --rm model-sync
docker compose restart sglang
```

---

## Hot-Swap: Replace Model Without Downtime

> When you finish training a new, better finetuned model and want to replace what the server is currently serving.

### Understanding the cycle

```
New checkpoint ready locally
        ↓
  upload to MinIO         ← model-uploader / mc cp
        ↓
  sync to Docker volume   ← model-sync (mc mirror --overwrite)
        ↓
  restart inference server  ← docker compose restart sglang/vllm
        ↓
  new model serving
```

### Method 1 — Script (recommended)

```bash
# Replace current model (default: ../gemma-1b-finetune, engine: sglang)
bash scripts/update_model.sh

# Replace with a model from a different directory
bash scripts/update_model.sh \
  --model-dir /path/to/gemma-1b-v2 \
  --model-name gemma-1b-v2 \
  --engine sglang

# Model already in MinIO — just resync + restart
bash scripts/update_model.sh --no-upload

# Update both SGLang and vLLM
bash scripts/update_model.sh --engine all
```

### Method 2 — Manual step-by-step

```bash
# 1. Upload new model to MinIO
docker compose run --rm \
  -e MODEL_SRC_DIR=/model-src \
  -e MODEL_PREFIX=gemma-1b-v2 \
  -v /your/new/model:/model-src:ro \
  model-uploader

# 2. Sync from MinIO → Docker volume (overwrites old weights)
docker compose run --rm \
  -e MODEL_PREFIX=gemma-1b-v2 \
  model-sync

# 3a. Restart SGLang
docker compose restart sglang
docker compose logs sglang -f      # watch until "Server is ready"

# 3b. Restart vLLM (if running)
docker compose restart vllm
docker compose logs vllm -f
```

### Changing the served model name

If your new model has a different name (e.g. `gemma-1b-v2` instead of `gemma-1b-finetune`):

1. Update `scripts/sglang_entrypoint.sh`:
   ```bash
   SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-1b-v2}"
   ```

2. Update `litellm/config.yaml` — change `model_name` entry or add a new one:
   ```yaml
   - model_name: gemma-1b-v2
     litellm_params:
       model: openai/gemma-1b-v2
       api_base: http://sglang:30002/v1
       api_key: "none"
   ```

3. Restart both:
   ```bash
   docker compose restart sglang litellm
   ```

### SGLang auto-reload note

> SGLang does **not** support live hot-reload of model weights (as of v0.4). The restart approach above brings it back in 2–5 minutes. During restart, LiteLLM will return 503 for SGLang-backed models — if you have vLLM as fallback, switch clients to `gemma-vllm` temporarily.

### vLLM auto-reload note

> vLLM also does **not** support live weight swapping. Same graceful restart approach applies. vLLM typically loads faster (~1–3 min) than SGLang for smaller models.

---

## API Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4001/v1",    # LiteLLM proxy
    api_key="sk-llmserving-master-key",     # from .env LITELLM_MASTER_KEY
)

# Chat completion
response = client.chat.completions.create(
    model="gemma-1b-finetune",
    messages=[{"role": "user", "content": "What tools can you call?"}],
    max_tokens=512,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gemma-1b-finetune",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Direct SGLang (bypass LiteLLM)

```python
client = OpenAI(base_url="http://localhost:30002/v1", api_key="none")
response = client.chat.completions.create(
    model="gemma-1b-finetune",
    messages=[{"role":"user","content":"Hello!"}]
)
```

### cURL

```bash
# Via LiteLLM proxy
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-llmserving-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-1b-finetune",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# List models
curl http://localhost:4001/v1/models \
  -H "Authorization: Bearer sk-llmserving-master-key"
```

### Function / Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gemma-1b-finetune",
    messages=[{"role": "user", "content": "What's the weather in Hanoi?"}],
    tools=tools,
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls)
```

---

## vLLM (Enabled by Default)

vLLM is an alternative inference engine with PagedAttention. It runs on port `30003` and is enabled by default:

```bash
# Start vLLM (uses VLLM_GPU_MEM=0.34 from .env — shares GPU with SGLang)
docker compose up -d vllm

# To use more VRAM when running vLLM ONLY (not alongside SGLang):
docker compose stop sglang
VLLM_GPU_MEM=0.9 docker compose up -d vllm

# Call via LiteLLM using the "gemma-vllm" model name
curl http://localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer sk-llmserving-master-key" \
  -d '{"model":"gemma-vllm","messages":[{"role":"user","content":"Hi!"}]}'
```

| | SGLang | vLLM |
|-|--------|------|
| Port | 30002 | 30003 |
| Memory policy | LPM scheduling | PagedAttention |
| Tool calling | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Best for | high concurrency | low latency |

---

## Monitoring: Grafana & Prometheus

### Access Grafana

- URL: `http://localhost:3000`
- Login: `admin` / `admin123` (or as set in `.env`)

### Pre-loaded Dashboards

| Dashboard | What it shows |
|-----------|---------------|
| **LLM Serving Overview** | SGLang req/s, P99 latency, queue, LiteLLM req/s, Redis memory |
| **NVIDIA DCGM GPU** | GPU util%, VRAM%, temperature, power draw per GPU |
| **Node Exporter Full** | CPU, RAM, disk I/O, network per host |
| **Redis Exporter** | Hit rate, memory, commands/s, connected clients |

### Download/refresh community dashboards

```bash
bash scripts/download_dashboards.sh
# Grafana picks up new files automatically (30s refresh interval)
```

### Key Prometheus metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `sglang_request_success_total` | SGLang | Total completions served |
| `sglang_request_duration_seconds` | SGLang | Latency histogram |
| `sglang_queue_size` | SGLang | Current queue depth |
| `DCGM_FI_DEV_GPU_UTIL` | DCGM | GPU utilization % |
| `DCGM_FI_DEV_FB_USED` | DCGM | VRAM used (bytes) |
| `DCGM_FI_DEV_GPU_TEMP` | DCGM | GPU temperature °C |
| `redis_memory_used_bytes` | Redis Exp. | Redis memory |
| `litellm_requests_total` | LiteLLM | Proxy requests |
| `node_cpu_seconds_total` | Node Exp. | Host CPU time |
| `node_memory_MemAvailable_bytes` | Node Exp. | Host RAM available |

---

## WandB Integration (Post-Training)

See [`docs/wandb_guide.md`](./docs/wandb_guide.md) for the full guide. Quick reference:

### Why WandB over MLflow for LLM

| | WandB | MLflow |
|-|-------|--------|
| Prompt/completion tables | ✅ Native | ⚠️ Plugin |
| S3 artifact references | ✅ `add_reference` | Partial |
| Model Registry (no ops) | ✅ Cloud SaaS | Self-host |
| Lineage DAG | ✅ Visual | Basic |
| HPO sweeps | ✅ | ⚠️ |

### Quick integration

```bash
pip install wandb
wandb login          # enter key from https://wandb.ai/authorize
export WANDB_API_KEY=your_key
```

Add to your training script:

```python
import wandb

# Before training
wandb.init(project="gemma-finetune", config={"lr": 2e-4, "lora_r": 16})

# In train loop
wandb.log({"train/loss": loss, "train/lr": lr}, step=step)

# After training — log artifact with MinIO reference
artifact = wandb.Artifact("gemma-1b-finetune", type="model",
    metadata={"s3_path": "s3://models/gemma-1b-finetune/",
              "endpoint": "http://localhost:9100"})
artifact.add_reference("s3://models/gemma-1b-finetune/")
wandb.log_artifact(artifact)

# Promote to registry when eval passes
artifact.link("entity/model-registry/gemma-1b-finetune", aliases=["production"])
wandb.finish()
```

For MinIO S3 reference to work, set:
```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export AWS_ENDPOINT_URL=http://localhost:9100
```

---

## File Structure

```
llm_serving/
├── docker-compose.yml              # 14-service orchestration
├── .env                            # Credentials (DO NOT commit)
│
├── train/                          # Host-run training pipeline (LoRA + MLflow)
│   ├── README.md
│   ├── train.py
│   ├── merge_model.py              # Download adapter from MLflow + merge
│   ├── config.json
│   └── requirements.txt
│
├── dockerfiles/
│   └── Dockerfile.uploader         # Python uploader image
│
├── scripts/
│   ├── setup.sh                    # 🔑 Zero-to-hero bootstrap (empty machine)
│   ├── install_gpu_docker.sh       # GPU-only fix script
│   ├── download_dashboards.sh      # Fetch Grafana community dashboards
│   ├── update_model.sh             # 🔄 Hot-swap model (upload + sync + restart)
│   ├── upload_model_to_minio.py    # Python MinIO uploader
│   └── sglang_entrypoint.sh        # SGLang launch arguments
│
├── litellm/
│   └── config.yaml                 # Model routing + Redis cache + Prometheus
│
├── prometheus/
│   ├── prometheus.yml              # Scrape jobs (SGLang, vLLM, LiteLLM, GPU, Redis, Node)
│   └── alert_rules.yml             # Alerting rules
│
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml
│   │   └── dashboards/dashboards.yml
│   └── dashboards/
│       ├── llm_serving_overview.json   # Custom: SGLang + LiteLLM + Redis
│       ├── nvidia_dcgm_gpu.json        # Downloaded: GPU metrics (ID 12239)
│       ├── node_exporter_full.json     # Downloaded: host metrics (ID 1860)
│       └── redis_exporter.json         # Downloaded: Redis (ID 763)
│
└── docs/
    ├── llmops_diagram.md           # Mermaid architecture diagram
    ├── wandb_guide.md              # WandB integration guide
    └── wandb_integration.py        # Code snippets
```

---

## Troubleshooting

### `could not select device driver "nvidia"`

nvidia-container-toolkit not installed or Docker not restarted:
```bash
sudo bash scripts/install_gpu_docker.sh
```

### `unknown or invalid runtime name: nvidia`

Remove any `runtime: nvidia` from docker-compose.yml (already fixed). Use only `deploy.resources` block.

### SGLang never becomes healthy

```bash
docker compose logs sglang -f      # watch loading progress
# Common causes:
# - Out of VRAM: reduce SGLang VRAM knobs in .env (SGLANG_MEM_FRACTION_STATIC / SGLANG_MAX_RUNNING_REQUESTS)
# - Wrong model path: docker compose exec sglang ls /model
# - Model not synced: docker compose run --rm model-sync
```

### MinIO upload times out

```bash
docker compose ps minio
curl http://localhost:9100/minio/health/live
docker compose logs minio
```

### vLLM OOM (out of VRAM)

```bash
# Lower GPU memory fraction in .env:
VLLM_GPU_MEM=0.3
docker compose restart vllm
```

### Grafana "No data"

```bash
# Check Prometheus targets
open http://localhost:9091/targets
# Check all services are running
docker compose ps
```

### Model volume empty after restart

The `model_weights` volume persists across `docker compose down` (without `-v`).  
If lost, re-sync from MinIO:
```bash
docker compose run --rm model-sync
docker compose restart sglang
```

---

## Stop / Clean Up

```bash
# Stop all containers (keep data volumes = model stays in MinIO)
docker compose down

# Stop + remove ALL volumes (full clean slate)
docker compose down -v

# Restart a single service
docker compose restart sglang

# View logs
docker compose logs sglang -f
docker compose logs litellm -f
docker compose logs grafana -f

# Show all running services + ports
docker compose ps
```
