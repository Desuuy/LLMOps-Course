#!/bin/bash
# ============================================================
#  update_model.sh — Hot-swap the served model without downtime
#
#  Use when you have a new (better) finetuned model checkpoint
#  and want to replace what SGLang / vLLM are serving.
#
#  Flow:
#   1. Upload new model files from LOCAL_MODEL_DIR → MinIO
#   2. Re-sync MinIO → model_weights Docker volume
#   3. Gracefully restart the inference server(s)
#
#  Usage:
#    bash scripts/update_model.sh [options]
#
#  Options:
#    --model-dir PATH     Local directory of new model (default: ../gemma-1b-finetune)
#    --model-name NAME    MinIO prefix / served-model-name (default: gemma-1b-finetune)
#    --engine ENGINE      Which engine to restart: sglang | vllm | all (default: sglang)
#    --no-upload          Skip upload, just re-sync and restart (model already in MinIO)
#
#  Examples:
#    # Replace with a new checkpoint in a different folder
#    bash scripts/update_model.sh --model-dir ../gemma-1b-v2 --model-name gemma-1b-v2
#
#    # Model already in MinIO, just resync + restart
#    bash scripts/update_model.sh --no-upload
#
#    # Restart vLLM instead of SGLang
#    bash scripts/update_model.sh --engine vllm
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-$(dirname "$SERVING_DIR")/gemma-1b-finetune}"
MODEL_NAME="${MODEL_NAME:-gemma-1b-finetune}"
ENGINE="${ENGINE:-sglang}"
SKIP_UPLOAD=0

# ── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)   MODEL_DIR="$2"; shift 2 ;;
    --model-name)  MODEL_NAME="$2"; shift 2 ;;
    --engine)      ENGINE="$2"; shift 2 ;;
    --no-upload)   SKIP_UPLOAD=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
info() { echo -e "${BLUE}ℹ️  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }

echo -e "\n${BOLD}🔄 Model Hot-Swap: ${MODEL_NAME} → ${ENGINE}${NC}\n"
cd "$SERVING_DIR"

# ── Step 1: Upload new model to MinIO ────────────────────────────────────────
if [ "$SKIP_UPLOAD" -eq 0 ]; then
  info "Step 1/3 — Uploading new model weights to MinIO..."
  if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
  fi

  # Upload via the uploader service (builds image if needed)
  docker compose run --rm \
    -e MODEL_SRC_DIR=/model-src \
    -e MODEL_PREFIX="$MODEL_NAME" \
    -v "${MODEL_DIR}:/model-src:ro" \
    model-uploader

  ok "Model uploaded: s3://models/${MODEL_NAME}/"
else
  info "Step 1/3 — Skipping upload (--no-upload)"
fi

# ── Step 2: Resync MinIO → Docker volume ────────────────────────────────────
info "Step 2/3 — Syncing MinIO → model_weights volume..."

# Run model-sync manually (handles the mc mirror operation)
docker compose run --rm \
  -e MODEL_PREFIX="$MODEL_NAME" \
  model-sync

ok "Volume updated with latest weights from s3://models/${MODEL_NAME}/"

# ── Step 3: Restart inference server(s) ─────────────────────────────────────
info "Step 3/3 — Restarting inference engine(s): ${ENGINE}"

restart_sglang() {
  warn "Restarting SGLang (brief downtime ~2-5 min while model loads)..."
  docker compose restart sglang
  echo "   Waiting for SGLang to be healthy..."
  TIMEOUT=300; COUNT=0
  until curl -sf http://localhost:30002/health >/dev/null 2>&1; do
    COUNT=$((COUNT+10))
    [ $COUNT -ge $TIMEOUT ] && { warn "Timeout. Check: docker compose logs sglang -f"; break; }
    echo "   [${COUNT}s] Loading..."
    sleep 10
  done
  curl -sf http://localhost:30002/health >/dev/null 2>&1 && ok "SGLang serving new model: ${MODEL_NAME}"
}

restart_vllm() {
  warn "Restarting vLLM (brief downtime ~1-3 min while model loads)..."
  docker compose restart vllm
  echo "   Waiting for vLLM to be healthy..."
  TIMEOUT=180; COUNT=0
  until curl -sf http://localhost:30003/health >/dev/null 2>&1; do
    COUNT=$((COUNT+10))
    [ $COUNT -ge $TIMEOUT ] && { warn "Timeout. Check: docker compose logs vllm -f"; break; }
    echo "   [${COUNT}s] Loading..."
    sleep 10
  done
  curl -sf http://localhost:30003/health >/dev/null 2>&1 && ok "vLLM serving new model: ${MODEL_NAME}"
}

case "$ENGINE" in
  sglang) restart_sglang ;;
  vllm)   restart_vllm ;;
  all)    restart_sglang; restart_vllm ;;
  *)      echo "❌ Unknown engine: $ENGINE (use: sglang | vllm | all)"; exit 1 ;;
esac

echo ""
echo -e "${BOLD}✅ Hot-swap complete!${NC}"
echo "   Model: ${MODEL_NAME}"
echo "   Engine: ${ENGINE}"
echo ""
echo "Verify with:"
echo "  curl http://localhost:4001/v1/models -H 'Authorization: Bearer \$LITELLM_MASTER_KEY'"
echo ""
echo "If you changed the model name, update litellm/config.yaml → restart litellm:"
echo "  docker compose restart litellm"
