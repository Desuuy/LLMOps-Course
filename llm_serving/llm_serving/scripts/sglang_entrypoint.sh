#!/bin/bash
# SGLang entrypoint — serves the Gemma-3 1B finetuned model
# Model architecture: Gemma3ForCausalLM (gemma3_text)
# --enable-metrics  : REQUIRED for Prometheus /metrics endpoint
# --served-model-name: must match the LiteLLM config model name

set -e

MODEL_PATH="${MODEL_PATH:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-1b-finetune}"
PORT="${SGLANG_PORT:-30002}"

# VRAM tuning defaults are intentionally conservative so SGLang can coexist
# with vLLM on the same GPU.
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-4096}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"

echo "=========================================="
echo "  SGLang — Starting Gemma-3 1B Finetune  "
echo "  Model  : ${MODEL_PATH}"
echo "  Name   : ${SERVED_MODEL_NAME}"
echo "  Port   : ${PORT}"
echo "  Metrics: http://0.0.0.0:${PORT}/metrics"
echo "  Config : mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"
echo "           max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}"
echo "           chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE}"
echo "           context_len=${SGLANG_CONTEXT_LENGTH}"
echo "           max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS}"
echo "=========================================="

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${PORT}" \
  --host "0.0.0.0" \
  --tp 1 \
  --schedule-policy lpm \
  --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}" \
  --context-length "${SGLANG_CONTEXT_LENGTH}" \
  --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --enable-metrics \
  --trust-remote-code
