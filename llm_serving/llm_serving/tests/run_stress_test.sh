#!/usr/bin/env bash
# ============================================================
# LLM Serving — Locust Stress Test Runner
# ============================================================
# Usage:
#   ./tests/run_stress_test.sh              # interactive web UI
#   ./tests/run_stress_test.sh headless     # quick headless smoke test
#   ./tests/run_stress_test.sh soak         # longer soak test
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/reports"
LOCUSTFILE="${SCRIPT_DIR}/locustfile.py"
HOST="${LLM_HOST:-http://localhost:4001}"
MODEL="${LLM_MODEL:-gemma-1b-finetune}"
API_KEY="${LLM_API_KEY:-sk-llmserving-master-key}"

mkdir -p "${REPORT_DIR}"

MODE="${1:-ui}"

export LLM_MODEL="${MODEL}" #"gemma-vllm" #
export LLM_API_KEY="${API_KEY}"

case "${MODE}" in
  ui)
    echo ">>> Starting Locust web UI at http://localhost:8089"
    echo ">>> Target host : ${HOST}"
    echo ">>> Model       : ${MODEL}"
    locust -f "${LOCUSTFILE}" \
           --host "${HOST}"
    ;;

  headless)
    # Quick smoke: 10 users, ramp 2/s, run 1 minute
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ">>> Headless smoke test — 10 users, 1 min"
    locust -f "${LOCUSTFILE}" \
           --host "${HOST}" \
           --headless \
           -u 10 -r 2 \
           --run-time 1m \
           --html "${REPORT_DIR}/smoke_${TIMESTAMP}.html" \
           --csv  "${REPORT_DIR}/smoke_${TIMESTAMP}"
    echo ">>> Report saved to ${REPORT_DIR}/smoke_${TIMESTAMP}.html"
    ;;

  medium)
    # Medium load: 30 users, ramp 3/s, run 5 minutes
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ">>> Medium load test — 30 users, 5 min"
    locust -f "${LOCUSTFILE}" \
           --host "${HOST}" \
           --headless \
           -u 30 -r 3 \
           --run-time 5m \
           --html "${REPORT_DIR}/medium_${TIMESTAMP}.html" \
           --csv  "${REPORT_DIR}/medium_${TIMESTAMP}"
    echo ">>> Report saved to ${REPORT_DIR}/medium_${TIMESTAMP}.html"
    ;;

  soak)
    # Soak test: 50 users, ramp 2/s, run 30 minutes
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ">>> Soak test — 50 users, 30 min"
    locust -f "${LOCUSTFILE}" \
           --host "${HOST}" \
           --headless \
           -u 50 -r 2 \
           --run-time 30m \
           --html "${REPORT_DIR}/soak_${TIMESTAMP}.html" \
           --csv  "${REPORT_DIR}/soak_${TIMESTAMP}"
    echo ">>> Report saved to ${REPORT_DIR}/soak_${TIMESTAMP}.html"
    ;;

  spike)
    # Spike test: ramp from 0 to 100 in 10s, then stop
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo ">>> Spike test — 100 users, ramp 10/s, run 2 min"
    locust -f "${LOCUSTFILE}" \
           --host "${HOST}" \
           --headless \
           -u 100 -r 10 \
           --run-time 2m \
           --html "${REPORT_DIR}/spike_${TIMESTAMP}.html" \
           --csv  "${REPORT_DIR}/spike_${TIMESTAMP}"
    echo ">>> Report saved to ${REPORT_DIR}/spike_${TIMESTAMP}.html"
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Valid modes: ui | headless | medium | soak | spike"
    exit 1
    ;;
esac
