#!/usr/bin/env bash
# ============================================================
# Compare SGLang vs vLLM with the same Locust workload.
#
# Writes:
#  - tests/reports/compare_<timestamp>_config.json
#  - tests/reports/compare_<timestamp>_sglang_stdout.txt
#  - tests/reports/compare_<timestamp>_vllm_stdout.txt
#  - tests/reports/compare_<timestamp>_summary.json
#  - (plus the Locust HTML/CVS reports produced by run_stress_test.sh)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/reports"
mkdir -p "${REPORT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="${REPORT_DIR}/compare_${TS}"

LLM_HOST="${LLM_HOST:-http://localhost:4001}"
LLM_API_KEY="${LLM_API_KEY:-sk-llmserving-master-key}"
CONDA_ENV="${CONDA_ENV:-cole}"

# LiteLLM model_name routing:
SGLANG_MODEL="${SGLANG_MODEL:-gemma-1b-finetune}"
VLLM_MODEL="${VLLM_MODEL:-gemma-vllm}"
export LLM_HOST LLM_API_KEY SGLANG_MODEL VLLM_MODEL CONDA_ENV

snapshot_config() {
  local out="$1"
  # Effective config snapshot:
  # - SGLang runtime env (inside container)
  # - vLLM command line (docker inspect)
  # - Compose resolved config block
  # Use a quoted heredoc so Bash doesn't expand $SGLANG_* inside the embedded command.
  python3 - <<'PY' > "${out}"
import json,subprocess,os

def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()

compose_dir="/mnt/oldssd/home/pc/Documents/cole_labs/llm_serving"

sglang_env = sh("cd %s && docker compose exec -T sglang bash -lc 'echo $SGLANG_MEM_FRACTION_STATIC $SGLANG_MAX_RUNNING_REQUESTS $SGLANG_CHUNKED_PREFILL_SIZE $SGLANG_CONTEXT_LENGTH $SGLANG_MAX_PREFILL_TOKENS'" % compose_dir)
sglang_vals = sglang_env.split()
sglang_cfg = {
  "SGLANG_MEM_FRACTION_STATIC": float(sglang_vals[0]) if sglang_vals else None,
  "SGLANG_MAX_RUNNING_REQUESTS": int(sglang_vals[1]) if len(sglang_vals)>1 else None,
  "SGLANG_CHUNKED_PREFILL_SIZE": int(sglang_vals[2]) if len(sglang_vals)>2 else None,
  "SGLANG_CONTEXT_LENGTH": int(sglang_vals[3]) if len(sglang_vals)>3 else None,
  "SGLANG_MAX_PREFILL_TOKENS": int(sglang_vals[4]) if len(sglang_vals)>4 else None,
}

vllm_cmd = sh("docker inspect llms_vllm --format '{{json .Config.Cmd}}'")
vllm_cmd_full = sh("docker inspect llms_vllm --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'")

compose_config = sh("cd %s && docker compose config" % compose_dir)

data = {
  "LLM_HOST": os.environ.get("LLM_HOST"),
  "models": {"sglang": os.environ.get("SGLANG_MODEL"), "vllm": os.environ.get("VLLM_MODEL")},
  "sglang": sglang_cfg,
  "vllm": {"cmd": vllm_cmd_full, "cmd_json": vllm_cmd},
  "docker_compose_config_text_truncated": compose_config[:20000],
}

print(json.dumps(data, indent=2))
PY
}

run_locust_for_model() {
  local model="$1"
  local mode="$2"
  local out_stdout="$3"

  # run_stress_test.sh reads LLM_MODEL / LLM_API_KEY
  export LLM_MODEL="${model}"
  export LLM_API_KEY="${LLM_API_KEY}"
  # Force headless mode duration label to remain consistent:
  # run_stress_test.sh uses its own -u/-r/-run-time depending on mode argument.
  (cd "${SCRIPT_DIR}/.." && \
    if command -v conda >/dev/null 2>&1; then
      conda run -n "${CONDA_ENV}" --no-capture-output bash tests/run_stress_test.sh "${mode}"
    else
      bash tests/run_stress_test.sh "${mode}"
    fi) 2>&1 | tee "${out_stdout}"
}

extract_summary() {
  local stdout_file="$1"
  python3 - <<PY
import re,json
text=open("$stdout_file","r",encoding="utf-8",errors="ignore").read()
def get(label):
    # Example: "E2E latency p99     : 1360 ms"
    m=re.search(rf"{re.escape(label)}\\s*:\\s*([0-9.]+)\\s*ms", text)
    return float(m.group(1)) if m else None
print(json.dumps({
  "requests_measured": int(re.search(r"Requests measured\\s*:\\s*([0-9]+)", text).group(1)) if re.search(r"Requests measured\\s*:\\s*([0-9]+)", text) else None,
  "e2e_p50_ms": get("E2E latency p50"),
  "e2e_p95_ms": get("E2E latency p95"),
  "e2e_p99_ms": get("E2E latency p99"),
  "e2e_max_ms": get("E2E latency max"),
  "timestamp": "${TS}",
}, indent=2))
PY
}

read_gpu_procs() {
  python3 - <<PY
import subprocess,re,json
out=subprocess.check_output("nvidia-smi", shell=True, text=True, stderr=subprocess.STDOUT)
def proc_mem(name_sub):
    # find line containing name_sub and capture last column (MiB)
    for line in out.splitlines():
        if name_sub in line:
            m=re.search(r"([0-9]+)MiB", line)
            if m: return int(m.group(1))
    return None
data={"sglang_scheduler_mib":proc_mem("sglang::scheduler"),"vllm_enginecore_mib":proc_mem("VLLM::EngineCore")}
print(json.dumps(data))
PY
}

main() {
  snapshot_config "${RUN_PREFIX}_config.json"

  before="$(read_gpu_procs)"
  echo "${before}" > "${RUN_PREFIX}_gpu_before.json"

  echo "=== Running SGLang model: ${SGLANG_MODEL} ==="
  sglang_stdout="${RUN_PREFIX}_sglang_stdout.txt"
  run_locust_for_model "${SGLANG_MODEL}" "headless" "${sglang_stdout}"
  sglang_summary="$(extract_summary "${sglang_stdout}")"

  echo "=== Running vLLM model: ${VLLM_MODEL} ==="
  vllm_stdout="${RUN_PREFIX}_vllm_stdout.txt"
  run_locust_for_model "${VLLM_MODEL}" "headless" "${vllm_stdout}"
  vllm_summary="$(extract_summary "${vllm_stdout}")"

  after="$(read_gpu_procs)"
  echo "${after}" > "${RUN_PREFIX}_gpu_after.json"

  python3 - <<PY > "${RUN_PREFIX}_summary.json"
import json
before=json.load(open("${RUN_PREFIX}_gpu_before.json"))
after=json.load(open("${RUN_PREFIX}_gpu_after.json"))
sg=json.loads('''${sglang_summary}''')
vl=json.loads('''${vllm_summary}''')
out={"sglang":sg,"vllm":vl,"gpu_before":before,"gpu_after":after,"run_prefix":"${RUN_PREFIX}"}
print(json.dumps(out, indent=2))
PY

  echo ""
  echo "Done."
  echo "Summary: ${RUN_PREFIX}_summary.json"
}

main

