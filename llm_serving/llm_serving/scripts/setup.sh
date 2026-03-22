#!/bin/bash
# ============================================================
#  setup.sh — Zero-to-Hero LLM Serving Stack Bootstrap
#  Works on a FRESH machine with nothing installed.
#
#  What it does:
#   1. Install Docker Engine (if missing)
#   2. Install Docker Compose plugin (if missing)
#   3. Install NVIDIA Driver check
#   4. Install nvidia-container-toolkit (if missing)
#   5. Configure Docker for GPU & restart daemon
#   6. Download Grafana dashboards
#   7. Start MinIO, upload model
#   8. Sync model from MinIO → shared volume
#   9. Start full stack
#
#  Run: sudo bash scripts/setup.sh
#       (sudo needed for system package installs)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

step() { echo -e "\n${BLUE}${BOLD}═══ $* ═══${NC}"; }
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
die()  { echo -e "${RED}❌ $*${NC}" >&2; exit 1; }

echo -e "\n${BOLD}╔══════════════════════════════════════════════════════════╗"
echo    "║     LLM Serving Stack — Full Bootstrap (Empty Machine)   ║"
echo -e "╚══════════════════════════════════════════════════════════╝${NC}\n"

# Detect OS
. /etc/os-release 2>/dev/null || die "Cannot detect OS"
OS_ID="${ID:-unknown}"
echo "   OS: ${PRETTY_NAME:-$OS_ID}"

# ─────────────────────────────────────────────────────────────────────────────
step "1/9  Docker Engine"
# ─────────────────────────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
  DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
  ok "Docker already installed: v${DOCKER_VER}"
else
  warn "Docker not found. Installing..."
  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
      apt-get update -qq
      apt-get install -y ca-certificates curl gnupg lsb-release
      install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://download.docker.com/linux/${OS_ID}/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      chmod a+r /etc/apt/keyrings/docker.gpg
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/${OS_ID} $(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list >/dev/null
      apt-get update -qq
      apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      ;;
    fedora|rhel|centos|rocky|almalinux)
      dnf -y install dnf-plugins-core
      dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
      dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      ;;
    arch|manjaro|endeavouros)
      pacman -Sy --noconfirm docker docker-compose
      ;;
    *)
      die "Unsupported OS: $OS_ID. Install Docker manually: https://docs.docker.com/engine/install/"
      ;;
  esac
  systemctl enable --now docker
  # Add current sudo user to docker group so non-root can use docker later
  REAL_USER="${SUDO_USER:-$USER}"
  usermod -aG docker "$REAL_USER" 2>/dev/null || true
  ok "Docker installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "2/9  Docker Compose plugin"
# ─────────────────────────────────────────────────────────────────────────────
if docker compose version &>/dev/null; then
  ok "Docker Compose: $(docker compose version --short)"
else
  warn "Docker Compose plugin missing. Installing..."
  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
      apt-get install -y docker-compose-plugin ;;
    fedora|rhel|centos|rocky|almalinux)
      dnf install -y docker-compose-plugin ;;
    arch|manjaro|endeavouros)
      pacman -Sy --noconfirm docker-compose ;;
  esac
  ok "Docker Compose installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "3/9  NVIDIA GPU driver"
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found.\nInstall the NVIDIA driver first:\n  Ubuntu: sudo ubuntu-drivers autoinstall && sudo reboot\n  Then re-run this script."
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
ok "GPU: ${GPU_INFO}"

# ─────────────────────────────────────────────────────────────────────────────
step "4/9  nvidia-container-toolkit"
# ─────────────────────────────────────────────────────────────────────────────
if command -v nvidia-ctk &>/dev/null && nvidia-ctk --version &>/dev/null 2>&1; then
  ok "nvidia-container-toolkit already installed"
else
  warn "Installing nvidia-container-toolkit..."
  case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
      apt-get install -y curl gnupg
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update -qq
      apt-get install -y nvidia-container-toolkit
      ;;
    fedora|rhel|centos|rocky|almalinux)
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
      dnf install -y nvidia-container-toolkit
      ;;
    arch|manjaro|endeavouros)
      pacman -Sy --noconfirm nvidia-container-toolkit ;;
    *)
      die "Unsupported OS for auto-install. See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
      ;;
  esac
  ok "nvidia-container-toolkit installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "5/9  Configure Docker GPU runtime"
# ─────────────────────────────────────────────────────────────────────────────
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 3
ok "Docker daemon restarted with NVIDIA runtime"

# Verify GPU in Docker
echo "   Testing GPU inside Docker container..."
if docker run --rm --gpus all \
    nvidia/cuda:12.1.0-base-ubuntu22.04 \
    nvidia-smi -L 2>&1 | grep -q "GPU 0"; then
  ok "Docker GPU access verified"
else
  die "Docker GPU test failed.\nTry: sudo reboot  then  sudo bash $0"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "6/9  Download Grafana dashboards"
# ─────────────────────────────────────────────────────────────────────────────
# Run as real user (not root) so file ownership is correct
REAL_USER="${SUDO_USER:-$USER}"
if [ "$REAL_USER" = "root" ]; then
  bash "$SCRIPT_DIR/download_dashboards.sh"
else
  su - "$REAL_USER" -c "bash '$SCRIPT_DIR/download_dashboards.sh'"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "7/9  Start MinIO & upload model"
# ─────────────────────────────────────────────────────────────────────────────
cd "$SERVING_DIR"
docker compose up -d minio

echo "   Waiting for MinIO to be healthy..."
TIMEOUT=60; COUNT=0
until docker compose exec -T minio mc ready local >/dev/null 2>&1; do
  COUNT=$((COUNT+2)); [ $COUNT -ge $TIMEOUT ] && die "MinIO never became healthy"
  sleep 2
done
ok "MinIO ready → http://localhost:9101"

echo "   Uploading model to MinIO (first time only)..."
docker compose run --rm model-uploader
ok "Model uploaded → s3://models/gemma-1b-finetune/"

# ─────────────────────────────────────────────────────────────────────────────
step "8/9  Sync model MinIO → Docker volume"
# ─────────────────────────────────────────────────────────────────────────────
docker compose run --rm model-sync
ok "Model volume populated with weights from MinIO"

# ─────────────────────────────────────────────────────────────────────────────
step "9/9  Start full stack"
# ─────────────────────────────────────────────────────────────────────────────
docker compose up -d

echo ""
echo "   ⏳ Waiting for SGLang to load model (may take 2-5 min)..."
TIMEOUT=300; COUNT=0
until curl -sf http://localhost:30002/health >/dev/null 2>&1; do
  COUNT=$((COUNT+10)); [ $COUNT -ge $TIMEOUT ] && { warn "SGLang health timeout. Check: docker compose logs sglang -f"; break; }
  echo "      [${COUNT}s/${TIMEOUT}s] Loading model..."
  sleep 10
done
curl -sf http://localhost:30002/health >/dev/null 2>&1 && ok "SGLang is serving!"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗"
echo    "║              🎉  Stack is Ready!                          ║"
echo    "╠══════════════════════════════════════════════════════════╣"
echo    "║  MinIO Console  → http://localhost:9101  (minioadmin)    ║"
echo    "║  SGLang API     → http://localhost:30002/v1              ║"
echo    "║  LiteLLM Proxy  → http://localhost:4001/v1               ║"
echo    "║  LiteLLM UI     → http://localhost:4001/ui               ║"
echo    "║  Prometheus     → http://localhost:9091                  ║"
echo    "║  Grafana        → http://localhost:3000  (admin)         ║"
echo    "╠══════════════════════════════════════════════════════════╣"
echo    "║  Quick test:                                             ║"
echo    "║  curl http://localhost:4001/v1/chat/completions  \\       ║"
echo    "║    -H 'Authorization: Bearer sk-llmserving-master-key'  ║"
echo    "║    -H 'Content-Type: application/json'  \\               ║"
echo    "║    -d '{\"model\":\"gemma-1b-finetune\",                    ║"
echo    "║         \"messages\":[{\"role\":\"user\",\"content\":\"Hi!\"}]}'  ║"
echo    "╠══════════════════════════════════════════════════════════╣"
echo    "║  vLLM API (port 30003):                                ║"
echo -e "║  docker compose stop vllm / start vllm               ║"
echo -e "╚══════════════════════════════════════════════════════════╝${NC}\n"
