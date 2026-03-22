#!/bin/bash
# ============================================================
#  install_gpu_docker.sh
#  Install nvidia-container-toolkit and configure Docker for GPU
#
#  Run ONCE before starting the LLM serving stack:
#    chmod +x scripts/install_gpu_docker.sh
#    sudo ./scripts/install_gpu_docker.sh
#
#  What this fixes:
#    "could not select device driver 'nvidia' with capabilities: [[gpu]]"
#    This error means Docker cannot find the nvidia runtime.
#    Solution: install nvidia-container-toolkit + configure Docker daemon.
# ============================================================
set -euo pipefail

# ── Must run as root ─────────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run as root: sudo $0"
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     NVIDIA Container Toolkit Setup for Docker GPU        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Verify GPU driver is installed ───────────────────────────────────
echo "🔍 [1/5] Checking NVIDIA driver..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "❌ nvidia-smi not found. Install the NVIDIA driver first:"
  echo "   Ubuntu:  sudo ubuntu-drivers autoinstall && sudo reboot"
  echo "   Or manually download from https://www.nvidia.com/Download/index.aspx"
  exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)
echo "   ✅ GPU detected: ${GPU_INFO}"

# ── Step 2: Detect OS ────────────────────────────────────────────────────────
echo ""
echo "🔍 [2/5] Detecting OS..."
if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS=$ID
  VERSION_ID=${VERSION_ID:-""}
  echo "   Found: ${PRETTY_NAME:-$OS}"
else
  echo "❌ Cannot detect OS. Exiting."
  exit 1
fi

# ── Step 3: Install nvidia-container-toolkit ─────────────────────────────────
echo ""
echo "📦 [3/5] Installing nvidia-container-toolkit..."

case "$OS" in
  ubuntu|debian|linuxmint|pop)
    # Check if already installed
    if dpkg -l | grep -q "nvidia-container-toolkit"; then
      echo "   ✅ nvidia-container-toolkit already installed:"
      dpkg -l | grep nvidia-container-toolkit | awk '{print "      ",$2,$3}'
    else
      echo "   Adding NVIDIA Container Toolkit apt repository..."

      apt-get install -y curl gnupg

      # Official NVIDIA repo setup (2024+ method)
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

      apt-get update -qq
      apt-get install -y nvidia-container-toolkit

      echo "   ✅ nvidia-container-toolkit installed"
    fi
    ;;

  fedora|rhel|centos|rocky|almalinux)
    if rpm -q nvidia-container-toolkit &>/dev/null; then
      echo "   ✅ nvidia-container-toolkit already installed"
    else
      echo "   Adding NVIDIA Container Toolkit yum/dnf repository..."
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
      dnf install -y nvidia-container-toolkit
      echo "   ✅ nvidia-container-toolkit installed"
    fi
    ;;

  arch|manjaro|endeavouros)
    if pacman -Q nvidia-container-toolkit &>/dev/null 2>&1; then
      echo "   ✅ nvidia-container-toolkit already installed"
    else
      pacman -Sy --noconfirm nvidia-container-toolkit
      echo "   ✅ nvidia-container-toolkit installed"
    fi
    ;;

  *)
    echo "⚠️  Unknown OS: $OS. Attempting generic apt install..."
    apt-get install -y nvidia-container-toolkit 2>/dev/null \
      || { echo "❌ Failed. Install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"; exit 1; }
    ;;
esac

# ── Step 4: Configure Docker daemon ─────────────────────────────────────────
echo ""
echo "⚙️  [4/5] Configuring Docker daemon for NVIDIA GPU support..."

# nvidia-ctk configures /etc/docker/daemon.json
nvidia-ctk runtime configure --runtime=docker

# Show what was written
if [ -f /etc/docker/daemon.json ]; then
  echo "   /etc/docker/daemon.json now contains:"
  cat /etc/docker/daemon.json | sed 's/^/   /'
fi

# Restart Docker
echo ""
echo "🔄 Restarting Docker daemon..."
if systemctl is-active --quiet docker; then
  systemctl restart docker
  sleep 3
  echo "   ✅ Docker restarted"
else
  echo "   ⚠️  Docker is not running via systemd. Start it manually: sudo dockerd"
fi

# ── Step 5: Verify GPU access inside Docker ──────────────────────────────────
echo ""
echo "🧪 [5/5] Verifying GPU access inside Docker..."

if docker run --rm --gpus all \
    nvidia/cuda:12.1.0-base-ubuntu22.04 \
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  ✅  GPU is accessible from Docker!                      ║"
  echo "║     You can now run: sudo docker compose up -d sglang   ║"
  echo "╚══════════════════════════════════════════════════════════╝"
else
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  ❌  GPU test failed inside Docker.                      ║"
  echo "║  Possible causes:                                        ║"
  echo "║   1. NVIDIA driver mismatch (try: sudo reboot)           ║"
  echo "║   2. Docker wasn't fully restarted                       ║"
  echo "║   3. Kernel module not loaded: sudo modprobe nvidia      ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  echo "Debug commands:"
  echo "  nvidia-smi                          # check driver on host"
  echo "  cat /etc/docker/daemon.json         # check daemon config"
  echo "  sudo systemctl status docker        # docker service status"
  echo "  sudo journalctl -u docker -n 50     # docker logs"
  exit 1
fi
