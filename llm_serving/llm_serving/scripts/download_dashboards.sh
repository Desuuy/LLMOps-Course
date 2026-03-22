#!/bin/bash
# ============================================================
#  download_dashboards.sh
#  Download community Grafana dashboards from grafana.com
#  Place them in grafana/dashboards/ for auto-provisioning
#
#  Usage (run from llm_serving/):
#    bash scripts/download_dashboards.sh
# ============================================================
set -euo pipefail

DASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/grafana/dashboards"
mkdir -p "$DASH_DIR"

echo "📊 Downloading Grafana dashboards → $DASH_DIR"
echo ""

# Helper: download one dashboard
download_dash() {
  local id="$1"
  local name="$2"
  local out="$DASH_DIR/${name}.json"

  if [ -f "$out" ]; then
    echo "   ✅ Already exists: ${name}.json"
    return
  fi

  echo "   ⬇️  Downloading: ${name} (ID=${id})..."
  # Grafana.com API returns the raw dashboard JSON at this URL
  if curl -fsSL \
      -H "Accept: application/json" \
      "https://grafana.com/api/dashboards/${id}/revisions/latest/download" \
      -o "$out"; then
    # Fix datasource reference: grafana.com dashboards use ${DS_PROMETHEUS} or ${DS_PROM}
    # Our provisioned datasource is named "Prometheus" — patch both formats
    sed -i 's/"${DS_PROMETHEUS}"/"Prometheus"/g; s/\${DS_PROMETHEUS}/Prometheus/g; s/"${DS_PROM}"/"Prometheus"/g; s/\${DS_PROM}/Prometheus/g' "$out" 2>/dev/null || true
    echo "      ✅ Saved: ${name}.json"
  else
    echo "      ❌ Failed to download ${name} (ID=${id}). Skipping."
    rm -f "$out"
  fi
}

# ── Community dashboards ──────────────────────────────────────────────────────
# NVIDIA DCGM GPU metrics
download_dash 12239 "nvidia_dcgm_gpu"

# Redis (by Prometheus Redis Exporter)
download_dash 763 "redis_exporter"

# Node Exporter Full (host: CPU, RAM, disk, network)
download_dash 1860 "node_exporter_full"

# Postgres (by pgbadger / prometheus-postgres-exporter)
# Note: not adding postgres exporter to compose, but useful if added later
# download_dash 9628 "postgres_overview"

echo ""
echo "✅ Dashboard download complete."
echo "   Grafana will auto-load them on next start."
echo "   Location: $DASH_DIR/"
ls -1 "$DASH_DIR/"
