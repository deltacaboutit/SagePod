#!/usr/bin/env bash
# Start SagePod.
#
#   ./run.sh            Public HTTPS link via Cloudflare (easiest for a phone).
#                       Requires a password:  SAGEPOD_PASSWORD='...' ./run.sh
#   ./run.sh studio     Through the SageMaker Studio (JupyterLab) proxy.
#   ./run.sh notebook   Through a SageMaker Notebook Instance proxy.
#   ./run.sh local      Plain http://localhost:8501 (for testing).
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-tunnel}"
PORT="${PORT:-8501}"
ARGS=(--server.port "$PORT" --server.headless true --browser.gatherUsageStats false
      --server.maxUploadSize 20 --client.toolbarMode minimal
      --server.fileWatcherType none)

case "$MODE" in
  tunnel)
    if [ -z "${SAGEPOD_PASSWORD:-}" ]; then
      echo "The tunnel link is reachable from the internet, so a password is required."
      echo "Run:  SAGEPOD_PASSWORD='pick-a-password' ./run.sh"
      exit 1
    fi
    [ -x bin/cloudflared ] || { echo "cloudflared missing; run ./setup.sh first."; exit 1; }
    bin/cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" > cloudflared.log 2>&1 &
    TUNNEL_PID=$!
    trap 'kill $TUNNEL_PID 2>/dev/null || true' EXIT
    ( for _ in $(seq 1 30); do
        URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' cloudflared.log | head -1 || true)
        if [ -n "$URL" ]; then
          echo; echo "=============================================================="
          echo " Open this on your phone:  $URL"
          echo "=============================================================="; echo
          break
        fi
        sleep 1
      done ) &
    streamlit run app.py "${ARGS[@]}" --server.address 127.0.0.1
    ;;
  studio)
    BASE="${BASE_PATH:-jupyterlab/default/proxy/absolute/$PORT}"
    echo "Open your JupyterLab address, but replace everything after the domain with:  /$BASE/"
    exec streamlit run app.py "${ARGS[@]}" --server.baseUrlPath "$BASE" \
      --server.enableCORS false --server.enableXsrfProtection false
    ;;
  notebook)
    BASE="${BASE_PATH:-proxy/absolute/$PORT}"
    echo "Open https://<notebook-name>.notebook.<region>.sagemaker.aws/$BASE/"
    exec streamlit run app.py "${ARGS[@]}" --server.baseUrlPath "$BASE" \
      --server.enableCORS false --server.enableXsrfProtection false
    ;;
  local)
    exec streamlit run app.py "${ARGS[@]}"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use: tunnel, studio, notebook or local."; exit 1 ;;
esac
