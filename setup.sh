#!/usr/bin/env bash
# One-time setup for SagePod on a SageMaker GPU instance.
# Usage:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Checking for a GPU"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (this will be very slow)')" \
  || { echo "PyTorch is missing. Use a SageMaker image that includes PyTorch."; exit 1; }

echo "==> Installing Python packages"
python -m pip install -q -r requirements.txt
python -m spacy download en_core_web_sm -q || true
# Some SageMaker images ship spacy-curated-transformers, which breaks with the
# newer spaCy packages installed above and stops the voice model from loading.
# SagePod does not use it, so remove it.
python -m pip uninstall -y -q spacy-curated-transformers 2>/dev/null || true

echo "==> Installing espeak-ng (helps the voice pronounce unusual words; optional)"
if ! command -v espeak-ng >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    (sudo apt-get update -qq && sudo apt-get install -y -qq espeak-ng) || echo "   (skipped: apt-get not allowed)"
  elif command -v conda >/dev/null 2>&1; then
    conda install -y -q -c conda-forge espeak-ng || echo "   (skipped: conda install failed)"
  fi
fi

echo "==> Downloading cloudflared (gives the app a private web address for your phone)"
mkdir -p bin
if [ ! -x bin/cloudflared ]; then
  curl -fsSL -o bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x bin/cloudflared
fi

echo "==> Downloading the models (about 15 GB, only needed once)"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
llm = os.environ.get("SAGEPOD_LLM", "Qwen/Qwen2.5-7B-Instruct")
snapshot_download(llm, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
snapshot_download("hexgrad/Kokoro-82M")
print("Models ready.")
PY

echo
echo "Setup complete. Start the app with:  SAGEPOD_PASSWORD='pick-a-password' ./run.sh"
