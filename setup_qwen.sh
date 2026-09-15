#!/usr/bin/env bash
set -euo pipefail

VLLM_VERSION="0.29.0"

export HF_HOME="/workspace/hf-cache"
export VLLM_CACHE_ROOT="/workspace/vllm-cache"
export TORCHINDUCTOR_CACHE_DIR="/workspace/torchinductor-cache"

mkdir -p \
  "$HF_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR"

echo "[1/3] Environment"
python3 --version
pip --version
nvidia-smi || true

echo
echo "[2/3] vLLM"

CURRENT_VLLM="$(
  python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true
)"

if [ "$CURRENT_VLLM" = "$VLLM_VERSION" ]; then
  echo "vLLM $VLLM_VERSION already installed."
else
  echo "Installing vLLM $VLLM_VERSION..."
  pip install "vllm==$VLLM_VERSION"
fi

echo
echo "[3/3] Versions"

python3 - <<'PY'
import torch
import vllm

print("torch       :", torch.__version__)
print("vllm        :", vllm.__version__)
print("CUDA        :", torch.version.cuda)
print("CUDA usable :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("Capability  :", torch.cuda.get_device_capability(0))
    print("VRAM GiB    :", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ))
PY

echo
echo "Setup complete."