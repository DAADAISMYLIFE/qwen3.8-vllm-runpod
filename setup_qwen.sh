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

# runpod 은 파드마다 드라이버가 다르다(예: 570/CUDA12.8 vs 580/CUDA13.0).
# 드라이버가 지원하는 CUDA 버전을 감지해 그에 맞는 torch 빌드를 설치한다.
# (cu130 을 12.8 드라이버에 깔면 "NVIDIA driver too old" 로 죽는다.)
CUDA_DRV="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
CUDA_MAJOR="${CUDA_DRV%%.*}"
if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
  TORCH_BACKEND="cu130"
else
  TORCH_BACKEND="cu128"   # 12.x 드라이버 (RTX 5090 은 12.8 부터 지원)
fi
echo "드라이버 CUDA=${CUDA_DRV:-unknown} -> torch backend=${TORCH_BACKEND}"

if [ "$CURRENT_VLLM" = "$VLLM_VERSION" ]; then
  echo "vLLM $VLLM_VERSION already installed."
else
  echo "Installing vLLM $VLLM_VERSION ($TORCH_BACKEND)..."
  # 1순위: vLLM 이 드라이버 맞춰 torch 백엔드 자동 선택
  pip install "vllm==$VLLM_VERSION" --torch-backend="$TORCH_BACKEND" || \
  pip install "vllm==$VLLM_VERSION" --torch-backend=auto || {
    # 폴백: torch 를 해당 backend 로 먼저 고정 설치 후 vllm
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}"
    pip install "vllm==$VLLM_VERSION"
  }
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