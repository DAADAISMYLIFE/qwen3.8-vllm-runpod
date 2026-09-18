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

# ── 드라이버 호환성 사전 점검 (중요) ─────────────────────────────
# vLLM 0.29.0 은 torch 2.13.0 을 요구하는데, torch 2.13.0 은 CUDA-13 전용 빌드다
# (nccl-cu13/cudnn-cu13 의존, cu128 빌드 자체가 없음). 따라서 드라이버가 CUDA 13 미만인
# 파드에서는 pip 으로 절대 못 돌린다("NVIDIA driver too old" / "undefined symbol: ncclCommResume").
# runpod 은 파드마다 드라이버가 다르므로(570/12.8 vs 580/13.0) 설치 전에 확인하고,
# 12.x 면 헛수고 대신 즉시 멈춰서 CUDA-13 파드로 다시 잡으라고 안내한다.
CUDA_DRV="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
CUDA_MAJOR="${CUDA_DRV%%.*}"
if [ -z "$CUDA_MAJOR" ]; then
  echo "경고: nvidia-smi 로 드라이버 CUDA 버전을 못 읽었다. GPU 파드가 맞는지 확인해라."
elif [ "$CUDA_MAJOR" -lt 13 ] 2>/dev/null; then
  echo "=================================================================="
  echo "  이 파드의 GPU 드라이버는 CUDA ${CUDA_DRV} 까지만 지원한다."
  echo "  vLLM ${VLLM_VERSION} 는 torch 2.13(=CUDA 13 전용)을 요구하므로 여기선 못 돈다."
  echo "  → runpod 에서 CUDA 13.0(드라이버 580+) 파드로 다시 잡아라."
  echo "    (드라이버는 호스트 소유라 컨테이너 안에서 못 바꾼다.)"
  echo "=================================================================="
  exit 1
fi
echo "드라이버 CUDA=${CUDA_DRV:-unknown} (>=13 OK)"

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