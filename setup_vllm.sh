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
MODEL_CACHE="$HF_HOME/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4"

# ── 드라이버 호환성 사전 점검 (중요) ─────────────────────────────
# vLLM 0.29.0 은 torch==2.13.0 을 고정하고, torch 2.13 은 cu129/cu130 빌드만 있다(cu128 없음).
# vLLM 자체도 0.26 이후로는 cu129/cu130 휠만 배포한다(GitHub 릴리스·wheels.vllm.ai 모두 cu128 없음).
# 드라이버 570(CUDA 12.8) 파드에서 실제로 시도한 결과:
#   - PyPI 기본(cu130): "NVIDIA driver too old"
#   - +cu129 휠: torch import/cuda.is_available() 은 통과하지만 모델 로드 시 Marlin 커널에서
#     "the provided PTX was compiled with an unsupported toolchain" (cudaErrorUnsupportedPtxVersion).
#     CUDA 마이너 호환은 SASS 에만 적용되고 PTX JIT 는 드라이버 툴체인 이하여야 하므로 우회 불가.
# 따라서 CUDA 13 미만 드라이버면 즉시 멈추고 CUDA 13.0(드라이버 580+) 파드로 다시 잡으라고 안내한다.
CUDA_DRV="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
CUDA_MAJOR="${CUDA_DRV%%.*}"
if [ -z "$CUDA_MAJOR" ]; then
  echo "경고: nvidia-smi 로 드라이버 CUDA 버전을 못 읽었다. GPU 파드가 맞는지 확인해라."
elif [ "$CUDA_MAJOR" -lt 13 ]; then
  echo "=================================================================="
  echo "  이 파드의 GPU 드라이버는 CUDA ${CUDA_DRV} 까지만 지원한다."
  echo "  vLLM ${VLLM_VERSION} 는 cu129/cu130 휠만 있고, 12.8 드라이버에선 cu129 도"
  echo "  모델 로드 시 PTX JIT(cudaErrorUnsupportedPtxVersion) 로 죽는다(실측)."
  echo "  → runpod 배포 화면 Additional Filters 에서 CUDA Version 13.0 을 걸고 다시 잡아라."
  echo "    (드라이버는 호스트 소유라 컨테이너 안에서 못 바꾼다.)"
  echo "=================================================================="
  exit 1
fi
echo "드라이버 CUDA=${CUDA_DRV:-unknown} (>=13 OK)"

# ── 디스크 사전 점검 ─────────────────────────────────────────────
# 모델 safetensors 합 21GB. runpod 기본 볼륨(/workspace) 20GB 면 다운로드 중 ENOSPC 로 죽는다.
NEED_GB=25
AVAIL_GB="$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')"
CACHED=0
if [ -d "$MODEL_CACHE" ]; then
  CACHED="$(find "$MODEL_CACHE" -name '*.safetensors' | wc -l)"
  INCOMPLETE="$(find "$MODEL_CACHE" -name '*.incomplete' | wc -l)"
else
  INCOMPLETE=0
fi
if [ "$CACHED" -ge 5 ] && [ "$INCOMPLETE" -eq 0 ]; then
  echo "모델 캐시 있음($MODEL_CACHE, safetensors ${CACHED}개) -> 디스크 점검 생략"
elif [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  echo "=================================================================="
  echo "  $HF_HOME 여유 ${AVAIL_GB}GB < 필요 ${NEED_GB}GB. 모델(21GB)을 못 받는다."
  echo "  → runpod 파드 Edit 에서 Volume Disk 를 60GB 이상으로 늘려라(늘리기만 가능, 데이터 유지)."
  echo "    재시작되면 이 스크립트를 다시 돌려라(컨테이너 디스크의 pip 설치는 날아간다)."
  echo "=================================================================="
  exit 1
else
  echo "디스크 여유 ${AVAIL_GB:-?}GB (>=${NEED_GB} OK)"
fi

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