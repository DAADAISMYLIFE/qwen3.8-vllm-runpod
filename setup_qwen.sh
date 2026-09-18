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
CURRENT_TORCH_CUDA="$(
  python3 -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true
)"
MODEL_CACHE="$HF_HOME/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4"

# ── 드라이버별 설치 경로 선택 ───────────────────────────────────
# vLLM 0.29.0 은 torch==2.13.0 을 고정하는데 torch 2.13 은 cu129/cu130 빌드만 있다(cu128 없음).
# PyPI 기본 휠은 cu130 이라 드라이버 570(CUDA 12.8) 파드에선 "NVIDIA driver too old" 로 죽는다.
# 대신 GitHub 릴리스의 +cu129 휠은 CUDA 12.x 마이너 호환으로 12.8 드라이버에서도 돈다
# (2026-09-18 RTX 5090 / 570.195 에서 torch 2.13.0+cu129 cuda.is_available()=True 확인).
#   드라이버 CUDA 13.x → pip 기본(cu130)
#   드라이버 CUDA 12.x → +cu129 휠 + pytorch cu129 인덱스
CUDA_DRV="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
CUDA_MAJOR="${CUDA_DRV%%.*}"
if [ -z "$CUDA_MAJOR" ]; then
  echo "경고: nvidia-smi 로 드라이버 CUDA 버전을 못 읽었다. GPU 파드가 맞는지 확인해라. (PyPI 기본 휠로 진행)"
  CUDA_MAJOR=13
fi
if [ "$CUDA_MAJOR" -ge 13 ]; then
  WHEEL="vllm==$VLLM_VERSION"
  EXTRA=()
  echo "드라이버 CUDA=${CUDA_DRV} -> PyPI 기본 휠(cu130)"
else
  WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_$(uname -m).whl"
  EXTRA=(--extra-index-url https://download.pytorch.org/whl/cu129)
  echo "드라이버 CUDA=${CUDA_DRV} -> +cu129 휠 (12.x 마이너 호환)"
  # 이전에 PyPI 기본(cu130) 이 깔려 있으면 버전이 같아도 다시 깐다
  if [ "$CURRENT_VLLM" = "$VLLM_VERSION" ] && [[ "$CURRENT_TORCH_CUDA" == 13* ]]; then
    echo "설치된 torch 가 cu${CURRENT_TORCH_CUDA} 라 이 드라이버에선 못 돈다 -> cu129 로 재설치"
    CURRENT_VLLM=""
  fi
fi

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
  pip install "$WHEEL" ${EXTRA[@]+"${EXTRA[@]}"}
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