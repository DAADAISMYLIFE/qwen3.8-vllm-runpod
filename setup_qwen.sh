#!/usr/bin/env bash
# Ollama 로 Qwen3.8-27B 를 올리는 설치 스크립트 (runpod).
# vLLM 은 0.12 이후 휠이 전부 CUDA 12.9/13.0 빌드라 드라이버 570(CUDA 12.8) 파드에서 못 돌지만
# (Marlin 커널 PTX JIT 에서 죽음), Ollama 는 CUDA 12 런타임을 내장해 12.x/13.x 어느 파드든 돈다.
# vLLM 경로는 setup_vllm.sh / start_vllm.sh 에 남겨뒀다 (CUDA 13.0 파드 전용).
set -euo pipefail

MODEL="${MODEL:-qwen3.8:27b}"     # q4_K_M 18GB, tools+thinking. 단일세션 속도 우선이면 qwen3.8:27b-mtp-q4_K_M
export OLLAMA_MODELS="/workspace/ollama"
mkdir -p "$OLLAMA_MODELS"

echo "[1/3] Environment"
nvidia-smi || true
CUDA_DRV="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
echo "드라이버 CUDA=${CUDA_DRV:-unknown} (Ollama 는 12.x/13.x 모두 OK)"

# 모델 매니페스트가 이미 있으면 디스크 점검 생략. 없으면 18GB + 여유 필요.
MANIFEST="$OLLAMA_MODELS/manifests/registry.ollama.ai/library/${MODEL%%:*}/${MODEL#*:}"
NEED_GB=22
AVAIL_GB="$(df -BG --output=avail "$OLLAMA_MODELS" | tail -1 | tr -dc '0-9')"
if [ -f "$MANIFEST" ]; then
  echo "모델 캐시 있음($MODEL) -> 디스크 점검 생략"
elif [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  echo "=================================================================="
  echo "  $OLLAMA_MODELS 여유 ${AVAIL_GB}GB < 필요 ${NEED_GB}GB. 모델(18GB)을 못 받는다."
  echo "  → runpod 파드 Edit 에서 디스크를 늘려라."
  echo "=================================================================="
  exit 1
else
  echo "디스크 여유 ${AVAIL_GB:-?}GB (>=${NEED_GB} OK)"
fi

echo
echo "[2/3] Ollama"
if command -v ollama >/dev/null 2>&1; then
  echo "ollama already installed: $(ollama --version 2>/dev/null | head -1)"
else
  # 컨테이너엔 systemd 가 없어서 설치 스크립트가 바이너리만 깐다(서비스 등록은 건너뜀). 드라이버는 건드리지 않는다.
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo
echo "[3/3] Model pull: $MODEL"
# pull 은 서버가 떠 있어야 한다. start_qwen.sh 가 8000 에 떠 있으면 그걸 쓰고, 아니면 임시 서버를 띄웠다 내린다.
if curl -s --max-time 2 http://127.0.0.1:8000/api/version >/dev/null 2>&1; then
  HOST="127.0.0.1:8000"; TMP_PID=""
else
  HOST="127.0.0.1:11435"
  OLLAMA_HOST="$HOST" ollama serve >/tmp/ollama-setup.log 2>&1 &
  TMP_PID=$!
  for _ in $(seq 1 30); do
    curl -s --max-time 2 "http://$HOST/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
fi
OLLAMA_HOST="$HOST" ollama pull "$MODEL"
OLLAMA_HOST="$HOST" ollama list
if [ -n "$TMP_PID" ]; then kill "$TMP_PID" 2>/dev/null || true; fi

echo
echo "Setup complete. -> bash start_qwen.sh"
