#!/usr/bin/env bash
# Ollama 서버 기동 (OpenAI 호환 /v1 API). vLLM 때와 같은 8000 포트라 runpod 프록시 주소 그대로.
set -euo pipefail

MODEL="${MODEL:-qwen3.8:27b}"

export OLLAMA_MODELS="/workspace/ollama"
export OLLAMA_HOST="0.0.0.0:8000"
export OLLAMA_CONTEXT_LENGTH=65536   # 기본값이면 revagent 20스텝 도구출력이 잘린다. 모델 네이티브 256k.
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0     # KV 절반 -> 64k 컨텍스트가 32GB 에 들어감. 품질 문제 있으면 f16.
export OLLAMA_KEEP_ALIVE=-1          # 유휴 시 모델 언로드 금지
export OLLAMA_NUM_PARALLEL=1         # revagent 는 순차 호출. 늘리면 KV 를 그만큼 더 먹는다.

# 주의: Ollama 는 API 키 인증이 없다. runpod 프록시 주소를 아는 사람은 누구나 호출 가능.
#       .secure 의 QWEN 키는 무시되므로 필요하면 포트를 TCP 로 열고 SSH 터널로 붙어라.

ollama serve &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 60); do
  curl -s --max-time 2 http://127.0.0.1:8000/api/version >/dev/null 2>&1 && break
  sleep 1
done

# 워밍업: 지금 모델을 VRAM 에 올려 첫 요청 지연(수십 초)을 없앤다.
echo "[start] loading $MODEL ..."
curl -s --max-time 600 http://127.0.0.1:8000/api/generate \
  -d "{\"model\":\"$MODEL\",\"keep_alive\":-1}" >/dev/null
curl -s http://127.0.0.1:8000/api/ps
echo
echo "[start] ready: http://0.0.0.0:8000/v1  model=$MODEL"
wait $PID
