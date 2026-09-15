#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/workspace/hf-cache"
export VLLM_CACHE_ROOT="/workspace/vllm-cache"
export TORCHINDUCTOR_CACHE_DIR="/workspace/torchinductor-cache"

export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="cyankiwi/Qwen3.8-27B-AWQ-INT4"

if [ -z "${VLLM_API_KEY:-}" ]; then
  echo "VLLM_API_KEY is not set."
  echo "Example:"
  echo "  export VLLM_API_KEY='your-secret-key'"
  exit 1
fi

# RTX 5090 32GB 기준.
#
# 컨텍스트/메모리
#   --max-model-len 65536   : 모델 네이티브 256k. 먼저 64k로 올리고, 부팅 로그의
#                             "GPU KV cache size: N tokens" 가 131072 이상이면 131072로 올려도 됨.
#   --max-num-seqs 4        : 하이브리드(Mamba) 모델은 SSM 상태 캐시가 시퀀스 수에 비례.
#                             128 -> 4 로 줄여서 그 VRAM을 KV 캐시에 넘김. (기존 OOM 원인)
#   --language-model-only   : 비전 인코더 로드 스킵 (텍스트 전용). VRAM 절약.
#   --kv-cache-memory 삭제   : 위 두 개로 자리 비웠으니 gpu-memory-utilization 기준 자동 산정.
#                             다시 OOM 나면 --gpu-memory-utilization 0.88 로 내려볼 것.
#   --max-cudagraph-capture-size 4 : mamba cache 에러 나면 줄이라는 게 공식 레시피.
#
# 에이전트용 (tool calling / thinking 분리)
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder : 함수 호출 파싱
#   --reasoning-parser qwen3                                 : thinking 을 content 와 분리
#
# 옵션 (필요하면 주석 해제)
#   --kv-cache-dtype fp8    : KV 메모리 절반 -> 토큰 예산 2배. 5090(Blackwell) 지원.
#   --speculative-config '{"method":"mtp","num_speculative_tokens":1}' : 단일 세션 tok/s 향상.

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-cudagraph-capture-size 4 \
  --gpu-memory-utilization 0.92 \
  --language-model-only \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --api-key "$VLLM_API_KEY"
