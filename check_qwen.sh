#!/usr/bin/env bash
# vLLM 서버 상태 검증 스크립트 (로컬 WSL에서 실행)
# 사용법: bash qwen3.8/check_qwen.sh
# 필요: 프로젝트 루트의 .secure (QWEN=키, URL=주소, MODEL=모델명)
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.secure; set +a
URL="${URL%/}"   # 끝 슬래시 제거: Ollama(Gin) 는 //v1/... 를 307 으로 돌린다

# runpod 프록시가 python urllib 기본 User-Agent 를 403 으로 막음 -> curl 사용
req() {  # req <json-body>
  curl -s --max-time 180 "$URL/v1/chat/completions" \
    -H "Authorization: Bearer $QWEN" -H "Content-Type: application/json" -d "$1"
}

echo "=== [1] 서버 생존 + 컨텍스트 길이 ==="
curl -s --max-time 20 -H "Authorization: Bearer $QWEN" "$URL/v1/models" \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print('models:',[m['id'] for m in d]); print('max_model_len:',d[0].get('max_model_len','(vLLM 전용 필드, Ollama 는 OLLAMA_CONTEXT_LENGTH 로 설정)'))"

echo; echo "=== [2] tool calling: 함수 호출을 뽑아내는가 ==="
req "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"서울 날씨 어때? 도구를 써서 알려줘.\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"도시의 현재 날씨를 조회\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
  \"tool_choice\":\"auto\",\"max_tokens\":300,
  \"chat_template_kwargs\":{\"enable_thinking\":false}
}" | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
tc=m.get('tool_calls')
print('PASS ->',tc[0]['function']['name'],tc[0]['function']['arguments']) if tc else print('FAIL: tool_calls 없음. content=',(m.get('content') or '')[:200])"

echo; echo "=== [3] thinking 분리: reasoning 필드로 빠지고 content 에 <think> 가 없는가 ==="
req "{
  \"model\":\"$MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"27 곱하기 43은? 답만 한 줄로.\"}],
  \"max_tokens\":800,
  \"chat_template_kwargs\":{\"enable_thinking\":true}
}" | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
r=m.get('reasoning') or m.get('reasoning_content') or ''
c=m.get('content') or ''
ok = bool(r) and '<think>' not in c and '</think>' not in c
print('PASS' if ok else 'FAIL', '| reasoning 길이:',len(r),'| content:',repr(c.strip()[:80]))"

echo; echo "=== [4] tool 왕복: 함수 결과를 먹이면 자연어로 답하는가 ==="
req "{
  \"model\":\"$MODEL\",
  \"messages\":[
    {\"role\":\"user\",\"content\":\"서울 날씨 어때?\"},
    {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \\\"서울\\\"}\"}}]},
    {\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"{\\\"temp_c\\\": 7, \\\"condition\\\": \\\"맑음\\\"}\"}
  ],
  \"max_tokens\":200,
  \"chat_template_kwargs\":{\"enable_thinking\":false}
}" | python3 -c "
import json,sys
c=(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')
print('PASS' if ('7' in c and '맑' in c) else 'FAIL','->',c.strip()[:120])"

echo; echo "=== [5] KV 캐시 사용률 (vLLM metrics; Ollama 는 없음) ==="
{ curl -s --max-time 20 -H "Authorization: Bearer $QWEN" "$URL/metrics" \
  | grep -E '^vllm:kv_cache_usage_perc|kv_cache_size_tokens' | sed -E 's/.*kv_cache_size_tokens="([0-9]+)".*/kv_cache_size_tokens=\1/' | head -2; } || true
echo "(비어 있으면 Ollama 서버)"
