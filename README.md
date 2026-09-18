# Qwen3.8-27B (AWQ-INT4) vLLM serving scripts

Scripts to install, run, and health-check a vLLM server for
`cyankiwi/Qwen3.8-27B-AWQ-INT4`, tuned for a single **RTX 5090 (32GB)** on runpod.
These are the flags that actually worked after fighting OOM and parser issues.

## Files
- `setup_qwen.sh` — install pinned vLLM (0.29.0) and print env/GPU info.
- `start_qwen.sh` — launch `vllm serve` with the working flag set.
- `check_qwen.sh` — smoke-test a running server: model list, tool calling,
  reasoning (`thinking`) separation, a full tool round-trip, and KV-cache metrics.

## Secrets
No keys are committed. Provide them at runtime:
- `start_qwen.sh` reads the API key from `$VLLM_API_KEY`.
- `check_qwen.sh` sources a local `.secure` file (git-ignored) with:
  ```
  QWEN=<api-key>
  URL=https://<your-endpoint>
  MODEL=cyankiwi/Qwen3.8-27B-AWQ-INT4
  ```

## Usage
```bash
bash setup_qwen.sh
export VLLM_API_KEY='your-secret-key'
bash start_qwen.sh
# in another shell, with .secure present:
bash check_qwen.sh
```

## Key flags and why (RTX 5090 32GB)
The model loads under vLLM as the `qwen3_5` architecture (release name is 3.8),
so the Qwen3.5 recipe applies.
- `--tool-call-parser qwen3_coder --enable-auto-tool-choice` — function calling.
- `--reasoning-parser qwen3` — split thinking into the `reasoning` field
  (otherwise `<think>` leaks into the message content).
- `--max-num-seqs 4` — **the OOM fix**: this hybrid (Gated-DeltaNet/Mamba) model's
  state cache scales with the sequence count; 128 exhausted VRAM. Lowering it
  frees room for the KV cache.
- `--language-model-only` — skip the vision encoder (text-only), saving VRAM.
- `--max-cudagraph-capture-size 4` — avoids mamba-cache capture errors.
- `--max-model-len 65536` — model is natively 256k, but the KV budget on 32GB
  is the real limit (~115k tokens total); 64k is a comfortable single-agent value.
  Optional `--kv-cache-dtype fp8` roughly doubles the token budget.

## Note
The runpod proxy rejects Python `urllib`'s default User-Agent with HTTP 403.
Use `curl` or `requests` (both fine) — not raw `urllib`.

## Driver note (runpod)
runpod gives different GPU drivers per pod (e.g. 570/CUDA12.8 vs 580/CUDA13.0).
`setup_qwen.sh` now auto-detects the driver's CUDA version and installs the matching
torch build (cu128 for 12.x, cu130 for 13.x), so it works regardless of which pod you get.
