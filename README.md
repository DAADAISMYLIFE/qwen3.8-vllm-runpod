# Qwen3.8-27B serving scripts (runpod, RTX 5090 32GB)

Scripts to install, run, and health-check an OpenAI-compatible server for Qwen3.8-27B on a
single **RTX 5090 (32GB)** runpod pod. Default path is **Ollama** (works on every runpod
driver); the **vLLM** path (`cyankiwi/Qwen3.8-27B-AWQ-INT4`) is kept for CUDA 13.0 pods.

## Which path
| Pod driver | Use | Why |
|---|---|---|
| any (570/CUDA 12.8 or 580/13.0) | **`setup_qwen.sh` + `start_qwen.sh` (Ollama)** | Ollama bundles its own CUDA 12 runtime, so it runs on every runpod driver. Same port 8000, same OpenAI `/v1` API, tools + thinking. |
| CUDA 13.0 only | `setup_vllm.sh` + `start_vllm.sh` (vLLM 0.29.0) | vLLM wheels since 0.12 are cu129/cu130 builds; on a 12.8 driver the Marlin AWQ kernel dies at PTX JIT (`cudaErrorUnsupportedPtxVersion`). Verified, not theoretical. |

## Files
- `setup_qwen.sh` — install Ollama, pull `qwen3.8:27b` (q4_K_M, 18 GB) into `/workspace/ollama`.
  Checks free disk first.
- `start_qwen.sh` — `ollama serve` on `0.0.0.0:8000` with 64k context, flash attention,
  q8_0 KV cache, model kept loaded; warms the model up before reporting ready.
- `check_qwen.sh` — smoke-test a running server (vLLM or Ollama): model list, tool calling,
  reasoning (`thinking`) separation, a full tool round-trip, and (vLLM only) KV-cache metrics.
- `setup_vllm.sh` / `start_vllm.sh` — the vLLM path, kept for CUDA 13.0 pods. `start_vllm.sh`
  holds the flag set that worked after fighting OOM and parser issues (see "Key flags").

## Secrets
- Ollama has **no API-key auth**. Anyone with the runpod proxy URL can call it. For anything
  beyond testing, expose the port as TCP and reach it over an SSH tunnel.
- `start_vllm.sh` reads the API key from `$VLLM_API_KEY`.
- `check_qwen.sh` sources a local `.secure` file (git-ignored) with:
  ```
  QWEN=<api-key>            # ignored by Ollama, required by vLLM
  URL=https://<your-endpoint>
  MODEL=qwen3.8:27b         # or cyankiwi/Qwen3.8-27B-AWQ-INT4 for vLLM
  ```

## Usage (Ollama, any pod)
```bash
bash setup_qwen.sh
bash start_qwen.sh
# in another shell, with .secure present (MODEL=qwen3.8:27b):
bash check_qwen.sh
```
`MODEL=qwen3.8:27b-mtp-q4_K_M bash setup_qwen.sh` pulls the MTP variant instead
(speculative decoding, faster single-stream); use the same `MODEL=` on `start_qwen.sh`.

## Usage (vLLM, CUDA 13.0 pod only)
```bash
bash setup_vllm.sh
export VLLM_API_KEY='your-secret-key'
bash start_vllm.sh
```

## Key flags and why (vLLM, RTX 5090 32GB)
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
runpod gives different GPU drivers per pod (e.g. 570/CUDA 12.8 vs 580/CUDA 13.0), and the
driver cannot be changed from inside the container. **You need a CUDA 13.0 pod.** When
deploying, open *Additional Filters* and set CUDA Version to 13.0.

Why there is no 12.x path: vLLM 0.29.0 pins `torch==2.13.0`, which only ships cu129/cu130
builds, and vLLM itself only publishes cu129/cu130 wheels (no cu128 anywhere for 0.26+).
Tested on a 570/12.8 pod (RTX 5090):
- PyPI default (cu130) → `NVIDIA driver too old`.
- `+cu129` wheel → torch imports and `cuda.is_available()` is True, but model load dies in
  the Marlin kernel with `the provided PTX was compiled with an unsupported toolchain`
  (`cudaErrorUnsupportedPtxVersion`). CUDA minor-version compatibility covers SASS only;
  PTX JIT needs a driver at least as new as the toolchain, so this cannot be worked around.

`setup_vllm.sh` reads the driver's CUDA version from `nvidia-smi` and stops early on 12.x.

## Disk note (runpod)
The model's safetensors total **21 GB**. runpod's default 20 GB `/workspace` volume is too
small and the download dies with `No space left on device`. Set the pod's Volume Disk to
**60 GB or more** (it can only be grown; data is kept, the pod restarts).
`setup_vllm.sh` checks free space under `HF_HOME` (needs 25 GB) and `setup_qwen.sh` under
`OLLAMA_MODELS` (needs 22 GB); both skip the check when the model is already cached.
