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

`setup_qwen.sh` reads the driver's CUDA version from `nvidia-smi` and stops early on 12.x.

## Disk note (runpod)
The model's safetensors total **21 GB**. runpod's default 20 GB `/workspace` volume is too
small and the download dies with `No space left on device`. Set the pod's Volume Disk to
**60 GB or more** (it can only be grown; data is kept, the pod restarts).
`setup_qwen.sh` checks free space under `HF_HOME` before installing and stops early if
there is less than 25 GB, unless the model is already cached.
