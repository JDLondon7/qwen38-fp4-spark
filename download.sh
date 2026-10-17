#!/usr/bin/env bash
# Fetch the checkpoint (~96 GiB) into MODEL_DIR. Resumable: re-run it after an interruption.
# If Hugging Face asks for it, accept the model terms in the browser and export HF_TOKEN first.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-FP4-Local}
mkdir -p "$MODEL_DIR"
free_gib=$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc 0-9)
[ "$free_gib" -ge 100 ] || echo "warning: ${free_gib} GiB free at $MODEL_DIR; the checkpoint needs about 96 GiB" >&2
if command -v hf >/dev/null; then
  hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir "$MODEL_DIR"
elif command -v huggingface-cli >/dev/null; then
  huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir "$MODEL_DIR"
else
  echo "install the Hugging Face CLI first: pip install -U huggingface_hub" >&2; exit 1
fi
echo "checkpoint ready at $MODEL_DIR"
