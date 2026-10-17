#!/bin/bash
# Snapshot of run01_baseline (backfilled 2026-10-17 01:49:34 BST from the settings it ran with). Re-run: bash run01_baseline.serve.sh
export MAX_LEN=65536
export EXTRA_ENV=-e\ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\ 
export EXTRA_ARGS=--kv-cache-memory-bytes\ 4G\ 
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local with the model card's DGX Spark recipe.
# Every knob is an env override so Mia-derived variants are one-line A/Bs.
set -euo pipefail
NAME=qwen38-fp4
PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.72}
MAX_LEN=${MAX_LEN:-262144}
MAX_SEQS=${MAX_SEQS:-8}
BATCHED=${BATCHED:-4096}
MTP_K=${MTP_K:-4}
KV_DTYPE=${KV_DTYPE:-fp8}
SSM_DTYPE=${SSM_DTYPE:-}            # Mia: bfloat16
GDN_ARGS=${GDN_ARGS:-"--gdn-decode-kernel b12x"}
COMPILE_CFG=${COMPILE_CFG:-'{"pass_config":{"fuse_act_quant":true}}'}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p ~/.cache/vllm ~/.cache/flashinfer ~/.triton ~/.cache/b12x
docker rm -f "$NAME" >/dev/null 2>&1 || true
SPEC=''
[ "$MTP_K" != 0 ] && SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}"
SSM=''
[ -n "$SSM_DTYPE" ] && SSM="--mamba-ssm-cache-dtype $SSM_DTYPE"
docker run -d --name "$NAME" --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer -v ~/.triton:/root/.triton -v ~/.cache/b12x:/root/.cache/b12x \
  -v $HOME/models:/models:ro \
  -e VLLM_PLE_TABLE_MEMORY=disk -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto -e VLLM_MXFP8_LM_HEAD=1 -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=1 $EXTRA_ENV \
  --entrypoint vllm eugr/spark-vllm-b12x:latest \
  serve /models/Qwen3.8-Flash-Next-FP4-Local --served-model-name nvidia/Qwen3.8-Flash-Next-FP4-Local \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
  --dtype bfloat16 --kv-cache-dtype "$KV_DTYPE" --quantization modelopt_mixed \
  --block-size 16 --load-format b12x --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCHED" $SPEC $SSM \
  $GDN_ARGS --linear-backend b12x --moe-backend b12x \
  --no-enable-flashinfer-autotune --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --compilation-config "$COMPILE_CFG" --gpu-memory-utilization "$GPU_UTIL" $EXTRA_ARGS
echo "started $NAME on :$PORT"
