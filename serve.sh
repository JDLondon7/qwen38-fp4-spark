#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local on one DGX Spark (GB10).
#
# The defaults are the measured winner (results/run21). Every knob is an
# environment override, so a variant is one line:
#   MTP_K=4 ./serve.sh
#   DRY_RUN=1 ./serve.sh        # print the docker command, start nothing
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

CONTAINER=${CONTAINER:-qwen38-fp4}
# The image the results were measured on. Set IMAGE=eugr/spark-vllm-b12x:latest to track upstream.
IMAGE=${IMAGE:-eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-FP4-Local}
SERVED_NAME=${SERVED_NAME:-nvidia/Qwen3.8-Flash-Next-FP4-Local}
HOST=${HOST:-127.0.0.1}             # 0.0.0.0 exposes the API to your network; pair it with API_KEY
PORT=${PORT:-8400}
API_KEY=${API_KEY:-}

MAX_LEN=${MAX_LEN:-262144}          # full native context
KV_BYTES=${KV_BYTES:-7G}            # fixed KV pool: 440,070 tokens, 1.68 full-length requests
MAX_SEQS=${MAX_SEQS:-8}
BATCHED=${BATCHED:-4096}
GPU_UTIL=${GPU_UTIL:-0.72}          # the card's value; the fixed KV pool decides the cache size
KV_DTYPE=${KV_DTYPE:-fp8}

MTP_K=${MTP_K:-3}                   # draft tokens per step; 0 disables speculative decoding
DRAFT_SAMPLING=${DRAFT_SAMPLING:-probabilistic}   # empty = vLLM default
SPEC_EXTRA=${SPEC_EXTRA:-}          # more speculative-config JSON fields, e.g. "disable_eagle_block_drop":true

SSM_DTYPE=${SSM_DTYPE:-bfloat16}    # recurrent (GDN) state; the b12x GDN prefill kernel needs float32
GDN_PREFILL=${GDN_PREFILL:-flashinfer}
GDN_DECODE=${GDN_DECODE:-cuda}
COMPILE_CFG=${COMPILE_CFG:-'{"pass_config":{"fuse_act_quant":true}}'}

GUARD_GIB=${GUARD_GIB:-4}           # stop this container if host MemAvailable drops below this; 0 = off
WAIT=${WAIT:-1}                     # wait for /health before returning
DRY_RUN=${DRY_RUN:-0}
EXTRA_DOCKER_ARGS=${EXTRA_DOCKER_ARGS:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

[ -f "$MODEL_DIR/config.json" ] || { echo "no checkpoint at $MODEL_DIR (run ./download.sh, or set MODEL_DIR)" >&2; exit 1; }

spec="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K"
[ -n "$DRAFT_SAMPLING" ] && spec+=",\"draft_sample_method\":\"$DRAFT_SAMPLING\""
[ -n "$SPEC_EXTRA" ] && spec+=",$SPEC_EXTRA"
spec+="}"

gdn=()
[ -n "$GDN_PREFILL" ] && gdn+=(--gdn-prefill-backend "$GDN_PREFILL")
[ -n "$GDN_DECODE" ] && gdn+=(--gdn-decode-kernel "$GDN_DECODE")

docker_args=(
  run -d --name "$CONTAINER" --gpus all --ipc host --network host --shm-size=16g
  --ulimit memlock=-1 --ulimit stack=67108864
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined
  -v "$MODEL_DIR:/model:ro"
  -v "$HOME/.cache/vllm:/root/.cache/vllm"
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer"
  -v "$HOME/.cache/b12x:/root/.cache/b12x"
  -v "$HOME/.triton:/root/.triton"
  # From the model card's DGX Spark recipe:
  -e VLLM_PLE_TABLE_MEMORY=disk -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a
  -e SAFETENSORS_FAST_GPU=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1
  -e B12X_POLICY_MODE=auto -e VLLM_MXFP8_LM_HEAD=1 -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=1
  # Ours: keeps the boot-time weight repack from fragmenting its way past the host's memory.
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
# shellcheck disable=SC2206
docker_args+=($EXTRA_DOCKER_ARGS --entrypoint vllm "$IMAGE")

vllm_args=(
  serve /model --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT" --tensor-parallel-size 1 --pipeline-parallel-size 1
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
  --dtype bfloat16 --kv-cache-dtype "$KV_DTYPE" --quantization modelopt_mixed
  --block-size 16 --load-format b12x --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCHED"
  --kv-cache-memory-bytes "$KV_BYTES" --gpu-memory-utilization "$GPU_UTIL"
  --mamba-ssm-cache-dtype "$SSM_DTYPE" "${gdn[@]}"
  --linear-backend b12x --moe-backend b12x
  --no-enable-flashinfer-autotune --mm-encoder-tp-mode data
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
  --compilation-config "$COMPILE_CFG"
)
[ "$MTP_K" != 0 ] && vllm_args+=(--speculative-config "$spec")
[ -n "$API_KEY" ] && vllm_args+=(--api-key "$API_KEY")
# shellcheck disable=SC2206
vllm_args+=($EXTRA_ARGS)

if [ "$DRY_RUN" = 1 ]; then
  printf 'docker'; printf ' %q' "${docker_args[@]}" "${vllm_args[@]}"; echo
  exit 0
fi

mkdir -p "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.cache/b12x" "$HOME/.triton" "$HERE/logs"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker "${docker_args[@]}" "${vllm_args[@]}" >/dev/null
echo "started $CONTAINER on $HOST:$PORT ($IMAGE)"

if [ "$GUARD_GIB" != 0 ]; then
  setsid nohup "$HERE/scripts/memguard.sh" "$CONTAINER" "$GUARD_GIB" > "$HERE/logs/memguard.log" 2>&1 < /dev/null &
  echo "memory guard on: stops $CONTAINER below ${GUARD_GIB} GiB MemAvailable (logs/memguard.log)"
fi

[ "$WAIT" = 1 ] || exit 0
t0=$(date +%s)
echo "waiting for /health (about 4 min warm, about 10 min on the first boot while b12x autotunes)"
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then
    echo "$CONTAINER exited during boot after $(( $(date +%s) - t0 ))s; last log lines:" >&2
    docker logs --tail 30 "$CONTAINER" >&2 || true
    cat "$HERE/logs/memguard.log" >&2 2>/dev/null || true
    exit 1
  fi
  sleep 5
done
echo "healthy after $(( $(date +%s) - t0 ))s: http://$HOST:$PORT/v1"
