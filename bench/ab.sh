#!/usr/bin/env bash
# ab.sh LABEL — boot one variant and measure it, the way every row in results/ was made.
# Env overrides pass straight through to serve.sh, so one change per run is one variable:
#   MTP_K=4 bench/ab.sh k4
# Writes results/runNN_LABEL.txt (numbers) and results/runNN_LABEL.serve.sh (the exact
# settings, re-runnable on their own). Stop any co-tenant first if you want standalone numbers.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); R=$HERE/results; mkdir -p "$R"
PORT=${PORT:-8400}; export PORT
N=$(( $(ls "$R" 2>/dev/null | grep -oE "^run[0-9]+" | sort -u | wc -l) + 1 ))
L=$(printf "run%02d_%s" "$N" "${1:?usage: ab.sh LABEL}"); O=$R/$L.txt; echo "$L"
KNOBS="IMAGE MODEL_DIR HOST PORT MAX_LEN KV_BYTES MAX_SEQS BATCHED GPU_UTIL KV_DTYPE MTP_K DRAFT_SAMPLING SPEC_EXTRA SSM_DTYPE GDN_PREFILL GDN_DECODE COMPILE_CFG EXTRA_DOCKER_ARGS EXTRA_ARGS"
{ echo "#!/usr/bin/env bash"; echo "# Snapshot of $L, $(date '+%F %T %Z'). Re-run from the repo root: bash results/$L.serve.sh"
  for v in $KNOBS; do [ -n "${!v+x}" ] && printf 'export %s=%q\n' "$v" "${!v}"; done
  echo 'exec "$(dirname "$0")/../serve.sh"'; } > "$R/$L.serve.sh"
chmod +x "$R/$L.serve.sh"
t0=$(date +%s)
"$R/$L.serve.sh" >/dev/null || { echo "boot failed after $(( $(date +%s) - t0 ))s" > "$O"; docker logs --tail 30 "${CONTAINER:-qwen38-fp4}" >> "$O" 2>&1; exit 1; }
{ echo "label=$L boot=$(( $(date +%s) - t0 ))s config=$L.serve.sh"
  docker logs "${CONTAINER:-qwen38-fp4}" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens, Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" | tail -1
  bash "$HERE/bench/bench4.sh" "$PORT" | grep -E "Run|Total"
  python3 -u "$HERE/bench/decode_sweep.py" "$PORT" 512 2
  docker logs "${CONTAINER:-qwen38-fp4}" 2>&1 | grep "SpecDecoding metrics" | tail -1 | grep -oE "Mean acceptance length: [0-9.]+|Per-position acceptance rate: [0-9., ]+"
  python3 -u "$HERE/bench/decode_sweep.py" "$PORT" 512 2 greedy
  docker logs "${CONTAINER:-qwen38-fp4}" 2>&1 | grep "SpecDecoding metrics" | tail -1 | grep -oE "Mean acceptance length: [0-9.]+|Per-position acceptance rate: [0-9., ]+"
  python3 -u "$HERE/bench/prefill_check.py" "$PORT" 8000 32000
  echo "steady: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' '); $(free -g | awk '/Mem/{print "avail "$7"G"}')"
} > "$O" 2>&1
echo done >> "$O"; cat "$O"
