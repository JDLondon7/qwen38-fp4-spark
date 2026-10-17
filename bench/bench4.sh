#!/usr/bin/env bash
# bench4.sh [PORT] — a quick four-part smoke benchmark against an OpenAI-compatible server:
# TTFT (5 tiny requests), throughput (3 x 100 tokens), 3 parallel requests, and 2 x 500 tokens.
# Wall-clock numbers include HTTP and prefill, so they read lower than decode_sweep.py's.
set -euo pipefail
PORT=${1:-8400}; BASE="http://localhost:$PORT"
for t in curl jq bc; do command -v $t >/dev/null || { echo "bench4.sh needs $t" >&2; exit 1; }; done
MODEL=$(curl -fsS "$BASE/v1/models" | jq -r ".data[0].id")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
ask() { curl -s "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2}" -o "$3"; }
now() { date +%s%3N; }

echo "4-benchmark suite, port $PORT, model $MODEL"
for i in 1 2 3; do ask "Hello" 10 /dev/null; done

echo; echo "=== 1. TTFT ==="
for i in 1 2 3 4 5; do s=$(now); ask "Say hello" 5 /dev/null; echo "Run $i: $(( $(now) - s ))ms"; done

echo; echo "=== 2. Throughput (100 tokens) ==="
for i in 1 2 3; do
  s=$(now); ask "Write a short paragraph about artificial intelligence" 100 "$TMP/tp$i.json"; e=$(( $(now) - s ))
  n=$(jq ".usage.completion_tokens" "$TMP/tp$i.json"); echo "Run $i: $n tokens in ${e}ms = $(echo "scale=2; $n*1000/$e" | bc) tok/s"
done

echo; echo "=== 3. Concurrent (3 parallel) ==="
s=$(now)
ask "Explain quantum computing briefly" 60 "$TMP/c1.json" &
ask "What is machine learning?" 60 "$TMP/c2.json" &
ask "Describe neural networks" 60 "$TMP/c3.json" &
wait; e=$(( $(now) - s ))
n=$(( $(jq ".usage.completion_tokens" "$TMP/c1.json") + $(jq ".usage.completion_tokens" "$TMP/c2.json") + $(jq ".usage.completion_tokens" "$TMP/c3.json") ))
echo "Total: $n tokens in ${e}ms = $(echo "scale=2; $n*1000/$e" | bc) tok/s aggregate"

echo; echo "=== 4. Long answer (500 tokens) ==="
for i in 1 2; do
  s=$(now); ask "Write a detailed essay about the history of computing, covering early mechanical calculators, ENIAC, the transistor revolution, personal computers, and modern AI. Include specific dates and names." 500 "$TMP/ctx$i.json"; e=$(( $(now) - s ))
  n=$(jq ".usage.completion_tokens" "$TMP/ctx$i.json"); echo "Run $i: $n tokens in ${e}ms = $(echo "scale=2; $n*1000/$e" | bc) tok/s"
done
