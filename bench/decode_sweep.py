#!/usr/bin/env python3
"""Streaming decode sweep: per-stream and aggregate tok/s at 1,2,4,8 streams.
decode_sweep.py PORT [TOKENS] [REPEATS] [greedy]. Times decode from first token to last,
so prefill/TTFT is reported separately and does not dilute decode tok/s."""
import json, sys, time, threading, urllib.request
PORT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
GREEDY = len(sys.argv) > 4 and sys.argv[4] == "greedy"
BASE = f"http://localhost:{PORT}"
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
PROMPTS = ["Write a detailed essay about the history of computing.",
           "Explain how a CPU pipeline works, step by step.",
           "Describe the causes and consequences of the 2008 financial crisis.",
           "Write a long story about a lighthouse keeper.",
           "Explain the theory of general relativity to a student.",
           "Describe how futures markets work, including margin and settlement.",
           "Write a guide to growing tomatoes at home.",
           "Explain how compilers turn source code into machine code."]

def one(prompt, out):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": N, "min_tokens": N, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}, **({"temperature": 0} if GREEDY else {})}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; toks = 0
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            if ch and first is None:
                delta = ch[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first = time.perf_counter()
    t1 = time.perf_counter()
    out.append((first - t0, toks, (toks - 1) / (t1 - first)))

print(f"model {MODEL}, {N} tokens/stream, {REP} repeats, {'greedy' if GREEDY else 'sampled'}")
print("streams  ttft_ms  per_stream_tok/s  aggregate_tok/s")
for c in (1, 2, 4, 8):
    rows = []
    for _ in range(REP):
        out = []; ts = [threading.Thread(target=one, args=(PROMPTS[i], out)) for i in range(c)]
        t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.perf_counter() - t0
        rows.append((sum(o[0] for o in out) / c, sum(o[2] for o in out) / c, sum(o[1] for o in out) / wall))
    ttft = sum(r[0] for r in rows) / REP; ps = sum(r[1] for r in rows) / REP; ag = sum(r[2] for r in rows) / REP
    print(f"{c:>7}  {ttft*1000:7.0f}  {ps:16.1f}  {ag:15.1f}")
