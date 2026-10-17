#!/usr/bin/env python3
"""Prefill throughput: prefill_check.py PORT [SIZES...]. Sends a unique long prompt per size
(no prefix-cache hits), max_tokens=1, and reports prompt tokens / TTFT."""
import json, sys, time, uuid, urllib.request
PORT = sys.argv[1]; SIZES = [int(s) for s in sys.argv[2:]] or [8000, 32000]
BASE = f"http://localhost:{PORT}"
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
words = ("market price order risk position futures contract volume liquidity spread "
         "trend support resistance session open close high low bid ask fill").split()
print("target_tokens  prompt_tokens  ttft_s  prefill_tok/s")
for n in SIZES:
    tag = uuid.uuid4().hex
    text = tag + " " + " ".join(words[(i * 7 + len(tag)) % len(words)] for i in range(int(n * 0.98)))
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text + "\nSummarise in one word."}],
                       "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); r = json.load(urllib.request.urlopen(req, timeout=600)); t = time.perf_counter() - t0
    p = r["usage"]["prompt_tokens"]
    print(f"{n:>13}  {p:>13}  {t:6.2f}  {p / t:13.0f}")
