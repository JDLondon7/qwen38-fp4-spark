# Qwen3.8-Flash-Next FP4 on one DGX Spark

**A measured serving recipe for [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) on a single DGX Spark (GB10).**
It starts from the model card's own vLLM recipe and keeps only the changes that won a one-change-per-run A/B. That came to 21 numbered runs, and every one ships here with its exact launcher and raw output.

<sub>by <a href="https://github.com/JDLondon7">JDLondon7</a> · measured 17 October 2026</sub>

|  | Model card recipe | **This recipe** |
|---|---:|---:|
| Decode, 1 stream | 34.6 tok/s | **39.7–42.6 tok/s** |
| Decode, 8 streams (aggregate) | 65.3 tok/s | **116.8 tok/s** (+79%) |
| Time to first token at 8 streams | 16.5 s (requests queue) | **0.52 s** |
| Context | 64K, as booted for the baseline | **262,144 (full native)** |
| Prefill at 125K tokens | not measured | **2,522 tok/s** |
| Host memory left at steady state | not recorded | **23 GiB**, or 13 GiB beside a ~9 GiB co-tenant |

*Sampled decode, 512 tokens per stream. Both columns use the same container image and the same benchmark, on the same machine.*

---

## Quick start

```bash
git clone https://github.com/JDLondon7/qwen38-fp4-spark && cd qwen38-fp4-spark
./download.sh        # ~96 GiB into ~/models/Qwen3.8-Flash-Next-FP4-Local (resumable)
./serve.sh           # ~10 min on the first boot (b12x autotunes), ~4 min after that
curl -s localhost:8400/v1/models
./stop.sh
```

The server speaks the OpenAI API on `127.0.0.1:8400`, with reasoning and tool calling on. To reach it from another machine, set `HOST=0.0.0.0 API_KEY=...`, or use an SSH tunnel.

You need Docker with the NVIDIA runtime, about 100 GiB of free disk, and nothing else large on the GPU while the model boots. The boot section below explains that last point.

## The recipe: five changes to the model card

All five are defaults in `serve.sh`. The run numbers point into [`results/`](results/).

**1. Fragmentation-proof allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.**
At boot the image repacks the MoE weights to W4A16, layer by layer, and the PyTorch caching allocator fragments as it goes. Host usage peaks around 92 GiB of the Spark's 121 GiB before settling at about 83 GiB. With our co-tenant resident, the card recipe was killed by the OOM killer at that peak. This variable, plus booting the model before the co-tenant, fixed it. Every run here uses it, including the baseline.

**2. A fixed KV pool sized to the context: `--kv-cache-memory-bytes 7G`.**
A fixed pool skips vLLM's memory profiling and makes the footprint the same on every boot. 7 GiB of FP8 KV holds **440,070 tokens**, 1.68 full 262K requests. The card's `--gpu-memory-utilization 0.72` still applies, but the fixed pool decides the cache size.
*Evidence:* run13 (4 GiB) measured the same speed as run07 (8 GiB) at 64K. run15 then sized the pool for full context.

**3. Three draft tokens, not four: MTP `num_speculative_tokens: 3`.**
With K=4, the fourth draft token was accepted only 12% of the time. Dropping it made every step cheaper for almost no loss of accepted tokens.
*Evidence:* run02. 8-stream aggregate went 65.3 → 79.1 tok/s, and mean acceptance rose from about 2.3 to 2.59.

**4. BF16 recurrent state on the FlashInfer/CUDA GDN kernels.**
`--mamba-ssm-cache-dtype bfloat16 --gdn-prefill-backend flashinfer --gdn-decode-kernel cuda`. The card's b12x GDN prefill kernel only accepts FP32 state (run03 fails to boot), and vLLM refuses to mix b12x and non-b12x GDN kernels, so both kernels move together. We measured the two halves separately in run11:
- The kernel swap alone, still with FP32 state, adds **+5% prefill**: 2,520 → 2,653 tok/s at 8K (run08 vs run11).
- BF16 state on top adds **1–7% decode**: greedy 111.4 → 119.7 tok/s at 8 streams (run11 vs run07).

It also halves the per-sequence state. With FP32 state, 8 streams queued for 13.5 s in a 4 GiB pool (run02). With BF16 state, the same pool ran 8 streams with no queue (run13).

**5. Probabilistic drafting: `"draft_sample_method": "probabilistic"`.**
The MTP head now samples its drafts instead of taking them greedily. When your requests sample (temperature above 0), the drafts agree with the target far more often: mean acceptance rose from **2.12 to 2.53**, and sampled decode gained **+8 / +11 / +13%** at 2, 4 and 8 streams (run16 vs run18). Greedy requests are unchanged, as expected.

Everything else is the card's recipe exactly: env vars, b12x linear and MoE backends, FP8 KV, `--max-num-seqs 8`, `--max-num-batched-tokens 4096`, and the compile config.

## Final numbers (run21)

These were measured beside a ~9 GiB TTS server sharing the GPU, which is how we run the box. The standalone run18 matches them within noise.

Decode sweep, 512 tokens per stream, two repeats per level ([`bench/decode_sweep.py`](bench/decode_sweep.py)):

| Streams | Sampled, aggregate tok/s | Sampled, per stream | Greedy, aggregate tok/s | TTFT, sampled |
|---:|---:|---:|---:|---:|
| 1 | 42.6 | 43.2 | 40.4 | 210 ms |
| 2 | 64.7 | 34.6 | 64.5 | 206 ms |
| 4 | 82.0 | 23.0 | 80.9 | 282 ms |
| 8 | 116.8 | 16.2 | 114.4 | 517 ms |

Prefill with unique prompts and no prefix-cache hits ([`bench/prefill_check.py`](bench/prefill_check.py)):

| Prompt tokens | Time to first token | Prefill tok/s |
|---:|---:|---:|
| 7,890 | 3.15 s | 2,503 |
| 31,409 | 12.15 s | 2,585 |
| 125,488 | 49.76 s | 2,522 |

Bench4, the quick wall-clock smoke test ([`bench/bench4.sh`](bench/bench4.sh)), against the card baseline:

| | Card baseline (run01) | This recipe (run21) |
|---|---|---|
| Time to first token, 5 requests | 331 / 198 / 196 / 261 / 203 ms | 358 / 224 / 218 / 214 / 217 ms |
| 100 tokens, three requests | 41.1 / 42.7 / 46.1 tok/s | 47.3 / 43.8 / 42.5 tok/s |
| 3 parallel requests | 58.6 tok/s | 64.9 tok/s |
| 500 tokens, two requests | 36.9 / 38.8 tok/s | **49.6 / 45.4 tok/s** |

Single-stream sampled decode varies by a few tok/s from boot to boot: 39.7 in run18, 42.6 in run21. Read it as roughly +15–23% over the baseline. The multi-stream gains are well outside that noise.

## Everything we tried

One change per run against the current best. Each `runNN_*.txt` holds the raw output and each `runNN_*.serve.sh` re-launches that exact configuration. Decode figures are sampled aggregate tok/s at 1 / 2 / 4 / 8 streams unless marked.

| Run | Change | Result | Verdict |
|---|---|---|---|
| [01](results/run01_baseline.txt) | Card recipe: K=4, 64K context, 4 GiB KV | 34.6 / 52.0 / 54.5 / 65.3; 8 streams queue (TTFT 16.5 s) | baseline |
| [02](results/run02_k3.txt) | MTP K=3 | 38.5 / 59.8 / 71.8 / 79.1; acceptance 2.59 | **kept** |
| [03](results/run03_k3_ssmbf16.txt) | BF16 state on the b12x GDN kernels | does not boot: b12x prefill needs FP32 state | rejected |
| [04](results/run04_k3_kv8.txt) | KV 8 GiB | 37.6 / 60.5 / 74.7 / 101.2; no queue at 8 | kept, later resized |
| [05](results/run05_k3_kv8_ssmbf16_gdnfi.txt) | BF16 state + FlashInfer/CUDA GDN kernels | 40.9 / 60.4 / 77.2 / 108.3 | **kept**; split into its two halves by run11 |
| [06](results/run06_fulldecodeonly.txt) | `cudagraph_mode: FULL_DECODE_ONLY` | 35.7 / 57.3 / 75.9 / 109.5: −13% at 1 stream | rejected |
| [07](results/run07_rerun_run05_ssmbf16_gdnfi.txt) | Rerun of 05, adding greedy and prefill | greedy 40.6 / 64.7 / 80.6 / 119.7; prefill 2,660 / 2,697 | confirmed |
| [08](results/run08_rerun_run04_b12xgdn.txt) | Rerun of 04, as a control | greedy 39.4 / 64.3 / 78.0 / 112.2; prefill 2,520 / 2,561 | control |
| [09](results/run09_ssmbf16_gdnfi_batched8192.txt) | `--max-num-batched-tokens 8192` | prefill flat (2,666 / 2,654), +3.9 GiB, slower boot | rejected |
| [10](results/run10_final_coresident.txt) | 07 booted beside the co-tenant | healthy, 12 GiB left | passed |
| [11](results/run11_gdnfi_fp32state.txt) | Swapped GDN kernels, FP32 state | prefill +5% vs 08; decode below 07 | both halves of 05 kept |
| [12](results/run12_final_coresident.txt) | 8 GiB KV beside the co-tenant | guard stopped the model at 4.1 GiB free while the co-tenant booted | rejected |
| [13](results/run13_winner_kv4.txt) | KV back to 4 GiB | greedy 41.3 / 68.5 / 82.4 / 120.2, 4 GiB less memory | **kept** |
| [14](results/run14_final_coresident_8400.txt) | 13 beside the co-tenant on port 8400 | healthy; a later partial sweep was aborted (noted in the file) | passed |
| [15](results/run15_fullctx_262k_kv7.txt) | 262K context, 7 GiB KV, beside the co-tenant | 440,070 tokens; prefill 2,557 / 2,576 / 2,536 at 8K / 32K / 128K; lowest free memory 4.6 GiB | **kept** |
| [16](results/run16_base_fullctx_standalone.txt) | 15 standalone (the new base) | 40.2 / 59.4 / 74.6 / 105.2; greedy 40.5 / 65.4 / 82.6 / 115.7 | base |
| [17](results/run17_blockdrop_off.txt) | `disable_eagle_block_drop` | acceptance 2.12 → 1.97; slower 32K prefill | rejected |
| [18](results/run18_draft_probabilistic.txt) | `draft_sample_method: probabilistic` | 39.7 / 64.3 / 82.7 / 118.5; acceptance 2.12 → 2.53 | **kept** |
| [19](results/run19_maxseqs4.txt) | `--max-num-seqs 4` | 8 streams queue (TTFT 12 s); prefill −23% | rejected |
| [20](results/run20_kschedule_k3to2.txt) | K=3 up to 4 streams, K=2 at 5–8 | +6% greedy at 8, but 2-stream TTFT 181 → 330 ms and lower sampled decode at 1–2 | rejected |
| [21](results/run21_final_coresident_probdraft.txt) | 18 booted beside the co-tenant | the final numbers above | **final** |

Ideas we did not port:
- **A reduced draft vocabulary.** This image already gives the MTP drafter an NVFP4 LM head (`VLLM_MTP_NVFP4_LM_HEAD=1` by default), which captures most of that saving.
- **PLE and FP8-KV compatibility patches.** The image ships both features natively; the card's env already sets `VLLM_PLE_TABLE_MEMORY=disk`.

## Memory on the GB10

The Spark's CPU and GPU share 121 GiB. Nothing stops a GPU allocation from starving the host, and the OOM killer takes whichever process it likes: often your desktop or a neighbour rather than vLLM.

| | Host memory |
|---|---|
| Qwen at steady state, 7 GiB KV | 82,980 MiB |
| Peak during boot (W4A16 repack) | ~92 GiB |
| Free at steady state, standalone | 23 GiB |
| Free at steady state, beside a 5.5 + 3.4 GiB TTS server | 13 GiB |
| Lowest free point while the co-tenant booted second | 5.0 GiB |

**Boot the model first, then anything else.** The repack peak and a co-tenant's own boot spike don't fit together. Our TTS neighbour peaks around 17.5 GiB at boot, and booting it first is what killed early attempts. Once the model is up its footprint is preallocated: runtime drift was about 1 GiB. Our order after a reboot:

```bash
docker stop my-other-gpu-service
./serve.sh                       # returns once /health is up
docker start my-other-gpu-service
```

**The memory guard.** `serve.sh` starts [`scripts/memguard.sh`](scripts/memguard.sh). It stops the model container if `MemAvailable` drops below `GUARD_GIB` (4 by default), so the model, not your desktop or co-tenant, is what goes. It fired once across the 21 runs, in run12, which is why the pool is 7 GiB and not 8. Set `GUARD_GIB=0` to turn it off.

**The first boot is slow, once.** b12x autotunes its kernels on the first launch, about 10 minutes, and caches the result in `~/.cache/b12x`. Later boots take 3.5–4 minutes. CUDA graphs cost only about 0.5 GiB.

## Configuration

Every knob is an environment variable for `serve.sh`. `DRY_RUN=1 ./serve.sh` prints the full docker command and starts nothing.

| Variable | Default | Notes |
|---|---|---|
| `MODEL_DIR` | `~/models/Qwen3.8-Flash-Next-FP4-Local` | where `download.sh` puts the checkpoint |
| `HOST` / `PORT` | `127.0.0.1` / `8400` | `0.0.0.0` exposes the API; set `API_KEY` with it |
| `API_KEY` | empty | passed to `--api-key` |
| `MAX_LEN` | `262144` | full native context |
| `KV_BYTES` | `7G` | fixed KV pool; grow it only if you have no co-tenant (8 GiB failed beside one in run12) |
| `MTP_K` | `3` | `0` turns speculative decoding off |
| `DRAFT_SAMPLING` | `probabilistic` | empty = vLLM's default |
| `SSM_DTYPE` | `bfloat16` | `float32` needs `GDN_PREFILL=` (empty) and `GDN_DECODE=b12x` |
| `GDN_PREFILL` / `GDN_DECODE` | `flashinfer` / `cuda` | |
| `MAX_SEQS` / `BATCHED` | `8` / `4096` | the card's values; both alternatives lost (runs 19, 09) |
| `GUARD_GIB` | `4` | memory-guard floor; `0` = off |
| `IMAGE` | `eugr/spark-vllm-b12x@sha256:03a9…3595` | the exact image measured; `:latest` tracks upstream |
| `CONTAINER` | `qwen38-fp4` | |
| `EXTRA_ARGS` / `EXTRA_DOCKER_ARGS` | empty | appended to `vllm serve` and `docker run` |

## Reproduce a run

```bash
MTP_K=4 bench/ab.sh k4     # boots the variant, measures it, writes results/run22_k4.{txt,serve.sh}
```

[`bench/ab.sh`](bench/ab.sh) runs Bench4, a sampled and a greedy decode sweep, the spec-decode acceptance counters, and a prefill check at 8K and 32K. Stop any co-tenant first if you want standalone numbers. The snapshots in `results/` predate `serve.sh`. Each one embeds the launcher it actually ran, binds `0.0.0.0` and mounts `~/models`, and can be re-run as-is with `bash results/runNN_*.serve.sh`.

## Test bench and honest gaps

- **Hardware and software:**
  - DGX Spark (GB10), driver 580.95.05, kernel 6.14.0-1013-nvidia;
  - image `eugr/spark-vllm-b12x` built 15 Oct 2026 (`sha256:03a95654…3595`), vLLM `0.1.dev21830+g8244d1c44`.
- **Scope of the evidence.** One machine and two repeats per level, all on short prompts apart from the prefill check. The numbers are solid for ranking configurations. Don't treat them as a spec for your box.
- **No quality evaluation was run.** Speculative decoding verifies every token against the target model. BF16 recurrent state is a precision change, though, and we have checked it only by reading outputs, not by a benchmark. If accuracy matters to you, A/B `SSM_DTYPE=float32 GDN_PREFILL= GDN_DECODE=b12x` on your own tasks.
- **No soak test.** Stability evidence is the benchmark sessions, which ran up to about 20 minutes per boot, not days of traffic.
- **`B12X_POLICY_MODE=auto` is kept from the card,** but we found nothing in this image that reads it.

## Credits

- **NVIDIA** for the checkpoint and the DGX Spark recipe on its [model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), which is our baseline.
- **eugr** and the **Local Inference Lab** community for the [`spark-vllm-b12x`](https://hub.docker.com/r/eugr/spark-vllm-b12x) image that makes this checkpoint run on a Spark.
- **Mia's AI Lab.** Their [single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) for a different checkpoint of the same model gave us several of the ideas to test: K=3, BF16 state, KV sizing and block drop. Some carried over; some didn't on this image.
- **vLLM** and **Qwen** for the engine and the model.

## License

The scripts and write-up in this repository are MIT ([`LICENSE`](LICENSE)). The model is not covered: it is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) and the Qwen license, as its model card states.
