# Results

One file pair per run, numbered in the order they ran on 17 October 2026:

- `runNN_LABEL.txt`: the raw output. Bench4, then the sampled decode sweep and its spec-decode acceptance, the greedy sweep and its acceptance, then the prefill check. Co-resident runs (10, 12, 14, 15, 21) start with boot times and the lowest free memory seen.
- `runNN_LABEL.serve.sh`: the exact launcher that run used, runnable as-is.
- `runNN_LABEL.docker`: from run04 on, the resolved container command and environment from `docker inspect`.

Runs 01–04 were backfilled into the snapshot format from the settings they ran with. Run04's sweep was re-run against the same live server after its first output was lost. Run14's partial co-resident completion was aborted to free the server; run15 and run21 supersede it.

The verdict for each run is in the [main README](../README.md#everything-we-tried).
