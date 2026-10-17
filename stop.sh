#!/usr/bin/env bash
# Stop the server gracefully (vLLM gets 30 s to release shared memory). The memory guard exits on its own.
CONTAINER=${CONTAINER:-qwen38-fp4}
docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER" && docker rm "$CONTAINER" >/dev/null
