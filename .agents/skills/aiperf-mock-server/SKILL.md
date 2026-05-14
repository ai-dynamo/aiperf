---
name: aiperf-mock-server
description: Use when an aiperf review, reproduction, or test scenario needs the in-repo OpenAI-compatible mock server running on a random localhost port. Boots tests/aiperf_mock_server on a free port, waits for /health to return ok, returns the URL + PID + log path + exact mock flags, exposes a teardown path. Other aiperf-* skills (aiperf-code-review reproductions, aiperf-correctness-testing, aiperf-adversarial-testing) compose with this; do NOT roll a one-off mock launch when this skill exists.
---

# AIPerf Mock Server

Bring up the in-repo `aiperf-mock-server` on a random localhost port, confirm readiness, hand the URL + PID + log path + flag record to the caller, and let the caller tear it down when done.

## When to use

- A skill needs to run the `aiperf` CLI against a mock backend (no real LLM server available, or determinism required).
- Reproducing a finding from `aiperf-code-review` requires hitting an OpenAI-compatible endpoint.
- Running `aiperf-correctness-testing` or `aiperf-adversarial-testing`.

## When NOT to use

- The user already has a real inference server running and asked to benchmark THAT — don't substitute the mock.
- The task is read-only static analysis with no network calls.

## Exemption — adversarial testing

`aiperf-adversarial-testing` invokes this skill ONCE PER SCENARIO with different mock flags (error-rate, latency, etc.) because per-scenario state isolation is the whole point. That's not "rolling your own launch" — that's composition. Other callers that need a single shared mock across a workload should invoke once and reuse.

## Pre-requisites

The mock server must be installed. It's part of `make first-time-setup`. If you're inside a workspace created by `aiperf-worktree` and setup succeeded, the binary is on `$PATH` of the workspace's venv.

Quick check:

```bash
which aiperf-mock-server || python -m aiperf_mock_server --help >/dev/null
```

If neither works, run `make install-mock-server` in the workspace before continuing. If that fails, surface the error — do not fall back to importing the package manually.

## Steps

### 1. Pick a free port

Don't hard-code `8000` — collisions with the user's other servers are a known gotcha. Bind-test a port:

```bash
PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
```

This races (the port can be claimed between `s.close()` and the server bind), but in practice it's reliable on a dev machine. If the bind in step 2 fails, retry once with a fresh port before erroring.

### 2. Launch on the chosen port

Run in the background with logs captured to a file the caller can grep:

```bash
LOG="$(mktemp -t aiperf-mock-XXXXXX.log)"

aiperf-mock-server --host 127.0.0.1 --port "$PORT" --fast \
  >"$LOG" 2>&1 &
PID=$!
```

Flag notes:
- `--fast` zeroes TTFT/ITL. Use for reproduction + correctness — repro doesn't need realistic latency, and determinism matters more.
- For `aiperf-adversarial-testing`, override with `--ttft <ms> --itl <ms> --error-rate <pct> --random-seed 42` per that skill's scenario matrix.
- For Prometheus / DCGM scenarios, no extra flags needed — endpoints are always on.

If the user invoked something exotic (`--workers 4`, custom backend), thread it through; the skill is a launcher, not a flag-policy gate.

### 3. Wait for `/health`

The server takes ~1-3s to come up. Poll, don't sleep blindly:

```bash
URL="http://127.0.0.1:${PORT}"

for i in $(seq 1 30); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "mock server died during startup; log: $LOG" >&2
    cat "$LOG" >&2
    exit 1
  fi
  sleep 0.2
done

curl -fsS "$URL/health" || { echo "health check failed after 6s; log: $LOG" >&2; exit 1; }
```

Six seconds covers cold-start on dev hardware; if it blows past, the failure mode is "import-time crash" or "port collision," not "slow startup" — let the error surface, don't extend the timeout.

### 4. Report back

```
MOCK_URL=http://127.0.0.1:<port>
MOCK_PID=<pid>
MOCK_LOG=<path>
MOCK_FLAGS=<exact flags passed, for the reproducibility record>
```

Calling skills point `aiperf` at `$MOCK_URL` (`--url $MOCK_URL` or `OPENAI_BASE_URL=$MOCK_URL/v1`).

### 5. Teardown

The calling skill MUST tear down when done — otherwise the user accumulates stale uvicorn processes across sessions:

```bash
kill "$MOCK_PID" 2>/dev/null || true
wait "$MOCK_PID" 2>/dev/null || true
```

If the caller might exit before reaching its teardown line (uncaught error, user Ctrl-C), set a `trap`:

```bash
trap 'kill $MOCK_PID 2>/dev/null; wait $MOCK_PID 2>/dev/null' EXIT
```

Append the log to the caller's artifact directory before deleting (only delete after the copy succeeds — otherwise an unset `$ARTIFACT_DIR` makes `cp` fail and `rm` still runs, losing the log):

```bash
if [ -n "$ARTIFACT_DIR" ]; then
  cp "$LOG" "$ARTIFACT_DIR/mock-server.log" && rm "$LOG"
fi
# If $ARTIFACT_DIR is unset, leave $LOG in place — the caller is responsible for cleanup.
```

## Endpoints quick-reference

The mock implements (see `tests/aiperf_mock_server/README.md` for the full matrix):

| Path | Purpose |
|---|---|
| `/v1/chat/completions` | OpenAI chat (streaming + non-streaming) |
| `/v1/completions` | OpenAI text completion |
| `/v1/embeddings` | OpenAI embeddings (768-dim) |
| `/v1/images/generations` | OpenAI image generation |
| `/v1/images/edits` | OpenAI image edit (multipart binary) |
| `/v1/image/infer` | NIM image retrieval |
| `/v1/ranking`, `/rerank`, `/v2/rerank` | NVIDIA/HuggingFace/Cohere reranking |
| `/generate`, `/generate_stream` | HuggingFace TGI |
| `/v1/custom-multimodal` | Custom multimodal format |
| `/rag/api/prompt` | Solido RAG |
| `/health` | Liveness probe (use for readiness gating) |
| `/metrics`, `/vllm/metrics`, `/sglang/metrics`, `/trtllm/metrics` | Prometheus scrape targets |
| `/dynamo_frontend/metrics` | Dynamo frontend metrics |
| `/dynamo_component/prefill/metrics` | Dynamo prefill worker metrics |
| `/dynamo_component/decode/metrics` | Dynamo decode worker metrics |
| `/dcgm{N}/metrics` | DCGM GPU telemetry; the route is parameterized (`{N}` is an int instance ID, currently 1..2) |

When a calling skill needs a specific endpoint, point `aiperf --endpoint-type <type>` at the right one. Valid `EndpointType` enum values (see `src/aiperf/plugin/enums.pyi:288-322`): `chat`, `chat_embeddings`, `cohere_rankings`, `completions`, `embeddings`, `hf_tei_rankings`, `huggingface_generate`, `image_generation`, `image_retrieval`, `nim_embeddings`, `nim_rankings`, `responses`, `solido_rag`, `template`, `video_generation`. (There is no `rankings`, `multimodal`, or `image_edit` enum value — multimodal workloads use `chat` with multimodal-prompt inputs, or `template` with a custom endpoint path.)

## Common mistakes

- **Hard-coding port 8000.** Other dev processes (or another mock from a prior run) bind it and the launch fails opaquely. Always pick a free port.
- **Using `time.sleep(5)` instead of polling `/health`.** Flaky on cold-start machines, slow on fast ones.
- **Forgetting `--fast` during reproduction.** Default TTFT/ITL adds ~50ms per token; a 1000-token repro takes a minute longer than it needs to and the latency numbers in artifacts become noise.
- **Not capturing the log.** When a repro fails, the mock-side stderr is the first thing to read. Without `$MOCK_LOG`, you'll re-run blind.
- **Leaving the process running.** Subsequent runs collide with the leaked uvicorn; user has to `pkill aiperf-mock-server` manually. Always teardown via `trap`.
- **Pointing `aiperf` at `http://localhost:<port>` while `NO_PROXY` is unset.** A corp `HTTP_PROXY` env routes localhost through the proxy and returns 405/502. Set `NO_PROXY=127.0.0.1,localhost` in the subprocess env if any HTTP proxy variable is set.
