# Realistic-latency mock inference server for `aiperf profile`

Goal: stand up a local OpenAI-compatible inference server with realistic latency
(~45 ms to first token, ~8 ms per output token), verify it responds, and produce
the exact `aiperf profile` command to benchmark it.

Per the operational constraint for this run, the server listens on **port 18020**
(not the mock's default 8000). The release binary `target/release/aiperf-mock-rs`
was already built.

---

## 1. Every shell command that was run

Launch the mock server with realistic analytic latency (TTFT 45 ms, ITL 8 ms),
no HF tokenizer download, deterministic seed, on port 18020:

```bash
./target/release/aiperf-mock-rs --no-tokenizer --port 18020 \
  --ttft 45 --itl 8 --random-seed 1 \
  > scratchpad/mock.log 2>&1 &
```

Wait for readiness, then confirm health + model list:

```bash
until curl -sf http://127.0.0.1:18020/health >/dev/null; do sleep 0.2; done
curl -sf http://127.0.0.1:18020/health && echo " OK"
curl -s http://127.0.0.1:18020/v1/models | head -c 400
```

Confirm the server process is alive and time a real (non-streaming) chat
completion to prove the latency model:

```bash
pgrep -af 'aiperf-mock-rs --no-tokenizer --port 18020'

curl -s -w '\nHTTP %{http_code}  total=%{time_total}s\n' \
  http://127.0.0.1:18020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hello there, tell me about latency"}],"max_tokens":20}'
```

Prove the streaming (SSE) path also works:

```bash
curl -s -N -o /dev/null -w 'TTFB(first-byte)=%{time_starttransfer}s  total=%{time_total}s  HTTP %{http_code}\n' \
  http://127.0.0.1:18020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"stream me a reply"}],"max_tokens":10,"stream":true}'
```

Clean up the server when finished:

```bash
kill 2522943   # the aiperf-mock-rs server pid
```

---

## 2. Output proving the server responded

`/health` (200, echoes the active config — note `"ttft":45.0`, `"itl":8.0`,
`"port":18020`, `"no_tokenizer":true`):

```
{"config":{...,"itl":8.0,...,"no_tokenizer":true,"port":18020,...,"ttft":45.0,...},"status":"healthy"} OK
```

`/v1/models`:

```
{"data":[{"created":1783963835,"id":"Qwen/Qwen3-0.6B","object":"model","owned_by":"aiperf-mock"},
{"created":1783963835,"id":"black-forest-labs/FLUX.1-dev","object":"model","owned_by":"aiperf-mock"},
{"created":1783963835,"id":"gpt-3.5-turbo","object":"model","owned_by":"aiperf-mock"},
{"created":1783963835,"id":"gpt-4","object":"model","owned_by":"aiperf-mock"},{"created":1783963835,"id":"gpt-4o",...
```

Server process alive:

```
2522943 ./target/release/aiperf-mock-rs --no-tokenizer --port 18020 --ttft 45 --itl 8 --random-seed 1
```

Timed non-streaming chat completion — 13 completion tokens, so the expected
latency is `45 ms + (13-1)*8 ms = 141 ms`; measured **151 ms** total (matches the
45 ms TTFT + 8 ms/token model):

```
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hello there, tell me about latencyhello there, tell me a","role":"assistant"}}],
"created":1783963885,"id":"chatcmpl-ea18e5c1-...","model":"mock-model","object":"chat.completion",
"usage":{"completion_tokens":13,"prompt_tokens":8,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":21}}
HTTP 200  total=0.151554s
```

Streaming (SSE) request succeeds (HTTP 200); the server flushes the SSE preamble
immediately, and total wall time for a short reply is ~0.10 s:

```
TTFB(first-byte)=0.000362s  total=0.102005s  HTTP 200
```

Both the OpenAI non-streaming and streaming `/v1/chat/completions` paths return
200 with realistic timing.

---

## 3. Exact `aiperf profile` command

```bash
aiperf profile \
  --url http://127.0.0.1:18020 \
  --model mock-model \
  --endpoint-type chat \
  --streaming \
  --request-count 20
```

Notes:
- Use the literal `127.0.0.1` (the mock binds IPv4 only; `localhost` may resolve
  to `::1` and fail with connection-refused).
- Any model name works — the mock echoes whatever `--model` you send. Use
  `--streaming` so AIPerf measures TTFT / ITL against the SSE stream.
- Add `--concurrency N` (or `--request-rate R`) to drive load; increase
  `--request-count` for a longer run.

---

## 4. Final answer for the user

Your realistic-latency stand-in server is the in-repo Rust mock,
`aiperf-mock-rs`, running with its analytic latency model set to **45 ms TTFT**
and **8 ms per output token**. Start it (listening on 127.0.0.1:18020) with:

```bash
cargo build --release -p aiperf-mock-rs   # once; already built here
./target/release/aiperf-mock-rs --no-tokenizer --port 18020 --ttft 45 --itl 8 --random-seed 1
```

`--ttft 45` and `--itl 8` give you the 45 ms-to-first-token / 8 ms-per-token
profile; `--no-tokenizer` avoids a Hugging Face download on first start;
`--random-seed 1` makes the timing reproducible. Verify it's live with
`curl -sf http://127.0.0.1:18020/health`. A verified 13-token completion took
151 ms total, matching `45 ms + 12 x 8 ms`.

Then benchmark it with:

```bash
aiperf profile \
  --url http://127.0.0.1:18020 \
  --model mock-model \
  --endpoint-type chat \
  --streaming \
  --request-count 20
```

(For a load sweep, add `--concurrency 8` or `--request-rate 10`. If you want the
mock to also model queueing/throughput saturation instead of a fixed per-request
cost, add `--scheduler-enabled --scheduler-max-batch-size 256`.)
