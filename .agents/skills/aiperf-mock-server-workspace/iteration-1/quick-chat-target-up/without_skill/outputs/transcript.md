# Transcript — Fake OpenAI-compatible LLM server (streaming), port 18011

Goal: stand up a local fake OpenAI-compatible LLM server, hit `/v1/chat/completions`
with `curl`, and get a streaming response back. Confirm it actually serves.

Solution used: the workspace's own `aiperf-mock-rs` crate — a standalone
OpenAI-compatible mock inference target (chat/completions/embeddings + real SSE).
Port forced to **18011** per the operational constraint. Server killed at the end.

---

## 1. Every shell command I ran

```bash
# Inspect the mock crate + how the port is configured
ls crates/
sed -n '1,120p' crates/mock-rs/src/main.rs
grep -nE 'port|host|default|8000|struct Config|#\[arg|#\[clap' crates/mock-rs/src/config.rs
# -> port flag: -p/--port (env MOCK_SERVER_PORT), default 8000; host default 127.0.0.1

# Build the release binary
cargo build --release -p aiperf-mock-rs
# -> Finished `release` profile [optimized] target(s)

# Start the server on the required port 18011 (backgrounded)
./target/release/aiperf-mock-rs --port 18011 --fast > .../scratchpad/mock.log 2>&1 &

# Confirm the process is alive and listening on 18011
pgrep -af 'aiperf-mock-rs.*18011'
ss -ltnp | grep 18011
# -> LISTEN 127.0.0.1:18011 users:(("aiperf-mock-rs",pid=2526255,fd=9))

# Health + model list
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:18011/health
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:18011/v1/models

# The actual ask: a STREAMING chat completion
curl -sS -N http://127.0.0.1:18011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","stream":true,"max_tokens":12,"messages":[{"role":"user","content":"Say hello in one short sentence."}]}'

# Clean up: kill the server, confirm the port is free
kill 2526255
ss -ltnp | grep 18011 || echo "port 18011 free"
pgrep -af 'aiperf-mock-rs.*18011' || echo "process gone"
```

---

## 2. Key output proving the server responded

### Health check (`GET /health`) — HTTP 200

```
{"...,"port":18011,...,"status":"healthy"}
HTTP 200
```

### Listener proof (`ss -ltnp`)

```
LISTEN 0 16384 127.0.0.1:18011 0.0.0.0:* users:(("aiperf-mock-rs",pid=2526255,fd=9))
```

### Streaming chat completion (`POST /v1/chat/completions`, `stream:true`)

Real Server-Sent Events — incremental `data:` chunks terminated by `data: [DONE]`:

```
data: {"id":"chatcmpl-102f0dba-...","object":"chat.completion.chunk","created":1783964363,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"ithin "}}]}

data: {"id":"chatcmpl-102f0dba-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"thine "}}]}

data: {"id":"chatcmpl-102f0dba-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"own "}}]}

... (12 delta chunks total) ...

data: {"id":"chatcmpl-102f0dba-...","object":"chat.completion.chunk","choices":[{"index":0,"finish_reason":"stop","delta":{"content":"ting"}}]}

data: [DONE]
```

(The generated text is drawn from the mock's Shakespeare corpus — content is
synthetic/deterministic, which is expected for a fake server. The wire format,
streaming behavior, and OpenAI schema are genuine.)

---

## 3. Did the server actually serve traffic?

**Yes.** Verified three independent ways against a live process (pid 2526255):
- `GET /health` returned HTTP 200 with `"status":"healthy"` and `"port":18011`.
- `GET /v1/models` returned a populated OpenAI-shaped model list (gpt-4o-mini, gpt-4, etc.).
- `POST /v1/chat/completions` with `stream:true` returned a real SSE stream of
  `chat.completion.chunk` deltas ending in `[DONE]`.

After testing, the server was killed and port 18011 was confirmed free
(`port 18011 free`, `process gone`).

---

## 4. Final answer to the user

Your fake OpenAI-compatible server is the in-repo `aiperf-mock-rs` crate — no
external service or API key needed. Run it and hit it like this:

```bash
# Build once
cargo build --release -p aiperf-mock-rs

# Run it (choose your port; I used 18011). --fast = minimal latency.
./target/release/aiperf-mock-rs --port 18011 --fast

# Streaming chat completion
curl -N http://127.0.0.1:18011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"Hello!"}]}'
```

You'll get a live SSE stream of `chat.completion.chunk` objects ending with
`data: [DONE]`, exactly like the real OpenAI API. It also serves `/health`,
`/v1/models`, `/v1/completions`, and `/v1/embeddings`. It accepts any `model`
id and any `Authorization` header (it's a mock — no real auth). The generated
text is synthetic (Shakespeare corpus), which is fine for wiring up and testing
a client against OpenAI's streaming contract. Convenience: `cargo run -p
aiperf-mock-rs -- --port 18011 --fast`.
