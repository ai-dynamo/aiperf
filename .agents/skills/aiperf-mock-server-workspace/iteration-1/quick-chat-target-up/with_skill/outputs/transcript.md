# Mock OpenAI-Compatible Server — Transcript

Task: get a fake OpenAI-compatible LLM server running locally, hit
`/v1/chat/completions` with curl, and get a streaming response back. Confirm it
actually serves. Operational constraint: server MUST listen on port **18010**,
and I must kill any process I start.

Skill followed: `.claude/skills/aiperf-mock-server/SKILL.md` (the in-repo
`aiperf-mock-rs` Rust mock server). Release binary was pre-built at
`./target/release/aiperf-mock-rs`.

## 1. Every shell command I ran

```bash
# Confirm the pre-built binary exists, then launch in the background on port 18010.
# --fast zeros latency (instant SSE), --no-tokenizer avoids the HF network download.
# NO_PROXY set so loopback curl isn't routed through an ambient proxy.
ls -la ./target/release/aiperf-mock-rs
export NO_PROXY=127.0.0.1,localhost
./target/release/aiperf-mock-rs --fast --no-tokenizer --port 18010 > .../scratchpad/mock.log 2>&1 &
# -> launched pid 2522894

# Wait for /health, then verify liveness (health code, model list, proc alive, log tail).
for i in $(seq 1 50); do curl -sf http://127.0.0.1:18010/health >/dev/null && break; sleep 0.2; done
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:18010/health
curl -s http://127.0.0.1:18010/v1/models | head -c 400
ps -o pid,cmd -p 2522894 | tail -1

# Confirm which PID owns port 18010 (other parallel agents also had mock servers up).
ss -ltnp 2>/dev/null | grep -E ':18010|:18020'
pgrep -af aiperf-mock-rs

# The actual deliverable: a streaming chat completion via curl (SSE), plus non-streaming.
curl -sN http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"Say hello in three words."}]}' | head -20

curl -s http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","stream":false,"messages":[{"role":"user","content":"hi"}]}'

# Cleanup: kill the process I started.
kill 2522894
```

## 2. Key output proving the server responded

### Health check (HTTP 200)
```
=== health ===  -> 200
```

### Model listing (`GET /v1/models`)
```json
{"data":[{"created":1783963835,"id":"Qwen/Qwen3-0.6B","object":"model","owned_by":"aiperf-mock"},
         {"id":"black-forest-labs/FLUX.1-dev",...},
         {"id":"gpt-3.5-turbo",...},{"id":"gpt-4",...},{"id":"gpt-4o",...}]}
```

### Listener ownership (my PID owns 18010)
```
LISTEN 127.0.0.1:18010  users:(("aiperf-mock-rs",pid=2522894,fd=9))
proc: 2522894 ./target/release/aiperf-mock-rs --fast --no-tokenizer --port 18010
```

### Streaming `POST /v1/chat/completions` (`stream: true`) — real SSE
```
data: {"id":"chatcmpl-518db480-...","object":"chat.completion.chunk","created":1783963902,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Say "}}]}

data: {"id":"chatcmpl-518db480-...","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hello "}}]}

data: {"id":"chatcmpl-518db480-...","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"in t"}}]}

data: {"id":"chatcmpl-518db480-...","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hree"}}]}

data: {"id":"chatcmpl-518db480-...","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","delta":{"content":" wor"}}]}

data: [DONE]
```

### Non-streaming comparison
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],
 "created":1783963902,"id":"chatcmpl-d86334fd-...","model":"gpt-4o","object":"chat.completion",
 "usage":{"completion_tokens":1,"prompt_tokens":1,"total_tokens":2}}```

## 3. Did the server actually serve traffic?

Yes. `/health` returned HTTP 200, `/v1/models` returned the OpenAI-style model
list, and `POST /v1/chat/completions` with `stream: true` returned a valid
multi-chunk SSE stream (`chat.completion.chunk` deltas, a terminal
`finish_reason: "stop"`, and a `data: [DONE]` sentinel). A non-streaming request
also returned a well-formed `chat.completion` object with a `usage` block. `ss`
confirmed my process (pid 2522894) was the listener bound to `127.0.0.1:18010`.
The process I started was killed during cleanup.

Notes:
- Used the literal `127.0.0.1` (not `localhost`) per the skill — the mock binds
  IPv4 only and `localhost` can resolve to `::1` first.
- `--fast --no-tokenizer` gave instant, offline responses (no HF download).
- Other `aiperf-mock-rs` processes (ports 18020/18030) were running from parallel
  agents in this shared worktree; I did not touch them and only killed my own PID.

## 4. Final answer to the user

Your fake OpenAI-compatible LLM server is up and confirmed serving. I launched the
in-repo Rust mock (`aiperf-mock-rs`) on `http://127.0.0.1:18010` in fast/offline
mode:

```bash
./target/release/aiperf-mock-rs --fast --no-tokenizer --port 18010
```

Hit it with streaming curl (use `127.0.0.1`, not `localhost`):

```bash
curl -sN http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"Say hello in three words."}]}'
```

It streamed back proper `chat.completion.chunk` SSE events ending in `data: [DONE]`,
and `/health` returned 200. Any model name works (the mock echoes it). I killed the
server I started; relaunch with the command above when you need it. For realistic
latency drop `--fast`; for real token counts drop `--no-tokenizer`.
