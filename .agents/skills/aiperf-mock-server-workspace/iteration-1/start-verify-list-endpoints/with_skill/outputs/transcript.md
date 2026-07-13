# aiperf-mock-rs — start, verify liveness, list endpoints

Skill followed: `.claude/skills/aiperf-mock-server/SKILL.md` (30-second path,
adapted to the required port **18030** instead of the default 8000).

## 1. Every shell command run

```bash
# Pre-flight: confirm release binary exists and port 18030 is free
ls -la /home/anthony/nvidia/projects/aiperf/ajc/rust/target/release/aiperf-mock-rs
ss -ltnp 2>/dev/null | grep -E ':18030' || echo "port 18030 free"

# Launch (background), fast + no HF tokenizer download, on the required port
cd /home/anthony/nvidia/projects/aiperf/ajc/rust
./target/release/aiperf-mock-rs --fast --no-tokenizer --port 18030 > <scratch>/mock.log 2>&1 &

# Liveness poll — wait for /health, then verify process + listening socket + log
for i in $(seq 1 50); do
  if curl -sf http://127.0.0.1:18030/health >/dev/null 2>&1; then echo "HEALTH OK after ${i} tries"; break; fi
  sleep 0.2
done
curl -s -i http://127.0.0.1:18030/health
pgrep -af 'aiperf-mock-rs --fast'
ss -ltnp 2>/dev/null | grep ':18030'
cat <scratch>/mock.log

# Confirm the model-list route serves, then read the authoritative routes from source
curl -s http://127.0.0.1:18030/v1/models
grep -nE '\.route\(' crates/mock-rs/src/app.rs

# Cleanup: kill the process we started
kill 2525365   # (the pid of the mock we launched on 18030)
```

Note: the mock binds IPv4 only, so all curls use the literal `127.0.0.1`
(never `localhost`, which can resolve to `::1` first). Per the skill, `--fast`
zeros all latency and bypasses the scheduler/cache, and `--no-tokenizer`
avoids the Hugging Face download on first start.

## 2. Liveness-check output (proof it was really up)

`/health` returned HTTP 200:

```
HTTP/1.1 200 OK
content-type: application/json
content-length: 1542

{"config":{...,"fast":true,"host":"127.0.0.1","no_tokenizer":true,"port":18030,...},"status":"healthy"}
```

Process alive and socket actually listening on 18030:

```
2525365 ./target/release/aiperf-mock-rs --fast --no-tokenizer --port 18030
LISTEN 0  16384  127.0.0.1:18030  0.0.0.0:*  users:(("aiperf-mock-rs",pid=2525365,fd=9))
```

Startup log (clean bind, no error/exit):

```
INFO Starting AIPerf Mock Server (Rust) host=127.0.0.1 port=18030 fast=true workers=32
INFO Listening addr=127.0.0.1:18030 backlog=16384
```

`/v1/models` also served a real OpenAI-style list (Qwen/Qwen3-0.6B, gpt-4o,
text-embedding-3-small, black-forest-labs/FLUX.1-dev, etc.), confirming the
routes work, not just `/health`.

(There was an unrelated second mock owned by another agent on port 18010; left
untouched. Only the pid we started, 2525365 on 18030, was killed.)

## 3. Endpoints exposed (authoritative — from `crates/mock-rs/src/app.rs`)

Meta / model listing:
- `GET  /`                       — root info
- `GET  /health`                 — health probe (200 = live)
- `GET  /v1/models`              — OpenAI model list
- `GET  /v1/models/{id}`         — single model

LLM inference (real SSE when `stream: true`):
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

Rerank:
- `POST /v1/ranking`             — NIM
- `POST /rerank`                 — HF TEI
- `POST /v2/rerank`              — Cohere

TGI:
- `POST /generate`
- `POST /generate_stream`

Images / multimodal / RAG:
- `POST /v1/images/generations`
- `POST /v1/image/infer`
- `POST /v1/custom-multimodal`
- `POST /rag/api/prompt`         — Solido RAG

Telemetry / metrics scrapers:
- `GET  /metrics`                — aiperf-mock's own
- `GET  /vllm/metrics`
- `GET  /sglang/metrics`
- `GET  /trtllm/metrics`
- `GET  /dynamo_frontend/metrics`
- `GET  /dynamo_component/prefill/metrics`
- `GET  /dynamo_component/decode/metrics`
- `GET  /dcgm1/metrics`          — synthetic DCGM GPU 1
- `GET  /dcgm2/metrics`          — synthetic DCGM GPU 2

25 routes total.

## 4. Final answer given to the user

The Rust mock inference server (`aiperf-mock-rs`) is up and verified live on
`http://127.0.0.1:18030` — `/health` returns 200 `{"status":"healthy"}`, the
process is running (pid 2525365) with a listening socket on 127.0.0.1:18030,
and `/v1/models` serves a real model list. I launched it with
`--fast --no-tokenizer` (instant latency, no HF download) and stopped it when
done.

It exposes 25 HTTP endpoints (all registered in `crates/mock-rs/src/app.rs`):

- Model listing: `GET /`, `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`
- LLM: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`
- Rerank: `POST /v1/ranking` (NIM), `POST /rerank` (HF TEI), `POST /v2/rerank` (Cohere)
- TGI: `POST /generate`, `POST /generate_stream`
- Images / multimodal / RAG: `POST /v1/images/generations`, `POST /v1/image/infer`, `POST /v1/custom-multimodal`, `POST /rag/api/prompt`
- Telemetry: `GET /metrics`, `/vllm/metrics`, `/sglang/metrics`, `/trtllm/metrics`, `/dynamo_frontend/metrics`, `/dynamo_component/prefill/metrics`, `/dynamo_component/decode/metrics`, `/dcgm1/metrics`, `/dcgm2/metrics`

Use `127.0.0.1`, not `localhost` — the mock binds IPv4 only.
