# Start, verify, and list endpoints — aiperf-mock-rs (Rust mock inference server)

Repo: `/home/anthony/nvidia/projects/aiperf/ajc/rust`
Constraint: server must listen on port **18031**; kill any started process before finishing.

## 1. Shell commands run

Discovery (locate crate, port flag, and routes):

```bash
# Locate the mock crate and its sources
ls crates/                                   # crate is at crates/mock-rs (package name: aiperf-mock-rs)
find . -maxdepth 4 -iname "*mock*" -type d | grep -v target
cat crates/mock-rs/Cargo.toml                # [[bin]] name = "aiperf-mock-rs"

# Port flag lives in config.rs: -p/--port, env MOCK_SERVER_PORT, default 8000; --host default 127.0.0.1
grep -n "port\|host\|struct MockServerConfig\|#\[arg\|default" crates/mock-rs/src/config.rs

# Routes registered in app.rs::build_router
grep -rn "\.route(" crates/mock-rs/src/*.rs
sed -n '14,65p' crates/mock-rs/src/app.rs
```

Build:

```bash
cargo build -p aiperf-mock-rs --bin aiperf-mock-rs
# -> Finished `dev` profile ... (compiled aiperf-mock-rs v0.0.0)
```

Start on port 18031 (background), logging to file:

```bash
MOCK_SERVER_PORT=18031 ./target/debug/aiperf-mock-rs --port 18031 --host 127.0.0.1 \
  > .tmp/mock-server-18031.log 2>&1 &
```

Liveness verification:

```bash
# process + listening socket
pgrep -af aiperf-mock-rs
ss -ltnp | grep 18031

# real HTTP requests
curl -sS -w "\nHTTP_STATUS:%{http_code}\n" http://127.0.0.1:18031/health
curl -sS -w "\nHTTP_STATUS:%{http_code}\n" http://127.0.0.1:18031/
curl -sS -w "\nHTTP_STATUS:%{http_code}\n" http://127.0.0.1:18031/v1/models
curl -sS -w "\nHTTP_STATUS:%{http_code}\n" http://127.0.0.1:18031/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

Teardown:

```bash
kill <pid-of-aiperf-mock-rs-on-18031>   # pid 2554411 in this run
```

## 2. Liveness-check output

Server startup log (`.tmp/mock-server-18031.log`):

```
INFO Starting AIPerf Mock Server (Rust) host=127.0.0.1 port=18031 fast=false workers=32
INFO Corpus loaded: 1151386 tokens from .../crates/mock-rs/../../src/aiperf/dataset/generator/assets/shakespeare.txt
INFO Listening addr=127.0.0.1:18031 backlog=16384
```

Process + socket:

```
2554411 ./target/debug/aiperf-mock-rs --port 18031 --host 127.0.0.1
LISTEN 0  16384  127.0.0.1:18031  0.0.0.0:*  users:(("aiperf-mock-rs",pid=2554411,fd=9))
```

HTTP responses (proves it actually serves, not just that the command returned):

- `GET /health` -> `HTTP_STATUS:200`, body `{"config":{...},"status":"healthy"}`
- `GET /` -> `HTTP_STATUS:200`, body `{...,"message":"AIPerf Mock Server","version":"2.0.0"}`
- `GET /v1/models` -> `HTTP_STATUS:200`, returns 10 models (Qwen/Qwen3-0.6B, gpt-4o, text-embedding-3-small, FLUX.1-dev, ...)
- `POST /v1/chat/completions` -> `HTTP_STATUS:200`, body:
  `{"choices":[{"finish_reason":"length","message":{"content":"le and thine image d","role":"assistant"}}],..."usage":{"completion_tokens":5,"prompt_tokens":1,"total_tokens":6}}`

Verdict: server is genuinely UP — it accepted TCP connections and returned valid HTTP 200 responses, including a real generated chat completion.

## 3. HTTP endpoints exposed

Source of truth: `crates/mock-rs/src/app.rs::build_router` (24 routes).

Core / OpenAI-compatible:
- `GET  /` — root info (server name + version + effective config)
- `GET  /health` — health check (status + config)
- `GET  /v1/models` — list models
- `GET  /v1/models/{id}` — get a single model
- `POST /v1/chat/completions` — OpenAI chat completions (streaming + non-streaming)
- `POST /v1/completions` — OpenAI text completions
- `POST /v1/embeddings` — OpenAI embeddings

Rerank / ranking:
- `POST /v1/ranking` — NVIDIA NIM ranking
- `POST /rerank` — HuggingFace TEI rerank
- `POST /v2/rerank` — Cohere rerank

TGI (Text Generation Inference):
- `POST /generate` — TGI generate
- `POST /generate_stream` — TGI streaming generate

Image / multimodal / RAG:
- `POST /v1/images/generations` — image generation
- `POST /v1/image/infer` — image retrieval
- `POST /v1/custom-multimodal` — custom multimodal
- `POST /rag/api/prompt` — Solido RAG

Metrics (Prometheus/OpenMetrics + DCGM):
- `GET  /metrics` — aiperf-mock metrics
- `GET  /vllm/metrics` — vLLM-dialect metrics
- `GET  /sglang/metrics` — SGLang-dialect metrics
- `GET  /trtllm/metrics` — TRT-LLM-dialect metrics
- `GET  /dynamo_frontend/metrics` — Dynamo frontend metrics
- `GET  /dynamo_component/prefill/metrics` — Dynamo prefill component metrics
- `GET  /dynamo_component/decode/metrics` — Dynamo decode component metrics
- `GET  /dcgm1/metrics` — DCGM GPU telemetry (exporter 1)
- `GET  /dcgm2/metrics` — DCGM GPU telemetry (exporter 2)

## 4. Final answer to the user

The Rust mock inference server (`aiperf-mock-rs`, crate `crates/mock-rs`) is up and verified on `http://127.0.0.1:18031`. I confirmed liveness beyond the command returning: the process is listening on 127.0.0.1:18031, and live HTTP requests returned `200` — `/health` reports `"status":"healthy"`, `/v1/models` lists 10 models, and a `POST /v1/chat/completions` actually generated tokens.

It exposes 24 HTTP endpoints:

- Core/OpenAI: `GET /`, `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`
- Rerank/ranking: `POST /v1/ranking` (NIM), `POST /rerank` (HF TEI), `POST /v2/rerank` (Cohere)
- TGI: `POST /generate`, `POST /generate_stream`
- Image/multimodal/RAG: `POST /v1/images/generations`, `POST /v1/image/infer`, `POST /v1/custom-multimodal`, `POST /rag/api/prompt`
- Metrics: `GET /metrics`, `GET /vllm/metrics`, `GET /sglang/metrics`, `GET /trtllm/metrics`, `GET /dynamo_frontend/metrics`, `GET /dynamo_component/prefill/metrics`, `GET /dynamo_component/decode/metrics`, `GET /dcgm1/metrics`, `GET /dcgm2/metrics`

(The server started for this check was stopped afterward.)
