// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock server configuration - CLI args + `MOCK_SERVER_*` env vars.

use clap::Parser;
use serde::{Deserialize, Serialize};

use crate::accuracy::{AccuracyFormat, AccuracyMatch};
use crate::grpc::GrpcBehavior;
use crate::prefix_cache::EvictionPolicy;

#[derive(Debug, Clone, Parser, Serialize, Deserialize)]
#[command(name = "aiperf-mock-server", about = "AIPerf Mock Server (Rust)")]
pub struct MockServerConfig {
    #[arg(short = 'p', long, env = "MOCK_SERVER_PORT", default_value_t = 8000)]
    pub port: u16,

    #[arg(long, env = "MOCK_SERVER_HOST", default_value = "127.0.0.1")]
    pub host: String,

    /// Optional KServe Open Inference Protocol (OIP) v2 gRPC listener port. When
    /// set, a second listener serves the KServe `GRPCInferenceService`
    /// (`ModelInfer`, `ModelStreamInfer`, `ModelReady`, `ServerLive`,
    /// `ServerReady`) over h2c on `--host:<this>`. Unset means no gRPC listener.
    /// The HTTP-only L4 balancer ignores this option with `--processes > 1`.
    #[arg(long, env = "MOCK_SERVER_GRPC_PORT")]
    pub grpc_port: Option<u16>,

    /// Optional Unix-domain socket path. When set, the server binds a
    /// `UnixListener` at this path and serves the axum router over HTTP/1.1
    /// (the runner's UDS transport is HTTP/1.1-only:
    /// `transport::http/client/connection.rs` connects a `UnixStream` and
    /// negotiates h1). A stale socket file at the path is unlinked first. The
    /// TCP frontend on `--port` continues in parallel. The TCP-only L4 balancer
    /// ignores this option with `--processes > 1`.
    #[arg(long, env = "MOCK_SERVER_UDS")]
    pub uds: Option<String>,

    /// Path to a PEM certificate chain enabling the HTTPS (and, with
    /// `--grpc-port`, `grpcs`) frontend. Paired with `--tls-key`; supplying one
    /// without the other is an error. When set, accepted TCP streams are wrapped
    /// in a rustls handshake advertising ALPN `h2`+`http/1.1` before hyper sees
    /// them, so the mock is a target for AIPerf's `https://`/`grpcs://`
    /// transports. Unset (and `--tls-self-signed` unset) keeps cleartext.
    #[arg(long, env = "MOCK_SERVER_TLS_CERT")]
    pub tls_cert: Option<String>,

    /// Path to the PEM private key (PKCS#8, RSA/PKCS#1, or SEC1) matching
    /// `--tls-cert`.
    #[arg(long, env = "MOCK_SERVER_TLS_KEY")]
    pub tls_key: Option<String>,

    /// Serve HTTPS (and `grpcs`) with an in-memory self-signed certificate for
    /// `127.0.0.1`/`localhost`, generated fresh at startup. Convenient for local
    /// TLS integration runs where the client disables verification
    /// (`endpoint.ssl_verify=false`); ignored when `--tls-cert`/`--tls-key` are
    /// given.
    #[arg(
        long,
        env = "MOCK_SERVER_TLS_SELF_SIGNED",
        default_value_t = false,
        action = clap::ArgAction::SetTrue
    )]
    pub tls_self_signed: bool,
    /// When set, the KServe gRPC `ModelInfer` handler behaves as a NON-LLM
    /// embedding model (like a Triton `python`-backend embedder): it consumes the
    /// input text tensor and returns a single `FP32` embedding tensor of this
    /// dimension (shape `[1, dim]`) instead of a generated `BYTES` text output.
    /// This exercises the `kserve_v2_embeddings` STRING-in / FP32-out contract.
    /// Only unary `ModelInfer` supports embeddings.
    #[arg(long, env = "MOCK_SERVER_GRPC_EMBEDDING_DIM")]
    pub grpc_embedding_dim: Option<usize>,

    /// Output-tensor behavior for the KServe gRPC (and HTTP `/v2/.../infer`)
    /// inference handler. `auto` (the default) inspects the request's INPUT
    /// tensor names to decide what to emit — a `passages` tensor means the
    /// `kserve_v2_rankings` path (emit a numeric `scores` FP32 tensor), a
    /// `prompt` tensor with no `text_input` means the `kserve_v2_images` path
    /// (emit a `generated_image` BYTES tensor), and anything else (`text_input`,
    /// with or without an `image` tensor for `kserve_v2_vlm`) generates text and
    /// emits a `text_output` BYTES tensor. The explicit `text` / `rankings` /
    /// `images` values force one behavior regardless of the input tensors, for a
    /// single-purpose target. `grpc_embedding_dim`, when set, still overrides all
    /// of these for unary `ModelInfer` (FP32 embedding output).
    #[arg(
        long,
        env = "MOCK_SERVER_GRPC_BEHAVIOR",
        value_enum,
        default_value = "auto"
    )]
    pub grpc_behavior: GrpcBehavior,

    /// Tokio worker-thread count. `0` (the default) means auto = nproc; any
    /// explicit value, including `1`, is honored verbatim.
    #[arg(short = 'w', long, env = "MOCK_SERVER_WORKERS", default_value_t = 0)]
    pub workers: usize,

    /// Number of server processes to run behind a built-in round-robin load
    /// balancer. `1` serves in one process. `N > 1`
    /// turns this process into a lightweight L4 (TCP) round-robin balancer: it
    /// binds `--host:--port`, spawns `N` child `aiperf-mock-server` processes on
    /// internal loopback ports carrying this exact config, and splices each
    /// accepted connection to the next backend in rotation. This lets the mock
    /// use independent allocators and tokio runtimes while exposing one
    /// OpenAI-compatible frontend. `0` = auto = number of
    /// CPUs. When auto-dividing, each child's tokio `--workers` defaults to
    /// `max(1, nproc / processes)` so the total worker-thread count stays bounded.
    #[arg(long, env = "MOCK_SERVER_PROCESSES", default_value_t = 1)]
    pub processes: usize,

    /// HTTP/2 `SETTINGS_MAX_CONCURRENT_STREAMS` advertised to clients. Bounds how
    /// many requests a single h2 connection may have in flight at once; hyper's
    /// default (~200) caps concurrent-request stress tests. Raise it (e.g.
    /// `2000000`) to hold hundreds of thousands / millions of simultaneous
    /// streams on one connection. `0` leaves hyper's default untouched.
    #[arg(long, env = "MOCK_SERVER_MAX_CONCURRENT_STREAMS", default_value_t = 0)]
    pub max_concurrent_streams: u32,

    #[arg(short = 't', long, env = "MOCK_SERVER_TTFT", default_value_t = 20.0)]
    pub ttft: f64,

    #[arg(long, env = "MOCK_SERVER_ITL", default_value_t = 5.0)]
    pub itl: f64,

    /// Per-ISL-token TTFT scaling (ms). Effective TTFT =
    /// `ttft + ttft_per_isl_token_ms * prompt_token_count`, modeling prefill
    /// cost that scales with prompt length (e.g. 0.05 ~= 50ms per 1k input
    /// tokens). Default 0.0 keeps TTFT constant.
    #[arg(long, env = "MOCK_SERVER_TTFT_PER_ISL_TOKEN_MS", default_value_t = 0.0)]
    pub ttft_per_isl_token_ms: f64,

    /// TTFT concurrency penalty (ms): `ttft_ms += ttft_concurrency_quad_ms *
    /// active_inflight^2`, modeling prefill contention that grows super-linearly
    /// with load. Analytic mode only (ignored when the scheduler is enabled).
    #[arg(
        long,
        env = "MOCK_SERVER_TTFT_CONCURRENCY_QUAD_MS",
        default_value_t = 0.0
    )]
    pub ttft_concurrency_quad_ms: f64,

    /// Per-OSL-token ITL scaling (ms): `itl_ms += itl_per_osl_token_ms *
    /// osl_tokens`. Analytic mode only.
    #[arg(long, env = "MOCK_SERVER_ITL_PER_OSL_TOKEN_MS", default_value_t = 0.0)]
    pub itl_per_osl_token_ms: f64,

    /// Concurrency-linear ITL penalty (ms): `itl_ms += itl_concurrency_lin_ms *
    /// active_inflight`. Analytic mode only.
    #[arg(
        long,
        env = "MOCK_SERVER_ITL_CONCURRENCY_LIN_MS",
        default_value_t = 0.0
    )]
    pub itl_concurrency_lin_ms: f64,

    /// Lognormal TTFT jitter coefficient of variation (stddev/mean). 0 = none.
    #[arg(long, env = "MOCK_SERVER_TTFT_JITTER_CV", default_value_t = 0.0)]
    pub ttft_jitter_cv: f64,

    /// Lognormal ITL jitter coefficient of variation (stddev/mean). 0 = none.
    #[arg(long, env = "MOCK_SERVER_ITL_JITTER_CV", default_value_t = 0.0)]
    pub itl_jitter_cv: f64,

    /// Enable the step-based batched scheduler. When true, requests compete for
    /// per-step decode and prefill slots, producing a realistic throughput-vs-
    /// concurrency saturation knee. When false, the closed-form analytic
    /// TTFT/ITL model (the `*_per_*` / `*_concurrency_*` knobs) is used.
    #[arg(long, env = "MOCK_SERVER_SCHEDULER_ENABLED", default_value_t = false, action = clap::ArgAction::SetTrue)]
    pub scheduler_enabled: bool,

    /// Virtual decode-step cadence (ms). Each step admits up to
    /// `scheduler_max_batch_size` decode tokens.
    #[arg(long, env = "MOCK_SERVER_SCHEDULER_STEP_MS", default_value_t = 5.0)]
    pub scheduler_step_ms: f64,

    /// Maximum concurrent decoders served per step (the effective batch size);
    /// the saturation knee lands at concurrency ~= this value.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_MAX_BATCH_SIZE",
        default_value_t = 256
    )]
    pub scheduler_max_batch_size: usize,

    /// Maximum prefill chunks admitted per step. Lower = prefill becomes the
    /// binding constraint, producing TTFT cliffs under concurrent arrivals.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_MAX_PREFILL_CHUNKS_PER_STEP",
        default_value_t = 8
    )]
    pub scheduler_max_prefill_chunks_per_step: usize,

    /// Tokens per prefill chunk; a prompt of P tokens needs ceil(P/chunk) chunks.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_PREFILL_CHUNK_TOKENS",
        default_value_t = 512
    )]
    pub scheduler_prefill_chunk_tokens: usize,

    /// Fixed prefill chunks per request, overriding the ISL-derived count when
    /// positive. Use this to make TTFT (queue wait) independent of prompt length —
    /// matching servers where TTFT is dominated by scheduling/queueing rather
    /// than raw prefill compute. 0 = derive from ISL and chunk size.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_PREFILL_CHUNKS_PER_REQUEST",
        default_value_t = 0
    )]
    pub scheduler_prefill_chunks_per_request: usize,

    /// Per-request lognormal (mean-1) coefficient of variation applied to the
    /// prefill chunk count, so queue-wait/TTFT spreads request-to-request
    /// (TTFT CV ~= this value). 0 = every request has identical prefill work.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_PREFILL_WORK_CV",
        default_value_t = 0.0
    )]
    pub scheduler_prefill_work_cv: f64,

    /// Per-step lognormal (mean-1) coefficient of variation applied to the
    /// decode and prefill admit budgets, so the server drains at a fluctuating
    /// rate instead of a perfectly smooth one — adds realistic throughput
    /// burstiness / temporal autocorrelation to TTFT and tok/s. 0 = smooth.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_ADMIT_JITTER_CV",
        default_value_t = 0.0
    )]
    pub scheduler_admit_jitter_cv: f64,

    /// Sublinear prefill-throughput exponent. The per-step prefill chunk budget
    /// scales as `(occupancy / ref)^exponent` (floored at the base budget),
    /// where `occupancy` is the live in-flight request count (prefill + decode
    /// waiters). This models chunked prefill piggybacking on a fuller decode
    /// batch: more load means more prefill throughput, so queue-wait/TTFT grows
    /// SUBLINEARLY with concurrency (`TTFT ~ C^(1 - exponent)`). 0 = fixed rate
    /// (TTFT grows linearly with concurrency). Derivable from two trace points:
    /// `exponent = 1 - log(ttft2/ttft1) / log(c2/c1)`.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_PREFILL_THROUGHPUT_EXPONENT",
        default_value_t = 0.0
    )]
    pub scheduler_prefill_throughput_exponent: f64,

    /// Reference occupancy (in-flight request count) at which the prefill budget
    /// equals its base. 0 = use `scheduler_max_batch_size`. Only matters when
    /// `scheduler_prefill_throughput_exponent > 0`; it sets where the sublinear
    /// scaling pivots and is absorbed by tuning, so the exponent alone fixes the
    /// concurrency SHAPE.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_PREFILL_THROUGHPUT_REF",
        default_value_t = 0
    )]
    pub scheduler_prefill_throughput_ref: usize,

    /// Enable goodput-collapse modeling: once the decode queue overloads, the
    /// per-step admit budget shrinks toward a floor, so aggregate tok/s actually
    /// drops past the knee (models preemption/admission thrash).
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_GOODPUT_COLLAPSE_ENABLED",
        default_value_t = false,
        action = clap::ArgAction::SetTrue
    )]
    pub scheduler_goodput_collapse_enabled: bool,

    /// Decode-queue overload ratio (queue_len / max_batch) at which goodput
    /// collapse begins.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_GOODPUT_COLLAPSE_THRESHOLD",
        default_value_t = 1.5
    )]
    pub scheduler_goodput_collapse_threshold: f64,

    /// How fast the effective batch shrinks past threshold:
    /// `shrink = (overload - threshold) * slope`, capped at `(1 - floor)`.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_GOODPUT_COLLAPSE_SLOPE",
        default_value_t = 0.5
    )]
    pub scheduler_goodput_collapse_slope: f64,

    /// Floor fraction of `max_batch_size` the effective budget bottoms at under
    /// full collapse.
    #[arg(
        long,
        env = "MOCK_SERVER_SCHEDULER_GOODPUT_COLLAPSE_FLOOR",
        default_value_t = 0.3
    )]
    pub scheduler_goodput_collapse_floor: f64,

    /// Disable content-addressed KV-cache (prefix) reuse. By default the mock
    /// models radix prefix caching like SGLang (which has it ON by default,
    /// `--disable-radix-cache=False`): a prompt's leading blocks that match a
    /// cached prefix skip prefill, lowering TTFT, and are reported as
    /// `usage.prompt_tokens_details.cached_tokens`. Hits occur only on genuinely
    /// shared prefixes such as multi-turn history and system prompts. Uses
    /// SGLang's `--disable-radix-cache`.
    #[arg(
        long,
        env = "MOCK_SERVER_DISABLE_PREFIX_CACHE",
        default_value_t = false,
        action = clap::ArgAction::SetTrue
    )]
    pub disable_prefix_cache: bool,

    /// Tokens per cache block (prefix-matching granularity). Default 1 matches
    /// SGLang's default `page_size=1` (token-granular radix matching). Larger =
    /// coarser matching, like a paged KV cache with bigger pages.
    #[arg(
        long,
        env = "MOCK_SERVER_PREFIX_CACHE_BLOCK_TOKENS",
        default_value_t = 1
    )]
    pub prefix_cache_block_tokens: usize,

    /// Cache capacity in tokens (LRU eviction). Bounds the reuse window: once
    /// exceeded, least-recently-used prefixes go cold and stop hitting. Uses
    /// SGLang's `max_total_num_tokens` KV pool (which it derives from
    /// mem-fraction-static x GPU memory); the default is a representative
    /// single-large-GPU pool. Scale to your deployment for fidelity.
    #[arg(
        long,
        env = "MOCK_SERVER_PREFIX_CACHE_CAPACITY_BLOCKS",
        default_value_t = 500_000
    )]
    pub prefix_cache_capacity_blocks: usize,

    /// Override: force this fraction (0..1) of every prompt to be served from
    /// cache, bypassing content addressing. Use to study a target hit rate on
    /// workloads without natural prefix sharing. 0 = content-addressed matching.
    #[arg(long, env = "MOCK_SERVER_PREFIX_CACHE_HIT_RATE", default_value_t = 0.0)]
    pub prefix_cache_hit_rate: f64,

    /// Let prefix-cache hits reduce prefill work (and thus TTFT). OFF by default:
    /// in a saturated, queue-bound regime (e.g. the calibrated dsv4 agentic
    /// profile) TTFT is contention-dominated and empirically independent of
    /// cache hits (measured R^2 ~ 0.04), so the cache is reported in usage but
    /// does not move latency. Enable for an unsaturated server where freeing
    /// prefill compute genuinely lowers TTFT.
    #[arg(
        long,
        env = "MOCK_SERVER_PREFIX_CACHE_LATENCY_AWARE",
        default_value_t = false,
        action = clap::ArgAction::SetTrue
    )]
    pub prefix_cache_latency_aware: bool,

    /// KV-cache eviction policy applied when the prefix cache is at capacity.
    /// SGLang-compatible radix eviction policy (default `lru`); choices:
    /// lru, lfu, fifo, mru, filo, priority, slru. Only observable under capacity
    /// pressure (set `--prefix-cache-capacity-blocks` below the working set);
    /// otherwise nothing is evicted and every policy behaves identically. The
    /// `priority` policy needs a per-request `priority` field in the payload.
    #[arg(
        long,
        env = "MOCK_SERVER_PREFIX_CACHE_EVICTION_POLICY",
        value_enum,
        default_value = "lru"
    )]
    pub prefix_cache_eviction_policy: EvictionPolicy,

    #[arg(
        long,
        env = "MOCK_SERVER_EMBEDDING_BASE_LATENCY",
        default_value_t = 10.0
    )]
    pub embedding_base_latency: f64,

    #[arg(
        long,
        env = "MOCK_SERVER_EMBEDDING_PER_INPUT_LATENCY",
        default_value_t = 2.0
    )]
    pub embedding_per_input_latency: f64,

    #[arg(long, env = "MOCK_SERVER_RANKING_BASE_LATENCY", default_value_t = 10.0)]
    pub ranking_base_latency: f64,

    #[arg(
        long,
        env = "MOCK_SERVER_RANKING_PER_PASSAGE_LATENCY",
        default_value_t = 1.0
    )]
    pub ranking_per_passage_latency: f64,

    #[arg(
        long,
        env = "MOCK_SERVER_IMAGE_RETRIEVAL_BASE_LATENCY",
        default_value_t = 10.0
    )]
    pub image_retrieval_base_latency: f64,

    #[arg(
        long,
        env = "MOCK_SERVER_IMAGE_RETRIEVAL_PER_IMAGE_LATENCY",
        default_value_t = 5.0
    )]
    pub image_retrieval_per_image_latency: f64,

    /// Actually HTTP-GET `image_url`/`video_url` values in chat and
    /// `/v1/image/infer` requests instead of treating them as opaque strings.
    /// Off by default so existing runs with fake/non-routable URLs are
    /// unaffected; enable to exercise an AIPerf content server (or any URL
    /// host) end to end, triggering its serving and transfer-record path.
    #[arg(long, env = "MOCK_SERVER_FETCH_CONTENT_URLS", default_value_t = false)]
    pub fetch_content_urls: bool,

    /// Per-request timeout (seconds) for content-URL fetches when
    /// `--fetch-content-urls` is enabled.
    #[arg(
        long,
        env = "MOCK_SERVER_CONTENT_FETCH_TIMEOUT",
        default_value_t = 30.0
    )]
    pub content_fetch_timeout: f64,

    #[arg(long, env = "MOCK_SERVER_LOG_LEVEL", default_value = "INFO")]
    pub log_level: String,

    #[arg(
        short = 'v',
        long,
        env = "MOCK_SERVER_VERBOSE",
        default_value_t = false
    )]
    pub verbose: bool,

    #[arg(short = 'f', long, env = "MOCK_SERVER_FAST", default_value_t = false)]
    pub fast: bool,

    #[arg(
        long,
        env = "MOCK_SERVER_ACCESS_LOGS",
        default_value_t = false,
        action = clap::ArgAction::SetTrue,
    )]
    pub access_logs: bool,

    #[arg(long, env = "MOCK_SERVER_ERROR_RATE", default_value_t = 0.0)]
    pub error_rate: f64,

    /// HTTP status codes injected when `--error-rate` fires. Comma-separated
    /// (e.g. `429,503,400,500`); the mock picks one per injected error via the
    /// seeded `mock.errors` RNG stream, so the sequence is reproducible under
    /// `--random-seed`. Defaults to `500`.
    #[arg(
        long,
        env = "MOCK_SERVER_ERROR_STATUS_CODES",
        value_delimiter = ',',
        default_value = "500"
    )]
    pub error_status_codes: Vec<u16>,

    /// `Retry-After` header value (whole seconds) emitted on injected `429` and
    /// `503` responses — the backoff hint a real rate-limited / overloaded
    /// backend returns and that AIPerf's retry policy reads.
    #[arg(long, env = "MOCK_SERVER_ERROR_RETRY_AFTER", default_value_t = 1)]
    pub error_retry_after: u64,

    /// Seeded probability (0.0–1.0) that a *streaming* chat request emits a few
    /// normal token frames and then a terminal mid-stream SSE error
    /// (`event: error`) instead of completing. Exercises the runner's
    /// mid-stream error path, which pre-stream injection never reaches. The
    /// decision is drawn from the seeded `mock.errors` stream.
    #[arg(long, env = "MOCK_SERVER_ERROR_MIDSTREAM_RATE", default_value_t = 0.0)]
    pub error_midstream_rate: f64,

    #[arg(long, env = "MOCK_SERVER_RANDOM_SEED")]
    pub random_seed: Option<u64>,

    #[arg(long, env = "MOCK_SERVER_DCGM_GPU_NAME", default_value = "h200")]
    pub dcgm_gpu_name: String,

    #[arg(long, env = "MOCK_SERVER_DCGM_NUM_GPUS", default_value_t = 2)]
    pub dcgm_num_gpus: u32,

    #[arg(long, env = "MOCK_SERVER_DCGM_MIN_THROUGHPUT", default_value_t = 100)]
    pub dcgm_min_throughput: u32,

    #[arg(long, env = "MOCK_SERVER_DCGM_WINDOW_SEC", default_value_t = 1.0)]
    pub dcgm_window_sec: f64,

    #[arg(long, env = "MOCK_SERVER_DCGM_HOSTNAME", default_value = "localhost")]
    pub dcgm_hostname: String,

    #[arg(long, env = "MOCK_SERVER_DCGM_SEED")]
    pub dcgm_seed: Option<u64>,

    #[arg(
        long,
        env = "MOCK_SERVER_DCGM_AUTO_LOAD",
        default_value_t = true,
        action = clap::ArgAction::Set,
    )]
    pub dcgm_auto_load: bool,

    #[arg(long, env = "MOCK_SERVER_TOKENIZER", default_value = "Qwen/Qwen3-0.6B")]
    pub tokenizer: String,

    #[arg(long, env = "MOCK_SERVER_TOKENIZER_REVISION", default_value = "main")]
    pub tokenizer_revision: String,

    #[arg(
        long,
        env = "MOCK_SERVER_TOKENIZER_TRUST_REMOTE_CODE",
        default_value_t = false
    )]
    pub tokenizer_trust_remote_code: bool,

    #[arg(long, env = "MOCK_SERVER_NO_TOKENIZER", default_value_t = false)]
    pub no_tokenizer: bool,

    /// Path to a JSONL accuracy dataset. Each line is an object with a prompt
    /// field (`prompt`/`question`/`input`/`text`) and a gold field
    /// (`ground_truth`/`answer`/`gold`/`target`), plus optional `task`,
    /// per-row `format`, and `choices`. When set, requests whose user text
    /// matches a row return the ground-truth answer (formatted for the grader)
    /// for a seeded fraction of requests; unmatched requests fall through to the
    /// normal corpus generation.
    #[arg(long, env = "MOCK_SERVER_ACCURACY_DATASET")]
    pub accuracy_dataset: Option<String>,

    /// Default grader format used to wrap the gold answer. A per-row `format`
    /// field overrides this.
    #[arg(
        long,
        env = "MOCK_SERVER_ACCURACY_FORMAT",
        value_enum,
        default_value = "passthrough"
    )]
    pub accuracy_format: AccuracyFormat,

    /// How an incoming request's user text is matched to a dataset row.
    /// All modes whitespace-normalize; `substring` (default) also matches when a
    /// row key is contained in the request (few-shot / system-prompt wrapping);
    /// the `_ci` variants case-fold. A row may carry a dedicated `match_key`
    /// (aliases `match`/`key`/`id`) — a stable fragment guaranteed to appear in
    /// the wire prompt — which is matched instead of the full prompt.
    #[arg(
        long,
        env = "MOCK_SERVER_ACCURACY_MATCH",
        value_enum,
        default_value = "substring"
    )]
    pub accuracy_match: AccuracyMatch,

    /// Seeded fraction of matched requests returned with the correct answer
    /// (0.0–1.0). The rest return a plausible wrong answer. The verdict is
    /// deterministic in `(random_seed, prompt)`, independent of arrival order.
    #[arg(long, env = "MOCK_SERVER_ACCURACY_CORRECT_RATE", default_value_t = 1.0)]
    pub accuracy_correct_rate: f64,

    /// Seeded fraction of matched requests rendered as chain-of-thought
    /// (reasoning plus the final answer) rather than a bare answer.
    #[arg(long, env = "MOCK_SERVER_ACCURACY_COT_RATE", default_value_t = 0.0)]
    pub accuracy_cot_rate: f64,

    /// Seeded fraction of matched requests rendered as an adversarial,
    /// parser-choking response shape (leading whitespace, reasoning-only
    /// content, wrong case, boxed wrap, conflicting answers, unicode, or a
    /// streaming `object: null` frame).
    #[arg(
        long,
        env = "MOCK_SERVER_ACCURACY_ADVERSARIAL_RATE",
        default_value_t = 0.0
    )]
    pub accuracy_adversarial_rate: f64,

    /// When rendering CoT, place the reasoning in a separate `reasoning_content`
    /// field (reasoning-model shape) instead of inline before the answer.
    #[arg(
        long,
        env = "MOCK_SERVER_ACCURACY_REASONING_FIELD",
        default_value_t = true,
        action = clap::ArgAction::Set,
    )]
    pub accuracy_reasoning_field: bool,

    /// Comma-separated list of models to advertise on `GET /v1/models`. When
    /// empty (the default) a small list of well-known LLM names is returned.
    /// Models seen via actual traffic are always appended.
    #[arg(long, env = "MOCK_SERVER_MODELS", value_delimiter = ',')]
    pub models: Vec<String>,

    // Zero-valued usage fields are omitted from the wire payload.
    /// Prompt tokens reported as written into the KV cache, emitted as top-level
    /// `cache_creation_input_tokens` (OpenAI) and in the Anthropic `messages`
    /// usage. Feeds AIPerf `usage_prompt_cache_write_tokens`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_CACHE_WRITE_TOKENS",
        default_value_t = 0
    )]
    pub usage_cache_write_tokens: usize,

    /// Prompt cache-miss count, emitted as top-level `prompt_cache_miss_tokens`.
    /// Feeds AIPerf `usage_prompt_cache_miss_tokens`.
    #[arg(long, env = "MOCK_SERVER_USAGE_CACHE_MISS_TOKENS", default_value_t = 0)]
    pub usage_cache_miss_tokens: usize,

    /// Anthropic disjoint cache-read count, emitted as `cache_read_input_tokens`
    /// in the `messages` usage object only.
    #[arg(long, env = "MOCK_SERVER_USAGE_CACHE_READ_TOKENS", default_value_t = 0)]
    pub usage_cache_read_tokens: usize,

    /// Audio tokens attributed to the prompt, emitted under
    /// `prompt_tokens_details.audio_tokens`. Feeds `usage_prompt_audio_tokens`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_PROMPT_AUDIO_TOKENS",
        default_value_t = 0
    )]
    pub usage_prompt_audio_tokens: usize,

    /// Audio tokens attributed to model output, emitted under
    /// `completion_tokens_details.audio_tokens`. Feeds
    /// `usage_completion_audio_tokens`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_COMPLETION_AUDIO_TOKENS",
        default_value_t = 0
    )]
    pub usage_completion_audio_tokens: usize,

    /// Prompt-audio duration in seconds, emitted as top-level
    /// `prompt_audio_seconds`. Feeds `usage_prompt_audio_seconds`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_PROMPT_AUDIO_SECONDS",
        default_value_t = 0.0
    )]
    pub usage_prompt_audio_seconds: f64,

    /// Accepted predicted-output tokens, emitted under
    /// `completion_tokens_details.accepted_prediction_tokens`. Feeds
    /// `usage_accepted_prediction_tokens`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_ACCEPTED_PREDICTION_TOKENS",
        default_value_t = 0
    )]
    pub usage_accepted_prediction_tokens: usize,

    /// Rejected predicted-output tokens, emitted under
    /// `completion_tokens_details.rejected_prediction_tokens`. Feeds
    /// `usage_rejected_prediction_tokens`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_REJECTED_PREDICTION_TOKENS",
        default_value_t = 0
    )]
    pub usage_rejected_prediction_tokens: usize,

    /// Tool-definition prompt tokens, emitted as top-level
    /// `toolUsePromptTokenCount` (the exact key AIPerf's `UsageView` reads).
    /// Feeds `usage_tool_use_prompt_tokens`.
    #[arg(
        long,
        env = "MOCK_SERVER_USAGE_TOOL_USE_PROMPT_TOKENS",
        default_value_t = 0
    )]
    pub usage_tool_use_prompt_tokens: usize,

    /// Seeded probability (0.0–1.0) that a *chat* request answers with a
    /// function tool call (`message.tool_calls` non-streaming, `delta.tool_calls`
    /// deltas streaming) instead of a plain assistant turn. The per-request draw
    /// comes from the seeded `mock.tool_calls` stream, so which requests emit a
    /// tool call is reproducible under `--random-seed`. When a tool call fires,
    /// the finish reason becomes `tool_calls` and the emitted `usage` carries
    /// `toolUsePromptTokenCount`. Default `0.0` disables tool-call emission
    /// entirely (a normal run's payload is byte-unchanged). Only affects the
    /// OpenAI-compatible chat endpoint.
    #[arg(long, env = "MOCK_SERVER_TOOL_CALL_RATE", default_value_t = 0.0)]
    pub tool_call_rate: f64,

    /// Function name used for emitted tool calls (deterministic). Paired with
    /// `--tool-call-arguments`; the runner parses `function.name` +
    /// `function.arguments` back into the captured record.
    #[arg(
        long,
        env = "MOCK_SERVER_TOOL_CALL_NAME",
        default_value = "get_weather"
    )]
    pub tool_call_name: String,

    /// JSON-encoded argument string for emitted tool calls (deterministic). Sent
    /// verbatim as `function.arguments` (OpenAI encodes tool-call arguments as a
    /// JSON *string*, not an object). In streaming mode it is split across two
    /// `delta.tool_calls` frames so the runner's argument-concatenation merge is
    /// exercised end to end.
    #[arg(
        long,
        env = "MOCK_SERVER_TOOL_CALL_ARGUMENTS",
        default_value = r#"{"location":"NYC"}"#
    )]
    pub tool_call_arguments: String,
}

impl MockServerConfig {
    /// True when any TLS flag selects an HTTPS frontend (explicit cert/key pair
    /// or `--tls-self-signed`).
    pub fn tls_enabled(&self) -> bool {
        self.tls_self_signed || self.tls_cert.is_some() || self.tls_key.is_some()
    }

    pub fn usage_fields_enabled(&self) -> bool {
        self.usage_cache_write_tokens != 0
            || self.usage_cache_miss_tokens != 0
            || self.usage_cache_read_tokens != 0
            || self.usage_prompt_audio_tokens != 0
            || self.usage_completion_audio_tokens != 0
            || self.usage_prompt_audio_seconds != 0.0
            || self.usage_accepted_prediction_tokens != 0
            || self.usage_rejected_prediction_tokens != 0
            || self.usage_tool_use_prompt_tokens != 0
    }
}

impl Default for MockServerConfig {
    fn default() -> Self {
        Self::parse_from::<_, &str>([])
    }
}

impl MockServerConfig {
    /// Apply `--fast` and `--verbose` configuration overrides.
    pub fn apply_flags(mut self) -> Self {
        if self.verbose {
            self.log_level = "DEBUG".to_string();
        }
        if self.fast {
            self.ttft = 0.0;
            self.itl = 0.0;
            self.ttft_per_isl_token_ms = 0.0;
            self.ttft_concurrency_quad_ms = 0.0;
            self.itl_per_osl_token_ms = 0.0;
            self.itl_concurrency_lin_ms = 0.0;
            self.ttft_jitter_cv = 0.0;
            self.itl_jitter_cv = 0.0;
            // `--fast` must bypass scheduler and cache latency as well.
            self.scheduler_enabled = false;
            self.disable_prefix_cache = true;
            self.prefix_cache_hit_rate = 0.0;
            self.embedding_base_latency = 0.0;
            self.embedding_per_input_latency = 0.0;
            self.ranking_base_latency = 0.0;
            self.ranking_per_passage_latency = 0.0;
            self.image_retrieval_base_latency = 0.0;
            self.image_retrieval_per_image_latency = 0.0;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_zeros_all_latencies() {
        let cfg = MockServerConfig {
            fast: true,
            ..MockServerConfig::default()
        }
        .apply_flags();
        assert_eq!(cfg.ttft, 0.0);
        assert_eq!(cfg.itl, 0.0);
        assert_eq!(cfg.ttft_per_isl_token_ms, 0.0);
        assert_eq!(cfg.embedding_base_latency, 0.0);
        assert_eq!(cfg.embedding_per_input_latency, 0.0);
        assert_eq!(cfg.ranking_base_latency, 0.0);
        assert_eq!(cfg.ranking_per_passage_latency, 0.0);
        assert_eq!(cfg.image_retrieval_base_latency, 0.0);
        assert_eq!(cfg.image_retrieval_per_image_latency, 0.0);
    }

    #[test]
    fn verbose_sets_debug_log_level() {
        let cfg = MockServerConfig {
            verbose: true,
            ..MockServerConfig::default()
        }
        .apply_flags();
        assert_eq!(cfg.log_level, "DEBUG");
    }

    #[test]
    fn default_values_are_stable() {
        let cfg = MockServerConfig::default();
        assert_eq!(cfg.port, 8000);
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.grpc_port, None);
        assert_eq!(cfg.uds, None);
        assert_eq!(cfg.tls_cert, None);
        assert_eq!(cfg.tls_key, None);
        assert!(!cfg.tls_self_signed);
        assert!(!cfg.tls_enabled());
        assert_eq!(cfg.processes, 1);
        assert_eq!(cfg.workers, 0);
        assert_eq!(cfg.ttft, 20.0);
        assert_eq!(cfg.itl, 5.0);
        assert_eq!(cfg.ttft_per_isl_token_ms, 0.0);
        assert_eq!(cfg.ttft_concurrency_quad_ms, 0.0);
        assert_eq!(cfg.itl_per_osl_token_ms, 0.0);
        assert_eq!(cfg.itl_concurrency_lin_ms, 0.0);
        assert!(!cfg.scheduler_enabled);
        assert_eq!(cfg.scheduler_step_ms, 5.0);
        assert_eq!(cfg.scheduler_max_batch_size, 256);
        assert_eq!(cfg.scheduler_max_prefill_chunks_per_step, 8);
        assert_eq!(cfg.scheduler_prefill_chunk_tokens, 512);
        assert_eq!(cfg.scheduler_prefill_throughput_exponent, 0.0);
        assert_eq!(cfg.scheduler_prefill_throughput_ref, 0);
        assert_eq!(cfg.scheduler_goodput_collapse_threshold, 1.5);
        assert!(!cfg.disable_prefix_cache);
        assert_eq!(cfg.prefix_cache_block_tokens, 1);
        assert_eq!(cfg.prefix_cache_hit_rate, 0.0);
        assert!(!cfg.prefix_cache_latency_aware);
        assert_eq!(cfg.embedding_base_latency, 10.0);
        assert_eq!(cfg.dcgm_gpu_name, "h200");
        assert_eq!(cfg.dcgm_num_gpus, 2);
        assert!(cfg.dcgm_auto_load);
        assert!(!cfg.fast);
        assert_eq!(cfg.error_rate, 0.0);
        assert_eq!(cfg.error_status_codes, vec![500u16]);
        assert_eq!(cfg.error_retry_after, 1);
        assert_eq!(cfg.error_midstream_rate, 0.0);
    }

    #[test]
    fn uds_flag_parses() {
        let cfg = MockServerConfig::parse_from(["aiperf-mock-server", "--uds", "/tmp/mock.sock"]);
        assert_eq!(cfg.uds.as_deref(), Some("/tmp/mock.sock"));
    }

    #[test]
    fn tool_call_defaults_are_disabled() {
        let cfg = MockServerConfig::default();
        assert_eq!(cfg.tool_call_rate, 0.0);
        assert_eq!(cfg.tool_call_name, "get_weather");
        assert_eq!(cfg.tool_call_arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn parses_error_status_code_menu() {
        let cfg = MockServerConfig::parse_from([
            "aiperf-mock-server",
            "--error-status-codes",
            "429,503,400,500",
        ]);
        assert_eq!(cfg.error_status_codes, vec![429u16, 503, 400, 500]);
    }
}
