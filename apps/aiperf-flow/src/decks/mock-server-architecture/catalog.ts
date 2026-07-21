/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Audited 64-page source/proof catalog for `aiperf-mock-server`, ported verbatim from the
//! hand-authored Cursor canvas `docs/canvases/mock-server-architecture.canvas.tsx` ("THE MOCK
//! FOUNDRY"). The chapter list, page ids, titles, invariant sentences, source/proof file paths,
//! node lists, steps, and modes are the real content ground truth and are preserved exactly —
//! nothing here is invented. Rendering components synthesize these facts into React Flow diagrams
//! and tables; the facts themselves live here.

export type ChapterId =
  | "orientation"
  | "ingress"
  | "llm"
  | "specialized"
  | "grpc"
  | "timing"
  | "scheduler"
  | "semantics"
  | "observability"
  | "proof";

export type PageStatus = "built" | "partial" | "boundary";
export type EvidenceTier = "raw-e2e" | "integration" | "unit" | "implementation";
export type VisualKind =
  | "flow"
  | "frames"
  | "timeline"
  | "tensor"
  | "conveyor"
  | "cache"
  | "instrument"
  | "topology"
  | "evidence";

export type FeaturePage = {
  id: string;
  chapter: ChapterId;
  title: string;
  kicker: string;
  status: PageStatus;
  evidence: EvidenceTier;
  visual: VisualKind;
  source: string;
  proof: string;
  invariant: string;
  nodes: readonly string[];
  steps: readonly string[];
  modes: readonly string[];
  metrics: readonly number[];
};

export type Chapter = {
  id: ChapterId;
  title: string;
  short: string;
  world: string;
};

export const CHAPTERS: readonly Chapter[] = [
  { id: "orientation", title: "Orientation", short: "Foundry map", world: "process cutaway" },
  { id: "ingress", title: "Runtime and ingress", short: "Ingress manifold", world: "listener + route manifold" },
  { id: "llm", title: "LLM protocols", short: "Glassworks", world: "transparent protocol pipes" },
  { id: "specialized", title: "Specialized endpoints", short: "Endpoint works", world: "looms, sorters, chambers" },
  { id: "grpc", title: "gRPC and Riva", short: "Switching yard", world: "protobuf yard + transducers" },
  { id: "timing", title: "Timing and generation", short: "Escapement", world: "TTFT / ITL escapement" },
  { id: "scheduler", title: "Scheduler and cache", short: "Foundry floor", world: "conveyor + cache library" },
  { id: "semantics", title: "Faults and semantics", short: "Fault lab", world: "injectors + verdict gates" },
  { id: "observability", title: "Observability and deployment", short: "Telemetry deck", world: "gauges + replicated foundries" },
  { id: "proof", title: "Proof and boundaries", short: "Proof machine", world: "exploded evidence" },
];

/** Per-chapter stage vocabulary the traveling "specimen" morphs into, from the canvas source. */
export const SPECIMEN_STAGE: Record<ChapterId, string> = {
  orientation: "Config v2",
  ingress: "HTTP bytes",
  llm: "protocol frame",
  specialized: "typed payload",
  grpc: "protobuf tensor",
  timing: "token stream",
  scheduler: "scheduled batch",
  semantics: "response state",
  observability: "metric facts",
  proof: "evidence link",
};

const S = "rust/mock-server/src/";
const T = "rust/mock-server/tests/";
const E = "rust/e2e/tests/";

function page(
  id: string,
  chapter: ChapterId,
  title: string,
  visual: VisualKind,
  source: string,
  proof: string,
  invariant: string,
  nodes: readonly string[],
  options: Partial<Pick<FeaturePage, "kicker" | "status" | "evidence" | "steps" | "modes" | "metrics">> = {},
): FeaturePage {
  return {
    id,
    chapter,
    title,
    visual,
    source,
    proof,
    invariant,
    nodes,
    kicker: options.kicker ?? "Implementation truth",
    status: options.status ?? "built",
    evidence: options.evidence ?? "integration",
    steps: options.steps ?? ["Inspect", "Advance", "Resolve"],
    modes: options.modes ?? ["primary", "compare"],
    metrics: options.metrics ?? [18, 42, 68, 86],
  };
}

export const PAGES: readonly FeaturePage[] = [
  page("process-boundary", "orientation", "Standalone target boundary", "topology", `${S}main.rs`, `${E}test_chat_endpoint.rs`, "The mock is launched independently; AIPerf sees an ordinary target.", ["aiperf", "HTTP or gRPC", "mock process"], { evidence: "raw-e2e" }),
  page("crate-dependency", "orientation", "Crate dependency direction", "evidence", "rust/mock-server/Cargo.toml", "rust/mock-server/Cargo.toml", "The mock depends on aiperf-runtime; the product execution path does not depend on the mock.", ["aiperf-runtime", "aiperf-mock-server", "product runner"], { evidence: "implementation" }),
  page("startup", "orientation", "Startup decision tree", "flow", `${S}main.rs`, `${T}balancer.rs`, "Configuration selects balancer or single-process listeners before serving.", ["parse config", "process gate", "build state", "serve"]),
  page("application-state", "orientation", "Shared application state", "topology", `${S}state.rs`, `${T}integration.rs`, "Router clones share immutable configuration and synchronized generators, metrics, cache, and accuracy state.", ["config", "token generator", "metrics", "prefix cache", "accuracy"]),
  page("request-journey", "orientation", "One request end to end", "flow", `${S}handlers.rs`, `${E}test_tuned_raw_timing.rs`, "A request crosses parsing, token budgeting, latency, streaming, and accounting in one server process.", ["listener", "router", "handler", "tokens", "latency", "SSE", "metrics"], { evidence: "raw-e2e", steps: ["Accept", "Route", "Budget", "Pace", "Stream", "Record"] }),
  page("architecture-atlas", "orientation", "Architecture atlas", "evidence", `${S}lib.rs`, `${E}test_tuned_raw_timing.rs`, "Source files form a testable feature graph rather than one monolithic handler.", ["entry", "protocols", "timing", "state", "proof"], { evidence: "raw-e2e" }),

  page("config", "ingress", "CLI and environment configuration", "frames", `${S}config.rs`, `${S}config.rs`, "Clap definitions are authoritative, but definition alone does not prove wiring.", ["CLI flag", "environment twin", "MockServerConfig", "consumer"], { evidence: "unit" }),
  page("tcp-listener", "ingress", "TCP listener", "flow", `${S}listener.rs`, `${T}integration.rs`, "The tuned Hyper listener accepts TCP and serves the shared router.", ["bind", "accept", "connection", "router"]),
  page("http2", "ingress", "HTTP/1.1 and HTTP/2", "frames", `${S}listener.rs`, `${T}integration.rs`, "One connection builder negotiates HTTP behavior and honors the configured h2 stream limit.", ["TCP", "HTTP/1.1", "h2c", "request"], { modes: ["HTTP/1.1", "h2c"] }),
  page("uds", "ingress", "Unix-domain HTTP listener", "topology", `${S}listener.rs`, `${E}test_uds.rs`, "UDS serves HTTP/1.1 directly and refuses to delete a non-socket path.", ["socket path", "UnixListener", "Hyper", "router"], { evidence: "raw-e2e", modes: ["valid socket", "path collision"] }),
  page("tls", "ingress", "TLS and ALPN", "frames", `${S}tls.rs`, `${E}test_tls.rs`, "TLS wraps accepted streams and advertises HTTP protocols through ALPN.", ["certificate", "rustls", "ALPN", "HTTPS"], { evidence: "raw-e2e", modes: ["HTTP/1.1", "HTTP/2"] }),
  page("router", "ingress", "Axum route surface", "topology", `${S}app.rs`, `${T}integration.rs`, "All HTTP dialects converge on shared state while retaining distinct wire shapes.", ["health", "OpenAI", "Anthropic", "KServe", "telemetry"]),
  page("models-health", "ingress", "Health and model discovery", "instrument", `${S}app.rs`, `${T}integration.rs`, "Health is a direct liveness response; observed model names extend model discovery.", ["GET /health", "GET /v1/models", "seen models"], { metrics: [200, 3, 7, 12] }),

  page("chat", "llm", "OpenAI chat completions", "frames", `${S}handlers.rs`, `${E}test_chat_endpoint.rs`, "Chat supports streaming and non-streaming responses with the requested model echoed.", ["request", "choice delta", "content", "finish"], { evidence: "raw-e2e", modes: ["stream", "non-stream"] }),
  page("completions", "llm", "Legacy completions", "frames", `${S}handlers.rs`, `${E}test_completions_endpoint.rs`, "The completions route uses text choices rather than chat message deltas.", ["prompt", "text choice", "usage", "done"], { evidence: "raw-e2e" }),
  page("sse", "llm", "SSE stream assembly", "frames", `${S}handlers.rs`, `${T}integration.rs`, "Generated token events precede terminal usage and the stream terminator.", ["headers", "data frame", "token frames", "usage frame", "[DONE]"], { steps: ["Headers", "First token", "Next token", "Usage", "Done"] }),
  page("terminal-usage", "llm", "Terminal usage frame", "timeline", `${S}handlers.rs`, `${E}test_usage_fields.rs`, "Usage is terminal accounting, not a generated token timing sample.", ["request start", "generated tokens", "usage", "done"], { evidence: "raw-e2e", metrics: [0, 25, 35, 35] }),
  page("messages", "llm", "Anthropic Messages", "frames", `${S}handlers.rs`, `${E}test_new_routes.rs`, "Anthropic event names and cache usage fields are emitted in Anthropic shapes.", ["message_start", "content_block_delta", "message_delta", "message_stop"], { evidence: "raw-e2e" }),
  page("responses", "llm", "OpenAI Responses API", "frames", `${S}handlers.rs`, `${E}test_new_routes.rs`, "Responses emits response-scoped output events rather than chat completion chunks.", ["response.created", "response.output_text.delta", "response.completed"], { evidence: "raw-e2e" }),
  page("reasoning", "llm", "Reasoning content", "frames", `${S}tokens.rs`, `${E}test_usage_fields.rs`, "Reasoning and output content remain distinguishable in supported response shapes.", ["reasoning", "visible output", "usage reconciliation"], { evidence: "raw-e2e" }),
  page("vllm-generate", "llm", "Token-native vLLM generate", "tensor", `${S}handlers.rs`, `${E}test_new_routes.rs`, "The vLLM route accepts token-native input and returns generated token data.", ["input token ids", "generation", "output token ids"], { evidence: "raw-e2e", modes: ["token ids", "text"] }),

  page("embeddings", "specialized", "Embeddings", "tensor", `${S}handlers.rs`, `${E}test_embeddings_endpoint.rs`, "Embedding dimensions and values are deterministic for the same input.", ["input text", "deterministic seed", "FP vector"], { evidence: "raw-e2e", modes: ["single", "batch"] }),
  page("rankings", "specialized", "Ranking dialects", "tensor", `${S}handlers.rs`, `${E}test_rankings_endpoint.rs`, "Ranking routes produce dialect-specific score and index shapes from shared deterministic scoring.", ["query", "documents", "scores", "rank order"], { evidence: "raw-e2e", modes: ["rerank", "rankings"] }),
  page("tgi", "specialized", "TGI generate compatibility", "frames", `${S}handlers.rs`, `${E}test_huggingface_generate_endpoint.rs`, "TGI request and stream shapes map onto the same generation seam.", ["inputs", "parameters", "token event", "generated_text"], { evidence: "raw-e2e" }),
  page("image-generation", "specialized", "Image generation", "tensor", `${S}handlers.rs`, `${E}test_image_generation_endpoint.rs`, "Image output is deterministic and returned in the requested endpoint shape.", ["prompt", "image generator", "encoded image"], { evidence: "raw-e2e", modes: ["URL-like", "base64"] }),
  page("image-edit", "specialized", "Image edit", "tensor", `${S}handlers.rs`, `${E}test_image_edit_endpoint.rs`, "Image and mask inputs feed the edit response path.", ["source image", "mask", "prompt", "edited output"], { evidence: "raw-e2e" }),
  page("image-retrieval", "specialized", "Image retrieval", "timeline", `${S}handlers.rs`, `${E}test_image_retrieval_endpoint.rs`, "Image retrieval latency is independently configurable from generation.", ["query", "retrieval delay", "ranked images"], { evidence: "raw-e2e", metrics: [0, 15, 45, 70] }),
  page("multimodal", "specialized", "Multimodal payloads", "frames", `${S}models.rs`, `${E}test_multimodal.rs`, "Text and media content parts are parsed without changing the shared response machinery.", ["text part", "image part", "message", "response"], { evidence: "raw-e2e", modes: ["image", "audio", "video"] }),
  page("rag-kserve-http", "specialized", "RAG and KServe HTTP", "flow", `${S}app.rs`, `${E}test_kserve.rs`, "RAG and KServe HTTP aliases remain HTTP routes over shared handlers.", ["HTTP alias", "request model", "handler", "typed response"], { evidence: "raw-e2e", modes: ["RAG", "v2 infer", "v1 predict"] }),

  page("grpc-unary", "grpc", "KServe unary ModelInfer", "tensor", `${S}grpc.rs`, `${T}grpc_integration.rs`, "Unary protobuf tensors are decoded and returned on the KServe inference service.", ["ModelInferRequest", "input tensor", "output tensor", "ModelInferResponse"]),
  page("grpc-stream", "grpc", "KServe ModelStreamInfer", "frames", `${S}grpc.rs`, `${T}grpc_integration.rs`, "The streaming RPC emits a sequence of protobuf inference responses.", ["RPC headers", "stream item", "terminal item", "trailers"]),
  page("grpc-readiness", "grpc", "KServe readiness services", "instrument", `${S}grpc.rs`, `${T}grpc_integration.rs`, "ModelReady, ServerLive, and ServerReady expose deterministic readiness.", ["ModelReady", "ServerLive", "ServerReady"], { metrics: [1, 1, 1, 1] }),
  page("grpc-embeddings", "grpc", "gRPC embedding tensor", "tensor", `${S}grpc.rs`, `${T}grpc_integration.rs`, "Embedding mode returns FP32 shape [1, configured dimension] with no token semantics.", ["STRING input", "embedding generator", "FP32 [1,N]"], { modes: ["N=8", "N=32", "N=128"] }),
  page("grpc-rankings", "grpc", "gRPC ranking behavior", "tensor", `${S}grpc.rs`, `${T}grpc_integration.rs`, "Ranking behavior selects ranking tensors instead of generated text.", ["query tensor", "candidate tensor", "scores", "indices"], { modes: ["auto", "rankings"] }),
  page("grpc-images", "grpc", "gRPC image and VLM behavior", "tensor", `${S}grpc.rs`, `${T}grpc_integration.rs`, "Behavior selection changes output tensor types while preserving the KServe RPC.", ["input tensors", "behavior gate", "image or VLM output"], { modes: ["images", "text", "auto"] }),
  page("riva-asr-tts", "grpc", "Riva ASR and TTS", "flow", `${S}grpc_riva.rs`, `${E}test_riva.rs`, "Riva speech services are exposed only through gRPC service methods.", ["audio bytes", "ASR transcript", "TTS waveform"], { evidence: "raw-e2e", modes: ["ASR", "TTS"] }),
  page("riva-nlp-boundary", "grpc", "Riva NLP is gRPC-only", "evidence", `${S}grpc_riva.rs`, `${E}test_riva.rs`, "Riva ASR, TTS, and NLP have no HTTP route in the mock router.", ["HTTP router", "gRPC Riva service", "NLP response"], { status: "boundary", evidence: "raw-e2e", modes: ["HTTP absent", "gRPC built"] }),

  page("tokenizer-truth", "timing", "Character and corpus tokenization", "flow", `${S}tokens.rs`, `${S}tokens.rs`, "Rust token generation is character/corpus based; it does not load a Hugging Face tokenizer.", ["prompt characters", "whitespace chunks", "corpus pieces", "token budget"], { evidence: "unit", status: "partial" }),
  page("tokenizer-flags", "timing", "HF tokenizer flags are unwired", "evidence", `${S}config.rs`, `${S}main.rs`, "--tokenizer, revision, and trust-remote-code are not consumed; --no-tokenizer only gates corpus loading.", ["HF identity flags", "not consumed", "--no-tokenizer", "corpus gate", "character tokenizer"], { status: "boundary", evidence: "implementation", modes: ["identity flags", "corpus gate"] }),
  page("token-budgets", "timing", "Deterministic token budgets", "tensor", `${S}tokens.rs`, `${E}test_deterministic_behavior.rs`, "Seeded inputs and fixed distributions yield repeatable input and output budgets.", ["seed", "ISL draw", "OSL draw", "generated corpus"], { evidence: "raw-e2e", modes: ["fixed", "distributed"] }),
  page("ttft-itl", "timing", "TTFT and ITL pacing", "timeline", `${S}latency.rs`, `${E}test_tuned_raw_timing.rs`, "First-token delay and generated-token gaps are independently paced.", ["request start", "TTFT", "token 1", "ITL", "token N"], { evidence: "raw-e2e", steps: ["Start", "Wait TTFT", "First token", "Wait ITL", "Complete"], metrics: [0, 20, 30, 40, 50] }),
  page("latency-scaling", "timing", "Concurrency and length scaling", "timeline", `${S}latency.rs`, `${E}test_rust_python_latency_parity.rs`, "Configured ISL, OSL, and concurrency coefficients alter analytic latency.", ["base latency", "ISL term", "OSL term", "concurrency term"], { evidence: "raw-e2e", modes: ["base", "loaded"] }),
  page("jitter-timerfd", "timing", "Seeded jitter and timerfd", "timeline", `${S}latency.rs`, `${T}integration.rs`, "Seeded jitter is reproducible and pacing sleeps through RealClock timerfd precision.", ["latency sample", "seeded CV", "sleep_ns", "emit"], { modes: ["zero jitter", "seeded jitter"], metrics: [0, 19, 31, 52] }),

  page("prefill-decode", "scheduler", "Prefill and decode stepping", "conveyor", `${S}scheduler.rs`, `${S}scheduler.rs`, "Scheduler ticks admit prefill work and emit decode tokens under configured capacities.", ["queue", "prefill chunk", "batch", "decode lane", "token"], { evidence: "unit", steps: ["Queue", "Prefill", "Batch", "Decode", "Emit"] }),
  page("batch-saturation", "scheduler", "Batch saturation", "conveyor", `${S}scheduler.rs`, `${E}test_recipe_collapse_knee.rs`, "The maximum batch size creates a visible admission and throughput knee.", ["waiting", "active batch", "capacity", "overflow"], { evidence: "raw-e2e", modes: ["below knee", "at knee", "above knee"], metrics: [12, 38, 71, 76, 62] }),
  page("goodput-collapse", "scheduler", "Goodput collapse", "conveyor", `${S}scheduler.rs`, `${E}test_recipe_collapse_knee.rs`, "Configured collapse behavior reduces service rate after saturation.", ["arrival rate", "saturation knee", "collapsed service", "backlog"], { evidence: "raw-e2e", metrics: [18, 41, 78, 55, 32] }),
  page("cache-blocks", "scheduler", "Prefix block hashing", "cache", `${S}prefix_cache.rs`, `${S}prefix_cache.rs`, "Prompt token blocks are hashed into bounded cache entries.", ["prompt tokens", "block split", "hash", "cache slot"], { evidence: "unit", steps: ["Split", "Hash", "Lookup", "Insert"] }),
  page("cache-eviction", "scheduler", "Prefix cache eviction", "cache", `${S}prefix_cache.rs`, `${S}prefix_cache.rs`, "Capacity pressure applies the selected LRU, LFU, or FIFO policy.", ["resident blocks", "new block", "victim", "replacement"], { evidence: "unit", modes: ["LRU", "LFU", "FIFO"] }),
  page("cache-latency-optin", "scheduler", "Cache latency effect is opt-in", "cache", `${S}handlers.rs`, `${S}prefix_cache.rs`, "Cache hits reduce latency only when --prefix-cache-latency-aware is enabled.", ["cache hit", "latency-aware gate", "cached token discount", "TTFT"], { status: "partial", evidence: "unit", modes: ["accounting only", "latency-aware"] }),

  page("status-injection", "semantics", "HTTP status injection", "frames", `${S}handlers.rs`, `${E}test_error_fidelity.rs`, "Seeded status injection chooses only from the configured status menu.", ["request", "error draw", "status menu", "error body"], { evidence: "raw-e2e", modes: ["success", "429", "500", "503"] }),
  page("retry-after", "semantics", "Retry-After fidelity", "frames", `${S}handlers.rs`, `${E}test_error_fidelity.rs`, "Configured Retry-After accompanies injected responses that support retry semantics.", ["status", "Retry-After header", "body"], { evidence: "raw-e2e" }),
  page("midstream", "semantics", "Mid-stream SSE failure", "frames", `${S}handlers.rs`, `${E}test_error_fidelity.rs`, "A stream can fail after generated output, preserving partial-response evidence.", ["headers", "token", "token", "error cut", "closed"], { evidence: "raw-e2e", steps: ["Open", "Token 1", "Token 2", "Inject error", "Close"] }),
  page("extended-usage", "semantics", "Extended usage accounting", "tensor", `${S}models.rs`, `${E}test_usage_fields.rs`, "Cache, audio, prediction, and tool-use usage fields preserve provider-specific names.", ["prompt usage", "cache usage", "audio usage", "prediction usage", "tool usage"], { evidence: "raw-e2e", modes: ["OpenAI", "Anthropic"] }),
  page("tool-calls", "semantics", "Tool-call emission", "frames", `${S}handlers.rs`, `${E}test_tool_calls.rs`, "Tool name and arguments appear in both complete and streamed tool-call shapes.", ["tool id", "function name", "argument fragments", "finish reason"], { evidence: "raw-e2e", modes: ["stream", "non-stream"] }),
  page("accuracy-verdicts", "semantics", "Accuracy formats and seeded verdicts", "flow", `${S}accuracy.rs`, `${T}accuracy_integration.rs`, "Prompt matching selects ground truth; seeded rates choose correct, CoT, and adversarial output forms.", ["prompt", "match strategy", "seeded verdict", "format renderer", "response"], { modes: ["MMLU", "GSM8K", "MATH", "exact"] }),
  page("accuracy-oracle", "semantics", "Live accuracy oracle", "instrument", `${S}accuracy.rs`, `${E}test_accuracy_mock.rs`, "/accuracy and Prometheus counters report the mock's actual served tally.", ["matched", "correct", "incorrect", "adversarial", "per task"], { evidence: "raw-e2e", metrics: [100, 73, 27, 8] }),

  page("prometheus", "observability", "Prometheus backend dialects", "instrument", `${S}prom.rs`, `${E}test_server_metrics.rs`, "One metrics endpoint can emit vLLM, SGLang, TRT-LLM, or Dynamo naming dialects.", ["request counters", "token counters", "cache counters", "dialect encoder"], { evidence: "raw-e2e", modes: ["vLLM", "SGLang", "TRT-LLM", "Dynamo"] }),
  page("dcgm", "observability", "Synthetic DCGM telemetry", "instrument", `${S}dcgm.rs`, `${E}test_dcgm_faker.rs`, "Per-GPU gauges and counters are deterministic under a configured seed.", ["GPU 0", "GPU 1", "power", "SM active", "energy"], { evidence: "raw-e2e", modes: ["idle", "loaded"] }),
  page("throughput-load", "observability", "Throughput-linked load", "instrument", `${S}throughput.rs`, `${E}test_telemetry_fills.rs`, "Synthetic GPU load follows observed request throughput within the configured window.", ["completed requests", "window rate", "load model", "DCGM gauges"], { evidence: "raw-e2e", metrics: [8, 24, 57, 83] }),
  page("multiprocess", "observability", "Multi-process L4 balancer", "topology", `${S}balancer.rs`, `${T}balancer.rs`, "The parent round-robins TCP connections across isolated child servers.", ["client connections", "L4 parent", "child 0", "child 1", "child N"], { modes: ["1 process", "4 processes"] }),
  page("limits-access", "observability", "Transport and access-log limits", "evidence", `${S}main.rs`, `${S}config.rs`, "--processes skips gRPC and UDS; --access-logs is defined but not connected to middleware.", ["process gate", "HTTP TCP", "gRPC skipped", "UDS skipped", "access logs unwired"], { status: "boundary", evidence: "implementation", modes: ["single process", "multi-process"] }),

  page("proof-graph", "proof", "Implementation-to-proof graph", "evidence", `${S}lib.rs`, `${E}test_tuned_raw_timing.rs`, "Raw-record e2e is strongest; integration, unit, then implementation-only evidence follow.", ["Rust source", "feature claim", "raw-record e2e", "integration", "unit"], { evidence: "raw-e2e", modes: ["strongest", "all tiers"] }),
  page("unsupported-matrix", "proof", "Unsupported combinations", "evidence", `${S}main.rs`, `${T}grpc_integration.rs`, "Boundaries are explicit: multi-process is TCP/HTTP-only and Riva is gRPC-only.", ["multi-process", "gRPC", "UDS", "Riva HTTP", "supported path"], { status: "boundary", evidence: "implementation", modes: ["deployment", "transport"] }),
  page("source-index", "proof", "Source and proof index", "topology", `${S}lib.rs`, `${T}integration.rs`, "Every atlas feature links to implementation and its strongest available proof.", ["configuration", "listeners", "handlers", "generation", "scheduler", "telemetry", "tests"], { modes: ["by module", "by evidence"] }),
];

/** All pages belonging to `chapter`, in catalog order. */
export function pagesForChapter(chapter: ChapterId): FeaturePage[] {
  return PAGES.filter((entry) => entry.chapter === chapter);
}

/** Single page by id (throws if unknown — ids are a fixed, audited set). */
export function pageById(id: string): FeaturePage {
  const found = PAGES.find((entry) => entry.id === id);
  if (found === undefined) {
    throw new Error(`Unknown Mock Foundry page: ${id}`);
  }
  return found;
}

/** Human-readable labels for the evidence tiers, strongest first. */
export const EVIDENCE_LABEL: Record<EvidenceTier, string> = {
  "raw-e2e": "raw-record e2e",
  integration: "integration",
  unit: "unit",
  implementation: "implementation-only",
};

export const STATUS_LABEL: Record<PageStatus, string> = {
  built: "built",
  partial: "partial",
  boundary: "boundary",
};
