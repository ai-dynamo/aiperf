import {
  Button,
  Callout,
  Code,
  Divider,
  Pill,
  Row,
  Select,
  Stack,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

// ═══════════════════════════════════════════════════════════════════════════
// THE MOCK FOUNDRY — one continuous living systems cutaway of aiperf-mock-server.
// Ten chapter-specific machine worlds; each of the 64 chambers has feature-
// specific geometry and an embedded interaction. A request specimen travels
// through the active machinery and morphs as it crosses process, listener,
// protocol, tensor, timing, scheduler, fault, telemetry, and proof boundaries.
// Facts, source paths, and boundary claims come from the audited catalog below.
// ═══════════════════════════════════════════════════════════════════════════

type Theme = ReturnType<typeof useHostTheme>;
type ChapterId =
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
type PageStatus = "built" | "partial" | "boundary";
type EvidenceTier = "raw-e2e" | "integration" | "unit" | "implementation";
type VisualKind =
  | "flow"
  | "frames"
  | "timeline"
  | "tensor"
  | "conveyor"
  | "cache"
  | "instrument"
  | "topology"
  | "evidence";
type Evt = { key: string; preventDefault: () => void };
type FeaturePage = {
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

const CHAPTERS: readonly { id: ChapterId; title: string; short: string; world: string }[] = [
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

const S = "rust/mock-server/src/";
const T = "rust/mock-server/tests/";
const E = "rust/e2e-tests/tests/";

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

// Audited 64-page source/proof catalog — preserved verbatim from the approved atlas.
const PAGES: readonly FeaturePage[] = [
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

const VISUALS: readonly VisualKind[] = ["flow", "frames", "timeline", "tensor", "conveyor", "cache", "instrument", "topology", "evidence"];

function validateCatalog(pages: readonly FeaturePage[]) {
  if (pages.length !== 64) throw new Error(`Mock Foundry requires exactly 64 pages; got ${pages.length}`);
  const ids = new Set<string>();
  for (const entry of pages) {
    if (ids.has(entry.id)) throw new Error(`Duplicate Mock Foundry page: ${entry.id}`);
    ids.add(entry.id);
    if (!CHAPTERS.some((chapter) => chapter.id === entry.chapter)) throw new Error(`Unknown chapter on ${entry.id}`);
    if (!entry.title || !entry.invariant || !entry.nodes.length || !entry.steps.length || !entry.modes.length) {
      throw new Error(`Incomplete Mock Foundry page: ${entry.id}`);
    }
    if (entry.status === "built" && !entry.source.startsWith("rust/")) throw new Error(`Built page lacks Rust source: ${entry.id}`);
    if (!entry.proof.startsWith("rust/") && entry.proof !== "Cargo.toml") throw new Error(`Invalid proof on ${entry.id}`);
  }
  for (const chapter of CHAPTERS) if (!pages.some((entry) => entry.chapter === chapter.id)) throw new Error(`Empty chapter: ${chapter.id}`);
  for (const visual of VISUALS) if (!pages.some((entry) => entry.visual === visual)) throw new Error(`Unused visual family: ${visual}`);
}
validateCatalog(PAGES);

// ── styling ──────────────────────────────────────────────────────────────────

const CSS = `
* { box-sizing: border-box; }
@keyframes fdry-flow { to { stroke-dashoffset: -32; } }
@keyframes fdry-pulse { 0%,100% { opacity:.5 } 50% { opacity:1 } }
@keyframes fdry-spin { to { transform: rotate(360deg); } }
@keyframes fdry-throb { 0%,100% { opacity:.35 } 50% { opacity:.9 } }
@keyframes fdry-rise { from { opacity:0; transform: translateY(6px) } to { opacity:1; transform: translateY(0) } }
.fdry-flow { animation: fdry-flow 1.7s linear infinite; }
.fdry-pulse { transform-box: fill-box; transform-origin: center; animation: fdry-pulse 1.8s ease-in-out infinite; }
.fdry-spin { transform-box: fill-box; transform-origin: center; animation: fdry-spin 9s linear infinite; }
.fdry-throb { animation: fdry-throb 2.2s ease-in-out infinite; }
.fdry-specimen { transition: transform .32s cubic-bezier(.3,.7,.3,1); transform-box: view-box; }
.fdry-part { cursor: pointer; outline: none; }
.fdry-part:focus-visible { outline: 2px solid; outline-offset: 2px; border-radius: 4px; }
.fdry-root { min-height: 100%; padding: 16px; outline: none; }
.fdry-wrap { width: min(1400px, 100%); margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.fdry-eyebrow { font: 700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .15em; text-transform: uppercase; }
.fdry-title { margin: 3px 0 0; font: 620 clamp(19px,2.4vw,28px)/1.04 ui-sans-serif, system-ui, sans-serif; letter-spacing: -.02em; }
.fdry-head { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; }
.fdry-strip { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 3px; }
.fdry-seg { flex: 1 1 0; min-width: 108px; border: 1px solid; border-radius: 9px; padding: 7px 10px; background: transparent; color: inherit; cursor: pointer; text-align: left; }
.fdry-seg-label { width: 100%; border: 0; padding: 0; background: transparent; color: inherit; cursor: pointer; text-align: left; }
.fdry-seg small { font: 500 8px ui-monospace, monospace; opacity: .6; }
.fdry-seg strong { display: block; font: 620 10.5px/1.2 ui-sans-serif, system-ui, sans-serif; margin-top: 2px; }
.fdry-ticks { display: flex; gap: 3px; margin-top: 6px; flex-wrap: wrap; }
.fdry-tick { width: 8px; height: 8px; border-radius: 50%; border: 1px solid; background: transparent; padding: 0; cursor: pointer; }
.fdry-stage { position: relative; border: 1px solid; border-radius: 12px; overflow: hidden; animation: fdry-rise .24s ease-out both; }
.fdry-drawer { position: absolute; top: 0; right: 0; height: 100%; width: min(354px, 88%); border-left: 1px solid; padding: 15px; overflow-y: auto; z-index: 6; animation: fdry-rise .18s ease-out both; }
.fdry-foot { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.fdry-hint { font: 500 10px/1.4 ui-monospace, monospace; }
@media (prefers-reduced-motion: reduce) {
  .fdry-flow, .fdry-pulse, .fdry-spin, .fdry-throb, .fdry-stage, .fdry-drawer { animation: none !important; }
  .fdry-specimen { transition: none !important; }
}
@media (max-width: 820px) {
  .fdry-drawer { width: 100%; height: auto; max-height: 66%; top: auto; bottom: 0; border-left: 0; border-top: 1px solid; }
  .fdry-seg { min-width: 90px; }
  .fdry-head { align-items: flex-start; }
}
`;

const stageStyle = (t: Theme) =>
  ({ width: "100%", height: "clamp(348px,54vh,592px)", display: "block", color: t.accent.primary } as const);

const SPECIMEN_STAGE: Record<ChapterId, string> = {
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

// ── shared atoms ─────────────────────────────────────────────────────────────

function activate(e: Evt, fn: () => void) {
  if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
    if (e.key !== "Enter") e.preventDefault();
    fn();
  }
}

function FoundryDefs({ t }: { t: Theme }) {
  return (
    <defs>
      <marker id="fa" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={t.accent.primary} /></marker>
      <marker id="fo" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={t.category.orange} /></marker>
      <marker id="fg" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={t.category.green} /></marker>
      <marker id="fp" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill={t.category.purple} /></marker>
    </defs>
  );
}

function Chamber({ t, label, children }: { t: Theme; label: string; children: unknown }) {
  return (
    <svg viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet" role="group" aria-label={label} style={stageStyle(t)}>
      {children as ReturnType<typeof FoundryDefs>}
    </svg>
  );
}

function Bay({
  t, x, y, w, h, label, sub, active, tone, onSelect,
}: {
  key?: string; t: Theme; x: number; y: number; w: number; h: number; label: string; sub?: string; active: boolean; tone?: string; onSelect: () => void;
}) {
  const stroke = active ? tone ?? t.accent.primary : t.stroke.primary;
  return (
    <g className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${label}`} onClick={onSelect} onKeyDown={(e: Evt) => activate(e, onSelect)}>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={active ? t.fill.secondary : t.fill.tertiary} stroke={stroke} strokeWidth={active ? 2.2 : 1.2} />
      <text x={x + w / 2} y={y + h / 2 - (sub ? 6 : 0)} textAnchor="middle" dominantBaseline="middle" fontSize={13} fontWeight={600} fill={t.text.primary}>{label}</text>
      {sub ? <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>{sub}</text> : null}
    </g>
  );
}

function Port({
  t, cx, cy, r, label, active, tone, onSelect,
}: {
  key?: string; t: Theme; cx: number; cy: number; r: number; label: string; active: boolean; tone?: string; onSelect: () => void;
}) {
  const stroke = active ? tone ?? t.accent.primary : t.stroke.primary;
  return (
    <g className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${label}`} onClick={onSelect} onKeyDown={(e: Evt) => activate(e, onSelect)}>
      <circle cx={cx} cy={cy} r={r} fill={active ? t.fill.secondary : t.fill.tertiary} stroke={stroke} strokeWidth={active ? 2.2 : 1.2} />
      <text x={cx} y={cy + r + 15} textAnchor="middle" fontSize={10.5} fontWeight={active ? 650 : 500} fill={active ? t.text.primary : t.text.secondary}>{label}</text>
    </g>
  );
}

function Conduit({ t, d, on, tone, marker }: { key?: string; t: Theme; d: string; on: boolean; tone?: string; marker?: string }) {
  return (
    <path d={d} fill="none" stroke={on ? tone ?? t.accent.primary : t.stroke.secondary} strokeWidth={on ? 2.4 : 1.4} strokeDasharray="8 8" className={on ? "fdry-flow" : undefined} markerEnd={on ? `url(#${marker ?? "fa"})` : undefined} />
  );
}

function Gauge({ t, cx, cy, r, value, label, active, tone, onSelect }: { key?: string; t: Theme; cx: number; cy: number; r: number; value: number; label: string; active: boolean; tone?: string; onSelect: () => void }) {
  const v = Math.max(0, Math.min(100, value));
  const ang = Math.PI * (1 - v / 100);
  const nx = cx + Math.cos(ang) * (r - 9);
  const ny = cy - Math.sin(ang) * (r - 9);
  const arc = `M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r},${cy}`;
  const accent = active ? t.accent.primary : tone ?? t.category.green;
  return (
    <g className="fdry-part" role="button" tabIndex={0} aria-label={`${label}: ${v} percent`} onClick={onSelect} onKeyDown={(e: Evt) => activate(e, onSelect)}>
      <path d={arc} fill="none" stroke={t.stroke.primary} strokeWidth={7} />
      <path d={arc} fill="none" stroke={accent} strokeWidth={7} strokeDasharray={`${(v / 100) * Math.PI * r} ${Math.PI * r}`} />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={active ? t.accent.primary : t.text.primary} strokeWidth={2.6} />
      <circle cx={cx} cy={cy} r={4} fill={t.text.primary} />
      <text x={cx} y={cy + 22} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>{label}</text>
      <text x={cx} y={cy + 38} textAnchor="middle" fontSize={13} fontWeight={700} fill={accent}>{v}%</text>
    </g>
  );
}

function Lamp({ t, cx, cy, label, on, onSelect }: { key?: string; t: Theme; cx: number; cy: number; label: string; on: boolean; onSelect: () => void }) {
  return (
    <g className="fdry-part" role="button" tabIndex={0} aria-label={`${label}: ${on ? "ready" : "off"}`} onClick={onSelect} onKeyDown={(e: Evt) => activate(e, onSelect)}>
      <circle cx={cx} cy={cy} r={26} fill={on ? t.fill.secondary : t.fill.tertiary} stroke={on ? t.category.green : t.stroke.primary} strokeWidth={on ? 2.4 : 1.2} className={on ? "fdry-throb" : undefined} />
      <circle cx={cx} cy={cy} r={9} fill={on ? t.category.green : t.stroke.secondary} />
      <text x={cx} y={cy + 44} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={t.text.primary}>{label}</text>
    </g>
  );
}

function Cap({ t, text }: { t: Theme; text: string }) {
  return <text x={500} y={532} textAnchor="middle" fontSize={10.5} fill={t.text.tertiary}>{text}</text>;
}

function specimenShape(chapter: ChapterId, t: Theme) {
  const c = { fill: t.accent.control, stroke: t.accent.primary, strokeWidth: 1.8 } as const;
  switch (chapter) {
    case "orientation": return <circle cx={0} cy={0} r={10} {...c} />;
    case "ingress": return <g><rect x={-12} y={-8} width={24} height={16} rx={3} {...c} /><line x1={-7} y1={-2} x2={7} y2={-2} stroke={t.accent.primary} /><line x1={-7} y1={3} x2={4} y2={3} stroke={t.accent.primary} /></g>;
    case "llm": return <g>{[-7, 0, 7].map((o) => <rect key={o} x={-12 + Math.abs(o) / 2} y={o - 2.5} width={24 - Math.abs(o)} height={5} rx={2} {...c} />)}</g>;
    case "specialized": return <polygon points="0,-12 12,0 0,12 -12,0" {...c} />;
    case "grpc": return <polygon points="-10,-8 0,-13 10,-8 10,8 0,13 -10,8" {...c} />;
    case "timing": return <g>{[-9, 0, 9].map((o) => <circle key={o} cx={o} cy={0} r={4} {...c} />)}</g>;
    case "scheduler": return <g>{[-9, 0, 9].map((o, i) => <rect key={o} x={o - 3.5} y={-5 - i * 2} width={7} height={10 + i * 4} rx={2} {...c} />)}</g>;
    case "semantics": return <g><circle cx={0} cy={0} r={11} {...c} /><path d="M-7,0 L-1,6 L8,-6" fill="none" stroke={t.accent.primary} strokeWidth={2.4} /></g>;
    case "observability": return <g><path d="M-11,6 A12,12 0 0 1 11,6" fill={t.accent.control} stroke={t.accent.primary} strokeWidth={1.8} /><line x1={0} y1={6} x2={7} y2={-4} stroke={t.accent.primary} strokeWidth={2.4} /></g>;
    case "proof": return <g><circle cx={-7} cy={0} r={6} {...c} /><circle cx={7} cy={0} r={6} {...c} /><line x1={-1} y1={0} x2={1} y2={0} stroke={t.accent.primary} strokeWidth={2.4} /></g>;
  }
}

function Specimen({ t, chapter, x, y }: { t: Theme; chapter: ChapterId; x: number; y: number }) {
  return (
    <g className="fdry-specimen" style={{ transform: `translate(${x}px,${y}px)` }} aria-hidden="true">
      <g className="fdry-pulse">{specimenShape(chapter, t)}</g>
    </g>
  );
}

function ControlDock({ t, page, step, mode, onStep, onMode }: { t: Theme; page: FeaturePage; step: number; mode: string; onStep: () => void; onMode: (v: string) => void }) {
  const stepLabel = page.steps[step % page.steps.length];
  let mx = 452;
  return (
    <g>
      <line x1={20} y1={548} x2={980} y2={548} stroke={t.stroke.tertiary} />
      <g className="fdry-part" role="button" tabIndex={0} aria-label={`Advance mechanism; current state ${stepLabel}`} onClick={onStep} onKeyDown={(e: Evt) => activate(e, onStep)}>
        <circle cx={44} cy={574} r={18} fill={t.fill.secondary} stroke={t.accent.primary} strokeWidth={1.6} />
        <polygon points="38,565 38,583 52,574" fill={t.accent.primary} />
      </g>
      <text x={70} y={569} fontSize={8.5} fill={t.text.tertiary} className="fdry-eyebrow">STATE</text>
      <text x={70} y={583} fontSize={11.5} fontWeight={600} fill={t.text.primary}>{stepLabel}</text>
      {page.modes.length > 1
        ? page.modes.map((m) => {
            const w = Math.max(58, m.length * 6.6 + 18);
            const bx = mx;
            mx += w + 7;
            const on = mode === m;
            return (
              <g key={m} className="fdry-part" role="button" tabIndex={0} aria-pressed={on} aria-label={`Compare mode ${m}`} onClick={() => onMode(m)} onKeyDown={(e: Evt) => activate(e, () => onMode(m))}>
                <rect x={bx} y={560} width={w} height={28} rx={7} fill={on ? t.accent.control : t.fill.tertiary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 1.8 : 1} />
                <text x={bx + w / 2} y={577} textAnchor="middle" fontSize={10} fontWeight={on ? 700 : 500} fill={on ? t.text.onAccent : t.text.secondary}>{m}</text>
              </g>
            );
          })
        : null}
    </g>
  );
}

type SceneProps = {
  t: Theme;
  page: FeaturePage;
  step: number;
  mode: string;
  selected: string;
  onSelect: (v: string) => void;
  onStep: () => void;
  onMode: (v: string) => void;
};

function ghost(t: Theme, x: number, y: number, w: number, h: number, label: string) {
  return (
    <g aria-hidden="true">
      <rect x={x} y={y} width={w} height={h} rx={7} fill={t.fill.quaternary} stroke={t.stroke.tertiary} strokeDasharray="4 4" />
      <text x={x + w / 2} y={y + h / 2 + 3} textAnchor="middle" fontSize={9} fill={t.text.quaternary}>{label}</text>
    </g>
  );
}

// ── Chapter 1 · Orientation — process cutaway ─────────────────────────────────

function OrientationFoundry(p: SceneProps) {
  const { t, page, step, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — process cutaway`;
  const dock = <ControlDock t={t} page={page} step={step} mode={p.mode} onStep={p.onStep} onMode={p.onMode} />;
  const cap = <Cap t={t} text={page.steps[step % page.steps.length]} />;

  switch (page.id) {
    case "process-boundary": {
      const at = [[150, 245], [500, 245], [800, 285]][Math.min(2, step % 3)];
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {ghost(t, 640, 130, 300, 300, "")}
          <text x={790} y={150} textAnchor="middle" fontSize={9} fill={t.text.quaternary}>mock process image</text>
          {ghost(t, 665, 175, 250, 44, "listener")}
          {ghost(t, 665, 232, 250, 44, "router")}
          {ghost(t, 665, 289, 250, 44, "handler + generation")}
          <line x1={520} y1={100} x2={520} y2={470} stroke={t.stroke.primary} strokeDasharray="3 6" />
          <text x={520} y={92} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>process boundary</text>
          <Conduit t={t} d="M230,245 L430,245" on={step >= 1} />
          <Conduit t={t} d="M580,245 L660,255" on={step >= 2} />
          <Bay t={t} x={60} y={205} w={170} h={90} label="aiperf" sub="external driver" active={sel("aiperf")} onSelect={() => onSelect("aiperf")} />
          <Bay t={t} x={430} y={210} w={90} h={70} label="HTTP" sub="or gRPC" active={sel("HTTP or gRPC")} onSelect={() => onSelect("HTTP or gRPC")} />
          <Bay t={t} x={640} y={130} w={300} h={300} label="" active={sel("mock process")} onSelect={() => onSelect("mock process")} />
          <text x={790} y={415} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>aiperf-mock-server process</text>
          <Specimen t={t} chapter="orientation" x={at[0]} y={at[1] - 15} />
          {cap}{dock}
        </Chamber>
      );
    }
    case "crate-dependency": {
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Conduit t={t} d="M330,200 L470,240" on tone={t.category.green} marker="fg" />
          <text x={400} y={205} fontSize={9.5} fill={t.category.green}>depends on</text>
          <path d="M470,410 L330,410" fill="none" stroke={t.category.red} strokeWidth={1.8} strokeDasharray="6 6" />
          <line x1={392} y1={396} x2={408} y2={424} stroke={t.category.red} strokeWidth={2.4} />
          <line x1={408} y1={396} x2={392} y2={424} stroke={t.category.red} strokeWidth={2.4} />
          <text x={400} y={440} fontSize={9.5} fill={t.category.red}>no dependency</text>
          <Bay t={t} x={470} y={170} w={220} h={90} label="aiperf-mock-server" sub="test target" active={sel("aiperf-mock-server")} onSelect={() => onSelect("aiperf-mock-server")} />
          <Bay t={t} x={110} y={170} w={220} h={90} label="aiperf-runtime" sub="library" active={sel("aiperf-runtime")} onSelect={() => onSelect("aiperf-runtime")} />
          <Bay t={t} x={110} y={370} w={220} h={90} label="product runner" sub="aiperf-cli path" active={sel("product runner")} onSelect={() => onSelect("product runner")} />
          <Bay t={t} x={470} y={370} w={220} h={90} label="aiperf-mock-server" sub="not on hot path" active={sel("aiperf-mock-server")} tone={t.category.red} onSelect={() => onSelect("aiperf-mock-server")} />
          <Specimen t={t} chapter="orientation" x={selected === "product runner" ? 220 : 400} y={selected === "product runner" ? 415 : 215} />
          <Cap t={t} text="The runner never links the mock; the mock is an ordinary target." />{dock}
        </Chamber>
      );
    }
    case "startup": {
      const multi = step % 2 === 1;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Conduit t={t} d="M180,150 L180,215" on marker="fa" />
          <Conduit t={t} d="M180,300 L180,340" on marker="fa" />
          <Conduit t={t} d="M235,373 L470,300" on={multi} tone={t.category.orange} marker="fo" />
          <Conduit t={t} d="M235,373 L470,440" on={!multi} marker="fa" />
          <Conduit t={t} d="M690,300 L760,360" on={multi} tone={t.category.orange} marker="fo" />
          <Conduit t={t} d="M690,440 L760,380" on={!multi} marker="fa" />
          <Bay t={t} x={95} y={110} w={170} h={44} label="parse config" active={sel("parse config")} onSelect={() => onSelect("parse config")} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect process gate" onClick={() => onSelect("process gate")} onKeyDown={(e: Evt) => activate(e, () => onSelect("process gate"))}>
            <polygon points="180,215 245,258 180,300 115,258" fill={sel("process gate") ? t.fill.secondary : t.fill.tertiary} stroke={sel("process gate") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("process gate") ? 2.2 : 1.2} />
            <text x={180} y={262} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>--processes?</text>
          </g>
          <Bay t={t} x={470} y={278} w={220} h={44} label="balancer (N > 1)" sub="round-robin parent" active={multi} tone={t.category.orange} onSelect={() => onSelect("process gate")} />
          <Bay t={t} x={470} y={418} w={220} h={44} label="single-process listeners" active={!multi} onSelect={() => onSelect("build state")} />
          <Bay t={t} x={760} y={338} w={170} h={84} label="build state" sub="then serve" active={sel("build state") || sel("serve")} onSelect={() => onSelect("serve")} />
          <Specimen t={t} chapter="orientation" x={180} y={180} />
          <Cap t={t} text={multi ? "N > 1 → L4 balancer branch (advance to compare)" : "N = 1 → listeners branch (advance to compare)"} />{dock}
        </Chamber>
      );
    }
    case "application-state": {
      const leaves = ["config", "token generator", "metrics", "prefix cache", "accuracy"];
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {leaves.map((n, i) => {
            const a = (Math.PI * 2 * i) / leaves.length - Math.PI / 2;
            const lx = 500 + Math.cos(a) * 300;
            const ly = 300 + Math.sin(a) * 165;
            return <g key={n}><Conduit t={t} d={`M500,300 L${lx},${ly}`} on={sel(n) || (selected === "" && i <= step % leaves.length)} tone={n === "config" ? t.category.gray : t.accent.primary} /></g>;
          })}
          <circle cx={500} cy={300} r={62} fill={t.fill.secondary} stroke={t.accent.primary} strokeWidth={2} className="fdry-spin" strokeDasharray="10 8" />
          <text x={500} y={296} textAnchor="middle" fontSize={12} fontWeight={650} fill={t.text.primary}>AppState</text>
          <text x={500} y={312} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>Arc clones</text>
          {leaves.map((n, i) => {
            const a = (Math.PI * 2 * i) / leaves.length - Math.PI / 2;
            const lx = 500 + Math.cos(a) * 300;
            const ly = 300 + Math.sin(a) * 165;
            return <Port key={n} t={t} cx={lx} cy={ly} r={30} label={n} active={sel(n)} tone={n === "config" ? t.category.gray : undefined} onSelect={() => onSelect(n)} />;
          })}
          <Specimen t={t} chapter="orientation" x={500} y={230} />
          <Cap t={t} text="Immutable config is shared read-only; generators, metrics, cache, and accuracy are synchronized." />{dock}
        </Chamber>
      );
    }
    case "request-journey": {
      const chain = page.nodes;
      const ai = step % chain.length;
      const xs = chain.map((_, i) => 90 + (i * 830) / (chain.length - 1));
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={70} y={250} width={870} height={90} rx={12} fill={t.fill.quaternary} stroke={t.stroke.primary} />
          <path d="M100,295 L910,295" stroke={t.accent.primary} strokeWidth={2} strokeDasharray="9 9" className="fdry-flow" markerEnd="url(#fa)" />
          {chain.map((n, i) => (
            <Bay key={n} t={t} x={xs[i] - 55} y={i % 2 ? 175 : 355} w={110} h={54} label={n} active={sel(n) || (selected === "" && i === ai)} onSelect={() => onSelect(n)} />
          ))}
          {chain.map((n, i) => <line key={n} x1={xs[i]} y1={i % 2 ? 229 : 355} x2={xs[i]} y2={i % 2 ? 250 : 340} stroke={sel(n) || (selected === "" && i === ai) ? t.accent.primary : t.stroke.secondary} />)}
          <Specimen t={t} chapter="orientation" x={xs[ai]} y={295} />
          <Cap t={t} text={`${page.steps[ai % page.steps.length]} · one server process, edge to edge`} />{dock}
        </Chamber>
      );
    }
    case "architecture-atlas": {
      const clusters = [
        { n: "entry", x: 180, y: 160 },
        { n: "protocols", x: 500, y: 130 },
        { n: "timing", x: 820, y: 200 },
        { n: "state", x: 340, y: 380 },
        { n: "proof", x: 720, y: 400 },
      ];
      const ai = step % clusters.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {clusters.map((c, i) => clusters.slice(i + 1).map((d) => <line key={c.n + d.n} x1={c.x} y1={c.y} x2={d.x} y2={d.y} stroke={t.stroke.tertiary} strokeDasharray="2 7" />))}
          {clusters.map((c, i) => (
            <g key={c.n}>
              {Array.from({ length: 4 }).map((_, k) => <rect key={k} x={c.x - 44 + (k % 2) * 46} y={c.y + 32 + Math.floor(k / 2) * 18} width={40} height={13} rx={3} fill={t.fill.tertiary} stroke={t.stroke.tertiary} />)}
              <Port t={t} cx={c.x} cy={c.y} r={34} label={c.n} active={sel(c.n) || (selected === "" && i === ai)} onSelect={() => onSelect(c.n)} />
            </g>
          ))}
          <Specimen t={t} chapter="orientation" x={clusters[ai].x} y={clusters[ai].y - 52} />
          <Cap t={t} text="Source files form a testable feature graph, not one monolithic handler." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 2 · Ingress — listener / TLS / route manifold ─────────────────────

function IngressManifold(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — ingress manifold`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "config": {
      const layers = page.nodes;
      const ai = step % layers.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {layers.map((n, i) => {
            const on = sel(n) || (selected === "" && i === ai);
            const y = 120 + i * 88;
            const inset = i * 40;
            return (
              <g key={n} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
                <rect x={130 + inset} y={y} width={620 - inset} height={64} rx={8} fill={on ? t.fill.secondary : i % 2 ? t.fill.tertiary : t.fill.quaternary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 2.2 : 1.2} />
                <text x={150 + inset} y={y + 27} fontSize={12} fontWeight={600} fill={t.text.primary}>{i + 1}. {n}</text>
                <text x={150 + inset} y={y + 46} fontSize={9.5} fill={t.text.tertiary}>{["clap flag", "AIPERF_* twin", "typed struct", "read at runtime"][i]}</text>
              </g>
            );
          })}
          <text x={790} y={150} fontSize={9.5} fill={t.category.yellow}>definition ≠ wiring</text>
          <text x={790} y={168} fontSize={9} fill={t.text.tertiary}>a defined flag is</text>
          <text x={790} y={182} fontSize={9} fill={t.text.tertiary}>not proof of use</text>
          <Specimen t={t} chapter="ingress" x={770} y={152 + ai * 88} />
          <Cap t={t} text="Clap definitions are authoritative, but definition alone does not prove wiring." />{dock}
        </Chamber>
      );
    }
    case "tcp-listener": {
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <path d="M120,180 L360,270 L360,330 L120,420 Z" fill={t.fill.quaternary} stroke={t.stroke.primary} />
          <text x={200} y={305} fontSize={9.5} fill={t.text.tertiary}>accept funnel</text>
          {chain.map((n, i) => <Conduit key={n} t={t} d={`M${170 + i * 200},300 L${330 + i * 200},300`} on={i < ai} />)}
          {chain.map((n, i) => <Bay key={n} t={t} x={90 + i * 200} y={272} w={130} h={56} label={n} active={sel(n) || (selected === "" && i === ai)} onSelect={() => onSelect(n)} />)}
          <Specimen t={t} chapter="ingress" x={155 + ai * 200} y={250} />
          <Cap t={t} text="The tuned Hyper listener accepts TCP and serves the shared router." />{dock}
        </Chamber>
      );
    }
    case "http2": {
      const h2 = mode === "h2c";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Conduit t={t} d="M200,300 L360,300" on marker="fa" />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect protocol switch" onClick={() => onSelect(h2 ? "h2c" : "HTTP/1.1")} onKeyDown={(e: Evt) => activate(e, () => onSelect(h2 ? "h2c" : "HTTP/1.1"))}>
            <circle cx={400} cy={300} r={30} fill={t.fill.secondary} stroke={t.accent.primary} strokeWidth={1.8} />
            <line x1={400} y1={300} x2={h2 ? 430 : 425} y2={h2 ? 320 : 278} stroke={t.accent.primary} strokeWidth={3} />
          </g>
          <text x={400} y={352} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>negotiate</text>
          <Conduit t={t} d="M430,282 L560,210" on={!h2} />
          <Conduit t={t} d="M430,318 L560,392" on={h2} />
          <Bay t={t} x={70} y={272} w={130} h={56} label="TCP" active={sel("TCP")} onSelect={() => onSelect("TCP")} />
          <Bay t={t} x={560} y={182} w={200} h={56} label="HTTP/1.1" active={sel("HTTP/1.1") || (!h2 && selected === "")} onSelect={() => onSelect("HTTP/1.1")} />
          <Bay t={t} x={560} y={364} w={200} h={56} label="h2c" sub="honors max stream limit" active={sel("h2c") || (h2 && selected === "")} onSelect={() => onSelect("h2c")} />
          <Bay t={t} x={800} y={272} w={130} h={56} label="request" active={sel("request")} onSelect={() => onSelect("request")} />
          <Conduit t={t} d="M760,210 L800,285" on={!h2} />
          <Conduit t={t} d="M760,392 L800,315" on={h2} />
          <Specimen t={t} chapter="ingress" x={h2 ? 620 : 620} y={h2 ? 392 : 210} />
          <Cap t={t} text="One connection builder negotiates HTTP behavior and honors the configured h2 stream limit." />{dock}
        </Chamber>
      );
    }
    case "uds": {
      const collide = mode === "path collision";
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={70} y={120} width={860} height={2} fill={t.stroke.tertiary} />
          <text x={90} y={112} fontSize={9} fill={t.text.quaternary}>main TCP plane (not used here)</text>
          {chain.map((n, i) => <Conduit key={n} t={t} d={`M${180 + i * 210},330 L${350 + i * 210},330`} on={i < ai && !collide} tone={t.category.pink} marker="fp" />)}
          {chain.map((n, i) => <Bay key={n} t={t} x={90 + i * 210} y={302} w={140} h={56} label={n} active={sel(n) || (selected === "" && i === ai)} tone={t.category.pink} onSelect={() => onSelect(n)} />)}
          {collide ? (
            <g>
              <line x1={168} y1={318} x2={196} y2={346} stroke={t.category.red} strokeWidth={3} />
              <line x1={196} y1={318} x2={168} y2={346} stroke={t.category.red} strokeWidth={3} />
              <text x={182} y={400} textAnchor="middle" fontSize={10} fill={t.category.red}>refuses to unlink a non-socket path</text>
            </g>
          ) : null}
          <Specimen t={t} chapter="ingress" x={collide ? 160 : 160 + ai * 210} y={310} />
          <Cap t={t} text="UDS serves HTTP/1.1 directly and refuses to delete a non-socket path." />{dock}
        </Chamber>
      );
    }
    case "tls": {
      const h2 = mode === "HTTP/2";
      const gates = page.nodes;
      const ai = step % gates.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {gates.map((n, i) => {
            const x = 130 + i * 210;
            const open = i < ai;
            return (
              <g key={n}>
                <Conduit t={t} d={`M${x + 90},300 L${x + 120},300`} on={open} />
                <line x1={x} y1={open ? 200 : 240} x2={x} y2={360} stroke={open ? t.accent.primary : t.stroke.primary} strokeWidth={3} />
                <Bay t={t} x={x - 60} y={365} w={130} h={48} label={n} active={sel(n) || (selected === "" && i === ai)} onSelect={() => onSelect(n)} />
              </g>
            );
          })}
          <text x={640} y={200} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>ALPN advertises: {h2 ? "h2, http/1.1" : "http/1.1"}</text>
          <Specimen t={t} chapter="ingress" x={165 + ai * 210} y={300} />
          <Cap t={t} text="TLS wraps accepted streams and advertises HTTP protocols through ALPN." />{dock}
        </Chamber>
      );
    }
    case "router": {
      const routes = page.nodes;
      const ai = step % routes.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={90} y={272} w={150} h={64} label="inbound" sub="one connection" active={sel("inbound")} onSelect={() => onSelect("inbound")} />
          {routes.map((n, i) => {
            const y = 110 + i * 82;
            const on = sel(n) || (selected === "" && i === ai);
            return (
              <g key={n}>
                <path d={`M240,304 C360,304 380,${y + 27} 500,${y + 27}`} fill="none" stroke={on ? t.accent.primary : t.stroke.secondary} strokeWidth={on ? 2.2 : 1.4} strokeDasharray="8 8" className={on ? "fdry-flow" : undefined} markerEnd={on ? "url(#fa)" : undefined} />
                <Bay t={t} x={500} y={y} w={220} h={54} label={n} sub="distinct wire shape" active={on} onSelect={() => onSelect(n)} />
              </g>
            );
          })}
          <line x1={760} y1={110} x2={760} y2={520} stroke={t.stroke.tertiary} strokeDasharray="3 6" />
          <text x={800} y={300} fontSize={9.5} fill={t.text.tertiary}>shared</text>
          <text x={800} y={314} fontSize={9.5} fill={t.text.tertiary}>state</text>
          <Specimen t={t} chapter="ingress" x={470} y={137 + ai * 82} />
          <Cap t={t} text="All HTTP dialects converge on shared state while retaining distinct wire shapes." />{dock}
        </Chamber>
      );
    }
    case "models-health": {
      const health = Math.min(100, 60 + step * 12);
      const models = page.metrics[3] ?? 12;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Gauge t={t} cx={300} cy={300} r={92} value={health} label="GET /health · 200" active={sel("GET /health")} onSelect={() => onSelect("GET /health")} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect model discovery" onClick={() => onSelect("GET /v1/models")} onKeyDown={(e: Evt) => activate(e, () => onSelect("GET /v1/models"))}>
            <rect x={600} y={220} width={280} height={160} rx={12} fill={sel("GET /v1/models") ? t.fill.secondary : t.fill.tertiary} stroke={sel("GET /v1/models") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("GET /v1/models") ? 2.2 : 1.2} />
            <text x={740} y={252} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>GET /v1/models</text>
            <text x={740} y={318} textAnchor="middle" fontSize={30} fontWeight={700} fill={t.accent.primary}>{models}</text>
            <text x={740} y={352} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>seen models (advance)</text>
          </g>
          <Specimen t={t} chapter="ingress" x={sel("GET /v1/models") ? 740 : 300} y={sel("GET /v1/models") ? 200 : 200} />
          <Cap t={t} text="Health is a direct liveness response; observed model names extend model discovery." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 3 · LLM protocols — transparent glassworks ────────────────────────

function ProtocolGlassworks(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — protocol glassworks`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  // A transparent pipe carrying page-specific frame shapes.
  const Pipe = ({ y }: { y: number }) => (
    <g aria-hidden="true">
      <rect x={70} y={y - 46} width={860} height={92} rx={46} fill={t.fill.quaternary} stroke={t.stroke.primary} />
      <path d={`M100,${y} L910,${y}`} stroke={t.accent.primary} strokeWidth={1.6} strokeDasharray="6 10" className="fdry-flow" opacity={0.55} />
    </g>
  );

  const framePipe = (nodes: readonly string[], y: number, subs: readonly string[]) => {
    const ai = step % nodes.length;
    const xs = nodes.map((_, i) => 130 + (i * 740) / Math.max(1, nodes.length - 1));
    return (
      <g>
        <Pipe y={y} />
        {nodes.map((n, i) => {
          const on = sel(n) || (selected === "" && i === ai);
          return (
            <g key={n} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect frame ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
              <rect x={xs[i] - 62} y={y - 34} width={124} height={68} rx={7} fill={on ? t.accent.control : t.fill.secondary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 2 : 1} opacity={i > ai && selected === "" ? 0.5 : 1} />
              <text x={xs[i]} y={y - 8} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={on ? t.text.onAccent : t.text.primary}>{n}</text>
              <text x={xs[i]} y={y + 12} textAnchor="middle" fontSize={8.5} fill={on ? t.text.onAccent : t.text.tertiary}>{subs[i] ?? ""}</text>
            </g>
          );
        })}
        <Specimen t={t} chapter="llm" x={xs[ai]} y={y - 62} />
      </g>
    );
  };

  switch (page.id) {
    case "chat": {
      const stream = mode !== "non-stream";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {stream
            ? framePipe(page.nodes, 300, ["POST /v1/chat", "delta.content", "assembled text", "finish_reason"])
            : (
              <g>
                <Pipe y={300} />
                <Bay t={t} x={360} y={250} w={280} h={100} label="single JSON body" sub="choices[0].message" active={sel("content")} onSelect={() => onSelect("content")} />
                <Specimen t={t} chapter="llm" x={500} y={230} />
              </g>
            )}
          <Cap t={t} text={stream ? "Streaming: choice deltas flow through the glass pipe." : "Non-stream: one settled JSON body, no deltas."} />{dock}
        </Chamber>
      );
    }
    case "completions":
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{framePipe(page.nodes, 300, ["prompt in", "choices[].text", "usage block", "[DONE]"])}<Cap t={t} text="Legacy completions carry text choices, not chat message deltas." />{dock}</Chamber>;
    case "sse": {
      const nodes = page.nodes;
      const ai = step % nodes.length;
      const xs = nodes.map((_, i) => 120 + (i * 760) / (nodes.length - 1));
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Pipe y={300} />
          {nodes.map((n, i) => {
            const on = i <= ai;
            const isTerm = n === "usage frame" || n === "[DONE]";
            return (
              <g key={n} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
                <rect x={xs[i] - 52} y={isTerm ? 258 : 270} width={104} height={isTerm ? 84 : 60} rx={7} fill={sel(n) ? t.accent.control : on ? t.fill.secondary : t.fill.tertiary} stroke={sel(n) ? t.accent.primary : isTerm ? t.category.yellow : on ? t.accent.primary : t.stroke.primary} strokeWidth={sel(n) ? 2 : 1.2} opacity={on ? 1 : 0.5} />
                <text x={xs[i]} y={300} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={sel(n) ? t.text.onAccent : t.text.primary}>{n}</text>
              </g>
            );
          })}
          <text x={640} y={230} textAnchor="middle" fontSize={9.5} fill={t.category.yellow}>usage arrives ~0 ms after the last token</text>
          <Specimen t={t} chapter="llm" x={xs[ai]} y={240} />
          <Cap t={t} text="Generated token events precede terminal usage and the stream terminator." />{dock}
        </Chamber>
      );
    }
    case "terminal-usage": {
      const vals = page.metrics;
      const max = Math.max(...vals, 1);
      const ai = step % vals.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <line x1={90} y1={330} x2={910} y2={330} stroke={t.stroke.primary} strokeWidth={2} />
          {vals.map((v, i) => {
            const x = 100 + (v / max) * 780;
            const on = sel(page.nodes[i]) || (selected === "" && i === ai);
            const usage = page.nodes[i] === "usage";
            return (
              <g key={i} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${page.nodes[i]} at ${v} ms`} onClick={() => onSelect(page.nodes[i])} onKeyDown={(e: Evt) => activate(e, () => onSelect(page.nodes[i]))}>
                <line x1={x} y1={usage ? 250 : 300} x2={x} y2={360} stroke={on ? t.accent.primary : usage ? t.category.yellow : t.stroke.secondary} strokeWidth={on ? 3 : 1.6} />
                <circle cx={x} cy={330} r={on ? 9 : 6} fill={on ? t.accent.primary : usage ? t.category.yellow : t.fill.primary} stroke={t.stroke.focused} className={on ? "fdry-pulse" : undefined} />
                <text x={x} y={250 - (i % 2) * 18} textAnchor="middle" fontSize={9} fill={on ? t.text.primary : t.text.tertiary}>{page.nodes[i]}</text>
                <text x={x} y={382} textAnchor="middle" fontSize={9} fill={t.text.secondary}>{v} ms</text>
              </g>
            );
          })}
          <text x={500} y={420} textAnchor="middle" fontSize={9.5} fill={t.category.yellow}>usage and done share a timestamp — accounting, not a timing sample</text>
          <Specimen t={t} chapter="llm" x={100 + (vals[ai] / max) * 780} y={310} />
          <Cap t={t} text="Usage is terminal accounting, not a generated token timing sample." />{dock}
        </Chamber>
      );
    }
    case "messages":
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{framePipe(page.nodes, 300, ["cache usage", "text delta", "stop reason", "final stop"])}<Cap t={t} text="Anthropic event names and cache usage fields are emitted in Anthropic shapes." />{dock}</Chamber>;
    case "responses":
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{framePipe(page.nodes, 300, ["response id", "output_text", "completed"])}<Cap t={t} text="Responses emits response-scoped output events, not chat completion chunks." />{dock}</Chamber>;
    case "reasoning": {
      const ai = step % 3;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Pipe y={210} />
          <Pipe y={390} />
          <text x={120} y={175} fontSize={9.5} fill={t.category.purple}>reasoning lane</text>
          <text x={120} y={355} fontSize={9.5} fill={t.text.tertiary}>visible output lane</text>
          <Bay t={t} x={220} y={176} w={220} h={68} label="reasoning" sub="reasoning_content" active={sel("reasoning") || (selected === "" && ai === 0)} tone={t.category.purple} onSelect={() => onSelect("reasoning")} />
          <Bay t={t} x={220} y={356} w={220} h={68} label="visible output" sub="content" active={sel("visible output") || (selected === "" && ai === 1)} onSelect={() => onSelect("visible output")} />
          <Conduit t={t} d="M440,210 L640,300" on tone={t.category.purple} />
          <Conduit t={t} d="M440,390 L640,300" on />
          <Bay t={t} x={640} y={266} w={240} h={68} label="usage reconciliation" active={sel("usage reconciliation") || (selected === "" && ai === 2)} onSelect={() => onSelect("usage reconciliation")} />
          <Specimen t={t} chapter="llm" x={ai === 0 ? 330 : ai === 1 ? 330 : 760} y={ai === 0 ? 155 : ai === 1 ? 335 : 245} />
          <Cap t={t} text="Reasoning and output content remain distinguishable in supported response shapes." />{dock}
        </Chamber>
      );
    }
    case "vllm-generate": {
      const tokens = mode !== "text";
      const ai = step % page.nodes.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Pipe y={300} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect input token ids" onClick={() => onSelect("input token ids")} onKeyDown={(e: Evt) => activate(e, () => onSelect("input token ids"))}>
            {Array.from({ length: 6 }).map((_, i) => <rect key={i} x={110 + i * 26} y={280} width={22} height={40} rx={3} fill={tokens ? t.accent.control : t.fill.tertiary} stroke={sel("input token ids") ? t.accent.primary : t.stroke.primary} />)}
            <text x={188} y={340} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>{tokens ? "input token ids" : "input text"}</text>
          </g>
          <Conduit t={t} d="M280,300 L400,300" on={ai >= 1} />
          <Bay t={t} x={400} y={266} w={200} h={68} label="generation" active={sel("generation") || (selected === "" && ai === 1)} onSelect={() => onSelect("generation")} />
          <Conduit t={t} d="M600,300 L700,300" on={ai >= 2} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect output token ids" onClick={() => onSelect("output token ids")} onKeyDown={(e: Evt) => activate(e, () => onSelect("output token ids"))}>
            {Array.from({ length: 6 }).map((_, i) => <rect key={i} x={710 + i * 26} y={280} width={22} height={40} rx={3} fill={sel("output token ids") ? t.accent.control : t.fill.secondary} stroke={sel("output token ids") ? t.accent.primary : t.category.green} />)}
            <text x={788} y={340} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>{tokens ? "output token ids" : "output text"}</text>
          </g>
          <Specimen t={t} chapter="llm" x={[188, 500, 788][ai]} y={250} />
          <Cap t={t} text="The vLLM route accepts token-native input and returns generated token data." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 4 · Specialized — looms / sorters / chambers ──────────────────────

function EndpointWorks(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — endpoint works`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "embeddings": {
      const rows = mode === "batch" ? 3 : 1;
      const cols = 8;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={80} y={250} w={150} h={90} label="input text" active={sel("input text")} onSelect={() => onSelect("input text")} />
          <Conduit t={t} d="M230,295 L300,295" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect deterministic seed" onClick={() => onSelect("deterministic seed")} onKeyDown={(e: Evt) => activate(e, () => onSelect("deterministic seed"))}>
            {Array.from({ length: 7 }).map((_, i) => <line key={i} x1={310} y1={230 + i * 22} x2={430} y2={230 + i * 22} stroke={sel("deterministic seed") ? t.accent.primary : t.stroke.primary} />)}
            <text x={370} y={410} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>loom · deterministic seed</text>
          </g>
          <Conduit t={t} d="M440,295 L500,295" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect FP vector" onClick={() => onSelect("FP vector")} onKeyDown={(e: Evt) => activate(e, () => onSelect("FP vector"))}>
            {Array.from({ length: rows * cols }).map((_, i) => {
              const c = i % cols;
              const r = Math.floor(i / cols);
              return <rect key={i} x={510 + c * 46} y={240 + r * 42} width={40} height={34} rx={4} fill={sel("FP vector") ? t.accent.control : t.fill.tertiary} stroke={sel("FP vector") ? t.accent.primary : t.category.green} />;
            })}
            <text x={510 + (cols * 46) / 2} y={240 + rows * 42 + 22} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>FP vector · shape [{rows}, {cols}]</text>
          </g>
          <Specimen t={t} chapter="specialized" x={sel("FP vector") ? 700 : 370} y={220} />
          <Cap t={t} text="Embedding dimensions and values are deterministic for the same input." />{dock}
        </Chamber>
      );
    }
    case "rankings": {
      const order = mode === "rankings" ? [2, 0, 3, 1] : [0, 2, 1, 3];
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={90} y={180} w={150} h={60} label="query" active={sel("query")} onSelect={() => onSelect("query")} />
          <text x={370} y={150} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>documents in</text>
          <text x={700} y={150} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>sorter → rank order</text>
          {[0, 1, 2, 3].map((d) => (
            <g key={d}>
              <Bay t={t} x={300} y={190 + d * 68} w={130} h={52} label={`doc ${d}`} sub={sel("documents") ? "candidate" : undefined} active={sel("documents")} onSelect={() => onSelect("documents")} />
              <Conduit t={t} d={`M430,${216 + d * 68} L610,${216 + order.indexOf(d) * 68}`} on tone={t.category.orange} marker="fo" />
            </g>
          ))}
          {order.map((d, rank) => (
            <Bay key={`r${rank}`} t={t} x={610} y={190 + rank * 68} w={150} h={52} label={`#${rank + 1} · doc ${d}`} active={sel("scores") || sel("rank order")} tone={t.category.orange} onSelect={() => onSelect("rank order")} />
          ))}
          <Specimen t={t} chapter="specialized" x={520} y={216} />
          <Cap t={t} text="Ranking routes produce dialect-specific score and index shapes from shared deterministic scoring." />{dock}
        </Chamber>
      );
    }
    case "tgi": {
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={70} y={210} width={860} height={2} fill={t.stroke.tertiary} />
          <text x={90} y={200} fontSize={9} fill={t.text.quaternary}>TGI request shape</text>
          <rect x={70} y={410} width={860} height={2} fill={t.stroke.tertiary} />
          <text x={90} y={432} fontSize={9} fill={t.text.quaternary}>shared generation seam</text>
          {chain.map((n, i) => {
            const x = 120 + (i * 760) / (chain.length - 1);
            const on = sel(n) || (selected === "" && i === ai);
            return (
              <g key={n}>
                <Bay t={t} x={x - 70} y={250} w={140} h={54} label={n} active={on} onSelect={() => onSelect(n)} />
                <Conduit t={t} d={`M${x},304 L${x},380`} on={on} />
                <circle cx={x} cy={390} r={7} fill={on ? t.accent.primary : t.stroke.secondary} />
              </g>
            );
          })}
          <Specimen t={t} chapter="specialized" x={120 + (ai * 760) / (chain.length - 1)} y={230} />
          <Cap t={t} text="TGI request and stream shapes map onto the same generation seam." />{dock}
        </Chamber>
      );
    }
    case "image-generation": {
      const b64 = mode === "base64";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={80} y={260} w={150} h={80} label="prompt" active={sel("prompt")} onSelect={() => onSelect("prompt")} />
          <Conduit t={t} d="M230,300 L320,300" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect image generator" onClick={() => onSelect("image generator")} onKeyDown={(e: Evt) => activate(e, () => onSelect("image generator"))}>
            <rect x={320} y={230} width={200} height={160} rx={10} fill={sel("image generator") ? t.fill.secondary : t.fill.tertiary} stroke={sel("image generator") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("image generator") ? 2.2 : 1.2} />
            {Array.from({ length: 16 }).map((_, i) => <rect key={i} x={340 + (i % 4) * 42} y={250 + Math.floor(i / 4) * 34} width={36} height={28} rx={2} fill={i % 3 === 0 ? t.fill.primary : t.fill.tertiary} stroke={t.stroke.tertiary} />)}
            <text x={420} y={410} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>deterministic chamber</text>
          </g>
          <Conduit t={t} d="M520,300 L620,300" on />
          <Bay t={t} x={620} y={260} w={260} h={80} label={b64 ? "base64 payload" : "URL-like reference"} sub="encoded image" active={sel("encoded image")} onSelect={() => onSelect("encoded image")} />
          <Specimen t={t} chapter="specialized" x={sel("encoded image") ? 750 : 420} y={210} />
          <Cap t={t} text="Image output is deterministic and returned in the requested endpoint shape." />{dock}
        </Chamber>
      );
    }
    case "image-edit": {
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={180} w={170} h={90} label="source image" active={sel("source image")} onSelect={() => onSelect("source image")} />
          <Bay t={t} x={70} y={330} w={170} h={90} label="mask" active={sel("mask")} onSelect={() => onSelect("mask")} />
          <Conduit t={t} d="M240,225 L420,280" on />
          <Conduit t={t} d="M240,375 L420,320" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect edit chamber" onClick={() => onSelect("prompt")} onKeyDown={(e: Evt) => activate(e, () => onSelect("prompt"))}>
            <rect x={420} y={250} width={180} height={100} rx={10} fill={sel("prompt") ? t.fill.secondary : t.fill.tertiary} stroke={sel("prompt") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("prompt") ? 2.2 : 1.2} />
            <text x={510} y={296} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>edit chamber</text>
            <text x={510} y={314} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>+ prompt</text>
          </g>
          <Conduit t={t} d="M600,300 L700,300" on />
          <Bay t={t} x={700} y={250} w={220} h={100} label="edited output" active={sel("edited output")} onSelect={() => onSelect("edited output")} />
          <Specimen t={t} chapter="specialized" x={510} y={230} />
          <Cap t={t} text="Image and mask inputs feed the edit response path." />{dock}
        </Chamber>
      );
    }
    case "image-retrieval": {
      const vals = page.metrics;
      const max = Math.max(...vals, 1);
      const ai = step % vals.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <line x1={90} y1={210} x2={910} y2={210} stroke={t.stroke.primary} strokeWidth={2} />
          <text x={90} y={196} fontSize={9} fill={t.text.tertiary}>retrieval delay track (independent of generation)</text>
          {vals.map((v, i) => {
            const x = 100 + (v / max) * 780;
            const on = sel(page.nodes[i % page.nodes.length]) || (selected === "" && i === ai);
            return (
              <g key={i} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${page.nodes[i % page.nodes.length]} at ${v} ms`} onClick={() => onSelect(page.nodes[i % page.nodes.length])} onKeyDown={(e: Evt) => activate(e, () => onSelect(page.nodes[i % page.nodes.length]))}>
                <line x1={x} y1={190} x2={x} y2={230} stroke={on ? t.accent.primary : t.stroke.secondary} strokeWidth={on ? 3 : 1.6} />
                <circle cx={x} cy={210} r={on ? 8 : 5} fill={on ? t.accent.primary : t.fill.primary} stroke={t.stroke.focused} />
                <text x={x} y={252} textAnchor="middle" fontSize={9} fill={t.text.secondary}>{v} ms</text>
              </g>
            );
          })}
          {Array.from({ length: 4 }).map((_, i) => (
            <g key={i}><rect x={160 + i * 180} y={310} width={140} height={110} rx={8} fill={t.fill.tertiary} stroke={t.stroke.primary} /><text x={230 + i * 180} y={370} textAnchor="middle" fontSize={10} fill={t.text.secondary}>ranked image {i + 1}</text></g>
          ))}
          <Specimen t={t} chapter="specialized" x={100 + (vals[ai] / max) * 780} y={190} />
          <Cap t={t} text="Image retrieval latency is independently configurable from generation." />{dock}
        </Chamber>
      );
    }
    case "multimodal": {
      const parts = mode === "audio" ? ["text part", "audio part"] : mode === "video" ? ["text part", "video part"] : ["text part", "image part"];
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={200} y={150} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>content parts</text>
          {parts.map((n, i) => (
            <g key={n}>
              <Bay t={t} x={90} y={190 + i * 110} w={220} h={80} label={n} active={sel(n) || sel(page.nodes[i])} onSelect={() => onSelect(n)} />
              <Conduit t={t} d={`M310,${230 + i * 110} L440,300`} on />
            </g>
          ))}
          <Bay t={t} x={440} y={258} w={180} h={84} label="message" sub="composer" active={sel("message")} onSelect={() => onSelect("message")} />
          <Conduit t={t} d="M620,300 L710,300" on />
          <Bay t={t} x={710} y={258} w={210} h={84} label="response" sub="shared machinery" active={sel("response")} onSelect={() => onSelect("response")} />
          <Specimen t={t} chapter="specialized" x={530} y={238} />
          <Cap t={t} text="Text and media content parts are parsed without changing the shared response machinery." />{dock}
        </Chamber>
      );
    }
    case "rag-kserve-http": {
      const chain = page.nodes;
      const ai = step % chain.length;
      const xs = chain.map((_, i) => 130 + (i * 740) / (chain.length - 1));
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={500} y={170} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>alias: {mode}</text>
          {chain.map((n, i) => (
            <g key={n}>
              {i < chain.length - 1 ? <Conduit t={t} d={`M${xs[i] + 60},300 L${xs[i + 1] - 60},300`} on={i < ai} /> : null}
              <Bay t={t} x={xs[i] - 70} y={266} w={140} h={68} label={n} active={sel(n) || (selected === "" && i === ai)} onSelect={() => onSelect(n)} />
            </g>
          ))}
          <Specimen t={t} chapter="specialized" x={xs[ai]} y={246} />
          <Cap t={t} text="RAG and KServe HTTP aliases remain HTTP routes over shared handlers." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 5 · gRPC / Riva — protobuf switching yard + transducers ───────────

function GrpcSwitchyard(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — protobuf switching yard`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  const Wagon = ({ x, y, w, name, on, onClick }: { key?: string; x: number; y: number; w: number; name: string; on: boolean; onClick: () => void }) => (
    <g className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${name}`} onClick={onClick} onKeyDown={(e: Evt) => activate(e, onClick)}>
      <rect x={x} y={y} width={w} height={58} rx={7} fill={on ? t.fill.secondary : t.fill.tertiary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 2.2 : 1.2} />
      <text x={x + w / 2} y={y + 34} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>{name}</text>
      <circle cx={x + 16} cy={y + 66} r={7} fill={t.fill.primary} stroke={t.stroke.primary} />
      <circle cx={x + w - 16} cy={y + 66} r={7} fill={t.fill.primary} stroke={t.stroke.primary} />
    </g>
  );

  const rails = (y: number) => <g aria-hidden="true"><line x1={60} y1={y} x2={940} y2={y} stroke={t.stroke.primary} /><line x1={60} y1={y + 6} x2={940} y2={y + 6} stroke={t.stroke.primary} /></g>;

  switch (page.id) {
    case "grpc-unary": {
      const chain = page.nodes;
      const ai = step % chain.length;
      const xs = chain.map((_, i) => 100 + i * 220);
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {rails(360)}
          {chain.map((n, i) => <Wagon key={n} x={xs[i]} y={295} w={180} name={n} on={sel(n) || (selected === "" && i === ai)} onClick={() => onSelect(n)} />)}
          {chain.slice(0, -1).map((_, i) => <Conduit key={`c${i}`} t={t} d={`M${xs[i] + 180},324 L${xs[i + 1]},324`} on={i < ai} />)}
          <Specimen t={t} chapter="grpc" x={xs[ai] + 90} y={275} />
          <Cap t={t} text="Unary protobuf tensors are decoded and returned on the KServe inference service." />{dock}
        </Chamber>
      );
    }
    case "grpc-stream": {
      const items = ["stream item", "stream item", "stream item", "terminal item"];
      const shown = Math.min(items.length, (step % items.length) + 1);
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {rails(360)}
          <Bay t={t} x={70} y={295} w={130} h={58} label="RPC headers" active={sel("RPC headers")} onSelect={() => onSelect("RPC headers")} />
          {items.map((n, i) => {
            const term = i === items.length - 1;
            const show = i < shown;
            return <g key={i} opacity={show ? 1 : 0.35}><Wagon x={230 + i * 150} y={295} w={130} name={term ? "terminal" : `item ${i + 1}`} on={sel(term ? "terminal item" : "stream item") && show} onClick={() => onSelect(term ? "terminal item" : "stream item")} /></g>;
          })}
          <Bay t={t} x={840} y={295} w={100} h={58} label="trailers" active={sel("trailers")} onSelect={() => onSelect("trailers")} />
          <Specimen t={t} chapter="grpc" x={230 + (shown - 1) * 150 + 65} y={275} />
          <Cap t={t} text="The streaming RPC emits a sequence of protobuf inference responses." />{dock}
        </Chamber>
      );
    }
    case "grpc-readiness":
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {["ModelReady", "ServerLive", "ServerReady"].map((n, i) => <Lamp key={n} t={t} cx={280 + i * 220} cy={290} label={n} on={sel(n) || selected === ""} onSelect={() => onSelect(n)} />)}
          <Specimen t={t} chapter="grpc" x={280 + (step % 3) * 220} y={240} />
          <Cap t={t} text="ModelReady, ServerLive, and ServerReady expose deterministic readiness." />{dock}
        </Chamber>
      );
    case "grpc-embeddings": {
      const dim = mode === "N=8" ? 8 : mode === "N=32" ? 32 : 128;
      const cols = 16;
      const shown = Math.min(dim, 48);
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {rails(180)}
          <Wagon x={90} y={130} w={200} name="STRING input" on={sel("STRING input")} onClick={() => onSelect("STRING input")} />
          <Conduit t={t} d="M290,159 L370,159" on />
          <Bay t={t} x={370} y={130} w={200} h={58} label="embedding generator" active={sel("embedding generator")} onSelect={() => onSelect("embedding generator")} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect FP32 tensor" onClick={() => onSelect("FP32 [1,N]")} onKeyDown={(e: Evt) => activate(e, () => onSelect("FP32 [1,N]"))}>
            {Array.from({ length: shown }).map((_, i) => <rect key={i} x={120 + (i % cols) * 46} y={290 + Math.floor(i / cols) * 34} width={40} height={28} rx={3} fill={sel("FP32 [1,N]") ? t.accent.control : t.fill.tertiary} stroke={sel("FP32 [1,N]") ? t.accent.primary : t.category.green} />)}
            <text x={500} y={290 + Math.ceil(shown / cols) * 34 + 22} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>FP32 tensor · shape [1, {dim}] · no token semantics</text>
          </g>
          <Specimen t={t} chapter="grpc" x={480} y={110} />
          <Cap t={t} text="Embedding mode returns FP32 shape [1, configured dimension] with no token semantics." />{dock}
        </Chamber>
      );
    }
    case "grpc-rankings": {
      const rank = mode === "rankings";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {rails(230)}
          <Wagon x={90} y={180} w={190} name="query tensor" on={sel("query tensor")} onClick={() => onSelect("query tensor")} />
          <Wagon x={310} y={180} w={210} name="candidate tensor" on={sel("candidate tensor")} onClick={() => onSelect("candidate tensor")} />
          <Conduit t={t} d="M520,209 L620,209" on tone={rank ? t.accent.primary : t.stroke.secondary} />
          <Bay t={t} x={620} y={200} w={130} h={60} label="scores" active={sel("scores") || rank} tone={rank ? t.accent.primary : undefined} onSelect={() => onSelect("scores")} />
          <Bay t={t} x={620} y={330} w={130} h={60} label="indices" active={sel("indices") || rank} tone={rank ? t.accent.primary : undefined} onSelect={() => onSelect("indices")} />
          <text x={500} y={430} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>{rank ? "ranking behavior selected → ranking tensors" : "auto behavior → generated text (advance mode to rankings)"}</text>
          <Specimen t={t} chapter="grpc" x={rank ? 685 : 415} y={rank ? 300 : 160} />
          <Cap t={t} text="Ranking behavior selects ranking tensors instead of generated text." />{dock}
        </Chamber>
      );
    }
    case "grpc-images": {
      const out = mode === "images" ? "image tensor" : mode === "text" ? "generated text" : "auto-selected";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {rails(300)}
          <Wagon x={90} y={270} w={190} name="input tensors" on={sel("input tensors")} onClick={() => onSelect("input tensors")} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect behavior gate" onClick={() => onSelect("behavior gate")} onKeyDown={(e: Evt) => activate(e, () => onSelect("behavior gate"))}>
            <polygon points="400,255 470,299 400,343 330,299" fill={sel("behavior gate") ? t.fill.secondary : t.fill.tertiary} stroke={sel("behavior gate") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("behavior gate") ? 2.2 : 1.2} />
            <text x={400} y={303} textAnchor="middle" fontSize={10} fontWeight={600} fill={t.text.primary}>behavior</text>
          </g>
          <Conduit t={t} d="M470,299 L560,299" on />
          <Bay t={t} x={560} y={270} w={280} h={58} label={out} sub="image or VLM output" active={sel("image or VLM output")} onSelect={() => onSelect("image or VLM output")} />
          <Specimen t={t} chapter="grpc" x={sel("image or VLM output") ? 700 : 400} y={250} />
          <Cap t={t} text="Behavior selection changes output tensor types while preserving the KServe RPC." />{dock}
        </Chamber>
      );
    }
    case "riva-asr-tts": {
      const tts = mode === "TTS";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect audio bytes" onClick={() => onSelect("audio bytes")} onKeyDown={(e: Evt) => activate(e, () => onSelect("audio bytes"))}>
            {Array.from({ length: 18 }).map((_, i) => { const h = 14 + Math.abs(Math.sin(i * 1.3)) * 60; return <rect key={i} x={110 + i * 12} y={300 - h / 2} width={7} height={h} rx={2} fill={sel("audio bytes") ? t.accent.primary : t.category.cyan} opacity={0.8} />; })}
            <text x={220} y={380} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>audio bytes</text>
          </g>
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect transducer" onClick={() => onSelect(tts ? "TTS waveform" : "ASR transcript")} onKeyDown={(e: Evt) => activate(e, () => onSelect(tts ? "TTS waveform" : "ASR transcript"))}>
            <rect x={400} y={250} width={180} height={100} rx={12} fill={t.fill.secondary} stroke={t.accent.primary} strokeWidth={1.8} />
            <text x={490} y={296} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>transducer</text>
            <text x={490} y={314} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>{tts ? "text → speech" : "speech → text"}</text>
          </g>
          <Conduit t={t} d={tts ? "M580,300 L640,300" : "M400,300 L340,300"} on marker={tts ? "fa" : "fa"} />
          <Bay t={t} x={640} y={250} w={230} h={100} label={tts ? "TTS waveform" : "ASR transcript"} active={sel(tts ? "TTS waveform" : "ASR transcript")} onSelect={() => onSelect(tts ? "TTS waveform" : "ASR transcript")} />
          <Specimen t={t} chapter="grpc" x={490} y={230} />
          <Cap t={t} text="Riva speech services are exposed only through gRPC service methods." />{dock}
        </Chamber>
      );
    }
    case "riva-nlp-boundary": {
      const grpc = mode !== "HTTP absent";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={70} y={140} width={380} height={180} rx={12} fill={t.fill.quaternary} stroke={t.stroke.primary} />
          <text x={260} y={168} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>HTTP router</text>
          <line x1={110} y1={220} x2={410} y2={300} stroke={t.category.red} strokeWidth={2.4} />
          <line x1={410} y1={220} x2={110} y2={300} stroke={t.category.red} strokeWidth={2.4} />
          <text x={260} y={350} textAnchor="middle" fontSize={9.5} fill={t.category.red}>no Riva route</text>
          <Bay t={t} x={70} y={140} w={380} h={180} label="" active={sel("HTTP router")} tone={t.category.red} onSelect={() => onSelect("HTTP router")} />
          <Bay t={t} x={560} y={160} w={360} h={70} label="gRPC Riva service" active={sel("gRPC Riva service") || grpc} tone={t.category.green} onSelect={() => onSelect("gRPC Riva service")} />
          <Conduit t={t} d="M740,230 L740,300" on={grpc} tone={t.category.green} marker="fg" />
          <Bay t={t} x={560} y={300} w={360} h={70} label="NLP response" active={sel("NLP response")} tone={t.category.green} onSelect={() => onSelect("NLP response")} />
          <Specimen t={t} chapter="grpc" x={740} y={140} />
          <Cap t={t} text="Riva ASR, TTS, and NLP have no HTTP route in the mock router." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 6 · Timing — the escapement ───────────────────────────────────────

function TimingEscapement(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — timing escapement`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "tokenizer-truth": {
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={500} y={150} textAnchor="middle" fontSize={9.5} fill={t.category.yellow}>character / corpus path — no Hugging Face tokenizer loaded</text>
          {chain.map((n, i) => {
            const x = 120 + (i * 760) / (chain.length - 1);
            const on = sel(n) || (selected === "" && i === ai);
            return (
              <g key={n}>
                {i < chain.length - 1 ? <Conduit t={t} d={`M${x + 60},300 L${x + 160},300`} on={i < ai} tone={t.category.yellow} marker="fo" /> : null}
                <Bay t={t} x={x - 60} y={266} w={140} h={68} label={n} active={on} tone={t.category.yellow} onSelect={() => onSelect(n)} />
              </g>
            );
          })}
          <Specimen t={t} chapter="timing" x={120 + (ai * 760) / (chain.length - 1)} y={246} />
          <Cap t={t} text="Rust token generation is character/corpus based; it does not load a Hugging Face tokenizer." />{dock}
        </Chamber>
      );
    }
    case "tokenizer-flags": {
      const gate = mode === "corpus gate";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={150} w={230} h={120} label="HF identity flags" sub="--tokenizer / revision / trust" active={sel("HF identity flags")} tone={t.category.red} onSelect={() => onSelect("HF identity flags")} />
          <path d="M300,210 L470,210" fill="none" stroke={t.category.red} strokeWidth={2} strokeDasharray="6 6" />
          <line x1={378} y1={196} x2={394} y2={224} stroke={t.category.red} strokeWidth={2.4} />
          <line x1={394} y1={196} x2={378} y2={224} stroke={t.category.red} strokeWidth={2.4} />
          <Bay t={t} x={470} y={150} w={200} h={120} label="not consumed" active={sel("not consumed")} tone={t.category.red} onSelect={() => onSelect("not consumed")} />
          <Bay t={t} x={70} y={340} w={230} h={110} label="--no-tokenizer" active={sel("--no-tokenizer")} onSelect={() => onSelect("--no-tokenizer")} />
          <Conduit t={t} d="M300,395 L470,395" on={gate} />
          <Bay t={t} x={470} y={340} w={200} h={110} label="corpus gate" sub="only gates corpus load" active={sel("corpus gate") || gate} onSelect={() => onSelect("corpus gate")} />
          <Conduit t={t} d="M670,395 L760,340" on={gate} />
          <Bay t={t} x={760} y={280} w={180} h={110} label="character tokenizer" active={sel("character tokenizer")} onSelect={() => onSelect("character tokenizer")} />
          <Specimen t={t} chapter="timing" x={gate ? 560 : 180} y={gate ? 320 : 130} />
          <Cap t={t} text="HF identity flags are not consumed; --no-tokenizer only gates corpus loading." />{dock}
        </Chamber>
      );
    }
    case "token-budgets": {
      const dist = mode === "distributed";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={80} y={260} w={140} h={80} label="seed" active={sel("seed")} onSelect={() => onSelect("seed")} />
          <Conduit t={t} d="M220,290 L300,240" on />
          <Conduit t={t} d="M220,310 L300,360" on />
          <Bay t={t} x={300} y={210} w={180} h={60} label="ISL draw" sub={dist ? "distribution" : "fixed"} active={sel("ISL draw")} onSelect={() => onSelect("ISL draw")} />
          <Bay t={t} x={300} y={330} w={180} h={60} label="OSL draw" sub={dist ? "distribution" : "fixed"} active={sel("OSL draw")} onSelect={() => onSelect("OSL draw")} />
          <Conduit t={t} d="M480,240 L560,290" on />
          <Conduit t={t} d="M480,360 L560,310" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect generated corpus" onClick={() => onSelect("generated corpus")} onKeyDown={(e: Evt) => activate(e, () => onSelect("generated corpus"))}>
            {Array.from({ length: 24 }).map((_, i) => <rect key={i} x={560 + (i % 8) * 44} y={250 + Math.floor(i / 8) * 34} width={38} height={28} rx={3} fill={sel("generated corpus") ? t.accent.control : t.fill.tertiary} stroke={sel("generated corpus") ? t.accent.primary : t.stroke.primary} />)}
            <text x={720} y={368} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>generated corpus · repeatable under seed</text>
          </g>
          <Specimen t={t} chapter="timing" x={410} y={dist ? 200 : 200} />
          <Cap t={t} text="Seeded inputs and fixed distributions yield repeatable input and output budgets." />{dock}
        </Chamber>
      );
    }
    case "ttft-itl": {
      const vals = page.metrics;
      const max = Math.max(...vals, 1);
      const ai = step % vals.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <line x1={90} y1={330} x2={910} y2={330} stroke={t.stroke.primary} strokeWidth={2} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect TTFT gate" onClick={() => onSelect("TTFT")} onKeyDown={(e: Evt) => activate(e, () => onSelect("TTFT"))}>
            <line x1={90 + (vals[1] / max) * 780} y1={230} x2={90 + (vals[1] / max) * 780} y2={330} stroke={sel("TTFT") ? t.accent.primary : t.category.orange} strokeWidth={4} />
            <text x={90 + (vals[1] / max) * 780} y={218} textAnchor="middle" fontSize={9.5} fill={t.category.orange}>TTFT gate</text>
          </g>
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect ITL gate" onClick={() => onSelect("ITL")} onKeyDown={(e: Evt) => activate(e, () => onSelect("ITL"))}>
            <line x1={90 + (vals[3] / max) * 780} y1={230} x2={90 + (vals[3] / max) * 780} y2={330} stroke={sel("ITL") ? t.accent.primary : t.category.purple} strokeWidth={4} />
            <text x={90 + (vals[3] / max) * 780} y={218} textAnchor="middle" fontSize={9.5} fill={t.category.purple}>ITL gate</text>
          </g>
          {vals.map((v, i) => {
            const x = 90 + (v / max) * 780;
            const on = sel(page.nodes[i]) || (selected === "" && i === ai);
            return (
              <g key={i} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${page.nodes[i]} at ${v} ms`} onClick={() => onSelect(page.nodes[i])} onKeyDown={(e: Evt) => activate(e, () => onSelect(page.nodes[i]))}>
                <circle cx={x} cy={330} r={on ? 9 : 6} fill={on ? t.accent.primary : t.fill.primary} stroke={t.stroke.focused} className={on ? "fdry-pulse" : undefined} />
                <text x={x} y={360} textAnchor="middle" fontSize={8.5} fill={t.text.secondary}>{page.nodes[i]}</text>
                <text x={x} y={376} textAnchor="middle" fontSize={8.5} fill={t.text.tertiary}>{v} ms</text>
              </g>
            );
          })}
          <text x={500} y={430} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>the escapement releases token 1 at TTFT, then every ITL — advance to scrub</text>
          <Specimen t={t} chapter="timing" x={90 + (vals[ai] / max) * 780} y={310} />
          <Cap t={t} text="First-token delay and generated-token gaps are independently paced." />{dock}
        </Chamber>
      );
    }
    case "latency-scaling": {
      const loaded = mode === "loaded";
      const terms = page.nodes;
      const base = [40, 30, 45, loaded ? 70 : 12];
      const max = Math.max(...base, 1);
      let acc = 90;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={90} y={160} fontSize={9.5} fill={t.text.tertiary}>analytic latency = stacked terms</text>
          {terms.map((n, i) => {
            const w = (base[i] / max) * 620;
            const x = acc;
            acc += w + 6;
            const on = sel(n) || (selected === "" && i === step % terms.length);
            return (
              <g key={n} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
                <rect x={x} y={250} width={w} height={90} rx={7} fill={on ? t.accent.control : i % 2 ? t.fill.secondary : t.fill.tertiary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 2 : 1} />
                <text x={x + w / 2} y={300} textAnchor="middle" fontSize={10} fontWeight={600} fill={on ? t.text.onAccent : t.text.primary}>{n}</text>
              </g>
            );
          })}
          <text x={90} y={400} fontSize={9.5} fill={loaded ? t.category.orange : t.text.tertiary}>{loaded ? "loaded: concurrency term dominates" : "base: minimal concurrency contribution"}</text>
          <Specimen t={t} chapter="timing" x={acc + 20} y={295} />
          <Cap t={t} text="Configured ISL, OSL, and concurrency coefficients alter analytic latency." />{dock}
        </Chamber>
      );
    }
    case "jitter-timerfd": {
      const jit = mode === "seeded jitter";
      const ai = step % page.nodes.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={266} w={150} h={68} label="latency sample" active={sel("latency sample") || ai === 0} onSelect={() => onSelect("latency sample")} />
          <Conduit t={t} d="M220,300 L320,300" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect seeded CV" onClick={() => onSelect("seeded CV")} onKeyDown={(e: Evt) => activate(e, () => onSelect("seeded CV"))}>
            <rect x={320} y={266} width={160} height={68} rx={8} fill={sel("seeded CV") ? t.fill.secondary : t.fill.tertiary} stroke={sel("seeded CV") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("seeded CV") ? 2.2 : 1.2} />
            <path d={jit ? "M330,300 q10,-24 20,0 t20,0 t20,0 t20,0 t20,0" : "M330,300 h150"} fill="none" stroke={jit ? t.category.orange : t.category.green} strokeWidth={2} />
            <text x={400} y={352} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>{jit ? "seeded CV > 0" : "zero jitter"}</text>
          </g>
          <Conduit t={t} d="M480,300 L580,300" on />
          <Bay t={t} x={580} y={266} w={150} h={68} label="sleep_ns" sub="timerfd precision" active={sel("sleep_ns") || ai === 2} onSelect={() => onSelect("sleep_ns")} />
          <Conduit t={t} d="M730,300 L800,300" on />
          <Bay t={t} x={800} y={266} w={130} h={68} label="emit" active={sel("emit") || ai === 3} onSelect={() => onSelect("emit")} />
          <Specimen t={t} chapter="timing" x={[145, 400, 655, 865][ai]} y={246} />
          <Cap t={t} text="Seeded jitter is reproducible and pacing sleeps through RealClock timerfd precision." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 7 · Scheduler — conveyor + cache library ─────────────────────────

function SchedulerFloor(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — foundry floor`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "prefill-decode": {
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={60} y={200} width={880} height={70} rx={10} fill={t.fill.quaternary} stroke={t.stroke.primary} />
          <text x={80} y={192} fontSize={9} fill={t.text.tertiary}>prefill conveyor</text>
          <path d="M90,235 L910,235" stroke={t.category.orange} strokeWidth={2} strokeDasharray="9 9" className="fdry-flow" markerEnd="url(#fo)" />
          <rect x={60} y={330} width={880} height={70} rx={10} fill={t.fill.quaternary} stroke={t.stroke.primary} />
          <text x={80} y={322} fontSize={9} fill={t.text.tertiary}>decode conveyor</text>
          <path d="M910,365 L90,365" stroke={t.category.green} strokeWidth={2} strokeDasharray="9 9" className="fdry-flow" markerEnd="url(#fg)" />
          {chain.map((n, i) => {
            const top = i < 3;
            const x = top ? 120 + i * 260 : 700 - (i - 3) * 260;
            const y = top ? 235 : 365;
            const on = sel(n) || (selected === "" && i === ai);
            return <Bay key={n} t={t} x={x - 70} y={y - 27} w={140} h={54} label={n} active={on} tone={top ? t.category.orange : t.category.green} onSelect={() => onSelect(n)} />;
          })}
          <Specimen t={t} chapter="scheduler" x={ai < 3 ? 120 + ai * 260 : 700 - (ai - 3) * 260} y={ai < 3 ? 210 : 405} />
          <Cap t={t} text="Scheduler ticks admit prefill work and emit decode tokens under configured capacities." />{dock}
        </Chamber>
      );
    }
    case "batch-saturation": {
      const knee = mode === "at knee" ? 71 : mode === "above knee" ? 62 : 38;
      const cap = 76;
      const occ = mode === "below knee" ? 3 : mode === "at knee" ? 6 : 6;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={90} y={200} width={520} height={90} rx={10} fill={t.fill.quaternary} stroke={t.stroke.primary} />
          <text x={110} y={192} fontSize={9} fill={t.text.tertiary}>active batch (capacity {6})</text>
          {Array.from({ length: 6 }).map((_, i) => <rect key={i} x={110 + i * 84} y={215} width={70} height={60} rx={7} fill={i < occ ? t.fill.secondary : t.fill.tertiary} stroke={i < occ ? t.accent.primary : t.stroke.primary} />)}
          {mode === "above knee" ? <g><rect x={630} y={215} width={90} height={60} rx={7} fill={t.fill.tertiary} stroke={t.category.red} strokeDasharray="5 5" /><text x={675} y={250} textAnchor="middle" fontSize={9} fill={t.category.red}>overflow</text></g> : null}
          <Bay t={t} x={90} y={330} w={140} h={54} label="waiting" active={sel("waiting")} onSelect={() => onSelect("waiting")} />
          <Bay t={t} x={280} y={330} w={140} h={54} label="active batch" active={sel("active batch")} onSelect={() => onSelect("active batch")} />
          <Bay t={t} x={470} y={330} w={140} h={54} label="capacity" active={sel("capacity")} onSelect={() => onSelect("capacity")} />
          <Bay t={t} x={660} y={330} w={140} h={54} label="overflow" active={sel("overflow")} tone={t.category.red} onSelect={() => onSelect("overflow")} />
          <Gauge t={t} cx={840} cy={270} r={72} value={Math.round((knee / cap) * 100)} label="throughput knee" active={selected === ""} tone={t.category.orange} onSelect={() => onSelect("capacity")} />
          <Specimen t={t} chapter="scheduler" x={110 + Math.min(occ, 5) * 84 + 35} y={180} />
          <Cap t={t} text="The maximum batch size creates a visible admission and throughput knee." />{dock}
        </Chamber>
      );
    }
    case "goodput-collapse": {
      const vals = page.metrics;
      const max = Math.max(...vals, 1);
      const pts = vals.map((v, i) => `${90 + (i * 780) / (vals.length - 1)},${430 - (v / max) * 260}`).join(" ");
      const ai = step % vals.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <line x1={90} y1={430} x2={910} y2={430} stroke={t.stroke.primary} />
          <line x1={90} y1={150} x2={90} y2={430} stroke={t.stroke.primary} />
          <text x={70} y={150} textAnchor="end" fontSize={9} fill={t.text.tertiary}>rate</text>
          <polyline points={pts} fill="none" stroke={t.category.orange} strokeWidth={2.4} />
          {vals.map((v, i) => {
            const x = 90 + (i * 780) / (vals.length - 1);
            const y = 430 - (v / max) * 260;
            const lbl = page.nodes[i % page.nodes.length];
            const on = sel(lbl) || (selected === "" && i === ai);
            return (
              <g key={i} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${lbl}`} onClick={() => onSelect(lbl)} onKeyDown={(e: Evt) => activate(e, () => onSelect(lbl))}>
                <circle cx={x} cy={y} r={on ? 9 : 5} fill={on ? t.accent.primary : t.fill.primary} stroke={t.stroke.focused} className={on ? "fdry-pulse" : undefined} />
                <text x={x} y={y - 14} textAnchor="middle" fontSize={8.5} fill={on ? t.text.primary : t.text.tertiary}>{lbl}</text>
              </g>
            );
          })}
          <text x={640} y={220} fontSize={9.5} fill={t.category.red}>service rate collapses past saturation</text>
          <Specimen t={t} chapter="scheduler" x={90 + (ai * 780) / (vals.length - 1)} y={430 - (vals[ai] / max) * 260 - 30} />
          <Cap t={t} text="Configured collapse behavior reduces service rate after saturation." />{dock}
        </Chamber>
      );
    }
    case "cache-blocks": {
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect prompt tokens" onClick={() => onSelect("prompt tokens")} onKeyDown={(e: Evt) => activate(e, () => onSelect("prompt tokens"))}>
            {Array.from({ length: 12 }).map((_, i) => <rect key={i} x={90 + i * 30} y={180} width={26} height={30} rx={3} fill={sel("prompt tokens") ? t.accent.control : t.fill.tertiary} stroke={i % 4 === 3 ? t.category.orange : t.stroke.primary} />)}
            <text x={270} y={230} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>prompt tokens → block split (every 4)</text>
          </g>
          <Conduit t={t} d="M270,250 L270,300" on={ai >= 2} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect hash" onClick={() => onSelect("hash")} onKeyDown={(e: Evt) => activate(e, () => onSelect("hash"))}>
            <rect x={190} y={300} width={160} height={54} rx={8} fill={sel("hash") ? t.fill.secondary : t.fill.tertiary} stroke={sel("hash") ? t.accent.primary : t.stroke.primary} />
            <text x={270} y={332} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>hash</text>
          </g>
          <Conduit t={t} d="M350,327 L470,327" on={ai >= 3} />
          <text x={640} y={180} fontSize={9} fill={t.text.tertiary}>cache library (bounded slots)</text>
          {Array.from({ length: 9 }).map((_, i) => <rect key={i} x={480 + (i % 3) * 150} y={200 + Math.floor(i / 3) * 70} width={130} height={54} rx={6} fill={i === 4 && ai >= 3 ? t.fill.secondary : t.fill.tertiary} stroke={i === 4 && ai >= 3 ? t.accent.primary : t.stroke.primary} />)}
          <text x={545} y={402} fontSize={9} fill={t.text.tertiary}>cache slot</text>
          <Specimen t={t} chapter="scheduler" x={ai < 2 ? 270 : ai === 2 ? 270 : 545} y={ai < 2 ? 160 : ai === 2 ? 285 : 350} />
          <Cap t={t} text="Prompt token blocks are hashed into bounded cache entries." />{dock}
        </Chamber>
      );
    }
    case "cache-eviction": {
      const policy = mode;
      const victim = policy === "LRU" ? 0 : policy === "LFU" ? 3 : 1;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={90} y={170} fontSize={9.5} fill={t.text.tertiary}>resident blocks · policy {policy}</text>
          {Array.from({ length: 6 }).map((_, i) => {
            const vic = i === victim;
            return (
              <g key={i} className="fdry-part" role="button" tabIndex={0} aria-label={`Resident block ${i}${vic ? " victim" : ""}`} onClick={() => onSelect(vic ? "victim" : "resident blocks")} onKeyDown={(e: Evt) => activate(e, () => onSelect(vic ? "victim" : "resident blocks"))}>
                <rect x={100 + i * 130} y={200} width={110} height={80} rx={8} fill={vic ? t.fill.secondary : t.fill.tertiary} stroke={vic ? t.category.red : t.stroke.primary} strokeWidth={vic ? 2.2 : 1.2} />
                <text x={155 + i * 130} y={245} textAnchor="middle" fontSize={10} fill={t.text.primary}>block {i}</text>
                {vic ? <text x={155 + i * 130} y={262} textAnchor="middle" fontSize={8.5} fill={t.category.red}>victim</text> : null}
              </g>
            );
          })}
          <Conduit t={t} d="M500,360 L500,290" on tone={t.category.orange} marker="fo" />
          <Bay t={t} x={420} y={360} w={160} h={60} label="new block" active={sel("new block")} tone={t.category.orange} onSelect={() => onSelect("new block")} />
          <Bay t={t} x={640} y={360} w={200} h={60} label="replacement" active={sel("replacement")} onSelect={() => onSelect("replacement")} />
          <Specimen t={t} chapter="scheduler" x={155 + victim * 130} y={180} />
          <Cap t={t} text="Capacity pressure applies the selected LRU, LFU, or FIFO policy." />{dock}
        </Chamber>
      );
    }
    case "cache-latency-optin": {
      const aware = mode === "latency-aware";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={80} y={266} w={150} h={68} label="cache hit" active={sel("cache hit")} onSelect={() => onSelect("cache hit")} />
          <Conduit t={t} d="M230,300 L340,300" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect latency-aware gate" onClick={() => onSelect("latency-aware gate")} onKeyDown={(e: Evt) => activate(e, () => onSelect("latency-aware gate"))}>
            <rect x={340} y={260} width={80} height={80} rx={8} fill={aware ? t.fill.secondary : t.fill.tertiary} stroke={aware ? t.accent.primary : t.category.yellow} strokeWidth={aware ? 2.2 : 1.6} />
            <line x1={380} y1={300} x2={aware ? 405 : 380} y2={aware ? 278 : 268} stroke={aware ? t.accent.primary : t.category.yellow} strokeWidth={3} />
            <text x={380} y={356} textAnchor="middle" fontSize={9} fill={t.text.tertiary}>gate</text>
          </g>
          <Conduit t={t} d="M420,300 L520,260" on={aware} />
          <Bay t={t} x={520} y={220} w={200} h={64} label="cached token discount" active={sel("cached token discount") || aware} onSelect={() => onSelect("cached token discount")} />
          <Conduit t={t} d="M420,300 L520,380" on={!aware} tone={t.category.yellow} marker="fo" />
          <Bay t={t} x={520} y={350} w={200} h={64} label="accounting only" sub="no latency effect" active={!aware} tone={t.category.yellow} onSelect={() => onSelect("latency-aware gate")} />
          <Conduit t={t} d="M720,252 L800,300" on={aware} />
          <Bay t={t} x={800} y={266} w={120} h={68} label="TTFT" active={sel("TTFT")} onSelect={() => onSelect("TTFT")} />
          <Specimen t={t} chapter="scheduler" x={aware ? 620 : 620} y={aware ? 200 : 430} />
          <Cap t={t} text="Cache hits reduce latency only when --prefix-cache-latency-aware is enabled." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 8 · Semantics — fault injector lab ────────────────────────────────

function FaultLab(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — fault lab`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "status-injection": {
      const menu = ["success", "429", "500", "503"];
      const chosen = menu.indexOf(mode) >= 0 ? mode : "success";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={266} w={150} h={68} label="request" active={sel("request")} onSelect={() => onSelect("request")} />
          <Conduit t={t} d="M220,300 L310,300" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect seeded error draw" onClick={() => onSelect("error draw")} onKeyDown={(e: Evt) => activate(e, () => onSelect("error draw"))}>
            <circle cx={360} cy={300} r={40} fill={sel("error draw") ? t.fill.secondary : t.fill.tertiary} stroke={sel("error draw") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("error draw") ? 2.2 : 1.4} className="fdry-spin" strokeDasharray="7 7" />
            <text x={360} y={304} textAnchor="middle" fontSize={9.5} fill={t.text.primary}>seeded draw</text>
          </g>
          <text x={520} y={170} fontSize={9.5} fill={t.text.tertiary}>configured status menu</text>
          {menu.map((n, i) => {
            const on = n === chosen;
            return (
              <g key={n}>
                <Conduit t={t} d={`M400,300 L${520 - 10},${200 + i * 62}`} on={on} tone={n === "success" ? t.category.green : t.category.red} marker={n === "success" ? "fg" : "fo"} />
                <Bay t={t} x={520} y={175 + i * 62} w={150} h={48} label={n} active={on || sel("status menu")} tone={n === "success" ? t.category.green : t.category.red} onSelect={() => onSelect("status menu")} />
              </g>
            );
          })}
          <Bay t={t} x={730} y={266} w={190} h={68} label="error body" active={sel("error body")} onSelect={() => onSelect("error body")} />
          <Specimen t={t} chapter="semantics" x={470} y={200 + menu.indexOf(chosen) * 62} />
          <Cap t={t} text="Seeded status injection chooses only from the configured status menu." />{dock}
        </Chamber>
      );
    }
    case "retry-after": {
      const ai = step % page.nodes.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={110} y={230} w={200} h={100} label="status" sub="retryable" active={sel("status") || ai === 0} tone={t.category.orange} onSelect={() => onSelect("status")} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect Retry-After header" onClick={() => onSelect("Retry-After header")} onKeyDown={(e: Evt) => activate(e, () => onSelect("Retry-After header"))}>
            <rect x={400} y={220} width={220} height={60} rx={8} fill={sel("Retry-After header") ? t.accent.control : t.fill.secondary} stroke={sel("Retry-After header") ? t.accent.primary : t.category.orange} strokeWidth={1.8} />
            <text x={510} y={256} textAnchor="middle" fontSize={11} fontWeight={600} fill={sel("Retry-After header") ? t.text.onAccent : t.text.primary}>Retry-After: N</text>
          </g>
          <Bay t={t} x={400} y={310} w={220} h={80} label="body" active={sel("body") || ai === 2} onSelect={() => onSelect("body")} />
          <Conduit t={t} d="M310,280 L400,250" on />
          <Conduit t={t} d="M310,300 L400,350" on />
          <Specimen t={t} chapter="semantics" x={510} y={200} />
          <Cap t={t} text="Configured Retry-After accompanies injected responses that support retry semantics." />{dock}
        </Chamber>
      );
    }
    case "midstream": {
      const nodes = page.nodes;
      const ai = step % nodes.length;
      const cut = ai >= 3;
      const xs = nodes.map((_, i) => 120 + (i * 760) / (nodes.length - 1));
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <rect x={70} y={260} width={860} height={80} rx={40} fill={t.fill.quaternary} stroke={t.stroke.primary} />
          {nodes.map((n, i) => {
            const isCut = n === "error cut";
            const on = i <= ai;
            return (
              <g key={i} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
                {isCut ? <line x1={xs[i]} y1={250} x2={xs[i]} y2={350} stroke={cut ? t.category.red : t.stroke.secondary} strokeWidth={cut ? 4 : 1.6} /> : null}
                <rect x={xs[i] - 48} y={272} width={96} height={56} rx={7} fill={sel(n) ? t.accent.control : on ? t.fill.secondary : t.fill.tertiary} stroke={sel(n) ? t.accent.primary : isCut ? t.category.red : on ? t.accent.primary : t.stroke.primary} opacity={on ? 1 : 0.4} />
                <text x={xs[i]} y={304} textAnchor="middle" fontSize={9} fontWeight={600} fill={sel(n) ? t.text.onAccent : t.text.primary}>{n}</text>
              </g>
            );
          })}
          {cut ? <text x={xs[3]} y={240} textAnchor="middle" fontSize={9.5} fill={t.category.red}>stream cut — partial response preserved</text> : null}
          <Specimen t={t} chapter="semantics" x={xs[Math.min(ai, cut ? 3 : ai)]} y={250} />
          <Cap t={t} text="A stream can fail after generated output, preserving partial-response evidence." />{dock}
        </Chamber>
      );
    }
    case "extended-usage": {
      const anth = mode === "Anthropic";
      const meters = page.nodes;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={500} y={160} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>{anth ? "Anthropic field names" : "OpenAI field names"}</text>
          {meters.map((n, i) => {
            const on = sel(n);
            const val = [90, 60, 40, 55, 30][i];
            const x = 110 + i * 165;
            return (
              <g key={n} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
                <rect x={x} y={200} width={130} height={180} rx={8} fill={t.fill.tertiary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 2.2 : 1.2} />
                <rect x={x + 14} y={380 - (val / 100) * 150} width={102} height={(val / 100) * 150} rx={4} fill={on ? t.accent.primary : t.category.cyan} opacity={0.75} />
                <text x={x + 65} y={400} textAnchor="middle" fontSize={9} fill={t.text.secondary}>{n}</text>
              </g>
            );
          })}
          <Specimen t={t} chapter="semantics" x={175} y={180} />
          <Cap t={t} text="Cache, audio, prediction, and tool-use usage fields preserve provider-specific names." />{dock}
        </Chamber>
      );
    }
    case "tool-calls": {
      const stream = mode !== "non-stream";
      const chain = page.nodes;
      const ai = step % chain.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={500} y={160} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>{stream ? "streamed argument fragments" : "single assembled tool call"}</text>
          {chain.map((n, i) => {
            const x = 120 + (i * 760) / (chain.length - 1);
            const on = sel(n) || (selected === "" && i === ai);
            const frag = n === "argument fragments";
            return (
              <g key={n}>
                {i < chain.length - 1 ? <Conduit t={t} d={`M${x + 65},300 L${x + 155},300`} on={i < ai} /> : null}
                <Bay t={t} x={x - 70} y={266} w={140} h={68} label={n} sub={frag ? (stream ? "N chunks" : "one blob") : undefined} active={on} onSelect={() => onSelect(n)} />
              </g>
            );
          })}
          <Specimen t={t} chapter="semantics" x={120 + (ai * 760) / (chain.length - 1)} y={246} />
          <Cap t={t} text="Tool name and arguments appear in both complete and streamed tool-call shapes." />{dock}
        </Chamber>
      );
    }
    case "accuracy-verdicts": {
      const chain = page.nodes;
      const ai = step % chain.length;
      const xs = chain.map((_, i) => 110 + (i * 780) / (chain.length - 1));
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={500} y={160} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>format: {mode}</text>
          {chain.map((n, i) => {
            const on = sel(n) || (selected === "" && i === ai);
            const gate = n === "seeded verdict";
            return (
              <g key={n}>
                {i < chain.length - 1 ? <Conduit t={t} d={`M${xs[i] + 60},310 L${xs[i + 1] - 60},310`} on={i < ai} /> : null}
                {gate
                  ? <g className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}><polygon points={`${xs[i]},270 ${xs[i] + 58},310 ${xs[i]},350 ${xs[i] - 58},310`} fill={on ? t.fill.secondary : t.fill.tertiary} stroke={on ? t.accent.primary : t.stroke.primary} strokeWidth={on ? 2.2 : 1.2} /><text x={xs[i]} y={306} textAnchor="middle" fontSize={9} fontWeight={600} fill={t.text.primary}>verdict</text><text x={xs[i]} y={320} textAnchor="middle" fontSize={8} fill={t.text.tertiary}>correct/CoT/adv</text></g>
                  : <Bay t={t} x={xs[i] - 62} y={276} w={124} h={68} label={n} active={on} onSelect={() => onSelect(n)} />}
              </g>
            );
          })}
          <Specimen t={t} chapter="semantics" x={xs[ai]} y={252} />
          <Cap t={t} text="Prompt matching selects ground truth; seeded rates choose correct, CoT, and adversarial output forms." />{dock}
        </Chamber>
      );
    }
    case "accuracy-oracle": {
      const vals = page.metrics;
      const bars = ["matched", "correct", "incorrect", "adversarial"];
      const max = Math.max(...vals, 1);
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <text x={90} y={170} fontSize={9.5} fill={t.text.tertiary}>GET /accuracy · served tally (oracle)</text>
          <line x1={110} y1={420} x2={910} y2={420} stroke={t.stroke.primary} />
          {bars.map((n, i) => {
            const h = (vals[i] / max) * 220;
            const x = 150 + i * 190;
            const on = sel(n);
            return (
              <g key={n} className="fdry-part" role="button" tabIndex={0} aria-label={`Inspect ${n}: ${vals[i]}`} onClick={() => onSelect(n)} onKeyDown={(e: Evt) => activate(e, () => onSelect(n))}>
                <rect x={x} y={420 - h} width={110} height={h} rx={6} fill={on ? t.accent.primary : t.category.green} opacity={0.8} />
                <text x={x + 55} y={440} textAnchor="middle" fontSize={9.5} fill={t.text.secondary}>{n}</text>
                <text x={x + 55} y={415 - h} textAnchor="middle" fontSize={10} fontWeight={700} fill={t.text.primary}>{vals[i]}</text>
              </g>
            );
          })}
          <Bay t={t} x={780} y={200} w={130} h={60} label="per task" sub="correct/matched" active={sel("per task")} onSelect={() => onSelect("per task")} />
          <Specimen t={t} chapter="semantics" x={205} y={180} />
          <Cap t={t} text="/accuracy and Prometheus counters report the mock's actual served tally." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 9 · Observability — telemetry deck + replicated foundries ─────────

function TelemetryDeck(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — telemetry deck`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "prometheus": {
      const counters = ["request counters", "token counters", "cache counters"];
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {counters.map((n, i) => (
            <g key={n}>
              <Bay t={t} x={80} y={190 + i * 90} w={200} h={64} label={n} active={sel(n)} onSelect={() => onSelect(n)} />
              <Conduit t={t} d={`M280,${222 + i * 90} L440,300`} on />
            </g>
          ))}
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect dialect encoder" onClick={() => onSelect("dialect encoder")} onKeyDown={(e: Evt) => activate(e, () => onSelect("dialect encoder"))}>
            <rect x={440} y={260} width={180} height={90} rx={10} fill={sel("dialect encoder") ? t.fill.secondary : t.fill.tertiary} stroke={sel("dialect encoder") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("dialect encoder") ? 2.2 : 1.2} />
            <text x={530} y={300} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>dialect encoder</text>
            <text x={530} y={318} textAnchor="middle" fontSize={9.5} fill={t.accent.primary}>{mode}</text>
          </g>
          <Conduit t={t} d="M620,305 L720,305" on />
          <Bay t={t} x={720} y={272} w={200} h={64} label={`${mode} names`} sub="one /metrics endpoint" active={selected === ""} onSelect={() => onSelect("dialect encoder")} />
          <Specimen t={t} chapter="observability" x={530} y={240} />
          <Cap t={t} text="One metrics endpoint can emit vLLM, SGLang, TRT-LLM, or Dynamo naming dialects." />{dock}
        </Chamber>
      );
    }
    case "dcgm": {
      const loaded = mode === "loaded";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {[0, 1].map((g) => (
            <g key={g}>
              <text x={200 + g * 460} y={160} textAnchor="middle" fontSize={10} fontWeight={600} fill={t.text.primary}>GPU {g}</text>
              <Gauge t={t} cx={110 + g * 460} cy={300} r={66} value={loaded ? 82 : 12} label="power" active={sel("power")} onSelect={() => onSelect("power")} />
              <Gauge t={t} cx={290 + g * 460} cy={300} r={66} value={loaded ? 91 : 5} label="SM active" active={sel("SM active")} tone={t.category.purple} onSelect={() => onSelect("SM active")} />
            </g>
          ))}
          <Bay t={t} x={400} y={420} w={200} h={54} label="energy" sub="deterministic under seed" active={sel("energy")} onSelect={() => onSelect("energy")} />
          <Specimen t={t} chapter="observability" x={110} y={220} />
          <Cap t={t} text="Per-GPU gauges and counters are deterministic under a configured seed." />{dock}
        </Chamber>
      );
    }
    case "throughput-load": {
      const vals = page.metrics;
      const ai = step % vals.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={266} w={170} h={70} label="completed requests" active={sel("completed requests")} onSelect={() => onSelect("completed requests")} />
          <Conduit t={t} d="M240,300 L320,300" on />
          <Bay t={t} x={320} y={266} w={150} h={70} label="window rate" active={sel("window rate")} onSelect={() => onSelect("window rate")} />
          <Conduit t={t} d="M470,300 L550,300" on />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect load model" onClick={() => onSelect("load model")} onKeyDown={(e: Evt) => activate(e, () => onSelect("load model"))}>
            <circle cx={600} cy={300} r={44} fill={sel("load model") ? t.fill.secondary : t.fill.tertiary} stroke={sel("load model") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("load model") ? 2.2 : 1.4} className="fdry-spin" strokeDasharray="8 8" />
            <text x={600} y={304} textAnchor="middle" fontSize={9.5} fill={t.text.primary}>load model</text>
          </g>
          <Conduit t={t} d="M644,300 L720,300" on />
          <Gauge t={t} cx={820} cy={320} r={72} value={Math.min(100, vals[ai])} label="DCGM gauges" active={sel("DCGM gauges")} onSelect={() => onSelect("DCGM gauges")} />
          <Specimen t={t} chapter="observability" x={[155, 395, 600, 820][Math.min(3, ai)]} y={246} />
          <Cap t={t} text="Synthetic GPU load follows observed request throughput within the configured window." />{dock}
        </Chamber>
      );
    }
    case "multiprocess": {
      const four = mode === "4 processes";
      const kids = four ? 4 : 1;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={266} w={160} h={70} label="client connections" active={sel("client connections")} onSelect={() => onSelect("client connections")} />
          <Conduit t={t} d="M230,300 L320,300" on />
          <Bay t={t} x={320} y={266} w={140} h={70} label="L4 parent" sub="round-robin" active={sel("L4 parent")} tone={t.category.orange} onSelect={() => onSelect("L4 parent")} />
          {Array.from({ length: kids }).map((_, i) => {
            const y = kids === 1 ? 300 : 170 + i * 90;
            const nm = i === 0 ? "child 0" : i === 1 ? "child 1" : "child N";
            return (
              <g key={i}>
                <Conduit t={t} d={`M460,300 L${640 - 20},${y + 27}`} on={i === (step % kids)} tone={t.category.orange} marker="fo" />
                <Bay t={t} x={640} y={y} w={200} h={54} label={`foundry ${i}`} sub={nm} active={sel(nm)} onSelect={() => onSelect(nm)} />
              </g>
            );
          })}
          <Specimen t={t} chapter="observability" x={545} y={four ? 170 + (step % 4) * 90 + 27 : 300} />
          <Cap t={t} text="The parent round-robins TCP connections across isolated child servers." />{dock}
        </Chamber>
      );
    }
    case "limits-access": {
      const multi = mode === "multi-process";
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <g className="fdry-part" role="button" tabIndex={0} aria-label="Inspect process gate" onClick={() => onSelect("process gate")} onKeyDown={(e: Evt) => activate(e, () => onSelect("process gate"))}>
            <polygon points="180,200 250,260 180,320 110,260" fill={sel("process gate") ? t.fill.secondary : t.fill.tertiary} stroke={sel("process gate") ? t.accent.primary : t.stroke.primary} strokeWidth={sel("process gate") ? 2.2 : 1.2} />
            <text x={180} y={264} textAnchor="middle" fontSize={10} fontWeight={600} fill={t.text.primary}>--processes</text>
          </g>
          <Conduit t={t} d="M250,260 L360,230" on tone={t.category.green} marker="fg" />
          <Bay t={t} x={360} y={200} w={180} h={60} label="HTTP TCP" active={sel("HTTP TCP")} tone={t.category.green} onSelect={() => onSelect("HTTP TCP")} />
          <Bay t={t} x={360} y={290} w={180} h={54} label="gRPC skipped" active={sel("gRPC skipped") || multi} tone={multi ? t.category.red : undefined} onSelect={() => onSelect("gRPC skipped")} />
          <Bay t={t} x={360} y={360} w={180} h={54} label="UDS skipped" active={sel("UDS skipped") || multi} tone={multi ? t.category.red : undefined} onSelect={() => onSelect("UDS skipped")} />
          {multi ? <g><line x1={300} y1={295} x2={360} y2={317} stroke={t.category.red} strokeWidth={2.4} /><line x1={300} y1={365} x2={360} y2={387} stroke={t.category.red} strokeWidth={2.4} /></g> : null}
          <Bay t={t} x={640} y={280} w={260} h={64} label="access logs unwired" sub="--access-logs defined, not middleware" active={sel("access logs unwired")} tone={t.category.yellow} onSelect={() => onSelect("access logs unwired")} />
          <Specimen t={t} chapter="observability" x={multi ? 450 : 450} y={multi ? 320 : 230} />
          <Cap t={t} text="--processes skips gRPC and UDS; --access-logs is defined but not connected to middleware." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

// ── Chapter 10 · Proof — exploded evidence machine ────────────────────────────

function ProofExplodedView(p: SceneProps) {
  const { t, page, step, mode, selected, onSelect } = p;
  const sel = (n: string) => selected === n;
  const label = `${page.title} — proof machine`;
  const dock = <ControlDock t={t} page={page} step={step} mode={mode} onStep={p.onStep} onMode={p.onMode} />;

  switch (page.id) {
    case "proof-graph": {
      const tiers = ["Rust source", "feature claim", "raw-record e2e", "integration", "unit"];
      const ai = step % tiers.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Bay t={t} x={70} y={266} w={150} h={70} label="Rust source" active={sel("Rust source")} onSelect={() => onSelect("Rust source")} />
          <Conduit t={t} d="M220,300 L300,300" on />
          <Bay t={t} x={300} y={266} w={150} h={70} label="feature claim" active={sel("feature claim")} onSelect={() => onSelect("feature claim")} />
          {["raw-record e2e", "integration", "unit"].map((n, i) => {
            const y = 180 + i * 100;
            const strongest = i === 0;
            const on = sel(n) || (mode === "strongest" ? strongest : true);
            return (
              <g key={n}>
                <Conduit t={t} d={`M450,300 L${540 - 10},${y + 27}`} on={on} tone={strongest ? t.category.green : t.stroke.secondary} marker={strongest ? "fg" : "fa"} />
                <Bay t={t} x={540} y={y} w={220} h={54} label={n} sub={strongest ? "strongest" : `tier ${i + 2}`} active={sel(n)} tone={strongest ? t.category.green : undefined} onSelect={() => onSelect(n)} />
              </g>
            );
          })}
          <Specimen t={t} chapter="proof" x={[145, 375, 650, 650, 650][ai]} y={[246, 246, 207, 307, 407][ai]} />
          <Cap t={t} text="Raw-record e2e is strongest; integration, unit, then implementation-only evidence follow." />{dock}
        </Chamber>
      );
    }
    case "unsupported-matrix": {
      const deployment = mode === "deployment";
      const cols = deployment ? ["HTTP", "gRPC", "UDS", "TLS"] : ["HTTP", "gRPC", "UDS", "Riva HTTP"];
      const rows = deployment ? ["single process", "multi-process"] : ["general inference", "Riva services"];
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          {cols.map((c, i) => <text key={c} x={300 + i * 140} y={190} textAnchor="middle" fontSize={10} fontWeight={600} fill={t.text.primary}>{c}</text>)}
          {rows.map((r, ri) => (
            <g key={r}>
              <text x={220} y={240 + ri * 100} textAnchor="end" fontSize={10} fill={t.text.secondary}>{r}</text>
              {cols.map((c, ci) => {
                const good = deployment
                  ? !(ri === 1 && (c === "gRPC" || c === "UDS"))
                  : ri === 0
                    ? c !== "Riva HTTP"
                    : c === "gRPC";
                const nm = good ? "supported path" : c;
                return (
                  <g key={c} className="fdry-part" role="button" tabIndex={0} aria-label={`${r} × ${c}: ${good ? "supported" : "unsupported"}`} onClick={() => onSelect(nm)} onKeyDown={(e: Evt) => activate(e, () => onSelect(nm))}>
                    <rect x={250 + ci * 140} y={215 + ri * 100} width={100} height={62} rx={8} fill={good ? t.fill.secondary : t.fill.tertiary} stroke={good ? t.category.green : t.category.red} strokeWidth={selected === nm ? 2.4 : 1.4} />
                    {good ? <path d={`M${272 + ci * 140},${248 + ri * 100} l12,12 l24,-26`} fill="none" stroke={t.category.green} strokeWidth={2.4} /> : <g><line x1={278 + ci * 140} y1={232 + ri * 100} x2={322 + ci * 140} y2={260 + ri * 100} stroke={t.category.red} strokeWidth={2.4} /><line x1={322 + ci * 140} y1={232 + ri * 100} x2={278 + ci * 140} y2={260 + ri * 100} stroke={t.category.red} strokeWidth={2.4} /></g>}
                  </g>
                );
              })}
            </g>
          ))}
          <text x={500} y={440} textAnchor="middle" fontSize={9.5} fill={t.text.tertiary}>multi-process is TCP/HTTP-only · Riva is gRPC-only</text>
          <Specimen t={t} chapter="proof" x={300} y={200} />
          <Cap t={t} text="Boundaries are explicit: multi-process is TCP/HTTP-only and Riva is gRPC-only." />{dock}
        </Chamber>
      );
    }
    case "source-index": {
      const mods = page.nodes;
      const ai = step % mods.length;
      return (
        <Chamber t={t} label={label}>
          <FoundryDefs t={t} />
          <Port t={t} cx={500} cy={290} r={44} label="lib.rs" active={selected === ""} onSelect={() => onSelect("configuration")} />
          {mods.map((n, i) => {
            const a = (Math.PI * 2 * i) / mods.length - Math.PI / 2;
            const lx = 500 + Math.cos(a) * 330;
            const ly = 290 + Math.sin(a) * 180;
            const on = sel(n) || (selected === "" && i === ai);
            return (
              <g key={n}>
                <Conduit t={t} d={`M500,290 L${lx},${ly}`} on={on} tone={t.category.purple} marker="fp" />
                <Bay t={t} x={lx - 70} y={ly - 24} w={140} h={48} label={n} active={on} tone={t.category.purple} onSelect={() => onSelect(n)} />
              </g>
            );
          })}
          <Specimen t={t} chapter="proof" x={500} y={220} />
          <Cap t={t} text="Every atlas feature links to implementation and its strongest available proof." />{dock}
        </Chamber>
      );
    }
    default:
      return <Chamber t={t} label={label}><FoundryDefs t={t} />{dock}</Chamber>;
  }
}

function Scene(props: SceneProps) {
  switch (props.page.chapter) {
    case "orientation": return <OrientationFoundry {...props} />;
    case "ingress": return <IngressManifold {...props} />;
    case "llm": return <ProtocolGlassworks {...props} />;
    case "specialized": return <EndpointWorks {...props} />;
    case "grpc": return <GrpcSwitchyard {...props} />;
    case "timing": return <TimingEscapement {...props} />;
    case "scheduler": return <SchedulerFloor {...props} />;
    case "semantics": return <FaultLab {...props} />;
    case "observability": return <TelemetryDeck {...props} />;
    case "proof": return <ProofExplodedView {...props} />;
  }
}

// ── shell ────────────────────────────────────────────────────────────────────

function statusTone(status: PageStatus): "success" | "warning" | "info" {
  if (status === "built") return "success";
  if (status === "partial") return "warning";
  return "info";
}

function HeaderSpecimen({ t, chapter }: { t: Theme; chapter: ChapterId }) {
  const index = CHAPTERS.findIndex((c) => c.id === chapter);
  const start = 14;
  const end = 150;
  const x = start + (index * (end - start)) / (CHAPTERS.length - 1);
  return (
    <svg viewBox="0 0 250 44" style={{ width: 250, height: 44, display: "block" }} aria-label={`Request specimen: ${SPECIMEN_STAGE[chapter]}`}>
      <line x1={start} y1={22} x2={end} y2={22} stroke={t.stroke.primary} strokeWidth={1.6} />
      <line x1={start} y1={22} x2={x} y2={22} stroke={t.accent.primary} strokeWidth={1.6} strokeDasharray="4 4" className="fdry-flow" />
      {CHAPTERS.map((c, i) => {
        const cx = start + (i * (end - start)) / (CHAPTERS.length - 1);
        return <circle key={c.id} cx={cx} cy={22} r={i === index ? 3.5 : 2} fill={i <= index ? t.accent.primary : t.fill.primary} stroke={t.bg.elevated} />;
      })}
      <g className="fdry-specimen" style={{ transform: `translate(${x}px,22px)` }}><g className="fdry-pulse">{specimenShape(chapter, t)}</g></g>
      <text x={162} y={19} fontSize={8} fill={t.text.tertiary} className="fdry-eyebrow">SPECIMEN</text>
      <text x={162} y={33} fontSize={10} fontWeight={600} fill={t.text.primary}>{SPECIMEN_STAGE[chapter]}</text>
    </svg>
  );
}

function ContextDrawer({ t, page, selected, onClose }: { t: Theme; page: FeaturePage; selected: string; onClose: () => void }) {
  const dispatch = useCanvasAction();
  return (
    <div className="fdry-drawer" style={{ background: t.bg.elevated, borderColor: t.stroke.primary }}>
      <Row gap={8} align="center">
        <Text size="small" tone="tertiary" weight="semibold">MACHINE PART</Text>
        <div style={{ flex: 1 }} />
        <Button variant="ghost" onClick={onClose}>Close</Button>
      </Row>
      <div style={{ marginTop: 8 }}>
        <Text weight="semibold">{selected || page.nodes[0]}</Text>
      </div>
      <Row gap={6} wrap style={{ marginTop: 8 }}>
        <Pill active>{page.status}</Pill>
        <Pill>{page.evidence}</Pill>
      </Row>
      <Divider />
      <Text size="small" tone="tertiary">INVARIANT</Text>
      <Text size="small">{page.invariant}</Text>
      <div style={{ marginTop: 12 }}>
        <Text size="small" tone="tertiary">SOURCE</Text>
        <Text size="small" truncate="start"><Code>{page.source}</Code></Text>
      </div>
      <div style={{ marginTop: 8 }}>
        <Text size="small" tone="tertiary">STRONGEST PROOF</Text>
        <Text size="small" truncate="start"><Code>{page.proof}</Code></Text>
      </div>
      <Row gap={6} wrap style={{ marginTop: 12 }}>
        <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: page.source })}>Open source</Button>
        <Button variant="ghost" onClick={() => dispatch({ type: "openFile", path: page.proof })}>Open proof</Button>
      </Row>
      <div style={{ marginTop: 12 }}>
        <Callout tone={statusTone(page.status)} title={page.status === "built" ? "Built behavior" : page.status === "partial" ? "Partial wiring" : "Implementation boundary"}>
          Evidence tier: {page.evidence}.
        </Callout>
      </div>
    </div>
  );
}

export default function MockFoundry() {
  const t = useHostTheme();
  const [persistedPage, setPersistedPage] = useCanvasState("mock-foundry.page", PAGES[0].id);
  const [step, setStep] = useCanvasState("mock-foundry.step", 0);
  const [mode, setMode] = useCanvasState("mock-foundry.mode", "primary");
  const [selected, setSelected] = useCanvasState("mock-foundry.selected", "");

  const page = PAGES.find((entry) => entry.id === persistedPage) ?? PAGES[0];
  const pageIndex = PAGES.findIndex((entry) => entry.id === page.id);
  const chapter = CHAPTERS.find((entry) => entry.id === page.chapter) ?? CHAPTERS[0];
  const chapterIndex = CHAPTERS.findIndex((entry) => entry.id === page.chapter);
  const chapterPages = PAGES.filter((entry) => entry.chapter === page.chapter);
  const inChapter = chapterPages.findIndex((entry) => entry.id === page.id);
  const effectiveMode = page.modes.includes(mode) ? mode : page.modes[0];

  const navigate = (id: string) => {
    const target = PAGES.find((entry) => entry.id === id) ?? PAGES[0];
    setPersistedPage(target.id);
    setStep(0);
    setSelected("");
    setMode(target.modes[0]);
  };
  const move = (delta: number) => navigate(PAGES[Math.max(0, Math.min(PAGES.length - 1, pageIndex + delta))].id);

  const handleKeyboard = (event: { key: string; target: EventTarget | null; preventDefault: () => void }) => {
    const tag = (event.target as { tagName?: string } | null)?.tagName?.toLowerCase();
    if (event.key === "Escape") { setSelected(""); return; }
    if (tag === "input" || tag === "select" || tag === "textarea" || tag === "button") return;
    if (event.key === "ArrowLeft" && pageIndex > 0) { event.preventDefault(); move(-1); }
    if (event.key === "ArrowRight" && pageIndex < PAGES.length - 1) { event.preventDefault(); move(1); }
  };

  const sceneProps: SceneProps = {
    t,
    page,
    step,
    mode: effectiveMode,
    selected,
    onSelect: (v: string) => setSelected(v),
    onStep: () => setStep((value) => (value + 1) % Math.max(page.steps.length, page.nodes.length)),
    onMode: (v: string) => setMode(v),
  };

  return (
    <div
      className="fdry-root"
      tabIndex={0}
      onKeyDown={handleKeyboard}
      style={{ background: t.bg.editor, color: t.text.primary }}
      aria-label="The Mock Foundry — aiperf-mock-server systems cutaway. Focus the canvas and use Left and Right arrows to travel; Escape closes the part inspector."
    >
      <style>{CSS}</style>
      <div className="fdry-wrap">
        <div className="fdry-head">
          <div style={{ flex: 1, minWidth: 260 }}>
            <div className="fdry-eyebrow" style={{ color: t.accent.primary }}>The Mock Foundry · {chapter.title} · {chapter.world}</div>
            <h1 className="fdry-title" style={{ color: t.text.primary }}>{page.title}</h1>
          </div>
          <HeaderSpecimen t={t} chapter={page.chapter} />
          <Row gap={6} align="center">
            <Pill active>{page.status}</Pill>
            <Pill>{pageIndex + 1} / {PAGES.length}</Pill>
          </Row>
        </div>

        <div className="fdry-strip" aria-label="Chapter filmstrip">
          {CHAPTERS.map((item, index) => {
            const pages = PAGES.filter((entry) => entry.chapter === item.id);
            const first = pages[0];
            const current = item.id === page.chapter;
            return (
              <div
                key={item.id}
                className="fdry-seg"
                style={{ borderColor: current ? t.accent.primary : t.stroke.primary, opacity: current ? 1 : 0.62 }}
              >
                <button
                  className="fdry-seg-label"
                  aria-current={current ? "page" : undefined}
                  onClick={() => navigate(first.id)}
                >
                  <small style={{ color: t.text.tertiary }}>{String(index + 1).padStart(2, "0")} · {pages.length}</small>
                  <strong style={{ color: current ? t.accent.primary : t.text.primary }}>{item.short}</strong>
                </button>
                <div className="fdry-ticks">
                  {pages.map((pg) => {
                    const active = pg.id === page.id;
                    return (
                      <button
                        key={pg.id}
                        className="fdry-tick"
                        aria-label={pg.title}
                        title={pg.title}
                        onClick={() => navigate(pg.id)}
                        style={{ borderColor: active ? t.accent.primary : t.stroke.primary, background: active ? t.accent.primary : "transparent" }}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="fdry-stage" style={{ borderColor: t.stroke.primary, background: t.bg.elevated }}>
          <Scene {...sceneProps} />
          {selected ? <ContextDrawer t={t} page={page} selected={selected} onClose={() => setSelected("")} /> : null}
        </div>

        <div className="fdry-foot">
          <Button variant="secondary" disabled={pageIndex === 0} onClick={() => move(-1)}>Back</Button>
          <Button variant="primary" disabled={pageIndex === PAGES.length - 1} onClick={() => move(1)}>Next</Button>
          <Pill>chapter {chapterIndex + 1} / {CHAPTERS.length}</Pill>
          <Pill>{inChapter + 1} / {chapterPages.length} in chapter</Pill>
          <div style={{ flex: 1, minWidth: 12 }} />
          <span className="fdry-hint" style={{ color: t.text.tertiary }}>select a part for evidence · valve advances state</span>
          <Select
            value={page.id}
            onChange={navigate}
            options={PAGES.map((entry, index) => ({
              value: entry.id,
              label: `${index + 1}. ${CHAPTERS.find((item) => item.id === entry.chapter)?.short} · ${entry.title}`,
            }))}
            style={{ minWidth: 300 }}
          />
        </div>
      </div>
    </div>
  );
}
