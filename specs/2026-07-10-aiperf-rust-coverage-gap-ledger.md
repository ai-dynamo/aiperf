# AIPerf-Rust: Coverage-Gap Ledger — what the specs MISS but is worth keeping

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** research synthesis / decision input. Several gaps catalogued below have
since been closed by dedicated specs (metrics accumulator + metric-catalog,
telemetry accumulators, RNG derive-system); those are marked **CLOSED** inline with a
pointer to the spec that covers them. The still-open gap areas are endpoints/exporters
(§2), config-v2 hidden algorithms (§3), timing-engine depth + the outer-loop
sweep/BO/SLA coordinator (§4), and the presentation/API/plot surfaces (§6).
**Method:** 7 parallel source-reading passes over the full 720-file Python tree
(`src/aiperf/`), each briefed on what the existing 10 specs already cover, tasked
with cataloguing only the DELTA and classifying every concept
PORT-EXACT / REDO-CLEANER / THROW-AWAY.
**Companions:** all files in `specs/`, especially
`2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` (the master ledger this
extends).

---

## 0. What the 10 existing specs already cover (the "spine")

Clock/engine boundary, transport (HTTP/SSE leaf + timing recording), graph-IR
runtime byte-exact port, dataset/segment-store seam, accuracy accumulator+analyzer,
scheduling-policy sketch, metric *formulas* + field-name contract (as a one-line
"port near-exact" row), arrival patterns, phase-as-record-dimension, NaN/Inf
discipline, shared-core + request-rate/tokenizer increments.

The spine is ~80% aligned. This ledger records the **large subsystems the spine
specs name in one line or not at all**, where the real risk lives. Each section
ends with the concrete GAPS to fold into the master ledger or a new sub-spec.

---

## 1. Metrics / records / post-processing plane — CLOSED

**CLOSED** by the metrics-accumulator and metric-catalog specs, and built as
`aiperf_runtime::metrics_core` (119-row catalog, NaN-sparse columnar storage, ragged ICL, the
sweep-line curves, phase windows, worker merge, typed native-v2 `Reporter`). The
findings below are retained as the research record of what those specs had to absorb;
they are no longer open gaps.

**Single biggest structural finding: there are TWO metric engines in the tree.**
Port `MetricsAccumulator` + `ColumnStore` (numpy-columnar, NaN-sparse; the
`records_manager.py:493` "SOLE summary" path). DELETE the legacy
`MetricResultsProcessor` per-instance replay and the `parse→aggregate→derive`
three-phase split — that split exists ONLY for distributed RecordProcessors.
In-process, RECORD metrics are per-record fn evals and AGGREGATE metrics collapse
to one `AggregationKind` fold (SUM/MAX/MIN).

**PORT-EXACT (earned-in-blood, specs miss these):**
- **The entire sweep-line subsystem** (`analysis/sweepline*.py`,
  `accumulator_sweeps.py`) — time-weighted concurrency / generation-concurrency /
  prefill-concurrency / throughput / prefill-throughput / per-user / tokens-in-flight
  step-function curves, with duration-weighted percentiles. Highest-value,
  highest-risk unspec'd code. `_sweep_line_cumsum` tie-break (ends-before-starts) +
  FP-roundoff-snap; throughput rate `(output_tokens-1)/gen_dur` (the −1 is subtle);
  KV-cache ICL variant (`sweepline_kv_cache.py`) with `np.nextafter` chunk-arrival
  clamping is the most bug-prone code in the repo.
- **Percentile kernel** (`metric_dicts.py:58-109`): band `[1,5,10,25,50,75,90,95,99]`,
  manual linear interpolation, `virtual_idx = q/100*(n-1)`. `ddof` split:
  **0=population for inference metrics, 1=sample for telemetry** — silent-wrong-std
  trap. Do NOT swap in a Rust quantile crate without matching the method.
- **adj_* error-inflated percentiles** (`derived_latency.py:173-259`, issue #688):
  only `request_latency`+`inter_token_latency` (flag
  `PERCENTILE_INCLUDES_FAILED_REQUESTS`); failed requests appended as `+inf`,
  `np.percentile(method="nearest")` (NOT linear), `std=None`.
- **`effective_latency`** (CO-aware, `end_ns − credit_issued_ns`) and
  **`credit_to_start_latency`** (queue wait) — distinct from the avg-ITL/goodput
  the spine already ported.
- **`observation_duration` window resolution** (`metric_dicts.py:221-242`) — why
  throughput/goodput stay correct under phase/timeslice windowing (explicit window
  bounds override benchmark_duration). Every rate metric depends on it.
- **Phase-tag-authoritative masking, NOT time-window** (`accumulator.py:179-209`) —
  a real Windows coarse-clock boundary fix; naive time-intersection drops boundary
  records.
- **`network_adjusted_*`** (constant-RTT shift, clamp-0, request-start-anchored
  metrics only — ITL/ICL deliberately excluded because RTT cancels in
  `latency−ttft`).
- **Dependency DAG + topo sort + type-tier rule** (RECORD→RECORD only,
  AGGREGATE→RECORD/AGGREGATE, DERIVED→any) — the real invariant behind "ITL depends
  on TTFT+latency+OSL."
- **`MetricFlags`** 18-bit set + **`MetricConsoleGroup`** (EFFECTIVE/ACTIVE/USAGE/
  CACHE/PREDICTION/AUDIO/REASONING/DEFAULT/NONE) + the unit system with `convert_to`
  chains — output/filtering contract.

**REDO-CLEANER:** `ColumnStore` sentinel encodings + handler-closure cache +
GrowableArray (Rust `Option<T>`/enums/`Vec` natively); `BaseUsageRecordMetric` (~30
field-reader metrics → one generic/table); `DerivedSumMetric` `total_*` family →
mostly `ColumnStore.numeric_sum(tag)`; HTTP-trace k6-named passthroughs (keep names,
generate). t-digest list aggregation is optional in single-process — keep exact
`RaggedSeries` (CSR flat-values+offsets, `grouped_cumsum` resets at request
boundaries) unless ramp-scale forces the sketch; if kept, Welford std is PORT-EXACT.

**Metric catalog:** ~120 concrete metrics; the non-obvious formulas are ITL
(`osl<2` guard), TTFO (first non-reasoning token), osl_mismatch (`min()` cap on
threshold), thinking-efficiency (per-record vs total), good_request_count (per-metric
`LARGER_IS_BETTER` direction), `cache_reporting_hint` (absent vs 0 = "cache off" vs
"on/0-hits").

**GAPS (now ABSORBED by the metrics-accumulator + metric-catalog specs):** sweep-line
curves (their own sub-spec); effective/credit-queue latencies; observation_duration;
phase-tag masking; AggregationKind-fold as the taxonomy answer (the "struct + fold
enum" choice over a heavyweight Metric trait won; see §8 decision 1).

---

## 2. Endpoints + exporters (request/response + output contracts)

The transport spec covers the HTTP leaf; the PAYLOAD-formatting and
RESPONSE-parsing zoo, and the OUTPUT-format contracts, are unspec'd.

**Endpoint registration is a static table** (today `plugins.yaml:184-424` +
`EndpointMetadata`): type → path → capability flags (`requires_polling`,
`requires_form_data`, `requires_inline_media`, `tokenizes_input`, `produces_tokens`,
`streaming_path`, `metrics_title`). **PORT-EXACT** as a Rust enum+struct table; the
flags change request LIFECYCLE, not just payload (form-data multipart,
video async submit→poll→download).

**PORT-EXACT endpoint quirks (silent corruption if redone naively):**
- **chat** (`openai_chat.py`): max_tokens vs max_completion_tokens switch
  (`use_legacy_max_tokens`); merge order payload < endpoint.extra < turn.extra_body
  (extra_body wins); `_ensure_include_usage` only when streaming+server-token-count;
  response precedence **reasoning > content+tool_calls > tool_calls > content**
  (a metric-comparability contract; the mixed content+tool_calls emit is the ~18%
  agentic-OSL-undercount fix); tool-call streaming reassembly by `index` with
  missing-index fallback = `len(dict)` not 0.
- **completions**: `prompt` is a list; max_tokens always literal.
- **embeddings**: no stream/max_tokens; parse RAISES on malformed (diverges from
  chat's degrade-to-None — note deliberately).
- **responses** (`openai_responses.py`, highest complexity): `input`/`instructions`/
  `max_output_tokens`; video rejected at format time; SSE event-type dispatch;
  replay `_REPLAY_UNSAFE_OUTPUT_ITEM_TYPES` filter; dedup-by-id union. The `-replay`
  file split is a Python file-size artifact — merge back in Rust.
- **Input-side ISL accounting** (`payload_extraction.py`) — tool-schema tokenization
  (`orjson.dumps(parameters)` to match server), pre-tokenised int-list counting,
  chat-template `messages` view. Feeds ISL metrics = comparability contract. The SSE
  leaf spec MISSES this entirely.

**Tier-2 endpoints (REDO-CLEANER, per-vendor field names PORT-EXACT):** rankings
family (nim/cohere/hf_tei — different payload+extraction shapes), image gen/edit
(base64 magic-byte MIME sniff), video (only async-poll lifecycle),
huggingface_generate (`token.text` not `generated_text`), raw/template (Jinja2 +
JMESPath — real dep cost, scope it), solido_rag (niche, skip until asked).

**Exporters — three distinct STAT_KEY orderings, all PORT-EXACT:** CSV
`STAT_KEYS`, JSON `JsonMetricResult` (genai-perf parity, `SCHEMA_VERSION="1.4"`,
`extra="allow"`), console `DEFAULT_STAT_KEYS`. `profile_export_aiperf.json` +
`.csv` field names/null-semantics are frozen downstream contracts. **Blood quirk:**
NaN/inf must round-trip via `model_dump + scrub_non_finite + orjson`, NOT
`model_dump_json` (pydantic coerces non-finite→null, colliding with explicit-None
"metric absent"). INTERNAL/EXPERIMENTAL filtering happens in the EXPORTER layer, not
the accumulator. Keep the two console warning exporters (OSL-mismatch,
usage-discrepancy) — earned thresholds + actionable fix-text.

**THROW-AWAY / DEFER:** `outputs_json_exporter` fragment-glob-and-merge (single
process: accumulate in RAM, write once — but keep the record schema + metric
allowlist); mlflow/wandb subprocess uploaders (defer); `exporter_manager` plugin
iteration (rebuild as ordered call list; keep the deferred-exporter + recorded
`profile_export_console.txt` concepts).

---

## 3. Config / CLI

**Headline: `config/` is already a from-scratch "Config v2.0" YAML system** with a
deliberate **envelope/body split** mirroring the K8s `AIPerfSweep` CRD
(`AIPerfConfig` envelope wraps `BenchmarkConfig` body). The spine's "keep Pydantic
ergonomics, redo in clap+serde" is right about the mechanism but the **type shape is
PORT-EXACT**, and several `config/` files are ALGORITHMS, not knobs.

**Hidden algorithm/policy living under `config/` (PORT-EXACT, claim in scheduling
spec):**
- `config/phases.py` — the SchedulingPolicy INPUT model (Concurrency/Poisson/Gamma/
  Constant/UserCentric/FixedSchedule phases; `prefill_concurrency`+ramp; stop
  conditions; seamless transitions; cancellation).
- `config/distributions.py` (531 LOC) — `Distribution` with runtime `.sample(rng)`
  (Fixed/Normal/LogNormal/Multimodal/Empirical). Reproducibility-critical.
- `config/adaptive_scale_phase.py` — closed-loop autoscaler config.
- `config/sweep/` (grid/zip/scenarios/Sobol/LatinHypercube + **AdaptiveSearch BO**)
  + `multi_run.py` (trials + `ConvergenceConfig`) + `config/resolution/` (BenchmarkPlan
  assembly + phase↔dataset compatibility).

**Config-loader ergonomics that are NOT free with serde (budget ~15 files):**
Jinja2 `variables:`, `${ENV:default}` substitution, duration strings (`5m`/`2h`),
singular↔plural + shorthand hoisting (`model→models`, `isl→prompts.*`), difflib typo
hints, consistent-seed auto-fill of 42, secret-redaction serializers on
api_key/headers/urls (MUST reproduce or artifacts leak secrets).

**CLI:** `profile` + `config` (init/expand/validate) + `plot` are MUST-HAVE; `chat`
is a cheap valuable smoke tool; `service` is THROW-AWAY. Flag zoo = flat `CLIConfig`,
**280 flags / 233 fields / 30 help groups** → direct clap target. **Non-mechanical
CLI policy the spec MISSES:** "magic lists" (`--concurrency 1,2,4` silently promotes
to a `sweep:` block) and `--search-recipe` expansion — must be re-decided, not
auto-derived.

**Runtime config split:** THROW-AWAY service_run_type/communication/workers_per_pod/
dataset_api_base_url + all `config/comm/**`; KEEP (as tokio knobs) ui / workers /
workers_min / record_processors / stats_interval / api_port.

---

## 4. Timing / orchestrator / search — the load-generation engine

**Critical disambiguation the specs conflate: there are TWO "adaptive" concepts.**
(1) **In-run AdaptiveScale** (`timing/strategies/adaptive_scale*.py`) — a
within-one-profiling-phase ramp-until-fail concurrency/users controller
(discover→sustain→complete SM, per-window SLA eval, sla-margin step sizing, sustain
recovery). THIS is the northstar's "adaptive workload." (2) **Outer-loop adaptive
search** (`orchestrator/` + `search_recipes/`, 62 files) — a multi-benchmark-run
SLA-boundary / Bayesian-optimization sweep coordinator. The northstar does not
describe (2) at all.

**PORT-EXACT timing semantics (specs miss the depth):**
- **UserCentricStrategy is entirely uncovered** — the 5th online strategy:
  virtual-history steady-state seeding, coprime-stagger turn spacing, heapq open-loop
  spawn schedule, late-response re-alignment. Also the substrate AdaptiveScale drives
  when `control_variable="users"`. Needs its own port design.
- **Cooperative yield-on-zero-interval** (`request_rate.py:160-163`) — asyncio
  deadlock guard in CONCURRENCY_BURST; needs a deliberate tokio `yield_now()`
  equivalent (return path as separate task), not a silent drop.
- **Two-dimension concurrency + debt/drain** (`concurrency.py`) — session slot
  (turn-0 acquire / final-turn release) + prefill slot (every-turn acquire /
  TTFT release); `DynamicConcurrencyLimit` debt-tracking for graceful drain on
  limit-decrease; global+phase layered limiter; `release_stuck_slots`. Spine has
  prefill-on-TTFT but not the debt/drain depth.
- **Multi-stage cancel-drain phase teardown** (`runner.py:621-710`): grace-timeout →
  cancel_all_credits → CANCEL_DRAIN_TIMEOUT → stuck-slot release → force-complete.
- **Ramp curve catalog** (`ramping.py`): Linear/Exponential/Poisson × stepped
  (concurrency, from 1) / continuous (rate). Richer than the generic "ramping" bullet.
- **Request cancellation simulator** (`request_cancellation.py`) — probabilistic
  client-disconnect, timer at request-fully-sent, warmup-disabled, derived RNG.
  Distinct from credit cancellation; unmentioned.
- Phase lifecycle SM, credit-counter atomicity (root vs total requests_sent for
  DAG/fork), ordered stop-condition chain (`applies_to_dag_children` split).

**`orchestrator/` is NOT ZMQ plumbing — it is the multi-run sweep coordinator
(real policy, keep):** variations×trials iteration (INDEPENDENT vs REPEATED),
ask/tell adaptive loop, artifact-dir tree (downstream plotters depend on
run_NNNN/trial_NNNN asymmetry), SHA-256 seed derivation, convergence criteria
(CIWidth/CV/Distribution ks_2samp), confidence aggregation, Pareto/SLA-filter.
SLA-boundary planners Monotonic + MultiTier are pure algorithms, cleanly portable.

**THROW-AWAY / scoping decisions:**
- `timing/manager.py`, `phase/publisher.py`, plugin lookups → tokio tasks + channels
  + static dispatch.
- **Subprocess-per-run isolation has no tokio analog yet** — today every sweep cell
  is a forked Python process because single-run `SystemController` calls `os._exit`.
  Single-process tokio needs a defined "run isolation" primitive (fresh `JoinSet` +
  scoped state teardown) or cell N's leaked tasks/slots poison cell N+1. Keep the
  `RunExecutor` trait + secret-redaction; drop the subprocess mechanism.
- **BO subsystem (SmoothIsotonic/Optuna/BoTorch, scipy)** — reimplementing BO in Rust
  is multi-month. Decision: port Monotonic/MultiTier + grid + convergence + confidence
  natively; leave SmoothIsotonic/Optuna/BoTorch as a thin **Python outer shell that
  shells out to the Rust single-run binary** (the `RunExecutor`/subprocess seam is
  already the natural FFI line).

**search_recipes/** = CLI-time preset → sweep-config compiler (declarative policy,
keep as a Rust enum+expander): max-throughput-{ttft,itl}-sla, concurrency-ramp,
prefill-ttft-curve, decode-itl-curve, max-concurrency-under-sla (5 styles),
max-goodput-under-slo, pareto-sweep. Numeric post-processors (knee detect, curve
fits, Pareto) are small REDO-CLEANER.

**BranchOrchestrator — SUPERSEDED by graph-IR.** Runtime DAG FORK/SPAWN dispatch
(`timing/branch_orchestrator.py` + 5 helpers, ~1200 LOC) is NOT a separate module to
keep — the graph-IR dataflow effort subsumes it. Conversation forking/spawning,
prerequisite fan-in, and the session-tree are the DAG the graph-IR runtime already
models. Do NOT port `branch_orchestrator.py` as its own thing; its semantics land in
the graph-IR port. (The credit-return interception path is therefore graph-IR's, not
a collision.)

---

## 5. Telemetry planes (gpu_telemetry / server_metrics / network_latency) — CLOSED

**CLOSED** by the telemetry-accumulators spec and built as `aiperf_runtime::gpu_telemetry`,
`aiperf_runtime::server_metrics`, and `aiperf_runtime::network_latency` (Clock-injected side-channel
accumulator modules feeding a shared accumulator seam). One authoritative revision to
note: the telemetry-accumulators spec's 2026-07-10 addendum replaces the
scrape-then-reconstruct window model described below with **phase-boundary counter
snapshots** — where the two conflict, the phase-boundary snapshot design is
authoritative and the `FINAL_SCRAPE_GRACE_NS` widening below is superseded. The
findings are retained as the research record of what the spec absorbed.

All three were ZMQ-service managers whose plumbing collapses to plain async tasks
writing into in-process accumulators. Domain logic below.

**Unify the counter-delta engine:** GPU energy and server counters implement the
SAME semantics — pre-window baseline + reset-clamp-to-0. The
`query_time_range(start,end)→mask` hook is IDENTICAL on both accumulators and is the
JOIN contract the accuracy-per-watt / EnergyEfficiencyAnalyzer spec rides on.
Preserve it.

**gpu_telemetry PORT-EXACT:** DCGM Prometheus scrape + fixed field/unit-scale table;
counter energy delta (pre-window baseline, reset-clamp); **`FINAL_SCRAPE_GRACE_NS`
≈666ms window-widening** (specs MISS this — the trailing counter scrape lands after
phase end; energy is systematically undercounted without it); cross-GPU efficiency
rollups (total_power, total_energy, output_tokens_per_joule, energy_per_user) = the
EnergyEfficiencyAnalyzer feedstock. PyNVML/AMDSMI collectors = REDO-CLEANER/defer
(need `nvml-wrapper`-style bindings; only if local-GPU/AMD is a target).

**server_metrics PORT-EXACT beyond the already-flagged OpenMetrics routing (all
spec GAPS):**
- **TRT-LLM JSON-at-`/metrics` reject → `/prometheus/metrics` fallback probe + URL
  swap** (empirically found).
- **Terminal auto-disable (anti-spiral)** — once classified non-Prometheus,
  short-circuit all future scrapes (else "30-min benchmark → 8 hr" parse-error spiral).
- **Histogram polynomial percentile estimator** (`histogram_percentiles.py`, 1250
  LOC, arXiv 2504.00001) — ~20% P99 error vs ~950% for standard linear interpolation.
  Large, self-contained, high-value; port near-verbatim.
- **vLLM/SGLang realtime metric atlas** (`accumulator.py:408-838`) — backend-specific
  name mappings (prefix-cache hit rate, KV-cache usage, queue depth, preemptions) with
  vLLM-first/SGLang-fallback + counter/gauge type-guards. Domain knowledge, keep.
- Unit inference (suffix + description-regex + ratio-vs-percent scale detection).
- NaN-dropping rationale is ZMQ-orjson-specific and DIES with ZMQ — revisit
  (`data_collector.py:363-500`), don't port the "taint whole histogram" behavior
  blindly.

**network_latency (in NO spec, but keep — easiest port, most orphaned):** TCP-connect
RTT calibration — fresh unpooled connection per probe, TLS skipped so http/https are
one uniform round-trip, DNS resolved once & cached, MIN_SAMPLES top-up at completion,
`--network-latency-mean` bypasses probing. Feeds the `network_adjusted_*` metrics
(§1). `set_network_rtt_ns` deliver-before-summarize becomes a direct hand-off in
single process.

---

## 6. Presentation plane (ui / api / plot) — the genuine spec blind spot

**Cross-cutting fact:** all three are passive consumers of the ZMQ bus via
`@on_message` tracker mixins whose STATE-AGGREGATION MATH is pure and
transport-agnostic (`ProgressTracker` phase%/req-s/ETA, worker/metric state). In
single-process tokio these consume a `broadcast`/`mpsc` channel off the collector.
**The math ports; the transport dies.** This is the seam that makes the presentation
plane "keep-thin" not "throw-away."

**OWN NOW (in-core):**
1. Console summary table + error/warning tables (`exporters/console_*`) — the primary
   CLI deliverable (PORT-EXACT conceptually; reproduce column groups/flag filtering
   with a Rust table lib).
2. Progress bars (`indicatif`) = the SIMPLE UI (REDO-CLEANER).
3. Tracker aggregation math fed by a tokio channel (REDO-CLEANER) — reusable heart of
   both UI and API.
4. **Parse + materialize `PlotEnvelopeConfig`** → `<artifact>/.aiperf-plot-config.yaml`
   receipt (PORT-EXACT; already in the Plot-Envelope spec — the one plot thing that
   belongs in the Rust core because it lives in the config plane).

**DEFER:**
5. `ratatui` live dashboard — thin view over the same channel; the Textual code is
   throw-away. Not parity-critical.
6. The entire **FastAPI `api/` server** — a k8s/Prometheus/dashboard backend for a
   multiprocess deployment (Prometheus text `/api/metrics`, `/healthz`/`/readyz`
   probes, `/api/results` streamed download). Capture the response-model contracts;
   build a thin `axum` server only if network observability is later required.
7. The entire **plot rendering engine** (~23k LOC Plotly/kaleido-Chrome/matplotlib/
   Dash) — delegate to the existing Python `aiperf plot` **sidecar** via the
   already-existing post-run `OnComplete` callback. Rust owns only the envelope config
   + receipt, passes preset dicts through opaquely.

---

## 7. Common substrate — domain nucleus vs multiprocess scaffolding

**Biggest miss: the real endpoint/timing/arrival CONTRACTS are not in `common/` —
they live in `plugin/enums.pyi`** (plugin-generated, but the stub has the concrete
literal sets): `EndpointType`, `TimingMode`
(adaptive_scale/fixed_schedule/request_rate/user_centric_rate), `ArrivalPattern`
(concurrency_burst/constant/gamma/poisson). Harvest the `.pyi` literals into Rust
enums.

**PORT-EXACT domain nucleus:**
- **`enums/metric_enums.py`** (807 LOC) — real unit-algebra: `MetricFlags` bitflags,
  time/size/over-time/power/energy/frequency/temperature units with `convert_to`
  chains, `MetricType`/`AggregationKind`/`MetricConsoleGroup`/`MetricValueType`.
  → Rust `bitflags!` + unit enum with conversion factors.
- **`CaseInsensitiveStrEnum`** dash/underscore/case folding (`foo-bar == foo_bar ==
  FOO_BAR`, hash on normalized form) — EVERY Rust enum `Deserialize`/`FromStr` must
  replicate or configs break.
- **`random_generator.py`** determinism substrate — SHA-256 seed derivation
  (`sha256(f"{root}:{identifier}")[:8]`), bounded rejection sampling, gammavariate
  arrival burstiness. **RESOLVED (RNG derive-system spec, CLOSED):** there is NO
  cross-language byte-parity requirement with the Python tool; native Rust seed
  derivation is locked to BLAKE3-derived order-independent seeds (`aiperf_runtime::rng`:
  `RngRoot::derive`, `RandomGenerator`, `HashIdRandomGenerator`), not Python SHA-256
  parity.
- **`models/sequence_distribution.py`** — 3-syntax ISL/OSL distribution parser
  (probabilities as percentages 0-100, a deliberate footgun-guard).
- **`path_safety.py`** — CWE-22 sanitizer (rejects symlink in leaf OR any parent —
  the non-obvious bit `resolve()` alone misses). Security-relevant, replicate exactly.
- **`redact.py`** — provider-auth-header frozenset + Bearer/Basic regexes.
- **`ConversationContextMode`** (4-way DELTAS/MESSAGE_ARRAY × with/without-responses)
  and **`ConversationBranchMode`** (FORK/SPAWN) + `PrerequisiteKind` — multi-turn
  accumulation + DAG-forking contracts. **SUPERSEDED by graph-IR** — these are the
  conversation-DAG semantics the graph-IR dataflow runtime already models; not a
  separate `common/` module to keep. The enum literals may still be needed as input
  vocabulary, but the STATE MACHINE lives in graph-IR.
- `compute_time_ns` (perf→wall reconciliation), `Usage`/`ErrorDetails` models,
  `compression.py` Accept-Encoding negotiation, `STAT_KEYS` ordering.

**REDO-CLEANER:** tokenizer routing (tiktoken-vs-HF `TIKTOKEN_ENCODING_NAMES` with
gpt2 excluded, `builtin`→o200k_base; HF-cache alias resolution) → `tokenizers` crate,
keep the routing DECISIONS; `AutoRoutedModel` discriminated-union router → free with
serde `#[serde(tag)]`; exceptions with behavioral semantics
(`IncompatibleMetricsEndpointError` triggers collector auto-disable).

**THROW-AWAY:** `protocols.py` (ZMQ socket topology), `hooks.py`
(@on_init/@on_message registry), all 20 `mixins/`, `messages/`, base_service/comms/
bootstrap/readiness_probe/singleton, custom logger (use `tracing` — but the custom
level set TRACE/NOTICE/SUCCESS is a log-parity UX contract).

---

## 8. Cross-cutting decisions this research surfaces (for the master ledger)

1. **Metric taxonomy = "struct + AggregationKind fold," not a heavyweight Metric
   trait** — unless user-extensible metrics are a hard requirement. The columnar
   engine collapses AGGREGATE to a single-column fold; most `total_*` derived metrics
   are `numeric_sum(tag)`. (Resolves the master ledger §4 open decision.)
2. **Cross-language byte-for-byte RNG reproducibility — RESOLVED: no** (RNG
   derive-system spec, CLOSED). No Python byte-parity requirement; native derivation is
   REDO-with-BLAKE3-order-independent-seeds (`aiperf_runtime::rng`), not PORT-EXACT SHA-256.
3. **BO/SLA-search subsystem: native Rust vs Python-outer-shell-shelling-out-to-Rust.**
   Recommend the shell for SmoothIsotonic/Optuna/BoTorch (multi-month to port);
   native for Monotonic/MultiTier/grid/convergence/confidence.
4. **"Run isolation" primitive for multi-run sweeps** — the single-process tokio
   replacement for subprocess-per-cell isolation. Undesigned; blocks the sweep layer.
5. **A `SchedulingPolicy` input model exists already — `config/phases.py`** — the
   scheduling spec should adopt it rather than invent one.
6. **Conversation/multi-turn/FORK state machine is SUPERSEDED by graph-IR** —
   `workers/session_manager.py`, `timing/branch_orchestrator.py`, `ConversationContextMode`,
   `ConversationBranchMode`/`PrerequisiteKind` are all the conversation-DAG the graph-IR
   dataflow runtime already models. Do NOT keep them as a separate module; fold the
   semantics into the graph-IR port. (The non-DAG, single-turn/linear dispatch path
   still needs its plain per-request execution loop, but that is not a "state machine.")
7. **UserCentricStrategy** needs its own port design (5th online strategy, uncovered).

---

## 9. One-line summary

The spine specs nailed the injection-seam architecture and the earned-in-blood
algorithms they named. This research found the risk in **five large unspec'd bodies**,
of which two are now CLOSED: (1) the sweep-line time-weighted metrics + the columnar
accumulator — **CLOSED** (metrics-accumulator + metric-catalog specs, built as
`aiperf_runtime::metrics_core`); (5) the telemetry counter-delta + histogram estimator +
backend metric atlas — **CLOSED** (telemetry-accumulators spec, built as
`aiperf_runtime::gpu_telemetry` / `server_metrics` / `network_latency`). The three still-open
bodies are (2) the endpoint payload/parse zoo + genai-perf export contracts, (3) the
already-v2 config system with hidden runtime algorithms, and (4) the timing engine's
depth (UserCentric, debt/drain, cancel-drain, yield-on-zero) plus the whole outer-loop
sweep/BO/SLA-search coordinator. (Multi-turn/FORK conversation semantics are NOT a
sixth body — they are superseded by the graph-IR dataflow port; the RNG-reproducibility
question is likewise resolved by the RNG derive-system spec.) Keep the presentation
plane thin (console + progress + tracker math in core; API and plot renderer
deferred/side-carred). The two decisions that most shape remaining scope are
**BO-native-vs-shell-out** and **the run-isolation primitive** for sweeps.
