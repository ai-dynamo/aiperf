<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf

> ✅ **CANONICAL PRODUCT ARCHITECTURE.** There is ONE product binary, `aiperf` (crate `aiperf-cli`): it is BOTH the human-facing entry point AND the execution engine. It owns `profile`/`config` natively (byte-exact Config v2), projects one protocol-v2 request, and drives execution by re-execing **itself** (`aiperf --execute`, an internal hidden mode) once per run/probe/cell over the stdio seam — preserving process/SIGINT/panic isolation with no separate executable. It delegates only peripheral subcommands (`plot`/`service`/`plugins`) back to Python — in-process embedded CPython in the `pyo3-embed` build, or a `python -m aiperf` subprocess in the lean default build (`AIPERF_NATIVE=0` falls the whole command back to the pure-Python frontend, which spawns the same `aiperf --execute`). Config v2's schema, outer orchestration, and presentation semantics stay Python-defined. `aiperf-mock-server` is a standalone developer/test inference target, not an orchestrated backend process. The `rust/runtime` package is a runtime library with no binary target (the binary is the `aiperf-cli` crate). There is no separate runner binary, no ZMQ, service mesh, multiprocess credit protocol, mmap dataset cache, or `plugins.yaml` on the native path. The `aiperf-rs` and `~/projects/aiperf-rust` trees remain **DEPRECATED**. Design record: [`specs/`](specs/); start-here index: [`llms.txt`](llms.txt).
>
> Rust execution truth is in `rust/` (the native CLI entry point is `rust/cli/`); canonical Python frontend truth includes `src/aiperf/config/`, `src/aiperf/orchestrator/`, and `src/aiperf/cli_runner/`. Other inherited Python controller/service code is legacy, not an alternate hot path.

## What this is

A Python-orchestrated, Rust-executed load generator + measurement front-end for inference servers. The runner dispatches OpenAI-compatible, Anthropic Messages, KServe, and NVIDIA Riva ASR/TTS/NLP requests over native HTTP/SSE or gRPC, records fine-grained timing (TTFT / ITL / TPOT / e2e / throughput / goodput), and serializes native results. The execution substrate is designed for **three interchangeable modes** over a single `{transport, clock}` seam:

1. **ONLINE-real** — real HTTP or gRPC to a real server, wall clock. *(built)*
2. **ONLINE-mock** — real HTTP to a mock server (`aiperf-mock-server`), wall clock. *(built — same code as (1), different target)*
3. **OFFLINE-mock** — in-process virtual-clock co-simulation of the Dynamo mocker engine, no sockets, deterministic. *(built behind the non-default `dynosim` Cargo feature; requires the sibling `dynamo-aiperf-native` checkout.)*

Online-real and online-mock are product-reachable through the native `aiperf` binary (entry point → self-exec `aiperf --execute`), or, with `AIPERF_NATIVE=0`, the pure-Python frontend spawning the same `aiperf --execute`. The workspace-owned `aiperf-mock-server` server is launched independently and supplies an ordinary online target URL; the frontend does not supervise it as part of a run. Offline-mock is product-reachable through the same Config-v2 path when the `aiperf` binary is built with `dynosim`; exact-image capabilities omit those transports from the base build.

## Canonical vs aspirational — the code is a walking skeleton

Ground every claim in `crate/src/file.rs`, not the specs: specs are design intent, **code is truth**. The current gaps between the north-star design and today's code:

- The north-star's `Backend` / `Engine` / `Harness` vocabulary is **aspirational**. Today's seam is `Clock` + `RequestSink<R>` / `RequestObserver` / `Dispatchable`.
- **One product entry path.** The native `aiperf` binary (or the Python frontend when `AIPERF_NATIVE=0`) projects one BenchmarkRun protocol-v2 request and re-execs itself (`aiperf --execute`); the execution engine is v2-only, rejects non-v2. Selects scheduled vs graph from dataset format; binds `cfg.transport.type`. Capabilities is an in-process `aiperf_cli::execute_mode::capabilities_catalog` function (no `--capabilities` argv mode). Unregistered transport/endpoint ids fail closed.
- **Native online transports are built.** HTTP: Clock-injected hyper (`transport_http`), h1/h2c/UDS/TLS/SSE, post-send cancellation. gRPC: Clock-injected Tonic (`transport_grpc`), KServe OIP + Riva ASR/TTS/NLP. Both on `current_thread` + `LocalSet`; no Python gRPC plugin.
- **Open endpoint registry, v2 validation, and frozen linked application are built.** `aiperf_runtime::endpoints`: open registry with 9 KServe + 9 Riva + `vllm_generate` factories, raw/effective config, worker-local prepared bindings. `aiperf_runtime::extensions` transactionally composes new dialects. `RunnerApplication` freezes the linked registry, input resolvers, independent transport and workload registries (the pair layer is deleted), and v2 coordinator at bootstrap.
- **Anthropic Messages parity is built.** `aiperf_runtime::endpoints::MessagesEndpoint`: exact PR 731 `/v1/messages` body, auth headers, all content shapes, streaming/non-streaming parsing, cache-usage reconciliation, thinking/signature replay. Graph transport remains Chat-shaped.
- **Offline and wall-clock Dynamo co-simulation are built** (behind `dynosim` feature). `aiperf_runtime::dynosim`: same `RequestSink` / `TurnDispatcher` / `GraphSink` over Dynamo's `SteppableReplay`. `dynosim_offline` = virtual clock; `dynosim_online` = wall clock, apples-to-apples with Dynamo's live driver. `dynamo-full` adds router/ZMQ/KV/AIC.
- **Dynamo replay is authored through `aiperf profile`**; no `aiperf dynosim` command. Select `transport.type: dynosim_offline|dynosim_online` in Config v2; trace file and concurrency/rate axes reuse the shared `dataset`/`phases` surface. Live mocker and replay-optimize sweep remain `python -m dynamo.*` tools.
- **Online scheduling policy is built.** `ScheduledRuntime` paces arrivals (Poisson/Gamma/constant/burst) with `SlotPool` admission and `StopChecker` bounds. `RequestRateWorkload`: one turn per tick, FIFO continuation priority. Graph mode: trait-backed root/arrival/admission/placement/failure policy.
- **Phase orchestration is built.** `aiperf_runtime::timing::phase`: `PhaseLifecycle`, `ClockPhaseRunner`/`Orchestrator`, duration→grace→cancel→drain→force escalation, warmup→profiling ordering, cancellation latch. Graph adapts through `PhaseExecutionFactory`.
- **Ancillary timing policy is built.** `aiperf_runtime::timing`: `RampStrategy`/`RampDriver`, seeded `CancellationPolicy`, `UrlSelector`. HTTP 499 anchored to send completion; in-process endpoint rejects URL selection; fixed schedules reject ramps.
- **Adaptive scale is built** as `aiperf_runtime::adaptive_core`: object-safe actuator/evaluator/step/window/controller; all four live actuators; `ramp_until_fail` controller; schema-v2 artifacts. Same futures online and offline.
- **Hash-derived RNG substrate is built** as `aiperf_runtime::rng`: `RngRoot::derive` / BLAKE3 seed derivation, `RandomGenerator`, `HashIdRandomGenerator`, sampler seams. Non-graph scheduler integration remains future work.
- **WEKA and Dynamo recorded-graph adapters are built.** `aiperf_runtime::graph::recorded`: strict WEKA/Dynamo decode, shared LCP-trie lowerer, dense segment interning. All content via `aiperf_runtime::rng` BLAKE3/PCG64; Python never parses or lowers either input.
- **Graph-IR trajectory-snapshot (t*) and warmup priming are built.** `aiperf_runtime::rng::numpy_pcg64` (byte-exact NumPy `SeedSequence`+PCG64, golden-vector proven), `aiperf_runtime::graph::tstar` (`WindowTStarSampler`/`TStarSampler`, per-trace seeded t* draw, cross-language parity), `aiperf_runtime::graph::snapshot` (`chop_trie_at_tstar` profiling, `rewrite_for_warmup` priming, `chop_trie_at_frontier` handoff resume; chains grouped by `conversation_id`), and `aiperf_runtime::graph::warmup_handoff` (`GraphWarmupHandoff`/`LaneHandoff`). The runner `graph_phase_runtime` splits t* per phase (warmup→priming, profiling→chop; default window = byte-identical full replay), emits a `trajectory_warmup_failed` v2 envelope on terminal warmup failure before profiling, runs a Clock-bounded in-runtime `GraphPressureRecycle` cache-pressure warmup with per-lane executed/return-wall observability, and consumes the warmup→profiling handoff once via `chop_trie_at_frontier`. The scenario config-lock stays in Python (`src/aiperf/common/scenario/**`); Rust consumes only the resolved v2 knobs (`trajectory_start_min_ratio`/`trajectory_start_max_ratio`/`t_star_random_seed`, `agentic_cache_warmup_duration`, projected by `rust_wire`). Current fidelity is a lane-0 baseline (per-lane t* salting / shuffle draws / bounded-recycle corpus-cursor continuation are documented future refinements); idle-gap warp parity vs Python `ActiveIdleWarp` is exact.
- **Performance metrics are built** as `aiperf_runtime::metrics_core`: 119-row catalog, NaN-sparse storage, ragged ICL, all sweep-lines, phase windows, worker merge, typed native-v2 `Reporter`. The native-Rust exporter plane (`aiperf_runtime::export`) is built and is the default sole emitter: aiperf-v1 (genai-perf-v1) JSON+CSV (including the GPU `telemetry_data` endpoint/GPU/metric summary), timeslice, server-metrics JSON/CSV/parquet, accuracy CSV, console.txt, OTLP per-record metrics, MLflow, and W&B — parity-verified byte-for-byte vs the retired Python exporters (`AIPERF_RUNTIME_NATIVE_EXPORT=0` restores legacy Python). An opt-in bounded-memory retention mode (`aiperf_runtime::metrics_core::MetricsStorageMode::Sketch`) streams each Record-metric value into a mergeable `cellular::sketch` t-digest instead of retaining it — exact counts/sums/min/max, approximate percentiles — for high-request-rate memory (`AIPERF_METRICS_SKETCH`, below).
- **Legacy static/stateful accuracy uses canonical Python evaluators.** Rust: scheduling, HTTP I/O, timing, metrics. Python worker: prompts, hidden tests, execution, scoring over JSONL stdio. Pinned: Harbor 0.18, AgentLab 0.4.2 + BrowserGym 0.14.3, MCPMark `cd45b7f57923b9b3985467f5139927575f83141c`. No Rust prompt builders, graders, or model clients. OSWorld/AppWorld still need canonical providers.
- **Compile-time extensibility is built.** One unified `AIPerfRegistry` / `AIPerfExtension` seam (`aiperf_runtime::extensions`) owns endpoints, dataset loaders, samplers, transports, workloads, exporters, and actuators over a shared `TransactionalRegistry<T>`. The stock composition is one ordered `with_builtin_extensions([...])` list whose only `#[cfg]` is feature-gate lines, and `--capabilities` auto-derives from the registered set. Static registration, duplicate rejection; no transport×workload pair map or compatibility predicate (any workload runs over any transport). `RunnerApplication` freezes at bootstrap. No `plugins.yaml`, runtime discovery, or dynamic ABI.
- **Dataset/segment unified store is built end to end** as `aiperf_runtime::dataset`. `Payload::TokenIds` + `Turn::raw_token_ids` for exact token arrays. `dag_jsonl` / `weka_trace` / `dynamo_trace` bypass the linear loader registry: runner-owned resolver → one compiler → frozen `SegmentStore`; no `Dataset` / `DagMetadata` intermediate.
- **Native content serving is built** (`aiperf_runtime::content_server`). `AIPERF_CONTENT_SERVER_ENABLED=true` + non-empty dir: stable image/video URLs, audio inline. gRPC, offline, and agentic/evaluation pairs reject the sidecar.
- **Telemetry archive/watch was removed; side-channel telemetry is built.** The legacy `aiperf-prometheus`, `aiperf-telemetry-archive`, `aiperf watch`, and `telemetry_watch` workload/pair are deleted. `aiperf_runtime::server_metrics` owns its own Prometheus/OpenMetrics parser. **GPU telemetry and network-latency calibration are built** as Clock-injected side-channel accumulator modules: `aiperf_runtime::gpu_telemetry` (DCGM-first source/decoder traits, canonical field scaling, boundary counter snapshots, cadence gauge distributions, supervised Python source, per-GPU series, energy/power/efficiency joins) and `aiperf_runtime::network_latency` (fresh TCP-connect RTT calibration with DNS resolution/fallback, per-target population stats, run-level mean, nonfatal structured failures). Both feed the shared side-channel accumulator seam. The feature-gated `aiperf_runtime::aic_runtime` builds an aiconfigurator timing engine onto the mocker's `perf_model` for the DynoSim path (part of the `dynamo-full` AIC surface).

When something is designed-but-not-built, this file says so. Do not assume a spec feature exists in the code.

## Crate workspace (`rust/`)

Workspace: `edition = "2024"`, `resolver = "3"`. Four product crates (plus `pyext`, a packaging-only pyo3 cdylib, and the `e2e` test harness); 16 former `aiperf-*` library crates are now modules of `aiperf` (see §Module organization below).

| Crate | Purpose | Key files |
|---|---|---|
| `loadgen-core` | Transport-neutral dispatch/measure seam + the collector. `Dispatchable`, `RequestSink<R>`, local-loop `RequestObserver` (no `Send`/`Sync`; f64-ms timestamps; optional `ObservedTokenKind` classification; terminal `ObservedUsage` with optional fields), `TraceCollector` → `TraceSimulationReport`; the `CollectorObserver` pure recorder. Zero engine/KV/HTTP deps. | `sink.rs`, `collector.rs`, `observer.rs` |
| `aiperf-runtime` | Library-only runtime composition used by the `aiperf` binary (crate `aiperf-cli`); there is no `src/main.rs` in this crate. It owns scheduled/transport composition, HTTP and gRPC prepared sinks, online pacing, datasets, exact token-array response observation/usage, the provider-neutral evaluation workload/typed host registry/fair arbiter/ledger/retry/scoped proxy/report join, legacy static and stateful accuracy seams, adaptive/ancillary policy, native report persistence, and the feature-gated direct raw-token Dynamo adapter. It also owns the single `AIPerfRegistry`/`AIPerfExtension` composition seam (`extensions/`) and — behind the `engine` Cargo feature — hosts the v2 execution engine `aiperf_runtime::engine` (protocol envelopes, the transport/workload registries, execution factories and `*_execution` drivers, coordinator, `RunnerApplication`, cellular controller/cell, control-plane HTTP, and the GPU/network/server side-channel accumulators); only `aiperf-cli` enables it, so `aiperf-mock-server`/`e2e` build without that layer. Sixteen former library crates are inlined as modules (see §Module organization). | `lib.rs`, `extensions/`, `engine/`, `evaluation.rs`, `evaluation/`, `dynosim.rs`, `ancillary.rs`, `metrics.rs`, `accuracy.rs`, `adaptive.rs`, `http.rs`, `grpc.rs`, `run.rs`, `phase_runtime.rs`, `scheduled.rs`, `report.rs` |
| `aiperf-mock-server` | Standalone online test/benchmark inference target: OpenAI chat/completions/embeddings, TGI, rerank, image, multimodal, and RAG routes; real SSE; analytic or batch-scheduler latency; deterministic token generation; prefix-cache policy; Prometheus backend dialects; and synthetic DCGM telemetry. Latency pacing runs on the `aiperf` RealClock `timerfd` (ns-precision `sleep_ns`, not the `tokio::time` 1 ms wheel). `--processes N` makes the launched binary a lightweight L4 round-robin balancer that spawns N child servers (same binary/config) on internal loopback ports and splices each connection to the next backend; `N=1` (default) is the unchanged single-process path. An optional `--grpc-port` (env `MOCK_SERVER_GRPC_PORT`) opens a second listener serving the KServe OIP v2 `GRPCInferenceService` (`ModelInfer`/`ModelStreamInfer`/`ModelReady`/`ServerLive`/`ServerReady`) over h2c, hand-routed on the shared hyper stack and reusing the same token/latency generation seam plus the `aiperf_runtime::transport_grpc::proto` prost messages the runner`\s gRPC client uses (no build-time protoc); it is HTTP-only in `--processes N` balancer mode (gRPC warned-and-skipped). An optional `--grpc-embedding-dim N` (env `MOCK_SERVER_GRPC_EMBEDDING_DIM`) turns the unary gRPC `ModelInfer` handler into a non-LLM embedder: it consumes the input text tensor and returns one `FP32` embedding tensor of dim N (shape `[1, N]`, deterministic, reusing the HTTP embeddings generator) instead of a generated `BYTES` text output, making the mock a target for AIPerf's `kserve_v2_embeddings` gRPC endpoint (STRING-in / FP32-out, no token semantics). An optional `--accuracy-dataset <path>` (env `MOCK_SERVER_ACCURACY_DATASET`) loads a JSONL `{prompt, ground_truth}` dataset and, for matched prompts, returns the grader-formatted correct answer for a seeded fraction (`--accuracy-correct-rate`): `--accuracy-format` selects the grader shape (mmlu/mmlu_pro/gsm8k/math/exact_match/passthrough), `--accuracy-cot-rate` renders chain-of-thought (in a separate `reasoning_content` field when `--accuracy-reasoning-field`), and `--accuracy-adversarial-rate` emits parser-choke shapes drawn from real issues (reasoning-only content #1136, an `object:null` stream frame #1010, plus whitespace/case/boxed/conflicting/unicode). Ground truth is loaded mock-side and keyed on the prompt — `--accuracy-match` selects the strategy (whitespace-normalized `substring` by default, plus `exact` and `_ci` case-insensitive variants; a per-row `match_key`/`id` matches a stable fragment of a few-shot-wrapped prompt) — AIPerf never sends it over the wire; verified e2e in `rust/e2e/tests/test_accuracy_mock.rs`. A live `GET /accuracy` endpoint (and `aiperf_mock_accuracy_*` Prometheus series on `/metrics`) reports the run's actual served tally — matched/correct/incorrect/unmatched/adversarial/cot and per-task `correct/matched` — as an oracle to compare against what AIPerf's grader reports. Additional built-out mock capabilities close runner-to-mock e2e gaps, each with an e2e raw-export test under `rust/e2e/tests/`: configurable error injection (`--error-status-codes` menu, `--error-retry-after`, mid-stream SSE errors via `--error-midstream-rate`); extended usage accounting (`--usage-*`: cache write/miss, audio tokens+seconds, accepted/rejected prediction, tool-use-prompt, Anthropic cache); `tool_calls`/function-call emission (`--tool-call-rate`); the OpenAI Responses API (`/v1/responses`), token-native vLLM generate (`/inference/v1/generate`), `/openai/v1/*` KServe aliases, and KServe HTTP v2-infer / v1-predict / `/v1/infer` routes; gRPC `ModelInfer` output-tensor variants for KServe rankings/images/VLM (`--grpc-behavior`); the NVIDIA Riva ASR/TTS/NLP gRPC services (`grpc_riva.rs`); a Unix-domain-socket HTTP listener (`--uds`); TLS/HTTPS + `grpcs` listeners (`--tls-cert`/`--tls-key`/`--tls-self-signed`, `tls.rs`); and the DCGM encoder/decoder/SM-active + vLLM external/CPU-cache + SGLang counter telemetry fields the runner decodes. It exports an Axum router for tests and a tuned Hyper server binary, but is not part of the runner dependency graph. | `app.rs`, `balancer.rs`, `grpc.rs`, `listener.rs`, `config.rs`, `handlers.rs`, `tokens.rs`, `latency.rs`, `scheduler.rs`, `prefix_cache.rs`, `metrics.rs`, `prom.rs`, `dcgm.rs`, `accuracy.rs`, `grpc_riva.rs`, `tls.rs`, `main.rs` |
| `aiperf-cli` | The ONE product binary `aiperf` (crate `aiperf-cli`, lib + bin): BOTH the native Rust entry point AND the execution engine. Owns `aiperf profile` (single run **and** sweeps) and `aiperf config init/validate/expand` natively — no Python on the profile/config path: idiomatic clap flags + serde Config v2 (byte-exact YAML keys/flags/defaults) → projection onto the `protocol_v2` request schema → re-execs ITSELF (`aiperf --execute`, an internal hidden mode intercepted before clap; also `--cell`/`--aggregator` for cellular) once per cell → aggregates → echoes the runner's `profile_export_console.txt` summary to the terminal. Comma-list flags expand to a native grid/zip sweep (`sweep/`, alpha-sorted axes, byte-exact per-cell requests + artifact-dir 5-row table + `base+idx` seeds), each cell a single-run request with the sweep envelope stamped. `--num-profile-runs N` repeats each variation over N trials (`--parameter-sweep-mode` REPEATED/INDEPENDENT, per-trial `run_NNNN`/`trial_NNNN` dirs, warmup dropped after trial 0), byte-exact vs the Python orchestrator. Single-trial cells aggregate into a byte-exact `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` (ported from `SweepAnalyzer` + the aggregate sweep exporters); the multi-trial confidence aggregate is **built** (`sweep/confidence.rs`): the non-sweep multi-run path writes `aggregate/profile_export_aiperf_aggregate.{json,csv}` (+ a collated `profile_export_aiperf_collated.json` under `--convergence-metric`), and the sweep path writes per-variation confidence aggregates plus the cross-variation `sweep_aggregate` with `best_configurations`/`pareto_optimal` and REPEATED/INDEPENDENT dir routing — the Student-t inverse CDF is a pure-Rust port (regularized incomplete beta + bisection, no scipy/pyo3), so CI bounds are close-but-not-bit-exact (no test pins them numerically). `config init` scaffolds from 28 embedded templates (`config/`), `config validate` resolves through the native YAML surface (which applies `${ENV}` substitution + Jinja2 `{{ }}`/`variables:` expansion via `expand.rs`, inline dataset `records:`, and the distribution discriminator), and a config-authored `sweep:` block (grid/zip over dotted paths with bare-name aliases, `sweep/yaml_sweep.rs`) expands to a native sweep previewable via `config expand`. Native profile/config are byte-exact vs Python for every flag they model (GenAI-Perf aliases included; zero pure-alias gaps; top-level `scenario`/`trajectory_start_*`/`unsafe_override`/`agentic_cache_warmup_duration` and the always-emitted `accuracy`/`endpoint_profiles`/`failure_policy` sections land natively). **The CLI NEVER shells out to the Python `profile` command** — full parity is being built out entirely in Rust (pure-Rust request projection for config/media/rankings/synthesis/scenario flags; accuracy request-projection flags are **built** (native), while accuracy **execution** uses the pinned Python evaluators — the sanctioned in-process/subprocess path per 'minimal pyo3 for the ML library and accuracy', never the Python `profile` command; the adaptive **SLA-search** subsystem is **built** — all four `max-concurrency-under-sla --search-style` styles run natively as dynamic ask-tell loops (one `aiperf --execute` child per probe, SLA metric read back from `native-v2.json`): `monotonic` is pure Rust; the default `smooth_isotonic` (PAVA+PCHIP), `bo` (BoTorchSampler+qLogNEI), and `optuna` (TPE/GP) run behind the opt-in `search-pyo3` feature, which embeds CPython and calls the real scipy/optuna — the shipped default binary stays Python-free). Unmodeled flags currently surface a clap error until ported. Peripheral subcommands are native pure-Rust (no pyo3): `analyze-trace` (mooncake ISL/OSL + cache-hit stats, byte-exact `PrefixAnalyzer` incl. NumPy percentile/std), `validate mooncake-trace` (schema validators), `speed-bench-report` (spec-decode acceptance matrix → byte-exact CRLF CSV), and `chat` (hyper streaming OpenAI client, http/https, + per-turn TTFT/TPS/ITL/cache stats; reuses the shared `read_sse` SSE reader, tiktoken/HF tokenizer, and an injected `Clock`), and `synthesize agentic-code` (the seeded multi-turn dataset generator; byte-exact `dataset.jsonl` on the byte-exact numpy `Generator` port `aiperf_runtime::rng::numpy_generator` — ziggurat normal + Lemire integers + weighted choice, verbatim from numpy's C source). The remaining subcommands wrap heavy Python subsystems — `plot` (matplotlib), `service` (the legacy Python mesh the native path replaces), `plugins` (registry/import validation): in the `pyo3-embed`/`search-pyo3` build they run the real `aiperf.entrypoint.main` **in-process via embedded CPython** (ZERO subprocess shell-out for the whole CLI); the lean Python-free build falls back to a `python -m aiperf` subprocess. Parity gated by golden vectors (`tools/parity/`: single-run `dump_golden.py`, sweep `dump_sweep.py`, sweep-aggregate `dump_sweep_aggregate.py`, and the dynamic SLA-search `dump_monotonic.py` / `dump_isotonic.py` / `dump_bayes.py` — byte-exact probe-sequence parity vs the real Python planners on the pure-Rust and seeded-TPE paths; scipy-bootstrap and botorch-GP paths behaviorally verified). | `main.rs`, `lib.rs`, `dispatch.rs`, `delegate.rs`, `profile.rs`, `flags.rs`, `load.rs`, `yaml.rs`, `sweep/`, `search.rs`, `isotonic.rs`, `bayes.rs`, `pyfit.rs`, `pyopt.rs`, `analyze_trace.rs`, `validate.rs`, `speed_bench.rs`, `chat.rs`, `synthesize/`, `config/`, `render.rs`, `execute.rs`, `exec_bin.rs`, `execute_mode.rs`, `mimalloc_options.c`/`build.rs` |
| `pyext` | Packaging-only pyo3 `cdylib`, compiled by maturin into `aiperf._native` for the **single `aiperf` wheel**. maturin requires a real binding target, so this tiny module is the wheel's compiled target; maturin also packages the `src/aiperf` frontend (`python-source = "src"`) but does NOT build or intern the unified `aiperf` binary. Instead `tools/wheel_repack.py` (run by `make wheel`) repacks the ONE unified `aiperf` binary (entry point + execution engine) directly into the wheel's scripts directory as `aiperf-<ver>.data/scripts/aiperf`, so the installed `aiperf` command IS the native binary (there is no `[project.scripts] aiperf` console-script entry). The pure-Python app is reachable as `aiperf-python` (`aiperf.entrypoint:main`) and `python -m aiperf`, and the native binary delegates every non-`profile`/`config` subcommand back to `python -m aiperf` (which stays pure Python — no exec loop). No `aiperf-runtime`/dynosim deps; carries only build metadata for `aiperf --version`/diagnostics; discovery uses `importlib.resources` and never imports this module. | `Cargo.toml`, `src/lib.rs` |

Dependency direction: `aiperf-cli` → {`aiperf-runtime` (with the `engine` feature), `loadgen-core`} and re-execs ITSELF (`aiperf --execute`) at runtime (a launched process, not a second binary); `aiperf-runtime` → {`loadgen-core`} plus optional `dynamo-mocker` only under `dynosim`; `aiperf-mock-server` → {`aiperf-runtime`}; `pyext` → {`pyo3`} only (no internal deps, and nothing depends on it). aiperf-cli and mock-rs do not depend on each other; real-network integration tests spawn the mock binary as an ordinary target.

## Module organization (`rust/runtime/src/`)

Sixteen former `aiperf-*` library crates are now `aiperf_runtime::<module>::` namespaces. All inter-module imports use `crate::<module>::` within `aiperf`; runner and mock-rs use `aiperf_runtime::<module>::`. Five modules use a `_core` or `transport_` prefix to avoid name conflicts (`metrics_core`, `adaptive_core`, `accuracy_core`, `transport_http`, `transport_grpc`). Full module table with purposes and key files: [`docs/module-organization.md`](docs/module-organization.md).

## The two seams (the whole architecture)

- **`{clock}`** (`aiperf-clock`): `RealClock` vs `SimClock` behind one `Clock` trait. `is_virtual()` selects the `drive_real` (tokio reactor drives) vs `drive_sim` (idle-pump: poll the `LocalSet` to quiescence draining all same-instant work → `advance_to(next_event_time)` waking heap-ordered sleepers → repeat) driver over the *same* executor. Virtual time is integer ns with an `(at_ns, seq_no)` deterministic tie-break — **never `tokio::time`** (its 1 ms timer wheel destroys µs/ns firing gates).
- **`{transport}`** (`loadgen-core::sink`): `RequestSink<R>::dispatch` drives a `Dispatchable` request to terminal and emits `on_arrival` / `on_admit` / `on_token` (or `on_classified_token` with `ObservedTokenKind::{Output,Reasoning}`) / terminal `on_usage(ObservedUsage)` / `on_terminal` through a `RequestObserver`. Classification defaults to `on_token`; usage defaults to a no-op and keeps unreported counts as absent fields. `RequestObserver` has no `Send`/`Sync` supertraits: each thread-per-core worker owns a local observer graph in `Rc`/`RefCell`; cross-thread consumers may still provide a thread-safe implementation. Real HTTP, native gRPC, mock HTTP, and the feature-gated in-process engine co-sim all feed this observer seam; `GrpcRequest` retains its prepared endpoint reference for `RequestSink` dispatch. TTFT is the first token callback; sinks emit no separate first-token event.

## Extensibility & porting discipline (non-negotiable)

- **Every extension point is a trait.** Anything that could ever have a second implementation — a transport, a clock, an accuracy evaluator, a request/response shape, an arrival pattern, a dataset loader, a sampler, a segment store, a metric accumulator, an analyzer, an exporter, an endpoint dialect, a tokenizer, a scheduling policy — MUST be an implementable `trait` (object-safe where it crosses a `dyn` boundary; generic where it is hot-path monomorphized) with at least one concrete impl behind it. Never hardcode a concrete type where a future variant is conceivable. If you are `match`-ing on an enum of "kinds" or branching on a string mode, that is a trait waiting to be extracted. In-tree precedent: `Clock`, `RequestSink<R>` / `RequestObserver` / `Dispatchable`, `AccuracyEvaluator`, `SegmentStore` / `PromptMaterializer`, `GraphSink`.
- **Always design ahead.** When you add code, add the seam the next plausible requirement will need — name the trait, take the trait (not the concrete) in signatures, thread the injection point — even if you ship exactly one impl today. The three-modes-for-free property only survives if features are written against the `{transport, clock}` seams, never against a specific backend/clock/transport; a feature that works in only one mode is a design bug. Note the extension you are leaving open in a `//!` / `///` doc comment.
- **Read the ENTIRE Python source before porting ANYTHING.** Before porting a behavior, read the WHOLE Python file end-to-end AND every file it meaningfully touches (its imports, the models it builds, the callers that consume its output, the tests that pin it). Never port from a docstring, a grep hit, a snippet, or a spec paragraph — those omit the earned-in-blood edge cases (SSE fast-paths, backward usage walks, finish/usage reconciliation, firing-gate rounding), which live in the parts of the file you assumed you could skip. Cite the exact Python `path:line` you ported from in the Rust `//!` / `///` docs, and guard it with a parity test wherever byte-exactness matters.

## Coding standards (Rust)

- **SPDX header** (`// SPDX-FileCopyrightText …` + `// SPDX-License-Identifier: Apache-2.0`) atop every source file. `//!` module docs; `///` doc comment on every public item.
- **Thread-per-core**, not work-stealing, on the hot path: N OS threads, each a `current_thread` tokio runtime + `LocalSet`; per-trace state is `Rc`/`RefCell`, futures are `!Send`, tasks are `spawn_local`; parallelism = many traces across threads. See `aiperf-graph/src/runtime.rs`, `transport_bench.rs`. The runner's online `run.rs` path follows the same `!Send` model on a single `current_thread` runtime + `LocalSet`, with `Rc` observers and dynamic `SlotPool` admission.
- **All time through `Clock`** in the clock-aware crates (`aiperf-transport-http`, `aiperf-transport-grpc`, `aiperf-graph`, and the runner/library online path): never `Instant::now()`, `SystemTime::now()`, or raw `tokio::time` for measurement or firing gates. The relocated OpenAI SSE chunk types (`aiperf-transport-http::sse`) and the `CollectorObserver` recorder (`loadgen-core::observer`) own no clock; callers supply Clock-derived timestamps.
- **No `Arc`/`Mutex` on hot paths.** Accumulate lock-free per-thread / per-worker and merge once at the end (the graph bench keeps a per-worker accumulator and merges at the join) — never contend a shared collector lock per token on the throughput path.
- **Content-addressed segments**: blake3, prefix-dependent (fold the parent id into the hash so shared prefixes dedup and identical text under different prefixes stays distinct); materialize = clone/concat pre-serialized bytes, never re-serialize.
- **mimalloc** is installed by the `aiperf` binary (crate `aiperf-cli`); per-request allocation churn in the graph executor and streaming client was the top profiled hotspot.
- **Loopback benchmarking**: the hyper transport never consults the ambient `HTTP_PROXY`, so localhost traffic is not proxied (an ambient proxy 405s localhost and tanks throughput).
- **SSE**: buffer raw bytes, split lines on the byte buffer, UTF-8-decode only complete lines (a multibyte char may straddle a TCP chunk boundary).
- **Authoritative token counts**: request `stream_options.include_usage` so `usage.completion_tokens` is returned; the HTTP sink emits it through `RequestObserver::on_usage`. Adaptive windows and the collector-wide native accumulator reconcile OSL and the `(last−first)/(osl−1)` ITL denominator to authoritative completion usage while preserving observed chunk timings.
- **Errors**: `anyhow` in the runner/app layer; library crates use plain error enums with hand-written `Display` (no `thiserror`).
- **Python defines the Config v2 schema** (mirrored byte-exact by `aiperf-cli`'s native serde Config v2). Rust runner requests use strict `serde` / `serde_json` DTOs with unknown-field rejection. Prefer a direct-serialized request body on the hot path.
- **NaN/Inf discipline**: numeric metric values crossing a serialization boundary must be finite or explicitly absent.
- **Comments explain *why*, never *what*.** No emojis in code. Read the actual code, never trust the docstrings or comments.

## Build, test, run

The (inherited-Python) `Makefile` has cargo-backed packaging targets (`make bundle-cli`, `make wheel`) plus native-entrypoint targets (`make native-cli` builds the pure-Rust `aiperf` binary; `make install-native` places it in `dist/native-bin/` for a Python-free profile/config flow); for everything else use cargo directly:

```bash
cargo build                  # debug build of the whole workspace
cargo build --release        # optimized — use for any throughput number
cargo test                   # all unit tests (self-contained: in-process axum mock, no external server)
cargo test -p aiperf-runtime # just the runtime library crate
cargo test -p aiperf-mock-server # standalone mock-server unit + HTTP integration suite
cargo clippy --all-targets   # lints
cargo fmt                    # format (rustfmt)

# Product runner with offline scheduled + graph pairs; expects the sibling checkout.
cargo build -p aiperf-cli --features dynosim
# Focused library algorithms remain independently testable.
cargo test -p aiperf-runtime --features dynosim --lib
# Complete native build: router runtime, ZMQ events, KV offload, AIC, and profile support.
cargo build -p aiperf-runtime --features dynamo-full

# Native entry point with the scipy/optuna-backed dynamic SLA-search styles
# (`--search-style smooth_isotonic|bo|optuna` for `max-concurrency-under-sla`).
# OFF by default so the shipped `aiperf` binary is Python-free; ON embeds CPython
# and calls the real scipy/optuna in-process (`monotonic` needs no feature).
cargo build -p aiperf-cli --features search-pyo3

# Base embed-CPython feature alone (implied by search-pyo3): routes every
# non-profile/config subcommand through in-process `aiperf.entrypoint.main`
# instead of a `python -m aiperf` subprocess — zero shell-out for the whole CLI.
cargo build -p aiperf-cli --features pyo3-embed

# Standalone online mock target for local integration runs.
cargo run -p aiperf-mock-server -- --fast

# Same mock, additionally serving the KServe OIP v2 gRPC service on :8001
# (target for `transport.type: grpc`, `grpc://127.0.0.1:8001`). HTTP-only under `--processes N`.
cargo run -p aiperf-mock-server -- --fast --grpc-port 8001
```

Package the single `aiperf` wheel (maturin backend, `pyproject.toml`). There is
**one** `aiperf` distribution: maturin compiles the `pyext` pyo3 module into
`aiperf._native` and packages the `src/aiperf` frontend (`python-source = "src"`);
`tools/wheel_repack.py` (run by `make wheel`) then repacks the one unified `aiperf`
executable into the wheel's scripts directory as `aiperf-<ver>.data/scripts/aiperf`
— no separate runner wheel. Because it carries a native binary
the wheel is platform + CPython-ABI specific (no longer `py3-none-any`); the full
execution surface (`--features full` = `dynosim` + `parquet` + `velo`) requires the
sibling `dynamo-aiperf-native` checkout at build time (the default `--features parquet`
build is sibling-free):

```bash
# Build the unified aiperf binary (lto=fat, CLI_FEATURES-selectable).
make bundle-cli
# bundle-cli + `maturin build` + `tools/wheel_repack.py` -> one aiperf wheel.
make wheel
# Editable dev install builds and installs the unified binary (make install-app runs bundle-cli);
# AIPERF_EXEC_BIN overrides the execution child to a dynosim/custom-features build.
```

Run the product:

```bash
# Generate, validate, and run through the human-facing `aiperf` entry point.
aiperf config init --template minimal --output benchmark.yaml
aiperf config validate benchmark.yaml
aiperf profile --config benchmark.yaml

# Externalize generated images/videos through the run-owned native server.
mkdir -p /tmp/aiperf-content
AIPERF_CONTENT_SERVER_ENABLED=true \
  AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
  aiperf profile --config benchmark.yaml

# Cellular (multi-process) mode: `--cells N` (or `runtime.cells: N`) makes the launched
# aiperf a controller that spawns N `aiperf --cell` children over a
# (cell_id, cell_count) budget partition and merges their records into one report.
# `--cells 1` (default) is the unchanged single-process path. Supported for synthetic
# and file/public runs over the http OR grpc transport with request-bounded phases (gRPC
# runs the SAME cell executor as http — the cell issuer + records shipper live above the
# transport); a run seed and single URL are NOT
# required (seedless auto-derives a shared seed, multi-URL round-robins cell-locally,
# ramps/rate/cancellation are aggregate-equivalent, warned). MULTI-TURN runs (a `sessions`
# / --num-conversations budget) partition per CONVERSATION and are supported on the
# exact-fold merge path (metrics-only concat merge, order-independent): the controller
# slices the sessions budget per cell (owned_positions, tiles exactly) and rejects
# multi-turn on the retain path (a live-reply inputs.json forces retain) with a clear
# message. Graph programs (dag_jsonl/weka_trace/dynamo_trace) also partition across cells
# (trace-level, concatenation-merged; a graph --sketch-metrics cell folds each record into
# its t-digest and ships a sketch StorePartition, merged associatively like exact-fold).
# Fails closed on the dynosim offline/online transport (a separate SimClock executor with
# no cell-issuer/records wiring) + scheduled duration/adaptive, and on multi-turn with a
# random sampler (sequential/shuffle only).
# E2e: test_cellular.rs + test_cellular_multiturn.rs + test_grpc_cellular.rs (scheduled) +
# test_graph_cellular.rs (incl. sketch store-ship).
aiperf profile --config benchmark.yaml --cells 4

# Native multi-value sweeps: any comma-list flag (concurrency/request-rate/request-count/
# isl/osl/num-conversations/benchmark-duration) expands to a grid (default) or zip sweep,
# one `aiperf --execute` per cell, aggregated into `<dir>/sweep_aggregate/` with a live table.
aiperf profile --model M --url 127.0.0.1:8000 --endpoint-type chat --concurrency 1,2,4 --request-count 100
aiperf profile --model M --url 127.0.0.1:8000 --endpoint-type chat --concurrency 2,4 --request-rate 5,10 --sweep-type zip

# Native config workflow (no Python): scaffold, validate.
aiperf config init --list
aiperf config init --template minimal --model M --url 127.0.0.1:8000 --output benchmark.yaml
aiperf config validate benchmark.yaml

# Capabilities/inventory is an in-process function now (no `--capabilities` argv
# mode): `aiperf_cli::execute_mode::capabilities_catalog()` composes the stock
# application and returns the linked catalog.

# Offline/online Dynamo replay is a Config-v2 transport, authored through `aiperf profile`.
# (transport.type: dynosim_offline|dynosim_online). The live mocker server and the
# replay-optimize sweep stay native tools: `python -m dynamo.mocker`, Dynamo's profiler.
aiperf profile --config dynosim_offline_replay.yaml
```

`cargo run -p aiperf-runtime` is invalid: the `aiperf-runtime` package has no binary (the native binary is the `aiperf-cli` crate). The entry point projects one side-effect-free authored-v2 request; an unregistered transport or workload fails closed without conversion to v1. `dag_jsonl`/`weka_trace`/`dynamo_trace` enter the runner-owned graph-input resolver once, call exactly one compiler, and never pass through a second registry. gRPC endpoints require `transport.type: grpc` with `grpc://`/`grpcs://` URLs. Offline/online replay requires `dynosim_offline`/`dynosim_online` and a feature-bearing runner.

Content server: `AIPERF_CONTENT_SERVER_ENABLED=true` + non-empty `AIPERF_CONTENT_SERVER_CONTENT_DIR`. See `docs/tutorials/content-server.md`.

Per-record columnar sidecars: the `artifacts.records` format list takes `jsonl`, `csv`, and `parquet` (any subset, e.g. `records: [jsonl, csv, parquet]`). `csv` writes a flat `profile_export_records.csv` (metadata cols + one column per metric named `{Header} ({unit})` — the summary-CSV unit convention via `RecordMetricColumn::csv_display_name` — + `error_code`/`error_type`/`error_message`, `trace_*` when `artifacts.trace`), ported from the legacy Python `RecordExportCsvResultsProcessor` but re-styled from that port's `{tag}_value`/`{tag}_unit` pairs to match `profile_export_aiperf.csv`. `parquet` emits a wide, columnar `profile_export.parquet` beside the per-request `profile_export.jsonl` (one row per request, one nullable column per catalog record-metric, units in `aiperf.units` file metadata; flat `trace_*` columns when `artifacts.trace`). Runner-owned artifact, not an `Exporter` over `NativeReport`: the per-record data lives only at the `CapturedRecord` callsites, so `rust/runtime/src/engine/records.rs::write_records_{parquet,csv}` drive the writers at both the scheduled and graph callsites (parquet via `aiperf_runtime::export::per_record_parquet`, gated on the `parquet` feature — a lite runner warns and skips; CSV is stdlib and always available). The metric column set is the shared `aiperf_runtime::metrics_core::record_metric_columns`. Independent of the JSONL and force-disabled on the DynoSim/sketch paths (`rust_wire` strips `records_parquet_path`/`records_csv_path`).

Execution-engine off switch: `AIPERF_RUNTIME_ENGINE=python` (default `rust`) routes a single `aiperf profile` run through the legacy pure-Python service mesh (`SystemController` + Worker/TimingManager/RecordsManager children) instead of the native `aiperf` execution path, for A/B benchmarking the old hot path against the Rust core on an identical `BenchmarkRun`. Enum field `Environment.RUNTIME.ENGINE` in `src/aiperf/common/environment.py`; `rust` (default) rejects unknown values.
RNG backend switch: `AIPERF_RNG_BACKEND=rust_parity` (default `legacy`) swaps the seeded random substrate from Python`\s Mersenne Twister + NumPy (SHA-256 seed derivation) to a pure-Python byte-exact port of the Rust `aiperf_runtime::rng` `Pcg64` + BLAKE3 substrate (`src/aiperf/common/rng_parity/`), so seeded Python and Rust produce identical streams in tests. Enum field `Environment.RNG.BACKEND` in `src/aiperf/common/environment.py`; `legacy` (default) is unchanged. Parity is proved against committed Rust golden vectors (`rust/runtime/examples/rng_parity_vectors.rs` -> `rust/runtime/tests/data/rng_parity_vectors.json`, replayed by `tests/unit/common/test_rng_parity.py`).

Export-plane off switch: `AIPERF_RUNTIME_NATIVE_EXPORT=0` (default `1`) restores the legacy Python emitters (the `ExporterManager` data/console exporters, the mlflow/wandb post-run uploaders, and the OTel live-streaming sidecar) instead of the native `aiperf_runtime::export` sink plane, for A/B verification (mirrors `AIPERF_RUNTIME_ENGINE=python`). By default the native Rust sinks are the sole emitter of `profile_export_aiperf.{json,csv}`, timeslices, `server_metrics.{json,csv,parquet}`, `accuracy_results.csv`, `profile_export_console.txt`, and the OTel/MLflow/W&B network sinks: the frontend projects `cfg.export` (`rust_wire._export`) whenever the config signal is present, suppresses the live-streaming sidecar (`rust_wire._live_streaming`), and skips `native_report.export_python_compatibility_reports`. Bool field `Environment.RUNTIME.NATIVE_EXPORT` in `src/aiperf/common/environment.py`.

Metrics sketch off/on switch: `AIPERF_METRICS_SKETCH=1` / `aiperf profile --sketch-metrics` (default `0`) opts one run into bounded-memory metric retention: the accumulator processes each record into a transient single-row scratch, harvests its finite values into a per-`(phase, tag)` t-digest (`aiperf_runtime::metrics_core::MetricsStorageMode::Sketch`, reusing `cellular::sketch::TDigest` at `AIPERF_METRICS_TDIGEST_COMPRESSION`) and clears the row, so accumulator memory is O(1) in the record count; the online path folds each finalized record into the sketch and drops it (`RunCapture::finish_fold_into`). Counts/sums/averages/min/max stay exact and rate derivations stay exact from the min/max timestamp aggregates; percentiles become approximate (with a streaming Welford `std`), and per-record artifacts (records/raw/outputs JSONL, per-record OTLP) plus per-row-only outputs (timeslices, per-model/endpoint inference series, sweep curves) are unavailable — dropped from the run request in `rust_wire` and fail-closed in `execute.rs::validate_plan`. Bool field `Environment.METRICS.SKETCH` in `src/aiperf/common/environment.py`. Fold-and-drop is per-worker on BOTH the single-thread and thread-per-core sharded paths — each worker folds every completed record into its own bounded accumulator and drops it as it streams (`ShardRecords::Folded`; the single-thread path skips the end-of-run drain), retaining only errored records — so per-cell peak RSS is O(shards × sketch + concurrency), not O(records); only the retain path (`AIPERF_RUNTIME_EXACT_FOLD=0`) keeps every record until drain, by design (byte-exact global-order merge + per-record artifacts). Cellular sketch runs are supported (tier T1 of the k6-parity plan): a `--cells N` sketch cell folds into its bounded sketch and ships the folded store (`CellMessage::StorePartition`, the same wire form exact-fold uses), which the controller merges associatively (`merge_store_partitions` → `ColumnStore::append_store` → t-digest merge) into an O(cells × sketch) report — counts/sums/extrema exact, percentiles approximate. The record total travels with the store (`ColumnStore::ingested_count`), since a sketch store retains no rows (`record_count() == 0`).

Unit tests use in-process axum endpoints. `tests/scheduled_real_mock.rs` retains real wall-clock library coverage; the unified-binary product coverage lives in `rust/cli/tests/` (spawning `aiperf --execute`).

## Feature-complete definition — e2e raw-records verification (MANDATORY)

A feature (new endpoint, metric, transport, dataset shape, cellular/graph capability, streaming/fold path,
…) is NOT "complete" — never call it done, shipped, or verified — until it has an **end-to-end test that
verifies the raw per-record JSONL against a properly-tuned mock server**. Unit tests, a smoke run, or a
summary/count-only assertion do NOT satisfy this bar.

- **Properly-tuned mock (determinism):** drive the canonical Python frontend (`aiperf profile`) against
  `aiperf-mock-server` configured so every value is exactly predictable — fixed `--ttft` and `--itl`,
  `--ttft-jitter-cv 0 --itl-jitter-cv 0`, analytic mode (scheduler off), and fixed synthetic ISL/OSL
  (`--synthetic-input-tokens-stddev 0`, `--output-tokens-stddev 0`, a pinned `--tokenizer`).
- **Verify TIMING** in `profile_export_raw.jsonl` per record: TTFT (first-token perf_ns − request start),
  ITL (mean gap between *generated-token* chunks only — EXCLUDE the terminal usage/`[DONE]` chunk, which
  arrives ~0 ms after the last token and otherwise dilutes ITL), and request_latency
  (≈ `ttft + (osl−1)·itl`), asserted against the mock's fixed values within a small transport-overhead
  tolerance (~1–2 ms) — NOT a wide band.
- **Verify DATA** per record: OSL = count of generated-token chunks equals the requested cap; ISL, model,
  streaming flag, response content, and status/error fields are present and correct.
- **Both timing and data**, at the raw-record level. The e2e harness (`rust/e2e/tests/common/`) exposes
  `raw_records()` (reads `profile_export_raw.jsonl`) for exactly this; use `--export-level raw`.

Use this as the definition of done in briefs, reviews, and before claiming any feature verified.

## Design specs (`specs/`)

Read for intent; verify against `rust/` for reality. Full index with status and one-liners: [`specs/README.md`](specs/README.md).

## Adding things

- **A new transport** → implement `RequestSink<YourReq>` (+ `Dispatchable` for `YourReq`) in its own crate; emit `on_classified_token` when output versus reasoning is known (otherwise `on_token`) and one terminal `on_usage` observation with optional fields; nothing in `loadgen-core` changes.
- **A new clock / execution mode** → implement `Clock`; `drive_sim` / `drive_real` already dispatch on `is_virtual()`.
- **A new graph feature** → the `executor` / `segment` / `channel_store` modules in `aiperf-graph`; keep firing-gate arithmetic byte-exact (see the graph-IR spec's parity contract).
- **A new accuracy benchmark** → for provider-neutral evaluation, implement semantics inside a pinned evaluator provider and register only an exact immutable distribution/task manifest, factory-owned public projections, required typed host operations, isolation proof, parity evidence, and product subprocess proof. Never add a Rust prompt builder, grader, or evaluator inference client. Until the benchmark's exact migration/deletion gates pass, static tasks remain in the pinned Python/Lighteval worker and stateful families remain behind their pinned Harbor, AgentLab/BrowserGym, or MCPMark `AgenticHarnessProvider` / `AgenticHarness` path.
- **A new metric** → add its `MetricSpec` in `aiperf-metrics::catalog`, implement its record/aggregate/derived computation in `store.rs` / `accumulator.rs`, and extend `RecordIngest` plus the runtime adapter only when a new raw fact is required. The native reporter consumes accumulator results without a per-metric serializer branch.
- **A new synthetic-media delivery method** → implement `SyntheticMediaPublisher`; keep codec/generation in `SyntheticMediaGenerator`, select the publisher through `SyntheticMediaGeneratorFactory`, and return exact endpoint-ready bytes to the composer.

## Keeping these docs current (MANDATORY)

These four agent files, `specs/README.md`, and root `llms.txt` are the architecture map. They go stale the instant code or specs change. **Whenever you add, modify, remove, or implement any architecture, update the map IN THE SAME CHANGE — it is part of the task, not optional follow-up.** Explicit triggers -> required edits -> verify:

| When you… | Edit | Verify |
|---|---|---|
| Add a spec to `specs/` | `specs/README.md` index row + `llms.txt` specs index + the "Design specs" section (all four agent files) | sync check |
| Modify / rename / remove a spec | the same three places (fix row, links, filename) | sync check |
| **Implement** a designed feature (designed -> built) | flip its flag in "Canonical vs aspirational" + the crate-table built/designed note + `specs/README.md` status column; delete the stale "not built" caveat | `cargo build`, sync check |
| Add / remove / rename a crate | crate topology table + the dependency-direction line (all four) + `llms.txt` crate table | `cargo build`, sync check |
| Change a seam (`Clock` / `RequestSink` / `RequestObserver` / `Dispatchable`) or a trait method | "The two seams" section + "Adding things" + `llms.txt` seam summary | `cargo build`, sync check |
| Add / change a Python CLI/config surface, runner wire feature, or build/run command | "Build, test, run" (all four) + `llms.txt` | run the command or runner subprocess proof |
| Deprecate / un-deprecate a sibling tree | the CANONICAL banner (all four + `README.md`) | sync check |
| Contradict / supersede a decision in an existing spec | append a dated `## Addendum` at the END of that spec (NEVER edit its body) + note the supersession in `specs/README.md` | — |

**Rules:**
- Edit ALL FOUR agent files together (identical body) and ALWAYS finish with `python tools/check_agent_files_sync.py` (or `make check-agent-files-sync`) — non-zero exit = bodies diverged; fix before committing.
- `specs/README.md` and `llms.txt` are NOT sync-checked but MUST move in lockstep — a spec/crate/seam change that leaves them stale is an INCOMPLETE change.
- Ground every claim in `crate/src/file.rs`. State designed-but-not-built explicitly; never describe intent as reality.
- **Never rewrite a shipped spec to contradict it.** Specs are an append-only historical record. If a decision or implementation supersedes, revises, or contradicts an already-written spec, do NOT edit that spec's body — append a dated `## Addendum — YYYY-MM-DD` at the END of the spec stating what changed, why, and which section/claim it supersedes. The original text stays; the addendum is authoritative where they conflict. Record the supersession in the `specs/README.md` status column.
- Put the doc updates in the SAME commit as the code/spec change they describe.
- **Enforced by tooling:** `tools/check_docs_current.py` fails a change that touches `specs/` or adds/removes a crate without also moving `specs/README.md` / `llms.txt` (and, for crates, the four agent files). It runs as the `check-docs-current` pre-commit hook and the "Rust Docs Guard" CI workflow; run `python tools/check_docs_current.py` locally before committing. Bypass only with `DOCS_GUARD_SKIP=1`, and justify it in the commit message.

### Agent-instruction file sync (mechanics)

`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` (name is Python-legacy; kept so the checker's target list matches) share a **byte-identical body** below their per-tool headers. Only the header differs: the cursor file keeps its YAML frontmatter (`alwaysApply: true`) then the SPDX comment; the other three start with the SPDX comment. The body begins at the first `# AIPerf` H1. Edit all four together and verify with `python tools/check_agent_files_sync.py` (or `make check-agent-files-sync`).
