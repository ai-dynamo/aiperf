<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python-to-Rust silent behavior change audit — consolidated findings

## Scope and method

Eleven parallel audits compared the Rust implementation in `rust/` against the
Python implementation, looking for one specific class of defect: a feature
present in **both** implementations where the same user input silently produces a
**different observable result**, with no error, warning, or documented notice.

Out of scope by construction: features only Rust has, architectural differences,
performance, and anything Rust refuses loudly.

### Baseline

The Python reference is **`origin/main` @ `bc359bf8fd`**, materialized as a
detached worktree at `/mnt/4tb/aiperf-parity-py-main`. `origin/main` contains no
`rust/` tree, so the port is entirely branch-local and the two sides cannot be
confused.

This matters: the first pass mistakenly used the branch working tree as the
Python reference. That tree is 4345 commits ahead of `origin/main` and has 132
locally-modified Python files concentrated in `config/` (41), `common/` (26), and
`dataset/` (21). Every finding below was re-verified against the correct
baseline, and each domain report carries a `## Withdrawn after baseline
correction` section.

## Reading the severity column

- **P0** — silently produces wrong results, invalid measurements, lost data, or a
  hung/never-terminating run.
- **P1** — materially changes workload semantics, reported numbers, artifacts, or
  operational behavior.
- **P2** — narrower modes, diagnostics, cosmetics with informational content.

## Domain reports

| # | Domain | Report | P0 | P1 | P2 |
|---|---|---|---|---|---|
| 01 | CLI flag surface | `01-cli-flags.md` | 1 | 10 | 3 |
| 02 | Config schema (YAML v2) | `02-config-schema.md` | 0 | 6 | 8 |
| 03 | Endpoint request payloads | `03-endpoint-payloads.md` | 1 | 3 | 6 |
| 04 | Response parsing & tokens | `04-response-tokens.md` | 2 | 5 | 4 |
| 05 | Dataset & prompt generation | `05-dataset-prompts.md` | 0 | 2 | 4 |
| 06 | Load generation & timing | `06-timing-load.md` | 0 | 9 | 1 |
| 07 | Metrics definitions & math | `07-metrics.md` | 0 | 3 | 5 |
| 08 | Artifacts & export formats | `08-artifacts-exports.md` | 1 | 11 | 2 |
| 09 | Console output & process behavior | `09-console-ux.md` | 2 | 7 | 4 |
| 10 | Telemetry side channels | `10-telemetry.md` | 1 | 4 | 4 |
| 11 | Accuracy & multi-run orchestration | `11-accuracy-sweep.md` | 1 | 2 | 5 |
| 12 | Control hooks | `12-control-hooks.md` | 0 | 3 | 1 |

Counts are post-deduplication within a domain but not across domains; see
"Cross-domain duplicates" below.

## P0 — silent invalid results

### 1. A user-centric run never terminates

Found independently by two agents from different directions (flag resolution and
phase construction), both tracing to the same root cause.

`--user-centric-rate 5 --num-users 8` is bounded at 10 requests upstream by the
default-request-count fallback, which excludes only `FIXED_SCHEDULE`. Rust
excludes `user_centric` from that fallback (`resolve.rs:1353-1359`) and builds the
phase from the raw count (`:1373`), so the benchmark runs forever. The user sees a
run that simply never finishes.

Reports: `01-cli-flags.md` (P0), `06-timing-load.md` (finding 6).

### 2. Output tokens are counted as response events, not tokenized text

Rust's client-side output-token count counts content-bearing response events;
upstream computes `len(tokenizer.encode("".join(output_texts)))` — one encode over
the whole generation, independent of event count
(`records/inference_result_parser.py:587-619`). Runtime-measured: a non-streaming
chat reply whose server reported 12 completion tokens yielded **OSL 1.0**.

Every output-token-derived metric is wrong, including output token throughput and
per-user throughput. Report: `04-response-tokens.md` (finding 1).

### 3. `audio_transcription` publishes fabricated token metrics

Upstream turns the endpoint descriptor's `produces_tokens: false` /
`tokenizes_input: false` into disallowed-metric flags
(`base_metrics_processor.py:38-48`). Rust consumes the same descriptor facts only
to skip loading a tokenizer, with no metric filter.

Runtime-measured on a 4-request transcription run: `input_sequence_length 550.0`,
`input_token_throughput 84,596 tok/s`, `total_token_throughput 84,596`,
`output_sequence_length 0.0`, and `osl_mismatch_diff_pct -100%` on 4 of 4
records. Applies to every `produces_tokens: false` descriptor.

Report: `04-response-tokens.md` (finding 2).

### 4. Consecutive runs silently overwrite each other's results

Upstream writes each run into an auto-generated per-run subdirectory,
`artifacts/<model>-<service_kind>-<endpoint_type>-<stimulus>/`, guarded by
`if "dir" not in cfg.artifacts.model_fields_set` (`resolvers.py:97-107`,
`_compute_artifact_name` at `:139-175`). Rust writes flat into `artifacts/`.

Two runs in the same working directory silently destroy the first run's results.
Report: `08-artifacts-exports.md` (finding 1).

### 5. A cancelled run's report is indistinguishable from a complete one

Upstream appends "The profile run was cancelled early. Results shown may be
incomplete or inaccurate." and tags the duration "(cancelled early)"
(`controller/system_controller.py:1185`, `:1230`). Rust's `render_console_txt`
never reads `was_cancelled` and exits 0.

A partial benchmark looks like a finished one, in the console and in the exit
code. Report: `09-console-ux.md` (finding 1).

### 6. A second Ctrl-C does nothing

Upstream prints "Press Ctrl+C again to force quit immediately" on the first
Ctrl-C (`system_controller.py:872`) and honors it (`:844-853`). In Rust, tokio's
handler has replaced SIGINT's default disposition and the listener task has
already completed, so the advertised escape hatch is gone — a stuck drain
requires SIGKILL from another terminal.

Report: `09-console-ux.md` (finding 2).

### 7. Most GPU energy-efficiency metrics vanish

Corroborated independently by two agents from different unchanged files.

Upstream emits 12 vendor-scoped tags per vendor (`nvidia_*` / `amd_*`). Rust
emits 4 unprefixed and drops 8 entirely: `average_gpu_power`,
`energy_per_output_token`, `energy_per_total_token`, `energy_per_request`,
`energy_delay_product`, `performance_per_watt`, `output_tps_per_watt`,
`goodput_per_watt`. Because the 4 survivors also lose their vendor prefix, **all
12 upstream tags read as missing** to a downstream consumer, with no error.

Reports: `10-telemetry.md` (finding 1, P0), `07-metrics.md` (finding 1, P1).

### 8. `--accuracy-benchmark` is accepted and silently ignored

`--accuracy-benchmark` is parsed and lowered into `cfg.accuracy`, then never
projected: `workload_kind()` returns only `scheduled`/`graph`
(`config/model/workload_kind.rs:102-114`), nothing emits the `static_accuracy` id
that is the sole producer of `NativeDatasetPlan::StaticAccuracy`, and
`protocol_v2.rs` never mentions accuracy.

Worse, Rust's evaluator path launches `python -u -m aiperf.accuracy.worker`
(`accuracy_core/worker.rs:105-112`) and **that module does not exist upstream**;
`rust/runtime/src/accuracy_core/` contains no native grader either. The run exits
zero as a plain synthetic perf run, with no grading and no accuracy artifacts.

Report: `11-accuracy-sweep.md` (finding 1).

### 9. OpenAI completions sends `prompt` as an array

Upstream deliberately sends a bare string for the single-prompt case —
`"prompt": prompts[0] if len(prompts) == 1 else prompts`, with the comment "some
gateways reject the list[str] wrapping" (`endpoints/openai_completions.py:43-49`).
Rust has no single-prompt branch and always sends a one-element array.

Report: `03-endpoint-payloads.md` (finding 1).

## P1 — material behavior change

Grouped by theme. See each domain report for evidence.

### Load generation

- Warmup inherits only concurrency in Rust; upstream also inherits
  `--request-rate`, `--arrival-pattern`, `--arrival-smoothness`,
  `--prefill-concurrency`, and all three ramps. `--request-rate 10
  --warmup-request-count 50` warms up open-loop at 10 rps upstream and
  closed-loop at concurrency 1 in Rust.
- Default `dispatch: global` replaces the Poisson renewal process with a jittered
  fixed-rate grid, so arrival burstiness — and every queueing-sensitive tail
  metric — varies with the host's core count.
- That same path bursts the full backlog after a stall, bypassing the catch-up
  bound that **both** engines otherwise honor identically
  (`AIPERF_TIMING_MAX_CATCHUP_SECONDS`, default 0.01, range `0..=10`).
- Any of nine stray `--warmup-*` flags builds a warmup phase whose only stop
  condition is lifecycle — an unbounded warmup. Upstream requires a
  count/session/duration trigger and errors otherwise.
- `--user-centric-rate` without `--num-users` is silently dropped (becomes
  concurrency-1, 10 requests); upstream errors in both directions.
- Rust's user-centric phase nulls cancellation, all three ramps, and prefill, so
  `--request-cancellation-rate 10` cancels ~10% upstream and 0% in Rust.
- `--num-users` below the worker count inflates to the worker count, distorting
  per-user think time.
- `--arrival-smoothness` / `--vllm-burstiness` without `--arrival-pattern`
  auto-promotes to a gamma process upstream; Rust falls to Poisson and drops the
  value.
- `--trajectory-start-min/max-ratio` default 0.25/0.75 upstream, 0.0/0.0 in Rust.

### Measurement

- Request latency ends at transport stream close in Rust, last content chunk
  upstream. Because `benchmark_duration` derives from the same boundary, the bias
  propagates into every rate denominator, ITL, and decode duration.
  (`07-metrics.md` finding 3, `04-response-tokens.md` finding 6.)
- Non-streaming runs publish `time_to_first_token` and `decode_duration` in Rust;
  upstream filters both via `STREAMING_ONLY`. Measured TTFT 1.482 ms against a
  request latency of 1.499 ms — a meaningless number presented as a real one.
- A chunk carrying both `reasoning_content` and `content` is attributed entirely
  to reasoning in Rust, moving tokens out of `output_token_count` and suppressing
  time-to-first-output-token.
- ISL falls back to the authored dataset count in Rust when a body has nothing
  tokenizable; upstream reports `input_sequence_length` absent.
- Streaming usage is merged per field in Rust vs last-non-empty-chunk upstream,
  and Rust derives `usage_total_tokens` when the server omits it.
- The fixed-schedule replay send-lag family (`replay_sched_lag_p50/p90/p99`,
  `replay_sched_degraded`) has no Rust counterpart, so a replay where AIPerf
  itself fell behind schedule looks identical to a clean one.
- A single failed GPU phase-boundary scrape discards the entire GPU telemetry
  section in Rust; upstream still reports every per-GPU gauge.
- Rust reports `endpoints_configured` and `endpoints_successful` as the same
  list, so an unreachable DCGM URL is indistinguishable from a healthy one.
- A missing server-metrics phase-start snapshot makes Rust report the counter's
  entire since-boot value as the phase delta; upstream falls back to the first
  in-window sample.
- Rust drops upstream's power-integration energy fallback
  (`power_w × duration_s`), so a collector exposing power but no cumulative
  energy counter loses `total_gpu_energy` and `output_tokens_per_joule` while
  `total_gpu_power` still prints.

### Input data

- Upstream guarantees a `sonnet` prompt re-encodes to exactly `--isl` via
  `_decode_to_exact_len`; Rust has no repair loop **and** reports the composer's
  intended count rather than the achieved one, so the wire drift is invisible.
- `--prefix-prompt-length` without `--num-prefix-prompts` raises upstream and
  silently produces zero prefix sharing in Rust, invalidating a cache-hit
  measurement with no warning.

### Configuration

- 6 of 27 shipped upstream templates fail to load on the native binary
  (verified by running it): they author the upstream `dataset.isl`/`dataset.osl`
  shorthand, which `DatasetSection`'s `deny_unknown_fields` rejects.
- `summary: ["json"]` — upstream's own default — means JSON **and** CSV upstream
  (`metrics_csv_exporter.py` has no summary gate; CSV is unconditional) but
  JSON-only in Rust, so **25 of 27** templates silently stop emitting
  `profile_export_aiperf.csv`. (`02-config-schema.md` finding 2,
  `08-artifacts-exports.md` finding 3 — converged.)
- A rate phase authored without `rate`/`rate_series` errors upstream and silently
  degrades to a concurrency-1 closed-loop phase in Rust.
- `fixed_schedule` with only `endOffset` keeps `auto_offset: true` upstream and
  infers `false` in Rust, moving schedule zero from the first timestamp to 0.0 —
  an epoch-millisecond trace then appears to hang. (Found by three agents:
  `01`, `02` finding 4, `06` finding 9.)
- Omitted `runtime.workers` resolves to `max(1, min(int(cpu*0.75)-1, 32))` then
  concurrency-capped upstream, versus full `available_parallelism()` uncapped in
  Rust.
- `models.items[]` is the one remaining place Rust's loader silently drops
  unknown keys where upstream's `extra="forbid"` rejects them; per-item `weight`
  is never read.

### Artifacts

- `--profile-export-prefix foo` yields `foo_aiperf.json` in Rust vs `foo.json`
  upstream, and the prefix is ignored entirely for the console, server-metrics,
  and network-latency files.
- Summary JSON emits `run_info: {}` and drops top-level `start_time`/`end_time`
  (losing the redacted CLI command and resolved seed) while still declaring
  `schema_version: "1.4"`.
- Distribution `sum` is never emitted — JSON key absent, CSV `sum` cell empty on
  every row.
- Scalar/derived metrics gain `min`/`max` (and `sum` for counters) that upstream
  never emitted, so `request_count.max` now exists and means nothing.
- Summary CSV lost upstream's third GPU-telemetry section
  (`Endpoint,GPU_Index,GPU_Name,GPU_UUID,…`).
- Per-record JSONL flipped `error`, `conversation_id`, and `credit_issued_ns`
  from omitted to explicit `null`, and serializes integer token counts as floats
  (`105.0`).
- `mlflow_export.json` is never written, yet `aiperf plot --mlflow-upload` — still
  Python-delegated — resolves its target run from that file.
- MLflow receives zero params (`params: BTreeMap::new()` is plumbed but never
  filled) versus upstream's ten, including `aiperf.cli_command`.
- Accuracy mode writes aggregate `accuracy_results.csv` instead of upstream's
  per-record `accuracy_export.jsonl` — different name, format, and granularity.
- OTLP push loses all 14 `aiperf.timing.*` series plus the `aiperf.<tag>`
  histogram upstream creates for every metric lacking a spec mapping.

### Console and process

- Three pre-run advisories are silent in Rust: `--osl` without
  `ignore_eos`/`min_tokens`, accuracy without `temperature=0`, and "SSL
  certificate verification is DISABLED" — the last while Rust installs a no-op
  cert verifier.
- When every request fails, Rust prints nothing to the terminal (one
  `tracing::error!`) despite having written a `profile_export_console.txt`
  containing the error table and an actionable advisory.
- `--show-trace-timing` shows no trace timing: all 14 `http_req_*` tags are
  `group: none`, so the flag's only effect is extra JSONL columns, and it is not
  in the loudly-refused flag table either.
- Logs moved stdout → stderr, breaking wrappers that grep stdout; `-v`/`-vv` no
  longer raise dependency log levels (`AIPERF_LOG` is the undisclosed
  replacement).
- The entire end-of-run footer is dropped: CLI command, benchmark duration,
  exported-file list, and log-file path.
- GPU console output cut from two vendor-attributed 12-row tables to four
  unattributed rows; the per-GPU table and cross-vendor comparability disclaimer
  are gone.
- An advisory instructs the user to set
  `AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD`, which nothing in Rust reads — the
  threshold is a hardcoded `const f64 = 5.0` at
  `export/console_txt.rs:36`, and the string occurs only in the message it prints
  (`:292`) and two golden files.

### Control hooks

The `#1332` timeout fix itself ported correctly (see non-findings). These
divergences are in the surrounding request and multi-origin plumbing, which that
commit did not touch. The authentication finding — the only one not gated on an
unusual configuration — is fixed; the remaining three are open.

- **Control POSTs carry no authentication — FIXED.** Control POSTs now send the
  endpoint's full auth header set, matching `endpoint_auth.py:22-38`: authored
  `endpoint.headers` pass through, then either `Authorization: Bearer <api_key>`
  or, for Messages, a hard-assigned `x-api-key` plus a defaulted
  `anthropic-version`. The dialect rules are single-sourced in
  `endpoints/implementation.rs` and `endpoints/anthropic.rs` and reused by
  `auth_headers_for_endpoint`; `control_hooks.rs` resolves the set once per
  prepare and installs it on the validated profile as a `ResolvedRequestHeaders`
  whose `Debug` exposes names only, so values never reach logs, errors, or
  descriptors. Both the typed and JSON entry points are wired. See finding 1 in
  `12-control-hooks.md`.
- **Multi-origin profiler stop aborts at the first failing origin.**
  `execute_control_hook` returns from inside the per-handle loop
  (`control_hooks.rs:641`), where upstream attempts every origin and aggregates
  the failures (`common/control_hooks.py:224-256`). Both surface an error, so the
  divergence is not a silent success: Rust leaves the remaining servers still
  profiling.
- **No reverse-order rollback after a partial profiler start.** Upstream stops
  whatever already started, in reverse order, and re-raises unchanged — under
  `BaseException`, so cancellation also triggers cleanup
  (`common/control_hooks.py:189-221`). Rust's `start_server_profiler` has no
  rollback path.
- **No origin de-duplication.** `prepare_handles` builds one handle per
  `endpoint.urls` entry (`control_hooks.rs:344-362`) after normalizing each to
  its origin, where upstream collapses them through `unique_endpoint_origins`
  (`common/control_hooks.py:61-71`). Two URLs sharing a host send doubled reset,
  start, and stop POSTs.

### Requests and multi-run

- `User-Agent` is `aiperf-transport-http/0` rather than `aiperf/<version>`.
- `--extra-inputs stream:<bool>` is a real streaming-mode switch in Rust (moves
  `Accept` and the response reader); upstream only edits the body, leaving
  `Accept` and TTFT parsing mismatched.
- A user-supplied `Content-Type` survives to the wire on Rust JSON endpoints and
  is overwritten upstream.
- `pareto-sweep` defaults concurrency to `(1, 4, 16, 64, 256)` upstream and
  `vec![1]` in Rust: 10 runs versus 2 from the same command line, yielding a
  Pareto frontier from a 5× sparser observation set. Upstream's single-point
  refusal is also absent.
- `--convergence-metric` is omitted from Rust's unimplemented-flag warnings and
  validated as if it enabled convergence, but only toggles an extra artifact;
  `plan_cells` materializes all trials with no early exit, so upstream may stop
  at 2 of 10 trials where Rust always runs 10.
- `--audio-sample-rates` diverges 1000× at or below 96 — upstream has a threshold
  normalizer (`v/1000 if v > 96 else v`) that Rust lacks — which is exactly the
  range upstream's help text advertises (`16`, `44.1`, `48`, `96`).
- `--auto-plot` is a true no-op in Rust; upstream runs the plotter.
- Per-loader preferred-sampling overrides (`random_pool`→SHUFFLE,
  `dag_jsonl`→RANDOM) have no Rust counterpart; Rust hardcodes `sequential`,
  changing prompt order and KV-cache hit rate.
- `--isl-stddev`/`--osl-stddev` are silently dropped without the paired mean, so
  `--isl-stddev 128` gives constant prompts and a reported stddev of 0.
- `--video-duration` default drops 5.0 s → 1.0 s; setting any one `--rankings-*`
  flag silently re-specifies the others' token means.

## Cross-domain duplicates, reconciled

| Finding | Reported by | Resolution |
|---|---|---|
| Unbounded user-centric run | `01` (P0), `06` (P1) | One bug, `resolve.rs:1353-1359`. Rated **P0**. |
| GPU energy metrics | `10` (P0), `07` (P1) | Same truncation from two files. Rated **P0**. |
| `artifacts.summary` CSV loss | `02`, `08` | Converged: the trap is solely the changed meaning of `json`, not disjoint vocabularies. 25 of 27 templates. |
| Request-latency boundary | `07`, `04` | One boundary difference with two downstream consequences. |
| `fixed_schedule` auto-offset | `01`, `02`, `06` | One resolution difference. |
| Accuracy artifact format | `08`, `11` | Distinct: `08` covers the file, `11` covers grading never running. |

## Notable verified non-findings

Recorded so they are not re-investigated. Each was suspected and disproved
against baseline.

- **ITL divisor and chunk inclusion.** Upstream divides by
  `OSL − first_content_chunk_tokens`, identical to Rust, with matching guards,
  matching `OSL − 1` fallback, matching once-per-process warning, and the same
  `--per-chunk-usage` opt-in and three validations. Terminal usage-only chunks
  and `[DONE]` are excluded on both.
- **`stream_options.include_usage`.** Upstream forces it on for every streaming
  run, exactly like Rust, with an explicit comment that gating it on
  `--use-server-token-count` would silently drop vLLM per-request metrics.
- **`X-Session-Affinity`.** Upstream does read it, under the field name
  `X_SESSION_AFFINITY_FROM_CORRELATION_ID`, default `True`. Both engines send it.
- **Synthetic special-token ISL subtraction.** Parity on all three arms. The
  paired `sequence_distribution` path subtracts nothing on either side (Rust's
  `input_token_subtraction` is a dead field there, read only in the `None` arm of
  `match paired_lengths`); the `random_range_ratio` path reproduces upstream's
  VLLM-fold / SGLANG-shift style split faithfully.
- **High-resolution pacing.** Upstream's `high_res_timer.py` uses the same
  absolute-deadline `CLOCK_MONOTONIC` timerfd mechanism as Rust's
  `real_clock.rs:116-143`, and both honor the same catch-up env var and default.
- **Adaptive `error_rate` SLA.** Upstream already computes percentage points over
  a successes+errors denominator that excludes cancellations — byte-equivalent to
  Rust.
- **`audio_transcription` request side.** Same multipart field set and order,
  same `{b64_data, filename, content_type}` descriptor, same `audio.<fmt>`
  filename, equivalent MIME tables including the mp3/mpga/mpeg collapse.
- **Percentile and statistics machinery.** Percentile band, linear interpolation,
  population standard deviation, nearest-rank `adj_*` construction, sweep-line
  `effective_*`/`active_*` tags and duration-CDF math, cancellation
  classification, and empty/all-failed edge behavior all equivalent.
- **Raw per-GPU telemetry plane.** All 20 shared DCGM/AMD source fields match on
  output name, unit, scale factor, and gauge/counter classification. URL
  normalization, default endpoints, the 0.333 s interval, the 10 s reachability
  timeout, enabled-by-default flags, and the TRT-LLM `/prometheus/metrics`
  fallback all match.
- **Credential redaction.** Same nine headers, same `<redacted>` sentinel, plus
  matching field serializers. No credential leak in either direction.
- **Flag name coverage.** Of 301 upstream long flag names, exactly two have no
  Rust counterpart (`--transport`, `--transport-type`), and both fail loudly.
- **Control-hook timeouts and retries (the `#1332` fix).** All nine crux values
  match. Critically, Rust's `DEFAULT_CONTROL_HOOK_TIMEOUT_NS = 30_000_000_000`
  (`engine/control_hooks.rs:30`) is a genuine constant fallback, not an
  inheritance of the 6-hour `endpoint.timeout`, and enforcement was traced rather
  than assumed: `attempt_control_request` sets
  `absolute_deadline_ns = now + 30s` (`:601`) and
  `NativeControlPlaneHttp::execute` takes
  `absolute_deadline_ns.min(cancellation.deadline_ns())`, racing the request
  against its own `Clock` sleep (`control_plane_http.rs:895-909`). The 6-hour
  `total_timeout_ns` is cloned into the control-plane `ClientConfig`
  (`control_hooks.rs:409-410`) but can only ever be the looser bound, so it
  cannot extend a control POST. Also matching exactly: the 60 s total retry
  budget, the `{409, 423, 429, 503}` retryable set with all other non-2xx never
  retried, total-budget retry semantics with per-attempt timeout reset, the
  1 s→8 s doubling backoff, `extra="forbid"` unknown-key rejection, the default
  start/stop paths, the failure policy, and hook phase placement and ordering.

## Separate concern: this branch's Python has regressed from `origin/main`

Not parity findings, but discovered during the audit and worth their own fix.
The branch working tree's Python differs from `origin/main` in ways that are
outright broken or that revert upstream behavior:

1. **Synthetic text generation raises `NameError`.**
   `dataset/composer/synthetic.py:216-217` passes `with_prefix=` and
   `exact_length=` to a `generate()` whose signature no longer accepts them, and
   both names are undefined in scope. The call site is live
   (`synthetic.py:144`), so Python cannot generate a synthetic text prompt at
   all on this branch. Upstream declares both as keyword-only params
   (`generator/prompt.py:428-436`) and defines both locals (`:225`, `:246`). A
   botched merge reverted the signature while keeping the call site.
2. **`openai_chat.py` lost 81 lines**, reverting the `include_usage`-always
   behavior that upstream deliberately added.
3. **Four upstream files were deleted**, each a user-facing surface:
   `endpoints/openai_audio_transcription.py` (123 lines),
   `timing/high_res_timer.py` (234), `config/dataset/system_prompt.py` (111),
   `orchestrator/export_helpers.py` (114).
4. **`common/models/sequence_distribution.py` lost 591 lines** and
   `common/tokenizer.py` lost 103.
5. **`timing/strategies/adaptive_scale_sla.py`** reverted `error_rate` to
   `errors / stats.total`, losing upstream's percentage-point form.

These regressions are why the first audit pass produced several false findings:
comparing Rust against a reverted Python makes Rust look like the deviant.

## Recommended next steps

1. Fix the P0s in the order listed; items 1, 4, 5, and 6 are small and
   high-impact.
2. Restore the branch's Python from `origin/main` for the five items above, or
   confirm each divergence was intentional.
3. Add cross-engine acceptance gates for the surfaces that drifted silently:
   artifact directory layout, summary JSON/CSV schema, the metric tag inventory,
   and the endpoint request body per endpoint type. Every P0 here would have been
   caught by a golden-file comparison.
4. Resolve the `## Unverified / needs runtime check` sections in each domain
   report; several need only a deterministic mock-server run.
