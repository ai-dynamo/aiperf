# Track A — verified regression-detection harness suite

**Date:** 2026-07-12
**Status:** Plan (grounded in the read-only tree at `/home/anthony/nvidia/projects/aiperf/ajc/rust`, branch `ajc/rust`). All prototyping goes under `~/tmp`; the repo is not modified by this plan.
**Guards the four verified refactors** (specs under `specs/2026-07-12-*.md`):

- **A1** scheduled worker-local accumulation (`scheduled-worker-local-accumulation.md`) — per-worker `NativeMetricsObserver`+`MetricsAccumulator`, merged once. Load-bearing change: `RunCapture::finish` positional record-zip → uuid-keyed join (`execute.rs:3200-3234`).
- **A2** single-observer compat projection (`single-observer-compat-projection.md`) — derive `TraceSimulationReport` from the native accumulator's retained records instead of a live `CollectorObserver`.
- **A3** `Connector` seam Tcp/Uds/Duplex (`http-connector-seam-uds-duplex.md`) — scheme-selected connection layer; UDS metric-identical to TCP; Duplex = in-process hermetic; pool `origin_key` keying.
- **A4** lean per-request hot path (`lean-per-request-hotpath.md`) — batched classified-token hook, interned dimensions, retention/ICL gate.

> Every file:line below was read from the working tree on 2026-07-12. "Code is truth" — re-verify before landing; some line numbers drift.

---

## 0. TL;DR — the one thing that matters

Today's infra has **strong cross-*engine* parity (offline AIPerf==Dynamo byte gate) and strong *unit-level* merge tests, but ZERO full-report value-parity regression golden for the online scheduled product path.** The handful of online tests pin ~3 scalar fields each (`num_requests`, a TTFT range). A1/A2/A4 can silently shift `p95 ttft`, `itl` distribution, `max_itl_ms`, `output_token_throughput_per_user`, goodput, and canceled-request counts with **no failing test today**. The first deliverable is therefore a **golden full-report snapshot captured from current HEAD, on a deterministic SimClock substrate**, before A1/A2/A4 land.

---

## 1. Inventory of existing infrastructure (what already catches regressions, and the gaps)

### 1.1 The online↔offline byte-parity gate (the "74-field common summary")

- **Mechanism:** `verify_parity` / `verify_parity_online` in `rust/runtime/src/engine/offline_execution.rs:2266-2317`. It calls `canonical_shared_metric_bytes()` on both the AIPerf compat report and the Dynamo report and asserts `aiperf_bytes == dynamo_bytes` (`offline_execution.rs:2271-2286`).
- **The trait + serialization:** `CanonicalSharedMetrics::canonical_shared_metric_bytes` — `rust/runtime/src/dynosim.rs:655-666`; the whole gate lives in `finish_shared_metrics_enforcing` (`dynosim.rs:907-969`), which sets `independently_accumulated` (`dynosim.rs:924-932`) and, when the AIPerf collector is unfed, substitutes `compatibility_report_from_dynamo` (`dynosim.rs:931, 986-1050`), then byte-compares (`dynosim.rs:957-968`) and counts `shared_fields` (`dynosim.rs:969`).
- **Field count is 74 base + 3 goodput = 77.** Product proof: `rust/cli/tests/offline_scheduled_stdio.rs:196,214` asserts `provenance["parity_shared_fields"] == "77"` and `native["run"]["dynamo"]["parity"]["shared_fields"] == 77`. Runner-level acceptance test `offline_execution.rs:2543` asserts `"74"` for the no-goodput case.
- **CRITICAL SCOPE LIMIT:** this gate proves **AIPerf-collector == Dynamo-engine** on the *offline SimClock* path only. It is **not** a golden-vs-refactored self-comparison, and it does **not** run on the online product path (the runner discards `report.performance` online — `execute.rs:3276-3282`, confirmed in the A2 spec §6). So it guards A2's *offline* projection (which projects from `DynamoSimulationReport`, unaffected by dropping the online collector) but does **nothing** for A2's online projection risk.

### 1.2 Accumulator / store merge tests (the A1 unit substrate — already strong)

- **`per_worker_merge_matches_single_accumulator_ingest_order`** — `rust/runtime/src/metrics_core/accumulator.rs:1781-1835`. Builds two records, ingests both into one `direct` accumulator, ingests one each into `left`/`right`, `left.merge(&right)`, and asserts `left.summarize() == direct.summarize()` **plus** per-endpoint `inference_series()` ordering and per-worker masks. **This is the closest existing A1 guard** — but it is 2 records, 2 workers, one merge order.
- **`worker_stores_merge_with_numeric_categorical_and_ragged_alignment`** — `rust/runtime/src/metrics_core/store.rs:1437-1471`. Pins `append_store` numeric/categorical re-interning + ragged ICL `append_shifted` alignment.
- **`ragged_series_preserves_out_of_order_rows_masks_and_shifted_merge`** — `store.rs:1267`.
- **Merge primitive:** `MetricsAccumulator::merge` (`accumulator.rs:485-514`), dense precondition asserted in `ColumnStore::append_store` (`store.rs:569-656`, dense check ~575-578).
- **Reference production merge:** `rust/runtime/src/graph/transport_bench.rs:385-395` (graph bench merges per-worker `native` accumulators at the join).
- **GAPS:** no test asserts (a) **merge-order independence** (reverse the fold), (b) **workers=1 vs workers=N produce the identical summary** for the same request set, (c) the runner's new **uuid-keyed `finish` join** (the A1 load-bearing change) reorders rows without changing report bytes. `accumulator.rs:1781` is order-sensitive by construction (it fixes ingest order); it does not prove order-*independence*.

### 1.3 Rust integration / product-path tests

- **`rust/runtime/tests/scheduled_real_mock.rs`** — real wall-clock against a spawned `aiperf-mock-server` binary (`--random-seed`, `spawn()` at :25, `mock_binary()` at :81). Asserts `num_requests`/`completed_requests` exactly (`:175-177,224-225`), scheduled offsets exactly (`:183-191`), and TTFT via **tolerance** ranges (`assert_real_ttft_and_lateness` :116-138: "12ms mock TTFT should remain recognizable"). This is the template for the real-mock tier.
- **`rust/runtime/tests/scheduled_sim.rs`, `request_rate_sim.rs`, `phase_runtime_sim.rs`** — SimClock deterministic proofs with fixed-latency test dispatchers. Assert **exact** ns offsets and `mean_ttft_ms == 10.0`, `total_output_tokens == 12` (`scheduled_sim.rs:282-346`). **This is the deterministic substrate the value-parity golden must build on.**
- **Runner stdio E2E** (`rust/cli/tests/*_stdio.rs`) — JSONL-over-stdio subprocess driver (`offline_scheduled_stdio.rs:29-62` writes request, splits stdout on `\n`, reads terminal frame). Assert `terminal["success"]`, `provenance`, `native["schema_version"] == "2.0"`. `online_v2_stdio.rs` uses an in-process axum fixture server; `thread_per_core_product.rs:66-180` proves **round-robin placement across 3 workers** and counts one server request per turn — the closest thing to a multi-worker online product test, but it asserts placement, not metric values.
- **Note:** `dag_stdio_e2e.rs` is referenced in `git status` but is **absent** from the tree today (renamed → `recorded_graph_stdio_e2e.rs`). Do not target the old name.
- **Graph parity:** `rust/runtime/tests/graph_recorded_adapter_parity.rs` (WEKA↔Dynamo lower to identical topology + materialized bytes, `:166-184`).

### 1.4 Throughput harnesses

- **`rust/runtime/examples/rps_bench.rs`** — thread-per-core RPS driver (`THREADS`/`CONNS`/`LANES`/`WARMUP_S`/`WINDOW_S`, h2c or `HTTP1=1`), prints achieved RPS against a running `aiperf-mock-server --fast`. **Not** wired to `origin_key`/UDS today (uses `establish` directly).
- **`fast_sse` (ultra-cheap fixed-SSE server, `UDS_PATH`-aware) — REMOVED.** It lived at `rust/aiperf-core/examples/fast_sse.rs`; the `aiperf-core` crate was dissolved into its proper seams and the example deleted (commit `1011863f4`). A `UDS_PATH`-listening pure-transport driver must be re-created (e.g. as an example under `rust/runtime/`) before it can serve as the TCP-vs-UDS win driver once A3 lands a `unix:`-scheme client path.
- **`rust/runtime/src/graph/transport_bench.rs`** — the per-worker-merge graph bench (also the A1 reference impl).
- **GAP:** no product-path (`aiperf profile`) throughput baseline with `records:false`, and no automated ratio/regression check — the benches print numbers, nothing asserts them.

### 1.5 Transport / connector tests (A3 substrate)

- **`rust/runtime/tests/transport_http/pool.rs`** — pool bounds, sticky-session serialization (`:224`), sim-clock timeout accounting (`:361`). **No `origin_key` collision test** (the A3 highest-attention risk).
- **`rust/runtime/tests/transport_http/connect.rs:12`** — establishes an h1 connection to the mock and records socket info (the shape a Duplex/UDS connect test mirrors).
- **`rust/runtime/tests/transport_http/no_direct_time.rs`** — guards against raw `Instant::now()` (Clock discipline); A3 impls must keep passing it.
- **`rust/runtime/tests/transport_http/tls.rs`, `h2c.rs`, `reuse.rs`, `cancel.rs`** — the dispatch-above-`Sender` behaviors A3 must leave byte-identical.

### 1.6 Python product-path tests

- `tests/ci/test_docs_end_to_end/test_runner.py` drives documented `aiperf` commands; `tests/component_integration/cli/` covers CLI surface. These exercise `aiperf profile` end-to-end but do not pin Rust metric values. Useful as a **smoke** tier (the run completes and emits a schema-2.0 native report), not a value gate.

### 1.7 Mock determinism knobs (the substrate for reproducible real-mock runs)

`rust/mock-server/src/config.rs`: `random_seed` (:339), closed-form **analytic** latency mode (:71 — deterministic given seed), `prefix_cache_*` (:217-273, disable via `disable_prefix_cache`), `dcgm_seed` (:357). With `--random-seed` + analytic latency + fixed token count, the mock's **token counts and body bytes are deterministic**; wall-clock arrival *timing* still jitters (RealClock), so latency distributions are tolerance-only against the real mock.

### 1.8 The single biggest detection gap

**There is no full-report value-parity regression golden for the online scheduled product path.** Everything that exists is either (a) a cross-engine offline gate, (b) unit merge tests on 2 records, or (c) integration tests pinning ~3 scalars. A1's uuid-join reorder, A2's percentile-algorithm/time-base change, and A4's batched-token/ICL-gate change all land on exactly the surface no golden covers. **Close this first (Harness A).**

---

## 2. Determinism analysis (the gating constraint for byte-parity)

| Substrate | Clock | Reproducible? | Use for |
|---|---|---|---|
| SimClock + fixed-latency test dispatcher (`scheduled_sim.rs` shape) | virtual, integer-ns, `(at_ns,seq_no)` tie-break | **Runner product path: byte-exact vs HEAD for ALL fields** (records-first re-ingest in dispatch order, Finding 6). Library merge-summary path: byte-exact integer aggregates + equal-latency floats, tolerance for varied-latency floats (§4.5). | The value-parity **golden** (runner: everything byte-identical; library: latency distributions per the split) |
| Offline dynosim scheduled (`offline_scheduled_stdio.rs`) | SimClock | **Byte-exact** (already gated 77 fields) | Cross-engine A2-offline guard (unchanged by A2) |
| Real `aiperf-mock-server` (`scheduled_real_mock.rs`) | RealClock | counts/tokens exact; **latency jitters** | Real-transport tier: byte-exact counts, tolerance-banded latency |
| A3 `DuplexConnector` in-process server | RealClock | counts/tokens exact; latency jitters | Hermetic connector-parity + fast CI transport tests |
| `rps_bench`/`transport_bench` | RealClock | throughput ±noise | Ratio-based throughput checks only |

**Determinism blocker + fix.** The online product path is RealClock, so a *full* online report is **not** byte-goldenable — latency distributions vary. **Fix (two-tier golden):**
1. **Deterministic tier (byte-exact):** capture the golden on **SimClock** with a fixed-latency dispatcher (the accumulation math that A1/A2/A4 touch runs *identically* under SimClock — the observer/accumulator/merge code is clock-agnostic). This makes **every** field of the native-v2 report and the projected `TraceSimulationReport` byte-comparable, including the A2 percentile/time-base fields. This is where A2's percentile-algorithm change is caught deterministically.
2. **Real-transport tier (mixed):** on the real mock (and A3 Duplex), assert **byte-exact** counts/tokens/throughput and **tolerance-banded** latency distributions, so the real hyper stack, SSE parse, and connection reuse are exercised without flaking on wall-clock jitter.

For the A1 float-summation-order caveat (spec §4.5): **this caveat is scoped to the
LIBRARY merge-summary path, NOT the runner product path (Finding 6).** The runner
is records-first — worker records are concatenated, uuid-joined into **dispatch
order**, and re-ingested into one fresh accumulator in that order
(`execute.rs:1557-1560`) — so the IEEE-754 fold is over the same dispatch order HEAD
uses, and **every runner field, including float distributions, is byte-identical to
HEAD at any worker count.** A runner float delta vs a HEAD golden is a real
regression, not accepted drift. The (a)/(b)/(c) split below therefore applies to the
**library** `ScheduledRuntime`/`phase_runtime` path (which merges accumulators and
`summarize()`s once, reordering the fold): (a) integer aggregates → byte-exact vs
HEAD; (b) library-path float distributions on a **varied**-latency profile →
ULP/relative tolerance vs HEAD, or re-baseline from A1 output; (c) library-path float
distributions on an **equal**-latency profile → byte-exact (order-invariant fold).
For the **runner** value-parity golden (Harness A SimClock tier), all fields are
byte-exact vs HEAD — no tolerance band. Merge-order-independence is asserted
separately on integer aggregates for the library merge path (Harness B).

---

## 3. Harness suite

Each harness names the risk it guards, where it lives, CI vs manual, the run command, and what a failure means.

### Harness A — METRIC VALUE-PARITY MATRIX (guards A1, A2, A4)

**Risk caught:** any change to accumulation, merge, projection, batching, or ICL retention that shifts a reported metric value. Specifically the A2 fields: `ttft/ttst/e2e` percentiles (native linear-interp vs collector nearest-rank), `itl` distribution + `max_itl_ms`, `output_token_throughput_per_user`, `num_requests` incl. canceled, good-only goodput; and A4 ICL ordering.

**Design:**
- **Location:** new deterministic test file `rust/runtime/tests/report_value_golden.rs` (SimClock tier) + assertions extended into `rust/runtime/tests/scheduled_real_mock.rs` (real tier). Golden fixtures as committed JSON under `rust/runtime/tests/golden/` (or, since repo is read-only for *this* plan, staged under `~/tmp/track-a/golden/` during capture, then committed by the implementer at land time).
- **Workload matrix** (each a row → one golden file): `{concurrency, poisson, constant/request-rate, fixed-schedule, user-centric, graph}` × `{streaming, non-streaming}` × `{records:true, records:false}`, **plus an error/cancellation-injection variant** of at least the concurrency and request-rate rows (force `WorkerCommand` send failure, worker-drop, and `PlacementCancellation` — turn_execution.rs:476,501-503,465-466,681-692 — and library Canceled/Failed synthesis scheduled.rs:951-999). Small deterministic dataset (fixed synthetic prompts), fixed RNG seed (`RngRoot::derive`), **fixed worker count = 2** (to exercise multi-worker record concatenation while staying byte-stable), fixed-latency dispatcher (ttft/itl constants like `scheduled_sim.rs:118`). **On the runner product path a single varied-latency profile suffices and is byte-exact vs HEAD (Finding 6)** — the records-first re-ingest folds in dispatch order, so no equal-vs-varied split is needed to dodge a reorder. **The two-latency-profile split ((i) equal, (ii) varied) is only required for the LIBRARY merge-summary path (Harness E / A2)** where the accumulator merge reorders the fold; on that path (i) is byte-exact and (ii) is tolerance-compared. **`records:false` rows additionally guard: (a) the uuid-join keys on the drain uuid, not the empty `correlation_id` (Finding 3 — assert the run does not abort/collapse in aggregate-only mode), and (b) TTST is still present/correct despite the `token_arrivals_ns` retention gate (Finding 5 — `second_token_ns` retained).**
- **Why the error/cancellation rows are mandatory (A1 spec §3.3/Risk 4):** A1 moves arrival+terminal worker-local, so a pre-worker failure produces an identity with no worker record. The coordinator-side fallback accumulator must synthesize the HEAD-identical errored/canceled `RecordIngest`. These rows pin `errored`, `canceled`, `ErrorRequestCount`, and admit/start/end timing against HEAD and prove the run does **not** abort fail-closed on a missing uuid-join lookup. Without them the entire pre-worker-failure surface is untested.
- **What is byte-identical vs what is tolerance-compared (deterministic SimClock tier) — A1 spec §4.5 + Finding 6:** **for the RUNNER product path, ALL fields — integer aggregates AND float latency distributions — must be byte-identical to the HEAD golden.** The runner is records-first: worker records concatenate, uuid-join into **dispatch order**, and re-ingest into one fresh accumulator in that order (`execute.rs:1557-1560`), so the IEEE-754 fold matches HEAD's single-observer dispatch-order fold and no reorder occurs. There is **no ULP tolerance and no A1-recapture for runner floats**; a runner float delta is a real regression. The two-latency-profile split ((i) equal-latency, (ii) varied-latency) and the ULP/relative-tolerance apparatus are relevant only to the **library merge-summary path** (Harness E / A2), where the accumulator merge reorders the fold: there varied-latency floats are (a) captured from A1 output or (b) ULP/tolerance-compared, and equal-latency floats stay byte-exact. **A2's percentile-algorithm and time-base choices are still fully gated on the library path here:** a records-exact A2 projection is byte-unchanged, while a summary-cheap one shows exactly which fields moved — forcing an intentional, justified golden update. Do not import the library-path reorder tolerance into the runner golden.
- **What is tolerance-banded (real-mock tier only):** latency distribution fields (`ttft/ttst/tpot/itl/e2e/*_per_user` mean/percentiles) within a band (e.g. ±15% of golden or an absolute ns floor, matching `scheduled_real_mock.rs:124-126` style). Counts/tokens/throughput-from-counts stay **exact**.
- **Compare mechanism:** reuse the existing `canonical_shared_metric_bytes` serialization surface (`dynosim.rs:655-666`) for the compat `TraceSimulationReport` half (this is the exact byte surface the offline gate already trusts), and add a native-v2 report canonical-bytes compare (serialize `Reporter` output to compact JSON, diff against golden). Build a tiny `assert_report_matches_golden(report, golden_path, tolerance_fields)` helper in the test crate; do **not** extend the AIPerf==Dynamo gate (that is a different, cross-engine comparison — see §1.1 scope limit).

**CI/manual:** SimClock tier = **CI** (fast, deterministic, no external server). Real-mock tier = **CI** (spawns `aiperf-mock-server` like `scheduled_real_mock.rs` already does) but latency assertions are tolerance-banded to stay green on a noisy box.

**Run command:**
```bash
cargo test -p aiperf-runtime --test report_value_golden          # SimClock byte-exact tier
cargo test -p aiperf-runtime --test scheduled_real_mock          # real-mock counts-exact + latency-band tier
```
**Failure meaning:** a metric value moved. SimClock-tier failure = a *deterministic* regression (or an intentional A2 semantic change → update golden with a documented reason). Real-tier count/token failure = a real bug; real-tier latency-band failure at >band = investigate before assuming noise.

**MUST be captured from current HEAD before A1/A2/A4 land.** This is the ordering pin (see §4). Caveat (A1 spec §4.5 + Finding 6): **for the runner product path the HEAD capture is the EXACT byte baseline for every field, including float distributions** — records-first re-ingest does not reorder the fold. The tolerance-baseline treatment applies only to the **library merge-summary path** (Harness E / A2): there varied-latency float distribution fields are re-captured from A1 output or ULP/relative-compared (the merge reorders the fold), while integer aggregates and equal-latency floats stay byte-exact.

---

### Harness B — MERGE DETERMINISM (guards A1)

**Risk caught:** worker-local aggregates that depend on worker count or merge order; the uuid-keyed `finish` join reordering rows in a way that changes report bytes.

**Design (extends the existing strong unit substrate):**
- **Location:** extend `rust/runtime/src/metrics_core/accumulator.rs` tests (alongside `per_worker_merge_matches_single_accumulator_ingest_order:1781`) and `store.rs` tests (alongside `:1437`). Add a runner-level uuid-join test in `rust/cli/tests/`.
- **Tests to add:**
  1. **workers=1 vs workers=N equality:** ingest the same fixed record set as (a) one accumulator, (b) N split accumulators merged. Assert `summarize()` equal for N ∈ {1,2,3,5}. Grounds directly on `MetricsAccumulator::merge` (`accumulator.rs:485-514`) + `append_store` (`store.rs:569-656`).
  2. **Merge-order independence (integer aggregates):** for ≥3 partial accumulators, assert `RequestCount`, `TotalOutputSequenceLength`, `TotalInputSequenceLength`, `GoodRequestCount`, `ErrorRequestCount` are identical across all merge permutations (spec §6 scratch already proved this in `~/tmp/a1-spec`; promote it to an in-tree test). Float percentiles are *not* asserted order-independent (documented sub-ULP caveat, spec §4.5) — pin worker order for those.
  3. **Dense-precondition guard:** a worker accumulator built with a global sparse `request_index` is rejected by `merge` (mirrors `~/tmp/a1-spec/src/bin/sparse.rs`); pins the "workers must carry dense local rows" design constraint (spec §4.2).
  4. **uuid-keyed finish join + global `request_index`:** a runner test feeding merged per-worker records whose concatenated order ≠ dispatch order (and each worker's records carrying **local** `request_index = Some(0..n_w)`), asserting that `RunCapture::finish` (the `execute.rs:3200-3234` replacement) (a) stamps each record's `request_index` to its **globally-unique, dense, monotonic dispatch ordinal** from the identity, (b) re-ingests **collision-free** (no `insert_record_at` panic — the Finding-1 abort), (c) yields rows that are **dense `0..N-1`** and in **dispatch order** (byte-identical to a single-worker run over the same identities — proves `None`/push was correctly rejected), and (d) never aborts on the positional-zip assertion. Ragged ICL alignment across workers is already pinned by `store.rs:1437`; add the cross-worker ICL-order case explicitly (guards A4 too).

**CI/manual:** **CI**, all unit-speed.
**Run command:** `cargo test -p aiperf-runtime && cargo test -p aiperf-cli --test <new_uuid_join_test>`
**Failure meaning:** the merge is not associative/commutative over aggregates, or the uuid-join reordered rows into the report — the A1 load-bearing refactor is broken; the run would fail closed in production.

---

### Harness C — THROUGHPUT REGRESSION + WIN (guards A1 lift, A3 UDS win; catches perf regressions)

**Risk caught:** A1 failing to remove the single-core accumulation ceiling; **A1 delivering little/no gain because the real ceiling is the per-request coordinator funnel, not per-token replay** (A1 spec §1.1.1); A3 UDS not actually faster; any perf regression from A2/A4.

**Token-bound vs request-bound — the harness must distinguish them (A1 spec §1.1.1).** A single long-output ratio floor cannot tell whether A1 lifted a per-token ceiling or whether the coordinator's per-request funnel (Box::pin + turn.clone + mpsc send + oneshot recv + select! loop, all single-threaded) was the limit all along. Run the A1-lift check on **two contrasting workloads back-to-back**: (1) **long-output streaming** (many tokens/request — expected token-bound, A1 should lift), and (2) **short-output / non-streaming / usage-only** (≈1 token/request — expected request-bound, A1 may NOT lift). If (1) lifts but (2) stays flat, that is the *expected* signature and confirms A1's scope; if **both** stay flat, A1 did not remove the ceiling on either and the per-request funnel (A4 #1A/#4b persistent lanes) is the real lever — surface this explicitly rather than passing on the long-output row alone.

**Design (ratio-based, never absolute — stable on a shared box):**
- **Drivers:**
  1. `rust/runtime/examples/rps_bench.rs` against `aiperf-mock-server --fast` (existing) — TCP baseline.
  2. Same `rps_bench` against a re-created `UDS_PATH`-listening pure-transport server (the removed `fast_sse` role — see §1.4) once A3 wires a `unix:`-scheme client — the **TCP-vs-UDS win**. **Neither piece exists yet:** the `fast_sse` example was deleted with `aiperf-core` and must be re-created (e.g. under `rust/runtime/examples/`), and `rps_bench` calls `establish` directly and is not wired to a `unix:` client (§1.4). Rewiring `rps_bench` to select a `UdsConnector` (or point it at a `unix:` URL through `select_connector`), plus re-authoring the UDS pure-transport target, is **required PR3 work that the connector spec must budget** — without it there is no UDS-win measurement and the connector seam risks landing test-only.
  3. A **product-path** `aiperf profile` run with `records:false` (A4/A1 aggregate-only mode) — the real ceiling-lift proof, since the ceiling is coordinator-side accumulation, not transport.
  4. `transport_bench.rs` (graph, already per-worker-merged) as the A1 reference upper bound.
- **What to pin (ratios, not absolutes):**
  - **A1 lift:** on the product path, RPS at `workers=N` / RPS at `workers=1` must exceed a floor (e.g. ≥ 2.5× at N=4) **after** A1 — captured as "before" on current HEAD to show the flat pre-A1 scaling (the ceiling). Pin the *scaling ratio*, not raw RPS.
  - **A3 UDS win:** UDS RPS / TCP RPS ≥ 1.1 (same box, same driver, back-to-back) — a relative check immune to absolute box speed.
  - **Regression floor:** post-refactor RPS ≥ 0.9 × a rolling baseline recorded in `~/.claude/benchmark-findings/` (per box), re-run back-to-back in the same invocation to cancel drift.
- **Stability technique:** always run baseline and candidate **in the same process invocation, back-to-back, same warmup/window**, and compare the *ratio*. Never compare against a number from a different day/box.

**CI/manual:** **manual / nightly** (throughput is noisy in shared CI). Provide a `~/tmp/track-a/bench.sh` wrapper that runs baseline+candidate back-to-back and prints ratios with pass/fail vs the floors.
**Run command:**
```bash
# TCP baseline vs UDS win (A3)
cargo run --release -p aiperf-mock-server -- --fast &          # pure-transport target
THREADS=8 CONNS=8 LANES=16 cargo run --release -p aiperf-runtime --example rps_bench
# UDS_PATH=/tmp/fast.sock <UDS pure-transport server> &        # fast_sse example was removed (aiperf-core dissolved); re-create before use
# (A3) point rps_bench at unix:/tmp/fast.sock, compare RPS
# A1 lift: product path, records:false, workers sweep
aiperf profile --config bench-records-false.yaml   # workers=1 then workers=4, compare
```
**Failure meaning:** A1 lift ratio below floor = accumulation still pegs one core (the ceiling was not removed). UDS ratio < 1.1 = the connector seam added overhead or UDS not selected. Regression floor breach = A2/A4 added per-token/per-request cost.

---

### Harness D — CONNECTOR CORRECTNESS (guards A3)

**Risk caught:** Tcp/Uds/Duplex producing different metrics; scheme→connector misselection; `origin_key` pool-key collision (the A3 highest-attention item, spec §4/§6).

**Design:**
- **Location:** `rust/runtime/tests/transport_http/` — new `connector.rs` (mirrors `connect.rs:12` shape) + extend `pool.rs`.
- **Tests:**
  1. **Metric-identical Tcp==Uds==Duplex:** drive the *same* request against an in-process server reachable via all three connectors (loopback `TcpListener`, `UnixListener`, `tokio::io::duplex`), assert identical parsed response, identical trace fields **except** the synthesized `SocketInfo` dummies (spec §3.2/§3.3), and identical downstream observer facts (token count, usage). The Duplex round-trip is already proven in `~/tmp/connector-spec/tests/e2e.rs`; promote it in-tree.
  2. **Scheme selection:** `select_connector` returns `TcpConnector` for `http(s)://`, `UdsConnector` for `unix:`, `DuplexConnector` for `mem://` (spec §4 table).
  3. **`origin_key` isolation (the flagged risk):** two different `unix:` paths → two distinct pool entries; the same path twice → one shared entry; two different `mem://` names → distinct entries. Assert against `pool.rs` `origin_key` (`:28-35`) and the sticky-binding path (`:191-203`). This is the **new** test the spec explicitly demands (§6 "add a unit test").
  4. **Duplex as hermetic substrate:** wire `DuplexConnector` into the Harness A real-transport tier so the connector-parity and value-parity tests run with **zero sockets** (fast, deterministic port-free CI).
  5. **Regression guards stay green:** `no_direct_time.rs`, `reuse.rs`, `cancel.rs`, `tls.rs` (TCP-only) must all still pass — A3 is strictly below the `Sender` (spec §5).

**CI/manual:** **CI** (Duplex needs no network; UDS/TCP loopback are already in-suite).
**Run command:** `cargo test -p aiperf-runtime`
**Failure meaning:** connectors diverge in metrics (UDS/Duplex not a drop-in), or the pool key aliases distinct endpoints (silent cross-talk) / fails sticky binding.

---

### Harness E — CROSS-MODE / CROSS-TRANSPORT PARITY (guards A2, A3, and online↔offline)

**Risk caught:** A2's online single-observer report diverging from the old dual-observer report; TCP≠UDS at the *report* level; online↔offline drift.

**Design:**
- **A2 dual→single equivalence (the core A2 gate):** BEFORE A2 lands, capture the current dual-observer (`CollectorObserver` + `NativeMetricsObserver`) `TraceSimulationReport` from a deterministic SimClock library run (`rust/runtime` `run.rs`/`scheduled.rs` path, which *does* feed the tee — unlike the runner). AFTER A2, assert the **projected** report equals the captured golden. If the implementer takes the spec's **records-first** projection (§6 recommendation), this must be **byte-identical**; if summary-cheap, the diff enumerates the percentile/time-base fields that changed (forcing an explicit decision + golden update). Location: `rust/runtime/tests/compat_projection_parity.rs` (new), golden captured from HEAD.
- **TCP==UDS report parity:** run the Harness A concurrency+streaming row over TCP and over UDS (real mock / the re-created UDS pure-transport target that replaces the removed `fast_sse`, see §1.4), assert byte-exact counts/tokens/throughput and latency within band. Reuses Harness A + Harness D.
- **online↔offline:** **reuse the existing gate as-is** — `offline_scheduled_stdio.rs` (77-field parity) and `verify_parity` (`offline_execution.rs:2266`). A2's offline path projects from `DynamoSimulationReport` (spec §4), so this gate is unaffected and simply must **keep passing** after A2 (it proves A2 didn't break the offline projection). No extension needed.

**CI/manual:** A2 equivalence + offline gate = **CI**; TCP==UDS report parity = **CI** (Duplex/loopback) with latency banded.
**Run command:**
```bash
cargo test -p aiperf-runtime --test compat_projection_parity     # A2 dual==single (or enumerated deltas)
cargo test -p aiperf-cli --test offline_scheduled_stdio   # existing online↔offline 77-field gate (must stay green)
```
**Failure meaning:** A2 changed a library-visible report value that a consumer pins (`run.rs:1693`, `scheduled_real_mock.rs`); or the offline byte gate broke (A2 touched the wrong projection).

---

## 4. Golden-capture ordering (what to snapshot from current HEAD, and before which refactor)

**Capture from current HEAD, in this order, BEFORE landing any refactor:**

1. **Harness A SimClock golden matrix** (all workload×streaming×records rows, worker-count pinned = 2) — **before A1, A2, A4**. This is the master value baseline. Without it, A1/A2/A4 land against nothing.
2. **Harness E A2 dual-observer `TraceSimulationReport` golden** (library SimClock run that feeds the tee) — **before A2**. A2 deletes the live `CollectorObserver`; this is the only way to prove the projection reproduces it.
3. **Harness C "before" scaling curve** (product-path RPS at workers=1..N with `records:false`, on the target box) — **before A1**. Captures the flat pre-A1 ceiling so the post-A1 lift ratio is provable. Save to `~/.claude/benchmark-findings/`.
4. **Harness A real-mock counts + latency bands** (baseline latency means to set the ±band) — **before A1/A2/A4**.

**Can be authored after (they assert invariants, not goldens):**
- Harness B (merge determinism) — pure invariants, land anytime; ideally **before A1** so A1 is developed test-first.
- Harness D (connector correctness) — invariants + the `~/tmp/connector-spec` promotion; land **with A3**.
- Harness E TCP==UDS — needs A3's `unix:` client; land **with A3**.
- Harness C UDS win — needs A3; land **with A3**.

**Gate matrix (harness must exist before refactor lands):**

| Refactor | Blocking harnesses (must pre-exist / pre-capture) |
|---|---|
| A1 | Harness A golden (SimClock+real), Harness B (all 4), Harness C "before" curve |
| A2 | Harness A golden, Harness E A2 dual-observer golden, offline gate green |
| A3 | Harness D (all), Harness E TCP==UDS, Harness C UDS driver |
| A4 | Harness A golden (esp. `records:false` + ICL rows), Harness B cross-worker ICL-order case |

---

## 5. Deliverable summary (repeated for the caller)

- **Biggest gap:** no full-report value-parity golden for the online scheduled product path — the exact surface A1/A2/A4 touch. Close with Harness A first.
- **Determinism blocker:** online product path is RealClock → latency distributions jitter, so a full online report is not byte-goldenable. **Fix:** two-tier golden — SimClock fixed-latency tier for byte-exact *everything* (this is where A2's percentile/time-base fields are deterministically gated), real-mock/Duplex tier for byte-exact counts + tolerance-banded latency. Pin worker count for byte-exact float sums.
- **Reuse, don't rebuild:** the 77-field `verify_parity` gate (`offline_execution.rs:2266`, `dynosim.rs:655-666`) already guards A2-offline; `per_worker_merge_matches_single_accumulator_ingest_order` (`accumulator.rs:1781`) + `worker_stores_merge…` (`store.rs:1437`) are the A1 unit substrate to *extend*, not replace; `~/tmp/a1-spec`, `~/tmp/a2-spec`, `~/tmp/connector-spec`, `~/tmp/a4-spec/proof` scratch proofs promote directly into Harnesses B/E/D.
