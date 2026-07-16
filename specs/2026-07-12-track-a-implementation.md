# Track A implementation plan — unify the measurement engine, wire UDS, go fast

Goal: lift the product's ~100k req/s ceiling and de-duplicate the measurement path
while preserving reported metrics **to the precision each field allows**. Precise
meaning (do not read "no change" as blanket byte-identity):

- **Integer aggregates and counts** (request/token/error/goodput counts, input/output
  token totals) stay **byte-identical** to HEAD.
- **Float latency distributions** (`avg`/`sum`/`std`/percentiles of
  `ttft`/`ttst`/`tpot`/`itl`/`e2e`/`otpu`): **the reorder drift applies ONLY to the
  library merge-summary path, NOT the runner product path** (A1 spec §4.5 scope
  correction / Finding 6). The **runner** is records-first — worker records are
  concatenated, uuid-joined into **dispatch order**, and re-ingested into one fresh
  accumulator in that order (`execute.rs:1557-1560`), so its float fields are
  **byte-identical to HEAD** at any worker count; a runner-path float delta is a real
  regression. Only the library `ScheduledRuntime`/`phase_runtime` summarize-directly
  path merges accumulators and thus reorders the IEEE-754 fold; there the float fields
  use a ULP/relative tolerance or an A1-captured golden. Do not blanket-attribute float
  drift to A1 — it is a property of the merge-summary mechanism the runner does not use.
- A2 has its **own** real (non-ULP) semantic deltas (tpot/e2e/total_output_tokens) that
  the records-first projection must neutralize — see PR5.

Grounded in four verified specs (each with a
green scratch compile) under `specs/2026-07-12-*.md`:

- `…-scheduled-worker-local-accumulation.md` (A1)
- `…-http-connector-seam-uds-duplex.md` (A3)
- `…-lean-per-request-hotpath.md` (A4)
- `…-single-observer-compat-projection.md` (A2)

Regression gates live in the companion plan `2026-07-12-track-a-regression-harnesses.md`.

## The correction that sets the order

The product `aiperf-cli` path **does not** use the `ScheduledRuntime` `ObserverTee`:
`ConfiguredDispatcher::dispatch_turn` (`execute.rs:3268-3311`) discards it and feeds a
**single** `RunCapture` `NativeMetricsObserver` (`execute.rs:3067`). Therefore:

- **A1 is the product-throughput lever** and is **independent of A2** — the compat
  `CollectorObserver` is not on the runner hot path.
- **A2 is a library/offline cleanup** (removes real double-recording on `run.rs`,
  offline Dynamo, accuracy-single-turn), not needed for product throughput, and has
  the trickiest parity surface — so it goes **last**.

Sequence: **PR1 → PR2 (A1)**, **PR3 (A3)** and **PR4 (A4)** in parallel after PR2,
**PR5 (A2)** last.

---

## PR1 — `RunCapture` uuid-keyed join (behavior-preserving prerequisite)

The one load-bearing refactor, isolated so it can land and bake before A1.

- **Where:** `rust/runtime/src/engine/execute.rs:3200-3234` (`RunCapture::finish`; crate `aiperf-runtime`, `engine` module).
- **Change:** replace the positional record↔identity zip (`ingest.correlation_id ==
  identity.uuid`, `:3211-3218`) with a **uuid-keyed join** keyed on the record's
  **true `Uuid`** (built from the dispatch identities; each record resolved by its
  drain-provided uuid).
- **Key on the drain uuid, NEVER on `ingest.correlation_id` (Finding 3 — hard
  requirement, not a nicety):** in aggregate-only (`records:false`) mode —
  which PR4 item 3 (A4 #6) proposes extending to the runner, and which the Harness A /
  Harness C matrix exercises — `register_metadata` sets
  `correlation_id = Some(String::new())` (`metrics.rs:261`) and `into_record` yields
  `""` (`metrics.rs:473-476`), so a `correlation_id`-keyed map collapses every record
  onto one `""` key (abort-on-duplicate or all-to-one). `finish_with_records` already
  has `entry.uuid` at drain (`metrics.rs:434`); the worker drain seam must return each
  record **paired with its `Uuid`** (`Vec<(Uuid, RecordIngest)>`) so the join keys on
  the real uuid. Today the conflict is latent only because `RunCapture` uses
  `NativeMetricsObserver::new` (`execute.rs:3080`, dimensions retained).
- **Why first:** A1's worker-local merge concatenates records **per worker**, not in
  global dispatch order, which breaks the positional invariant. Removing that invariant
  as a standalone, no-behavior-change PR de-risks A1 entirely.
- **Design the join to tolerate a fallback record (for PR2):** the uuid-join must not
  assume every identity has exactly one merged worker record. Under A1 (PR2) some
  identities fail **before any worker observer registers them** (send failure, worker
  drop, placement cancellation) and are finalized by a coordinator-side fallback
  accumulator (A1 spec §3.3/Risk 4). Build the join so each identity resolves to a
  record from **worker-merge OR fallback**, and never abort on a legitimately
  fallback-served identity. In PR1 (no A1 yet) every identity still has one record, so
  behavior is unchanged; the join shape is just made forward-compatible.
- **Verify:** existing runner integration tests + the metric value-parity golden
  (captured from current HEAD before this lands). PR1 changes only row *ordering* logic
  with one record per identity, so its output must be byte-identical to HEAD.
- **Rollback:** trivial — single function.

## PR2 — A1 worker-local accumulation (the perf PR)

- **Change:** give each thread-per-core worker its own `NativeMetricsObserver`;
  **delete** the `BufferedObserver` event buffer + coordinator replay
  (`turn_execution.rs:164-223`, `:516-518`; gRPC twin `grpc_turn_execution.rs:376-378`).
  The token-accumulation O(tokens) cost moves to the worker (`on_token` local).
- **RUNNER MECHANISM IS RECORDS-FIRST, NOT ACCUMULATOR-MERGE (A1 spec §2.1 /
  Findings 1+2 — supersedes any "merge accumulators once" wording):** the runner
  builds its report by **re-ingesting `RecordIngest`s into a fresh accumulator in
  dispatch order** (`execute.rs:1557-1560`), after `finish` rewrites each record's
  `admit_ns` to the coordinator credit-issued time (`execute.rs:3220-3224`).
  Therefore:
  - The worker drain seam returns per-worker **records**
    (`drain_records -> Vec<(Uuid, RecordIngest)>`), **not** `MetricsAccumulator`s.
    Do **not** call `MetricsAccumulator::merge`/`summarize()` on the runner path.
  - **Why merge-summary is wrong here:** `admit_ns` is stamped worker-side at
    dispatch (`http.rs:895`; gRPC `grpc.rs:313`, "admit == dispatch time") and
    feeds `CreditToStartLatency`/`EffectiveLatency`/`CreditDropLatency`
    (`store.rs:1097-1103`). A pre-summarized worker accumulator bakes those in
    with queue≈0 and the post-run rewrite becomes impossible → those three metrics
    report ~0 on every request-rate/adaptive/back-pressured run (whole-metric
    corruption, not sub-ULP). Records-first keeps the rewrite intact.
  - **`request_index` MUST be a globally-unique, dense, monotonic dispatch index
    (DECIDED — not `None`/push).** Each worker's `finish_with_records` today stamps
    `request_index = Some(local slot 0..n_w)` (`metrics.rs:472`); concatenated,
    `worker0.record[0]` and `worker1.record[0]` both re-ingest to row 0 →
    `insert_record_at_with_token_arrivals` `assert!(!occupied[row])`
    (`store.rs:553-556`) panics (Finding 1). Fix: the coordinator uuid-join stamps
    each record's `request_index` = its **global dispatch ordinal** (the single
    issuance counter's value for that identity), unique across workers and dense
    `0..N-1` (every issued index filled by a worker record OR the fallback
    accumulator — no holes). Re-ingest at that index is collision-free, hole-free,
    and in HEAD dispatch order.
  - **Two rejected alternatives:** `request_index = None`/`push_record` gives
    drain/concatenation order (`[w0 ∥ w1 ∥ …]`), *not* HEAD's interleaved dispatch
    order → float fields drift; and a worker placing records at the global index in
    its **own** store holes that store → panic. So workers accumulate **dense-local**
    internally (`RunCapture` already does this, `execute.rs:3090` sets no
    `request_index`) and the **global** ordinal is applied only at the coordinator
    join — the two indices are decoupled.
  - **Single global counter:** the ordinal is `record_index = recorder.begin(...)`
    (`scheduled.rs:868`), which must stay a **single coordinator-owned** issuance
    counter (not reset per worker) — consistent with keeping admission/issuance
    coordinator-single-threaded and only measurement worker-local.
  - Because re-ingest is thus in **HEAD dispatch order**, the runner report (incl.
    float distributions) is **byte-identical to HEAD** — no ULP tolerance needed on
    the runner path (Finding 6; contrast the library merge-summary path).
- **gRPC parity (Finding 2):** the gRPC twin stamps/needs the same `admit_ns`
  rewrite (`grpc.rs:313`); it must also be records-first, and the regression matrix
  must add a gRPC parity row (currently absent).
- **Files:** `turn_execution.rs`, `grpc_turn_execution.rs`, `execute.rs`
  (`RunCapture` now builds one observer per worker + the records-first
  concatenate/uuid-join/`request_index`-reassign/`admit_ns`-rewrite/re-ingest),
  `online_execution.rs` driver.
- **Design boundary (DECIDED): single issuer now; shard *only* accumulation.**
  Issuance/admission (credit mint + `SlotPool` slots + the global dispatch-index
  counter) stays a **single coordinator-owned issuer**; only **measurement** goes
  worker-local. This is aiperf-v2's **Tier 0 "Direct"** (`docs/deps/aiperf-v2-
  cellular-runtime.md`, REQ 5 `IssuanceTier`); **distributed per-cell issuers
  (Tier 2 "Cellular Autonomous") are deferred** to ultra-scale. Rationale: a single
  Rust issuer clears typical high loads with large headroom — a central dispatcher
  thread caps ~1.7 M/s (cross-thread-sync bound) and the HTTP data plane is the real
  limit far below that (`~/.claude/benchmark-findings/rust-singular-dispatcher-vs-
  worksteal-credit-throughput.md`); issuance only becomes the wall near ultra-scale.
- **Single issuer buys two correctness wins — keep them:**
  1. **Exact global concurrency** — one `SlotPool` enforces the exact global limit
     (Tier 0/1 are "exact"; only Cellular Autonomous relaxes to per-cell/partition-
     bounded). Matches HEAD semantics.
  2. **Deterministic, byte-parity-friendly `request_index`** — because a single
     issuer assigns the global dispatch ordinal **sequentially and deterministically**
     (request *k* always gets index *k* in a seed-driven order), the re-ingest fold
     order is reproducible → runner byte-parity comes for free. **Implement issuance
     as a single central assignment point, NOT a timing-dependent shared-atomic
     self-issue** — a shared atomic gives unique+dense indices but *non-deterministic*
     assignment (which request gets which index varies run-to-run), reordering the
     IEEE-754 fold and breaking run-to-run float reproducibility.
- **Put the issuer behind an `IssuanceAuthority` seam** (one trait, `Direct` impl
  today) so the Cellular-Autonomous distributed issuer is a future drop-in — matching
  both the repo's extensibility discipline and aiperf-v2's `IssuanceTier` so the
  single-process and eventual cellular designs stay one model at two scales.
- **Pre-worker failure fallback (fail-closed hazard — A1 spec §3.3/Risk 4):** requests
  that fail before a worker observer registers them (send failure `turn_execution.rs:476`,
  worker-drop `:501-503`, placement cancellation `:465-466,681-692`) have **no** worker
  record. The coordinator must synthesize the same errored/canceled `RecordIngest` HEAD
  produces (execute.rs:3314-3328; scheduled.rs:951-999) as **coordinator-owned fallback
  records**, keyed by their uuid and folded into the same records-first re-ingest set
  (§2.1) so every identity resolves to a worker record **or** a fallback record and
  `errored`/`canceled`/`ErrorRequestCount`/timing stay HEAD-identical. Without this the
  uuid-join misses those identities and the run aborts fail-closed.
- **Live results sink (non-consuming — A1 spec §3.3/Risk 3):** `CapturePhaseProcessor`
  today reads a **non-consuming** `snapshot_record` from the coordinator observer. With
  a `live_sink` attached the worker must return a **cloned** `RecordIngest` in
  `WorkerReply.live_record`; **do NOT use `drain_terminal_record`** (it `take_terminal`s
  and would drop each live-emitted request from the final merge, undercounting). Test:
  a `--live` run's end-of-run counts == a non-live run's.
- **Evidence gate before landing (A1 spec §1.1.1):** capture coordinator flamegraphs on
  BOTH a long-output streaming workload and a short-output/non-streaming/usage-only
  workload, confirming the buffered-observer replay (not the per-request coordinator
  funnel) is the dominant cost. If the funnel dominates on short-output, A1 will not
  lift that workload's ceiling — the persistent-lane restructure (A4 #1A/#4b) is the
  real lever there.
- **Depends on:** PR1. **Independent of** A2.
- **Verify:** value-parity golden — **on the runner path ALL fields (integer AND
  float distributions) are byte-exact vs HEAD**, because records-first re-ingest folds
  in dispatch order (Finding 6); a runner float delta is a real regression, not accepted
  drift. The ULP/tolerance apparatus (A1 §4.5) applies only to the library merge-summary
  path, not the runner product path. Add: the merge-determinism unit harness
  (library-path merge-order independence for integer aggregates), the
  `request_index`-reassignment test (Finding 1 — concatenated per-worker records
  re-ingest without a slot-collision panic), an **aggregate-only (`records:false`)
  uuid-join row** (Finding 3 — join keys on uuid, not the empty `correlation_id`),
  **error/cancellation-injection rows** proving the fallback path, a live-sink
  count-equivalence test, a **gRPC parity row** (Finding 2), and the throughput-win
  harness on both long- and short-output workloads.

## PR3 — A3 Connector seam (Tcp/Uds/Duplex)

- **Trait (compile-verified):** `#[async_trait::async_trait(?Send)] trait Connector {
  async fn connect(&self, url, cfg, clock, trace) -> Result<(Sender, SocketInfo)> }`,
  held as `Rc<dyn Connector>` (native async-fn-in-trait is not dyn-compatible → boxed;
  `Arc` is wrong — impls hold `!Send` `Rc` resolvers).
- **Impls:** `TcpConnector` (owns the `DnsResolver` + TLS + socket opts, today's
  `connection.rs:446-508`), `UdsConnector` (the extracted `if let Some(uds_path)`
  branch `:428-444`), `DuplexConnector` (`tokio::io::duplex` → `spawn_local` in-process
  handler via `Rc<dyn DuplexEndpoint>`).
- **Touch points:** `connection.rs` (delete UDS branch, `establish*` → shims/timeout
  wrapper — **keep as shims: `transport_bench.rs:516` calls `establish` directly**),
  `pool.rs:116` (`resolver` field → `connector`), `pool.rs:309/400`
  (`establish_with_resolver` → `connector.connect`), **`pool.rs:28-35` `origin_key`
  must key UDS/Duplex by path/name, not the synthetic host — highest-risk item**,
  `http_client.rs:276/330`, `config/defaults.rs:199-202` (drop `uds_path`), and
  **`rust/runtime/src/graph/transport_bench.rs:462-506,516` — the ONE live non-`None` writer
  of `uds_path` (NOT dead code, correcting the earlier claim; it sets `uds_path` via
  struct-shorthand which a `grep "uds_path:"` misses). Migrate it to `unix:`-scheme
  connector selection in this SAME PR or the `aiperf-runtime` crate's `graph` module (formerly
  the `aiperf-graph` crate) fails to compile (field drop) /
  silently TCP-connects its dummy `http://localhost` URL (branch drop).**
- **Selection:** scheme→connector at the endpoint-prepare composition root
  (`http(s)://`→Tcp, `unix:/path`→Uds), **not** a config flag threaded through layers.
  `mem://`→Duplex is **test/bench-only** — no Config-v2 surface supplies a
  `DuplexEndpoint`, so Duplex is not product-reachable by design.
- **UDS socket path ≠ HTTP route+Host — retain a separate route URL (Finding 4,
  correctness bug otherwise):** `build_request_with_method` derives the request line
  from `url.path()` and the `HOST` header from `url.authority()`, **both from the same
  URL** (`http_client.rs:408-416`). Today UDS keeps two URLs apart —
  `uds_path`=socket, a distinct `http://localhost/v1/chat/completions`=route+Host
  (`transport_bench.rs:462-466`). Selecting `UdsConnector` from a `unix:/run/x.sock`
  URL and then feeding that same URL to request-building POSTs to `/run/x.sock` with an
  **empty Host** (404/400). So `select_connector` must return the connector **and** a
  separate route URL (or the prepared endpoint retains one); dropping `uds_path` removes
  only the connect-selector flag, not the route/Host source. Migrated `transport_bench`
  must keep its `http://localhost/...` route URL for request-building while the `unix:`
  URL only selects/targets the socket. See A3 spec §3.2 correction.
- **Python side is NOT a one-line whitelist (correcting scope):** `endpoint.py`
  (~460-475) rejects `unix:` on **three** gates — `not netloc/hostname` (462-466),
  scheme-not-in-whitelist (467-471), and the port-parse block. Making `unix:`
  product-reachable requires relaxing the netloc/hostname requirement for `unix:`,
  adding `unix` to the whitelist, skipping port parsing for `unix:`, then threading the
  URL through `rust_wire`/`EndpointProfileConfigV2` (no new `uds_path` field). Budget
  this as real Python work.
- **UDS-win throughput driver is unbudgeted:** `rps_bench` (Harness C driver) calls
  `establish` directly and is not wired to a `unix:` client (regression plan §1.4).
  Rewiring it to a `UdsConnector` (the `fast_sse` example that formerly exposed a
  `UDS_PATH` + `unix:` client has since been removed, so a new `unix:` client driver is
  needed) to
  measure the headline UDS win is additional PR3 work; without it the seam risks
  landing test-only with no measured product win.
- **Independent of** A1/A2/A4.
- **Caveat (verified):** `DuplexConnector` is clean for an online in-process mock on
  `RealClock` but does **not** unify the offline `SimClock` sim (needs a Clock-driven
  steppable server — separate, later).
- **Verify:** connector-correctness harness (Tcp/Uds/Duplex metric-identical), UDS
  throughput-win harness, `origin_key` isolation test.

## PR4 — A4 lean per-request (after PR2)

Ranked do-first (high-impact, localized):

1. **#1B batched classified-token hook** — add one **backward-compatible defaulted**
   method to `RequestObserver` in `loadgen-core/src/sink.rs` (precedent: `on_output_tokens`,
   `grpc.rs:388`); collapses the per-token `Uuid` hashmap lookup to **one lookup/request**
   while **preserving token-arrival order for ICL**. **Reasoning-model TTFOT trap
   (A4 spec Finding #1 risk):** the classified-batch override must set
   `first_output_token_ns` on the **first `Output`-kind element**, guarding on kind like
   the per-chunk `on_classified_token` (metrics.rs:607-611) — NOT blindly `batch[0]` as
   `on_output_tokens` does (metrics.rs:632-634, valid only because that path is
   output-only). A reasoning-led mixed batch `[(t0,Reasoning),(t1,Output),…]` must yield
   `first_output=t1`. Add a test feeding a reasoning-led batch and asserting TTFOT ==
   first Output arrival.
2. **#3 memoize `InferenceDimensions`** sink-side (`http.rs:1257/1283/1332`) — intern
   the constant `endpoint_url`+`model` instead of allocating two `String`s/request.
3. **#6 retention/ICL gate** — extend `new_aggregate_only`/`retain_record_dimensions`
   (`metrics.rs:240-270`) to skip `token_arrivals_ns` + dimension clones when records/ICL
   are off; **add an ICL toggle to `MetricsConfig`, fail-closed** (only `InterChunkLatency`
   consumes the *full* arrival vector, `store.rs:1057-1063`). **The retention keep-list
   MUST also retain `second_token_ns` (Finding 5) — not just `first_token_ns`,
   `first_output_token_ns`, `last_arrival_ns`, and the counts.** TTST (`catalog` id 200,
   a shipped metric) is computed from `second_token_ns.zip(first_token_ns)`
   (`store.rs:1009-1013`), sourced from `token_arrivals_ns.get(1)` (`metrics.rs:490`).
   Dropping the arrival vector without separately keeping the second arrival makes
   `second_token_ns=None` → TTST absent/NaN in exactly the fast-path aggregate-only mode
   the optimization targets, while TTFT/e2e stay present — a silent shipped-metric loss.
   Keep a scalar `second_arrival_ns` alongside `first`/`last` when the vector is skipped.

- **Defer:** #2 inline-SSE-parse (a transport restructure overlapping A3, not a
  constant-factor tweak) and #5 integer-ns timestamps (cross-observer seam change,
  low priority).
- **Verify:** value-parity golden — **ICL must stay byte-identical** (the batched-token
  ordering risk) — plus a retention-gate test **and a reasoning-led mixed-batch TTFOT
  test** (first_output = first Output-kind element, not batch[0]) **and a
  `records:false`/no-ICL run asserting TTST is still present and correct (Finding 5 —
  `second_token_ns` retained), so the retention gate does not silently drop a shipped
  metric.**

## PR5 — A2 records-first compat projection (library/offline dedup, LAST)

- **Change:** derive `TraceSimulationReport` from the single native observer's retained
  `RecordIngest` records (**records-first**, replaying the collector's exact math),
  removing the parallel `CollectorObserver` from the `run.rs`/offline/accuracy tees.
- **Non-negotiable parity (verified risks):** the projection must reproduce the
  collector's **nearest-rank** percentiles (native uses linear interpolation,
  `kernel.rs:62-123` vs `collector.rs:1119-1132`) and its **credit-issued** latency time
  base (native uses transport-start, `metrics.rs:447-456` vs `scheduled.rs:896`). A
  summary-only projection silently changes library-visible p50/p90/p99 and TTFT — do
  **records-first**. Fill field gaps from records: `itl` dist + `max_itl`,
  `output_token_throughput_per_user`, `num_requests` (incl. canceled), good-only goodput.
- **Real (non-ULP) semantic deltas the field table previously mis-marked "OK" — these
  MUST be records-first too (A2 spec §5/§6.2):**
  - **`tpot`** — native `InterTokenLatency = (end_ns − start_ns − ttft)/(osl−1)` with
    `osl = usage.completion_tokens` diverges from collector
    `(last_token − first_token)/(streamed_count − 1)` on BOTH numerator
    (`end_ns ≠ last_token`) and denominator (`completion_tokens ≠ streamed count`).
    Recompute from `token_arrival_ns`.
  - **`total_output_tokens`** — native sums `completion_tokens.or(output count)`;
    collector sums the **streamed-token count** (`token_times_ms.len()`). Recompute the
    streamed count from records (or accept native usage-based totals and re-pin the
    golden + `run.rs`/`scheduled_real_mock.rs`).
  - **`e2e` back-tail** — native `RequestLatency` ends at terminal `end_ns` (after the
    final content token/usage frame); collector ends at `last_token`. Fixing only the
    front arrival base is insufficient — project `last_token − arrival` from records,
    do **not** substitute `end_ns` for `last_token`.
  These diverge as the **norm** under CLAUDE.md's authoritative-usage design, not as an
  edge case, so summary-only projection changes library-visible throughput and latency.
- **Reuse:** `phase_runtime.rs:287,775-786` already conditionally drops the collector
  (`collect_performance_summary`) — the seam exists.
- **Files:** `run.rs:434/897/1151`, `dynosim.rs`, `phase_runtime.rs`, `report.rs`,
  new projection fn.
- **Verify:** cross-mode parity harness — new-projection report **== byte-identical**
  old dual-observer report across the workload matrix. This is the gate.

---

## Cross-cutting (every PR)

- **Doc sync (mandatory):** update the four agent files + `CLAUDE.md` +
  `specs/README.md` + `llms.txt` in the same PR (repo doc-guard enforces it). Also fix
  the stale claims the audit found (protocol-v1 not "isolated"; `EndpointType` not
  "compatibility-only"; missing `aiperf-telemetry-archive`/`aiperf-prometheus`; the
  offline feature is `dynosim`, not `dynamo-offline`).
- **Gating:** no PR merges without its regression harness green (companion plan).
- **Golden capture:** snapshot the value-parity golden from **current HEAD before PR1**,
  so PR1→PR5 are all diffed against pre-refactor behavior.
