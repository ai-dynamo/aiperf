# Lean per-request / per-token online scheduled hot path — verified optimization spec

**Date:** 2026-07-12
**Scope:** constant-factor allocation / lookup / round-trip reductions on AIPerf's
**online scheduled HTTP** dispatch path (`rust/runtime/src/scheduled.rs` →
`rust/runtime/src/http.rs` → observer chain `ObserverTee` →
`{CollectorObserver, NativeMetricsObserver}`). No algorithmic or metric-value
change is proposed; every item preserves the current report bytes.
**Status:** design / not implemented. Repo was read-only for this analysis.

Every cost below was confirmed at an exact `file:line` in the current tree
(branch `ajc/rust`). Two non-obvious compile shapes were proven in a scratch
crate at `~/tmp/a4-spec/proof/` (`cargo run` → `ALL PROOFS OK`): a
backward-compatible batched-classified observer hook (default method, object
safe) and slot-handle threading into a moved dispatch task.

---

## 0. The hot path, as it actually runs

One scheduled request flows through:

1. `ScheduledRuntime::issue_turn_internal`
   (`rust/runtime/src/scheduled.rs:808`) — per request: computes
   `dimensions = self.dispatcher.inference_dimensions(&turn)`
   (`scheduled.rs:892`), `recorder.borrow_mut().begin(&turn, …)`
   (`scheduled.rs:868`, clones at `scheduled.rs:358-359`),
   `native_metrics.register_metadata(...)` (`scheduled.rs:875`), then
   `self.scheduler.execute_async(Box::pin(async move { … }))`
   (`scheduled.rs:905`) with `turn.clone()` inside
   (`scheduled.rs:924` / `:933`).
2. `TransportSink::dispatch_collect_record_with_hooks`
   (`rust/runtime/src/http.rs:871`) — streams via the transport, then a
   **post-hoc loop over `rec.responses`** (`http.rs:948-1006`) that, per SSE
   chunk, parses `ChatChunk` (`http.rs:961`) and calls
   `obs.on_classified_token(uuid, self.ms(msg.perf_ns), kind)` (`http.rs:978`).
3. `obs` is an `ObserverTee` (`rust/runtime/src/metrics.rs:688`, built at
   `scheduled.rs:486`) fanning **every** token callback to
   `CollectorObserver` (`rust/loadgen-core/src/observer.rs:49`) **and**
   `NativeMetricsObserver` (`rust/runtime/src/metrics.rs:546`).

Key structural fact: **the transport does not parse while streaming.** The
streaming callback only runs the first-token filter until the first meaningful
token is seen (`rust/runtime/src/transport_http/client/http_client.rs:701-709`,
gated by `!first_seen`); all SSE messages are buffered into `record.responses`
(`http_client.rs:709`) and the real parse happens later in the `http.rs:948`
loop. This changes the true shape of Finding #2 (see below).

Per-token lookups happen in **both** observers:
- `NativeMetricsObserver::on_classified_token` → `request_mut(uuid)` →
  `request_slots.get(&uuid)` (FxHashMap) then `requests.get_mut(slot)`
  (`metrics.rs:603-614`, helper at `metrics.rs:176-179`).
- `CollectorObserver::on_token` → `TraceCollector::on_token` →
  `self.requests.get_mut(&uuid)` (FxHashMap)
  (`observer.rs:56` → `rust/loadgen-core/src/collector.rs:740-746`).

So "one UUID hashmap lookup per token" is really **two** (tee → both maps),
each on the tight post-hoc loop.

---

## Finding #1 — per-token UUID map lookup in the observers

**Confirmed cost.** `NativeMetricsObserver::request_mut`
(`rust/runtime/src/metrics.rs:176-179`) does `request_slots.get(&uuid)?`
(FxHashMap<Uuid,usize>) + `requests.get_mut(slot)`. Called for every token at
`metrics.rs:603-614` (`on_classified_token`) and `metrics.rs:605`. The
`ObserverTee` (`metrics.rs:707-711`) also drives `CollectorObserver::on_token`
→ `TraceCollector::on_token` FxHashMap lookup (`collector.rs:740`). Both run in
the post-hoc loop at `http.rs:978`, i.e. **2 hash lookups × output-token
count** per request. The slot index is already known before dispatch:
`record_index` (`scheduled.rs:868`) is passed as
`RequestMetricMetadata.request_index` (`scheduled.rs:878`) and becomes the
observer's dense slot at `metrics.rs:559` (`request_index.get_or_insert`).

**Precedent for the cheap fix already in-tree.** The batched hook
`RequestObserver::on_output_tokens(uuid, &[f64])` exists
(`rust/loadgen-core/src/sink.rs:112`) and is already used by the gRPC path
(`rust/runtime/src/grpc.rs:388`), the endpoint-aware HTTP path
(`rust/runtime/src/http/endpoint_dispatch.rs:416`), and dynosim
(`rust/runtime/src/dynosim.rs:2099`). `NativeMetricsObserver::on_output_tokens`
(`metrics.rs:617-635`) already does **one** lookup for the whole batch.
`transport_bench.rs:191,219` is the graph-lane exemplar: accumulate arrivals in
a **local `Vec`** during the stream, hand off once.

**Proposed change — two independent options:**

- **Option B (localized, recommended first).** In the `http.rs:948` post-hoc
  loop, accumulate token arrivals into a local buffer instead of calling
  `on_classified_token` per chunk, then emit **one** batched observer call at
  the end. Because output/reasoning can (rarely) interleave and ICL windows
  depend on **global arrival order** (`store.rs:1057-1063`), the existing
  `on_output_tokens` (output-only) is insufficient to preserve exact order in
  the mixed case. Add an **order-preserving classified batch hook** with a
  default that replays per-token (so no other impl must change):

  ```rust
  fn on_classified_tokens(&self, uuid: Uuid, batch: &[(f64, ObservedTokenKind)]) {
      for &(at, kind) in batch { self.on_classified_token(uuid, at, kind); }
  }
  ```

  `NativeMetricsObserver` overrides it to do one lookup + one `reserve`;
  `CollectorObserver` overrides it to one lookup + push loop; `ObserverTee`
  forwards. Result: **2 lookups per request** instead of 2 × N.

- **Option A (seam-carried slot handle).** Thread the already-known
  `record_index` into the dispatch task and into a per-request observer handle
  so token callbacks index `requests: Vec<Option<PendingEntry>>` directly with
  **zero** map lookups. This is a larger seam change (a new
  slot-keyed observer entry point, or a `RequestToken`/handle threaded through
  `dispatch_turn`) and overlaps the A1 lane restructure.

**Seam impact.** Option B extends `loadgen-core::RequestObserver` with **one
defaulted method** — backward compatible, object-safe (proven in the scratch
crate via `&dyn RequestObserver`). It requires editing `ObserverTee`
(`metrics.rs`), `NativeMetricsObserver`, `CollectorObserver`, and the
`aiperf_runtime::adaptive_core` tee (`rust/runtime/src/adaptive_core/observer.rs:91`) to override
for full benefit, but nothing breaks if they don't. Option A is a deeper
cross-crate seam change.

**Expected impact.** High. Collapses the dominant per-token work (2 hash
lookups + 2 `RefCell` borrows) to per-request. Scales with output length.
**Blast radius.** Option B: medium (one crate seam + 3-4 observer impls,
default keeps the rest correct). Option A: large.
**Risk.** Low-medium for B, with **one sharp correctness trap on reasoning
models.** The claim that first-output bookkeeping "already handles the batch
case" is **only true for the OUTPUT-ONLY path**: `on_output_tokens` sets
`first_output_token_ns.get_or_insert(token_arrivals_ns[len − at_ms.len()])` —
the **first appended element** (metrics.rs:632–634) — which is correct there only
because that path assumes *every* element is `Output`. The new classified batch
hook must NOT copy that logic. A mixed batch can lead with a reasoning token,
e.g. `[(t0, Reasoning), (t1, Output), …]`, and `TimeToFirstOutputToken` must be
`t1`, not `t0`. The per-chunk `on_classified_token` gets this right by guarding
`first_output_token_ns.get_or_insert` **on `ObservedTokenKind::Output`**
(metrics.rs:607–611); the batch override must replicate that guard — scan the
batch for the first element whose kind is `Output` and only then set
`first_output_token_ns` — never blindly take `batch[0]`. Modeling the classified
override on `on_output_tokens` (metrics.rs:632–634) would silently corrupt TTFOT
(and every TTFOT-derived metric) on reasoning-then-output streams. Verify with a
test that feeds a reasoning-led mixed batch and asserts `first_output_token_ns`
== the first Output arrival. Otherwise B preserves order/first-token correctly.
Overlaps **A1** (if lanes are restructured, Option A falls out naturally).

---

## Finding #2 — "double SSE parse" (reframed after reading the transport)

**Confirmed, but narrower than reported.** The filter closure at `http.rs:929`
calls `is_meaningful_chat_token(message)` (`http.rs:60-65`) which parses
`ChatChunk`; the post-hoc loop re-parses `ChatChunk` at `http.rs:961`. **But**
the filter closure is only invoked until the first meaningful token
(`http_client.rs:701` guard `!first_seen`), so the *double* parse touches only
the **leading** messages (a role-only chunk + the first content chunk), not
every chunk. The reported "deserializes each chunk … then RE-deserializes"
overstates the per-chunk overlap.

The **real** cost is structural, and matches the fix the finding proposes: the
transport buffers **all** SSE messages into `record.responses`
(`Vec<Response>`, each owning the `data` string) and parses them in a **second
pass** (`http.rs:948-1006`), instead of parsing inline during the stream. The
graph lane already does the lean thing — parse `ChatChunk` inside the streaming
`on_msg` callback and never accumulate `Response` objects
(`rust/runtime/src/graph/transport_bench.rs:206-232`). The redundant generic
`serde_json::Value` parse was already gated behind `capture_wire_responses`
(`http.rs:956`, default set to skip for perf runs per `http.rs:670-680`), so
that half is done.

**Proposed change.** Move `ChatChunk` decode + delta/usage extraction into the
streaming SSE callback (mirror `transport_bench.rs:206-232`) and surface the
parsed facts (delta text, kind, `perf_ns`, terminal usage) to the collector,
eliminating the `record.responses` buffer + second pass for scheduled perf
runs. Combine with Finding #1 Option B so inline parsing feeds one batched
observer handoff.

**Seam impact.** This is **not** a `loadgen-core` change — it is a **transport
restructure** in `aiperf_runtime::transport_http` + `rust/runtime/src/http.rs`. The
transport currently returns `RequestRecord { responses: Vec<Response>, … }`
consumed by many callers (evaluation/agentic read `wire_responses`; endpoint
dispatch decodes `record.responses` at `endpoint_dispatch.rs:368`). A streaming
parse path must either (a) be an additional lean entry point used only by the
scheduled perf sink, or (b) keep `responses` capture optional. The
first-token-filter closure signature (`FnMut(i64, &SseMessage) -> bool`,
`http_transport.rs:158`) would grow to surface the parsed chunk (or be replaced
by a richer streaming handler like `RecordingSseHandler`,
`http_client.rs:691`).

**Expected impact.** Medium-high on long streams (drops N `Response` heap
objects + N re-parses per request), low on the leading-chunk double-parse
itself. **Blast radius.** Large — transport API surface + every
`record.responses` consumer. **Risk.** Medium-high; byte-exact SSE parse scars
(`sse.rs:53-69` JSON-continuation, `[DONE]`, role-only, usage-only) must be
preserved and covered by the existing loopback parity tests. **This is the
"connector spec" overlap** — best done as its own transport-streaming redesign,
not folded into the constant-factor batch.

---

## Finding #3 — per-request `String` allocations for constant dimensions

**Confirmed cost.** `dispatcher.inference_dimensions(&turn)` is called **every
request** at `scheduled.rs:892`. For `TransportSink` (the `TurnDispatcher` impl,
`http.rs:1283-1293`) it allocates: `selected_url(...)` which for the common
`endpoint_path=None` returns `selected_url.clone()` — a fresh `String`
(`http.rs:814`, `:807`) — plus `Some(self.model.clone())` (`http.rs:1291`) /
`turn.effective_model.clone()`. Both are constant per endpoint/model for the
vast majority of runs. Additionally:
- `recorder.begin` clones `turn.conversation_id` + `turn.x_correlation_id`
  every captured request (`scheduled.rs:358-359`).
- `register_metadata` builds `RequestMetricMetadata` cloning `conversation_id`
  and `request_correlation_id` (`scheduled.rs:884-889`), then may null them
  (`metrics.rs:258-262`) — already skipped when `retains_record_dimensions()`
  is false.

**Proposed change.** Intern the constant `InferenceDimensions` once. Because
`InferenceDimensions { endpoint_url: Option<String>, model: Option<String> }`
is cloned into every `RecordIngest.dimensions` (`metrics.rs:481`) and used as a
grouping key for per-endpoint/model series, the cheapest zero-risk move is:
- cache the resolved dimensions for the `url_index=None`/no-`endpoint_path`
  common case on the sink (compute lazily once per `(url_index, model)` pair),
  returning clones of interned `Rc<str>`-backed values; or
- change `InferenceDimensions` to carry `Option<Rc<str>>` so the per-request
  clone is a refcount bump, not a heap `String` copy. (This touches
  `aiperf_runtime::metrics_core`'s `InferenceDimensions` — a wider but mechanical change.)

The conversation/correlation clones are genuinely per-request (unique per
turn/session) and cannot be interned; leave them (they are already elided in
aggregate-only mode).

**Seam impact.** Localized to `rust/runtime/src/http.rs` if done as a sink-side
cache returning cloned interned strings. If `InferenceDimensions` is changed to
`Rc<str>`, it touches `aiperf_runtime::metrics_core` (not `loadgen-core`).
**Expected impact.** Medium — removes 2 `String` allocs/request on the steady
path. **Blast radius.** Small (sink cache) to medium (`InferenceDimensions`
type change ripples through `store.rs`/`report.rs`). **Risk.** Low; values are
identical, only ownership changes. Do the **sink-side memoization** first.

---

## Finding #4 — boxed future + full `turn.clone()` per request

**Confirmed cost.** `self.scheduler.execute_async(Box::pin(async move { … }))`
(`scheduled.rs:905`) heap-allocates one boxed future per request. Inside,
`turn.clone()` (`scheduled.rs:924` or `:933`, one branch runs) deep-clones
`TurnToSend` — Strings (`conversation_id`, `x_correlation_id`,
`request_correlation_id`), `messages: Vec`, `request_headers: BTreeMap`,
`request_parameters: BTreeMap`, optional `request_body: Bytes`. The clone
exists because `dispatch_turn(turn, …)` consumes the turn
(`http.rs:1295`/`PreparedHttpTurn::from_turn` needs owned data) while the async
block still needs `turn` afterward for the lifecycle observer
(`scheduled.rs:1002`, `on_terminal(&turn, …)`) and `turn.uuid`
(`scheduled.rs:977`, `:983`). `runtime = self.clone()` (`scheduled.rs:904`) is a
cheap `Rc` bump (method is `self: &Rc<Self>`), **not** a concern.

**Proposed change.**
- Wrap the turn in `Rc<TurnToSend>` (or split the small "post-dispatch" facts —
  `uuid`, `conversation_id`, `is_final_turn` — out before the move) so the
  dispatcher borrows/refcounts instead of deep-cloning. `PreparedHttpTurn::from_turn`
  would take `&TurnToSend` (it already reads fields and clones selectively,
  `http.rs:431-486`).
- The boxed future is intrinsic to `execute_async`'s `Pin<Box<dyn Future>>`
  task-queue contract; removing it requires the **A1 persistent-lane
  restructure** (a long-lived per-lane task pulling from a queue, like
  `transport_bench.rs:561-603`), not a local edit.

**Seam impact.** `turn` ownership is localized to `rust/runtime/src/{scheduled,
http}.rs`. The `Box::pin` removal is **not** localized — it is the A1 lane
restructure. **Expected impact.** Medium for the clone removal (one deep clone
of a multi-String/BTreeMap struct per request); the `Box::pin` itself is a
single small alloc/request (low). **Blast radius.** Medium (dispatcher
signatures `dispatch_turn`/`dispatch_turn_streaming` → `&turn` or `Rc<turn>`).
**Risk.** Low-medium. **Overlaps A1** for the boxed-future half.

---

## Finding #5 — `ns → ms(f64) → ns` round-trip per token

**Confirmed cost.** The transport stamps tokens as f64 ms:
`obs.on_classified_token(uuid, self.ms(msg.perf_ns), kind)` where
`self.ms(ns) = (ns - start_ns) as f64 / 1_000_000.0` (`http.rs:795-797`,
`:978`). `NativeMetricsObserver` immediately converts back:
`relative_ns_from_ms(ms) = (ms * 1_000_000.0).round_ties_even() as i64`
(`metrics.rs:392-397`, called `metrics.rs:604`). So each token does a divide +
multiply + round it would not need if the seam carried integer ns. The
`RequestObserver` timestamp type is `f64` ms across the whole seam
(`rust/loadgen-core/src/sink.rs:87-116`).

**Proposed change.** Add integer-ns observer entry points (proven in scratch:
`on_classified_token_ns(uuid, at_ns, kind)` with a default that divides to
preserve every existing impl). `NativeMetricsObserver` overrides to store
`at_ns` directly (it already stores i64 ns internally). `CollectorObserver`
stores ms (`collector.rs:744`) so it keeps the divide — meaning the round-trip
is only fully removed once the collector also moves to ns, or if the native
observer is the only ns-sensitive consumer.

**Seam impact.** This **is** a `loadgen-core::RequestObserver` signature change
(new ns methods across all impls: `NativeMetricsObserver`, `CollectorObserver`,
`ObserverTee`, `aiperf_runtime::adaptive_core` window sampler, `aiperf_runtime::graph`, offline
`dynosim`, plus the `Dispatchable` test doubles). Object-safe with defaults.
**Expected impact.** Low-medium — one FP multiply+round per token removed; small
next to the hash lookups. **Blast radius.** Large (every observer impl and every
call site that stamps ms). **Risk.** Medium — must not perturb the existing
`round_ties_even` rounding that report parity depends on; keeping f64 ms as the
default path avoids changing collector math. **Coordinate with `loadgen-core`;
do after #1 (the batch already amortizes the lookup, shrinking #5's relative
value).**

---

## Finding #6 — `records:false` retention: also skip `token_arrivals_ns` + dims

**Confirmed seam exists.** `NativeMetricsObserver::new_aggregate_only`
(`metrics.rs:240-254`) sets `retain_record_dimensions=false`; `register_metadata`
already nulls `worker_id`/`conversation_id`/`correlation_id` in that mode
(`metrics.rs:258-262`), and `scheduled.rs:882-889` already skips the
conversation/correlation clones when `retains_record_dimensions()` is false.
The finding asks to extend this to also drop the growing
`token_arrivals_ns: Vec<i64>` (`metrics.rs:93`, `PendingRequest`; allocated with
`Vec::with_capacity(requested_output_length)` at `metrics.rs:578`, pushed per
token at `metrics.rs:606`).

**Feasibility — verified, with a caveat.** `token_arrivals_ns` feeds three
things when a record is finalized (`metrics.rs:489-492`): `first_token_ns`
(`.first()`), `second_token_ns` (`.get(1)`), and the whole vector →
`RecordIngest.token_arrival_ns` → `store.populate_raw_metrics`. In the store the
**only** consumer that needs the *full* vector is **InterChunkLatency**
(`store.rs:1057-1063`, consecutive-gap ragged series). TTFT/TTFO/TTST come from
the separately-stored `first_token_ns`/`second_token_ns`/`first_output_token_ns`
scalars (`store.rs:1000-1021`); OSL/ITL/TPOT come from usage
`completion_tokens`/OSL (`store.rs:1029-1044`), not the arrival vector. So the
vector is droppable **iff InterChunkLatency is not being reported**. But ICL is
currently **always** computed for the online path — there is no
`MetricsConfig` flag to disable it (`MetricsConfig`,
`rust/runtime/src/metrics_core/accumulator.rs:109-127`, has no ICL toggle), and the
online metrics test asserts ICL is present (`metrics.rs:1108-1112`).

**Proposed change.** Add an explicit `retain_token_arrivals` / "ICL enabled"
policy (thread from `records:false` + no-ICL-sweepline into
`MetricsConfig`/observer). When off: keep `first_token_ns`, **`second_token_ns`**,
`first_output_token_ns`, a running `last_arrival_ns`, and the output/reasoning
counts (already tracked at `metrics.rs:609-612`); do not allocate/push the
`Vec`. `into_record` then emits an empty `token_arrival_ns` and the store skips
the ICL ragged series for those rows. This also lets Finding #1 Option B skip
the per-request batch buffer entirely in aggregate mode.

> **CORRECTION (Finding 5) — the keep-list MUST include the second token
> timestamp, or TTST (a shipped metric) is silently dropped.** An earlier keep-list
> read "keep only `first_token_ns`, `first_output_token_ns`, a running
> `last_arrival_ns`, and the output/reasoning counts" — omitting `second_token_ns`.
> But `TimeToSecondToken` (TTST, `catalog` id 200, a real shipped metric) is
> computed from `record.second_token_ns.zip(record.first_token_ns)`
> (`store.rs:1009-1013`), and `second_token_ns` comes from
> `token_arrivals_ns.get(1)` (`metrics.rs:490`). If the arrival vector is skipped
> without separately retaining the second arrival, `second_token_ns = None` → TTST
> becomes absent/NaN in exactly the aggregate-only/no-ICL fast path this
> optimization targets, while TTFT/e2e stay present. So the observer must keep a
> scalar **second** arrival (feeding `second_token_ns`) alongside first/last when it
> stops pushing the vector. The feasibility paragraph above already notes "TTST come
> from the separately-stored `second_token_ns` scalar" — the keep-list must honor
> that. Guard with a `records:false`/no-ICL test asserting TTST is present and
> correct, in addition to the ICL-enabled test.

**Seam impact.** Localized to `rust/runtime/src/metrics.rs` +
`rust/runtime/src/metrics_core` (a `MetricsConfig` bool and a store guard). **Not** a
`loadgen-core` change. **Expected impact.** Medium — removes the largest
per-request allocation (a `Vec<i64>` sized to output length) for aggregate-only
/ no-ICL runs. **Blast radius.** Small-medium (config plumb + store guard +
tests that assert ICL). **Risk.** Medium: must fail closed — if ICL *is*
requested, the vector must be retained; a wrong gate silently drops a shipped
metric. Guard with a test that ICL still appears whenever the flag is on.

---

## Ranked plan

| # | Optimization | Impact | Localized? | `loadgen-core` seam change? | Do-first |
|---|---|---|---|---|---|
| **1B** | Batched classified token handoff (1 lookup/request) | **High** | Mostly (1 defaulted seam method + 3-4 observer impls) | **Yes** — 1 backward-compatible defaulted method | ✅ #1 |
| **3** | Intern/memoize constant `InferenceDimensions` (sink-side) | Medium | **Yes** (sink cache) | No | ✅ #2 |
| **6** | Skip `token_arrivals_ns` Vec + dims when no records/ICL | Medium | **Yes** (`aiperf`/`aiperf_runtime::metrics_core`) | No | ✅ #3 |
| **4a** | Avoid `turn.clone()` (Rc/borrow) | Medium | Yes (dispatcher sig) | No | next |
| **5** | Integer-ns token timestamps | Low-med | No (all observer impls) | **Yes** — new ns methods everywhere | seam-coordinated |
| **2** | Inline SSE parse, drop `Vec<Response>` second pass | Med-high | No (transport restructure) | No (transport API, not loadgen-core) | connector spec |
| **1A / 4b** | Slot-handle + persistent lanes (no boxed future) | High | No | Yes (slot-keyed seam) | **A1 restructure** |

**Do-first (high-impact + localized):** **#1 Option B**, **#3 sink-side
memoization**, **#6 retention skip**. These three are independent, need at most
one backward-compatible `loadgen-core` addition (#1B), and land the bulk of the
per-token / per-request constant-factor win with low risk.

**Need `loadgen-core` coordination:** **#1** (one defaulted method — safe) and
**#5** (new ns timestamp methods across every observer — larger, do after #1).

**Restructure overlaps (call out explicitly):**
- **A1 (persistent lanes):** Finding **#1 Option A** (slot handle, zero lookups)
  and Finding **#4b** (eliminate the per-request `Box::pin`) both fall out of the
  A1 long-lived-lane restructure (`transport_bench.rs:561-603`); do not attempt
  them as isolated edits.
- **Connector spec (transport streaming):** Finding **#2** is a transport-level
  streaming-parse redesign (parse inline, stop buffering `record.responses`),
  not a constant-factor tweak; it belongs with the connector/streaming spec and
  must preserve the SSE parse scars + `wire_responses` consumers.

---

## Verification evidence

- **All six costs** confirmed at the `file:line`s cited inline above (read from
  the working tree, branch `ajc/rust`, 2026-07-12).
- **Finding #1/#5 compile shapes** proven in `~/tmp/a4-spec/proof/`
  (`cargo run` → `ALL PROOFS OK`): (a) a defaulted, object-safe
  `on_classified_tokens(uuid, &[(f64,Kind)])` batch hook exercised via
  `&dyn RequestObserver` with a one-lookup override; (b) a defaulted integer-ns
  `on_classified_token_ns`; (c) a per-request `SlotHandle { slot }` threaded
  into a `move` closure indexing `Vec<Option<Entry>>` directly (Option A / A1
  shape).
- **Finding #2 reframe** proven by reading `http_client.rs:701-709` (filter runs
  only until `first_seen`) and `transport_bench.rs:206-232` (the lean inline
  parse that never buffers `Response`).
- **Finding #6 feasibility** proven by `store.rs:1000-1063`: only
  InterChunkLatency consumes the full arrival vector; all other token-timing
  metrics use separately-stored scalars — but `MetricsConfig`
  (`accumulator.rs:109-127`) has no ICL toggle today, so a gate must be added.
- **In-tree batch precedent** (`grpc.rs:388`, `endpoint_dispatch.rs:416`,
  `dynosim.rs:2099`; native impl `metrics.rs:617-635`) confirms `on_output_tokens`
  batching is already the accepted pattern — #1B generalizes it to the classified
  case for order preservation.
