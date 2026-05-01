# AgentX RFC L0 Tests

## Purpose

Minimum implementation-agnostic L0 trust tests for any AgentX RFC trace-playback load generator.

A passing L0 run proves that a predefined full-functionality AgentX validation trace can be replayed against a mock or instrumented server with auditable agreement between:

- trace input
- request plan
- client ledger
- server ledger
- exported metrics
- methodology metadata

If these cannot be reconciled, the run is not trustworthy.

## Absolute requirements

An AgentX trace-playback load generator must provide all of the following before its results should be trusted:

1. **A predefined full-functionality validation trace** that is public or shipped with the test harness, not the private benchmark trace set.
2. **Multiple root users/conversations** in that validation trace, each with growing multi-turn context.
3. **Agent topology coverage**: at least one spawned subagent, one join, context isolation, and total in-flight requests exceeding root-user concurrency.
4. **Timing coverage**: think-time replay with original response time excluded, delay-cap behavior, warmup, measured window, cooldown/drain, and visible lateness when the player falls behind.
5. **Recycle/cache coverage**: one cache-busted recycled trace execution and one cache-preserving recycled trace execution.
6. **A deterministic mock or instrumented server run** with scripted streaming, failure, timeout, pause, zero-output, and normal responses.
7. **Stable correlation IDs** joining trace input, request plan, client ledger, server ledger, metrics, and methodology metadata.
8. **Client/server metrics reconciliation** for request counts, success/failure counts, token counts, TTFT, TPOT, retries, failures, and metrics-window inclusion.
9. **Methodology metadata** that states what was measured, what was not measured, scheduling mode, timing policy, cache policy, retry policy, trace identity, and output-length policy.
10. **Manual server-metrics validation for publishable runs** when real engine metrics are part of the claim; engine-specific cache/offload counters are required only when exposed and claimed.

If any item is missing, the L0 result is incomplete.

## Validation trace identity

The L0 trace must be a predefined, hard-coded full-functionality AgentX validation trace. It is a conformance fixture for proving loadgen behavior, not the private benchmark trace set. Actual benchmark users may run any trace pack.

The validation trace identity must be pinned before L0 results are accepted:

- validation trace name
- source path or URI
- schema version
- validation trace version
- SHA256 digest or equivalent immutable content digest
- expected event count
- expected request-plan manifest path
- expected topology manifest path
- expected timing manifest path
- expected mock-server response script path

If the validation trace identity is not pinned, the L0 run is incomplete.

## L0 harnesses

Run the same validation trace through these harnesses:

1. **Golden replay harness** — compares generated request plan, topology ledger, timing ledger, and coverage report against expected manifests.
2. **Deterministic mock-server harness** — replays against scripted streaming, failure, timeout, pause, and zero-output responses; reconciles client metrics to server ledger.
3. **Recycle/cache harness** — exercises cache-bust and cache-preserve recycle modes and verifies prefix identity changes or preservation.
4. **Methodology harness** — validates metadata labels, non-measured claims, policies, digests, and artifact join keys.
5. **Manual server-metrics harness** — for publishable runs, reconciles client metrics against real engine metrics where exposed; Prometheus-only engines may provide aggregate, not per-request, reconciliation.

## Required L0 validation trace

The validation trace must include:

- multiple root conversations/users, each with growing multi-turn context
- enough model-invocation events per root to prove multi-turn replay, not just one request per user
- at least one context-only tool/system event
- at least one streaming response
- at least one controlled failure response
- at least one inter-turn gap where original response time must be excluded from replayed think time
- at least one inter-turn gap above the delay cap
- at least one warmup request outside the measured window
- at least one measured request
- at least one cooldown/drain request outside the measured window
- one parent branch, one spawned subagent branch, and one join
- one parallel subagent moment where total in-flight requests can exceed configured root conversation concurrency
- one repeated content/hash/prefix unit that should materialize identically within a trace execution
- one recycled trace execution where cache-bust policy should prevent unintended cross-cycle prefix reuse
- one recycled trace execution where cache reuse is intentionally preserved

## Global artifact rule

Every per-request or per-event artifact must include stable trace, event, request, and correlation identifiers sufficient to join trace input, request plan, client ledger, server ledger, and metrics exports. If trace contents are secret, these identifiers may be redacted or HMAC-derived, but they must remain stable across all L0 artifacts for the same run.

## L0.1 Trace-to-request fidelity

Must verify:

- model-invocation events become the intended model requests
- context-only events do not produce model requests
- message/event order is preserved where ordered by the trace
- prior context grows according to each root user's trace history rule
- context does not bleed between root users unless explicitly encoded by the trace
- tool/system/context segments are preserved
- model, endpoint, temperature, max output, stream flag, and routing fields are preserved
- no silent truncation, mutation, deduplication, role collapse, or field loss occurs

Required artifact:

- request ledger keyed by stable correlation id

## L0.2 Subagent topology fidelity

Must verify:

- parent lineage is preserved
- spawned subagents use isolated context unless explicitly inherited
- subagent context is not pulled back into parent lineage unless explicitly declared
- joins wait for the correct prerequisites
- background/terminal branches do not incorrectly block parent flow
- total in-flight requests can exceed root conversation concurrency during parallel subagents
- root conversation concurrency remains stable while total request concurrency rises

Required artifact:

- topology ledger keyed by stable correlation id

## L0.3 Timing semantics

Must verify:

- replayed think time excludes original model response time when required
- delay cap is applied exactly
- scheduled send time and actual send time are recorded
- lateness is recorded when the player falls behind
- topology dependency waits are distinguishable from accidental client-side serialization
- warmup, measured, and cooldown windows include/exclude the correct requests
- a controlled mock-server pause is visible in lateness or tail-latency artifacts

Required artifact:

- timeline ledger keyed by stable correlation id

## L0.4 Warmup and sweep semantics

Must verify:

- initial conversations can start from configured random offsets
- warmup requests are excluded from measurement
- measurement starts only after the declared warmup boundary
- cooldown/drain requests are excluded unless explicitly included by methodology
- fixed-duration runs report unique traces, repeated traces, turns, and measured model invocations
- root conversation concurrency is reported separately from total request concurrency including subagents
- multiple root users can be active in the same run without serializing into a single-user replay

Required artifact:

- warmup/sweep ledger

## L0.5 Determinism

Must verify:

- same seed, trace set, and config produce the same logical request sequence
- trace selection, random start offsets, and recycling order are deterministic when configured
- synthetic prompt materialization is stable when used
- runtime nondeterminism cannot change request identity
- metadata includes seed, validation-trace digest, config digest, player version, validation-trace version, and methodology version

Required artifact:

- workload fingerprint and request-ledger hash

## L0.6 Synthetic content and cache realism

Must verify:

- repeated content/hash/prefix units materialize identically when intended
- distinct content/hash/prefix units do not accidentally collide
- recycled trace executions apply the declared cache-bust policy
- cache-busted recycled executions do not accidentally reuse prior-cycle prefixes
- cache-preserving recycled executions keep the expected reusable prefix identity
- modeled reuse granularity is reported
- cache-hit claims do not use set-intersection shortcuts when contiguous-prefix behavior is required
- trace-derived reuse potential is distinguished from server-observed cache behavior
- server-observed cache/offload validation is required only for engines that expose those counters; otherwise cache/offload claims are marked unsupported

Required artifact:

- content/cache ledger

## L0.7 Metrics truth against mock server

Must verify:

- sent request count equals server-observed request count
- success and failure counts match
- streaming TTFT uses first token-bearing content chunk
- TPOT and output token metrics use the declared authoritative token counter
- zero-output responses are classified explicitly
- failed, timed-out, cancelled, retried, and malformed responses are not silently dropped
- retries are represented explicitly and not counted as independent workload demand
- metrics inclusion windows match request inclusion flags
- histogram/count/sum exports reconcile with per-request ledger

Required artifact:

- client-vs-server metrics reconciliation keyed by stable correlation id

## L0.8 Failure honesty

Must reject with diagnostic errors:

- missing parent branch
- orphan join
- cyclic branch dependency
- impossible timestamp ordering unless explicitly supported
- unsupported request type
- invalid endpoint/model mapping
- malformed streaming telemetry
- missing token-length fields when methodology depends on them
- duplicate correlation ids
- filters that remove required prerequisites
- filters that make warmup, measurement, or cooldown empty without reporting it

Required artifact:

- expected-failure snapshots with relevant trace/event/branch/correlation ids when available

## L0.9 Methodology metadata

Must export:

- trace-playback proxy vs closed-loop agent harness label
- open-loop vs closed-loop scheduling label
- validation-trace source and digest for L0 conformance runs; benchmark trace-pack source/digest only when disclosed or provided as controlled-access metadata
- synthetic/anonymized/real content label
- tokenizer/content materialization policy
- prefix-cache realism disclosure, with engine-specific cache/offload claims marked engine-specific rather than universal
- recycle cache-bust policy
- output length policy
- timing policy, including original-response-time exclusion
- delay-cap policy
- root conversation concurrency vs total request concurrency policy
- random-start/warmup policy
- retry policy
- warmup/cooldown policy
- metrics inclusion policy
- official-duration vs developer-short-run label
- explicit non-measured claims: task success, quality, real agent feedback loop, real tool latency, MoE routing realism, speculative-decoding acceptance realism when synthetic content is used

Required artifact:

- methodology report

## L0.10 Coverage and anti-gaming visibility

Must verify:

- low-concurrency runs report which traces were exercised
- fixed-duration runs report coverage fraction
- recycled traces are marked as repeats
- recycled trace executions report whether cache-bust or cache-preserve mode was applied
- randomized order is reconstructable from seed
- skipped/filter/rejected traces are reported with reasons
- unique traces are distinguished from repeated executions
- low-N percentile/sample cases are exposed

Required artifact:

- trace coverage report

## L0.11 Server-metrics cross-check

This can be validated manually for publishable runs when no automated per-request server-ledger integration exists. Prometheus-only metrics may support aggregate reconciliation rather than full correlation-id reconciliation.

Must verify:

- every client ledger row either joins to server ledger rows by correlation id or carries an explicit no-server-observation reason
- request counts reconcile after accounting for retries and connection failures that never reached server
- success/failure classification reconciles with server status/error records
- input/output token counts reconcile with server usage accounting or declared authoritative counter
- streaming chunk count and first token-bearing timestamp reconcile with server emission log
- client-observed TTFT is greater than or equal to server-side TTFT within declared tolerance
- client-observed end-to-end latency is greater than or equal to server receive-to-finish latency within declared tolerance
- server-side queueing/backpressure/token counters reconcile when exposed and claimed
- engine-specific cache/offload counters reconcile when exposed and when the result claims cache/offload behavior
- server cancellations/timeouts are reflected in client metrics
- client and server metrics use the same warmup/measured/cooldown inclusion window
- tolerances are declared and justified

Required artifact:

- server-metrics reconciliation table

## L0.12 Histogram and percentile integrity

Must verify:

- counts, sums, and histogram buckets reconcile with per-request ledger
- percentiles come from raw samples or aggregatable histograms, not averaged precomputed percentiles
- bucket boundaries or native-histogram config are declared
- duration units are consistent
- failed/timed-out request inclusion policy is explicit
- P50/P75/P90/P99 tables include sample count and measured-window request count
- low-sample percentile claims are marked unstable or omitted by policy

Required artifact:

- histogram reconciliation report

## Passing bar

The L0 suite passes only if all required artifacts reconcile:

- request ledger
- topology ledger
- timeline ledger
- warmup/sweep ledger
- workload fingerprint
- content/cache ledger
- metrics reconciliation
- failure snapshots
- methodology report
- trace coverage report
- server-metrics reconciliation
- histogram reconciliation
