# AgentX RFC L0 Trust Tests for Agent Trace Playback Load Generators

## Purpose

These are implementation-agnostic L0 tests for any load generator used to implement the AgentX RFC's trace-playback MVP.

The RFC's purpose is to move beyond fixed-shape single-turn scenarios toward an agentic-coding workload proxy: high-turn multi-turn conversations, linearly growing context, tool/context segments, short inter-turn gaps, subagent spawning, high prefix-cache pressure, steady-state warmup, and concurrency sweeps.

The L0 goal is trust and correctness. A human reviewer should be able to take a predefined full-functionality AgentX validation trace, replay it against a mock or instrumented server, and reconstruct exactly what the player did, why it did it, what it measured, and what it did not measure.

A run that cannot reconcile its trace input, client request ledger, server ledger, exported metrics, and methodology report is not trustworthy or publishable as an AgentX RFC result.

## Scope

These tests do not prove that trace replay is a true closed-loop agentic-coding benchmark. They prove that, within the RFC's trace-playback MVP class, the player faithfully implements the claimed workload and reports auditable metrics.

The L0 suite must explicitly prevent two overclaims:

- Calling trace replay a closed-loop agent benchmark when future prompts are not generated from the model-under-test's actual outputs.
- Treating a plausible aggregate metric table as trustworthy when request topology, timing, token accounting, cache assumptions, or server-side observations cannot be reconciled.

## Definitions

- **AgentX trace**: A trace intended to represent agentic coding behavior: multi-turn context growth, tool/context segments, subagents, and short inter-turn think-time gaps.
- **Trace event**: One record in the source trace. It may represent a model invocation, tool result, system/context mutation, branch operation, warmup marker, or metadata.
- **Model-invocation event**: A trace event that should produce an HTTP/gRPC/model-server request.
- **Context event**: A trace event that should be preserved in future request context but should not itself produce a model-server request.
- **Replay player**: A load generator that plays a recorded or synthesized trace against a model server.
- **Open-loop replay**: Requests are scheduled from trace time or a configured arrival process. A slow response does not delay later independent scheduled sends, except where the trace topology explicitly requires dependency ordering.
- **Closed-loop replay**: Later sends are gated on prior responses completing. This can be valid when it matches the intended workload, but it must be labeled because it can hide latency during stalls.
- **Closed-loop agent harness**: A real agent loop where future prompts are generated from the model-under-test's actual outputs and tool results. A trace replay player is not this unless it actually runs such a loop.
- **Think time**: The inter-turn gap the RFC wants replayed after excluding original model response time.
- **TTFT**: Time to first token-bearing content chunk. Metadata-only, role-only, heartbeat, or usage-only chunks do not count as first token unless the methodology explicitly defines TTFT differently.
- **TPOT**: Time per output token over token-bearing output after TTFT, using the declared authoritative token counter.
- **Client ledger**: The player-side per-request record.
- **Server ledger**: The mock or instrumented server's per-request record.
- **Correlation id**: A stable id present in the trace-derived request plan, client ledger, server ledger, metrics export, and reconciliation report. For secret traces, this may be a redacted or HMAC-derived identifier.

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

## Required AgentX L0 validation trace

The validation trace must be broad enough to exercise the RFC trust boundary:

- Multiple root conversations/users, each with context growth across turns.
- Enough model-invocation events per root to prove multi-turn replay, not just one request per user.
- At least one context-only event representing tool/system/context material that should appear in a later prompt.
- At least one streaming response.
- At least one controlled failure response.
- At least one inter-turn gap where original response time must be excluded from replayed think time.
- At least one inter-turn gap above the RFC delay cap so the cap is exercised.
- At least one warmup request outside the measured window.
- At least one measured request.
- At least one cooldown or drain request outside the measured window.
- At least one parent branch, spawned subagent branch, and join.
- At least one subagent that runs in parallel with the parent or another subagent, so total in-flight requests can briefly exceed configured root conversation concurrency.
- At least one repeated content/hash/prefix unit that should materialize identically within a trace execution.
- At least one recycled trace execution where cache-bust policy should prevent unintended cross-cycle prefix reuse.
- At least one recycled trace execution where cache reuse is intentionally preserved.

## L0.1 RFC workload-shape fidelity

The player must prove that the replayed workload has the shape the RFC claims.

Must test:

- High-turn multi-turn conversations remain multi-turn after parsing, filtering, warmup positioning, and replay planning.
- Context grows according to each root user's trace history rule; turn N contains the intended prior user/assistant/tool/system context for that root.
- Context does not bleed between root users unless explicitly encoded by the trace.
- Tool, system, and context segments are preserved as context rather than discarded or flattened into unrelated user text.
- Model-invocation events become exactly one intended model request unless the trace explicitly encodes batching, fanout, retry, or another non-1:1 policy.
- Context-only events do not produce model requests.
- Event order is preserved wherever the trace declares an ordering dependency.
- Per-invocation parameters are preserved: model, endpoint, temperature, max output, stream/non-stream, and request-specific routing fields.
- No silent truncation, mutation, deduplication, role collapse, or field loss occurs.

Human verification artifact:

- A compact request ledger with correlation id, trace id, conversation id, event id, request id, roles/segments, declared lengths, context-history summary, parameters, endpoint, and model.

## L0.2 Subagent topology fidelity

The player must prove it preserves the RFC's subagent behavior, not just a flat request list.

Must test:

- Parent conversation lineage is preserved.
- Spawned subagents get isolated context unless the trace explicitly declares inherited context.
- Subagent context is not pulled back into the parent lineage unless the trace explicitly declares that behavior.
- Branch joins wait for the right prerequisites.
- Background or terminal branches do not incorrectly block parent flow.
- Parallel subagents can increase total in-flight requests beyond root conversation concurrency when the topology allows it.
- Root conversation concurrency remains stable even when subagent requests temporarily increase total in-flight request count.
- Branch ids, parent ids, join ids, and correlation ids are stable and auditable.

Human verification artifact:

- A topology ledger showing correlation id, parent event, spawned branches, joins, terminal branches, dependency edges, and whether each branch inherited or isolated context.

## L0.3 RFC timing semantics

The player must prove its replay clock implements the RFC timing model. Latency tests must guard against coordinated omission: a player that stops observing or stops sending during stalls can report plausible but misleading tail latency.

Must test:

- Think time is computed from the trace by excluding original model response time when the RFC mode requires it.
- The RFC inter-turn delay cap is applied exactly.
- Scheduled send time and actual send time are measured.
- If the player falls behind schedule, lateness is recorded instead of hidden.
- Open-loop and closed-loop behavior are separately tested, or unsupported modes are explicitly rejected.
- Dependency waits caused by trace topology are distinguishable from accidental client-side serialization.
- Warmup, measurement, and cooldown windows include or exclude the right requests.
- A controlled mock-server pause is visible in lateness/tail-latency artifacts rather than disappearing from the distribution.

Human verification artifact:

- A timeline ledger with correlation id, trace timestamp, original response-time component, computed think time, capped delay, scheduled time, actual send time, server receive time when available, first token-bearing chunk time, response finish time, lateness, and metrics-included flag.

## L0.4 Steady-state warmup and sweep semantics

The player must prove it implements the RFC's steady-state start and sweep semantics, because these directly shape headline throughput and tail-latency results.

Must test:

- Initial conversations can start from configured random offsets inside their trace rather than all starting at turn zero.
- Warmup requests needed to establish steady-state context are excluded from measurement.
- Measurement starts only after the declared warmup boundary.
- Cooldown/drain requests are excluded from measurement unless the methodology explicitly says otherwise.
- Fixed-duration runs report how many unique traces, repeated traces, turns, and model invocations were actually measured.
- Low-concurrency sweep points report smaller sample counts instead of hiding low-N uncertainty.
- Per-concurrency runs preserve the configured root conversation concurrency while separately reporting total request concurrency including subagents.
- Multiple root users can be active in the same run without serializing into a single-user replay.

Human verification artifact:

- A sweep/warmup ledger showing configured root concurrency, total in-flight high-water mark, warmup requests, measured requests, cooldown requests, unique trace coverage, repeated trace count, and sample count per metric.

## L0.5 Determinism and reproducibility

The player must prove the same logical RFC run can be repeated when configured to be deterministic.

Must test:

- Same seed, trace set, and config produce the same logical request sequence.
- Trace selection, random start offsets, and recycling order are deterministic when configured.
- Synthetic prompt materialization is stable when used.
- Non-deterministic map/set iteration, runtime hash randomization, thread scheduling, or process ordering cannot change request identity.
- Run metadata includes seed, validation-trace digest, config digest, player version, validation-trace version, and methodology version.

Human verification artifact:

- A workload fingerprint and request-ledger hash.

## L0.6 Synthetic content and cache-realism disclosure

The RFC's trace-playback MVP depends on reconstructing prompts from anonymized or synthetic material. The player must prove that materialization is stable and disclose what cache realism it can and cannot claim.

Must test:

- Repeated trace content/hash/prefix units materialize identically when they are supposed to model reuse.
- Distinct trace content/hash/prefix units do not accidentally collide.
- Recycled trace executions apply the declared cache-bust policy.
- Cache-busted recycled executions do not accidentally reuse prior-cycle prefixes.
- Cache-preserving recycled executions keep the expected reusable prefix identity.
- The player reports the granularity at which reuse is modeled, such as token, block, hash block, or text-prefix unit.
- The player does not report cache-hit rate using a set-intersection shortcut when the claimed server behavior requires contiguous prefix matching.
- Cache-related reported metrics distinguish trace-derived reuse potential from actual server-observed cache behavior.
- Server-observed cache/offload validation is required only for engines that expose those counters; otherwise cache/offload claims are marked unsupported.
- Output text is captured or server-side metrics are available when the methodology claims anything about MoE routing, speculative-decoding acceptance, or gibberish/coherence effects.

Human verification artifact:

- A content/cache ledger showing original trace unit id, materialized unit digest, reuse group, modeled reuse granularity, and any server-observed cache metric used for reconciliation.

## L0.7 Mock-server metrics truth

The player must prove reported client-side metrics match what a known mock server emitted and what the client observed.

Must test against a mock server:

- Sent request count equals observed request count.
- Success and failure counts match.
- Streaming TTFT is computed from first token-bearing content chunk, not request start, request end, metadata-only chunk, role-only chunk, heartbeat, or usage-only chunk.
- TPOT and output-token metrics use the declared authoritative token counter.
- Zero-output responses are classified according to explicit policy and cannot become successful requests with artificial `ttft=0` unless that behavior is explicitly reported as degraded.
- Failed, timed-out, cancelled, retried, and malformed responses are not silently dropped.
- Retry attempts are represented explicitly and cannot be mistaken for independent workload demand.
- Metrics windows match request inclusion flags.
- Histogram/count/sum exports reconcile with the per-request ledger; precomputed percentile summaries are not averaged across instances.

Human verification artifact:

- A player-summary vs. mock-server-ledger comparison, keyed by correlation id.

## L0.8 Failure honesty

The player must reject untrustworthy RFC inputs loudly.

Must test:

- Missing parent branch.
- Orphan join.
- Cyclic branch dependency.
- Impossible timestamp ordering, unless explicitly supported and reported.
- Unsupported request type.
- Invalid endpoint/model mapping.
- Malformed streaming telemetry.
- Missing token-length fields when the methodology depends on them.
- Duplicate correlation ids.
- Trace filters that remove required prerequisites.
- Trace filters that make warmup, measurement, or cooldown windows empty without reporting it.

Human verification artifact:

- Expected-failure snapshots with error messages that identify all available relevant identifiers, such as trace id, event id, turn id, branch id, correlation id, and violated invariant.

## L0.9 Methodology disclosure

The player must not let users overclaim what the AgentX RFC MVP measured.

Must test that exported metadata includes:

- Trace-playback proxy vs. closed-loop agent harness label.
- Open-loop vs. closed-loop scheduling label.
- Validation-trace source and digest for L0 conformance runs; benchmark trace-pack source/digest only when the benchmark runner chooses to disclose or provide controlled-access metadata.
- Whether content is synthetic, anonymized, or real.
- Tokenizer/content materialization policy.
- Prefix-cache realism disclosure, including modeled reuse granularity, block size when relevant, hashing, salting, multimodal, LoRA, or full-block-only assumptions that affect cache behavior.
- Recycle cache-bust policy.
- Engine-specific cache/offload claims marked engine-specific rather than universal.
- Output length policy: enforced, natural, ignore-eos, retry-on-empty, etc.
- Timing policy, including whether original response time was excluded from think time.
- Delay-cap policy.
- Concurrency policy, including root conversation concurrency vs. total in-flight request concurrency.
- Random start/warmup policy.
- Retry policy.
- Warmup/cooldown policy.
- Metrics inclusion policy.
- Duration policy and whether the run is an official-duration run or a developer-short proxy run.
- Known non-measured things, such as task success, quality, real agent feedback loop, real tool latency, MoE routing realism, and speculative-decoding acceptance realism when synthetic content is used.

Human verification artifact:

- A one-page methodology report generated from run metadata.

## L0.10 Anti-gaming and subset visibility

The player must make it obvious what workload actually ran, especially because the RFC's fixed-duration sweeps naturally produce smaller samples at low concurrency.

Must test:

- Low-concurrency runs report which traces were exercised.
- Fixed-duration runs report coverage fraction.
- Recycled traces are marked as repeats.
- Recycled trace executions report whether cache-bust or cache-preserve mode was applied.
- Randomized order is reconstructable from seed.
- No trace can be silently skipped without being reported.
- Filters report selected, skipped, repeated, and rejected traces with reasons.
- Coverage reporting distinguishes unique traces from repeated trace executions.
- The methodology report exposes low-N cases rather than presenting their percentiles as equally stable to high-concurrency points.

Human verification artifact:

- A trace coverage report listing selected, skipped, repeated, filtered, and rejected traces with reasons.

## L0.11 Server-metrics cross-check

The player must prove its client-side metrics are consistent with what the server says happened. Client and server durations are different perspectives: client duration includes client scheduling, serialization, network/proxy behavior, retries, and response transit; server duration measures the server's receive-to-finish view. L0 should reconcile them explicitly, not pretend they are identical.

Must test with a mock or instrumented server. This can be validated manually for publishable runs when no automated per-request server-ledger integration exists. Prometheus-only metrics may support aggregate reconciliation rather than full correlation-id reconciliation.

- Every client ledger row either joins to server ledger rows through correlation id or carries an explicit no-server-observation reason.
- Client request count matches server request count after accounting for retries and connection-level failures that never reached the server.
- Client success/failure classification reconciles with server status codes and error records.
- Client input/output token counts reconcile with server-side usage accounting or the declared authoritative token counter.
- Streaming chunk count and first token-bearing timestamp are consistent with the server emission log.
- Client-observed TTFT is greater than or equal to server-side time-to-first-token, except for declared clock-skew/timestamp-source tolerances.
- Client-observed end-to-end latency is greater than or equal to server-side receive-to-finish latency, except for declared clock-skew/timestamp-source tolerances.
- Server-side queueing, backpressure, and processing time are separated from client-side scheduling lateness where the server exposes enough detail.
- Server-side queueing, backpressure, and token counters reconcile when exposed and claimed.
- Engine-specific cache/offload counters reconcile when exposed and when the result claims cache/offload behavior.
- Server-side cancellations and timeouts are reflected in client metrics.
- Server metrics and client metrics use the same inclusion window: warmup, measured, and cooldown.
- Any tolerance used for reconciliation is declared in the methodology report and justified by timestamp source, clock synchronization, mock-server precision, or transport overhead.

Human verification artifact:

| Field | Client ledger | Server ledger | Status |
|---|---:|---:|---|
| requests_received | 120 | 120 | match |
| successful_requests | 118 | 118 | match |
| failed_requests | 2 | 2 | match |
| input_tokens | 1,024,000 | 1,024,000 | match |
| output_tokens | 128,000 | 128,000 | match |
| cache_reuse_events_if_exposed | 94 | 94 | match |
| p50_ttft_ms | 42.1 | 40.8 | client >= server, within tolerance |
| p99_e2e_ms | 912.4 | 908.0 | client >= server, within tolerance |

Failure rule:

- If client and server disagree, the run is not publishable until reconciled or explicitly marked degraded.

## L0.12 Histogram, percentile, and sample-size integrity

The player must prove summary statistics are derived from auditable distributions, not from lossy or invalid aggregation.

Must test:

- Count, sum, min/max when exported, and histogram buckets reconcile with the per-request ledger.
- Percentiles are derived from raw samples or aggregatable histograms, not by averaging precomputed percentile summaries across workers.
- Bucket boundaries or native-histogram configuration are declared.
- Units are declared and consistent, preferably seconds for duration metrics.
- Tail percentiles include failed or timed-out requests according to explicit policy; excluded requests must be reported separately.
- P50/P75/P90/P99 tables include sample count and measured-window request count.
- Low-sample percentile claims are marked unstable or omitted according to explicit policy.

Human verification artifact:

- A histogram reconciliation report showing sample count, included/excluded requests, bucket counts, percentile source, and low-N warnings.

## L0 trust bar

A passing L0 run must prove this statement:

> Given the predefined full-functionality AgentX validation trace and a mock or instrumented server, the player emits the intended multi-turn context growth, subagent topology, think-time replay, steady-state measurement window, metrics, server reconciliation, and methodology metadata exactly; if any trust-critical invariant is violated, the test fails loudly with auditable evidence.

The minimum human-auditable outputs are:

- Request ledger.
- Topology ledger.
- Timeline ledger.
- Sweep/warmup ledger.
- Workload fingerprint.
- Content/cache ledger.
- Client/server correlation map.
- Mock/server metrics reconciliation.
- Histogram reconciliation report.
- Trace coverage report.
- Methodology report.

The benchmark is not trustworthy if those artifacts cannot be reconciled by a human reviewer.

## Research and memory anchors

- AgentX RFC workload shape from prior analysis: high-turn multi-turn conversations, linearly growing context, tool/context segments, short think-time gaps, subagent spawning, high KV-cache pressure, steady-state warmup, and concurrency sweeps.
- AgentX RFC methodology from prior analysis: replay original trace speed while excluding original response time, cap long inter-turn delays, start initial conversations from random offsets, exclude warmup and cooldown from metrics, sweep root conversation concurrency, and allow subagents to briefly raise total in-flight requests above configured concurrency.
- Prior correctness findings from trace-replay audits: TTFT and TPOT are easy to compute incorrectly; zero-output retries can hide failures; cache-hit calculations can overstate reuse if they ignore contiguous-prefix behavior; fixed-duration low-concurrency runs expose fewer requests by construction.
- Brave New Geek, "Everything You Know About Latency Is Wrong": latency distributions, tail latency, and coordinated omission risks in load testing.
- OpenTelemetry HTTP semantic conventions: separate client/server duration metrics, attribute alignment, and client/server measurement caveats.
- Prometheus histogram practices: histogram aggregation, quantiles, bucket selection, and why precomputed summary quantiles are not aggregatable.
- vLLM prefix-caching documentation: prefix caching uses full-block cache behavior, hashing inputs, salts, and other implementation details that must be disclosed before claiming cache realism.
