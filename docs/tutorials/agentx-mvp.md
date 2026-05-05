<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# InferenceX AgentX MVP Benchmark

> **Status: Work-in-progress MVP.** This is the first AIPerf implementation of the
> SemiAnalysis InferenceX AgentX-MVP benchmark. The scenario, the rules it locks,
> and the output fields described here may change as the spec stabilizes. Don't
> treat any result you produce today as "final" — treat it as a useful
> apples-to-apples comparison run.

This page walks you through running the **AgentX MVP** benchmark in AIPerf. It's
aimed at someone who hasn't worked with the scenario before — you'll get a
copy-pasteable command first, then explanations of what it actually does and why.

---

## What Is AgentX MVP?

AgentX MVP is a multi-turn, agentic-coding benchmark proposed by SemiAnalysis as
part of their InferenceX effort. The idea: instead of measuring an inference
server with synthetic 1-turn prompts, measure it with realistic *coding-agent
sessions* — long conversations with subagents, KV-cache reuse, and inter-turn
think time. Sessions come from the public **WEKA agentic-coding trace corpus**
captured by Callan Fox ([kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)),
which records real Claude Code sessions byte-for-byte.

AgentX MVP is essentially a *recipe* on top of those traces: a fixed set of
replay rules so two different teams running on two different servers produce
results you can actually compare. Things like "inter-turn delays cap at 60
seconds", "the server must be allowed to generate full responses (no early
stop)", "warm up the cache before measuring", and so on.

AIPerf bundles every one of those rules into a single CLI flag:
`--scenario inferencex-agentx-mvp`. When you pass that flag, AIPerf locks the
relevant settings, rejects conflicting flags, and stamps a `submission_valid`
field onto the JSON output (both the per-run `profile_export.json` and, when
you pass `--num-profile-runs >= 2`, the aggregate file) so you can see at a
glance whether the run followed the rules.

---

## Quick Start

You'll need:

- An **OpenAI-compatible inference server** running and reachable.
- AIPerf installed (`make first-time-setup` if you're working from this repo).

The trace corpus is fetched automatically from HuggingFace
(`semianalysisai/cc-traces-weka-042026`, public, no auth, ~657 MB compressed,
739 traces) — no manual clone required. HF caches it locally so re-runs are
near-instant.

Then:

```bash
aiperf profile \
    --scenario inferencex-agentx-mvp \
    --url localhost:8000 \
    --model claude-opus-4-5-20251101 \
    --model claude-haiku-4-5-20251001 \
    --endpoint-type chat \
    --streaming \
    --public-dataset semianalysis_cc_traces_weka \
    --num-dataset-entries 739 \
    --concurrency 100 \
    --benchmark-duration 900 \
    --num-profile-runs 3
```

That's the whole thing. A few notes:

- **`--scenario inferencex-agentx-mvp`** is the only flag that's specific to
  this benchmark. Everything else is normal AIPerf.
- `--model` is whatever you're actually serving — you don't have to match
  the model names baked into the trace corpus. AIPerf rewrites every trace
  request's `model` field to one of your configured models. With one
  `--model`, every request goes to that model. With two (e.g. one for the
  parent agent, one for subagents), the trace's "main" model maps to the
  first `--model` and other distinct trace models map to the rest in
  first-appearance order. See [Per-Trace Model Rewriting](weka-trace.md#per-trace-model-rewriting)
  in the Weka tutorial for the full behavior.
- **`--benchmark-duration 900`** is the minimum AgentX MVP allows (15 minutes).
  Longer is fine. AIPerf will reject anything shorter.
- **`--concurrency`** is up to you and reflects the load you want to sustain.
  100 is a reasonable starting point.
- **`--streaming`** is not forced by the scenario — pass it yourself for chat
  endpoints. The WEKA traces were captured against streaming responses, so
  streaming replay matches the recorded request shape.
- **`--num-profile-runs 3`** is recommended for confidence-interval reporting.
  The `submission_valid` field is stamped on every run with `--scenario` set
  (single-run `profile_export.json` and, when `--num-profile-runs >= 2`, the
  aggregate file). With a single run you still get the validity stamp on the
  per-run file; multi-run adds the aggregate. See
  [Reading the Result](#reading-the-result-submission_valid) below.
- **`--num-dataset-entries 739`** loads the full 739-trace corpus. Without
  this flag, the loader caps at the AIPerf default of 100 rows and you'll
  benchmark against a 100-trace subset (the loader logs `Loading 100/739
  traces` at INFO so you can spot it). For a canonical AgentX MVP submission,
  use 739 (or higher — extra rows are silently ignored).

You don't need to pass `--ignore-trace-delays`, `--use-think-time-only`,
`--inter-turn-delay-cap-seconds`, `--fixed-schedule`, or anything related to
warmup. The scenario sets all of those for you. If you *do* pass one of them
with the wrong value, AIPerf will tell you up front rather than silently
producing an invalid result.

> **Optional: `--apply-chat-template`.** The scenario doesn't lock this
> either way. Pass it if you want AIPerf's reported ISL to count the full
> wire-token total — chat-template wrapping plus the cache-bust marker —
> instead of the bare prompt text. With the flag on, the synthetic-side
> compensation makes the metric directly comparable to a server's
> `usage.prompt_tokens`. Off (default), the metric counts the bare text
> the composer generated. Either is a valid AgentX MVP submission; pick
> whichever your reporting wants. See
> [Input Sequence Length (ISL) Tokenization](../reference/isl-tokenization.md)
> for the full picture.

> **Optional: `--use-server-token-count` (OSL mismatch fix).** By default
> AIPerf computes output sequence length (OSL) by re-tokenizing the
> server's response with the model's local tokenizer. If that tokenizer
> disagrees with the server's own tokenizer — different revision, vendor
> BPE merges, a different chat template — the reported OSL can drift from
> the server's actual emitted token count, and the per-run console will
> show an "Output Sequence Length Mismatch Warning" panel even though
> `ignore_eos=true` is locked and the server really did emit
> `max_tokens`. Pass `--use-server-token-count` to make AIPerf trust the
> server's `usage.completion_tokens` instead of re-tokenizing locally;
> the mismatch goes away. The scenario does not lock this flag either
> way — it's safe to add to an AgentX MVP submission.

---

## What `--scenario inferencex-agentx-mvp` Locks for You

When you pass the scenario flag, AIPerf checks (and in some cases sets) the
following settings before the run starts. If any of them conflict with what you
asked for, the run errors immediately with a clear message naming the offending
flag.

| Locked setting | What it means | Why it matters |
|---|---|---|
| `timing_mode` is `agentic_replay` | Use the multi-turn agentic-replay scheduler (locked in by the scenario; not a user-selectable flag) | This is the scheduling discipline AgentX MVP requires (warmup → steady-state, FIFO trace recycle, 60s clamp). |
| `extra_inputs.ignore_eos = true` | Server is told to ignore its end-of-stream token and generate the full requested length | Without this, models stop early and you measure their decision to stop, not the server. |
| `--use-think-time-only` is on | Inter-turn delays use the agent's recorded "think time" only, not "send-to-send time" | Send-to-send delays include the *previous* server's response time, which would unfairly slow your replay if your server is faster than the recording. |
| `--ignore-trace-delays` is off | Trace-derived inter-turn delays (the recorded `think_time`, see the row above) are not stripped — only clamped (see below) | The whole point of replay is to preserve the agent's pacing. |
| `--inter-turn-delay-cap-seconds = 60` | Any single inter-turn delay over 60s is clamped to 60s | Real coding sessions have 10-minute coffee-break gaps that would distort steady-state measurement. |
| `--cache-bust system_prefix` | Inject a unique per-conversation marker into the system message at the start of every play (or, if there is no system message, into the first user turn) | Without this, every time a trace is recycled the server's prefix cache would warm up further on identical content, and steady-state cache-hit rates would inflate the longer the run goes. The marker forces every recycled play of a trace to have a fresh prompt prefix. Auto-injected when you don't pass `--cache-bust` yourself. |
| Loader is `semianalysis_cc_traces_weka` or `weka_trace` | The dataset is the public `semianalysisai/cc-traces-weka-042026` HF dataset (via `--public-dataset semianalysis_cc_traces_weka`) or a local copy of the same corpus replayed via `--custom-dataset-type weka_trace --input-file <dir>` (the file-based `weka_trace` loader; `--input-file` alone won't auto-detect, you must pass the explicit type). Both produce byte-identical conversations — see [the Weka tutorial](weka-trace.md#file-based-vs-huggingface-which-to-use). | The benchmark is defined against this exact, hash-verifiable corpus so submissions are reproducible. |
| `--benchmark-duration ≥ 900` | The run lasts at least 15 minutes | Steady-state needs time to stabilize; short runs are noise. |
| No client-side input truncation | `--synthesis-max-isl` is rejected (it drops traces whose input length exceeds the cap, falsifying the workload) | Truncating prompts on the client side would falsify the workload. |
| `--random-seed` is set | If you didn't pass one, AIPerf picks a strong random one and logs it | Reproducibility — every replayed result can be regenerated. |

If you forgot to pass `ignore_eos`, `--use-think-time-only`, `--cache-bust`,
or `--random-seed`, AIPerf injects the locked value and tells you at INFO log
level. The same goes for `--inter-turn-delay-cap-seconds` when you didn't set
it explicitly. If you *did* pass one of these explicitly with a value that
conflicts with the scenario, AIPerf errors with all the violations listed at
once — you don't have to fix them one at a time.

---

## Reading the Result: `submission_valid`

When you use `--scenario`, AIPerf stamps a submission-validity flag onto every
JSON output for the run. The per-run `profile_export.json` carries it under
its `metadata` block, and when you also pass `--num-profile-runs >= 2` the
aggregate file (`profile_export_aiperf_aggregate.json`, in a per-run aggregate
subdirectory under your artifact directory) carries it too:

```json
{
  "metadata": {
    "scenario": "inferencex-agentx-mvp",
    "submission_valid": true,
    ...
  },
  "metrics": { ... },
  ...
}
```

Three possible states for `submission_valid`:

- **`submission_valid: true`** — the run honored every scenario rule and
  finished cleanly. This is the result you want.
- **`submission_valid: false`** — something went wrong (or you forced
  something). The same metadata block also contains `submission_invalid_reasons`,
  a list of short tags explaining why. Common values:
  - `"unsafe_override"` — you passed `--unsafe-override` along with one or
    more rule-breaking flags. See [`--unsafe-override`](#--unsafe-override) below.
  - `"context_overflow_rate_exceeded"` — more than 1% of the responses came
    back with a context-overflow error from the server, which means the server
    is rejecting prompts the benchmark requires it to handle. This usually
    points at the server being started with a reduced max model length;
    AgentX MVP requires the model's default.
- **Field absent** — you ran without `--scenario`. The submission-validity
  machinery is gated on the scenario flag.

If you see `submission_valid: false`, look at `submission_invalid_reasons` and
the AIPerf log. The reasons map one-to-one to either a scenario rule you broke
or a runtime threshold you crossed.

---

## How It Actually Runs

### Warmup Phase: Trajectories and `k_i`

Before AIPerf measures anything, it runs a **warmup phase** that primes the
server's KV cache. This isn't the standard generic AIPerf warmup — it's a
trajectory-based warmup specific to the agentic-replay scheduler.

Here's the picture. You set `--concurrency 100`. The scheduler picks 100
distinct conversations (call them *trajectories*) from the trace pool. For
each trajectory, it samples a random "starting turn" `k_i` somewhere in
roughly the first 70% of that conversation's turns (clamped to leave at
least one profile turn after warmup). Then, in the warmup phase, it
dispatches exactly *one* request per trajectory: turn `k_i` of conversation
`i`, with the full prefix history (turns 0 through `k_i-1`) attached as
message context.

The point is that the server's prefix cache fills with a realistic mix of
multi-turn coding contexts before any measurement starts. When the profiling
phase begins, every trajectory resumes from `k_i + 1` — and the server's cache
already holds the prefix.

The `k_i` values are deterministic given the random seed: same dataset + same
seed = same trajectories + same start points + same recycle order, on any
machine. That's why the scenario insists on a seed.

The warmup phase ends when **every** warmup request has resolved (success or
failure). If any warmup request fails terminally (after retries), AIPerf
aborts the run with a `TrajectoryWarmupFailedError` and lists the failed
trace IDs — the philosophy is "don't quietly start metrics on a degraded
warmup". Slow warmups are not aborted automatically: the warmup grace
period defaults to no limit, so the run will wait until every warmup
request resolves. If the warmup is taking longer than you expect, that's a
signal worth investigating in the server logs.

### Profiling Phase: Replay, Recycle, 60s Clamp

After warmup, the profiling phase opens. Now you're measuring. Each trajectory
keeps replaying its conversation from turn `k_i + 1` onward, honoring the
trace's recorded inter-turn think times — except any single delay over 60
seconds is silently clamped to 60 seconds.

When a trajectory finishes its conversation (last turn dispatched and
acknowledged), its trace ID goes back into a **FIFO recycle queue**, and the
slot picks up the next trace ID from the head of the queue. The recycle queue
starts pre-populated with every trace in the corpus that *isn't* currently a
trajectory. So as long as the corpus is larger than the trajectory count, every
trace gets played at least once before any trace is replayed twice.

A few wrinkles worth knowing:

- **Recycled traces start at turn 0**, not at a random `k_i`. The "start
  somewhere in the middle" rule applies only to the initial trajectories — the
  intent is to spread the *initial* state across the conversation length, not
  to keep injecting mid-conversation jumps forever.
- **Each play of a trace gets a fresh cache-bust marker.** When a trace ID is
  recycled (or first dispatched as a trajectory), AIPerf prepends a unique
  short tag like `[rid:8a3f2c1b9e7d]\n\n` to the conversation's system
  message (or, if the trace has no system message, to its first user turn —
  one injection per play, shared across all turns of that play). The tag is
  derived deterministically *within a single run* from the run's
  auto-generated benchmark ID, the recycle pass for that slot, the
  trajectory index, and the trace ID. The trace ID is part of the digest by
  design — without it, two different traces landing on the same
  `(recycle_pass, trajectory_index)` pair would collide on the same marker
  (~33% rate at MVP scale). Within one run, the same trace plays out with
  the same marker on every turn, and a different marker each time it
  recycles. Across runs, the markers differ (the benchmark ID is a fresh
  UUID each time), which is intentional — the whole point is that the
  server's KV-cache prefix doesn't get progressively warmer on identical
  content as the run goes on, because every recycled play has a fresh
  prompt prefix. Locked to `system_prefix` under the scenario.
- **Warmup and profiling share the marker for a given play.** The digest is
  intentionally phase-agnostic: a trajectory's warmup turn `k_i` and its
  first profiling turn `k_i+1` carry the *same* `[rid:…]`. That's how the
  KV-cache prefix work done during warmup transfers into measurement
  instead of being thrown away. (If `phase` were folded into the digest,
  warmup would prime a prefix the profiling phase never sees.)
- **Concurrency must fit your corpus.** AIPerf rejects runs at startup when
  `--concurrency` exceeds the number of usable trajectories (pool size minus
  traces too short to split into a warmup + profiling turn): each lane is
  pinned to a distinct trajectory, so the requested concurrency simply cannot
  be honoured. Pick a `--concurrency` that fits your corpus, or use a larger
  trace corpus.
- **Profiling ends** when `--benchmark-duration` elapses. Anything in flight
  finishes during a cooldown window and is included in the metrics; nothing
  *new* starts after the duration ends.

### Subagents

AgentX MVP traces include subagent invocations — the parent agent spawns a
helper, the helper runs to completion, the parent picks up where it left off.
AIPerf models these as `SPAWN` / `SPAWN_JOIN` branches in the conversation
DAG, replays the helper as a separate concurrent session, and waits for it
before the parent's next turn. Because subagents run *alongside* their parent,
your in-flight request count can briefly exceed `--concurrency`. That's
expected and correct; concurrency bounds the number of *parent* trajectories,
not the total open requests.

For more on the trace format, the SPAWN/JOIN mechanics, and how AIPerf maps
nested subagents, see the [Weka Traces tutorial](weka-trace.md).

---

## Live vs. Pre-Canned Assistant Turns (`AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES`)

By default, the weka loader emits each turn's delta with the trace's
**pre-canned** assistant content (synthesized from `prev_out_tokens` and
the recorded hash_ids) so the wire prompt's hash chain matches the
original recording byte-for-byte. The downside: the assistant tokens the
server *actually generates* on turn N never appear in turn N+1's prompt,
so the server's just-built KV blocks for the assistant region are
invalidated every turn — measured cache-hit rate underweights the
assistant prefix.

Set `AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1` to flip the
trade-off:

- The loader emits **user-only deltas** (assistant segments are still
  tracked internally for LCP / truncation correctness, but never sent on
  the wire).
- The conversation context mode becomes `DELTAS_WITHOUT_RESPONSES`, so
  the worker captures the server's live assistant response and threads
  it into the session's `turn_list` for the next turn's prompt.
- Cache-hit rate now reflects what a real agentic user would experience:
  the prior turn's KV is still valid because the server is reading back
  exactly the tokens it just emitted.

Caveat: server-generated assistant length will not exactly match the
trace's recorded `output_length`, so the boundary between assistant
blocks and the next user turn shifts by a few tokens each turn. Hash-id
equality past turn 0 is **not** preserved. For metrics that care about
the cache reuse pattern (cache-hit rate, prefill/decode mix, end-to-end
latency) this drift is harmless. For tooling that compares per-block
hits against the trace's recorded `hash_ids`, it isn't.

The default (`False`) is unchanged.

---

## `--unsafe-override`

Sometimes you intentionally want to break a scenario rule — to study the
sensitivity of one variable, to run a 1-minute smoke test instead of a
15-minute proper run, to see what happens with a smaller model. For that:

```bash
aiperf profile \
    --scenario inferencex-agentx-mvp \
    --unsafe-override \
    --benchmark-duration 60 \
    ...
```

What `--unsafe-override` does:

- **Converts every scenario rule violation from an error into a warning.** The
  run starts.
- **Stamps `submission_valid: false`** in every JSON output (per-run and, when
  `--num-profile-runs >= 2`, the aggregate file), with `"unsafe_override"` in
  `submission_invalid_reasons` — but only when at least one rule was actually
  broken. Passing the flag without breaking any rule is a no-op.

Once the flag was on AND a rule was broken, the run is marked invalid forever —
you cannot un-set the flag at runtime, you cannot launder a result through
post-processing. The flag is a no-op without `--scenario` (since there's no rule
set to override).

Use this for development. Don't use it for anything you want to compare
against other AgentX MVP runs.

---

## Troubleshooting

**`UnknownScenarioError: Unknown scenario 'inferencex-agentx-mvp'. Valid scenarios: …`**
Re-run `make generate-all-plugin-files` and reinstall (`make install`) —
your local plugin registry is out of date.

**`EmptyTracePoolError: Loader produced 0 traces; trajectories cannot be built.`**
The HF dataset download or row validation produced no usable traces. Check
your network connectivity to `huggingface.co` and confirm the dataset name
is `semianalysis_cc_traces_weka`. The shipped corpus has 739 traces.

**`TrajectoryWarmupFailedError: Trajectory warmup failed for N trace(s): …`**
Your inference server rejected one or more warmup requests after AIPerf's
normal retry budget. Check the server logs — common causes are an
authentication or model-name mismatch (e.g. `--model` doesn't match what
the server is serving), the server's `max-model-len` set lower than the
trace's requested context, or the server simply not running. AgentX MVP
deliberately aborts on warmup failure rather than producing a partial result.

**Run completes but `submission_valid: false` with `"context_overflow_rate_exceeded"`**
Your server is rejecting prompts as too long for more than 1% of requests.
The most common cause is starting the server with a reduced `--max-model-len`
(or equivalent flag) — AgentX MVP requires the model's default. Restart the
server without overriding the max length and try again. The exact overflow
count and total response count are in the same metadata block, so you can
see how close you were to the threshold.

**"scenario `'inferencex-agentx-mvp'` requires loader=any of …"**
The AgentX MVP scenario is defined against the public
`semianalysisai/cc-traces-weka-042026` corpus, replayed via either the
HuggingFace loader (`semianalysis_cc_traces_weka`, selected by
`--public-dataset`) or the explicit local file-based loader (`weka_trace`,
selected by `--custom-dataset-type weka_trace --input-file <dir>` of the
same JSON traces). Pass one of:

- `--public-dataset semianalysis_cc_traces_weka` (zero-setup; HF download), or
- `--custom-dataset-type weka_trace --input-file <local-trace-dir>` (offline;
  the dir must contain the same Weka trace JSON files). `--input-file` alone
  does NOT auto-detect weka trace directories — you have to pass the explicit
  `--custom-dataset-type weka_trace`.

If you're trying to replay a *different* corpus under this scenario, that's
not a supported submission — but you can pass `--unsafe-override` to run
anyway; the result will be marked `submission_valid=false`.

**"scenario requires `cache_bust.target=system_prefix`; got `<other>`"**
You explicitly passed `--cache-bust <other>` (e.g. `system_suffix` or `none`)
alongside `--scenario`, and AIPerf refuses to silently override an explicit
user choice. If you didn't pass `--cache-bust` at all, the validator
auto-injects `system_prefix` and you'll never see this error. If you
genuinely need a different cache-bust target for an ablation
study, pass `--unsafe-override` and accept the
`submission_valid: false` stamp.

**"scenario `'inferencex-agentx-mvp'` forbids client-side input truncation; `--synthesis-max-isl` …"**
You passed `--synthesis-max-isl <N>`, which drops any trace whose recorded
input length exceeds `N` tokens. Under the scenario that's forbidden because
it changes the replayed workload (a different subset of the corpus, with the
hardest prompts removed). Either drop the flag (and let the server handle
its own context-length errors, which the scenario tracks via the
`context_overflow_rate_exceeded` reason), or pass `--unsafe-override` and
accept `submission_valid: false`.

**"My run finished but I can't find `submission_valid` anywhere"**
You probably ran without `--scenario`. The validity stamp is gated on the
scenario flag — re-run with `--scenario inferencex-agentx-mvp` and look in
the per-run `profile_export.json` (under `metadata`). If you also passed
`--num-profile-runs >= 2`, it'll also appear in the aggregate file at
`profile_export_aiperf_aggregate.json` in the aggregate subdirectory.

**Run is slower than I expected**
Two common causes. First, the warmup phase replays one full conversation
prefix per trajectory before profiling starts; with deep histories that's
a meaningful chunk of wall time on its own. Second, subagent spawns mean
in-flight count exceeds `--concurrency` at peaks; if your server is
concurrency-limited, that adds queueing. Drop `--concurrency` or raise the
server's limit.

**Results vary run-to-run on the same server**
Two runs with different `--random-seed` values will land on different
trajectories and different `k_i`s, so some variation is expected. To
reproduce exactly, capture the seed AIPerf logged at startup and pass it
back via `--random-seed`. Note that AgentX MVP doesn't lock generation
temperature, so server-side sampling stochasticity also contributes;
average over enough requests for percentiles to stabilize.

---

## See Also

- [Weka Agentic Coding Traces](weka-trace.md) — the underlying trace format
  and SPAWN/JOIN subagent mechanics.
- [Timing Modes Reference](../benchmark-modes/timing-modes-reference.md) —
  where `agentic_replay` fits among the other AIPerf timing modes.
- [Warmup Phase tutorial](warmup.md) — the generic AIPerf warmup mechanism
  (the agentic-replay warmup is a specialization of this).
- [Input Sequence Length (ISL) Tokenization](../reference/isl-tokenization.md) —
  how `--isl`, `--apply-chat-template`, and the locked `--cache-bust system_prefix`
  marker interact in the reported metric.
- [ISL Budget Compensation Derivation](../reference/isl-budget-compensation.md) —
  the math behind chat-template + marker overhead compensation.
- [CLI Options Reference](../cli-options.md) — the auto-generated reference
  for `--scenario`, `--unsafe-override`, `--inter-turn-delay-cap-seconds`,
  and every other flag.
