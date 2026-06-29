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
sessions* — long conversations with KV-cache reuse and inter-turn think time.
Sessions come from the public **WEKA agentic-coding trace corpus** captured by
Callan Fox ([kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)),
which records real Claude Code sessions byte-for-byte. AgentX MVP runs against
the current **with-subagents** corpus, where parent coding sessions can spawn
helper conversations that rejoin the parent before it resumes; see
[the Weka tutorial](weka-trace.md) for the source format and the SPAWN/JOIN
mapping.

AgentX MVP is essentially a *recipe* on top of those traces: a fixed set of
replay rules so two different teams running on two different servers produce
results you can actually compare. Things like "long request-start idle gaps are
compressed to 10 seconds", "the server must be allowed to generate full
responses (no early stop)", "warm up the cache before measuring", and so on.

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
(`semianalysisai/cc-traces-weka-062126`, public, no auth) — no
manual clone required. HF caches it locally so re-runs are near-instant.

Then:

```bash
uv run aiperf profile \
    --scenario inferencex-agentx-mvp \
    --url localhost:8000 \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --max-context-length 128_000 \
    --endpoint-type chat \
    --streaming \
    --use-server-token-count \
    --public-dataset semianalysis_cc_traces_weka_with_subagents \
    --concurrency 32 \
    --ui simple
```

That's the whole thing. A few notes:

- **`--scenario inferencex-agentx-mvp`** is the only flag that's specific to
  this benchmark. Everything else is normal AIPerf.
- `--model` is whatever you're actually serving — you don't have to match
  the model names baked into the trace corpus. The example uses a single
  model, so AIPerf rewrites every trace request's `model` field to
  `deepseek-ai/DeepSeek-V4-Pro`. With multiple `--model` values, the trace's
  "main" model maps to the first `--model` and other distinct trace models map
  to the rest in first-appearance order. See
  [Per-Trace Model Rewriting](weka-trace.md#per-trace-model-rewriting) in the
  Weka tutorial for the full behavior.
- **`--max-context-length 128_000`** drops traces whose peak input length exceeds
  128k tokens before replay. This should match the maximum context your server
  is configured to accept.
- **`--benchmark-duration`** defaults to 1800 seconds (30 minutes) under the
  scenario when you don't pass it. You can set it explicitly; 900 seconds
  (15 minutes) is the minimum AgentX MVP allows and AIPerf will reject
  anything shorter.
- **`--concurrency`** is up to you and reflects the load you want to sustain,
  but it must be a single integer under `--scenario`; comma-list sweeps are
  rejected. 32 is a reasonable starting point.
- **`--streaming`** is not forced by the scenario — pass it yourself for chat
  endpoints. The WEKA traces were captured against streaming responses, so
  streaming replay matches the recorded request shape.
- **`--num-profile-runs 3`** is optional but recommended for final
  confidence-interval reporting. The `submission_valid` field is stamped on
  every run with `--scenario` set (single-run `profile_export.json` and, when
  `--num-profile-runs >= 2`, the aggregate file). With a single run you still
  get the validity stamp on the per-run file; multi-run adds the aggregate. See
  [Reading the Result](#reading-the-result-submission_valid) below.
- **`--num-dataset-entries`** is optional for Weka corpus loaders. When omitted,
  the loader automatically loads the full corpus (233 traces for the pinned
  with-subagents corpus) and logs `Loading all 233 traces` at INFO. Pass an
  explicit value only if you intentionally want a subset — the loader will log
  `Loading N/233 traces` so you can see the cap in effect.

You don't need to pass `--ignore-trace-delays`,
`--trace-idle-gap-cap-seconds=10`, `--fixed-schedule`, or anything related to
warmup. The scenario sets those for you. The idle-gap cap compresses recorded
request-start gaps over 10 seconds within each trace; it is not a
think-time-only mode or a blanket clamp on every individual parent-turn delay.
If you *do* pass a locked option with the wrong value, AIPerf will tell you up
front rather than silently producing an invalid result.

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
| `timing_mode` is `agentic_replay` | Use the multi-turn agentic-replay scheduler (locked in by the scenario; not a user-selectable flag) | This is the scheduling discipline AgentX MVP requires (warmup → steady-state, sampler-driven trace recycle, per-session-tree concurrency, trace idle-gap compression). |
| `extra_inputs.ignore_eos = true` | Server is told to ignore its end-of-stream token and generate the full requested length | Without this, models stop early and you measure their decision to stop, not the server. |
| `--ignore-trace-delays` is off | Trace-derived delays are preserved, with long idle gaps capped by the trace idle-gap rule below | The whole point of replay is to preserve the agent's pacing without letting coffee-break gaps dominate steady-state. |
| `--trace-idle-gap-cap-seconds = 10` | Gaps between recorded request starts over 10s are compressed to 10s per trace | Real coding sessions have long idle gaps; capping request-start gaps preserves relative subagent overlap better than clamping each parent turn delay independently. |
| `--cache-bust first_turn_prefix` | Inject a unique per-conversation marker at the start of the first user turn for every play | Without this, every time a trace is recycled the server's prefix cache would warm up further on identical content, and steady-state cache-hit rates would inflate the longer the run goes. The marker forces every recycled play of a trace to have a fresh prompt prefix. Auto-injected when you don't pass `--cache-bust` yourself. |
| Loader is `semianalysis_cc_traces_weka_062126`, `weka_trace`, or constrained `weka_hf` | The dataset is the public `semianalysisai/cc-traces-weka-062126` HF dataset (via `--public-dataset semianalysis_cc_traces_weka_062126`), a local compatible Weka-format corpus replayed via `--custom-dataset-type weka_trace --input-file <dir>` (the file-based `weka_trace` loader; under the scenario, pass the explicit type so the scenario validator sees `detected_loader=weka_trace` before dataset auto-detection runs), or `--hf-weka-dataset semianalysisai/cc-traces-weka-062126` (which auto-selects the generic `weka_hf` loader). These paths produce byte-identical conversations when given the same source rows — see [the Weka tutorial](weka-trace.md#file-based-vs-huggingface-which-to-use). | The benchmark is defined against exact, hash-verifiable corpora so submissions are reproducible. |
| `--benchmark-duration ≥ 900` (defaults to 1800 when unset) | The run lasts at least 15 minutes; omitted, it runs for 30 | Steady-state needs time to stabilize; short runs are noise. |
| No client-side input truncation | `--synthesis-max-isl` is rejected (it drops traces whose input length exceeds the cap, falsifying the workload) | Truncating prompts on the client side would falsify the workload. |
| `--random-seed` is set | If you didn't pass one, AIPerf picks a strong random one and logs it | Reproducibility — every replayed result can be regenerated. |

If you forgot to pass `ignore_eos`, `--cache-bust`, or `--random-seed`, AIPerf injects the locked value and tells you at INFO log level. The same goes for `--trace-idle-gap-cap-seconds` and `--benchmark-duration` (1800s default) when you didn't set them explicitly. If you did pass one of these explicitly with a value that conflicts with the scenario, AIPerf errors with all the violations listed at once — you don't have to fix them one at a time.

---

## Reading the Result: `submission_valid`

When you use `--scenario`, AIPerf stamps a submission-validity flag onto every
JSON output for the run. The per-run `profile_export.json` carries it under
its `metadata` block, and when you also pass `--num-profile-runs >= 2` the
aggregate file (`aggregate/profile_export_aiperf_aggregate.json` under your
artifact directory) carries it too:

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
  - `"run_cancelled"` — the run was cancelled early (Ctrl+C). AIPerf still
    writes the export files with whatever partial metrics it collected, but
    a cancelled run is never a valid submission.
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

Here's the picture. You set `--concurrency 100`. The scheduler builds 100
active trajectory lanes. It uses distinct conversations when enough usable
traces exist; if the usable pool is smaller than the requested concurrency,
it wrap-fills the remaining lanes by cycling through the usable traces with
deterministic per-lane start positions. For each lane, it samples a random
"starting turn" `k_i` somewhere between 0% and 100% of that conversation's
turns (the default `--trajectory-start-min-ratio` / `--trajectory-start-max-ratio`
window, clamped to leave at least one profile turn after warmup). Then, in the warmup phase, it dispatches the warmup turn(s) per lane:
turn `k_i` for simple (non-subagent) trajectories, with the full prefix history
(turns 0 through `k_i-1`) attached as message context. Lanes with live subagent
branches at `k_i` may dispatch one warmup credit per ready branch.

The point is that the server's prefix cache fills with a realistic mix of
multi-turn coding contexts before any measurement starts. When the profiling
phase begins, every trajectory resumes from `k_i + 1` — and the server's cache
already holds the prefix.

The `k_i` values are deterministic given the random seed: same dataset + same
seed = same trajectories + same start points + same recycle order, on any
machine. That's why the scenario insists on a seed.

AIPerf aborts the run as soon as **any** warmup request fails terminally
(after retries) — it does not wait for the rest of the warmup to drain. A
single terminal failure means PROFILING would start on a degraded trajectory
pool, so AIPerf cancels in-flight warmup immediately, logs the failing trace
at `WARNING` ("aborting run early"), shuts down cleanly as a cancelled run,
and exits **non-zero** so CI/automation sees a failure rather than an empty
"success". The philosophy is "don't quietly start metrics on a degraded
warmup". Slow-but-healthy warmups are *not* aborted: the warmup grace period
defaults to no limit, so a warmup that is merely slow runs to completion. If
the warmup is taking longer than you expect, that's a signal worth
investigating in the server logs.

#### Optional cache-pressure warmup

Set `--agentic-cache-warmup-duration SECONDS` to add a sustained cache-pressure
stage after the snapshot warmup. AIPerf continues the same live trajectory
trees for that duration with recorded idle delays removed and every request
limited to one output token. When the duration expires, it stops issuing new
requests, drains requests already on the wire, snapshots each live root,
subagent, and unresolved join, and starts profiling from that exact state.

These requests remain part of `warmup`, so they are excluded from exported
request metrics. The option is disabled by default.

### Profiling Phase: Replay, Recycle, Idle-Gap Compression

After warmup, the profiling phase opens. Now you're measuring. Each trajectory
keeps replaying its conversation from turn `k_i + 1` onward, honoring the
trace's recorded request-start schedule after applying the 10-second idle-gap
compression rule. When a recorded gap between consecutive request starts in the
same trace exceeds 10 seconds, the later request and everything after it are
shifted earlier so that idle gap becomes 10 seconds while local subagent overlap
is preserved.

Concurrency here is **per session tree**: each lane holds one slot for a whole
tree — the root conversation plus every subagent it spawns (children,
subchildren, background `::fa:` flat-async streams, `::aux:` sidecars). A lane
recycles only once its **entire tree drains** — the root has sent its last turn
*and* every subagent has finished — not merely when the root's final turn is
acknowledged. So a background subagent that outlives its root keeps the lane's
slot, and exactly `--concurrency` trees stay live at all times (never more — a
new root can't start until a tree fully drains; never less — rootless/gated
lanes hold the slot too). The shared tree id (`root_correlation_id`) is written
to every record in `profile_export.jsonl`, so `aiperf analyze swim-lane` groups
each tree under one lane and renders exactly `--concurrency` slots.

When a tree drains, the lane recycles by drawing the next root from the **dataset
sampler** (the same sampler that built the initial trajectories, honoring the
dataset's `sampling_strategy`). As long as the corpus is larger than the
trajectory count, a sequential/shuffle sampler plays every trace at least once
before replaying any trace.

A few wrinkles worth knowing:

- **Recycled traces start at turn 0**, not at a random `k_i`. The "start
  somewhere in the middle" rule applies only to the initial trajectories — the
  intent is to spread the *initial* state across the conversation length, not
  to keep injecting mid-conversation jumps forever.
- **Each play of a trace gets a fresh cache-bust marker.** When a trace ID is
  recycled (or first dispatched as a trajectory), AIPerf prepends a unique
  short tag like `[rid:8a3f2c1b9e7d]\n\n` to the first user turn — one
  injection per play, shared across all turns of that play. The tag is
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
  prompt prefix. Locked to `first_turn_prefix` under the scenario.
- **Warmup and profiling share the marker for a given play.** The digest is
  intentionally phase-agnostic: a trajectory's warmup turn `k_i` and its
  first profiling turn `k_i+1` carry the *same* `[rid:…]`. That's how the
  KV-cache prefix work done during warmup transfers into measurement
  instead of being thrown away. (If `phase` were folded into the digest,
  warmup would prime a prefix the profiling phase never sees.)
- **Concurrency can exceed your distinct usable trajectories.** AIPerf first
  builds the distinct usable trajectory pool (skipping traces too short to
  split into a warmup + profiling turn), then wrap-fills extra lanes by
  cycling through that pool until it reaches `--concurrency`. Reused lanes keep
  deterministic start selection and recycle behavior, so the requested
  concurrency is honoured even when multiple lanes share a source trace. A
  too-small usable pool is only invalid when it is empty.
- **Profiling ends** when `--benchmark-duration` elapses. Anything in flight
  finishes during a cooldown window and is included in the metrics; nothing
  *new* starts after the duration ends.

### Subagents

The AgentX MVP corpus is the current **with-subagents** variant. Parent turns
can spawn one or more helper conversations, and the parent's next anchored turn
waits on the corresponding `SPAWN_JOIN` prerequisite before resuming. `--concurrency`
controls the number of live session **trees** — one slot per root *plus all the
subagents it spawns*, held until the whole tree drains. Those helper conversations
run alongside their parent, so the instantaneous in-flight *request* count can rise
above `--concurrency` at a fan-out point; the concurrency setting is a cap on
concurrent trees, not on every parent-plus-subagent request. Because the slot is
held for the entire tree, a background subagent that outlives its parent does not
free the lane early — exactly `--concurrency` trees stay live throughout.

AIPerf constructs this topology from `WekaSubagentEntry` blocks in the trace:
subagents with preceding and following parent anchors become SPAWN/JOIN
branches, background subagents with no following anchor do not block the parent,
and adjacent subagents sharing the same anchors collapse into one multi-child
branch.

Within each subagent entry, nested hash-id LCP chain detection splits the
inner requests into per-context-chain children (the subagent's own thread
`::sa:<agent_id>` plus `:c000`, `:c001`, ... siblings for one-shot disjoint
calls, parallel forks, and flattened worker threads — on the 060826 corpus the
615 subagent entries expand to ~3.1k chain children). Each chain child
dispatches at its **recorded offset** from the spawn rather than bursting when
the parent turn completes, so the in-subagent request schedule replays on the
recorded timeline. The parent's SPAWN_JOIN waits on all of a subagent's chain
children. For the format details, detection rules, and SPAWN/JOIN mechanics,
see the [Weka Traces tutorial](weka-trace.md).

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
is `semianalysis_cc_traces_weka_062126` or `weka_hf` with
`--hf-weka-dataset semianalysisai/cc-traces-weka-062126`.

**Run aborts early: `aborting run early (broadcasting ProfileCancelCommand)` / warmup failure**
Your inference server rejected a warmup request after AIPerf's normal retry
budget. AgentX MVP aborts on the **first** terminal warmup failure rather than
producing a partial result: the run cancels immediately and exits non-zero
(the failing trace is named in the `WARNING` log; the legacy
`TrajectoryWarmupFailedError` is still raised as a backstop if the live abort
path is unavailable). Check the server logs — common causes are an
authentication or model-name mismatch (e.g. `--model` doesn't match what the
server is serving), the server's `max-model-len` set lower than the trace's
requested context, or the server simply not running.

**Run completes but `submission_valid: false` with `"context_overflow_rate_exceeded"`**
Your server is rejecting prompts as too long for more than 1% of requests.
The most common cause is starting the server with a reduced `--max-model-len`
(or equivalent flag) — AgentX MVP requires the model's default. Restart the
server without overriding the max length and try again. The exact overflow
count appears in the `metrics` block as `context_overflow_count` (unit: requests),
so you can see how close you were to the threshold.

**"scenario `'inferencex-agentx-mvp'` requires loader=any of …"**
The AgentX MVP scenario is defined against the public
`semianalysisai/cc-traces-weka-062126` corpus, replayed via the
pinned HuggingFace loader (`semianalysis_cc_traces_weka_062126`,
selected by `--public-dataset`), the explicit local file-based loader
(`weka_trace`, selected by `--custom-dataset-type weka_trace --input-file
<dir>`), or the generic HuggingFace Weka loader constrained to the same repo.
Pass one of:

- `--public-dataset semianalysis_cc_traces_weka_062126` (zero-setup; HF download),
- `--custom-dataset-type weka_trace --input-file <local-trace-dir>` (offline;
  the dir must contain compatible Weka trace JSON files), or
- `--hf-weka-dataset semianalysisai/cc-traces-weka-062126` (auto-selects `weka_hf`).

`--input-file` alone can auto-detect Weka trace directories in ordinary custom-dataset runs, but it does not populate the scenario validator's `detected_loader` field before AgentX MVP locks are checked. Under `--scenario inferencex-agentx-mvp`, pass the explicit `--custom-dataset-type weka_trace`. If you're trying to replay a *different* corpus under this scenario, that's not a supported submission — but you can pass `--unsafe-override` to run anyway; the result will be marked `submission_valid=false`.

**"scenario requires `cache_bust.target=first_turn_prefix`; got `<other>`"**
You explicitly passed `--cache-bust <other>` (e.g. `system_suffix` or `none`)
alongside `--scenario`, and AIPerf refuses to silently override an explicit
user choice. If you didn't pass `--cache-bust` at all, the validator
auto-injects `first_turn_prefix` and you'll never see this error. If you
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
`aggregate/profile_export_aiperf_aggregate.json` under the artifact directory.

**Run is slower than I expected**
The warmup phase replays one full conversation prefix per trajectory before
profiling starts; with deep histories in real coding traces, that's a meaningful
chunk of wall time on its own. Subagent fan-out can also create additional
in-flight requests beyond the active parent trajectory count. If your server is
concurrency-limited and you raised `--concurrency` above its limit, you'll also
see queueing. Drop `--concurrency` or raise the server's limit.

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
  how `--isl`, `--apply-chat-template`, and the locked `--cache-bust first_turn_prefix`
  marker interact in the reported metric.
- [ISL Budget Compensation Derivation](../reference/isl-budget-compensation.md) —
  the math behind chat-template + marker overhead compensation.
- [CLI Options Reference](../cli-options.md) — the auto-generated reference
  for `--scenario`, `--unsafe-override`, `--inter-turn-delay-cap-seconds`,
  and every other flag.
