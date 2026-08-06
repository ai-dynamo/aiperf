<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# InferenceX AgentX MVP Benchmark

> **Status: Work-in-progress MVP.** This is the first AIPerf implementation of the
> SemiAnalysis InferenceX AgentX-MVP benchmark. The scenario, the rules it locks,
> and the output fields described here may change as the spec stabilizes.

This page walks you through running the **AgentX MVP** benchmark in AIPerf. It's
aimed at someone who hasn't worked with the scenario before — after a short
orientation, you'll get a copy-pasteable command, then explanations of what it
does and why.

> **Want the under-the-hood mechanics instead of a how-to?** See
> [SemiAnalysis AgentX: How the Benchmark Works (FAQ)](../benchmark-modes/semianalysis-agentx-faq.md)
> — a serving-engineer's guide to what load it puts on your server, how it reproduces
> KV-cache structure, what `t*`/warmup/cache-busting do, and what makes a run valid.

---

## What Is AgentX MVP?

AgentX MVP is a multi-turn, agentic-coding benchmark proposed by SemiAnalysis as
part of their InferenceX effort. The idea: instead of measuring an inference
server with synthetic 1-turn prompts, measure it with realistic *coding-agent
sessions* — long conversations with KV-cache reuse and inter-turn think time.
Sessions come from the public **Weka agentic-coding trace corpus** captured by
Callan Fox ([kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)),
which records real Claude Code sessions byte-for-byte. AgentX MVP runs against
the current **with-subagents** corpus, where parent coding sessions can spawn
helper conversations that rejoin the parent before it resumes; see
[the Weka tutorial](weka-trace.md) for the source format and the SPAWN/JOIN
mapping.

AgentX MVP is essentially a *recipe* on top of those traces: a fixed set of
replay rules so two different teams running on two different servers produce
results you can compare. Things like "preserve each trace's original request
timing", "never leave the whole system idle for more than 10 seconds", "the
server must be allowed to generate full responses (no early stop)", "warm up
the cache before measuring", and so on.

AIPerf bundles every one of those rules into a single CLI flag:
`--scenario inferencex-agentx-mvp`. When you pass that flag, AIPerf locks the
relevant settings and rejects conflicting flags. It also stamps a
`submission_valid` field onto the JSON outputs (per-run and aggregate) as a
quick sanity check that the run followed the locked rules.

---

## Quick Start

You'll need:

- An **OpenAI-compatible inference server** running and reachable. `--url`
  takes a bare `host:port` (`http://` is prepended automatically) or a full
  scheme-prefixed URL; the endpoint path (e.g. `/v1/chat/completions`) is
  appended automatically based on `--endpoint-type`. If the server requires
  authentication, add `--api-key <key>` (sent as a `Bearer` token).
- AIPerf installed: `pip install aiperf` (or run it ad hoc with
  `uvx aiperf`).
- The model's **HuggingFace tokenizer** resolvable from `--model`: the Weka
  loaders rebuild every prompt through it. For a gated repo, export
  `HF_TOKEN`; if `--model` isn't a resolvable HF repo name (a local server
  alias, a private build), pass `--tokenizer <hf-repo-or-local-path>`
  explicitly. See
  [Tokenizer Auto-Detection](../reference/tokenizer-auto-detection.md).

The trace corpus is fetched automatically from HuggingFace
(`semianalysisai/cc-traces-weka-062126`, public, no auth) — no
manual clone required; HF caches it locally.
The `semianalysis_cc_traces_weka_with_subagents` value in the command below
is a *rolling* alias tracking the current with-subagents corpus (presently
the `062126` drop); the date-pinned `semianalysis_cc_traces_weka_062126`
locks the corpus for reproducibility. Prefer the date-pinned alias for any
run you intend to compare or submit — the rolling alias advances when a new
corpus drops, and two runs on different drops are not comparable. Each corpus
also has a `_256k` variant (e.g.
`semianalysis_cc_traces_weka_with_subagents_256k`) that pre-drops individual
requests whose input + output exceeds 256k tokens — pick it when your
server's context window is ~256k (see the
[AgentX FAQ §3](../benchmark-modes/semianalysis-agentx-faq.md#3-how-realistic-are-the-prompts-and-token-counts)).

Then:

```bash
aiperf profile \
    --scenario inferencex-agentx-mvp \
    --url localhost:8000 \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --max-context-length 128000 \
    --endpoint-type chat \
    --public-dataset semianalysis_cc_traces_weka_with_subagents \
    --concurrency 32 \
    --use-server-token-count \
    --streaming \
    --extra-inputs ignore_eos:true \
    --cache-bust first_turn_prefix \
    --system-idle-gap-cap-seconds 10 \
    --trajectory-start-min-ratio 0.0 \
    --trajectory-start-max-ratio 1.0 \
    --benchmark-duration 1800 \
    --random-seed 20260707 \
    --ui simple
```

The command mixes three kinds of flags — knowing which is which tells you
what you must set, what you may tune, and what you shouldn't touch:

**Flags you choose** — your server, model, and load:

- **`--scenario inferencex-agentx-mvp`** is the only flag that's specific to
  this benchmark. Everything else is normal AIPerf.
- `--model` is whatever you're serving — you don't have to match
  the model names baked into the trace corpus; AIPerf rewrites every trace
  request's `model` field. See
  [Per-Trace Model Rewriting](weka-trace.md#per-trace-model-rewriting) in the
  Weka tutorial for how multiple `--model` values map.
- **`--max-context-length 128000`** drops traces whose peak **input + output**
  — the prompt plus that turn's requested `max_tokens` — exceeds 128k tokens
  before replay. Filtering happens before `--num-dataset-entries` is applied,
  so the cap keeps the first N *eligible* traces.
  Set it to the maximum context your server accepts — i.e. the
  model's native maximum, since AgentX MVP expects the server to run at its
  default max length. Because it mirrors the server's real capacity, the run
  replays only traces the server can serve — which is why the scenario allows
  it while rejecting the arbitrary client-side cap `--synthesis-max-isl`.
  The flag is optional and the scenario doesn't check it: omit it and no
  client-side filtering happens, so over-length traces fail at the server
  and count toward the 1% context-overflow threshold instead.
- **`--concurrency`** sets how many session trees stay live throughout the
  run, i.e. the sustained load. It must be a single integer under
  `--scenario`; comma-list sweeps are rejected. When concurrency exceeds the
  number of distinct loaded traces, the trace pool wraps — the scenario locks
  `--cache-bust first_turn_prefix`, and an active cache-bust target satisfies
  the wrap opt-in. Without cache-bust, wrapping still requires
  `--allow-dataset-wrap`.
- `--url`, `--endpoint-type chat`, `--use-server-token-count`, and `--ui`
  round out the group ([`--use-server-token-count`](#tokenization-options---apply-chat-template-and---use-server-token-count)
  is explained below).

**Flags the scenario hard-locks** — `--streaming`,
`--extra-inputs ignore_eos:true`,
`--cache-bust first_turn_prefix`, and `--system-idle-gap-cap-seconds 10`. You
can omit all of them (AIPerf fills in exactly these values under
`--scenario`), but writing them out makes the command document what runs. If
you pass one of them with a *conflicting* value, AIPerf errors up front
rather than silently producing an invalid result. The scenario forbids
`--inter-turn-delay-cap-seconds`; `--trace-idle-gap-cap-seconds` remains an
optional CLI control and is unset by default.

**Flags the scenario only defaults** — auto-filled when omitted, but an
explicit value is honored *silently*, with no error and no change to the
`submission_valid` stamp:

- **`--trajectory-start-min-ratio 0.0` / `--trajectory-start-max-ratio 1.0`**
  set the window where each lane's starting instant `t*` is sampled. The
  scenario never validates these — override them and the run still stamps
  `submission_valid: true` while replaying a materially different workload.
  Leave them at `0.0`/`1.0` for any run you intend to compare.
- **`--benchmark-duration 1800`** (30 minutes) is the scenario default;
  900 seconds (15 minutes) is the enforced minimum, and AIPerf rejects
  anything shorter. Longer values are accepted without complaint.
- **`--random-seed`**: omitted, AIPerf picks a fresh random seed and logs it;
  passing your own (any integer) makes the run reproducible up front.
  Pinning a seed also keys the reconstructed-dataset disk cache: unseeded
  scenario runs draw a fresh seed each time and repay the full multi-minute
  corpus reconstruction on every run (see the
  [AgentX FAQ §8](../benchmark-modes/semianalysis-agentx-faq.md#8-running-the-benchmark-and-why-the-first-run-is-slow)).

**Optional extras** (not in the command above):

- **`--num-profile-runs N`** repeats the benchmark N times and adds an
  aggregate file with confidence intervals across the runs. See
  [Reading the Result](#reading-the-result-submission_valid) below for where
  the `submission_valid` stamp lands.
- The full corpus (393 traces) loads by default; `--num-dataset-entries N`
  keeps the first N *eligible* traces after `--max-context-length` filtering
  (filter-then-cap). Reducing the corpus changes the replayed workload and is
  *not* caught by the scenario locks — the run still stamps
  `submission_valid: true` — so use it only for smoke tests, never for runs
  you intend to compare against other AgentX MVP results. If your smoke
  concurrency exceeds N, the locked cache-bust target lets the pool wrap; with
  cache-bust off you would need `--allow-dataset-wrap` or a lower concurrency.

You don't need to touch the scheduling or warmup knobs: the scenario picks
the agentic-replay scheduler and auto-fills the values shown above. Don't
pass `--fixed-schedule`, `--request-rate`, or `--ignore-trace-delays` — they
conflict with the locked scheduling mode and the run will error.

### Tokenization options: `--apply-chat-template` and `--use-server-token-count`

**Optional: `--apply-chat-template`.** With the flag off (the default),
the reported ISL is the bare-text encode of the wire payload. With it on,
the record processor re-tokenizes each request's wire payload through the
tokenizer's own `apply_chat_template`, so the reported ISL counts the full
wire-token total — chat-template wrapping plus the cache-bust marker — and
is directly comparable to a server's `usage.prompt_tokens`. See
[Input Sequence Length (ISL) Tokenization](../reference/isl-tokenization.md)
for the full picture.

**`--use-server-token-count` (included in the Quick Start command; fixes
OSL mismatch).** Without it, AIPerf computes output sequence length (OSL)
by re-tokenizing the server's response with the model's local tokenizer.
If that tokenizer disagrees with the server's own — a different revision,
different BPE merges, a different chat template — the reported OSL drifts
from the server's actual emitted token count, and the per-run console
shows an "Output Sequence Length Mismatch Warning" panel even though
`ignore_eos=true` is locked and the server really did emit `max_tokens`.
With the flag, AIPerf trusts the server's `usage.completion_tokens` (and
`usage.prompt_tokens`) and the mismatch goes away.

### Benchmarking through a router (multiple replicas)

Make the routing conversation-aware, or cross-replica scatter will destroy
the prefix-cache reuse this benchmark exists to measure. Server side: SGLang
Model Gateway `--policy cache_aware` (or `--policy manual`) / Dynamo
`--router-mode kv`. Client side, AIPerf keeps a stable per-conversation ID
and exposes it as an additive session-affinity header via an environment
variable: `AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=1` (SGLang
`manual`), `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1` (Dynamo
session affinity), or `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1` (any
other router). Details and full launch commands: the
[AgentX FAQ §9](../benchmark-modes/semianalysis-agentx-faq.md#9-multi-replica-serving-conversation-aware-routing-sglang-dynamo).

### What you should see

A run moves through four externally visible stages; knowing them keeps you
from killing a healthy run (a Ctrl+C stamps it `run_cancelled` / invalid):

1. **Dataset configuration.** On the first run this includes the corpus
   download from HuggingFace plus a CPU-heavy reconstruction of every trace
   into a tokenized, cache-structured dataset — expect several minutes of
   apparent silence, and see
   [Troubleshooting](#troubleshooting) if it exceeds the default
   configuration timeout. Later runs with the same corpus, settings, and a
   pinned `--random-seed` restore it from the on-disk cache in seconds.
2. **Warmup.** Each lane replays one deep-prefix turn to prime the server's
   KV cache. With deep histories in real coding traces this is a meaningful
   chunk of wall time on its own.
3. **Profiling** for `--benchmark-duration` (1800 s by default) — with
   `--ui simple`, per-phase progress and request counts tick along as
   traffic flows.
4. **Drain and export.** In-flight requests finish during a grace-period
   drain, then the console prints the metrics tables and the exact artifact
   paths (under `./artifacts/` unless you set `--artifact-dir`).

End to end, a cold first run therefore takes noticeably longer than the 30
minutes the duration flag suggests — reconstruction, warmup, and drain all
sit on top of it.

---

## What `--scenario inferencex-agentx-mvp` Locks for You

When you pass the scenario flag, AIPerf checks (and in some cases sets) the
following settings before the run starts. If any of them conflict with what you
asked for, the run errors immediately with a clear message naming the offending
flag.

| Locked setting | What it means | Why it matters |
|---|---|---|
| `timing_mode` is `agentic_replay` | Use the multi-turn agentic-replay scheduler (locked in by the scenario; not a user-selectable flag) | This is the scheduling discipline AgentX MVP requires (warmup → steady-state — the [profiling phase](#profiling-phase-faithful-replay-recycle-global-idle-guard) — with sampler-driven trace recycle and per-session-tree concurrency). |
| `extra_inputs.ignore_eos = true` | Server is told to ignore its end-of-stream token and generate the full requested length | Without this, models stop early and you measure their decision to stop, not the server. |
| `--streaming` is on | Responses stream token-by-token (auto-enabled when unset; explicit `--no-streaming` errors) | The per-token latency metrics (TTFT, ITL) are core to this benchmark and need streaming responses. |
| Replay delays are end-to-start (always on, no flag) | Each turn's replay delay is the recorded idle gap from the previous response's *end* to the next request's *start*, not the start-to-start delta. This is unconditional for weka trace replay — there is no toggle. | Replay dispatches each turn after the previous one completes, so start-to-start deltas would double-count the previous request's server time, making every session drift later turn by turn and overstating how many sessions overlap at once. |
| `--ignore-trace-delays` is off | Trace-derived delays are preserved | The replay retains the captured agent pacing and KV-cache reuse intervals. |
| Per-trace idle-gap cap is optional | `--trace-idle-gap-cap-seconds` is unset by default and accepts an explicit CLI value | Setting it bounds observed runtime idle across the root and every descendant stream without rewriting dataset timestamps or bypassing spawn/join dependencies. |
| Per-turn cap is forbidden | `--inter-turn-delay-cap-seconds` must remain unset | Independently clamping parent and subagent delays can distort their relative timing. |
| `--system-idle-gap-cap-seconds = 10` | When no request is active or ready, all pending replay timers shift uniformly so the next request arrives within 10 seconds | The benchmark avoids measuring long periods with no server work while preserving request order and relative spacing across every pending trajectory. |
| `--cache-bust first_turn_prefix` | A unique per-conversation marker is injected at the start of the first user turn for every play (each dispatch of a trace, initial or recycled) | Without this, every time a trace is recycled the server's prefix cache would warm up further on identical content, and steady-state cache-hit rates would inflate the longer the run goes. The marker gives every recycled play a fresh prompt prefix. |
| Loader is a pinned Weka with-subagents corpus | The dataset must be a with-subagents `--public-dataset` alias or `--hf-weka-dataset semianalysisai/cc-traces-weka-062126` (`weka_hf`). A local `weka_trace` directory is format-compatible but unpinned — the run refuses unless you pass `--unsafe-override` (which stamps `submission_valid: false`). The [Troubleshooting](#troubleshooting) entry for this lock lists the exact flag forms. | Submission validity requires a known public corpus identity; arbitrary local dirs are not hash-verifiable. |
| `--benchmark-duration >= 900` (defaults to 1800 when unset) | The run lasts at least 15 minutes; if omitted, it runs for 30 minutes | Steady-state needs time to stabilize; short runs are noise. |
| No client-side input truncation | `--synthesis-max-isl` — the file-based synthesis ISL filter — is rejected because it drops traces whose input length exceeds the cap (the `--public-dataset` corpus has no synthesis filter, so there the flag has no effect either way) | Truncating prompts on the client side would falsify the workload. |
| `--random-seed` is set | If you didn't pass one, AIPerf picks a strong random one and logs it | Reproducibility — every replayed result can be regenerated. |

If you forget to pass the `ignore_eos` extra-input, `--streaming`,
`--cache-bust`, or `--random-seed`, AIPerf
injects the locked value and tells you at INFO log level. The same goes for
`--system-idle-gap-cap-seconds` and `--benchmark-duration` (1800s default)
when you don't set them explicitly. If you pass one of these explicitly with
a value that conflicts with the scenario, AIPerf errors with all the
violations listed at once — you don't have to fix them one at a time.

The trajectory-start ratios (`--trajectory-start-min-ratio` /
`--trajectory-start-max-ratio`) are the soft exception: the scenario
auto-fills `0.0`/`1.0` but never validates them, so an explicit different
value is honored silently — no error, no change to `submission_valid` — even
though it materially changes the workload. Treat them as fixed for any run
you intend to compare (see the flag groups in [Quick Start](#quick-start)).

---

## Reading the Result: `submission_valid`

All output files land under the artifact directory — `./artifacts/` by
default, relative to where you ran `aiperf`; override it with
`--artifact-dir`. AIPerf prints the exact output locations at the end of the
run.

When you use `--scenario`, AIPerf stamps a submission-validity flag onto the
AIPerf metrics JSON outputs for the run. The per-run `profile_export_aiperf.json` carries it
under its `metadata` block, and when you pass `--num-profile-runs >= 2` (and
at least two runs complete successfully) the aggregate file
(`aggregate/profile_export_aiperf_aggregate.json` under your artifact
directory) carries it too:

```json
{
  "metadata": {
    "scenario": "inferencex-agentx-mvp",
    "submission_valid": true,
    ...
  },
  "request_throughput": { ... },
  "request_latency": { ... },
  ...
}
```

(In the per-run file, metric results are top-level fields alongside
`metadata`; the aggregate file nests them under a `metrics` key instead.)

Three possible states for `submission_valid`:

- **`submission_valid: true`** — the run honored every scenario rule AIPerf
  checks, and finished cleanly.
- **`submission_valid: false`** — something went wrong (or you forced
  something). The same metadata block also contains `submission_invalid_reasons`,
  a list of short tags explaining why. Possible values:
  - `"unsafe_override"` — you passed `--unsafe-override` along with one or
    more rule-breaking flags. See [`--unsafe-override`](#--unsafe-override) below.
  - `"context_overflow_rate_exceeded"` — more than 1% of the responses came
    back with a context-overflow error from the server, which means the server
    is rejecting prompts the benchmark requires it to handle. This usually
    means the server was started with a reduced max model length;
    AgentX MVP requires the model's default.
  - `"run_cancelled"` — the run was cancelled early (Ctrl+C). On a single
    Ctrl+C AIPerf cancels gracefully and still writes the export files with
    whatever partial metrics it collected (a second Ctrl+C force-quits
    without writing); a cancelled run is always stamped invalid.
  - `"scenario_reresolve_failed"` — multi-run aggregate export could not
    re-apply the scenario lock on the base config (import/env breakage).
    Fail-closed: the aggregate stamps `submission_valid: false` rather than
    assuming the lock held.
- **Field absent** — you ran without `--scenario`. The submission-validity
  machinery is gated on the scenario flag.

If you see `submission_valid: false`, look at `submission_invalid_reasons` and
the AIPerf log. The reasons map one-to-one to either a scenario rule you broke
or a runtime threshold you crossed.

Treat the stamp as a guideline, not a certification. It reflects only the
checks AIPerf can perform itself — the flag locks applied at startup plus the
runtime thresholds it tracks. `true` means AIPerf detected no rule violation;
it can't attest to anything outside its view (a server started with a reduced
context window, for example, only shows up indirectly through the
context-overflow rate). Whether two results are genuinely comparable still
depends on the full setup on both sides, and the MVP rule set itself is
evolving (see the status note at the top).

---

## How a Run Executes

### Warmup Phase: Trajectories and `k_i`

Before AIPerf measures anything, it runs a **warmup phase** that primes the
server's KV cache. This isn't the generic AIPerf warmup — it's a
trajectory-based warmup specific to the agentic-replay scheduler.

Here's the picture. You set `--concurrency 100`. The scheduler builds 100
active trajectory lanes, drawing traces from the dataset sampler. Filling more
lanes than distinct loaded roots requires `--allow-dataset-wrap` or an active
`--cache-bust` target (which the scenario locks on); without either, an
undersized pool is capped with a warning rather than silently wrap-filled. When wrapping is enabled, the same
trace can back multiple lanes, each with a deterministic per-lane start
position. For each lane, it samples a random starting instant `t*` somewhere
between 0% and 100% of that trace's recorded duration (the
`--trajectory-start-min-ratio` / `--trajectory-start-max-ratio` window the
scenario auto-fills, widening the generic 25%–75% default) and
derives the "starting turn" `k_i` from it, always leaving at least one
profiling turn after warmup. Then it dispatches the warmup turn(s) per lane:
turn `k_i` for simple (non-subagent) trajectories, with the full prefix
history (turns 0 through `k_i-1`) attached as message context. Lanes whose
subagent branches are already live at `k_i` dispatch one warmup request per
active stream.

The point is that the server's prefix cache fills with a realistic mix of
multi-turn coding contexts before any measurement starts. When the profiling
phase begins, every trajectory resumes from `k_i + 1` — and the server's cache
already holds the prefix.

The `k_i` values are deterministic given the random seed: same dataset + same
seed = same trajectories + same start points + same recycle order, on any
machine. That's why the scenario insists on a seed.

AIPerf aborts the run as soon as any **root-conversation** warmup request
fails terminally — each warmup request gets exactly one attempt; there is no
retry — and it does not wait for the rest of the warmup to drain. A single
terminal failure means profiling would start on a degraded trajectory pool,
so AIPerf cancels in-flight warmup immediately, logs the failing trace at
`WARNING` ("aborting run early"), and shuts down as a cancelled run stamped
`submission_valid: false` (reason `run_cancelled`). A *subagent* stream's
warmup failure does not trigger the abort — only root (depth-0)
conversations gate it. Slow-but-healthy warmups are also *not* aborted: the
warmup grace period (`--agentic-warmup-grace-period`; under `--scenario`,
`--warmup-grace-period` aliases onto it when the dedicated flag is unset —
see the [Warmup Phase tutorial](warmup.md)) has no limit by default, so a
warmup that is merely slow runs to completion.

#### Optional cache-pressure warmup

Set `--agentic-cache-warmup-duration SECONDS` to add a sustained
cache-pressure stage after the initial trajectory warmup described above.
AIPerf continues the same live session trees for that duration with recorded
idle delays removed and every request limited to one output token. When the
duration expires, it stops issuing new requests, drains requests already on
the wire, snapshots each live root, subagent, and unresolved join, and starts
profiling from that exact state. Each stream's full next-turn delay is carried
across the phase boundary; time spent waiting for the global warmup drain does
not consume it.

For repeatable warmup depth, use
`--warmup-requests-per-lane REQUESTS` instead. After the mandatory snapshot
primers complete, each concurrency lane sends exactly that many additional
warmup wire requests. For example, `--concurrency 16` with
`--warmup-requests-per-lane 10` adds 160 cache-pressure requests after the
primers, with a strict 10-additional-request quota on every lane.

At handoff, timestamped root, subagent, and background streams are restored to
their next request's position on one shared per-trajectory dataset clock. The
earliest pending request starts profiling immediately; the remaining streams
retain their flattened cross-stream order and relative start spacing. For a
timestamp-less dataset, AIPerf falls back to each stream's recorded
end-to-start delay.

`--agentic-cache-warmup-duration` and `--warmup-requests-per-lane` are mutually
exclusive: choose a time-bounded warmup or a deterministic request-bounded
warmup.

These requests remain part of warmup, so they are excluded from exported
request metrics.

### Profiling Phase: Faithful Replay, Recycle, Global Idle Guard

After warmup, the profiling phase opens. Now you're measuring. Each trajectory
keeps replaying its conversation from turn `k_i + 1` onward, honoring the
original recorded inter-turn gaps as end-to-start delays counted from the
moment the previous turn completes. At the phase boundary, AIPerf subtracts
one global minimum from the pending first-request offsets: the earliest
request starts immediately, while every other trajectory keeps the same
relative spacing. With
`--trace-idle-gap-cap-seconds S`, a per-trajectory watchdog covers initial
future work and restarts when the last in-flight request across the root and
all descendant streams completes.
If that tree remains idle for `S` seconds, AIPerf uniformly advances only its
pending timers; dataset timestamps, relative timer spacing, request order, and
spawn/join gates remain unchanged. If every request completes while future
requests remain scheduled more than 10 seconds away, AIPerf
shifts all pending request timers earlier by the same amount. This bounds true
system-idle time without changing request order or the relative spacing among
pending trajectories. A scheduled turn may become retained by a cross-stream
replay barrier instead of reaching the wire; AIPerf re-evaluates the guard
after that scheduler task drains so a nearby blocked turn cannot hide a much
later dispatchable timer. The phase-end log reports how many global jumps
occurred and how many seconds they skipped.

Concurrency here is **per session tree**: each lane holds one slot for a
whole tree — the root conversation plus every subagent worker stream it
spawns. A lane recycles only once its **entire tree drains** — the root has
sent its last turn *and* every subagent has finished — not merely when the
root's final turn is acknowledged. Exactly `--concurrency` trees stay live at
all times. Never more: a new root can't start until a tree fully drains.
Never fewer: a lane whose root has finished but whose subagents are still
draining keeps its slot. The shared tree id (`root_correlation_id`) is
written to every record in `profile_export.jsonl` (in the artifact
directory), so `aiperf analyze swim-lane` groups each tree under one lane and
renders exactly `--concurrency` slots as a per-lane timeline.

When a tree drains, the lane recycles by drawing the next root from the **dataset
sampler** (the same sampler that built the initial trajectories, honoring the
dataset's `sampling_strategy`). As long as the corpus is larger than the
trajectory count, a sequential/shuffle sampler plays every trace at least
once before repeating any.

A few wrinkles worth knowing:

- **Recycled traces start at turn 0**, not at a random `k_i`. The "start
  somewhere in the middle" rule applies only to the initial trajectories — the
  intent is to spread the *initial* state across the conversation length, not
  to keep injecting mid-conversation jumps forever.
- **Each play of a trace gets a fresh cache-bust marker.** When a trace is
  first dispatched (or recycled), AIPerf prepends a unique short tag like
  `[rid:8a3f2c1b9e7d]\n\n` to the first user turn. The marker is generated
  once per play, reused on every turn of that play, and replaced when the
  trace recycles into a new play — that's what keeps the server's prefix
  cache from warming progressively on identical content as the run goes on.
  Markers also differ across runs, since each run's auto-generated benchmark
  ID feeds the marker digest. See
  [AgentX FAQ §4](../benchmark-modes/semianalysis-agentx-faq.md#4-the-kv-cache-story-warmup-t-and-cache-busting)
  for marker placement and the warmup-to-profiling handoff.
- **Warmup and profiling share the marker for a given play.** A trajectory's
  warmup turn `k_i` and its first profiling turn `k_i+1` carry the *same*
  `[rid:…]` — that's how the KV-cache prefix work done during warmup
  transfers into measurement instead of being thrown away.
- **Concurrency may exceed the number of usable traces only with wrap enabled.**
  Traces too short to split into a warmup + profiling turn are skipped (the
  pool is capped with a warning when it cannot fill after skips). Filling
  more lanes than distinct loaded roots requires `--allow-dataset-wrap` or an
  active `--cache-bust` target. With wrap on, one source trace
  can back several lanes (each keeping its own start position and recycle
  behavior). An *empty* pool after filtering is still an error.
- **Profiling ends** when `--benchmark-duration` elapses. Anything in flight
  finishes during a grace-period drain and is included in the metrics; nothing
  *new* starts after the duration ends.

### Subagents

The AgentX MVP corpus is the current **with-subagents** variant. Parent turns
can spawn one or more helper conversations, and the parent's next anchored
turn waits on the corresponding `SPAWN_JOIN` prerequisite before resuming. As
covered in the
[Profiling Phase](#profiling-phase-faithful-replay-recycle-global-idle-guard)
section, `--concurrency` counts live session **trees**, with each slot held
until the whole tree drains. The new wrinkle here is the *request* count:
helper conversations run alongside their parent, so the instantaneous number
of in-flight requests can rise above `--concurrency` at a fan-out point — the
cap is on concurrent trees, not on individual in-flight requests.

AIPerf constructs this topology from `WekaSubagentEntry` blocks in the trace:
subagents with preceding and following parent anchors become SPAWN/JOIN
branches, background subagents with no following anchor do not block the
parent, and adjacent subagents sharing the same anchors collapse into one
multi-child branch.

Within each subagent entry, AIPerf additionally detects nested context
chains, so one recorded subagent may replay as several parallel child
streams (helper sidecars, worker groups, solo workers). Two behaviors
matter for the load: each child stream dispatches at its **recorded offset**
from the spawn rather than bursting when the parent turn completes, so the
in-subagent request schedule replays on the recorded timeline; and the
parent's SPAWN_JOIN waits on *all* of a subagent's child streams. For the
detection rules and the `::sa:`/`::aux:`/`::wg:`/`::fa:` stream-naming
scheme, see the [Weka Traces tutorial](weka-trace.md).

---

## `--unsafe-override`

Sometimes you intentionally want to break a scenario rule — to study the
sensitivity of one variable, to run a 1-minute smoke test instead of a
15-minute proper run, to try a different cache-bust target for an ablation.
For that:

```bash
aiperf profile \
    --scenario inferencex-agentx-mvp \
    --unsafe-override \
    --benchmark-duration 60 \
    ...
```

What `--unsafe-override` does:

- **Converts most scenario rule violations from an error into a warning.** The
  run starts.
- **Does not bypass a missing weka loader.** If you omit `--public-dataset` /
  `--input-file`, the CLI defaults to synthetic; AgentX still refuses to start
  (continuing would look like a corpus load while producing 1-turn sessions
  that empty the trajectory pool). Pass an allowed weka loader explicitly.
- **Stamps `submission_valid: false`** in every JSON output (per-run and, when
  `--num-profile-runs >= 2`, the aggregate file), with `"unsafe_override"` in
  `submission_invalid_reasons` — but only when at least one rule was actually
  broken. Passing the flag without breaking any rule is a no-op.

Once the flag is on and a rule is broken, the stamp stays `false` in that
run's outputs — there's no way to flip it after the fact, so the override is
always visible in the result. The flag is a no-op without `--scenario` (since
there are no scenario rules to override).

Use this for development and ablation studies. Don't use it for anything you
want to compare against other AgentX MVP runs.

---

## Troubleshooting

**`UnknownScenarioError: Unknown scenario 'inferencex-agentx-mvp'. Valid scenarios: …`**
The AIPerf you're running predates this scenario — scenarios ship with the
package and are registered in code, not in a data file you can regenerate.
Upgrade AIPerf (`pip install -U aiperf`) and re-run.

**`EmptyTracePoolError: Loader produced 0 traces; trajectories cannot be built.`**
The HF dataset download or row validation produced no usable traces. Check
your network connectivity to `huggingface.co` and confirm the
`--public-dataset` value is a with-subagents alias (e.g.
`semianalysis_cc_traces_weka_with_subagents` or
`semianalysis_cc_traces_weka_062126`), or `weka_hf` with
`--hf-weka-dataset semianalysisai/cc-traces-weka-062126`. On an offline
machine, replay a local Weka-format trace directory via
`--custom-dataset-type weka_trace --input-file <dir> --unsafe-override`
(offline smoke only — `submission_valid: false`) — see the
[Weka Traces tutorial](weka-trace.md) for how to obtain or capture one, and
note the model tokenizer must also be available offline (pre-populate the HF
cache and set `HF_HUB_OFFLINE=1`, or point `--tokenizer` at a local path).

**`ScenarioLockError` … `got synthetic` / `cannot be bypassed with --unsafe-override`**
You set `--scenario inferencex-agentx-mvp` (and maybe `--unsafe-override` for
a short smoke duration) but omitted a weka dataset source. The CLI then
defaults to synthetic prompts (`session_000000`, 1 turn each). AgentX refuses
that combination even under `--unsafe-override`. Add
`--public-dataset semianalysis_cc_traces_weka_062126` (or another allowed
alias / `--hf-weka-dataset semianalysisai/cc-traces-weka-062126`).

**The tokenizer can't be resolved or downloaded**
The Weka loaders rebuild every prompt through a HuggingFace tokenizer
derived from `--model`, so the dataset build fails if that repo can't be
reached. If the model repo is gated, export `HF_TOKEN`; if `--model` isn't a
resolvable HF repo name (a local server alias, a private build), pass
`--tokenizer <hf-repo-or-local-path>` explicitly. See
[Tokenizer Auto-Detection](../reference/tokenizer-auto-detection.md).

**Run aborts early: `aborting run early (broadcasting ProfileCancelCommand)` / warmup failure**
Your inference server rejected a warmup request. Each warmup request gets
exactly one attempt — there is no retry — and AgentX MVP aborts on the
**first** terminal root-conversation warmup failure rather than producing a
partial result: the run cancels immediately and the failing trace is named
in the `WARNING` log. Check the AIPerf and
server logs — the underlying cause is usually a connection error
(`Connection refused`: wrong `--url` or the server isn't running), an auth
failure (`401`/`403`: pass `--api-key`), a model-name mismatch
(`404` / `model not found`: `--model` doesn't match what the server is
serving), or the server's `max-model-len` set lower than the trace's
requested context.

**Run completes but `submission_valid: false` with `"context_overflow_rate_exceeded"`**
Your server is rejecting prompts as too long for more than 1% of requests.
The most common cause is starting the server with a reduced `--max-model-len`
(or equivalent flag) — AgentX MVP requires the model's default. Restart the
server without overriding the max length and try again. If the model's
*native* window is itself around 256k (MiniMax-class models), the fix is a
matching `_256k` corpus instead — e.g.
`--public-dataset semianalysis_cc_traces_weka_with_subagents_256k`; see the
[AgentX FAQ §3](../benchmark-modes/semianalysis-agentx-faq.md#3-how-realistic-are-the-prompts-and-token-counts)
for why that beats capping client-side. The number of requests that
overflowed appears as the `context_overflow_count` metric (a raw count, not
a rate), top-level in `profile_export_aiperf.json` and under `metrics` in
the aggregate file — divide it by
`request_count + error_request_count + skipped_context_overflow_count`
(the same denominator the aggregate exporter uses for the 1% threshold)
to see how close you were to the limit.

**"scenario `'inferencex-agentx-mvp'` requires loader=any of …" / cannot verify corpus identity for a local weka_trace directory**
The AgentX MVP scenario stamps `submission_valid: true` only for a pinned
public SemiAnalysis Weka corpus (`semianalysisai/cc-traces-weka-062126`),
replayed via with-subagents `--public-dataset` aliases (rolling or
date-pinned; the legacy no-subagents aliases are rejected) or the generic
HuggingFace Weka loader (`weka_hf`) constrained to that repo. Pass one of:

- `--public-dataset semianalysis_cc_traces_weka_with_subagents` (zero-setup;
  rolling alias for the current corpus),
- `--public-dataset semianalysis_cc_traces_weka_062126` (zero-setup;
  date-pinned), or
- `--hf-weka-dataset semianalysisai/cc-traces-weka-062126` (auto-selects `weka_hf`).

A local `--custom-dataset-type weka_trace --input-file <dir>` (or a bare
`--input-file <weka-dir>` that auto-detects as `weka_trace`) is accepted for
offline smoke tests only with `--unsafe-override` — the result is marked
`submission_valid: false`, because AIPerf cannot fingerprint an arbitrary
local directory as the public corpus. The same override path applies if you
intentionally replay a *different* Weka-format corpus under this scenario.

**"scenario `'inferencex-agentx-mvp'` requires `cache_bust.target=first_turn_prefix`; got `<other>`"**
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
the per-run `profile_export_aiperf.json` (under `metadata`). If you passed
`--num-profile-runs >= 2`, it appears in the aggregate file at
`aggregate/profile_export_aiperf_aggregate.json` under the artifact directory
too.

**Configuration times out before any traffic starts**
The first run on a machine reconstructs the whole corpus into a tokenized,
cache-structured dataset before sending anything, and for the with-subagents
corpora that reconstruction can exceed the default 300-second configuration
timeout. Raise `AIPERF_DATASET_CONFIGURATION_TIMEOUT` and
`AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` (which must be at least as large)
to ~1800 seconds for a cold run. The cost is one-time: the reconstructed
dataset lands in an on-disk cache (default `~/.cache/aiperf/dataset_mmap`)
whose key includes the random seed, so pin `--random-seed` and later runs
restore it in seconds. See
[AgentX FAQ §8](../benchmark-modes/semianalysis-agentx-faq.md#8-running-the-benchmark-and-why-the-first-run-is-slow).

**Run is slower than I expected**
On a cold first run, the corpus reconstruction described in the previous
entry happens before any traffic — budget several extra minutes for it.
After that, the warmup phase replays a full conversation prefix for every
trajectory stream before profiling starts; with deep histories in real
coding traces, that's a meaningful chunk of wall time on its own. Subagent
fan-out can also create additional in-flight requests beyond the active
parent trajectory count. If your server is concurrency-limited and you
raised `--concurrency` above its limit, you'll also see queueing. Drop
`--concurrency` or raise the server's limit.

**Results vary run-to-run on the same server**
Two runs with different `--random-seed` values will land on different
trajectories and different `k_i`s, so some variation is expected. To
reproduce exactly, capture the seed AIPerf logged at startup and pass it
back via `--random-seed`. Server-side sampling stochasticity (generation
temperature) also contributes; more profile runs (`--num-profile-runs`) give
the percentiles more data to stabilize on.

---

## See Also

- [Weka Agentic Coding Traces](weka-trace.md) — the underlying trace format
  and SPAWN/JOIN subagent mechanics.
- [Metrics Reference](../metrics-reference.md) — definitions and formulas for
  TTFT, ITL, throughput, and every other reported metric.
- [Effective vs Active Metrics](../reference/effective-vs-active-metrics.md) —
  the time-weighted EFFECTIVE and ACTIVE console tables printed at the end of
  a run.
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
  for `--scenario`, `--unsafe-override`, `--system-idle-gap-cap-seconds`,
  and every other flag.
