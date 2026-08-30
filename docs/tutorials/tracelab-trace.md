<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile with TraceLab Agentic Coding Traces

AIPerf can replay [TraceLab](https://github.com/uw-syfi/TraceLab), a public corpus of real agentic coding sessions captured from 52 developers using Claude Code and Codex against production endpoints. It records 665,453 LLM rounds across 8,058 sessions and 743,819 tool calls, licensed CC BY 4.0.

TraceLab is a good fit when you want a KV-cache workload whose shape was set by people rather than by a generator: long sessions, deep prefix reuse, agent fan-out, and multi-minute human pauses between turns.

This guide covers fetching the corpus and replaying it, and is explicit about which parts of a session are reproduced and which are reconstructed.

---

## Start a vLLM Server

Launch a vLLM server with a chat model:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

Verify the server is ready:
```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Download the Corpus

The corpus is published as a GitHub release asset. It is about 100 MB compressed and roughly 650 MB inflated, and AIPerf reads it gzipped, so there is no need to decompress it:

```bash
curl -L -o syfi_coding_trace.jsonl.gz \
  https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz
```

For a first run, take a slice rather than the whole corpus:

```bash
zcat syfi_coding_trace.jsonl.gz | head -20000 | gzip > tracelab_slice.jsonl.gz
```

---

## Profile

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --tokenizer Qwen/Qwen3-0.6B \
  --url localhost:8000 \
  --endpoint-type chat \
  --custom-dataset-type tracelab \
  --input-file tracelab_slice.jsonl.gz \
  --num-dataset-entries 50 \
  --concurrency 4 \
  --no-fixed-schedule \
  --inter-turn-delay-cap-seconds 5
```

`--no-fixed-schedule` keeps the run in concurrency mode so `--inter-turn-delay-cap-seconds` takes effect. Without it, AIPerf auto-promotes TraceLab (which carries per-round timestamps) to fixed-schedule replay, where the recorded timestamps are authoritative and the cap has no influence.

`--inter-turn-delay-cap-seconds` is worth setting deliberately. The corpus records real human pauses, some of them tens of minutes long, and AIPerf will sleep for them faithfully. Capping makes a run finish in reasonable time; not capping is the more honest replay. Decide which you want rather than inheriting the default.

---

## Fixed-Schedule Replay

Every round carries an absolute timestamp, so the corpus can be replayed on its
recorded arrival schedule:

```bash
aiperf profile ... --custom-dataset-type tracelab --fixed-schedule \
  --fixed-schedule-end-offset 300000
```

Bound the window. A fixed-schedule run without one replays the sessions at their
real offsets, and a single TraceLab session can span hours of wall clock, so the
run takes that long by design. `--fixed-schedule-end-offset` is in milliseconds.

---

## TraceLab Format

Each JSONL line is one LLM round, not one session. AIPerf groups rounds into sessions by `session_id` and replays each session as a conversation. The fields it reads:

- `session_id`: session this round belongs to, formatted `<provider>:<uuid>`
- `round_index`: the corpus's own round ordering
- `input_tokens_total`, `prefix_tokens`, `newly_append_tokens`: engine-reported input decomposition, where `input_tokens_total = prefix_tokens + newly_append_tokens`
- `output_tokens`, `reasoning_output_tokens`: generated token counts
- `model`, `provider`, `user`, `project`: identity fields
- `timing_events[]`: per-event absolute ISO-8601 timestamps (`user_message`, `tool_result`, `text`, `reasoning`, `tool_call`)
- `tools[]`: tool calls with `tool_name`, `emitted_at`, `result_at`, `tool_wall_latency_ms`
- `first_input_event_type`: what fed this round

The corpus contains no prompt or response text: the longest string in it is a 503-character command skeleton. AIPerf synthesizes prompt content to the recorded token counts, which is the same approach every hash-id trace format uses.

---

## What Is Reproduced, and What Is Reconstructed

Replay fidelity differs by property, and the difference matters when you interpret results.

Reproduced from the record:

- input and output token counts per round
- absolute submission times, so inter-turn think time is the real human gap
- which model served each round
- whether a round ended by requesting a tool

Reconstructed, because the corpus does not record it:

- **KV-cache block IDs.** TraceLab carries no content hashes at all. AIPerf mints per-session virtual block IDs from the `prefix_tokens` / `newly_append_tokens` split, which reproduces the recorded prefix reuse to block granularity and degrades correctly when an agent compacts its context. Cross-session sharing is not recoverable: the corpus has no cross-session content identity.
- **Subagent parent/child links.** A subagent round is filed under its own top-level `session_id` with no parent reference of any kind. AIPerf recovers the link by timing containment: a session sharing the same user and project whose entire span falls inside a spawning tool call's window is taken to be that call's subagent, and the tightest enclosing window wins. This is a containment rate, not an accuracy: the corpus has no ground truth for the join. Sessions that match no window are replayed as independent traces rather than attached to a guessed parent.
- **`api_time`.** Derived from the span between a round's last input event and its last model-emitted event. The corpus records no server-reported latency and no TTFT, so this is a proxy with nothing to check it against.

Not recoverable at all: the subagent type requested at spawn, which lived in tool-call arguments the corpus strips.

---

## Block Size

Because the block IDs are synthesized rather than recorded, the block size is a real knob here. It defaults to 64 tokens; set `--isl-block-size` to match the engine under test:

```bash
aiperf profile ... --custom-dataset-type tracelab --isl-block-size 128
```

Changing it changes the synthesized ID chain, and therefore the prefix-cache hit rate the replay presents to the server.

---

## Subagent Join Controls

The join runs by default. Three environment variables control it:

| Variable | Default | Effect |
| --- | --- | --- |
| `AIPERF_DATASET_TRACELAB_SUBAGENT_JOIN` | `true` | Set `false` to replay every recorded session as an independent flat trace, which is the shape the corpus literally records |
| `AIPERF_DATASET_TRACELAB_CODEX_SUBAGENT_JOIN` | `true` | Set `false` to keep only the precise blocking-tool-call join |
| `AIPERF_DATASET_TRACELAB_MIN_SPAWN_MS` | `10000` | Minimum spawning-call wall latency treated as a subagent round-trip |

The Codex control exists because the two providers are not equally joinable. Claude Code spawns a subagent through a single blocking tool call, so that call's window is exactly the child's lifetime. Codex uses an async `spawn_agent` / `wait_agent` / `close_agent` lifecycle whose handles live in the stripped tool arguments, so a spawn cannot be paired to its own wait and only a coarse session-level window is available. A Codex session that fans out several agents collapses them into one window.

---

## Related Tutorials

- [Replay Weka Agentic Coding Traces](weka-trace.md)
- [Profile with Bailian Traces](bailian-trace.md)
- [Trace Benchmarking](../benchmark-modes/trace-replay.md)
