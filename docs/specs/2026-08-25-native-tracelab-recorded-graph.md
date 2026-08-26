# Native TraceLab Recorded-Graph Input

## Status

Design for origin/main tracker #44, exact upstream commit
`082a51827eb9f755aacb93122cafe0383cf99f6e`.

## Problem

TraceLab publishes one JSONL row per real Claude Code or Codex LLM round. The
native runtime can execute WEKA-shaped recorded graphs, but cannot acquire or
convert TraceLab rows and does not route the `tracelab` format to the graph
workload. Treating the corpus as a conventional multi-turn dataset would lose
recorded request timing, cache-prefix identity and recovered subagent
concurrency.

## Architecture

Add `graph::recorded::tracelab` beside the native WEKA compiler. Its public
entry point accepts `RecordedTraceInputConfig`, acquires a local plain/gzip
JSONL source (plus existing inline/byte/remote row sources), groups rows in
first-session order, converts root sessions to strict WEKA-shaped `Value`
documents, and invokes `compile_weka_trace_input` against those in-memory
documents. It rewrites only `GraphInputMetadata.format` to `tracelab`.

This keeps one implementation of cache-content synthesis, strict schema
validation, selection, Graph-IR lowering and subagent execution. Conversion
uses no temporary files and rereads no caller source.

## Conversion contract

- A TraceLab row is detected by the distinctive required-key subset
  `session_id`, `round_index`, `input_tokens_total`, `prefix_tokens`,
  `newly_append_tokens`, and `timing_events`.
- ISO-8601 timestamps accept `Z`, explicit offsets, and naive values interpreted
  as UTC. Submission is the latest `user_message`/`tool_result`; request API
  time extends to the latest `text`/`reasoning`/`tool_call` and floors at zero.
- Rows without a usable session ID are ignored. Sessions with no dated round are
  omitted. Empty output after conversion is an error.
- Within a session, rounds sort by submission time, then `round_index`, then
  stable source order.
- `hash_ids` contains `input_tokens_total / block_size` whole blocks. A round
  reuses at most `min(prefix_tokens / block_size, previous_blocks,
  current_blocks)` leading identities and mints positive trace-local identities
  for the rest. Parent and nested children share one minter.
- Input and output lengths floor at one. Reasoning output tokens add to visible
  output length. `first_input_event_type == tool_result` selects the WEKA
  `tool_result` input type. Any `tool_call` output event selects `tool_use`;
  otherwise the stop reason is `end_turn`.
- Request timestamps are seconds relative to the root's first submission.
  Think time is the non-negative gap from the previous request completion.
- Trace/agent IDs replace unsafe runs with `_` and truncate to 150 bytes without
  splitting UTF-8.

## Subagent recovery

Default policy enables both joins and uses a minimum Claude spawn duration of
10,000 ms. Authored graph options may set `subagent_join`,
`codex_subagent_join`, and `min_spawn_ms`; `block_size` defaults to 64.

Claude windows come from blocking `Agent` or `Task` tool calls carrying valid
`emitted_at`, `result_at`, and sufficient `tool_wall_latency_ms`. Codex uses one
session window from the earliest `spawn_agent.emitted_at` to the latest
`wait_agent.result_at` and can be disabled independently.

A child must differ from the parent, share the first row's `user` and `project`,
and have its complete event/tool-call span within the candidate window. The
tightest window wins, then earliest start and parent ID provide deterministic
ties. Only children whose parent is a root are nested; grandchildren remain
standalone. A nested marker is placed after the last parent request at or before
the spawn. A child with no dated requests or no anchor remains standalone.

## Validation and configuration

`tracelab` joins the one authoritative graph-format inventory and graph adapter
resolver. All cellular and artifact-shipping format predicates derive from or
are updated with that inventory so cross-host runs transfer the original source
and partition whole traces.

The CLI/config resolver treats `--custom-dataset-type tracelab` as the authored
format and routes `--isl-block-size` to `dataset.options.block_size`. The
compiler rejects zero block size, non-boolean join options, negative minimum
spawn duration, unknown options, directories, missing files, malformed JSON,
invalid UTF-8, corrupt gzip and duplicate safe trace IDs with contextual errors.

Graph timing is already authoritative inside each trace. Native does not
translate TraceLab to the scheduled fixed-schedule workload; graph execution is
the native equivalent and `--ignore-trace-delays` remains the explicit way to
run structure without recorded pacing.

## Tests

Unit tests port the applicable conversion/source behaviors in three focused
behavior suites. Runtime integration loads real plain and gzip fixtures through
the public compiler and built-in graph adapter, validating Graph-IR node counts,
cache identity and subagent edges. CLI resolution proves workload routing and
block-size projection. A native-binary dry-run test proves the original source
is accepted outside test-only converter APIs.

## Scope

No Python loader, plugin entry, mmap cache key, generated Python CLI docs, or
intermediate converted file is added. Native graph acquisition is run-owned and
already immutable/cached at its execution boundary. The ancestry merge records
the exact upstream commit without importing its Python tree.
