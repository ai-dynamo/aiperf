<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex CLI and Claude Code import into recorded-agent Graph-IR

## Purpose

This record specifies the first native port of the Python Codex CLI and Claude
Code session adapters. The public Graph-IR format remains
`agent_recording`. A source-normalization layer imports either session-log
format and lowers it to the same `GraphInputBundle`, replay metadata, driver,
sampling, placement, and reporting boundaries used by recorded-agent replay.

The importers do **not** pretend that Codex or Claude Code JSONL is a
Mini-SWE-Agent recording. Those logs do not contain the exact provider request,
tool schemas, request usage, executable command contract, or complete timing
needed by the Mini-SWE schema. Source-specific parsing therefore ends at a new
provider-neutral session IR; a dedicated session lowerer produces canonical
recorded-agent Graph-IR. The strict Mini-SWE wire DTOs, discovery, and lowering
remain separate.

## Source anchors

This design is grounded in:

- Python Codex parsing at
  `ajc/dag-v3:src/aiperf/dataset/loader/graph/adapters/codex.py:38-396`.
- Python Claude Code parsing at
  `ajc/dag-v3:src/aiperf/dataset/loader/graph/adapters/claude_code.py:47-474`.
- The shared Python session IR and union-graph builder at
  `ajc/dag-v3:src/aiperf/dataset/loader/graph/adapters/_agent_session.py:52-438`.
- The existing native `agent_recording` DTOs, discovery, and lowering at
  `rust/runtime/src/graph/recorded/agent_recording/schema.rs:11-198`,
  `rust/runtime/src/graph/recorded/agent_recording/discovery.rs:23-685`, and
  `rust/runtime/src/graph/recorded/agent_recording/lowering.rs:30-684`.
- The direct graph-input adapter boundary at
  `rust/runtime/src/engine/graph_input.rs:193-221,439-692`.

## Built

The native `agent_recording` input accepts strict Mini-SWE recordings as before
and native Codex CLI and Claude Code JSONL session imports. `source_format`
selects `auto`, `mini_swe_agent`, `codex`, or `claude_code`; `auto` recognizes
one JSONL file while directory imports require an explicit provider format.

Codex directories recurse over sorted session JSONL files. Claude directories
select sorted main sessions and their direct `subagents/agent-*.jsonl` files
when `include_subagents` is enabled. Discovery canonicalizes every selected
path and rejects symlinks and root escapes. Cellular controllers revalidate and
stream-copy the exact selected files through no-follow descriptors into private
scratch before parsing or serving them. Cross-host cells rebuild only that
scratch-backed manifest; co-located cells receive the same scratch paths. A
caller source replacement after snapshotting therefore cannot alter a cell's
input, and session histories are never retained wholesale in controller memory.

The importers normalize provider histories into non-executable session IR,
lower it to canonical recorded-agent Graph-IR, and preserve observed tool
relationships without creating executable tool nodes. Codex imports replay
through chat-compatible endpoints; Claude imports retain Messages content and
require a Messages-compatible endpoint. Imported request bodies receive the
prepared context tokenizer for exact local input-token accounting. Deterministic
mock-server E2E covers Codex and Claude request history, ISL/OSL, streaming
TTFT/ITL, and privacy-safe diagnostics.

## Historical design context

## Problem

AIPerf replays strict Mini-SWE-Agent recordings and imports native Codex CLI
and Claude Code session JSONL through the same `agent_recording` graph input.
The Python adapters provided discovery and
parsing rules, including Codex recursive session discovery, Claude Code
sidechain filtering, parallel tool correlation, and subagent discovery. Their
output model is not the correct native target:

- Python constructs one shared union graph across sessions, with
  `assistant_K` LLM nodes, conditional tool branches, and replay nodes
  (`_agent_session.py:177-237,265-315`). Native recorded-agent lowering instead
  produces an independently owned linear graph program per source trace
  (`rust/runtime/src/graph/recorded/agent_recording/lowering.rs:176-208,267-395`).
- Python retains only the first user input in trace initial state and tool
  results in replay outputs (`_agent_session.py:373-423`). Later plain user
  messages and observed assistant history are not represented as exact request
  snapshots.
- Python's Codex parser attaches pending function calls and results to the
  subsequent assistant text event (`codex.py:172-191`). In the source event
  order, the function call is itself model output, its result is tool output,
  and the following assistant message is another model completion.
- Native Mini-SWE lowering preserves exact authored `provider_request.messages`
  and `provider_request.tools` bytes (`lowering.rs:406-519`). Codex and Claude
  session logs contain observed messages and tool uses, not that provider
  request envelope.

The port must therefore preserve source evidence without claiming wire parity,
must remain safe for untrusted tool payloads, and must integrate with existing
recorded-agent execution rather than adding two parallel workload formats.

## Goals

The implementation shall:

1. Accept one Codex or Claude Code JSONL file and the documented directory
   layouts.
2. Deterministically discover, parse, validate, digest, and normalize every
   selected session exactly once.
3. Produce one canonical `agent_recording` `GraphTraceProgram` per session,
   including stable replay identity and explicit fidelity annotations.
4. Reconstruct the best-supported request history before each observed model
   completion without inventing unavailable request fields.
5. Represent observed tool time without executing arbitrary imported tool
   arguments.
6. Preserve existing local, thread-per-core, and cellular ownership rules.
7. Prevent directory imports and cross-host shipping from reading or serving
   unrelated session files.
8. Report malformed or ambiguous inputs with source file, line, record type,
   and safe identifiers, without echoing prompt or tool payload content.

## Non-goals

The first delivery does not:

- Execute Codex or Claude Code tools, shell commands, MCP calls, subagents, or
  permission flows.
- Recreate either CLI as a live agent or use live replies to choose subsequent
  actions.
- Claim byte-identical provider request replay.
- Translate Claude's provider-native tool blocks into OpenAI tool-call wire
  format.
- Restore private or encrypted reasoning.
- Infer missing tool schemas, sampling settings, token usage, or generation
  caps.
- Port the Python union-graph topology, trace tags, partial anonymization,
  min/max-turn filtering, or empirical/lognormal shared wait distributions.
- Change the `recorded-agent-default` canonical scenario, its fixture digests,
  or its Mini-SWE/PinchBench/SWE-Bench environment behavior.
- Add a conversion command or write an intermediate normalized file.

## Public configuration and CLI contract

The authored dataset remains a file dataset with
`format: agent_recording`. The strict `RecordedAgentGraphConfig` gains source
selection and the only source-specific control needed by the first delivery:

```yaml
datasets:
  - type: file
    format: agent_recording
    path: /path/to/session-or-project
    sampling: sequential
    graph:
      source_format: codex
      include_subagents: true
```

`source_format` accepts:

| Config value | Meaning |
|---|---|
| `auto` | Preserve Mini-SWE JSON/manifest detection and sniff a single JSONL file. This is the default. |
| `mini_swe_agent` | Require the existing recording, directory, or replay-manifest contract. |
| `codex` | Require Codex CLI JSONL and Codex directory discovery rules. |
| `claude_code` | Require Claude Code JSONL and Claude project/subagent discovery rules. |

`include_subagents` defaults to `true`, applies only to `claude_code`, and is
rejected for the other explicit source formats rather than silently ignored.

The equivalent CLI is:

```text
aiperf profile \
  --input-file /path/to/session-or-project \
  --graph-format agent_recording \
  --graph-recording-source codex
```

The CLI adds:

| CLI | Config v2 | Default |
|---|---|---|
| `--graph-recording-source auto|mini-swe-agent|codex|claude-code` | `dataset.graph.source_format` | `auto` |
| `--graph-include-subagents[=bool]` | `dataset.graph.include_subagents` | `true` |

The fields belong in `RecordedAgentGraphConfig`, whose strict extension point
is `rust/runtime/src/config/model/dataset.rs:343-380`. CLI declarations belong
beside the existing recorded-agent flags at
`rust/cli/src/flags.rs:900-939`, and projection belongs beside the existing
`RecordedAgentGraphConfig` construction at `rust/cli/src/load.rs:212-230`.
YAML already carries the typed `dataset.graph` block at
`rust/cli/src/yaml.rs:1037-1041`.

The following validation is mandatory:

- `dataset.format` remains exactly `agent_recording`; `dataset.graph` is already
  restricted to that format at `rust/runtime/src/config/validate.rs:301-312`.
- Sampling remains `sequential`, options remain empty, and dataset wrap remains
  disabled, matching `rust/runtime/src/engine/graph_input.rs:544-567,684-690`.
- `execute_tools=true` is rejected for `codex` and `claude_code`. Existing tool
  images and tool lifecycle settings are consequently rejected as inapplicable
  for these source formats.
- `scenario: recorded-agent-default` rejects imported sessions before source
  loading. That scenario is bound to the canonical manifest and recording
  digests, not arbitrary session logs.
- `auto` is supported for a single file. A directory requires an explicit
  source format because Codex and Claude have different recursive read sets.

No Python adapter timing controls enter the first public contract. Python
offers factor/default/override/clamp, `observed|max|sum` bundle reduction, and
fixed/empirical/lognormal strategies
(`_agent_session_duration.py:27-38,41-82,85-153`). Native static graph edges
carry one scalar delay (`rust/runtime/src/graph/model.rs:88-101`), not the
shared union-node distribution Python authors. Adding a different timing model
requires a separate design.

## Detection and discovery

### Single files

For `source_format: auto`:

1. Preserve existing Mini-SWE JSON, gzip, and manifest detection from
   `rust/runtime/src/engine/graph_input.rs:858-886`.
2. For `.jsonl`, scan at most 20 non-empty JSON-object records.
3. A Codex match requires a supported top-level `type` and object `payload`,
   following `codex.py:363-396`.
4. A Claude match requires `sessionId` plus either `parentUuid` or a supported
   Claude marker record. A bounded multi-record scan is necessary because
   resumed logs may begin with non-discriminating records
   (`claude_code.py:38-44,428-474`).
5. Fail if neither or more than one source matches. Explicit source selection
   bypasses sniffing but not strict source validation.

### Directories

Codex recursively discovers sorted `*.jsonl` files, as the Python adapter does
at `codex.py:93-99`.

Claude discovers sorted top-level `*.jsonl` main sessions and, when enabled,
sorted `<session>/subagents/agent-*.jsonl` files
(`claude_code.py:107-115,154-161`). Nested subagent files are never rediscovered
as main sessions.

The native implementation must make the discovered read set a first-class
value shared with cellular shipping. Every path is resolved beneath the
canonical selected root, every component is checked with `symlink_metadata`,
and symlinks are rejected. These rules extend the existing root containment
and symlink protections at
`rust/runtime/src/graph/recorded/agent_recording/discovery.rs:140-180,183-227,615-669`.

Cellular import shipping consumes only the exact discovered read set; it never
reuses replay-root traversal for a user's `~/.codex` or `~/.claude` tree.

## Normalization data flow

The load path is:

```text
RecordedAgentRunnerGraphInputAdapter
  -> decode strict dataset.graph configuration
  -> resolve and validate source_format
  -> discover an exact root-contained file set
  -> stream and parse source JSONL
  -> normalize each file into ImportedAgentSession
  -> lower ImportedAgentSession values into GraphInputBundle
  -> attach canonical recorded-agent preparation policy
```

The adapter resolver is intentionally responsible only for format identity and
one strict load, as specified at `rust/runtime/src/engine/graph_input.rs:193-221`.
The `agent_recording` adapter remains registered once in the built-in
composition (`graph_input.rs:244-260,443-458`). Source selection occurs inside
that adapter and does not register `codex` or `claude_code` as new graph-input
formats.

### Internal session IR

The new internal IR is not a public wire contract:

```text
ImportedAgentSession
  session_id: String
  source: Codex | ClaudeCode
  source_path: PathBuf
  source_digest: String
  model: Option<String>
  system_prompt: Option<RawJsonMessage>
  cwd_present: bool
  git_branch_present: bool
  parent: Option<ImportedSubagentParent>
  calls: Vec<ImportedModelCall>
  observed_tool_count: u64
  ignored_record_count: u64

ImportedModelCall
  source_id: String
  request_messages: Vec<RawJsonMessage>
  model: Option<String>
  delay_after_previous_us: Option<f64>
  tool_schema_available: bool
  output_tokens: Option<u64>
```

`RawJsonMessage` retains the normalized message object bytes that the importer
authors. It does not claim to retain bytes of an unavailable original provider
request. Cwd and git branch are retained only as presence facts; their raw
values are neither lowered nor logged.

Parsers maintain an ordered conversation history. Immediately before each
observed assistant completion, they snapshot that history as one
`ImportedModelCall.request_messages`. After the snapshot, the observed
assistant content and correlated tool results extend history for later calls.
This gives every native LLM node a self-contained static request history, which
matches the existing recorded-agent lowerer's use of full source request
snapshots rather than dynamic channel splices
(`rust/runtime/src/graph/recorded/agent_recording/lowering.rs:406-488`).

## Canonical lowering contract

Each imported session becomes one `GraphTraceProgram`:

- `profiling.trace.id` is the stable session ID.
- Nodes are `llm_0` through `llm_N` in inferred source-call order.
- Edges are `START -> llm_0 -> ... -> llm_N -> END`.
- Each node's output channel uses `ChannelType::Messages` and
  `ReducerName::AddMessages`, matching existing recorded-agent lowering at
  `rust/runtime/src/graph/recorded/agent_recording/lowering.rs:328-349`.
- Each `LlmNode.items` is the normalized request-message snapshot interned in
  the segment pool. Role validation remains the same non-empty-role check used
  for strict recordings at `lowering.rs:417-446`.
- `streaming` and `fallback_max_tokens` use the existing adapter settings.
  Because source output usage is unavailable, every imported call uses the
  fallback cap and records target output tokens as zero/unavailable. Existing
  strict lowering also falls back when completion usage is absent or zero at
  `lowering.rs:468-488,628-635`.
- `LlmRequestSpec.tools` is absent. The session logs record tool uses, not the
  tool-definition array sent to the model.
- `LlmRequestSpec.model` is present only when existing
  `use_recorded_model=true` is selected.
- Recorded sampling is unavailable. `use_recorded_sampling=true` is rejected
  for imported sources rather than silently doing nothing.
- No `ToolNode` or trace environment is created. Imported tool payloads are
  observations, not authorized commands.
- A positive observed tool bundle duration becomes
  `delay_after_predecessor_us` on the edge to the next inferred model call.
  Plain assistant-to-assistant wall time is not replayed because it combines
  unknown model latency, user think time, and client overhead.
- `warmup` and `environment` are absent, and the driver is
  `TraceDriverSpec::recorded_replay()`.

The bundle has `GraphInputMetadata.format == "agent_recording"`, one root per
session, an immutable shared segment store, and deduplicated warning facts.
The prepared input retains `random_seed: None`, no wrapping, default `t*`, and
no cache-bust target, exactly as the existing adapter returns at
`rust/runtime/src/engine/graph_input.rs:668-691`.

Replay metadata uses the existing `ReplayTraceMetadata` fields at
`rust/runtime/src/graph/driver.rs:94-119`:

- `manifest_ordinal`: deterministic sorted session ordinal.
- `identity.adapter`: `codex` or `claude_code`.
- `identity.family`: `session` for main sessions and `subagent` for Claude
  subagents.
- `identity.task_id`: stable session/derived subagent ID.
- `source_digest`: BLAKE3 of the exact source JSONL file bytes.
- `target_output_tokens`: zero for every inferred call.
- `expected_llm_node_count`: inferred call count.
- `expected_tool_node_count`: observed completed tool-use count, even though
  tool execution is disabled. Existing Mini-SWE lowering similarly records
  completed source tool count independently of emitted tool nodes at
  `lowering.rs:273-290,382-390`.
- `request_profile_identity`: `recorded-agent:codex` or
  `recorded-agent:claude_code`.
- `comparability_annotations`: non-secret, typed fidelity facts described
  below, plus Claude subagent parent identity where applicable.

The built-in request profile resolver recognizes both adapter names without an
unknown-adapter warning. Neither gains SWE-Bench sampling fields; those are
specific to `swebench` at
`rust/runtime/src/graph/recorded/agent_recording/lowering.rs:114-149`.

## Codex source mapping

Codex uses one JSONL file per session and recursively walks directories
(`ajc/dag-v3:src/aiperf/dataset/loader/graph/adapters/codex.py:3-7,93-99`).

The native mapping is:

| Codex source | Native meaning |
|---|---|
| `session_meta.payload.id` | Required session/trace ID. |
| `session_meta.payload.cwd` | Presence-only private provenance. |
| `session_meta.payload.model` | Recorded model candidate. |
| `session_meta.payload.base_instructions.text` | Initial `system` history message. |
| `session_meta.payload.git.branch` | Presence-only private provenance. |
| `turn_context.payload.model` | Model fallback when session metadata has none. |
| user `message` blocks of type `input_text` or `text` | User history messages in block order. |
| developer `message` | Developer history message; unlike Python, it is not discarded. |
| assistant `message` blocks of type `output_text` or `text` | One assistant completion and subsequent history. |
| consecutive `function_call` records before outputs | One assistant tool-use completion; preserve name, raw arguments, and `call_id` in canonical observed history. |
| `function_call_output` | Tool-result history correlated by `call_id`. |
| `reasoning.summary` | Counted as omitted sensitive reasoning; not lowered. |
| other `event_msg` or response item types | Ignored with a non-secret count unless required correlation is affected. |

The metadata fields and Python fallback rules are implemented at
`codex.py:160-169,244-268`; supported content collapse is at
`codex.py:271-295`; call/result correlation and observed durations are at
`codex.py:204-227,308-335`.

The native call boundary intentionally differs from Python. A function-call
bundle is model output and therefore creates an inferred LLM call before its
tool results. A later assistant text message after those results creates the
next inferred call. Python instead flushes the pending calls and results only
when it encounters that later assistant message (`codex.py:172-191`). The
native rule preserves causal order and shall be asserted by golden fixtures.

Consecutive function calls are one parallel bundle until the first correlated
output or another model-output boundary. Results may arrive in any order but
are appended to history in source record order. Duplicate `call_id` values are
errors. An interrupted session may end with unmatched calls; it remains
loadable, is annotated `tool_results_complete=false`, and receives no invented
result body.

## Claude Code source mapping

Claude Code stores main session files at the project directory root and
subagent files below a session-specific `subagents` directory
(`ajc/dag-v3:src/aiperf/dataset/loader/graph/adapters/claude_code.py:3-8,107-115`).

The native mapping is:

| Claude source | Native meaning |
|---|---|
| first stable `sessionId` | Required main session identity. Later values must agree. |
| first `cwd` and `gitBranch` | Presence-only private provenance. |
| assistant `message.model` | Recorded model candidate. |
| plain string user `message.content` | User history message. |
| assistant text blocks | Assistant history in block order. |
| assistant `tool_use` blocks | Provider-native assistant content retained in history and correlated by block `id`. |
| user `tool_result` blocks | Provider-native user content retained in history and correlated by `tool_use_id`. |
| `permission-mode`, `file-history-snapshot`, `summary`, `attachment` | Ignored metadata count. |
| `isSidechain` | Separates main from subagent records. |
| first subagent `parentToolUseId` | Required parent link for a discovered subagent. |

The Python parser's metadata, user/tool-result, and assistant/tool-use handling
is at `claude_code.py:260-326`. Its sidechain filtering is at
`claude_code.py:270-278`.

Each distinct assistant `message.id`, falling back to record `uuid`, is one
inferred model call. Multiple records for the same ID may be repeated snapshots
or chunks. The importer merges non-conflicting blocks by stable block ID and
text position, suppresses exact duplicates, and rejects conflicting reuse of a
tool-use ID. Python merely appends every record until the message ID changes
(`claude_code.py:299-326`); the native rule avoids double-counting repeated
snapshots while failing closed on ambiguity.

When `include_subagents=true`, subagent files are parsed only from the exact
documented pattern and linked through `parentToolUseId`. A matching parent
`Task` tool call produces a sibling session with ID
`<main-session>#sa#<parentToolUseId>`, matching Python's stable ID at
`claude_code.py:72-87`. The subagent program has family `subagent` and records
the parent session and tool-use IDs in comparability annotations. It is not
inlined into the parent graph and its tool call is not executed. An unmatched
or multiply matched subagent is an error, rather than being silently dropped.

Claude logs do not carry a system prompt; Python explicitly sets it to `None`
at `claude_code.py:329-361`. Anthropic content blocks remain provider-native.
Running a Claude import against an endpoint dialect that cannot accept those
message blocks must fail during endpoint/input compatibility preflight, not
silently translate them.

## Fidelity contract

Every imported program carries these stable comparability annotations:

```text
source_format = "codex" | "claude_code"
request_wire_exact = false
tool_schema_available = false
output_tokens_available = false
model_latency_available = false
reasoning_included = false
tool_results_complete = true | false
subagent_topology = "none" | "sibling"
```

The following semantics cannot be faithfully recovered and must never be
implied by metrics or provenance:

1. **Exact provider request wire.** The logs record conversation events, not
   the complete request body. Reconstructed messages are new canonical bytes.
2. **Tool schemas.** Tool uses include names and arguments, but not necessarily
   the tool-definition array presented to the model. `request.tools` is absent.
3. **Sampling and generation caps.** Temperature, top-p, top-k, min-p, stop
   rules, and per-call maximum tokens are generally absent.
4. **Token usage.** The fixtures and parser contracts do not supply reliable
   per-call completion usage. Native fallback generation caps therefore do not
   represent observed OSL.
5. **Model latency.** Event timestamps are typically completion or observation
   times; request-start time is unavailable. Assistant-to-assistant gaps mix
   model latency, user think time, tool work, and client overhead.
6. **Reasoning.** Codex exposes reasoning summaries, not complete reasoning;
   Claude may contain thinking or redacted blocks with provider-specific
   handling. Neither is added to replay prompts by this design.
7. **Executable tool semantics.** Generic Bash, Read, patch, Task, MCP, and
   custom tool payloads are observations, not an authorized sandbox recipe.
   The existing executable path requires explicit command and environment
   semantics (`rust/runtime/src/graph/recorded/agent_recording/lowering.rs:276-290,586-603,637-646`).
8. **Parallel timing.** A first-call/last-result bundle interval is observable,
   but independent tool start/end intervals can be missing or share one message
   timestamp.
9. **Branches, resume, and compaction.** Claude `parentUuid`, sidechains,
   summaries, and resumed-session headers do not by themselves establish one
   complete executable branch. The first delivery accepts one validated linear
   main chain plus separately linked subagents.

These limitations are why imported sessions are not eligible for
`recorded-agent-default` parity claims.

## Security and privacy

Session logs routinely contain user prompts, system/developer instructions,
assistant output, reasoning summaries, local paths, branch names, tool
arguments, tool results, environment data, and secrets. The implementation
shall:

- Never include raw prompt, reasoning, tool argument, tool result, cwd, branch,
  or ignored record contents in errors, tracing, warning facts, trace IDs, or
  comparability annotations.
- Retain raw content only in the immutable segment store where it is required
  to construct benchmark requests.
- Keep cwd and git branch as booleans indicating presence; do not recreate the
  Python tags that embed their values (`_agent_session.py:406-415`).
- Omit Codex reasoning summaries entirely. Errors may identify only their
  record type and safe source coordinates.
- Reject `execute_tools=true`; imported actions have no trusted environment or
  command policy.
- Reject symlinks and root escapes and share one exact file enumeration between
  local loading and cross-host shipping.
- Digest exact source file bytes with BLAKE3, but do not attach a full source
  snapshot to serialized program metadata.
- Avoid a public `anonymize` switch in the first delivery. Python's option only
  anonymizes initial user messages and tool results
  (`_agent_session.py:379-386,426-438`) and therefore does not provide a
  defensible privacy boundary.

## Error policy

All import errors use a source-specific wrapper with this safe context when
available:

```text
<path>: line <n>: source <codex|claude_code>: record <type>: <detail>
```

The detail must not contain the serialized record or user-controlled content.

The importer rejects:

- unreadable files, invalid UTF-8, invalid JSON, and non-object JSONL records;
- an empty discovered file set or a file with no usable records;
- missing, empty, or inconsistent session IDs;
- ambiguous auto-detection;
- duplicate source paths or resolved trace IDs;
- duplicate/conflicting tool-use IDs or subagent parent matches;
- invalid/non-finite timestamps when a timestamp is used for tool timing;
- a normalized session with no inferred model calls;
- source/config combinations declared invalid in the public contract;
- paths outside the selected canonical root or any symlink in the selected
  path set.

Unknown additive fields and unknown record types are not errors by themselves;
the CLI formats evolve independently of AIPerf. They are ignored only after
required identity and correlation fields are validated, and their count is
retained as a non-secret annotation. This is intentionally less rigid than the
Mini-SWE manifest DTOs, which use `deny_unknown_fields` for strict replay
defaults and tasks (`rust/runtime/src/graph/recorded/agent_recording/schema.rs:11-86`).

Missing tool results at an interrupted end of file are non-fatal and annotated.
Malformed or conflicting results are fatal. Missing/invalid timestamps disable
that tool bundle's delay only when the timestamp is not required for another
validation; no synthetic timestamp or duration is invented.

## Module ownership

The implementation adds focused modules without weakening the Mini-SWE types:

```text
rust/runtime/src/graph/recorded/agent_recording/
  import/
    mod.rs          source enum, shared IR, public dispatcher, common error
    discovery.rs    exact deterministic file sets and format detection
    codex.rs        Codex DTOs and state machine
    claude_code.rs  Claude DTOs, sidechain parsing, subagent linking
    lowering.rs     ImportedAgentSession -> GraphInputBundle
  discovery.rs      unchanged ownership: Mini-SWE recording/manifest discovery
  schema.rs         unchanged ownership: Mini-SWE wire DTOs
  lowering.rs       unchanged ownership: Mini-SWE lowering
  mod.rs            exports both strict replay and session-import entry points
```

Other ownership:

- `rust/runtime/src/config/model/dataset.rs`: public typed config and serde
  defaults.
- `rust/runtime/src/config/validate.rs`: source/config compatibility.
- `rust/cli/src/flags.rs`, `rust/cli/src/load.rs`, and `rust/cli/src/yaml.rs`:
  native CLI and YAML projection.
- `rust/runtime/src/engine/graph_input.rs`: select Mini-SWE or session import,
  then retain one canonical prepared-input policy. It must not contain source
  parsing state machines.
- `rust/runtime/src/engine/cellular_controller.rs` and
  `rust/runtime/src/engine/cellular_cell.rs`: ship and preflight the exact
  discovered set; they must not reimplement discovery patterns.

The package root documentation in
`rust/runtime/src/graph/recorded/agent_recording/mod.rs:4` must be widened from
"Strict Mini-SWE-Agent recording input" to recorded-agent sources generally,
while the existing Mini-SWE functions keep their names and behavior.

## Verification strategy

### Fixtures

Port these source fixtures from `ajc/dag-v3` into
`rust/runtime/tests/fixtures/recorded_agent_session_import/`:

- `tests/fixtures/graph/codex/linear.jsonl`
- `tests/fixtures/graph/codex/with_tools.jsonl`
- `tests/fixtures/graph/claude_code/linear.jsonl`
- `tests/fixtures/graph/claude_code/parallel_tools.jsonl`
- `tests/fixtures/graph/claude_code/with_subagent/main.jsonl`
- `tests/fixtures/graph/claude_code/with_subagent/main/subagents/agent-aaa.jsonl`

Add focused adversarial fixtures for a Claude lead-in record, repeated assistant
snapshot, mismatched session IDs, duplicate tool IDs, dangling tool result,
unmatched subagent, invalid timestamp, mixed-format directory, root escape, and
symlink.

### Unit and integration tests

1. **Detection/discovery:** explicit and automatic single-file detection;
   Codex recursive sorting; Claude shallow main discovery; optional exact
   subagent pattern; empty/mixed/ambiguous input errors; root/symlink checks.
2. **Codex normalization:** metadata fallback, system/developer/user history,
   function-call boundary before results, parallel correlation, missing-result
   annotation, and omission of reasoning content.
3. **Claude normalization:** main-chain filtering, distinct message-ID calls,
   duplicate snapshot suppression, parallel tool ordering, provider-native
   content blocks, and stable sibling subagent identity.
4. **Golden session IR:** assert complete normalized calls and safe fidelity
   annotations. The golden explicitly records intended differences from the
   Python union graph.
5. **Lowering:** follow the established public lowering tests at
   `rust/runtime/tests/recorded_agent_lowering.rs:53-183`; assert graph node and
   edge order, interned message wires, delay placement, fallback token caps,
   absence of tool nodes/environment, replay metadata, segment reuse, and
   `format == "agent_recording"`.
6. **Engine/config:** extend the adapter tests near
   `rust/runtime/src/engine/graph_input.rs:1882-1904`; test strict config decode,
   CLI/YAML projection, invalid source/options, recorded model behavior, and
   canonical prepared-input policy.
7. **Cellular:** prove the controller-owned discovered snapshot is the only
   source for cell planning and downloads, excludes unrelated JSONL, and rejects
   symlink replacement.
8. **Privacy:** assert errors, warnings, debug output, trace identity, and
   annotations do not contain fixture prompt text, cwd, branch, arguments,
   results, or reasoning.

The Python unit coverage that motivates the baseline fixture cases is at
`ajc/dag-v3:tests/unit/dataset/loader/graph/adapters/test_codex.py:19-112` and
`ajc/dag-v3:tests/unit/dataset/loader/graph/adapters/test_claude_code.py:19-126`.

### Product E2E

Add deterministic product tests against `aiperf-mock-server`, because new graph
input behavior requires raw per-record verification rather than summary-only
checks:

- Codex through the OpenAI chat endpoint.
- Claude through the Anthropic messages endpoint; the runtime provides that
  dialect at `rust/runtime/src/endpoints/anthropic.rs:48-115`.

Pin model, tokenizer, input, output cap, streaming mode, TTFT/ITL, and jitter.
Inspect raw request records for exact normalized message arrays, inferred call
count, stable trace identity, model override policy, fallback max tokens,
streaming mode, response content, status, and errors. For tool-bearing fixtures,
assert the graph contains no executable tool node and that the deterministic
clock-level test—not a wall-clock tolerance—owns observed bundle-delay
verification.

## Delivered milestones

### Delivered: source contract and safe discovery

Add the typed source enum/config, validation, common import IR, error type,
deterministic file enumeration, single-file detection, source digests, and
adversarial discovery tests.

### Delivered: Codex local import

Add the Codex state machine, causal function-call normalization, dedicated
session lowering, config/CLI projection, golden fixtures, lowering tests, and a
single-file chat E2E. The deliverable is a complete local Codex session replay
with explicit fidelity annotations.

### Delivered: Claude main sessions

Add Claude main-chain parsing, message-ID consolidation, provider-native
content preservation, parallel tool correlation, messages-endpoint preflight,
golden fixtures, and a messages-endpoint E2E.

### Delivered: Claude subagents

Add exact subagent discovery, parent `Task` correlation, stable sibling
identity, privacy-safe parent annotations, unmatched/duplicate-parent errors,
and directory fixture coverage.

### Delivered: cellular exact-set shipping and documentation

Make local discovery's selected file set the sole source for every cellular
delivery. Remote cells fetch the captured bytes from their routable controller
authority; co-located cells compile a controller-owned scratch materialization
when HTTP delivery is unavailable. Source replacement after discovery cannot
change planning or execution.

Each milestone is independently reviewable and keeps `agent_recording` as the
only public graph-input format.

## Recorded decisions

The following records distinguish shipped behavior from future design work.

### Shipped decisions

1. **Cross-host directory support:** implemented as exact-set shipping, never
   replay-root traversal. The controller snapshots discovered bytes before it
   binds the artifact server; the server and every remote cell consume those
   bytes. Co-located no-HTTP cells receive a controller scratch materialization
   of the same bytes rather than the caller path.
2. **Claude endpoint compatibility:** provider-native Claude content is
   accepted only by the Messages endpoint. No implicit OpenAI wire translation
   is performed.
3. **System prompt representation:** Codex `base_instructions.text` becomes a
   system history message; Claude imports remain system-less because their
   recording has no equivalent authored request field.
4. **Observed tool authority:** imported tool calls and results are replay
   evidence only. The imported graph never executes a tool, shell command,
   MCP call, or subagent.

### Open questions

1. **Codex function-call grouping:** confirm against representative current
   Codex logs whether consecutive calls before the first output always form one
   model completion. If no stable boundary exists, require an annotation and a
   documented deterministic heuristic rather than claiming exact call count.
   The shipped behavior is fixture-backed by Codex E2E.
2. **Claude repeated message records:** determine whether current Claude Code
   writes streaming chunks, cumulative snapshots, or both for one `message.id`.
   The conflict/deduplication rule is fixture-backed.
3. **Interrupted-session policy:** this record allows a missing terminal tool
   result with an annotation. Confirm whether benchmark consumers prefer
   dropping such sessions by default; do not substitute placeholder tool
   content.
4. **Observed tool delay:** confirm whether the product wants raw observed
   first-use-to-last-result duration only or a later configurable scalar
   reduction. Empirical/lognormal distributions remain out of scope.
5. **Source-version gating:** neither Python adapter validates CLI version
   beyond structural fields. Decide whether provenance records a non-secret
   version and whether known incompatible major versions fail closed.
6. **Comprehensive redaction:** if users require a privacy transform, design a
   separate explicit policy covering system/developer/user/assistant content,
   tool arguments/results, paths, metadata, and derived segments. The Python
   partial `anonymize` behavior is not an acceptable contract.

Decisions that change provider wire shape, timing semantics, executable tool
authority, or privacy behavior require an update to this record before
implementation.
