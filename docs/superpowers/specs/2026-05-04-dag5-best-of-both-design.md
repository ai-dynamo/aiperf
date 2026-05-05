# `ajc/dag5` — Best-of-Both DAG Branch Design

**Status:** approved design, pending implementation plan
**Date:** 2026-05-04
**Author:** Anthony Casagrande (acasagrande@nvidia.com)

## Goal

A new feature branch `ajc/dag5` that combines the advanced DAG framework from
`ajc/inferencex-agentx-mvp` with the targeted refinements from `ajc/dag4`,
deliberately omitting the InferenceX/AgentX scenario, the Weka loader stack,
the cache-bust marker injection, and the agentic-replay strategy. The branch
should be a clean, mergeable-shaped DAG-only delta against `origin/main`.

## Branch Baseline

Branch from `origin/main` (HEAD as of 2026-05-04). Single feature track:
DAG benchmark mode (FORK + pre-session SPAWN), plus the endpoint refactor
that supports DAG context replay.

### In-Scope

- DAG framework: `BranchOrchestrator` with fan-in, multi-gate, K-delayed
  joins, pre-session SPAWN, prereq walking; FORK pin refcounting; sticky
  routing via `parent_correlation_id`; `agent_depth`/`parent_correlation_id`
  flow through `Credit` / `TurnToSend` / `RequestInfo` / `RequestRecord`.
- `dag_jsonl` input type with `forks:` AND `spawns:` shorthand, prereq
  validation, cycle / multi-parent / non-terminal-fork checks.
- Endpoint refactor: `response_mixin.py`, generic turn→messages building,
  `BaseEndpoint.build_assistant_turn` for context replay, function call
  arguments SSE handling on Responses, mixed `content + tool_calls` on Chat,
  single-pass `extract_payload_inputs` for media counting.
- Supporting input types that travel with the endpoint refactor: `inputs_json`
  loader, `raw_payload` loader, `raw` endpoint, mooncake-trace `payload` mode.
- DAG-aware `--num-conversations` autodefault (`_count_dag_root_entries` +
  `_is_forking_dataset`).
- `--request-count` caps DAG children too (dag4 semantics — see Behavior
  Decisions below).
- `--no-fixed-schedule` (`InputConfig.disable_auto_fixed_schedule`) — generic
  loadgen flag, not Weka-specific.
- `BranchStats` published via `CreditPhaseCompleteMessage` and exported to
  `profile_export_aiperf.json`. `joins_suppressed` counter for stop-condition
  drains.
- `_DagSettings.FAIL_FAST` env var (`AIPERF_DAG_FAIL_FAST`).
- DAG-aware completion gates in records-tracker and phase-runner; child
  HTTP requests count toward `requests_sent`; children honor cancellation
  and duration stop conditions.
- Documentation: `docs/benchmark-modes/dag.md`,
  `docs/tutorials/inputs-json-replay.md`, `docs/tutorials/raw-payload-replay.md`,
  + the three-file sync of `dag_jsonl` mention (CLAUDE.md +
  `.github/copilot-instructions.md` + `.cursor/rules/python.mdc`).

### Out-of-Scope (explicit)

- AgentX scenario, `_AgentXSettings`, `--scenario`/`--unsafe-override`,
  `AGENTIC` mode wiring.
- Weka loaders (`weka_trace.py`, `weka_parallel_convert.py`, `weka_synth_buf.py`,
  `weka_trace_models.py`, `semianalysis_cc_traces_weka.py`, etc.),
  `--use-think-time-only`, weka delta-context.
- Cache-bust marker injection: `_apply_cache_bust_*` in
  `src/aiperf/workers/worker.py`, `cache_bust_marker` / `cache_bust_target`
  plumbing, `validate_cache_bust_compatibility`.
- Agentic-replay strategy (`src/aiperf/timing/strategies/agentic_replay.py`),
  `src/aiperf/timing/trajectory_source.py`, AGENTIC_REPLAY mode.
- `Turn.reset_context` (only consumer was Weka delta-context).
- Plugin-categories split (`accumulator` / `stream_exporter` / `analyzer`) —
  stay on main's single `ResultsProcessorType`. `BranchStats` exporter sits
  on the older pipeline (~50 lines of adaptation).
- Realtime stats overhaul (`_render_realtime_block`, `AccumulatorMetricsSummary`,
  dynamic `realtime_metrics_interval(ui_type)` resolver).
- ASR loader stack, SageMaker capture loader, additional accuracy benchmarks
  beyond what is on `main`.
- `--ignore-trace-delays` (Weka-flavored).

## Component-Level Architecture

The DAG surface layers into five concerns. Each is independently understandable
and the interfaces between layers are message-bus-typed.

### a. Loader Layer

- `dataset/loader/dag_jsonl.py` + `dataset/loader/dag_jsonl_models.py`
- Pure parse/validate; no timing or worker awareness.
- Outputs `Conversation` objects with:
  - `branches: list[ConversationBranchInfo]` (mode FORK or SPAWN; SPAWN
    branches set `dispatch_timing="pre"` for background pre-session
    dispatch — the `ConversationBranchInfo.dispatch_timing` field defaults
    to `"post"` and `"pre"` is reserved for SPAWN per the field validator)
  - `agent_depth` stamped via topology walk
  - `prerequisites` attached to `Turn` instances
- Sister DAG-adjacent loaders (`inputs_json.py`, `raw_payload.py`,
  `mooncake_trace.py`) share the same `Conversation` output shape and ship
  alongside DAG because they were part of the same endpoint-refactor commit
  on `dag4` and are needed for byte-exact DAG replay.
- `can_load` rejects non-dict first records via an explicit
  `isinstance(data, dict)` guard (current
  `src/aiperf/dataset/loader/dag_jsonl.py:122-127`); dag4 lacks the guard
  and raises `AttributeError` on non-dict probes. Keep current's guard.

### b. Data Models

- `common/models/dataset_models.py`: `Turn`, `TurnMetadata`, `Conversation`.
  - `Turn` carries `prerequisites: list[TurnPrerequisite] | None`.
  - `Conversation.metadata()` projects `prerequisites=turn.prerequisites`
    into each `TurnMetadata` it builds (dag4 line 378). Current
    `Conversation.metadata()` (lines 457–471) omits the field — latent bug
    confirmed. Note: `Turn.metadata()` projects prereqs on both branches;
    only the `Conversation.metadata()` walk differs. Take dag4's projection.
  - `TurnMetadata.has_forks` stamped at load time so the sticky router can
    defer eviction until all forks have spawned.
- `common/models/branch.py`: `ConversationBranchInfo` with
  `dispatch_timing: Literal["pre","post"]` (default `"post"`; `"pre"` is
  reserved for background SPAWN branches per the field validator).
- `common/models/branch_stats.py`: `BranchStats` with `joins_suppressed`
  counter for stop-condition-suppressed joins.
- `common/enums/enums.py`: `ConversationBranchMode` (FORK | SPAWN) enum.
- `dataset/loader/dag_jsonl_models.py`: `DagSpawn` model backing the
  `spawns:` shorthand in dag_jsonl input.
- `common/models/prerequisites.py`: `TurnPrerequisite`, `PrerequisiteKind`.

### c. Timing Layer

- `BranchOrchestrator` (in `timing/branch_orchestrator.py`): full advanced
  version from current — fan-in, multi-gate, K-delayed joins, pre-session
  SPAWN, prereq walking, child-error handling gated by
  `AIPERF_DAG_FAIL_FAST`, marker minting.
- `ConversationSource.start_branch_child` and
  `ConversationSource.start_pre_session_child` — child SampledSession builders
  that inherit sticky routing from parent.
- `src/aiperf/timing/strategies/request_rate.py` threads the orchestrator
  and routes `requests_sent`-cap refusals to `on_child_stopped` so parent
  joins drain (dag4 method `_issue_child_continuation_or_release` ported
  and adapted to current's fan-in/multi-gate logic).
- `phase/credit_counter.py`: child credits flip `is_final_credit` once
  `requests_sent` crosses the cap (dag4).
- `phase/stop_conditions.py`: `RequestCountStopCondition.applies_to_dag_children = True`;
  `SessionCountStopCondition` stays root-only (dag4 design).
- `TimingManager._on_dataset_configuration_failed` +
  `_wait_for_dataset_or_failure` (current — `src/aiperf/timing/manager.py:95`
  and `:140`; absent on dag4) — fixes a 300s hang when DatasetManager
  configure fails.

The orchestrator is the only component aware of DAG topology; everything
below it sees ordinary credits with `agent_depth` / `parent_correlation_id`
populated.

### d. Worker Layer

- FORK pin refcount on `UserSession`; child credits inherit parent's session
  slot for credit counting and sticky routing.
- `UserSession.is_fork_parent` stamped at `create_and_store` time (dag4 fix)
  so PAYLOAD_BYTES round-trips do not lose the fork-parent flag.
- `inference_client.py`: payload_bytes JSON validation guard;
  `RecordContext` propagation for `RequestInfo` downcast at the ZMQ hop
  (full Turn list never crosses the wire).
- Refcount-based FORK-pin eviction; parent session evicts only when refcount
  hits zero (no more pending forks).

### e. Endpoint Layer

- `endpoints/response_mixin.py`: shared SSE / chat / responses parsing.
  Improved JMESPath compile-failure log explaining the auto-detect fallback.
- `endpoints/base_endpoint.py.build_assistant_turn`: captures the assistant
  reply for context replay so subsequent turns and FORK-mode children see
  what the model actually said. Default implementation captures plain text
  and the `content` field of `ReasoningResponseData`; `reasoning` itself is
  dropped (most chat templates do not round-trip it). Endpoints with
  structured response fields override.
- `endpoints/openai_responses.py`: surfaces
  `response.function_call_arguments.delta` and `.done` SSE events as
  `ToolCallResponseData` so TTFO fires and OSL is correct on tool-using
  turns. `_extract_response_content` walks `output[]` for `function_call`
  items in non-streaming responses (precedence: `reasoning > message > function_call`).
  `instructions` no longer double-inserted as a synthetic `{role: "system"}` message.
- `endpoints/openai_chat.py`: `ToolCallResponseData(tool_call_text=..., content=...)`
  for chunks carrying both prose and tool-call deltas (~18% of agentic
  turns); rename `text` → `tool_call_text` field.
- `endpoints/raw_endpoint.py`: byte-exact payload replay; raw payloads
  decoupled from the raw endpoint so existing endpoints can also pre-format
  raw payloads.
- Assistant-turn handling added to `endpoints/openai_chat.py` and
  `endpoints/openai_responses.py`; Responses `instructions` handling lives
  on the `openai_responses` override.

## Data Flow

### FORK (child shares parent's session and history)

1. **Loader**: parses turn with `branches=[BranchInfo(mode=FORK, target_turn_ids=[...])]`,
   stamps `TurnMetadata.has_forks=True` on the parent turn, walks topology
   to stamp `Conversation.agent_depth` on each child.
2. **Phase 0 (warmup)** ignores DAG (root credits only).
3. **Phase 1 (steady)**:
   - `CreditIssuer` dispatches root credit for parent session.
   - Worker executes parent turn → parses response → calls
     `build_assistant_turn` → appends to session history.
   - Worker emits `CreditCompleted` with child-fork hints derived from
     `TurnMetadata.has_forks`.
   - `BranchOrchestrator` receives → for each `forks:` target, calls
     `ConversationSource.start_branch_child(parent_correlation_id, target_turn_id)`:
     - builds `SampledSession` sharing parent's `session_id`
     - inherits parent's `UserSession` slot (refcount++)
     - copies sticky routing (`parent_correlation_id` flows through to `RequestInfo`)
   - Child credit dispatched at `agent_depth=parent.depth+1`; each fork
     target gets its own child credit; orchestrator records pending joins.
   - When all child credits complete → `BranchOrchestrator` fires join →
     session refcount--.
   - Parent session evicts only when refcount hits zero.

### SPAWN (pre-session, child gets a brand-new session, dispatched before parent)

1. **Loader**: parses `Conversation` with `spawns: [DagSpawn(...)]` at root.
   `DagJsonlLoader._inline_pre_session_spawns` hoists the spawn target into
   its own `SampledSession` with `dispatch_timing="pre"` before the credit
   phase opens.
2. **Phase 0/1 entry**:
   - `PhaseRunner` consults the orchestrator for pre-session entries.
   - `ConversationSource.start_pre_session_child` builds the `SampledSession`.
   - Pre-session children dispatch immediately — no parent gating.
   - Their `CreditCompleted` closes the prereq gate that holds the dependent
     root credit.

### Prereq gating

- When a `Turn` has `prerequisites=[TurnPrerequisite(kind=...)]`, the
  orchestrator blocks credit dispatch on that turn until all prereqs are
  satisfied.
- Orchestrator advances when the gating events arrive on the message bus.

### Cap-induced cancellation

- Cancellation, child errors, and the `--request-count` cap all unwind
  through the orchestrator's join-tracking state.
- `AIPERF_DAG_FAIL_FAST=1` aborts the whole run on first child error.
- Default tolerates and counts via `BranchStats.errors`.
- Cap-gated children route via `_issue_child_continuation_or_release` →
  `on_child_stopped` so parent joins drain cleanly. `BranchStats.joins_suppressed`
  tracks how many joins ended this way.

## Behavior Decisions

### `--request-count` caps DAG children (dag4 semantics)

`--request-count 30` means **30 wire requests, period.** Children count
against the cap. Rationale: "request count" is wire-level vocabulary in the
rest of the tool — token counts, latency percentiles, and throughput are all
wire-level — so a topology-aware cap would surprise users. The escape hatch
for "I want 30 sessions regardless of forks" is `--num-conversations 30`,
which is plumbed correctly.

This is paired with:
- `RequestCountStopCondition.applies_to_dag_children = True`
- `phase/credit_counter.py` `is_final_credit` flip when `requests_sent`
  crosses cap
- `request_rate._issue_child_continuation_or_release` to drain joins on cap

`SessionCountStopCondition` remains root-only (current and dag4 agree).

### `--num-conversations` autodefault for `dag_jsonl`

Per dag4: when input is `dag_jsonl`, default `--num-conversations` to the
count of root entries (`_count_dag_root_entries`). Refuses to default
`--request-count` for forking datasets so the cap does not truncate
mid-tree.

### `Turn.reset_context` dropped

Only consumer was Weka delta-context (LCP cuts). With Weka out of scope, the
field has no consumer, so it is dropped to keep the model surface clean.
Future delta-context loaders can reintroduce it as a deliberate add.

## Error Handling & Failure Modes

### Loader-time validation (fail at config-load, not at runtime)

- Cycles in the DAG, multi-parent without explicit fork target, non-terminal
  forks, dangling prereq references, undefined `forks:`/`spawns:` targets:
  rejected with file:line in the error.
- Non-dict first record in `can_load` returns False (auto-detection robustness).
- Dataset-configuration failure raises a `DatasetConfigurationFailed` event;
  `TimingManager._wait_for_dataset_or_failure` catches it and aborts cleanly
  instead of hanging 300s for the configure timeout.

### Runtime — child errors

- **Default**: count the error in `BranchStats.errors`, release the join
  slot, drain pending siblings, continue. Parent session refcount decremented.
- **`AIPERF_DAG_FAIL_FAST=1`**: cancel all pending children of the same
  parent, raise to `PhaseRunner`, terminate phase.

### Runtime — cap-induced cancellation

When `--request-count` is hit mid-tree, the gated child routes through
`on_child_stopped` (not as an error) so parent joins still drain.
`BranchStats.joins_suppressed` counter tracks the count (reportable but
not a failure).

### Runtime — payload-bytes round-trip pitfalls

- `UserSession.is_fork_parent` stamped at `create_and_store` time (dag4 fix),
  not lazily recomputed from `conversation.branches` — survives PAYLOAD_BYTES
  round-trip where `branches` is dropped.
- `Conversation.metadata()` projects `prerequisites=turn.prerequisites` into
  `TurnMetadata` (dag4 fix).

### Worker-level

- Sticky-routing miss for a child (parent's `correlation_id` evicted before
  child dispatched): worker logs and falls back to load-balanced routing;
  `BranchStats` records the slip.
- Tool-call response with neither `content` nor `tool_calls`: emit no
  `ParsedResponse`, log at debug; does not kill the credit.

### Endpoint

- Malformed `response_field` JMESPath: log warning, fall back to auto-detect
  (dag4's improved error message).
- Streaming tool call with delta-only content: surfaced via
  `ToolCallResponseData`; TTFO fires correctly.

## Testing Strategy

Three tiers, each independently runnable. Per repo conventions, each tier
is its own pytest invocation with `-n auto`, with the **tier subfolder as
the path argument** — never narrower paths, never combined tiers in one
invocation:

```bash
uv run pytest tests/unit/ -n auto
uv run pytest -m component_integration -n auto
uv run pytest -m integration -n auto
```

### Unit (`tests/unit/`)

- **Models**: `test_branch_model.py`, `test_prerequisite_model.py`,
  `test_dataset_models_prereq.py` — Turn / Conversation / TurnMetadata
  round-trips with prereqs, fork/spawn fields, agent_depth stamping.
- **Loader**: `test_dag_jsonl.py`, `test_dag_jsonl_prereq{,_adversarial}.py`,
  `test_dag_jsonl_topology_pathological.py` — every reject path
  (cycles, multi-parent, non-terminal forks, dangling prereqs, malformed
  `forks:`/`spawns:`).
- **Config**: `test_user_config_dag_default.py` — `--num-conversations`
  autodefault on `dag_jsonl` input.
- **Endpoints**: `test_openai_responses.py` (function call arguments
  delta/done), `test_openai_chat.py` (mixed content+tool_calls precedence),
  `test_response_mixin.py` (JMESPath fallback), `test_base_endpoint.py`
  (`build_assistant_turn` for chat / responses / completions).
- **BranchOrchestrator**: `test_branch_orchestrator.py` plus the eight
  scenario files
  `test_branch_orchestrator_{fan_in,multi_gate,delayed,join,phase0,pre_session,adversarial,adversarial_full}.py`
  — port all 9 from current's `tests/unit/timing/`.

### Component-integration (`tests/component_integration/`, single-process)

- **DAG cross-cutting**: `test_dag_cross_component.py`,
  `test_dag_concurrency_pathology.py`, `test_dag_combined_pathology.py`,
  `test_dag_timing_pathology.py`, `test_dag_join_end_to_end.py`,
  `test_dag_v1_adversarial.py`, `test_dag_adversarial_timing_modes.py`.
- **Hard cap**: `tests/component_integration/timing/test_dag_hard_cap.py`
  (dag4) — verifies `--request-count 30` produces exactly 30 wire requests
  across forks.
- **Multi-root**: `tests/component_integration/timing/test_dag_multi_root_payload_bytes.py`
  (dag4) — multi-root DAG round-trips through PAYLOAD_BYTES context mode.
- **Prereqs**: `test_prerequisites.py`, `test_prerequisites_adversarial.py`,
  `test_prereq_metadata_adversarial.py`.

### Integration (`tests/integration/`, multi-process)

- One reference DAG topology end-to-end against the in-repo mock server,
  asserting BranchStats numbers, child credit counts, fork-pin refcount
  drained, output `profile_export_aiperf.json` round-trip.
- `inputs_json` + `raw_payload` byte-exact replay tests.

### Fixtures

Copy `tests/fixtures/dag/` from current (multi-fork, deep-chain, fan-in,
multi-gate, K-join, pre-session-spawn) plus dag4's
`multi_root_single_turn.dag.jsonl`.

### Skip

All `test_agentic_replay_*` (12 files), all `test_weka_*` (~20 files),
cache-bust test family — those features are not in `dag5`.

## Plugin Registry

`src/aiperf/plugin/plugins.yaml` entries:

- **Loaders**: `dag_jsonl`, `inputs_json`, `raw_payload`, `mooncake_trace`,
  plus whatever is on `main`. **Not** added: `weka_trace`,
  `semianalysis_cc_traces_weka`, `sagemaker_data_capture`, ASR loader set,
  `agentic_replay`.
- **Endpoints**: `raw` endpoint added; existing endpoints retained.
- **Stop conditions**: `RequestCountStopCondition.applies_to_dag_children`
  reflected in plugin metadata if applicable.
- Plugin categories: stay on `main`'s shape (no `accumulator` /
  `stream_exporter` / `analyzer` split).

Validate with `make validate-plugin-schemas` and regenerate enums/overloads
with `make generate-all-plugin-files`.

## Documentation Updates

- `docs/benchmark-modes/dag.md` — full DAG mode reference (FORK + SPAWN,
  prereqs, request-count semantics, fail-fast env var).
- `docs/tutorials/inputs-json-replay.md` — `inputs.json` verbatim replay.
- `docs/tutorials/raw-payload-replay.md` — `raw_payload` byte-exact replay.
- `README.md` — add new tutorials to index.
- Three-file sync per project rule (`CLAUDE.md`,
  `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`): mention
  `dag_jsonl` input type as already done in the existing CLAUDE.md tip
  line; verify all three files match after edit. (dag4's commit added the
  same mention to `AGENTS.md` for a "four-file sync"; project CLAUDE.md
  only mandates the three-file sync, so AGENTS.md is optional and only
  updated if it already participates in this repo's sync rule when the
  port lands.)
- `docs/cli-options.md` — auto-regenerated via `make generate-cli-docs` after
  CLI surface lands.
- `docs/environment-variables.md` — auto-regenerated via
  `make generate-env-vars-docs` after `_DagSettings.FAIL_FAST` lands.
- `docs/architecture.md`, `docs/dev/patterns.md` — DAG framework references
  and example patterns.

## Open Questions

None. All design decisions resolved during brainstorming session 2026-05-04.

## Implementation Plan

To be drafted via the `superpowers:writing-plans` skill in a follow-up
session, using this spec as the contract.
