# DAG5 Plan 2 — Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the DAG runtime on top of Plan 1's foundation — `BranchOrchestrator` with full FORK + pre-session SPAWN semantics, `dag_jsonl` loader with topology validation, worker FORK pin refcount, request-rate / credit-counter / stop-conditions changes for `--request-count` cap-applies-to-children semantics, plus the `--num-conversations` autodefault, `BranchStats` publication, full unit + component-integration + integration test suite, and DAG mode docs.

**Architecture:** Builds on `ajc/dag5` HEAD (Plan 1 done). Each task is small and TDD-style. End-state: a complete DAG benchmark mode end-to-end against the in-repo mock server, with FORK + pre-session SPAWN, prereq gating, fan-in / multi-gate / K-delayed-join, sticky routing via `parent_correlation_id`, `AIPERF_DAG_FAIL_FAST` gating, and `--request-count` semantics that cap children too.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest + pytest-asyncio + pytest-xdist, uv, ZMQ message bus.

---

## Source-of-Truth Pointers

Plan 2 ports content from two source branches into `ajc/dag5`:

- **`ajc/inferencex-agentx-mvp`** (the advanced DAG framework) is the source for: `BranchOrchestrator` (1042-line full version with fan-in / multi-gate / K-delayed-join / pre-session SPAWN / prereq walking / child-error gating), `ConversationSource.start_branch_child` and `start_pre_session_child`, the `dag_jsonl` loader and `dag_jsonl_models.py` (including the `spawns:` shorthand and `_inline_pre_session_spawns` topology rewrite), the worker FORK pin refcount on `UserSession`, `inference_client._enrich_request_record` with `RecordContext` downcasting, the `RecordContext` / `RequestInfo` split on `RequestRecord`, `agent_depth` / `parent_correlation_id` plumbing on `Credit` / `TurnToSend` / `RequestInfo` / `RequestRecord`, `TimingManager._on_dataset_configuration_failed` + `_wait_for_dataset_or_failure`, and the BranchOrchestrator unit-test family (9 files) plus DAG cross-component integration tests (~7 files).

- **`ajc/dag4`** (targeted refinements) is the source for: `request_rate.py` orchestrator threading + `_issue_child_continuation_or_release`, `phase/credit_counter.py` `is_final_credit` flip when `requests_sent` crosses cap, `phase/stop_conditions.py` `RequestCountStopCondition.applies_to_dag_children = True`, the `UserSession.is_fork_parent` stamped at `create_and_store` fix, `--num-conversations` autodefault for `dag_jsonl` (`_count_dag_root_entries` + `_is_forking_dataset` + the autodefault block), `--no-fixed-schedule` (`InputConfig.disable_auto_fixed_schedule`), and the `tests/component_integration/timing/test_dag_hard_cap.py` + `test_dag_multi_root_payload_bytes.py` test pair.

- **`origin/main`** is the starting point for everything else. Plan 1 already cherry-picked or ported the data models, endpoint refactor, sister loaders, and tutorials — see Plan 1's "Spec Coverage" table for what's already on `ajc/dag5` HEAD.

For verbatim file ports, the implementation step uses `git show <branch>:<path> > <path>`. When the source-branch file requires drift adjustments against the current `ajc/dag5` shape (e.g. references to `cache_bust_marker` / `cache_bust_target` / `Turn.reset_context` / agentic-replay imports — all of which are explicitly out of scope per the spec), the implementation step shows the targeted patch applied after the `git show` write. The audit notes for each task call out which lines need scrubbing.

The authoritative spec is [`docs/superpowers/specs/2026-05-04-dag5-best-of-both-design.md`](../specs/2026-05-04-dag5-best-of-both-design.md). Plan 2 covers the spec's §"In-Scope" items not already shipped in Plan 1 — see "Spec Coverage" at the bottom of this file for the mapping.

---

### Task 1: `agent_depth` and `parent_correlation_id` on `Credit` and `TurnToSend`

**Files:**
- Modify: `src/aiperf/credit/structs.py`
- Test: `tests/unit/credit/test_credit_dag_fields.py`

**Audit notes:**
- `inferencex-agentx-mvp` adds `agent_depth`, `parent_correlation_id`, `has_forks`, `branch_mode`, `cache_bust_marker`, `cache_bust_target` to both `Credit` and `TurnToSend`. The cache-bust pair is OUT of scope per spec §"Out-of-Scope (explicit)" — drop them. `dag4` has the same DAG-only diff without cache-bust; that's the clean version to port.
- `TurnToSend.from_previous_credit` gains an optional `next_meta: TurnMetadata | None = None` argument so the sticky router can propagate `has_forks` for the new turn. See `dag4_credit_structs.py` lines 132–148.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import ConversationBranchMode, CreditPhase
  from aiperf.credit.structs import Credit, TurnToSend


  class TestCreditDagFields:
      def test_credit_default_dag_fields(self):
          c = Credit(
              credit_num=0,
              credit_phase=CreditPhase.PROFILING,
              conversation_id="c",
              x_correlation_id="x",
          )
          assert c.agent_depth == 0
          assert c.parent_correlation_id is None
          assert c.has_forks is False
          assert c.branch_mode is ConversationBranchMode.FORK

      def test_credit_explicit_dag_fields(self):
          c = Credit(
              credit_num=1,
              credit_phase=CreditPhase.PROFILING,
              conversation_id="c",
              x_correlation_id="child-1",
              agent_depth=2,
              parent_correlation_id="root-corr",
              has_forks=True,
              branch_mode=ConversationBranchMode.SPAWN,
          )
          assert c.agent_depth == 2
          assert c.parent_correlation_id == "root-corr"
          assert c.has_forks is True
          assert c.branch_mode is ConversationBranchMode.SPAWN

      def test_turn_to_send_dag_fields_propagate_from_credit(self):
          c = Credit(
              credit_num=2,
              credit_phase=CreditPhase.PROFILING,
              conversation_id="c",
              x_correlation_id="child",
              agent_depth=1,
              parent_correlation_id="root",
              has_forks=False,
              branch_mode=ConversationBranchMode.FORK,
          )
          tts = TurnToSend.from_previous_credit(c)
          assert tts.agent_depth == 1
          assert tts.parent_correlation_id == "root"
          assert tts.branch_mode is ConversationBranchMode.FORK
          # has_forks defaults False on the produced turn (next_meta omitted).
          assert tts.has_forks is False

      def test_turn_to_send_has_forks_from_next_meta(self):
          from aiperf.common.models.dataset_models import TurnMetadata

          c = Credit(
              credit_num=3,
              credit_phase=CreditPhase.PROFILING,
              conversation_id="c",
              x_correlation_id="x",
          )
          meta = TurnMetadata(
              conversation_id="c",
              turn_index=2,
              has_forks=True,
          )
          tts = TurnToSend.from_previous_credit(c, next_meta=meta)
          assert tts.has_forks is True
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/credit/test_credit_dag_fields.py -v`

  Expected: FAIL — fields don't exist yet on `Credit`/`TurnToSend`, and `from_previous_credit` doesn't accept `next_meta`.

- [ ] **Step 3: Write minimal implementation**

  Edit `src/aiperf/credit/structs.py`:

  1. Add to imports near the top:

     ```python
     from typing import TYPE_CHECKING

     from aiperf.common.enums import ConversationBranchMode, CreditPhase

     if TYPE_CHECKING:
         from aiperf.common.models.dataset_models import TurnMetadata
     ```

  2. Add to the `Credit` struct field list (preserve msgspec field-order — append after the existing fields, all with defaults):

     ```python
         agent_depth: int = 0
         parent_correlation_id: str | None = None
         has_forks: bool = False
         branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
         """DAG branch mode for this credit. Ignored when parent_correlation_id is None
         (i.e. for root sessions). FORK = inherit parent turn_list; SPAWN =
         fresh context. Default FORK keeps wire footprint small via msgspec omit_defaults."""
     ```

  3. Add the same four fields (without the docstring repetition) to `TurnToSend`.

  4. Replace the existing `from_previous_credit` classmethod with the dag4 version:

     ```python
         @classmethod
         def from_previous_credit(
             cls, credit: Credit, next_meta: "TurnMetadata | None" = None
         ) -> Self:
             """Create the next turn to send from the previous turn's credit.

             Args:
                 credit: The previous turn's credit.
                 next_meta: Metadata for the NEW turn being built. When provided, the
                     ``has_forks`` flag is derived from it so the sticky
                     router can defer parent-entry eviction until DAG children drain.
             """
             return cls(
                 credit_num=credit.credit_num,
                 credit_phase=credit.credit_phase,
                 conversation_id=credit.conversation_id,
                 x_correlation_id=credit.x_correlation_id,
                 turn_index=credit.turn_index + 1,
                 should_cancel=credit.should_cancel,
                 cancel_after_ns=credit.cancel_after_ns,
                 drop_perf_ns=credit.drop_perf_ns,
                 agent_depth=credit.agent_depth,
                 parent_correlation_id=credit.parent_correlation_id,
                 has_forks=next_meta.has_forks if next_meta is not None else False,
                 branch_mode=credit.branch_mode,
             )
     ```

  Reference: `git show ajc/dag4:src/aiperf/credit/structs.py` (138 lines — diff against main at lines 9–10, 14–17, 54–60, 108–111, 118–137).

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — new tests green; no regressions on the existing credit / orchestrator suite (the new fields are all defaults).

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/credit/structs.py tests/unit/credit/test_credit_dag_fields.py
  git commit -s -m "$(cat <<'EOF'
  feat(credit): add agent_depth, parent_correlation_id, branch_mode to Credit / TurnToSend

  Adds the four DAG wire fields (agent_depth, parent_correlation_id,
  has_forks, branch_mode) to both Credit and TurnToSend. All default to
  the root-session value so non-DAG runs are unaffected on the wire
  (msgspec omit_defaults keeps the bytes off).

  TurnToSend.from_previous_credit gains an optional next_meta argument
  so the sticky router can propagate has_forks for the new turn —
  without it, parent eviction would race the next fork.

  Cache-bust marker fields are intentionally NOT included; cache-bust
  injection is out of scope for dag5 per the spec.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 2: `RecordContext` base + `RequestInfo` / `RequestRecord` DAG fields

**Files:**
- Modify: `src/aiperf/common/models/record_models.py`
- Test: `tests/unit/common/models/test_record_context.py`

**Audit notes:**
- Plan 1 deferred this explicitly (Plan 1 Task 9 docstring: "no RecordContext refactor in Plan 1").
- The full split lives in `inferencex_record_models.py` lines 564–762: `RecordContext` is a new base class carrying the slim post-transport fields; `RequestInfo(RecordContext)` adds the worker-only transport fields; `RequestRecord.request_info` is re-typed `RecordContext | None` so the worker can downcast before the ZMQ hop.
- DROP `cache_bust_marker` and `cache_bust_target` from the `RecordContext` definition — they're out of scope.
- Add `agent_depth` / `parent_correlation_id` to `RecordContext` (plus `MetricRecordMetadata` which mirrors them — see inferencex lines 111–120).

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import CreditPhase
  from aiperf.common.models.record_models import (
      RecordContext,
      RequestInfo,
      RequestRecord,
  )


  def _make_record_context(**overrides) -> RecordContext:
      defaults = dict(
          credit_num=0,
          credit_phase=CreditPhase.PROFILING,
          conversation_id="c",
          turn_index=0,
          x_request_id="r",
          x_correlation_id="x",
      )
      defaults.update(overrides)
      return RecordContext(**defaults)


  class TestRecordContext:
      def test_default_dag_fields(self):
          ctx = _make_record_context()
          assert ctx.agent_depth == 0
          assert ctx.parent_correlation_id is None
          assert ctx.payload_bytes is None
          assert ctx.max_tokens is None
          assert ctx.audio_duration_seconds is None

      def test_explicit_dag_fields(self):
          ctx = _make_record_context(
              agent_depth=3,
              parent_correlation_id="root",
          )
          assert ctx.agent_depth == 3
          assert ctx.parent_correlation_id == "root"


  class TestRequestInfoIsRecordContext:
      def test_request_info_inherits_record_context(self):
          assert issubclass(RequestInfo, RecordContext)

      def test_request_info_has_transport_extras(self):
          # RequestInfo has model_endpoint / turns / endpoint_headers / etc.
          # that RecordContext does not.
          ri_fields = set(RequestInfo.model_fields.keys())
          ctx_fields = set(RecordContext.model_fields.keys())
          extras = ri_fields - ctx_fields
          assert {"model_endpoint", "turns", "endpoint_headers"}.issubset(extras)


  class TestRequestRecordHoldsRecordContext:
      def test_record_context_assignable_to_request_info_field(self):
          ctx = _make_record_context(agent_depth=2)
          rr = RequestRecord(request_info=ctx)
          assert rr.request_info is ctx
          assert rr.request_info.agent_depth == 2

      def test_request_info_subclass_assignable(self):
          # RequestInfo IS A RecordContext, so it must also assign cleanly.
          # Build a minimal RequestInfo via duck-typing the parent class only.
          ctx = _make_record_context()
          rr = RequestRecord(request_info=ctx)
          # Round-trip through model_dump / model_validate must preserve the slim shape.
          dumped = rr.model_dump()
          rebuilt = RequestRecord.model_validate(dumped)
          assert rebuilt.request_info is not None
          assert rebuilt.request_info.x_correlation_id == "x"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_record_context.py -v`

  Expected: FAIL — `RecordContext` does not exist; `RequestInfo` is a top-level class on main, not a subclass.

- [ ] **Step 3: Write minimal implementation**

  Edit `src/aiperf/common/models/record_models.py`. The change is structural: replace the existing `class RequestInfo(AIPerfBaseModel):` block (currently around line 469 on `ajc/dag5` HEAD) with two classes — `RecordContext` containing the slim post-transport fields, and `RequestInfo(RecordContext)` containing the transport-only extras.

  Use the inferencex shape as the source. Pull the verbatim block:

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/common/models/record_models.py | sed -n '564,762p' > /tmp/record_context_block.py
  ```

  Then replace the existing `RequestInfo` class definition in the local file with the contents of `/tmp/record_context_block.py`, scrubbing the cache-bust fields:

  - Delete the `cache_bust_marker` field block on `RecordContext` (inferencex lines 678–685).
  - Delete the `cache_bust_target` field block on `RecordContext` (inferencex lines 686–691).
  - Leave the `agent_depth` / `parent_correlation_id` block intact.

  Also re-type `RequestRecord.request_info`:

  ```python
      request_info: RecordContext | None = Field(
          default=None,
          description="Slim per-record context (see ``RecordContext``). Built "
          "by ``inference_client._enrich_request_record`` from the full "
          "``RequestInfo`` that drove the request — stripping the transport-"
          "only extras so only the fields the record processor actually "
          "reads cross ZMQ.",
      )
  ```

  Also add `agent_depth` / `parent_correlation_id` to `MetricRecordMetadata` (inferencex lines 111–120) so downstream metric lookup has the fields:

  ```python
      agent_depth: int = Field(
          default=0,
          description="The DAG agent depth of the session that produced this record. 0 for root sessions, "
          "incremented by 1 for each nested subagent fork. Use to filter records by DAG layer.",
      )
      parent_correlation_id: str | None = Field(
          default=None,
          description="The x_correlation_id of the parent session that spawned this record's session via a "
          "DAG subagent fork. None for root sessions. Use to group sibling branches of the same DAG.",
      )
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS. The split is structurally compatible because `RequestInfo` still has every field it had on main; the new `RecordContext` parent just exposes a slim subset.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/record_models.py tests/unit/common/models/test_record_context.py
  git commit -s -m "$(cat <<'EOF'
  feat(models): split RequestInfo into RecordContext base + transport extras

  Introduces RecordContext as a slim post-transport context attached to
  RequestRecord, with RequestInfo subclassing it for the worker-only
  transport extras (model_endpoint, turns, headers, drop_perf_ns, etc.).
  This keeps the full Turn list off the ZMQ hop to the record processor;
  inference_client._enrich_request_record (Task 8) downcasts before the
  hop so only the slim fields cross the wire.

  Adds agent_depth and parent_correlation_id to RecordContext and to
  MetricRecordMetadata so DAG provenance flows end-to-end through the
  metric pipeline.

  Cache-bust marker fields are intentionally NOT included; cache-bust
  injection is out of scope for dag5 per the spec.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 3: `_DagSettings.FAIL_FAST` env var

**Files:**
- Modify: `src/aiperf/common/environment.py`
- Modify: `src/aiperf/common/__init__.py` or wherever the settings singleton is exported (verify with `git grep -n "_DatasetSettings\b" src/aiperf/common/environment.py`)
- Test: `tests/unit/common/test_dag_settings.py`

**Audit notes:**
- main has no `_DagSettings` class; fresh add. Pattern follows `_DatasetSettings` / `_DeveloperSettings` blocks at lines 141 and 185 of `main_environment.py`.
- Environment-variable prefix per project convention is `AIPERF_DAG_*` (verify `env_prefix=` on neighbouring `_DatasetSettings`).

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import importlib

  import pytest


  class TestDagSettings:
      def test_default_fail_fast_false(self, monkeypatch):
          monkeypatch.delenv("AIPERF_DAG_FAIL_FAST", raising=False)
          import aiperf.common.environment as env_mod

          importlib.reload(env_mod)
          assert env_mod.DagSettings().FAIL_FAST is False

      @pytest.mark.parametrize("raw,expected", [
          ("1", True),
          ("true", True),
          ("True", True),
          ("0", False),
          ("false", False),
      ])
      def test_env_override(self, monkeypatch, raw: str, expected: bool):
          monkeypatch.setenv("AIPERF_DAG_FAIL_FAST", raw)
          import aiperf.common.environment as env_mod

          importlib.reload(env_mod)
          assert env_mod.DagSettings().FAIL_FAST is expected
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/test_dag_settings.py -v`

  Expected: FAIL — `DagSettings` (or the equivalent module-level singleton) does not exist.

- [ ] **Step 3: Write minimal implementation**

  Add to `src/aiperf/common/environment.py` after `_DeveloperSettings`:

  ```python
  class _DagSettings(BaseSettings):
      """Settings for DAG benchmark mode (`dag_jsonl` input type).

      Toggles runtime behavior of `BranchOrchestrator` and dependent
      timing components. All fields default to non-DAG-aware behaviour
      so non-DAG runs are unaffected.
      """

      model_config = SettingsConfigDict(
          env_prefix="AIPERF_DAG_",
          extra="ignore",
      )

      FAIL_FAST: bool = Field(
          default=False,
          description="When True, abort the whole run on the first DAG child "
          "error (cancel pending siblings, raise to PhaseRunner, terminate "
          "phase). Default False — the orchestrator counts the error in "
          "BranchStats.errors, releases the join slot, drains pending "
          "siblings, and continues the run. Set via "
          "AIPERF_DAG_FAIL_FAST=1 for strict CI assertions.",
      )


  DagSettings = _DagSettings
  ```

  Then add the module-level singleton export at the bottom of the file alongside the existing settings exports:

  ```python
  dag_settings = _DagSettings()
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/environment.py tests/unit/common/test_dag_settings.py
  git commit -s -m "$(cat <<'EOF'
  feat(env): add _DagSettings.FAIL_FAST env var (AIPERF_DAG_FAIL_FAST)

  Introduces the AIPERF_DAG_FAIL_FAST environment variable. Default
  False (count errors in BranchStats and continue); set to 1 to abort
  the run on first child error, cancel pending siblings, and terminate
  the phase.

  Used by BranchOrchestrator (Task 17) to gate child-error handling
  between count-and-continue and fail-fast modes.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 4: `dag_jsonl_models.py` — `DagSpawn` and helpers

**Files:**
- Create: `src/aiperf/dataset/loader/dag_jsonl_models.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_models.py`

**Audit notes:**
- Verbatim port from `inferencex-agentx-mvp` (152 lines).
- Defines `DagSpawn` (the validated record backing the `spawns:` shorthand) plus any helpers used by the loader (`_TopologyState`, branch-id minting, etc. — verify by reading the source).
- No drift adjustments needed; the file is loader-internal and references only spec-In-Scope models (`Conversation`, `Turn`, `ConversationBranchInfo`, `TurnPrerequisite`).

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.dataset.loader.dag_jsonl_models import DagSpawn


  class TestDagSpawn:
      def test_minimal_construction(self):
          s = DagSpawn(target="child-session-1")
          assert s.target == "child-session-1"

      def test_round_trip(self):
          s = DagSpawn(target="t1")
          dumped = s.model_dump()
          rebuilt = DagSpawn.model_validate(dumped)
          assert rebuilt == s

      def test_target_is_required(self):
          with pytest.raises(Exception):  # pydantic.ValidationError
              DagSpawn()  # type: ignore[call-arg]
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/dataset/loader/test_dag_jsonl_models.py -v`

  Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

  Verbatim port from `inferencex-agentx-mvp`:

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/dataset/loader/dag_jsonl_models.py > src/aiperf/dataset/loader/dag_jsonl_models.py
  ```

  Then verify with `cat src/aiperf/dataset/loader/dag_jsonl_models.py`. If the file imports anything not on `ajc/dag5` HEAD (run `uv run python -c "from aiperf.dataset.loader.dag_jsonl_models import DagSpawn"`), patch the import line — Plan 1 already shipped `ConversationBranchMode`, `Turn`, `Conversation`, `ConversationBranchInfo`, `TurnPrerequisite` so the imports should resolve.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/dataset/loader/dag_jsonl_models.py tests/unit/dataset/loader/test_dag_jsonl_models.py
  git commit -s -m "$(cat <<'EOF'
  feat(loaders): add dag_jsonl_models with DagSpawn record

  Internal models for the dag_jsonl loader (Task 5). DagSpawn is the
  validated record backing the `spawns:` shorthand at conversation root;
  helpers cover topology-walk state and branch-id minting.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 5: `dag_jsonl` loader

**Files:**
- Create: `src/aiperf/dataset/loader/dag_jsonl.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_topology_pathological.py`
- Test fixtures: copy `tests/fixtures/dag/` from `inferencex-agentx-mvp`

**Audit notes:**
- Verbatim port from `inferencex-agentx-mvp` (499 lines). Includes `forks:` shorthand, `spawns:` shorthand, `_inline_pre_session_spawns` topology rewrite, cycle detection, multi-parent rejection, non-terminal-fork rejection, dangling-prereq rejection, agent-depth stamping via topology walk.
- Spec §a explicitly requires keeping the `isinstance(data, dict)` guard in `can_load` (current's lines 122–127 — note: this was on the `inferencex-agentx-mvp` version of the file; dag4 lacks the guard).
- The loader produces `Conversation` objects whose `branches:` carry `mode=FORK` or `mode=SPAWN` and whose SPAWN branches set `dispatch_timing="pre"`.
- Plan 1 Task 4 already stamped `Conversation.agent_depth` via the data-model walk; this loader populates the actual values from the parsed file's topology.

- [ ] **Step 1: Write the failing test**

  Three test files. Start with `test_dag_jsonl.py` (the happy-path + can_load suite):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest

  from aiperf.common.config.user_config import UserConfig
  from aiperf.common.enums import ConversationBranchMode
  from aiperf.dataset.loader.dag_jsonl import DagJsonlDatasetLoader


  def _write_jsonl(path: Path, records: list[dict]) -> None:
      with open(path, "wb") as f:
          for r in records:
              f.write(orjson.dumps(r))
              f.write(b"\n")


  @pytest.fixture
  def simple_fork_file(tmp_path: Path) -> Path:
      p = tmp_path / "simple_fork.dag.jsonl"
      _write_jsonl(p, [
          {
              "session_id": "root",
              "turns": [
                  {
                      "messages": [{"role": "user", "content": "ask"}],
                      "forks": ["child-a", "child-b"],
                  }
              ],
          },
          {
              "session_id": "child-a",
              "turns": [{"messages": [{"role": "user", "content": "child-a"}]}],
          },
          {
              "session_id": "child-b",
              "turns": [{"messages": [{"role": "user", "content": "child-b"}]}],
          },
      ])
      return p


  @pytest.fixture
  def spawn_root_file(tmp_path: Path) -> Path:
      p = tmp_path / "spawn_root.dag.jsonl"
      _write_jsonl(p, [
          {
              "session_id": "root",
              "spawns": [{"target": "pre-warm"}],
              "turns": [{"messages": [{"role": "user", "content": "main"}]}],
          },
          {
              "session_id": "pre-warm",
              "turns": [{"messages": [{"role": "user", "content": "warmup"}]}],
          },
      ])
      return p


  class TestCanLoad:
      def test_dict_with_session_id_and_turns_accepted(self):
          assert DagJsonlDatasetLoader.can_load(
              data={"session_id": "x", "turns": [{"messages": []}]}
          )

      def test_non_dict_first_record_rejected(self):
          # spec §a explicit: keep the isinstance(data, dict) guard.
          assert not DagJsonlDatasetLoader.can_load(data=["not", "a", "dict"])
          assert not DagJsonlDatasetLoader.can_load(data="bare-string")

      def test_messages_only_record_rejected(self):
          # Owned by raw_payload / inputs_json.
          assert not DagJsonlDatasetLoader.can_load(
              data={"messages": [{"role": "user", "content": "hi"}]}
          )


  class TestSimpleFork:
      def test_load_produces_root_plus_two_children(
          self, simple_fork_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=simple_fork_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          ids = {c.conversation_id for c in conversations}
          assert ids == {"root", "child-a", "child-b"}

      def test_root_branches_are_fork_mode(
          self, simple_fork_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=simple_fork_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          root = next(c for c in conversations if c.conversation_id == "root")
          assert root.turns[0].branches is not None
          assert all(
              b.mode is ConversationBranchMode.FORK for b in root.turns[0].branches
          )

      def test_agent_depth_stamped_via_topology_walk(
          self, simple_fork_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=simple_fork_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          by_id = {c.conversation_id: c for c in conversations}
          assert by_id["root"].agent_depth == 0
          assert by_id["child-a"].agent_depth == 1
          assert by_id["child-b"].agent_depth == 1

      def test_has_forks_stamped_on_parent_turn(
          self, simple_fork_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=simple_fork_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          root = next(c for c in conversations if c.conversation_id == "root")
          # The parent turn carries forks → its TurnMetadata should have has_forks True.
          meta = root.metadata().turns[0]
          assert meta.has_forks is True


  class TestSpawnsShorthand:
      def test_spawn_marked_pre_dispatch(
          self, spawn_root_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=spawn_root_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          by_id = {c.conversation_id: c for c in conversations}
          # The spawn target should exist as its own conversation, with one of
          # its branches (or the root's spawns) flagged dispatch_timing="pre".
          root = by_id["root"]
          spawn_branches = [
              b for t in root.turns for b in (t.branches or [])
              if b.mode is ConversationBranchMode.SPAWN
          ]
          assert any(b.dispatch_timing == "pre" for b in spawn_branches)

      def test_spawn_target_is_loadable_as_separate_conversation(
          self, spawn_root_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=spawn_root_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          ids = {c.conversation_id for c in conversations}
          assert "pre-warm" in ids
  ```

  Then `test_dag_jsonl_topology_pathological.py` (the reject-path suite):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest

  from aiperf.common.config.user_config import UserConfig
  from aiperf.dataset.loader.dag_jsonl import DagJsonlDatasetLoader


  def _write_jsonl(path: Path, records: list[dict]) -> None:
      with open(path, "wb") as f:
          for r in records:
              f.write(orjson.dumps(r))
              f.write(b"\n")


  @pytest.fixture
  def cyclic_file(tmp_path: Path) -> Path:
      p = tmp_path / "cyclic.dag.jsonl"
      _write_jsonl(p, [
          {"session_id": "a", "turns": [{"messages": [{"role": "user", "content": "a"}], "forks": ["b"]}]},
          {"session_id": "b", "turns": [{"messages": [{"role": "user", "content": "b"}], "forks": ["a"]}]},
      ])
      return p


  @pytest.fixture
  def multi_parent_file(tmp_path: Path) -> Path:
      p = tmp_path / "multi_parent.dag.jsonl"
      _write_jsonl(p, [
          {"session_id": "p1", "turns": [{"messages": [{"role": "user", "content": "p1"}], "forks": ["c"]}]},
          {"session_id": "p2", "turns": [{"messages": [{"role": "user", "content": "p2"}], "forks": ["c"]}]},
          {"session_id": "c", "turns": [{"messages": [{"role": "user", "content": "c"}]}]},
      ])
      return p


  @pytest.fixture
  def non_terminal_fork_file(tmp_path: Path) -> Path:
      p = tmp_path / "non_terminal_fork.dag.jsonl"
      _write_jsonl(p, [
          {
              "session_id": "p",
              "turns": [
                  # Fork in a non-terminal turn — followed by another turn in the same session.
                  {"messages": [{"role": "user", "content": "t1"}], "forks": ["c"]},
                  {"messages": [{"role": "user", "content": "t2"}]},
              ],
          },
          {"session_id": "c", "turns": [{"messages": [{"role": "user", "content": "c"}]}]},
      ])
      return p


  @pytest.fixture
  def dangling_fork_file(tmp_path: Path) -> Path:
      p = tmp_path / "dangling.dag.jsonl"
      _write_jsonl(p, [
          {"session_id": "p", "turns": [{"messages": [{"role": "user", "content": "p"}], "forks": ["nonexistent"]}]},
      ])
      return p


  @pytest.fixture
  def dangling_prereq_file(tmp_path: Path) -> Path:
      p = tmp_path / "dangling_prereq.dag.jsonl"
      _write_jsonl(p, [
          {
              "session_id": "p",
              "turns": [
                  {
                      "messages": [{"role": "user", "content": "p"}],
                      "prerequisites": [{"kind": "session_complete", "target": "ghost"}],
                  }
              ],
          },
      ])
      return p


  class TestRejectPaths:
      def test_cycle_rejected(self, cyclic_file: Path, default_user_config: UserConfig):
          loader = DagJsonlDatasetLoader(filename=cyclic_file, user_config=default_user_config)
          with pytest.raises(Exception, match="(?i)cycle"):
              loader.convert_to_conversations(loader.load_dataset())

      def test_multi_parent_rejected(self, multi_parent_file: Path, default_user_config: UserConfig):
          loader = DagJsonlDatasetLoader(filename=multi_parent_file, user_config=default_user_config)
          with pytest.raises(Exception, match="(?i)multi"):
              loader.convert_to_conversations(loader.load_dataset())

      def test_non_terminal_fork_rejected(
          self, non_terminal_fork_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=non_terminal_fork_file, user_config=default_user_config
          )
          with pytest.raises(Exception, match="(?i)terminal|non-terminal"):
              loader.convert_to_conversations(loader.load_dataset())

      def test_dangling_fork_rejected(
          self, dangling_fork_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=dangling_fork_file, user_config=default_user_config
          )
          with pytest.raises(Exception, match="(?i)nonexistent|undefined|dangling"):
              loader.convert_to_conversations(loader.load_dataset())

      def test_dangling_prereq_rejected(
          self, dangling_prereq_file: Path, default_user_config: UserConfig
      ):
          loader = DagJsonlDatasetLoader(
              filename=dangling_prereq_file, user_config=default_user_config
          )
          with pytest.raises(Exception, match="(?i)ghost|prereq|undefined"):
              loader.convert_to_conversations(loader.load_dataset())
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/dataset/loader/test_dag_jsonl.py tests/unit/dataset/loader/test_dag_jsonl_topology_pathological.py -v`

  Expected: FAIL — `aiperf.dataset.loader.dag_jsonl` does not exist.

- [ ] **Step 3: Write minimal implementation**

  Verbatim port from `inferencex-agentx-mvp`, then copy fixtures:

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/dataset/loader/dag_jsonl.py > src/aiperf/dataset/loader/dag_jsonl.py
  # Copy the entire DAG fixture directory from inferencex.
  git checkout ajc/inferencex-agentx-mvp -- tests/fixtures/dag/
  # Pull the dag4-only multi-root fixture too — required by Task 30.
  git checkout ajc/dag4 -- tests/fixtures/dag/multi_root_single_turn.dag.jsonl
  ```

  If any imports in `dag_jsonl.py` reference Weka / agentic-replay / `Turn.reset_context` / cache-bust (per spec §"Out-of-Scope"), patch them out. Run `uv run python -c "from aiperf.dataset.loader.dag_jsonl import DagJsonlDatasetLoader"` to flush import errors and fix one-by-one.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/dataset/loader/dag_jsonl.py tests/unit/dataset/loader/test_dag_jsonl.py tests/unit/dataset/loader/test_dag_jsonl_topology_pathological.py tests/fixtures/dag/
  git commit -s -m "$(cat <<'EOF'
  feat(loaders): add dag_jsonl loader for DAG-shaped benchmarks

  JSONL loader where each line is a Conversation. Supports forks: and
  spawns: shorthand at the turn level; runs a topology walk that stamps
  Conversation.agent_depth and TurnMetadata.has_forks; rejects cycles,
  multi-parent fan-in, non-terminal forks, and dangling fork / prereq
  references at parse time so DAG errors surface before the run starts.

  can_load preserves the isinstance(data, dict) guard so non-dict probes
  fall through to the next loader cleanly.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 6: `dag_jsonl` plugin registry entry

**Files:**
- Modify: `src/aiperf/plugin/plugins.yaml`
- Auto-regenerate: `src/aiperf/common/enums/plugins.py` and overload artifacts via `make generate-all-plugin-files`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from aiperf.common import plugins
  from aiperf.common.enums import CustomDatasetType, PluginType


  class TestDagJsonlRegistered:
      def test_enum_member_present(self):
          assert "dag_jsonl" in {m.value for m in CustomDatasetType}

      def test_loader_class_resolves(self):
          cls = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, "dag_jsonl")
          assert cls is not None
          assert cls.__name__ == "DagJsonlDatasetLoader"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/test_plugin_registry.py -v -k dag_jsonl` (the test file is already present from Plan 1; if not, run via the path of the new test file).

  Expected: FAIL — `dag_jsonl` not in registry.

- [ ] **Step 3: Write minimal implementation**

  Edit `src/aiperf/plugin/plugins.yaml` and add under the `loaders:` section (the exact YAML key may be `loaders:` or nested under a top-level grouping — match Plan 1's `inputs_json` / `raw_payload` entries):

  ```yaml
    - name: dag_jsonl
      class: aiperf.dataset.loader.dag_jsonl.DagJsonlDatasetLoader
      description: |
        DAG-shaped benchmark loader. Each JSONL line is a Conversation;
        supports forks: and spawns: shorthand at the turn level;
        topology walk validates and stamps agent_depth.
      metadata:
        is_trace_dataset: false
  ```

  Then regenerate the plugin artifacts and validate:

  ```bash
  make generate-all-plugin-files
  make validate-plugin-schemas
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/plugin/plugins.yaml src/aiperf/common/enums/ tests/unit/common/test_plugin_registry.py
  git commit -s -m "$(cat <<'EOF'
  feat(plugins): register dag_jsonl loader

  Adds the dag_jsonl loader to the plugin registry so --custom-dataset-type
  dag_jsonl resolves at config-load. Regenerates the plugin enum and
  overload artifacts.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 7: `UserSession.is_fork_parent` stamped at `create_and_store`

**Files:**
- Modify: `src/aiperf/workers/session_manager.py`
- Test: `tests/unit/workers/test_session_manager_fork_parent.py`

**Audit notes:**
- Compare `dag4_session_manager.py` against `main_session_manager.py`. The dag4 fix stamps `is_fork_parent` at `create_and_store` time so it survives PAYLOAD_BYTES round-trips where `conversation.branches` is dropped.
- main has no `is_fork_parent` field on `UserSession` at all — this task adds the field plus the stamp.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import ConversationBranchMode
  from aiperf.common.models.dataset_models import (
      Conversation,
      ConversationBranchInfo,
      Turn,
  )
  from aiperf.workers.session_manager import SessionManager, UserSession


  def _conv_with_fork(session_id: str = "root") -> Conversation:
      child_branch = ConversationBranchInfo(
          mode=ConversationBranchMode.FORK,
          target_turn_ids=["child-x"],
      )
      return Conversation(
          conversation_id=session_id,
          turns=[
              Turn(
                  messages=[{"role": "user", "content": "ask"}],
                  branches=[child_branch],
              )
          ],
      )


  def _conv_no_fork(session_id: str = "linear") -> Conversation:
      return Conversation(
          conversation_id=session_id,
          turns=[Turn(messages=[{"role": "user", "content": "ask"}])],
      )


  class TestIsForkParent:
      def test_stamped_true_for_forking_conversation(self):
          mgr = SessionManager()
          conv = _conv_with_fork()
          session = mgr.create_and_store(conv)
          assert session.is_fork_parent is True

      def test_stamped_false_for_linear_conversation(self):
          mgr = SessionManager()
          conv = _conv_no_fork()
          session = mgr.create_and_store(conv)
          assert session.is_fork_parent is False

      def test_survives_payload_bytes_round_trip(self):
          mgr = SessionManager()
          conv = _conv_with_fork()
          session = mgr.create_and_store(conv)
          # Simulate the PAYLOAD_BYTES path that drops conversation.branches.
          session.conversation.turns[0].branches = None
          assert session.is_fork_parent is True  # still stamped
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/workers/test_session_manager_fork_parent.py -v`

  Expected: FAIL — `is_fork_parent` doesn't exist on `UserSession`.

- [ ] **Step 3: Write minimal implementation**

  The clean source is `dag4_session_manager.py`. Diff against main and apply:

  ```bash
  diff /tmp/dag5plan/main_session_manager.py /tmp/dag5plan/dag4_session_manager.py
  ```

  Apply the diff hunks: add the `is_fork_parent: bool` field to `UserSession`, add the `_compute_is_fork_parent(conv) -> bool` helper, and call it from `create_and_store` to stamp the field.

  ```bash
  # If the dag4 file has no out-of-scope drift (no cache-bust, no Weka, no agentic-replay), port verbatim:
  git show ajc/dag4:src/aiperf/workers/session_manager.py > src/aiperf/workers/session_manager.py
  ```

  After write, audit:

  ```bash
  grep -nE "cache_bust|weka|agentic_replay|reset_context" src/aiperf/workers/session_manager.py
  ```

  Expected output: empty. If any matches, scrub them.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/workers/session_manager.py tests/unit/workers/test_session_manager_fork_parent.py
  git commit -s -m "$(cat <<'EOF'
  fix(workers): stamp UserSession.is_fork_parent at create_and_store time

  Computes the fork-parent flag once at session creation from the
  conversation's branches and stores it on UserSession, instead of
  recomputing it lazily from `conversation.branches` on every read.
  This is required because the PAYLOAD_BYTES context-mode path drops
  the branches field for wire-size, which would otherwise lose the
  flag and break sticky-routing eviction logic.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 8: Worker FORK pin refcount on `UserSession`

**Files:**
- Modify: `src/aiperf/workers/session_manager.py` (refcount field + bump/decrement helpers)
- Modify: `src/aiperf/workers/worker.py` (call sites: bump on child seed, decrement on child join, evict-when-zero)
- Test: `tests/unit/workers/test_session_fork_refcount.py`

**Audit notes:**
- inferencex's worker.py is 1171 lines (vs main's 716) — it carries cache-bust, Weka delta-context, agentic-replay logic, all explicitly out of scope. The FORK pin refcount is the only piece to lift; do NOT verbatim-port worker.py.
- The dag4 worker has the cache-bust marker injection too, also out of scope. So this task hand-writes the refcount additions.
- The refcount lives on `UserSession` (added here) and is incremented by the orchestrator-driven child-seed path (Task 11+) and decremented when a child join arrives (Task 11+). Eviction in `SessionManager.evict_if_unpinned` checks both `is_fork_parent` and `refcount == 0`.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import ConversationBranchMode
  from aiperf.common.models.dataset_models import (
      Conversation,
      ConversationBranchInfo,
      Turn,
  )
  from aiperf.workers.session_manager import SessionManager


  def _conv_with_n_forks(n: int, session_id: str = "root") -> Conversation:
      branches = [
          ConversationBranchInfo(
              mode=ConversationBranchMode.FORK, target_turn_ids=[f"child-{i}"]
          )
          for i in range(n)
      ]
      return Conversation(
          conversation_id=session_id,
          turns=[
              Turn(
                  messages=[{"role": "user", "content": "ask"}],
                  branches=branches,
              )
          ],
      )


  class TestForkRefcount:
      def test_default_refcount_zero(self):
          mgr = SessionManager()
          session = mgr.create_and_store(_conv_with_n_forks(2))
          assert session.fork_refcount == 0

      def test_increment_per_child_seed(self):
          mgr = SessionManager()
          session = mgr.create_and_store(_conv_with_n_forks(3))
          mgr.pin_for_fork_child(session.session_id)
          mgr.pin_for_fork_child(session.session_id)
          mgr.pin_for_fork_child(session.session_id)
          assert session.fork_refcount == 3

      def test_decrement_on_child_join(self):
          mgr = SessionManager()
          session = mgr.create_and_store(_conv_with_n_forks(2))
          mgr.pin_for_fork_child(session.session_id)
          mgr.pin_for_fork_child(session.session_id)
          mgr.release_fork_child(session.session_id)
          assert session.fork_refcount == 1

      def test_evict_only_when_refcount_zero(self):
          mgr = SessionManager()
          session = mgr.create_and_store(_conv_with_n_forks(2))
          mgr.pin_for_fork_child(session.session_id)
          # Try to evict while children still pending.
          mgr.evict_if_unpinned(session.session_id)
          assert mgr.has(session.session_id)
          mgr.release_fork_child(session.session_id)
          mgr.evict_if_unpinned(session.session_id)
          assert not mgr.has(session.session_id)
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/workers/test_session_fork_refcount.py -v`

  Expected: FAIL — `fork_refcount`, `pin_for_fork_child`, `release_fork_child`, `evict_if_unpinned` missing.

- [ ] **Step 3: Write minimal implementation**

  Edit `src/aiperf/workers/session_manager.py`:

  1. Add to `UserSession`:

     ```python
         fork_refcount: int = Field(
             default=0,
             description="Refcount of pending DAG-FORK children that pin this "
             "session in the manager so its history is still resident when "
             "each child credit dispatches. Incremented at child-seed time "
             "by `pin_for_fork_child`; decremented on child join by "
             "`release_fork_child`. Eviction (`evict_if_unpinned`) is a no-op "
             "while this is non-zero.",
         )
     ```

  2. Add to `SessionManager`:

     ```python
         def pin_for_fork_child(self, session_id: str) -> None:
             """Increment the fork-pin refcount on the session."""
             session = self._sessions.get(session_id)
             if session is None:
                 raise KeyError(f"No session {session_id} to pin")
             session.fork_refcount += 1

         def release_fork_child(self, session_id: str) -> None:
             """Decrement the fork-pin refcount on the session."""
             session = self._sessions.get(session_id)
             if session is None:
                 return
             session.fork_refcount = max(0, session.fork_refcount - 1)

         def evict_if_unpinned(self, session_id: str) -> None:
             """Evict the session if its fork refcount has reached zero."""
             session = self._sessions.get(session_id)
             if session is None:
                 return
             if session.fork_refcount > 0:
                 return
             del self._sessions[session_id]

         def has(self, session_id: str) -> bool:
             return session_id in self._sessions
     ```

  Worker call-site wiring lives in Task 11+ (where the orchestrator dispatches child credits). For this task only the SessionManager API surfaces are added.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/workers/session_manager.py tests/unit/workers/test_session_fork_refcount.py
  git commit -s -m "$(cat <<'EOF'
  feat(workers): add fork pin refcount to UserSession / SessionManager

  Adds fork_refcount on UserSession plus pin_for_fork_child /
  release_fork_child / evict_if_unpinned on SessionManager. Eviction
  is now refcount-gated so a parent session stays resident across N
  forks and is only freed when the last child joins.

  This is the storage half of FORK pin refcounting; the worker's
  orchestrator-driven call sites (bump on child seed, decrement on
  child join) land in the orchestrator wiring task.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 9: `inference_client._enrich_request_record` propagates RecordContext

**Files:**
- Modify: `src/aiperf/workers/inference_client.py`
- Test: `tests/unit/workers/test_inference_client_enrich_record.py`

**Audit notes:**
- `inferencex_inference_client.py` is 276 lines vs main's 211. The diff that matters: `_enrich_request_record` builds a pure `RecordContext` from the worker's `RequestInfo` (downcasting via `model_dump` on the slim subset of fields) and assigns it to `record.request_info`, ensuring the worker-only transport extras (`model_endpoint`, `turns`, headers) never cross ZMQ.
- The cache-bust enrichment fields are out of scope — drop them.
- `agent_depth` and `parent_correlation_id` flow from the originating `Credit` → `RequestInfo` → `RecordContext` (the inheritance is automatic since `RequestInfo` extends `RecordContext` from Task 2; the downcast just preserves the existing values).

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import CreditPhase
  from aiperf.common.models.record_models import (
      RecordContext,
      RequestInfo,
      RequestRecord,
  )
  from aiperf.workers.inference_client import InferenceClient


  def _make_request_info(**overrides) -> RequestInfo:
      from aiperf.common.models.model_endpoint_info import ModelEndpointInfo

      defaults = dict(
          credit_num=0,
          credit_phase=CreditPhase.PROFILING,
          conversation_id="c",
          turn_index=0,
          x_request_id="r",
          x_correlation_id="x",
          agent_depth=2,
          parent_correlation_id="root",
          model_endpoint=ModelEndpointInfo.model_construct(),  # bare shell
          turns=[],
      )
      defaults.update(overrides)
      return RequestInfo(**defaults)


  class TestEnrichRequestRecord:
      def test_record_context_replaces_request_info_on_record(self):
          ri = _make_request_info()
          record = RequestRecord()
          # Direct call to the static/instance helper that does the downcast.
          enriched = InferenceClient._enrich_request_record(record, ri)
          assert enriched.request_info is not None
          # Crucially: must be a pure RecordContext, NOT the full RequestInfo.
          assert type(enriched.request_info) is RecordContext

      def test_dag_fields_propagate(self):
          ri = _make_request_info(agent_depth=3, parent_correlation_id="p")
          record = RequestRecord()
          enriched = InferenceClient._enrich_request_record(record, ri)
          assert enriched.request_info.agent_depth == 3
          assert enriched.request_info.parent_correlation_id == "p"

      def test_transport_extras_dropped(self):
          ri = _make_request_info()
          record = RequestRecord()
          enriched = InferenceClient._enrich_request_record(record, ri)
          # The downcast strips model_endpoint / turns / endpoint_headers.
          assert not hasattr(enriched.request_info, "model_endpoint") or \
              enriched.request_info.model_dump().get("model_endpoint") is None
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/workers/test_inference_client_enrich_record.py -v`

  Expected: FAIL — `_enrich_request_record` either doesn't exist or doesn't perform the downcast.

- [ ] **Step 3: Write minimal implementation**

  Edit `src/aiperf/workers/inference_client.py`. Add the static helper near the `RequestRecord`-creation site:

  ```python
      @staticmethod
      def _enrich_request_record(
          record: RequestRecord, request_info: RequestInfo
      ) -> RequestRecord:
          """Attach a slim RecordContext (downcast from RequestInfo) to the
          record before the ZMQ hop to the record processor.

          The full RequestInfo carries transport-only extras (model_endpoint,
          turns, headers, drop_perf_ns, etc.) that the record-processor
          pipeline never reads; downcasting saves ~500-900 bytes per record
          at high throughput.
          """
          ctx_field_names = set(RecordContext.model_fields.keys())
          ri_dump = request_info.model_dump(include=ctx_field_names)
          record.request_info = RecordContext.model_validate(ri_dump)
          return record
  ```

  Then update the existing call site in `inference_client` that previously assigned `record.request_info = request_info` (find via `grep -n "request_info\s*=" src/aiperf/workers/inference_client.py`) to call the new helper:

  ```python
      self._enrich_request_record(record, request_info)
  ```

  Verify by reading inferencex's version for the exact placement: `git show ajc/inferencex-agentx-mvp:src/aiperf/workers/inference_client.py | grep -n "_enrich_request_record"`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/workers/inference_client.py tests/unit/workers/test_inference_client_enrich_record.py
  git commit -s -m "$(cat <<'EOF'
  feat(workers): downcast RequestInfo to RecordContext before ZMQ hop

  Adds InferenceClient._enrich_request_record which builds a pure
  RecordContext (the slim post-transport context) from the worker's
  RequestInfo and assigns it to RequestRecord.request_info. The
  transport-only extras (model_endpoint, full turns list, headers,
  drop_perf_ns) never cross the ZMQ hop to the record processor.

  agent_depth and parent_correlation_id are preserved on the slim
  RecordContext so DAG provenance flows end-to-end through metrics.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 10: `ConversationSource.start_branch_child` and `start_pre_session_child`

**Files:**
- Modify: `src/aiperf/timing/conversation_source.py`
- Test: `tests/unit/timing/test_conversation_source_dag.py`

**Audit notes:**
- `inferencex_conversation_source.py` is 239 lines vs main's 114. The two new public methods are at lines 148–183 (`start_branch_child`) and 184–218 (`start_pre_session_child`).
- Both methods build child `SampledSession` instances. `start_branch_child` shares the parent's `session_id` (FORK semantics — child inherits parent's history); `start_pre_session_child` mints a fresh `session_id` (SPAWN semantics — child has no parent context).
- Both copy sticky routing via `parent_correlation_id`.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.models.dataset_models import (
      Conversation,
      Turn,
  )
  from aiperf.timing.conversation_source import ConversationSource, SampledSession


  @pytest.fixture
  def parent_conv() -> Conversation:
      return Conversation(
          conversation_id="root-session",
          turns=[Turn(messages=[{"role": "user", "content": "ask"}])],
      )


  @pytest.fixture
  def source_with_parent(parent_conv: Conversation) -> ConversationSource:
      # Construct a minimal ConversationSource indexing the parent + a child.
      child = Conversation(
          conversation_id="child-session",
          turns=[Turn(messages=[{"role": "user", "content": "child"}])],
          agent_depth=1,
      )
      return ConversationSource(conversations=[parent_conv, child])


  class TestStartBranchChild:
      def test_inherits_parent_session_id(self, source_with_parent: ConversationSource):
          child = source_with_parent.start_branch_child(
              parent_correlation_id="parent-corr",
              parent_session_id="root-session",
              target_turn_id="child-session",
          )
          assert isinstance(child, SampledSession)
          assert child.routing_key == "root-session"  # FORK shares parent's slot

      def test_carries_parent_correlation_id(self, source_with_parent: ConversationSource):
          child = source_with_parent.start_branch_child(
              parent_correlation_id="parent-corr",
              parent_session_id="root-session",
              target_turn_id="child-session",
          )
          # The first turn out of the child should carry parent_correlation_id.
          tts = child.build_first_turn()
          assert tts.parent_correlation_id == "parent-corr"

      def test_agent_depth_propagates_from_loaded_conversation(
          self, source_with_parent: ConversationSource
      ):
          child = source_with_parent.start_branch_child(
              parent_correlation_id="parent-corr",
              parent_session_id="root-session",
              target_turn_id="child-session",
          )
          tts = child.build_first_turn()
          assert tts.agent_depth == 1


  class TestStartPreSessionChild:
      def test_fresh_session_id(self, source_with_parent: ConversationSource):
          child = source_with_parent.start_pre_session_child(
              target_turn_id="child-session",
          )
          assert isinstance(child, SampledSession)
          # Pre-session children get their own routing slot.
          assert child.routing_key != "root-session"
          assert child.routing_key  # non-empty

      def test_no_parent_correlation_id(self, source_with_parent: ConversationSource):
          child = source_with_parent.start_pre_session_child(
              target_turn_id="child-session",
          )
          tts = child.build_first_turn()
          assert tts.parent_correlation_id is None
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_conversation_source_dag.py -v`

  Expected: FAIL — `start_branch_child` and `start_pre_session_child` missing on `ConversationSource`.

- [ ] **Step 3: Write minimal implementation**

  Diff `inferencex_conversation_source.py` against `main_conversation_source.py` and apply:

  1. Add `routing_key` property to `SampledSession` if not already there (inferencex line 59).
  2. Add `build_turn_at_index(turn_index)` to `SampledSession` (inferencex line 85) so child credits can advance.
  3. Add `start_branch_child(parent_correlation_id, parent_session_id, target_turn_id) -> SampledSession` to `ConversationSource` (inferencex lines 148–183).
  4. Add `start_pre_session_child(target_turn_id) -> SampledSession` (inferencex lines 184–218).

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/timing/conversation_source.py > /tmp/dag5_conversation_source.py
  diff src/aiperf/timing/conversation_source.py /tmp/dag5_conversation_source.py
  ```

  Apply the inferencex content verbatim — the file has no out-of-scope dependencies (no cache-bust, no Weka, no agentic-replay).

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/timing/conversation_source.py > src/aiperf/timing/conversation_source.py
  grep -nE "cache_bust|weka|agentic_replay|reset_context" src/aiperf/timing/conversation_source.py
  ```

  Expected output: empty.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/timing/conversation_source.py tests/unit/timing/test_conversation_source_dag.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): add ConversationSource.start_branch_child and start_pre_session_child

  Adds two new SampledSession builders on ConversationSource:

  - start_branch_child: FORK semantics — child shares parent's
    session_id, inheriting the parent's stored history. Sticky routing
    via parent_correlation_id flows through to TurnToSend.

  - start_pre_session_child: SPAWN semantics — child gets a fresh
    session_id and a free routing slot. No parent_correlation_id.

  These are the entry points the BranchOrchestrator calls when
  dispatching DAG children.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 11: `BranchOrchestrator` core (FORK-only subset)

**Files:**
- Create: `src/aiperf/timing/branch_orchestrator.py`
- Create: helper module(s) if the verbatim port trips ergonomics linter (file size, complexity, nesting depth) — the inferencex source is 1042 lines; if `make check-ergonomics` flags it, split off `_branch_orchestrator_state.py` (PrereqState, PendingBranchJoin, ChildJoinEntry dataclasses) and `_branch_orchestrator_helpers.py` (the marker-minting and prereq-walking helpers) into sibling modules.
- Test: `tests/unit/timing/test_branch_orchestrator.py` (the core suite, fork-only scenarios)

**Audit notes:**
- Verbatim port from `inferencex_branch_orchestrator.py`. The advanced features (fan-in, multi-gate, K-delayed-join, pre-session SPAWN, prereq walking, child-error handling) are wired in subsequent tasks (12-17) via additional test files; the core port lands the entire 1042-line module here so all hooks exist, then later tasks port the test files that exercise each feature.
- DROP every reference to `cache_bust_marker` / `cache_bust_target` / `_apply_cache_bust_*` / `agentic_replay` / `Turn.reset_context` / `weka_*` if any appear (run `grep -nE "cache_bust|weka|agentic_replay|reset_context" src/aiperf/timing/branch_orchestrator.py` after the port).
- The orchestrator references `Credit`, `TurnToSend`, `RecordContext`, `ConversationSource.start_branch_child` / `start_pre_session_child`, `SessionManager.pin_for_fork_child` / `release_fork_child`, `_DagSettings.FAIL_FAST` — all defined in Tasks 1, 2, 3, 8, 10. If any of these names diverge in the inferencex source, patch the references after the `git show` write.
- Use the inferencex `tests/unit/timing/test_branch_orchestrator.py` file verbatim for the core test (port via `git show`).

- [ ] **Step 1: Write the failing test**

  Verbatim-port the inferencex test file as the unit-test surface for this task. Read it first to confirm shape:

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator.py | head -40
  ```

  Then write it to disk in Step 1:

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator.py > tests/unit/timing/test_branch_orchestrator.py
  ```

  If the test file imports anything from `agentic_replay` / `weka` / cache-bust modules, patch those imports out — the test should only import `BranchOrchestrator`, the data models, and `Credit` / `TurnToSend`.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_branch_orchestrator.py -v`

  Expected: FAIL — `aiperf.timing.branch_orchestrator` does not exist.

- [ ] **Step 3: Write minimal implementation**

  Verbatim port:

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/timing/branch_orchestrator.py > src/aiperf/timing/branch_orchestrator.py
  grep -nE "cache_bust|weka|agentic_replay|reset_context" src/aiperf/timing/branch_orchestrator.py
  ```

  Expected output: empty. If any matches, patch them out. Common patterns to scrub:
  - `import` lines pulling cache-bust / agentic-replay / weka modules -> delete
  - References to `credit.cache_bust_marker` / `credit.cache_bust_target` -> delete the field reads
  - `if turn.reset_context:` blocks -> delete the entire block
  - Any `_apply_cache_bust_*` method or call site -> delete

  If `make check-ergonomics` flags the file (size > 800 lines or complexity > thresholds — Plan 1 hit this on Tasks 11-13), extract the dataclasses to `_branch_orchestrator_state.py` and the standalone helpers to `_branch_orchestrator_helpers.py`. Don't re-baseline the linter — split.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — the core orchestrator unit suite goes green; subsequent feature suites (Tasks 12-17) bring the full test surface online.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/timing/branch_orchestrator.py src/aiperf/timing/_branch_orchestrator_*.py tests/unit/timing/test_branch_orchestrator.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): add BranchOrchestrator (core)

  Lands the BranchOrchestrator module that owns DAG-topology
  awareness end-to-end: pending child-credit tracking, parent join
  bookkeeping, sticky-routing slot management, prereq gating,
  and child-error handling gated by AIPERF_DAG_FAIL_FAST.

  This commit lands the full module + the core test surface.
  Feature scenarios (fan-in, multi-gate, K-delayed-join,
  pre-session SPAWN, prereq walking, child-error gating) come
  online via subsequent test-port commits as the orchestrator's
  hooks are wired into request_rate / phase-runner / records-tracker.

  cache_bust marker injection, agentic_replay, Weka delta-context,
  and Turn.reset_context are intentionally NOT included; all out
  of scope per the spec.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 12: BranchOrchestrator fan-in scenario

**Files:**
- Test: `tests/unit/timing/test_branch_orchestrator_fan_in.py`

- [ ] **Step 1: Write the failing test**

  Port the inferencex fan-in suite verbatim:

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_fan_in.py > tests/unit/timing/test_branch_orchestrator_fan_in.py
  ```

  Expected behavior under test: a single parent fans out to N children; the orchestrator records N pending join slots; only when all N child credits complete does the orchestrator fire the parent join callback. Refcount on the parent session must hit zero exactly once.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_branch_orchestrator_fan_in.py -v`

  Expected: PASS already (the orchestrator landed in Task 11 with fan-in support); if FAIL, audit the patched-out lines from Task 11 — fan-in tracking is core to `BranchOrchestrator._ensure_future_join` (line 643 of inferencex source).

- [ ] **Step 3: Write minimal implementation**

  No new implementation. The orchestrator is already complete from Task 11. Step 3 is `cat src/aiperf/timing/branch_orchestrator.py | grep -n "_ensure_future_join\|_pop_future_join"` — confirm the join methods are present.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/timing/test_branch_orchestrator_fan_in.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port BranchOrchestrator fan-in scenario suite

  N-way fan-out -> fan-in: a parent forks to N children, orchestrator
  tracks N pending joins, parent-join fires only after the last
  child completes. Verifies refcount drains exactly once.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 13: BranchOrchestrator multi-gate scenario

**Files:**
- Test: `tests/unit/timing/test_branch_orchestrator_multi_gate.py`

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_multi_gate.py > tests/unit/timing/test_branch_orchestrator_multi_gate.py
  ```

  Expected behavior: a parent has multiple gating points across distinct turns; each gate blocks its dependent children until that gate satisfies. Multiple independent gate clusters operate in parallel without crosstalk.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_branch_orchestrator_multi_gate.py -v`

  Expected: PASS (orchestrator is complete); if FAIL, audit the prereq-state walk in `BranchOrchestrator._build_prereq_index` (inferencex line 254).

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/timing/test_branch_orchestrator_multi_gate.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port BranchOrchestrator multi-gate scenario suite

  Multiple gating points within one DAG: each gate clusters its
  dependent children; clusters resolve independently without crosstalk.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 14: BranchOrchestrator K-delayed-join scenario

**Files:**
- Test: `tests/unit/timing/test_branch_orchestrator_delayed.py`
- Test: `tests/unit/timing/test_branch_orchestrator_join.py`

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_delayed.py > tests/unit/timing/test_branch_orchestrator_delayed.py
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_join.py > tests/unit/timing/test_branch_orchestrator_join.py
  ```

  Expected behavior: K-delayed-join fires when >=K child credits complete (rather than waiting for all N). The `delayed` test exercises K=1, K=2, and K=N edge cases. The `join` test exercises explicit join-callback invocation order.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_branch_orchestrator_delayed.py tests/unit/timing/test_branch_orchestrator_join.py -v`

  Expected: PASS; if FAIL, the K-threshold logic lives on `PendingBranchJoin.is_satisfied` (inferencex line 168) — verify the field is present.

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/timing/test_branch_orchestrator_delayed.py tests/unit/timing/test_branch_orchestrator_join.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port BranchOrchestrator K-delayed-join + join callback suites

  K-delayed-join: parent join fires when >=K children complete (K=1,
  K=2, K=N edge cases). join: explicit join-callback invocation
  order under interleaved child completions.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 15: BranchOrchestrator prereq walking + adversarial tests

**Files:**
- Test: `tests/unit/common/test_prerequisites.py`
- Test: `tests/unit/common/test_prerequisites_adversarial.py`
- Test: `tests/unit/common/test_prereq_metadata_adversarial.py`

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/common/test_prerequisites.py > tests/unit/common/test_prerequisites.py
  git show ajc/inferencex-agentx-mvp:tests/unit/common/test_prerequisites_adversarial.py > tests/unit/common/test_prerequisites_adversarial.py
  git show ajc/inferencex-agentx-mvp:tests/unit/common/test_prereq_metadata_adversarial.py > tests/unit/common/test_prereq_metadata_adversarial.py
  ```

  Expected behavior: when a `Turn` carries `prerequisites`, the orchestrator blocks dispatch of that turn's credit until every prereq is satisfied. The adversarial suite covers race conditions, double-satisfaction, prereq-on-prereq chains, and message-bus reordering.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/test_prerequisites.py tests/unit/common/test_prerequisites_adversarial.py tests/unit/common/test_prereq_metadata_adversarial.py -v`

  Expected: PASS (PrereqState and `_build_prereq_index` are core orchestrator surface from Task 11). If any test references `Turn.reset_context` or cache-bust, patch those out per spec section "Out-of-Scope".

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/common/test_prerequisites.py tests/unit/common/test_prerequisites_adversarial.py tests/unit/common/test_prereq_metadata_adversarial.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port prereq-walking suite (happy path + adversarial)

  Prerequisites block credit dispatch until every prereq satisfies.
  Adversarial coverage: race conditions, double-satisfaction,
  prereq-on-prereq chains, message-bus reordering.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 16: BranchOrchestrator pre-session SPAWN + phase0 scenarios

**Files:**
- Test: `tests/unit/timing/test_branch_orchestrator_pre_session.py`
- Test: `tests/unit/timing/test_branch_orchestrator_phase0.py`

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_pre_session.py > tests/unit/timing/test_branch_orchestrator_pre_session.py
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_phase0.py > tests/unit/timing/test_branch_orchestrator_phase0.py
  ```

  Expected behavior:
  - Pre-session SPAWN: targets are dispatched before the credit phase opens; their `CreditCompleted` closes the prereq gate that holds the dependent root credit.
  - Phase 0 (warmup): orchestrator ignores DAG branches; only root credits dispatch.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_branch_orchestrator_pre_session.py tests/unit/timing/test_branch_orchestrator_phase0.py -v`

  Expected: PASS — `_inline_pre_session_spawns` lives in `dag_jsonl.py` (Task 5) and the orchestrator's pre-session entry point is in the Task 11 port. If FAIL, audit `start_pre_session_child` integration (Task 10).

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/timing/test_branch_orchestrator_pre_session.py tests/unit/timing/test_branch_orchestrator_phase0.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port pre-session SPAWN + phase0 suites for BranchOrchestrator

  Pre-session SPAWN: orchestrator dispatches spawn targets before the
  credit phase opens; their CreditCompleted closes the gating prereq.
  Phase 0 (warmup): orchestrator ignores DAG topology; only roots dispatch.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 17: BranchOrchestrator child-error handling + AIPERF_DAG_FAIL_FAST

**Files:**
- Test: `tests/unit/timing/test_branch_orchestrator_adversarial.py`
- Test: `tests/unit/timing/test_branch_orchestrator_adversarial_full.py`

**Audit notes:**
- Both suites cover the orchestrator's child-error handling. Per spec section "Error Handling - Runtime - child errors":
  - Default: count the error in `BranchStats.errors`, release the join slot, drain pending siblings, continue.
  - `AIPERF_DAG_FAIL_FAST=1`: cancel pending siblings of the same parent, raise to PhaseRunner, terminate phase.

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_adversarial.py > tests/unit/timing/test_branch_orchestrator_adversarial.py
  git show ajc/inferencex-agentx-mvp:tests/unit/timing/test_branch_orchestrator_adversarial_full.py > tests/unit/timing/test_branch_orchestrator_adversarial_full.py
  ```

  Both files exercise: child error -> BranchStats.errors increment; pending sibling drain; FAIL_FAST=1 cancels siblings and raises. Verify the suite reads `_DagSettings.FAIL_FAST` via `monkeypatch.setenv("AIPERF_DAG_FAIL_FAST", "1")` and reloads the settings module — if the inferencex tests use a different env-var or a hardcoded constant (the inferencex branch carried `_AgentXSettings`), patch the import to use Plan 2's `_DagSettings` instead.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_branch_orchestrator_adversarial.py tests/unit/timing/test_branch_orchestrator_adversarial_full.py -v`

  Expected: PASS if the orchestrator handles `_DagSettings.FAIL_FAST` correctly. If FAIL on a missing env-var or an `_AgentXSettings`-shaped constant, fix the inferencex port: replace any `_AgentXSettings` reference with `_DagSettings` and `AIPERF_AGENTX_*` with `AIPERF_DAG_*`.

- [ ] **Step 3: Write minimal implementation**

  Audit `branch_orchestrator.py`:

  ```bash
  grep -nE "AGENTX|_AgentXSettings|fail_fast|FAIL_FAST" src/aiperf/timing/branch_orchestrator.py
  ```

  Replace any `_AgentXSettings` reference with `_DagSettings`. The runtime branch is straightforward: when child error fires, check `dag_settings.FAIL_FAST` — if True, cancel pending siblings via `_cancel_pending_for_parent(parent_correlation_id)` and re-raise; if False, increment `BranchStats.errors`, release the join slot, continue.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/timing/test_branch_orchestrator_adversarial.py tests/unit/timing/test_branch_orchestrator_adversarial_full.py src/aiperf/timing/branch_orchestrator.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): wire BranchOrchestrator child-error handling to AIPERF_DAG_FAIL_FAST

  Default behavior: count error in BranchStats.errors, release the join
  slot, drain pending siblings, continue.

  AIPERF_DAG_FAIL_FAST=1: cancel pending siblings of the same parent,
  raise to PhaseRunner, terminate phase.

  Replaces any inferencex residue referencing _AgentXSettings with
  the dag5 _DagSettings shape.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 18: `request_rate.py` orchestrator threading + `_issue_child_continuation_or_release`

**Files:**
- Modify: `src/aiperf/timing/strategies/request_rate.py`
- Test: `tests/unit/timing/strategies/test_request_rate_dag_continuation.py`

**Audit notes:**
- `dag4_request_rate.py` (302 lines) carries the cap-gated child routing logic. Compare against `main_request_rate.py` (243 lines).
- Key method: `_issue_child_continuation_or_release(self, credit) -> None`. When `requests_sent` >= cap, the gated child routes through `on_child_stopped` (instead of erroring) so parent joins drain.
- The strategy gains a constructor parameter `branch_orchestrator: BranchOrchestrator | None = None` so non-DAG runs continue to work unchanged.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from unittest.mock import MagicMock

  import pytest

  from aiperf.common.enums import CreditPhase
  from aiperf.credit.structs import Credit
  from aiperf.timing.strategies.request_rate import RequestRateStrategy


  @pytest.fixture
  def child_credit() -> Credit:
      return Credit(
          credit_num=99,
          credit_phase=CreditPhase.PROFILING,
          conversation_id="c",
          x_correlation_id="child-1",
          parent_correlation_id="root",
          agent_depth=1,
      )


  class TestIssueChildContinuationOrRelease:
      @pytest.mark.asyncio
      async def test_below_cap_dispatches_normally(self, child_credit: Credit):
          orch = MagicMock()
          orch.on_child_stopped = MagicMock()
          strategy = RequestRateStrategy(
              # Minimal construction — most args are placeholders for the unit boundary.
              # Use the strategy's actual signature; only branch_orchestrator + cap_state matter for this path.
              branch_orchestrator=orch,
              request_count_cap=100,
              requests_sent=10,
          )
          dispatched = await strategy._issue_child_continuation_or_release(child_credit)
          assert dispatched is True
          orch.on_child_stopped.assert_not_called()

      @pytest.mark.asyncio
      async def test_at_cap_routes_to_on_child_stopped(self, child_credit: Credit):
          orch = MagicMock()
          orch.on_child_stopped = MagicMock()
          strategy = RequestRateStrategy(
              branch_orchestrator=orch,
              request_count_cap=100,
              requests_sent=100,
          )
          dispatched = await strategy._issue_child_continuation_or_release(child_credit)
          assert dispatched is False
          orch.on_child_stopped.assert_called_once_with(child_credit)
  ```

  (The fixture-construction signature is approximate — when porting the actual strategy, match its real `__init__` from main and add `branch_orchestrator` as a new keyword arg.)

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/strategies/test_request_rate_dag_continuation.py -v`

  Expected: FAIL — `_issue_child_continuation_or_release` does not exist; `branch_orchestrator` is not a constructor parameter.

- [ ] **Step 3: Write minimal implementation**

  Apply the dag4 changes to `request_rate.py`:

  ```bash
  diff /tmp/dag5plan/main_request_rate.py /tmp/dag5plan/dag4_request_rate.py
  ```

  Apply hunks: add `branch_orchestrator` constructor parameter; add `_issue_child_continuation_or_release` method; route the child-credit dispatch path through it.

  ```bash
  git show ajc/dag4:src/aiperf/timing/strategies/request_rate.py > src/aiperf/timing/strategies/request_rate.py
  grep -nE "cache_bust|weka|agentic_replay" src/aiperf/timing/strategies/request_rate.py
  ```

  Expected output: empty.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/timing/strategies/request_rate.py tests/unit/timing/strategies/test_request_rate_dag_continuation.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): thread BranchOrchestrator into RequestRateStrategy

  Adds branch_orchestrator constructor param + _issue_child_continuation_or_release.
  When requests_sent crosses the --request-count cap, gated child credits
  route through on_child_stopped (not as errors) so pending parent joins
  drain cleanly. BranchStats.joins_suppressed counts how many joins ended
  this way.

  Non-DAG runs are unaffected: the new branch_orchestrator parameter
  defaults to None.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 19: `phase/credit_counter.py` `is_final_credit` flip

**Files:**
- Modify: `src/aiperf/timing/phase/credit_counter.py`
- Test: `tests/unit/timing/phase/test_credit_counter_dag_final_flip.py`

**Audit notes:**
- `dag4_credit_counter.py` (285 lines) vs `main_credit_counter.py` (237 lines). The diff: a child credit flips `is_final_credit` to True once `requests_sent` crosses `--request-count`, signalling the orchestrator to stop dispatching new children for the parent.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import CreditPhase
  from aiperf.credit.structs import Credit
  from aiperf.timing.phase.credit_counter import CreditCounter


  @pytest.fixture
  def credit() -> Credit:
      return Credit(
          credit_num=10,
          credit_phase=CreditPhase.PROFILING,
          conversation_id="c",
          x_correlation_id="x",
          parent_correlation_id="root",
          agent_depth=1,
      )


  class TestIsFinalCreditFlip:
      def test_below_cap_not_final(self, credit: Credit):
          counter = CreditCounter(request_count=100, requests_sent=10)
          assert counter.is_final_credit(credit) is False

      def test_at_cap_marks_final(self, credit: Credit):
          counter = CreditCounter(request_count=100, requests_sent=100)
          assert counter.is_final_credit(credit) is True

      def test_root_credit_not_affected_by_dag_cap(self, credit: Credit):
          # Root credits use the existing per-conversation final-credit semantics;
          # the DAG cap flip only fires for children.
          credit.parent_correlation_id = None
          credit.agent_depth = 0
          counter = CreditCounter(request_count=100, requests_sent=100)
          # Root behavior is governed by other rules — this test only asserts
          # that the new DAG-cap branch does not over-trigger.
          # Concrete root behaviour stays as-is on main; assert no exception.
          counter.is_final_credit(credit)
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/phase/test_credit_counter_dag_final_flip.py -v`

  Expected: FAIL — the cap-based flip on child credits is not present.

- [ ] **Step 3: Write minimal implementation**

  ```bash
  git show ajc/dag4:src/aiperf/timing/phase/credit_counter.py > src/aiperf/timing/phase/credit_counter.py
  grep -nE "cache_bust|weka|agentic_replay" src/aiperf/timing/phase/credit_counter.py
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/timing/phase/credit_counter.py tests/unit/timing/phase/test_credit_counter_dag_final_flip.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): flip is_final_credit on DAG child credits at cap

  When --request-count is set and requests_sent crosses the cap, child
  credits are flagged is_final_credit=True so the orchestrator stops
  dispatching new children for that parent. Pairs with
  RequestCountStopCondition.applies_to_dag_children=True (Task 20) and
  request_rate._issue_child_continuation_or_release (Task 18) to give
  --request-count literal wire-cap semantics including DAG fan-out.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 20: `phase/stop_conditions.py` `RequestCountStopCondition.applies_to_dag_children = True`

**Files:**
- Modify: `src/aiperf/timing/phase/stop_conditions.py`
- Test: `tests/unit/timing/phase/test_stop_conditions_dag_applies.py`

**Audit notes:**
- Per spec section "c. Timing Layer": `RequestCountStopCondition.applies_to_dag_children = True`; `SessionCountStopCondition` stays root-only.
- `dag4_stop_conditions.py` carries this flag. Diff against main and apply.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.timing.phase.stop_conditions import (
      RequestCountStopCondition,
      SessionCountStopCondition,
  )


  class TestAppliesToDagChildren:
      def test_request_count_applies(self):
          assert RequestCountStopCondition.applies_to_dag_children is True

      def test_session_count_root_only(self):
          # SessionCountStopCondition stays root-only per dag4 design.
          assert SessionCountStopCondition.applies_to_dag_children is False
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/phase/test_stop_conditions_dag_applies.py -v`

  Expected: FAIL — `applies_to_dag_children` attribute missing.

- [ ] **Step 3: Write minimal implementation**

  ```bash
  git show ajc/dag4:src/aiperf/timing/phase/stop_conditions.py > src/aiperf/timing/phase/stop_conditions.py
  grep -nE "applies_to_dag_children" src/aiperf/timing/phase/stop_conditions.py
  ```

  Expected output: at least one match (on `RequestCountStopCondition`).

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/timing/phase/stop_conditions.py tests/unit/timing/phase/test_stop_conditions_dag_applies.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): RequestCountStopCondition.applies_to_dag_children = True

  Marks the request-count stop condition as applying to DAG child
  credits (caps wire requests, period). SessionCountStopCondition
  remains root-only. Pairs with credit_counter.is_final_credit flip
  (Task 19) and request_rate._issue_child_continuation_or_release
  (Task 18).

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 21: `TimingManager._on_dataset_configuration_failed` + `_wait_for_dataset_or_failure`

**Files:**
- Modify: `src/aiperf/timing/manager.py`
- Test: `tests/unit/timing/test_timing_manager_dataset_failure.py`

**Audit notes:**
- `inferencex_timing_manager.py` (209 lines) vs `main_timing_manager.py` (161 lines). The diff: two new methods that listen for `DatasetConfigurationFailed` events and abort the wait cleanly instead of hanging for the 300s configure timeout.
- These methods are listed under spec section "c. Timing Layer".

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import asyncio

  import pytest


  class TestDatasetFailureHandling:
      @pytest.mark.asyncio
      async def test_wait_for_dataset_or_failure_returns_on_failure_event(self):
          from aiperf.timing.manager import TimingManager

          mgr = TimingManager.__new__(TimingManager)  # bypass full bootstrap
          mgr._dataset_ready = asyncio.Event()
          mgr._dataset_failed = asyncio.Event()

          async def fail_soon():
              await asyncio.sleep(0)
              mgr._dataset_failed.set()

          asyncio.create_task(fail_soon())

          # Should return promptly (not hang 300s) once _dataset_failed is set.
          result = await asyncio.wait_for(mgr._wait_for_dataset_or_failure(), timeout=2.0)
          assert result is False  # False == failure path

      @pytest.mark.asyncio
      async def test_on_dataset_configuration_failed_sets_failure_event(self):
          from aiperf.common.messages import DatasetConfigurationFailedMessage
          from aiperf.timing.manager import TimingManager

          mgr = TimingManager.__new__(TimingManager)
          mgr._dataset_ready = asyncio.Event()
          mgr._dataset_failed = asyncio.Event()

          msg = DatasetConfigurationFailedMessage(reason="bad config")
          await mgr._on_dataset_configuration_failed(msg)
          assert mgr._dataset_failed.is_set()
  ```

  (The test uses `__new__` to bypass the full bootstrap; if the manager's events have different names in the inferencex port, adjust to match.)

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_timing_manager_dataset_failure.py -v`

  Expected: FAIL — methods don't exist.

- [ ] **Step 3: Write minimal implementation**

  Diff inferencex against main and apply:

  ```bash
  diff /tmp/dag5plan/main_timing_manager.py /tmp/dag5plan/inferencex_timing_manager.py
  ```

  Lift the two methods plus the `_dataset_failed` asyncio.Event field. If the inferencex source pulls in `agentic_replay` or other out-of-scope imports, scrub. Verify after:

  ```bash
  grep -nE "cache_bust|weka|agentic_replay|reset_context" src/aiperf/timing/manager.py
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/timing/manager.py tests/unit/timing/test_timing_manager_dataset_failure.py
  git commit -s -m "$(cat <<'EOF'
  fix(timing): abort cleanly on DatasetConfigurationFailed instead of hanging 300s

  Adds TimingManager._on_dataset_configuration_failed (handler for
  DatasetConfigurationFailedMessage) and _wait_for_dataset_or_failure
  (awaits either the ready event or the failure event). Without these,
  a malformed dataset hangs the run for the full 300s configure timeout
  before raising; with them, the run aborts immediately when the
  DatasetManager publishes the failure event.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 22: DAG-aware completion gates in records-tracker + phase-runner

**Files:**
- Modify: `src/aiperf/records/records_tracker.py` (the actual file — `git grep records_tracker` may resolve to a different name; verify)
- Modify: `src/aiperf/timing/phase/phase_runner.py`
- Test: `tests/unit/timing/phase/test_phase_runner_dag_completion.py`

**Audit notes:**
- The completion gate must count child HTTP requests toward `requests_sent` so `--request-count` truncates DAGs correctly.
- The phase-runner waits for `requests_sent >= request_count` AND no pending DAG joins on the orchestrator before declaring the phase complete.
- inferencex's records-tracker / phase-runner carry both edits — diff and apply.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from unittest.mock import MagicMock

  import pytest


  class TestPhaseRunnerDagCompletion:
      def test_completion_blocks_while_orchestrator_has_pending_joins(self):
          from aiperf.timing.phase.phase_runner import PhaseRunner

          orch = MagicMock()
          orch.has_pending_branch_work.return_value = True
          runner = PhaseRunner.__new__(PhaseRunner)
          runner._branch_orchestrator = orch
          runner._requests_sent = 100
          runner._request_count_cap = 100
          # Even at cap, completion should not fire while joins are pending.
          assert runner._is_phase_complete() is False

      def test_completion_fires_when_orchestrator_drained(self):
          from aiperf.timing.phase.phase_runner import PhaseRunner

          orch = MagicMock()
          orch.has_pending_branch_work.return_value = False
          runner = PhaseRunner.__new__(PhaseRunner)
          runner._branch_orchestrator = orch
          runner._requests_sent = 100
          runner._request_count_cap = 100
          assert runner._is_phase_complete() is True


  class TestRecordsTrackerCountsChildren:
      def test_child_request_increments_requests_sent(self):
          from aiperf.records.records_tracker import RecordsTracker
          from aiperf.common.enums import CreditPhase
          from aiperf.credit.structs import Credit

          tracker = RecordsTracker.__new__(RecordsTracker)
          tracker._requests_sent_by_phase = {CreditPhase.PROFILING: 0}

          child_credit = Credit(
              credit_num=5,
              credit_phase=CreditPhase.PROFILING,
              conversation_id="c",
              x_correlation_id="child",
              parent_correlation_id="root",
              agent_depth=1,
          )
          tracker.on_credit_dispatched(child_credit)
          assert tracker._requests_sent_by_phase[CreditPhase.PROFILING] == 1
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/phase/test_phase_runner_dag_completion.py -v`

  Expected: FAIL — `has_pending_branch_work` consultation in `_is_phase_complete` is not present; tracker may not count children.

- [ ] **Step 3: Write minimal implementation**

  Patch the records-tracker and phase-runner per the inferencex source:

  ```bash
  # Locate the actual records-tracker file.
  git grep -lE "class RecordsTracker" -- 'src/aiperf/**.py'
  # Diff the relevant slice against main.
  ```

  Edit both files to:
  - Records tracker: `on_credit_dispatched(credit)` increments `requests_sent_by_phase[credit.credit_phase]` regardless of `agent_depth` (no early-return for children).
  - Phase runner: `_is_phase_complete` returns False if `branch_orchestrator.has_pending_branch_work()` is True, even if `requests_sent >= request_count`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/records/records_tracker.py src/aiperf/timing/phase/phase_runner.py tests/unit/timing/phase/test_phase_runner_dag_completion.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): DAG-aware completion gates in records-tracker + phase-runner

  Records tracker: child HTTP requests count toward requests_sent so
  --request-count truncates DAGs correctly.

  Phase runner: completion blocks while branch_orchestrator reports
  pending joins, even if requests_sent has reached --request-count.
  Without this, the phase would close mid-DAG, dropping in-flight
  children.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 23: `BranchStats` publication via `CreditPhaseCompleteMessage` + export

**Files:**
- Modify: `src/aiperf/common/messages/credit_phase_complete.py` (or wherever `CreditPhaseCompleteMessage` lives — `git grep -l "CreditPhaseCompleteMessage" src/aiperf/common/messages/`)
- Modify: `src/aiperf/timing/manager.py` (publish `BranchStats` on phase-complete)
- Modify: the `profile_export_aiperf.json` exporter (`git grep -l "profile_export_aiperf" src/aiperf/`)
- Test: `tests/unit/common/test_credit_phase_complete_branch_stats.py`
- Test: `tests/unit/exporters/test_profile_export_branch_stats.py`

**Audit notes:**
- Plan 1 already shipped the `BranchStats` model (Task 7 of Plan 1) but did NOT wire it through publication or export. That wiring is this task.
- Spec section "In-Scope" + "Behavior Decisions": `BranchStats.joins_suppressed` is the counter for stop-condition-suppressed joins.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.messages import CreditPhaseCompleteMessage
  from aiperf.common.models.branch_stats import BranchStats


  class TestCreditPhaseCompleteBranchStats:
      def test_field_present(self):
          stats = BranchStats(
              children_spawned=4,
              children_completed=4,
              errors=0,
              joins_suppressed=0,
          )
          msg = CreditPhaseCompleteMessage(branch_stats=stats)
          assert msg.branch_stats == stats

      def test_default_none_for_non_dag_runs(self):
          msg = CreditPhaseCompleteMessage()
          assert msg.branch_stats is None

      def test_round_trip(self):
          stats = BranchStats(
              children_spawned=2,
              children_completed=2,
              errors=1,
              joins_suppressed=1,
          )
          msg = CreditPhaseCompleteMessage(branch_stats=stats)
          rebuilt = CreditPhaseCompleteMessage.model_validate(msg.model_dump())
          assert rebuilt.branch_stats == stats
  ```

  And the exporter test:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import orjson
  import pytest

  from aiperf.common.models.branch_stats import BranchStats


  class TestProfileExportBranchStats:
      def test_branch_stats_in_json_when_present(self, tmp_path):
          from aiperf.exporters.profile_exporter import write_profile_export_aiperf_json

          stats = BranchStats(
              children_spawned=3,
              children_completed=3,
              errors=0,
              joins_suppressed=0,
          )
          out = tmp_path / "profile_export_aiperf.json"
          write_profile_export_aiperf_json(out, branch_stats=stats)
          dumped = orjson.loads(out.read_bytes())
          assert "branch_stats" in dumped
          assert dumped["branch_stats"]["children_spawned"] == 3
          assert dumped["branch_stats"]["joins_suppressed"] == 0

      def test_branch_stats_omitted_for_non_dag_runs(self, tmp_path):
          from aiperf.exporters.profile_exporter import write_profile_export_aiperf_json

          out = tmp_path / "profile_export_aiperf.json"
          write_profile_export_aiperf_json(out, branch_stats=None)
          dumped = orjson.loads(out.read_bytes())
          assert dumped.get("branch_stats") is None
  ```

  (The `write_profile_export_aiperf_json` import path is illustrative — match the actual function signature in the exporter; if the exporter takes a `ProfileResults` object that already contains `branch_stats`, adapt the test to construct that object.)

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/test_credit_phase_complete_branch_stats.py tests/unit/exporters/test_profile_export_branch_stats.py -v`

  Expected: FAIL — `branch_stats` field missing on the message; exporter doesn't write the section.

- [ ] **Step 3: Write minimal implementation**

  1. Add `branch_stats: BranchStats | None = Field(default=None, description=...)` to `CreditPhaseCompleteMessage`.
  2. In `TimingManager` (or wherever the phase-complete message is published — verify with `git grep "CreditPhaseCompleteMessage(" src/aiperf/`), set `branch_stats=self._branch_orchestrator.snapshot_branch_stats() if self._branch_orchestrator else None`.
  3. Add `snapshot_branch_stats() -> BranchStats` to `BranchOrchestrator` (it's already tracking the counters internally per the inferencex source — wire the snapshot accessor).
  4. In the exporter, splice `branch_stats` into the output JSON when non-None.

  Reference: `git grep -n "branch_stats" ajc/inferencex-agentx-mvp -- 'src/aiperf/**.py'` to find the inferencex wiring sites.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/messages/ src/aiperf/timing/manager.py src/aiperf/timing/branch_orchestrator.py src/aiperf/exporters/ tests/unit/common/test_credit_phase_complete_branch_stats.py tests/unit/exporters/test_profile_export_branch_stats.py
  git commit -s -m "$(cat <<'EOF'
  feat(timing): publish BranchStats on CreditPhaseCompleteMessage and export to JSON

  Wires BranchOrchestrator's running counters (children_spawned,
  children_completed, errors, joins_suppressed) through to the
  CreditPhaseCompleteMessage and the profile_export_aiperf.json
  output. Non-DAG runs omit the section.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 24: CLI `--num-conversations` autodefault for `dag_jsonl`

**Files:**
- Modify: `src/aiperf/common/config/user_config.py`
- Test: `tests/unit/common/config/test_user_config_dag_default.py`

**Audit notes:**
- `dag4_user_config.py` lines 415-425 (`_is_forking_dataset`) and 474-513 (`_count_dag_root_entries`).
- The autodefault block lives at lines 195-236 (the request-rate validator branch). When `--request-count` and `--num-conversations` are both unset and the dataset is forking, default `--num-conversations` to the root count.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest

  from aiperf.common.config.user_config import UserConfig


  def _write_dag_jsonl(path: Path, root_count: int, total: int) -> None:
      assert total >= root_count
      records: list[dict] = []
      # First N are roots (no incoming forks).
      for i in range(root_count):
          children = [f"child-{i}-{j}" for j in range(total // root_count - 1) if total > root_count]
          records.append({
              "session_id": f"root-{i}",
              "turns": [{"messages": [{"role": "user", "content": "x"}], "forks": children}]
                       if children else [{"messages": [{"role": "user", "content": "x"}]}],
          })
      # Remaining are children (referenced).
      for i in range(root_count):
          for j in range(total // root_count - 1) if total > root_count else []:
              records.append({
                  "session_id": f"child-{i}-{j}",
                  "turns": [{"messages": [{"role": "user", "content": "c"}]}],
              })
      with open(path, "wb") as f:
          for r in records:
              f.write(orjson.dumps(r))
              f.write(b"\n")


  class TestDagAutodefault:
      def test_num_conversations_defaults_to_root_count(self, tmp_path):
          dag_file = tmp_path / "x.dag.jsonl"
          _write_dag_jsonl(dag_file, root_count=3, total=9)  # 3 roots, 6 children

          config = UserConfig(
              endpoint={"model_names": ["test-model"], "url": "http://localhost:8000/v1"},
              input={"file": str(dag_file), "custom_dataset_type": "dag_jsonl"},
              loadgen={"concurrency": 4},
          )
          # Autodefault should size --num-conversations by *root* count, not total.
          assert config.input.conversation.num == 3

      def test_request_count_not_defaulted_for_forking_dataset(self, tmp_path):
          dag_file = tmp_path / "x.dag.jsonl"
          _write_dag_jsonl(dag_file, root_count=2, total=4)

          config = UserConfig(
              endpoint={"model_names": ["test-model"], "url": "http://localhost:8000/v1"},
              input={"file": str(dag_file), "custom_dataset_type": "dag_jsonl"},
              loadgen={"concurrency": 4},
          )
          # --request-count must NOT be auto-defaulted (would truncate mid-tree).
          # Number of conversations IS defaulted; --request-count stays None.
          assert config.loadgen.request_count is None
          assert config.input.conversation.num == 2
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/config/test_user_config_dag_default.py -v`

  Expected: FAIL — `_count_dag_root_entries` and the autodefault branch missing.

- [ ] **Step 3: Write minimal implementation**

  Diff `dag4_user_config.py` against `main_user_config.py` and apply the two new methods (`_is_forking_dataset` lines 415-425, `_count_dag_root_entries` lines 474-513) plus the autodefault branch (lines 195-236).

  Use `git show` to write a scratch file, then port the targeted hunks (don't replace the whole `user_config.py` — main has drifted in unrelated places):

  ```bash
  diff /tmp/dag5plan/main_user_config.py /tmp/dag5plan/dag4_user_config.py | head -200
  ```

  Apply the hunks with `Edit` calls anchored on stable surrounding context. After editing, verify:

  ```bash
  grep -nE "_is_forking_dataset|_count_dag_root_entries" src/aiperf/common/config/user_config.py
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/config/user_config.py tests/unit/common/config/test_user_config_dag_default.py
  git commit -s -m "$(cat <<'EOF'
  feat(cli): autodefault --num-conversations to dag_jsonl root count

  When the input is dag_jsonl and neither --request-count nor
  --num-conversations is provided, default --num-conversations to the
  count of root entries (sessions not referenced by any other session's
  forks). Refuses to default --request-count for forking datasets so
  the cap does not truncate the DAG mid-tree.

  Adds _is_forking_dataset and _count_dag_root_entries helpers ported
  from dag4.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 25: CLI `--no-fixed-schedule` (`InputConfig.disable_auto_fixed_schedule`)

**Files:**
- Modify: `src/aiperf/common/config/input_config.py`
- Modify: `src/aiperf/common/config/user_config.py` (consume the flag in the `_should_use_fixed_schedule_for_trace_dataset` path)
- Test: `tests/unit/common/config/test_no_fixed_schedule.py`

**Audit notes:**
- Spec section "In-Scope" calls this out as a generic loadgen flag, not Weka-specific.
- The flag gates the auto-promotion of trace datasets with timestamps to fixed-schedule mode.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import pytest

  from aiperf.common.config.user_config import UserConfig


  class TestNoFixedSchedule:
      def test_default_false(self):
          config = UserConfig(
              endpoint={"model_names": ["m"], "url": "http://localhost:8000/v1"},
          )
          assert config.input.disable_auto_fixed_schedule is False

      def test_flag_disables_auto_promotion(self, tmp_path):
          # Create a trace dataset that *would* otherwise auto-promote to fixed schedule.
          trace_file = tmp_path / "trace.jsonl"
          trace_file.write_text('{"timestamp": 0, "messages": [{"role": "user", "content": "x"}]}\n')
          config = UserConfig(
              endpoint={"model_names": ["m"], "url": "http://localhost:8000/v1"},
              input={
                  "file": str(trace_file),
                  "custom_dataset_type": "mooncake_trace",
                  "disable_auto_fixed_schedule": True,
              },
              loadgen={"concurrency": 4},
          )
          # With the flag set, the timing mode should NOT be FIXED_SCHEDULE.
          from aiperf.common.enums import TimingMode

          assert config._timing_mode != TimingMode.FIXED_SCHEDULE
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/config/test_no_fixed_schedule.py -v`

  Expected: FAIL — `disable_auto_fixed_schedule` field missing.

- [ ] **Step 3: Write minimal implementation**

  1. Add to `InputConfig`:

     ```python
         disable_auto_fixed_schedule: Annotated[
             bool,
             Field(
                 description="When True, suppress the auto-promotion of trace "
                 "datasets with timestamps to fixed-schedule timing mode. "
                 "Honour the explicit timing flags on --concurrency / "
                 "--request-rate even for timestamped traces. Equivalent CLI "
                 "alias: --no-fixed-schedule.",
                 alias="no-fixed-schedule",
             ),
         ] = False
     ```

  2. In `user_config.py`'s `_should_use_fixed_schedule_for_trace_dataset` (or wherever the auto-promotion lives), short-circuit:

     ```python
     if self.input.disable_auto_fixed_schedule:
         return False
     ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/config/input_config.py src/aiperf/common/config/user_config.py tests/unit/common/config/test_no_fixed_schedule.py
  git commit -s -m "$(cat <<'EOF'
  feat(cli): add --no-fixed-schedule (disable_auto_fixed_schedule)

  Generic loadgen flag that suppresses auto-promotion of timestamped
  trace datasets to fixed-schedule timing mode. Honors the explicit
  --concurrency / --request-rate selection even for timestamped
  traces. Useful when re-running a trace with a different load
  pattern than it was captured under.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 26: Worker call-sites for FORK pin refcount

**Files:**
- Modify: `src/aiperf/workers/worker.py` (add the pin/release call sites; the SessionManager API itself landed in Task 8)
- Test: `tests/unit/workers/test_worker_fork_pin_call_sites.py`

**Audit notes:**
- This task wires the orchestrator-driven pin/release into the worker's child-credit handling path. Avoid the inferencex worker's cache-bust / Weka / agentic-replay paths — only lift the pin/release calls.
- On FORK child credit dispatch (parent_correlation_id is set, branch_mode is FORK), the worker calls `session_manager.pin_for_fork_child(parent_session_id)` before sending. On the child's terminal turn (or on cancellation), the worker calls `session_manager.release_fork_child(parent_session_id)`.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from unittest.mock import MagicMock

  import pytest

  from aiperf.common.enums import ConversationBranchMode, CreditPhase
  from aiperf.credit.structs import Credit
  from aiperf.workers.worker import Worker


  class TestForkPinCallSites:
      @pytest.mark.asyncio
      async def test_pin_called_on_fork_child_dispatch(self):
          worker = Worker.__new__(Worker)
          worker._session_manager = MagicMock()
          # The parent session was created earlier; resolve_parent returns its id.
          worker._resolve_parent_session_id = MagicMock(return_value="parent-sid")

          fork_credit = Credit(
              credit_num=1,
              credit_phase=CreditPhase.PROFILING,
              conversation_id="x",
              x_correlation_id="child-corr",
              parent_correlation_id="parent-corr",
              agent_depth=1,
              branch_mode=ConversationBranchMode.FORK,
          )
          await worker._on_fork_child_credit_dispatch(fork_credit)
          worker._session_manager.pin_for_fork_child.assert_called_once_with("parent-sid")

      @pytest.mark.asyncio
      async def test_release_called_on_child_terminal(self):
          worker = Worker.__new__(Worker)
          worker._session_manager = MagicMock()
          worker._resolve_parent_session_id = MagicMock(return_value="parent-sid")

          await worker._on_fork_child_terminal("parent-corr")
          worker._session_manager.release_fork_child.assert_called_once_with("parent-sid")
  ```

  (`_on_fork_child_credit_dispatch` and `_on_fork_child_terminal` are placeholder names — match the actual hooks the inferencex worker uses, e.g. by reading the inferencex source for `pin_for_fork_child` call sites: `git show ajc/inferencex-agentx-mvp:src/aiperf/workers/worker.py | grep -n "pin_for_fork_child\|release_fork_child"`.)

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/workers/test_worker_fork_pin_call_sites.py -v`

  Expected: FAIL — call sites not wired.

- [ ] **Step 3: Write minimal implementation**

  Locate the inferencex pin/release call sites:

  ```bash
  git show ajc/inferencex-agentx-mvp:src/aiperf/workers/worker.py | grep -nB2 -A8 "pin_for_fork_child\|release_fork_child"
  ```

  Lift just those call sites — DO NOT verbatim-port the entire 1171-line inferencex worker.py. Apply targeted Edit calls against the local 716-line worker.py.

  After editing:

  ```bash
  grep -nE "cache_bust|weka|agentic_replay|reset_context" src/aiperf/workers/worker.py
  ```

  Expected output: empty.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/workers/worker.py tests/unit/workers/test_worker_fork_pin_call_sites.py
  git commit -s -m "$(cat <<'EOF'
  feat(workers): wire FORK pin refcount into worker child-credit path

  Worker calls SessionManager.pin_for_fork_child(parent_session_id)
  on FORK child dispatch and release_fork_child on child terminal /
  cancellation. Parent session evicts only when the refcount hits
  zero.

  cache-bust marker injection paths from inferencex are intentionally
  NOT lifted; only the pin/release call sites are.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 27: DAG cross-component tests (timing/)

**Files:**
- Test: `tests/unit/timing/test_dag_cross_component.py`
- Test: `tests/unit/timing/test_dag_concurrency_pathology.py`
- Test: `tests/component_integration/timing/test_dag_combined_pathology.py`
- Test: `tests/component_integration/timing/test_dag_timing_pathology.py`
- Test: `tests/component_integration/timing/test_dag_join_end_to_end.py`
- Test: `tests/component_integration/timing/test_dag_v1_adversarial.py`
- Test: `tests/component_integration/timing/test_dag_adversarial_timing_modes.py`

- [ ] **Step 1: Write the failing test**

  Verbatim port from inferencex:

  ```bash
  for f in tests/unit/timing/test_dag_cross_component.py tests/unit/timing/test_dag_concurrency_pathology.py tests/component_integration/timing/test_dag_combined_pathology.py tests/component_integration/timing/test_dag_timing_pathology.py tests/component_integration/timing/test_dag_join_end_to_end.py tests/component_integration/timing/test_dag_v1_adversarial.py tests/component_integration/timing/test_dag_adversarial_timing_modes.py; do
    git show ajc/inferencex-agentx-mvp:$f > $f
  done
  ```

  Audit each for out-of-scope imports (cache_bust / weka / agentic_replay / reset_context):

  ```bash
  grep -lE "cache_bust|weka|agentic_replay|reset_context" tests/unit/timing/test_dag_cross_component.py tests/unit/timing/test_dag_concurrency_pathology.py tests/component_integration/timing/test_dag_combined_pathology.py tests/component_integration/timing/test_dag_timing_pathology.py tests/component_integration/timing/test_dag_join_end_to_end.py tests/component_integration/timing/test_dag_v1_adversarial.py tests/component_integration/timing/test_dag_adversarial_timing_modes.py
  ```

  Patch any flagged files to drop the OOS references.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/timing/test_dag_cross_component.py tests/unit/timing/test_dag_concurrency_pathology.py -v`

  Expected: PASS — the orchestrator runtime is complete by this point. If any FAIL, the failure points to a wiring gap; fix the underlying orchestrator / records-tracker / phase-runner code before continuing.

- [ ] **Step 3: Write minimal implementation**

  No new impl. If a test fails, fix the runtime code (not the test).

- [ ] **Step 4: Run test to verify it passes**

  Run unit and component-integration in two separate invocations per project rule:

  ```bash
  uv run pytest tests/unit/ -n auto
  uv run pytest -m component_integration -n auto
  ```

  Expected: PASS on both.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/timing/test_dag_cross_component.py tests/unit/timing/test_dag_concurrency_pathology.py tests/component_integration/timing/test_dag_*.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port DAG cross-component + adversarial test suites

  Cross-component orchestrator-to-records-tracker coverage; concurrency
  pathology (interleaved fork explosions, racing joins); combined
  pathology (DAG + cancellation + cap); adversarial timing modes
  (DAG under request-rate, fixed-schedule, gamma); v1 adversarial
  (legacy edge cases preserved); join end-to-end.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 28: dag4-only `test_dag_hard_cap.py` and `test_dag_multi_root_payload_bytes.py`

**Files:**
- Test: `tests/component_integration/timing/test_dag_hard_cap.py`
- Test: `tests/component_integration/timing/test_dag_multi_root_payload_bytes.py`

**Audit notes:**
- These tests assert exactly the dag4 semantics:
  - hard_cap: `--request-count 30` produces exactly 30 wire requests across forks
  - multi_root: multi-root DAG round-trips through PAYLOAD_BYTES context mode without losing `is_fork_parent`

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/dag4:tests/component_integration/timing/test_dag_hard_cap.py > tests/component_integration/timing/test_dag_hard_cap.py
  git show ajc/dag4:tests/component_integration/timing/test_dag_multi_root_payload_bytes.py > tests/component_integration/timing/test_dag_multi_root_payload_bytes.py
  ```

  Verify the multi-root fixture is in place (Task 5 already copied `tests/fixtures/dag/multi_root_single_turn.dag.jsonl` from dag4).

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest -m component_integration -n auto -k "test_dag_hard_cap or test_dag_multi_root_payload_bytes"`

  Expected: PASS — Tasks 18, 19, 20 (request-rate / credit-counter / stop-conditions wiring) and Task 7 (`is_fork_parent` stamping) are the implementation that makes these tests pass.

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest -m component_integration -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/component_integration/timing/test_dag_hard_cap.py tests/component_integration/timing/test_dag_multi_root_payload_bytes.py
  git commit -s -m "$(cat <<'EOF'
  test(timing): port dag4 hard-cap and multi-root payload-bytes tests

  hard_cap: --request-count 30 on a forking dataset produces exactly
  30 wire requests across the DAG fanout (validates the cap-applies-to-
  children semantics from Tasks 18-20).

  multi_root_payload_bytes: a multi-root DAG round-trips through
  PAYLOAD_BYTES context mode without losing UserSession.is_fork_parent
  (validates the dag4 stamp-at-create-time fix from Task 7).

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 29: End-to-end DAG integration test against in-repo mock server

**Files:**
- Test: `tests/integration/test_dag_full_topology.py`
- Test: `tests/integration/test_dag_spawn.py`

**Audit notes:**
- These tests run a complete benchmark against the in-repo mock server with a DAG topology, asserting BranchStats numbers, child credit counts, fork-pin refcount drained at end, and `profile_export_aiperf.json` round-trip.
- inferencex carries both files; port verbatim.

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/integration/test_dag_full_topology.py > tests/integration/test_dag_full_topology.py
  git show ajc/inferencex-agentx-mvp:tests/integration/test_dag_spawn.py > tests/integration/test_dag_spawn.py
  grep -lE "cache_bust|weka|agentic_replay|reset_context" tests/integration/test_dag_full_topology.py tests/integration/test_dag_spawn.py
  ```

  Patch any matches.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest -m integration -n auto -k "test_dag_full_topology or test_dag_spawn"`

  Expected: PASS — full runtime is in place.

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest -m integration -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/integration/test_dag_full_topology.py tests/integration/test_dag_spawn.py
  git commit -s -m "$(cat <<'EOF'
  test(integration): port DAG full-topology and spawn end-to-end tests

  Runs a complete benchmark against the in-repo mock server with a
  fork-and-fan-in DAG topology and a pre-session SPAWN topology.
  Asserts BranchStats numbers, child credit counts, fork-pin refcount
  drained at end, and profile_export_aiperf.json round-trip.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 30: DAG-metadata tagging unit test + records test port

**Files:**
- Test: `tests/unit/records/test_dag_metadata_tagging.py`

- [ ] **Step 1: Write the failing test**

  ```bash
  git show ajc/inferencex-agentx-mvp:tests/unit/records/test_dag_metadata_tagging.py > tests/unit/records/test_dag_metadata_tagging.py
  ```

  This test asserts that `MetricRecordMetadata.agent_depth` and `parent_correlation_id` are correctly populated from the originating credit/record path.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/records/test_dag_metadata_tagging.py -v`

  Expected: PASS (Task 2 added the fields to `MetricRecordMetadata`; Task 9 ensures they propagate via `_enrich_request_record`).

- [ ] **Step 3: Write minimal implementation**

  No new impl.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/records/test_dag_metadata_tagging.py
  git commit -s -m "$(cat <<'EOF'
  test(records): port DAG metadata tagging test

  Asserts agent_depth and parent_correlation_id flow from Credit through
  RequestInfo / RecordContext into MetricRecordMetadata so metric
  records can be filtered by DAG layer or parent.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 31: dag_jsonl prereq + delayed/fan_in/multi_gate/pre_session/pathological loader tests

**Files:**
- Test: `tests/unit/dataset/loader/test_dag_jsonl_prereq.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_prereq_adversarial.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_delayed.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_fan_in.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_multi_gate.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_pre_session.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_property.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_adversarial_full.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_pathological.py`
- Test: `tests/unit/dataset/loader/test_dag_jsonl_plugin.py`

- [ ] **Step 1: Write the failing test**

  ```bash
  for f in tests/unit/dataset/loader/test_dag_jsonl_prereq.py \
           tests/unit/dataset/loader/test_dag_jsonl_prereq_adversarial.py \
           tests/unit/dataset/loader/test_dag_jsonl_delayed.py \
           tests/unit/dataset/loader/test_dag_jsonl_fan_in.py \
           tests/unit/dataset/loader/test_dag_jsonl_multi_gate.py \
           tests/unit/dataset/loader/test_dag_jsonl_pre_session.py \
           tests/unit/dataset/loader/test_dag_jsonl_property.py \
           tests/unit/dataset/loader/test_dag_jsonl_adversarial_full.py \
           tests/unit/dataset/loader/test_dag_jsonl_pathological.py \
           tests/unit/dataset/loader/test_dag_jsonl_plugin.py; do
    git show ajc/inferencex-agentx-mvp:$f > $f
  done
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/dataset/loader/ -v -k dag_jsonl`

  Expected: PASS — Task 5 ported the loader; these are the full coverage suites for prereq parsing, fan-in / multi-gate / delayed / pre-session topology variants, property-based fuzzing, full-adversarial paths, pathological inputs, and plugin-registry resolution.

- [ ] **Step 3: Write minimal implementation**

  No new impl. If any test fails, fix the loader (not the test).

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/unit/dataset/loader/test_dag_jsonl_*.py
  git commit -s -m "$(cat <<'EOF'
  test(loaders): port full dag_jsonl loader test suite

  Prereq parsing (happy + adversarial), topology variants
  (delayed, fan-in, multi-gate, pre-session), property-based fuzzing,
  full adversarial paths, pathological inputs, plugin-registry resolution.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 32: `docs/benchmark-modes/dag.md`

**Files:**
- Create: `docs/benchmark-modes/dag.md`

**Audit notes:**
- inferencex's `docs/benchmark-modes/dag.md` is the source. dag4 may not have it (it landed on inferencex via the same commit family that introduced the `spawns:` shorthand).
- If neither branch has the file, write it fresh from spec section "Data Flow" + section "Behavior Decisions".

- [ ] **Step 1: Write the failing test**

  No test. Documentation task. The "test" is human review against the spec.

- [ ] **Step 2: Run test to verify it fails**

  Run: `ls docs/benchmark-modes/dag.md 2>&1`

  Expected: file not found.

- [ ] **Step 3: Write minimal implementation**

  ```bash
  # Try inferencex first.
  if git -C . rev-parse --verify ajc/inferencex-agentx-mvp:docs/benchmark-modes/dag.md >/dev/null 2>&1; then
    git show ajc/inferencex-agentx-mvp:docs/benchmark-modes/dag.md > docs/benchmark-modes/dag.md
  elif git -C . rev-parse --verify ajc/dag4:docs/benchmark-modes/dag.md >/dev/null 2>&1; then
    git show ajc/dag4:docs/benchmark-modes/dag.md > docs/benchmark-modes/dag.md
  else
    # Write fresh from spec — sections to cover, in order:
    # 1. Overview (FORK vs SPAWN)
    # 2. dag_jsonl format (forks: shorthand, spawns: shorthand, prerequisites)
    # 3. Topology validation rules
    # 4. agent_depth and parent_correlation_id semantics
    # 5. --request-count behavior (caps children)
    # 6. --num-conversations autodefault
    # 7. AIPERF_DAG_FAIL_FAST env var
    # 8. BranchStats output in profile_export_aiperf.json
    # 9. Worked example: 3-root file with 2-way fork
    echo "(write fresh — see spec section 'Data Flow' and section 'Behavior Decisions')"
  fi
  ```

  After writing, audit for out-of-scope mentions (cache-bust / Weka / agentic-replay / reset_context); patch any out.

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  ls docs/benchmark-modes/dag.md
  ```

  Expected: file present.

  Optionally run `make generate-all-docs` to verify the docs build cleanly.

- [ ] **Step 5: Commit**

  ```bash
  git add docs/benchmark-modes/dag.md
  git commit -s -m "$(cat <<'EOF'
  docs(benchmark-modes): add dag.md DAG benchmark mode reference

  Full reference for the dag_jsonl input type: format syntax (forks:
  / spawns: shorthand, prerequisites), topology validation rules,
  agent_depth / parent_correlation_id semantics, --request-count
  cap-applies-to-children behavior, --num-conversations autodefault,
  AIPERF_DAG_FAIL_FAST env var, BranchStats output schema, and a
  worked example.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 33: Four-file sync of `dag_jsonl` mention + docs regeneration

**Files:**
- Modify: `CLAUDE.md` (mention `dag_jsonl` in the existing tip line)
- Modify: `.github/copilot-instructions.md` (mirror)
- Modify: `.cursor/rules/python.mdc` (mirror)
- Modify: `AGENTS.md` if it participates in the sync rule per origin/main's CLAUDE.md (verify with `head -30 CLAUDE.md`)
- Auto-regenerate: `docs/cli-options.md` and `docs/environment-variables.md`

**Audit notes:**
- Plan 1's CLAUDE.md mentions `dag_jsonl` already (the project-CLAUDE.md tip line: "`dag_jsonl` input type: conversation DAG benchmarks (fork + spawn modes); see `docs/benchmark-modes/dag.md`"). Verify that mention is consistent with the other three (or four) files.
- Per origin/main's CLAUDE.md "Three-File Sync Rule" / "Four-File Sync Rule" (depends on which version of main is checked out), AGENTS.md may or may not participate. Verify by reading the rule on `ajc/dag5` HEAD (which already merged Plan 1's CLAUDE.md changes).

- [ ] **Step 1: Write the failing test**

  ```bash
  # Verify the four-file (or three-file) sync rule on dag5.
  grep -A5 "Sync Rule" CLAUDE.md
  # The "test" is `make check-agent-files-sync` if such a target exists; otherwise `pre-commit run --all-files`.
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  make check-agent-files-sync 2>&1 || true
  ```

  Expected: either the target exists and passes (no edits needed), or it fails / doesn't exist and we manually inspect.

- [ ] **Step 3: Write minimal implementation**

  If the existing `dag_jsonl` mention in CLAUDE.md is missing from any of the sister files, propagate it. The exact text already in CLAUDE.md on `ajc/dag5`:

  > `dag_jsonl` input type: conversation DAG benchmarks (fork + spawn modes); see `docs/benchmark-modes/dag.md`

  Ensure the same line is present (verbatim where the project's sync rule mandates verbatim, or paraphrased where the rule allows differences in headers/frontmatter only) in:

  - `.github/copilot-instructions.md`
  - `.cursor/rules/python.mdc`
  - `AGENTS.md` (if AGENTS.md participates per the rule on dag5 HEAD)

  Then regenerate the auto-generated docs:

  ```bash
  make generate-cli-docs
  make generate-env-vars-docs
  ```

  These pick up the new `--no-fixed-schedule` flag and the `AIPERF_DAG_FAIL_FAST` env var.

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  make check-agent-files-sync 2>&1 || diff CLAUDE.md .github/copilot-instructions.md
  pre-commit run --all-files
  ```

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc AGENTS.md docs/cli-options.md docs/environment-variables.md
  git commit -s -m "$(cat <<'EOF'
  docs: sync dag_jsonl mention across instruction files; regenerate auto-docs

  Three-file (or four-file) sync per project rule: ensures CLAUDE.md,
  .github/copilot-instructions.md, .cursor/rules/python.mdc (and
  AGENTS.md where the sync rule mandates) carry the same dag_jsonl
  reference line.

  Regenerates docs/cli-options.md (picks up --no-fixed-schedule) and
  docs/environment-variables.md (picks up AIPERF_DAG_FAIL_FAST).

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Spec Coverage

Mapping of in-scope items from the spec's "In-Scope" (or "Behavior Decisions" / "Error Handling") that Plan 2 lands. Items already shipped in Plan 1 are listed in Plan 1's "Spec Coverage" table and are not duplicated here.

| In-Scope item (spec section) | Task |
|------------------------------|------|
| `agent_depth` / `parent_correlation_id` / `branch_mode` on `Credit` and `TurnToSend` (b. Data Models, Data Flow) | 1 |
| `RecordContext` base + `RequestInfo` / `RequestRecord` re-typing (d. Worker Layer, Data Flow) | 2 |
| `agent_depth` / `parent_correlation_id` on `MetricRecordMetadata` (Data Flow) | 2 |
| `_DagSettings.FAIL_FAST` env var `AIPERF_DAG_FAIL_FAST` (In-Scope + Error Handling) | 3 |
| `dag_jsonl_models.py` with `DagSpawn` (b. Data Models) | 4 |
| `dag_jsonl` loader with `forks:` + `spawns:` shorthand, `_inline_pre_session_spawns`, topology walk that stamps `agent_depth`, validation rejects (a. Loader Layer + Data Flow) | 5 |
| Plugin registry: `dag_jsonl` loader registered (Plugin Registry) | 6 |
| `UserSession.is_fork_parent` stamped at `create_and_store` time (Error Handling — payload-bytes round-trip pitfalls) | 7 |
| Worker FORK pin refcount on `UserSession` (d. Worker Layer + Data Flow) | 8 |
| `inference_client._enrich_request_record` propagates `RecordContext` (d. Worker Layer) | 9 |
| `ConversationSource.start_branch_child` and `start_pre_session_child` (c. Timing Layer + Data Flow) | 10 |
| `BranchOrchestrator` (full advanced version) — core port (c. Timing Layer) | 11 |
| `BranchOrchestrator` fan-in scenarios (Testing Strategy + In-Scope advanced features) | 12 |
| `BranchOrchestrator` multi-gate scenarios (Testing Strategy) | 13 |
| `BranchOrchestrator` K-delayed-join scenarios (In-Scope + Testing Strategy) | 14 |
| `BranchOrchestrator` prereq walking (happy + adversarial) (Data Flow — Prereq gating + Testing Strategy) | 15 |
| `BranchOrchestrator` pre-session SPAWN + phase0 (Data Flow — SPAWN + In-Scope) | 16 |
| `BranchOrchestrator` child-error handling gated by `AIPERF_DAG_FAIL_FAST` (Error Handling — child errors) | 17 |
| `request_rate.py` orchestrator threading + `_issue_child_continuation_or_release` (c. Timing Layer + Behavior Decisions — `--request-count` caps DAG children) | 18 |
| `phase/credit_counter.py` `is_final_credit` flip on cap (c. Timing Layer + Behavior Decisions) | 19 |
| `phase/stop_conditions.py` `RequestCountStopCondition.applies_to_dag_children = True` (c. Timing Layer + Behavior Decisions) | 20 |
| `TimingManager._on_dataset_configuration_failed` + `_wait_for_dataset_or_failure` (c. Timing Layer + Error Handling — Loader-time validation) | 21 |
| DAG-aware completion gates in records-tracker + phase-runner; child HTTP requests count toward `requests_sent` (In-Scope) | 22 |
| `BranchStats` published via `CreditPhaseCompleteMessage` and exported to `profile_export_aiperf.json` (In-Scope) | 23 |
| `--num-conversations` autodefault (`_count_dag_root_entries` + `_is_forking_dataset`); refusal to default `--request-count` for forking datasets (Behavior Decisions) | 24 |
| `--no-fixed-schedule` (`InputConfig.disable_auto_fixed_schedule`) (In-Scope) | 25 |
| Worker FORK pin refcount call sites (`pin_for_fork_child` / `release_fork_child` in worker.py) (d. Worker Layer) | 26 |
| DAG cross-component tests + adversarial timing-mode tests (Testing Strategy — Component-integration) | 27 |
| `test_dag_hard_cap.py` and `test_dag_multi_root_payload_bytes.py` (Testing Strategy — Component-integration) | 28 |
| End-to-end DAG integration test against in-repo mock server (Testing Strategy — Integration) | 29 |
| DAG-metadata tagging records test (Testing Strategy — Unit + Records) | 30 |
| Full `dag_jsonl` loader test suite (prereq, delayed, fan-in, multi-gate, pre-session, property-based, adversarial-full, pathological, plugin) (Testing Strategy — Unit + Loader) | 31 |
| `docs/benchmark-modes/dag.md` (Documentation Updates) | 32 |
| Three-file (or four-file) sync of `dag_jsonl` mention; auto-regenerated `docs/cli-options.md` and `docs/environment-variables.md` (Documentation Updates) | 33 |

## Out of scope

The following items are deliberately NOT included in Plan 2 per spec section "Out-of-Scope (explicit)":

- Cache-bust marker injection (`_apply_cache_bust_*` in worker.py, `cache_bust_marker` / `cache_bust_target` plumbing on `Credit` / `TurnToSend` / `RecordContext`, `validate_cache_bust_compatibility`)
- AgentX scenario, `_AgentXSettings`, `--scenario` / `--unsafe-override`, `AGENTIC` mode wiring (the `AIPERF_AGENTX_*` env-var family is replaced by `AIPERF_DAG_*` in the orchestrator port)
- Weka loaders (`weka_trace.py`, `weka_parallel_convert.py`, `weka_synth_buf.py`, `weka_trace_models.py`, `semianalysis_cc_traces_weka.py`), `--use-think-time-only`, weka delta-context
- Agentic-replay strategy (`src/aiperf/timing/strategies/agentic_replay.py`), `src/aiperf/timing/trajectory_source.py`, `AGENTIC_REPLAY` mode
- `Turn.reset_context` (only consumer was Weka delta-context; future delta-context loaders can reintroduce it as a deliberate add)
- Plugin-categories split (`accumulator` / `stream_exporter` / `analyzer`) — stay on main's single `ResultsProcessorType`
- Realtime stats overhaul (`_render_realtime_block`, `AccumulatorMetricsSummary`, dynamic `realtime_metrics_interval(ui_type)` resolver)
- ASR loader stack, SageMaker capture loader, additional accuracy benchmarks beyond what is on `main`
- `--ignore-trace-delays` (Weka-flavored)
