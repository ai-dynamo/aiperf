# DAG5 Plan 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the foundation layer of dag5 — data models, endpoint refactor, sister loaders, plugin registry, and tutorials — onto a fresh branch from origin/main, without yet introducing the DAG runtime.

**Architecture:** Single Python branch `ajc/dag5` cut from `origin/main`. Plan 1 lands as a sequence of small, TDD-style commits. Each task is independently testable. End-state: byte-exact replay (`inputs.json`, `raw_payload`) works; tool-call TTFO/OSL is correct on Chat and Responses endpoints; data models for DAG topology exist but are not yet wired into orchestrator (Plan 2).

**Tech Stack:** Python 3.10+, Pydantic v2, pytest + pytest-asyncio + pytest-xdist, uv.

---

## Source-of-Truth Pointers

Plan 1 ports content from `ajc/dag4` to a fresh `ajc/dag5` branch cut from `origin/main`. For verbatim file ports, the implementation step uses `git show ajc/dag4:<path> > <path>` (the dag4 file is already correct AND has no surrounding-code drift on main). When dag4's file diverges from main only in targeted ways, the implementation step shows the targeted diff applied against main's version.

The authoritative spec is [`docs/superpowers/specs/2026-05-04-dag5-best-of-both-design.md`](../specs/2026-05-04-dag5-best-of-both-design.md). Plan 1 is a strict subset of its In-Scope list — see "Spec Coverage" at the bottom of this file for the mapping.

---

### Task 1: Branch creation from origin/main + spec port via cherry-pick

**Files:**
- Create: branch `ajc/dag5` (cut from `origin/main`)
- Cherry-pick: `docs/superpowers/specs/2026-05-04-dag5-best-of-both-design.md` (commits `43b473cb1` and `350ea9bb0` from `ajc/inferencex-agentx-mvp`)

- [ ] **Step 1: Write the failing test**

  No test for branch creation itself. The smoke test is `make first-time-setup` succeeding plus `uv run pytest tests/unit/ -n auto` running clean against an unmodified main. Capture pre-port baseline pass count for later regression comparison.

  ```bash
  git fetch origin
  # Verify the two spec commits exist on the source branch.
  git log --oneline ajc/inferencex-agentx-mvp -- docs/superpowers/specs/ | head -5
  ```

  Expected: at least two commits whose subjects match `docs(specs): dag5 best-of-both DAG branch design` and `docs(specs): fix attribution and path errors in dag5 spec`.

- [ ] **Step 2: Run test to verify it fails**

  Skip — there's nothing to fail before the branch exists. Proceed directly to Step 3.

- [ ] **Step 3: Write minimal implementation**

  ```bash
  git fetch origin
  git checkout -b ajc/dag5 origin/main
  # Cherry-pick the two spec commits in chronological order.
  git cherry-pick 43b473cb1 350ea9bb0
  # Verify the spec landed.
  ls docs/superpowers/specs/2026-05-04-dag5-best-of-both-design.md
  # Refresh the env so any new deps pulled by main since the last setup are present.
  make first-time-setup
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — the unit suite must be green on freshly-cut `ajc/dag5` before any porting begins. Save the pass count to compare against after Plan 1 completes.

- [ ] **Step 5: Commit**

  No commit needed — the cherry-picks already produced commits. Verify with:

  ```bash
  git log --oneline -5
  ```

  Expected: HEAD is `350ea9bb0` (or its rewritten cherry-pick SHA), parent is `43b473cb1`, and the rest are origin/main commits.

---

### Task 2: `ConversationBranchMode` enum

**Files:**
- Modify: `src/aiperf/common/enums/enums.py`
- Test: `tests/unit/common/enums/test_conversation_branch_mode.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import ConversationBranchMode


  class TestConversationBranchMode:
      def test_members_present(self):
          assert ConversationBranchMode.FORK == "fork"
          assert ConversationBranchMode.SPAWN == "spawn"

      def test_string_round_trip(self):
          # ExtensibleStrEnum / CaseInsensitiveStrEnum should accept lower-case and
          # upper-case variants and round-trip through str().
          assert ConversationBranchMode("fork") is ConversationBranchMode.FORK
          assert ConversationBranchMode("FORK") is ConversationBranchMode.FORK
          assert str(ConversationBranchMode.SPAWN) == "spawn"

      @pytest.mark.parametrize(
          "raw,expected",
          [
              ("fork", ConversationBranchMode.FORK),
              ("FORK", ConversationBranchMode.FORK),
              ("Fork", ConversationBranchMode.FORK),
              ("spawn", ConversationBranchMode.SPAWN),
              ("SPAWN", ConversationBranchMode.SPAWN),
          ],
      )
      def test_case_insensitive(self, raw: str, expected: ConversationBranchMode):
          assert ConversationBranchMode(raw) is expected
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/enums/test_conversation_branch_mode.py -v`

  Expected: FAIL with `ImportError: cannot import name 'ConversationBranchMode' from 'aiperf.common.enums'`.

- [ ] **Step 3: Write minimal implementation**

  In `src/aiperf/common/enums/enums.py`, add the enum near the other `Conversation*` enums (immediately above `class ConversationContextMode`):

  ```python
  class ConversationBranchMode(CaseInsensitiveStrEnum):
      """Mode discriminator for ``ConversationBranchInfo``.

      Distinguishes two kinds of DAG branches sharing one primitive:

      - ``FORK``: child inherits the parent's accumulated message context and
        sticky-routes to the parent's worker (prefix-cache locality). Used by
        aiperf's native DAG conversation-forking semantics.
      - ``SPAWN``: child starts with a fresh context, free routing. Used for
        pre-session sub-agent dispatch.
      """

      FORK = "fork"
      """Child inherits parent's turn_list (accumulated message history + captured
      live responses); sticky-routes to parent's worker for prefix-cache locality."""

      SPAWN = "spawn"
      """Child gets a fresh context; free routing (no sticky pin to parent)."""
  ```

  Also add `"ConversationBranchMode"` to the `__all__` list at the bottom of the file (alphabetically placed). Then export it from `src/aiperf/common/enums/__init__.py` by adding the symbol to the existing `from aiperf.common.enums.enums import (...)` block and to its `__all__`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/common/enums/test_conversation_branch_mode.py -v`

  Expected: PASS, all 5 cases.

  Then run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — no regressions.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/enums/enums.py src/aiperf/common/enums/__init__.py tests/unit/common/enums/test_conversation_branch_mode.py
  git commit -s -m "$(cat <<'EOF'
  feat(enums): add ConversationBranchMode enum (FORK / SPAWN)

  Mode discriminator for the upcoming DAG ConversationBranchInfo model.
  FORK = child inherits parent context + sticky route; SPAWN = fresh
  context, free routing. Enum is wired through __init__ and exported.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 3: `prerequisites.py` models

**Files:**
- Create: `src/aiperf/common/enums/enums.py` (add `PrerequisiteKind` enum)
- Create: `src/aiperf/common/models/prerequisites.py`
- Test: `tests/unit/common/models/test_prerequisite_model.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest
  from pydantic import ValidationError

  from aiperf.common.enums import PrerequisiteKind
  from aiperf.common.models.prerequisites import TurnPrerequisite


  class TestPrerequisiteKind:
      def test_members(self):
          assert PrerequisiteKind.SPAWN_JOIN == "spawn_join"
          assert PrerequisiteKind.CHILD_SESSION_COMPLETE == "child_session_complete"
          assert PrerequisiteKind.TIMER == "timer"
          assert PrerequisiteKind.EXTERNAL_EVENT == "external_event"
          assert PrerequisiteKind.BARRIER == "barrier"


  class TestTurnPrerequisite:
      def test_construct_spawn_join(self):
          p = TurnPrerequisite(
              kind=PrerequisiteKind.SPAWN_JOIN,
              branch_id="root:0",
          )
          assert p.kind is PrerequisiteKind.SPAWN_JOIN
          assert p.branch_id == "root:0"
          assert p.child_conversation_ids is None
          assert p.barrier_id is None
          assert p.timer_seconds is None
          assert p.event_name is None

      def test_serialization_round_trip(self):
          p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
          dumped = p.model_dump()
          restored = TurnPrerequisite.model_validate(dumped)
          assert restored == p

      def test_extra_fields_forbidden(self):
          with pytest.raises(ValidationError):
              TurnPrerequisite(
                  kind=PrerequisiteKind.SPAWN_JOIN,
                  branch_id="root:0",
                  unknown_field="boom",
              )

      def test_frozen(self):
          p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
          with pytest.raises(ValidationError):
              p.branch_id = "other:1"

      def test_construct_timer_reserved(self):
          # Reserved kinds construct fine — orchestrator-side validation
          # (validate_for_orchestrator_v1) is a separate concern handled in Plan 2.
          p = TurnPrerequisite(kind=PrerequisiteKind.TIMER, timer_seconds=1.5)
          assert p.kind is PrerequisiteKind.TIMER
          assert p.timer_seconds == 1.5
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_prerequisite_model.py -v`

  Expected: FAIL with `ImportError: cannot import name 'PrerequisiteKind' from 'aiperf.common.enums'`.

- [ ] **Step 3: Write minimal implementation**

  Add `PrerequisiteKind` to `src/aiperf/common/enums/enums.py` immediately after `ConversationBranchMode` (verbatim from `git show ajc/dag4:src/aiperf/common/enums/enums.py | sed -n '120,142p'`):

  ```python
  class PrerequisiteKind(CaseInsensitiveStrEnum):
      """Types of conditions that can gate a turn's dispatch.

      Extensible: v1 orchestrator only honors SPAWN_JOIN; the remaining values
      are reserved and rejected at load time by
      ``validate_for_orchestrator_v1``. Each deferred value is pinned to a
      future orchestrator capability in the DAG prereq-gating design doc.
      """

      SPAWN_JOIN = "spawn_join"
      """All blocking children from a named branch have completed."""

      CHILD_SESSION_COMPLETE = "child_session_complete"
      """A specific child runtime session has completed (reserved)."""

      TIMER = "timer"
      """Wall-clock delay has elapsed (reserved)."""

      EXTERNAL_EVENT = "external_event"
      """Named external signal has been received (reserved)."""

      BARRIER = "barrier"
      """Runtime-diamond join on a shared barrier_id (reserved)."""
  ```

  Add `"PrerequisiteKind"` to enums `__all__` and export from `src/aiperf/common/enums/__init__.py`.

  Then create `src/aiperf/common/models/prerequisites.py` verbatim from dag4:

  ```bash
  git show ajc/dag4:src/aiperf/common/models/prerequisites.py > src/aiperf/common/models/prerequisites.py
  ```

  Verify the file contents:

  ```bash
  cat src/aiperf/common/models/prerequisites.py
  ```

  Expected: defines `TurnPrerequisite(AIPerfBaseModel)` with `model_config = ConfigDict(extra="forbid", frozen=True)` and fields `kind`, `branch_id`, `child_conversation_ids`, `barrier_id`, `timer_seconds`, `event_name`.

  Finally, export `TurnPrerequisite` from `src/aiperf/common/models/__init__.py`:

  ```python
  from aiperf.common.models.prerequisites import TurnPrerequisite
  ```

  Add `"TurnPrerequisite"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 5 new prereq tests, with no regressions.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/enums/enums.py src/aiperf/common/enums/__init__.py src/aiperf/common/models/prerequisites.py src/aiperf/common/models/__init__.py tests/unit/common/models/test_prerequisite_model.py
  git commit -s -m "$(cat <<'EOF'
  feat(models): add PrerequisiteKind enum and TurnPrerequisite model

  Models the prerequisite condition that can gate a Turn's dispatch.
  V1 orchestrator (Plan 2) honors only SPAWN_JOIN; remaining kinds are
  reserved and rejected at load time. Frozen + extra=forbid so authoring
  errors fail loudly.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 4: `Turn.prerequisites` field + `TurnMetadata.has_forks` + `Conversation.agent_depth`

**Files:**
- Modify: `src/aiperf/common/models/dataset_models.py`
- Test: `tests/unit/common/models/test_dataset_models_dag_fields.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from aiperf.common.enums import PrerequisiteKind
  from aiperf.common.models import (
      Conversation,
      Text,
      Turn,
      TurnMetadata,
      TurnPrerequisite,
  )


  class TestTurnPrerequisitesField:
      def test_default_is_empty_list(self):
          t = Turn(texts=[Text(contents=["hi"])])
          assert t.prerequisites == []

      def test_round_trip_with_prereqs(self):
          prereq = TurnPrerequisite(
              kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
          )
          t = Turn(texts=[Text(contents=["hi"])], prerequisites=[prereq])
          dumped = t.model_dump()
          restored = Turn.model_validate(dumped)
          assert restored.prerequisites == [prereq]


  class TestTurnMetadataHasForks:
      def test_default_false(self):
          m = TurnMetadata()
          assert m.has_forks is False

      def test_set_true(self):
          m = TurnMetadata(has_forks=True)
          assert m.has_forks is True

      def test_round_trip(self):
          m = TurnMetadata(has_forks=True, timestamp_ms=1000.0)
          dumped = m.model_dump()
          restored = TurnMetadata.model_validate(dumped)
          assert restored.has_forks is True
          assert restored.timestamp_ms == 1000.0


  class TestConversationAgentDepth:
      def test_default_zero(self):
          c = Conversation(session_id="s1", turns=[Turn(texts=[Text(contents=["hi"])])])
          assert c.agent_depth == 0

      def test_set_depth(self):
          c = Conversation(
              session_id="s1",
              turns=[Turn(texts=[Text(contents=["hi"])])],
              agent_depth=2,
          )
          assert c.agent_depth == 2

      def test_round_trip(self):
          c = Conversation(
              session_id="s1",
              turns=[Turn(texts=[Text(contents=["hi"])])],
              agent_depth=3,
          )
          dumped = c.model_dump()
          restored = Conversation.model_validate(dumped)
          assert restored.agent_depth == 3
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_dataset_models_dag_fields.py -v`

  Expected: FAIL — `Turn` has no `prerequisites` field, `TurnMetadata` has no `has_forks` field, `Conversation` has no `agent_depth` field. Also `TurnPrerequisite` is imported from `aiperf.common.models` (re-export check — must already work after Task 3).

- [ ] **Step 3: Write minimal implementation**

  In `src/aiperf/common/models/dataset_models.py`:

  Add the import at the top of the file (after the existing imports from `aiperf.common.models.base_models`):

  ```python
  from aiperf.common.models.prerequisites import TurnPrerequisite
  ```

  Inside `class TurnMetadata(AIPerfBaseModel)`, after the existing `delay_ms` field and before the close of the class, add:

  ```python
      has_forks: bool = Field(
          default=False,
          description="True if this turn triggers any FORK-mode branch. Stamped at "
          "load time by the dag_jsonl loader's topology walk so the sticky router "
          "can defer parent-session eviction until all forks have spawned. Stays "
          "False on non-DAG datasets.",
      )
  ```

  Inside `class Turn(AIPerfBaseModel)`, after the existing `videos` field, add:

  ```python
      prerequisites: list[TurnPrerequisite] = Field(
          default_factory=list,
          description="Conditions gating dispatch of this turn (DAG authoring). "
          "Attached to the gated turn; resolved against branch_ids declared on "
          "prior turns. Empty on non-DAG datasets.",
      )
  ```

  Inside `class Conversation(AIPerfBaseModel)`, after the existing `accuracy_task` field, add:

  ```python
      agent_depth: int = Field(
          default=0,
          description="Static DAG nesting level — 0 for sampleable roots, "
          "``parent_depth + 1`` for fork-spawned descendants. Stamped at "
          "load time by the dag_jsonl loader's topology walk; non-DAG "
          "conversations stay at the default 0. The sampler treats "
          "``agent_depth == 0`` as the root predicate (children are seeded "
          "from their parent's worker context, never sampled directly).",
      )
  ```

  Note: `Conversation.metadata()` and `Turn.metadata()` are NOT touched in this task — Task 5 handles the metadata projection.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 7 new dataset_models tests, no regressions.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/dataset_models.py tests/unit/common/models/test_dataset_models_dag_fields.py
  git commit -s -m "$(cat <<'EOF'
  feat(models): add DAG fields to Turn, TurnMetadata, Conversation

  Three additive fields landed together because they share a single test
  surface and are all default-zero/False/empty on non-DAG datasets:

  - Turn.prerequisites — conditions gating this turn's dispatch
  - TurnMetadata.has_forks — sticky-router defer hint stamped at load time
  - Conversation.agent_depth — static DAG nesting level for root sampling

  No projection logic yet — Conversation.metadata() lands in the next
  commit so the failure mode (silently dropped prereqs) shows up
  separately.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 5: `Conversation.metadata()` prerequisites projection fix

**Files:**
- Modify: `src/aiperf/common/models/dataset_models.py`
- Test: `tests/unit/common/models/test_conversation_metadata_prereq_projection.py`

This is the dag4 latent-bug fix mentioned in the spec (§b. Data Models): on main, `Conversation.metadata()` builds `TurnMetadata` directly without projecting `turn.prerequisites`, so prereqs are silently dropped on the way to consumers that read `ConversationMetadata.turns[i].prerequisites`.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from aiperf.common.enums import PrerequisiteKind
  from aiperf.common.models import (
      Conversation,
      Text,
      Turn,
      TurnPrerequisite,
  )


  class TestConversationMetadataProjectsPrereqs:
      def test_metadata_carries_prereqs(self):
          prereq = TurnPrerequisite(
              kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
          )
          conv = Conversation(
              session_id="conv-a",
              turns=[
                  Turn(texts=[Text(contents=["root"])]),
                  Turn(texts=[Text(contents=["join"])], prerequisites=[prereq]),
              ],
          )
          meta = conv.metadata()
          assert meta.conversation_id == "conv-a"
          assert len(meta.turns) == 2
          assert meta.turns[0].prerequisites == []
          assert meta.turns[1].prerequisites == [prereq]

      def test_metadata_default_empty_prereqs(self):
          conv = Conversation(
              session_id="conv-b",
              turns=[Turn(texts=[Text(contents=["only"])])],
          )
          meta = conv.metadata()
          assert meta.turns[0].prerequisites == []
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_conversation_metadata_prereq_projection.py -v`

  Expected: FAIL on `assert meta.turns[1].prerequisites == [prereq]` — prereqs are dropped because `Turn.metadata()` does not include them.

- [ ] **Step 3: Write minimal implementation**

  In `src/aiperf/common/models/dataset_models.py`, modify `Turn.metadata()` from:

  ```python
      def metadata(self) -> TurnMetadata:
          """Get the metadata of the turn."""
          return TurnMetadata(
              timestamp_ms=self.timestamp,
              delay_ms=self.delay,
          )
  ```

  to:

  ```python
      def metadata(self) -> TurnMetadata:
          """Get the metadata of the turn."""
          return TurnMetadata(
              timestamp_ms=self.timestamp,
              delay_ms=self.delay,
              prerequisites=list(self.prerequisites),
          )
  ```

  Also add the matching field to `TurnMetadata` (since `Turn.metadata()` returns one):

  ```python
      prerequisites: list["TurnPrerequisite"] = Field(
          default_factory=list,
          description="Conditions gating dispatch of this turn (DAG projection). "
          "Mirrors ``Turn.prerequisites`` so consumers of "
          "``ConversationMetadata`` can reach prereqs without holding the full "
          "Turn list.",
      )
  ```

  (The forward reference `"TurnPrerequisite"` resolves through the module-level import added in Task 4.)

  `Conversation.metadata()` itself does not need changes — it already calls `turn.metadata()`, which now carries prereqs through.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — both new tests, plus Task 4's tests (which now exercise the projection path), plus no regressions.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/dataset_models.py tests/unit/common/models/test_conversation_metadata_prereq_projection.py
  git commit -s -m "$(cat <<'EOF'
  fix(models): project Turn.prerequisites into TurnMetadata

  Conversation.metadata() builds ConversationMetadata.turns by calling
  Turn.metadata(), which until now returned a TurnMetadata with only
  timestamp_ms / delay_ms set. Prereqs declared on a Turn were silently
  dropped, so the orchestrator (Plan 2) would never see SPAWN_JOIN
  gates that travelled through the metadata path.

  Add prerequisites field to TurnMetadata and project it from
  Turn.metadata().

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 6: `ConversationBranchInfo` model with `dispatch_timing` literal

**Files:**
- Create: `src/aiperf/common/models/branch.py`
- Test: `tests/unit/common/models/test_branch_model.py`

This task **adapts** dag4's `branch.py`: the spec mandates a `dispatch_timing: Literal["pre", "post"]` field (default `"post"`, `"pre"` reserved for SPAWN), not dag4's `is_background: bool` + `subagent_type: SubagentType | None` shape (SubagentType is the AgentX surface, explicitly out-of-scope per spec §"Out-of-Scope").

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest
  from pydantic import ValidationError

  from aiperf.common.enums import ConversationBranchMode
  from aiperf.common.models.branch import ConversationBranchInfo


  class TestConversationBranchInfoDefaults:
      def test_fork_default_dispatch_post(self):
          b = ConversationBranchInfo(
              branch_id="root:0",
              child_conversation_ids=["c1"],
              mode=ConversationBranchMode.FORK,
          )
          assert b.dispatch_timing == "post"
          assert b.mode is ConversationBranchMode.FORK

      def test_spawn_default_dispatch_post(self):
          b = ConversationBranchInfo(
              branch_id="root:0",
              child_conversation_ids=["c1"],
              mode=ConversationBranchMode.SPAWN,
          )
          assert b.dispatch_timing == "post"

      def test_spawn_can_set_pre(self):
          b = ConversationBranchInfo(
              branch_id="root:0",
              child_conversation_ids=["c1"],
              mode=ConversationBranchMode.SPAWN,
              dispatch_timing="pre",
          )
          assert b.dispatch_timing == "pre"


  class TestConversationBranchInfoValidator:
      def test_fork_rejects_pre(self):
          # FORK + dispatch_timing="pre" is invalid: only SPAWN can be pre-dispatched.
          with pytest.raises(ValidationError) as exc_info:
              ConversationBranchInfo(
                  branch_id="root:0",
                  child_conversation_ids=["c1"],
                  mode=ConversationBranchMode.FORK,
                  dispatch_timing="pre",
              )
          # The error message should explain why and how to fix.
          assert "SPAWN" in str(exc_info.value) or "spawn" in str(exc_info.value)

      def test_invalid_dispatch_value(self):
          with pytest.raises(ValidationError):
              ConversationBranchInfo(
                  branch_id="root:0",
                  child_conversation_ids=["c1"],
                  mode=ConversationBranchMode.SPAWN,
                  dispatch_timing="bogus",
              )


  class TestConversationBranchInfoSerialization:
      def test_round_trip(self):
          b = ConversationBranchInfo(
              branch_id="root:0",
              child_conversation_ids=["c1", "c2"],
              mode=ConversationBranchMode.SPAWN,
              dispatch_timing="pre",
          )
          dumped = b.model_dump()
          restored = ConversationBranchInfo.model_validate(dumped)
          assert restored == b
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_branch_model.py -v`

  Expected: FAIL with `ImportError: cannot import name 'ConversationBranchInfo' from 'aiperf.common.models.branch'` (the file does not exist).

- [ ] **Step 3: Write minimal implementation**

  Create `src/aiperf/common/models/branch.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  from typing import Literal

  from pydantic import Field, ValidationInfo, field_validator

  from aiperf.common.enums import ConversationBranchMode
  from aiperf.common.models.base_models import AIPerfBaseModel


  class ConversationBranchInfo(AIPerfBaseModel):
      """Describes a DAG branch from a parent turn to one or more child conversations.

      One primitive unifies aiperf's native FORK semantics (child inherits
      parent turn_list + sticky-routes to parent worker) with pre-session
      SPAWN semantics (fresh context, free routing, optionally dispatched
      before the parent's first turn). The ``mode`` field discriminates
      the two; the ``dispatch_timing`` field gates pre-session SPAWN.
      """

      branch_id: str = Field(
          description="Deterministic branch ID, shape "
          "'<parent_session_id>:<parent_turn_index>'.",
      )
      child_conversation_ids: list[str] = Field(
          description="Child conversation_ids dispatched when this branch fires.",
      )
      mode: ConversationBranchMode = Field(
          description="FORK = child inherits parent context; "
          "SPAWN = fresh context.",
      )
      dispatch_timing: Literal["pre", "post"] = Field(
          default="post",
          description="When the children dispatch relative to the parent's "
          "first turn. ``post`` (default) fires after the parent turn that "
          "declares the branch completes — both FORK and SPAWN children. "
          "``pre`` fires the children before the parent's first turn — "
          "reserved for SPAWN (background pre-session sub-agent dispatch); "
          "the field validator rejects ``pre`` when mode is FORK.",
      )

      @field_validator("dispatch_timing")
      @classmethod
      def _validate_pre_requires_spawn(
          cls, v: Literal["pre", "post"], info: ValidationInfo
      ) -> Literal["pre", "post"]:
          if v == "pre" and info.data.get("mode") == ConversationBranchMode.FORK:
              raise ValueError(
                  "dispatch_timing='pre' is reserved for SPAWN-mode branches "
                  "(background pre-session sub-agent dispatch). FORK children "
                  "inherit the parent's context and must dispatch after the "
                  "parent turn — drop dispatch_timing or change mode to SPAWN."
              )
          return v
  ```

  Export from `src/aiperf/common/models/__init__.py`:

  ```python
  from aiperf.common.models.branch import ConversationBranchInfo
  ```

  Add `"ConversationBranchInfo"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 6 new branch tests, no regressions.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/branch.py src/aiperf/common/models/__init__.py tests/unit/common/models/test_branch_model.py
  git commit -s -m "$(cat <<'EOF'
  feat(models): add ConversationBranchInfo with dispatch_timing literal

  One primitive carries both FORK (inherit parent context, sticky route)
  and SPAWN (fresh context). dispatch_timing="pre" is the pre-session
  SPAWN escape hatch for background sub-agent dispatch; field validator
  rejects pre+FORK with a fix-it message.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 7: `BranchStats` model

**Files:**
- Create: `src/aiperf/common/models/branch_stats.py`
- Test: `tests/unit/common/models/test_branch_stats_model.py`

This task **adapts** dag4's `branch_stats.py`: the spec calls out a `joins_suppressed` counter (§Behavior Decisions, "BranchStats.joins_suppressed tracks how many joins ended this way") for stop-condition drains. The dag4 version names the equivalent counter `children_truncated`. We use the spec's name (`joins_suppressed`) since this is the dag5 contract; everything else mirrors dag4.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from aiperf.common.models.branch_stats import BranchStats


  class TestBranchStatsDefaults:
      def test_all_counters_zero(self):
          s = BranchStats()
          assert s.children_spawned == 0
          assert s.children_completed == 0
          assert s.children_errored == 0
          assert s.parents_suspended == 0
          assert s.parents_resumed == 0
          assert s.parents_failed_due_to_child_error == 0
          assert s.joins_suppressed == 0


  class TestBranchStatsIncrement:
      def test_set_and_serialize(self):
          s = BranchStats(
              children_spawned=5,
              children_completed=4,
              children_errored=1,
              parents_suspended=2,
              parents_resumed=2,
              joins_suppressed=3,
          )
          d = s.stats_dict()
          assert d["children_spawned"] == 5
          assert d["children_completed"] == 4
          assert d["joins_suppressed"] == 3
          assert d["parents_failed_due_to_child_error"] == 0

      def test_round_trip(self):
          s = BranchStats(joins_suppressed=7, children_spawned=10)
          restored = BranchStats.model_validate(s.model_dump())
          assert restored == s
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_branch_stats_model.py -v`

  Expected: FAIL with `ImportError: cannot import name 'BranchStats' from 'aiperf.common.models.branch_stats'`.

- [ ] **Step 3: Write minimal implementation**

  Create `src/aiperf/common/models/branch_stats.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pydantic import Field

  from aiperf.common.models.base_models import AIPerfBaseModel


  class BranchStats(AIPerfBaseModel):
      """Counters for DAG branch orchestration observability.

      Exported as part of ``ProfileResults.branch_stats`` so DAG-shaped runs
      (FORK or SPAWN mode) can be inspected (how many children dispatched,
      how many parents resumed after joins, etc.). Stats are mode-agnostic.
      """

      children_spawned: int = Field(
          default=0,
          description="Number of DAG child sessions that were successfully dispatched.",
      )
      children_completed: int = Field(
          default=0,
          description="Number of DAG child sessions that reached their leaf turn "
          "and were joined back.",
      )
      children_errored: int = Field(
          default=0,
          description="Number of DAG child sessions that terminated with an error.",
      )
      parents_suspended: int = Field(
          default=0,
          description="Number of parent sessions that paused to await an outstanding "
          "branch join.",
      )
      parents_resumed: int = Field(
          default=0,
          description="Number of parent sessions that resumed with a join turn after "
          "all children completed.",
      )
      parents_failed_due_to_child_error: int = Field(
          default=0,
          description="Number of parent sessions that were aborted because a child "
          "errored under AIPERF_DAG_FAIL_FAST=true.",
      )
      joins_suppressed: int = Field(
          default=0,
          description="Number of joins released without firing because a stop "
          "condition (typically the --request-count cap) blocked the gated child "
          "from dispatching. Counts each join once. Reportable but not a failure.",
      )

      def stats_dict(self) -> dict[str, int]:
          """Snapshot the counters as a plain dict (stable shape for exporters)."""
          return self.model_dump()
  ```

  Export from `src/aiperf/common/models/__init__.py`:

  ```python
  from aiperf.common.models.branch_stats import BranchStats
  ```

  Add `"BranchStats"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 3 new BranchStats tests, no regressions.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/branch_stats.py src/aiperf/common/models/__init__.py tests/unit/common/models/test_branch_stats_model.py
  git commit -s -m "$(cat <<'EOF'
  feat(models): add BranchStats with joins_suppressed counter

  Counters for DAG branch orchestration observability — exported to
  ProfileResults.branch_stats. joins_suppressed tracks joins released
  without firing due to a stop-condition cap (per dag5 spec); other
  counters mirror dag4 verbatim.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 8: `ToolCallResponseData` rename `text` → `tool_call_text` + add `content`

**Files:**
- Modify: `src/aiperf/common/models/record_models.py`
- Test: `tests/unit/common/models/test_tool_call_response_data.py`

This is a hard rename — no `Field(alias=...)` for back-compat. Every consumer of `ToolCallResponseData.text` must be updated in the same task. The dag4 reference shape is in `git show ajc/dag4:src/aiperf/common/models/record_models.py` lines 811-840.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.models.record_models import ToolCallResponseData


  class TestToolCallResponseDataShape:
      def test_required_field_renamed(self):
          # Old name `text` no longer exists; `tool_call_text` is the field.
          d = ToolCallResponseData(tool_call_text="get_weather('SF')")
          assert d.tool_call_text == "get_weather('SF')"
          assert d.content is None

      def test_old_name_rejected(self):
          # Rename is hard — passing the old kwarg must fail.
          with pytest.raises(TypeError):
              ToolCallResponseData(text="get_weather('SF')")

      def test_with_prose_content(self):
          d = ToolCallResponseData(
              tool_call_text="get_weather('SF')",
              content="Let me check the weather.",
          )
          assert d.tool_call_text == "get_weather('SF')"
          assert d.content == "Let me check the weather."

      def test_get_text_combines_content_then_tool_call(self):
          d = ToolCallResponseData(
              tool_call_text="get_weather('SF')",
              content="Let me check the weather. ",
          )
          # get_text returns content first, then tool_call_text — matches the
          # observed wire order (prose precedes the dispatch in agent traces).
          assert d.get_text() == "Let me check the weather. get_weather('SF')"

      def test_get_text_pure_tool_call(self):
          d = ToolCallResponseData(tool_call_text="get_weather('SF')")
          assert d.get_text() == "get_weather('SF')"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_tool_call_response_data.py -v`

  Expected: FAIL — `text` is the field name on main; `tool_call_text` and `content` are absent.

- [ ] **Step 3: Write minimal implementation**

  In `src/aiperf/common/models/record_models.py`, replace the existing `ToolCallResponseData` class with the dag4 version:

  ```python
  class ToolCallResponseData(BaseResponseData):
      """Parsed tool-call response data (streaming delta or complete message).

      Mirrors the ``ReasoningResponseData`` shape — two fields, one for the
      type's primary content and one for any prose that arrived alongside
      it. Both contribute to client-side OSL via :meth:`get_text`; the
      distinct fields let downstream metrics that want to categorise output
      (e.g. "what fraction of OSL was tool-call dispatch?") read each
      portion separately.
      """

      tool_call_text: str
      """Combined model-generated text from tool calls — every call's
      ``function.name`` and ``function.arguments`` concatenated in
      ``output[]`` order."""

      content: str | None = None
      """Optional prose ``content`` emitted alongside the tool calls in the
      same chunk/message. Carries the prose portion when the model talks
      while dispatching a tool (~18% of turns in agentic traffic) so
      client-side OSL counts both portions and matches the server's
      ``usage.completion_tokens``. ``None`` when the response is pure
      tool-call (no prose accompanying the dispatch)."""

      def get_text(self) -> str:
          """Return ``content`` followed by ``tool_call_text`` — the
          combined string the tokeniser sees for this response."""
          return (self.content or "") + self.tool_call_text
  ```

  Then sweep the rest of the codebase for `ToolCallResponseData(text=...)` callers and rename. On origin/main, the only producer is `src/aiperf/endpoints/openai_chat.py`:

  ```bash
  grep -rn "ToolCallResponseData" src/ tests/ | grep -v ".dag4"
  ```

  Update each call site. Most likely only `src/aiperf/endpoints/openai_chat.py` line ~265 (`return ToolCallResponseData(text=tool_call_text)`) — change `text=` to `tool_call_text=`. Tasks 12 and 13 will fully refactor those endpoint files; this task only changes the kwarg name so the model layer stays internally consistent.

  Also sweep tests:

  ```bash
  grep -rn "ToolCallResponseData" tests/
  ```

  Update any test that constructs it with the old kwarg.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — the 5 new ToolCallResponseData tests, plus existing endpoint tests still pass after the rename.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/record_models.py src/aiperf/endpoints/openai_chat.py tests/
  git commit -s -m "$(cat <<'EOF'
  refactor(models): rename ToolCallResponseData.text -> tool_call_text + content

  Mirror ReasoningResponseData's two-field shape so chunks carrying both
  prose content AND a tool_call dispatch (~18% of agent turns) keep both
  portions on the parsed response. get_text() returns content followed by
  tool_call_text — the order the wire produced them.

  Hard rename, no alias — the only producer on main is openai_chat.py and
  it's updated in this commit to keep the model layer consistent before
  the larger endpoint refactor in later tasks.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 9: `Turn.raw_payload` field (no RecordContext refactor in Plan 1)

**Files:**
- Modify: `src/aiperf/common/models/dataset_models.py`
- Test: `tests/unit/common/models/test_turn_raw_payload.py`

The original task brief mentioned `RecordContext` as a possible inclusion. Reviewed dag4's `record_models.py` diff: introducing `RecordContext` as a new `RequestInfo` superclass is a **major structural refactor** that touches `inference_client._enrich_request_record`, the worker → record-processor ZMQ hop, and downstream consumers — all DAG-orchestrator-aware territory. **Defer the entire `RecordContext` work to Plan 2.**

What Plan 1 *does* need is `Turn.raw_payload` — required by the `inputs_json` and `raw_payload` loaders (Tasks 15-16) and the raw endpoint (Task 14). On origin/main `Turn.raw_payload` does not exist; on dag4 it does. Add it here so the loader tasks can construct turns.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from aiperf.common.models import Turn


  class TestTurnRawPayload:
      def test_default_none(self):
          t = Turn()
          assert t.raw_payload is None

      def test_set_and_round_trip(self):
          payload = {
              "messages": [{"role": "user", "content": "hi"}],
              "model": "Qwen/Qwen3-0.6B",
              "max_tokens": 32,
          }
          t = Turn(role="user", raw_payload=payload)
          assert t.raw_payload == payload
          restored = Turn.model_validate(t.model_dump())
          assert restored.raw_payload == payload

      def test_raw_payload_does_not_disturb_other_fields(self):
          t = Turn(role="user", raw_payload={"messages": []})
          assert t.texts == []
          assert t.images == []
          assert t.raw_messages is None
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/common/models/test_turn_raw_payload.py -v`

  Expected: FAIL — `Turn` has no `raw_payload` field; constructor rejects the kwarg.

- [ ] **Step 3: Write minimal implementation**

  In `src/aiperf/common/models/dataset_models.py`, inside `class Turn(AIPerfBaseModel)`, after the `videos` field (and before `prerequisites` from Task 4), add:

  ```python
      raw_payload: dict[str, Any] | None = Field(
          default=None,
          description="Complete pre-built API request payload for verbatim replay. "
          "When set, bypasses all endpoint payload construction (format_payload) "
          "and sends this dict directly to the transport. Populated by the "
          "raw_payload, inputs_json, and mooncake_trace (payload mode) loaders. "
          "Mutually exclusive with normal turn-content fields in spirit, but no "
          "validator enforces that — loaders construct one or the other.",
      )
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/common/models/dataset_models.py tests/unit/common/models/test_turn_raw_payload.py
  git commit -s -m "$(cat <<'EOF'
  feat(models): add Turn.raw_payload for verbatim API replay

  Pre-built API request payload sent directly to the transport without
  endpoint formatting. Populated by the upcoming raw_payload, inputs_json,
  and mooncake_trace (payload mode) loaders. Endpoint-side consumption
  lands with the raw endpoint and chat/responses refactors.

  RecordContext / RequestInfo split (also in dag4) is deferred to Plan 2 —
  it's a structural worker-side refactor, not a model-only addition.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 10: `response_mixin.py` port

**Files:**
- Create: `src/aiperf/endpoints/response_mixin.py`
- Test: `tests/unit/endpoints/test_response_mixin.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from unittest.mock import MagicMock

  import pytest

  from aiperf.common.models import (
      InferenceServerResponse,
      TextResponseData,
  )
  from aiperf.endpoints.response_mixin import JMESPathResponseMixin


  class _StubMixinHost(JMESPathResponseMixin):
      """Minimal subclass that supplies the attributes the mixin reads.

      The mixin reads ``self.model_endpoint.endpoint.extra`` for the
      ``response_field`` config and uses ``self.info`` / ``self.error`` /
      ``self.warning`` for logging. Stub them all.
      """

      def __init__(self, response_field: str | None):
          extra = [("response_field", response_field)] if response_field else []
          self.model_endpoint = MagicMock()
          self.model_endpoint.endpoint.extra = extra
          self.logged_info: list[str] = []
          self.logged_error: list[str] = []
          self.logged_warning: list[str] = []
          self._init_response_parser()

      def info(self, msg: str) -> None:
          self.logged_info.append(msg)

      def error(self, msg: str) -> None:
          self.logged_error.append(msg)

      def warning(self, msg: str) -> None:
          self.logged_warning.append(msg)

      # Auto-detect implementation just wraps the JSON in a TextResponseData
      # so the test can verify the fallback path fires.
      def auto_detect_and_extract(self, json_obj):
          if isinstance(json_obj, dict) and "auto" in json_obj:
              return TextResponseData(text=json_obj["auto"])
          return None

      def make_text_response_data(self, text: str):
          return TextResponseData(text=text)

      def convert_to_response_data(self, value):
          return TextResponseData(text=str(value))


  def _build_response(payload: bytes | str | None, json_obj=None):
      r = InferenceServerResponse(perf_ns=42, payload=payload)
      if json_obj is not None:
          # Pre-cache the parsed json so get_json returns it.
          r._cached_json = json_obj  # noqa: SLF001
      return r


  class TestJMESPathResponseMixinCompile:
      def test_no_response_field_compiles_to_none(self):
          host = _StubMixinHost(response_field=None)
          assert host._compiled_jmespath is None
          # No "Compiled JMESPath" log line.
          assert all("Compiled JMESPath" not in m for m in host.logged_info)

      def test_valid_response_field_compiles(self):
          host = _StubMixinHost(response_field="result.text")
          assert host._compiled_jmespath is not None
          assert any("Compiled JMESPath query" in m for m in host.logged_info)

      def test_malformed_response_field_logs_and_falls_back(self):
          # JMESPath compile errors must NOT raise; the mixin logs and leaves
          # _compiled_jmespath as None so parse_response degrades to auto-detect.
          host = _StubMixinHost(response_field="!!! not valid jmespath !!!")
          assert host._compiled_jmespath is None
          assert host.logged_error, "Expected an error log on compile failure"
          # The error log must mention the auto-detect fallback so the user
          # knows behaviour is degraded but functional.
          assert any(
              "auto-detect" in m.lower() for m in host.logged_error
          ), f"Expected auto-detect mention; got logs={host.logged_error!r}"


  class TestJMESPathResponseMixinParse:
      def test_falls_back_to_auto_detect_on_search_failure(self):
          host = _StubMixinHost(response_field="result.text")
          # JSON missing the expected path -> JMESPath search yields None ->
          # mixin falls back to auto_detect_and_extract.
          r = _build_response(payload=b'{"auto":"hello"}', json_obj={"auto": "hello"})
          parsed = host.parse_response(r)
          assert parsed is not None
          assert isinstance(parsed.data, TextResponseData)
          assert parsed.data.text == "hello"

      def test_jmespath_match_wins(self):
          host = _StubMixinHost(response_field="result.text")
          r = _build_response(
              payload=b'{"result":{"text":"jp"}}',
              json_obj={"result": {"text": "jp"}},
          )
          parsed = host.parse_response(r)
          assert parsed is not None
          assert parsed.data.text == "jp"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/endpoints/test_response_mixin.py -v`

  Expected: FAIL — `aiperf.endpoints.response_mixin` does not exist.

- [ ] **Step 3: Write minimal implementation**

  Port verbatim from dag4:

  ```bash
  git show ajc/dag4:src/aiperf/endpoints/response_mixin.py > src/aiperf/endpoints/response_mixin.py
  cat src/aiperf/endpoints/response_mixin.py
  ```

  Expected file content (full text — verify after the `git show`):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import jmespath

  from aiperf.common.models import InferenceServerResponse, ParsedResponse


  class JMESPathResponseMixin:
      """Mixin: JMESPath + auto-detect response parsing.

      Reads optional ``response_field`` from endpoint.extra to compile a JMESPath
      query used during response parsing.  Falls back to auto-detection when no
      query is configured or when the query fails to match.
      """

      def _init_response_parser(self) -> None:
          extra = self.model_endpoint.endpoint.extra
          extra_dict = dict(extra) if extra else {}
          response_field = extra_dict.get("response_field")
          self._compiled_jmespath = None
          if response_field:
              try:
                  self._compiled_jmespath = jmespath.compile(response_field)
                  self.info(f"Compiled JMESPath query: '{response_field}'")
              except (jmespath.exceptions.JMESPathError, TypeError) as e:
                  self.error(
                      f"Failed to compile JMESPath query {response_field!r}: {e!r}. "
                      "Falling back to auto-detect response parsing — fix or remove "
                      "endpoint.extra.response_field to silence this log."
                  )

      def parse_response(
          self, response: InferenceServerResponse
      ) -> ParsedResponse | None:
          json_obj = response.get_json()
          if not json_obj:
              if text := response.get_text():
                  return ParsedResponse(
                      perf_ns=response.perf_ns, data=self.make_text_response_data(text)
                  )
              return None

          response_data = None
          if self._compiled_jmespath:
              try:
                  if value := self._compiled_jmespath.search(json_obj):
                      response_data = self.convert_to_response_data(value)
              except (jmespath.exceptions.JMESPathError, TypeError) as e:
                  self.warning(f"JMESPath search failed: {e!r}. Trying auto-detection.")

          if not response_data:
              response_data = self.auto_detect_and_extract(json_obj)

          return (
              ParsedResponse(perf_ns=response.perf_ns, data=response_data)
              if response_data
              else None
          )
  ```

  No edits required against dag4 — main and dag4 share the same `InferenceServerResponse` and `ParsedResponse` shapes.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 5 new mixin tests.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/endpoints/response_mixin.py tests/unit/endpoints/test_response_mixin.py
  git commit -s -m "$(cat <<'EOF'
  feat(endpoints): add JMESPathResponseMixin for shared parsing

  Shared mixin used by the upcoming raw endpoint (and any future endpoint
  that wants JMESPath-driven response extraction). Compile failure
  degrades to auto-detect with an actionable error log instead of
  crashing the endpoint at construction.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 11: `BaseEndpoint.build_assistant_turn` + extract_payload_inputs

**Files:**
- Modify: `src/aiperf/endpoints/base_endpoint.py`
- Test: `tests/unit/endpoints/test_base_endpoint_assistant_turn.py`

The dag4 `base_endpoint.py` is 616 lines vs main's 252 lines — the diff is large. Most of the additions cluster into three concerns:

1. `build_assistant_turn(record) -> Turn | None` — the hook this task is named after.
2. Generic `build_messages` / `_render_*_part` skeleton — used by Tasks 12 and 13.
3. `extract_payload_inputs` for media counting — also used by Tasks 12 and 13.

All three land here in one commit because they share imports (`MediaType`, `ExtractedPayload`, `Text`, `Turn`, `ReasoningResponseData`) and the chat/responses refactors immediately depend on them.

`extract_response_data(record)` already exists on main's `BaseEndpoint` — `build_assistant_turn` calls it, so that import path is already correct.

`ExtractedPayload` is the model `extract_payload_inputs` returns. Verify it exists on main:

```bash
git show origin/main:src/aiperf/common/models/__init__.py | grep ExtractedPayload
```

If the symbol is absent (which is likely on main), add a minimal model alongside this task — see step 3.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from unittest.mock import MagicMock

  import pytest

  from aiperf.common.models import (
      InferenceServerResponse,
      ParsedResponse,
      ReasoningResponseData,
      RequestRecord,
      TextResponseData,
      Turn,
  )
  from aiperf.endpoints.base_endpoint import BaseEndpoint


  class _StubEndpoint(BaseEndpoint):
      """Minimum viable concrete BaseEndpoint subclass for assistant-turn tests.

      ``parse_response`` is the only abstract method; we wire it to read the
      pre-canned responses we attach in each test.
      """

      def __init__(self):
          # Bypass real init — BaseEndpoint constructor wants a model_endpoint;
          # the methods under test don't read it. Use object.__setattr__ to
          # short-circuit any pydantic model machinery.
          pass

      def format_payload(self, request_info):
          raise NotImplementedError

      def parse_response(self, response: InferenceServerResponse):
          # The test attaches a pre-built ParsedResponse to each
          # InferenceServerResponse via response._test_parsed; return that.
          return getattr(response, "_test_parsed", None)


  def _resp_with_parsed(parsed: ParsedResponse | None) -> InferenceServerResponse:
      r = InferenceServerResponse(perf_ns=0, payload=b"")
      r._test_parsed = parsed  # noqa: SLF001
      return r


  class TestBuildAssistantTurnDefault:
      def test_text_only_record(self):
          ep = _StubEndpoint()
          record = RequestRecord(
              responses=[
                  _resp_with_parsed(
                      ParsedResponse(perf_ns=0, data=TextResponseData(text="Hello"))
                  ),
                  _resp_with_parsed(
                      ParsedResponse(perf_ns=1, data=TextResponseData(text=", world"))
                  ),
              ]
          )
          turn = ep.build_assistant_turn(record)
          assert turn is not None
          assert turn.role == "assistant"
          assert len(turn.texts) == 1
          assert turn.texts[0].contents == ["Hello, world"]

      def test_reasoning_drops_reasoning_keeps_content(self):
          ep = _StubEndpoint()
          record = RequestRecord(
              responses=[
                  _resp_with_parsed(
                      ParsedResponse(
                          perf_ns=0,
                          data=ReasoningResponseData(
                              content="visible answer",
                              reasoning="hidden chain of thought",
                          ),
                      )
                  ),
              ]
          )
          turn = ep.build_assistant_turn(record)
          assert turn is not None
          assert turn.texts[0].contents == ["visible answer"]
          # reasoning is intentionally dropped — most chat templates do not
          # round-trip it on replay.

      def test_empty_record_returns_none(self):
          ep = _StubEndpoint()
          record = RequestRecord(responses=[])
          assert ep.build_assistant_turn(record) is None

      def test_responses_with_no_data_return_none(self):
          ep = _StubEndpoint()
          record = RequestRecord(
              responses=[_resp_with_parsed(ParsedResponse(perf_ns=0, data=None))]
          )
          assert ep.build_assistant_turn(record) is None
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/endpoints/test_base_endpoint_assistant_turn.py -v`

  Expected: FAIL — `BaseEndpoint` has no `build_assistant_turn` method on main.

- [ ] **Step 3: Write minimal implementation**

  First, ensure `ExtractedPayload` is available. If `git show origin/main:src/aiperf/common/models/__init__.py | grep ExtractedPayload` returns nothing, add to `src/aiperf/common/models/record_models.py` (or a small new file `src/aiperf/common/models/extracted_payload.py`):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pydantic import Field

  from aiperf.common.models.base_models import AIPerfBaseModel


  class ExtractedPayload(AIPerfBaseModel):
      """Single-pass extraction result for tokenisation + media accounting.

      Returned by ``BaseEndpoint.extract_payload_inputs``: tokenisable text in
      ``texts`` plus per-modality counts populated as the walk encounters
      ``image_url`` / ``input_audio`` / ``video_url`` parts. One ``orjson.loads``
      plus one O(n) walk yields everything downstream needs.
      """

      texts: list[str] = Field(
          default_factory=list,
          description="Tokenisable text strings (prompt content, instructions, "
          "tool schemas, replayed assistant tool_calls).",
      )
      image_count: int = Field(
          default=0,
          description="Count of image content parts in the payload.",
      )
      audio_count: int = Field(
          default=0,
          description="Count of audio content parts in the payload.",
      )
      video_count: int = Field(
          default=0,
          description="Count of video content parts in the payload.",
      )
  ```

  Re-export from `src/aiperf/common/models/__init__.py`.

  Then port the dag4 `BaseEndpoint` additions (build_assistant_turn + build_messages skeleton + content-part hooks + extract_payload_inputs) to main's `base_endpoint.py`. The cleanest way is to start from dag4's whole file:

  ```bash
  git show ajc/dag4:src/aiperf/endpoints/base_endpoint.py > /tmp/base_endpoint.dag4.py
  ```

  Diff it against the current main's file and apply only the additive pieces (do **not** drop main-side code that dag4 lacks — main may have endpoint helpers dag4 removed as part of unrelated refactors). The additive surface to add:

  - Imports at top: `from typing import Any, ClassVar`, `from aiperf.common.enums import MediaType`, and add `ExtractedPayload`, `ReasoningResponseData`, `Text`, `Turn` to the existing `from aiperf.common.models import (...)` block.
  - Insert the `build_assistant_turn`, generic `build_messages` skeleton, all six `_render_*_part` hooks, the `PART_TYPES: ClassVar` dict, and the `extract_payload_inputs` method exactly as in dag4 lines 76-432 (between `extract_response_data` and `make_text_response_data`).

  Reference: full dag4 source at `git show ajc/dag4:src/aiperf/endpoints/base_endpoint.py` (616 lines). The crucial method bodies are reproduced in `/tmp/dag5plan/base_endpoint.dag4.py` if you cached the diff.

  Do **not** copy dag4's `RecordContext` / `RequestInfo` consumption changes — those belong to Plan 2.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 4 new build_assistant_turn tests, no regressions on existing `test_base_endpoint.py` or other endpoint tests.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/endpoints/base_endpoint.py src/aiperf/common/models/record_models.py src/aiperf/common/models/__init__.py tests/unit/endpoints/test_base_endpoint_assistant_turn.py
  git commit -s -m "$(cat <<'EOF'
  feat(endpoints): add build_assistant_turn, build_messages skeleton, extract_payload_inputs

  Three additions land together because chat/responses refactors in the
  next commits depend on all three:

  - build_assistant_turn(record) -> Turn | None: capture the model's reply
    for context replay (subsequent turns + FORK-mode DAG children).
    Default keeps text + ReasoningResponseData.content; drops reasoning.
  - build_messages + _render_*_part hooks: generic Turn -> wire-message
    skeleton; openai_chat / openai_responses override the part-type names.
  - extract_payload_inputs: single O(n) walk yielding tokenisable text +
    per-modality counts, replacing endpoint-specific double-walks.

  Adds ExtractedPayload model so single-pass extraction has a typed
  return shape.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 12: `openai_responses.py` refactor

**Files:**
- Modify: `src/aiperf/endpoints/openai_responses.py`
- Test: `tests/unit/endpoints/test_openai_responses_dag5.py` (new file; don't disturb the existing `test_responses_endpoint.py`)

Three behaviours land together:

1. **SSE: `response.function_call_arguments.delta` and `.done`** surface as `ToolCallResponseData` so TTFO fires and OSL is correct on tool-using turns.
2. **Non-streaming: walk `output[]` for `function_call` items**, with precedence `reasoning > message > function_call`.
3. **Drop the synthetic `instructions -> {role: "system"}` message insertion** in `format_payload` — `instructions` lives in the Responses-API top-level field, not in `input[]`. The `texts.insert(0, instructions)` in `extract_payload_inputs` is preserved (separate concern: tokenisation).

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.enums import ModelSelectionStrategy
  from aiperf.common.models import (
      InferenceServerResponse,
      ToolCallResponseData,
  )
  from aiperf.common.models.model_endpoint_info import (
      EndpointInfo,
      ModelEndpointInfo,
      ModelInfo,
      ModelListInfo,
  )
  from aiperf.common.models.record_models import (
      ReasoningResponseData,
      TextResponseData,
  )
  from aiperf.endpoints.openai_responses import ResponsesEndpoint
  from aiperf.plugin.enums import EndpointType
  from tests.unit.endpoints.conftest import (
      create_endpoint_with_mock_transport,
      create_model_endpoint,
      create_request_info,
  )


  def _sse(json_obj):
      """Build an InferenceServerResponse pre-cached with a parsed SSE event."""
      import orjson

      payload = orjson.dumps(json_obj)
      r = InferenceServerResponse(perf_ns=42, payload=payload)
      return r


  @pytest.fixture
  def streaming_endpoint():
      ep_info = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
      return create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)


  @pytest.fixture
  def endpoint():
      ep_info = create_model_endpoint(EndpointType.RESPONSES)
      return create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)


  class TestFunctionCallArgumentsDelta:
      def test_delta_event_emits_tool_call_data(self, streaming_endpoint):
          event = {
              "type": "response.function_call_arguments.delta",
              "delta": '{"city":"SF"',
          }
          parsed = streaming_endpoint._parse_streaming_response(event, perf_ns=1)
          assert parsed is not None
          assert isinstance(parsed.data, ToolCallResponseData)
          assert parsed.data.tool_call_text == '{"city":"SF"'

      def test_delta_event_with_no_delta_returns_none(self, streaming_endpoint):
          event = {"type": "response.function_call_arguments.delta"}
          parsed = streaming_endpoint._parse_streaming_response(event, perf_ns=1)
          assert parsed is None


  class TestExtractResponseContentFunctionCallWalk:
      def test_function_call_alone_emits_tool_call_data(self, endpoint):
          json_obj = {
              "object": "response",
              "output": [
                  {
                      "type": "function_call",
                      "name": "get_weather",
                      "arguments": '{"city":"SF"}',
                  }
              ],
          }
          data = endpoint._extract_response_content(json_obj)
          assert isinstance(data, ToolCallResponseData)
          assert "get_weather" in data.tool_call_text
          assert '"city":"SF"' in data.tool_call_text
          assert data.content is None

      def test_message_and_function_call_message_wins(self, endpoint):
          # Precedence: reasoning > message > function_call.
          # message present -> TextResponseData, function_call dropped.
          json_obj = {
              "object": "response",
              "output": [
                  {
                      "type": "message",
                      "content": [{"type": "output_text", "text": "Sure!"}],
                  },
                  {
                      "type": "function_call",
                      "name": "get_weather",
                      "arguments": '{}',
                  },
              ],
          }
          data = endpoint._extract_response_content(json_obj)
          assert isinstance(data, TextResponseData)
          assert data.text == "Sure!"

      def test_reasoning_wins_over_message_and_function_call(self, endpoint):
          json_obj = {
              "object": "response",
              "output": [
                  {
                      "type": "reasoning",
                      "summary": [{"type": "summary_text", "text": "thinking..."}],
                  },
                  {
                      "type": "message",
                      "content": [{"type": "output_text", "text": "answer"}],
                  },
                  {
                      "type": "function_call",
                      "name": "f",
                      "arguments": "{}",
                  },
              ],
          }
          data = endpoint._extract_response_content(json_obj)
          assert isinstance(data, ReasoningResponseData)
          assert data.reasoning == "thinking..."
          assert data.content == "answer"


  class TestInstructionsHandling:
      def test_instructions_no_longer_inserted_as_system_message(
          self, endpoint
      ):
          # ``instructions`` lives in the top-level Responses field, not as a
          # synthetic {"role":"system"} input item. format_payload must NOT
          # add it to ``input``.
          from aiperf.common.models import Text, Turn

          turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
          ep_info = create_model_endpoint(
              EndpointType.RESPONSES,
              extra=[("instructions", "You are helpful.")],
          )
          ep = create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)
          request_info = create_request_info(
              model_endpoint=ep_info, turns=[turn]
          )
          payload = ep.format_payload(request_info)
          # instructions belongs at top level (passed through endpoint.extra).
          # input[] must NOT contain a synthetic system message duplicating it.
          for item in payload.get("input", []):
              if isinstance(item, dict):
                  content = item.get("content")
                  if isinstance(content, str):
                      assert content != "You are helpful."
                  if isinstance(content, list):
                      for part in content:
                          if isinstance(part, dict):
                              assert part.get("text") != "You are helpful."
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/endpoints/test_openai_responses_dag5.py -v`

  Expected: FAIL — `ToolCallResponseData` not produced from function_call_arguments delta on main; `_extract_response_content` does not walk `function_call` items; `instructions` may still be inserted (depends on main's exact behaviour — verify before assuming).

- [ ] **Step 3: Write minimal implementation**

  The cleanest path is to overwrite `src/aiperf/endpoints/openai_responses.py` with the dag4 version, then verify against main's diff to ensure no main-only behaviour is dropped:

  ```bash
  git show ajc/dag4:src/aiperf/endpoints/openai_responses.py > src/aiperf/endpoints/openai_responses.py
  ```

  Then audit the diff against `origin/main:src/aiperf/endpoints/openai_responses.py`:

  ```bash
  diff -u <(git show origin/main:src/aiperf/endpoints/openai_responses.py) src/aiperf/endpoints/openai_responses.py | less
  ```

  Items that should appear (positive list):
  - `import orjson`
  - `from aiperf.common.enums import MediaType`
  - `RequestRecord, ToolCallResponseData` added to model imports
  - `PART_TYPES: ClassVar[dict[MediaType, set[str]]]` for `input_text` / `input_image` / `input_audio`
  - `extract_payload_inputs` override that prepends `instructions` into the texts list
  - `_render_text_part` / `_render_image_part` / `_render_audio_part` emit Responses-API shapes
  - `format_payload` no longer constructs a synthetic system message from `instructions`; instead just builds `input_items` from optional `user_context_message` + `build_messages(turns)`
  - `_parse_streaming_response` handles `response.function_call_arguments.delta` (and `.done` is documented as carrying no replayable content)
  - `_extract_response_content` walks `output[]` for `function_call` items with `reasoning > message > function_call` precedence
  - `build_assistant_turn` override that captures all output items via union of `response.completed.output[]` + `response.output_item.done`

  Items that should NOT appear (this is dag5, not dag4 wholesale):
  - Anything referencing `RecordContext` (Plan 2)
  - Anything referencing `branches`, `agent_depth`, `parent_correlation_id` on `RequestInfo` (Plan 2)

  If dag4's version drifted from main on something orthogonal (e.g. base URL handling), preserve main's behaviour and only port the four targeted changes above.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 5 new dag5 responses tests, plus the existing `test_responses_endpoint.py` suite (which exercises `format_payload`, `parse_response`, multimodal content).

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/endpoints/openai_responses.py tests/unit/endpoints/test_openai_responses_dag5.py
  git commit -s -m "$(cat <<'EOF'
  feat(endpoints): refactor openai_responses for tool-call OSL + payload cleanup

  Three behavioural changes:

  - response.function_call_arguments.delta/.done SSE events emit
    ToolCallResponseData so TTFO fires and OSL is correct on tool-using
    turns (~64% of streaming agent turns previously had NO data-bearing
    event).
  - _extract_response_content walks output[] for function_call items
    with precedence reasoning > message > function_call, matching
    ChatEndpoint precedence.
  - instructions is no longer double-inserted as a synthetic
    {"role":"system"} input item — Responses-API contract puts it at
    the top level. extract_payload_inputs still prepends it for token
    accounting.

  Build_assistant_turn captures every output item (message, function_call,
  reasoning, etc.) via a union of response.completed.output[] and
  response.output_item.done (deduplicated by item id), so FORK-mode
  children see the parent's full output on replay.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 13: `openai_chat.py` refactor

**Files:**
- Modify: `src/aiperf/endpoints/openai_chat.py`
- Test: `tests/unit/endpoints/test_openai_chat_dag5.py`

Two behaviours:

1. **Mixed `content + tool_calls` chunks** return `ToolCallResponseData(tool_call_text=..., content=...)` so the prose portion (~18% of agentic turns) is preserved.
2. **`tool_call_text` rename** consumer-side (the producer was already updated in Task 8; this commit just keeps the file consistent).

Plus the `build_assistant_turn` override that re-assembles streaming `tool_calls` deltas keyed by `index`, and the generic `format_payload` simplification (delegating to `build_messages`).

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.models import (
      RequestRecord,
      Text,
      Turn,
  )
  from aiperf.common.models.record_models import (
      InferenceServerResponse,
      ParsedResponse,
      ReasoningResponseData,
      TextResponseData,
      ToolCallResponseData,
  )
  from aiperf.endpoints.openai_chat import ChatEndpoint
  from aiperf.plugin.enums import EndpointType
  from tests.unit.endpoints.conftest import (
      create_endpoint_with_mock_transport,
      create_model_endpoint,
      create_request_info,
  )


  @pytest.fixture
  def chat_endpoint():
      ep_info = create_model_endpoint(EndpointType.CHAT, streaming=True)
      return create_endpoint_with_mock_transport(ChatEndpoint, ep_info)


  class TestParseChunkContentAndToolCalls:
      def test_chunk_with_content_only(self, chat_endpoint):
          json_obj = {
              "object": "chat.completion.chunk",
              "choices": [{"delta": {"content": "Hello"}}],
          }
          data = chat_endpoint._parse_chunk_data(json_obj)
          assert isinstance(data, TextResponseData)
          assert data.text == "Hello"

      def test_chunk_with_tool_calls_only(self, chat_endpoint):
          json_obj = {
              "object": "chat.completion.chunk",
              "choices": [
                  {
                      "delta": {
                          "tool_calls": [
                              {
                                  "index": 0,
                                  "function": {
                                      "name": "get_weather",
                                      "arguments": '{"city":"SF"}',
                                  },
                              }
                          ]
                      }
                  }
              ],
          }
          data = chat_endpoint._parse_chunk_data(json_obj)
          assert isinstance(data, ToolCallResponseData)
          assert "get_weather" in data.tool_call_text
          assert '"city":"SF"' in data.tool_call_text
          assert data.content is None

      def test_chunk_with_both_content_and_tool_calls(self, chat_endpoint):
          # Mixed content+tool_calls: ~18% of agent turns. Both portions
          # must be preserved so client-OSL matches usage.completion_tokens.
          json_obj = {
              "object": "chat.completion.chunk",
              "choices": [
                  {
                      "delta": {
                          "content": "Let me check. ",
                          "tool_calls": [
                              {
                                  "index": 0,
                                  "function": {
                                      "name": "get_weather",
                                      "arguments": '{"city":"SF"}',
                                  },
                              }
                          ],
                      }
                  }
              ],
          }
          data = chat_endpoint._parse_chunk_data(json_obj)
          assert isinstance(data, ToolCallResponseData)
          assert data.content == "Let me check. "
          assert "get_weather" in data.tool_call_text
          assert data.get_text() == "Let me check. get_weather" + '{"city":"SF"}'

      def test_chunk_with_reasoning_wins(self, chat_endpoint):
          json_obj = {
              "object": "chat.completion.chunk",
              "choices": [
                  {
                      "delta": {
                          "content": "answer",
                          "reasoning": "thinking...",
                          "tool_calls": [
                              {
                                  "index": 0,
                                  "function": {"name": "f", "arguments": "{}"},
                              }
                          ],
                      }
                  }
              ],
          }
          data = chat_endpoint._parse_chunk_data(json_obj)
          assert isinstance(data, ReasoningResponseData)
          assert data.reasoning == "thinking..."
          assert data.content == "answer"


  class TestBuildAssistantTurnReassemblesToolCallDeltas:
      def test_streaming_tool_calls_index_keyed_concat(self, chat_endpoint):
          import orjson

          # Three streaming chunks splitting one tool call across deltas.
          chunks = [
              {
                  "object": "chat.completion.chunk",
                  "choices": [
                      {
                          "delta": {
                              "tool_calls": [
                                  {
                                      "index": 0,
                                      "id": "call_1",
                                      "type": "function",
                                      "function": {"name": "get_weather"},
                                  }
                              ]
                          }
                      }
                  ],
              },
              {
                  "object": "chat.completion.chunk",
                  "choices": [
                      {
                          "delta": {
                              "tool_calls": [
                                  {
                                      "index": 0,
                                      "function": {"arguments": '{"city":'},
                                  }
                              ]
                          }
                      }
                  ],
              },
              {
                  "object": "chat.completion.chunk",
                  "choices": [
                      {
                          "delta": {
                              "tool_calls": [
                                  {
                                      "index": 0,
                                      "function": {"arguments": '"SF"}'},
                                  }
                              ]
                          }
                      }
                  ],
              },
          ]
          responses = [
              InferenceServerResponse(perf_ns=i, payload=orjson.dumps(c))
              for i, c in enumerate(chunks)
          ]
          record = RequestRecord(responses=responses)
          turn = chat_endpoint.build_assistant_turn(record)
          assert turn is not None
          assert turn.role == "assistant"
          assert turn.raw_messages is not None
          assert len(turn.raw_messages) == 1
          msg = turn.raw_messages[0]
          assert msg["role"] == "assistant"
          tool_calls = msg.get("tool_calls")
          assert tool_calls and len(tool_calls) == 1
          tc = tool_calls[0]
          assert tc["id"] == "call_1"
          assert tc["function"]["name"] == "get_weather"
          assert tc["function"]["arguments"] == '{"city":"SF"}'
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/endpoints/test_openai_chat_dag5.py -v`

  Expected: FAIL — main's `_parse_chunk_data` returns `TextResponseData` for content-only and bare `ToolCallResponseData(text=...)` for tool-only; mixed chunks return TextResponseData (dropping the tool_call). `build_assistant_turn` doesn't exist on main's ChatEndpoint.

- [ ] **Step 3: Write minimal implementation**

  Overwrite the chat endpoint with the dag4 version:

  ```bash
  git show ajc/dag4:src/aiperf/endpoints/openai_chat.py > src/aiperf/endpoints/openai_chat.py
  ```

  Verify diff against main:

  ```bash
  diff -u <(git show origin/main:src/aiperf/endpoints/openai_chat.py) src/aiperf/endpoints/openai_chat.py | less
  ```

  Required additions to verify present:
  - `_create_messages` and `_set_message_content` removed (delegated to `BaseEndpoint.build_messages` skeleton from Task 11).
  - `format_payload` directly prepends `system_message` and `user_context_message`, then calls `self.build_messages(turns)`.
  - `format_payload` also applies `turns[-1].extra_body` if present (the dag4 addition for non-native per-turn fields).
  - `_parse_chunk_data` precedence is `reasoning > content_alone -> Text > content+tool_calls -> ToolCall(content=, tool_call_text=) > tool_calls_alone -> ToolCall(tool_call_text=) > content_alone -> Text > None`. Verify with the diff above.
  - `build_assistant_turn` reassembles streaming `tool_calls` keyed by `index`.

  Items NOT to copy from dag4:
  - `RecordContext` integration (Plan 2)
  - `branches` / `agent_depth` references (Plan 2)

  Note: dag4 introduces a `Turn.extra_body` field that the chat endpoint reads. This task adds the field at the same time (3 lines in `dataset_models.py`):

  ```python
      extra_body: dict[str, Any] | None = Field(
          default=None,
          description="Non-native per-turn request-body fields (temperature, "
          "top_p, seed, stop, vendor tunables like ignore_eos/min_tokens). "
          "Merged into the top level of the chat-completions payload at "
          "dispatch time, matching the OpenAI SDK's extra_body convention.",
      )
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 5 new chat tests, plus the existing `test_chat_endpoint.py` and `test_chat_endpoint_parse_response.py`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/endpoints/openai_chat.py src/aiperf/common/models/dataset_models.py tests/unit/endpoints/test_openai_chat_dag5.py
  git commit -s -m "$(cat <<'EOF'
  feat(endpoints): refactor openai_chat for mixed content+tool_calls + replay

  Mixed content+tool_calls chunks (~18% of agent turns) now produce
  ToolCallResponseData(tool_call_text=..., content=...) so client-OSL
  preserves both portions and matches the server's
  usage.completion_tokens. Pure content / pure tool_call paths unchanged.

  format_payload now delegates Turn flattening to BaseEndpoint.build_messages
  and prepends system_message / user_context_message at the top of
  ``messages``. Adds Turn.extra_body merge into the payload.

  build_assistant_turn reassembles streaming tool_calls deltas keyed by
  index so a FORK-mode DAG child replaying the parent's history sees the
  full {role: assistant, content, tool_calls} message.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 14: `raw_endpoint.py`

**Files:**
- Create: `src/aiperf/endpoints/raw_endpoint.py`
- Test: `tests/unit/endpoints/test_raw_endpoint.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.common.models import Turn
  from aiperf.endpoints.raw_endpoint import RawEndpoint
  from aiperf.plugin.enums import EndpointType
  from tests.unit.endpoints.conftest import (
      create_endpoint_with_mock_transport,
      create_model_endpoint,
      create_request_info,
  )


  @pytest.fixture
  def raw_endpoint():
      ep_info = create_model_endpoint(EndpointType.RAW)
      return create_endpoint_with_mock_transport(RawEndpoint, ep_info)


  class TestRawEndpointFormatPayload:
      def test_returns_raw_payload_from_last_turn(self, raw_endpoint, ):
          payload = {
              "messages": [{"role": "user", "content": "hi"}],
              "model": "Qwen/Qwen3-0.6B",
              "max_tokens": 16,
          }
          turn = Turn(role="user", raw_payload=payload)
          ep_info = create_model_endpoint(EndpointType.RAW)
          request_info = create_request_info(model_endpoint=ep_info, turns=[turn])
          assert raw_endpoint.format_payload(request_info) == payload

      def test_no_raw_payload_raises(self, raw_endpoint):
          turn = Turn(role="user")
          ep_info = create_model_endpoint(EndpointType.RAW)
          request_info = create_request_info(model_endpoint=ep_info, turns=[turn])
          with pytest.raises(NotImplementedError) as exc_info:
              raw_endpoint.format_payload(request_info)
          assert "raw_payload" in str(exc_info.value)

      def test_no_turns_raises(self, raw_endpoint):
          ep_info = create_model_endpoint(EndpointType.RAW)
          request_info = create_request_info(model_endpoint=ep_info, turns=[])
          with pytest.raises(NotImplementedError):
              raw_endpoint.format_payload(request_info)
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/endpoints/test_raw_endpoint.py -v`

  Expected: FAIL — `aiperf.endpoints.raw_endpoint` does not exist; `EndpointType.RAW` may also be missing.

- [ ] **Step 3: Write minimal implementation**

  First, add `RAW = "raw"` to `EndpointType` enum (in `src/aiperf/plugin/enums.py` — or wherever `EndpointType` lives; `grep -rn "class EndpointType" src/aiperf/` to locate). This is required so `EndpointType.RAW` round-trips through plugin discovery in Task 18.

  Then port the file verbatim from dag4:

  ```bash
  git show ajc/dag4:src/aiperf/endpoints/raw_endpoint.py > src/aiperf/endpoints/raw_endpoint.py
  cat src/aiperf/endpoints/raw_endpoint.py
  ```

  Expected file content:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  from typing import Any

  from aiperf.common.models import RequestInfo
  from aiperf.endpoints.base_endpoint import BaseEndpoint
  from aiperf.endpoints.response_mixin import JMESPathResponseMixin


  class RawEndpoint(JMESPathResponseMixin, BaseEndpoint):
      """Fallback endpoint for non-standard APIs.

      Does not format payloads or append a URL path.  Parses responses using
      auto-detection with optional JMESPath extraction via ``response_field``
      in endpoint.extra.  Prefer a regular endpoint type (e.g. chat) when the
      target API is supported -- raw payloads bypass formatting regardless of
      endpoint type, and regular endpoints provide structured response parsing.
      """

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self._init_response_parser()

      def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
          if request_info.turns:
              turn = request_info.turns[-1]
              if turn.raw_payload is not None:
                  return turn.raw_payload
          raise NotImplementedError(
              "RawEndpoint does not construct payloads and no raw_payload "
              "found on request turns. Use raw_payload or inputs_json dataset types."
          )
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/endpoints/raw_endpoint.py src/aiperf/plugin/enums.py tests/unit/endpoints/test_raw_endpoint.py
  git commit -s -m "$(cat <<'EOF'
  feat(endpoints): add RawEndpoint for byte-exact payload replay

  Fallback endpoint for non-standard APIs. Does not format payloads or
  append a URL path; parses responses via auto-detection with optional
  JMESPath extraction (response_field in endpoint.extra).

  Pairs with the upcoming raw_payload / inputs_json loaders. Plugin
  registry entry lands in a later commit.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 15: `inputs_json` loader

**Files:**
- Create: `src/aiperf/dataset/loader/inputs_json.py`
- Modify: `src/aiperf/dataset/loader/base_loader.py` (add `BaseRawPayloadLoader` class + `get_preferred_sampling_strategy` hook on `BaseLoader`)
- Modify: `src/aiperf/dataset/loader/models.py` (add `InputsJsonSession` model)
- Test: `tests/unit/dataset/loader/test_inputs_json_payload.py`

This task introduces the shared `BaseRawPayloadLoader` because Task 16 (`raw_payload`) needs it too. `get_preferred_sampling_strategy` is also added to `BaseLoader` since it does not exist on origin/main.

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest

  from aiperf.common.config.user_config import UserConfig
  from aiperf.common.enums import ConversationContextMode
  from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader


  @pytest.fixture
  def inputs_json_file(tmp_path: Path) -> Path:
      data = {
          "data": [
              {
                  "session_id": "session-001",
                  "payloads": [
                      {
                          "messages": [{"role": "user", "content": "Hello"}],
                          "model": "Qwen/Qwen3-0.6B",
                          "max_tokens": 32,
                      },
                      {
                          "messages": [
                              {"role": "user", "content": "Hello"},
                              {"role": "assistant", "content": "Hi"},
                              {"role": "user", "content": "How are you?"},
                          ],
                          "model": "Qwen/Qwen3-0.6B",
                          "max_tokens": 64,
                      },
                  ],
              },
              {
                  "session_id": "session-002",
                  "payloads": [
                      {
                          "messages": [{"role": "user", "content": "Bye"}],
                          "model": "Qwen/Qwen3-0.6B",
                      }
                  ],
              },
          ]
      }
      p = tmp_path / "inputs.json"
      p.write_bytes(orjson.dumps(data))
      return p


  class TestInputsJsonCanLoad:
      def test_can_load_with_data_key(self):
          assert InputsJsonPayloadLoader.can_load(
              data={"data": [{"session_id": "s", "payloads": [{}]}]}
          )

      def test_rejects_non_dict_data(self):
          assert not InputsJsonPayloadLoader.can_load(data={"data": "not a list"})

      def test_rejects_missing_payloads_key(self):
          assert not InputsJsonPayloadLoader.can_load(
              data={"data": [{"session_id": "s"}]}
          )

      def test_can_load_from_file(self, inputs_json_file: Path):
          assert InputsJsonPayloadLoader.can_load(filename=inputs_json_file)

      def test_rejects_empty_dict(self):
          assert not InputsJsonPayloadLoader.can_load(data={})


  class TestInputsJsonLoad:
      def test_load_preserves_session_ids_and_turns(
          self, inputs_json_file: Path, default_user_config: UserConfig
      ):
          loader = InputsJsonPayloadLoader(
              filename=inputs_json_file, user_config=default_user_config
          )
          data = loader.load_dataset()
          assert set(data.keys()) == {"session-001", "session-002"}
          assert len(data["session-001"][0].payloads) == 2
          assert len(data["session-002"][0].payloads) == 1

      def test_convert_produces_raw_payload_turns(
          self, inputs_json_file: Path, default_user_config: UserConfig
      ):
          loader = InputsJsonPayloadLoader(
              filename=inputs_json_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          assert len(conversations) == 2
          conv = next(c for c in conversations if c.session_id == "session-001")
          assert (
              conv.context_mode
              == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
          )
          assert len(conv.turns) == 2
          for turn in conv.turns:
              assert turn.raw_payload is not None
              assert "messages" in turn.raw_payload
  ```

  This test depends on a `default_user_config` fixture. If `tests/unit/dataset/loader/conftest.py` does not already provide one, add it:

  ```python
  # In tests/unit/dataset/loader/conftest.py (add or extend)
  import pytest

  from aiperf.common.config.user_config import UserConfig


  @pytest.fixture
  def default_user_config() -> UserConfig:
      return UserConfig.model_validate({
          "endpoint": {"model_names": ["test-model"]},
          "input": {"random_seed": 42},
      })
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/dataset/loader/test_inputs_json_payload.py -v`

  Expected: FAIL — `aiperf.dataset.loader.inputs_json` does not exist.

- [ ] **Step 3: Write minimal implementation**

  Add `BaseRawPayloadLoader` to `src/aiperf/dataset/loader/base_loader.py`. First, ensure `get_preferred_sampling_strategy` exists on `BaseLoader` (origin/main does not have it). Append to base_loader.py:

  ```python
  from aiperf.plugin.enums import DatasetSamplingStrategy

  # Inside BaseLoader, add classmethod:

      @classmethod
      def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
          """Loader's preferred sampling strategy when the user does not set one."""
          return DatasetSamplingStrategy.SHUFFLE
  ```

  Then add at the bottom of base_loader.py:

  ```python
  class BaseRawPayloadLoader(BaseFileLoader):
      """Base for loaders that produce verbatim raw_payload conversations.

      Provides shared defaults: MESSAGE_ARRAY_WITH_RESPONSES context mode and
      SEQUENTIAL sampling.
      """

      @classmethod
      def get_default_context_mode(cls) -> ConversationContextMode | None:
          return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

      @classmethod
      def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
          return DatasetSamplingStrategy.SEQUENTIAL
  ```

  Add `InputsJsonSession` and `RawPayload` models to `src/aiperf/dataset/loader/models.py`:

  ```python
  class RawPayload(AIPerfBaseModel):
      """A single raw API request payload for verbatim replay."""

      payload: dict[str, Any] = Field(description="Complete API request payload.")


  class InputsJsonSession(AIPerfBaseModel):
      """A session from the InputsFile format with pre-formatted payloads."""

      session_id: str = Field(description="Session ID of the conversation.")
      payloads: list[dict[str, Any]] = Field(
          min_length=1, description="Ordered list of per-turn payloads."
      )
  ```

  Add both to the `CustomDatasetT` union near the bottom of models.py.

  Then port the loader verbatim from dag4:

  ```bash
  git show ajc/dag4:src/aiperf/dataset/loader/inputs_json.py > src/aiperf/dataset/loader/inputs_json.py
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/dataset/loader/inputs_json.py src/aiperf/dataset/loader/base_loader.py src/aiperf/dataset/loader/models.py tests/unit/dataset/loader/test_inputs_json_payload.py tests/unit/dataset/loader/conftest.py
  git commit -s -m "$(cat <<'EOF'
  feat(loaders): add inputs_json loader for verbatim AIPerf inputs.json replay

  Loads AIPerf's InputsFile artifact format (top-level data list, each
  entry has session_id + payloads list) as raw_payload turns. Each
  payload is sent verbatim via the transport — no endpoint formatting.

  Adds BaseRawPayloadLoader (MESSAGE_ARRAY_WITH_RESPONSES + SEQUENTIAL
  defaults) for the inputs_json + raw_payload loaders to share, plus
  the get_preferred_sampling_strategy hook on BaseLoader.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 16: `raw_payload` loader

**Files:**
- Create: `src/aiperf/dataset/loader/raw_payload.py`
- Test: `tests/unit/dataset/loader/test_raw_payload.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest

  from aiperf.common.config.user_config import UserConfig
  from aiperf.common.enums import ConversationContextMode
  from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader


  def _write_jsonl(path: Path, records: list[dict]) -> None:
      with open(path, "wb") as f:
          for r in records:
              f.write(orjson.dumps(r))
              f.write(b"\n")


  @pytest.fixture
  def jsonl_file(tmp_path: Path) -> Path:
      p = tmp_path / "payloads.jsonl"
      _write_jsonl(
          p,
          [
              {
                  "messages": [{"role": "user", "content": "hi"}],
                  "model": "Qwen/Qwen3-0.6B",
                  "max_tokens": 16,
              },
              {
                  "messages": [{"role": "user", "content": "bye"}],
                  "model": "Qwen/Qwen3-0.6B",
                  "max_tokens": 16,
              },
          ],
      )
      return p


  @pytest.fixture
  def jsonl_dir(tmp_path: Path) -> Path:
      d = tmp_path / "convs"
      d.mkdir()
      _write_jsonl(
          d / "session_001.jsonl",
          [
              {"messages": [{"role": "user", "content": "t1"}]},
              {
                  "messages": [
                      {"role": "user", "content": "t1"},
                      {"role": "assistant", "content": "r"},
                      {"role": "user", "content": "t2"},
                  ]
              },
          ],
      )
      _write_jsonl(
          d / "session_002.jsonl",
          [{"messages": [{"role": "user", "content": "single"}]}],
      )
      return d


  class TestRawPayloadCanLoad:
      def test_messages_array_accepted(self):
          assert RawPayloadDatasetLoader.can_load(
              data={"messages": [{"role": "user", "content": "hi"}]}
          )

      def test_conversation_id_rejected(self):
          # Trajectory records carry conversation_id and are owned by another loader.
          assert not RawPayloadDatasetLoader.can_load(
              data={
                  "messages": [{"role": "user", "content": "hi"}],
                  "conversation_id": "abc",
              }
          )

      def test_data_list_rejected(self):
          # That's the InputsJson shape, not raw_payload.
          assert not RawPayloadDatasetLoader.can_load(
              data={"messages": [], "data": [{"session_id": "s", "payloads": []}]}
          )

      def test_directory_with_jsonl_accepted(self, jsonl_dir: Path):
          assert RawPayloadDatasetLoader.can_load(filename=jsonl_dir)


  class TestRawPayloadLoad:
      def test_single_file_one_session_per_line(
          self, jsonl_file: Path, default_user_config: UserConfig
      ):
          loader = RawPayloadDatasetLoader(
              filename=jsonl_file, user_config=default_user_config
          )
          data = loader.load_dataset()
          assert len(data) == 2  # two lines -> two sessions
          for payloads in data.values():
              assert len(payloads) == 1  # single-turn each

      def test_directory_one_file_per_session_multi_turn(
          self, jsonl_dir: Path, default_user_config: UserConfig
      ):
          loader = RawPayloadDatasetLoader(
              filename=jsonl_dir, user_config=default_user_config
          )
          data = loader.load_dataset()
          assert len(data) == 2  # two .jsonl files -> two sessions
          turn_counts = sorted(len(payloads) for payloads in data.values())
          assert turn_counts == [1, 2]

      def test_convert_produces_raw_payload_turns(
          self, jsonl_file: Path, default_user_config: UserConfig
      ):
          loader = RawPayloadDatasetLoader(
              filename=jsonl_file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          assert all(
              c.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
              for c in conversations
          )
          for c in conversations:
              for turn in c.turns:
                  assert turn.raw_payload is not None
                  assert "messages" in turn.raw_payload
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/dataset/loader/test_raw_payload.py -v`

  Expected: FAIL — `aiperf.dataset.loader.raw_payload` does not exist.

- [ ] **Step 3: Write minimal implementation**

  Port verbatim from dag4:

  ```bash
  git show ajc/dag4:src/aiperf/dataset/loader/raw_payload.py > src/aiperf/dataset/loader/raw_payload.py
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/dataset/loader/raw_payload.py tests/unit/dataset/loader/test_raw_payload.py
  git commit -s -m "$(cat <<'EOF'
  feat(loaders): add raw_payload loader for verbatim API replay

  JSONL loader with two modes:
  - Single file: each line is a one-turn conversation.
  - Directory: each .jsonl file is one multi-turn conversation, lines
    are ordered turns.

  Each line is sent verbatim via the transport — no endpoint formatting.
  can_load rejects records with conversation_id or a data list to avoid
  collisions with the trajectory and inputs_json formats.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 17: `mooncake_trace` payload mode

**Files:**
- Modify: `src/aiperf/dataset/loader/mooncake_trace.py`
- Modify: `src/aiperf/dataset/loader/models.py` (add `payload` field to `MooncakeTrace`)
- Test: `tests/unit/dataset/loader/test_mooncake_payload_mode.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest

  from aiperf.common.config.user_config import UserConfig
  from aiperf.dataset.loader.models import MooncakeTrace


  class TestMooncakeTracePayloadMode:
      def test_payload_field_accepted(self):
          t = MooncakeTrace(
              payload={"prompt": "Hello", "max_tokens": 50},
              timestamp=1000,
          )
          assert t.payload == {"prompt": "Hello", "max_tokens": 50}

      def test_payload_mutually_exclusive_with_input_length(self):
          with pytest.raises(ValueError):
              MooncakeTrace(
                  payload={"prompt": "Hello"},
                  input_length=10,
              )

      def test_payload_mutually_exclusive_with_messages(self):
          with pytest.raises(ValueError):
              MooncakeTrace(
                  payload={"prompt": "Hello"},
                  messages=[{"role": "user", "content": "x"}],
              )

      def test_empty_payload_rejected(self):
          with pytest.raises(ValueError):
              MooncakeTrace(payload={})


  class TestMooncakeTraceLoaderPayload:
      def test_payload_traces_produce_raw_payload_turns(
          self, tmp_path: Path, default_user_config: UserConfig
      ):
          from aiperf.dataset.loader.mooncake_trace import MooncakeDatasetLoader

          file = tmp_path / "trace.jsonl"
          with open(file, "wb") as f:
              for i in range(3):
                  f.write(
                      orjson.dumps(
                          {
                              "timestamp": 100 * i,
                              "payload": {
                                  "prompt": f"prompt-{i}",
                                  "max_tokens": 40,
                              },
                          }
                      )
                  )
                  f.write(b"\n")

          loader = MooncakeDatasetLoader(
              filename=file, user_config=default_user_config
          )
          conversations = loader.convert_to_conversations(loader.load_dataset())
          assert len(conversations) >= 1
          for conv in conversations:
              for turn in conv.turns:
                  assert turn.raw_payload is not None
                  assert turn.raw_payload["prompt"].startswith("prompt-")

      def test_mixed_payload_and_messages_in_session_rejected(
          self, tmp_path: Path, default_user_config: UserConfig
      ):
          from aiperf.dataset.loader.mooncake_trace import MooncakeDatasetLoader

          file = tmp_path / "mixed.jsonl"
          with open(file, "wb") as f:
              # Same session_id with both payload and messages — must raise.
              f.write(
                  orjson.dumps(
                      {
                          "session_id": "s1",
                          "payload": {"prompt": "p"},
                      }
                  )
              )
              f.write(b"\n")
              f.write(
                  orjson.dumps(
                      {
                          "session_id": "s1",
                          "messages": [{"role": "user", "content": "m"}],
                      }
                  )
              )
              f.write(b"\n")

          loader = MooncakeDatasetLoader(
              filename=file, user_config=default_user_config
          )
          with pytest.raises(ValueError, match="payload.*messages"):
              loader.convert_to_conversations(loader.load_dataset())
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/dataset/loader/test_mooncake_payload_mode.py -v`

  Expected: FAIL — `MooncakeTrace.payload` does not exist.

- [ ] **Step 3: Write minimal implementation**

  In `src/aiperf/dataset/loader/models.py`, in the `MooncakeTrace` class, add the `payload` field after `messages` and update the existing `model_validator(mode="after")` validators per the dag4 diff. The minimal patch (positive diff against main):

  ```python
      payload: dict[str, Any] | None = Field(
          None,
          description="Complete pre-built API request payload sent verbatim "
          "to the transport. Bypasses all endpoint formatting. Cannot be "
          "combined with other input modes.",
      )
  ```

  Update `validate_input` to include `payload` in the mutual-exclusivity count and add a hash_ids guard for payload mode:

  ```python
      @model_validator(mode="after")
      def validate_input(self) -> "MooncakeTrace":
          input_modes = [
              self.input_length is not None,
              self.text_input is not None,
              self.messages is not None,
              self.payload is not None,
          ]
          input_mode_count = sum(input_modes)
          if input_mode_count == 0:
              raise ValueError(
                  "Exactly one of 'input_length', 'text_input', 'messages', "
                  "or 'payload' must be provided"
              )
          if input_mode_count > 1:
              raise ValueError(
                  "'input_length', 'text_input', 'messages', and 'payload' "
                  "are mutually exclusive. Use only one of them."
              )
          if self.hash_ids is not None and self.input_length is None:
              raise ValueError(
                  "'hash_ids' is only allowed when 'input_length' is "
                  "provided, not when 'text_input', 'messages', or "
                  "'payload' are provided"
              )
          return self
  ```

  Add a separate `validate_payload` validator:

  ```python
      @model_validator(mode="after")
      def validate_payload(self) -> "MooncakeTrace":
          if self.payload is not None and not self.payload:
              raise ValueError("'payload' must be a non-empty dict")
          return self
  ```

  In `src/aiperf/dataset/loader/mooncake_trace.py`, update `_infer_context_mode`, `_get_text_input`, and `_build_turn` per the dag4 diff. The required edits (apply to main's existing methods):

  ```python
      def _infer_context_mode(
          self, traces: list[MooncakeTrace]
      ) -> ConversationContextMode | None:
          """Auto-detect MESSAGE_ARRAY_WITH_RESPONSES for pre-built content.

          Traces with ``messages`` or ``payload`` are self-contained and use
          MESSAGE_ARRAY_WITH_RESPONSES. Mixing different input modes (payload
          vs messages vs synthesized) in the same session is unsupported.
          """
          payload_count = sum(1 for t in traces if t.payload is not None)
          messages_count = sum(1 for t in traces if t.messages is not None)

          if payload_count and messages_count:
              raise ValueError(
                  "Mixed Mooncake sessions with both 'payload' and 'messages' "
                  "traces are unsupported. Use one mode per session."
              )

          self_contained = payload_count + messages_count
          if self_contained == len(traces):
              return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
          if self_contained > 0:
              raise ValueError(
                  "Mixed Mooncake sessions with both raw content "
                  "(messages/payload) and synthesized prompts are unsupported."
              )
          return None

      def _get_text_input(self, trace: MooncakeTrace) -> str | None:
          if trace.messages is not None or trace.payload is not None:
              return ""
          return trace.text_input

      def _build_turn(self, trace: MooncakeTrace, prompt: str) -> Turn:
          if trace.payload is not None:
              return Turn(
                  timestamp=trace.timestamp,
                  delay=trace.delay,
                  max_tokens=trace.output_length,
                  raw_payload=trace.payload,
              )
          # ... existing messages / synthesized branches unchanged ...
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — including the 6 new mooncake-payload tests, plus existing `test_mooncake_trace_messages.py`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/dataset/loader/mooncake_trace.py src/aiperf/dataset/loader/models.py tests/unit/dataset/loader/test_mooncake_payload_mode.py
  git commit -s -m "$(cat <<'EOF'
  feat(loaders): add payload mode to mooncake_trace

  Fourth input mode for MooncakeTrace alongside input_length / text_input
  / messages: a complete pre-built API request payload sent verbatim
  through the transport. Mutually exclusive with the other three.

  Updates context-mode inference and turn building to route payload
  traces through Turn.raw_payload. Mixed payload+messages sessions are
  rejected with a clear error.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 18: Plugin registry update

**Files:**
- Modify: `src/aiperf/plugin/plugins.yaml`
- Run: `make validate-plugin-schemas` and `make generate-all-plugin-files`
- Test: `tests/unit/plugin/test_dag5_loader_plugin_entries.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest

  from aiperf.plugin import plugins
  from aiperf.plugin.enums import CustomDatasetType, EndpointType, PluginType


  class TestRawEndpointRegistered:
      def test_raw_endpoint_class_resolves(self):
          cls = plugins.get_class(PluginType.ENDPOINT, EndpointType.RAW)
          assert cls is not None
          assert cls.__name__ == "RawEndpoint"


  class TestLoaderEntriesRegistered:
      @pytest.mark.parametrize(
          "loader_name,expected_cls",
          [
              ("raw_payload", "RawPayloadDatasetLoader"),
              ("inputs_json", "InputsJsonPayloadLoader"),
          ],
      )
      def test_loader_resolves(self, loader_name: str, expected_cls: str):
          cls = plugins.get_class(
              PluginType.CUSTOM_DATASET, CustomDatasetType(loader_name)
          )
          assert cls is not None
          assert cls.__name__ == expected_cls
  ```

  Note: the actual `plugins.get_class` and `PluginType` API names may differ; consult `src/aiperf/plugin/__init__.py` and adjust imports. The behavioural assertion (the loader class is reachable through the plugin registry) is the thing that matters.

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/unit/plugin/test_dag5_loader_plugin_entries.py -v`

  Expected: FAIL — `raw`, `raw_payload`, `inputs_json` are not in `plugins.yaml`.

- [ ] **Step 3: Write minimal implementation**

  Edit `src/aiperf/plugin/plugins.yaml`:

  Under `endpoints:` add:

  ```yaml
    raw:
      class: aiperf.endpoints.raw_endpoint:RawEndpoint
      description: |
        Fallback endpoint for non-standard APIs. Does not format payloads or
        append a URL path. Parses responses using auto-detection with optional
        JMESPath extraction via response_field. Prefer a regular endpoint type
        when the target API is supported.
      metadata:
        endpoint_path: null
        supports_streaming: true
        produces_tokens: true
        tokenizes_input: true
        supports_audio: true
        supports_images: true
        supports_videos: true
        metrics_title: LLM Metrics
  ```

  Under `custom_datasets:` (or wherever loaders are registered) add:

  ```yaml
    raw_payload:
      class: aiperf.dataset.loader.raw_payload:RawPayloadDatasetLoader
      description: |
        Raw payload JSONL loader for verbatim API replay. Each line is a
        complete API request body sent directly to the transport with zero
        formatting. Supports single file (one conversation per line) and
        directory mode (one JSONL file per multi-turn conversation).

    inputs_json:
      class: aiperf.dataset.loader.inputs_json:InputsJsonPayloadLoader
      description: |
        Inputs JSON payload loader for verbatim API replay. Loads AIPerf
        InputsFile format with pre-formatted payloads. Preserves multi-turn
        session structure and sends each payload directly to the transport
        without endpoint formatting.
  ```

  No new entry needed for mooncake `payload` mode — it's just a new field on the existing loader.

  Then run the validators:

  ```bash
  make validate-plugin-schemas
  make generate-all-plugin-files
  ```

  Inspect any auto-generated files that changed (enum extensions, overload stubs) and stage them with the commit.

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — the 3 new plugin tests plus everything else.

- [ ] **Step 5: Commit**

  ```bash
  git add src/aiperf/plugin/plugins.yaml src/aiperf/plugin/ tests/unit/plugin/test_dag5_loader_plugin_entries.py
  git commit -s -m "$(cat <<'EOF'
  feat(plugins): register raw endpoint, raw_payload + inputs_json loaders

  Plugin registry entries for the three new components. Mooncake payload
  mode reuses the existing mooncake_trace registration (the field is
  internal to the loader). Auto-generated plugin enums / overloads
  refreshed via make generate-all-plugin-files.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

### Task 19: Tutorials

**Files:**
- Create: `docs/tutorials/inputs-json-replay.md`
- Create: `docs/tutorials/raw-payload-replay.md`
- Modify: `README.md` (add both tutorials to the index)

- [ ] **Step 1: Write the failing test**

  Tutorial files have no test harness; this step is a presence + linkage check that doubles as a smoke test:

  ```bash
  test ! -f docs/tutorials/inputs-json-replay.md && echo "MISSING inputs-json"
  test ! -f docs/tutorials/raw-payload-replay.md && echo "MISSING raw-payload"
  grep -F "inputs-json-replay.md" README.md || echo "MISSING readme link inputs-json"
  grep -F "raw-payload-replay.md" README.md || echo "MISSING readme link raw-payload"
  ```

  Expected: all four "MISSING" lines printed before Step 3 lands.

- [ ] **Step 2: Run test to verify it fails**

  Same shell snippet as Step 1. Expected: four MISSING lines.

- [ ] **Step 3: Write minimal implementation**

  Port both tutorials verbatim from dag4 (they exist there and were authored against the same loaders we just ported):

  ```bash
  git show ajc/dag4:docs/tutorials/inputs-json-replay.md > docs/tutorials/inputs-json-replay.md
  git show ajc/dag4:docs/tutorials/raw-payload-replay.md > docs/tutorials/raw-payload-replay.md
  ```

  Update `README.md`'s Tutorials index. In the "Workloads and Data" section, add two entries (alphabetical insertion next to the other custom-dataset tutorials):

  ```markdown
  - [Inputs JSON Replay](docs/tutorials/inputs-json-replay.md) - Verbatim multi-turn replay of AIPerf inputs.json artifacts
  - [Raw Payload Replay](docs/tutorials/raw-payload-replay.md) - Verbatim JSONL payload replay (single file or directory)
  ```

- [ ] **Step 4: Run test to verify it passes**

  Re-run the Step 1 shell snippet. Expected: no output (all four checks pass).

  Then run: `uv run pytest tests/unit/ -n auto`

  Expected: PASS — final regression check.

- [ ] **Step 5: Commit**

  ```bash
  git add docs/tutorials/inputs-json-replay.md docs/tutorials/raw-payload-replay.md README.md
  git commit -s -m "$(cat <<'EOF'
  docs(tutorials): add inputs-json and raw-payload replay guides

  Two tutorials covering the new verbatim replay loaders shipped in this
  branch: AIPerf inputs.json artifact replay (multi-turn, named sessions)
  and JSONL raw_payload replay (single file or directory). Both linked
  from README.md's Workloads and Data tutorial index.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Spec Coverage

Mapping of in-scope items from the spec's §"In-Scope" to the task that lands them. Items marked with §-section pointers are the spec's implicit requirements (called out elsewhere in the spec body).

| In-Scope item (spec §"In-Scope" or referenced section) | Task |
|--------------------------------------------------------|------|
| `ConversationBranchMode` enum (§b. Data Models) | 2 |
| `TurnPrerequisite`, `PrerequisiteKind` (§b. Data Models) | 3 |
| `Turn.prerequisites` (§b. Data Models) | 4 |
| `TurnMetadata.has_forks` stamped at load time (§b. Data Models) | 4 |
| `Conversation.agent_depth` (§b. Data Models) | 4 |
| `Conversation.metadata()` projects `prerequisites` into `TurnMetadata` (§b. Data Models — dag4 fix) | 5 |
| `ConversationBranchInfo` with `dispatch_timing: Literal["pre","post"]` and field validator (§b. Data Models) | 6 |
| `BranchStats` with `joins_suppressed` counter (spec §"In-Scope" + §"Behavior Decisions") | 7 |
| `ToolCallResponseData` rename (`text` → `tool_call_text`) + add `content` (spec §e. Endpoint Layer + §"Endpoint" failure modes) | 8 |
| `Turn.raw_payload` field (required by raw / inputs_json / mooncake-payload loaders below) | 9 |
| `endpoints/response_mixin.py` + improved JMESPath compile-failure log (§e. Endpoint Layer) | 10 |
| `endpoints/base_endpoint.py.build_assistant_turn` + generic `build_messages` skeleton + `extract_payload_inputs` (§e. Endpoint Layer) | 11 |
| `endpoints/openai_responses.py`: `response.function_call_arguments.delta`/`.done` SSE handling (§e. Endpoint Layer) | 12 |
| `endpoints/openai_responses.py`: `_extract_response_content` walks `output[]` for `function_call` items (§e. Endpoint Layer) | 12 |
| `endpoints/openai_responses.py`: drop `instructions` synthetic system-message insertion (§e. Endpoint Layer) | 12 |
| `endpoints/openai_chat.py`: mixed `content + tool_calls` produces `ToolCallResponseData(tool_call_text=, content=)` (§e. Endpoint Layer) | 13 |
| `endpoints/openai_chat.py`: rename `text` → `tool_call_text` consumer-side (§e. Endpoint Layer) | 13 |
| `endpoints/openai_chat.py`: assistant-turn handling (§e. Endpoint Layer) | 13 |
| `endpoints/raw_endpoint.py`: byte-exact payload replay (§e. Endpoint Layer) | 14 |
| `dataset/loader/inputs_json.py` (§a. Loader Layer — sister DAG-adjacent loaders) | 15 |
| `dataset/loader/raw_payload.py` (§a. Loader Layer — sister DAG-adjacent loaders) | 16 |
| `dataset/loader/mooncake_trace.py` payload mode (§a. Loader Layer — sister DAG-adjacent loaders) | 17 |
| Plugin registry: `raw` endpoint, `inputs_json`, `raw_payload`, mooncake `payload` mode (§"Plugin Registry") | 18 |
| `docs/tutorials/inputs-json-replay.md` (§"Documentation Updates") | 19 |
| `docs/tutorials/raw-payload-replay.md` (§"Documentation Updates") | 19 |
| `README.md` tutorial index update (§"Documentation Updates") | 19 |

## Deferred to Plan 2

The following spec §"In-Scope" items are intentionally NOT covered in Plan 1 because they introduce or depend on the DAG runtime (BranchOrchestrator + topology walk + worker FORK refcount + credit/timing changes), which is the entire point of Plan 2:

- `BranchOrchestrator` and its 8-scenario test family (`fan_in`, `multi_gate`, `delayed`, `join`, `phase0`, `pre_session`, `adversarial`, `adversarial_full`)
- `ConversationSource.start_branch_child` / `start_pre_session_child`
- `dag_jsonl` loader and `dag_jsonl_models.py` (`DagSpawn`, `forks:` / `spawns:` shorthand, topology walk that stamps `agent_depth`, cycle / multi-parent / non-terminal-fork / dangling-prereq validation)
- `request_rate.py` orchestrator threading + `_issue_child_continuation_or_release`
- `phase/credit_counter.py` `is_final_credit` flip
- `phase/stop_conditions.py` `RequestCountStopCondition.applies_to_dag_children`
- `TimingManager._on_dataset_configuration_failed` + `_wait_for_dataset_or_failure` (verify on main first; if absent, port in Plan 2)
- Worker FORK pin refcount on `UserSession`; `UserSession.is_fork_parent` stamped at `create_and_store` time
- `RecordContext` / `RequestInfo` split, `inference_client._enrich_request_record`, `agent_depth` / `parent_correlation_id` on `Credit` / `TurnToSend` / `RequestInfo` / `RequestRecord`
- `_DagSettings.FAIL_FAST` (`AIPERF_DAG_FAIL_FAST`)
- DAG-aware completion gates in records-tracker and phase-runner; child HTTP requests count toward `requests_sent`
- `BranchStats` published via `CreditPhaseCompleteMessage` and exported to `profile_export_aiperf.json` (the model exists from Plan 1; the publish/export wiring is Plan 2)
- `--num-conversations` autodefault (`_count_dag_root_entries` + `_is_forking_dataset`); refusal to default `--request-count` for forking datasets
- `--no-fixed-schedule` (`InputConfig.disable_auto_fixed_schedule`)
- `docs/benchmark-modes/dag.md`
- Three-file sync of `dag_jsonl` mention in `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`
- Auto-regenerated `docs/cli-options.md` and `docs/environment-variables.md` (no Plan 1 CLI surface lands)
- BranchOrchestrator unit tests, DAG cross-component tests, integration end-to-end test, hard-cap test, multi-root test, prereq adversarial tests
