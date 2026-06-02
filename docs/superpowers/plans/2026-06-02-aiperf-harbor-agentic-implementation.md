<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf + Harbor Agentic Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new AIPerf services — `ProxyService` (FastAPI reverse proxy that source-tags requests and preserves TTFT/ITL across SSE) and `TaskRunnerService` (supervises `harbor run` subprocesses for SWE-bench / Terminal-bench) — wired into the existing records pipeline so a single benchmark run reports per-source request perf alongside per-task agent outcomes.

**Architecture:** Two new `BaseComponentService` peers registered in `src/aiperf/plugin/plugins.yaml`. Harbor coupling is loose: subprocess invocation + vendored Pydantic schema (`extra="forbid"`) for harbor's rollout JSON. Two new `MessageType` members (`PROXY_REQUEST_RECORD`, `TASK_RECORD`) flow into `RecordProcessor` with two new `@on_message` handlers. Spec: `docs/superpowers/specs/2026-05-29-aiperf-harbor-agentic-design.md`.

**Tech Stack:** Python 3.10+, async/await throughout, Pydantic v2, FastAPI + uvicorn (already in deps), httpx (already in deps), ZMQ message bus, pytest + pytest-asyncio + pytest-httpserver. Build/test via `uv` only (never pip).

**Linear:** AIP-920

---

## File Structure

### New files

```
src/aiperf/
  proxy/
    __init__.py
    service.py                        # ProxyService(BaseComponentService)
  task_runner/
    __init__.py
    service.py                        # TaskRunnerService(BaseComponentService)
    harbor_schema.py                  # vendored HarborRolloutResult Pydantic model
  common/messages/
    proxy_messages.py                 # ProxyRequestRecordMessage(BaseServiceMessage)
    task_messages.py                  # TaskRecordMessage(BaseServiceMessage)

tests/
  fixtures/harbor/v0_9_0/
    resolved.json                     # successful SWE-bench rollout
    failed_task.json                  # resolved=False, valid schema
    errored.json                      # rollout with mid-run error field set
    schema_drift.json                 # has extra_field, must fail validation
    README.md                         # how to refresh fixtures from real harbor
  harness/
    mock_harbor.py                    # PATH-shim for component_integration tests
  unit/proxy/
    __init__.py
    test_proxy_service.py
    test_proxy_records.py
  unit/task_runner/
    __init__.py
    test_harbor_schema.py
    test_task_runner_service.py
  unit/messages/
    __init__.py
    test_new_message_types.py
  unit/property/
    test_source_tag_invariant.py
  component_integration/
    test_proxy_in_process.py
    test_task_runner_with_mock_harbor.py
  integration/
    test_agentic_against_mock_server.py
    test_agentic_plus_replay_against_mock_server.py
    test_proxy_failure_aborts_run.py

docs/
  benchmark-modes/agentic-harbor.md   # user-facing guide
```

### Modified files

| Path | Change |
|---|---|
| `src/aiperf/common/enums/enums.py` | Add `PROXY_REQUEST_RECORD` and `TASK_RECORD` to `MessageType` |
| `src/aiperf/common/messages/__init__.py` | Export new message classes |
| `src/aiperf/records/record_processor_service.py` | Two new `@on_message` handlers + source-dimension aggregation + harbor-task aggregator |
| `src/aiperf/plugin/plugins.yaml` | Two new `service:` entries (`proxy`, `task_runner`) |
| `src/aiperf/config/flags/cli_config.py` | Eight new flags (see §5.6 of the spec) |
| `src/aiperf/config/flags/_section_fields.py` | Group the new flags into a CLI section |
| `docs/architecture.md` | Add new services to components and data-flow |
| `docs/dev/patterns.md` | New "Long-running task runner" pattern entry |
| `docs/index.yml` | Register `docs/benchmark-modes/agentic-harbor.md` |
| `AGENTS.md` / `CLAUDE.md` / `.github/copilot-instructions.md` / `.cursor/rules/python.mdc` | Four-file sync: add the new service category to the "Tips" / "Adding a New Service" notes |

### Reference files (read before starting)

- `src/aiperf/api/api_service.py` — FastAPI + `BaseComponentService` reference pattern
- `src/aiperf/common/messages/inference_messages.py` — message class pattern with `Field(description=...)`
- `src/aiperf/records/record_processor_service.py` — existing `@on_message` handlers to mirror
- `src/aiperf/dataset/loader/bailian_trace.py` and `dag_jsonl.py` — third-party trace loader precedents (for parsing patterns)
- `tests/harness/subprocess.py` — process-group helpers for clean subprocess teardown

---

## Pre-flight (do this once before Milestone 1)

- [ ] **Step P.1: Confirm uv environment is ready**

  Run: `uv sync && uv run pytest tests/unit/ -n auto -q | tail -20`
  Expected: tests pass (no new code yet; baseline confirmation).

- [ ] **Step P.2: Confirm pre-commit is wired**

  Run: `PATH="$HOME/.local/bin:$PATH" pre-commit --version`
  Expected: prints a version. If `pre-commit not found`, install with `uv tool install pre-commit`.

- [ ] **Step P.3: Capture a real harbor rollout fixture (one time)**

  Per spec §10 open question #1, we don't yet have the real harbor 0.9.x rollout JSON. Run harbor once against any SWE-bench instance pointed at any OpenAI-compatible endpoint, capture the resulting JSON, and place it at `tests/fixtures/harbor/v0_9_0/_captured_raw.json` (gitignored prefix `_`). Compare against the assumed schema in spec §5.3. If the real schema differs significantly, update spec §5.3 and the Pydantic model in Milestone 3 to match. Do NOT proceed with hypothetical fields.

  If real harbor cannot be run in your environment (no sandbox provider creds), document the assumption explicitly and proceed; the schema validation will catch divergence at first real use.

---

## Milestone 1: Add new MessageType enum values

Tiny but load-bearing — every downstream task imports from `MessageType`.

### Task 1.1: Extend MessageType enum

**Files:**
- Modify: `src/aiperf/common/enums/enums.py`

- [ ] **Step 1.1.1: Read the current MessageType class**

  Open `src/aiperf/common/enums/enums.py` and locate `class MessageType(CaseInsensitiveStrEnum):`. Note the alphabetical / grouped pattern of existing members.

- [ ] **Step 1.1.2: Add two members**

  Add (preserving the file's existing grouping convention):

  ```python
  PROXY_REQUEST_RECORD = "proxy_request_record"
  TASK_RECORD = "task_record"
  ```

- [ ] **Step 1.1.3: Verify enum imports cleanly**

  Run: `uv run python -c "from aiperf.common.enums import MessageType; print(MessageType.PROXY_REQUEST_RECORD, MessageType.TASK_RECORD)"`
  Expected: `MessageType.PROXY_REQUEST_RECORD MessageType.TASK_RECORD`

- [ ] **Step 1.1.4: Commit**

  ```bash
  git add src/aiperf/common/enums/enums.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(enums): add PROXY_REQUEST_RECORD and TASK_RECORD MessageType (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 2: Define the new message classes

### Task 2.1: ProxyRequestRecordMessage

**Files:**
- Create: `src/aiperf/common/messages/proxy_messages.py`
- Test: `tests/unit/messages/test_new_message_types.py`
- Modify: `src/aiperf/common/messages/__init__.py`

- [ ] **Step 2.1.1: Write the failing test**

  Create `tests/unit/messages/__init__.py` (empty file) and `tests/unit/messages/test_new_message_types.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import orjson
  from pytest import param
  import pytest

  from aiperf.common.enums import MessageType
  from aiperf.common.messages.proxy_messages import (
      ProxyRequestRecordMessage,
      ProxyRequestSource,
  )


  def _proxy_payload() -> dict:
      return {
          "service_id": "proxy-1",
          "source": "agent",
          "request_id": "req-001",
          "upstream_path": "chat/completions",
          "method": "POST",
          "status_code": 200,
          "send_time_ns": 1_000_000_000,
          "first_byte_time_ns": 1_050_000_000,
          "final_time_ns": 1_200_000_000,
          "inter_chunk_times_ns": [1_060_000_000, 1_080_000_000, 1_120_000_000],
          "input_tokens": 42,
          "output_tokens": 17,
          "partial": False,
          "error_class": None,
      }


  @pytest.mark.parametrize(
      "source",
      [
          param("agent", id="agent"),
          param("replay", id="replay"),
      ],
  )  # fmt: skip
  def test_proxy_request_record_message_round_trips(source: str) -> None:
      payload = _proxy_payload() | {"source": source}
      msg = ProxyRequestRecordMessage.model_validate(payload)
      assert msg.message_type == MessageType.PROXY_REQUEST_RECORD
      assert msg.source == ProxyRequestSource(source)
      blob = orjson.dumps(msg.model_dump(mode="json"))
      back = ProxyRequestRecordMessage.model_validate(orjson.loads(blob))
      assert back == msg


  def test_proxy_request_record_rejects_unknown_source() -> None:
      payload = _proxy_payload() | {"source": "worker"}  # vestigial, rejected
      with pytest.raises(ValueError):
          ProxyRequestRecordMessage.model_validate(payload)
  ```

- [ ] **Step 2.1.2: Run the test to verify it fails**

  Run: `uv run pytest tests/unit/messages/test_new_message_types.py -v`
  Expected: FAIL — `ModuleNotFoundError: No module named 'aiperf.common.messages.proxy_messages'`

- [ ] **Step 2.1.3: Implement `proxy_messages.py`**

  Create `src/aiperf/common/messages/proxy_messages.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  from pydantic import Field

  from aiperf.common.enums import CaseInsensitiveStrEnum, MessageType
  from aiperf.common.messages.service_messages import BaseServiceMessage
  from aiperf.common.types import MessageTypeT


  class ProxyRequestSource(CaseInsensitiveStrEnum):
      """Source bucket for a proxied request. Inferred from the URL path."""

      AGENT = "agent"
      REPLAY = "replay"


  class ProxyRequestRecordMessage(BaseServiceMessage):
      """Per-request perf record emitted by ProxyService."""

      message_type: MessageTypeT = MessageType.PROXY_REQUEST_RECORD

      source: ProxyRequestSource = Field(
          description="Which client class issued the request (agent or replay), "
          "inferred from the URL path the proxy received the request on."
      )
      request_id: str = Field(
          description="Proxy-generated identifier, also stamped on the upstream request "
          "as the X-AIPerf-Request-Id header."
      )
      upstream_path: str = Field(
          description="Path forwarded to the upstream endpoint, e.g. 'chat/completions'."
      )
      method: str = Field(description="HTTP method (always POST in v1).")
      status_code: int = Field(
          description="HTTP status returned to the client (mirrors upstream status)."
      )
      send_time_ns: int = Field(
          description="Monotonic ns timestamp when the proxy forwarded the request."
      )
      first_byte_time_ns: int | None = Field(
          default=None,
          description="Monotonic ns of the first byte received from upstream. "
          "None if the request errored before any response.",
      )
      final_time_ns: int | None = Field(
          default=None,
          description="Monotonic ns of the final byte received from upstream. "
          "None if the connection was cut mid-stream.",
      )
      inter_chunk_times_ns: list[int] = Field(
          default_factory=list,
          description="Monotonic ns timestamps for each SSE chunk after the first byte. "
          "Empty for non-streaming responses.",
      )
      input_tokens: int | None = Field(
          default=None,
          description="Input tokens parsed from the request payload (None if not parseable).",
      )
      output_tokens: int | None = Field(
          default=None,
          description="Output tokens parsed from the response usage block "
          "(None if streaming and usage not present).",
      )
      partial: bool = Field(
          default=False,
          description="True if the stream was cut before completion. Aggregates exclude "
          "partial records from latency percentiles but count them in error rate.",
      )
      error_class: str | None = Field(
          default=None,
          description="Categorical error tag (timeout / connection_reset / upstream_5xx) "
          "if the request did not complete cleanly.",
      )
  ```

- [ ] **Step 2.1.4: Export the new class**

  Add to `src/aiperf/common/messages/__init__.py` (find the existing exports and follow the pattern):

  ```python
  from aiperf.common.messages.proxy_messages import (
      ProxyRequestRecordMessage,
      ProxyRequestSource,
  )
  ```

  And add both names to `__all__` if present.

- [ ] **Step 2.1.5: Run tests, verify pass**

  Run: `uv run pytest tests/unit/messages/test_new_message_types.py -v`
  Expected: 3 passes (2 parametrized + 1 negative).

- [ ] **Step 2.1.6: Commit**

  ```bash
  git add src/aiperf/common/messages/proxy_messages.py \
          src/aiperf/common/messages/__init__.py \
          tests/unit/messages/__init__.py \
          tests/unit/messages/test_new_message_types.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(messages): add ProxyRequestRecordMessage (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 2.2: TaskRecordMessage

**Files:**
- Create: `src/aiperf/common/messages/task_messages.py`
- Modify: `tests/unit/messages/test_new_message_types.py`
- Modify: `src/aiperf/common/messages/__init__.py`

- [ ] **Step 2.2.1: Append failing tests for TaskRecordMessage**

  Append to `tests/unit/messages/test_new_message_types.py`:

  ```python
  from aiperf.common.messages.task_messages import (
      TaskErrorCategory,
      TaskRecordMessage,
  )


  def _task_payload() -> dict:
      return {
          "service_id": "task-runner-1",
          "instance_id": "swe-bench-django__django-13710",
          "benchmark": "swe-bench",
          "resolved": True,
          "wall_clock_seconds": 312.4,
          "total_input_tokens": 18_400,
          "total_output_tokens": 2_100,
          "step_count": 14,
          "error_category": None,
          "error_message": None,
          "stderr_tail": None,
          "harbor_version": "0.9.0",
      }


  def test_task_record_message_round_trips() -> None:
      msg = TaskRecordMessage.model_validate(_task_payload())
      assert msg.message_type == MessageType.TASK_RECORD
      assert msg.resolved is True
      blob = orjson.dumps(msg.model_dump(mode="json"))
      back = TaskRecordMessage.model_validate(orjson.loads(blob))
      assert back == msg


  @pytest.mark.parametrize(
      "category",
      [
          param("harbor_crash", id="harbor_crash"),
          param("schema_mismatch", id="schema_mismatch"),
          param("task_failure", id="task_failure"),
          param("shutdown_killed", id="shutdown_killed"),
      ],
  )  # fmt: skip
  def test_task_record_categories_round_trip(category: str) -> None:
      payload = _task_payload() | {
          "resolved": False,
          "error_category": category,
          "error_message": "demo",
      }
      msg = TaskRecordMessage.model_validate(payload)
      assert msg.error_category == TaskErrorCategory(category)
  ```

- [ ] **Step 2.2.2: Run, verify failure**

  Run: `uv run pytest tests/unit/messages/test_new_message_types.py -v`
  Expected: collection error / `ModuleNotFoundError: aiperf.common.messages.task_messages`.

- [ ] **Step 2.2.3: Implement `task_messages.py`**

  Create `src/aiperf/common/messages/task_messages.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  from pydantic import Field

  from aiperf.common.enums import CaseInsensitiveStrEnum, MessageType
  from aiperf.common.messages.service_messages import BaseServiceMessage
  from aiperf.common.types import MessageTypeT


  class TaskErrorCategory(CaseInsensitiveStrEnum):
      """Classification of why a task did not resolve cleanly.

      ``task_failure`` is signal — the agent legitimately failed the task.
      The other three indicate infrastructure or framework problems.
      """

      HARBOR_CRASH = "harbor_crash"
      SCHEMA_MISMATCH = "schema_mismatch"
      TASK_FAILURE = "task_failure"
      SHUTDOWN_KILLED = "shutdown_killed"


  class TaskRecordMessage(BaseServiceMessage):
      """Per-rollout record emitted by TaskRunnerService after a harbor subprocess exits."""

      message_type: MessageTypeT = MessageType.TASK_RECORD

      instance_id: str = Field(
          description="Benchmark instance identifier (e.g. a SWE-bench task ID)."
      )
      benchmark: str = Field(
          description="Benchmark name passed to harbor (e.g. 'swe-bench', 'terminal-bench')."
      )
      resolved: bool = Field(
          description="Whether the agent successfully completed the task."
      )
      wall_clock_seconds: float = Field(
          description="Total wall-clock duration of the rollout, as reported by harbor."
      )
      total_input_tokens: int = Field(
          description="Sum of input tokens across every model call in the rollout."
      )
      total_output_tokens: int = Field(
          description="Sum of output tokens across every model call in the rollout."
      )
      step_count: int = Field(
          description="Number of agent steps in the rollout."
      )
      error_category: TaskErrorCategory | None = Field(
          default=None,
          description="Set when resolved=False. None when resolved=True.",
      )
      error_message: str | None = Field(
          default=None,
          description="Short human-readable error string. None when resolved=True.",
      )
      stderr_tail: str | None = Field(
          default=None,
          description="Last 4 KB of harbor stderr if the subprocess crashed; otherwise None.",
      )
      harbor_version: str = Field(
          description="Harbor version that produced this rollout, used for fixture pinning."
      )
  ```

- [ ] **Step 2.2.4: Update `__init__.py` exports**

  Add to `src/aiperf/common/messages/__init__.py`:

  ```python
  from aiperf.common.messages.task_messages import (
      TaskErrorCategory,
      TaskRecordMessage,
  )
  ```

- [ ] **Step 2.2.5: Run, verify pass**

  Run: `uv run pytest tests/unit/messages/test_new_message_types.py -v`
  Expected: 8 passes total.

- [ ] **Step 2.2.6: Commit**

  ```bash
  git add src/aiperf/common/messages/task_messages.py \
          src/aiperf/common/messages/__init__.py \
          tests/unit/messages/test_new_message_types.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(messages): add TaskRecordMessage (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 3: Vendored harbor rollout schema

### Task 3.1: HarborRolloutResult model + fixtures

**Files:**
- Create: `src/aiperf/task_runner/__init__.py` (empty)
- Create: `src/aiperf/task_runner/harbor_schema.py`
- Create: `tests/fixtures/harbor/v0_9_0/resolved.json`
- Create: `tests/fixtures/harbor/v0_9_0/failed_task.json`
- Create: `tests/fixtures/harbor/v0_9_0/errored.json`
- Create: `tests/fixtures/harbor/v0_9_0/schema_drift.json`
- Create: `tests/fixtures/harbor/README.md`
- Create: `tests/unit/task_runner/__init__.py` (empty)
- Create: `tests/unit/task_runner/test_harbor_schema.py`

- [ ] **Step 3.1.1: Create the fixtures**

  Create `tests/fixtures/harbor/v0_9_0/resolved.json`:

  ```json
  {
    "instance_id": "django__django-13710",
    "benchmark": "swe-bench",
    "resolved": true,
    "wall_clock_seconds": 312.4,
    "total_input_tokens": 18400,
    "total_output_tokens": 2100,
    "step_count": 14,
    "error": null,
    "harbor_version": "0.9.0"
  }
  ```

  `tests/fixtures/harbor/v0_9_0/failed_task.json`:

  ```json
  {
    "instance_id": "django__django-99999",
    "benchmark": "swe-bench",
    "resolved": false,
    "wall_clock_seconds": 410.1,
    "total_input_tokens": 22000,
    "total_output_tokens": 3400,
    "step_count": 22,
    "error": null,
    "harbor_version": "0.9.0"
  }
  ```

  `tests/fixtures/harbor/v0_9_0/errored.json`:

  ```json
  {
    "instance_id": "django__django-broken",
    "benchmark": "swe-bench",
    "resolved": false,
    "wall_clock_seconds": 12.0,
    "total_input_tokens": 200,
    "total_output_tokens": 0,
    "step_count": 1,
    "error": "sandbox provisioning failed: e2b quota exceeded",
    "harbor_version": "0.9.0"
  }
  ```

  `tests/fixtures/harbor/v0_9_0/schema_drift.json` (intentionally has an extra field):

  ```json
  {
    "instance_id": "django__django-future",
    "benchmark": "swe-bench",
    "resolved": true,
    "wall_clock_seconds": 200.0,
    "total_input_tokens": 5000,
    "total_output_tokens": 500,
    "step_count": 8,
    "error": null,
    "harbor_version": "0.10.0",
    "future_field_we_dont_know_about": "surprise"
  }
  ```

  `tests/fixtures/harbor/README.md`:

  ```markdown
  # Harbor rollout fixtures

  One subdirectory per supported harbor minor version. Refresh when bumping the
  vendored schema:

  1. Install harbor: `uv tool install harbor`
  2. Pick one canonical task: e.g. `harbor run swe-bench --instance django__django-13710 --model <test-model> --base-url <test-endpoint> --output-dir /tmp/harbor-fixture`
  3. Copy the resulting per-rollout JSON into `vX_Y_Z/resolved.json`, redacting any sensitive bits.
  4. Hand-author the `failed_task.json`, `errored.json`, and `schema_drift.json`
     variants by mutating the captured baseline.
  ```

- [ ] **Step 3.1.2: Write failing schema tests**

  Create `tests/unit/task_runner/test_harbor_schema.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from pathlib import Path

  import orjson
  import pytest
  from pydantic import ValidationError
  from pytest import param

  from aiperf.task_runner.harbor_schema import HarborRolloutResult

  FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "harbor" / "v0_9_0"


  @pytest.mark.parametrize(
      "fixture, expected_resolved, expected_error_present",
      [
          param("resolved.json", True, False, id="resolved"),
          param("failed_task.json", False, False, id="failed_task"),
          param("errored.json", False, True, id="errored"),
      ],
  )  # fmt: skip
  def test_harbor_rollout_parses_v0_9_0(
      fixture: str, expected_resolved: bool, expected_error_present: bool
  ) -> None:
      payload = orjson.loads((FIXTURES / fixture).read_bytes())
      result = HarborRolloutResult.model_validate(payload)
      assert result.resolved is expected_resolved
      assert (result.error is not None) is expected_error_present
      assert result.harbor_version == "0.9.0"


  def test_harbor_rollout_rejects_unknown_field() -> None:
      payload = orjson.loads((FIXTURES / "schema_drift.json").read_bytes())
      with pytest.raises(ValidationError) as exc:
          HarborRolloutResult.model_validate(payload)
      assert "future_field_we_dont_know_about" in str(exc.value)
  ```

- [ ] **Step 3.1.3: Run, verify failure**

  Run: `uv run pytest tests/unit/task_runner/test_harbor_schema.py -v`
  Expected: collection error / `ModuleNotFoundError: aiperf.task_runner.harbor_schema`.

- [ ] **Step 3.1.4: Implement the schema**

  Create empty `src/aiperf/task_runner/__init__.py`.

  Create `src/aiperf/task_runner/harbor_schema.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  from pydantic import ConfigDict, Field

  from aiperf.common.models.base_models import AIPerfBaseModel


  class HarborRolloutResult(AIPerfBaseModel):
      """Vendored shape of harbor's per-rollout JSON output.

      ``extra="forbid"`` is intentional: any new field harbor adds in a future
      version fails parsing loudly, surfacing as a ``schema_mismatch`` task
      error. When that fires, capture a fresh fixture under
      ``tests/fixtures/harbor/v<MAJOR>_<MINOR>_<PATCH>/`` and update this model.
      """

      model_config = ConfigDict(extra="forbid")

      instance_id: str = Field(
          description="Benchmark instance identifier (e.g. a SWE-bench task ID)."
      )
      benchmark: str = Field(
          description="Benchmark name (e.g. 'swe-bench' or 'terminal-bench')."
      )
      resolved: bool = Field(
          description="Whether the agent successfully completed the task."
      )
      wall_clock_seconds: float = Field(
          description="Total wall-clock duration of the rollout."
      )
      total_input_tokens: int = Field(
          description="Sum of input tokens across all model calls."
      )
      total_output_tokens: int = Field(
          description="Sum of output tokens across all model calls."
      )
      step_count: int = Field(
          description="Number of agent steps in the rollout."
      )
      error: str | None = Field(
          default=None,
          description="Error message if the rollout failed mid-run; null on success.",
      )
      harbor_version: str = Field(
          description="Harbor version that produced this rollout."
      )
  ```

- [ ] **Step 3.1.5: Run, verify pass**

  Run: `uv run pytest tests/unit/task_runner/test_harbor_schema.py -v`
  Expected: 4 passes (3 parametrized + 1 schema-drift assertion).

- [ ] **Step 3.1.6: Commit**

  ```bash
  git add src/aiperf/task_runner/ tests/fixtures/harbor/ tests/unit/task_runner/
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(task_runner): vendor HarborRolloutResult schema with v0_9_0 fixtures (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 4: ProxyService skeleton (lifecycle + port bind)

### Task 4.1: Service shell that boots and stops cleanly

**Files:**
- Create: `src/aiperf/proxy/__init__.py` (empty)
- Create: `src/aiperf/proxy/service.py`
- Create: `tests/unit/proxy/__init__.py` (empty)
- Create: `tests/unit/proxy/test_proxy_service.py`

Use `src/aiperf/api/api_service.py` as the lifecycle reference. The proxy doesn't need routers; it has hardcoded routes.

- [ ] **Step 4.1.1: Write the failing lifecycle test**

  Create `tests/unit/proxy/test_proxy_service.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import asyncio
  import socket

  import httpx
  import pytest
  from pytest_httpserver import HTTPServer

  from aiperf.proxy.service import ProxyService


  def _free_port() -> int:
      with socket.socket() as s:
          s.bind(("127.0.0.1", 0))
          return s.getsockname()[1]


  @pytest.mark.asyncio
  async def test_proxy_service_binds_and_stops(make_run, mock_message_bus) -> None:
      """ProxyService binds its port on start and releases it on stop."""
      port = _free_port()
      svc = ProxyService(
          run=make_run(proxy_port=port, upstream_url="http://127.0.0.1:1"),
          service_id="proxy-test",
      )
      await svc.initialize()
      await svc.start()
      try:
          async with httpx.AsyncClient() as client:
              with pytest.raises(httpx.HTTPError):
                  # not yet routed, but the port should accept the connection
                  await client.post(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
      finally:
          await svc.stop()
      # Port released after stop
      with socket.socket() as s:
          s.bind(("127.0.0.1", port))
  ```

  This test depends on two fixtures (`make_run`, `mock_message_bus`) that don't exist yet. Add them to `tests/unit/proxy/conftest.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import pytest

  # Reuse the existing fake-comms harness; check tests/harness/fake_communication.py
  # for the exact factory the project uses. Mirror what tests/unit/api/ does.

  @pytest.fixture
  def mock_message_bus():
      """Stand-in message bus that records published messages without ZMQ."""
      from tests.harness.fake_communication import FakeCommunication
      return FakeCommunication()


  @pytest.fixture
  def make_run():
      """Factory that produces a BenchmarkRun with proxy/task_runner fields populated."""
      from aiperf.config.resolution.plan import BenchmarkRun
      def _make(**overrides):
          # Minimal BenchmarkRun with the new CLI fields defaulted.
          # See tests/unit/api/conftest.py for the exact construction style;
          # the project uses a helper like build_minimal_run() — reuse it.
          from tests.unit.api.conftest import build_minimal_run
          return build_minimal_run(**overrides)
      return _make
  ```

  > **Reference check:** if `build_minimal_run` doesn't exist under that exact name, look at how `tests/unit/api/` constructs a `BenchmarkRun` for its tests and mirror that. Do not invent a new construction pattern.

- [ ] **Step 4.1.2: Run, verify failure**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py -v`
  Expected: collection error / `ModuleNotFoundError: aiperf.proxy.service`.

- [ ] **Step 4.1.3: Implement the skeleton**

  Create `src/aiperf/proxy/service.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import asyncio
  from contextlib import suppress
  from typing import TYPE_CHECKING

  import uvicorn
  from fastapi import FastAPI

  from aiperf.common.base_component_service import BaseComponentService
  from aiperf.common.hooks import on_start, on_stop

  if TYPE_CHECKING:
      from aiperf.config.resolution.plan import BenchmarkRun


  class ProxyService(BaseComponentService):
      """FastAPI reverse proxy between LLM clients and the upstream endpoint.

      v1 surface: passthrough routing with source tagging by URL path
      (``/v1/agent/*``, ``/v1/replay/*``). Subsequent milestones add streaming
      capture, record emission, and error handling.
      """

      def __init__(
          self,
          run: BenchmarkRun,
          service_id: str | None = None,
          **kwargs,
      ) -> None:
          super().__init__(run=run, service_id=service_id, **kwargs)
          self._proxy_port = run.cfg.proxy.port
          self._upstream_url = run.cfg.proxy.upstream_url
          self.app = FastAPI(title="AIPerf Proxy")
          self._server: uvicorn.Server | None = None
          self._server_task: asyncio.Task | None = None

      @on_start
      async def _start_server(self) -> None:
          config = uvicorn.Config(
              self.app,
              host="127.0.0.1",
              port=self._proxy_port,
              log_level="warning",
              access_log=False,
          )
          self._server = uvicorn.Server(config)
          self._server_task = asyncio.create_task(
              self._server.serve(), name="proxy-uvicorn"
          )
          # Wait until uvicorn flips `started` so callers know we're bound.
          while not self._server.started:
              await asyncio.sleep(0.01)

      @on_stop
      async def _stop_server(self) -> None:
          if self._server is not None:
              self._server.should_exit = True
          if self._server_task is not None:
              with suppress(asyncio.CancelledError):
                  await self._server_task
          self._server = None
          self._server_task = None
  ```

  The `run.cfg.proxy.port` / `run.cfg.proxy.upstream_url` accessors don't exist yet; they're introduced in Milestone 14 (CLI flags). For now this milestone's test will need a CLI surface shim. **Pause here** — proceed only if `BenchmarkRun` already exposes a `cfg.proxy` namespace (it does not). Two options:

  1. Land Milestone 14 first (CLI flags + config plumbing), then resume Milestone 4.
  2. Temporarily accept `proxy_port` / `upstream_url` as direct kwargs into `ProxyService.__init__` and rewire to `run.cfg.proxy` in Milestone 14.

  Pick option 1 — landing config first keeps every subsequent test using the final API surface and avoids a midflight rewire. Mark this task **blocked on Milestone 14**, skip ahead, and return here.

  > **Plan note:** The order in this document is logical, not strictly executable. The actual dependency graph is: M1, M2, M3, M14, M15, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M16, M17, M18, M19. A subagent executor should respect the dependency order, not the document order. The reason this plan is presented logically (service-by-service) rather than dependency-ordered is reader comprehension — but **execute in dependency order**.

- [ ] **Step 4.1.4: When unblocked, run the failing test again**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py::test_proxy_service_binds_and_stops -v`
  Expected: PASS.

- [ ] **Step 4.1.5: Commit**

  ```bash
  git add src/aiperf/proxy/ tests/unit/proxy/__init__.py \
          tests/unit/proxy/conftest.py tests/unit/proxy/test_proxy_service.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(proxy): ProxyService skeleton with lifecycle-managed uvicorn server (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 5: ProxyService passthrough routing (non-streaming)

### Task 5.1: Two source-tagged routes that forward JSON requests

**Files:**
- Modify: `src/aiperf/proxy/service.py`
- Modify: `tests/unit/proxy/test_proxy_service.py`

- [ ] **Step 5.1.1: Add failing routing test**

  Append to `tests/unit/proxy/test_proxy_service.py`:

  ```python
  @pytest.mark.asyncio
  @pytest.mark.parametrize("source_path", ["agent", "replay"])
  async def test_proxy_forwards_json_request(
      make_run, mock_message_bus, httpserver: HTTPServer, source_path: str
  ) -> None:
      httpserver.expect_request("/chat/completions").respond_with_json(
          {"id": "abc", "choices": [{"message": {"content": "hi"}}],
           "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
      )
      port = _free_port()
      svc = ProxyService(
          run=make_run(proxy_port=port, upstream_url=httpserver.url_for("")),
          service_id="proxy-test",
      )
      await svc.initialize()
      await svc.start()
      try:
          async with httpx.AsyncClient() as client:
              resp = await client.post(
                  f"http://127.0.0.1:{port}/v1/{source_path}/chat/completions",
                  json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  timeout=5.0,
              )
          assert resp.status_code == 200
          assert resp.json()["choices"][0]["message"]["content"] == "hi"
      finally:
          await svc.stop()
  ```

- [ ] **Step 5.1.2: Run, verify failure (404 or connection refused)**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py::test_proxy_forwards_json_request -v`
  Expected: FAIL.

- [ ] **Step 5.1.3: Implement the two routes**

  Add to `src/aiperf/proxy/service.py` after the imports:

  ```python
  import httpx
  from fastapi import Request
  from fastapi.responses import Response
  ```

  And inside `__init__` after `self.app = FastAPI(...)`:

  ```python
  self._client = httpx.AsyncClient(base_url=self._upstream_url, timeout=None)
  self._register_routes()
  ```

  Add a method `_register_routes`:

  ```python
  def _register_routes(self) -> None:
      from aiperf.common.messages.proxy_messages import ProxyRequestSource

      async def _passthrough(
          request: Request, path: str, source: ProxyRequestSource
      ) -> Response:
          body = await request.body()
          upstream = await self._client.request(
              method=request.method,
              url=f"/{path}",
              content=body,
              headers={
                  k: v
                  for k, v in request.headers.items()
                  if k.lower() not in {"host", "content-length"}
              },
          )
          return Response(
              content=upstream.content,
              status_code=upstream.status_code,
              headers={
                  k: v
                  for k, v in upstream.headers.items()
                  if k.lower() not in {"transfer-encoding", "connection"}
              },
          )

      @self.app.post("/v1/agent/{path:path}")
      async def _agent(request: Request, path: str) -> Response:
          return await _passthrough(request, path, ProxyRequestSource.AGENT)

      @self.app.post("/v1/replay/{path:path}")
      async def _replay(request: Request, path: str) -> Response:
          return await _passthrough(request, path, ProxyRequestSource.REPLAY)
  ```

  Extend `_stop_server` to close the httpx client:

  ```python
  await self._client.aclose()
  ```

- [ ] **Step 5.1.4: Run, verify pass**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py -v`
  Expected: all pass.

- [ ] **Step 5.1.5: Commit**

  ```bash
  git add src/aiperf/proxy/service.py tests/unit/proxy/test_proxy_service.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(proxy): JSON passthrough routes for /v1/agent and /v1/replay (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 6: ProxyService faithful SSE streaming + timing capture

### Task 6.1: Stream chunks as they arrive; record TTFT/ITL

**Files:**
- Modify: `src/aiperf/proxy/service.py`
- Modify: `tests/unit/proxy/test_proxy_service.py`

- [ ] **Step 6.1.1: Add failing streaming test**

  Append to `tests/unit/proxy/test_proxy_service.py`:

  ```python
  @pytest.mark.asyncio
  async def test_proxy_streams_sse_with_timing(
      make_run, mock_message_bus, httpserver: HTTPServer
  ) -> None:
      # pytest-httpserver doesn't natively do SSE; use a chunked text response.
      sse_body = (
          b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
          b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
          b'data: [DONE]\n\n'
      )
      httpserver.expect_request("/chat/completions").respond_with_data(
          sse_body, content_type="text/event-stream"
      )
      port = _free_port()
      svc = ProxyService(
          run=make_run(proxy_port=port, upstream_url=httpserver.url_for("")),
          service_id="proxy-test",
      )
      await svc.initialize()
      await svc.start()
      try:
          async with httpx.AsyncClient() as client:
              async with client.stream(
                  "POST",
                  f"http://127.0.0.1:{port}/v1/agent/chat/completions",
                  json={"stream": True},
                  timeout=5.0,
              ) as resp:
                  chunks = [c async for c in resp.aiter_bytes()]
          merged = b"".join(chunks)
          assert b'"content":"hi"' in merged
          assert b'[DONE]' in merged
      finally:
          await svc.stop()
  ```

- [ ] **Step 6.1.2: Run, verify fail (response not chunked)**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py::test_proxy_streams_sse_with_timing -v`
  Expected: FAIL — single buffered chunk rather than streamed.

- [ ] **Step 6.1.3: Convert passthrough to streaming**

  In `_register_routes`, replace `_passthrough` with:

  ```python
  from fastapi.responses import StreamingResponse
  import time
  import uuid

  async def _passthrough(
      request: Request, path: str, source: ProxyRequestSource
  ) -> StreamingResponse:
      body = await request.body()
      request_id = str(uuid.uuid4())
      send_time_ns = time.monotonic_ns()
      inter_chunk_times_ns: list[int] = []
      first_byte_time_ns: int | None = None
      headers = {
          k: v
          for k, v in request.headers.items()
          if k.lower() not in {"host", "content-length"}
      }
      headers["X-AIPerf-Request-Id"] = request_id

      async def _stream():
          nonlocal first_byte_time_ns
          async with self._client.stream(
              method=request.method,
              url=f"/{path}",
              content=body,
              headers=headers,
          ) as upstream:
              # Capture upstream status / headers via closure for outer caller.
              _stream.status_code = upstream.status_code
              _stream.upstream_headers = dict(upstream.headers)
              async for chunk in upstream.aiter_raw():
                  now = time.monotonic_ns()
                  if first_byte_time_ns is None:
                      first_byte_time_ns = now
                  else:
                      inter_chunk_times_ns.append(now)
                  yield chunk
          final_time_ns = time.monotonic_ns()
          # Stash for record emission (Milestone 7).
          self._last_record_fields = {
              "source": source,
              "request_id": request_id,
              "upstream_path": path,
              "method": request.method,
              "status_code": _stream.status_code,
              "send_time_ns": send_time_ns,
              "first_byte_time_ns": first_byte_time_ns,
              "final_time_ns": final_time_ns,
              "inter_chunk_times_ns": inter_chunk_times_ns,
          }

      gen = _stream()
      # Prime to learn the upstream status before returning headers.
      first_chunk = await gen.__anext__()

      async def _wrapped():
          yield first_chunk
          async for c in gen:
              yield c

      return StreamingResponse(
          _wrapped(),
          status_code=_stream.status_code,
          headers={
              k: v
              for k, v in _stream.upstream_headers.items()
              if k.lower() not in {"transfer-encoding", "connection", "content-length"}
          },
      )
  ```

  > **Note for the implementer:** the priming-via-`__anext__` dance is required because StreamingResponse takes its `status_code` synchronously but the upstream status isn't known until the first byte arrives. If the upstream returns no body at all (e.g., HEAD or 204), this raises `StopAsyncIteration` — handle that by treating the response as empty. Cover in error-handling milestone.

- [ ] **Step 6.1.4: Run, verify pass**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py -v`
  Expected: all pass including new streaming test.

- [ ] **Step 6.1.5: Commit**

  ```bash
  git add src/aiperf/proxy/service.py tests/unit/proxy/test_proxy_service.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(proxy): faithful SSE streaming with per-chunk timing capture (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 7: Emit ProxyRequestRecordMessage on every request

### Task 7.1: Wire timing capture into message publication

**Files:**
- Modify: `src/aiperf/proxy/service.py`
- Create: `tests/unit/proxy/test_proxy_records.py`

- [ ] **Step 7.1.1: Write failing emission test**

  Create `tests/unit/proxy/test_proxy_records.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import socket

  import httpx
  import pytest
  from pytest_httpserver import HTTPServer
  from pytest import param

  from aiperf.common.enums import MessageType
  from aiperf.common.messages.proxy_messages import ProxyRequestSource
  from aiperf.proxy.service import ProxyService


  def _free_port() -> int:
      with socket.socket() as s:
          s.bind(("127.0.0.1", 0))
          return s.getsockname()[1]


  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "source_path, expected_source",
      [
          param("agent", ProxyRequestSource.AGENT, id="agent"),
          param("replay", ProxyRequestSource.REPLAY, id="replay"),
      ],
  )  # fmt: skip
  async def test_proxy_emits_record_with_source_tag(
      make_run, mock_message_bus, httpserver: HTTPServer,
      source_path: str, expected_source: ProxyRequestSource,
  ) -> None:
      httpserver.expect_request("/chat/completions").respond_with_json(
          {"choices": [{"message": {"content": "x"}}],
           "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
      )
      port = _free_port()
      svc = ProxyService(
          run=make_run(proxy_port=port, upstream_url=httpserver.url_for("")),
          service_id="proxy-test",
      )
      svc.publish = mock_message_bus.record_publish  # type: ignore[assignment]
      await svc.initialize()
      await svc.start()
      try:
          async with httpx.AsyncClient() as client:
              await client.post(
                  f"http://127.0.0.1:{port}/v1/{source_path}/chat/completions",
                  json={"messages": []},
              )
      finally:
          await svc.stop()
      published = mock_message_bus.messages_of_type(MessageType.PROXY_REQUEST_RECORD)
      assert len(published) == 1
      msg = published[0]
      assert msg.source == expected_source
      assert msg.status_code == 200
      assert msg.input_tokens == 7
      assert msg.output_tokens == 3
      assert msg.first_byte_time_ns is not None
      assert msg.final_time_ns is not None
      assert msg.send_time_ns < msg.first_byte_time_ns <= msg.final_time_ns
  ```

  Extend `tests/unit/proxy/conftest.py`'s `FakeCommunication` with `record_publish` and `messages_of_type` helpers if it doesn't already have them (mirror the existing pattern in `tests/harness/fake_communication.py`).

- [ ] **Step 7.1.2: Run, verify failure**

  Run: `uv run pytest tests/unit/proxy/test_proxy_records.py -v`
  Expected: FAIL — no messages published yet.

- [ ] **Step 7.1.3: Emit the record**

  In `src/aiperf/proxy/service.py`, after the streaming generator finishes:

  ```python
  async def _publish_record_after_stream(fields: dict, body_for_tokens: bytes) -> None:
      from aiperf.common.messages.proxy_messages import ProxyRequestRecordMessage
      input_tokens, output_tokens = _parse_token_counts(body_for_tokens, fields)
      msg = ProxyRequestRecordMessage(
          service_id=self.service_id,
          input_tokens=input_tokens,
          output_tokens=output_tokens,
          **fields,
      )
      await self.publish(msg)
  ```

  Replace the bare assignment `self._last_record_fields = ...` with an `await self._publish_record_after_stream(fields, captured_body)` call. To do that, accumulate the response body bytes alongside streaming for token parsing:

  ```python
  body_chunks: list[bytes] = []
  async for chunk in upstream.aiter_raw():
      body_chunks.append(chunk)
      now = time.monotonic_ns()
      if first_byte_time_ns is None:
          first_byte_time_ns = now
      else:
          inter_chunk_times_ns.append(now)
      yield chunk
  await _publish_record_after_stream(
      fields={...},
      body_for_tokens=b"".join(body_chunks),
  )
  ```

  Add a module-private `_parse_token_counts(body: bytes, fields: dict) -> tuple[int | None, int | None]`:

  ```python
  def _parse_token_counts(body: bytes, fields: dict) -> tuple[int | None, int | None]:
      """Best-effort extraction of usage tokens.

      Non-streaming JSON: parse top-level 'usage' block.
      SSE: look for a usage block in the final non-[DONE] data line.
      Returns (None, None) when usage isn't surfaced (e.g., streaming
      without inline usage; addressed by spec §10 open question #3).
      """
      import orjson
      try:
          if body.lstrip().startswith(b"{"):
              data = orjson.loads(body)
              usage = data.get("usage") or {}
              return usage.get("prompt_tokens"), usage.get("completion_tokens")
          # SSE: walk backwards for the last non-[DONE] data block
          for line in reversed(body.splitlines()):
              line = line.strip()
              if line.startswith(b"data:") and not line.endswith(b"[DONE]"):
                  payload = line[len(b"data:"):].strip()
                  if payload.startswith(b"{"):
                      data = orjson.loads(payload)
                      usage = data.get("usage") or {}
                      if usage:
                          return usage.get("prompt_tokens"), usage.get("completion_tokens")
      except Exception:
          return None, None
      return None, None
  ```

- [ ] **Step 7.1.4: Run, verify pass**

  Run: `uv run pytest tests/unit/proxy/ -v`
  Expected: all pass.

- [ ] **Step 7.1.5: Commit**

  ```bash
  git add src/aiperf/proxy/service.py tests/unit/proxy/test_proxy_records.py \
          tests/unit/proxy/conftest.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(proxy): emit PROXY_REQUEST_RECORD with source/timing/tokens (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 8: ProxyService error handling

### Task 8.1: Upstream 5xx, timeouts, client disconnect

**Files:**
- Modify: `src/aiperf/proxy/service.py`
- Modify: `tests/unit/proxy/test_proxy_service.py`

- [ ] **Step 8.1.1: Write failing error-path tests**

  Append to `tests/unit/proxy/test_proxy_service.py`:

  ```python
  @pytest.mark.asyncio
  async def test_proxy_records_upstream_5xx(
      make_run, mock_message_bus, httpserver: HTTPServer
  ) -> None:
      httpserver.expect_request("/x").respond_with_data("boom", status=503)
      port = _free_port()
      svc = ProxyService(
          run=make_run(proxy_port=port, upstream_url=httpserver.url_for("")),
          service_id="proxy-test",
      )
      svc.publish = mock_message_bus.record_publish  # type: ignore
      await svc.initialize()
      await svc.start()
      try:
          async with httpx.AsyncClient() as client:
              resp = await client.post(f"http://127.0.0.1:{port}/v1/agent/x", json={})
          assert resp.status_code == 503
      finally:
          await svc.stop()
      msg = mock_message_bus.messages_of_type(MessageType.PROXY_REQUEST_RECORD)[0]
      assert msg.status_code == 503
      assert msg.error_class == "upstream_5xx"


  @pytest.mark.asyncio
  async def test_proxy_records_upstream_timeout(
      make_run, mock_message_bus, httpserver: HTTPServer, monkeypatch
  ) -> None:
      # Use a port nothing listens on to provoke a connection error.
      port = _free_port()
      dead_port = _free_port()
      svc = ProxyService(
          run=make_run(proxy_port=port, upstream_url=f"http://127.0.0.1:{dead_port}"),
          service_id="proxy-test",
      )
      svc.publish = mock_message_bus.record_publish  # type: ignore
      await svc.initialize()
      await svc.start()
      try:
          async with httpx.AsyncClient() as client:
              resp = await client.post(f"http://127.0.0.1:{port}/v1/agent/x", json={})
          assert resp.status_code in (502, 504)  # proxy returns a synthetic gateway error
      finally:
          await svc.stop()
      msg = mock_message_bus.messages_of_type(MessageType.PROXY_REQUEST_RECORD)[0]
      assert msg.error_class in {"connection_error", "timeout"}
      assert msg.partial is False  # nothing was ever received
  ```

- [ ] **Step 8.1.2: Run, verify failure**

  Run: `uv run pytest tests/unit/proxy/test_proxy_service.py -k "upstream_5xx or upstream_timeout" -v`
  Expected: FAIL.

- [ ] **Step 8.1.3: Implement error-class tagging**

  In `_passthrough`, wrap the upstream stream in a try/except, returning a synthetic `Response(status_code=502, content=b"upstream error")` on connection error / timeout and capturing `error_class` accordingly. Cases:

  - `httpx.ConnectError` / `httpx.ConnectTimeout` → `error_class="connection_error"`, status 502
  - `httpx.ReadTimeout` → `error_class="timeout"`, status 504
  - upstream `5xx` status passes through unchanged with `error_class="upstream_5xx"`
  - upstream `4xx` passes through with `error_class="upstream_4xx"`
  - upstream `2xx` clean stream → `error_class=None`
  - `asyncio.CancelledError` from client disconnect mid-stream → `partial=True`, `error_class="client_disconnect"`, still publish the record with whatever timing was captured

  Each branch publishes a `ProxyRequestRecordMessage` before returning so failures show up in metrics.

- [ ] **Step 8.1.4: Run, verify pass**

  Run: `uv run pytest tests/unit/proxy/ -v`
  Expected: all pass.

- [ ] **Step 8.1.5: Commit**

  ```bash
  git add src/aiperf/proxy/service.py tests/unit/proxy/test_proxy_service.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(proxy): error-class tagging for 5xx/timeout/disconnect paths (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 9: RecordProcessor handles PROXY_REQUEST_RECORD

### Task 9.1: New @on_message handler with source dimension

**Files:**
- Modify: `src/aiperf/records/record_processor_service.py`
- Modify: existing record-processor tests under `tests/unit/records/` (path TBD — locate first)

- [ ] **Step 9.1.1: Locate the existing RecordProcessor tests**

  Run: `find tests -path '*/records/*' -name 'test_*.py' | head`
  Expected: at least one file. Read its `@on_message` test for `MetricRecordsMessage` to mirror the pattern.

- [ ] **Step 9.1.2: Write failing handler test**

  Add a new test file `tests/unit/records/test_proxy_record_handler.py` mirroring the patterns you found:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  import pytest
  from aiperf.common.enums import MessageType
  from aiperf.common.messages.proxy_messages import (
      ProxyRequestRecordMessage,
      ProxyRequestSource,
  )

  # Test must construct a RecordProcessor instance with the harness factory
  # already used in other tests in this directory (find it via the existing
  # tests in step 9.1.1).

  @pytest.mark.asyncio
  async def test_record_processor_aggregates_by_source(make_record_processor) -> None:
      proc = make_record_processor()
      msgs = [
          ProxyRequestRecordMessage(
              service_id="proxy", source=ProxyRequestSource.AGENT,
              request_id=f"r-{i}", upstream_path="chat/completions",
              method="POST", status_code=200,
              send_time_ns=i*10, first_byte_time_ns=i*10+5, final_time_ns=i*10+9,
              inter_chunk_times_ns=[], input_tokens=10, output_tokens=2,
          )
          for i in range(3)
      ] + [
          ProxyRequestRecordMessage(
              service_id="proxy", source=ProxyRequestSource.REPLAY,
              request_id=f"r-{i}", upstream_path="chat/completions",
              method="POST", status_code=200,
              send_time_ns=i*10, first_byte_time_ns=i*10+3, final_time_ns=i*10+8,
              inter_chunk_times_ns=[], input_tokens=5, output_tokens=1,
          )
          for i in range(2)
      ]
      for m in msgs:
          await proc.on_proxy_request_record(m)
      summary = proc.summarize_proxy_metrics()
      assert summary.count_total == 5
      assert summary.by_source[ProxyRequestSource.AGENT].count == 3
      assert summary.by_source[ProxyRequestSource.REPLAY].count == 2
      # Invariant: per-source counts must sum to total
      assert sum(s.count for s in summary.by_source.values()) == summary.count_total
  ```

- [ ] **Step 9.1.3: Run, verify failure**

  Run: `uv run pytest tests/unit/records/test_proxy_record_handler.py -v`
  Expected: FAIL (handler / summarize methods do not exist).

- [ ] **Step 9.1.4: Implement the handler**

  In `src/aiperf/records/record_processor_service.py`:

  - Add `@on_message(MessageType.PROXY_REQUEST_RECORD)` decorated method `on_proxy_request_record(self, msg: ProxyRequestRecordMessage)` that updates internal counters keyed by `(source, metric_dimension)`.
  - Reuse the existing TTFT/ITL/throughput aggregator if its types match the new message shape; otherwise wrap the new shape into the existing per-request type at the handler boundary.
  - Add a `summarize_proxy_metrics()` method returning a typed dataclass with `count_total: int` and `by_source: dict[ProxyRequestSource, _PerSourceMetrics]` where `_PerSourceMetrics` has `count`, `ttft_p50`, `ttft_p95`, `itl_p50`, `throughput_rps`, `error_rate`.

  > **Be careful here.** The existing RecordProcessor likely has assumptions about a single per-request stream. Read its body end-to-end before deciding whether to extend the existing aggregator type or add a parallel one. Whichever choice you make, exclude `partial=True` records from latency percentiles but include them in error-rate.

- [ ] **Step 9.1.5: Run, verify pass**

  Run: `uv run pytest tests/unit/records/test_proxy_record_handler.py tests/unit/records/ -v`
  Expected: new handler test passes; existing record-processor tests still pass.

- [ ] **Step 9.1.6: Commit**

  ```bash
  git add src/aiperf/records/record_processor_service.py \
          tests/unit/records/test_proxy_record_handler.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(records): handle PROXY_REQUEST_RECORD with source-dimension aggregates (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 10: TaskRunnerService subprocess supervision

### Task 10.1: Dispatcher + per-instance worker coroutines

**Files:**
- Create: `src/aiperf/task_runner/service.py`
- Create: `tests/harness/mock_harbor.py`
- Create: `tests/unit/task_runner/test_task_runner_service.py`

- [ ] **Step 10.1.1: Write the mock harbor script**

  Create `tests/harness/mock_harbor.py`:

  ```python
  #!/usr/bin/env python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  """Stand-in for the harbor CLI used by component-integration and unit tests.

  Supports two subcommands:
    list <benchmark>          → prints instance ids, one per line, to stdout
    run <benchmark> --instance <id> --model <m> --base-url <u> --output-dir <d>
                              → writes <d>/rollout.json with a deterministic shape

  Behavior is controlled by env vars:
    MOCK_HARBOR_INSTANCES        comma-separated instance ids for 'list'
    MOCK_HARBOR_RESOLVED_FRAC    float in [0,1] — fraction of runs marked resolved
    MOCK_HARBOR_SLEEP_SECONDS    sleep before writing rollout (simulates duration)
    MOCK_HARBOR_EXIT_CODE        non-zero to simulate crash; default 0
    MOCK_HARBOR_MISSING_OUTPUT   if set, exit 0 without writing rollout.json
    MOCK_HARBOR_BROKEN_JSON      if set, write malformed JSON to rollout.json
  """

  from __future__ import annotations

  import argparse
  import json
  import os
  import sys
  import time
  from pathlib import Path


  def _do_list(benchmark: str) -> int:
      instances = os.environ.get("MOCK_HARBOR_INSTANCES", "instance-0,instance-1").split(",")
      for i in instances:
          print(i)
      return 0


  def _do_run(args: argparse.Namespace) -> int:
      time.sleep(float(os.environ.get("MOCK_HARBOR_SLEEP_SECONDS", "0")))
      exit_code = int(os.environ.get("MOCK_HARBOR_EXIT_CODE", "0"))
      if exit_code != 0:
          print("mock_harbor: simulated crash", file=sys.stderr)
          return exit_code
      out = Path(args.output_dir)
      out.mkdir(parents=True, exist_ok=True)
      if os.environ.get("MOCK_HARBOR_MISSING_OUTPUT"):
          return 0
      target = out / "rollout.json"
      if os.environ.get("MOCK_HARBOR_BROKEN_JSON"):
          target.write_text("{not json")
          return 0
      resolved_frac = float(os.environ.get("MOCK_HARBOR_RESOLVED_FRAC", "1.0"))
      # Deterministic resolved/unresolved per instance id hash
      resolved = (abs(hash(args.instance)) % 100) / 100 < resolved_frac
      payload = {
          "instance_id": args.instance,
          "benchmark": args.benchmark,
          "resolved": resolved,
          "wall_clock_seconds": float(os.environ.get("MOCK_HARBOR_SLEEP_SECONDS", "0")),
          "total_input_tokens": 100,
          "total_output_tokens": 20,
          "step_count": 3,
          "error": None,
          "harbor_version": "0.9.0-mock",
      }
      target.write_text(json.dumps(payload))
      return 0


  def main() -> int:
      parser = argparse.ArgumentParser(prog="mock_harbor")
      sub = parser.add_subparsers(dest="cmd", required=True)
      sub_list = sub.add_parser("list")
      sub_list.add_argument("benchmark")
      sub_run = sub.add_parser("run")
      sub_run.add_argument("benchmark")
      sub_run.add_argument("--instance", required=True)
      sub_run.add_argument("--model", required=True)
      sub_run.add_argument("--base-url", required=True)
      sub_run.add_argument("--output-dir", required=True)
      args, _ = parser.parse_known_args()
      if args.cmd == "list":
          return _do_list(args.benchmark)
      if args.cmd == "run":
          return _do_run(args)
      return 2


  if __name__ == "__main__":
      raise SystemExit(main())
  ```

  Make it executable: `chmod +x tests/harness/mock_harbor.py`.

- [ ] **Step 10.1.2: Write failing task-runner test**

  Create `tests/unit/task_runner/test_task_runner_service.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import os
  from pathlib import Path

  import pytest

  from aiperf.common.enums import MessageType
  from aiperf.task_runner.service import TaskRunnerService


  @pytest.fixture
  def harbor_on_path(monkeypatch, tmp_path: Path):
      """Make `harbor` resolve to our mock_harbor.py for the duration of the test."""
      src = Path(__file__).resolve().parents[2] / "harness" / "mock_harbor.py"
      shim = tmp_path / "bin"
      shim.mkdir()
      target = shim / "harbor"
      target.symlink_to(src)
      monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
      return shim


  @pytest.mark.asyncio
  async def test_task_runner_dispatches_concurrent_rollouts(
      make_run, mock_message_bus, harbor_on_path, tmp_path: Path, monkeypatch
  ) -> None:
      monkeypatch.setenv("MOCK_HARBOR_INSTANCES", "i-0,i-1,i-2,i-3")
      monkeypatch.setenv("MOCK_HARBOR_RESOLVED_FRAC", "1.0")
      svc = TaskRunnerService(
          run=make_run(
              task_concurrency=2,
              harbor_benchmark="swe-bench",
              harbor_output_dir=str(tmp_path),
              model_name="m",
              proxy_port=9999,
          ),
          service_id="tr",
      )
      svc.publish = mock_message_bus.record_publish  # type: ignore
      await svc.initialize()
      await svc.start()
      await svc.wait_until_idle()
      await svc.stop()
      records = mock_message_bus.messages_of_type(MessageType.TASK_RECORD)
      assert {r.instance_id for r in records} == {"i-0", "i-1", "i-2", "i-3"}
      assert all(r.resolved for r in records)
      assert all(r.error_category is None for r in records)
  ```

- [ ] **Step 10.1.3: Run, verify failure**

  Run: `uv run pytest tests/unit/task_runner/test_task_runner_service.py -v`
  Expected: FAIL — `aiperf.task_runner.service` not found.

- [ ] **Step 10.1.4: Implement the service**

  Create `src/aiperf/task_runner/service.py`:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0

  from __future__ import annotations

  import asyncio
  import shutil
  from pathlib import Path
  from typing import TYPE_CHECKING

  import orjson
  from pydantic import ValidationError

  from aiperf.common.base_component_service import BaseComponentService
  from aiperf.common.enums import MessageType
  from aiperf.common.hooks import background_task, on_start, on_stop
  from aiperf.common.messages.task_messages import (
      TaskErrorCategory,
      TaskRecordMessage,
  )
  from aiperf.task_runner.harbor_schema import HarborRolloutResult

  if TYPE_CHECKING:
      from aiperf.config.resolution.plan import BenchmarkRun


  class TaskRunnerService(BaseComponentService):
      """Supervises ``harbor run`` subprocesses (one per benchmark instance)."""

      def __init__(self, run: BenchmarkRun, service_id: str | None = None, **kw) -> None:
          super().__init__(run=run, service_id=service_id, **kw)
          cfg = run.cfg.task_runner
          self._concurrency = cfg.concurrency
          self._benchmark = cfg.harbor_benchmark
          self._output_dir = Path(cfg.harbor_output_dir)
          self._extra_args = cfg.harbor_extra_args or ""
          self._fail_fast = cfg.fail_fast_ratio
          self._shutdown_grace = cfg.shutdown_grace_seconds
          self._model = run.cfg.model_name
          self._proxy_port = run.cfg.proxy.port
          self._instances: list[str] = []
          self._stop_dispatch = asyncio.Event()
          self._idle = asyncio.Event()
          self._idle.set()
          self._sem = asyncio.Semaphore(self._concurrency)
          self._inflight: set[asyncio.Task] = set()
          self._results: list[bool] = []  # True = ok (resolved or task_failure)

      @on_start
      async def _start_dispatch(self) -> None:
          self._instances = await self._resolve_instances()
          self._idle.clear()
          asyncio.create_task(self._dispatch_loop(), name="task-runner-dispatch")

      @on_stop
      async def _stop_dispatch(self) -> None:
          self._stop_dispatch.set()
          # Grace period for in-flight subprocesses
          try:
              await asyncio.wait_for(self._idle.wait(), timeout=self._shutdown_grace)
          except asyncio.TimeoutError:
              for t in list(self._inflight):
                  t.cancel()
              await asyncio.gather(*self._inflight, return_exceptions=True)

      async def wait_until_idle(self) -> None:
          await self._idle.wait()

      async def _resolve_instances(self) -> list[str]:
          proc = await asyncio.create_subprocess_exec(
              "harbor", "list", self._benchmark,
              stdout=asyncio.subprocess.PIPE,
              stderr=asyncio.subprocess.PIPE,
          )
          stdout, stderr = await proc.communicate()
          if proc.returncode != 0:
              raise RuntimeError(
                  f"harbor list {self._benchmark} failed: {stderr.decode()}"
              )
          return [line.strip() for line in stdout.decode().splitlines() if line.strip()]

      async def _dispatch_loop(self) -> None:
          try:
              for instance_id in self._instances:
                  if self._stop_dispatch.is_set() or self._exceeds_fail_fast():
                      break
                  await self._sem.acquire()
                  task = asyncio.create_task(
                      self._run_one(instance_id), name=f"task-{instance_id}"
                  )
                  self._inflight.add(task)
                  task.add_done_callback(self._on_subtask_done)
              if self._inflight:
                  await asyncio.gather(*self._inflight, return_exceptions=True)
          finally:
              self._idle.set()

      def _on_subtask_done(self, task: asyncio.Task) -> None:
          self._inflight.discard(task)
          self._sem.release()

      def _exceeds_fail_fast(self) -> bool:
          if not self._results:
              return False
          infra_failures = sum(1 for ok in self._results if not ok)
          return infra_failures / len(self._results) >= self._fail_fast

      async def _run_one(self, instance_id: str) -> None:
          instance_out = self._output_dir / instance_id
          instance_out.mkdir(parents=True, exist_ok=True)
          cmd = [
              "harbor", "run", self._benchmark,
              "--instance", instance_id,
              "--model", self._model,
              "--base-url", f"http://127.0.0.1:{self._proxy_port}/v1/agent",
              "--output-dir", str(instance_out),
          ]
          if self._extra_args:
              cmd.extend(self._extra_args.split())
          proc = await asyncio.create_subprocess_exec(
              *cmd,
              stdout=asyncio.subprocess.DEVNULL,
              stderr=asyncio.subprocess.PIPE,
          )
          try:
              _, stderr = await proc.communicate()
          except asyncio.CancelledError:
              proc.kill()
              await proc.wait()
              await self._publish_failure(
                  instance_id, TaskErrorCategory.SHUTDOWN_KILLED,
                  "subprocess killed by shutdown", b""
              )
              raise
          if proc.returncode != 0:
              await self._publish_failure(
                  instance_id, TaskErrorCategory.HARBOR_CRASH,
                  f"harbor exit code {proc.returncode}",
                  stderr[-4096:],
              )
              return
          rollout_path = instance_out / "rollout.json"
          if not rollout_path.exists():
              await self._publish_failure(
                  instance_id, TaskErrorCategory.HARBOR_CRASH,
                  "rollout.json not written", stderr[-4096:],
              )
              return
          try:
              raw = orjson.loads(rollout_path.read_bytes())
              result = HarborRolloutResult.model_validate(raw)
          except (orjson.JSONDecodeError, ValidationError) as exc:
              await self._publish_failure(
                  instance_id, TaskErrorCategory.SCHEMA_MISMATCH,
                  str(exc), stderr[-4096:],
              )
              return
          await self._publish_success(result)

      async def _publish_success(self, result: HarborRolloutResult) -> None:
          self._results.append(True)
          await self.publish(TaskRecordMessage(
              service_id=self.service_id,
              instance_id=result.instance_id,
              benchmark=result.benchmark,
              resolved=result.resolved,
              wall_clock_seconds=result.wall_clock_seconds,
              total_input_tokens=result.total_input_tokens,
              total_output_tokens=result.total_output_tokens,
              step_count=result.step_count,
              error_category=None if result.resolved else TaskErrorCategory.TASK_FAILURE,
              error_message=result.error,
              stderr_tail=None,
              harbor_version=result.harbor_version,
          ))

      async def _publish_failure(
          self, instance_id: str, category: TaskErrorCategory,
          message: str, stderr_tail: bytes,
      ) -> None:
          self._results.append(category is TaskErrorCategory.TASK_FAILURE)
          await self.publish(TaskRecordMessage(
              service_id=self.service_id,
              instance_id=instance_id,
              benchmark=self._benchmark,
              resolved=False,
              wall_clock_seconds=0.0,
              total_input_tokens=0,
              total_output_tokens=0,
              step_count=0,
              error_category=category,
              error_message=message,
              stderr_tail=stderr_tail.decode(errors="replace") if stderr_tail else None,
              harbor_version="unknown",
          ))
  ```

- [ ] **Step 10.1.5: Run, verify pass**

  Run: `uv run pytest tests/unit/task_runner/test_task_runner_service.py -v`
  Expected: PASS.

- [ ] **Step 10.1.6: Commit**

  ```bash
  git add src/aiperf/task_runner/service.py tests/harness/mock_harbor.py \
          tests/unit/task_runner/test_task_runner_service.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(task_runner): TaskRunnerService dispatcher with semaphore + rollout parsing (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 10.2: Cover the four failure categories

**Files:**
- Modify: `tests/unit/task_runner/test_task_runner_service.py`

- [ ] **Step 10.2.1: Add parametrized failure tests**

  Append four parametrized tests covering each `MOCK_HARBOR_*` env-var path: `harbor_crash` (non-zero exit), `harbor_crash` (missing output), `schema_mismatch` (broken JSON), and `task_failure` (mock writes `resolved=false`). Assert the expected `error_category` on the emitted `TASK_RECORD`.

- [ ] **Step 10.2.2: Add a fail-fast test**

  Set `MOCK_HARBOR_EXIT_CODE=1` and `fail_fast_ratio=0.4` with 10 instances; assert fewer than 10 `TASK_RECORD` messages emitted because dispatch stopped early.

- [ ] **Step 10.2.3: Run, verify pass**

  Run: `uv run pytest tests/unit/task_runner/ -v`
  Expected: all pass.

- [ ] **Step 10.2.4: Commit**

  ```bash
  git add tests/unit/task_runner/test_task_runner_service.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(task_runner): cover all error categories + fail-fast (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 11: RecordProcessor handles TASK_RECORD

### Task 11.1: New @on_message handler with per-benchmark aggregate

**Files:**
- Modify: `src/aiperf/records/record_processor_service.py`
- Create: `tests/unit/records/test_task_record_handler.py`

- [ ] **Step 11.1.1: Write failing aggregate test**

  Create `tests/unit/records/test_task_record_handler.py` asserting:
  - `count_total` == number of `TASK_RECORD` messages
  - `resolved_count` == count where `resolved=True`
  - `failure_breakdown` returns counts keyed by `TaskErrorCategory`
  - `wall_clock_p50` and `wall_clock_p95` computed only over resolved + task_failure (not infra failures)
  - `tokens_per_task_mean` computed over all completed rollouts

- [ ] **Step 11.1.2: Run, verify failure**

  Run: `uv run pytest tests/unit/records/test_task_record_handler.py -v`
  Expected: FAIL.

- [ ] **Step 11.1.3: Implement**

  Add `@on_message(MessageType.TASK_RECORD)` to `RecordProcessor`. The handler appends to an internal list (memory cost bounded by benchmark size, fine). Add a `summarize_task_metrics()` method returning a typed dataclass with the fields above.

- [ ] **Step 11.1.4: Run, verify pass**

  Run: `uv run pytest tests/unit/records/ -v`
  Expected: all pass.

- [ ] **Step 11.1.5: Commit**

  ```bash
  git add src/aiperf/records/record_processor_service.py \
          tests/unit/records/test_task_record_handler.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(records): handle TASK_RECORD with per-benchmark task aggregate (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 12: Surface the new aggregates in profile_export.json + console

### Task 12.1: Records manager exports `proxy_request_metrics` + `harbor_task_metrics`

**Files:**
- Modify: `src/aiperf/records/records_manager.py` (or wherever final report assembly happens — locate first)
- Modify: relevant exporter file under `src/aiperf/exporters/`

- [ ] **Step 12.1.1: Locate the final-report assembly point**

  Run: `grep -rn "profile_export" src/aiperf/ --include='*.py' -l | head`
  Read the export builder. Identify where to add new top-level keys.

- [ ] **Step 12.1.2: Add failing exporter test**

  Add a test that constructs a `RecordProcessor` with seeded proxy + task records, drives the export, and asserts:
  - `profile_export["proxy_request_metrics"]["all"]["ttft_p95"]` is set
  - `profile_export["proxy_request_metrics"]["by_source"]["agent"]["count"]` matches input
  - `profile_export["harbor_task_metrics"]["resolved_count"]` matches input

- [ ] **Step 12.1.3: Implement**

  Wire `summarize_proxy_metrics()` and `summarize_task_metrics()` into the export payload. Mirror existing console-table rendering for the new aggregates. Use `MetricConsoleGroup.NONE` if you do NOT want them in the main console table; the spec wants them visible, so render as the two new tables from §6.2 of the spec.

- [ ] **Step 12.1.4: Run, verify pass**

  Run: `uv run pytest tests/unit/records/ tests/unit/exporters/ -v`
  Expected: all pass.

- [ ] **Step 12.1.5: Commit**

  ```bash
  git add src/aiperf/records/records_manager.py src/aiperf/exporters/ tests/unit/exporters/
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(exporters): proxy_request_metrics + harbor_task_metrics in profile_export (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 13: Plugin registration (plugins.yaml)

### Task 13.1: Register both services

**Files:**
- Modify: `src/aiperf/plugin/plugins.yaml`

- [ ] **Step 13.1.1: Add entries**

  Under `service:`, add:

  ```yaml
    proxy:
      class: aiperf.proxy.service:ProxyService
      description: |
        Reverse proxy that sits between LLM clients (harbor agents, replay workers)
        and the configured inference endpoint, emitting per-request perf records
        tagged by source path (/v1/agent, /v1/replay).
      metadata:
        required: false
        auto_start: false

    task_runner:
      class: aiperf.task_runner.service:TaskRunnerService
      description: |
        Supervises long-running agent task rollouts via the harbor subprocess.
        Spawns up to --task-concurrency harbor processes in parallel, parses
        each rollout's JSON output on subprocess exit, and emits per-task records.
      metadata:
        required: false
        auto_start: false
  ```

- [ ] **Step 13.1.2: Validate the registry**

  Run: `uv run make validate-plugin-schemas`
  Expected: pass.

  Run: `uv run python -c "from aiperf.plugin import plugins; from aiperf.plugin.enums import PluginType, ServiceType; print(plugins.get_class(PluginType.SERVICE, 'proxy')); print(plugins.get_class(PluginType.SERVICE, 'task_runner'))"`
  Expected: prints both class references.

- [ ] **Step 13.1.3: Commit**

  ```bash
  git add src/aiperf/plugin/plugins.yaml
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(plugin): register proxy and task_runner services (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 14: CLI flags + config plumbing

This milestone unblocks Milestone 4. Execute it third (after M1–M3) per the dependency note.

### Task 14.1: Define new flags and config sections

**Files:**
- Modify: `src/aiperf/config/flags/cli_config.py`
- Modify: `src/aiperf/config/flags/_section_fields.py`
- Modify: `src/aiperf/config/resolution/plan.py` (or wherever `BenchmarkRun.cfg` lives)
- Modify: `docs/cli-options.md` (auto-regenerated)

- [ ] **Step 14.1.1: Read the existing flag-definition pattern**

  Read `src/aiperf/config/flags/cli_config.py` end-to-end to understand the existing add-a-flag pattern. The CLI is flat (CLAUDE.md: "CLIConfig is flat; never add a nested config class").

  > Important: "flat CLI" doesn't mean the *runtime* config has to be flat. CLI flags map to a flat namespace, then `BenchmarkRun.cfg` may group them into sub-objects internally (e.g., `cfg.proxy.port`). The spec assumes this grouping; verify the existing convention by reading how other subsystems do it (e.g., `cfg.runtime.api_host` from `api_service.py`).

- [ ] **Step 14.1.2: Add the eight flags**

  Following the existing pattern, add to `cli_config.py`:

  - `task_runner: Literal["harbor"] | None` (None disables; only `"harbor"` accepted in v1)
  - `harbor_benchmark: str | None`
  - `task_concurrency: int = 1`
  - `harbor_output_dir: Path | None` (default resolved from `artifact_dir`)
  - `harbor_extra_args: str | None`
  - `proxy_port: int = 18888`
  - `task_runner_fail_fast: float = 0.5`
  - `task_runner_shutdown_grace: int = 60`

  Each gets a `Field(description=...)` per CLAUDE.md. Group them into a new CLI section called `"Agentic"` in `_section_fields.py`.

- [ ] **Step 14.1.3: Add runtime config grouping**

  In whatever module owns `BenchmarkRun.cfg` (likely under `src/aiperf/config/`), add nested config classes `ProxyConfig` (port, upstream_url derived from `--url`) and `TaskRunnerConfig` (concurrency, harbor_benchmark, harbor_output_dir, harbor_extra_args, fail_fast_ratio, shutdown_grace_seconds). Populate them in the resolution layer that turns flat CLI flags into the runtime `cfg`.

- [ ] **Step 14.1.4: Add resolver validation**

  In the config resolution code, validate:
  - `--task-runner harbor` requires `--harbor-benchmark`.
  - `--task-concurrency` must be ≥ 1.
  - `--proxy-port` must be 1–65535.
  - `--task-runner-fail-fast` must be in [0, 1].

  Raise `ConfigValidationError` (or whatever exception type the project uses) with a clear message on each.

- [ ] **Step 14.1.5: Regenerate CLI docs**

  Run: `make generate-cli-docs`
  Expected: `docs/cli-options.md` updated with the new flags.

- [ ] **Step 14.1.6: Run unit tests**

  Run: `uv run pytest tests/unit/config/ -n auto -v`
  Expected: existing tests pass; if you added new config validation, add tests for it under `tests/unit/config/`.

- [ ] **Step 14.1.7: Commit**

  ```bash
  git add src/aiperf/config/ docs/cli-options.md tests/unit/config/
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(cli): add 8 flags for harbor task-runner + proxy (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 15: SystemController activates the new services

### Task 15.1: Start services when `--task-runner harbor` is set

**Files:**
- Modify: wherever SystemController instantiates services (probably `src/aiperf/controller/` — locate)
- Modify: lifecycle ordering to enforce: Proxy `RUNNING` before TaskRunner spawns; TaskRunner `RUNNING` after Workers ready

- [ ] **Step 15.1.1: Locate SystemController service-startup code**

  Run: `grep -rn "auto_start\|ServiceType\." src/aiperf/controller/ --include='*.py' | head -20`
  Read the existing startup orchestration.

- [ ] **Step 15.1.2: Wire in the new services**

  When `cfg.task_runner.enabled` is True (i.e., `--task-runner harbor` was set), include `ProxyService` and `TaskRunnerService` in the service-start sequence. Enforce the ordering from spec §6.3:
  1. ProxyService → RUNNING (await port bind via the lifecycle event the service emits)
  2. TaskRunnerService → INITIALIZED (instance list resolved)
  3. Workers, DatasetManager, TimingManager → RUNNING
  4. TaskRunnerService → RUNNING (dispatch begins)

- [ ] **Step 15.1.3: Add fatal-on-proxy-crash policy**

  If `ProxyService` transitions to `FAILED`, SystemController emits a fatal lifecycle event. Reuse the existing required-service-failure path; do not invent new policy. Add a test under `tests/integration/test_proxy_failure_aborts_run.py` (covered in Milestone 17).

- [ ] **Step 15.1.4: Run integration test from Milestone 17 to confirm**

  (Will fail until M17 is written; that's expected.) For now, verify unit tests pass:
  Run: `uv run pytest tests/unit/controller/ -n auto -v`
  Expected: pass.

- [ ] **Step 15.1.5: Commit**

  ```bash
  git add src/aiperf/controller/
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  feat(controller): start ProxyService + TaskRunnerService when --task-runner harbor (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 16: Component-integration tests

### Task 16.1: Proxy + RecordProcessor in one process

**Files:**
- Create: `tests/component_integration/test_proxy_in_process.py`

- [ ] **Step 16.1.1: Write the test**

  Boot `ProxyService` + `RecordProcessor` + `RecordsManager` via the real (in-process) message bus harness used elsewhere in `tests/component_integration/`. Hit the proxy with `httpx` to a pytest-httpserver upstream. Assert aggregated metrics from `RecordsManager` match the expected values within tolerance.

- [ ] **Step 16.1.2: Run, debug to pass**

  Run: `uv run pytest tests/component_integration/test_proxy_in_process.py -m component_integration -n auto -v`
  Expected: pass.

- [ ] **Step 16.1.3: Commit**

  ```bash
  git add tests/component_integration/test_proxy_in_process.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(component_integration): proxy + record processor wiring (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 16.2: TaskRunnerService with mock harbor

**Files:**
- Create: `tests/component_integration/test_task_runner_with_mock_harbor.py`

- [ ] **Step 16.2.1: Write the test**

  Boot `TaskRunnerService` + `RecordProcessor` + `RecordsManager` with `mock_harbor` on `PATH`. Assert `harbor_task_metrics` aggregates show the expected resolved count.

- [ ] **Step 16.2.2: Run, debug to pass**

  Run: `uv run pytest tests/component_integration/test_task_runner_with_mock_harbor.py -m component_integration -n auto -v`
  Expected: pass.

- [ ] **Step 16.2.3: Commit**

  ```bash
  git add tests/component_integration/test_task_runner_with_mock_harbor.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(component_integration): task runner with mock_harbor end-to-end (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 17: Integration tests against the in-repo mock inference server

### Task 17.1: Agentic-only run

**Files:**
- Create: `tests/integration/test_agentic_against_mock_server.py`

- [ ] **Step 17.1.1: Write the test**

  Use the existing `tests/aiperf_mock_server/` (mock inference server) and `mock_harbor` (PATH shim) to drive a full multi-process AIPerf run: `aiperf profile <model> --url <mock-server-url> --task-runner harbor --harbor-benchmark swe-bench --task-concurrency 2 --num-requests <small>`. Assert:
  - Exit code 0
  - `profile_export.json` contains both `proxy_request_metrics` and `harbor_task_metrics`
  - Per-source breakdown for `agent` has nonzero count
  - `<artifact_dir>/harbor/<instance_id>/rollout.json` files exist

- [ ] **Step 17.1.2: Run, debug to pass**

  Run: `uv run pytest tests/integration/test_agentic_against_mock_server.py -m integration -n auto -v`
  Expected: pass.

- [ ] **Step 17.1.3: Commit**

  ```bash
  git add tests/integration/test_agentic_against_mock_server.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(integration): agentic run against mock server (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 17.2: Agentic + replay co-traffic

**Files:**
- Create: `tests/integration/test_agentic_plus_replay_against_mock_server.py`

- [ ] **Step 17.2.1: Write the test**

  Same shape as 17.1 plus `--input-file <small.dag.jsonl>`. Assert both `agent` and `replay` source counts are nonzero, and the all-sources count equals the sum.

- [ ] **Step 17.2.2: Run, debug to pass**

  Run: `uv run pytest tests/integration/test_agentic_plus_replay_against_mock_server.py -m integration -n auto -v`
  Expected: pass.

- [ ] **Step 17.2.3: Commit**

  ```bash
  git add tests/integration/test_agentic_plus_replay_against_mock_server.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(integration): mixed agentic + replay traffic, source bucketing (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 17.3: Proxy failure aborts the run

**Files:**
- Create: `tests/integration/test_proxy_failure_aborts_run.py`

- [ ] **Step 17.3.1: Write the test**

  Start a full AIPerf run, then deliberately kill the `ProxyService` process (or simulate FAILED state via an injected exception in `_start_server`). Assert SystemController emits a fatal lifecycle event and exits non-zero. Check the log for a clear "ProxyService failed" message.

- [ ] **Step 17.3.2: Run, pass**

  Run: `uv run pytest tests/integration/test_proxy_failure_aborts_run.py -m integration -v`
  Expected: pass.

- [ ] **Step 17.3.3: Commit**

  ```bash
  git add tests/integration/test_proxy_failure_aborts_run.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(integration): proxy failure causes fatal abort of the run (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 18: Property test for source-tag invariant

### Task 18.1: Mechanical invariant

**Files:**
- Create: `tests/unit/property/test_source_tag_invariant.py`

- [ ] **Step 18.1.1: Write the property test**

  Generate random sequences of `ProxyRequestRecordMessage` (via `hypothesis` or simple parameterized) and feed them to the RecordProcessor; assert `count_total == sum(by_source[s].count for s in sources)` for every input. Skim `tests/unit/property/test_finite_invariants.py` for the project's existing property-style invariants pattern.

- [ ] **Step 18.1.2: Run, pass**

  Run: `uv run pytest tests/unit/property/test_source_tag_invariant.py -v`
  Expected: pass.

- [ ] **Step 18.1.3: Commit**

  ```bash
  git add tests/unit/property/test_source_tag_invariant.py
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  test(property): source-tag count invariant for proxy aggregates (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Milestone 19: Documentation

### Task 19.1: User-facing benchmark-modes guide

**Files:**
- Create: `docs/benchmark-modes/agentic-harbor.md`
- Modify: `docs/index.yml`

- [ ] **Step 19.1.1: Write the doc**

  Create `docs/benchmark-modes/agentic-harbor.md` covering:
  - What `--task-runner harbor` does in one paragraph
  - Prerequisites: install harbor (`uv tool install harbor`), have Daytona/Modal/e2b credentials configured per harbor's docs, an OpenAI-compatible inference endpoint to benchmark
  - Worked example for SWE-bench: full `aiperf profile ...` command
  - Worked example for Terminal-bench: full command
  - Worked example with co-traffic: `--task-runner harbor` + `--input-file <dag_jsonl>`
  - How to read the report (per-source table + task aggregate)
  - Troubleshooting: `harbor_crash`, `schema_mismatch`, `task_failure` — when each happens
  - Limitations: v1 non-goals (no per-step request-ID correlation, no crash recovery, harbor manages sandboxes)

  Use mermaid for any diagrams, not ASCII. No emojis. Reference the spec from §1's "see also" for engineers needing internals.

- [ ] **Step 19.1.2: Add to Fern index**

  In `docs/index.yml`, add the new doc under the appropriate section (likely under "Benchmark Modes"). Match the formatting of the other entries.

- [ ] **Step 19.1.3: Verify check_docs_index**

  Run: `uv run python tools/check_docs_index.py`
  Expected: pass.

- [ ] **Step 19.1.4: Commit**

  ```bash
  git add docs/benchmark-modes/agentic-harbor.md docs/index.yml
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  docs(benchmark-modes): agentic-harbor user guide (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 19.2: Update architecture and patterns

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/dev/patterns.md`

- [ ] **Step 19.2.1: Update architecture.md**

  Add `ProxyService` and `TaskRunnerService` to the components section. Update the data-flow diagram (mermaid) to show the proxy on the critical path when `--task-runner harbor` is on.

- [ ] **Step 19.2.2: Update patterns.md**

  Add a "Long-running task runner" entry pointing at `TaskRunnerService` as the reference implementation: dispatcher coroutine + per-instance worker coroutines + asyncio.Semaphore, with the failure-category contract.

- [ ] **Step 19.2.3: Commit**

  ```bash
  git add docs/architecture.md docs/dev/patterns.md
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  docs: add proxy + task-runner services to architecture + patterns (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 19.3: Four-file agent docs sync

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`

- [ ] **Step 19.3.1: Add a "Long-running task runners" tip**

  In the "Tips" section of each file, add a one-liner pointing at `TaskRunnerService` as the reference for adding additional task runners.

- [ ] **Step 19.3.2: Verify sync**

  Run: `uv run make check-agent-files-sync`
  Expected: pass.

- [ ] **Step 19.3.3: Commit**

  ```bash
  git add AGENTS.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
  PATH="$HOME/.local/bin:$PATH" git commit -s -m "$(cat <<'EOF'
  docs(agents): four-file sync for task-runner reference (AIP-920)

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Final verification

- [ ] **Step F.1: Full unit-test sweep**

  Run: `uv run pytest tests/unit/ -n auto -q`
  Expected: all pass.

- [ ] **Step F.2: Component integration sweep**

  Run: `uv run pytest -m component_integration -n auto -q`
  Expected: all pass.

- [ ] **Step F.3: Integration sweep**

  Run: `uv run pytest -m integration -n auto -q`
  Expected: all pass.

- [ ] **Step F.4: Pre-commit on the full tree**

  Run: `PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files`
  Expected: pass (codespell, ruff, ergonomics-baseline, schema validation, etc.)

- [ ] **Step F.5: Plugin schema validation**

  Run: `uv run make validate-plugin-schemas`
  Expected: pass.

- [ ] **Step F.6: Doc generation regen check**

  Run: `uv run make generate-all-docs`
  Expected: docs regenerated; `git diff` shows no unstaged changes (everything you generated is already committed).

- [ ] **Step F.7: Push branch**

  Run: `git push -u origin dbermudez/aip-920-harbor-integration-for-agentic-benchmarking`

- [ ] **Step F.8: Open PR**

  Use `gh pr create` per the project workflow. PR title: `feat: harbor integration for agentic benchmarking (AIP-920)`. PR body should reference the spec at `docs/superpowers/specs/2026-05-29-aiperf-harbor-agentic-design.md` and the Linear issue AIP-920.

---

## Dependency-ordered execution sequence

For a subagent executing this plan, the dependency-correct order is:

```
M1 → M2 → M3 → M14 → M15 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M12 → M13 → M16 → M17 → M18 → M19 → F.*
```

The document presents milestones in a service-by-service order for reader comprehension. Always check the dependency annotation on each milestone before starting it.
