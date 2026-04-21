# Msgspec Primitives P2: Message Base Flip — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip `Message` from `AIPerfBaseModel`/`AutoRoutedModel` to `msgspec.Struct(tag_field="message_type")`, convert every `Message` subclass (~40 envelopes across 9 modules) into a tagged-union member, delete `AutoRoutedModel`, detach `AIPerfBaseModel` from it, and retire `PydanticStructMixin` from every struct that only appears inside `Message` payloads.

**Architecture:** The `Message` class currently uses `AutoRoutedModel`'s `__init_subclass__` registry to discriminate on `message_type`. msgspec supports this natively via `tag_field="message_type"` + per-subclass `tag=MessageType.X`. To keep diff blast radius manageable, `Message` retains shim methods (`from_json`, `to_json_bytes`, `model_dump_json`, `model_dump`, `__str__`) implemented over msgspec — transport clients, tests, and the JSON/msgpack codecs continue calling the same API. The two non-Message `AutoRoutedModel` consumers (`TraceDataExport`, `BaseServerMetricData`) never exercise `from_json` routing — they only set the ClassVar — so they detach cleanly with a one-line removal. `PydanticStructMixin` stays defined for the final export boundary (`ErrorDetailsCount` inside `JsonExportData`/`TelemetryExportData`) but is removed as a mixin from every other struct once no Pydantic envelope carries them.

**Tech Stack:** msgspec 0.18+ (`msgspec.Struct`, `msgspec.json`, `msgspec.msgpack`, `msgspec.convert`, `msgspec.to_builtins`), Pydantic v2 (`BaseModel`, `ConfigDict`), orjson (already used), pytest/pytest-asyncio.

---

## File Structure

**Modified (convert Message subclass envelopes to msgspec):**
- `src/aiperf/common/messages/base_messages.py` — `Message`, `RequiresRequestNSMixin`, `ErrorMessage`
- `src/aiperf/common/messages/service_messages.py` — 8 classes
- `src/aiperf/common/messages/worker_messages.py` — 4 classes
- `src/aiperf/common/messages/progress_messages.py` — 5 classes
- `src/aiperf/common/messages/inference_messages.py` — 2 classes
- `src/aiperf/common/messages/dataset_messages.py` — 6 classes
- `src/aiperf/common/messages/telemetry_messages.py` — 3 classes
- `src/aiperf/common/messages/server_metrics_messages.py` — 3 classes
- `src/aiperf/credit/messages.py` — 6 `Message` subclasses (Router↔Worker structs already msgspec; untouched)

**Modified (AutoRoutedModel / AIPerfBaseModel detach):**
- `src/aiperf/common/models/base_models.py` — drop `AutoRoutedModel` parent from `AIPerfBaseModel`
- `src/aiperf/common/models/trace_models.py` — drop `discriminator_field` ClassVar on `TraceDataExport`
- `src/aiperf/common/models/server_metrics_models.py` — drop `discriminator_field` ClassVar on `BaseServerMetricData`
- `src/aiperf/common/models/__init__.py` — remove `AutoRoutedModel` export
- `src/aiperf/common/models/record_models.py` — comment at line 505 references `AutoRoutedModel discrimination`; update since `BaseTraceData.from_json` uses msgspec directly

**Modified (PydanticStructMixin removal from structs no longer in Pydantic parents):**
- `src/aiperf/common/models/error_models.py` — `ErrorDetails`, `ExitErrorInfo` drop mixin; `ErrorDetailsCount` KEEPS mixin (still in `JsonExportData.error_summary`/`TelemetryExportData.error_summary`)
- `src/aiperf/common/models/record_models.py` — 8 `PydanticStructMixin` bases removed
- `src/aiperf/common/models/progress_models.py` — 2 removed
- `src/aiperf/common/models/dataset_models.py` — 7 removed
- `src/aiperf/common/models/credit_models.py` — 2 removed
- `src/aiperf/common/models/health_models.py` — 5 removed
- `src/aiperf/common/models/service_models.py` — 1 removed
- `src/aiperf/common/models/worker_models.py` — 1 removed
- `src/aiperf/timing/config.py` — `CreditPhaseConfig` drops mixin

**Deleted:**
- `src/aiperf/common/models/auto_routed_model.py`

**Test additions / changes:**
- Create: `tests/unit/common/messages/test_tagged_union_roundtrip.py` — msgpack and JSON round-trip via tagged-union decode
- Modify: `tests/unit/common/models/test_auto_routed_messages.py` — convert to tagged-union semantics OR delete and inline coverage into the new roundtrip test
- Modify: `tests/unit/common/messages/test_messages.py` — tests that rely on Pydantic-specific behavior need adjustment
- Modify: `tests/unit/zmq/test_router_reply_client.py`, `test_sub_client.py`, `test_pull_client.py`, `test_dealer_request_client.py` — verify `to_json_bytes`/`model_dump_json` shims still produce byte-equivalent output

**Docs update:**
- `docs/superpowers/specs/2026-04-20-msgspec-primitives-migration-design.md` — mark P2 landed, add commit hash to overview status matrix after the PR merges

---

## Phase A: Preparation & Failing Tests

### Task A1: Write failing msgpack tagged-union round-trip test

**Files:**
- Create: `tests/unit/common/messages/test_tagged_union_roundtrip.py`

- [ ] **Step 1: Write the failing test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tagged-union round-trip tests for Message after P2 msgspec flip."""
import msgspec
import pytest

from aiperf.common.enums import LifecycleState, MessageType, ServiceType
from aiperf.common.messages import (
    HeartbeatMessage,
    Message,
    StatusMessage,
)


def test_heartbeat_message_msgpack_roundtrip_via_base_decoder():
    """Encoding a HeartbeatMessage and decoding as Message routes to HeartbeatMessage."""
    msg = HeartbeatMessage(
        service_id="svc-1",
        service_type=ServiceType.WORKER,
        state=LifecycleState.RUNNING,
    )
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=Message)

    restored = decoder.decode(encoder.encode(msg))

    assert isinstance(restored, HeartbeatMessage)
    assert restored.service_id == "svc-1"
    assert restored.service_type == ServiceType.WORKER
    assert restored.state == LifecycleState.RUNNING


def test_json_roundtrip_preserves_tag_field_name():
    """JSON emits the 'message_type' tag so external consumers stay compatible."""
    msg = StatusMessage(
        service_id="svc-2",
        service_type=ServiceType.TIMING_MANAGER,
        state=LifecycleState.STOPPED,
    )
    encoded = msgspec.json.encode(msg)
    as_dict = msgspec.json.decode(encoded)
    assert as_dict["message_type"] == MessageType.STATUS


def test_from_json_compat_wrapper_accepts_bytes_and_dict():
    """Message.from_json preserves AutoRoutedModel's dual-input API over msgspec."""
    heartbeat_dict = {
        "message_type": MessageType.HEARTBEAT,
        "service_id": "svc-3",
        "service_type": ServiceType.WORKER,
        "state": LifecycleState.RUNNING,
    }
    from_dict = Message.from_json(heartbeat_dict)
    import orjson
    from_bytes = Message.from_json(orjson.dumps(heartbeat_dict))
    assert isinstance(from_dict, HeartbeatMessage)
    assert isinstance(from_bytes, HeartbeatMessage)


def test_to_json_bytes_shim_emits_message_type_tag():
    """`to_json_bytes()` stays wire-compatible with the prior Pydantic path."""
    msg = HeartbeatMessage(
        service_id="svc-4",
        service_type=ServiceType.WORKER,
        state=LifecycleState.RUNNING,
    )
    import orjson
    decoded = orjson.loads(msg.to_json_bytes())
    assert decoded["message_type"] == MessageType.HEARTBEAT
    assert decoded["service_id"] == "svc-4"
    # request_id default (None) is excluded
    assert "request_id" not in decoded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/common/messages/test_tagged_union_roundtrip.py -v`
Expected: All four tests FAIL. `Message` is currently Pydantic; `msgspec.msgpack.Decoder(type=Message)` raises because `Message` is not a `Struct`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/unit/common/messages/test_tagged_union_roundtrip.py
git commit -s -m "test(primitives): add failing tagged-union roundtrip for P2"
```

---

## Phase B: Detach Non-Message AutoRoutedModel Consumers

The two non-Message AutoRoutedModel consumers (`TraceDataExport`, `BaseServerMetricData`) set `discriminator_field` but never actually exercise `AutoRoutedModel.from_json` routing — `TraceDataExport` uses a manual `_EXPORT_LOOKUP` (`trace_models.py:189`), and `BaseServerMetricData` subclasses are unioned via Pydantic's native `GaugeMetricData | CounterMetricData | HistogramMetricData` annotations in exporters. Dropping the ClassVar is a no-op at runtime; the register-in-parent step in `AutoRoutedModel.__init_subclass__` fires but nothing consumes the registry.

This phase is independent of Phase C and ships in its own commit so regressions here won't conflate with the message-flip commit.

### Task B1: Remove `discriminator_field` ClassVar from TraceDataExport

**Files:**
- Modify: `src/aiperf/common/models/trace_models.py:25-32`

- [ ] **Step 1: Write regression test**

Add to `tests/unit/common/models/test_trace_models.py`:
```python
def test_trace_data_export_has_no_autoroutedmodel_registry():
    """Sanity: TraceDataExport no longer populates AutoRoutedModel's lookup table."""
    # After P2-B1, TraceDataExport has no discriminator_field ClassVar.
    assert "discriminator_field" not in TraceDataExport.__dict__


def test_base_trace_data_to_export_still_routes_via_export_lookup():
    """to_export() uses the manual _EXPORT_LOOKUP, not AutoRoutedModel."""
    from aiperf.common.models.trace_models import AioHttpTraceData, AioHttpTraceDataExport
    data = AioHttpTraceData()
    assert isinstance(data.to_export(), AioHttpTraceDataExport)
```

- [ ] **Step 2: Run test — passes (sanity) or fails (trace_type dict assertion before edit)**

Run: `uv run pytest tests/unit/common/models/test_trace_models.py::test_trace_data_export_has_no_autoroutedmodel_registry -v`
Expected: FAIL (pre-edit the ClassVar still exists).

- [ ] **Step 3: Edit `src/aiperf/common/models/trace_models.py`**

Delete lines 25-27 (the `# For auto-routed-model serialization...` comment and the `discriminator_field: ClassVar[str] = "trace_type"` line). Keep the `trace_type: str = Field(...)` field — it remains the free-form export label read by `_EXPORT_LOOKUP`.

New state of the class head:
```python
class TraceDataExport(AIPerfBaseModel):
    """Export model with wall-clock timestamps following k6 and HAR conventions.

    All timestamps are converted from perf_counter to wall-clock time (time.time_ns())
    for correlation with logs, metadata, and cross-system analysis.

    Create from BaseTraceData using trace_data.to_export() method.
    """

    trace_type: str = Field(
        ...,
        description="The type of the trace. This is typically the name of the library used "
        "and must match the trace_type of the corresponding trace data model.",
    )
    ...
```

- [ ] **Step 4: Run regression tests**

Run: `uv run pytest tests/unit/common/models/test_trace_models.py tests/unit/metrics/test_http_trace_metrics.py -v`
Expected: all pass.

### Task B2: Remove `discriminator_field` ClassVar from BaseServerMetricData

**Files:**
- Modify: `src/aiperf/common/models/server_metrics_models.py:514-523`

- [ ] **Step 1: Write regression test**

Add to `tests/unit/server_metrics/test_json_exporter.py` (or adjacent test file):
```python
def test_base_server_metric_data_has_no_autoroutedmodel_registry():
    assert "discriminator_field" not in BaseServerMetricData.__dict__


def test_gauge_counter_histogram_unions_still_deserialize():
    """Pydantic's native Union discriminates on `type` field."""
    import msgspec
    from aiperf.common.models.server_metrics_models import (
        CounterMetricData,
        GaugeMetricData,
        HistogramMetricData,
    )
    payload = {"type": "gauge", "description": "d", "series": []}
    # Pydantic Union validation path
    from pydantic import TypeAdapter
    union_adapter = TypeAdapter(GaugeMetricData | CounterMetricData | HistogramMetricData)
    assert isinstance(union_adapter.validate_python(payload), GaugeMetricData)
```

- [ ] **Step 2: Run regression test**

Run: `uv run pytest tests/unit/server_metrics/ -v -k "has_no_autoroutedmodel or unions_still_deserialize"`
Expected: one new test fails (ClassVar still present).

- [ ] **Step 3: Edit `src/aiperf/common/models/server_metrics_models.py`**

Delete line 521 (`discriminator_field: ClassVar[str] = "type"`) and remove the `ClassVar` import if unused elsewhere in the file.

New state:
```python
class BaseServerMetricData(AIPerfBaseModel):
    """Base metric data with type, description, unit, and base series stats.

    Used in hybrid export format where metrics are keyed by name for O(1) lookup,
    but stats within each series are flattened for easy access.
    """

    type: PrometheusMetricType = Field(description="Metric type")

    description: str = Field(description="Metric description from HELP text")
    unit: str | None = Field(
        default=None,
        description="Unit inferred from metric name suffix (_seconds, _bytes, etc.)",
    )
```

- [ ] **Step 4: Run regression tests**

Run: `uv run pytest tests/unit/server_metrics/ -n auto`
Expected: all pass.

### Task B3: Commit Phase B

- [ ] **Step 1: Run broader suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all pass.

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/common/models/trace_models.py \
        src/aiperf/common/models/server_metrics_models.py \
        tests/unit/common/models/test_trace_models.py \
        tests/unit/server_metrics/
git commit -s -m "refactor(primitives): detach trace/server-metrics exports from AutoRoutedModel (P2-B)"
```

---

## Phase C: Flip Message Base + Convert Envelopes (Atomic)

This is the largest single logical change in P2 and MUST land in one commit — the tagged-union decoder on `Message` requires every subclass to be defined at import time as a `Struct` with `tag=`. Partial conversion leaves the module tree unimportable.

### Task C1: Convert `Message`, `RequiresRequestNSMixin`, `ErrorMessage` to msgspec.Struct tagged-union root

**Files:**
- Modify: `src/aiperf/common/messages/base_messages.py`

- [ ] **Step 1: Rewrite base_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Any, ClassVar

import msgspec
import orjson
from typing_extensions import Self

from aiperf.common.enums import MessageType
from aiperf.common.models.error_models import ErrorDetails

_JSON_ENCODER = msgspec.json.Encoder()
_MSGPACK_ENCODER = msgspec.msgpack.Encoder()


class Message(
    msgspec.Struct,
    tag_field="message_type",
    kw_only=True,
    omit_defaults=True,
):
    """Base message class — msgspec.Struct tagged union on ``message_type``.

    Subclasses register their tag via ``class Foo(Message, tag=MessageType.FOO):``.
    msgspec resolves the concrete subclass on decode with zero registry machinery.

    Compatibility shims (``from_json``, ``to_json_bytes``, ``model_dump_json``,
    ``model_dump``, ``__str__``) preserve the prior Pydantic-backed API so
    transport clients, codecs, and tests need no call-site changes.
    """

    request_ns: int | None = None
    request_id: str | None = None

    # Subclass lookup for the dict-accepting variant of ``from_json``.
    # msgspec populates a tag->type mapping on the parent Struct class via
    # its own internals, but exposing a stable classmethod keeps the public
    # API intact. Discovery is lazy to avoid ordering issues at import time.
    _decoder_cache: ClassVar[msgspec.json.Decoder | None] = None

    @classmethod
    def _json_decoder(cls) -> msgspec.json.Decoder:
        if cls._decoder_cache is None:
            cls._decoder_cache = msgspec.json.Decoder(type=cls)
        return cls._decoder_cache

    @classmethod
    def from_json(cls, json_or_dict: str | bytes | bytearray | dict[str, Any]) -> Self:
        """Decode bytes/str/dict into the correct tagged-union subclass."""
        if isinstance(json_or_dict, dict):
            return msgspec.convert(json_or_dict, cls, strict=False)
        return cls._json_decoder().decode(json_or_dict)

    def to_json_bytes(self) -> bytes:
        """Serialize to JSON bytes (wire-compatible with prior Pydantic path)."""
        return _JSON_ENCODER.encode(self)

    def to_msgpack_bytes(self) -> bytes:
        """Serialize to msgpack bytes — used by the P3 single-codec path."""
        return _MSGPACK_ENCODER.encode(self)

    def model_dump(
        self,
        *,
        exclude_none: bool = False,
        mode: str | None = None,
        by_alias: bool = False,
    ) -> dict[str, Any]:
        """Pydantic-compat shim — ``mode`` and ``by_alias`` are accepted but ignored.

        ``omit_defaults=True`` on the Struct already drops None defaults, so the
        ``exclude_none`` branch is a no-op for the common path. Explicit non-None
        values are preserved.
        """
        data = msgspec.to_builtins(self)
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data

    def model_dump_json(self, *, exclude_none: bool = True, indent: int | None = None) -> str:
        """Pydantic-compat shim returning a JSON string."""
        encoded = _JSON_ENCODER.encode(self)
        if indent is not None:
            # Rare path — only used by debug/export; fall back to orjson.
            return orjson.dumps(orjson.loads(encoded), option=orjson.OPT_INDENT_2).decode()
        return encoded.decode()

    def __str__(self) -> str:
        return self.to_json_bytes().decode()

    @property
    def message_type(self) -> MessageType:
        """Expose the msgspec tag as the legacy ``message_type`` attribute.

        msgspec stores the tag on the ``__struct_tag__`` slot; surface it under
        the field name every existing caller already uses (``msg.message_type``).
        """
        return self.__struct_tag__  # type: ignore[attr-defined]


class RequiresRequestNSMixin(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Struct-inherited mixin: concrete subclasses must set ``request_ns``.

    msgspec supports multiple Struct parents provided field layouts don't
    conflict. Subclasses that compose this with ``Message`` or
    ``BaseServiceMessage`` end up with ``request_ns`` promoted to a required
    field with a default factory of ``time.time_ns``.
    """

    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]


class ErrorMessage(Message, tag=MessageType.ERROR):
    """Envelope carrying an ErrorDetails payload."""

    error: ErrorDetails
```

**Known risk — msgspec multiple-Struct inheritance:** the project memory flags `gotcha_msgspec_multiple_struct_inheritance.md` — "msgspec.Struct rejects multiple Struct parents". `RequiresRequestNSMixin` is currently used by `WorkerStartupStateMessage(BaseServiceMessage, RequiresRequestNSMixin)` and `AllRecordsReceivedMessage(BaseServiceMessage, RequiresRequestNSMixin)`. If msgspec rejects this multi-parent pattern, flatten the mixin at each use site by moving its single `request_ns` field directly onto the subclass with the same default_factory. This is the recommended fallback — the mixin has only one field, so duplication is trivial. Before starting Task C2, run a small smoke import from a Python REPL to confirm the multi-parent pattern works; if it doesn't, skip the mixin and inline `request_ns: int = msgspec.field(default_factory=time.time_ns)` on `AllRecordsReceivedMessage` and `WorkerStartupStateMessage` directly.

**Known risk — ExtensibleStrEnum tags:** project memory `gotcha_msgspec_extensible_str_enum.md` warns that `MessageType` (an `ExtensibleStrEnum`) needs enc/dec hooks for msgspec encode/decode round-trips. msgspec tag_field routing uses the `tag=` argument as a literal value at class setup — because `ExtensibleStrEnum` inherits from `str`, msgspec treats the tag as its str value, which is what external consumers expect. Round-trip decoding emits a `str` and matches against the tag registry. However, if a call site accesses `msg.message_type` and compares with `MessageType.HEARTBEAT`, the property returns whatever msgspec stored on `__struct_tag__` — which is the raw str. Because `MessageType` inherits from `str`, `raw_str == MessageType.HEARTBEAT` is True, so call sites keep working. If any test asserts `isinstance(msg.message_type, MessageType)`, it fails — convert those assertions to string equality. Run a targeted smoke test after Task C1: `HeartbeatMessage(...).message_type == MessageType.HEARTBEAT` should be True, `type(HeartbeatMessage(...).message_type)` should be `str`.

- [ ] **Step 2: Smoke-import verification**

Run: `uv run python -c "from aiperf.common.messages import Message, ErrorMessage; print(ErrorMessage.__struct_tag__)"`
Expected: prints `MessageType.ERROR`. If import fails with "instance lay-out conflict", apply the flatten-mixin fallback described above before proceeding to C2.

### Task C2: Convert `service_messages.py` (BaseServiceMessage + 7 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/service_messages.py`

- [ ] **Step 1: Rewrite service_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import msgspec

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.memory_tracker import MemoryPhase
from aiperf.common.messages.base_messages import Message
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import ServiceTypeT


class BaseServiceMessage(Message, kw_only=True, omit_defaults=True):
    """Any message originating from a specific service; requires ``service_id``."""

    service_id: str


class BaseStatusMessage(BaseServiceMessage, kw_only=True, omit_defaults=True):
    """Lifecycle status message — ``request_ns`` defaults to ``time.time_ns``."""

    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]
    state: LifecycleState
    service_type: ServiceTypeT


class StatusMessage(BaseStatusMessage, tag=MessageType.STATUS):
    """Service status report."""


class RegistrationMessage(BaseStatusMessage, tag=MessageType.REGISTRATION):
    """Service self-registration."""


class HeartbeatMessage(BaseStatusMessage, tag=MessageType.HEARTBEAT):
    """Service heartbeat."""


class MemoryReportMessage(BaseServiceMessage, tag=MessageType.MEMORY_REPORT):
    """Self-reported memory snapshot from a service process."""

    pid: int
    service_type: ServiceTypeT
    phase: MemoryPhase
    pss_bytes: int
    rss_bytes: int | None = None
    uss_bytes: int | None = None
    shared_bytes: int | None = None


class ConnectionProbeMessage(BaseServiceMessage, tag=MessageType.CONNECTION_PROBE):
    """ZMQ slow-joiner self-echo probe."""


class BaseServiceErrorMessage(BaseServiceMessage, tag=MessageType.SERVICE_ERROR):
    """Service-level error envelope."""

    error: ErrorDetails
```

- [ ] **Step 2: Smoke-import verification**

Run: `uv run python -c "from aiperf.common.messages import HeartbeatMessage, StatusMessage; h = HeartbeatMessage(service_id='s', service_type='worker', state='running'); print(h.to_json_bytes())"`
Expected: JSON bytes emitted with `message_type: "heartbeat"`.

### Task C3: Convert `worker_messages.py` (4 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/worker_messages.py`

- [ ] **Step 1: Rewrite worker_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import msgspec

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats


class WorkerHealthMessage(BaseServiceMessage, tag=MessageType.WORKER_HEALTH):
    """Worker health check."""

    health: ProcessHealth
    task_stats: WorkerTaskStats

    @property
    def error_rate(self) -> float:
        if self.task_stats.total == 0:
            return 0
        return self.task_stats.failed / self.task_stats.total


class WorkerStatusSummaryMessage(
    BaseServiceMessage, tag=MessageType.WORKER_STATUS_SUMMARY
):
    """Aggregate worker status by worker_id."""

    worker_statuses: dict[str, WorkerStatus]
    worker_startup_states: dict[str, WorkerStartupState] = msgspec.field(
        default_factory=dict
    )


class WorkerPodStateMessage(BaseServiceMessage, tag=MessageType.WORKER_POD_STATE):
    """Controller-facing aggregate snapshot for a Kubernetes worker pod."""

    pod_index: str
    declared_workers: int
    declared_record_processors: int
    pod_state: str
    admission_state: str
    benchmark_generation: str | None = None
    dataset_generation: str | None = None
    router_connected_workers: int = 0
    dispatchable_workers: int = 0
    ready_workers: int = 0
    ready_record_processors: int = 0
    degraded_workers: int = 0
    degraded_record_processors: int = 0


class WorkerStartupStateMessage(
    BaseServiceMessage, tag=MessageType.WORKER_STARTUP_STATE
):
    """Worker startup lifecycle transition.

    Inlines ``request_ns`` with a default_factory to avoid the
    multi-Struct-inheritance pattern (see gotcha_msgspec_multiple_struct_inheritance).
    """

    startup_state: WorkerStartupState
    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]
```

Note: the inlined `request_ns` field replaces the prior `RequiresRequestNSMixin` dual-parent pattern.

### Task C4: Convert `progress_messages.py` (5 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/progress_messages.py`

- [ ] **Step 1: Rewrite progress_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import PhaseRecordsStats, WorkerProcessingStats
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults


class RecordsProcessingStatsMessage(BaseServiceMessage, tag=MessageType.PROCESSING_STATS):
    """Per-phase processing stats from the RecordsManager."""

    processing_stats: PhaseRecordsStats
    worker_stats: dict[str, WorkerProcessingStats] = msgspec.field(default_factory=dict)


class ProfileResultsMessage(BaseServiceMessage, tag=MessageType.PROFILE_RESULTS):
    """Final profile results."""

    profile_results: ProfileResults


class AllRecordsReceivedMessage(
    BaseServiceMessage, tag=MessageType.ALL_RECORDS_RECEIVED
):
    """All parsed records received; final stats available."""

    final_processing_stats: PhaseRecordsStats
    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]


class ProcessRecordsResultMessage(
    BaseServiceMessage, tag=MessageType.PROCESS_RECORDS_RESULT
):
    """Record-processor batch result."""

    results: ProcessRecordsResult


class BenchmarkCompleteMessage(BaseServiceMessage, tag=MessageType.BENCHMARK_COMPLETE):
    """Benchmark completion signal."""

    was_cancelled: bool = False
```

### Task C5: Convert `inference_messages.py` (2 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/inference_messages.py`

- [ ] **Step 1: Rewrite inference_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import RequestRecord
from aiperf.common.models.record_models import MetricResult


class InferenceResultsMessage(BaseServiceMessage, tag=MessageType.INFERENCE_RESULTS):
    """Single inference result record."""

    record: RequestRecord


class RealtimeMetricsMessage(BaseServiceMessage, tag=MessageType.REALTIME_METRICS):
    """Real-time metrics summary."""

    metrics: list[MetricResult]
```

### Task C6: Convert `dataset_messages.py` (6 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/dataset_messages.py`

- [ ] **Step 1: Rewrite dataset_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    Conversation,
    DatasetClientMetadata,
    DatasetMetadata,
    MemoryMapClientMetadata,
    Turn,
)


class ConversationRequestMessage(
    BaseServiceMessage, tag=MessageType.CONVERSATION_REQUEST
):
    """Request a full conversation by ID."""

    conversation_id: str
    credit_phase: CreditPhase | None = None


class ConversationResponseMessage(
    BaseServiceMessage, tag=MessageType.CONVERSATION_RESPONSE
):
    """Full conversation payload."""

    conversation: Conversation


class ConversationTurnRequestMessage(
    BaseServiceMessage, tag=MessageType.CONVERSATION_TURN_REQUEST
):
    """Request a single turn by (conversation_id, turn_index)."""

    conversation_id: str
    turn_index: int


class ConversationTurnResponseMessage(
    BaseServiceMessage, tag=MessageType.CONVERSATION_TURN_RESPONSE
):
    """Single turn payload."""

    turn: Turn


class DatasetConfiguredNotification(
    BaseServiceMessage, tag=MessageType.DATASET_CONFIGURED_NOTIFICATION
):
    """Broadcast that dataset configuration is complete."""

    metadata: DatasetMetadata
    client_metadata: DatasetClientMetadata
    benchmark_generation: str
    dataset_generation: str


class DatasetDownloadedNotification(
    BaseServiceMessage, tag=MessageType.DATASET_DOWNLOADED_NOTIFICATION
):
    """Pod-scoped dataset download complete."""

    client_metadata: MemoryMapClientMetadata
    pod_index: str | None = None
    success: bool = True
    error_message: str | None = None
```

### Task C7: Convert `telemetry_messages.py` (3 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/telemetry_messages.py`

- [ ] **Step 1: Rewrite telemetry_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import MetricResult, ProcessTelemetryResult


class ProcessTelemetryResultMessage(
    BaseServiceMessage, tag=MessageType.PROCESS_TELEMETRY_RESULT
):
    """Processed telemetry results envelope."""

    telemetry_result: ProcessTelemetryResult


class TelemetryStatusMessage(BaseServiceMessage, tag=MessageType.TELEMETRY_STATUS):
    """Telemetry availability report."""

    enabled: bool
    reason: str | None = None
    endpoints_configured: list[str] = msgspec.field(default_factory=list)
    endpoints_reachable: list[str] = msgspec.field(default_factory=list)


class RealtimeTelemetryMetricsMessage(
    BaseServiceMessage, tag=MessageType.REALTIME_TELEMETRY_METRICS
):
    """Real-time GPU telemetry metrics."""

    metrics: list[MetricResult]
```

### Task C8: Convert `server_metrics_messages.py` (3 subclasses)

**Files:**
- Modify: `src/aiperf/common/messages/server_metrics_messages.py`

- [ ] **Step 1: Rewrite server_metrics_messages.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models.server_metrics_models import (
    ProcessServerMetricsResult,
    ServerMetricsEndpointSummary,
)


class ServerMetricsStatusMessage(
    BaseServiceMessage, tag=MessageType.SERVER_METRICS_STATUS
):
    """Server-metrics availability report."""

    enabled: bool
    reason: str | None = None
    endpoints_configured: list[str] = msgspec.field(default_factory=list)
    endpoints_reachable: list[str] = msgspec.field(default_factory=list)


class ProcessServerMetricsResultMessage(
    BaseServiceMessage, tag=MessageType.PROCESS_SERVER_METRICS_RESULT
):
    """Processed server-metrics results envelope."""

    server_metrics_result: ProcessServerMetricsResult


class RealtimeServerMetricsMessage(
    BaseServiceMessage, tag=MessageType.REALTIME_SERVER_METRICS
):
    """Real-time per-endpoint server metrics."""

    endpoint_summaries: dict[str, ServerMetricsEndpointSummary]
```

### Task C9: Convert `credit/messages.py` Message subclasses (6 classes)

**Files:**
- Modify: `src/aiperf/credit/messages.py:19-62`

- [ ] **Step 1: Edit the 6 Message-tagged envelopes at the top of the file**

Leave the router↔worker structs (WorkerConnected, WorkerDispatchable, etc., lines 69+) untouched — they're already `msgspec.Struct` tagged unions on their own `t` field and are not routed through the main `Message` bus.

Replace the Pydantic envelopes:

```python
# top of file, update imports
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import BaseServiceMessage
from aiperf.common.models import CreditPhaseStats
from aiperf.timing.config import CreditPhaseConfig


class CreditPhasesConfiguredMessage(
    BaseServiceMessage, tag=MessageType.CREDIT_PHASES_CONFIGURED
):
    """Credit phase configuration announcement."""

    configs: list[CreditPhaseConfig]


class CreditPhaseStartMessage(BaseServiceMessage, tag=MessageType.CREDIT_PHASE_START):
    """Credit phase start announcement."""

    stats: CreditPhaseStats
    config: CreditPhaseConfig


class CreditPhaseProgressMessage(
    BaseServiceMessage, tag=MessageType.CREDIT_PHASE_PROGRESS
):
    """Credit phase progress update."""

    stats: CreditPhaseStats


class CreditPhaseSendingCompleteMessage(
    BaseServiceMessage, tag=MessageType.CREDIT_PHASE_SENDING_COMPLETE
):
    """Credit phase has finished sending (but may still be awaiting returns)."""

    stats: CreditPhaseStats


class CreditPhaseCompleteMessage(
    BaseServiceMessage, tag=MessageType.CREDIT_PHASE_COMPLETE
):
    """Credit phase is fully complete."""

    stats: CreditPhaseStats


class CreditsCompleteMessage(BaseServiceMessage, tag=MessageType.CREDITS_COMPLETE):
    """All credit phases complete."""
```

Remove the `from pydantic import Field` import at the top. Do not touch lines 69+ (router↔worker structs).

### Task C10: Update JsonMessageCodec + PydanticMsgpackCodec to work over msgspec

**Files:**
- Modify: `src/aiperf/common/message_codecs.py`

The codec layer must keep working through P2 — the P3 deletion happens later. Rewrite the two Pydantic-centric codecs to use msgspec under the hood but preserve their public API and cache_keys.

- [ ] **Step 1: Rewrite message_codecs.py (P3 will delete two of these)**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable message codecs for transport clients."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import msgspec
from pydantic import BaseModel

from aiperf.common.messages import Message


def _enc_hook(obj: Any) -> Any:
    """Encoder fallback for legacy Pydantic fields embedded in msgspec structs.

    After P2 this hook is unreachable for the records path — kept for channels
    still passing Pydantic export models. P3 deletes this.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"Object of type {type(obj).__name__} is not msgpack-encodable")


def _dec_hook(t: type, obj: Any) -> Any:
    """Decoder fallback symmetric to ``_enc_hook`` — retired in P3."""
    if isinstance(obj, dict) and isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_validate(obj)
    raise NotImplementedError(
        f"Unsupported msgspec decode target {t!r} for value {type(obj).__name__}"
    )


@runtime_checkable
class MessageCodecProtocol(Protocol):
    cache_key: str

    def encode(self, message: Any) -> bytes: ...
    def decode(self, data: bytes) -> Any: ...


class JsonMessageCodec:
    """JSON codec (msgspec-backed). Wire-equivalent to the prior Pydantic path."""

    cache_key = "json-message"

    def __init__(self) -> None:
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder(type=Message)

    def encode(self, message: Message) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Message:
        return self._decoder.decode(data)


class PydanticMsgpackCodec:
    """Transitional: msgpack encode of the tagged ``Message`` union.

    Kept alive through P2 so channels that explicitly opted into this codec
    don't need a simultaneous rewrite. P3 deletes this and collapses all
    traffic onto ``MsgspecStructCodec(decode_type=Message)``.
    """

    def __init__(
        self,
        *,
        cache_key: str,
        message_base_type: type[Message] = Message,
    ) -> None:
        self.cache_key = cache_key
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(type=message_base_type)

    def encode(self, message: Message) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Message:
        return self._decoder.decode(data)


class MsgspecStructCodec:
    """Typed msgpack codec — primary codec for records/raw-inference channels."""

    def __init__(self, *, decode_type: Any, cache_key: str) -> None:
        self.cache_key = cache_key
        self._encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)
        self._decoder = msgspec.msgpack.Decoder(type=decode_type, dec_hook=_dec_hook)

    def encode(self, message: Any) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Any:
        return self._decoder.decode(data)


JSON_MESSAGE_CODEC = JsonMessageCodec()


def codec_cache_key(codec: MessageCodecProtocol | None) -> str:
    return codec.cache_key if codec is not None else JSON_MESSAGE_CODEC.cache_key
```

### Task C11: Run focused test suites after the atomic set of edits

- [ ] **Step 1: Run new tagged-union test**

Run: `uv run pytest tests/unit/common/messages/test_tagged_union_roundtrip.py -v`
Expected: all four tests PASS.

- [ ] **Step 2: Run messages/codec suites**

Run: `uv run pytest tests/unit/common/messages/ tests/unit/common/test_message_codecs.py tests/unit/common/models/test_auto_routed_messages.py -v`
Expected: most pass. Any failures fall into three categories, addressed in Task C12:
1. Tests asserting Pydantic-specific error types (`pydantic.ValidationError`) — switch to `msgspec.ValidationError` / `msgspec.DecodeError`.
2. Tests constructing a message with unknown extra kwargs (Pydantic's `extra="allow"` permitted this; msgspec's `forbid_unknown_fields` default rejects) — remove the extra kwargs or drop the test.
3. Tests that call `message.model_construct(...)` or other Pydantic-only methods — replace with direct struct construction.

- [ ] **Step 3: Run ZMQ transport tests**

Run: `uv run pytest tests/unit/zmq/ -v`
Expected: `test_router_reply_client.py`, `test_sub_client.py`, `test_pull_client.py`, `test_dealer_request_client.py` pass because they call the `to_json_bytes()` / `model_dump_json()` shims on the new Message base and then decode via `Message.from_json`. If drift appears, the shim methods need adjustment — do not rewrite the transport clients.

### Task C12: Fix test drift (targeted, not speculative)

**Files:**
- Modify: any test from Task C11 that surfaced a real failure.

- [ ] **Step 1: Enumerate failures**

Run with quiet output to get the concise list:
`uv run pytest tests/unit/common/messages/ tests/unit/common/models/test_auto_routed_messages.py tests/unit/zmq/ 2>&1 | grep -E "(FAILED|ERROR)"`

- [ ] **Step 2: Fix each one in place**

Per the three failure categories above. Do not re-run until all files in the failure list are edited; then run once.

- [ ] **Step 3: Re-run**

Run: `uv run pytest tests/unit/common/messages/ tests/unit/common/models/test_auto_routed_messages.py tests/unit/zmq/ -v`
Expected: all pass.

### Task C13: Commit Phase C

- [ ] **Step 1: Final broad check**

Run: `uv run pytest tests/unit/ -n auto 2>&1 | tail -20`
Expected: 0 failed.

- [ ] **Step 2: Pre-commit**

Run: `ruff format . && ruff check --fix . && pre-commit run --all-files`
Fix any drift — esp. `test-imports`, `generate-plugin-artifacts`.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/common/messages/ \
        src/aiperf/common/message_codecs.py \
        src/aiperf/credit/messages.py \
        tests/
git commit -s -m "refactor(primitives): flip Message base to msgspec.Struct tagged union (P2-C)

Every envelope in base_messages, service_messages, worker_messages,
progress_messages, inference_messages, dataset_messages, telemetry_messages,
server_metrics_messages, and credit/messages.py becomes a tagged-union
member on message_type. JsonMessageCodec and PydanticMsgpackCodec are
rewritten over msgspec but keep their public API; P3 collapses them.
Message keeps from_json / to_json_bytes / model_dump_json / model_dump /
__str__ as msgspec-backed shims so transport clients and tests need no
call-site changes."
```

---

## Phase D: Delete `AutoRoutedModel`, Detach `AIPerfBaseModel`

### Task D1: Detach AIPerfBaseModel

**Files:**
- Modify: `src/aiperf/common/models/base_models.py`

- [ ] **Step 1: Edit base_models.py**

Replace the `AutoRoutedModel` parent with plain `BaseModel`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from pathlib import PurePath
from typing import Any

import msgspec
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class AIPerfBaseModel(BaseModel):
    """Base model for all AIPerf Pydantic models.

    This class is configured to allow arbitrary types to be used as fields
    to allow for more flexible model definitions by end users without breaking
    existing code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


# _msgspec_enc_hook / _msgspec_dec_hook / PydanticStructMixin: unchanged.
# (Leave the rest of the file as-is.)
```

- [ ] **Step 2: Verify import graph**

Run: `uv run python -c "from aiperf.common.models import AIPerfBaseModel, ErrorDetails; print(AIPerfBaseModel.__mro__)"`
Expected: MRO shows `AIPerfBaseModel -> BaseModel -> object`. No `AutoRoutedModel` in the chain.

### Task D2: Delete AutoRoutedModel module + export

**Files:**
- Delete: `src/aiperf/common/models/auto_routed_model.py`
- Modify: `src/aiperf/common/models/__init__.py:14,147`

- [ ] **Step 1: Edit models/__init__.py**

Remove the import (line 14) and the `"AutoRoutedModel"` entry from `__all__` (line 147).

- [ ] **Step 2: Delete the module**

```bash
git rm src/aiperf/common/models/auto_routed_model.py
```

- [ ] **Step 3: Verify no residual imports**

Run: `rg -n 'AutoRoutedModel|auto_routed_model' src/ tests/`
Expected: zero hits except in this plan file and the primitives spec doc.

If the grep surfaces live references, fix them in place (typically a leftover `from aiperf.common.models import AutoRoutedModel` in a file not yet touched).

### Task D3: Update the record_models.py stale comment

**Files:**
- Modify: `src/aiperf/common/models/record_models.py:505`

- [ ] **Step 1: Update the comment**

Before:
```python
        # Parity with the former Pydantic field_validator on trace_data:
        # route dict payloads to the correct BaseTraceData subclass via
        # AutoRoutedModel discrimination.
```

After:
```python
        # Parity with the former Pydantic field_validator on trace_data:
        # route dict payloads to the correct BaseTraceData subclass via
        # msgspec tagged-union decoding (see BaseTraceData.from_json).
```

### Task D4: Commit Phase D

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all pass.

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/common/models/ tests/
git commit -s -m "refactor(primitives): delete AutoRoutedModel, detach AIPerfBaseModel (P2-D)"
```

---

## Phase E: Retire `PydanticStructMixin` From Structs That No Longer Need It

The mixin stays *defined* in `base_models.py` and stays *applied* on `ErrorDetailsCount` (used in `JsonExportData.error_summary` and `TelemetryExportData.error_summary`, both `AIPerfBaseModel` subclasses). Every other struct that mixed it in for a Message payload loses the mixin now that Messages are msgspec-native.

### Task E1: Enumerate each struct and its remaining Pydantic-parent usage

- [ ] **Step 1: Gather the list**

Run: `rg -n 'PydanticStructMixin,' src/aiperf/common/models/ src/aiperf/timing/config.py`
Expected output (exact lines from the pre-P2 audit):
- `error_models.py`: ErrorDetails (13), ExitErrorInfo (103), ErrorDetailsCount (127)
- `record_models.py`: 8 sites (51, 122, 152, 217, 238, 279, 311, 426, 463) — all msgspec envelopes for request records
- `progress_models.py`: 15, 37 — PhaseRecordsStats, WorkerProcessingStats
- `dataset_models.py`: 16, 128, 164, 177, 254, 268, 313, 347, 361 — DatasetMetadata, Conversation, Turn, client-metadata tagged union, etc.
- `credit_models.py`: 13, 208 — CreditPhaseStats et al.
- `health_models.py`: 10, 43, 75, 88, 99, 108 — ProcessHealth + phase stats
- `service_models.py`: 15 — ServiceRunInfo/WorkerTaskStats
- `worker_models.py`: 11 — WorkerTaskStats
- `timing/config.py`: 97 — CreditPhaseConfig

- [ ] **Step 2: For each candidate, confirm the struct no longer appears as a field on any Pydantic (AIPerfBaseModel) class**

For a struct `T` at `src/.../foo_models.py:N`, run:
`rg -n ": list\[T\]|: T|: T \|" src/ tests/` and verify every hit lands in another msgspec.Struct (not a Pydantic class). Expected: all remaining hits are msgspec payloads of now-msgspec Message envelopes.

**Keep PydanticStructMixin on ErrorDetailsCount** — `rg -n "ErrorDetailsCount" src/aiperf/common/models/export_models.py` shows `JsonExportData.error_summary: list[ErrorDetailsCount] | None` and `TelemetryExportData.error_summary: list[ErrorDetailsCount] | None`. Both envelopes are AIPerfBaseModel (Pydantic); the mixin is still load-bearing for `model_dump_json` to serialize them.

### Task E2: Remove the mixin from each struct not in the keep-list

**Files:**
- Modify: every file enumerated in E1 except `error_models.py` (which keeps `ErrorDetailsCount`).

For each struct:
```python
class Foo(
    PydanticStructMixin,  # <-- delete this line
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
```
Becomes:
```python
class Foo(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
```

Remove the `from aiperf.common.models.base_models import PydanticStructMixin` import where it becomes unused.

- [ ] **Step 1: Edit `error_models.py`**

Remove `PydanticStructMixin,` line from `ErrorDetails` (line 13) and `ExitErrorInfo` (line 103). Keep it on `ErrorDetailsCount` (line 127).

- [ ] **Step 2: Edit `record_models.py`**

Remove `PydanticStructMixin,` from all 8 struct heads (lines ~51, 122, 152, 217, 238, 279, 311, 426, 463). Remove the import if now unused.

- [ ] **Step 3: Edit `progress_models.py`, `dataset_models.py`, `credit_models.py`, `health_models.py`, `service_models.py`, `worker_models.py`, `timing/config.py`**

Same pattern: remove `PydanticStructMixin,` from each struct head, remove the import if unused.

### Task E3: Run the envelope roundtrip + exporter suites

- [ ] **Step 1: Run roundtrip tests**

Run: `uv run pytest tests/unit/common/messages/ tests/unit/credit/test_progress_envelope_roundtrip.py tests/unit/common/models/ -v`
Expected: all pass. Records/dataset/credit/lifecycle envelope roundtrip tests confirm the Message envelopes still serialize cleanly without the struct-level mixin.

- [ ] **Step 2: Run exporter tests**

Run: `uv run pytest tests/unit/server_metrics/ tests/unit/exporters/ -v` (or whatever the project's exporter test path is — `rg -l 'JsonExportData\|TelemetryExportData' tests/` yields the actual set).
Expected: all pass. If `ErrorDetailsCount` serialization is broken, the mixin was inadvertently removed from it — restore.

### Task E4: Commit Phase E

- [ ] **Step 1: Full unit + component-integration suites**

Run: `uv run pytest tests/unit/ -n auto && uv run pytest -m component_integration -n auto`
Expected: all pass.

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/common/models/ src/aiperf/timing/config.py tests/
git commit -s -m "refactor(primitives): remove PydanticStructMixin from structs only used in msgspec envelopes (P2-E)

ErrorDetailsCount keeps the mixin — still a field on
JsonExportData.error_summary / TelemetryExportData.error_summary."
```

---

## Phase F: Integration Tests + Spec Status

### Task F1: Run the full integration suite

- [ ] **Step 1: Integration tests**

Run: `uv run pytest -m integration -n auto 2>&1 | tee /tmp/p2-integration.log`
Expected: 0 failed.

If failures appear, they are almost always from:
- A transport-level test that assumes JSON output (fix with a JSON-specific assertion path)
- A service test that mocked `Message.model_validate(...)` (now a classmethod on the Struct — `msgspec.convert` is the replacement)

Fix each in place; do not revert any earlier phase's changes.

### Task F2: Mark P2 complete in the specs + status matrix

**Files:**
- Modify: `docs/superpowers/specs/2026-04-20-msgspec-primitives-migration-design.md:1-10` — flip status from "Partial (P1 landed, P2+P3 pending)" to "Partial (P1+P2 landed, P3 pending)". Add the P2 commit hash placeholder to the Commits line (fill in post-merge).
- Modify: `docs/superpowers/specs/2026-04-20-msgspec-zmq-migration-overview.md:206` — update the Primitives row to "In progress (P1 landed in 41e53697e, P2 landed in <hash>; P3 pending)".

- [ ] **Step 1: Edit primitives spec header**

```markdown
**Status:** Partial (P1+P2 landed, P3 pending)
**Commits so far:** `41e53697e` (P1), `<P2 hash>` (P2)
```

- [ ] **Step 2: Edit overview status matrix**

```markdown
| Primitives (terminal) | In progress (P1 landed in 41e53697e, P2 landed in <hash>; P3 pending) | `msgspec-primitives-migration-design.md` | — |
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/
git commit -s -m "docs(specs): mark P2 landed; note P3 as remaining primitives work"
```

---

## Verification Checklist (read-only, post-commit)

- [ ] `rg -n 'AutoRoutedModel' src/ tests/` returns 0 hits.
- [ ] `rg -n 'class .*\(AIPerfBaseModel, AutoRoutedModel\)' src/ tests/` returns 0 hits.
- [ ] `rg -n 'discriminator_field' src/ tests/` returns 0 hits (outside specs/plans).
- [ ] `rg -n 'PydanticStructMixin' src/` returns exactly 3 hits: the class definition in `base_models.py`, its docstring, and its usage on `ErrorDetailsCount` in `error_models.py`.
- [ ] `uv run python -c "from aiperf.common.messages import Message; print(Message.__struct_tag_field__)"` prints `message_type`.
- [ ] `uv run python -c "from aiperf.common.messages import Message, HeartbeatMessage; import msgspec; print(msgspec.msgpack.Decoder(type=Message).decode(msgspec.msgpack.Encoder().encode(HeartbeatMessage(service_id='x', service_type='worker', state='running'))).__class__.__name__)"` prints `HeartbeatMessage`.
- [ ] `uv run pytest tests/unit/ -n auto` — 0 failed.
- [ ] `uv run pytest -m component_integration -n auto` — 0 failed.
- [ ] `uv run pytest -m integration -n auto` — 0 failed.
- [ ] `pre-commit run --all-files` — clean.
- [ ] Phase C commit is a single commit (not a merge of multiple partial flips).

---

## Rollback Strategy

If Phase C cannot land atomically (e.g., msgspec multi-Struct inheritance blocks `RequiresRequestNSMixin` in a way not caught by C1's smoke test and the flatten-fallback also fails for a reason unforeseen), abort by `git reset --hard <Phase B commit>`. Phase A (failing test) and Phase B (non-Message detach) stand alone and do not require Phase C to ship. The failing test in Phase A can be marked `@pytest.mark.xfail(reason="pending P2 flip")` until a retry.

Phases D and E are sequential after C; they do not require a standalone rollback because their commits are small and revertable with `git revert <hash>`.
