# Decouple Server Metrics & GPU Telemetry from RecordsManager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop routing GPU telemetry and server metrics records through RecordsManager. Each manager accumulates locally and publishes its own result message. Eliminate the ZMQ wire types for those records.

**Architecture:** Introduce two new plugin categories (`GPU_TELEMETRY_PROCESSOR`, `SERVER_METRICS_PROCESSOR`), move the existing accumulator + JSONL writer plugins into them, and make each manager load and drive its own processors. The collector callbacks already exist — we just fan out in-process instead of pushing over ZMQ. System controller's shutdown coordinator already waits on three independent result messages; it's agnostic to publisher identity.

**Tech Stack:** Python 3.10+ async, Pydantic v2, msgspec, ZMQ (for what remains), pytest.

---

## Pre-flight verification

- [ ] **Confirm shutdown coordination is publisher-agnostic**

Read `src/aiperf/controller/system_controller.py:1018-1090` — the three `@on_message` handlers for `PROCESS_RECORDS_RESULT`, `PROCESS_TELEMETRY_RESULT`, `PROCESS_SERVER_METRICS_RESULT` set their own latch flags (`_profile_results_received`, `_should_wait_for_telemetry`, `_should_wait_for_server_metrics`) and don't verify `service_id` matches RecordsManager. Changing publisher is transparent.

- [ ] **Confirm PROFILE_COMPLETE already reaches both side-channel managers**

`system_controller.py:968-983` (`_handle_profile_complete_relay`) already relays `CommandType.PROFILE_COMPLETE` to `ServiceType.GPU_TELEMETRY_MANAGER` and `ServiceType.SERVER_METRICS_MANAGER`. Both managers already have `@on_command(CommandType.PROFILE_COMPLETE)` handlers that do a final scrape (`gpu_telemetry/manager.py:378`, `server_metrics/manager.py:291`). That's our publish trigger.

---

## Task 1: Introduce new plugin categories

**Goal:** Add `GPU_TELEMETRY_PROCESSOR` and `SERVER_METRICS_PROCESSOR` as first-class plugin categories so each manager owns its own plugin set, and records_manager iterates only request-record processors.

**Files:**
- Modify: `src/aiperf/plugin/enums.py` (add `PluginType.GPU_TELEMETRY_PROCESSOR`, `PluginType.SERVER_METRICS_PROCESSOR`, plus `GPUTelemetryProcessorType` and `ServerMetricsProcessorType` extensible enums mirroring `ResultsProcessorType`)
- Modify: `src/aiperf/plugin/plugins.yaml` (move 4 entries: `gpu_telemetry_accumulator`, `gpu_telemetry_jsonl_writer`, `server_metrics_accumulator`, `server_metrics_jsonl_writer` out of `results_processor` into the two new categories)
- Modify: `src/aiperf/plugin/plugin_schema.py` or wherever plugin category schema is declared — add the two categories so `make validate-plugin-schemas` passes
- Generated: `src/aiperf/plugin/enums.pyi` via `make generate-all-plugin-files`

- [ ] **Step 1.1: Locate plugin category declarations**

Run:
```bash
grep -rn "RESULTS_PROCESSOR\|results_processor" src/aiperf/plugin/ | head -20
```
This should surface the PluginType enum, the per-category enum file, and the YAML schema validator. Read the file containing `class PluginType` and the file containing `class ResultsProcessorType` to mirror the pattern.

- [ ] **Step 1.2: Add PluginType entries**

In the PluginType enum (likely `src/aiperf/plugin/enums.py`):
```python
GPU_TELEMETRY_PROCESSOR = "gpu_telemetry_processor"
SERVER_METRICS_PROCESSOR = "server_metrics_processor"
```

- [ ] **Step 1.3: Add per-category name enums**

Create `GPUTelemetryProcessorType` and `ServerMetricsProcessorType` mirroring how `ResultsProcessorType` is declared (it's an `ExtensibleStrEnum`). Initial members:
```python
class GPUTelemetryProcessorType(ExtensibleStrEnum):
    ACCUMULATOR = "gpu_telemetry_accumulator"
    JSONL_WRITER = "gpu_telemetry_jsonl_writer"

class ServerMetricsProcessorType(ExtensibleStrEnum):
    ACCUMULATOR = "server_metrics_accumulator"
    JSONL_WRITER = "server_metrics_jsonl_writer"
```

- [ ] **Step 1.4: Register category→enum mapping**

Find where `ResultsProcessorType` is registered with `PluginType.RESULTS_PROCESSOR` (grep for `register_plugin_type` or `plugins.register_category` or similar). Register the two new pairs the same way.

- [ ] **Step 1.5: Update plugin YAML schema**

Find the JSON schema or validator that enumerates legal top-level categories. Add `gpu_telemetry_processor` and `server_metrics_processor` as valid top-level keys. Run `make validate-plugin-schemas` to confirm no schema error.

- [ ] **Step 1.6: Move plugin YAML entries**

In `src/aiperf/plugin/plugins.yaml`, remove these four entries from `results_processor`:
- `gpu_telemetry_accumulator` (lines ~708-712)
- `gpu_telemetry_jsonl_writer` (lines ~714-719)
- `server_metrics_accumulator` (lines ~740-744)
- `server_metrics_jsonl_writer` (lines ~746-750)

Add new top-level sections:
```yaml
gpu_telemetry_processor:
  gpu_telemetry_accumulator:
    class: aiperf.gpu_telemetry.accumulator:GPUTelemetryAccumulator
    description: |
      GPU telemetry accumulator that aggregates GPU telemetry records and computes
      metrics in a hierarchical structure. Loaded when telemetry is enabled.

  gpu_telemetry_jsonl_writer:
    class: aiperf.gpu_telemetry.jsonl_writer:GPUTelemetryJSONLWriter
    description: |
      GPU telemetry JSONL writer that exports per-record GPU telemetry data to
      JSONL files as it arrives from GPUTelemetryManager. Enabled with telemetry
      export config.

server_metrics_processor:
  server_metrics_accumulator:
    class: aiperf.server_metrics.accumulator:ServerMetricsAccumulator
    description: |
      Server metrics accumulator that aggregates Prometheus server metrics records
      and computes summary statistics. Supports Gauge, Counter, and Histogram metrics.

  server_metrics_jsonl_writer:
    class: aiperf.server_metrics.jsonl_writer:ServerMetricsJSONLWriter
    description: |
      Server metrics JSONL writer that exports per-record server metrics data to
      JSONL files in slim format.
```

- [ ] **Step 1.7: Regenerate artifacts**

```bash
make generate-all-plugin-files
make validate-plugin-schemas
```
Expected: clean exit, `src/aiperf/plugin/enums.pyi` now lists the new PluginType members and the new per-category enums.

- [ ] **Step 1.8: Commit**

```bash
git add -A
git commit -s -m "feat(plugins): add GPU_TELEMETRY_PROCESSOR and SERVER_METRICS_PROCESSOR categories"
```

---

## Task 2: Wire local processor loading in GPUTelemetryManager

**Goal:** GPUTelemetryManager loads its own GPU telemetry processors, fans each collector callback to every processor in-process, tracks its own errors, and publishes `ProcessTelemetryResultMessage` on `PROFILE_COMPLETE`. ZMQ push path still present — we remove it in Task 4.

**Files:**
- Modify: `src/aiperf/gpu_telemetry/manager.py`
- Reference: `src/aiperf/records/records_manager.py:170-207` (plugin loading pattern), `:400-421` (telemetry fan-out), `:814-862` (export + publish pattern)

- [ ] **Step 2.1: Add imports + state**

In `src/aiperf/gpu_telemetry/manager.py`, add imports near the top:
```python
from collections import defaultdict
from dataclasses import dataclass, field

from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages import ProcessTelemetryResultMessage
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ProcessTelemetryResult,
)
from aiperf.gpu_telemetry.protocols import (
    GPUTelemetryAccumulatorProtocol,
    GPUTelemetryProcessorProtocol,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, GPUTelemetryProcessorType
```

- [ ] **Step 2.2: Create processors in `__init__`**

After `super().__init__`, add:
```python
self._processors: list[GPUTelemetryProcessorProtocol] = []
self._accumulator: GPUTelemetryAccumulatorProtocol | None = None
self._error_counts: dict[ErrorDetails, int] = defaultdict(int)

for entry in plugins.iter_entries(PluginType.GPU_TELEMETRY_PROCESSOR):
    try:
        ProcessorClass = plugins.get_class(
            PluginType.GPU_TELEMETRY_PROCESSOR, entry.name
        )
        processor = ProcessorClass(
            service_id=self.service_id,
            run=self.run,
            pub_client=self.pub_client,
        )
        self.attach_child_lifecycle(processor)
        self._processors.append(processor)
        if entry.name == GPUTelemetryProcessorType.ACCUMULATOR:
            self._accumulator = processor
        self.debug(
            f"Created GPU telemetry processor: {entry.name}: "
            f"{processor.__class__.__name__}"
        )
    except PostProcessorDisabled:
        self.debug(
            f"GPU telemetry processor {entry.name} is disabled and will not be used"
        )
    except Exception as e:
        self.error(f"Failed to create GPU telemetry processor {entry.name}: {e}")
```

Delete `self.records_push_client = self.comms.create_push_client(...)` — gone.

- [ ] **Step 2.3: Replace `_on_telemetry_records` callback body**

Replace the existing `_on_telemetry_records` (currently pushes `TelemetryRecordsWireMessage`) with a direct fan-out:
```python
async def _on_telemetry_records(
    self, records: list[TelemetryRecord], collector_id: str
) -> None:
    """Fan out telemetry records to all loaded GPU telemetry processors."""
    if not records or not self._processors:
        return

    errors = await asyncio.gather(
        *[
            processor.process_telemetry_record(record)
            for processor in self._processors
            for record in records
        ],
        return_exceptions=True,
    )
    for error in errors:
        if isinstance(error, BaseException):
            self.exception(f"Failed to process telemetry record: {error!r}")
            self._error_counts[ErrorDetails.from_exception(error)] += 1
```

- [ ] **Step 2.4: Replace `_on_telemetry_error` callback body**

```python
async def _on_telemetry_error(self, error: ErrorDetails, collector_id: str) -> None:
    """Track collector-level telemetry errors locally."""
    self._error_counts[error] += 1
```

- [ ] **Step 2.5: Publish result message at PROFILE_COMPLETE**

Extend `_handle_profile_complete_command` to publish after the final scrape completes:
```python
@on_command(CommandType.PROFILE_COMPLETE)
async def _handle_profile_complete_command(self, message: Command) -> None:
    if self._collectors:
        self.info("GPU Telemetry: Profiling complete, capturing final metrics...")
        for dcgm_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
                self.debug(f"GPU Telemetry: Captured final state from {dcgm_url}")
            except Exception as e:
                self.warning(
                    f"GPU Telemetry: Failed to capture final state from {dcgm_url}: {e}"
                )
        await self._stop_all_collectors()

    # Parse start_ns / end_ns from PROFILE_COMPLETE command payload (sent by records_manager).
    start_ns: int | None = None
    if message.payload:
        try:
            parsed = orjson.loads(message.payload)
            start_ns = parsed.get("start_ns")
        except Exception:
            pass

    await self._publish_telemetry_result(start_ns=start_ns)

async def _publish_telemetry_result(self, start_ns: int | None) -> None:
    error_summary = [
        ErrorDetailsCount(error_details=err, count=count)
        for err, count in self._error_counts.items()
    ]
    if not self._accumulator:
        await self.publish(
            ProcessTelemetryResultMessage(
                service_id=self.service_id,
                telemetry_result=ProcessTelemetryResult(results=None),
            )
        )
        return
    export_data = self._accumulator.export_results(
        start_ns=start_ns or 0,
        error_summary=error_summary,
    )
    await self.publish(
        ProcessTelemetryResultMessage(
            service_id=self.service_id,
            telemetry_result=ProcessTelemetryResult(results=export_data),
        )
    )
```

**Note on `start_ns`:** RecordsManager currently computes `start_ns` from `_records_tracker.get_results_time_window()`. The GPU telemetry manager doesn't have that info locally. Plumb it via the `PROFILE_COMPLETE` command payload — extend the request in records_manager (`records_manager.py:354-360`) to pass `start_ns` in the command payload (Task 4.2). If `start_ns` is `None`, accumulator semantics already treat `0` / `None` as "include all data" (see `records_manager.py:838` comment).

- [ ] **Step 2.6: Move `START_REALTIME_TELEMETRY` command handler into this manager**

Add:
```python
@on_command(CommandType.START_REALTIME_TELEMETRY)
async def _on_start_realtime_telemetry_command(self, message: Command) -> None:
    if self._accumulator:
        self._accumulator.start_realtime_telemetry()
    else:
        self.error(
            "GPU telemetry accumulator not found, cannot start realtime telemetry"
        )
```

- [ ] **Step 2.7: Commit**

```bash
git add src/aiperf/gpu_telemetry/manager.py
git commit -s -m "feat(gpu-telemetry): accumulate and publish results locally"
```

---

## Task 3: Wire local processor loading in ServerMetricsManager

**Goal:** Same shape as Task 2, for server metrics.

**Files:**
- Modify: `src/aiperf/server_metrics/manager.py`
- Reference: `src/aiperf/records/records_manager.py:423-443` and `:864-921`

- [ ] **Step 3.1: Add imports + state**

Mirror Task 2.1 using:
```python
from aiperf.common.messages import ProcessServerMetricsResultMessage
from aiperf.common.models import ProcessServerMetricsResult
from aiperf.server_metrics.protocols import (
    ServerMetricsAccumulatorProtocol,
    ServerMetricsProcessorProtocol,
)
from aiperf.plugin.enums import PluginType, ServerMetricsProcessorType
```

- [ ] **Step 3.2: Load processors in `__init__`**

Mirror Task 2.2 but iterate `PluginType.SERVER_METRICS_PROCESSOR` and match `ServerMetricsProcessorType.ACCUMULATOR`. Delete `self.records_push_client`.

- [ ] **Step 3.3: Replace `_on_server_metrics_records`**

```python
async def _on_server_metrics_records(
    self, records: list[ServerMetricsRecord], collector_id: str
) -> None:
    if not records or not self._processors:
        return
    for record in records:
        errors = await asyncio.gather(
            *[
                processor.process_server_metrics_record(record)
                for processor in self._processors
            ],
            return_exceptions=True,
        )
        for error in errors:
            if isinstance(error, BaseException):
                self.exception(f"Failed to process server metrics record: {error!r}")
                self._error_counts[ErrorDetails.from_exception(error)] += 1
```

- [ ] **Step 3.4: Replace `_on_server_metrics_error`**

```python
async def _on_server_metrics_error(
    self, error: ErrorDetails, collector_id: str
) -> None:
    self._error_counts[error] += 1
```

- [ ] **Step 3.5: Publish result message at PROFILE_COMPLETE**

At the end of `_handle_profile_complete_command`, parse `start_ns` / `end_ns` from the command payload and call `self._accumulator.export_results(...)`, then publish `ProcessServerMetricsResultMessage`:

```python
start_ns: int | None = None
end_ns: int | None = None
if message.payload:
    try:
        parsed = orjson.loads(message.payload)
        start_ns = parsed.get("start_ns")
        end_ns = parsed.get("end_ns")
    except Exception:
        pass

await self._publish_server_metrics_result(start_ns=start_ns, end_ns=end_ns)

async def _publish_server_metrics_result(
    self, start_ns: int | None, end_ns: int | None
) -> None:
    error_summary = [
        ErrorDetailsCount(error_details=err, count=count)
        for err, count in self._error_counts.items()
    ]
    if not self._accumulator:
        await self.publish(
            ProcessServerMetricsResultMessage(
                service_id=self.service_id,
                server_metrics_result=ProcessServerMetricsResult(
                    results=None, error_summary=error_summary
                ),
            )
        )
        return
    export_data = await self._accumulator.export_results(
        start_ns=start_ns or time.time_ns(),
        end_ns=end_ns or time.time_ns(),
        error_summary=error_summary,
    )
    await self.publish(
        ProcessServerMetricsResultMessage(
            service_id=self.service_id,
            server_metrics_result=ProcessServerMetricsResult(
                results=export_data, error_summary=error_summary
            ),
        )
    )
```

- [ ] **Step 3.6: Commit**

```bash
git add src/aiperf/server_metrics/manager.py
git commit -s -m "feat(server-metrics): accumulate and publish results locally"
```

---

## Task 4: Remove side-channel paths from RecordsManager + delete ZMQ wire types

**Goal:** Everything related to telemetry/server-metrics records leaves RecordsManager and the wire protocol. Build must stay green.

**Files:**
- Modify: `src/aiperf/records/records_manager.py` — delete large swaths
- Modify: `src/aiperf/common/metric_records_wire.py` — delete `TelemetryRecordsWireMessage`, `ServerMetricsRecordWireMessage`
- Modify: `src/aiperf/common/channel_codecs.py` — remove those types from the union
- Modify: `src/aiperf/common/enums/enums.py` — remove `MessageType.TELEMETRY_RECORDS`, `MessageType.SERVER_METRICS_RECORD` (confirm no other consumers)
- Modify: `src/aiperf/records/records_manager.py` imports (strip newly-unused)

- [ ] **Step 4.1: Delete RecordsManager side-channel handlers**

Remove from `records_manager.py`:
- lines 246-269: `@on_pull_message(MessageType.TELEMETRY_RECORDS)` + `_on_telemetry_records`
- lines 270-297: `@on_pull_message(MessageType.SERVER_METRICS_RECORD)` + `_on_server_metrics_records`
- lines 400-443: `_send_telemetry_to_results_processors`, `_send_server_metrics_to_results_processors`
- lines 576-589: `@on_command(CommandType.START_REALTIME_TELEMETRY)` (moved to gpu_telemetry/manager.py in Task 2.6)
- lines 791-809 inside `_process_results`: the `_publish_telemetry_results` / `_publish_server_metrics_results` calls
- lines 814-924: `_process_telemetry_results`, `_publish_telemetry_results`, `_process_server_metrics_results`, `_publish_server_metrics_results`
- In `__init__`: `_telemetry_state`, `_server_metrics_state`, `_gpu_telemetry_processors`, `_server_metrics_processors`, `_gpu_telemetry_accumulator`, `_server_metrics_accumulator` (lines 159-167)
- In `__init__`: the two `isinstance(..., GPUTelemetryProcessorProtocol)` and `ServerMetricsProcessorProtocol` branches of the plugin loop (lines 182-197); leave only the `_metric_results_processors.append(results_processor)` case.

- [ ] **Step 4.2: Update RecordsManager PROFILE_COMPLETE command payload**

In `_finalize_and_process_results` (`records_manager.py:354-360`), change the `Command` construction to include `start_ns`/`end_ns` so side-channel managers can emit their result messages with the correct time window:
```python
start_ns, end_ns = self._records_tracker.get_results_time_window()
response = await self.control_client.request(
    Command(
        cid=uuid.uuid4().hex,
        cmd=CommandType.PROFILE_COMPLETE,
        payload=orjson.dumps(
            {"start_ns": start_ns, "end_ns": end_ns}
        ).decode(),
    ),
    timeout=10.0,
)
```

- [ ] **Step 4.3: Delete wire messages**

In `src/aiperf/common/metric_records_wire.py`, delete `TelemetryRecordsWireMessage` (line 202) and `ServerMetricsRecordWireMessage` (line 224). Delete the `_error_to_wire`/`wire_error_to_domain_error` helpers only if no other callers remain (check: records_manager still uses `wire_error_to_domain_error` on line 218 for the non-side-channel path — keep it, but trace whether the module still has remaining users via grep).

```bash
grep -rn "TelemetryRecordsWireMessage\|ServerMetricsRecordWireMessage\|_error_to_wire" src/aiperf/ tests/
```
Expected: no matches in `src/aiperf/`. If tests still reference them, delete those tests (Task 5).

- [ ] **Step 4.4: Update channel codec**

In `src/aiperf/common/channel_codecs.py:12-26`, remove `TelemetryRecordsWireMessage` / `ServerMetricsRecordWireMessage` from imports and the codec union.

- [ ] **Step 4.5: Remove MessageType enum values**

In `src/aiperf/common/enums/enums.py`, confirm no other references to `MessageType.TELEMETRY_RECORDS` / `MessageType.SERVER_METRICS_RECORD`:
```bash
grep -rn "TELEMETRY_RECORDS\|SERVER_METRICS_RECORD" src/aiperf/ tests/
```
Remove both enum members if clean.

- [ ] **Step 4.6: Verify build**

```bash
ruff format . && ruff check --fix .
uv run python -c "import aiperf"
```
Expected: no ImportError, no ruff errors.

- [ ] **Step 4.7: Commit**

```bash
git add -A
git commit -s -m "refactor(records): remove telemetry and server-metrics record routing"
```

---

## Task 5: Tests

**Files:**
- Delete or update: `tests/unit/records/test_records_manager.py` — any test that pushes `TelemetryRecordsWireMessage` / `ServerMetricsRecordWireMessage` or asserts on `_telemetry_state` / `_server_metrics_state`
- Create: `tests/unit/gpu_telemetry/test_manager_local_accumulation.py`
- Create: `tests/unit/server_metrics/test_manager_local_accumulation.py`
- Update: any component-integration test that verifies the full PROFILE_COMPLETE → results flow; target is: driving a fake collector must still result in `ProcessTelemetryResultMessage` landing at the controller

- [ ] **Step 5.1: Find existing records_manager tests for side-channel paths**

```bash
grep -rln "TELEMETRY_RECORDS\|SERVER_METRICS_RECORD\|_on_telemetry_records\|_on_server_metrics_records\|_gpu_telemetry_accumulator\|_server_metrics_accumulator" tests/
```
For each hit, delete if test exclusively exercises the deleted path, or trim if it also covers kept behavior.

- [ ] **Step 5.2: Write a failing test — GPUTelemetryManager accumulates and publishes**

Create `tests/unit/gpu_telemetry/test_manager_local_accumulation.py`:

```python
import pytest

from aiperf.common.enums import CommandType, MessageType
from aiperf.common.messages import ProcessTelemetryResultMessage
from aiperf.common.models import TelemetryRecord
from aiperf.gpu_telemetry.manager import GPUTelemetryManager


@pytest.mark.asyncio
async def test_telemetry_manager_publishes_result_on_profile_complete(
    telemetry_manager: GPUTelemetryManager,
    fake_telemetry_records: list[TelemetryRecord],
    publish_sink,
):
    await telemetry_manager._on_telemetry_records(
        fake_telemetry_records, collector_id="collector_0"
    )

    await telemetry_manager._handle_profile_complete_command(
        make_command(
            CommandType.PROFILE_COMPLETE,
            payload={"start_ns": 0, "end_ns": None},
        )
    )

    published = publish_sink.messages_of_type(MessageType.PROCESS_TELEMETRY_RESULT)
    assert len(published) == 1
    assert isinstance(published[0], ProcessTelemetryResultMessage)
    assert published[0].telemetry_result.results is not None
    assert published[0].telemetry_result.results.endpoints
```

Use existing fixtures where possible — grep `tests/unit/gpu_telemetry/` for `conftest.py` and a `telemetry_manager` fixture; if absent, assemble one that constructs `GPUTelemetryManager` with a minimal `BenchmarkRun` and a stub `pub_client` that records published messages.

- [ ] **Step 5.3: Run it**

```bash
uv run pytest tests/unit/gpu_telemetry/test_manager_local_accumulation.py -v
```
Expected: PASS (Task 2 made it work).

- [ ] **Step 5.4: Mirror for server_metrics**

Create `tests/unit/server_metrics/test_manager_local_accumulation.py` with the analogous test using a fake Prometheus snapshot.

- [ ] **Step 5.5: Run full unit suite**

```bash
uv run pytest tests/unit/ -n auto
```
Expected: all pass. Fix any now-obsolete records_manager tests by deleting or rescoping.

- [ ] **Step 5.6: Run component-integration**

```bash
uv run pytest -m component_integration -n auto
```
Expected: all pass. A failure here most likely means the PROFILE_COMPLETE payload shape change broke someone — inspect and adjust.

- [ ] **Step 5.7: Run integration**

```bash
uv run pytest -m integration -n auto
```
Expected: all pass.

- [ ] **Step 5.8: Commit**

```bash
git add tests/
git commit -s -m "test: cover local accumulation in gpu_telemetry and server_metrics managers"
```

---

## Task 6: Docs + final sweep

**Files:**
- Modify: `docs/architecture.md` — update the data flow section that shows records manager receiving telemetry/server-metrics
- Verify: CLAUDE.md three-file-sync — this change doesn't touch standards, so probably no sync needed; confirm

- [ ] **Step 6.1: Update architecture diagram/prose**

Locate the section in `docs/architecture.md` that mentions telemetry/server metrics flowing to records_manager. Update the mermaid diagram and prose to show:
- `GPUTelemetryManager` with local accumulator → `ProcessTelemetryResultMessage` → controller
- `ServerMetricsManager` with local accumulator → `ProcessServerMetricsResultMessage` → controller
- `RecordsManager` receives only request records

- [ ] **Step 6.2: Grep for stale references**

```bash
grep -rn "TelemetryRecordsWireMessage\|ServerMetricsRecordWireMessage\|RecordsManager.*telemetry\|RecordsManager.*server metrics" docs/
```
Fix any stale doc references.

- [ ] **Step 6.3: Run pre-commit on all files**

```bash
pre-commit run --all-files
```
Expected: all hooks pass.

- [ ] **Step 6.4: Commit**

```bash
git add docs/
git commit -s -m "docs(architecture): record local accumulation for side-channel metrics"
```
