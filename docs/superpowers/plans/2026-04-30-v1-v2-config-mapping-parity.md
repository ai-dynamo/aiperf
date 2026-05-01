# v1/v2 Config Mapping Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit every v1 config option that still exists in `src/aiperf/config/v1/`, fix missing v1 -> v2 converter mappings, and cover each fix with focused regression tests.

**Architecture:** Keep the current converter decomposition: endpoint/model fields in `_converter_endpoint.py`, runtime/service fields in `_converter_runtime.py`, telemetry fields in `_converter_telemetry.py`, and full-object validation through `convert_user_to_aiperf()` in `converter.py`. Add parity tests beside the existing v1 converter tests rather than introducing a new framework.

**Tech Stack:** Python 3.10+, Pydantic v2 models, pytest with xdist (`uv run pytest ... -n auto`), AIPerf `UserConfig`/`ServiceConfig` v1 DTOs, canonical v2 `AIPerfConfig`.

---

## Files

- Modify: `src/aiperf/config/v1/_converter_endpoint.py` — add endpoint field mappings that have matching v2 `EndpointConfig` fields.
- Modify: `src/aiperf/config/v1/_converter_telemetry.py` — preserve v1 GPU telemetry mode/collector parsing semantics where the v2 runtime still consumes them.
- Modify: `tests/unit/config/v1/test_converter_endpoint_models.py` — endpoint parity regressions.
- Modify: `tests/unit/config/v1/test_converter_telemetry.py` — telemetry parity regressions.
- Modify: `tests/unit/config/v1/test_converter_full.py` — end-to-end v1 `UserConfig` + `ServiceConfig` to v2 `AIPerfConfig` regression.
- Read-only audit: `src/aiperf/config/v1/*.py`, `src/aiperf/config/{config.py,endpoint.py,artifacts.py,_models_runtime.py,zmq.py}`.

Do not create commits unless the user explicitly asks; this repository has active local changes.

---

### Task 1: Endpoint v1 -> v2 parity

**Files:**
- Modify: `tests/unit/config/v1/test_converter_endpoint_models.py`
- Modify: `src/aiperf/config/v1/_converter_endpoint.py`

- [ ] **Step 1: Add failing endpoint mapping tests**

Append these tests to `tests/unit/config/v1/test_converter_endpoint_models.py`:

```python
def test_build_endpoint_maps_ready_check_interval_and_mode():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["m"],
                "urls": ["http://x"],
                "ready_check_timeout": 30.0,
                "ready_check_interval": 2.5,
                "ready_check_mode": "both",
            },
        }
    )

    out = build_endpoint(user)

    assert out["ready_check_timeout"] == 30.0
    assert out["ready_check_interval"] == 2.5
    assert out["ready_check_mode"] == "both"


def test_build_endpoint_maps_video_request_options():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["m"],
                "urls": ["http://x"],
                "download_video_content": True,
                "request_content_type": "multipart/form-data",
                "type": "image_to_video",
            },
        }
    )

    out = build_endpoint(user)

    assert out["download_video_content"] is True
    assert str(out["request_content_type"]) == "multipart/form-data"
```

- [ ] **Step 2: Run the endpoint tests and verify failure**

Run:

```bash
uv run pytest tests/unit/config/v1/test_converter_endpoint_models.py -n auto
```

Expected before implementation: at least one assertion fails because `_ENDPOINT_FIELD_MAP` currently omits `ready_check_interval`, `ready_check_mode`, `download_video_content`, or `request_content_type`.

- [ ] **Step 3: Add the endpoint mappings**

Update `_ENDPOINT_FIELD_MAP` in `src/aiperf/config/v1/_converter_endpoint.py` so it includes every v1 endpoint field that has a matching v2 endpoint field:

```python
_ENDPOINT_FIELD_MAP: dict[str, str] = {
    "url_selection_strategy": "url_strategy",
    "type": "type",
    "streaming": "streaming",
    "custom_endpoint": "path",
    "api_key": "api_key",
    "timeout_seconds": "timeout",
    "ready_check_timeout": "ready_check_timeout",
    "ready_check_mode": "ready_check_mode",
    "ready_check_interval": "ready_check_interval",
    "transport": "transport",
    "use_legacy_max_tokens": "use_legacy_max_tokens",
    "use_server_token_count": "use_server_token_count",
    "connection_reuse_strategy": "connection_reuse",
    "download_video_content": "download_video_content",
    "request_content_type": "request_content_type",
}
```

- [ ] **Step 4: Run endpoint tests and verify pass**

Run:

```bash
uv run pytest tests/unit/config/v1/test_converter_endpoint_models.py -n auto
```

Expected after implementation: all tests in the file pass.

---

### Task 2: GPU telemetry v1 token parity

**Files:**
- Modify: `tests/unit/config/v1/test_converter_telemetry.py`
- Modify: `src/aiperf/config/v1/_converter_telemetry.py`

- [ ] **Step 1: Add failing telemetry token tests**

Append these tests to `tests/unit/config/v1/test_converter_telemetry.py`:

```python
def test_gpu_telemetry_pynvml_token_sets_collector_private_attr():
    user = UserConfig.model_validate({"gpu_telemetry": ["pynvml"]})

    out = build_gpu_telemetry(user)

    assert out["enabled"] is True
    assert user._gpu_telemetry_collector_type == "pynvml"
    assert out["urls"] == []


def test_gpu_telemetry_dashboard_token_sets_mode_private_attr():
    user = UserConfig.model_validate({"gpu_telemetry": ["dashboard", "node1:9400"]})

    out = build_gpu_telemetry(user)

    assert out["enabled"] is True
    assert user._gpu_telemetry_mode == "dashboard"
    assert out["urls"] == ["http://node1:9400"]
```

- [ ] **Step 2: Run telemetry tests and verify failure**

Run:

```bash
uv run pytest tests/unit/config/v1/test_converter_telemetry.py -n auto
```

Expected before implementation: at least one assertion fails because `build_gpu_telemetry()` currently ignores `pynvml` and `dashboard` tokens.

- [ ] **Step 3: Parse v1 telemetry magic tokens**

Update `build_gpu_telemetry()` in `src/aiperf/config/v1/_converter_telemetry.py` with explicit token parsing:

```python
def build_gpu_telemetry(user: UserConfig) -> dict[str, Any]:
    """Translate v1 ``--gpu-telemetry`` magic-list into the v2 telemetry dict."""
    from aiperf.common.enums import GPUTelemetryMode
    from aiperf.plugin.enums import GPUTelemetryCollectorType

    if user.no_gpu_telemetry:
        return {"enabled": False}
    if not user.gpu_telemetry:
        return {"enabled": True}
    urls: list[str] = []
    metrics_file: Path | None = None
    for item in user.gpu_telemetry:
        token = item.lower()
        if token == "pynvml":
            user._gpu_telemetry_collector_type = GPUTelemetryCollectorType.PYNVML
        elif token == "dashboard":
            user._gpu_telemetry_mode = GPUTelemetryMode.DASHBOARD
        elif item.endswith(".csv"):
            metrics_file = Path(item)
        elif item.startswith("http") or ":" in item:
            urls.append(_url(item))
    gpu_telemetry: dict[str, Any] = {"enabled": True, "urls": urls}
    if metrics_file is not None:
        gpu_telemetry["metrics_file"] = metrics_file
    return gpu_telemetry
```

If enum member names differ, inspect `aiperf.common.enums.GPUTelemetryMode` and `aiperf.plugin.enums.GPUTelemetryCollectorType` and use the existing member whose string value is `dashboard` / `pynvml`.

- [ ] **Step 4: Run telemetry tests and verify pass**

Run:

```bash
uv run pytest tests/unit/config/v1/test_converter_telemetry.py -n auto
```

Expected after implementation: all tests in the file pass.

---

### Task 3: Full conversion regression for fixed mappings

**Files:**
- Modify: `tests/unit/config/v1/test_converter_full.py`

- [ ] **Step 1: Add an end-to-end conversion test**

Append this test to `tests/unit/config/v1/test_converter_full.py`:

```python
def test_convert_user_to_aiperf_preserves_endpoint_parity_fields():
    user = UserConfig.model_validate(
        {
            "endpoint": {
                "model_names": ["video-model"],
                "urls": ["http://server:8000"],
                "type": "image_to_video",
                "ready_check_timeout": 30.0,
                "ready_check_interval": 2.5,
                "ready_check_mode": "both",
                "download_video_content": True,
                "request_content_type": "multipart/form-data",
            },
            "input": {"prompt": "make a test video"},
        }
    )
    service = ServiceConfig()

    config = convert_user_to_aiperf(user, service)

    assert config.endpoint.ready_check_timeout == 30.0
    assert config.endpoint.ready_check_interval == 2.5
    assert config.endpoint.ready_check_mode == "both"
    assert config.endpoint.download_video_content is True
    assert str(config.endpoint.request_content_type) == "multipart/form-data"
```

Ensure the file imports the needed names:

```python
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import convert_user_to_aiperf
```

- [ ] **Step 2: Run the full converter test file**

Run:

```bash
uv run pytest tests/unit/config/v1/test_converter_full.py -n auto
```

Expected: the new test passes with Task 1 implemented. If v2 validation rejects the endpoint type string, inspect `EndpointType` and use the image/video endpoint enum value that supports `multipart/form-data` in `EndpointConfig._validate_request_content_type()`.

---

### Task 4: Mechanical parity audit and classification

**Files:**
- Read: `src/aiperf/config/v1/*.py`
- Read: `src/aiperf/config/{config.py,endpoint.py,artifacts.py,_models_runtime.py,models.py,zmq.py}`
- Modify only if a same-shape field is found missing from a converter.

- [ ] **Step 1: Produce a local inventory from model fields**

Run this read-only command:

```bash
uv run python - <<'PY'
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._accuracy import AccuracyConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.config.v1._input import InputConfig
from aiperf.config.v1._loadgen import LoadGeneratorConfig
from aiperf.config.v1._output import OutputConfig
from aiperf.config.v1._tokenizer import TokenizerConfig

models = [
    UserConfig,
    ServiceConfig,
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    OutputConfig,
    TokenizerConfig,
    AccuracyConfig,
]
for model in models:
    print(f"[{model.__name__}]")
    for name in model.model_fields:
        print(name)
    print()
PY
```

Expected: printed field names for the top-level v1 DTOs and major nested DTOs.

- [ ] **Step 2: Classify each field**

Use the printed inventory and converter modules to classify fields as:

```text
converted: explicitly mapped into endpoint/models/datasets/phases/artifacts/runtime/logging/telemetry/accuracy/tokenizer/multi_run/slos/sweep
runtime-only: private v1 parse scratch fields or CLI-only generated fields like config_file/benchmark_id/cli_command
v2-default: intentionally omitted so v2 Pydantic defaults apply
unsupported: no v2 consumer exists and preserving it would be dead config
missing: v1 field has a v2 field or behavior but no converter/test coverage
```

Expected: no `missing` fields remain after Tasks 1-3. If any `missing` field remains, add a focused failing test in the relevant `tests/unit/config/v1/test_converter_*.py` file, implement the smallest converter mapping, and rerun that file with `-n auto`.

- [ ] **Step 3: Verify runtime/service parity remains covered**

Run:

```bash
uv run pytest tests/unit/config/v1/test_converter_runtime.py -n auto
```

Expected: pass. This confirms API host/port, UI/logging, workers, record processors, and ZMQ discriminator behavior still works.

---

### Task 5: Final focused verification

**Files:**
- Test-only verification.

- [ ] **Step 1: Run all v1 config tests**

Run:

```bash
uv run pytest tests/unit/config/v1 -n auto
```

Expected: all v1 config tests pass.

- [ ] **Step 2: Run broader config tests**

Run:

```bash
uv run pytest tests/unit/config -n auto
```

Expected: all config tests pass. If unrelated existing failures appear, isolate by rerunning the failing test file alone and report it as pre-existing only after confirming the new v1 converter tests pass.

- [ ] **Step 3: Run formatting/lint on touched files**

Run:

```bash
ruff format src/aiperf/config/v1/_converter_endpoint.py src/aiperf/config/v1/_converter_telemetry.py tests/unit/config/v1/test_converter_endpoint_models.py tests/unit/config/v1/test_converter_telemetry.py tests/unit/config/v1/test_converter_full.py && ruff check --fix src/aiperf/config/v1/_converter_endpoint.py src/aiperf/config/v1/_converter_telemetry.py tests/unit/config/v1/test_converter_endpoint_models.py tests/unit/config/v1/test_converter_telemetry.py tests/unit/config/v1/test_converter_full.py
```

Expected: ruff completes successfully. If ruff changes files, rerun `uv run pytest tests/unit/config/v1 -n auto`.

---

## Self-Review

- Spec coverage: The plan inventories v1 fields, fixes missing same-shape mappings, prioritizes endpoint readiness/video fields, telemetry magic tokens, runtime/API/ZMQ behavior, and sweep magic-list behavior via the existing converter tests.
- Placeholder scan: No TBD/TODO/fill-in-later instructions remain. The only conditional instruction is enum-name inspection if an existing enum member name differs from its string value.
- Type consistency: Tests use existing `UserConfig`, `ServiceConfig`, `build_endpoint`, `build_gpu_telemetry`, and `convert_user_to_aiperf` symbols. Commands use `uv run pytest ... -n auto` per repository preference.
