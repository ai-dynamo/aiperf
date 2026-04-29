# Restore UserConfig/ServiceConfig as CLI-Only Input Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore `UserConfig` / `ServiceConfig` (from `origin/main`) as the CLI's input layer with **all validators stripped**, route everything through a single `convert_user_to_aiperf()` boundary, keep `AIPerfConfig` as the sole validation gate, and mechanically prevent v1 types from leaking into runtime code via a TID251 import ban.

**Architecture:** The v1 DTOs live in a quarantined package `src/aiperf/config/v1/` and only carry `Field` + `CLIParameter` annotations — no `@field_validator` / `@model_validator`. cyclopts walks `UserConfig` to synthesize CLI flags exactly as it did on `origin/main`. After parse, `convert_user_to_aiperf(user_config, service_config)` (sibling of v1 DTOs) maps into the existing `AIPerfConfig` schema, reusing the section-builder logic that today lives in `_cli_sections.py` / `_cli_dataset.py` (rerouted to read from the nested v1 shape). Anywhere downstream of `cli_commands/`, only `AIPerfConfig` / `BenchmarkPlan` / `BenchmarkRun` flow — never `UserConfig`. A ruff TID251 ban makes this fence machine-checked.

**Tech Stack:** Python 3.10+, Pydantic v2, cyclopts, ruff (TID251), pytest (-n auto).

**Hard rules (encoded in `src/aiperf/config/v1/__init__.py` docstring):**
1. New CLI flags that fit an existing v1 nested class (`EndpointConfig`, `InputConfig`, `LoadGeneratorConfig`, `OutputConfig`, `TokenizerConfig`, `AccuracyConfig`) — add the field there.
2. New CLI flags that don't fit any existing nested class — add as a **top-level field on `UserConfig`** (no new nested classes, ever).
3. Validators on v1 classes are forbidden. Validation lives on `AIPerfConfig`.
4. Converter is the **only** module that reads v1 attributes outside `cli_commands/`.

---

## File Structure

**New files:**
- `src/aiperf/config/v1/__init__.py` — package init + `__all__` + the four hard rules in module docstring.
- `src/aiperf/config/v1/_endpoint.py` — `EndpointConfig` (15 fields, no validators).
- `src/aiperf/config/v1/_input.py` — `InputConfig` + nested `ConversationConfig`, `TurnConfig`, `TurnDelayConfig`, `PromptConfig`, `InputTokensConfig`, `OutputTokensConfig`, `PrefixPromptConfig`, `ImageConfig`, `ImageHeightConfig`, `ImageWidthConfig`, `AudioConfig`, `AudioLengthConfig`, `VideoConfig`, `VideoAudioConfig`, `RankingsConfig`, `RankingsPassagesConfig`, `RankingsQueryConfig`, `SynthesisConfig`. (Multiple files acceptable if line budget pressures it; group by domain if split.)
- `src/aiperf/config/v1/_loadgen.py` — `LoadGeneratorConfig` (35 fields, no validators).
- `src/aiperf/config/v1/_output.py` — `OutputConfig` (no validators).
- `src/aiperf/config/v1/_tokenizer.py` — `TokenizerConfig` (no validators).
- `src/aiperf/config/v1/_accuracy.py` — `AccuracyConfig` (no validators).
- `src/aiperf/config/v1/_zmq.py` — `ZMQTCPConfig`, `ZMQIPCConfig`, `ZMQDualBindConfig` + proxy variants (no validators).
- `src/aiperf/config/v1/_workers.py` — `WorkersConfig` (no validators).
- `src/aiperf/config/v1/user_config.py` — `UserConfig` (no validators; flat top-level fields permitted).
- `src/aiperf/config/v1/service_config.py` — `ServiceConfig` (no validators).
- `src/aiperf/config/v1/converter.py` — `convert_user_to_aiperf(user, service)` entrypoint + section-builders (`build_endpoint`, `build_models`, `build_profiling`, `build_warmup`, `build_dataset`, `build_artifacts`, `build_gpu_telemetry`, `build_server_metrics`, `build_logging_runtime`, `build_tokenizer`, `build_accuracy`, `build_multi_run`). Split into `_converter_<section>.py` files if line budget pressure.
- `tests/unit/config/v1/test_v1_user_config_no_validators.py` — assert no `*Validator` decorators on v1 classes.
- `tests/unit/config/v1/test_convert_user_to_aiperf_*.py` — golden tests for each section-builder.
- `tests/unit/config/v1/test_backwards_compat_regression.py` — known v1 CLI invocation strings → expected `AIPerfConfig` shape.

**Modified files:**
- `src/aiperf/cli_commands/profile.py` — `cli_model: CLIModel` → `user_config: UserConfig, service_config: ServiceConfig | None`.
- `src/aiperf/cli_commands/config_cli.py` — same swap.
- `src/aiperf/cli_commands/kube/generate.py`, `kube/profile.py`, `kube/sweep.py`, `kube/_kube_common.py` — same swap; the helper `resolve_config(cli_model, ...)` becomes `resolve_config(user_config, service_config, ...)`.
- `src/aiperf/config/__init__.py` — export `UserConfig`, `ServiceConfig`, `convert_user_to_aiperf` from `aiperf.config.v1`. Remove `CLIModel` re-export.
- `pyproject.toml` — add `flake8-tidy-imports.banned-api` entry banning `aiperf.config.v1.*` outside the allowlist; add `[tool.ruff.lint.per-file-ignores]` allowlist for `src/aiperf/cli_commands/**` and `src/aiperf/config/v1/**`.
- `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc` — add a "Config v1 (CLI input layer)" section per the three-file sync rule.

**Deleted files:**
- `src/aiperf/config/cli_model.py`
- `src/aiperf/config/cli_converter.py`
- `src/aiperf/config/_cli_sections.py`
- `src/aiperf/config/_cli_dataset.py`

---

## Phase 1 — Restore v1 DTOs (validator-free)

The DTOs are *structurally* the v1 shape from `origin/main` but every validator is removed. CLIParameter annotations are preserved verbatim from the existing `cli_model.py` so cyclopts produces the same flag set.

### Task 1: v1 package skeleton + UserConfig/ServiceConfig top-level

**Files:**
- Create: `src/aiperf/config/v1/__init__.py`
- Create: `src/aiperf/config/v1/user_config.py`
- Create: `src/aiperf/config/v1/service_config.py`
- Test: `tests/unit/config/v1/test_v1_user_config_no_validators.py`

- [ ] **Step 1: Write failing test** (`test_v1_user_config_no_validators.py`)

```python
import inspect
from aiperf.config.v1 import UserConfig, ServiceConfig

def test_user_config_has_no_validators():
    decorators = [
        m for m in inspect.getmembers(UserConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not decorators, f"UserConfig must have NO validators (found: {decorators})"

def test_service_config_has_no_validators():
    decorators = [
        m for m in inspect.getmembers(ServiceConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not decorators, f"ServiceConfig must have NO validators (found: {decorators})"

def test_user_config_imports_from_v1_package():
    from aiperf.config.v1.user_config import UserConfig as UC
    assert UC is UserConfig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_v1_user_config_no_validators.py -v -n auto`
Expected: FAIL — `aiperf.config.v1` does not exist.

- [ ] **Step 3: Create `src/aiperf/config/v1/__init__.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config v1 — CLI-only input layer.

UserConfig and ServiceConfig are the cyclopts-facing input DTOs. They carry CLI
flag annotations and Pydantic field metadata, but NO validators — AIPerfConfig
is the single validation gate.

Hard rules (enforced by code review + the TID251 ban in pyproject.toml):

1. New CLI flags that fit an existing v1 nested class (EndpointConfig,
   InputConfig, LoadGeneratorConfig, OutputConfig, TokenizerConfig,
   AccuracyConfig) — add the field there.
2. New CLI flags that don't fit any existing nested class — add as a top-level
   field on UserConfig itself. NEVER add new nested classes to v1.
3. NO validators on v1 classes. Validation lives on AIPerfConfig.
4. The converter (aiperf.config.v1.converter) is the only module outside
   cli_commands/ that may read v1 attributes.

Anywhere downstream of cli_commands/, only AIPerfConfig / BenchmarkPlan /
BenchmarkRun flow.
"""

from aiperf.config.v1.service_config import ServiceConfig
from aiperf.config.v1.user_config import UserConfig

__all__ = ["ServiceConfig", "UserConfig"]
```

- [ ] **Step 4: Restore `user_config.py` and `service_config.py` skeletons**

Pull the class shells from `origin/main`:

```bash
git show origin/main:src/aiperf/common/config/user_config.py > /tmp/origin_user_config.py
git show origin/main:src/aiperf/common/config/service_config.py > /tmp/origin_service_config.py
```

Hand-port the class declarations to `src/aiperf/config/v1/{user_config,service_config}.py`. Field types referencing nested classes (`endpoint: EndpointConfig`, etc.) get stub forward-references — the nested classes are filled in by Tasks 2-7 below. **Strip every `@model_validator`, `@field_validator`, and any custom `__init__` / private helper method that exists only to feed a validator.** Keep `Field(...)` and `Annotated[..., CLIParameter(...)]` annotations as-is.

**Use forward-reference STRING types** for all nested-class fields so Tasks 2–6 only need to *create* their nested-class file (no re-edit of `user_config.py`/`service_config.py`). This unlocks parallel execution of Tasks 2–6.

```python
# user_config.py
from __future__ import annotations
from typing import TYPE_CHECKING
from pydantic import Field
from aiperf.config.base import BaseConfig

if TYPE_CHECKING:
    from aiperf.config.v1._endpoint import EndpointConfig
    from aiperf.config.v1._input import InputConfig
    from aiperf.config.v1._loadgen import LoadGeneratorConfig
    from aiperf.config.v1._output import OutputConfig
    from aiperf.config.v1._tokenizer import TokenizerConfig
    from aiperf.config.v1._accuracy import AccuracyConfig

class UserConfig(BaseConfig):
    """v1 user-facing CLI input. CLI-only. Validators forbidden."""
    endpoint: "EndpointConfig | None" = Field(default=None, description="Endpoint config")
    input: "InputConfig | None" = Field(default=None, description="Input config")
    output: "OutputConfig | None" = Field(default=None, description="Output config")
    tokenizer: "TokenizerConfig | None" = Field(default=None, description="Tokenizer config")
    loadgen: "LoadGeneratorConfig | None" = Field(default=None, description="Load gen config")
    accuracy: "AccuracyConfig | None" = Field(default=None, description="Accuracy config")
    # + top-level fields preserved from origin/main: cli_command, benchmark_id,
    #   gpu_telemetry, no_gpu_telemetry, server_metrics, no_server_metrics,
    #   server_metrics_formats. Pull these from `git show origin/main:...user_config.py`.
```

Same pattern for `service_config.py` (forward-refs to `ZMQTCPConfig`, `ZMQIPCConfig`, `ZMQDualBindConfig`, `WorkersConfig`).

`__init__.py` calls `UserConfig.model_rebuild()` and `ServiceConfig.model_rebuild()` after importing the nested classes (which are added by Tasks 2–6). For Task 1's standalone test, only the no-validators check runs — instantiating UserConfig with nested-class data isn't required until later tasks supply those classes.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_v1_user_config_no_validators.py -v -n auto`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/v1/ tests/unit/config/v1/
git commit -s -m "feat(config-v1): introduce v1 package skeleton and no-validator contract"
```

---

### Task 2: Restore EndpointConfig

**Files:**
- Create: `src/aiperf/config/v1/_endpoint.py`
- Modify: `src/aiperf/config/v1/user_config.py` — replace `endpoint: Any` with `EndpointConfig`
- Test: `tests/unit/config/v1/test_v1_endpoint_config.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig

def test_endpoint_config_round_trip():
    uc = UserConfig.model_validate({
        "endpoint": {"model_names": ["x"], "url": ["http://localhost:8000"]},
    })
    assert isinstance(uc.endpoint, EndpointConfig)
    assert uc.endpoint.model_names == ["x"]

def test_endpoint_config_has_no_validators():
    import inspect
    bad = [
        m for m in inspect.getmembers(EndpointConfig)
        if hasattr(m[1], "__pydantic_decorator_info__")
    ]
    assert not bad, bad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_v1_endpoint_config.py -v -n auto`
Expected: FAIL — `_endpoint` module missing.

- [ ] **Step 3: Restore `EndpointConfig` from `origin/main`**

Source: `git show origin/main:src/aiperf/common/config/endpoint_config.py`. Copy the class definition; **delete every validator method**. Preserve all `CLIParameter` annotations and `Field(...)` calls. Update imports for the new location.

- [ ] **Step 4: Wire into UserConfig**

Replace `endpoint: Any = None` with `endpoint: EndpointConfig = Field(default_factory=EndpointConfig, description="Endpoint configuration")` in `user_config.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/ -v -n auto`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/v1/_endpoint.py src/aiperf/config/v1/user_config.py tests/unit/config/v1/
git commit -s -m "feat(config-v1): restore EndpointConfig (no validators)"
```

---

### Task 3: Restore InputConfig + nested input children

**Files:**
- Create: `src/aiperf/config/v1/_input.py` (or split: `_input.py`, `_input_prompt.py`, `_input_media.py` if line budget pressures it; ergonomics line budget is 500)
- Modify: `src/aiperf/config/v1/user_config.py`
- Test: `tests/unit/config/v1/test_v1_input_config.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._input import InputConfig, ConversationConfig, PromptConfig, ImageConfig

def test_input_config_nested_round_trip():
    uc = UserConfig.model_validate({
        "input": {
            "conversation": {"num_turns": 3},
            "prompt": {"input_tokens": {"mean": 128, "stddev": 16}},
            "image": {"width": {"mean": 1024}, "height": {"mean": 768}},
        }
    })
    assert isinstance(uc.input, InputConfig)
    assert uc.input.conversation.num_turns == 3
    assert uc.input.prompt.input_tokens.mean == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_v1_input_config.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Restore from `origin/main`**

Sources: `conversation_config.py`, `prompt_config.py`, `image_config.py`, `audio_config.py`, `video_config.py`, `rankings_config.py`, `synthesis_config.py`, `input_config.py`. Concatenate into `_input.py` (or split as noted above), strip every validator. Preserve CLIParameter annotations.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_v1_input_config.py -v -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/_input*.py src/aiperf/config/v1/user_config.py tests/unit/config/v1/
git commit -s -m "feat(config-v1): restore InputConfig and nested children (no validators)"
```

---

### Task 4: Restore LoadGeneratorConfig

**Files:**
- Create: `src/aiperf/config/v1/_loadgen.py`
- Modify: `src/aiperf/config/v1/user_config.py`
- Test: `tests/unit/config/v1/test_v1_loadgen_config.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._loadgen import LoadGeneratorConfig

def test_loadgen_config_carries_warmup_fields():
    uc = UserConfig.model_validate({
        "loadgen": {
            "concurrency": 100,
            "request_count": 1000,
            "warmup_concurrency": 10,
            "warmup_request_count": 50,
        },
    })
    assert isinstance(uc.loadgen, LoadGeneratorConfig)
    assert uc.loadgen.concurrency == 100
    assert uc.loadgen.warmup_concurrency == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_v1_loadgen_config.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Restore `LoadGeneratorConfig` from `origin/main`**

Source: `git show origin/main:src/aiperf/common/config/loadgen_config.py`. Strip every validator (notably `validate_timing_mode`, `validate_must_have_stop_condition`, ramp normalizers — these all move to `AIPerfConfig` indirectly). Preserve all 35 fields including the `warmup_*` set, ramp fields, `num_users`, `user_centric_rate`, `num_profile_runs`, convergence fields, `request_cancellation_*`, `fixed_schedule*`. Preserve CLIParameter annotations.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_v1_loadgen_config.py -v -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/_loadgen.py src/aiperf/config/v1/user_config.py tests/unit/config/v1/
git commit -s -m "feat(config-v1): restore LoadGeneratorConfig (no validators)"
```

---

### Task 5: Restore OutputConfig + TokenizerConfig + AccuracyConfig

**Files:**
- Create: `src/aiperf/config/v1/_output.py`, `_tokenizer.py`, `_accuracy.py`
- Modify: `src/aiperf/config/v1/user_config.py`
- Test: `tests/unit/config/v1/test_v1_small_configs.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._output import OutputConfig
from aiperf.config.v1._tokenizer import TokenizerConfig
from aiperf.config.v1._accuracy import AccuracyConfig

def test_small_configs_round_trip():
    uc = UserConfig.model_validate({
        "output": {"artifact_directory": "/tmp/x"},
        "tokenizer": {"name": "gpt2"},
        "accuracy": {"benchmark_type": "lm_eval"},
    })
    assert isinstance(uc.output, OutputConfig)
    assert isinstance(uc.tokenizer, TokenizerConfig)
    assert isinstance(uc.accuracy, AccuracyConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_v1_small_configs.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Restore each from `origin/main`**

Sources: `output_config.py`, `tokenizer_config.py`, `accuracy_config.py`. Strip validators, preserve CLIParameter annotations.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_v1_small_configs.py -v -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/_output.py src/aiperf/config/v1/_tokenizer.py src/aiperf/config/v1/_accuracy.py src/aiperf/config/v1/user_config.py tests/unit/config/v1/
git commit -s -m "feat(config-v1): restore OutputConfig, TokenizerConfig, AccuracyConfig"
```

---

### Task 6: Restore ZMQ + Workers configs (ServiceConfig children)

**Files:**
- Create: `src/aiperf/config/v1/_zmq.py`, `_workers.py`
- Modify: `src/aiperf/config/v1/service_config.py`
- Test: `tests/unit/config/v1/test_v1_service_config.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import ServiceConfig
from aiperf.config.v1._zmq import ZMQTCPConfig, ZMQIPCConfig
from aiperf.config.v1._workers import WorkersConfig

def test_service_config_round_trip():
    sc = ServiceConfig.model_validate({
        "zmq_tcp": {"host": "127.0.0.1", "port_min": 50000, "port_max": 50100},
        "workers": {"max": 8},
        "log_level": "INFO",
    })
    assert isinstance(sc.zmq_tcp, ZMQTCPConfig)
    assert isinstance(sc.workers, WorkersConfig)
    assert sc.log_level == "INFO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_v1_service_config.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Restore from `origin/main`**

Sources: `zmq_config.py`, `worker_config.py`, `service_config.py`. Strip the 4 ServiceConfig validators (`validate_log_level_from_verbose_flags`, `validate_ui_type`, `validate_comm_config`, `validate_api_host_requires_port`) — their behaviors move to the converter (Phase 2, Task 14). Preserve CLIParameter annotations.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_v1_service_config.py -v -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/_zmq.py src/aiperf/config/v1/_workers.py src/aiperf/config/v1/service_config.py tests/unit/config/v1/
git commit -s -m "feat(config-v1): restore ServiceConfig + zmq/workers children"
```

---

### Task 7: cyclopts smoke test — UserConfig produces the same flag set as CLIModel

**Files:**
- Test: `tests/unit/config/v1/test_v1_cyclopts_flag_parity.py`

- [ ] **Step 1: Write the test**

The test enumerates flags produced by cyclopts when given `UserConfig` + `ServiceConfig` and compares the set against flags produced when given `CLIModel`. Use cyclopts' internal flag-discovery utility, or invoke `App` and dump its parser. Allowed differences: only flags that are documented as v2-only (e.g. anything related to `phases` lists, `datasets` lists, sweep, multi-run-via-yaml).

```python
import cyclopts
from aiperf.config.cli_model import CLIModel
from aiperf.config.v1 import UserConfig, ServiceConfig

def _flags_for(*models):
    app = cyclopts.App()
    @app.command
    def f(*models): pass
    # Use cyclopts introspection — adjust to actual cyclopts API in this repo.
    return _collect_flags(app)

def test_user_config_flag_parity_with_cli_model():
    v1_flags = _flags_for(UserConfig, ServiceConfig)
    flat_flags = _flags_for(CLIModel)
    missing = flat_flags - v1_flags
    assert not missing, f"Flags lost in v1 restoration: {missing}"
```

(Implementation note: if cyclopts doesn't expose flag enumeration cleanly, dispatch a minimal cyclopts app and capture `--help` output via `capsys`.)

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/unit/config/v1/test_v1_cyclopts_flag_parity.py -v -n auto`
Expected: PASS (or FAIL with a list of missing flags — fix Tasks 2-6 to add them).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/v1/test_v1_cyclopts_flag_parity.py
git commit -s -m "test(config-v1): cyclopts flag parity between UserConfig and CLIModel"
```

---

## Phase 2 — Build the converter

Each section-builder gets a golden test, then the implementation reroutes from `cli: BaseModel (CLIModel)` to `(user_config: UserConfig, service_config: ServiceConfig)`. The behavior is preserved; the inputs are restructured.

### Task 8: Converter entrypoint skeleton

**Files:**
- Create: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_entrypoint.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig, ServiceConfig
from aiperf.config.v1.converter import convert_user_to_aiperf
from aiperf.config.config import AIPerfConfig

def test_minimal_convert_returns_aiperf_config():
    user = UserConfig.model_validate({
        "endpoint": {"model_names": ["m"], "url": ["http://x"]},
        "loadgen": {"concurrency": 1, "request_count": 1},
    })
    service = ServiceConfig()
    result = convert_user_to_aiperf(user, service)
    assert isinstance(result, AIPerfConfig)
    assert result.endpoint.urls == ["http://x"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_converter_entrypoint.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Implement minimal entrypoint that delegates to current `build_aiperf_config`**

Adapter shim: build a one-off `CLIModel` instance from the v1 inputs (flatten back), call `build_aiperf_config`, return result. This is throwaway — it lets the test pass while later tasks reroute each section properly. Mark it `# noqa: temporary shim — replaced in Tasks 9-19`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_converter_entrypoint.py -v -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/converter.py tests/unit/config/v1/test_converter_entrypoint.py
git commit -s -m "feat(config-v1): converter entrypoint with adapter shim"
```

---

### Task 9: Reroute `build_endpoint`

**Files:**
- Modify: `src/aiperf/config/v1/converter.py` — add `build_endpoint(user, s_endpoint)`
- Test: `tests/unit/config/v1/test_converter_endpoint.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_endpoint

def test_build_endpoint_maps_urls_and_model_strategy():
    user = UserConfig.model_validate({
        "endpoint": {
            "model_names": ["llama-3"],
            "url": ["localhost:8000"],
            "endpoint_type": "chat",
            "streaming": True,
            "extra_inputs": ["temperature:0.7"],
        }
    })
    out = build_endpoint(user)
    assert out["urls"] == ["http://localhost:8000"]
    assert out["type"] == "chat"
    assert out["streaming"] is True
    assert out["extra"] == {"temperature": 0.7}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_converter_endpoint.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Port logic from `_cli_sections.build_endpoint`**

Read from `user.endpoint.<field>` instead of `cli.<field>`. Preserve URL normalization, template extraction, extra-inputs parsing.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/config/v1/test_converter_endpoint.py -v -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/converter.py tests/unit/config/v1/test_converter_endpoint.py
git commit -s -m "feat(config-v1): reroute build_endpoint to read from UserConfig.endpoint"
```

---

### Task 10: Reroute `build_models`

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_models.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_models

def test_build_models_maps_names_and_strategy():
    user = UserConfig.model_validate({
        "endpoint": {"model_names": ["a", "b"], "model_selection_strategy": "random"}
    })
    out = build_models(user)
    assert [m["name"] for m in out["names"]] == ["a", "b"]
    assert out["strategy"] == "random"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/config/v1/test_converter_models.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Port logic from `_cli_sections.build_models`**

Source reads `user.endpoint.model_names` and `user.endpoint.model_selection_strategy` (both fields live on `EndpointConfig` per Task 2).

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_models"
```

---

### Task 11: Reroute `build_profiling` (the timing-mode discriminator)

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_profiling.py`

The `_profiling_phase_type()` helper in current `_cli_sections.py` replicates v1's `validate_timing_mode` decision tree (request_rate → REQUEST_RATE, fixed_schedule → FIXED_SCHEDULE, user_centric_rate → USER_CENTRIC_RATE, else CONCURRENCY). This is the most logic-heavy reroute.

- [ ] **Step 1: Write failing tests** (one per timing-mode branch)

```python
import pytest
from pytest import param
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_profiling

@pytest.mark.parametrize("loadgen,expected_type", [
    param({"concurrency": 100, "request_count": 1000}, "concurrency", id="concurrency"),
    param({"request_rate": 100.0, "request_count": 1000}, "poisson", id="request-rate-default-poisson"),
    param({"request_rate": 100.0, "arrival_pattern": "gamma", "arrival_smoothness": 2.0, "request_count": 1000}, "gamma", id="request-rate-gamma"),
    param({"user_centric_rate": 5.0, "num_users": 50, "request_count": 1000}, "user_centric", id="user-centric"),
    param({"fixed_schedule": True, "fixed_schedule_auto_offset": True}, "fixed_schedule", id="fixed-schedule"),
])  # fmt: skip
def test_build_profiling_picks_phase_type(loadgen, expected_type):
    user = UserConfig.model_validate({"loadgen": loadgen})
    out = build_profiling(user)
    assert out["type"] == expected_type
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/config/v1/test_converter_profiling.py -v -n auto`
Expected: FAIL.

- [ ] **Step 3: Port `build_profiling` + `_profiling_phase_type` from `_cli_sections.py`**

Read fields from `user.loadgen.<field>`. Preserve ramp normalization, request_cancellation wiring.

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_profiling with timing-mode discriminator"
```

---

### Task 12: Reroute `build_warmup`

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_warmup.py`

- [ ] **Step 1: Write failing test**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_warmup

def test_build_warmup_returns_none_when_no_warmup_fields():
    user = UserConfig.model_validate({"loadgen": {"concurrency": 1, "request_count": 1}})
    assert build_warmup(user) is None

def test_build_warmup_packages_warmup_fields_as_phase():
    user = UserConfig.model_validate({"loadgen": {
        "warmup_concurrency": 10,
        "warmup_request_count": 50,
    }})
    out = build_warmup(user)
    assert out["type"] == "concurrency"
    assert out["concurrency"] == 10
    assert out["requests"] == 50
    assert out["exclude_from_results"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL.

- [ ] **Step 3: Port `build_warmup` from `_cli_sections.py`**

Read warmup fields from `user.loadgen.warmup_*`. Strip the leading "warmup_" prefix when assembling the phase. Force `exclude_from_results=True`.

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_warmup"
```

---

### Task 13: Reroute `build_dataset` (synthetic / file / public / synthesis)

**Files:**
- Modify: `src/aiperf/config/v1/converter.py` (or a sibling `_converter_dataset.py` if line budget pressures)
- Test: `tests/unit/config/v1/test_converter_dataset.py`

This is the second-most-complex reroute. The current `_cli_dataset.py` (433 lines) reads ~70 dataset-related CLI fields. Most live in `user.input` on v1 — verify each.

- [ ] **Step 1: Write failing tests for each dataset shape**

```python
import pytest
from pytest import param
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_dataset

@pytest.mark.parametrize("input_cfg,expected_type", [
    param({"prompt": {"input_tokens": {"mean": 128}}}, "synthetic", id="synthetic-default"),
    param({"file": "/tmp/data.jsonl"}, "file", id="file"),
    param({"public_dataset": "sharegpt"}, "public", id="public"),
    param({"file": "/tmp/data.jsonl", "synthesis": {"strategy": "augment"}}, "composed", id="composed"),
])  # fmt: skip
def test_build_dataset_picks_correct_type(input_cfg, expected_type):
    user = UserConfig.model_validate({"input": input_cfg})
    out = build_dataset(user)
    assert out["type"] == expected_type

def test_build_dataset_public_uses_dataset_field_not_name():
    user = UserConfig.model_validate({"input": {"public_dataset": "sharegpt"}})
    out = build_dataset(user)
    assert out["dataset"] == "sharegpt"
    # name is the dataset's identity in the list, NOT the source — that's `dataset`.
    # The wrapping in `[{"name": "main", **out}]` happens in convert_user_to_aiperf.
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Port `_cli_dataset.build_dataset` to read from `user.input`**

Check every field touched by current `build_dataset` — most are on `user.input` directly or under `user.input.{conversation,prompt,image,audio,video,rankings,synthesis}`. The `random_seed` and `dataset_sampling_strategy` may be top-level on `user.input` or on `user.loadgen` — verify against `origin/main`'s field locations.

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_dataset across synthetic/file/public/composed shapes"
```

---

### Task 14: Reroute `build_artifacts` + `build_logging_runtime` (ServiceConfig integration)

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_artifacts.py`, `test_converter_logging_runtime.py`

**Notable:** the stripped ServiceConfig validators (`validate_log_level_from_verbose_flags`, `validate_ui_type`, `validate_comm_config`, `validate_api_host_requires_port`) get re-applied **here** as plain code. `build_logging_runtime(user, service)` returns `(logging_dict, runtime_dict)`.

- [ ] **Step 1: Write failing tests**

```python
from aiperf.config.v1 import UserConfig, ServiceConfig
from aiperf.config.v1.converter import build_logging_runtime, build_artifacts

def test_logging_runtime_verbose_flag_promotes_to_debug():
    user = UserConfig()
    service = ServiceConfig.model_validate({"verbose": True})
    log, _runtime = build_logging_runtime(user, service)
    assert log["level"] == "DEBUG"

def test_logging_runtime_zmq_choice_picks_communication_type():
    user = UserConfig()
    service = ServiceConfig.model_validate({"zmq_tcp": {"host": "127.0.0.1"}})
    _log, runtime = build_logging_runtime(user, service)
    assert runtime["communication"]["type"] == "tcp"

def test_artifacts_combines_user_output_with_user_top_level():
    user = UserConfig.model_validate({
        "output": {"artifact_directory": "/tmp/a", "export_level": "summary"},
        "cli_command": "aiperf profile ...",
        "benchmark_id": "abc-123",
    })
    out = build_artifacts(user)
    assert out["dir"] == "/tmp/a"
    assert out["summary"] is True
    assert out["records"] is False
    assert out["cli_command"] == "aiperf profile ..."
    assert out["benchmark_id"] == "abc-123"
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Port `build_artifacts` + `build_logging_runtime`**

`build_logging_runtime` now takes `(user, service)`. It applies the four old ServiceConfig validators inline:
- `verbose=True` → `level="DEBUG"`; `extra_verbose=True` → `level="TRACE"`.
- `ui_type` — pass through as-is to `runtime["ui"]`; if absent, default `auto`.
- `zmq_tcp` set → `communication={"type": "tcp", ...}`; else `zmq_ipc` → `ipc`; else `dual_bind`.
- `api_host` set without `api_port` → raise (or, since validators forbidden on v1, raise here when assembling the dict — this is the converter's job).

`build_artifacts` reads from `user.output` (legacy nested) and merges in top-level `user.cli_command` / `user.benchmark_id` (from origin/main UserConfig top-level fields). Translates `export_level` enum → individual booleans.

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_artifacts and build_logging_runtime; fold stripped ServiceConfig validators into converter"
```

---

### Task 15: Reroute `build_gpu_telemetry` + `build_server_metrics`

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_telemetry.py`

These read from top-level UserConfig fields (`user.gpu_telemetry`, `user.no_gpu_telemetry`, `user.server_metrics`, `user.no_server_metrics`, `user.server_metrics_formats`). All are list-of-strings with magic tokens (`pynvml`/`dashboard`/URLs).

- [ ] **Step 1: Write failing tests**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_gpu_telemetry, build_server_metrics

def test_gpu_telemetry_parses_pynvml_token():
    user = UserConfig.model_validate({"gpu_telemetry": ["pynvml"]})
    out = build_gpu_telemetry(user)
    assert out["enabled"] is True
    assert "pynvml" in out.get("_mode", "")  # adjust to actual schema

def test_no_server_metrics_disables_collection():
    user = UserConfig.model_validate({"no_server_metrics": True})
    out = build_server_metrics(user)
    assert out["enabled"] is False

def test_server_metrics_formats_passed_through():
    user = UserConfig.model_validate({"server_metrics_formats": ["prometheus", "json"]})
    out = build_server_metrics(user)
    assert out["formats"] == ["prometheus", "json"]
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Port from `_cli_sections.build_gpu_telemetry` / `build_server_metrics`**

Read from `user.<top_level_field>`.

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_gpu_telemetry and build_server_metrics"
```

---

### Task 16: Reroute `build_tokenizer`, `build_accuracy`, `build_multi_run`

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_optionals.py`

- [ ] **Step 1: Write failing tests**

```python
from aiperf.config.v1 import UserConfig
from aiperf.config.v1.converter import build_tokenizer, build_accuracy, build_multi_run

def test_build_tokenizer_passthrough():
    user = UserConfig.model_validate({"tokenizer": {"name": "gpt2", "revision": "main"}})
    out = build_tokenizer(user)
    assert out == {"name": "gpt2", "revision": "main"}

def test_build_accuracy_passthrough():
    user = UserConfig.model_validate({"accuracy": {"benchmark_type": "lm_eval"}})
    out = build_accuracy(user)
    assert out["benchmark_type"] == "lm_eval"

def test_build_multi_run_pulls_from_loadgen():
    user = UserConfig.model_validate({"loadgen": {
        "num_profile_runs": 3,
        "confidence_level": 0.95,
        "convergence_metric": "ttft",
    }})
    out = build_multi_run(user)
    assert out["num_profile_runs"] == 3
    assert out["confidence_level"] == 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Port `build_tokenizer`, `build_accuracy`, `build_multi_run`**

`build_tokenizer` reads from `user.tokenizer`; `build_accuracy` from `user.accuracy`; `build_multi_run` from `user.loadgen.{num_profile_runs, profile_run_cooldown_seconds, confidence_level, set_consistent_seed, convergence_*, profile_run_disable_warmup_after_first}`.

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): reroute build_tokenizer, build_accuracy, build_multi_run"
```

---

### Task 17: Wire `convert_user_to_aiperf` to use the rerouted builders (drop adapter shim)

**Files:**
- Modify: `src/aiperf/config/v1/converter.py`
- Test: `tests/unit/config/v1/test_converter_full.py`

- [ ] **Step 1: Write a failing end-to-end test**

```python
from aiperf.config.v1 import UserConfig, ServiceConfig
from aiperf.config.v1.converter import convert_user_to_aiperf

def test_full_conversion_concurrency_run():
    user = UserConfig.model_validate({
        "endpoint": {"model_names": ["llama"], "url": ["http://localhost:8000"]},
        "loadgen": {"concurrency": 100, "request_count": 1000},
        "output": {"artifact_directory": "/tmp/x"},
        "cli_command": "aiperf profile --model llama --concurrency 100 --request-count 1000",
        "benchmark_id": "test-1",
    })
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    assert cfg.endpoint.urls == ["http://localhost:8000"]
    assert len(cfg.phases) == 1
    assert cfg.phases[0].name == "profiling"
    assert cfg.phases[0].type == "concurrency"
    assert cfg.phases[0].concurrency == 100
    assert cfg.datasets[0].name == "main"
    assert cfg.artifacts.benchmark_id == "test-1"

def test_full_conversion_with_warmup_phase():
    user = UserConfig.model_validate({
        "endpoint": {"model_names": ["m"], "url": ["http://x"]},
        "loadgen": {"concurrency": 10, "request_count": 100, "warmup_concurrency": 2, "warmup_request_count": 10},
    })
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)
    names = [p.name for p in cfg.phases]
    assert names == ["warmup", "profiling"]
    assert cfg.phases[0].exclude_from_results is True
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL or PASS-via-shim (depending on shim accuracy).

- [ ] **Step 3: Replace the shim with direct builder composition**

Mirror `cli_converter.build_aiperf_config`'s assembly: call rerouted builders, assemble the nested dict (`endpoint`, `models`, `phases`, `datasets`, `artifacts`, `gpu_telemetry`, `server_metrics`, optional `logging`/`runtime`/`tokenizer`/`accuracy`/`multi_run`/`random_seed`/`slos`), then `AIPerfConfig(**nested)`.

Delete the adapter-shim. The converter now reads exclusively from `UserConfig` / `ServiceConfig`.

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "feat(config-v1): drop converter shim; full UserConfig→AIPerfConfig path live"
```

---

## Phase 3 — CLI cutover

### Task 18: Migrate `cli_commands/profile.py`

**Files:**
- Modify: `src/aiperf/cli_commands/profile.py`
- Test: `tests/unit/cli_commands/test_profile_cli_v1_signature.py` (new) + existing `tests/unit/cli_commands/`

- [ ] **Step 1: Write failing test**

```python
import inspect
from aiperf.cli_commands.profile import profile
from aiperf.config.v1 import UserConfig, ServiceConfig

def test_profile_cli_takes_user_and_service_config():
    sig = inspect.signature(profile)
    annots = {p.name: p.annotation for p in sig.parameters.values()}
    assert annots["user_config"] is UserConfig
    assert annots["service_config"] in (ServiceConfig, ServiceConfig | None)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL.

- [ ] **Step 3: Update `profile.py`**

```python
# src/aiperf/cli_commands/profile.py
from aiperf.config.v1 import UserConfig, ServiceConfig
from aiperf.config.v1.converter import convert_user_to_aiperf

def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    if service_config is None:
        service_config = ServiceConfig()
    aiperf_config = convert_user_to_aiperf(user_config, service_config)
    plan = BenchmarkPlan.from_config(aiperf_config)
    cli_runner.run_benchmark(plan)
```

- [ ] **Step 4: Run unit suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -s -am "refactor(cli): profile() takes UserConfig+ServiceConfig and routes via v1 converter"
```

---

### Task 19: Migrate `cli_commands/config_cli.py`, `kube/{generate,profile,sweep}.py`, `kube/_kube_common.py`

**Files:**
- Modify: `src/aiperf/cli_commands/config_cli.py`, `kube/generate.py`, `kube/profile.py`, `kube/sweep.py`, `kube/_kube_common.py`

- [ ] **Step 1: For each file, swap signature**

```python
# Before
def resolve_config(cli_model: CLIModel, config_file: Path | None) -> AIPerfConfig:
    return ... build_aiperf_config(cli_model)

# After
def resolve_config(
    user_config: UserConfig,
    service_config: ServiceConfig,
    config_file: Path | None,
) -> AIPerfConfig:
    return ... convert_user_to_aiperf(user_config, service_config)
```

- [ ] **Step 2: Update all call-sites** in the same files.

- [ ] **Step 3: Run unit + component_integration**

Run: `uv run pytest tests/unit/ -n auto`
Then: `uv run pytest -m component_integration -n auto`
Expected: both green.

- [ ] **Step 4: Commit**

```bash
git commit -s -am "refactor(cli-kube): swap CLIModel for UserConfig+ServiceConfig across kube CLI commands"
```

---

### Task 20: Update `aiperf.config` package exports

**Files:**
- Modify: `src/aiperf/config/__init__.py`

- [ ] **Step 1: Replace re-exports**

```python
# Remove
from aiperf.config.cli_model import CLIModel  # delete

# Add
from aiperf.config.v1 import UserConfig, ServiceConfig
from aiperf.config.v1.converter import convert_user_to_aiperf
```

Update `__all__`.

- [ ] **Step 2: Run unit suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: green (any consumers of `aiperf.config.CLIModel` should already be migrated by Tasks 18-19).

- [ ] **Step 3: Commit**

```bash
git commit -s -am "refactor(config): export UserConfig/ServiceConfig/converter from aiperf.config package"
```

---

## Phase 4 — Strip dead code

### Task 21: Delete `cli_model.py`, `cli_converter.py`, `_cli_sections.py`, `_cli_dataset.py`

**Files:**
- Delete: `src/aiperf/config/cli_model.py`, `cli_converter.py`, `_cli_sections.py`, `_cli_dataset.py`
- Delete: corresponding tests under `tests/unit/config/` (any that reference `CLIModel` or `build_aiperf_config` directly — cyclopts-flag tests should be re-pointed at v1, not deleted).

- [ ] **Step 1: Grep for stragglers**

```bash
grep -rn "CLIModel\|build_aiperf_config\|cli_model\|_cli_sections\|_cli_dataset" src/aiperf tests
```

Expected: zero hits in `src/`, possibly some in `tests/` referring to deleted test files (resolve by deleting those tests).

- [ ] **Step 2: Delete the four files**

```bash
git rm src/aiperf/config/cli_model.py src/aiperf/config/cli_converter.py src/aiperf/config/_cli_sections.py src/aiperf/config/_cli_dataset.py
```

- [ ] **Step 3: Run full unit + component_integration suites**

Run: `uv run pytest tests/unit/ -n auto`
Then: `uv run pytest -m component_integration -n auto`
Expected: both green.

- [ ] **Step 4: Commit**

```bash
git commit -s -am "refactor(config): delete CLIModel and flat-converter (replaced by v1 layer)"
```

---

## Phase 5 — Enforce isolation

### Task 22: TID251 ban on `aiperf.config.v1` imports outside CLI / converter

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add TID rule + per-file allowlist**

```toml
[tool.ruff.lint]
# ... existing select ...
select = [
    "E", "F", "UP", "B", "SIM", "I",
    "TID",  # flake8-tidy-imports — enforce config-v1 isolation
]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"aiperf.config.v1".msg = "Config v1 is the CLI input layer only. Downstream code must use AIPerfConfig / BenchmarkPlan / BenchmarkRun. Allowed only in cli_commands/** and config/v1/**."
"aiperf.config.v1.user_config".msg = "..."
"aiperf.config.v1.service_config".msg = "..."

[tool.ruff.lint.per-file-ignores]
"src/aiperf/cli_commands/**" = ["TID251"]
"src/aiperf/config/v1/**" = ["TID251"]
"tests/unit/config/v1/**" = ["TID251"]
"tests/unit/cli_commands/**" = ["TID251"]
```

- [ ] **Step 2: Run ruff to verify zero violations**

Run: `uv run ruff check src/ tests/ --select TID251`
Expected: zero violations (because no production code imports v1 — the only consumers are the allowlisted dirs).

- [ ] **Step 3: Add a deliberate violation to verify the ban fires**

In a temporary file (e.g. `src/aiperf/common/test_violation.py`), add `from aiperf.config.v1 import UserConfig`. Run ruff. Expected: TID251 violation. Delete the temp file.

- [ ] **Step 4: Commit**

```bash
git commit -s -am "build(ruff): TID251 ban on aiperf.config.v1 outside CLI + converter"
```

---

### Task 23: Three-file CLAUDE.md sync — document v1 layer

**Files:**
- Modify: `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`

- [ ] **Step 1: Add identical "Config v1 (CLI input layer)" section to all three**

Add a section at the top level (before "Pre-Commit Checklist"):

```markdown
## Config v1 (CLI input layer)

`UserConfig` / `ServiceConfig` (`src/aiperf/config/v1/`) are the cyclopts-facing
CLI input DTOs. They carry CLI flag annotations and Pydantic field metadata,
but **NO validators** — `AIPerfConfig` is the single validation gate.

The converter (`src/aiperf/config/v1/converter.py`) is the only allowed v1→v2
boundary. Downstream of `cli_commands/`, only `AIPerfConfig` / `BenchmarkPlan`
/ `BenchmarkRun` flow. Enforced by ruff TID251.

Hard rules for adding new CLI flags:
1. Fits an existing v1 nested class (Endpoint/Input/LoadGen/Output/Tokenizer/
   Accuracy)? Add the field there.
2. Doesn't fit? Add as top-level field on `UserConfig`. NEVER add new nested
   classes to v1.
3. NO validators on v1 classes — ever.
```

- [ ] **Step 2: Diff the three files to confirm sync**

```bash
diff <(grep -A 20 "## Config v1" CLAUDE.md) <(grep -A 20 "## Config v1" .github/copilot-instructions.md)
diff <(grep -A 20 "## Config v1" CLAUDE.md) <(grep -A 20 "## Config v1" .cursor/rules/python.mdc)
```

Expected: no diffs.

- [ ] **Step 3: Commit**

```bash
git commit -s -am "docs(claude.md): document config v1 CLI input layer (3-file sync)"
```

---

## Phase 6 — Backwards-compat regression suite

### Task 24: Backwards-compat golden tests for representative v1 CLI invocations

**Files:**
- Test: `tests/unit/config/v1/test_backwards_compat_regression.py`

Pick 6-8 representative CLI invocations from `origin/main`'s tutorial / examples / test fixtures. Each test invokes the cyclopts parser on the flag string, then runs `convert_user_to_aiperf`, and asserts the resulting `AIPerfConfig` matches an expected hand-written shape.

- [ ] **Step 1: Identify representative invocations**

Grep `origin/main` for example CLI invocations:

```bash
git show origin/main:README.md | grep -A 1 "aiperf profile" | head -20
git show origin/main:docs/tutorials/ | grep -A 1 "aiperf profile" | head
git ls-tree -r origin/main | grep "tutorials/.*\.md" | xargs -I{} git show origin/main:{} | grep -A 1 "aiperf profile"
```

Pick 6-8 covering: concurrency, request_rate, fixed_schedule, user_centric, public_dataset, file dataset, synthesis, warmup phase.

- [ ] **Step 2: Write tests**

```python
import pytest
from pytest import param
from cyclopts import App
from aiperf.config.v1 import UserConfig, ServiceConfig
from aiperf.config.v1.converter import convert_user_to_aiperf

# Helper: parse a CLI string into (user_config, service_config) using the same
# cyclopts setup that profile() uses.
def _parse_cli(args: list[str]) -> tuple[UserConfig, ServiceConfig]:
    ...  # see existing cli.py wiring

@pytest.mark.parametrize("args,expected", [
    param(
        ["aiperf", "profile", "--model", "llama", "-u", "http://localhost:8000",
         "--concurrency", "100", "--request-count", "1000"],
        {"phases[0].type": "concurrency", "phases[0].concurrency": 100, "phases[0].requests": 1000},
        id="concurrency-basic",
    ),
    param(
        ["aiperf", "profile", "--model", "llama", "-u", "http://x",
         "--request-rate", "50", "--benchmark-duration", "30"],
        {"phases[0].type": "poisson", "phases[0].duration": 30},
        id="request-rate-poisson",
    ),
    # ... 4-6 more
])  # fmt: skip
def test_v1_cli_invocation_produces_expected_aiperf_config(args, expected):
    user, service = _parse_cli(args)
    cfg = convert_user_to_aiperf(user, service)
    for path, value in expected.items():
        assert _resolve(cfg, path) == value, f"{path}: expected {value}, got {_resolve(cfg, path)}"
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/config/v1/test_backwards_compat_regression.py -v -n auto`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git commit -s -am "test(config-v1): backwards-compat regression suite for representative CLI invocations"
```

---

### Task 25: Final shipping gate

**Files:** none — verification only.

- [ ] **Step 1: Run full unit suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all green.

- [ ] **Step 2: Run component_integration suite**

Run: `uv run pytest -m component_integration -n auto`
Expected: all green.

- [ ] **Step 3: Run integration suite**

Run: `uv run pytest -m integration -n auto`
Expected: all green (or pre-existing-flaky-only per `gotcha_*` memories).

- [ ] **Step 4: Run mechanical ergonomics + ruff baselines**

Run: `make check-ergonomics && make check-ruff-baselined && uv run ruff check src/ tests/`
Expected: all green; no new entries in baselines.

- [ ] **Step 5: Run pre-commit on changed files**

Run: `pre-commit run --all-files`
Expected: green.

- [ ] **Step 6: Commit verification record**

```bash
git commit -s --allow-empty -m "verify(config-v1): final shipping gate green for UserConfig/ServiceConfig CLI layer restoration"
```

---

## Self-Review Checklist

- [x] Spec coverage: every requirement from the brainstorm (restore v1 DTOs / strip validators / converter / TID251 fence / new top-level fields rule / no new nested classes rule) is covered by a task.
- [x] No placeholders: each task has concrete code + file paths + expected output.
- [x] Type consistency: `UserConfig`, `ServiceConfig`, `convert_user_to_aiperf` are named identically across all tasks. Section-builder names (`build_endpoint`, `build_models`, etc.) match `_cli_sections.py`'s current names.
- [x] Three-file sync rule (CLAUDE.md / copilot-instructions / cursor) is honored in Task 23.
- [x] No re-introduction of `dict[str, X]` ordered-config shapes (datasets/phases stay list-with-name on the v2 side; v1 doesn't have these concepts).
- [x] Verification gates: end-of-phase test runs, end-of-plan full suite + baselines + pre-commit.

## Risks and Mitigations

1. **cyclopts flag-discovery API may differ** — Task 7 may need a different approach (capture `--help` output if no programmatic enumeration). Mitigated by writing the test before counting on a specific API.
2. **Hidden v1 validator behaviors** — some `@model_validator` calls on `origin/main`'s UserConfig do *non-trivial* normalization (e.g. `_compute_artifact_directory`, `generate_benchmark_id`, GPU telemetry parsing). Tasks 14 and 15 catch the easy ones; the long tail must be audited at the end of Phase 1 (additional task can be inserted: "audit every stripped validator and confirm its behavior is replicated in either the converter or AIPerfConfig").
3. **`tokenizer.resolved_names` runtime mutation** — `validate_tokenizer_early(user_config, logger)` is called from `cli_runner.py` on `origin/main`. Verify whether this still exists post-cutover or has moved into `AIPerfConfig` post-load processing. If it lives outside the converter, document where; if it's gone, confirm the v2 path covers it.
4. **Sweep / multi_run from YAML, not CLI** — sweep config is YAML-only on v2. Confirm Task 19 (`kube/sweep.py`) still loads from YAML and only uses UserConfig for the per-run-CLI-overrides path.
5. **Existing tests using `CLIModel`** — Task 21 may break tests not yet migrated. Mitigated by Task 18-19 running unit suites green before Task 21.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-restore-userconfig-cli-layer.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
