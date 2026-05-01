# Tokenizer Bundle Path — Test Out + Harden

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is one subagent dispatch.
>
> **Branch:** `ajc/k8s` (do not branch off main; do not use worktrees).
>
> **Test discipline:** ONE `uv run pytest -n auto tests/unit/` per task; no per-subfolder splits, no `pre-commit run --all-files`.
>
> **Subagent model:** Always pass `model="opus"`; never let default Sonnet be used.
>
> **Commits:** Sequential tasks here, so plain `git commit -s`. Run `ruff format .` + `ruff check --fix .` manually inside the task before committing.

**Goal:** Harden the K8s tokenizer bundle distribution path: (1) delete the dead `TokenizerBundleRegistry` plumbing, (2) flip `bootstrap.py` to a controller-pod opt-out gate so worker pods get an HF air-gap belt, (3) make `download_tokenizer` crash-atomic, (4) add a cross-process round-trip test that catches the §9.1 regression class. Then run a DGX smoke per the spec §4 checklist.

**Architecture:** Controller pod's `api` container prewarms a shared `HF_HOME` emptyDir; worker pods pull `tar+zstd` bundles via `GET /api/tokenizer/{name:path}/bundle` and load via `AutoTokenizer.from_pretrained(local_path)`. Hardening preserves this topology. The registry was a cross-container Python global — dead in production — and gets removed. Air-gap moves from "application code" into a belt+suspenders env-var gate.

**Tech Stack:** Python 3.10+, asyncio, FastAPI, aiohttp, `zstandard`, `tarfile` (stdlib), `multiprocessing.spawn` for cross-process tests, pytest + pytest-asyncio.

**Spec:** [`docs/superpowers/specs/2026-04-30-tokenizer-bundle-hardening-design.md`](../specs/2026-04-30-tokenizer-bundle-hardening-design.md)

---

## Phase C — Code hardening (sequential)

### Task C-1: Delete TokenizerBundleRegistry plumbing

**Files:**
- Delete: `src/aiperf/common/tokenizer_bundle_registry.py`
- Delete: `tests/unit/common/test_tokenizer_bundle_registry.py`
- Delete: `tests/unit/common/test_tokenizer_validator_registry.py`
- Modify: `src/aiperf/common/tokenizer_validator.py` (drop `_DEFAULT_REGISTRY`, `set_default_registry`, `get_default_registry`, registry branches in `_partition_cached_names` and `_prefetch_tokenizers`)
- Modify: `src/aiperf/api/routers/tokenizer.py` (drop the `registry` parameter on `build_tokenizer_router` and `_resolve_snapshot_dir`; drop the `TYPE_CHECKING` import)
- Modify: `tests/unit/common/test_tokenizer_validator.py` (drop the `set_default_registry` import and any test cases that pass a registry)
- Rewrite: `tests/component_integration/test_tokenizer_distribution_round_trip.py` (hermetic `HF_HOME` instead of registry)

**Why:** The registry is constructed in the `control-plane`/`dataset-manager` containers but the router runs in the `api` container — separate processes, no shared globals. `api_service.py:133` calls `build_tokenizer_router()` with no arguments today; the registry path in the router is unreachable in production. Removing it eliminates ~150 lines of dead code and makes the production path the only path.

- [ ] **Step 1: Read the current state of files-to-modify so the edits below are exact.**

```bash
cat src/aiperf/api/routers/tokenizer.py
cat src/aiperf/common/tokenizer_validator.py | sed -n '1,200p'
cat tests/unit/common/test_tokenizer_validator.py | head -120
cat tests/component_integration/test_tokenizer_distribution_round_trip.py
```

- [ ] **Step 2: Delete the registry module + its two test files**

```bash
rm src/aiperf/common/tokenizer_bundle_registry.py
rm tests/unit/common/test_tokenizer_bundle_registry.py
rm tests/unit/common/test_tokenizer_validator_registry.py
```

- [ ] **Step 3: Strip registry plumbing from `tokenizer_validator.py`**

Remove these symbols and their callers:

- The top-level import `from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry`.
- `_DEFAULT_REGISTRY: TokenizerBundleRegistry | None = None` and the `set_default_registry`/`get_default_registry` functions (the `# Default registry hook` block).
- The registry-using branch inside `_partition_cached_names` (the `if already_cached:` block currently does logging + registry registration; keep the logging line, drop the registry-snapshot-download work).
- The registry-using branch inside `_prefetch_tokenizers` (the pre-loop `register_pending` and the per-future `mark_ready` that resolves snapshot dirs — drop both; keep the `try/except` and rich-panel-on-failure path).

After the edit, `tokenizer_validator.py` should have NO references to `TokenizerBundleRegistry`, `snapshot_download`, `_DEFAULT_REGISTRY`, `set_default_registry`, or `get_default_registry`. Verify with:

```bash
grep -nE 'Registry|snapshot_download|_DEFAULT_REGISTRY|set_default_registry|get_default_registry' src/aiperf/common/tokenizer_validator.py
# expected: zero matches
```

- [ ] **Step 4: Strip the registry parameter from `tokenizer.py` router**

Edit `src/aiperf/api/routers/tokenizer.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- serves tar+zstd of HF snapshot dirs from the shared cache.

The api container has zero HF egress at request time. Snapshots are populated
by the api container's own ``_prewarm_tokenizers`` (which runs before uvicorn
binds), writing into the shared ``tokenizer-cache`` emptyDir mounted at
``HF_HOME``. This router calls ``snapshot_download(local_files_only=True)``
against that shared cache and streams the resulting directory back as a single
``application/zstd`` payload.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment

_CHUNK_SIZE = 1 << 16  # 64 KiB


def _materialize_bundle(snapshot_dir: Path) -> bytes:
    """Build the full tar+zstd payload for ``snapshot_dir`` once."""
    import io as _io
    import tarfile as _tarfile

    cctx = zstandard.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
    with _io.BytesIO() as raw_tar:
        with _tarfile.open(fileobj=raw_tar, mode="w", dereference=True) as tar:
            for entry in sorted(snapshot_dir.iterdir()):
                tar.add(entry, arcname=entry.name)
        return cctx.compress(raw_tar.getvalue())


def _stream_bytes(payload: bytes) -> AsyncIterator[bytes]:
    async def _iter() -> AsyncIterator[bytes]:
        for i in range(0, len(payload), _CHUNK_SIZE):
            yield payload[i : i + _CHUNK_SIZE]

    return _iter()


async def _resolve_snapshot_dir(name: str) -> Path:
    """Return the local snapshot dir for ``name`` from the shared HF cache.

    Returns 503 when the cache is cold (worker pods retry through this) and
    404 when HF Hub doesn't recognise the name. Never reaches the network at
    request time.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        path = await asyncio.to_thread(
            snapshot_download,
            repo_id=name,
            repo_type="model",
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"tokenizer '{name}' not yet warmed in shared HF cache",
            headers={"Retry-After": "1"},
        ) from exc
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"tokenizer '{name}' not configured for this run",
        ) from exc
    return Path(path)


def build_tokenizer_router() -> APIRouter:
    """Return an APIRouter exposing ``GET /api/tokenizer/{name:path}/bundle``."""
    router = APIRouter(
        prefix="/api/tokenizer", tags=["Tokenizer"], include_in_schema=False
    )
    bundle_cache: dict[str, bytes] = {}
    cache_lock = asyncio.Lock()

    async def _get_bundle_bytes(name: str) -> bytes:
        cached = bundle_cache.get(name)
        if cached is not None:
            return cached
        async with cache_lock:
            cached = bundle_cache.get(name)
            if cached is not None:
                return cached
            snapshot_dir = await _resolve_snapshot_dir(name)
            payload = await asyncio.to_thread(_materialize_bundle, snapshot_dir)
            bundle_cache[name] = payload
            return payload

    @router.get("/{name:path}/bundle")
    async def get_tokenizer_bundle(name: str) -> StreamingResponse:
        payload = await _get_bundle_bytes(name)
        return StreamingResponse(_stream_bytes(payload), media_type="application/zstd")

    return router
```

Note: the 404 detail string is changed from `"tokenizer '{name}' not found on HuggingFace Hub: {exc}"` to `"tokenizer '{name}' not configured for this run"` — this is the C-3 cleanup folded in here so the test fixture rewrite is one task.

- [ ] **Step 5: Strip the registry-using cases from `test_tokenizer_validator.py`**

Edit `tests/unit/common/test_tokenizer_validator.py`:

- Drop the `from aiperf.common.tokenizer_validator import set_default_registry` import (and the `set_default_registry(None)` cleanup calls that pair with it). Look at lines around 20 and 105-110 of the current file.

After the edit, the file should have no references to `set_default_registry` or `TokenizerBundleRegistry`. Verify with:

```bash
grep -nE 'Registry|set_default_registry' tests/unit/common/test_tokenizer_validator.py
# expected: zero matches
```

- [ ] **Step 6: Rewrite `test_tokenizer_distribution_round_trip.py` to use a hermetic HF_HOME**

Replace the file's contents:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration round-trip: prewarm a hermetic HF_HOME, serve the
bundle through the FastAPI ``TokenizerRouter``, download via
``download_tokenizer``, and verify ``AutoTokenizer.from_pretrained(local_path)``
produces token IDs identical to the warmer's tokenizer.

This mirrors the production path (api container's ``_prewarm_tokenizers``
populating a shared ``HF_HOME`` emptyDir, then serving via the router from
that cache).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer

from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture
async def running_api(unused_tcp_port: int, tmp_path: Path, monkeypatch):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    # Prewarm the hermetic cache (mirrors api_service._prewarm_tokenizers).
    AutoTokenizer.from_pretrained("gpt2")

    app = FastAPI()
    app.include_router(build_tokenizer_router())
    config = uvicorn.Config(
        app, host="127.0.0.1", port=unused_tcp_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}"
    finally:
        server.should_exit = True
        await task


async def test_round_trip_gpt2(running_api, tmp_path: Path, monkeypatch) -> None:
    base_url = running_api
    expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")

    local_path = await download_tokenizer(
        api_base_url=base_url,
        name="gpt2",
        dest_root=tmp_path / "dl",
        max_retries=3,
        logger=logging.getLogger("test"),
    )

    # Force HF offline for the local-path load — proves the bundle is
    # self-contained and no Hub call leaks in.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
    assert actual == expected
```

- [ ] **Step 7: Run the unit suite**

```bash
ruff format src/aiperf/api/routers/tokenizer.py src/aiperf/common/tokenizer_validator.py tests/unit/common/test_tokenizer_validator.py tests/component_integration/test_tokenizer_distribution_round_trip.py
ruff check --fix src/aiperf/api/routers/tokenizer.py src/aiperf/common/tokenizer_validator.py tests/unit/common/test_tokenizer_validator.py tests/component_integration/test_tokenizer_distribution_round_trip.py
uv run pytest -n auto tests/unit/
```

Expected: all pass. Two test files were deleted; the remaining `tokenizer_validator` tests still cover the validator's behaviour.

- [ ] **Step 8: Run the component-integration round-trip locally**

```bash
uv run pytest -n auto -m component_integration tests/component_integration/test_tokenizer_distribution_round_trip.py -v
```

Expected: 1 passed. (gpt2 download is ~500KB; tolerable for one-off verification. If pre-cached on this machine, it's instant.)

- [ ] **Step 9: Commit**

```bash
git add -u src/aiperf/common/tokenizer_validator.py src/aiperf/api/routers/tokenizer.py tests/unit/common/test_tokenizer_validator.py tests/component_integration/test_tokenizer_distribution_round_trip.py
git add src/aiperf/common/tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_validator_registry.py 2>/dev/null || true
git rm src/aiperf/common/tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_validator_registry.py
git commit -s -m "refactor(tokenizer): drop dead TokenizerBundleRegistry plumbing

The registry was constructed in control-plane/dataset-manager containers
but the router runs in the api container -- separate processes, separate
Python globals. api_service.py calls build_tokenizer_router() with no
args; the registry path was unreachable in production. Removing it
eliminates ~150 lines of dead code; the component-integration round-trip
now exercises the production HF_HOME prewarm path. Also tightens the
404 detail to not echo HF Hub error formatting."
```

---

### Task C-2: Controller-pod opt-out for HF offline mode

**Files:**
- Modify: `src/aiperf/common/bootstrap.py` (replace the `AIPERF_JOB_ID` gate with `AIPERF_CONTROLLER_POD`)
- Modify: `src/aiperf/kubernetes/jobset_helpers.py` (add `controller_pod: bool` parameter to `build_env_vars`, inject the env var when set)
- Modify: `src/aiperf/kubernetes/jobset_builder.py` (pass `controller_pod=True` at every controller-pod call site; thread it through `_create_env_vars` and `_create_container`)
- Modify: `src/aiperf/kubernetes/jobset.py` (thread `controller_pod` through the `_create_env_vars` shim — used by the K8s runtime entry, mirrors `jobset_builder.py`)
- Create: `tests/unit/common/test_bootstrap_offline_gate.py` (new file)
- Modify: `tests/unit/kubernetes/test_jobset.py` (add env-var presence/absence assertions)

**Why:** Today `bootstrap.py:66` skips offline mode when `AIPERF_JOB_ID` is set, but `AIPERF_JOB_ID` is set on every pod (controller + workers). Worker air-gap therefore depends entirely on `download_tokenizer` not regressing. Flipping to a controller-pod opt-out preserves local-mode behaviour (offline by default) and adds a real belt for worker pods.

- [ ] **Step 1: Read the current state of files-to-modify**

```bash
sed -n '30,70p' src/aiperf/common/bootstrap.py
sed -n '170,245p' src/aiperf/kubernetes/jobset_helpers.py
sed -n '110,160p' src/aiperf/kubernetes/jobset_builder.py
sed -n '260,360p' src/aiperf/kubernetes/jobset_builder.py
sed -n '195,225p' src/aiperf/kubernetes/jobset.py
head -40 tests/unit/kubernetes/test_jobset.py
```

- [ ] **Step 2: Write the failing bootstrap gate test**

Create `tests/unit/common/test_bootstrap_offline_gate.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF offline-mode gate in bootstrap.py: controller-pod containers opt out;
worker pods and local mode default to offline."""
from __future__ import annotations

import multiprocessing
from unittest.mock import patch

import pytest

from aiperf.common import bootstrap


def _run_gate(env_vars: dict[str, str], parent_present: bool) -> dict[str, str]:
    """Drive _configure_child_process with a stubbed env + parent presence."""
    with patch.object(multiprocessing, "parent_process", return_value=object() if parent_present else None):
        # Snapshot the env vars we care about before/after.
        captured = {}

        def fake_signal(*_a, **_kw) -> None:
            return None

        with patch("signal.signal", side_effect=fake_signal):
            with patch.dict("os.environ", env_vars, clear=False):
                bootstrap._configure_child_process()
                captured["HF_HUB_OFFLINE"] = __import__("os").environ.get("HF_HUB_OFFLINE", "")
                captured["TRANSFORMERS_OFFLINE"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE", "")
        return captured


@pytest.fixture(autouse=True)
def _scrub_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    yield


def test_local_mode_enables_offline() -> None:
    captured = _run_gate({}, parent_present=True)
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert captured["TRANSFORMERS_OFFLINE"] == "1"


def test_worker_pod_enables_offline() -> None:
    # Worker pod: AIPERF_JOB_ID set, AIPERF_CONTROLLER_POD unset.
    captured = _run_gate({"AIPERF_JOB_ID": "job-7"}, parent_present=True)
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert captured["TRANSFORMERS_OFFLINE"] == "1"


def test_controller_pod_skips_offline() -> None:
    captured = _run_gate(
        {"AIPERF_JOB_ID": "job-7", "AIPERF_CONTROLLER_POD": "1"},
        parent_present=True,
    )
    assert captured["HF_HUB_OFFLINE"] == ""
    assert captured["TRANSFORMERS_OFFLINE"] == ""


def test_main_process_skips_signal_and_gate() -> None:
    # parent_process() is None → the function returns before touching env.
    captured = _run_gate({"AIPERF_CONTROLLER_POD": "1"}, parent_present=False)
    assert captured["HF_HUB_OFFLINE"] == ""
    assert captured["TRANSFORMERS_OFFLINE"] == ""
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
uv run pytest -n auto tests/unit/common/test_bootstrap_offline_gate.py -v
```

Expected: `test_worker_pod_enables_offline` FAILS (current code skips offline when `AIPERF_JOB_ID` is set).

- [ ] **Step 4: Edit `bootstrap.py` to flip the gate**

In `src/aiperf/common/bootstrap.py`, change `_configure_child_process` (around lines 46-67):

```python
def _configure_child_process() -> None:
    """Prepare a child-process environment: signals and HF offline mode.

    Ignore SIGINT and SIGTERM in child processes. SIGINT is ignored so only
    the parent handles Ctrl+C. SIGTERM is ignored because graceful shutdown is
    handled via the message bus (ShutdownCommand); process.terminate() is only
    called after the message bus path has already timed out, and the manager
    falls through to SIGKILL after the join timeout anyway. Ignoring SIGTERM
    prevents SIGSEGV crashes that occur when SIGTERM arrives while C extension
    code (uvloop, zmq, aiohttp, orjson) is executing.
    """
    if multiprocessing.parent_process() is None:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # HF offline-mode gate: enable everywhere EXCEPT controller-pod containers.
    # The controller pod's api / dataset-manager containers need HF egress for
    # prewarming the shared cache and for synthetic-dataset prompt generation;
    # every other context (worker pods, local mode) defaults to offline so a
    # regression that re-introduces ``from_pretrained(name)`` blows up
    # immediately instead of silently re-establishing HF egress.
    if os.environ.get("AIPERF_CONTROLLER_POD") != "1":
        _enable_hf_offline_mode()
```

- [ ] **Step 5: Run the bootstrap test, expect green**

```bash
uv run pytest -n auto tests/unit/common/test_bootstrap_offline_gate.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Add the `controller_pod` parameter to `build_env_vars`**

In `src/aiperf/kubernetes/jobset_helpers.py`, edit `build_env_vars`:

```python
def build_env_vars(
    *,
    job_id: str,
    namespace: str,
    pod_template: PodTemplateConfig,
    controller_host: str | None = None,
    include_pod_index: bool = True,
    controller_pod: bool = False,
) -> list[dict[str, Any]]:
    """Create environment variables for a container.

    Args:
        controller_pod: When True, this container runs inside the controller
            pod (api / dataset-manager / etc.) — emit ``AIPERF_CONTROLLER_POD=1``
            so ``bootstrap.py`` skips HF offline-mode (the controller needs
            HF egress for prewarming the shared cache). Worker-pod containers
            default to False and inherit offline mode.
    """
    # ... (existing body unchanged through the env list construction) ...
    env: list[dict[str, Any]] = [
        # ... (existing vars unchanged) ...
    ]

    if controller_pod:
        env.append({"name": "AIPERF_CONTROLLER_POD", "value": "1"})

    # ... (rest of the existing body unchanged) ...
    return env
```

The exact placement of the new `if controller_pod:` block: after the initial `env: list[dict[str, Any]] = [...]` literal (at the spot where the existing `if not has_hf_home:` block lives), before the existing `if include_pod_index:` block. Order doesn't matter for correctness, but keeping the new block tight to the static env-list literal makes it scan first.

- [ ] **Step 7: Thread `controller_pod` through the two `_create_env_vars` shims**

In `src/aiperf/kubernetes/jobset_builder.py` (around lines 117-129):

```python
def _create_env_vars(
    self,
    controller_host: str | None = None,
    include_pod_index: bool = True,
    controller_pod: bool = False,
) -> list[dict[str, Any]]:
    """Create environment variables for a container."""
    return build_env_vars(
        job_id=self.spec.job_id,
        namespace=self.spec.namespace,
        pod_template=self.spec.pod_template,
        controller_host=controller_host,
        include_pod_index=include_pod_index,
        controller_pod=controller_pod,
    )
```

Same change in `src/aiperf/kubernetes/jobset.py` (around lines 203-216):

```python
def _create_env_vars(
    self,
    controller_host: str | None = None,
    include_pod_index: bool = True,
    controller_pod: bool = False,
) -> list[dict[str, Any]]:
    from aiperf.kubernetes.jobset_helpers import build_env_vars

    return build_env_vars(
        job_id=self.job_id,
        namespace=self.namespace,
        pod_template=self.pod_template,
        controller_host=controller_host,
        include_pod_index=include_pod_index,
        controller_pod=controller_pod,
    )
```

- [ ] **Step 8: Add `controller_pod` parameter to `_create_container` and pass `True` at controller-pod call sites**

In `src/aiperf/kubernetes/jobset_builder.py`, edit `_create_container` (around lines 133-156):

```python
def _create_container(
    self,
    name: str,
    service_type: str,
    health_port: int | None,
    resources: dict[str, dict[str, str]] | None,
    *,
    api_port: int | None = None,
    controller_host: str | None = None,
    service_id: str | None = None,
    extra_env: list[dict[str, Any]] | None = None,
    include_pod_index: bool = True,
    controller_pod: bool = False,
    skip_readiness_probe: bool = False,
    skip_startup_probe: bool = False,
    skip_liveness_probe: bool = False,
) -> AIPerfContainerSpec:
    """Create a container spec with standard AIPerf configuration."""
    args = build_container_args(service_type, health_port, api_port, service_id)
    ports = build_container_ports(health_port, api_port)

    env = self._create_env_vars(
        controller_host=controller_host,
        include_pod_index=include_pod_index,
        controller_pod=controller_pod,
    )
    if extra_env:
        env.extend(extra_env)
    # ... (rest of the existing body unchanged) ...
```

Then, at every controller-pod container construction site (currently every `include_pod_index=False` call), add `controller_pod=True`. The sites to edit are in this same file:

- `_create_event_bus_proxy_container` — `_create_env_vars(include_pod_index=False)` → `_create_env_vars(include_pod_index=False, controller_pod=True)` (line 225).
- `_create_control_plane_containers` — every `_create_container(... include_pod_index=False, ...)` call gains `controller_pod=True`. There are five entries (CONTROL_PLANE / DATASET_MANAGER / TIMING_MANAGER / RECORDS_MANAGER / API). Use grep to find them; add the kwarg right after `include_pod_index=False`.
- Any other site in the file with `include_pod_index=False` (e.g. server-metrics-manager, gpu-telemetry-manager) — those are also controller-pod containers; add `controller_pod=True`.

After the edits, verify:

```bash
grep -nE 'include_pod_index=False' src/aiperf/kubernetes/jobset_builder.py
# Every match should be on a line that ALSO has `controller_pod=True`
# (or on the next/previous line in the same kwarg list).
grep -nB1 -A1 'controller_pod=True' src/aiperf/kubernetes/jobset_builder.py | head -40
```

Worker-pod construction sites (the `_create_worker_container` / `_create_record_processor_container` etc., which use the default `include_pod_index=True`) MUST NOT pass `controller_pod=True`. Sanity-grep:

```bash
grep -nE '_create_container\(' src/aiperf/kubernetes/jobset_builder.py | head -30
```

For each match, eyeball the surrounding kwargs: any container in a worker-pod path keeps `controller_pod` defaulted (i.e. omit the kwarg).

- [ ] **Step 9: Add jobset env-var assertion tests**

In `tests/unit/kubernetes/test_jobset.py`, add the following tests (place them after existing `build_env_vars`-touching tests; if none exist, add at the end of the file):

```python
def test_build_env_vars_controller_pod_emits_marker():
    """Controller-pod call sets AIPERF_CONTROLLER_POD=1."""
    from aiperf.kubernetes.jobset_helpers import build_env_vars
    from aiperf.kubernetes.environment import K8sEnvironment

    env = build_env_vars(
        job_id="job-7",
        namespace="ns",
        pod_template=K8sEnvironment.JOBSET.DEFAULT_POD_TEMPLATE,
        controller_pod=True,
    )
    names = {e["name"]: e.get("value") for e in env}
    assert names.get("AIPERF_CONTROLLER_POD") == "1"


def test_build_env_vars_worker_pod_omits_marker():
    """Worker-pod call (default controller_pod=False) does not set the marker."""
    from aiperf.kubernetes.jobset_helpers import build_env_vars
    from aiperf.kubernetes.environment import K8sEnvironment

    env = build_env_vars(
        job_id="job-7",
        namespace="ns",
        pod_template=K8sEnvironment.JOBSET.DEFAULT_POD_TEMPLATE,
    )
    names = [e["name"] for e in env]
    assert "AIPERF_CONTROLLER_POD" not in names
```

If `K8sEnvironment.JOBSET.DEFAULT_POD_TEMPLATE` doesn't exist, copy the construction style from the nearest existing `build_env_vars`-using test in the same file. The pod_template must be a real `PodTemplateConfig` with at least an empty `env: list[Any] = []` attribute.

- [ ] **Step 10: Run the full unit suite**

```bash
ruff format src/aiperf/common/bootstrap.py src/aiperf/kubernetes/jobset_helpers.py src/aiperf/kubernetes/jobset_builder.py src/aiperf/kubernetes/jobset.py tests/unit/common/test_bootstrap_offline_gate.py tests/unit/kubernetes/test_jobset.py
ruff check --fix src/aiperf/common/bootstrap.py src/aiperf/kubernetes/jobset_helpers.py src/aiperf/kubernetes/jobset_builder.py src/aiperf/kubernetes/jobset.py tests/unit/common/test_bootstrap_offline_gate.py tests/unit/kubernetes/test_jobset.py
uv run pytest -n auto tests/unit/
```

Expected: all pass. The four new bootstrap-gate tests + two new jobset env tests are green.

- [ ] **Step 11: Commit**

```bash
git add -u src/aiperf/common/bootstrap.py src/aiperf/kubernetes/jobset_helpers.py src/aiperf/kubernetes/jobset_builder.py src/aiperf/kubernetes/jobset.py tests/unit/common/test_bootstrap_offline_gate.py tests/unit/kubernetes/test_jobset.py
git add tests/unit/common/test_bootstrap_offline_gate.py
git commit -s -m "feat(bootstrap,k8s): controller-pod opt-out gate for HF offline mode

bootstrap.py defaulted to skipping HF offline mode whenever AIPERF_JOB_ID
was set, but that var is on every pod (controller + workers). Worker air-
gap was therefore enforced only by application code. Flip to a positive
controller-pod-only opt-out (AIPERF_CONTROLLER_POD=1) so worker pods and
local mode both default to offline. The operator now injects the marker
on every controller-pod container via the new build_env_vars(..., \\
controller_pod=True) flag, plumbed through jobset_builder/jobset shims."
```

---

### Task C-3: Atomic extract for `download_tokenizer`

**Files:**
- Modify: `src/aiperf/workers/worker_pod_tokenizer_download.py`
- Modify: `tests/unit/workers/test_worker_pod_tokenizer_download.py`

**Why:** Today `_extract_bundle(compressed, dest)` writes directly into `dest`, then `sentinel.write_text("ok")` lands at `dest_root/{slug}/.ready`. A crash mid-tar leaves a half-populated `dest/` with no sentinel; on retry the next `extractall` runs on top of the partial tree. Atomic rename eliminates the half-state.

- [ ] **Step 1: Write the failing crash-mid-tar test**

Append to `tests/unit/workers/test_worker_pod_tokenizer_download.py`:

```python
@pytest.mark.asyncio
async def test_extract_crash_then_retry_succeeds(stub_server, tmp_path: Path, monkeypatch) -> None:
    """A crash during extraction must not leave a partial bundle dir."""
    from aiperf.workers import worker_pod_tokenizer_download as wptd

    server, state = stub_server
    state["bundle"] = _make_bundle({"tokenizer.json": b'{"v":1}', "vocab.json": b"{}"})

    real_extract = wptd._extract_bundle
    calls = {"n": 0}

    def crashing_extract(compressed: bytes, dest: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate a partial extract: write one file, then raise.
            (dest / "tokenizer.json").write_bytes(b'{"v":1}')
            raise RuntimeError("simulated extract crash")
        real_extract(compressed, dest)

    monkeypatch.setattr(wptd, "_extract_bundle", crashing_extract)

    # First attempt crashes mid-extract; the helper raises.
    with pytest.raises(RuntimeError, match="simulated"):
        await wptd.download_tokenizer(
            api_base_url=str(server.make_url("")),
            name="gpt2",
            dest_root=tmp_path,
            max_retries=1,
            logger=logging.getLogger("test"),
        )

    # No half-state left at the final dest.
    final = tmp_path / wptd.slug_for_tokenizer("gpt2")
    assert not final.exists() or not any(final.iterdir()), (
        f"extract crash left partial files at {final}"
    )

    # Second attempt (real extractor) succeeds.
    out = await wptd.download_tokenizer(
        api_base_url=str(server.make_url("")),
        name="gpt2",
        dest_root=tmp_path,
        max_retries=2,
        logger=logging.getLogger("test"),
    )
    assert (out / "tokenizer.json").read_bytes() == b'{"v":1}'
    assert (out / "vocab.json").exists()
    assert (out / ".ready").exists()
```

- [ ] **Step 2: Run the test, expect failure**

```bash
uv run pytest -n auto tests/unit/workers/test_worker_pod_tokenizer_download.py::test_extract_crash_then_retry_succeeds -v
```

Expected: FAIL — the partial dir is left behind, OR the second attempt fails because it's loading on top of the partial tree.

- [ ] **Step 3: Edit `download_tokenizer` for atomic extract**

In `src/aiperf/workers/worker_pod_tokenizer_download.py`, replace the body of `download_tokenizer` (the lock-acquired section, currently lines ~62-115) with:

```python
async def download_tokenizer(
    *,
    api_base_url: str,
    name: str,
    dest_root: Path,
    max_retries: int,
    logger: logging.Logger,
) -> Path:
    """Download and extract one tokenizer bundle. Returns the snapshot dir.

    Extraction is crash-atomic: the tar is unpacked into ``{slug}.tmp/``,
    a ``.ready`` sentinel is written inside, then the directory is renamed
    to ``{slug}/`` via ``os.replace``. A crash mid-extraction leaves the
    tmp dir behind (cleaned up on next retry) but no half-populated final
    dir; readers always see a fully-extracted bundle or nothing.

    Raises:
        RuntimeError: 404 from server, or retries exhausted.
    """
    import os
    import shutil

    base = api_base_url.rstrip("/")
    url = f"{base}/api/tokenizer/{name}/bundle"
    logger.info(f"download_tokenizer: starting for '{name}' from {url}")
    slug = slug_for_tokenizer(name)
    dest = dest_root / slug
    dest_root.mkdir(parents=True, exist_ok=True)
    sentinel = dest / ".ready"
    if sentinel.exists():
        logger.info(f"download_tokenizer: '{name}' already extracted at {dest}")
        return dest

    lock_path = dest_root / f"{slug}.lock"
    logger.info(f"download_tokenizer: acquiring bundle lock at {lock_path}")
    async with _bundle_lock(lock_path):
        logger.info(f"download_tokenizer: lock acquired for '{name}'")
        if sentinel.exists():
            return dest

        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None
        request_timeout = aiohttp.ClientTimeout(total=300.0)
        async with aiohttp.ClientSession(
            connector=create_tcp_connector(), timeout=request_timeout
        ) as session:
            for attempt in range(1, max_retries + 1):
                try:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            raise RuntimeError(
                                f"tokenizer '{name}' not registered on operator API "
                                f"(HTTP 404 from {url})"
                            )
                        if resp.status == 503:
                            logger.info(
                                f"tokenizer '{name}' not ready (503), "
                                f"attempt {attempt}/{max_retries}"
                            )
                            await asyncio.sleep(min(backoff, _MAX_BACKOFF_S))
                            backoff *= 2
                            continue
                        resp.raise_for_status()
                        compressed = await resp.read()
                    logger.info(
                        f"download_tokenizer: '{name}' fetched "
                        f"({len(compressed)} bytes), extracting atomically"
                    )
                    tmp_dest = dest_root / f"{slug}.tmp"
                    if tmp_dest.exists():
                        shutil.rmtree(tmp_dest)
                    tmp_dest.mkdir(parents=True)
                    try:
                        _extract_bundle(compressed, tmp_dest)
                        (tmp_dest / ".ready").write_text("ok")
                    except BaseException:
                        # Clean up the partial tmp dir; final dest is untouched.
                        shutil.rmtree(tmp_dest, ignore_errors=True)
                        raise
                    # Atomic swap; survives crashes on either side of the rename.
                    if dest.exists():
                        shutil.rmtree(dest)
                    os.replace(tmp_dest, dest)
                    logger.info(f"download_tokenizer: '{name}' ready at {dest}")
                    return dest
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                    logger.warning(
                        f"transient error downloading tokenizer '{name}' "
                        f"({type(exc).__name__}: {exc}); attempt {attempt}/{max_retries}"
                    )
                    await asyncio.sleep(min(backoff, _MAX_BACKOFF_S))
                    backoff *= 2

        raise RuntimeError(
            f"failed to download tokenizer '{name}' after {max_retries} attempts: {last_exc}"
        )
```

Key changes vs current code:

- `dest.mkdir(parents=True, exist_ok=True)` is replaced by `dest_root.mkdir(parents=True, exist_ok=True)`; `dest` is created only via the rename.
- Extraction goes to `tmp_dest = dest_root / f"{slug}.tmp"`; if it pre-exists from a prior crash, it's wiped first.
- `.ready` is written inside `tmp_dest`, then `os.replace` swaps it in atomically. The sentinel survives the rename automatically.
- Any `BaseException` during extraction or sentinel-write triggers `shutil.rmtree(tmp_dest, ignore_errors=True)`. The final `dest` is never touched until after a successful extract.

- [ ] **Step 4: Run the new test**

```bash
uv run pytest -n auto tests/unit/workers/test_worker_pod_tokenizer_download.py::test_extract_crash_then_retry_succeeds -v
```

Expected: PASS.

- [ ] **Step 5: Run the full file to verify no regressions in existing cases**

```bash
uv run pytest -n auto tests/unit/workers/test_worker_pod_tokenizer_download.py -v
```

Expected: all pass.

- [ ] **Step 6: Run the full unit suite**

```bash
ruff format src/aiperf/workers/worker_pod_tokenizer_download.py tests/unit/workers/test_worker_pod_tokenizer_download.py
ruff check --fix src/aiperf/workers/worker_pod_tokenizer_download.py tests/unit/workers/test_worker_pod_tokenizer_download.py
uv run pytest -n auto tests/unit/
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -u src/aiperf/workers/worker_pod_tokenizer_download.py tests/unit/workers/test_worker_pod_tokenizer_download.py
git commit -s -m "fix(workers): crash-atomic extract in download_tokenizer

Extract into {slug}.tmp/, write the .ready sentinel inside, then
os.replace into {slug}/. A crash mid-tar now leaves no half-populated
final dir; the next retry sees a clean state. Lock + sentinel-short-
circuit semantics are preserved."
```

---

## Phase B — Cross-process round-trip test

### Task B-2: Component-integration cross-process test

**Files:**
- Create: `tests/component_integration/test_tokenizer_router_cross_process.py`

**Why:** Every existing test runs the warmer and the router in one Python process. The production topology is two containers with separate interpreters sharing only `HF_HOME` on disk. A future regression that re-introduces process-local Python state (a global, an in-memory mutex, a registry) would silently regress the cross-container contract. This test catches the §9.1 class.

- [ ] **Step 1: Write the test**

Create `tests/component_integration/test_tokenizer_router_cross_process.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process round-trip: prove the router serves bundles from a shared
HF_HOME populated by a different Python process — mirrors the production
topology where the api container's interpreter is separate from the
controller-plane container that prewarms.

Spawns the FastAPI app via ``multiprocessing.get_context("spawn")`` so the
child cannot inherit the parent's already-imported ``transformers`` modules
or any module-level globals.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import socket
import time
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_router(hf_home: str, port: int) -> None:
    """Subprocess entry: serve the tokenizer router with HF_HOME injected.

    Module-level so spawn can pickle it.
    """
    import os

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import uvicorn
    from fastapi import FastAPI

    from aiperf.api.routers.tokenizer import build_tokenizer_router

    app = FastAPI()
    app.include_router(build_tokenizer_router())
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _wait_for_tcp(host: str, port: int, deadline_s: float = 15.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"router subprocess did not bind {host}:{port} within {deadline_s}s")


async def test_router_serves_from_subprocess_populated_hf_home(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    hf_home.mkdir()

    # Parent process primes the hermetic cache by loading gpt2 with HF_HOME
    # pointed at the shared dir. The child process will read from the same
    # on-disk tree without sharing any Python state.
    import os
    os.environ["HF_HOME"] = str(hf_home)
    AutoTokenizer.from_pretrained("gpt2")

    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_serve_router, args=(str(hf_home), port), daemon=True)
    proc.start()
    try:
        await asyncio.to_thread(_wait_for_tcp, "127.0.0.1", port)

        local_path = await download_tokenizer(
            api_base_url=f"http://127.0.0.1:{port}",
            name="gpt2",
            dest_root=tmp_path / "dl",
            max_retries=3,
            logger=logging.getLogger("test"),
        )
        # Round-trip: load the downloaded snapshot offline, verify token IDs.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
        expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")
        assert actual == expected
    finally:
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest -n auto -m component_integration tests/component_integration/test_tokenizer_router_cross_process.py -v
```

Expected: 1 passed. If it fails with timeout, the spawn-based child may need extra startup time — bump `_wait_for_tcp` deadline to 30s. If it fails because `gpt2` isn't cached, the parent's `AutoTokenizer.from_pretrained("gpt2")` should populate it in `tmp_path/hf` (network egress required on first run; cached for subsequent runs).

- [ ] **Step 3: Run the unit suite + the existing component-integration round-trip to confirm no regressions**

```bash
ruff format tests/component_integration/test_tokenizer_router_cross_process.py
ruff check --fix tests/component_integration/test_tokenizer_router_cross_process.py
uv run pytest -n auto tests/unit/
uv run pytest -n auto -m component_integration tests/component_integration/test_tokenizer_distribution_round_trip.py tests/component_integration/test_tokenizer_router_cross_process.py
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/component_integration/test_tokenizer_router_cross_process.py
git commit -s -m "test(tokenizer): cross-process round-trip via spawn-spawned router

Catches the spec §9.1 regression class: any change that re-introduces
process-local Python state for the warmer↔router contract will fail
this test because the child process literally cannot see parent globals.
Production parity: HF_HOME is the only shared state."
```

---

## Phase A — DGX smoke (operational, not committed code)

> This phase is operator-driven, not a code commit. Track outcomes in this checklist; if anything fails, fold the fix into a follow-up commit on `ajc/k8s` and re-roll.

- [ ] **A-1: Build + push image off `ajc/k8s` HEAD.**

Use the existing image-build workflow under `~/.claude/workflows/aiperf-dgx/`. (Read `~/.claude/workflows/aiperf-dgx/index.md` first; pick the playbook that matches "build + push aiperf operator image".)

- [ ] **A-2: Roll the operator on DGX.**

`helm upgrade aiperf-operator ...` with the new image tag, against the active DGX cluster (`--context <dgx-context>` per the `feedback_kind_explicit_context` durable rule applies to kind only; for DGX use the workflow's documented context name).

- [ ] **A-3: Run a 60s `gpt2` smoke against the mock-server config.**

```bash
aiperf kube run --model gpt2 --tokenizer gpt2 --duration 60s ...
```

(Exact flags per the active `aiperf kube` template — read `dev/scripts/smoke_100k_conc.py` for the current shape and adapt to a small `gpt2` config.)

- [ ] **A-4: Validate the §4 checklist for the gpt2 run:**

```bash
# Worker pod env (any worker pod):
kubectl exec -n <ns> <worker-pod> -c <worker-container> -- env | grep -E 'AIPERF_CONTROLLER_POD|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|AIPERF_JOB_ID'
# Expected: AIPERF_CONTROLLER_POD unset; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; AIPERF_JOB_ID set.

# Controller pod env (api container):
kubectl exec -n <ns> <controller-pod> -c api -- env | grep -E 'AIPERF_CONTROLLER_POD|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE'
# Expected: AIPERF_CONTROLLER_POD=1; HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE unset (or whatever the api container's bootstrap leaves).

# Worker pod logs: zero references to huggingface.co.
kubectl logs -n <ns> <worker-pod> --all-containers | grep -iE 'huggingface\.co|connection refused.*hf' || echo "PASS: no HF references"

# Bundle fetch evidence: exactly one successful GET per worker pod.
kubectl logs -n <ns> <worker-pod> -c <worker-container> | grep -iE 'download_tokenizer.*ready|/api/tokenizer/.*bundle' | head
```

- [ ] **A-5: Run a 60s `meta-llama/Llama-3.1-8B-Instruct` smoke; repeat the §4 checklist.**

- [ ] **A-6: Mark spec §6 (post-smoke amendment) with the outcome:**
   - If everything passes: append a 1-paragraph "Smoke passed YYYY-MM-DD" section to the spec.
   - If anything fails: file the fix as a new follow-up commit on `ajc/k8s`; capture the failure mode + fix in spec §6 like the predecessor's §9 did.

---

## Out-of-scope reminders (no tasks)

- Persistent (PVC) tokenizer cache across pod restarts.
- Tokenizer revision pinning in the bundle URL.
- Real-streaming download (bundles <10MB).
- Cache-bounding the per-name `bundle_cache` (already bounded by run cardinality).
- K8s chaos test extension — covered by a separate spec.
