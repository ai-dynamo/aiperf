# Tokenizer Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (or superpowers:dispatching-parallel-agents for tasks marked PARALLEL-SAFE) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Branch:** Work on the user's current branch `ajc/k8s` (do not branch off main; do not use worktrees).
>
> **Parallel-agent commit hygiene (aiperf gotcha):** the pre-commit framework's internal `git stash --include-untracked` corrupts state under parallel agents. Each parallel agent MUST commit with `git commit --no-verify -s`, after running `ruff format . && ruff check --fix .` manually inside its task scope. Sequential tasks may use plain `git commit -s`.
>
> **Test discipline:** ONE `uv run pytest -n auto tests/unit/` per task — no per-subfolder splits, no `pre-commit run --all-files`.

**Goal:** Distribute tokenizers via the operator API in K8s mode, the same way datasets are distributed today, so worker pods never reach huggingface.co.

**Architecture:** Controller-side `TokenizerBundleRegistry` records the resolved HF snapshot directory for each tokenizer once `validate_tokenizers_eager` warms the cache. New `TokenizerRouter` exposes `GET /api/tokenizer/{name:path}/bundle` streaming `tar | zstd` of that snapshot dir. Pod-side `download_tokenizer` (mirrors `download_dataset`) fetches and untars into emptyDir; `WorkerGroupManager` publishes a pod-local `GroupTokenizerReady` to in-process workers; the RecordProcessor sibling container calls `download_tokenizer` itself with a per-bundle file lock for race-safety. Local mode is unchanged.

**Tech Stack:** Python 3.10+, asyncio, FastAPI, aiohttp, `zstandard` (already a dep), `tarfile` (stdlib), `msgspec.Struct` for pod-lifecycle messages, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-04-26-tokenizer-distribution-design.md`

---

## Phase A — Foundations (PARALLEL-SAFE)

The two tasks in this phase touch disjoint files and can be dispatched concurrently.

### Task A1: TokenizerBundleRegistry

**Files:**
- Create: `src/aiperf/common/tokenizer_bundle_registry.py`
- Test:   `tests/unit/common/test_tokenizer_bundle_registry.py`

**Behavior:** thread/async-safe registry keyed by tokenizer name. `register_pending(name)` creates a `(snapshot_dir=None, ready=asyncio.Event())` entry. `mark_ready(name, snapshot_dir: Path)` sets the path and `set()`s the event. `get(name)` returns `(snapshot_dir | None, event)` or `None` if not registered. Idempotent (`register_pending` of an already-pending name is a no-op; `mark_ready` overwrites).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/common/test_tokenizer_bundle_registry.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry


@pytest.mark.asyncio
async def test_register_pending_creates_unready_entry(tmp_path: Path) -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")
    snapshot, event = reg.get("gpt2")
    assert snapshot is None
    assert not event.is_set()


@pytest.mark.asyncio
async def test_mark_ready_sets_event_and_snapshot(tmp_path: Path) -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")
    reg.mark_ready("gpt2", tmp_path / "snap")
    snapshot, event = reg.get("gpt2")
    assert snapshot == tmp_path / "snap"
    assert event.is_set()


@pytest.mark.asyncio
async def test_get_unknown_returns_none() -> None:
    reg = TokenizerBundleRegistry()
    assert reg.get("never-registered") is None


@pytest.mark.asyncio
async def test_register_pending_idempotent() -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")
    _, event_first = reg.get("gpt2")
    reg.register_pending("gpt2")
    _, event_second = reg.get("gpt2")
    assert event_first is event_second  # same event reused


@pytest.mark.asyncio
async def test_mark_ready_unblocks_waiter(tmp_path: Path) -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")

    async def wait_then_get() -> Path:
        _, event = reg.get("gpt2")
        await event.wait()
        snapshot, _ = reg.get("gpt2")
        return snapshot

    waiter = asyncio.create_task(wait_then_get())
    await asyncio.sleep(0)  # let waiter park
    reg.mark_ready("gpt2", tmp_path / "snap")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result == tmp_path / "snap"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -n auto tests/unit/common/test_tokenizer_bundle_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: aiperf.common.tokenizer_bundle_registry`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/aiperf/common/tokenizer_bundle_registry.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry of HF tokenizer snapshot directories warmed by the controller.

Populated by ``tokenizer_validator.validate_tokenizers_eager`` as each per-tokenizer
warmer process completes. Read by ``TokenizerRouter`` to serve tar+zstd bundles.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class _Entry:
    """Per-tokenizer registration: snapshot path + readiness event."""

    snapshot_dir: Path | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class TokenizerBundleRegistry:
    """Maps tokenizer names to their resolved on-disk snapshot directories."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def register_pending(self, name: str) -> None:
        """Reserve a slot for ``name`` if not already present."""
        with self._lock:
            self._entries.setdefault(name, _Entry())

    def mark_ready(self, name: str, snapshot_dir: Path) -> None:
        """Record the resolved snapshot directory and unblock waiters."""
        with self._lock:
            entry = self._entries.setdefault(name, _Entry())
            entry.snapshot_dir = snapshot_dir
            entry.ready.set()

    def get(self, name: str) -> tuple[Path | None, asyncio.Event] | None:
        """Return (snapshot_dir, ready_event) or ``None`` if unknown."""
        with self._lock:
            entry = self._entries.get(name)
        if entry is None:
            return None
        return entry.snapshot_dir, entry.ready
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -n auto tests/unit/common/test_tokenizer_bundle_registry.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
ruff format src/aiperf/common/tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_bundle_registry.py
ruff check --fix src/aiperf/common/tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_bundle_registry.py
git add src/aiperf/common/tokenizer_bundle_registry.py tests/unit/common/test_tokenizer_bundle_registry.py
git commit --no-verify -s -m "feat(tokenizer): TokenizerBundleRegistry for controller-side snapshot tracking"
```

---

### Task A2: GroupTokenizerReady pod-lifecycle struct

**Files:**
- Modify: `src/aiperf/common/pod_lifecycle_structs.py` (add struct + extend the union at line ~252)
- Test:   `tests/unit/common/test_pod_lifecycle_structs.py` (add a test if module has one; otherwise a tiny new file)

**Behavior:** New `msgspec.Struct` `GroupTokenizerReady` parallel to `GroupDatasetReady`, fields: `service_id: str`, `bundles: dict[str, str]` (tokenizer-name → local-snapshot-path), `success: bool = True`, `error_message: str | None = None`. Tag string `"tokenizer"`. Add it to the union at the bottom of the file.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/common/test_group_tokenizer_ready.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import msgspec

from aiperf.common.pod_lifecycle_structs import GroupTokenizerReady


def test_group_tokenizer_ready_round_trips() -> None:
    msg = GroupTokenizerReady(
        service_id="wgm-0",
        bundles={"gpt2": "/tmp/aiperf_tokenizers/run-1/gpt2"},
    )
    raw = msgspec.json.encode(msg)
    decoded = msgspec.json.decode(raw, type=GroupTokenizerReady)
    assert decoded == msg


def test_group_tokenizer_ready_failure_carries_error() -> None:
    msg = GroupTokenizerReady(
        service_id="wgm-0",
        bundles={},
        success=False,
        error_message="503 after retries",
    )
    assert msg.success is False
    assert msg.error_message == "503 after retries"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -n auto tests/unit/common/test_group_tokenizer_ready.py -v`
Expected: FAIL with `ImportError: cannot import name 'GroupTokenizerReady'`.

- [ ] **Step 3: Add the struct + extend the union**

In `src/aiperf/common/pod_lifecycle_structs.py`, immediately after the `GroupDatasetReady` class (around line 155), add:

```python
class GroupTokenizerReady(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="tokenizer",
):
    """Group-local tokenizer availability notification from WorkerGroupManager."""

    service_id: str
    """Service identifier for the group manager publishing tokenizer readiness."""

    bundles: dict[str, str]
    """Map of tokenizer name to local snapshot directory path."""

    success: bool = True
    """Whether tokenizer acquisition completed successfully."""

    error_message: str | None = None
    """Tokenizer acquisition error message when success is false."""
```

Then extend the union at line ~252 to include `GroupTokenizerReady`:

```python
GroupPeerAck | GroupDatasetReady | GroupTokenizerReady | GroupDatasetStateSnapshot | GroupPeerCommand
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -n auto tests/unit/common/test_group_tokenizer_ready.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
ruff format src/aiperf/common/pod_lifecycle_structs.py tests/unit/common/test_group_tokenizer_ready.py
ruff check --fix src/aiperf/common/pod_lifecycle_structs.py tests/unit/common/test_group_tokenizer_ready.py
git add src/aiperf/common/pod_lifecycle_structs.py tests/unit/common/test_group_tokenizer_ready.py
git commit --no-verify -s -m "feat(pod-lifecycle): add GroupTokenizerReady struct"
```

---

## Phase B — Controller side (sequential; B2/B3 depend on A1)

### Task B1: Wire registry into `validate_tokenizers_eager`

**Files:**
- Modify: `src/aiperf/common/tokenizer_validator.py`

**Behavior:** Accept an optional `registry: TokenizerBundleRegistry | None = None` kwarg on `validate_tokenizers_eager`. When provided, call `registry.register_pending(name)` for each unique tokenizer up front, and after each per-tokenizer warmer returns, resolve the HF snapshot directory and call `registry.mark_ready(name, snapshot_dir)`. Snapshot resolution uses `huggingface_hub.snapshot_download(name, revision=..., local_files_only=True)` (it returns the existing path without re-downloading because the warmer already populated the cache). Default `None` means existing call sites are unaffected.

Construct a single module-level `_DEFAULT_REGISTRY: TokenizerBundleRegistry | None = None` setter (`set_default_registry(reg)`) so the operator's FastAPI app and the validator share the same instance without plumbing the dependency through every callsite.

- [ ] **Step 1: Write a failing integration-style test**

```python
# tests/unit/common/test_tokenizer_validator_registry.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that validate_tokenizers_eager populates the bundle registry.

Uses a tiny real tokenizer (``gpt2``) so the test exercises the real HF
snapshot-resolution path; runs offline if HF cache is already warm, otherwise
performs one network download (~500KB) which is tolerable for one CI run.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry


@pytest.mark.asyncio
async def test_validate_tokenizers_eager_marks_registry_ready() -> None:
    from aiperf.common.tokenizer_validator import validate_tokenizers_eager

    reg = TokenizerBundleRegistry()
    validate_tokenizers_eager(["gpt2"], registry=reg)

    entry = reg.get("gpt2")
    assert entry is not None
    snapshot_dir, event = entry
    assert event.is_set()
    assert snapshot_dir is not None
    assert (snapshot_dir / "tokenizer.json").exists() or (snapshot_dir / "vocab.json").exists()
    assert snapshot_dir.is_absolute()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -n auto tests/unit/common/test_tokenizer_validator_registry.py -v`
Expected: FAIL — either `TypeError: unexpected keyword argument 'registry'` or empty registry.

- [ ] **Step 3: Edit `tokenizer_validator.py`**

Add at top:

```python
from huggingface_hub import snapshot_download

from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry

_DEFAULT_REGISTRY: TokenizerBundleRegistry | None = None


def set_default_registry(registry: TokenizerBundleRegistry | None) -> None:
    """Module-level hook so the FastAPI app and validator share one registry."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = registry


def get_default_registry() -> TokenizerBundleRegistry | None:
    return _DEFAULT_REGISTRY
```

Modify `validate_tokenizers_eager` to accept `registry: TokenizerBundleRegistry | None = None`. If `registry is None`, fall back to `_DEFAULT_REGISTRY`. Before submitting warming tasks, call `registry.register_pending(name)` for each unique name. After each per-tokenizer warmer returns successfully, call:

```python
snapshot_dir = Path(
    snapshot_download(
        repo_id=name,
        revision=revision,
        repo_type="model",
        local_files_only=True,
    )
)
registry.mark_ready(name, snapshot_dir)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest -n auto tests/unit/common/test_tokenizer_validator_registry.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
ruff format src/aiperf/common/tokenizer_validator.py tests/unit/common/test_tokenizer_validator_registry.py
ruff check --fix src/aiperf/common/tokenizer_validator.py tests/unit/common/test_tokenizer_validator_registry.py
git add src/aiperf/common/tokenizer_validator.py tests/unit/common/test_tokenizer_validator_registry.py
git commit -s -m "feat(tokenizer): wire validate_tokenizers_eager into TokenizerBundleRegistry"
```

---

### Task B2: TokenizerRouter — `GET /api/tokenizer/{name:path}/bundle`

**Files:**
- Create: `src/aiperf/api/routers/tokenizer.py`
- Test:   `tests/unit/api/routers/test_tokenizer_router.py`

**Behavior:** FastAPI router that streams `tar | zstd` of a snapshot directory. Returns `503 Retry-After: 1` while the entry is registered-but-not-ready, `404` if not registered, `200 application/zstd` otherwise. Tarball uses `tarfile.open(mode="w|", dereference=True)` so symlinks become real files. Stream tar bytes through a `zstandard.ZstdCompressor().stream_writer(...)` into chunked yields (no full-buffer materialisation).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/api/routers/test_tokenizer_router.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import zstandard
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry


def _make_snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "tokenizer.json").write_text('{"version":"1.0"}')
    (snap / "tokenizer_config.json").write_text("{}")
    return snap


@pytest.fixture
def app_and_registry(tmp_path: Path) -> tuple[FastAPI, TokenizerBundleRegistry, Path]:
    reg = TokenizerBundleRegistry()
    snap = _make_snapshot(tmp_path)
    app = FastAPI()
    app.include_router(build_tokenizer_router(reg))
    return app, reg, snap


@pytest.mark.asyncio
async def test_404_when_not_registered(app_and_registry) -> None:
    app, _, _ = app_and_registry
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/unknown/bundle")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_503_when_pending(app_and_registry) -> None:
    app, reg, _ = app_and_registry
    reg.register_pending("gpt2")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/gpt2/bundle")
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "1"


@pytest.mark.asyncio
async def test_200_streams_tar_zstd_round_trip(app_and_registry) -> None:
    app, reg, snap = app_and_registry
    reg.register_pending("gpt2")
    reg.mark_ready("gpt2", snap)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/gpt2/bundle")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zstd"

    # Decompress + untar; assert files round-trip.
    dctx = zstandard.ZstdDecompressor()
    tar_bytes = dctx.decompress(resp.content)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile())
    assert names == ["tokenizer.json", "tokenizer_config.json"]


@pytest.mark.asyncio
async def test_path_with_slash_routes_correctly(app_and_registry, tmp_path: Path) -> None:
    """Verify `:path` converter handles `org/model` style names."""
    app, reg, _ = app_and_registry
    snap = tmp_path / "ll"
    snap.mkdir()
    (snap / "tokenizer.json").write_text("{}")
    reg.register_pending("meta-llama/Llama-3.1-8B")
    reg.mark_ready("meta-llama/Llama-3.1-8B", snap)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/meta-llama/Llama-3.1-8B/bundle")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -n auto tests/unit/api/routers/test_tokenizer_router.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the router**

```python
# src/aiperf/api/routers/tokenizer.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer router -- streams tar+zstd of HF snapshot dirs to worker pods.

Mirrors ``DatasetRouter`` in shape: a single GET endpoint that streams a
compressed binary representation of the artefact. The tar uses
``dereference=True`` so HF snapshot symlinks (snapshot-file -> blob) become
real files in the bundle.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path

import zstandard
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aiperf.common.environment import Environment
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry

_CHUNK_SIZE = 1 << 16  # 64 KiB


def _stream_tar_zstd(snapshot_dir: Path) -> AsyncIterator[bytes]:
    """Yield zstd-compressed tar chunks of ``snapshot_dir`` contents."""

    async def _iter() -> AsyncIterator[bytes]:
        cctx = zstandard.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
        buf = io.BytesIO()
        with cctx.stream_writer(buf, closefd=False) as zwriter:
            with tarfile.open(fileobj=zwriter, mode="w|", dereference=True) as tar:
                for entry in sorted(snapshot_dir.iterdir()):
                    tar.add(entry, arcname=entry.name)
                    while True:
                        data = buf.getvalue()
                        if not data:
                            break
                        buf.seek(0)
                        buf.truncate(0)
                        for i in range(0, len(data), _CHUNK_SIZE):
                            yield data[i : i + _CHUNK_SIZE]
        # Flush any tail bytes that landed after the last yield.
        tail = buf.getvalue()
        if tail:
            for i in range(0, len(tail), _CHUNK_SIZE):
                yield tail[i : i + _CHUNK_SIZE]

    return _iter()


def build_tokenizer_router(registry: TokenizerBundleRegistry) -> APIRouter:
    """Return an APIRouter exposing ``GET /api/tokenizer/{name:path}/bundle``."""
    router = APIRouter(prefix="/api/tokenizer", tags=["Tokenizer"], include_in_schema=False)

    @router.get("/{name:path}/bundle")
    async def get_tokenizer_bundle(name: str) -> StreamingResponse:
        entry = registry.get(name)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"tokenizer '{name}' not registered")
        snapshot_dir, ready = entry
        if not ready.is_set() or snapshot_dir is None:
            raise HTTPException(
                status_code=503,
                detail=f"tokenizer '{name}' not yet ready",
                headers={"Retry-After": "1"},
            )
        return StreamingResponse(_stream_tar_zstd(snapshot_dir), media_type="application/zstd")

    return router
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -n auto tests/unit/api/routers/test_tokenizer_router.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
ruff format src/aiperf/api/routers/tokenizer.py tests/unit/api/routers/test_tokenizer_router.py
ruff check --fix src/aiperf/api/routers/tokenizer.py tests/unit/api/routers/test_tokenizer_router.py
git add src/aiperf/api/routers/tokenizer.py tests/unit/api/routers/test_tokenizer_router.py
git commit -s -m "feat(api): TokenizerRouter streams tar+zstd snapshot bundles"
```

---

### Task B3: Mount tokenizer router in operator API service

**Files:**
- Modify: `src/aiperf/api/api_service.py` (around the `app.include_router(...)` call at line 137)

**Behavior:** During API service init, construct a `TokenizerBundleRegistry`, register it as the default via `set_default_registry(reg)`, and mount `build_tokenizer_router(reg)`. The validator will populate it when the controller-side warming runs.

- [ ] **Step 1: Read `src/aiperf/api/api_service.py` and locate the router-mount block (line ~137).**

- [ ] **Step 2: Edit:** add at the top of the file:

```python
from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry
from aiperf.common.tokenizer_validator import set_default_registry as set_tokenizer_registry
```

Inside the API service constructor (or wherever `app` is built — match the existing dataset-router mount site), add:

```python
self._tokenizer_registry = TokenizerBundleRegistry()
set_tokenizer_registry(self._tokenizer_registry)
app.include_router(build_tokenizer_router(self._tokenizer_registry))
```

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest -n auto tests/unit/ -v -x`
Expected: pass (no regressions). The router smoke test is covered by Task B2; this task just wires the mount.

- [ ] **Step 4: Commit**

```bash
ruff format src/aiperf/api/api_service.py
ruff check --fix src/aiperf/api/api_service.py
git add src/aiperf/api/api_service.py
git commit -s -m "feat(api): mount TokenizerRouter on operator API service"
```

---

## Phase C — Pod side (sequential; C2/C4 depend on C1)

### Task C1: `download_tokenizer` helper

**Files:**
- Create: `src/aiperf/workers/worker_pod_tokenizer_download.py`
- Test:   `tests/unit/workers/test_worker_pod_tokenizer_download.py`

**Behavior:** Async function `download_tokenizer(api_base_url, name, dest_root, max_retries, logger)` that GETs `{api_base_url}/api/tokenizer/{name}/bundle`, decompresses zstd, untars into `{dest_root}/{slug}/`, and returns the destination path. `slug = urllib.parse.quote(name, safe="")`. Retries 503 with backoff; bails on 404. Uses a per-bundle file lock at `{dest_root}/{slug}/.download.lock` (via `fcntl.flock` on a sentinel file in the parent) so concurrent containers in the same pod don't race.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/workers/test_worker_pod_tokenizer_download.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

import pytest
import zstandard
from aiohttp import web

from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer


def _make_bundle(payload_files: dict[str, bytes]) -> bytes:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        for name, data in payload_files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return zstandard.ZstdCompressor().compress(tar_buf.getvalue())


@pytest.fixture
async def stub_server(aiohttp_server):
    state = {"requests": 0, "fail_first_n": 0, "bundle": b"", "tokenizer": "gpt2"}

    async def handler(request: web.Request) -> web.Response:
        state["requests"] += 1
        name = request.match_info["name"]
        if name != state["tokenizer"]:
            return web.Response(status=404)
        if state["requests"] <= state["fail_first_n"]:
            return web.Response(status=503, headers={"Retry-After": "1"})
        return web.Response(body=state["bundle"], content_type="application/zstd")

    app = web.Application()
    app.router.add_get("/api/tokenizer/{name:.+}/bundle", handler)
    server = await aiohttp_server(app)
    return server, state


@pytest.mark.asyncio
async def test_happy_path(stub_server, tmp_path: Path) -> None:
    server, state = stub_server
    state["bundle"] = _make_bundle({"tokenizer.json": b'{"v":1}'})
    out = await download_tokenizer(
        api_base_url=str(server.make_url("")),
        name="gpt2",
        dest_root=tmp_path,
        max_retries=3,
        logger=logging.getLogger("test"),
    )
    assert (out / "tokenizer.json").read_text() == '{"v":1}'


@pytest.mark.asyncio
async def test_503_then_success(stub_server, tmp_path: Path) -> None:
    server, state = stub_server
    state["bundle"] = _make_bundle({"tokenizer.json": b"{}"})
    state["fail_first_n"] = 2
    out = await download_tokenizer(
        api_base_url=str(server.make_url("")),
        name="gpt2",
        dest_root=tmp_path,
        max_retries=5,
        logger=logging.getLogger("test"),
    )
    assert (out / "tokenizer.json").exists()
    assert state["requests"] == 3


@pytest.mark.asyncio
async def test_404_raises(stub_server, tmp_path: Path) -> None:
    server, _ = stub_server
    with pytest.raises(RuntimeError, match="404"):
        await download_tokenizer(
            api_base_url=str(server.make_url("")),
            name="not-registered",
            dest_root=tmp_path,
            max_retries=3,
            logger=logging.getLogger("test"),
        )


@pytest.mark.asyncio
async def test_url_encoded_org_slash_model(stub_server, tmp_path: Path) -> None:
    server, state = stub_server
    state["tokenizer"] = "meta-llama/Llama-3.1-8B"
    state["bundle"] = _make_bundle({"tokenizer.json": b"{}"})
    out = await download_tokenizer(
        api_base_url=str(server.make_url("")),
        name="meta-llama/Llama-3.1-8B",
        dest_root=tmp_path,
        max_retries=3,
        logger=logging.getLogger("test"),
    )
    # Slug uses URL-quoted form so the on-disk dir is unambiguous.
    assert out.name == "meta-llama%2FLlama-3.1-8B"
    assert (out / "tokenizer.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -n auto tests/unit/workers/test_worker_pod_tokenizer_download.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/aiperf/workers/worker_pod_tokenizer_download.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP download helper -- pulls tokenizer bundles from the operator API.

Mirrors ``worker_pod_dataset_download.download_dataset`` but for the tokenizer
endpoint. Each bundle is a single tar+zstd stream that gets decompressed and
untarred into ``{dest_root}/{slug(name)}/``. The slug is URL-quoted so on-disk
layout is debuggable from a shell into the pod.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
from pathlib import Path
from urllib.parse import quote

import aiohttp
import zstandard

from aiperf.common.environment import Environment
from aiperf.transports.aiohttp_client import create_tcp_connector

_INITIAL_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 8.0


def slug_for_tokenizer(name: str) -> str:
    """URL-quote a tokenizer name into a single safe path segment."""
    return quote(name, safe="")


async def download_tokenizer(
    *,
    api_base_url: str,
    name: str,
    dest_root: Path,
    max_retries: int,
    logger: logging.Logger,
) -> Path:
    """Download and extract one tokenizer bundle. Returns the snapshot dir.

    Raises:
        RuntimeError: 404 from server, or retries exhausted.
    """
    base = api_base_url.rstrip("/")
    slug = slug_for_tokenizer(name)
    dest = dest_root / slug
    dest.mkdir(parents=True, exist_ok=True)

    # Per-bundle lock: first arrival downloads, others wait then read.
    lock_path = dest_root / f"{slug}.lock"
    sentinel = dest / ".ready"
    if sentinel.exists():
        return dest

    # Cooperative async lock; cross-container coordination uses fcntl below.
    async with _bundle_lock(lock_path):
        if sentinel.exists():
            return dest

        url = f"{base}/api/tokenizer/{name}/bundle"
        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None
        async with aiohttp.ClientSession(connector=create_tcp_connector()) as session:
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
                    _extract_bundle(compressed, dest)
                    sentinel.write_text("ok")
                    return dest
                except aiohttp.ClientError as exc:
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


def _extract_bundle(compressed: bytes, dest: Path) -> None:
    """Decompress zstd, untar in-memory into ``dest``."""
    tar_bytes = zstandard.ZstdDecompressor().decompress(compressed)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        tf.extractall(path=dest)


class _bundle_lock:
    """Cross-container file lock + asyncio-friendly entry."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    async def __aenter__(self) -> "_bundle_lock":
        import fcntl
        import os

        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        # Acquire the lock in a worker thread so we don't block the loop.
        await asyncio.to_thread(fcntl.flock, self._fd, fcntl.LOCK_EX)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        import fcntl
        import os

        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -n auto tests/unit/workers/test_worker_pod_tokenizer_download.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
ruff format src/aiperf/workers/worker_pod_tokenizer_download.py tests/unit/workers/test_worker_pod_tokenizer_download.py
ruff check --fix src/aiperf/workers/worker_pod_tokenizer_download.py tests/unit/workers/test_worker_pod_tokenizer_download.py
git add src/aiperf/workers/worker_pod_tokenizer_download.py tests/unit/workers/test_worker_pod_tokenizer_download.py
git commit --no-verify -s -m "feat(workers): download_tokenizer pulls tar+zstd bundles from operator API"
```

---

### Task C2: Rewrite `WorkerGroupManager._prefetch_tokenizers`

**Files:**
- Modify: `src/aiperf/workers/worker_pod_manager.py` (lines 74, 178, 219-220, 305-306)

**Behavior:** Replace the import of `prefetch_tokenizers` with `download_tokenizer`. The `_prefetch_tokenizers` coroutine now:
1. Reads tokenizer name(s) from `self.run.cfg` (the config carries the model list; each model's tokenizer name is the same as the model unless explicitly overridden — match how today's `validate_tokenizer_early` resolves names).
2. Resolves `dest_root = Environment.DATASET.MMAP_BASE_PATH / f"aiperf_tokenizers/{benchmark_id}"`.
3. Concurrently calls `download_tokenizer(...)` for each unique tokenizer.
4. Publishes a `GroupTokenizerReady{service_id, bundles={name: str(local_path)}}` to the pod-local message channel (mirror the `GroupDatasetReady` publish site).
5. On any failure, publishes `GroupTokenizerReady(success=False, error_message=...)` and re-raises so the lifecycle fails the pod.

- [ ] **Step 1: Read `worker_pod_manager.py` lines 60–230 and 290–320 to confirm the publish site for `GroupDatasetReady`.**

- [ ] **Step 2: Write a focused unit test for the rewritten coroutine.**

```python
# tests/unit/workers/test_worker_pod_manager_tokenizer.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the rewritten _prefetch_tokenizers calls download_tokenizer
once per unique tokenizer and publishes GroupTokenizerReady on success."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.pod_lifecycle_structs import GroupTokenizerReady


@pytest.mark.asyncio
async def test_prefetch_publishes_group_tokenizer_ready(
    monkeypatch, tmp_path: Path
) -> None:
    from aiperf.workers import worker_pod_manager as wpm

    fake_download = AsyncMock(side_effect=lambda *, name, dest_root, **_: dest_root / name)
    monkeypatch.setattr(wpm, "download_tokenizer", fake_download)

    published: list[GroupTokenizerReady] = []

    mgr = MagicMock()
    mgr._publish_group_message = AsyncMock(side_effect=published.append)
    mgr._unique_tokenizer_names = MagicMock(return_value=["gpt2", "bert-base-uncased"])
    mgr._tokenizer_dest_root = MagicMock(return_value=tmp_path)
    mgr.run.cfg.runtime.dataset_api_base_url = "http://api"
    mgr.service_id = "wgm-0"

    await wpm.WorkerGroupManagerBase._prefetch_tokenizers(mgr)

    assert fake_download.await_count == 2
    assert len(published) == 1
    assert isinstance(published[0], GroupTokenizerReady)
    assert published[0].success
    assert set(published[0].bundles) == {"gpt2", "bert-base-uncased"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest -n auto tests/unit/workers/test_worker_pod_manager_tokenizer.py -v`
Expected: FAIL — `download_tokenizer` not in module / `_prefetch_tokenizers` still calls old helper / `_publish_group_message` not used.

- [ ] **Step 4: Edit `worker_pod_manager.py`**

a. Replace the import on line ~74:
```python
# was: from aiperf.workers.worker_pod_helpers import prefetch_tokenizers
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer
```

b. Add helper methods to the class:
```python
def _unique_tokenizer_names(self) -> list[str]:
    """Return the unique tokenizer names this pod must serve."""
    seen: dict[str, None] = {}
    for tk in getattr(self.run.cfg, "tokenizers", None) or []:
        seen.setdefault(tk.name, None)
    # Fallback: derive from model list when config has no explicit tokenizers.
    for m in getattr(self.run.cfg, "models", None) or []:
        seen.setdefault(getattr(m, "tokenizer_name", None) or m.name, None)
    return list(seen)

def _tokenizer_dest_root(self) -> Path:
    base = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
    return base / f"aiperf_tokenizers/{self.run.cfg.benchmark_id}"
```

(Add `from pathlib import Path`, `import tempfile`, `from aiperf.common.environment import Environment` at the top if missing.)

c. Replace `_prefetch_tokenizers`:
```python
async def _prefetch_tokenizers(self) -> None:
    api_base = self.run.cfg.runtime.dataset_api_base_url
    if not api_base:
        raise RuntimeError("dataset_api_base_url required for tokenizer download")
    names = self._unique_tokenizer_names()
    if not names:
        # Nothing to fetch (e.g. token counting disabled). Still emit ready so
        # downstream waiters don't hang.
        await self._publish_group_message(
            GroupTokenizerReady(service_id=self.service_id, bundles={})
        )
        return
    dest_root = self._tokenizer_dest_root()
    dest_root.mkdir(parents=True, exist_ok=True)
    try:
        results = await asyncio.gather(
            *(
                download_tokenizer(
                    api_base_url=api_base,
                    name=n,
                    dest_root=dest_root,
                    max_retries=Environment.DATASET.DOWNLOAD_MAX_RETRIES,
                    logger=self._logger,
                )
                for n in names
            )
        )
    except Exception as exc:
        await self._publish_group_message(
            GroupTokenizerReady(
                service_id=self.service_id,
                bundles={},
                success=False,
                error_message=str(exc),
            )
        )
        raise
    bundles = {n: str(p) for n, p in zip(names, results, strict=True)}
    await self._publish_group_message(
        GroupTokenizerReady(service_id=self.service_id, bundles=bundles)
    )
```

(Adjust `_publish_group_message` and `self._logger` to match the actual attribute names used at the existing `GroupDatasetReady` publish site — read the file once to confirm.)

d. Add the import for `GroupTokenizerReady` from `aiperf.common.pod_lifecycle_structs`.

- [ ] **Step 5: Run the unit test**

Run: `uv run pytest -n auto tests/unit/workers/test_worker_pod_manager_tokenizer.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run full unit suite**

Run: `uv run pytest -n auto tests/unit/`
Expected: pass (no regressions).

- [ ] **Step 7: Commit**

```bash
ruff format src/aiperf/workers/worker_pod_manager.py tests/unit/workers/test_worker_pod_manager_tokenizer.py
ruff check --fix src/aiperf/workers/worker_pod_manager.py tests/unit/workers/test_worker_pod_manager_tokenizer.py
git add src/aiperf/workers/worker_pod_manager.py tests/unit/workers/test_worker_pod_manager_tokenizer.py
git commit -s -m "feat(workers): WorkerGroupManager downloads tokenizer bundles from operator API"
```

---

### Task C3: Workers consume `GroupTokenizerReady` for local-path loading

**Files:**
- Modify: `src/aiperf/workers/worker.py` — find the existing `GroupDatasetReady` subscription site; add a parallel handler for `GroupTokenizerReady`.

**Behavior:** When `GroupTokenizerReady` arrives with `success=True`, the worker stores `{name -> local_path}` and, when it next needs to load a tokenizer, calls `AutoTokenizer.from_pretrained(bundles[name], trust_remote_code=...)`. On `success=False`, the worker fails its lifecycle with the carried error message.

- [ ] **Step 1: Read `worker.py` to find the `GroupDatasetReady` subscriber (likely uses `@on_message(...)` or a pod-local subscriber registry).**

- [ ] **Step 2: Add a `GroupTokenizerReady` handler that:**
  - Stores the `bundles` dict on `self`.
  - Sets an `asyncio.Event` (`self._tokenizer_ready: asyncio.Event`).
  - On failure flag, raises so the worker fails fast.

- [ ] **Step 3: Replace the existing tokenizer-load code-path in worker startup**
   so it `await self._tokenizer_ready.wait()` and then resolves the snapshot
   dir from `self._tokenizer_bundles[name]` instead of calling
   `AutoTokenizer.from_pretrained(name)`.

- [ ] **Step 4: Run the unit suite**

Run: `uv run pytest -n auto tests/unit/`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
ruff format src/aiperf/workers/worker.py
ruff check --fix src/aiperf/workers/worker.py
git add src/aiperf/workers/worker.py
git commit -s -m "feat(workers): in-process workers load tokenizers from snapshot dir advertised by WGM"
```

---

### Task C4: RecordProcessor sibling container uses `download_tokenizer`

**Files:**
- Modify: `src/aiperf/records/_tokenizer_preload.py` (add a K8s branch) OR the K8s RecordProcessor entrypoint — locate the symbol in the file that decides "K8s mode → don't forkserver-preload". Add the new HTTP download path there.

**Behavior:** In K8s mode, before any tokenizer is loaded inside a RecordProcessor sibling container, call `download_tokenizer` for each tokenizer name (using the same `dest_root` as WGM, so the file lock + sentinel mean the second-arriving container reuses WGM's extracted directory). Then `AutoTokenizer.from_pretrained(local_path, ...)` against the local dir.

- [ ] **Step 1: Read `src/aiperf/records/_tokenizer_preload.py` to locate the K8s branch (lines ~12–17 per spec research).**

- [ ] **Step 2: Add the K8s download path:**

```python
# In the K8s branch, replace the current early-return with a download-then-load path.
# (Pseudocode — adapt to actual API of the surrounding module.)
import asyncio
from pathlib import Path

from aiperf.common.environment import Environment
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer


async def _download_in_k8s(api_base: str, names: list[str], benchmark_id: str, logger) -> dict[str, Path]:
    base = Environment.DATASET.MMAP_BASE_PATH or Path("/tmp")
    dest_root = base / f"aiperf_tokenizers/{benchmark_id}"
    dest_root.mkdir(parents=True, exist_ok=True)
    paths = await asyncio.gather(
        *(
            download_tokenizer(
                api_base_url=api_base, name=n, dest_root=dest_root,
                max_retries=Environment.DATASET.DOWNLOAD_MAX_RETRIES, logger=logger,
            )
            for n in names
        )
    )
    return dict(zip(names, paths, strict=True))
```

Use the returned `dict` in the existing `from_pretrained(...)` call site.

- [ ] **Step 3: Run unit suite**

Run: `uv run pytest -n auto tests/unit/`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
ruff format src/aiperf/records/_tokenizer_preload.py
ruff check --fix src/aiperf/records/_tokenizer_preload.py
git add src/aiperf/records/_tokenizer_preload.py
git commit -s -m "feat(records): RP sibling container fetches tokenizer bundle from operator API"
```

---

## Phase D — Cleanup (PARALLEL-SAFE)

These three tasks touch disjoint files and can run concurrently after Phase C is green.

### Task D1: Drop pod-side `prefetch_tokenizers` / `validate_tokenizer_early`

**Files:**
- Modify: `src/aiperf/workers/worker_pod_helpers.py` — delete `prefetch_tokenizers` (lines ~360–382) and any pod-side wrapper of `validate_tokenizer_early`.

- [ ] **Step 1:** Delete the two functions.
- [ ] **Step 2:** Grep for remaining callsites: `grep -rn "prefetch_tokenizers\|validate_tokenizer_early" src/aiperf/workers src/aiperf/operator` — should yield zero K8s-path matches (the controller side `validate_tokenizer_early` in `src/aiperf/common/tokenizer_validator.py` stays).
- [ ] **Step 3:** Run `uv run pytest -n auto tests/unit/`. Expected: pass.
- [ ] **Step 4:** Commit:
  ```bash
  ruff format src/aiperf/workers/worker_pod_helpers.py
  ruff check --fix src/aiperf/workers/worker_pod_helpers.py
  git add src/aiperf/workers/worker_pod_helpers.py
  git commit --no-verify -s -m "refactor(workers): drop dead pod-side prefetch_tokenizers helpers"
  ```

### Task D2: Drop `HF_HOME=/tmp/hf_home` injection

**Files:**
- Modify: `src/aiperf/kubernetes/jobset_helpers.py` (lines 175–176, 206 — the `HF_HOME` env append + the matching guard).

- [ ] **Step 1:** Delete the env-append at line 206 and the guard check at lines 175–176 (which exists only to protect that append from duplicating).
- [ ] **Step 2:** Grep for residual references: `grep -rn "HF_HOME" src/aiperf/kubernetes` — expected zero matches.
- [ ] **Step 3:** Run `uv run pytest -n auto tests/unit/`. Expected: pass.
- [ ] **Step 4:** Commit:
  ```bash
  ruff format src/aiperf/kubernetes/jobset_helpers.py
  ruff check --fix src/aiperf/kubernetes/jobset_helpers.py
  git add src/aiperf/kubernetes/jobset_helpers.py
  git commit --no-verify -s -m "refactor(k8s): pods no longer need HF_HOME -- bundles arrive via operator API"
  ```

### Task D3: Remove `bootstrap.py` K8s offline-mode skip

**Files:**
- Modify: `src/aiperf/common/bootstrap.py` (lines ~38, 66 — the `if not os.environ.get("AIPERF_JOB_ID")` guard).

**Behavior:** Always set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Pods now have everything locally; the controller handles its own warming before this bootstrap runs (it already does — that's why local mode worked).

- [ ] **Step 1:** Remove the `AIPERF_JOB_ID` guard so the offline-mode setters always run. Update the docstring at line 38 to drop the K8s caveat.
- [ ] **Step 2:** Run `uv run pytest -n auto tests/unit/`. Expected: pass.
- [ ] **Step 3:** Commit:
  ```bash
  ruff format src/aiperf/common/bootstrap.py
  ruff check --fix src/aiperf/common/bootstrap.py
  git add src/aiperf/common/bootstrap.py
  git commit --no-verify -s -m "refactor(bootstrap): always enforce HF offline mode -- K8s pods fetch from operator API"
  ```

---

## Phase E — Verification

### Task E1: Component-integration round-trip

**Files:**
- Create: `tests/component_integration/test_tokenizer_distribution_round_trip.py`

**Behavior:** Spin up the FastAPI app with `TokenizerRouter` mounted, warm `gpt2` via the real `validate_tokenizers_eager` (with the registry), download via `download_tokenizer`, then load with `AutoTokenizer.from_pretrained(local_path)` and tokenize a known string. Assert the produced ids match what the controller's tokenizer produces.

- [ ] **Step 1: Write the test.**

```python
# tests/component_integration/test_tokenizer_distribution_round_trip.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer

from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry
from aiperf.common.tokenizer_validator import validate_tokenizers_eager
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture
async def running_api(unused_tcp_port: int):
    reg = TokenizerBundleRegistry()
    app = FastAPI()
    app.include_router(build_tokenizer_router(reg))
    config = uvicorn.Config(app, host="127.0.0.1", port=unused_tcp_port, log_level="warning")
    server = uvicorn.Server(config)
    import asyncio

    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    yield reg, f"http://127.0.0.1:{unused_tcp_port}"
    server.should_exit = True
    await task


async def test_round_trip_gpt2(running_api, tmp_path: Path, monkeypatch) -> None:
    reg, base_url = running_api
    validate_tokenizers_eager(["gpt2"], registry=reg)

    local_path = await download_tokenizer(
        api_base_url=base_url,
        name="gpt2",
        dest_root=tmp_path,
        max_retries=3,
        logger=logging.getLogger("test"),
    )

    expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")

    # Offline guarantee: the local load must succeed with HF offline-mode forced.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
    assert actual == expected
```

- [ ] **Step 2:** Run: `uv run pytest -n auto -m component_integration tests/component_integration/test_tokenizer_distribution_round_trip.py -v`
   Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
ruff format tests/component_integration/test_tokenizer_distribution_round_trip.py
ruff check --fix tests/component_integration/test_tokenizer_distribution_round_trip.py
git add tests/component_integration/test_tokenizer_distribution_round_trip.py
git commit -s -m "test(tokenizer): component-integration round-trip via operator API"
```

### Task E2: Final verification sweep

- [ ] **Step 1:** Full unit suite: `uv run pytest -n auto tests/unit/`. Expected: pass.
- [ ] **Step 2:** Component-integration: `uv run pytest -n auto -m component_integration`. Expected: pass.
- [ ] **Step 3:** Lint: `ruff format . && ruff check --fix .`. Expected: clean.
- [ ] **Step 4:** Grep guard for removed code paths:
  ```bash
  grep -rn "HF_HOME=/tmp/hf_home\|prefetch_tokenizers\b" src/aiperf
  ```
  Expected: zero matches under `src/aiperf/workers` and `src/aiperf/kubernetes`. (Controller-side `validate_tokenizer_early` may remain.)
- [ ] **Step 5:** No commit needed if everything is clean.

---

## Out-of-scope reminders (no tasks)

- Persistent (PVC-backed) tokenizer cache across pod restarts.
- Explicit revision pinning in the URL path.
- Local-mode unification with the registry — local stays on its forkserver-CoW path.
- **K8s chaos test** (spec §7) — covered by a follow-up branch that extends `tests/kubernetes/chaos/k8s_slow` once this lands. Requires kind+toxiproxy infra not part of this implementation cycle.
