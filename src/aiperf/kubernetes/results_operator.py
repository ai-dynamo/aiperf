# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-PVC result retrieval flows for jobs and sweep aggregates."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiofiles
import aiohttp
import orjson

from aiperf.common.environment import Environment
from aiperf.common.results_markers import READY_MARKER_NAME
from aiperf.kubernetes.client import find_operator_pod
from aiperf.kubernetes.console import (
    _human_size,
    print_error,
    print_file_table,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.operator.environment import OperatorEnvironment

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


RESULTS_SERVER_PORT = OperatorEnvironment.RESULTS.SERVER_PORT
_REDIRECT_STATUSES = {301, 302, 307, 308}


# ============================================================
# Shared download primitives
# ============================================================


if "_JobDownloadOutcome" not in globals():

    @dataclass(frozen=True, slots=True)
    class _JobDownloadOutcome:
        """Result of downloading every advertised file for one run."""

        downloaded: list[tuple[str, int]]
        """(display name, size in bytes) for each file that landed on disk."""

        failed: list[str]
        """Display names the server advertised but did not deliver."""

        @property
        def complete(self) -> bool:
            """True when every advertised file was retrieved."""
            return not self.failed


def _is_refused_name(display_name: str) -> bool:
    """True when we decline to write this name, regardless of the server.

    Dot-files (the results-ready marker among them), absolute paths and
    parent traversals are refused by policy. They are advertised in listings
    but never downloaded, so they are skips rather than failures.
    """
    normalized = Path(display_name)
    leaf = normalized.name
    return (
        not leaf
        or leaf.startswith(".")
        or normalized.is_absolute()
        or ".." in normalized.parts
    )


def _get_no_redirects(
    session: aiohttp.ClientSession,
    url: str,
    **kwargs: object,
) -> object:
    """Start a GET request without redirects, including test-double fallback."""
    try:
        return session.get(url, allow_redirects=False, **kwargs)
    except TypeError as e:
        if "allow_redirects" not in str(e):
            raise
        return session.get(url, **kwargs)


def _get_with_request_timeout(
    session: aiohttp.ClientSession,
    url: str,
) -> object:
    """Start a short result API request with its configured timeout."""
    timeout = aiohttp.ClientTimeout(
        total=K8sEnvironment.RESULTS.REQUEST_TIMEOUT_SECONDS
    )
    try:
        return session.get(url, timeout=timeout)
    except TypeError as e:
        if "timeout" not in str(e):
            raise
        return session.get(url)


async def _download_and_decompress(
    response: aiohttp.ClientResponse,
    dest_path: Path,
    content_encoding: str,
) -> None:
    """Stream an optionally encoded response to ``dest_path`` atomically."""
    import zlib

    if content_encoding == "zstd":
        import zstandard

        decompressor = zstandard.ZstdDecompressor().decompressobj()
    elif content_encoding == "gzip":
        decompressor = zlib.decompressobj(wbits=31)
    else:
        decompressor = None

    temp_path = dest_path.with_name(f".{dest_path.name}.{uuid.uuid4().hex}.tmp")
    replaced = False
    try:
        async with aiofiles.open(temp_path, "wb") as file:
            async for chunk in response.content.iter_chunked(
                Environment.COMPRESSION.CHUNK_SIZE
            ):
                if decompressor is not None:
                    chunk = decompressor.decompress(chunk)
                if chunk:
                    await file.write(chunk)
            if decompressor is not None:
                remaining = decompressor.flush()
                if remaining:
                    await file.write(remaining)
        await asyncio.to_thread(os.replace, temp_path, dest_path)
        replaced = True
    finally:
        if not replaced:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)


async def _verify_operator_health(api_base: str) -> bool:
    """Return whether the operator results server passes its health endpoint."""
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(
        total=K8sEnvironment.RESULTS.CONTROL_REQUEST_TIMEOUT_SECONDS
    )
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            async with session.get(f"{api_base}/healthz") as response:
                if response.status != 200:
                    print_error("Operator results server not healthy")
                    return False
        except aiohttp.ClientError as e:
            print_error(f"Could not connect to operator results server: {e}")
            return False
    return True


@asynccontextmanager
async def _operator_results_api(
    api: ApiClient,
    *,
    operator_namespace: str,
    local_port: int,
    results_port: int,
    kubeconfig: str | None,
    kube_context: str | None,
) -> AsyncIterator[str | None]:
    """Yield a port-forwarded, health-checked results-server base URL.

    Yields ``None`` when the operator pod is missing or unhealthy so callers
    report failure without distinguishing which of the two happened.
    """
    pod_info = await find_operator_pod(api, namespace=operator_namespace)
    if not pod_info:
        print_error("Operator pod not found")
        print_info(f"Looked in namespace: {operator_namespace}")
        yield None
        return

    pod_name, pod_phase = pod_info
    print_info(f"Found operator pod: {pod_name} (status: {pod_phase})")

    async with port_forward_with_status(
        operator_namespace,
        pod_name,
        local_port,
        remote_port=results_port,
        verify_api=False,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    ) as port:
        api_base = f"http://localhost:{port}"
        if not await _verify_operator_health(api_base):
            yield None
            return
        yield api_base


def _download_session() -> aiohttp.ClientSession:
    """Open a session sized for bulk artifact downloads, decompressed by us."""
    from aiperf.transports.aiohttp_client import create_tcp_connector

    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=K8sEnvironment.RESULTS.DOWNLOAD_TIMEOUT_SECONDS
        ),
        connector=create_tcp_connector(),
        auto_decompress=False,
    )


async def _collect_downloads(
    available: list[dict],
    download: Callable[[dict], Awaitable[tuple[str, int] | None]],
) -> _JobDownloadOutcome:
    """Download every advertised entry, recording rather than raising failures.

    A per-file failure is reported in the outcome instead of discarding the
    files that did land, so a mostly-complete directory survives while the
    caller can still refuse to claim success.
    """
    downloaded: list[tuple[str, int]] = []
    failed: list[str] = []
    for file_info in available:
        result = await download(file_info)
        if result is not None:
            downloaded.append(result)
        elif not _is_refused_name(str(file_info.get("name", ""))):
            failed.append(str(file_info.get("name", "<unnamed>")))
    return _JobDownloadOutcome(downloaded=downloaded, failed=failed)


# ============================================================
# Job results
# ============================================================


def _result_base_url(
    api_base: str, namespace: str, job_id: str, run: str | None
) -> str:
    """Return the results URL prefix for a job, pinned to a run when given."""
    base = f"{api_base}/api/v1/results/{quote(namespace, safe='')}/{quote(job_id, safe='')}"
    if run is not None:
        return f"{base}/runs/{quote(run, safe='')}"
    return base


async def _download_operator_file(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    namespace: str,
    job_id: str,
    file_info: dict,
    output_dir: Path,
    run: str | None = None,
) -> tuple[str, int] | None:
    """Download a single file from the operator results server."""
    display_name = file_info["name"]
    if _is_refused_name(display_name):
        print_warning(f"Refusing unsafe filename: {display_name!r}")
        return None
    quoted_name = quote(display_name, safe="/")
    base_url = _result_base_url(api_base, namespace, job_id, run)
    download_url = f"{base_url}/{quoted_name}"
    headers = {"Accept-Encoding": "zstd, gzip, identity"}

    try:
        async with _get_no_redirects(session, download_url, headers=headers) as resp:
            if resp.status == 404:
                print_warning(f"File not found: {display_name}")
                return None
            if resp.status in _REDIRECT_STATUSES:
                print_warning(f"Refusing redirected download for {display_name}")
                return None
            resp.raise_for_status()

            dest_path = output_dir / display_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            content_encoding = resp.headers.get("Content-Encoding", "identity")

            await _download_and_decompress(resp, dest_path, content_encoding)

            file_size = dest_path.stat().st_size
            print_success(f"Downloaded: {display_name} ({_human_size(file_size)})")
            return (display_name, file_size)
    except aiohttp.ClientError as e:
        print_warning(f"Failed to download {display_name}: {e}")
        return None


async def _list_operator_files(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    namespace: str,
    job_id: str,
    run: str | None = None,
) -> list[dict] | None:
    list_url = _result_base_url(api_base, namespace, job_id, run)
    try:
        async with _get_with_request_timeout(session, list_url) as resp:
            if resp.status == 404:
                print_error(f"No results stored for {namespace}/{job_id}")
                return None
            resp.raise_for_status()
            list_data = await resp.json(loads=orjson.loads)
    except aiohttp.ClientError as e:
        print_error(f"Failed to list results: {e}")
        return None

    if not isinstance(list_data, dict):
        print_error("Operator results listing had invalid JSON shape")
        return None
    if list_data.get("ready") is False:
        print_warning(
            f"No result files found for {namespace}/{job_id}: "
            f"{READY_MARKER_NAME} is missing; retry after the run completes"
        )
        return None
    available = list_data.get("files", [])
    if not available:
        print_warning("No result files found")
        return None
    if not isinstance(available, list):
        print_error("Operator results listing 'files' must be a list")
        return None
    for file_info in available:
        if not isinstance(file_info, dict) or not isinstance(
            file_info.get("name"), str
        ):
            print_error("Operator results listing contained an invalid file entry")
            return None
    return available


async def _resolve_operator_run(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    namespace: str,
    job_id: str,
    run: str | None,
) -> str | None:
    if run is not None:
        return run

    runs_url = f"{api_base}/api/v1/results/{namespace}/{job_id}/runs"
    try:
        async with _get_with_request_timeout(session, runs_url) as resp:
            if resp.status == 404:
                print_error(f"No runs found for {namespace}/{job_id}")
                return None
            resp.raise_for_status()
            payload = await resp.json(loads=orjson.loads)
    except aiohttp.ClientError as e:
        print_error(f"Failed to resolve latest run: {e}")
        return None

    if not isinstance(payload, dict):
        print_error(f"Malformed runs response for {namespace}/{job_id}")
        return None
    latest = payload.get("latest_epoch")
    if not isinstance(latest, str) or not latest:
        print_error(f"No latest run available for {namespace}/{job_id}")
        return None
    return latest


async def _download_all_operator_files(
    *,
    api_base: str,
    namespace: str,
    job_id: str,
    output_dir: Path,
    run: str | None = None,
) -> _JobDownloadOutcome | None:
    """List and download every result file for a job.

    Returns ``None`` only when the listing itself could not be obtained.
    """
    async with _download_session() as session:
        resolved_run = await _resolve_operator_run(
            session,
            api_base=api_base,
            namespace=namespace,
            job_id=job_id,
            run=run,
        )
        if resolved_run is None:
            return None

        available = await _list_operator_files(
            session,
            api_base=api_base,
            namespace=namespace,
            job_id=job_id,
            run=resolved_run,
        )
        if available is None:
            return None

        print_step(f"Downloading {len(available)} files...")

        return await _collect_downloads(
            available,
            lambda file_info: _download_operator_file(
                session,
                api_base=api_base,
                namespace=namespace,
                job_id=job_id,
                file_info=file_info,
                output_dir=output_dir,
                run=resolved_run,
            ),
        )


async def retrieve_results_from_operator(
    job_id: str,
    namespace: str,
    output_dir: Path,
    api: ApiClient,
    *,
    local_port: int = 0,
    operator_namespace: str = DEFAULT_OPERATOR_NAMESPACE,
    results_port: int = RESULTS_SERVER_PORT,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
    run: str | None = None,
) -> bool:
    """Retrieve results from the operator results server sidecar."""
    try:
        async with _operator_results_api(
            api,
            operator_namespace=operator_namespace,
            local_port=local_port,
            results_port=results_port,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        ) as api_base:
            if api_base is None:
                return False

            outcome = await _download_all_operator_files(
                api_base=api_base,
                namespace=namespace,
                job_id=job_id,
                output_dir=output_dir,
                run=run,
            )
            if outcome is None:
                return False

            if outcome.downloaded:
                print_file_table(outcome.downloaded)
            if not outcome.complete:
                print_error(
                    f"{len(outcome.failed)} of "
                    f"{len(outcome.downloaded) + len(outcome.failed)} files failed "
                    f"to download: {', '.join(outcome.failed)}"
                )
                if outcome.downloaded:
                    print_info(f"Partial results saved to: {output_dir}")
                return False
            if outcome.downloaded:
                print_success(f"Results saved to: {output_dir}")
                return True
            print_error("No files downloaded")
            return False

    except (TimeoutError, aiohttp.ClientError, OSError, RuntimeError) as e:
        print_error(f"Error connecting to operator: {e!r}")
        return False


# ============================================================
# Sweep aggregate artifacts
# ============================================================


def _sweep_artifacts_base_url(
    api_base: str, namespace: str, sweep_name: str, run: str
) -> str:
    return "/".join(
        [
            f"{api_base}/api/v1/sweeps",
            quote(namespace, safe=""),
            quote(sweep_name, safe=""),
            "epochs",
            quote(run, safe=""),
            "artifacts",
        ]
    )


def _safe_sweep_artifact_path(output_dir: Path, display_name: str) -> Path | None:
    relative = Path(display_name)
    if relative.is_absolute() or not relative.parts:
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    if any(part.startswith(".") for part in relative.parts):
        return None
    return output_dir / relative


async def _list_sweep_operator_files(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    namespace: str,
    sweep_name: str,
    run: str,
) -> list[dict] | None:
    list_url = _sweep_artifacts_base_url(api_base, namespace, sweep_name, run)
    try:
        async with _get_with_request_timeout(session, list_url) as resp:
            if resp.status == 404:
                print_warning(
                    f"No aggregate artifacts stored for sweep {namespace}/{sweep_name} run {run}"
                )
                return None
            resp.raise_for_status()
            list_data = await resp.json(loads=orjson.loads)
    except aiohttp.ClientError as e:
        print_warning(f"Failed to list sweep aggregate artifacts: {e}")
        return None

    if not isinstance(list_data, dict):
        print_warning("Operator sweep artifacts listing had invalid JSON shape")
        return None
    available = list_data.get("files", [])
    if not available:
        print_warning("No sweep aggregate artifact files found")
        return None
    return available


async def _download_sweep_operator_file(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    namespace: str,
    sweep_name: str,
    run: str,
    file_info: dict,
    output_dir: Path,
) -> tuple[str, int] | None:
    display_name = file_info["name"]
    dest_path = _safe_sweep_artifact_path(output_dir, display_name)
    if dest_path is None:
        print_warning(f"Refusing unsafe sweep artifact filename: {display_name!r}")
        return None

    quoted_name = quote(display_name, safe="/")
    download_url = f"{_sweep_artifacts_base_url(api_base, namespace, sweep_name, run)}/{quoted_name}"
    headers = {"Accept-Encoding": "zstd, gzip, identity"}

    try:
        async with _get_no_redirects(session, download_url, headers=headers) as resp:
            if resp.status == 404:
                print_warning(f"Sweep aggregate artifact not found: {display_name}")
                return None
            if resp.status in _REDIRECT_STATUSES:
                print_warning(
                    f"Refusing redirected sweep aggregate artifact for {display_name}"
                )
                return None
            resp.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            content_encoding = resp.headers.get("Content-Encoding", "identity")
            await _download_and_decompress(resp, dest_path, content_encoding)
            file_size = dest_path.stat().st_size
            print_success(f"Downloaded: {display_name} ({_human_size(file_size)})")
            return (display_name, file_size)
    except aiohttp.ClientError as e:
        print_warning(
            f"Failed to download sweep aggregate artifact {display_name}: {e}"
        )
        return None


async def _download_all_sweep_operator_files(
    *,
    api_base: str,
    namespace: str,
    sweep_name: str,
    output_dir: Path,
    run: str,
) -> _JobDownloadOutcome | None:
    """List and download every sweep-aggregate file for one epoch.

    Returns ``None`` only when the listing itself could not be obtained.
    """
    async with _download_session() as session:
        available = await _list_sweep_operator_files(
            session,
            api_base=api_base,
            namespace=namespace,
            sweep_name=sweep_name,
            run=run,
        )
        if available is None:
            return None

        print_step(f"Downloading {len(available)} sweep aggregate files...")

        return await _collect_downloads(
            available,
            lambda file_info: _download_sweep_operator_file(
                session,
                api_base=api_base,
                namespace=namespace,
                sweep_name=sweep_name,
                run=run,
                file_info=file_info,
                output_dir=output_dir,
            ),
        )


async def retrieve_sweep_aggregate_artifacts_from_operator(
    sweep_name: str,
    namespace: str,
    output_dir: Path,
    api: ApiClient,
    *,
    local_port: int = 0,
    operator_namespace: str = DEFAULT_OPERATOR_NAMESPACE,
    results_port: int = RESULTS_SERVER_PORT,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
    run: str,
) -> bool:
    """Download sweep-level aggregate artifacts for a specific sweep epoch."""
    if not run:
        print_warning(
            f"Sweep {namespace}/{sweep_name}: missing sweep epoch for aggregate artifacts"
        )
        return False

    try:
        async with _operator_results_api(
            api,
            operator_namespace=operator_namespace,
            local_port=local_port,
            results_port=results_port,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        ) as api_base:
            if api_base is None:
                return False

            downloaded_files = await _download_all_sweep_operator_files(
                api_base=api_base,
                namespace=namespace,
                sweep_name=sweep_name,
                output_dir=output_dir,
                run=run,
            )
            if downloaded_files is None:
                return False
            if downloaded_files.downloaded:
                print_file_table(downloaded_files.downloaded)
            if not downloaded_files.complete:
                # Report the partial set rather than silently exiting 0: the
                # files that did land are still on disk and useful, but the
                # caller must not claim success.
                print_error(
                    "Sweep aggregate download incomplete; missing "
                    f"{len(downloaded_files.failed)} file(s): "
                    + ", ".join(downloaded_files.failed[:10])
                )
                return False
            if downloaded_files.downloaded:
                return True
            print_warning("No sweep aggregate files downloaded")
            return False
    except (TimeoutError, aiohttp.ClientError, OSError, RuntimeError) as e:
        print_error(
            f"Error connecting to operator for sweep aggregate artifacts: {e!r}"
        )
        return False
