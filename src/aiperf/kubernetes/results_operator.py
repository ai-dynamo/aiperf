# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-PVC result retrieval flow."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiofiles
import aiohttp
import orjson

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
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.kubernetes.results_operator_common import (
    _is_refused_name,
    _JobDownloadOutcome,
)
from aiperf.kubernetes.results_operator_sweeps import (
    _download_all_sweep_operator_files as _download_all_sweep_operator_files,
)
from aiperf.kubernetes.results_operator_sweeps import (
    _download_sweep_operator_file as _download_sweep_operator_file,
)
from aiperf.kubernetes.results_operator_sweeps import (
    _list_sweep_operator_files as _list_sweep_operator_files,
)
from aiperf.kubernetes.results_operator_sweeps import (
    retrieve_sweep_aggregate_artifacts_from_operator as retrieve_sweep_aggregate_artifacts_from_operator,
)
from aiperf.kubernetes.results_sidecar import READY_MARKER_NAME

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


RESULTS_SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SERVER_PORT", "8081"))
_REDIRECT_STATUSES = {301, 302, 307, 308}


def _result_base_url(
    api_base: str, namespace: str, job_id: str, run: str | None
) -> str:
    """Return the results URL prefix for a job, pinned to a run when given."""
    base = f"{api_base}/api/v1/results/{quote(namespace, safe='')}/{quote(job_id, safe='')}"
    if run is not None:
        return f"{base}/runs/{quote(run, safe='')}"
    return base


def _get_no_redirects(
    session: aiohttp.ClientSession,
    url: str,
    **kwargs: object,
) -> object:
    """Start a GET request without following redirects, with test-double fallback."""
    try:
        return session.get(url, allow_redirects=False, **kwargs)
    except TypeError as e:
        if "allow_redirects" not in str(e):
            raise
        return session.get(url, **kwargs)


async def _download_and_decompress(
    resp: aiohttp.ClientResponse, dest_path: Path, content_encoding: str
) -> None:
    import zlib

    if content_encoding == "zstd":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        decompressor = dctx.decompressobj()
    elif content_encoding == "gzip":
        decompressor = zlib.decompressobj(wbits=31)
    else:
        decompressor = None

    temp_path = dest_path.with_name(f".{dest_path.name}.{uuid.uuid4().hex}.tmp")
    replaced = False
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                if decompressor is not None:
                    chunk = decompressor.decompress(chunk)
                if chunk:
                    await f.write(chunk)
            if decompressor is not None:
                remaining = decompressor.flush()
                if remaining:
                    await f.write(remaining)
        await asyncio.to_thread(os.replace, temp_path, dest_path)
        replaced = True
    finally:
        if not replaced:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)


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


async def _verify_operator_health(api_base: str) -> bool:
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=10)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            async with session.get(f"{api_base}/healthz") as resp:
                if resp.status != 200:
                    print_error("Operator results server not healthy")
                    return False
        except aiohttp.ClientError as e:
            print_error(f"Could not connect to operator results server: {e}")
            return False
    return True


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
        async with session.get(list_url) as resp:
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
        async with session.get(runs_url) as resp:
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

    Returns ``None`` only when the listing itself could not be obtained. A
    per-file failure is reported in the outcome rather than discarding the
    files that did land, so a mostly-complete results directory survives while
    the caller can still refuse to claim success.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=300)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
    ) as session:
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

        downloaded: list[tuple[str, int]] = []
        failed: list[str] = []
        for file_info in available:
            result = await _download_operator_file(
                session,
                api_base=api_base,
                namespace=namespace,
                job_id=job_id,
                file_info=file_info,
                output_dir=output_dir,
                run=resolved_run,
            )
            if result is not None:
                downloaded.append(result)
            elif not _is_refused_name(str(file_info.get("name", ""))):
                failed.append(str(file_info.get("name", "<unnamed>")))
        return _JobDownloadOutcome(downloaded=downloaded, failed=failed)


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
    pod_info = await find_operator_pod(api, namespace=operator_namespace)
    if not pod_info:
        print_error("Operator pod not found")
        print_info(f"Looked in namespace: {operator_namespace}")
        return False

    pod_name, pod_phase = pod_info
    print_info(f"Found operator pod: {pod_name} (status: {pod_phase})")

    try:
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
