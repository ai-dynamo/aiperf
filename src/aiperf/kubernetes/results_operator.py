# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-PVC result retrieval flow.

Handles downloading benchmark results from the operator's results-server
sidecar, which is backed by a persistent volume and therefore survives
JobSet deletion.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiofiles
import aiohttp

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
from aiperf.kubernetes.port_forward import port_forward_with_status

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


# Default port for the results server sidecar container
RESULTS_SERVER_PORT = 8081


async def _download_and_decompress(
    resp: aiohttp.ClientResponse, dest_path: Path, content_encoding: str
) -> None:
    """Download response body, decompressing if needed."""
    import zlib

    if content_encoding == "zstd":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        decompressor = dctx.decompressobj()
    elif content_encoding == "gzip":
        decompressor = zlib.decompressobj(wbits=31)
    else:
        decompressor = None

    async with aiofiles.open(dest_path, "wb") as f:
        async for chunk in resp.content.iter_chunked(64 * 1024):
            if decompressor is not None:
                chunk = decompressor.decompress(chunk)
            if chunk:
                await f.write(chunk)
        if decompressor is not None:
            remaining = decompressor.flush()
            if remaining:
                await f.write(remaining)


async def _download_operator_file(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    namespace: str,
    job_id: str,
    file_info: dict,
    output_dir: Path,
) -> tuple[str, int] | None:
    """Download a single file from the operator's results server.

    Returns ``(safe_name, size_bytes)`` on success, ``None`` if skipped or
    the download failed.
    """
    display_name = file_info["name"]
    # Defend against a compromised/buggy controller returning a traversal
    # path like ``../../etc/foo``. Strip to the basename and skip empty /
    # dotfile results.
    safe_name = Path(display_name).name
    if not safe_name or safe_name.startswith("."):
        print_warning(f"Refusing unsafe filename: {display_name!r}")
        return None
    quoted_name = quote(safe_name, safe="")
    download_url = f"{api_base}/api/v1/results/{namespace}/{job_id}/{quoted_name}"
    headers = {"Accept-Encoding": "zstd, gzip, identity"}

    try:
        async with session.get(download_url, headers=headers) as resp:
            if resp.status == 404:
                print_warning(f"File not found: {safe_name}")
                return None
            resp.raise_for_status()

            dest_path = output_dir / safe_name
            content_encoding = resp.headers.get("Content-Encoding", "identity")

            await _download_and_decompress(resp, dest_path, content_encoding)

            file_size = dest_path.stat().st_size
            print_success(f"Downloaded: {safe_name} ({_human_size(file_size)})")
            return (safe_name, file_size)
    except aiohttp.ClientError as e:
        print_warning(f"Failed to download {safe_name}: {e}")
        return None


async def _verify_operator_health(api_base: str) -> bool:
    """Check that the operator's results server is reachable and healthy."""
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
) -> list[dict] | None:
    """List available result files for a job. Returns None on error."""
    list_url = f"{api_base}/api/v1/results/{namespace}/{job_id}"
    try:
        async with session.get(list_url) as resp:
            if resp.status == 404:
                print_error(f"No results stored for {namespace}/{job_id}")
                return None
            resp.raise_for_status()
            list_data = await resp.json()
    except aiohttp.ClientError as e:
        print_error(f"Failed to list results: {e}")
        return None

    available = list_data.get("files", [])
    if not available:
        print_warning("No result files found")
        return None
    return available


async def _download_all_operator_files(
    *,
    api_base: str,
    namespace: str,
    job_id: str,
    output_dir: Path,
) -> list[tuple[str, int]] | None:
    """List and download every result file for a job.

    Returns the list of downloaded ``(name, size)`` tuples, or ``None`` if
    listing failed / no files were available.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=300)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
    ) as session:
        available = await _list_operator_files(
            session, api_base=api_base, namespace=namespace, job_id=job_id
        )
        if available is None:
            return None

        print_step(f"Downloading {len(available)} files...")

        downloaded: list[tuple[str, int]] = []
        for file_info in available:
            result = await _download_operator_file(
                session,
                api_base=api_base,
                namespace=namespace,
                job_id=job_id,
                file_info=file_info,
                output_dir=output_dir,
            )
            if result is not None:
                downloaded.append(result)
        return downloaded


async def retrieve_results_from_operator(
    job_id: str,
    namespace: str,
    output_dir: Path,
    api: ApiClient,
    *,
    local_port: int = 0,
    operator_namespace: str = "aiperf-system",
    results_port: int = RESULTS_SERVER_PORT,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Retrieve results from the operator's results server sidecar (PVC-backed).

    Port-forwards to the operator pod's results-server sidecar and downloads
    all available files for the specified job. Works even after the benchmark
    JobSet has been deleted, since results are stored on the operator's PVC.

    Returns True if results were successfully retrieved.
    """
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

            downloaded_files = await _download_all_operator_files(
                api_base=api_base,
                namespace=namespace,
                job_id=job_id,
                output_dir=output_dir,
            )
            if downloaded_files is None:
                return False

            if downloaded_files:
                print_file_table(downloaded_files)
                print_success(f"Results saved to: {output_dir}")
                return True
            print_error("No files downloaded")
            return False

    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        print_error(f"Error connecting to operator: {e!r}")
        return False
