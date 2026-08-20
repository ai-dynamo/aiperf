# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-PVC sweep aggregate artifact retrieval flow."""

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

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


RESULTS_SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SERVER_PORT", "8081"))
_REDIRECT_STATUSES = {301, 302, 307, 308}


def _get_no_redirects(
    session: aiohttp.ClientSession,
    url: str,
    **kwargs: object,
) -> object:
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
        async with session.get(list_url) as resp:
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

    Returns ``None`` only when the listing itself could not be obtained. A
    per-file failure is reported in the outcome rather than discarding the
    files that already landed -- this used to return None on the first
    failure, throwing away a mostly-complete aggregate directory and telling
    the user nothing about which file was missing. The job-level twin has been
    partial-tolerant since 5a51031db5.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=300)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
    ) as session:
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

        downloaded: list[tuple[str, int]] = []
        failed: list[str] = []
        for file_info in available:
            result = await _download_sweep_operator_file(
                session,
                api_base=api_base,
                namespace=namespace,
                sweep_name=sweep_name,
                run=run,
                file_info=file_info,
                output_dir=output_dir,
            )
            if result is not None:
                downloaded.append(result)
            elif not _is_refused_name(str(file_info.get("name", ""))):
                failed.append(str(file_info.get("name", "<unnamed>")))
        return _JobDownloadOutcome(downloaded=downloaded, failed=failed)


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
