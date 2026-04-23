# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-artifacts retrieval flow via the controller API.

Lists every result file exposed by the controller API (``/api/results/list``)
and downloads each one to a local directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiohttp

from aiperf.kubernetes.client import find_retrievable_pod
from aiperf.kubernetes.console import (
    _human_size,
    print_action,
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

    from aiperf.kubernetes.models import JobSetInfo


API_RESULTS_FILES_PATH = "/api/results/files"
API_RESULTS_LIST_PATH = "/api/results/list"


async def _download_artifact(
    session: aiohttp.ClientSession,
    files_base: str,
    filename: str,
    output_dir: Path,
    *,
    max_retries: int = 2,
) -> tuple[str, int] | None:
    """Download one artifact by name with retries.

    Returns ``(dest_name, size_bytes)`` on success, ``None`` on skip/error.
    """
    # Sanitize server-provided filename to block path traversal
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename.startswith("."):
        print_warning(f"Refusing unsafe filename: {filename!r}")
        return None
    quoted = quote(safe_filename, safe="")

    for attempt in range(1 + max_retries):
        try:
            async with session.get(f"{files_base}/{quoted}") as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()

                x_filename = resp.headers.get("x-filename")
                raw_dest = x_filename or safe_filename
                dest_name = Path(raw_dest).name or safe_filename
                dest_path = output_dir / dest_name

                content = await resp.read()
                expected = resp.content_length
                if expected is not None and len(content) != expected:
                    print_warning(
                        f"{dest_name}: expected {expected} bytes "
                        f"but received {len(content)}"
                    )
                    if attempt < max_retries:
                        continue
                    print_warning(
                        f"Skipping {dest_name}: incomplete download after retries"
                    )
                    return None

                await asyncio.to_thread(dest_path.write_bytes, content)
                file_size = len(content)
                print_success(f"Downloaded: {dest_name} ({_human_size(file_size)})")
                return (dest_name, file_size)
        except aiohttp.ClientConnectionError:
            if attempt < max_retries:
                continue
            print_warning("Lost connection to API service")
            return None
        except aiohttp.ClientResponseError as e:
            print_warning(f"Failed to download {filename}: {e.status}")
            return None
    return None


async def _list_available_artifacts(
    session: aiohttp.ClientSession, api_base: str, job_id: str
) -> list[str] | None:
    """List artifact filenames from the controller API. Returns None on error."""
    list_url = f"{api_base}{API_RESULTS_LIST_PATH}"
    try:
        async with session.get(list_url) as list_resp:
            list_resp.raise_for_status()
            list_data = await list_resp.json()
            return [f["name"] for f in list_data.get("files", [])]
    except (aiohttp.ClientError, KeyError) as e:
        print_error(f"Failed to list available results for job {job_id}: {e!r}")
        return None


async def _download_all_artifacts(
    api_base: str, job_id: str, output_dir: Path
) -> list[tuple[str, int]] | None:
    """Download every artifact listed by the controller API.

    Returns the list of downloaded ``(name, size)`` tuples, or ``None`` if
    listing failed.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(total=300)
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        available_files = await _list_available_artifacts(session, api_base, job_id)
        if available_files is None:
            return None

        files_base = f"{api_base}{API_RESULTS_FILES_PATH}"
        downloaded: list[tuple[str, int]] = []
        for filename in available_files:
            result = await _download_artifact(session, files_base, filename, output_dir)
            if result is not None:
                downloaded.append(result)
        return downloaded


async def retrieve_all_artifacts(
    job_id: str,
    namespace: str,
    output_dir: Path,
    jobset_info: JobSetInfo | None,
    api: ApiClient,
    local_port: int,
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Retrieve all artifacts via API by downloading files individually.

    Uses port-forward to connect to the API service, lists available files,
    then downloads each one.

    Returns:
        True if artifacts were successfully downloaded.
    """
    if not jobset_info:
        print_error(f"No job found with ID: {job_id}")
        print_info("The --all flag requires the JobSet to still exist.")
        return False

    pod = await find_retrievable_pod(api, namespace, job_id)
    if not pod:
        print_error(f"No controller pod found for job {job_id}")
        print_info("The --all flag requires the controller pod to be running.")
        print_action("Use --from-pod if pod terminated, or ConfigMap if job completed.")
        return False

    pod_name, pod_phase = pod
    print_success(f"Found controller pod: {pod_name} (status: {pod_phase})")

    try:
        async with port_forward_with_status(
            namespace,
            pod_name,
            local_port,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        ) as port:
            api_base = f"http://localhost:{port}"
            print_step("Downloading artifacts...")

            downloaded_files = await _download_all_artifacts(
                api_base, job_id, output_dir
            )
            if downloaded_files is None:
                return False

            if downloaded_files:
                print_file_table(downloaded_files)
                print_success(f"Artifacts saved to: {output_dir}")
                return True
            print_error("No artifacts found. Benchmark may still be running.")
            return False

    except aiohttp.ClientConnectionError:
        print_error("Could not connect to API. Is the pod running?")
        return False
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        print_error(f"Error downloading artifacts: {e!r}")
        return False
