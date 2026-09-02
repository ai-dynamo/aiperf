# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-artifacts retrieval flow via the controller API.

Lists every result file exposed by the controller API (``/api/results/list``)
and downloads each one to a local directory.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote, unquote

import aiofiles
import aiohttp
import orjson

from aiperf.common.environment import Environment
from aiperf.common.results_markers import READY_MARKER_NAME
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
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.kubernetes.results_operator import (
    _REDIRECT_STATUSES,
    _get_no_redirects,
    _get_with_request_timeout,
    _JobDownloadOutcome,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

    from aiperf.kubernetes.models import JobSetInfo


API_RESULTS_FILES_PATH = "/api/results/files"
API_RESULTS_LIST_PATH = "/api/results/list"


async def _response_json(response: aiohttp.ClientResponse) -> dict:
    """Parse JSON from aiohttp responses and lightweight test doubles."""
    try:
        return await response.json(loads=orjson.loads)
    except TypeError as e:
        if "loads" not in str(e):
            raise
        return await response.json()


def _response_destination(
    resp: aiohttp.ClientResponse, safe_filename: str
) -> str | None:
    raw_dest = resp.headers.get("x-filename") or safe_filename
    # Decode only when the server declares the value percent-encoded, so a
    # literal % in an ASCII name survives. Path(...).name below still runs
    # after decoding, so an encoded traversal cannot escape the directory.
    if resp.headers.get("x-filename-encoding") == "percent":
        raw_dest = unquote(raw_dest)
    # Drop the override when it resolves to an unsafe name, matching
    # ProgressClient._resolve_dest_path. Path("..").name is ".." -- the .name
    # call alone does not stop a server steering the write one directory up,
    # and the containment that follows is accidental (os.replace refusing a
    # directory target), not a check.
    dest_name = Path(raw_dest).name
    if dest_name in ("", ".", ".."):
        dest_name = safe_filename
    if dest_name == READY_MARKER_NAME:
        print_warning(f"Refusing reserved filename: {raw_dest!r}")
        return None
    return dest_name


def _incomplete_download_action(
    *,
    dest_name: str,
    expected: int | None,
    actual: int,
    attempt: int,
    max_retries: int,
) -> Literal["write", "retry", "skip"]:
    """Decide what to do with a downloaded body given its content-length.

    Returns ``"write"`` when the body is complete (or length is unknown),
    ``"retry"`` when the body is short but retries remain, and ``"skip"``
    when the body is short and retries are exhausted. ``"skip"`` must NOT be
    written to disk -- conflating it with ``"write"`` silently persists a
    truncated artifact.
    """
    if expected is None or actual == expected:
        return "write"
    print_warning(f"{dest_name}: expected {expected} bytes but received {actual}")
    if attempt < max_retries:
        return "retry"
    print_warning(f"Skipping {dest_name}: incomplete download after retries")
    return "skip"


def _sanitized_artifact_relpath(filename: str) -> str | None:
    """Normalise an artifact filename to a safe relative POSIX path.

    Server-supplied names are untrusted: traversal segments (``.``/``..``),
    empty names, and hidden basenames are refused (with a warning) by
    returning ``None``.
    """
    parts = [p for p in filename.replace("\\", "/").split("/") if p]
    if (
        not parts
        or any(p in ("", ".", "..") for p in parts)
        or parts[-1].startswith(".")
    ):
        print_warning(f"Refusing unsafe filename: {filename!r}")
        return None
    return "/".join(parts)


async def _stream_artifact_response(
    resp: aiohttp.ClientResponse,
    safe_filename: str,
    output_dir: Path,
    *,
    attempt: int,
    max_retries: int,
) -> tuple[str, int] | Literal["retry", "refused"] | None:
    """Validate and atomically stream one artifact response to disk."""
    if resp.status == 404:
        return None
    if resp.status in _REDIRECT_STATUSES:
        print_warning(f"Refusing redirected download for {safe_filename}")
        return None
    resp.raise_for_status()

    dest_name = _response_destination(resp, Path(safe_filename).name)
    if dest_name is None:
        return "refused"

    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    dest_path = output_dir / dest_name
    temp_path = dest_path.with_name(f".{dest_path.name}.{uuid.uuid4().hex}.tmp")
    replaced = False
    received = 0
    try:
        async with aiofiles.open(temp_path, "wb") as file:
            async for chunk in resp.content.iter_chunked(
                Environment.COMPRESSION.CHUNK_SIZE
            ):
                received += len(chunk)
                await file.write(chunk)
        action = _incomplete_download_action(
            dest_name=dest_name,
            expected=resp.content_length,
            actual=received,
            attempt=attempt,
            max_retries=max_retries,
        )
        if action == "retry":
            return "retry"
        if action == "skip":
            return None
        await asyncio.to_thread(os.replace, temp_path, dest_path)
        replaced = True
        return (dest_name, received)
    finally:
        if not replaced:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)


async def _download_artifact(
    session: aiohttp.ClientSession,
    files_base: str,
    filename: str,
    output_dir: Path,
    *,
    max_retries: int = K8sEnvironment.RESULTS.DOWNLOAD_MAX_RETRIES,
) -> tuple[str, int] | Literal["refused"] | None:
    """Download one artifact by name with retries.

    ``filename`` may carry nested subdirs (e.g. ``aggregate/...`` or
    ``checkpoints/...``) as emitted by the results sidecar; the relative
    layout is preserved on disk so two artifacts sharing a basename do not
    overwrite each other.

    Returns ``(dest_name, size_bytes)`` on success, ``"refused"`` when the
    name is declined by policy (unsafe path, hidden basename, reserved
    destination) and ``None`` when the artifact was advertised but could not
    be retrieved. Callers MUST keep the last two apart: a refusal is an
    expected skip, a ``None`` is a failed download that must not be reported
    as success.
    """
    safe_filename = _sanitized_artifact_relpath(filename)
    if safe_filename is None:
        return "refused"
    quoted = quote(safe_filename, safe="/")
    rel_path = Path(safe_filename)

    for attempt in range(1 + max_retries):
        try:
            async with _get_no_redirects(session, f"{files_base}/{quoted}") as resp:
                outcome = await _stream_artifact_response(
                    resp,
                    safe_filename,
                    output_dir / rel_path.parent,
                    attempt=attempt,
                    max_retries=max_retries,
                )
                if outcome == "retry":
                    continue
                if outcome == "refused":
                    return "refused"
                if outcome is None:
                    return None
                dest_name, file_size = outcome
                dest_rel = (rel_path.parent / dest_name).as_posix()
                print_success(f"Downloaded: {dest_rel} ({_human_size(file_size)})")
                return (dest_rel, file_size)
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
        async with _get_with_request_timeout(session, list_url) as list_resp:
            list_resp.raise_for_status()
            list_data = await _response_json(list_resp)
            return [f["name"] for f in list_data.get("files", [])]
    except (aiohttp.ClientError, KeyError) as e:
        print_error(f"Failed to list available results for job {job_id}: {e!r}")
        return None


async def _download_all_artifacts(
    api_base: str, job_id: str, output_dir: Path
) -> _JobDownloadOutcome | None:
    """Download every artifact listed by the controller API.

    Returns the downloaded/failed split, or ``None`` if listing itself failed.
    Names refused by policy (dot-files, traversal segments, reserved
    destinations) are neither downloaded nor counted as failures.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    timeout = aiohttp.ClientTimeout(
        total=K8sEnvironment.RESULTS.DOWNLOAD_TIMEOUT_SECONDS
    )
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        available_files = await _list_available_artifacts(session, api_base, job_id)
        if available_files is None:
            return None

        files_base = f"{api_base}{API_RESULTS_FILES_PATH}"
        downloaded: list[tuple[str, int]] = []
        failed: list[str] = []
        for filename in available_files:
            result = await _download_artifact(session, files_base, filename, output_dir)
            if result == "refused":
                continue
            if result is None:
                failed.append(filename)
            else:
                downloaded.append(result)
        return _JobDownloadOutcome(downloaded=downloaded, failed=failed)


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
        print_action(
            "Use --from-pods if pod terminated, or ConfigMap if job completed."
        )
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

            outcome = await _download_all_artifacts(api_base, job_id, output_dir)
            if outcome is None:
                return False

            if outcome.downloaded:
                print_file_table(outcome.downloaded)
            if not outcome.complete:
                # Some advertised files never landed. Reporting success here
                # would hand the caller a silently incomplete artifact
                # directory (and a zero exit code) for a partial download.
                for name in outcome.failed:
                    print_warning(f"Failed to download: {name}")
                print_error(
                    f"{len(outcome.failed)} of "
                    f"{len(outcome.downloaded) + len(outcome.failed)} artifacts "
                    f"could not be downloaded to: {output_dir}"
                )
                return False
            if outcome.downloaded:
                print_success(f"Artifacts saved to: {output_dir}")
                return True
            print_error("No artifacts found. Benchmark may still be running.")
            return False

    except aiohttp.ClientConnectionError:
        print_error("Could not connect to API. Is the pod running?")
        return False
    except (TimeoutError, aiohttp.ClientError, OSError, RuntimeError) as e:
        print_error(f"Error downloading artifacts: {e!r}")
        return False
