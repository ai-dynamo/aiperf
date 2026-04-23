# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result retrieval from Kubernetes benchmark pods and API service."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiofiles
import aiohttp
import orjson

from aiperf.kubernetes.client import (
    find_controller_pod,
    find_operator_pod,
    find_retrievable_pod,
)
from aiperf.kubernetes.console import (
    _human_size,
    console,
    print_action,
    print_error,
    print_file_table,
    print_header,
    print_info,
    print_metrics_summary,
    print_step,
    print_success,
    print_warning,
)
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.port_forward import port_forward_with_status
from aiperf.kubernetes.subproc import (
    run_command,
    start_streaming_process,
    terminate_process,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

    from aiperf.kubernetes.models import JobSetInfo

# Subset of key result files for quick retrieval (default `results` command)
KEY_RESULT_FILES = [
    "metrics.json",
    "profile_export_aiperf.json",
    "profile_export_console.txt",
    "profile_export_console.ansi",
]

# API path segments for URL construction
API_RESULTS_FILES_PATH = "/api/results/files"
API_RESULTS_LIST_PATH = "/api/results/list"


def _kubectl_kube_args(kubeconfig: str | None, kube_context: str | None) -> list[str]:
    """Build --kubeconfig/--context args for kubectl subprocesses."""
    args: list[str] = []
    if kubeconfig:
        args.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        args.extend(["--context", kube_context])
    return args


# Default port for the results server sidecar container
RESULTS_SERVER_PORT = 8081


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
    from aiperf.transports.aiohttp_client import create_tcp_connector

    pod_info = await find_operator_pod(
        api,
        namespace=operator_namespace,
    )
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
            downloaded_files: list[tuple[str, int]] = []

            # Verify connectivity
            timeout = aiohttp.ClientTimeout(total=10)
            connector = create_tcp_connector()
            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                try:
                    async with session.get(f"{api_base}/healthz") as resp:
                        if resp.status != 200:
                            print_error("Operator results server not healthy")
                            return False
                except aiohttp.ClientError as e:
                    print_error(f"Could not connect to operator results server: {e}")
                    return False

            # List and download files
            timeout = aiohttp.ClientTimeout(total=300)
            connector = create_tcp_connector()
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                auto_decompress=False,
            ) as session:
                list_url = f"{api_base}/api/v1/results/{namespace}/{job_id}"
                try:
                    async with session.get(list_url) as resp:
                        if resp.status == 404:
                            print_error(f"No results stored for {namespace}/{job_id}")
                            return False
                        resp.raise_for_status()
                        list_data = await resp.json()
                except aiohttp.ClientError as e:
                    print_error(f"Failed to list results: {e}")
                    return False

                available = list_data.get("files", [])
                if not available:
                    print_warning("No result files found")
                    return False

                print_step(f"Downloading {len(available)} files...")

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
                        downloaded_files.append(result)

            if downloaded_files:
                print_file_table(downloaded_files)
                print_success(f"Results saved to: {output_dir}")
                return True
            else:
                print_error("No files downloaded")
                return False

    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        print_error(f"Error connecting to operator: {e!r}")
        return False


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


async def _write_api_response(
    response: aiohttp.ClientResponse,
    filename: str,
    output_dir: Path,
    *,
    attempt: int,
    max_retries: int,
) -> bool | None:
    """Write a 200-response body to disk and emit a metric summary if relevant.

    Returns:
        True on success, False after exhausted retries on incomplete body,
        None to signal "retry this attempt" (incomplete body, retries remain).
    """
    content = await response.read()
    expected = response.content_length
    if expected is not None and len(content) != expected:
        print_warning(
            f"{filename}: expected {expected} bytes but received {len(content)}"
        )
        if attempt < max_retries:
            return None
        print_warning(f"Skipping {filename}: incomplete download after retries")
        return False

    output_file = output_dir / filename
    await asyncio.to_thread(output_file.write_bytes, content)
    print_success(f"Downloaded: {filename}")

    if filename == "metrics.json":
        try:
            metrics = orjson.loads(content)
            print_metrics_summary(metrics)
        except (orjson.JSONDecodeError, KeyError, TypeError) as e:
            print_warning(f"Could not parse metrics: {e}")
    return True


async def _download_api_file(
    session: aiohttp.ClientSession,
    files_base: str,
    filename: str,
    output_dir: Path,
    *,
    max_retries: int = 2,
) -> bool:
    """Download one key result file with retries.

    Returns True if the file was written successfully (including the metrics
    summary side-effect when the filename is ``metrics.json``).
    """
    for attempt in range(1 + max_retries):
        try:
            async with session.get(f"{files_base}/{filename}") as response:
                if response.status == 200:
                    outcome = await _write_api_response(
                        response,
                        filename,
                        output_dir,
                        attempt=attempt,
                        max_retries=max_retries,
                    )
                    if outcome is None:
                        continue
                    return outcome
                if response.status != 404:
                    print_warning(f"Failed to download {filename}: {response.status}")
                return False
        except aiohttp.ClientConnectorError:
            if attempt < max_retries:
                continue
            print_warning(f"Could not connect to API service for file {filename}")
            return False
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
            if attempt < max_retries:
                continue
            print_warning(f"Error downloading {filename}: {e}")
            return False
    return False


async def retrieve_results_from_api(
    job_id: str,
    namespace: str,
    output_dir: Path,
    jobset_info: JobSetInfo | None,
    api: ApiClient,
    *,
    local_port: int = 0,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Retrieve results from the API service via port-forward.

    Downloads metrics.json and other key result files from the running controller pod.

    Returns True if results were successfully retrieved, False otherwise.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    if not jobset_info:
        return False

    pod = await find_retrievable_pod(api, namespace, job_id)
    if not pod:
        return False

    pod_name, pod_phase = pod
    print_info(f"Found controller pod: {pod_name} (status: {pod_phase})")

    try:
        async with port_forward_with_status(
            namespace,
            pod_name,
            local_port,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        ) as port:
            files_base = f"http://localhost:{port}{API_RESULTS_FILES_PATH}"

            downloaded_any = False
            timeout = aiohttp.ClientTimeout(total=30)
            connector = create_tcp_connector()
            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                for filename in KEY_RESULT_FILES:
                    ok = await _download_api_file(
                        session, files_base, filename, output_dir
                    )
                    downloaded_any = downloaded_any or ok

            return downloaded_any

    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        print_warning(f"Error connecting to API: {e}")
        return False


async def retrieve_results_from_pod(
    job_id: str,
    namespace: str,
    output_dir: Path,
    jobset_info: JobSetInfo | None,
    api: ApiClient,
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Retrieve results by copying from pod.

    Returns:
        True if results were successfully copied, False otherwise.
    """
    if not jobset_info:
        print_error(f"No job found with ID: {job_id}")
        print_info("Results can only be retrieved from pods while the JobSet exists.")
        return False

    pod_info = await find_controller_pod(api, namespace, job_id)
    if not pod_info:
        print_error(f"No controller pod found for job {job_id}")
        print_info("The job may have completed and pods were cleaned up.")
        print_action("Use --ttl-seconds=0 during deploy to keep pods after completion.")
        return False

    pod_name, pod_phase = pod_info
    container = Containers.CONTROL_PLANE

    print_success(f"Found controller pod: {pod_name} (status: {pod_phase})")

    if not pod_phase.is_retrievable:
        print_error(f"Pod is in '{pod_phase}' state. Cannot retrieve results.")
        if pod_phase == PodPhase.PENDING:
            print_info("Wait for the pod to start running.")
        elif pod_phase == PodPhase.FAILED:
            print_action("Check logs with 'aiperf kube logs'.")
        return False

    print_step(f"Copying results from {pod_name}:/results to {output_dir}")

    if not await kubectl_copy_results(
        namespace,
        pod_name,
        container,
        output_dir,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    ):
        return False

    return display_copied_results(output_dir, jobset_info)


async def kubectl_copy_results(
    namespace: str,
    pod_name: str,
    container: str,
    output_dir: Path,
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Copy results from pod to local directory via kubectl cp.

    Args:
        namespace: Kubernetes namespace.
        pod_name: Controller pod name.
        container: Container name to copy from.
        output_dir: Local directory to copy results into.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        True if copy succeeded, False otherwise.
    """
    kube_args = _kubectl_kube_args(kubeconfig, kube_context)
    cp_result = await run_command(
        [
            "kubectl",
            "cp",
            "-n",
            namespace,
            "-c",
            container,
            f"{pod_name}:/results/.",
            str(output_dir),
            *kube_args,
        ]
    )

    if cp_result.ok:
        if cp_result.stdout:
            console.print(cp_result.stdout)
        return True

    print_error(f"Error copying results: {cp_result.stderr}")
    print_info("Trying to list available files...")

    ls_result = await run_command(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            "-c",
            container,
            pod_name,
            *kube_args,
            "--",
            "ls",
            "-la",
            "/results",
        ]
    )
    if ls_result.ok:
        console.print(ls_result.stdout)
    else:
        print_error("Could not list results directory.")
    return False


def display_copied_results(output_dir: Path, jobset_info: JobSetInfo) -> bool:
    """Display summary of copied result files.

    Args:
        output_dir: Directory containing copied results.
        jobset_info: JobSet info object with status.

    Returns:
        True if files were found, False otherwise.
    """
    copied_files = list(output_dir.glob("*"))
    if not copied_files:
        print_warning("No files found in results directory.")
        print_info("The benchmark may still be running. Try again after completion.")
        return False

    print_file_table([(f.name, f.stat().st_size) for f in copied_files], verb="Copied")

    summary_file = output_dir / "profile_export_aiperf.json"
    if summary_file.exists():
        try:
            summary = orjson.loads(summary_file.read_bytes())
            if "summary" in summary:
                print_header("Benchmark Summary")
                for key, value in summary["summary"].items():
                    console.print(f"  [dim]{key:<30}[/dim]  {value}")
        except (orjson.JSONDecodeError, KeyError, TypeError, OSError) as e:
            print_warning(f"Could not parse summary: {e}")

    print_info(f"Job status: {jobset_info.status}")
    print_success(f"Results saved to: {output_dir}")
    return True


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
    from aiperf.transports.aiohttp_client import create_tcp_connector

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
            downloaded_files: list[tuple[str, int]] = []

            print_step("Downloading artifacts...")

            timeout = aiohttp.ClientTimeout(total=300)
            connector = create_tcp_connector()
            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                list_url = f"{api_base}{API_RESULTS_LIST_PATH}"
                try:
                    async with session.get(list_url) as list_resp:
                        list_resp.raise_for_status()
                        list_data = await list_resp.json()
                        available_files = [
                            f["name"] for f in list_data.get("files", [])
                        ]
                except (aiohttp.ClientError, KeyError) as e:
                    print_error(
                        f"Failed to list available results for job {job_id}: {e!r}"
                    )
                    return False

                files_base = f"{api_base}{API_RESULTS_FILES_PATH}"
                for filename in available_files:
                    result = await _download_artifact(
                        session, files_base, filename, output_dir
                    )
                    if result is not None:
                        downloaded_files.append(result)

            if downloaded_files:
                print_file_table(downloaded_files)
                print_success(f"Artifacts saved to: {output_dir}")
                return True
            else:
                print_error("No artifacts found. Benchmark may still be running.")
                return False

    except aiohttp.ClientConnectionError:
        print_error("Could not connect to API. Is the pod running?")
        return False
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        print_error(f"Error downloading artifacts: {e!r}")
        return False


async def shutdown_api_service(
    job_id: str,
    namespace: str,
    api: ApiClient,
    local_port: int = 0,
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> bool:
    """Send shutdown request to the API service via port-forward.

    Args:
        job_id: AIPerf job ID.
        namespace: Kubernetes namespace.
        api: Connected kubernetes_asyncio ApiClient.
        local_port: Local port for port-forward.

    Returns:
        True if shutdown was successfully requested.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    api_shutdown_path = "/api/shutdown"

    pod = await find_retrievable_pod(api, namespace, job_id, require_running=True)
    if not pod:
        pod_info = await find_controller_pod(api, namespace, job_id)
        if not pod_info:
            print_warning(
                f"Controller pod not found for job {job_id}, cannot send shutdown signal"
            )
            return False
        print_info("Controller pod is not running, no shutdown needed")
        return True

    pod_name, _ = pod

    try:
        async with port_forward_with_status(
            namespace,
            pod_name,
            local_port,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        ) as port:
            timeout = aiohttp.ClientTimeout(total=10)
            connector = create_tcp_connector()
            async with (
                aiohttp.ClientSession(timeout=timeout, connector=connector) as session,
                session.post(f"http://localhost:{port}{api_shutdown_path}") as response,
            ):
                if response.status == 200:
                    print_success("API service shutdown requested")
                    return True
                if response.status == 409:
                    print_warning("Benchmark is still running. Cannot shut down yet.")
                    return False
                print_warning(
                    f"Unexpected response from shutdown endpoint: {response.status}"
                )
                return False
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        print_warning(f"Could not send shutdown signal: {e}")
        return False


async def stream_controller_logs(
    namespace: str,
    pod_name: str,
    *,
    container: str = Containers.CONTROL_PLANE,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Stream logs from controller pod until completion.

    Args:
        namespace: Kubernetes namespace
        pod_name: Pod name to stream logs from
        container: Container name within the pod
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.
    """
    cmd = [
        "kubectl",
        "logs",
        "-f",
        "-n",
        namespace,
        "-c",
        container,
        pod_name,
    ]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        cmd.extend(["--context", kube_context])
    proc = await start_streaming_process(cmd, merge_stderr=True)

    try:
        while True:
            if proc.stdout is None:
                break
            line = await proc.stdout.readline()
            if not line:
                await proc.wait()
                break
            print(line.decode().rstrip())
    except asyncio.CancelledError:
        proc.terminate()
        raise
    finally:
        await terminate_process(proc)
