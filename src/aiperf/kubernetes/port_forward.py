# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable helper functions for Kubernetes CLI commands."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp

from aiperf.kubernetes.console import print_info, print_warning
from aiperf.kubernetes.environment import K8sEnvironment

# Re-exported for backward compatibility; see progress_stream.py for implementation.
from aiperf.kubernetes.progress_stream import (
    _consume_ws_messages,  # noqa: F401
    stream_progress_from_api,  # noqa: F401
)

# Port-forward tunables -- moved to K8sEnvironment.PORT_FORWARD; aliases kept
# so internal callers don't have to spell the long path. Tests can monkeypatch
# these attributes if they need to shrink timeouts.
_PORT_FORWARD_TIMEOUT = K8sEnvironment.PORT_FORWARD.TIMEOUT_SECONDS
_API_INITIAL_DELAY = K8sEnvironment.PORT_FORWARD.API_INITIAL_DELAY_SECONDS
_API_RETRY_DELAY = K8sEnvironment.PORT_FORWARD.API_RETRY_DELAY_SECONDS
_API_MAX_RETRIES = K8sEnvironment.PORT_FORWARD.API_MAX_RETRIES
_PROCESS_CLEANUP_TIMEOUT = K8sEnvironment.PORT_FORWARD.PROCESS_CLEANUP_TIMEOUT_SECONDS


async def _drain_stream(stream: asyncio.StreamReader | None) -> None:
    """Continuously discard lines from a subprocess stream until EOF.

    kubectl port-forward writes a "Handling connection for <port>" line to
    stdout for every forwarded connection. Once readiness is parsed, nothing
    reads ``proc.stdout`` again, so on a busy/long-lived tunnel the OS pipe
    buffer (~64KB) fills, kubectl blocks on the write, and the tunnel stalls.
    Draining for the lifetime of the forward keeps the pipe empty.
    """
    if stream is None:
        return
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
    except asyncio.CancelledError:
        pass


async def _monitor_pod_liveness(
    namespace: str,
    pod_name: str,
    proc: asyncio.subprocess.Process,
    *,
    check_interval: float = 10.0,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Background task: kill port-forward if pod disappears."""
    cmd_base = ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "name"]
    if kubeconfig:
        cmd_base.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        cmd_base.extend(["--context", kube_context])

    from aiperf.kubernetes.subproc import run_command

    try:
        while proc.returncode is None:
            await asyncio.sleep(check_interval)
            try:
                # Bound the probe so a stuck kubectl (network partition,
                # throttled apiserver) cannot pin the liveness monitor.
                check_result = await run_command(cmd_base, timeout=check_interval)
                if check_result.returncode != 0:
                    # Only tear down on a genuine "pod gone" signal. A nonzero
                    # exit from a transient apiserver 5xx, throttling, token
                    # refresh, or network blip would otherwise kill a healthy
                    # tunnel, so treat anything that is not NotFound like the
                    # TimeoutError/OSError branches and retry next interval.
                    if "not found" not in check_result.stderr.lower():
                        continue
                    print_warning(
                        f"Pod {pod_name} no longer exists, closing port-forward"
                    )
                    proc.terminate()
                    return
            except TimeoutError:
                # Probe timed out (run_command already terminated the kubectl
                # subprocess); skip this tick and try again next interval.
                continue
            except OSError:  # noqa: BLE001 - watchdog must never die on a single-check failure
                # Transient subprocess/OS error on the probe path; drop this
                # check and try again on the next interval.
                continue
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def port_forward_to_controller(
    namespace: str,
    pod_name: str,
    local_port: int = 0,
    remote_port: int = 9090,
    *,
    verify_api: bool = True,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> AsyncIterator[int]:
    """Async context manager: start port-forward, yield actual port, cleanup on exit.

    Args:
        namespace: Kubernetes namespace.
        pod_name: Controller pod name.
        local_port: Local port to bind. 0 (default) picks an ephemeral port.
        remote_port: Remote port on pod.
        verify_api: If True, verify API responds before yielding.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Yields:
        The actual local port number.
    """
    proc, actual_port = await start_port_forward(
        namespace,
        pod_name,
        local_port,
        remote_port,
        verify_api=verify_api,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    )
    monitor_task = asyncio.create_task(
        _monitor_pod_liveness(
            namespace,
            pod_name,
            proc,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        )
    )
    # Drain kubectl's per-connection stdout/stderr chatter for the lifetime of
    # the forward; otherwise the pipe buffer fills and kubectl blocks (see
    # _drain_stream). Readiness parsing is already done, so it is safe to start.
    drain_tasks = [
        asyncio.create_task(_drain_stream(proc.stdout)),
        asyncio.create_task(_drain_stream(proc.stderr)),
    ]
    try:
        yield actual_port
    finally:
        monitor_task.cancel()
        for task in drain_tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await monitor_task
        for task in drain_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await cleanup_port_forward(proc)


async def _start_port_forward_process(
    namespace: str,
    pod_name: str,
    local_port: int,
    remote_port: int,
    *,
    timeout: float,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> tuple[asyncio.subprocess.Process, int]:
    """Start kubectl port-forward subprocess and wait for the ready message.

    Returns:
        Tuple of (process handle, actual local port).

    Raises:
        RuntimeError: If port-forward fails to start or times out.
    """
    cmd = [
        "kubectl",
        "port-forward",
        "-n",
        namespace,
        f"pod/{pod_name}",
        f"{local_port}:{remote_port}",
    ]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        cmd.extend(["--context", kube_context])

    from aiperf.kubernetes.subproc import start_streaming_process

    proc = await start_streaming_process(cmd)

    try:
        actual_port = await asyncio.wait_for(
            _wait_for_port_forward_ready(proc),
            timeout=timeout,
        )
        if actual_port is None:
            stderr = ""
            if proc.stderr:
                stderr = (await proc.stderr.read()).decode()
            raise RuntimeError(
                f"Port-forward exited unexpectedly: {stderr.strip() or 'no error output'}"
            )
    except TimeoutError as exc:
        stderr = ""
        if proc.stderr:
            try:
                raw = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
                stderr = raw.decode().strip()
            except TimeoutError:
                pass
        from aiperf.kubernetes.subproc import terminate_process

        await terminate_process(proc)
        detail = f" kubectl stderr: {stderr}" if stderr else ""
        raise RuntimeError(
            f"Port-forward did not become ready within {timeout}s.{detail}\n"
            f"  Check that the pod is running: kubectl get pod {pod_name} -n {namespace}\n"
            f"  Check that port {local_port} is not already in use"
        ) from exc

    return proc, actual_port


async def _verify_api_with_retries(
    proc: asyncio.subprocess.Process,
    actual_port: int,
    *,
    namespace: str,
    pod_name: str,
    local_port: int,
    remote_port: int,
    timeout: float,
    start_time: float,
    kubeconfig: str | None,
    kube_context: str | None,
) -> tuple[asyncio.subprocess.Process, int]:
    """Probe the forwarded API, restarting the port-forward on failure.

    Returns the (possibly new) process and port once the API responds, or
    raises ``RuntimeError`` once the retry budget or time budget is exhausted.
    """
    for attempt in range(_API_MAX_RETRIES + 1):
        elapsed = asyncio.get_running_loop().time() - start_time
        remaining_timeout = max(timeout - elapsed, 0.0)
        if remaining_timeout <= 0:
            await cleanup_port_forward(proc)
            raise RuntimeError(
                f"Port-forward API verification exceeded budget ({timeout}s) "
                f"after {attempt} attempts."
            )
        try:
            await asyncio.wait_for(
                _wait_for_api_ready(actual_port, proc),
                timeout=remaining_timeout,
            )
            return proc, actual_port
        except (TimeoutError, RuntimeError) as err:
            await cleanup_port_forward(proc)
            if attempt >= _API_MAX_RETRIES:
                raise RuntimeError(
                    f"Port-forward failed after {_API_MAX_RETRIES} retries. "
                    f"The API service may not be listening on port {remote_port}."
                ) from err
            print_info(
                f"API not ready, restarting port-forward... "
                f"({attempt + 1}/{_API_MAX_RETRIES})"
            )
            await asyncio.sleep(_API_RETRY_DELAY)
            elapsed = asyncio.get_running_loop().time() - start_time
            remaining_timeout = max(timeout - elapsed, 0.0)
            if remaining_timeout <= 0:
                raise RuntimeError(
                    f"Port-forward API verification exceeded budget ({timeout}s)."
                ) from err
            proc, actual_port = await _start_port_forward_process(
                namespace,
                pod_name,
                local_port,
                remote_port,
                timeout=remaining_timeout,
                kubeconfig=kubeconfig,
                kube_context=kube_context,
            )
    # Unreachable: loop either returns or raises.
    return proc, actual_port


async def start_port_forward(
    namespace: str,
    pod_name: str,
    local_port: int = 0,
    remote_port: int = 9090,
    *,
    timeout: float = _PORT_FORWARD_TIMEOUT,
    verify_api: bool = True,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> tuple[asyncio.subprocess.Process, int]:
    """Start kubectl port-forward and wait for it to be ready.

    Uses asyncio subprocess and waits for kubectl's "Forwarding from" message,
    then optionally verifies the API is actually responding. If the port-forward
    process dies during API verification (common when the API isn't listening yet),
    automatically restarts the port-forward and retries.

    Args:
        namespace: Kubernetes namespace
        pod_name: Pod to forward to
        local_port: Local port to bind. 0 (default) picks an ephemeral port.
        remote_port: Remote port on pod
        timeout: Max seconds to wait for port-forward and API to be ready
        verify_api: If True, verify API responds before returning
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        Tuple of (process handle, actual local port).

    Raises:
        RuntimeError: If port-forward fails to start or times out
    """
    start_time = asyncio.get_running_loop().time()

    proc, actual_port = await _start_port_forward_process(
        namespace,
        pod_name,
        local_port,
        remote_port,
        timeout=timeout,
        kubeconfig=kubeconfig,
        kube_context=kube_context,
    )

    if verify_api:
        proc, actual_port = await _verify_api_with_retries(
            proc,
            actual_port,
            namespace=namespace,
            pod_name=pod_name,
            local_port=local_port,
            remote_port=remote_port,
            timeout=timeout,
            start_time=start_time,
            kubeconfig=kubeconfig,
            kube_context=kube_context,
        )

    return proc, actual_port


async def _wait_for_api_ready(
    local_port: int,
    proc: asyncio.subprocess.Process,
    check_interval: float = 1.0,
) -> None:
    """Wait for the API service to respond to HTTP requests.

    Args:
        local_port: Local port to check
        proc: The port-forward process to monitor
        check_interval: Seconds between checks
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    url = f"http://127.0.0.1:{local_port}/health"

    # Give kubectl a moment to establish the tunnel
    await asyncio.sleep(_API_INITIAL_DELAY)

    connector = create_tcp_connector()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=5), connector=connector
    ) as session:
        while True:
            # Check if port-forward process died
            if proc.returncode is not None:
                stderr = ""
                if proc.stderr:
                    stderr = (await proc.stderr.read()).decode()
                raise RuntimeError(
                    f"Port-forward process exited (code {proc.returncode}) while waiting for API. "
                    f"stderr: {stderr.strip() or 'no output'}"
                )

            try:
                async with session.get(url) as resp:
                    if resp.status in (200, 404):
                        # 200 = health endpoint exists, 404 = API running but no health endpoint
                        return
            except aiohttp.ClientError:
                pass  # API not ready yet

            await asyncio.sleep(check_interval)


async def _wait_for_port_forward_ready(
    proc: asyncio.subprocess.Process,
) -> int | None:
    """Wait for kubectl port-forward to output its ready message.

    Args:
        proc: The port-forward subprocess

    Returns:
        The actual local port number if ready, None if process exited.
    """
    import re

    if proc.stdout is None:
        return None

    while True:
        # Check if process has exited
        if proc.returncode is not None:
            return None

        line = await proc.stdout.readline()
        if not line:
            # EOF - process likely exited
            await asyncio.sleep(0.1)
            if proc.returncode is not None:
                return None
            continue

        line_str = line.decode().strip()
        # kubectl outputs: "Forwarding from 127.0.0.1:<port> -> <port>"
        match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+)", line_str)
        if match:
            return int(match.group(1))


async def cleanup_port_forward(
    process: asyncio.subprocess.Process,
    timeout: float = _PROCESS_CLEANUP_TIMEOUT,
) -> None:
    """Gracefully terminate port-forward subprocess.

    Args:
        process: Port-forward asyncio subprocess handle
        timeout: Seconds to wait for graceful termination
    """
    from aiperf.kubernetes.subproc import terminate_process

    await terminate_process(process, timeout)


def port_forward_with_status(
    namespace: str,
    pod_name: str,
    local_port: int = 0,
    *,
    remote_port: int | None = None,
    verify_api: bool = True,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> contextlib.AbstractAsyncContextManager[int]:
    """Port-forward with status logging and success message.

    Wraps port_forward_to_controller with status feedback. Usage::

        async with port_forward_with_status(ns, pod) as port:
            url = f"http://localhost:{port}/..."

    Args:
        namespace: Kubernetes namespace.
        pod_name: Controller pod name.
        local_port: Local port to bind. 0 (default) picks an ephemeral port.
        remote_port: Remote port on pod. Defaults to API_SERVICE port.
        verify_api: If True, verify API responds before yielding.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Yields:
        The actual local port number.
    """
    from aiperf.kubernetes.console import print_success, status_log
    from aiperf.kubernetes.environment import K8sEnvironment

    if remote_port is None:
        remote_port = K8sEnvironment.PORTS.API_SERVICE

    @contextlib.asynccontextmanager
    async def _inner():
        with status_log(f"Starting port-forward to {pod_name}..."):
            async with port_forward_to_controller(
                namespace,
                pod_name,
                local_port,
                remote_port,
                verify_api=verify_api,
                kubeconfig=kubeconfig,
                kube_context=kube_context,
            ) as actual_port:
                print_success(f"Port-forward ready on localhost:{actual_port}")
                yield actual_port

    return _inner()
