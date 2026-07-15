# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube dashboard command: open the operator results server UI in a browser."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="dashboard")


@app.default
async def dashboard(
    *,
    manage_options: KubeManageOptions | None = None,
    port: Annotated[
        int,
        Parameter(
            name="--port",
            help="Local port to bind (default: 0 = ephemeral).",
        ),
    ] = 0,
    operator_namespace: Annotated[
        str | None,
        Parameter(
            name="--operator-namespace",
            help="Namespace where the operator is deployed. "
            "Auto-detected (cluster-wide pod search) when omitted.",
        ),
    ] = None,
    no_browser: Annotated[
        bool,
        Parameter(
            name="--no-browser",
            help="Print the URL instead of opening a browser.",
        ),
    ] = False,
) -> None:
    """Open the operator results server UI in your browser.

    Port-forwards to the operator's results server and opens the dashboard.
    The port-forward stays open until you press Ctrl+C and auto-reconnects
    (with backoff) on transient failures, pinning the local port across
    reconnects so the open browser tab keeps working.

    Examples:
        # Open dashboard in browser
        aiperf kube dashboard

        # Just print the URL (don't open browser)
        aiperf kube dashboard --no-browser

        # Use a specific local port
        aiperf kube dashboard --port 8081
    """
    from aiperf import cli_utils

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Opening Dashboard"):
        resolved = await _resolve_operator(manage_options, operator_namespace)
        if resolved is None:
            return
        ns, pod_name = resolved
        await _serve_dashboard(
            manage_options,
            ns,
            pod_name,
            port=port,
            no_browser=no_browser,
        )


async def _resolve_operator(
    manage_options: KubeManageOptions, operator_namespace: str | None
) -> tuple[str, str] | None:
    """Resolve (namespace, pod_name) for the operator. Returns ``None`` if not found."""
    from aiperf.kubernetes.client import (
        find_operator_pod,
        k8s_client,
        resolve_operator_namespace,
    )
    from aiperf.kubernetes.console import print_error, print_info
    from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

    async with k8s_client(
        kubeconfig=manage_options.kubeconfig,
        context=manage_options.kube_context,
    ) as api:
        ns = await resolve_operator_namespace(api, explicit=operator_namespace)
        if operator_namespace is None and ns != DEFAULT_OPERATOR_NAMESPACE:
            print_info(f"Auto-detected operator namespace: {ns}")
        pod_info = await find_operator_pod(api, namespace=ns)
        if not pod_info:
            print_error("Operator pod not found")
            print_info(f"Looked in namespace: {ns}")
            return None

        pod_name, pod_phase = pod_info
        print_info(f"Found operator pod: {pod_name} (status: {pod_phase})")
        return ns, pod_name


async def _refresh_operator_pod(
    manage_options: KubeManageOptions,
    operator_namespace: str,
    *,
    fallback: str,
) -> str:
    """Re-resolve the operator pod name across reconnects.

    The operator runs in a Deployment, so a pod restart yields a new pod name.
    Falls back to the last-known name on transient apiserver errors so the
    reconnect loop can keep trying — the next start_port_forward call will
    surface a clear "pod not found" error if the fallback is also gone.
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import find_operator_pod, k8s_client

    try:
        async with k8s_client(
            kubeconfig=manage_options.kubeconfig,
            context=manage_options.kube_context,
        ) as api:
            pod_info = await find_operator_pod(api, namespace=operator_namespace)
    except (ApiException, OSError):
        return fallback
    if pod_info is None:
        return fallback
    return pod_info[0]


async def _serve_dashboard(
    manage_options: KubeManageOptions,
    operator_namespace: str,
    pod_name: str,
    *,
    port: int,
    no_browser: bool,
) -> None:
    import asyncio
    import contextlib
    import webbrowser

    from aiperf.kubernetes.console import print_info, print_success, print_warning
    from aiperf.kubernetes.port_forward import (
        _drain_stream,
        cleanup_port_forward,
        start_port_forward,
    )
    from aiperf.kubernetes.results import RESULTS_SERVER_PORT

    initial_backoff = 1.0
    max_backoff = 30.0

    bound_port: int | None = None
    backoff = initial_backoff

    try:
        while True:
            try:
                proc, actual_port = await start_port_forward(
                    operator_namespace,
                    pod_name,
                    port if bound_port is None else bound_port,
                    RESULTS_SERVER_PORT,
                    verify_api=True,
                    kubeconfig=manage_options.kubeconfig,
                    kube_context=manage_options.kube_context,
                )
            except RuntimeError as exc:
                print_warning(
                    f"Port-forward failed: {exc}. Retrying in {backoff:.0f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                pod_name = await _refresh_operator_pod(
                    manage_options, operator_namespace, fallback=pod_name
                )
                continue

            if bound_port is None:
                bound_port = actual_port
                url = f"http://localhost:{bound_port}"
                if no_browser:
                    print_success(f"Dashboard available at: {url}")
                else:
                    webbrowser.open(url)
                    print_success(f"Dashboard opened at: {url}")
                print_info("Press Ctrl+C to stop. Auto-reconnects on transient errors.")
            else:
                print_success(f"Reconnected on localhost:{bound_port}")
            backoff = initial_backoff

            drain_tasks = [
                asyncio.create_task(_drain_stream(proc.stdout)),
                asyncio.create_task(_drain_stream(proc.stderr)),
            ]
            try:
                await proc.wait()
            finally:
                for task in drain_tasks:
                    task.cancel()
                for task in drain_tasks:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                await cleanup_port_forward(proc)
            print_warning(
                f"Port-forward disconnected (kubectl exit {proc.returncode}); "
                "reconnecting..."
            )

            pod_name = await _refresh_operator_pod(
                manage_options, operator_namespace, fallback=pod_name
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
