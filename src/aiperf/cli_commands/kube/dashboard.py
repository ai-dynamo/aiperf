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
        str,
        Parameter(
            name="--operator-namespace",
            help="Namespace where the operator is deployed.",
        ),
    ] = "aiperf-system",
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
    The port-forward stays open until you press Ctrl+C.

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
        pod_name = await _resolve_operator_pod(manage_options, operator_namespace)
        if pod_name is None:
            return
        await _serve_dashboard(
            manage_options,
            operator_namespace,
            pod_name,
            port=port,
            no_browser=no_browser,
        )


async def _resolve_operator_pod(
    manage_options: KubeManageOptions, operator_namespace: str
) -> str | None:
    from aiperf.kubernetes.client import find_operator_pod, k8s_client
    from aiperf.kubernetes.console import print_error, print_info

    async with k8s_client(
        kubeconfig=manage_options.kubeconfig,
        context=manage_options.kube_context,
    ) as api:
        pod_info = await find_operator_pod(api, namespace=operator_namespace)
        if not pod_info:
            print_error("Operator pod not found")
            print_info(f"Looked in namespace: {operator_namespace}")
            return None

        pod_name, pod_phase = pod_info
        print_info(f"Found operator pod: {pod_name} (status: {pod_phase})")
        return pod_name


async def _serve_dashboard(
    manage_options: KubeManageOptions,
    operator_namespace: str,
    pod_name: str,
    *,
    port: int,
    no_browser: bool,
) -> None:
    import asyncio
    import webbrowser

    from aiperf.kubernetes.console import print_info, print_success
    from aiperf.kubernetes.port_forward import port_forward_with_status
    from aiperf.kubernetes.results import RESULTS_SERVER_PORT

    async with port_forward_with_status(
        operator_namespace,
        pod_name,
        port,
        remote_port=RESULTS_SERVER_PORT,
        verify_api=False,
        kubeconfig=manage_options.kubeconfig,
        kube_context=manage_options.kube_context,
    ) as actual_port:
        url = f"http://localhost:{actual_port}"

        if no_browser:
            print_success(f"Dashboard available at: {url}")
        else:
            webbrowser.open(url)
            print_success(f"Dashboard opened at: {url}")

        print_info("Press Ctrl+C to stop port-forward")

        try:
            # Keep alive until interrupted
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
