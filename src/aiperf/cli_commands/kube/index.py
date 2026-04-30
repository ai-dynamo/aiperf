# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``aiperf kube index`` — manual control of the operator's runs index.

Wraps ``GET /admin/index/stats`` and ``POST /admin/index/rebuild`` exposed by
the operator's results-server FastAPI app. Rebuild is the recovery hatch
when the SQLite index has fallen out of sync with the PVC (e.g. after an
external file restore); stats confirms the index is populated post-startup.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import cyclopts
import httpx
import orjson

from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes import console as kube_console

app = cyclopts.App(
    name="index",
    help="Manage the operator's runs/sweep_variations SQLite index.",
)


@app.command(name="rebuild")
async def rebuild(
    *,
    output: Annotated[
        Literal["text", "json"],
        cyclopts.Parameter(help="Output format."),
    ] = "text",
    api_url: Annotated[
        str,
        cyclopts.Parameter(help="Operator HTTP API base URL."),
    ] = "http://localhost:38465",
    options: KubeManageOptions | None = None,
) -> None:
    """Rebuild the operator's runs index from the PVC."""
    if output == "json":
        logging.getLogger("aiperf.kube").setLevel(logging.WARNING)

    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=300.0) as client:
            resp = await client.post("/admin/index/rebuild")
            resp.raise_for_status()
            data = resp.json()
        if output == "json":
            kube_console.console.print(
                orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
            )
        else:
            kube_console.console.print(
                f"Indexed {data['runs_indexed']} runs and "
                f"{data['sweep_variations_indexed']} sweep variations "
                f"in {data['duration_seconds']:.2f}s"
            )
    finally:
        if output == "json":
            logging.getLogger("aiperf.kube").setLevel(logging.INFO)


@app.command(name="stats")
async def stats(
    *,
    output: Annotated[
        Literal["text", "json"],
        cyclopts.Parameter(help="Output format."),
    ] = "text",
    api_url: Annotated[
        str,
        cyclopts.Parameter(help="Operator HTTP API base URL."),
    ] = "http://localhost:38465",
    options: KubeManageOptions | None = None,
) -> None:
    """Show runs index statistics."""
    if output == "json":
        logging.getLogger("aiperf.kube").setLevel(logging.WARNING)
    try:
        async with httpx.AsyncClient(base_url=api_url) as client:
            resp = await client.get("/admin/index/stats")
            resp.raise_for_status()
            data = resp.json()
        if output == "json":
            kube_console.console.print(
                orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
            )
        else:
            kube_console.console.print(
                f"runs={data['runs_count']} "
                f"sweep_variations={data['sweep_variations_count']} "
                f"size={data['db_bytes']}B "
                f"schema_version={data['schema_version']} "
                f"last_bootstrap_unix={data['last_bootstrap_unix']}"
            )
    finally:
        if output == "json":
            logging.getLogger("aiperf.kube").setLevel(logging.INFO)
