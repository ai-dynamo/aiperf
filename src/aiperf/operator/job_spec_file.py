# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone CR-spec disk dump.

Writes ``<run_dir>/job_spec.json`` so the PVC is self-describing under
``kubectl cp`` recovery, independent of the runs_index DB. The index
stores the same spec as a column, but a standalone file makes the run
dir interpretable when the DB is missing.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any

import orjson

from aiperf.common.redact import REDACTED_VALUE, redact_headers
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.results_layout import run_dir

logger = logging.getLogger(__name__)


def _redact_spec_for_disk(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``spec`` with endpoint credentials redacted.

    The on-disk ``job_spec.json`` is served verbatim by the results file
    route, the run listing, and the ``.zip`` bundle — none of which apply
    redaction. Mirrors the ``/config`` endpoint's ``_redact_exposed_spec``
    contract so an inline ``endpoint.api_key`` or a bearer token in
    ``endpoint.headers`` never reaches disk in cleartext.
    """
    redacted = deepcopy(spec)
    benchmark = redacted.get("benchmark")
    if not isinstance(benchmark, dict):
        return redacted
    endpoint = benchmark.get("endpoint")
    if not isinstance(endpoint, dict):
        return redacted
    if endpoint.get("api_key") is not None:
        endpoint["api_key"] = REDACTED_VALUE
    headers = endpoint.get("headers")
    if isinstance(headers, dict):
        endpoint["headers"] = redact_headers(headers) or {}
    return redacted


async def save_job_spec_file(
    namespace: str,
    job_id: str,
    spec: dict[str, Any],
    *,
    epoch: str,
) -> None:
    """Persist ``spec`` as ``job_spec.json`` in the run directory.

    Endpoint credentials are redacted before write so the standalone file
    cannot leak secrets through the results file-serving API.
    """
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    path = dest_dir / "job_spec.json"
    payload = orjson.dumps(_redact_spec_for_disk(spec), option=orjson.OPT_INDENT_2)

    def _write() -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    await asyncio.to_thread(_write)
    logger.info("Saved CR spec to %s", path)
