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
from typing import Any

import orjson

from aiperf.common.redact import redact_endpoint_spec
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.results_layout import run_dir

logger = logging.getLogger(__name__)


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
    payload = orjson.dumps(redact_endpoint_spec(spec), option=orjson.OPT_INDENT_2)

    def _write() -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    await asyncio.to_thread(_write)
    logger.info("Saved CR spec to %s", path)
