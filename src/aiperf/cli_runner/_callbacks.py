# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-run callback machinery for aiperf.cli_runner.

``CompletedRun`` is the payload, ``OnComplete`` is the signature, and
``_invoke_callbacks`` isolates failures so a buggy hook can't break the
whole run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Payload passed to post-run callbacks after a successful benchmark.

    Carries the resolved artifact directory so callbacks (e.g. auto-plot) can
    locate the run's outputs without re-deriving them from BenchmarkConfig.
    """

    artifact_dir: Path


# Post-run hook signature. Invoked once per successful run/sweep with a
# CompletedRun payload. Each callback runs in isolation: an exception is
# logged with a full traceback, the run is forced to exit non-zero, and
# subsequent callbacks still run. Set ``AIPERF_RAISE_ON_CALLBACK_ERROR=true``
# to re-raise the first failure (after running the remaining callbacks) for
# strict-mode pipelines that want the exception to surface.
OnComplete = Callable[[CompletedRun], None]


def _invoke_callbacks(
    callbacks: list[OnComplete],
    completed: CompletedRun,
    exit_code: int,
    logger: Any,
) -> int:
    """Run every OnComplete callback, isolating failures.

    Each callback is invoked even if a prior one raised. On any failure the
    traceback is logged and ``exit_code`` is forced non-zero (preserving an
    already non-zero ``exit_code``). When the opt-in env var
    ``AIPERF_RAISE_ON_CALLBACK_ERROR`` is true, the first captured exception
    is re-raised after every callback has been attempted, providing a
    strict-mode contract where callback failures propagate to the caller.

    Returns the (possibly elevated) exit code so the caller can pass it to
    ``os._exit``.
    """
    # Read through the Pydantic Settings registry so the env var goes
    # through the project-wide validation/coercion pipeline (booleans
    # accept ``on``/``true``/``1``/``yes`` consistently). Reading raw
    # ``os.environ`` here would diverge from Pydantic's bool coercion
    # and silently default-False values like ``=on``. Instantiated at
    # call-time (not import-time) so unit tests that set the env var
    # via ``monkeypatch.setenv`` see the updated value.
    from aiperf.common.environment import _CLIRunnerSettings

    raise_on_error = _CLIRunnerSettings().RAISE_ON_CALLBACK_ERROR
    first_exc: BaseException | None = None
    for callback in callbacks:
        try:
            callback(completed)
        except Exception as exc:
            logger.exception(
                f"OnComplete callback {getattr(callback, '__name__', callback)!r} "
                f"failed; continuing with remaining callbacks"
            )
            if first_exc is None:
                first_exc = exc
            if exit_code == 0:
                exit_code = 1
    if first_exc is not None and raise_on_error:
        raise first_exc
    return exit_code
