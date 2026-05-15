#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import os

from tests.scripts.chaos.chaos_cases import build_cases as build_chaos_cases
from tests.scripts.chaos.config_loader_cases import build_config_loader_cases
from tests.scripts.chaos.fuzz_cases import build_fuzz_cases
from tests.scripts.chaos.harness import (
    Case,
    Context,
    create_context,
    run_cases,
    start_mock_server,
)
from tests.scripts.chaos.local_cases import build_cases as build_local_cases
from tests.scripts.chaos.resource_cases import build_resource_cases


def build_cases(ctx: Context | None = None) -> list[Case]:
    context = ctx or create_context()
    return [
        *build_local_cases(context),
        *build_chaos_cases(),
        *build_config_loader_cases(),
        *build_fuzz_cases(),
        *build_resource_cases(),
    ]


def main() -> None:
    # If the user already pinned a server via env, use it as-is. Otherwise spawn
    # a fresh aiperf-mock-server on a random free port for this run.
    if os.environ.get("AIPERF_ADVERSARIAL_URL"):
        ctx = create_context()
        run_cases(build_cases(ctx), ctx, "LOCAL_ADVERSARIAL")
        return

    # Pre-create the context root so the mock-server log lands next to case logs.
    ctx = create_context()
    mock_log = ctx.logs / "_mock_server.log"
    print(f"MOCK_SERVER_LOG={mock_log}", flush=True)
    with start_mock_server(mock_log) as url:
        print(f"MOCK_SERVER_URL={url}", flush=True)
        # Propagate the spawned URL into the existing Context (and to any child
        # cases that read AIPERF_ADVERSARIAL_URL out of env).
        ctx.url = url
        ctx.env["AIPERF_ADVERSARIAL_URL"] = url
        with contextlib.suppress(SystemExit):
            run_cases(build_cases(ctx), ctx, "LOCAL_ADVERSARIAL")
            return
        # SystemExit was raised by run_cases (some cases verdicted BUG). Re-raise
        # after the context manager has cleanly torn the server down.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
