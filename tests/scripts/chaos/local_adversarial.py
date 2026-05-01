#!/usr/bin/env python3
from __future__ import annotations

from tests.scripts.chaos.chaos_cases import build_cases as build_chaos_cases
from tests.scripts.chaos.harness import Case, Context, create_context, run_cases, run_cmd
from tests.scripts.chaos.local_cases import build_cases as build_local_cases


def build_cases(ctx: Context | None = None) -> list[Case]:
    context = ctx or create_context()
    return [*build_local_cases(context), *build_chaos_cases()]


def main() -> None:
    ctx = create_context()
    run_cases(build_cases(ctx), ctx, "LOCAL_ADVERSARIAL")


if __name__ == "__main__":
    main()
