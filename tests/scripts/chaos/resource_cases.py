from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from tests.scripts.chaos.harness import Case, Context, run_cmd, skip_case


def _aiperf_profile_cmd(ctx: Context, name: str) -> list[str]:
    return [
        "uv",
        "run",
        "aiperf",
        "profile",
        "--model",
        "mock-model",
        "--url",
        ctx.url,
        "--endpoint-type",
        "chat",
        "--request-count",
        "2",
        "--concurrency",
        "1",
        "--tokenizer",
        "builtin",
        "--ui",
        "none",
        "--request-timeout-seconds",
        "10",
        "--wait-for-model-timeout",
        "0",
        "--workers-max",
        "1",
        "--no-gpu-telemetry",
        "--artifact-dir",
        str(ctx.artifacts / name),
    ]


def has_systemd_run() -> bool:
    binary = shutil.which("systemd-run")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--user", "--scope", "--quiet", "true"],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def has_prlimit() -> bool:
    return shutil.which("prlimit") is not None


def is_linux() -> bool:
    return sys.platform == "linux"


def case_cpu_quota(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--property=CPUQuota=20%",
        *_aiperf_profile_cmd(ctx, name),
    ]
    return run_cmd(cmd, log, ctx, 180)


def case_memory_cap(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--property=MemoryMax=256M",
        *_aiperf_profile_cmd(ctx, name),
    ]
    return run_cmd(cmd, log, ctx, 180)


def case_low_fd_limit(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    cmd = ["prlimit", "--nofile=64", "--", *_aiperf_profile_cmd(ctx, name)]
    return run_cmd(cmd, log, ctx, 120)


_CGROUP_SPECS: tuple[tuple[str, str, callable], ...] = (
    (
        "resource-cpu-quota",
        "FLAG_FOR_REVIEW",
        case_cpu_quota,
    ),
    (
        "resource-memory-cap",
        "FLAG_FOR_REVIEW",
        case_memory_cap,
    ),
)

_PRLIMIT_SPECS: tuple[tuple[str, str, callable], ...] = (
    (
        "resource-low-fd-limit",
        "GRACEFUL_FAILURE_REQUIRED",
        case_low_fd_limit,
    ),
)


def build_resource_cases() -> list[Case]:
    if not is_linux():
        return [
            skip_case(name, "resource starvation cases require Linux")
            for name, _, _ in _CGROUP_SPECS + _PRLIMIT_SPECS
        ]
    cases: list[Case] = []
    if has_systemd_run():
        for name, expected, fn in _CGROUP_SPECS:
            cases.append(
                Case(
                    name=name,
                    expected=expected,  # type: ignore[arg-type]
                    run=fn,
                    why=f"aiperf must degrade gracefully under {name.split('-', 1)[1]}",
                )
            )
    else:
        cases.extend(
            skip_case(name, "systemd-run --user --scope unavailable")
            for name, _, _ in _CGROUP_SPECS
        )
    if has_prlimit():
        for name, expected, fn in _PRLIMIT_SPECS:
            cases.append(
                Case(
                    name=name,
                    expected=expected,  # type: ignore[arg-type]
                    run=fn,
                    why="aiperf must fail gracefully when fd limit is exhausted",
                )
            )
    else:
        cases.extend(
            skip_case(name, "prlimit not on PATH") for name, _, _ in _PRLIMIT_SPECS
        )
    return cases
