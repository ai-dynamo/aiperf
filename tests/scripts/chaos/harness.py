from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BASE = Path(__file__).resolve().parents[3]
LOCAL_DEFAULT_URL = "http://127.0.0.1:36037"
MOCK_SERVER_READY_TIMEOUT_SECONDS = 30

Expected = Literal[
    "PASS_REQUIRED",
    "GRACEFUL_FAILURE_REQUIRED",
    "INTERRUPT_OK",
    "FLAG_FOR_REVIEW",
    "SKIP_UNSUPPORTED",
]

CRASH_MARKERS = (
    "Traceback (most recent call last)",
    "TypeError:",
    "AttributeError:",
    "KeyError:",
    "IndexError:",
    "AssertionError:",
    "RuntimeError:",
    "Exception ignored in:",
)


@dataclass(slots=True)
class Context:
    """Per-run chaos harness state.

    ``create_context`` creates the run directories and seeds ``env`` with
    unbuffered Python plus localhost proxy bypasses so child commands write logs
    promptly and do not route mock-server traffic through HTTP_PROXY.
    """

    base: Path
    url: str
    root: Path
    logs: Path
    artifacts: Path
    fixtures: Path
    env: dict[str, str]


@dataclass(slots=True)
class Case:
    """Single chaos case definition.

    ``run`` receives the shared context, stable case name, and log path, then
    returns ``(return_code, combined_output)``. ``expected`` controls how
    ``verdict_for`` classifies pass/fail/timeout/crash outcomes.
    """

    name: str
    expected: Expected
    run: Callable[[Context, str, Path], tuple[int, str]]
    why: str


def create_context(root_name: str = "aiperf-local-adversarial") -> Context:
    url = os.environ.get("AIPERF_ADVERSARIAL_URL", LOCAL_DEFAULT_URL)
    root = BASE / "tests" / "scripts" / ".chaos_runs" / time.strftime("%Y%m%d-%H%M%S")
    if root_name != "aiperf-local-adversarial":
        root = (
            BASE
            / "tests"
            / "scripts"
            / ".chaos_runs"
            / root_name
            / time.strftime("%Y%m%d-%H%M%S")
        )
    logs = root / "logs"
    artifacts = root / "artifacts"
    fixtures = root / "fixtures"
    for path in (logs, artifacts, fixtures):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return Context(
        base=BASE,
        url=url,
        root=root,
        logs=logs,
        artifacts=artifacts,
        fixtures=fixtures,
        env=env,
    )


def free_port() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = str(sock.getsockname()[1])
    sock.close()
    return port


@contextlib.contextmanager
def start_mock_server(
    log_path: Path | None = None,
    ready_timeout: float = MOCK_SERVER_READY_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Spawn `aiperf-mock-server` on a free port, yield its base URL, kill on exit.

    The server's stdout/stderr is redirected to ``log_path`` (or ``/dev/null``
    if not given). Readiness is detected by polling ``GET /health`` until 200.
    """
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = ["uv", "run", "aiperf-mock-server", "--port", port, "--host", "127.0.0.1"]
    with contextlib.ExitStack() as stack:
        if log_path is not None:
            log_handle = stack.enter_context(log_path.open("w"))
        else:
            log_handle = stack.enter_context(open(os.devnull, "w"))  # noqa: SIM115
        log_handle.write(f"$ {shlex.join(cmd)}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=BASE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        health_url = f"{url}/health"
        deadline = time.monotonic() + ready_timeout
        # Bypass any HTTP_PROXY/http_proxy env so the loopback probe doesn't get
        # routed through a sandbox proxy that returns 405 for our health URL.
        no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            last_err: str = ""
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"mock server exited rc={proc.returncode} before becoming ready; "
                        f"see {log_path or '/dev/null'}"
                    )
                try:
                    with no_proxy_opener.open(health_url, timeout=1) as resp:
                        if 200 <= resp.status < 300:
                            break
                except (urllib.error.URLError, ConnectionError, OSError) as exc:
                    last_err = repr(exc)
                time.sleep(0.2)
            else:
                raise RuntimeError(
                    f"mock server at {url} did not become ready within "
                    f"{ready_timeout}s (last error: {last_err}); see "
                    f"{log_path or '/dev/null'}"
                )
            yield url
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


def run_cmd(
    cmd: list[str],
    log: Path,
    ctx: Context,
    timeout: int = 30,
    *,
    start_new_session: bool = True,
) -> tuple[int, str]:
    """Run a command from the repo root and return its rc plus captured log text.

    Output is streamed to ``log``. On timeout, the child process group is sent
    SIGTERM then SIGKILL when ``start_new_session`` is true; otherwise only the
    child process is terminated. The command inherits ``ctx.env``.
    """
    with log.open("w") as out:
        out.write(f"$ {shlex.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ctx.base,
            env=ctx.env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=start_new_session,
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            out.write(f"\nTIMEOUT after {timeout}s; terminating process group\n")
            if start_new_session:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                out.write("\nSIGTERM timeout; killing process group\n")
                if start_new_session:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=5)
                rc = 124
    return rc, log.read_text(errors="replace")


def tuple_case(item: tuple[str, list[str], int, str]) -> Case:
    name, cmd, timeout, expected = item
    match expected:
        case "pass":
            return pass_case(name, cmd, timeout)
        case "fail":
            return graceful_failure_case(name, cmd, timeout)
        case _:
            raise ValueError(f"unknown legacy case expectation {expected!r} for {name}")


def command_case(
    name: str,
    cmd: list[str],
    expected: Expected,
    why: str,
    timeout: int = 60,
) -> Case:
    def _run(ctx: Context, case_name: str, log: Path) -> tuple[int, str]:
        return run_cmd(cmd, log, ctx, timeout)

    return Case(name=name, expected=expected, run=_run, why=why)


def pass_case(
    name: str,
    cmd: list[str],
    timeout: int = 60,
    why: str = "expected successful command",
) -> Case:
    return command_case(name, cmd, "PASS_REQUIRED", why, timeout)


def graceful_failure_case(
    name: str, cmd: list[str], timeout: int = 60, why: str = "expected graceful failure"
) -> Case:
    return command_case(name, cmd, "GRACEFUL_FAILURE_REQUIRED", why, timeout)


def skip_case(name: str, why: str) -> Case:
    def _run(ctx: Context, case_name: str, log: Path) -> tuple[int, str]:
        log.write_text(f"SKIPPED: {why}\n")
        return 0, log.read_text()

    return Case(name=name, expected="SKIP_UNSUPPORTED", run=_run, why=why)


def verdict_for(expected: Expected, rc: int, text: str) -> str:
    non_debug = "\n".join(line for line in text.splitlines() if " DEBUG " not in line)
    crashed = any(marker in non_debug for marker in CRASH_MARKERS)
    if expected == "PASS_REQUIRED":
        return (
            "OK_PASS"
            if rc == 0 and not crashed
            else ("BUG_CRASH" if crashed else "BUG_UNEXPECTED_FAIL")
        )
    if expected == "GRACEFUL_FAILURE_REQUIRED":
        return (
            "OK_GRACEFUL_FAILURE"
            if rc != 0 and rc != 124 and not crashed
            else ("BUG_CRASH" if crashed else "BUG_UNEXPECTED_PASS_OR_TIMEOUT")
        )
    if expected == "INTERRUPT_OK":
        return (
            "OK_INTERRUPT"
            if rc != 0 and rc != 124 and not crashed
            else ("BUG_CRASH" if crashed else "BUG_INTERRUPT_HUNG_OR_ZERO")
        )
    if expected == "FLAG_FOR_REVIEW":
        return (
            "FLAG_FOR_REVIEW"
            if not crashed and rc != 124
            else ("BUG_CRASH" if crashed else "BUG_TIMEOUT")
        )
    if expected == "SKIP_UNSUPPORTED":
        return "OK_SKIP"
    raise AssertionError(expected)


def run_cases(cases: list[Case], ctx: Context, label: str) -> None:
    results: list[dict[str, object]] = []
    print(f"{label}_ROOT={ctx.root}", flush=True)
    for idx, case in enumerate(cases, 1):
        log = ctx.logs / f"{idx:03d}-{case.name}.log"
        print(
            f"[{idx:03d}/{len(cases):03d}] RUN {case.name} expected={case.expected} why={case.why}",
            flush=True,
        )
        started = time.monotonic()
        try:
            rc, text = case.run(ctx, case.name, log)
        except Exception as exc:
            rc = 125
            with log.open("w") as out:
                out.write(f"HARNESS_EXCEPTION: {exc!r}\n")
            text = log.read_text(errors="replace")
        verdict = verdict_for(case.expected, rc, text)
        elapsed = time.monotonic() - started
        print(
            f"[{idx:03d}/{len(cases):03d}] {verdict} rc={rc} elapsed={elapsed:.1f}s log={log}",
            flush=True,
        )
        results.append(
            {
                "name": case.name,
                "expected": case.expected,
                "why": case.why,
                "rc": rc,
                "verdict": verdict,
                "elapsed_seconds": elapsed,
                "log": str(log),
            }
        )
    summary = ctx.root / "summary.json"
    summary.write_text(
        json.dumps(
            {"root": str(ctx.root), "url": ctx.url, "results": results}, indent=2
        )
    )
    bugs = [r for r in results if str(r["verdict"]).startswith("BUG")]
    flags = [r for r in results if r["verdict"] == "FLAG_FOR_REVIEW"]
    skips = [r for r in results if r["verdict"] == "OK_SKIP"]
    print(f"SUMMARY={summary}", flush=True)
    print(
        f"OK={len(results) - len(bugs) - len(flags) - len(skips)} "
        f"FLAGS={len(flags)} SKIPS={len(skips)} BUGS={len(bugs)} TOTAL={len(results)}",
        flush=True,
    )
    if flags:
        print("FLAG_NAMES=" + ",".join(str(r["name"]) for r in flags), flush=True)
    if bugs:
        print(
            "BUG_NAMES="
            + ",".join(str(r["name"]) + ":" + str(r["verdict"]) for r in bugs),
            flush=True,
        )
        raise SystemExit(1)
