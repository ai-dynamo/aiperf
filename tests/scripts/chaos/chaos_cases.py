from __future__ import annotations

import contextlib
import http.server
import json
import os
import signal
import socket
import socketserver
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent

from tests.scripts.chaos.harness import Case, Context, free_port, run_cmd, skip_case
from tests.scripts.chaos.toxiproxy_local import Toxic, chaos_proxy, find_toxiproxy_bin


def base_profile(
    ctx: Context,
    name: str,
    *args: str,
    url: str | None = None,
    artifact: Path | None = None,
) -> list[str]:
    return [
        "uv",
        "run",
        "aiperf",
        "profile",
        "--model",
        "mock-model",
        "--url",
        url or ctx.url,
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
        "--artifact-dir",
        str(artifact or ctx.artifacts / name),
        "--no-gpu-telemetry",
        *args,
    ]


class MalformedJsonHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send()

    def do_POST(self) -> None:
        self._send()

    def _send(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"not valid json"')

    def log_message(self, format: str, *args: object) -> None:
        return


class Always500Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send()

    def do_POST(self) -> None:
        self._send()

    def _send(self) -> None:
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"forced chaos"}')

    def log_message(self, format: str, *args: object) -> None:
        return


class SlowHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send()

    def do_POST(self) -> None:
        self._send()

    def _send(self) -> None:
        time.sleep(3)
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"id":"slow","object":"chat.completion","choices":[{"message":{"content":"slow"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
            )

    def log_message(self, format: str, *args: object) -> None:
        return


@contextlib.contextmanager
def http_server(handler: type[http.server.BaseHTTPRequestHandler]) -> Iterator[str]:
    """Run a one-off loopback HTTP server for a chaos handler.

    Starts ``handler`` on a free localhost port in a daemon thread, yields the
    base URL, then shuts down the TCPServer and joins the thread on context exit.
    """
    port = int(free_port())
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def case_artifact_path_is_file(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    artifact = ctx.artifacts / "artifact-path-is-file"
    artifact.write_text("not a directory")
    return run_cmd(
        base_profile(ctx, name, "--endpoint-type", "chat", artifact=artifact),
        log,
        ctx,
        60,
    )


def case_read_only_artifact_parent(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    parent = ctx.artifacts / "readonly-parent"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        return run_cmd(
            base_profile(
                ctx, name, "--endpoint-type", "chat", artifact=parent / "child"
            ),
            log,
            ctx,
            60,
        )
    finally:
        parent.chmod(0o700)


def case_duplicate_api_port(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = str(sock.getsockname()[1])
    try:
        return run_cmd(
            base_profile(
                ctx,
                name,
                "--endpoint-type",
                "chat",
                "--api-host",
                "127.0.0.1",
                "--api-port",
                port,
            ),
            log,
            ctx,
            60,
        )
    finally:
        sock.close()


def case_bad_config_unknown_nested(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    cfg = ctx.fixtures / "unknown_nested.yaml"
    cfg.write_text(
        dedent(f"""
        model: mock-model
        endpoint:
          urls: [{ctx.url}]
          type: chat
          unexpectedNested: true
        dataset:
          type: synthetic
          entries: 1
          prompts:
            isl: 16
            osl: 8
        phases:
          type: concurrency
          concurrency: 1
          requests: 1
        artifacts:
          dir: {ctx.artifacts / name}
        tokenizer:
          name: builtin
        runtime:
          ui: none
    """).strip()
    )
    return run_cmd(
        ["uv", "run", "aiperf", "config", "validate", str(cfg)], log, ctx, 30
    )


def case_bad_template_response_field(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    cfg = ctx.fixtures / "bad_template_response_field.yaml"
    cfg.write_text(
        dedent(f"""
        model: mock-model
        endpoint:
          urls:
            - {ctx.url}
          type: template
          path: /v1/custom-multimodal
          template:
            body: '{{{{ {{"inference_params": {{"model_id": model}}, "modality_bundle": {{"text_fragments": [text]}}}} | tojson }}}}'
            responseField: definitely.missing.path
        dataset:
          type: synthetic
          entries: 1
          prompts:
            isl: 16
            osl: 8
        phases:
          type: concurrency
          concurrency: 1
          requests: 1
        artifacts:
          dir: {ctx.artifacts / name}
        tokenizer:
          name: builtin
        runtime:
          ui: none
        gpuTelemetry:
          enabled: false
    """).strip()
    )
    return run_cmd(
        ["uv", "run", "aiperf", "profile", "--config", str(cfg)], log, ctx, 60
    )


def case_custom_endpoint_404(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    return run_cmd(
        base_profile(
            ctx,
            name,
            "--endpoint-type",
            "chat",
            "--custom-endpoint",
            "/v1/definitely-missing",
        ),
        log,
        ctx,
        60,
    )


def case_server_500(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    with http_server(Always500Handler) as url:
        return run_cmd(
            base_profile(ctx, name, "--endpoint-type", "chat", url=url), log, ctx, 60
        )


def case_malformed_json(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    with http_server(MalformedJsonHandler) as url:
        return run_cmd(
            base_profile(ctx, name, "--endpoint-type", "chat", url=url), log, ctx, 60
        )


def case_slow_timeout(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    with http_server(SlowHandler) as url:
        cmd = base_profile(ctx, name, "--endpoint-type", "chat", url=url)
        cmd[cmd.index("--request-timeout-seconds") + 1] = "1"
        return run_cmd(cmd, log, ctx, 25)


def case_interrupt_profile(ctx: Context, name: str, log: Path) -> tuple[int, str]:
    with http_server(SlowHandler) as url:
        cmd = base_profile(ctx, name, "--endpoint-type", "chat", url=url)
        cmd[cmd.index("--request-timeout-seconds") + 1] = "30"
        with log.open("w") as out:
            out.write(f"$ {' '.join(cmd)}\n\n")
            proc = subprocess.Popen(
                cmd,
                cwd=ctx.base,
                env=ctx.env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            time.sleep(3)
            os.killpg(proc.pid, signal.SIGINT)
            try:
                rc = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = 124
                out.write("\nTIMEOUT waiting after SIGINT\n")
    return rc, log.read_text(errors="replace")


def case_same_artifact_concurrent(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    shared = ctx.artifacts / "shared-artifact-dir"
    cmd1 = base_profile(ctx, name + "-a", "--endpoint-type", "chat", artifact=shared)
    cmd2 = base_profile(ctx, name + "-b", "--endpoint-type", "chat", artifact=shared)
    with log.open("w") as out:
        out.write(f"$ {' '.join(cmd1)}\n$ {' '.join(cmd2)}\n\n")
        p1 = subprocess.Popen(
            cmd1,
            cwd=ctx.base,
            env=ctx.env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        time.sleep(0.5)
        p2 = subprocess.Popen(
            cmd2,
            cwd=ctx.base,
            env=ctx.env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            rc1 = p1.wait(timeout=45)
            rc2 = p2.wait(timeout=45)
        except subprocess.TimeoutExpired:
            for proc in (p1, p2):
                if proc.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(2)
            for proc in (p1, p2):
                if proc.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
            out.write("\nTIMEOUT waiting for concurrent artifact runs\n")
            return 124, log.read_text(errors="replace")
        out.write(f"\nRC1={rc1} RC2={rc2}\n")
    if rc1 == 0 and rc2 == 0:
        return 0, log.read_text(errors="replace")
    return 1, log.read_text(errors="replace")


def case_success_output_integrity(
    ctx: Context, name: str, log: Path
) -> tuple[int, str]:
    artifact = ctx.artifacts / name
    rc, text = run_cmd(
        base_profile(ctx, name, "--endpoint-type", "chat", artifact=artifact),
        log,
        ctx,
        60,
    )
    if rc != 0:
        return rc, text
    try:
        json.loads((artifact / "profile_export_aiperf.json").read_text())
        csv_text = (artifact / "profile_export_aiperf.csv").read_text()
        if "Metric," not in csv_text and "Request Count" not in csv_text:
            with log.open("a") as out:
                out.write(
                    "\nOUTPUT_INTEGRITY_ERROR: CSV missing expected header text\n"
                )
            return 1, log.read_text(errors="replace")
    except Exception as exc:
        with log.open("a") as out:
            out.write(f"\nOUTPUT_INTEGRITY_ERROR: {exc!r}\n")
        return 1, log.read_text(errors="replace")
    return 0, text


def _network_case(toxics: list[Toxic], request_timeout: str, harness_timeout: int):
    def _run(ctx: Context, name: str, log: Path) -> tuple[int, str]:
        with chaos_proxy(ctx.url, toxics) as proxied:
            cmd = base_profile(ctx, name, "--endpoint-type", "chat", url=proxied)
            cmd[cmd.index("--request-timeout-seconds") + 1] = request_timeout
            return run_cmd(cmd, log, ctx, harness_timeout)

    return _run


NETWORK_CASE_SPECS: tuple[tuple[str, str, str, list[Toxic], str, int], ...] = (
    (
        "network-latency-2s",
        "FLAG_FOR_REVIEW",
        "2s downstream latency on every request",
        [Toxic("latency", {"latency": 2000, "jitter": 100})],
        "10",
        90,
    ),
    (
        "network-bandwidth-cap",
        "FLAG_FOR_REVIEW",
        "50 KB/s downstream bandwidth cap",
        [Toxic("bandwidth", {"rate": 50})],
        "30",
        120,
    ),
    (
        "network-reset-peer",
        "GRACEFUL_FAILURE_REQUIRED",
        "TCP RST after 100ms; aiperf must fail gracefully",
        [Toxic("reset_peer", {"timeout": 100})],
        "10",
        60,
    ),
    (
        "network-slow-close",
        "FLAG_FOR_REVIEW",
        "5s slow_close on every connection teardown",
        [Toxic("slow_close", {"delay": 5000})],
        "30",
        90,
    ),
    (
        "network-timeout-toxic",
        "GRACEFUL_FAILURE_REQUIRED",
        "200ms timeout toxic; expect request retries to honor timeout",
        [Toxic("timeout", {"timeout": 200})],
        "5",
        60,
    ),
)


def build_network_cases() -> list[Case]:
    if find_toxiproxy_bin() is None:
        return [
            skip_case(name, f"toxiproxy-server not on PATH ({why})")
            for name, _expected, why, _toxics, _rt, _ht in NETWORK_CASE_SPECS
        ]
    return [
        Case(
            name=name,
            expected=expected,  # type: ignore[arg-type]
            run=_network_case(toxics, request_timeout, harness_timeout),
            why=why,
        )
        for name, expected, why, toxics, request_timeout, harness_timeout in NETWORK_CASE_SPECS
    ]


def build_cases() -> list[Case]:
    return [
        Case(
            "artifact-path-is-file",
            "GRACEFUL_FAILURE_REQUIRED",
            case_artifact_path_is_file,
            "artifact dir path already exists as a file",
        ),
        Case(
            "read-only-artifact-parent",
            "GRACEFUL_FAILURE_REQUIRED",
            case_read_only_artifact_parent,
            "artifact parent denies writes",
        ),
        Case(
            "duplicate-local-api-port",
            "GRACEFUL_FAILURE_REQUIRED",
            case_duplicate_api_port,
            "local API port is already bound",
        ),
        Case(
            "bad-config-unknown-nested",
            "GRACEFUL_FAILURE_REQUIRED",
            case_bad_config_unknown_nested,
            "config contains unknown nested endpoint field",
        ),
        Case(
            "bad-template-response-field",
            "GRACEFUL_FAILURE_REQUIRED",
            case_bad_template_response_field,
            "template response field does not exist",
        ),
        Case(
            "custom-endpoint-404",
            "GRACEFUL_FAILURE_REQUIRED",
            case_custom_endpoint_404,
            "server returns 404 for valid command with custom path",
        ),
        Case(
            "server-500",
            "GRACEFUL_FAILURE_REQUIRED",
            case_server_500,
            "server returns HTTP 500",
        ),
        Case(
            "malformed-json",
            "GRACEFUL_FAILURE_REQUIRED",
            case_malformed_json,
            "server returns malformed JSON",
        ),
        Case(
            "slow-timeout",
            "FLAG_FOR_REVIEW",
            case_slow_timeout,
            "server exceeds request timeout; bounded by harness timeout",
        ),
        Case(
            "interrupt-profile",
            "INTERRUPT_OK",
            case_interrupt_profile,
            "SIGINT during active profile",
        ),
        Case(
            "same-artifact-concurrent",
            "FLAG_FOR_REVIEW",
            case_same_artifact_concurrent,
            "two successful runs share one artifact dir concurrently",
        ),
        Case(
            "success-output-integrity",
            "PASS_REQUIRED",
            case_success_output_integrity,
            "successful chat run exports parseable outputs",
        ),
        *build_network_cases(),
    ]
