# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator-provider v2 worker, host ledger, and stock-pair proofs."""

from __future__ import annotations

import asyncio
import builtins
import importlib.metadata
import os
import re
import select
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from aiperf.accuracy.evaluation import resource_bootstrap
from aiperf.accuracy.evaluation.canonical import (
    CanonicalJsonError,
    canonical_dumps,
    canonical_loads,
    canonical_sha256,
)
from aiperf.accuracy.evaluation.contracts import (
    CallContext,
    EvaluationQueueCredits,
    HostOperationEvent,
    HostOperationRequest,
    ResponseMode,
    ScopedProxyBinding,
)
from aiperf.accuracy.evaluation.distributions import (
    MAX_PROCESSES,
    NEMO_EVALUATOR_DISTRIBUTION,
    OPENBENCH_DISTRIBUTION,
    RESOURCE_BOOTSTRAP,
    DistributionEvidence,
    executable_tasks,
    task_manifest,
)
from aiperf.accuracy.evaluation.host import PipeEvaluationHost
from aiperf.accuracy.evaluation.operation_schemas import (
    MODEL_GENERATE_SCHEMA,
    MODEL_RESPONSES_SCHEMA,
    OPERATION_DIRECTION_SCHEMA_SHA256,
    OPERATION_SCHEMA_SHA256,
)
from aiperf.accuracy.evaluation.providers.nemo_evaluator import (
    _binary_public_reward,
)
from aiperf.accuracy.evaluation.providers.openbench import _binary_public_score
from aiperf.accuracy.evaluation.worker import EvaluatorWorker, WorkerProtocolError
from tools.generate_stock_evaluator_manifest import materialize

_ROOT = Path(__file__).resolve().parents[3]
_STOCK_MANIFEST = (
    _ROOT / "src/aiperf/accuracy/evaluation/manifests/stock_distributions.json"
)
_ASSET_SHA256 = "fc9b5c03206d193c0013baf2d6344a133fe0096a2b47cd1eafdcee297dfd398a"
_ASSET_REVISION = (
    "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866:main:test:first5"
)
_NEMO_PROVIDER_ROOT = Path(
    os.environ.get(
        "AIPERF_TEST_NEMO_PROVIDER_ROOT",
        _ROOT / "tools/stock_evaluators/nemo/.venv",
    )
)
_OPENBENCH_PROVIDER_ROOT = Path(
    os.environ.get(
        "AIPERF_TEST_OPENBENCH_PROVIDER_ROOT",
        _ROOT / "tools/stock_evaluators/openbench/.venv",
    )
)


class _OpenAiUdsFixture:
    """One-request OpenAI terminal fixture for the pinned Inspect SDK path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.request: dict[str, Any] | None = None
        self.headers: dict[str, str] = {}
        self.error: BaseException | None = None
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=10)
        assert not self._thread.is_alive(), "OpenAI UDS fixture did not terminate"
        if self.error is not None:
            raise self.error

    def _serve(self) -> None:
        try:
            connection, _ = self._listener.accept()
            with connection, connection.makefile("rb") as stream:
                request_line = stream.readline().decode("ascii")
                assert request_line.startswith("POST /v1/chat/completions HTTP/1.1")
                while True:
                    line = stream.readline()
                    if line == b"\r\n":
                        break
                    name, value = line.decode("ascii").split(":", 1)
                    self.headers[name.lower()] = value.strip()
                length = int(self.headers["content-length"])
                self.request = canonical_loads(stream.read(length))
                body = canonical_dumps(
                    {
                        "id": "chatcmpl-openbench-proof",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "candidate",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Reasoning complete. Answer: 18",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 7,
                            "total_tokens": 17,
                        },
                    }
                )
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode("ascii")
                    + b"Connection: close\r\n\r\n"
                    + body
                )
        except BaseException as error:  # noqa: BLE001 - re-raised on the test thread.
            self.error = error
        finally:
            self._listener.close()


def test_canonical_json_rejects_duplicate_keys_and_normalizes_numbers() -> None:
    assert canonical_loads(b'{"b":-0.0,"a":1e+00}') == {"a": 1.0, "b": -0.0}
    assert canonical_dumps({"b": -0.0, "a": 1.0}) == b'{"a":1.0,"b":0}'
    with pytest.raises(CanonicalJsonError, match="duplicate"):
        canonical_loads(b'{"a":1,"a":2}')


def test_proxy_binding_has_prelaunch_purpose_authority_but_no_case_authority() -> None:
    wire = {
        "local_locator": "unix:///run/aiperf/evaluator-proxy.sock",
        "grant": {
            "grant_id": "grant-1",
            "session_id": "session-1",
            "secret": "s" * 32,
            "service_ids": ["candidate"],
            "purposes": ["primary"],
            "semantic_operation_ids": ["model.generate"],
            "process_scope_sha256": "a" * 64,
            "max_operations": 2,
            "max_concurrent_operations": 1,
            "max_request_bytes": 1024,
            "max_response_bytes": 2048,
            "max_stream_events": 4,
            "expires_after_ms": 1000,
        },
    }
    binding = ScopedProxyBinding.from_wire(wire)
    assert binding.purposes == ("primary",)
    assert not hasattr(binding, "case_ids")

    wire["grant"]["case_ids"] = ["case-1"]
    with pytest.raises(ValueError, match="unknown"):
        ScopedProxyBinding.from_wire(wire)
    wire["grant"].pop("case_ids")
    wire["host_socket_path"] = "/tmp/host.sock"
    with pytest.raises(ValueError, match="unknown"):
        ScopedProxyBinding.from_wire(wire)
    wire.pop("host_socket_path")
    wire["local_locator"] = "http://127.0.0.1:1234"
    with pytest.raises(ValueError, match="contained AIPerf socket"):
        ScopedProxyBinding.from_wire(wire)


def test_stock_product_pairs_and_schema_fingerprints_are_exact() -> None:
    assert executable_tasks(NEMO_EVALUATOR_DISTRIBUTION) == ("gsm8k",)
    assert executable_tasks(OPENBENCH_DISTRIBUTION) == ("gsm8k",)
    assert NEMO_EVALUATOR_DISTRIBUTION.config_schema_sha256 == (
        "b501baba9601933a8239e15b34fba57aa06ebaa6deb4d3132544ae5c5c9b47c4"
    )
    assert OPENBENCH_DISTRIBUTION.config_schema_sha256 == (
        "2c1a1a970a9695dc8d741096f2fc1b92cd57c7823b6b6c53b20f37e78f4da57b"
    )
    assert OPERATION_SCHEMA_SHA256["model.generate"] == (
        "5077609c909bb093f7e1db8617318fffc90947f6799aae9f8d7aced35107f416"
    )
    assert dict(OPERATION_DIRECTION_SCHEMA_SHA256["model.generate"]) == {
        "request": "0d89f49e356ebae5f637768b43dc0e957f25b1e5ee5a9148df3ddbb2e932c96d",
        "response": "1c2284478c7e01acaa3a88e611cd7d09f38f8374e08213582898d94ad56cf297",
        "stream": "025ddc6243a449525a8d9db0cf3afc4d311265329b96ea73e9014da224100582",
    }
    assert OPERATION_SCHEMA_SHA256["model.responses"] == (
        "b7441ef4a0fd0ea2cbb2410c09f3f18ad8fab00fdcea08c59bfc34fb1368cc9b"
    )
    assert dict(OPERATION_DIRECTION_SCHEMA_SHA256["model.responses"]) == {
        "request": "6afa8c604041566bb843367664c8ffff6961d0f17a5c25031b6a745991219f96",
        "response": "530ae988b7935903f54292c9c20ec7118a02688402d18bd66e9e4a94df5e7086",
        "stream": "800695bec0f214e79c9eb0b469ce020fc2931167126cb0d4e0ca7cce2d2f262e",
    }
    for descriptor in (
        NEMO_EVALUATOR_DISTRIBUTION,
        OPENBENCH_DISTRIBUTION,
    ):
        argv = descriptor.fixed_argv
        assert argv[:4] == (
            "-I",
            RESOURCE_BOOTSTRAP,
            "--max-processes",
            str(MAX_PROCESSES),
        )
        assert RESOURCE_BOOTSTRAP.startswith("/runtime/libexec/")
        assert "-m" not in argv
        assert "--stdio" not in argv
        assert argv[argv.index("--read-fd") + 1] == "3"
        assert argv[argv.index("--write-fd") + 1] == "4"
        environment = descriptor.clean_environment
        assert set(environment).isdisjoint(
            {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "OPENAI_API_KEY"}
        )
        assert environment["HOME"].startswith("/staging/")
        assert environment["TMPDIR"].startswith("/staging/")
        assert environment["XDG_DATA_HOME"].startswith("/staging/")


def test_resource_bootstrap_installs_limit_before_worker_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    limits = iter(
        [
            (resource_bootstrap.resource.RLIM_INFINITY,) * 2,
            (MAX_PROCESSES, MAX_PROCESSES),
        ]
    )

    def getrlimit(resource_id: int) -> tuple[int, int]:
        assert resource_id == resource_bootstrap.resource.RLIMIT_NPROC
        events.append("getrlimit")
        return next(limits)

    def setrlimit(resource_id: int, value: tuple[int, int]) -> None:
        assert resource_id == resource_bootstrap.resource.RLIMIT_NPROC
        events.append(("setrlimit", value))

    worker_module = type(
        "WorkerModule",
        (),
        {"main": staticmethod(lambda argv: events.append(("worker", argv)))},
    )
    original_import = builtins.__import__

    def import_module(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "aiperf.accuracy.evaluation.worker":
            events.append("worker-import")
            return worker_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(resource_bootstrap.resource, "getrlimit", getrlimit)
    monkeypatch.setattr(resource_bootstrap.resource, "setrlimit", setrlimit)
    monkeypatch.setattr(builtins, "__import__", import_module)

    worker_args = ["--provider", "nemo_evaluator"]
    resource_bootstrap.main(["--max-processes", str(MAX_PROCESSES), *worker_args])

    assert events == [
        "getrlimit",
        ("setrlimit", (MAX_PROCESSES, MAX_PROCESSES)),
        "getrlimit",
        "worker-import",
        ("worker", worker_args),
    ]


@pytest.mark.parametrize("value", ["", "0", "01", "+1", "-1", "1.0", "one"])
def test_resource_bootstrap_rejects_noncanonical_process_limits(value: str) -> None:
    with pytest.raises(ValueError, match="canonical positive integer"):
        resource_bootstrap._parse_max_processes(value)


def test_resource_bootstrap_fails_when_inherited_hard_limit_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource_bootstrap.resource,
        "getrlimit",
        lambda _: (MAX_PROCESSES - 1, MAX_PROCESSES - 1),
    )
    with pytest.raises(RuntimeError, match="cannot satisfy"):
        resource_bootstrap._install_process_limit(MAX_PROCESSES)


def test_model_operation_schema_confines_inline_images_to_strict_raster_data() -> None:
    generate_content = MODEL_GENERATE_SCHEMA["request"]["properties"]["messages"][
        "items"
    ]["properties"]["content"]["oneOf"][1]["items"]["oneOf"]
    responses_content = MODEL_RESPONSES_SCHEMA["request"]["properties"]["input"][
        "items"
    ]["properties"]["content"]["oneOf"][1]["items"]["oneOf"]
    image = next(
        block
        for block in generate_content
        if block["properties"]["type"].get("const") == "image_url"
    )
    assert image in responses_content
    url_schema = image["properties"]["image_url"]["properties"]["url"]
    assert url_schema["maxLength"] == 1_048_576
    pattern = re.compile(url_schema["pattern"])
    for url in (
        "data:image/gif;base64,R0lGODlhAQABAAAAACw=",
        "data:image/jpeg;base64,/9j/2Q==",
        "data:image/png;base64,aW5lcnQ=",
        "data:image/webp;base64,UklGRg==",
    ):
        assert pattern.fullmatch(url), url
    for url in (
        "",
        "https://model.invalid/image.png",
        "file:///etc/passwd",
        "data:image/svg+xml;base64,PHN2Zz4=",
        "data:image/png;base64,not-base64!",
        "data:image/png;base64,A===",
    ):
        assert pattern.fullmatch(url) is None, url


def test_python_operation_fingerprints_match_rust_stock_registry() -> None:
    rust_source = (_ROOT / "crates/aiperf-accuracy/src/provider.rs").read_text(
        encoding="utf-8"
    )
    rows = {
        operation: {
            "combined": combined,
            "request": request,
            "response": response,
            "stream": stream,
        }
        for operation, combined, request, response, stream in re.findall(
            r'StockEvaluationOperationSchema\s*\{\s*operation_id:\s*"([^"]+)",'
            r'\s*combined_schema_sha256:\s*"([0-9a-f]{64})",'
            r'\s*request_schema_sha256:\s*"([0-9a-f]{64})",'
            r'\s*response_schema_sha256:\s*"([0-9a-f]{64})",'
            r'\s*canonical_stream_schema_sha256:\s*"([0-9a-f]{64})",',
            rust_source,
        )
    }
    assert set(rows) == set(OPERATION_SCHEMA_SHA256)
    for operation, combined in OPERATION_SCHEMA_SHA256.items():
        assert rows[operation] == {
            "combined": combined,
            **dict(OPERATION_DIRECTION_SCHEMA_SHA256[operation]),
        }


def test_stock_provider_public_score_schema_is_exact_binary_object() -> None:
    for validator in (_binary_public_reward, _binary_public_score):
        assert validator(0) == {"value": 0.0}
        assert validator(1.0) == {"value": 1.0}
        for value in (0.5, -1, 2, True, "1", float("nan"), float("inf")):
            with pytest.raises(RuntimeError, match="binary"):
                validator(value)

    for descriptor in (NEMO_EVALUATOR_DISTRIBUTION, OPENBENCH_DISTRIBUTION):
        manifest = task_manifest(descriptor)
        entries = manifest.get("environments", manifest.get("tasks"))
        score = entries["gsm8k"]["public_projection"]["score_schemas"][0]
        assert score["projection_id"] == "gsm8k_binary_score_v1"
        assert score["schema_sha256"] == (
            "d156e6577305139bac7f48946996fa35d489a381a87bce4c58d18c47d8d9eeb5"
        )
        assert canonical_sha256(score["schema"]) == score["schema_sha256"]
        assert score["schema"] == {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": False,
            "properties": {"value": {"enum": [0, 1], "type": "number"}},
            "required": ["value"],
            "type": "object",
        }


@pytest.mark.asyncio
async def test_pipe_host_enforces_terminal_uniqueness() -> None:
    emitted: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def emit(event: dict[str, Any]) -> None:
        await emitted.put(event)

    host = PipeEvaluationHost(
        emit,
        EvaluationQueueCredits(
            units=1,
            host_operations=1,
            host_operations_per_unit=1,
            stream_events=4,
            sandboxes=0,
            processes=0,
            artifacts=1,
            artifact_bytes=1024,
        ),
    )
    request = HostOperationRequest(
        operation_id="operation-1",
        context=CallContext(
            session_id="session-1",
            unit_id="unit-1",
            case_id="case-1",
            semantic_attempt_id="attempt-1",
            logical_call_id="call-1",
        ),
        service_id="candidate",
        purpose="primary",
        semantic_operation_id="model.generate",
        payload={"messages": [], "generation": {"max_tokens": 1}},
        response_mode=ResponseMode.TERMINAL,
        idempotency_key="idempotency-1",
    )
    execution = asyncio.create_task(host.execute(request))
    emitted_request = await emitted.get()
    assert emitted_request["kind"] == "host_operation_requested"
    terminal = HostOperationEvent.from_wire(
        _terminal_event("operation-1", "attempt-1", "Answer: 18")
    )
    assert await host.submit_events((terminal,)) == ("operation-1",)
    assert (await execution).result is not None
    with pytest.raises(ValueError, match="late/duplicate"):
        await host.submit_events((terminal,))
    assert host.is_drained
    await host.close()


@pytest.mark.asyncio
async def test_worker_hello_rejects_regressing_request_ids(tmp_path: Path) -> None:
    worker = EvaluatorWorker(
        NEMO_EVALUATOR_DISTRIBUTION,
        tmp_path,
        evidence=DistributionEvidence("1" * 64, "2" * 64, "3" * 64),
    )
    await worker.dispatch(
        {
            "op": "hello",
            "id": 1,
            "protocol": 2,
            "max_message_bytes": 1024 * 1024,
            "max_collection_items": 1024,
            "launch_nonce": "n" * 32,
        }
    )
    with pytest.raises(WorkerProtocolError, match="regressing"):
        await worker.dispatch(
            {
                "op": "plan_session",
                "id": 1,
                "request": {},
            }
        )


@pytest.mark.parametrize(
    ("provider", "distribution", "config", "requirements"),
    [
        (
            "nemo_evaluator",
            NEMO_EVALUATOR_DISTRIBUTION.distribution_id,
            {
                "environment": "gsm8k",
                "solver": "chat",
                "solver_config": {"max_tokens": 64},
                "selection": {"limit": 1, "seed": 0},
            },
            {"nemo-evaluator": "0.4.0"},
        ),
        (
            "openbench",
            OPENBENCH_DISTRIBUTION.distribution_id,
            {"task": "gsm8k", "task_args": {}, "epochs": 1, "limit": 1},
            {"openbench": "0.5.3", "inspect-ai": "0.3.141"},
        ),
    ],
    ids=("nemo_evaluator_gsm8k_canary", "openbench_gsm8k_canary"),
)
def test_stock_provider_over_dedicated_fds(
    provider: str,
    distribution: str,
    config: dict[str, Any],
    requirements: dict[str, str],
    tmp_path: Path,
) -> None:
    """Run the real pinned provider lifecycle over inherited one-way pipes."""
    provider_root = (
        _NEMO_PROVIDER_ROOT
        if provider == "nemo_evaluator"
        else _OPENBENCH_PROVIDER_ROOT
    )
    if not _requirements_present(requirements, provider_root):
        _skip_optional_stock_proof(
            "exact external evaluator distribution is not installed"
        )
    descriptor = (
        NEMO_EVALUATOR_DISTRIBUTION
        if provider == "nemo_evaluator"
        else OPENBENCH_DISTRIBUTION
    )
    worker_root = tmp_path / "worker-root"
    materialize(
        distribution,
        worker_root,
        nemo_root=_NEMO_PROVIDER_ROOT,
        openbench_root=_OPENBENCH_PROVIDER_ROOT,
    )
    for relative in ("work", "staging", "proc", "dev", "run/aiperf"):
        mountpoint = worker_root / relative
        assert mountpoint.is_dir()
        assert not mountpoint.is_symlink()
        assert not tuple(mountpoint.iterdir())
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    asset = worker_root / "assets/gsm8k_canary.jsonl"
    assert asset.is_file()
    request_read_fd, request_write_fd = os.pipe()
    response_read_fd, response_write_fd = os.pipe()
    environment = dict(descriptor.clean_environment)
    environment["PATH"] = str(worker_root / "runtime/bin")
    for key in ("HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        directory = staging_root / Path(environment[key]).relative_to("/staging")
        directory.mkdir(parents=True, exist_ok=True)
        environment[key] = str(directory)
    worker_command = [
        str(worker_root / "runtime/bin/python3.12"),
        "-I",
        "-m",
        "aiperf.accuracy.evaluation.worker",
        "--provider",
        provider,
        "--distribution",
        distribution,
        "--read-fd",
        str(request_read_fd),
        "--write-fd",
        str(response_write_fd),
        "--staging-root",
        str(staging_root),
    ]
    proxy_fixture: _OpenAiUdsFixture | None = None
    if provider == "openbench":
        bubblewrap = Path("/usr/bin/bwrap")
        if not bubblewrap.is_file():
            _skip_optional_stock_proof("registered Bubblewrap is unavailable")
        preflight = subprocess.run(
            [
                str(bubblewrap),
                "--unshare-all",
                "--ro-bind",
                "/",
                "/",
                "--",
                "/bin/true",
            ],
            check=False,
            capture_output=True,
        )
        if preflight.returncode != 0:
            _skip_optional_stock_proof(
                "unprivileged Bubblewrap namespaces are unavailable"
            )
        proxy_socket = tmp_path / "evaluator-proxy.sock"
        proxy_fixture = _OpenAiUdsFixture(proxy_socket)
        proxy_fixture.start()
        environment = dict(descriptor.clean_environment)
        worker_command = [
            str(bubblewrap),
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--ro-bind",
            str(worker_root),
            "/",
            "--bind",
            str(staging_root),
            "/staging",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/run/aiperf",
            "--ro-bind",
            str(proxy_socket),
            "/run/aiperf/evaluator-proxy.sock",
            "--chdir",
            "/work",
            "--",
            "/runtime/bin/python3.12",
            "-I",
            "-m",
            "aiperf.accuracy.evaluation.worker",
            "--provider",
            provider,
            "--distribution",
            distribution,
            "--read-fd",
            str(request_read_fd),
            "--write-fd",
            str(response_write_fd),
            "--staging-root",
            "/staging",
        ]
    process = subprocess.Popen(
        worker_command,
        env=environment,
        pass_fds=(request_read_fd, response_write_fd),
        cwd=worker_root / "work",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.close(request_read_fd)
    os.close(response_write_fd)
    request_writer = os.fdopen(request_write_fd, "wb", buffering=0)
    response_reader = os.fdopen(response_read_fd, "rb", buffering=0)
    request_id = 0

    def call(operation: str, **fields: Any) -> dict[str, Any]:
        nonlocal request_id
        request_id += 1
        request_writer.write(
            canonical_dumps({"op": operation, "id": request_id, **fields}) + b"\n"
        )
        readable, _, _ = select.select([response_reader], [], [], 10)
        if not readable:
            assert process.stderr is not None
            stderr_ready, _, _ = select.select([process.stderr], [], [], 0)
            diagnostic = (
                os.read(process.stderr.fileno(), 65_536).decode(
                    "utf-8", errors="replace"
                )
                if stderr_ready
                else ""
            )
            pytest.fail(
                f"worker timed out during {operation}; poll={process.poll()}; {diagnostic}"
            )
        raw = response_reader.readline()
        if not raw:
            assert process.stderr is not None
            diagnostic = process.stderr.read().decode("utf-8", errors="replace")
            pytest.fail(
                f"worker pipe closed during {operation}; poll={process.poll()}; "
                f"{diagnostic}"
            )
        response = canonical_loads(raw)
        assert response["id"] == request_id
        assert response["ok"], response.get("error")
        return response["result"]

    try:
        hello = call(
            "hello",
            protocol=2,
            max_message_bytes=8 * 1024 * 1024,
            max_collection_items=16_384,
            launch_nonce="dedicated-fd-proof-" + "n" * 32,
        )
        assert hello["provider_id"] == provider
        stock = canonical_loads(_STOCK_MANIFEST.read_bytes(), max_bytes=8 * 1024 * 1024)
        registered = next(
            item
            for item in stock["distributions"]
            if item["distribution_id"] == distribution
        )
        for field in (
            "provider_source_sha256",
            "worker_source_sha256",
            "dependency_lock_sha256",
        ):
            assert hello[field] == registered[field]
        plan = call(
            "plan_session",
            request={
                "session_id": f"{provider}-proof",
                "provider_id": provider,
                "distribution_id": distribution,
                "config_schema_version": descriptor.config_schema_version,
                "config_schema_sha256": descriptor.config_schema_sha256,
                "provider_config": config,
                "reproducible": True,
            },
        )
        assert plan["finite_case_count"] == 1
        contained_asset = (
            "/assets/gsm8k_canary.jsonl" if provider == "openbench" else str(asset)
        )
        bind_fields: dict[str, Any] = {
            "assets": [
                {
                    "asset_id": "openai_gsm8k_main_test_canary",
                    "contained_path": contained_asset,
                    "content_sha256": _ASSET_SHA256,
                    "immutable_revision": _ASSET_REVISION,
                    "media_type": "application/x-ndjson",
                }
            ],
            "host_binding": {
                "host": {
                    "runner_sha256": "4" * 64,
                    "capability_inventory_sha256": "5" * 64,
                    "schema_inventory_sha256": "6" * 64,
                    "isolation_proof_sha256": "7" * 64,
                },
                "route_map_sha256": "8" * 64,
                "prepared_endpoints_sha256": "9" * 64,
            },
        }
        if provider == "openbench":
            bind_fields["proxy"] = {
                "local_locator": "unix:///run/aiperf/evaluator-proxy.sock",
                "grant": {
                    "grant_id": "openbench-proof-grant",
                    "session_id": "openbench-proof",
                    "secret": "s" * 48,
                    "service_ids": ["candidate"],
                    "purposes": ["primary"],
                    "semantic_operation_ids": ["model.generate"],
                    "process_scope_sha256": "a" * 64,
                    "max_operations": 1,
                    "max_concurrent_operations": 1,
                    "max_request_bytes": 8 * 1024 * 1024,
                    "max_response_bytes": 8 * 1024 * 1024,
                    "max_stream_events": 1,
                    "expires_after_ms": 60_000,
                },
            }
        response_content = (
            "Reasoning complete.\nThe answer is 18"
            if provider == "nemo_evaluator"
            else "Reasoning complete. Answer: 18"
        )
        call(
            "bind_assets",
            **bind_fields,
        )
        page = call("next_units", offset=0, limit=4)
        unit_ids = [item["unit_id"] for item in page["items"]]
        call("start_units", unit_ids=unit_ids)
        drained = False
        for _ in range(60):
            polled = call("poll_events", limit=64, wait_ms=500)
            for envelope in polled["events"]:
                event = envelope["event"]
                if event["kind"] != "host_operation_requested":
                    continue
                request = event["request"]
                call(
                    "submit_host_events",
                    events=[
                        _terminal_event(
                            request["operation_id"],
                            request["context"]["semantic_attempt_id"],
                            response_content,
                        )
                    ],
                )
            if polled["drained"]:
                drained = True
                break
        assert drained
        candidate = call("finalize_session")
        assert candidate["outcomes"][0]["outcome"]["kind"] == "completed"
        score_name = (
            "reward" if provider == "nemo_evaluator" else "grade_school_math_scorer"
        )
        assert candidate["outcomes"][0]["outcome"]["completed"]["scores"][score_name][
            "public_projection"
        ] == {"value": 1.0}
        assert candidate["aggregates"]
        aggregates = {item["metric"]: item for item in candidate["aggregates"]}
        if provider == "openbench":
            assert set(aggregates) == {"accuracy", "stderr"}
            assert aggregates["accuracy"]["definition"] == {
                "metric_params": {},
                "params": {},
                "score_name": "grade_school_math_scorer",
            }
        else:
            assert set(aggregates) == {"reward"}
            assert aggregates["reward"]["definition"] == {
                "exclude_cancelled": True,
                "exclude_infrastructure": True,
            }
        assert call("shutdown") == {"shutdown": True}
    finally:
        request_writer.close()
        response_reader.close()
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr.decode("utf-8", errors="replace")
    assert stdout == b""
    if proxy_fixture is not None:
        proxy_fixture.join()
        assert proxy_fixture.request is not None
        assert proxy_fixture.request["model"] == "candidate"
        assert proxy_fixture.headers["x-aiperf-proxy-grant"] == "openbench-proof-grant"
        assert proxy_fixture.headers["x-aiperf-case-id"] == "openbench-gsm8k-case-0"
        assert proxy_fixture.headers["x-aiperf-semantic-attempt-id"] == (
            "attempt-openbench-gsm8k-case-0-0"
        )


def _terminal_event(
    operation_id: str, semantic_attempt_id: str, content: str
) -> dict[str, Any]:
    return {
        "kind": "terminal",
        "terminal": {
            "operation_id": operation_id,
            "semantic_attempt_id": semantic_attempt_id,
            "disposition": "completed",
            "result": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                        "stop_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 7},
            },
            "usage": {"prompt_tokens": 10, "completion_tokens": 7},
            "observed_output": True,
        },
    }


def _requirements_present(requirements: dict[str, str], root: Path) -> bool:
    site_packages = root / "lib/python3.12/site-packages"
    if not site_packages.is_dir():
        return False
    installed = {
        item.metadata["Name"].lower().replace("_", "-"): item.version
        for item in importlib.metadata.distributions(path=[str(site_packages)])
    }
    return all(
        installed.get(distribution.lower().replace("_", "-")) == version
        for distribution, version in requirements.items()
    )


def _skip_optional_stock_proof(reason: str) -> None:
    if os.environ.get("AIPERF_REQUIRE_STOCK_PROVIDER_PROOF") == "1":
        pytest.fail(reason)
    pytest.skip(reason)
