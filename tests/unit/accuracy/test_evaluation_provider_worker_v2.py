# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator-provider v2 worker, host ledger, and stock-pair proofs."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import select
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aiperf.accuracy.evaluation.canonical import (
    CanonicalJsonError,
    canonical_dumps,
    canonical_loads,
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
    NEMO_EVALUATOR_DISTRIBUTION,
    OPENBENCH_DISTRIBUTION,
    DistributionEvidence,
    executable_tasks,
)
from aiperf.accuracy.evaluation.host import PipeEvaluationHost
from aiperf.accuracy.evaluation.operation_schemas import (
    OPERATION_DIRECTION_SCHEMA_SHA256,
    OPERATION_SCHEMA_SHA256,
)
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
        "d468bbc4f1fdbbc54360cede8194732b2ebaabbdfb55490bc572c4bb44f89cdf"
    )
    assert dict(OPERATION_DIRECTION_SCHEMA_SHA256["model.generate"]) == {
        "request": "c2f30f5396f4af6e44025d80294b2685916492c23dd730cd1e2a6ebdb6ae5d21",
        "response": "6c8d726e5a0c05a22de946ce2495d6a4bcf3b3b7bb7a48e5c39bad07ff954ca0",
        "stream": "84a861ea0a983368cd48e6db2fa4ac71b8219d7685065718859f0bfc4ea49206",
    }
    for descriptor in (
        NEMO_EVALUATOR_DISTRIBUTION,
        OPENBENCH_DISTRIBUTION,
    ):
        argv = descriptor.fixed_argv
        assert argv[:3] == ("-I", "-m", "aiperf.accuracy.evaluation.worker")
        assert "--stdio" not in argv
        assert argv[argv.index("--read-fd") + 1] == "3"
        assert argv[argv.index("--write-fd") + 1] == "4"
        environment = descriptor.clean_environment
        assert set(environment).isdisjoint(
            {"HOME", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "OPENAI_API_KEY"}
        )
        assert environment["XDG_DATA_HOME"].startswith("/staging/")


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
    if not _requirements_present(requirements):
        pytest.skip("exact external evaluator distribution is not installed")
    descriptor = (
        NEMO_EVALUATOR_DISTRIBUTION
        if provider == "nemo_evaluator"
        else OPENBENCH_DISTRIBUTION
    )
    worker_root = tmp_path / "worker-root"
    materialize(distribution, worker_root)
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
    environment["XDG_DATA_HOME"] = str(staging_root / ".xdg-data")
    environment["XDG_CACHE_HOME"] = str(staging_root / ".xdg-cache")
    process = subprocess.Popen(
        [
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
        ],
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
        assert raw, f"worker pipe closed during {operation}"
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
        call(
            "bind_assets",
            assets=[
                {
                    "asset_id": "openai_gsm8k_main_test_canary",
                    "contained_path": str(asset),
                    "content_sha256": _ASSET_SHA256,
                    "immutable_revision": _ASSET_REVISION,
                    "media_type": "application/x-ndjson",
                }
            ],
            host_binding={
                "host": {
                    "runner_sha256": "4" * 64,
                    "capability_inventory_sha256": "5" * 64,
                    "schema_inventory_sha256": "6" * 64,
                    "isolation_proof_sha256": "7" * 64,
                },
                "route_map_sha256": "8" * 64,
                "prepared_endpoints_sha256": "9" * 64,
            },
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
                            "Reasoning complete. Answer: 18. The answer is 18",
                        )
                    ],
                )
            if polled["drained"]:
                drained = True
                break
        assert drained
        candidate = call("finalize_session")
        assert candidate["outcomes"][0]["outcome"]["kind"] == "completed"
        assert candidate["aggregates"]
        assert call("shutdown") == {"shutdown": True}
    finally:
        request_writer.close()
        response_reader.close()
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr.decode("utf-8", errors="replace")
    assert stdout == b""


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


def _requirements_present(requirements: dict[str, str]) -> bool:
    try:
        return all(
            importlib.metadata.version(distribution) == version
            for distribution, version in requirements.items()
        )
    except importlib.metadata.PackageNotFoundError:
        return False
