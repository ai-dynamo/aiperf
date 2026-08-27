# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from aiperf.accuracy import worker as worker_module
from aiperf.accuracy.graders._codegen_worker_client import CodegenWorkerError
from aiperf.rust_shims.accuracy_worker import AccuracyWorker, _Problem, _Registration


def test_stdio_reserves_stdout_for_correlated_protocol_messages() -> None:
    requests = "\n".join(
        [
            json.dumps({"id": 7, "op": "hello", "protocol": 1}),
            json.dumps({"id": 8, "op": "shutdown"}),
            "",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-u", "-m", "aiperf.rust_shims.accuracy_worker"],
        input=requests,
        text=True,
        capture_output=True,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [7, 8]
    assert all(response["ok"] for response in responses)
    assert responses[0]["result"]["protocol"] == 1
    assert responses[1]["result"] == {"shutdown": True}
    assert completed.stderr == ""


def test_stdio_redirects_python_and_file_descriptor_noise_to_stderr() -> None:
    script = """
import os
from aiperf.accuracy import worker

original_hello = worker.AccuracyWorker.hello

def noisy_hello(self, protocol):
    print("python-noise", flush=True)
    os.write(1, b"descriptor-noise\\n")
    return original_hello(self, protocol)

worker.AccuracyWorker.hello = noisy_hello
raise SystemExit(worker.serve())
"""
    requests = "\n".join(
        [
            json.dumps({"id": 1, "op": "hello", "protocol": 1}),
            json.dumps({"id": 2, "op": "shutdown"}),
            "",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-u", "-c", script],
        input=requests,
        text=True,
        capture_output=True,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert "python-noise" in completed.stderr
    assert "descriptor-noise" in completed.stderr


def test_handshake_reports_source_lock_packages_and_runtime() -> None:
    identity = AccuracyWorker().hello(1)
    source_root = Path(worker_module.__file__).resolve().parent
    source_digest = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root).as_posix().encode()
        payload = source.read_bytes()
        source_digest.update(len(relative).to_bytes(8, "big"))
        source_digest.update(relative)
        source_digest.update(len(payload).to_bytes(8, "big"))
        source_digest.update(payload)
    assert identity["worker_source_sha256"] == source_digest.hexdigest()
    assert identity["dependency_lock_sha256"] is None
    assert identity["python_executable"] == sys.executable
    assert set(worker_module._LOCKED_PACKAGE_VERSIONS) <= set(identity["packages"])


def test_pinned_dataset_adapter_injects_and_enforces_revision() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def loader(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "dataset"

    pinned = worker_module._pinned_dataset_loader(loader, "a" * 40)
    assert pinned("repo", split="test") == "dataset"
    assert calls == [(("repo",), {"split": "test", "revision": "a" * 40})]
    with pytest.raises(ValueError, match="canonical worker requires"):
        pinned("repo", revision="b" * 40)


def test_lcb_release_file_mapping_matches_pinned_dataset_script() -> None:
    assert worker_module._lcb_release_files("v1") == ["test.jsonl"]
    assert worker_module._lcb_release_files("v4_v5") == [
        "test4.jsonl",
        "test5.jsonl",
    ]
    assert worker_module._lcb_release_files("release_v3") == [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
    ]
    assert worker_module._lcb_release_files("release_latest")[-1] == "test6.jsonl"
    with pytest.raises(ValueError, match="unsupported LiveCodeBench release"):
        worker_module._lcb_release_files("v6_v2")


def test_lcb_adapter_uses_pinned_raw_files_without_remote_script() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def loader(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "dataset"

    registration = worker_module._REGISTRATIONS["lcb-codegeneration"]
    adapted = worker_module._lcb_script_free_dataset_loader(loader, registration)
    assert (
        adapted(
            registration.dataset_repository,
            "v4_v5",
            split="test",
            trust_remote_code=True,
        )
        == "dataset"
    )
    base = (
        "https://huggingface.co/datasets/"
        f"{registration.dataset_repository}/resolve/{registration.dataset_revision}"
    )
    assert calls == [
        (
            ("json",),
            {
                "data_files": {"test": [f"{base}/test4.jsonl", f"{base}/test5.jsonl"]},
                "split": "test",
            },
        )
    ]


def test_every_inherited_dataset_has_an_immutable_revision() -> None:
    for registration in worker_module._REGISTRATIONS.values():
        assert len(registration.dataset_revision) == 40
        int(registration.dataset_revision, 16)
        assert registration.dataset_repository
        assert registration.evaluation_splits


def test_problem_pages_never_expose_worker_ground_truth() -> None:
    worker = AccuracyWorker()
    worker._benchmark = "fixture"
    worker._problems = [
        _Problem(
            problem_id="opaque",
            task="task",
            prompt="prompt",
            messages=[{"role": "user", "content": "prompt"}],
            generation={
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop": [],
            },
            ground_truth="private",
        )
    ]
    page = worker.next_problems(0, 10)
    assert page["done"] is True
    assert "ground_truth" not in page["items"][0]


def test_locked_environment_mismatch_is_an_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_package_version", lambda _name: None)
    with pytest.raises(RuntimeError, match="does not match"):
        worker_module._verify_locked_environment()


@pytest.mark.asyncio
async def test_mmlu_pro_routes_through_inherited_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MMLU-Pro is now the custom TIGER-Lab benchmark (MMLUProBenchmark +
    # MMLUProGrader) loaded via the standard registration path, not the old
    # lighteval special case. It routes through _load_inherited like any
    # other benchmark and accepts a grader override.
    monkeypatch.setattr(worker_module, "_verify_locked_environment", lambda: None)
    captured: list[tuple[str, str | None]] = []
    worker = AccuracyWorker()

    async def load_inherited(
        benchmark: str, _config: dict[str, Any], grader: str | None
    ) -> None:
        captured.append((benchmark, grader))
        worker._problems = [
            _Problem(
                problem_id="opaque",
                task="math",
                prompt="prompt",
                messages=[{"role": "user", "content": "prompt"}],
                generation={
                    "max_tokens": 4000,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                ground_truth="B",
            )
        ]
        worker._dataset_identity = {
            "provider": "fixture",
            "revision": "fixture-revision",
            "evaluation_splits": ["validation", "test"],
        }

    monkeypatch.setattr(worker, "_load_inherited", load_inherited)
    result = await worker.load({"benchmark": "mmlu-pro", "config": {}})
    assert captured == [("mmlu-pro", None)]
    assert result["problem_count"] == 1


@pytest.mark.asyncio
async def test_inherited_benchmark_routes_grader_override_to_python_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_verify_locked_environment", lambda: None)
    captured: list[tuple[str, str | None]] = []
    worker = AccuracyWorker()

    async def load_inherited(
        benchmark: str, _config: dict[str, Any], grader: str | None
    ) -> None:
        captured.append((benchmark, grader))
        worker._problems = [
            _Problem(
                problem_id="opaque",
                task="task",
                prompt="prompt",
                messages=[{"role": "user", "content": "prompt"}],
                generation={
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stop": [],
                },
                ground_truth="answer",
            )
        ]
        worker._dataset_identity = {
            "provider": "fixture",
            "revision": "fixture-revision",
            "evaluation_splits": ["test"],
        }

    monkeypatch.setattr(worker, "_load_inherited", load_inherited)
    result = await worker.load(
        {"benchmark": "mmlu", "config": {}, "grader": "exact_match"}
    )

    assert captured == [("mmlu", "exact_match")]
    assert result["problem_count"] == 1


def test_dynamic_subset_identity_is_explicit() -> None:
    registration = _Registration(
        benchmark_class="module:Benchmark",
        grader_class="module:Grader",
        dataset_repository="repo",
        dataset_revision="a" * 40,
        evaluation_splits=("test",),
    )
    assert worker_module._resolved_subset("mmlu", registration, ["math"]) == "math"
    assert (
        worker_module._resolved_subset("bigbench", registration, None)
        == "all canonical task subsets"
    )


@pytest.mark.asyncio
async def test_livecodebench_delegates_one_request_per_batch_and_reuses_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.accuracy.graders import code_execution

    def payload_to_cases(payload: dict[str, Any]) -> tuple[list[str], list[str], None]:
        count = int(payload["case_count"])
        return (
            [f"input-{index}" for index in range(count)],
            [f"output-{index}" for index in range(count)],
            None,
        )

    monkeypatch.setattr(code_execution, "_payload_to_test_cases", payload_to_cases)
    monkeypatch.setattr(
        code_execution,
        "_build_evaluation_sample",
        lambda inputs, _outputs, _fn_name: [{"input_output": f"{len(inputs)} cases"}],
    )

    def direct_pool_must_not_run(*_args: Any) -> tuple[dict[str, Any], Any]:
        raise AssertionError("native LCB called _run_codegen_metrics directly")

    monkeypatch.setattr(
        code_execution,
        "_run_codegen_metrics",
        direct_pool_must_not_run,
        raising=False,
    )
    child = SimpleNamespace(
        grade_codegen=AsyncMock(
            side_effect=[
                {"pass@1": 0.5, "detail": {"pass@1": {"0": 0.0, "1": 1.0}}},
                {"pass@1": 1.0, "detail": {"pass@1": {"0": 1.0}}},
            ]
        ),
        aclose=AsyncMock(),
    )
    child_factory = MagicMock(return_value=child)
    monkeypatch.setattr(
        worker_module, "CodegenGradingWorker", child_factory, raising=False
    )
    problems = [
        _Problem(
            problem_id=f"opaque-{index}",
            task="lcb",
            prompt="prompt",
            messages=[{"role": "user", "content": "prompt"}],
            generation={"max_tokens": 1},
            ground_truth=json.dumps({"case_count": case_count}),
        )
        for index, case_count in enumerate((2, 3, 4))
    ]
    accuracy_worker = AccuracyWorker()
    accuracy_worker._benchmark = "lcb-codegeneration"
    accuracy_worker._problems = problems
    accuracy_worker._by_id = {problem.problem_id: problem for problem in problems}
    accuracy_worker._grader = SimpleNamespace(extract_answer=lambda response: response)
    accuracy_worker._uses_lcb_batch_grader = True

    result = await accuracy_worker.grade_batch(
        [
            {"problem_id": "opaque-2", "response": "code-2"},
            {"problem_id": "opaque-1", "response": ""},
            {"problem_id": "opaque-0", "response": "code-0"},
        ]
    )
    second_result = await accuracy_worker.grade_batch(
        [{"problem_id": "opaque-0", "response": "code-0-again"}]
    )

    child_factory.assert_called_once_with()
    assert child.grade_codegen.await_args_list == [
        call(
            [{"input_output": "4 cases"}, {"input_output": "2 cases"}],
            [["code-2"], ["code-0"]],
            timeout=code_execution._derive_grade_timeout(9),
        ),
        call(
            [{"input_output": "2 cases"}],
            [["code-0-again"]],
            timeout=code_execution._derive_grade_timeout(2),
        ),
    ]
    assert [item["problem_id"] for item in result["items"]] == [
        "opaque-2",
        "opaque-1",
        "opaque-0",
    ]
    assert [item["correct"] for item in result["items"]] == [False, False, True]
    assert [item["unparsed"] for item in result["items"]] == [False, True, False]
    assert second_result["items"][0]["correct"] is True
    assert all("ground_truth" not in item for item in result["items"])


@pytest.mark.asyncio
async def test_livecodebench_codegen_worker_error_is_evaluator_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.accuracy.graders import code_execution

    monkeypatch.setattr(
        code_execution,
        "_payload_to_test_cases",
        lambda _payload: (["input"], ["output"], None),
    )
    monkeypatch.setattr(
        code_execution,
        "_build_evaluation_sample",
        lambda _inputs, _outputs, _fn_name: [{"input_output": "fixture"}],
    )
    child = SimpleNamespace(
        grade_codegen=AsyncMock(side_effect=CodegenWorkerError("sandbox died")),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        worker_module, "CodegenGradingWorker", MagicMock(return_value=child)
    )
    problem = _Problem(
        problem_id="opaque",
        task="lcb",
        prompt="prompt",
        messages=[{"role": "user", "content": "prompt"}],
        generation={"max_tokens": 1},
        ground_truth="{}",
    )
    accuracy_worker = AccuracyWorker()
    accuracy_worker._benchmark = "lcb-codegeneration"
    accuracy_worker._problems = [problem]
    accuracy_worker._by_id = {problem.problem_id: problem}
    accuracy_worker._grader = SimpleNamespace(extract_answer=lambda response: response)
    accuracy_worker._uses_lcb_batch_grader = True

    with pytest.raises(
        RuntimeError,
        match="canonical LiveCodeBench grading worker failed: sandbox died",
    ) as caught:
        await accuracy_worker.grade_batch(
            [{"problem_id": "opaque", "response": "code"}]
        )

    assert isinstance(caught.value.__cause__, CodegenWorkerError)


@pytest.mark.asyncio
async def test_codegen_worker_close_is_idempotent() -> None:
    accuracy_worker = AccuracyWorker()
    close_child = SimpleNamespace(aclose=AsyncMock())
    accuracy_worker._codegen_worker = close_child

    await accuracy_worker.close()
    await accuracy_worker.close()

    close_child.aclose.assert_awaited_once_with()
    assert accuracy_worker._codegen_worker is None


@pytest.mark.asyncio
async def test_codegen_worker_is_closed_before_replacement_non_lcb_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accuracy_worker = AccuracyWorker()
    accuracy_worker._benchmark = "lcb-codegeneration"
    reload_child = SimpleNamespace(aclose=AsyncMock())
    accuracy_worker._codegen_worker = reload_child
    child_factory = MagicMock()
    monkeypatch.setattr(worker_module, "CodegenGradingWorker", child_factory)
    monkeypatch.setattr(worker_module, "_verify_locked_environment", lambda: None)

    async def load_inherited(
        _benchmark: str, _config: dict[str, Any], _grader: str | None
    ) -> None:
        accuracy_worker._grader = SimpleNamespace()
        accuracy_worker._problems = [
            _Problem(
                problem_id="opaque",
                task="fixture",
                prompt="prompt",
                messages=[{"role": "user", "content": "prompt"}],
                generation={"max_tokens": 1},
                ground_truth="answer",
            )
        ]
        accuracy_worker._dataset_identity = {
            "provider": "fixture",
            "revision": "fixture-revision",
            "evaluation_splits": ["test"],
        }

    monkeypatch.setattr(accuracy_worker, "_load_inherited", load_inherited)

    result = await accuracy_worker.load({"benchmark": "mmlu", "config": {}})

    assert result["benchmark"] == "mmlu"
    reload_child.aclose.assert_awaited_once_with()
    assert accuracy_worker._codegen_worker is None
    child_factory.assert_not_called()
