# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from aiperf.accuracy import worker as worker_module
from aiperf.accuracy.worker import AccuracyWorker, _Problem, _Registration


def test_stdio_reserves_stdout_for_correlated_protocol_messages() -> None:
    requests = "\n".join(
        [
            json.dumps({"id": 7, "op": "hello", "protocol": 1}),
            json.dumps({"id": 8, "op": "shutdown"}),
            "",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-u", "-m", "aiperf.accuracy.worker"],
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


def test_handshake_reports_source_lock_packages_and_runtime() -> None:
    identity = AccuracyWorker().hello(1)
    lock = Path(worker_module.__file__).resolve().parents[3] / (
        "requirements/accuracy-worker.txt"
    )
    assert (
        identity["worker_source_sha256"]
        == hashlib.sha256(Path(worker_module.__file__).read_bytes()).hexdigest()
    )
    assert (
        identity["dependency_lock_sha256"]
        == hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    assert identity["python_executable"] == sys.executable
    assert set(worker_module._LOCKED_PACKAGE_VERSIONS) <= set(identity["packages"])
    lock_text = lock.read_text()
    for package, version in worker_module._LOCKED_PACKAGE_VERSIONS.items():
        assert f"{package}=={version} " in lock_text


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
async def test_load_rejects_grader_override_before_benchmark_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_verify_locked_environment", lambda: None)
    worker = AccuracyWorker()
    with pytest.raises(ValueError, match="grader overrides are disabled"):
        await worker.load(
            {
                "benchmark": "mmlu-pro",
                "config": {"grader": "bespoke-rust-equivalent"},
            }
        )


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
async def test_mmlu_pro_delegates_prompt_dataset_and_metric_to_lighteval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = SimpleNamespace(
        id="0",
        generation_size=99,
        stop_sequences=["Question:"],
    )

    class Metric:
        def compute_sample(self, *, doc: Any, model_response: Any) -> dict[str, float]:
            assert doc is not None
            return {
                "extractive_match": 1.0
                if model_response.text == ["The answer is (B)"]
                else 0.0
            }

    task = SimpleNamespace(
        evaluation_split=("test",),
        dataset={"test": [{"category": "math"}]},
        metrics=[Metric()],
        dataset_path="TIGER-Lab/MMLU-Pro",
        dataset_config_name="default",
        dataset_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        version=1,
        get_docs=lambda max_samples=None: [doc][:max_samples],
    )

    class Registry:
        def __init__(self, *, tasks: str) -> None:
            assert tasks == "mmlu_pro|5"

        def load_tasks(self) -> dict[str, Any]:
            return {"mmlu_pro|5": task}

    class LightevalTask:
        @staticmethod
        def load_datasets(
            tasks: dict[str, Any], dataset_loading_processes: int
        ) -> None:
            assert tasks == {"mmlu_pro|5": task}
            assert dataset_loading_processes == 1

    class PromptManager:
        def __init__(
            self, *, use_chat_template: bool, system_prompt: str | None
        ) -> None:
            assert use_chat_template is False
            assert system_prompt is None

        def prepare_prompt_api(self, _doc: Any) -> list[dict[str, str]]:
            return [{"role": "user", "content": "canonical prompt"}]

        def prepare_prompt(self, _doc: Any) -> str:
            return "canonical prompt"

    class ModelResponse:
        def __init__(self, *, text: list[str]) -> None:
            self.text = text

    modules = {
        "lighteval.tasks.lighteval_task": {"LightevalTask": LightevalTask},
        "lighteval.tasks.prompt_manager": {"PromptManager": PromptManager},
        "lighteval.tasks.registry": {"Registry": Registry},
        "lighteval.models.model_output": {"ModelResponse": ModelResponse},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    worker = AccuracyWorker()
    await worker._load_mmlu_pro({"max_problems": 1})
    assert worker._dataset_identity["revision"] == task.dataset_revision
    problem = worker._problems[0]
    assert problem.problem_id.startswith("mmlu-pro:00000000:")
    assert problem.messages == [{"role": "user", "content": "canonical prompt"}]
    assert problem.generation["max_tokens"] == 99
    grade = await worker._grade_one(problem, "The answer is (B)")
    assert grade["correct"] is True
    assert grade["confidence"] == 1.0


@pytest.mark.asyncio
async def test_livecodebench_reuses_one_lighteval_pool_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.accuracy.graders import code_execution

    calls: list[tuple[list[Any], list[list[str]]]] = []

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

    def run_pool(
        samples: list[Any], generations: list[list[str]]
    ) -> tuple[dict[str, Any], Any]:
        calls.append((samples, generations))
        return {
            "pass@1": 0.5,
            "detail": {"pass@1": {0: 1.0, 1: 0.0}},
        }, None

    monkeypatch.setattr(code_execution, "_run_codegen_metrics", run_pool)
    problems = [
        _Problem(
            problem_id=f"opaque-{index}",
            task="lcb",
            prompt="prompt",
            messages=[{"role": "user", "content": "prompt"}],
            generation={"max_tokens": 1},
            ground_truth="{}",
        )
        for index in range(2)
    ]
    worker = AccuracyWorker()
    worker._benchmark = "lcb-codegeneration"
    worker._problems = problems
    worker._by_id = {problem.problem_id: problem for problem in problems}
    worker._grader = SimpleNamespace(extract_answer=lambda response: response)
    result = await worker.grade_batch(
        [
            {"problem_id": problem.problem_id, "response": f"code-{index}"}
            for index, problem in enumerate(problems)
        ]
    )
    assert len(calls) == 1
    assert calls[0][1] == [["code-0"], ["code-1"]]
    assert [item["correct"] for item in result["items"]] == [True, False]
