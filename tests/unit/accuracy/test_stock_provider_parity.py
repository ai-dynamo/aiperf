# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone/hosted parity proofs for the two stock GSM8K providers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

from aiperf.accuracy.evaluation.distributions import (
    NEMO_EVALUATOR_DISTRIBUTION,
    OPENBENCH_DISTRIBUTION,
    source_tree_sha256,
)
from tools.generate_stock_evaluator_manifest import materialize

_ROOT = Path(__file__).resolve().parents[3]
_NEMO_ROOT = Path(
    os.environ.get(
        "AIPERF_TEST_NEMO_PROVIDER_ROOT",
        _ROOT / "tools/stock_evaluators/nemo/.venv",
    )
)
_OPENBENCH_ROOT = Path(
    os.environ.get(
        "AIPERF_TEST_OPENBENCH_PROVIDER_ROOT",
        _ROOT / "tools/stock_evaluators/openbench/.venv",
    )
)
_NEMO_LOCK = _ROOT / "tools/stock_evaluators/nemo/uv.lock"
_OPENBENCH_LOCK = _ROOT / "tools/stock_evaluators/openbench/uv.lock"
_MANIFESTS = _ROOT / "src/aiperf/accuracy/evaluation/manifests"


@pytest.fixture(scope="module")
def stock_rootfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Materialize both exact provider closures once for subprocess proofs."""
    for root in (_NEMO_ROOT, _OPENBENCH_ROOT):
        if not (root / "bin/python").is_file():
            if os.environ.get("AIPERF_REQUIRE_STOCK_PROVIDER_PROOF") == "1":
                pytest.fail(f"stock provider environment is unavailable: {root}")
            pytest.skip(f"stock provider environment is unavailable: {root}")
    parent = tmp_path_factory.mktemp("stock-provider-parity")
    result = {"nemo_evaluator": parent / "nemo", "openbench": parent / "openbench"}
    materialize(
        NEMO_EVALUATOR_DISTRIBUTION.distribution_id,
        result["nemo_evaluator"],
        nemo_root=_NEMO_ROOT,
        openbench_root=_OPENBENCH_ROOT,
    )
    materialize(
        OPENBENCH_DISTRIBUTION.distribution_id,
        result["openbench"],
        nemo_root=_NEMO_ROOT,
        openbench_root=_OPENBENCH_ROOT,
    )
    return result


def test_provider_uv_locks_are_tracked_and_bound_into_generated_locks() -> None:
    """The two dependency universes are committed inputs, never ambient state."""
    relative = (
        "tools/stock_evaluators/nemo/uv.lock",
        "tools/stock_evaluators/openbench/uv.lock",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *relative],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, tracked.stderr
    for path, lock_name in (
        (_NEMO_LOCK, "nemo_evaluator.lock.json"),
        (_OPENBENCH_LOCK, "openbench.lock.json"),
    ):
        generated = json.loads((_MANIFESTS / lock_name).read_text())
        assert generated["resolution"] == {
            "artifact_content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "environment_record_set_sha256": generated["resolution"][
                "environment_record_set_sha256"
            ],
            "kind": "uv-lock-v1",
        }


def test_semantic_source_tree_v1_does_not_absorb_overlay_patch_assets(
    tmp_path: Path,
) -> None:
    """Patch bytes have their own raw-digest domain and do not redefine v1."""
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    patch = tmp_path / "overlay.patch"
    patch.write_text("first\n")
    initial = source_tree_sha256(tmp_path)
    patch.write_text("second\n")
    assert source_tree_sha256(tmp_path) == initial
    (tmp_path / "module.py").write_text("VALUE = 2\n")
    assert source_tree_sha256(tmp_path) != initial


@pytest.mark.parametrize(
    ("provider", "program", "needle"),
    (
        (
            "nemo_evaluator",
            "import sys; sys.argv=['nel','--help']; "
            "from nemo_evaluator.cli.main import cli; cli()",
            "NeMo Evaluator CLI",
        ),
        (
            "openbench",
            "import sys; sys.argv=['bench','--help']; "
            "from openbench._cli import main; main()",
            "Usage: bench",
        ),
    ),
)
def test_overlaid_provider_clis_retain_standalone_help(
    stock_rootfs: dict[str, Path], provider: str, program: str, needle: str
) -> None:
    completed = _contained(stock_rootfs[provider], program)
    assert completed.returncode == 0, completed.stderr
    assert needle in completed.stdout


def test_nemo_actual_cli_run_spec_path_preserves_quick_mode(
    stock_rootfs: dict[str, Path],
) -> None:
    program = r"""
import json

from click.testing import CliRunner
import nemo_evaluator.executors as executors
from nemo_evaluator.cli.main import cli

observed = {}


class RecordingExecutor:
    def run(self, config, **kwargs):
        observed["config"] = config.model_dump(mode="json", exclude_none=True)
        observed["kwargs"] = kwargs


executors.get_executor = lambda _kind: RecordingExecutor()
result = CliRunner().invoke(
    cli,
    [
        "eval",
        "run",
        "--bench",
        "gsm8k",
        "--model-url",
        "http://frozen.invalid/v1",
        "--model-id",
        "candidate",
        "--repeats",
        "3",
        "--max-problems",
        "4",
        "--temperature",
        "0",
        "--max-tokens",
        "64",
        "--output-dir",
        "./frozen-results",
        "--dry-run",
    ],
)
assert result.exit_code == 0, result.output
config = observed["config"]
service = config["services"]["model"]
benchmark = config["benchmarks"][0]
assert service["url"] == "http://frozen.invalid/v1/chat/completions"
assert service["model"] == "candidate"
assert service["generation"] == {"temperature": 0.0, "max_tokens": 64}
assert benchmark["name"] == "gsm8k"
assert benchmark["repeats"] == 3
assert benchmark["max_problems"] == 4
assert observed["kwargs"] == {
    "dry_run": True,
    "resume": False,
    "background": False,
    "submit": False,
}
print("RESULT:" + json.dumps({"service": service, "benchmark": benchmark}, sort_keys=True))
"""
    completed = _contained(stock_rootfs["nemo_evaluator"], program)
    assert completed.returncode == 0, completed.stderr
    run_spec = _result(completed)
    assert run_spec["benchmark"]["name"] == "gsm8k"


def test_openbench_actual_cli_run_spec_path_preserves_standalone_options(
    stock_rootfs: dict[str, Path],
) -> None:
    program = r"""
import json

from openbench._cli import eval_command as command

task = object()
captured = {}
display_patch_calls = []
command.load_task = lambda name, allow_alpha=False: (
    task if (name, allow_alpha) == ("gsm8k", False) else None
)
command.patch_display_results = lambda: display_patch_calls.append(True)
command.eval = lambda **kwargs: captured.update(kwargs) or []

logs = command.run_eval(
    ["gsm8k"],
    model=["aiperf/candidate"],
    epochs=2,
    limit="2",
    retry_on_error=0,
    max_retries=0,
    max_tokens=64,
    temperature=0.0,
    display=command.DisplayType.NONE,
    log_dir="./frozen-logs",
)
assert logs == []
assert captured["tasks"] == [task]
assert captured["model"] == ["aiperf/candidate"]
assert captured["epochs"] == 2
assert captured["limit"] == 2
assert captured["max_retries"] == 0
assert captured["retry_on_error"] is None
assert captured["max_tokens"] == 64
assert captured["temperature"] == 0.0
assert captured["display"] == "none"
assert display_patch_calls == [True]
print("RESULT:" + json.dumps({
    key: captured[key]
    for key in (
        "model",
        "epochs",
        "limit",
        "max_retries",
        "retry_on_error",
        "max_tokens",
        "temperature",
        "display",
    )
}, sort_keys=True))
"""
    completed = _contained(stock_rootfs["openbench"], program)
    assert completed.returncode == 0, completed.stderr
    run_spec = _result(completed)
    assert run_spec["limit"] == 2
    assert run_spec["retry_on_error"] is None


def test_nemo_run_evaluation_recording_host_preserves_normalized_semantics(
    stock_rootfs: dict[str, Path],
) -> None:
    program = r"""
import asyncio
import json

from nemo_evaluator.benchmarks.gsm8k import gsm8k_scorer
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.host import RecordingEvaluationHost
from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.environments.custom import BenchmarkDefinition, ByobEnvironment
from nemo_evaluator.solvers.chat import ChatSolver


class ScriptedHost:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute_model(self, operation):
        return {
            "id": "chatcmpl-parity",
            "model": "candidate",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Reasoning complete. The answer is 18",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7},
        }


def rows():
    return [{"question": "What is 12 + 6?", "answer": "18"}]


def normalize(bundle):
    results = []
    for raw in bundle["_results"]:
        item = {key: value for key, value in raw.items() if key != "latency_ms"}
        trajectories = []
        for trajectory in item.get("trajectory", []):
            trajectories.append({
                key: value for key, value in trajectory.items() if key != "session_id"
            })
        item["trajectory"] = trajectories
        results.append(item)
    benchmark = dict(bundle["benchmark"])
    scores = dict(benchmark["scores"])
    scores.pop("runtime", None)
    benchmark["scores"] = scores
    return {
        "normalization": "nemo-evaluator-gsm8k-semantic-v1",
        "sdk_version": bundle["sdk_version"],
        "config_hash": bundle["config_hash"],
        "config": bundle["config"],
        "benchmark": benchmark,
        "n_results": bundle["n_results"],
        "results": results,
    }


async def one(host):
    environment = ByobEnvironment(BenchmarkDefinition(
        name="gsm8k",
        dataset=rows,
        prompt="Solve: {question}",
        target_field="answer",
        scorer_fn=gsm8k_scorer,
    ))
    solver = ChatSolver(ModelClient(
        base_url="https://standalone.invalid/v1",
        model="candidate",
        api_key="standalone-only-sentinel",
        max_tokens=64,
    ))
    return await run_evaluation(
        environment,
        solver,
        max_problems=1,
        evaluation_host=host,
    )


async def main():
    direct = await one(ScriptedHost())
    recording = RecordingEvaluationHost(ScriptedHost())
    recorded = await one(recording)
    operations = recording.operations
    assert normalize(direct) == normalize(recorded)
    assert len(operations) == 1
    operation = operations[0]
    assert (
        operation.semantic_operation_id,
        operation.service_id,
        operation.purpose,
        operation.path,
    ) == ("model.generate", "candidate", "primary", "/chat/completions")
    normalized = normalize(recorded)
    assert normalized["results"][0]["reward"] == 1.0
    print("RESULT:" + json.dumps(normalized, sort_keys=True, allow_nan=False))


asyncio.run(main())
"""
    completed = _contained(stock_rootfs["nemo_evaluator"], program)
    assert completed.returncode == 0, completed.stderr
    normalized = _result(completed)
    assert normalized["normalization"] == "nemo-evaluator-gsm8k-semantic-v1"
    assert normalized["benchmark"]["scores"]["summary"]["mean"] == 1.0


def test_nemo_selection_shuffle_shard_repeat_and_seed_once_are_deterministic(
    stock_rootfs: dict[str, Path],
) -> None:
    program = r"""
import asyncio
import json
import re

from nemo_evaluator.benchmarks.gsm8k import gsm8k_scorer
from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.host import RecordingEvaluationHost
from nemo_evaluator.engine.model_client import ModelClient
from nemo_evaluator.environments.custom import BenchmarkDefinition, ByobEnvironment
from nemo_evaluator.solvers.chat import ChatSolver


class CountingEnvironment(ByobEnvironment):
    def __init__(self, definition):
        super().__init__(definition)
        self.seed_calls = []

    async def seed(self, index):
        self.seed_calls.append(index)
        return await super().seed(index)


class ScriptedHost:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute_model(self, operation):
        content = operation.payload["messages"][-1]["content"]
        answer = re.search(r"Problem (\d+)", content).group(1)
        return {
            "id": "chatcmpl-multi",
            "model": "candidate",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"The answer is {answer}",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4},
        }


def rows():
    return [
        {"question": f"Problem {index}", "answer": str(index)}
        for index in range(6)
    ]


async def run_once():
    environment = CountingEnvironment(BenchmarkDefinition(
        name="gsm8k",
        dataset=rows,
        prompt="Solve {question}",
        target_field="answer",
        scorer_fn=gsm8k_scorer,
    ))
    all_results = []
    configs = []
    operation_count = 0
    for shard_index in range(2):
        host = RecordingEvaluationHost(ScriptedHost())
        solver = ChatSolver(ModelClient(
            base_url="https://standalone.invalid/v1",
            model="candidate",
            api_key="standalone-only-sentinel",
            max_tokens=32,
        ))
        config = {"model": "candidate", "repeats": 3}
        bundle = await run_evaluation(
            environment,
            solver,
            n_repeats=3,
            max_problems=4,
            config=config,
            max_concurrent=1,
            shard_info=(shard_index, 2),
            shuffle_seed=17,
            evaluation_host=host,
        )
        all_results.extend(bundle["_results"])
        configs.append(config)
        operation_count += len(host.operations)
    return environment.seed_calls, all_results, configs, operation_count


async def main():
    first = await run_once()
    second = await run_once()
    for seed_calls, results, configs, operation_count in (first, second):
        assert seed_calls == [0, 5, 1, 2]
        assert operation_count == 12
        assert [(item["problem_idx"], item["repeat"]) for item in results] == [
            (problem, repeat)
            for problem in (0, 5, 1, 2)
            for repeat in range(3)
        ]
        assert all(item["reward"] == 1.0 for item in results)
        assert [config["shard"]["range"] for config in configs] == [[0, 2], [2, 4]]
        assert all(config["shuffle"] == {
            "seed": 17,
            "applied": True,
            "ds_full_size": 6,
            "ds_effective_size": 4,
        } for config in configs)
    assert first[0] == second[0]
    assert [
        (item["problem_idx"], item["repeat"], item["reward"])
        for item in first[1]
    ] == [
        (item["problem_idx"], item["repeat"], item["reward"])
        for item in second[1]
    ]
    print("RESULT:" + json.dumps({
        "seed_calls": first[0],
        "occurrences": [
            [item["problem_idx"], item["repeat"]] for item in first[1]
        ],
        "operation_count": first[3],
    }, sort_keys=True))


asyncio.run(main())
"""
    completed = _contained(stock_rootfs["nemo_evaluator"], program, timeout=180)
    assert completed.returncode == 0, completed.stderr
    evidence = _result(completed)
    assert evidence["seed_calls"] == [0, 5, 1, 2]
    assert evidence["operation_count"] == 12


def test_openbench_public_inspect_host_parity_and_fail_closed_policies(
    stock_rootfs: dict[str, Path],
) -> None:
    program = r"""
import asyncio
import json
import tempfile

from inspect_ai import Task
from inspect_ai.dataset import json_dataset
from inspect_ai.extensions import entry_point_loading
from inspect_ai.model import GenerateConfig, Model, ModelCallContext
from inspect_ai.solver import generate
import inspect_ai._util.entrypoints as inspect_entrypoints
from openbench.evals.gsm8k import record_to_sample
from openbench.model._providers.aiperf_pipe import AiperfPipeModelAPI
from openbench.runtime import RecordingEvaluationHost, eval_async_hosted, normalize_eval_log
from openbench.scorers.grade_school_math import grade_school_math_scorer

ASSET = "/assets/gsm8k_canary.jsonl"
OUTPUT = {
    "model": "candidate",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Reasoning complete. Answer: 18. The answer is 18",
        },
        "stop_reason": "stop",
    }],
    "usage": {"input_tokens": 10, "output_tokens": 7, "total_tokens": 17},
}


def forbidden_entry_points(*_args, **_kwargs):
    raise AssertionError("installed Inspect entry points were enumerated")


inspect_entrypoints.entry_points = forbidden_entry_points


def task_and_config():
    config = GenerateConfig(
        temperature=0.0,
        max_tokens=2048,
        batch=False,
        attempt_timeout=None,
        max_retries=0,
    )
    task = Task(
        dataset=json_dataset(
            ASSET,
            sample_fields=record_to_sample,
            auto_id=True,
            limit=1,
            name="openbench_gsm8k_canary",
        ),
        solver=[generate()],
        scorer=grade_school_math_scorer(),
        config=config,
    )
    return task, config


async def success():
    task, config = task_and_config()
    host = RecordingEvaluationHost([OUTPUT])
    with entry_point_loading("deny"):
        model = Model(AiperfPipeModelAPI(host=host), config)
    with tempfile.TemporaryDirectory() as directory:
        logs = await eval_async_hosted(
            tasks=task,
            model=model,
            log_dir=directory,
            log_format="eval",
            limit=1,
            epochs=1,
            fail_on_error=False,
            continue_on_fail=True,
            retry_on_error=0,
            max_sandboxes=None,
            log_samples=True,
            log_realtime=False,
            log_images=False,
            score_display=False,
        )
    return normalize_eval_log(logs[0]), host.effects


async def failure():
    class FailingHost:
        def __init__(self):
            self.effects = []

        async def model_generate(self, effect):
            self.effects.append(effect)
            raise RuntimeError("scripted host infrastructure failure")

    task, config = task_and_config()
    host = FailingHost()
    with entry_point_loading("deny"):
        model = Model(AiperfPipeModelAPI(host=host), config)
    with tempfile.TemporaryDirectory() as directory:
        logs = await eval_async_hosted(
            tasks=task,
            model=model,
            log_dir=directory,
            log_format="eval",
            limit=1,
            epochs=1,
            fail_on_error=False,
            continue_on_fail=True,
            retry_on_error=0,
            max_sandboxes=None,
            log_samples=True,
            log_realtime=False,
            log_images=False,
            score_display=False,
        )
    return normalize_eval_log(logs[0]), host.effects


async def fail_closed_api_checks():
    host = RecordingEvaluationHost([OUTPUT])
    _, config = task_and_config()
    api = AiperfPipeModelAPI(host=host)
    try:
        await api.generate([], [], None, config)
    except RuntimeError as error:
        assert "active sample context" in str(error)
    else:
        raise AssertionError("missing ModelCallContext was accepted")
    assert host.effects == []
    try:
        with entry_point_loading("deny"):
            model = Model(api, config)
        await model.generate("hello", cache=True)
    except RuntimeError as error:
        assert "forbids cache access" in str(error)
    else:
        raise AssertionError("measured-mode cache access was accepted")
    assert host.effects == []

    context = ModelCallContext.activate(
        task_name="task",
        task_id="task-id",
        run_id="run-id",
        sample_id=1,
        epoch=1,
        semantic_attempt=0,
    )
    with context:
        assert context.claim_call().call_ordinal == 0
        assert context.claim_call().call_ordinal == 1
    for action in (context.claim_call, context.__enter__):
        try:
            action()
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale/reused ModelCallContext was accepted")

    try:
        await eval_async_hosted(tasks="gsm8k", model=object())
    except TypeError as error:
        assert "explicit Inspect Task" in str(error)
    else:
        raise AssertionError("hosted runtime accepted registry task lookup")


async def main():
    await fail_closed_api_checks()
    first, first_effects = await success()
    second, _ = await success()
    assert first == second
    json.dumps(first, sort_keys=True, allow_nan=False)
    assert len(first_effects) == 1
    effect = first_effects[0]
    assert (
        effect.sample_id,
        effect.epoch,
        effect.semantic_attempt,
        effect.call_ordinal,
        effect.service_id,
        effect.purpose,
    ) == (1, 1, 0, 0, "candidate", "primary")
    score = first["samples"][0]["scores"]["grade_school_math_scorer"]
    assert score["value"] == 1.0
    metrics = first["results"]["scores"][0]["metrics"]
    assert metrics["accuracy"]["value"] == 1.0
    assert metrics["stderr"]["value"] == 0.0

    failed, failed_effects = await failure()
    assert len(failed_effects) == 1
    assert len(failed["samples"]) == 1
    assert failed["samples"][0]["error"] is not None
    assert failed["samples"][0]["scores"] == {}
    print("RESULT:" + json.dumps(first, sort_keys=True, allow_nan=False))


asyncio.run(main())
"""
    completed = _contained(
        stock_rootfs["openbench"],
        program,
        replacements={"/assets/gsm8k_canary.jsonl": "assets/gsm8k_canary.jsonl"},
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    normalized = _result(completed)
    assert normalized["normalization"] == "openbench-inspect-semantic-v1"
    assert (
        normalized["samples"][0]["scores"]["grade_school_math_scorer"]["value"] == 1.0
    )


def test_integrated_provider_sources_do_not_mutate_private_inspect_or_own_http(
    stock_rootfs: dict[str, Path],
) -> None:
    nemo = stock_rootfs["nemo_evaluator"] / (
        "runtime/lib/python3.12/site-packages/nemo_evaluator"
    )
    openbench = stock_rootfs["openbench"] / (
        "runtime/lib/python3.12/site-packages/openbench"
    )
    for relative in ("engine/host.py", "engine/session.py", "engine/model_client.py"):
        source = (nemo / relative).read_text()
        assert "import aiohttp" not in source
        assert "hosts.local" not in source
    integrated = "\n".join(
        path.read_text()
        for path in (
            openbench / "runtime/inspect_session.py",
            openbench / "model/_providers/aiperf_pipe.py",
        )
    )
    for forbidden in (
        "_registry",
        "client.create =",
        "file_recorder",
        "display =",
        "httpx",
        "openai",
    ):
        assert forbidden not in integrated


def _contained(
    root: Path,
    program: str,
    *,
    replacements: dict[str, str] | None = None,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
    for original, replacement in (replacements or {}).items():
        program = program.replace(original, str(root / replacement))
    state = root / "test-state"
    environment = os.environ.copy()
    for name, relative in (
        ("HOME", "home"),
        ("TMPDIR", "tmp"),
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        directory = state / relative
        directory.mkdir(parents=True, exist_ok=True)
        environment[name] = str(directory)
    environment["PATH"] = str(root / "runtime/bin")
    return subprocess.run(
        [str(root / "runtime/bin/python3.12"), "-I", "-c", textwrap.dedent(program)],
        cwd=root / "work",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _result(completed: subprocess.CompletedProcess[str]) -> Any:
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith("RESULT:")
    ]
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0].removeprefix("RESULT:"))
