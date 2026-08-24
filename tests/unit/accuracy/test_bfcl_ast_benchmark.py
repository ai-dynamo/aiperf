# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``BFCLASTBenchmark`` (BFCL Prompt-mode loader).

Pins:
1. The validator gates: Prompt mode is zero-shot, has no CoT, cannot take a
   global system-prompt override, and names the deliberately out-of-scope BFCL
   category families rather than reporting them as unknown.
2. ``task`` is the BFCL category (not the benchmark name), which is what drives
   the existing per-task console/CSV breakdown.
3. ``ground_truth`` is the orjson payload ``ToolCallASTGrader`` consumes, with
   ``possible_answer`` passed through verbatim.
4. The version pin is enforced, with ``any`` as the documented escape hatch.

Dataset access is redirected at the ``_bfcl_compat`` seam to tmp fixture files
in the bundled JSON-lines shape, so these run without the ``[bfcl]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import orjson
import pytest
from pytest import param

from aiperf.accuracy.benchmarks.bfcl_ast import (
    DEFAULT_CATEGORIES,
    DEFAULT_GENERATION_SIZE,
    LIVE_CATEGORIES,
    NON_LIVE_CATEGORIES,
    SUPPORTED_CATEGORIES,
    TASK_NAME,
    BFCLASTBenchmark,
)
from aiperf.accuracy.graders import _bfcl_compat
from aiperf.accuracy.graders._bfcl_compat import ast_check as _real_ast_check
from aiperf.accuracy.graders._bfcl_compat import (
    bfcl_available as _real_bfcl_available_fn,
)
from aiperf.accuracy.graders._bfcl_compat import (
    check_version_pin as _real_check_version_pin,
)
from aiperf.accuracy.graders._bfcl_compat import decode_calls as _real_decode_calls
from aiperf.accuracy.graders._bfcl_compat import (
    installed_version as _real_installed_version,
)
from aiperf.accuracy.graders._bfcl_compat import require_bfcl as _real_require_bfcl
from aiperf.common.environment import Environment
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.harness import fake_bfcl
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _make_run(**accuracy: Any) -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.BFCL_AST, **accuracy},
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(orjson.dumps(row).decode("utf-8") for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def bfcl_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialize the fake dataset in BFCL's bundled on-disk layout."""
    data_dir = tmp_path / "data"
    answer_dir = data_dir / "possible_answer"
    for category, rows in fake_bfcl.DATA_ROWS.items():
        _write_jsonl(data_dir / f"{fake_bfcl.VERSION_PREFIX}_{category}.json", rows)
    for category, rows in fake_bfcl.POSSIBLE_ANSWER_ROWS.items():
        _write_jsonl(answer_dir / f"{fake_bfcl.VERSION_PREFIX}_{category}.json", rows)
    monkeypatch.setattr(_bfcl_compat, "data_dir", lambda: data_dir)
    monkeypatch.setattr(_bfcl_compat, "possible_answer_dir", lambda: answer_dir)
    return data_dir


class TestValidatorGates:
    """Unsupported constructs fail loudly instead of silently no-op'ing."""

    @pytest.mark.asyncio
    async def test_load_problems_nonzero_n_shots_raises(self) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(NotImplementedError, match=f"^{TASK_NAME}: "):
            await benchmark.load_problems(tasks=None, n_shots=5, enable_cot=False)

    @pytest.mark.asyncio
    async def test_load_problems_enable_cot_raises(self) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(NotImplementedError, match="enable-cot"):
            await benchmark.load_problems(tasks=None, n_shots=0, enable_cot=True)

    @pytest.mark.asyncio
    async def test_load_problems_with_system_prompt_override_raises(self) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run(system_prompt="Be helpful."))
        with pytest.raises(NotImplementedError, match="per-problem system prompt"):
            await benchmark.load_problems(tasks=None, n_shots=0, enable_cot=False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "task,reason_fragment",
        [
            param("exec_simple", "execution sandbox", id="exec"),
            param("multi_turn_base", "stateless and per-record", id="multi_turn"),
            param("web_search_base", "live web search", id="web_search"),
            param("memory_kv", "persistent state", id="memory"),
            param("format_sensitivity", "non-scoring", id="format_sensitivity"),
        ],
    )  # fmt: skip
    async def test_load_problems_out_of_scope_task_raises_with_reason(
        self, task: str, reason_fragment: str
    ) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(NotImplementedError, match=reason_fragment):
            await benchmark.load_problems(tasks=[task], n_shots=0, enable_cot=False)

    @pytest.mark.asyncio
    async def test_load_problems_unknown_task_lists_supported(self) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(NotImplementedError, match="simple_python"):
            await benchmark.load_problems(
                tasks=["not_a_category"], n_shots=0, enable_cot=False
            )


class TestProblemShape:
    """Each entry becomes a chat problem the dataset loader can consume."""

    @pytest.mark.asyncio
    async def test_load_problems_task_label_is_bfcl_category(
        self, bfcl_data: Path
    ) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python", "parallel"], n_shots=0, enable_cot=False
        )
        assert {p.task for p in problems} == {"simple_python", "parallel"}

    @pytest.mark.asyncio
    async def test_load_problems_emits_system_then_user_messages(
        self, bfcl_data: Path
    ) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python"], n_shots=0, enable_cot=False
        )
        roles = [m["role"] for m in problems[0].raw_messages]
        assert roles == ["system", "user"]
        assert "get_weather" in problems[0].raw_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_load_problems_flat_prompt_joins_all_messages(
        self, bfcl_data: Path
    ) -> None:
        """The completions endpoint uses the flat prompt, not raw_messages."""
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python"], n_shots=0, enable_cot=False
        )
        for message in problems[0].raw_messages:
            assert message["content"] in problems[0].prompt

    @pytest.mark.asyncio
    async def test_load_problems_sets_metadata_for_grading(
        self, bfcl_data: Path
    ) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python"], n_shots=0, enable_cot=False
        )
        metadata = problems[0].metadata
        assert metadata["bfcl_id"] == "simple_python_0"
        assert metadata["language"] == "python"
        assert metadata["test_category"] == "simple_python"
        assert metadata["generation_size"] == DEFAULT_GENERATION_SIZE

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "category,expected_language",
        [
            param("simple_python", "python", id="python"),
            param("simple_java", "java", id="java"),
            param("simple_javascript", "javascript", id="javascript"),
            param("parallel", "python", id="parallel_defaults_python"),
        ],
    )  # fmt: skip
    async def test_language_is_derived_from_category(
        self, category: str, expected_language: str
    ) -> None:
        from aiperf.accuracy.benchmarks.bfcl_ast import _language_for

        assert _language_for(category) == expected_language

    @pytest.mark.asyncio
    async def test_load_problems_defaults_to_non_live_categories(
        self, bfcl_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting --accuracy-tasks evaluates the non-live set."""
        requested: list[str] = []
        original = BFCLASTBenchmark._load_categories

        def _spy(self, categories):
            requested.extend(categories)
            return original(
                self, tuple(c for c in categories if c in fake_bfcl.DATA_ROWS)
            )

        monkeypatch.setattr(BFCLASTBenchmark, "_load_categories", _spy)
        benchmark = BFCLASTBenchmark(run=_make_run())
        await benchmark.load_problems(tasks=None, n_shots=0, enable_cot=False)
        assert tuple(requested) == DEFAULT_CATEGORIES


class TestGroundTruthSerialization:
    """The blob is the grader's entire input; it must round-trip exactly."""

    @pytest.mark.asyncio
    async def test_ground_truth_roundtrips_possible_answer_verbatim(
        self, bfcl_data: Path
    ) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python"], n_shots=0, enable_cot=False
        )
        payload = orjson.loads(problems[0].ground_truth)
        assert (
            payload["possible_answer"]
            == fake_bfcl.POSSIBLE_ANSWER_ROWS["simple_python"][0]["ground_truth"]
        )
        assert payload["test_category"] == "simple_python"
        assert payload["language"] == "python"
        assert payload["function"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_irrelevance_ground_truth_has_no_possible_answer(
        self, bfcl_data: Path
    ) -> None:
        """Hallucination categories ship no possible_answer file upstream."""
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["irrelevance"], n_shots=0, enable_cot=False
        )
        assert orjson.loads(problems[0].ground_truth)["possible_answer"] is None


class TestDatasetErrors:
    """Bad installs fail loudly rather than yielding a short problem set."""

    @pytest.mark.asyncio
    async def test_missing_category_file_raises_actionable(
        self, bfcl_data: Path
    ) -> None:
        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(RuntimeError, match=r"aiperf\[bfcl\]"):
            await benchmark.load_problems(
                tasks=["live_simple"], n_shots=0, enable_cot=False
            )

    @pytest.mark.asyncio
    async def test_malformed_jsonl_raises_naming_the_line(
        self, bfcl_data: Path
    ) -> None:
        path = bfcl_data / f"{fake_bfcl.VERSION_PREFIX}_simple_python.json"
        path.write_text('{"id": "ok"}\nnot json\n', encoding="utf-8")
        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(RuntimeError, match="line 2"):
            await benchmark.load_problems(
                tasks=["simple_python"], n_shots=0, enable_cot=False
            )


class TestVersionPin:
    """Dataset and checker ship together, so the package version is the pin."""

    def test_version_pin_mismatch_raises_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_bfcl_compat, "installed_version", lambda: "1.2.3")
        monkeypatch.setattr(Environment.ACCURACY, "BFCL_VERSION_PIN", "2026.3.23")
        with pytest.raises(RuntimeError) as exc:
            _real_check_version_pin()
        assert "1.2.3" in str(exc.value)
        assert "2026.3.23" in str(exc.value)
        assert "uv pip install" in str(exc.value)

    def test_version_pin_any_skips_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _explode() -> str:
            raise AssertionError("installed_version must not be consulted")

        monkeypatch.setattr(_bfcl_compat, "installed_version", _explode)
        monkeypatch.setattr(
            Environment.ACCURACY, "BFCL_VERSION_PIN", _bfcl_compat.ANY_VERSION
        )
        _real_check_version_pin()


class TestCategoryDrift:
    """Our category tuples mirror upstream's lists."""

    def test_supported_categories_are_non_live_plus_live(self) -> None:
        assert SUPPORTED_CATEGORIES == NON_LIVE_CATEGORIES + LIVE_CATEGORIES

    def test_default_categories_exclude_live(self) -> None:
        assert not any(c.startswith("live_") for c in DEFAULT_CATEGORIES)


class TestCheckAvailable:
    """Preflight gate for the optional extra."""

    def test_check_available_without_bfcl_raises_actionable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_bfcl_compat, "require_bfcl", _real_require_bfcl)
        monkeypatch.setattr(_bfcl_compat, "bfcl_available", lambda: False)
        with pytest.raises(RuntimeError, match=r"aiperf\[bfcl\]"):
            BFCLASTBenchmark.check_available()


class TestGroundTruthJoinIntegrity:
    """A question with no matching gold must fail loudly, not score 0%."""

    @pytest.mark.asyncio
    async def test_missing_possible_answer_entry_raises(self, bfcl_data: Path) -> None:
        """Dropping the gold row leaves the question with no ground truth;
        grading it would report a silent 0% for the whole category."""
        answer_path = (
            bfcl_data
            / "possible_answer"
            / f"{fake_bfcl.VERSION_PREFIX}_simple_python.json"
        )
        answer_path.write_text("", encoding="utf-8")

        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(RuntimeError, match="has no matching entry"):
            await benchmark.load_problems(
                tasks=["simple_python"], n_shots=0, enable_cot=False
            )

    @pytest.mark.asyncio
    async def test_mismatched_id_raises_naming_the_question(
        self, bfcl_data: Path
    ) -> None:
        answer_path = (
            bfcl_data
            / "possible_answer"
            / f"{fake_bfcl.VERSION_PREFIX}_simple_python.json"
        )
        row = dict(fake_bfcl.POSSIBLE_ANSWER_ROWS["simple_python"][0])
        row["id"] = "simple_python_999"
        answer_path.write_text(
            orjson.dumps(row).decode("utf-8") + "\n", encoding="utf-8"
        )

        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(RuntimeError, match="simple_python_0"):
            await benchmark.load_problems(
                tasks=["simple_python"], n_shots=0, enable_cot=False
            )

    @pytest.mark.asyncio
    async def test_hallucination_categories_are_exempt_from_the_join(
        self, bfcl_data: Path
    ) -> None:
        """These ship no possible_answer file upstream by design."""
        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["irrelevance"], n_shots=0, enable_cot=False
        )
        assert len(problems) == 1


class TestSymlinkedInstallPath:
    """site-packages is routinely reached through a symlink."""

    @pytest.mark.asyncio
    async def test_dataset_reachable_through_a_symlinked_directory(
        self, bfcl_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pyenv, conda and symlinked venvs all produce this layout; rejecting
        it would make the bundled dataset unreadable with a misleading error."""
        link = tmp_path / "linked_data"
        link.symlink_to(bfcl_data, target_is_directory=True)
        monkeypatch.setattr(_bfcl_compat, "data_dir", lambda: link)
        monkeypatch.setattr(
            _bfcl_compat, "possible_answer_dir", lambda: link / "possible_answer"
        )

        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python"], n_shots=0, enable_cot=False
        )

        assert len(problems) == 1
        assert problems[0].task == "simple_python"


class TestPreflightVersionPin:
    """The pin is enforced up front, as the docs promise."""

    def test_check_available_enforces_the_version_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mismatch invalidates the run's numbers, so it must fail in
        preflight rather than after a full service bootstrap."""
        monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
        monkeypatch.setattr(_bfcl_compat, "check_version_pin", _real_check_version_pin)
        monkeypatch.setattr(_bfcl_compat, "installed_version", lambda: "1.2.3")
        monkeypatch.setattr(Environment.ACCURACY, "BFCL_VERSION_PIN", "2026.3.23")

        with pytest.raises(RuntimeError, match="2026.3.23"):
            BFCLASTBenchmark.check_available()

    def test_check_available_passes_when_pinned_version_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
        monkeypatch.setattr(_bfcl_compat, "check_version_pin", _real_check_version_pin)
        monkeypatch.setattr(_bfcl_compat, "installed_version", lambda: "2026.3.23")
        monkeypatch.setattr(Environment.ACCURACY, "BFCL_VERSION_PIN", "2026.3.23")

        BFCLASTBenchmark.check_available()


class TestCompatFailureVisibility:
    """Upstream drift must surface as an error, not as a graded verdict."""

    def test_unknown_language_raises_instead_of_reporting_unparsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ReturnFormat member is broken wiring, not a bad response.

        Folded into the decode failure it would mark every problem in the
        affected language ``unparsed`` — reading as a model that never emits a
        parseable call rather than as an integration bug.
        """
        monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
        monkeypatch.setattr(
            _bfcl_compat, "_resolve", lambda candidates, what: fake_bfcl.ast_parse
        )

        def _no_member(_language: str) -> Any:
            raise ValueError("'klingon' is not a valid ReturnFormat")

        monkeypatch.setattr(_bfcl_compat, "_return_format_enum", _no_member)

        with pytest.raises(ValueError, match="ReturnFormat"):
            _real_decode_calls("[get_weather(city='SF')]", "klingon")

    def test_version_for_message_degrades_instead_of_masking_the_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``installed_version`` raises without distribution metadata; inside
        another error's message that would replace the real cause."""

        def _no_metadata() -> str:
            raise RuntimeError("no distribution metadata")

        monkeypatch.setattr(_bfcl_compat, "installed_version", _no_metadata)
        assert _bfcl_compat._version_for_message() == "<version unavailable>"

    def test_unresolvable_symbol_error_names_every_path_tried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original 'cannot locate' cause must survive a missing version."""

        def _no_metadata() -> str:
            raise RuntimeError("no distribution metadata")

        monkeypatch.setattr(_bfcl_compat, "installed_version", _no_metadata)
        monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)

        with pytest.raises(RuntimeError) as exc:
            _bfcl_compat._resolve(
                (("aiperf.nonexistent.module", "nope"),), "a made-up symbol"
            )

        assert "cannot locate a made-up symbol" in str(exc.value)
        assert "aiperf.nonexistent.module:nope" in str(exc.value)


class TestCompatResolvers:
    """The lazy resolvers themselves, which the fake normally stands in for."""

    def test_bfcl_available_is_false_when_find_spec_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-installed package can make ``find_spec`` raise rather than
        return None; that still means "not usable"."""
        import importlib.util

        def _raise(_name: str) -> Any:
            raise ValueError("__spec__ is not set")

        monkeypatch.setattr(importlib.util, "find_spec", _raise)
        assert _real_bfcl_available_fn() is False

    def test_installed_version_without_metadata_raises_actionable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata

        def _raise(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError("bfcl-eval")

        monkeypatch.setattr(importlib.metadata, "version", _raise)
        with pytest.raises(RuntimeError, match=r"aiperf\[bfcl\]"):
            _real_installed_version()

    def test_package_root_without_search_locations_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A namespace package or broken install exposes no directory to read
        the bundled dataset from."""
        import importlib.util

        monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda _name: SimpleNamespace(submodule_search_locations=None),
        )
        with pytest.raises(RuntimeError, match="no package directory"):
            _bfcl_compat.package_root()

    def test_data_dirs_are_derived_from_the_package_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_bfcl_compat, "package_root", lambda: tmp_path)
        assert _bfcl_compat.data_dir() == tmp_path / "data"
        assert (
            _bfcl_compat.possible_answer_dir() == tmp_path / "data" / "possible_answer"
        )

    def test_ast_check_refuses_a_checker_missing_a_known_parameter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling positionally past a renamed parameter would mis-grade the
        whole run in silence, so the shim refuses instead."""

        def _renamed_checker(
            func_description, model_output, gold, language, test_category, model_name
        ):  # noqa: ANN001, ANN202
            return {"valid": True}

        monkeypatch.setattr(_bfcl_compat, "require_bfcl", lambda: None)
        monkeypatch.setattr(_bfcl_compat, "installed_version", lambda: "9.9.9")
        monkeypatch.setattr(
            _bfcl_compat, "_resolve", lambda candidates, what: _renamed_checker
        )
        monkeypatch.setattr(_bfcl_compat, "_language_enum", lambda language: language)

        with pytest.raises(RuntimeError, match="no parameter for 'possible_answer'"):
            _real_ast_check(
                func_description=[],
                model_output=[],
                possible_answer=[],
                language="python",
                test_category="simple_python",
                model_name="aiperf",
            )


class TestJsonlReadEdgeCases:
    """Reading the bundled JSON-lines files."""

    @pytest.mark.asyncio
    async def test_blank_lines_are_skipped(self, bfcl_data: Path) -> None:
        """Upstream's files end with a trailing newline; a blank line is not a
        malformed record."""
        path = bfcl_data / f"{fake_bfcl.VERSION_PREFIX}_simple_python.json"
        row = orjson.dumps(fake_bfcl.DATA_ROWS["simple_python"][0]).decode("utf-8")
        path.write_text(f"\n{row}\n\n", encoding="utf-8")

        benchmark = BFCLASTBenchmark(run=_make_run())
        problems = await benchmark.load_problems(
            tasks=["simple_python"], n_shots=0, enable_cot=False
        )

        assert len(problems) == 1

    @pytest.mark.asyncio
    async def test_unresolvable_path_raises_naming_the_file(
        self, bfcl_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path that cannot be canonicalized (e.g. a symlink loop) must name
        the file rather than surfacing a bare OSError."""
        real_resolve = Path.resolve

        def _raise(self, *args, **kwargs):  # noqa: ANN001, ANN202
            if self.name.endswith("_simple_python.json"):
                raise OSError("Too many levels of symbolic links")
            return real_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", _raise)

        benchmark = BFCLASTBenchmark(run=_make_run())
        with pytest.raises(RuntimeError, match="cannot resolve"):
            await benchmark.load_problems(
                tasks=["simple_python"], n_shots=0, enable_cot=False
            )
