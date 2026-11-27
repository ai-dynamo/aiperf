# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nightly check that reports removable uv overrides."""

import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest import param

import tools.check_uv_overrides as module
from tools.check_uv_overrides import (
    EXIT_NEEDED,
    EXIT_PYPI_ERROR,
    EXIT_REMOVABLE,
    LatestRelease,
    PyPIError,
    check_overrides,
    main,
)

_LOCK = """\
version = 1

[manifest]
overrides = [{ name = "nltk", specifier = "==3.10.4" }]

[[package]]
name = "aiperf"
version = "0.14.0"
source = { editable = "." }
dependencies = [{ name = "nltk" }, { name = "lighteval" }]

[[package]]
name = "lighteval"
version = "0.13.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "nltk" }, { name = "rouge-score" }]

[[package]]
name = "Rouge_Score"
version = "0.1.2"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "absl-py" }]

[package.optional-dependencies]
text = [{ name = "NLTK" }]

[[package]]
name = "nltk"
version = "3.10.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "absl-py"
version = "2.3.1"
source = { registry = "https://pypi.org/simple" }
"""

_PYPROJECT = """\
[project]
name = "aiperf"
version = "0.14.0"

[tool.uv]
override-dependencies = ["nltk==3.10.4"]
"""

_ROUGE_OK = LatestRelease(version="0.1.2", requires_dist=["absl-py", "nltk"])


def _fake_pypi(
    releases: dict[str, LatestRelease],
) -> Callable[[str], LatestRelease]:
    def fetch(name: str) -> LatestRelease:
        return releases[name.lower().replace("_", "-")]

    return fetch


def _load(text: str) -> dict:
    return tomllib.loads(text)


@pytest.mark.parametrize(
    "lighteval_requires, removable",
    [
        param(["nltk==3.9.1"], False, id="exact-old-pin"),
        param(["nltk==3.10.4"], True, id="exact-new-pin"),
        param(["nltk>=3.10,<4"], True, id="range-admits"),
        param(["nltk<3.10"], False, id="range-excludes"),
        param(["torch"], True, id="dependency-dropped"),
        param(["nltk>=3.10; python_version >= '3.12'", "nltk==3.9.1; python_version < '3.12'"], False, id="marker-split-any-exclusion-keeps"),
        param(["nltk==3.9.1; extra == 'legacy'", "nltk>=3.10"], False, id="extra-only-exclusion-keeps"),
    ],
)  # fmt: skip
def test_check_overrides_lighteval_requirement_sets_removable(
    lighteval_requires: list[str], removable: bool
) -> None:
    fetch = _fake_pypi(
        {
            "lighteval": LatestRelease(
                version="0.14.0", requires_dist=lighteval_requires
            ),
            "rouge-score": _ROUGE_OK,
        }
    )

    [status] = check_overrides(_load(_PYPROJECT), _load(_LOCK), fetch)

    assert status.package == "nltk"
    assert status.locked_versions == ["3.10.4"]
    assert status.removable is removable


def test_check_overrides_finds_dependents_by_canonical_name_and_extras() -> None:
    """The project root is skipped; `Rouge_Score`'s extra naming `NLTK` counts."""
    fetch = _fake_pypi(
        {
            "lighteval": LatestRelease(version="0.13.0", requires_dist=["nltk==3.9.1"]),
            "rouge-score": _ROUGE_OK,
        }
    )

    [status] = check_overrides(_load(_PYPROJECT), _load(_LOCK), fetch)

    assert [(d.name, d.locked_version, d.admits) for d in status.dependents] == [
        ("lighteval", "0.13.0", False),
        ("rouge-score", "0.1.2", True),
    ]
    assert status.dependents[0].excluding == ["nltk==3.9.1"]


def test_check_overrides_no_overrides_returns_empty() -> None:
    def fetch(name: str) -> LatestRelease:
        raise AssertionError(f"PyPI queried for {name} with no overrides")

    assert (
        check_overrides(_load('[project]\nname = "aiperf"\n'), _load(_LOCK), fetch)
        == []
    )


@pytest.fixture
def repo_files(tmp_path: Path) -> list[str]:
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_text(_PYPROJECT)
    lock.write_text(_LOCK)
    return ["--pyproject", str(pyproject), "--lock", str(lock)]


@pytest.mark.parametrize(
    "lighteval_requires, expected_rc, expected_error",
    [
        param(["nltk==3.9.1"], EXIT_NEEDED, None, id="still-needed"),
        param(["nltk==3.10.4"], EXIT_REMOVABLE, "uv override 'nltk==3.10.4' is no longer needed", id="removable"),
    ],
)  # fmt: skip
def test_main_exit_code_and_annotation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    repo_files: list[str],
    lighteval_requires: list[str],
    expected_rc: int,
    expected_error: str | None,
) -> None:
    monkeypatch.setattr(
        module,
        "fetch_latest_release",
        _fake_pypi(
            {
                "lighteval": LatestRelease(
                    version="0.14.0", requires_dist=lighteval_requires
                ),
                "rouge-score": _ROUGE_OK,
            }
        ),
    )

    rc = main(repo_files)

    out = capsys.readouterr().out
    assert rc == expected_rc
    if expected_error is None:
        assert "::error::" not in out
    else:
        assert f"::error::{expected_error}" in out
    assert "lighteval locked 0.13.0, latest 0.14.0" in out


def test_main_pypi_failure_exits_2_with_distinct_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    repo_files: list[str],
) -> None:
    def fetch(name: str) -> LatestRelease:
        raise PyPIError(f"https://pypi.org/pypi/{name}/json: timed out")

    monkeypatch.setattr(module, "fetch_latest_release", fetch)

    rc = main(repo_files)

    out = capsys.readouterr().out
    assert rc == EXIT_PYPI_ERROR
    assert "::error::Could not query PyPI, override status unknown" in out
    assert "no longer needed" not in out


def test_fetch_latest_release_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(*_args: object, **_kwargs: object) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)

    with pytest.raises(PyPIError, match="connection refused"):
        module.fetch_latest_release("lighteval")


def test_repo_overrides_parse_against_repo_lock() -> None:
    """The real pyproject/uv.lock pair parses, and every override is locked."""
    repo = Path(__file__).resolve().parents[3]

    def fetch(name: str) -> LatestRelease:
        return LatestRelease(version="0", requires_dist=[])

    statuses = check_overrides(
        _load((repo / "pyproject.toml").read_text()),
        _load((repo / "uv.lock").read_text()),
        fetch,
    )

    for status in statuses:
        assert status.locked_versions, f"{status.override} not found in uv.lock"
        assert status.dependents, f"{status.override} has no locked dependents"
