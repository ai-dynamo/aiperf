# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for docs-e2e ``setup-file-`` fixtures.

Guides driven by ``--config foo.yaml`` print the config in the page but have no
way to put it on disk, so they cannot be covered by the end-to-end test at all.
A ``setup-file-`` tag materializes that block before the guide's commands run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import param

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from parser import MarkdownParser  # noqa: E402

SERVER = "vllm-default-openai"


def _parse(tmp_path: Path, body: str) -> MarkdownParser:
    doc = tmp_path / "guide.md"
    doc.write_text(body, encoding="utf-8")
    parser = MarkdownParser()
    parser._parse_file(str(doc))
    return parser


def test_file_fixture_is_captured_with_its_path(tmp_path: Path) -> None:
    parser = _parse(
        tmp_path,
        f"""
<!-- setup-file-{SERVER}-endpoint-server path=multi-phase.yaml -->
```yaml
phases:
  - name: warmup
    kind: warmup
```
<!-- /setup-file-{SERVER}-endpoint-server -->
""",
    )
    files = parser.servers[SERVER].files
    assert len(files) == 1
    assert files[0].path == "multi-phase.yaml"
    assert "kind: warmup" in files[0].content


@pytest.mark.parametrize(
    "language",
    [param("yaml", id="yaml"), param("json", id="json"), param("jsonl", id="jsonl")],
)  # fmt: skip
def test_file_fixture_accepts_any_language(tmp_path: Path, language: str) -> None:
    """``_extract_bash_block`` only accepts ```bash, which would drop configs."""
    parser = _parse(
        tmp_path,
        f"""
<!-- setup-file-{SERVER}-endpoint-server path=data.{language} -->
```{language}
{{"a": 1}}
```
<!-- /setup-file-{SERVER}-endpoint-server -->
""",
    )
    assert parser.servers[SERVER].files[0].content.strip() == '{"a": 1}'


def test_file_fixture_without_a_path_is_ignored(tmp_path: Path) -> None:
    parser = _parse(
        tmp_path,
        f"""
<!-- setup-file-{SERVER}-endpoint-server -->
```yaml
a: 1
```
""",
    )
    assert parser.servers.get(SERVER) is None or not parser.servers[SERVER].files


def test_run_tags_still_parse_alongside_file_tags(tmp_path: Path) -> None:
    """setup-file- is a longer prefix of setup-; neither may shadow the other."""
    parser = _parse(
        tmp_path,
        f"""
<!-- setup-file-{SERVER}-endpoint-server path=c.yaml -->
```yaml
a: 1
```
<!-- /setup-file-{SERVER}-endpoint-server -->

<!-- aiperf-run-{SERVER}-endpoint-server -->
```bash
aiperf profile --config c.yaml
```
<!-- /aiperf-run-{SERVER}-endpoint-server -->
""",
    )
    server = parser.servers[SERVER]
    assert [f.path for f in server.files] == ["c.yaml"]
    assert len(server.aiperf_commands) == 1
    assert server.setup_command is None


def test_a_file_tag_is_not_mistaken_for_a_server_setup(tmp_path: Path) -> None:
    parser = _parse(
        tmp_path,
        f"""
<!-- setup-file-{SERVER}-endpoint-server path=c.yaml -->
```yaml
a: 1
```
""",
    )
    # A setup-file- tag must not register itself as the server's boot command.
    assert parser.servers[SERVER].setup_command is None


class _Server:
    def __init__(self, files):
        self.name = "s"
        self.files = files


class _Fixture:
    def __init__(self, path):
        self.path = path
        self.content = "a: 1"
        self.file_path = "docs/x.md"
        self.start_line = 1


@pytest.mark.parametrize(
    "path",
    [
        param("/etc/passwd", id="absolute"),
        param("../outside.yaml", id="parent-traversal"),
        param("nested/../../escape.yaml", id="nested-traversal"),
    ],
)  # fmt: skip
def test_materialize_refuses_to_escape_the_working_directory(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doc is untrusted input to CI; a fixture path must stay contained."""
    from test_runner import EndToEndTestRunner

    called = False

    def _fail(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("docker exec must not run for a rejected path")

    monkeypatch.setattr("test_runner.subprocess.run", _fail)
    runner = EndToEndTestRunner.__new__(EndToEndTestRunner)
    runner.aiperf_container_id = "cid"

    assert runner._materialize_files(_Server([_Fixture(path)])) is False
    assert not called
