# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UserFile model: path validation, format inference, content typing."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aiperf.config.user_files import (
    RunMeta,
    UserFile,
    UserFileError,
    build_user_file_context,
    materialize_user_files,
)

# --- Path validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "input_config.json",
        "meta/notes.md",
        "deep/nested/info.txt",
    ],
)
def test_valid_paths_accepted(path):
    f = UserFile(path=path, content="ok")
    assert f.path == path


@pytest.mark.parametrize(
    "path,reason_substring",
    [
        ("/etc/passwd", "absolute"),
        ("../escape.json", ".."),
        ("foo/../bar.json", ".."),
        ("", "empty"),
        ("with\x00null.json", "control"),
    ],
)
def test_invalid_paths_rejected(path, reason_substring):
    with pytest.raises(ValidationError) as exc_info:
        UserFile(path=path, content="ok")
    assert reason_substring in str(exc_info.value).lower()


# --- Format inference ---------------------------------------------------------


def test_format_inferred_text_for_string_content():
    f = UserFile(path="x.txt", content="hello")
    assert f.format == "text"


def test_format_inferred_json_for_dict_content():
    f = UserFile(path="x.json", content={"a": 1})
    assert f.format == "json"


def test_format_inferred_json_for_list_content():
    f = UserFile(path="x.json", content=[1, 2, 3])
    assert f.format == "json"


def test_explicit_yaml_format_with_dict_content():
    f = UserFile(path="x.yaml", format="yaml", content={"a": 1})
    assert f.format == "yaml"


# --- Format/content mismatch --------------------------------------------------


def test_json_format_with_string_content_rejected():
    with pytest.raises(ValidationError) as exc_info:
        UserFile(path="x.json", format="json", content="raw string")
    assert "structured" in str(exc_info.value).lower()


def test_text_format_with_dict_content_rejected():
    with pytest.raises(ValidationError) as exc_info:
        UserFile(path="x.txt", format="text", content={"a": 1})
    assert "string" in str(exc_info.value).lower()


# --- build_user_file_context --------------------------------------------------


def _stub_config(variables=None, model="m", url="http://x"):
    """Minimal duck-typed config for build_user_file_context."""
    from types import SimpleNamespace

    return SimpleNamespace(
        variables=variables or {},
        benchmark=SimpleNamespace(
            models=[model],
            endpoint=SimpleNamespace(urls=[url]),
        ),
    )


def test_context_includes_injected_names(tmp_path):
    config = _stub_config(variables={"isl": 1024})
    meta = RunMeta(epoch="1714", job_name="run-1", namespace="ns")
    ctx = build_user_file_context(config, meta, run_dir=tmp_path)
    assert ctx["epoch"] == "1714"
    assert ctx["job_name"] == "run-1"
    assert ctx["namespace"] == "ns"
    assert ctx["model"] == "m"
    assert ctx["endpoint_url"] == "http://x"
    assert ctx["artifact_dir"] == str(tmp_path)
    assert ctx["isl"] == 1024


def test_collision_injected_wins_and_warns(caplog):
    config = _stub_config(variables={"epoch": "user-supplied"})
    meta = RunMeta(epoch="1714", job_name="r", namespace="")
    with caplog.at_level("WARNING"):
        ctx = build_user_file_context(config, meta, run_dir=Path("/tmp"))
    assert ctx["epoch"] == "1714"
    assert any(
        "epoch" in r.message and "shadow" in r.message.lower() for r in caplog.records
    )


# --- materialize_user_files ---------------------------------------------------


def test_materialize_json_renders_int_as_int(tmp_path):
    files = [UserFile(path="a.json", format="json", content={"n": "{{ x }}"})]
    materialize_user_files(files, run_dir=tmp_path, context={"x": 42})
    data = json.loads((tmp_path / "a.json").read_text())
    assert data == {"n": 42}  # not "42"


def test_materialize_yaml_round_trip(tmp_path):
    files = [UserFile(path="a.yaml", format="yaml", content={"k": "{{ v }}"})]
    materialize_user_files(files, run_dir=tmp_path, context={"v": "hello"})
    data = yaml.safe_load((tmp_path / "a.yaml").read_text())
    assert data == {"k": "hello"}


def test_materialize_text_preserves_newlines(tmp_path):
    files = [UserFile(path="notes.md", content="line {{ n }}\nend")]
    materialize_user_files(files, run_dir=tmp_path, context={"n": 1})
    assert (tmp_path / "notes.md").read_text() == "line 1\nend"


def test_materialize_subdir_creates_intermediate_dirs(tmp_path):
    files = [UserFile(path="meta/sub/a.json", content={"a": 1})]
    materialize_user_files(files, run_dir=tmp_path, context={})
    assert (tmp_path / "meta" / "sub" / "a.json").exists()


def test_materialize_undefined_variable_raises_with_path(tmp_path):
    files = [UserFile(path="a.txt", content="{{ missing }}")]
    with pytest.raises(UserFileError) as exc_info:
        materialize_user_files(files, run_dir=tmp_path, context={})
    msg = str(exc_info.value)
    assert "missing" in msg
    assert "a.txt" in msg


def test_materialize_overwrites_existing(tmp_path):
    (tmp_path / "a.txt").write_text("old")
    files = [UserFile(path="a.txt", content="new")]
    materialize_user_files(files, run_dir=tmp_path, context={})
    assert (tmp_path / "a.txt").read_text() == "new"


def test_materialize_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "evil").symlink_to(outside)
    files = [UserFile(path="evil/a.txt", content="x")]
    with pytest.raises(UserFileError) as exc_info:
        materialize_user_files(files, run_dir=tmp_path, context={})
    assert (
        "outside run dir" in str(exc_info.value).lower()
        or "escape" in str(exc_info.value).lower()
    )


def test_materialize_write_failure_raises(tmp_path):
    import os

    (tmp_path / "a.txt").write_text("seed")
    os.chmod(tmp_path, 0o500)  # read+exec only
    try:
        files = [UserFile(path="b.txt", content="x")]
        with pytest.raises(UserFileError):
            materialize_user_files(files, run_dir=tmp_path, context={})
    finally:
        os.chmod(tmp_path, 0o700)


def test_materialize_jinja_comment_is_stripped(tmp_path):
    files = [UserFile(path="a.txt", content="hello {# c #} world")]
    materialize_user_files(files, run_dir=tmp_path, context={})
    assert (tmp_path / "a.txt").read_text() == "hello  world"


def test_materialize_intermediate_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside_2"
    outside.mkdir(exist_ok=True)
    (tmp_path / "evil").symlink_to(outside)
    files = [UserFile(path="evil/sub/a.txt", content="x")]
    with pytest.raises(UserFileError) as exc_info:
        materialize_user_files(files, run_dir=tmp_path, context={})
    msg = str(exc_info.value).lower()
    assert "outside run dir" in msg or "escape" in msg
    # Critical: outside dir should NOT have a `sub` subdir created.
    assert not (outside / "sub").exists()


def test_materialize_json_preserves_literal_strings(tmp_path):
    """Literal '42' (no jinja) stays a string; rendered '{{ x }}' with x=42 becomes int."""
    files = [
        UserFile(
            path="a.json",
            format="json",
            content={
                "literal_id": "42",  # user wrote a string literal
                "rendered_id": "{{ x }}",  # context: x=42
            },
        )
    ]
    materialize_user_files(files, run_dir=tmp_path, context={"x": 42})
    data = json.loads((tmp_path / "a.json").read_text())
    assert data == {"literal_id": "42", "rendered_id": 42}
