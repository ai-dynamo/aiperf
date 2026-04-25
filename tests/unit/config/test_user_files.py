# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UserFile model: path validation, format inference, content typing."""

import pytest
from pydantic import ValidationError

from aiperf.config.user_files import UserFile

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
