# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the yaml-editor.js highlightYaml tokeniser."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
_YAML_EDITOR_PATH = _UI_DIR / "components" / "yaml-editor.js"


def _script(body: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({json.dumps(str(_YAML_EDITOR_PATH))}, 'utf8');
        const helpers = source
          .replace(/^import .*$/gm, '')
          .replace(/^export /gm, '');
        eval(helpers + '\\n' + {json.dumps(body)});
    """


def _highlight(yaml: str) -> list[dict]:
    script = _script(f"console.log(JSON.stringify(highlightYaml({json.dumps(yaml)})));")
    return json.loads(run_node(script))


def test_yaml_key_value_pair_is_tokenised_correctly() -> None:
    tokens = _highlight("concurrency: 128")
    clss = [t["cls"] for t in tokens]
    assert "key" in clss
    assert "num" in clss
    assert "punct" in clss
    key_tok = next(t for t in tokens if t["cls"] == "key")
    num_tok = next(t for t in tokens if t["cls"] == "num")
    assert key_tok["text"] == "concurrency"
    assert num_tok["text"] == "128"


def test_yaml_boolean_values_are_tokenised_as_bool() -> None:
    for val in ("true", "false"):
        tokens = _highlight(f"streaming: {val}")
        bool_tok = next((t for t in tokens if t["cls"] == "bool"), None)
        assert bool_tok is not None, f"expected bool token for value '{val}'"
        assert bool_tok["text"] == val


def test_yaml_null_values_are_tokenised_as_null() -> None:
    for null_val in ("null", "~"):
        tokens = _highlight(f"value: {null_val}")
        null_tok = next((t for t in tokens if t["cls"] == "null"), None)
        assert null_tok is not None, f"expected null token for value '{null_val}'"
        assert null_tok["text"] == null_val


def test_yaml_quoted_strings_are_tokenised_as_string() -> None:
    tokens = _highlight('endpoint: "http://svc.ns:8000"')
    str_tok = next((t for t in tokens if t["cls"] == "string"), None)
    assert str_tok is not None
    assert str_tok["text"] == '"http://svc.ns:8000"'


def test_yaml_full_line_comment_is_tokenised_as_comment() -> None:
    tokens = _highlight("# This is a comment")
    comment_tok = next((t for t in tokens if t["cls"] == "comment"), None)
    assert comment_tok is not None
    assert comment_tok["text"] == "# This is a comment"


def test_yaml_inline_comment_is_split_from_value() -> None:
    tokens = _highlight("workers: 4 # inline comment")
    clss = [t["cls"] for t in tokens if t["cls"] is not None]
    assert "key" in clss
    assert "num" in clss
    assert "comment" in clss
    comment_tok = next(t for t in tokens if t["cls"] == "comment")
    assert "# inline comment" in comment_tok["text"]


def test_yaml_doc_marker_is_tokenised_as_meta() -> None:
    tokens = _highlight("---")
    assert tokens[0]["cls"] == "meta"
    assert tokens[0]["text"] == "---"


def test_yaml_sequence_dash_is_tokenised_as_dash() -> None:
    tokens = _highlight("  - my-model")
    dash_tok = next((t for t in tokens if t["cls"] == "dash"), None)
    assert dash_tok is not None
    assert "- " in dash_tok["text"]


def test_yaml_float_value_is_tokenised_as_num() -> None:
    tokens = _highlight("ratio: 1.5")
    num_tok = next((t for t in tokens if t["cls"] == "num"), None)
    assert num_tok is not None
    assert num_tok["text"] == "1.5"


def test_yaml_flow_inline_sequence_is_tokenised_as_flow() -> None:
    tokens = _highlight("models: [gpt4, llama3]")
    flow_tok = next((t for t in tokens if t["cls"] == "flow"), None)
    assert flow_tok is not None


def test_yaml_empty_key_emits_key_and_punct() -> None:
    tokens = _highlight("spec:")
    key_tok = next((t for t in tokens if t["cls"] == "key"), None)
    assert key_tok is not None
    assert key_tok["text"] == "spec"


def test_yaml_multiline_newlines_are_emitted_between_lines() -> None:
    tokens = _highlight("a: 1\nb: 2")
    newlines = [t for t in tokens if t["cls"] is None and t["text"] == "\n"]
    assert len(newlines) >= 2


def test_yaml_plain_url_value_is_not_recoloured() -> None:
    tokens = _highlight("url: http://example.com")
    key_tok = next(t for t in tokens if t["cls"] == "key")
    assert key_tok["text"] == "url"
    url_tok = next(
        (t for t in tokens if t["cls"] is None and "http" in t["text"]), None
    )
    assert url_tok is not None


def test_yaml_editor_source_contains_no_raw_html_sinks() -> None:
    source = _YAML_EDITOR_PATH.read_text()
    assert "innerHTML" not in source
    assert "dangerouslySetInnerHTML" not in source
    assert "insertAdjacentHTML" not in source
