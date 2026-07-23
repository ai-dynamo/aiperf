# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial can_load auto-detection tests for WekaTraceLoader."""

from pathlib import Path

import orjson

from aiperf.dataset.loader.weka_trace import WekaTraceLoader

_VALID = {
    "id": "t1",
    "models": ["m"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [{"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1}],
}


def test_can_load_empty_dict_returns_false(tmp_path: Path):
    """An empty JSON object lacks all required WekaTrace fields."""
    p = tmp_path / "x.json"
    p.write_bytes(orjson.dumps({}))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_non_weka_json_with_type_n_returns_false(tmp_path: Path):
    """Top-level dict with only ``type: "n"`` is not a WekaTrace (missing id/models/etc)."""
    p = tmp_path / "x.json"
    p.write_bytes(orjson.dumps({"type": "n"}))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_nonexistent_directory_returns_false():
    """Nonexistent paths must return False without raising."""
    assert WekaTraceLoader.can_load(filename="/tmp/does_not_exist_xyz_123_abc") is False


def test_can_load_non_json_extension_with_valid_content_returns_false(tmp_path: Path):
    """``_probe_file`` requires a ``.json`` suffix even for otherwise-valid content."""
    p = tmp_path / "x.txt"
    p.write_bytes(orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_ndjson_returns_false(tmp_path: Path):
    """NDJSON (two concatenated JSON objects) is not a single JSON document."""
    p = tmp_path / "x.json"
    p.write_bytes(orjson.dumps(_VALID) + b"\n" + orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_js_comment_prefix_returns_false(tmp_path: Path):
    """orjson rejects JS-style ``//`` comments that precede otherwise-valid JSON."""
    p = tmp_path / "x.json"
    p.write_bytes(b"// comment\n" + orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_empty_directory_returns_false(tmp_path: Path):
    """Directories with no ``*.json`` entries must return False."""
    assert WekaTraceLoader.can_load(filename=tmp_path) is False


def test_can_load_directory_with_only_empty_json_files_returns_false(tmp_path: Path):
    """First-glob probe fails when all candidate JSON files are 0-byte."""
    (tmp_path / "a.json").write_bytes(b"")
    (tmp_path / "b.json").write_bytes(b"")
    assert WekaTraceLoader.can_load(filename=tmp_path) is False


def test_can_load_char_device_path_returns_false():
    """``/dev/null`` is neither a regular file nor directory; can_load must not raise."""
    assert WekaTraceLoader.can_load(filename=Path("/dev/null")) is False


def test_can_load_directory_first_json_alphabetically_is_mooncake_returns_false(
    tmp_path: Path,
):
    """Documents the single-probe mis-route gap."""
    (tmp_path / "a_mooncake.json").write_bytes(
        orjson.dumps({"timestamp": 0, "input_length": 10, "output_length": 5})
    )
    (tmp_path / "b_weka.txt").write_bytes(orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=tmp_path) is False
