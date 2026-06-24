# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial I/O and filesystem tests for WekaTraceLoader."""

import gzip
import os
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader

_VALID = {
    "id": "t1",
    "models": ["m"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [{"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1}],
}


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "t"
    uc.endpoint.model_names = ["m"]
    return uc


# ---------------------------------------------------------------------------
# File-content attacks
# ---------------------------------------------------------------------------


def test_can_load_zero_byte_file_returns_false(tmp_path: Path):
    """A zero-byte .json file isn't valid JSON; can_load must swallow the
    decode error and return False rather than raise."""
    p = tmp_path / "empty.json"
    p.write_bytes(b"")
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_json_null_returns_false(tmp_path: Path):
    """`null` parses as JSON but isn't a dict; can_load must reject non-dict
    top-level values."""
    p = tmp_path / "null.json"
    p.write_bytes(b"null")
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_json_array_returns_false(tmp_path: Path):
    """A top-level array parses as JSON but Weka traces are dicts; reject."""
    p = tmp_path / "arr.json"
    p.write_bytes(b"[]")
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_json_trailing_garbage_returns_false(tmp_path: Path):
    """orjson rejects trailing garbage after a valid object; can_load must
    return False, not raise."""
    p = tmp_path / "garbage.json"
    p.write_bytes(b'{"id":"t"} garbage')
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_json_with_bom_returns_false(tmp_path: Path):
    """UTF-8 BOM prefixes are not stripped by orjson; a BOM-prefixed valid
    trace must be rejected rather than parsed."""
    p = tmp_path / "bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_utf16_encoded_file_returns_false(tmp_path: Path):
    """UTF-16 (with BOM) encoded bytes are not valid UTF-8 JSON; reject."""
    p = tmp_path / "utf16.json"
    p.write_bytes(orjson.dumps(_VALID).decode("utf-8").encode("utf-16"))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_gzipped_bytes_with_json_extension_returns_false(tmp_path: Path):
    """Gzipped payload masquerading as .json: orjson can't decode raw gzip
    bytes, so can_load must return False."""
    p = tmp_path / "gz.json"
    p.write_bytes(gzip.compress(orjson.dumps(_VALID)))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_concatenated_json_objects_returns_false(tmp_path: Path):
    """NDJSON / concatenated objects aren't valid single JSON documents;
    orjson rejects them and can_load must return False."""
    p = tmp_path / "cat.json"
    p.write_bytes(orjson.dumps(_VALID) + orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=p) is False


# ---------------------------------------------------------------------------
# Filesystem attacks
# ---------------------------------------------------------------------------


def test_can_load_uppercase_json_extension_rejected(tmp_path: Path):
    """`_probe_file` checks `path.suffix != '.json'` case-sensitively, so
    `trace.JSON` must be rejected even if its contents would validate."""
    p = tmp_path / "trace.JSON"
    p.write_bytes(orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_nonexistent_path_returns_false():
    """A path that doesn't exist is not a file or dir; can_load returns
    False without raising."""
    assert WekaTraceLoader.can_load(filename="/does/not/exist_xyz.json") is False


def test_can_load_path_that_is_neither_file_nor_dir_returns_false():
    """/dev/null is a character device - neither a regular file nor a dir;
    can_load must return False."""
    assert WekaTraceLoader.can_load(filename="/dev/null") is False


def test_can_load_broken_symlink_returns_false(tmp_path: Path):
    """A dangling symlink resolves to a missing target; can_load returns
    False rather than raising."""
    link = tmp_path / "link.json"
    os.symlink(tmp_path / "missing.json", link)
    assert WekaTraceLoader.can_load(filename=link) is False


# ---------------------------------------------------------------------------
# Directory-mode attacks
# ---------------------------------------------------------------------------


def test_can_load_directory_single_probe_invalid_returns_false(tmp_path: Path):
    """Directory detection is single-probe (``next(sorted(glob(...)))``), not an
    exhaustive scan. A directory whose alphabetically-first JSON fails
    validation returns False even if other valid files exist — this
    documents the O(1) probe contract."""
    # After the sorted-glob fix, "a_bad.json" is deterministically probed
    # before "b_good.json" on all filesystems.
    (tmp_path / "a_bad.json").write_bytes(b"{}")
    (tmp_path / "b_good.json").write_bytes(orjson.dumps(_VALID))
    assert WekaTraceLoader.can_load(filename=tmp_path) is False


def test_can_load_directory_single_probe_valid_first_returns_true(tmp_path: Path):
    """Inverse of single_probe_invalid: alphabetically-first is valid → True
    even if later files are invalid. Determinism depends on the sorted-glob
    fix in can_load."""
    (tmp_path / "a_good.json").write_bytes(orjson.dumps(_VALID))
    (tmp_path / "b_bad.json").write_bytes(b"{}")
    assert WekaTraceLoader.can_load(filename=tmp_path) is True


def test_load_dataset_duplicate_id_across_files_raises(tmp_path: Path):
    """Two files with the same trace id in one directory must raise -
    trace ids form the dict key and silent overwrite would lose data."""
    (tmp_path / "a.json").write_bytes(orjson.dumps(_VALID))
    (tmp_path / "b.json").write_bytes(orjson.dumps(_VALID))
    loader = WekaTraceLoader(filename=str(tmp_path), user_config=_mk_user_config())
    with pytest.raises(ValueError, match="Duplicate trace id 't1'"):
        loader.load_dataset()


def test_load_dataset_ignores_non_json_siblings(tmp_path: Path):
    """Directory enumeration uses `*.json` glob, so sibling README/txt files
    are ignored and load_dataset returns only the valid trace."""
    (tmp_path / "trace.json").write_bytes(orjson.dumps(_VALID))
    (tmp_path / "readme.txt").write_bytes(b"hello")
    loader = WekaTraceLoader(filename=str(tmp_path), user_config=_mk_user_config())
    data = loader.load_dataset()
    assert set(data.keys()) == {"t1"}


def test_load_dataset_does_not_recurse_into_subdirs(tmp_path: Path):
    """`*.json` glob is non-recursive; JSON files in subdirectories must be
    ignored so nested fixtures can't smuggle extra traces."""
    (tmp_path / "a.json").write_bytes(orjson.dumps(_VALID))
    sub = tmp_path / "sub"
    sub.mkdir()
    other = dict(_VALID)
    other["id"] = "t2"
    (sub / "b.json").write_bytes(orjson.dumps(other))
    loader = WekaTraceLoader(filename=str(tmp_path), user_config=_mk_user_config())
    data = loader.load_dataset()
    assert set(data.keys()) == {"t1"}
