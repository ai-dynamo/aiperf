# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial coverage for RawPayloadDatasetLoader.

Exercises can_load boundary inputs (non-dict data, malformed JSONL, directory
discrimination rules) and load_dataset line-parsing edge cases that the
shipped unit tests do not cover. Task 1 of the raw-payload adversarial pass.
"""

import sys
from unittest.mock import MagicMock

import orjson
import pytest
from pytest import param

from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader


def _make_loader(filename):
    """Construct a loader bypassing __init__ to avoid UserConfig wiring."""
    loader = RawPayloadDatasetLoader.__new__(RawPayloadDatasetLoader)
    loader.filename = str(filename)
    loader.session_id_generator = MagicMock()
    loader.session_id_generator.next.side_effect = [f"s{i}" for i in range(100)]
    loader.info = MagicMock()
    loader.debug = MagicMock()
    return loader


class TestCanLoadDataShape:
    @pytest.mark.parametrize(
        "bad_data",
        [param([]), param("string"), param(123)],
    )  # fmt: skip
    def test_can_load_non_dict_data_returns_false(self, bad_data):
        """can_load guards against non-dict inputs and returns False cleanly.
        Auto-detection plugins feed arbitrary first-record shapes here; prior
        to the defensive guard, non-dict data raised AttributeError at
        ``data.get()`` and broke the detection chain mid-walk.
        """
        assert RawPayloadDatasetLoader.can_load(data=bad_data) is False

    def test_can_load_data_dict_without_messages_key_returns_false(self):
        assert RawPayloadDatasetLoader.can_load(data={"not_messages": []}) is False

    def test_can_load_data_dict_messages_not_a_list_returns_false(self):
        assert (
            RawPayloadDatasetLoader.can_load(data={"messages": "not-a-list"}) is False
        )

    def test_can_load_data_dict_with_conversation_id_returns_false(self):
        """Agentic trajectory records must be rejected even with messages."""
        assert (
            RawPayloadDatasetLoader.can_load(
                data={"messages": [], "conversation_id": "x"}
            )
            is False
        )

    def test_can_load_data_dict_with_top_level_data_list_returns_false(self):
        """InputsFile shape (top-level data=list) must not match raw-payload."""
        assert (
            RawPayloadDatasetLoader.can_load(data={"messages": [], "data": []}) is False
        )


class TestCanLoadDirectoryPeek:
    def test_can_load_file_with_zero_byte_jsonl_returns_false(self, tmp_path):
        d = tmp_path / "empty_line"
        d.mkdir()
        (d / "empty.jsonl").write_bytes(b"")
        assert RawPayloadDatasetLoader.can_load(filename=d) is False

    def test_can_load_file_with_null_first_line_returns_false(self, tmp_path):
        d = tmp_path / "null_line"
        d.mkdir()
        (d / "null.jsonl").write_bytes(b"null\n")
        assert RawPayloadDatasetLoader.can_load(filename=d) is False

    def test_can_load_file_with_json_array_first_line_returns_false(self, tmp_path):
        d = tmp_path / "array_line"
        d.mkdir()
        (d / "arr.jsonl").write_bytes(b"[1,2,3]\n")
        assert RawPayloadDatasetLoader.can_load(filename=d) is False

    def test_can_load_directory_with_non_jsonl_extension_returns_false(self, tmp_path):
        """Directory with only .json (not .jsonl) files must not match."""
        d = tmp_path / "wrong_ext"
        d.mkdir()
        (d / "payload.json").write_bytes(
            orjson.dumps({"messages": [{"role": "user", "content": "x"}]})
        )
        assert RawPayloadDatasetLoader.can_load(filename=d) is False

    def test_can_load_directory_with_first_jsonl_malformed_returns_false_currently(
        self, tmp_path
    ):
        """Documents _dir_has_raw_payload_jsonl silent-swallow behavior.

        The helper catches bare Exception on orjson parse errors and continues
        to the next file. BUT: the happy-path `return` inside the try-block
        only fires when parsing succeeds. On a malformed first file, control
        falls through to the next file. Here the malformed file is the only
        file, so the final `return False` fires.

        Candidate for Wave 2: narrow the except to orjson.JSONDecodeError so
        downstream fall-through is explicit rather than swallowing everything.
        """
        d = tmp_path / "malformed_only"
        d.mkdir()
        (d / "bad.jsonl").write_bytes(b"{not valid json\n")
        assert RawPayloadDatasetLoader.can_load(filename=d) is False


class TestLoadDatasetLineEdgeCases:
    def test_load_dataset_directory_unsorted_multiple_files_all_emitted(self, tmp_path):
        """Three single-turn sessions in a dir must all be emitted (sorted)."""
        d = tmp_path / "sessions"
        d.mkdir()
        # Write in non-alphabetical creation order to exercise sorted(glob).
        for name in ("zulu", "alpha", "mike"):
            (d / f"{name}.jsonl").write_bytes(
                orjson.dumps({"messages": [{"role": "user", "content": name}]}) + b"\n"
            )
        loader = _make_loader(d)
        data = loader.load_dataset()
        assert len(data) == 3
        for payloads in data.values():
            assert len(payloads) == 1
            assert "messages" in payloads[0].payload

    def test_load_dataset_single_file_with_multiple_lines_emits_one_conversation_per_line(
        self, tmp_path
    ):
        """In single-file mode, each line becomes its own single-turn session."""
        p = tmp_path / "three.jsonl"
        lines = [
            orjson.dumps({"messages": [{"role": "user", "content": f"L{i}"}]})
            for i in range(3)
        ]
        p.write_bytes(b"\n".join(lines) + b"\n")
        loader = _make_loader(p)
        data = loader.load_dataset()
        assert len(data) == 3
        for payloads in data.values():
            assert len(payloads) == 1

    def test_load_dataset_file_with_blank_line_in_middle_skips_blank(self, tmp_path):
        """Blank (whitespace-only) lines in the middle of a JSONL file are skipped."""
        p = tmp_path / "blanks.jsonl"
        first = orjson.dumps({"messages": [{"role": "user", "content": "a"}]})
        second = orjson.dumps({"messages": [{"role": "user", "content": "b"}]})
        p.write_bytes(first + b"\n\n" + second + b"\n")
        loader = _make_loader(p)
        data = loader.load_dataset()
        # Exactly two sessions: the blank line must not produce a phantom entry.
        assert len(data) == 2
        contents = sorted(
            payloads[0].payload["messages"][0]["content"] for payloads in data.values()
        )
        assert contents == ["a", "b"]

    def test_load_dataset_file_with_trailing_newline_parses_clean(self, tmp_path):
        """Trailing \\n must not create a phantom empty conversation."""
        p = tmp_path / "trail.jsonl"
        p.write_bytes(
            orjson.dumps({"messages": [{"role": "user", "content": "only"}]}) + b"\n"
        )
        loader = _make_loader(p)
        data = loader.load_dataset()
        assert len(data) == 1

    def test_convert_to_conversations_turn_carries_raw_payload_verbatim(self, tmp_path):
        """Turn.raw_payload must preserve the entire source dict, including
        caller-supplied extra keys beyond the standard chat schema.
        """
        p = tmp_path / "extras.jsonl"
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "custom_field": "verbatim",
            "nested": {"temperature": 0.7, "seed": 42},
        }
        p.write_bytes(orjson.dumps(payload) + b"\n")
        loader = _make_loader(p)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        turn = conversations[0].turns[0]
        assert turn.raw_payload == payload
        assert turn.raw_payload["custom_field"] == "verbatim"
        assert turn.raw_payload["nested"]["seed"] == 42


class TestWave2BugCandidates:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows uses ACLs not POSIX permission bits; os.chmod(0o000) does not block reads",
    )
    def test_can_load_directory_with_unreadable_jsonl_raises_post_fix(self, tmp_path):
        """Unreadable .jsonl file in a directory.

        Post Wave-2 fix: _dir_has_raw_payload_jsonl narrows its except to
        orjson.JSONDecodeError, letting PermissionError/OSError propagate so
        misconfigured inputs fail loudly rather than silently miscategorize.

        Today: the broad `except Exception: continue` swallows the OSError
        and can_load returns False. The xfail-strict marker flips as soon as
        Wave 2 ships, alerting us to remove the xfail.
        """
        import os

        d = tmp_path / "locked"
        d.mkdir()
        p = d / "blocked.jsonl"
        p.write_bytes(
            orjson.dumps({"messages": [{"role": "user", "content": "x"}]}) + b"\n"
        )
        os.chmod(p, 0o000)
        try:
            with pytest.raises((PermissionError, OSError)):
                RawPayloadDatasetLoader.can_load(filename=d)
        finally:
            os.chmod(p, 0o644)
