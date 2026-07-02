# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the turn-messages viewer: interning, zstd round-trip, fzstd safety.

Focus: every turn re-sends accumulated history, so unique ``(role, content)``
messages must be interned into a shared ``msgs`` table referenced by integer
``ids``; and the embedded payload must decode under a fzstd-safe ``windowLog``
window (the inlined decoder silently corrupts windows > 26).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import orjson
import pytest
import zstandard

from aiperf.analysis.turn_messages import (
    ZSTD_WINDOW_LOG,
    TurnMessagesError,
    _flatten_record,
    build_payload,
    write_turn_messages_html,
)
from aiperf.cli_commands.turn_messages import turn_messages as cli_turn_messages

NS = 1_000_000_000


def _rec(
    conv: str,
    turn: int,
    messages: list[dict],
    *,
    start_s: float = 0.0,
    depth: int = 0,
    x_corr: str | None = None,
    parent_corr: str | None = None,
    model: str = "test-model",
) -> dict:
    """Build a minimal --export-level raw record (one turn's request)."""
    return {
        "metadata": {
            "conversation_id": conv,
            "turn_index": turn,
            "request_start_ns": int(start_s * NS),
            "agent_depth": depth,
            "x_correlation_id": x_corr or f"{conv}#0",
            "parent_correlation_id": parent_corr,
            "benchmark_phase": "profiling",
        },
        "payload": {"model": model, "messages": messages},
    }


def _accumulated(conv: str, n_turns: int) -> list[dict]:
    """One conversation of ``n_turns`` turns, each re-sending full history."""
    history: list[dict] = [{"role": "system", "content": "you are a test agent"}]
    recs = []
    for t in range(n_turns):
        history = history + [{"role": "user", "content": f"question {t}"}]
        recs.append(_rec(conv, t, list(history), start_s=float(t)))
        history = history + [{"role": "assistant", "content": f"answer {t}"}]
    return recs


def _extract_payload(html_path: Path) -> dict:
    """Mirror the viewer's client-side decode: base64 -> zstd -> JSON.

    Caps the decoder window at ``ZSTD_WINDOW_LOG`` so this also asserts the
    frame is within the fzstd-safe window (a larger window would raise here).
    """
    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html_path.read_text())
    assert match, "embedded payload not found"
    dctx = zstandard.ZstdDecompressor(max_window_size=1 << ZSTD_WINDOW_LOG)
    return orjson.loads(dctx.decompress(base64.b64decode(match.group(1))))


def _write_raw(run_dir: Path, recs: list[dict]) -> None:
    (run_dir / "profile_export_raw.jsonl").write_bytes(
        b"\n".join(orjson.dumps(r) for r in recs)
    )


class TestBuildPayload:
    def test_interns_repeated_messages(self):
        # 4 turns re-sending history: 1 system + 4 user + 3 assistant = 8 unique,
        # but the raw message refs are far more (system+user repeat every turn).
        recs = [_flatten_record(r) for r in _accumulated("c1", 4)]
        total_refs = sum(len(r["messages"]) for r in recs)
        payload = build_payload(recs, 1, cap=8000, limit=40, max_turns=60)
        assert len(payload["msgs"]) == 8
        assert total_refs > len(payload["msgs"])  # interning actually dedups
        # every turn references the shared table by id, never inlines content
        turns = payload["convs"][0]["turns"]
        for turn in turns:
            assert "ids" in turn and "msgs" not in turn
            assert all(0 <= i < len(payload["msgs"]) for i in turn["ids"])

    def test_content_cap_truncates_and_records_remainder(self):
        big = "x" * 5000
        recs = [_flatten_record(_rec("c1", 0, [{"role": "user", "content": big}]))]
        payload = build_payload(recs, 1, cap=100, limit=40, max_turns=60)
        (msg,) = payload["msgs"]
        assert len(msg["body"]) == 100
        assert msg["len"] == 5000
        assert msg["trunc"] == 4900

    def test_limit_and_max_turns_summarize_overflow(self):
        recs = [
            _flatten_record(r) for r in _accumulated("c1", 5) + _accumulated("c2", 5)
        ]
        payload = build_payload(recs, 1, cap=8000, limit=1, max_turns=3)
        assert "2 conversations (1 shown)" in payload["stat"]
        conv = payload["convs"][0]
        assert conv["nt"] == 5
        assert conv["moreTurns"] == 2
        assert len(conv["turns"]) == 3

    def test_negative_args_are_clamped_to_zero(self):
        recs = [_flatten_record(_rec("c1", 0, [{"role": "user", "content": "abcdef"}]))]
        payload = build_payload(recs, 1, cap=-10, limit=40, max_turns=60)
        (msg,) = payload["msgs"]
        assert msg["body"] == "" and msg["trunc"] == 6  # cap clamped to 0
        # negative limit clamps to 0 -> nothing shown, no crash
        assert build_payload(recs, 1, cap=8000, limit=-1, max_turns=-1)["convs"] == []


class TestWriteHtml:
    def test_end_to_end_payload_decodes_within_safe_window(self, tmp_path: Path):
        _write_raw(tmp_path, _accumulated("c1", 6))
        out = write_turn_messages_html(tmp_path)
        assert out == tmp_path / "turn_messages.html"

        payload = _extract_payload(out)  # also asserts windowLog <= 26
        assert len(payload["convs"]) == 1
        assert payload["msgs"]  # interned table present
        ids = payload["convs"][0]["turns"][0]["ids"]
        assert all(0 <= i < len(payload["msgs"]) for i in ids)

    def test_self_contained_inline_decoder_no_decompressionstream(self, tmp_path: Path):
        _write_raw(tmp_path, _accumulated("c1", 3))
        html = write_turn_messages_html(tmp_path).read_text()
        assert ".fzstd=f()" in html  # decoder inlined as a global
        assert "fzstd.decompress(" in html
        assert "DecompressionStream" not in html  # no native gzip dependency
        assert "__FZSTD_JS__" not in html and "__PAYLOAD_B64__" not in html

    def test_reads_raw_records_shard_folder(self, tmp_path: Path):
        shard_dir = tmp_path / "raw_records"
        shard_dir.mkdir()
        (shard_dir / "shard0.jsonl").write_bytes(
            b"\n".join(orjson.dumps(r) for r in _accumulated("c1", 2))
        )
        out = write_turn_messages_html(tmp_path)
        assert _extract_payload(out)["convs"][0]["id"] == "c1"

    def test_missing_raw_records_raises(self, tmp_path: Path):
        with pytest.raises(TurnMessagesError, match="no raw records"):
            write_turn_messages_html(tmp_path)

    def test_explicit_out_path_respected(self, tmp_path: Path):
        _write_raw(tmp_path, _accumulated("c1", 2))
        dest = tmp_path / "nested" / "viewer.html"
        dest.parent.mkdir()
        assert write_turn_messages_html(tmp_path, out=dest) == dest
        assert dest.is_file()


def _write_lines(run_dir: Path, lines: list[bytes]) -> None:
    """Write raw (possibly malformed) jsonl lines verbatim."""
    (run_dir / "profile_export_raw.jsonl").write_bytes(b"\n".join(lines))


class TestAdversarial:
    def test_injected_html_in_content_cannot_escape_payload_blob(self, tmp_path: Path):
        # Message content is attacker-controlled; it must end up inside the
        # base64 zstd blob and be rendered via textContent, never as live markup.
        evil = '</script><script>alert(String.fromCharCode(88,83,83))</script>"\\'
        _write_raw(tmp_path, [_rec("c1", 0, [{"role": "user", "content": evil}])])
        out = write_turn_messages_html(tmp_path)
        html = out.read_text()
        # the raw injection never appears outside the (base64) payload literal
        assert evil not in html
        assert "alert(String.fromCharCode" not in html
        # ...and it round-trips byte-for-byte through the payload
        assert _extract_payload(out)["msgs"][0]["body"] == evil

    def test_template_placeholder_strings_in_content_roundtrip(self, tmp_path: Path):
        # Content equal to the template placeholders must not corrupt the file
        # (base64 cannot contain '_', so the substitution can't be fooled).
        for i, marker in enumerate(("__PAYLOAD_B64__", "__FZSTD_JS__")):
            d = tmp_path / f"run{i}"
            d.mkdir()
            _write_raw(d, [_rec("c1", 0, [{"role": "user", "content": marker}])])
            out = write_turn_messages_html(d)
            html = out.read_text()
            assert "__PAYLOAD_B64__" not in html and "__FZSTD_JS__" not in html
            assert _extract_payload(out)["msgs"][0]["body"] == marker

    def test_injected_conversation_id_cannot_escape(self, tmp_path: Path):
        cid = "<img src=x onerror=alert(1)>::sa:7"
        _write_raw(tmp_path, [_rec(cid, 0, [{"role": "user", "content": "hi"}])])
        out = write_turn_messages_html(tmp_path)
        assert cid not in out.read_text()
        assert _extract_payload(out)["convs"][0]["id"] == cid

    def test_non_object_json_lines_are_skipped(self, tmp_path: Path):
        # Valid JSON that is not an object (int/str/array/null) must not crash.
        good = orjson.dumps(_rec("c1", 0, [{"role": "user", "content": "ok"}]))
        _write_lines(
            tmp_path,
            [
                b"not json{",
                b"",
                b"   ",
                b"123",
                b'"a string"',
                b"[1,2,3]",
                b"null",
                good,
            ],
        )
        payload = _extract_payload(write_turn_messages_html(tmp_path))
        assert len(payload["convs"]) == 1
        assert payload["msgs"][0]["body"] == "ok"

    def test_messages_not_a_list_is_tolerated(self, tmp_path: Path):
        # payload.messages as a non-list (str/dict/int) must not be iterated raw.
        recs = [
            _rec("c1", 0, "i am a string not a list"),
            _rec("c2", 0, {"role": "user", "content": "a dict not a list"}),
            _rec("c3", 0, 42),
            _rec("c4", 0, [{"role": "user", "content": "real"}]),
        ]
        _write_raw(tmp_path, recs)
        payload = _extract_payload(write_turn_messages_html(tmp_path))
        assert {m["body"] for m in payload["msgs"]} == {"real"}
        assert len(payload["convs"]) == 4

    def test_non_dict_message_items_are_tolerated(self, tmp_path: Path):
        msgs = ["bare string", 42, None, {"role": "user", "content": "real"}]
        _write_raw(tmp_path, [_rec("c1", 0, msgs)])
        payload = _extract_payload(write_turn_messages_html(tmp_path))
        bodies = {m["body"] for m in payload["msgs"]}
        assert "real" in bodies and "bare string" in bodies and "42" in bodies

    def test_missing_metadata_and_payload_fields(self, tmp_path: Path):
        _write_lines(
            tmp_path,
            [
                orjson.dumps({}),
                orjson.dumps({"metadata": {}, "payload": {}}),
                orjson.dumps({"metadata": None, "payload": None}),
                orjson.dumps(
                    {"payload": {"messages": [{"role": "user", "content": "hi"}]}}
                ),
                orjson.dumps(_rec("c1", 0, [{"role": "user", "content": "ok"}])),
            ],
        )
        out = write_turn_messages_html(tmp_path)
        assert out.is_file()
        assert "ok" in {m["body"] for m in _extract_payload(out)["msgs"]}

    def test_content_type_variants(self, tmp_path: Path):
        msgs = [
            {"role": "system", "content": None},
            {"role": "user", "content": 42},
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}],
            },
            {"role": "user", "content": [{"type": "image"}]},
        ]
        _write_raw(tmp_path, [_rec("c1", 0, msgs)])
        bodies = {
            m["body"]
            for m in _extract_payload(write_turn_messages_html(tmp_path))["msgs"]
        }
        assert "" in bodies
        assert "42" in bodies
        assert "hello\n<image_url>" in bodies
        assert "<image>" in bodies

    def test_unicode_and_zero_content_cap(self, tmp_path: Path):
        body = "héllo 世界 🤖\n\ttabs\x00nul"
        _write_raw(tmp_path, [_rec("c1", 0, [{"role": "user", "content": body}])])
        out = write_turn_messages_html(tmp_path, content_cap=0)
        msg = _extract_payload(out)["msgs"][0]
        assert msg["body"] == ""
        assert msg["len"] == len(body)
        assert msg["trunc"] == len(body)

    def test_all_invalid_records_raises(self, tmp_path: Path):
        _write_lines(tmp_path, [b"garbage", b"{bad json", b"42", b"null"])
        with pytest.raises(TurnMessagesError, match="no valid raw records"):
            write_turn_messages_html(tmp_path)

    def test_replayed_conversation_splits_into_lanes(self, tmp_path: Path):
        recs = [
            _rec(
                "c1", 0, [{"role": "user", "content": "a"}], x_corr="lane-A", start_s=0
            ),
            _rec(
                "c1", 0, [{"role": "user", "content": "b"}], x_corr="lane-B", start_s=1
            ),
        ]
        _write_raw(tmp_path, recs)
        payload = _extract_payload(write_turn_messages_html(tmp_path))
        assert len(payload["convs"]) == 2
        tags = {c["replayTag"] for c in payload["convs"]}
        assert "" in tags and any("replay" in t for t in tags)


class TestCli:
    def test_out_with_multiple_run_dirs_exits_2(self, tmp_path: Path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        _write_raw(a, _accumulated("c1", 1))
        _write_raw(b, _accumulated("c2", 1))
        with pytest.raises(SystemExit) as e:
            cli_turn_messages([a, b], out=tmp_path / "x.html")
        assert e.value.code == 2

    def test_all_invalid_run_dirs_exit_1(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit) as e:
            cli_turn_messages([tmp_path / "missing1", tmp_path / "missing2"])
        assert e.value.code == 1
        assert capsys.readouterr().err.count("skip") == 2

    def test_partial_failure_exits_zero_and_writes_good(self, tmp_path: Path):
        good = tmp_path / "good"
        good.mkdir()
        _write_raw(good, _accumulated("c1", 1))
        # one good + one missing -> not all failed -> returns normally (no exit)
        cli_turn_messages([good, tmp_path / "missing"])
        assert (good / "turn_messages.html").is_file()

    def test_accepts_direct_jsonl_file_writes_beside_it(self, tmp_path: Path):
        _write_raw(tmp_path, _accumulated("c1", 2))
        cli_turn_messages([tmp_path / "profile_export_raw.jsonl"])
        assert (tmp_path / "turn_messages.html").is_file()

    def test_unwritable_out_is_reported_not_traceback(self, tmp_path: Path, capsys):
        good = tmp_path / "good"
        good.mkdir()
        _write_raw(good, _accumulated("c1", 1))
        with pytest.raises(SystemExit) as e:
            cli_turn_messages([good], out=tmp_path / "missing_dir" / "x.html")
        assert e.value.code == 1
        assert "could not write" in capsys.readouterr().err
