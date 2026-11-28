# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for structural multimodal ``dag_jsonl`` turns."""

from __future__ import annotations

import base64
import hashlib
import io
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import orjson
import pytest
from PIL import Image as PILImage
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.loader import dag_jsonl_media
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.dataset.loader.dag_jsonl_media import (
    DagMultimodalMaterializer,
    _encode_mp4,
    default_media_uuid,
    encode_deterministic_mp4,
    find_ffmpeg_binary,
    iter_deterministic_rgb_frames,
    synth_deterministic_png_bytes,
    verify_mp4_decodable,
)
from aiperf.dataset.loader.dag_jsonl_models import DagMediaSegment, DagTurn

IMG = {"media_hash": "shared_img", "kind": "image", "width": 64, "height": 48}
VID = {
    "media_hash": "shared_vid",
    "kind": "video",
    "width": 32,
    "height": 16,
    "fps": 0.5,
    "num_frames": 3,
}
IMG_VT = {"media_hash": "vt_img", "kind": "image", "vis_tokens": 576}
VID_VT = {"media_hash": "vt_vid", "kind": "video", "vis_tokens": 1920, "num_frames": 16}

# Stand-in ffmpeg: "encodes" by counting streamed RGB bytes into a tiny text
# file, "decodes" that file back into a YUV4MPEG2 stream, and floods stderr on
# both paths to prove the pipes are drained.
FAKE_FFMPEG = r"""
import os, pathlib, sys
args = sys.argv[1:]
if "yuv4mpegpipe" in args:
    w, h, n, rate = pathlib.Path(args[args.index("-i") + 1]).read_text().split()
    sys.stderr.buffer.write(b"W" * 300_000 + b"\ndecode_tail_marker\n")
    out = sys.stdout.buffer
    out.write(f"YUV4MPEG2 W{w} H{h} F{round(float(rate) * 100)}:100 Ip C444\n".encode())
    for _ in range(int(n)):
        out.write(b"FRAME\n" + b"\x80" * (int(w) * int(h) * 3))
    sys.exit(int(os.environ.get("FAKE_FFMPEG_DECODE_RC", "0")))
w, h = args[args.index("-s") + 1].split("x")
total = 0
while chunk := sys.stdin.buffer.read(65536):
    total += len(chunk)
    sys.stderr.buffer.write(b"E" * 131072)
codec = args[args.index("-c:v") + 1]
if codec in os.environ.get("FAKE_FFMPEG_FAIL_CODECS", "").split(","):
    sys.stderr.buffer.write(f"\n{codec} unavailable\n".encode())
    sys.exit(1)
frames = total // (int(w) * int(h) * 3)
pathlib.Path(args[-1]).write_text(f"{w} {h} {frames} {args[args.index('-r') + 1]}")
"""


@pytest.fixture
def fake_ffmpeg(tmp_path: Path) -> Iterator[tuple[str, list[list[str]]]]:
    """Patch ``subprocess.Popen`` so ``fake_ffmpeg`` runs under this interpreter."""
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(FAKE_FFMPEG, encoding="utf-8")
    calls: list[list[str]] = []
    real_popen = subprocess.Popen

    def _popen(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        calls.append(list(cmd))
        if cmd[0] == str(script):
            cmd = [sys.executable, *cmd]
        return real_popen(cmd, *args, **kwargs)

    with patch("subprocess.Popen", side_effect=_popen):
        yield str(script), calls


def _write_dag(tmp_path: Path, rows: list[dict]) -> Path:
    out = tmp_path / "mm.dag.jsonl"
    out.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")
    return out


def _two_tokens_per_long_word(text: str) -> list[int]:
    return [0 for word in text.split() for _ in range(1 if len(word) == 1 else 2)]


class _Tokenizer:
    """Tokenizer stub: 2 tokens per multi-char word, 1 per single char."""

    _encode_args: dict[str, Any] = {}

    def encode(self, text: str) -> list[int]:
        return _two_tokens_per_long_word(text)


def _media_span(media_hash: str, idx: int | None = None) -> dict:
    return {"kind": "media", "span_id": media_hash, "media_index": idx}


class TestModels:
    def test_video_geometry_and_image_frames_validated(self) -> None:
        with pytest.raises(ValidationError, match="even dimensions"):
            DagMediaSegment.model_validate({**VID, "width": 33})
        with pytest.raises(ValidationError, match="fps >= 0.01"):
            DagMediaSegment.model_validate({**VID, "fps": 0.004})
        with pytest.raises(ValidationError, match="num_frames=1"):
            DagMediaSegment.model_validate({**IMG, "num_frames": 2})
        with pytest.raises(ValidationError, match="not a boolean"):
            DagMediaSegment.model_validate({**IMG, "width": True})

    def test_dropped_framing_tokens_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            DagMediaSegment.model_validate({**IMG, "framing_tokens": 5})

    @pytest.mark.parametrize(
        "segment, expected",
        [
            param(IMG_VT, (1024, 576), id="qwen3-image"),
            param(VID_VT, (640, 384), id="qwen3-video"),
            param({**VID_VT, "vision_model": "qwen2_5_vl"}, (560, 336), id="qwen25-video"),
            param({**IMG_VT, "width": 1024, "height": 576}, (1024, 576), id="consistent-explicit"),
        ],
    )  # fmt: skip
    def test_vis_tokens_resolve_geometry(
        self, segment: dict, expected: tuple[int, int]
    ) -> None:
        seg = DagMediaSegment.model_validate(segment)
        assert (seg.width, seg.height) == expected

    @pytest.mark.parametrize(
        "segment, match",
        [
            param({"media_hash": "x", "kind": "image"}, "requires width and height", id="no-geometry"),
            param({**IMG, "height": None}, "requires width and height", id="width-only"),
            param({**IMG_VT, "width": 1024}, "both width and height or neither", id="vt-width-only"),
            param({**IMG, "vision_model": "qwen2_5_vl"}, "without vis_tokens", id="model-only"),
            param({**IMG_VT, "width": 1280, "height": 720}, "encodes to 880", id="mismatch"),
            param({**IMG_VT, "vis_tokens": 577}, "unreachable", id="unreachable"),
            param({**VID_VT, "vis_tokens": 1921}, "not divisible by grid_t=8", id="grid-t"),
            param({**IMG_VT, "vis_tokens": True}, "not a boolean", id="bool"),
        ],
    )  # fmt: skip
    def test_vis_tokens_validation_rejects(self, segment: dict, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            DagMediaSegment.model_validate(segment)

    def test_span_tokens_required_for_text_and_informational_for_media(self) -> None:
        with pytest.raises(ValidationError, match="requires 'tokens'"):
            DagTurn.model_validate({"prompt_spans": [{"kind": "text", "span_id": "t"}]})
        turn = DagTurn.model_validate(
            {
                "media_segments": [IMG],
                "prompt_spans": [
                    {**_media_span("shared_img"), "tokens": 578},
                    _media_span("shared_img", 0),
                ],
            }
        )
        assert [s.tokens for s in turn.prompt_spans] == [578, None]

    @pytest.mark.parametrize(
        "turn, match",
        [
            param({"messages": [{"role": "user", "content": "x"}], "in_text_tokens": 4},
                  "mutually exclusive", id="messages-and-in-text"),
            param({"media_segments": [IMG], "messages": [{"role": "user", "content": "x"}]},
                  "no .* content part references them", id="unreferenced-segments"),
            param({"media_segments": [IMG], "messages": [{"role": "assistant", "content": [
                  {"type": "media_ref", "media_index": 0}]}]}, "role='user'", id="assistant-media-ref"),
            param({"media_segments": [IMG], "messages": [{"role": "user", "content": [
                  {"type": "media_ref", "media_hash": "missing"}]}]}, "not found", id="unknown-hash"),
            param({"media_segments": [IMG], "prompt_spans": [
                  {**_media_span("shared_img"), "role": "system"}]}, "role='user'", id="system-media-span"),
            param({"prompt_spans": [{"kind": "text", "span_id": "t", "tokens": 1, "role": "tool"}]},
                  "role", id="tool-span"),
            param({"media_segments": [IMG, {**IMG, "width": 128}],
                   "prompt_spans": [_media_span("shared_img")]}, "ambiguous", id="ambiguous-hash"),
            param({"media_segments": [IMG], "prompt_spans": [_media_span("x", 3)]},
                  "does not resolve", id="index-out-of-range"),
        ],
    )  # fmt: skip
    def test_turn_validation_rejects(self, turn: dict, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            DagTurn.model_validate(turn)

    def test_identical_duplicate_segments_are_not_ambiguous(self) -> None:
        DagTurn.model_validate(
            {"media_segments": [IMG, IMG], "prompt_spans": [_media_span("shared_img")]}
        )

    @pytest.mark.parametrize(
        "turn, expected",
        [
            param({"in_text_tokens": 8}, True, id="in-text-tokens"),
            param({"media_segments": [IMG]}, True, id="media-segments"),
            param({"in_text_tokens": 0}, False, id="zero-in-text-tokens"),
            param({"in_text_tokens": True}, False, id="bool-in-text-tokens"),
            param({"text": "hi", "media_segments": [IMG]}, False, id="multi-turn-key"),
            param({"image_uuids": ["u"], "in_text_tokens": 4}, False, id="multi-turn-uuids"),
        ],
    )  # fmt: skip
    def test_can_load_precision(self, turn: dict, expected: bool) -> None:
        record = {"session_id": "s", "turns": [turn]}
        assert DagJsonlLoader.can_load(record) is expected


class TestSynthesis:
    def test_png_is_deterministic_and_exact_size(self) -> None:
        png = synth_deterministic_png_bytes(65, 33, "seed")
        assert png == synth_deterministic_png_bytes(65, 33, "seed")
        assert png != synth_deterministic_png_bytes(65, 33, "other")
        with PILImage.open(io.BytesIO(png)) as img:
            assert img.size == (65, 33) and img.mode == "RGB"

    def test_rgb_frames_stream_deterministically(self) -> None:
        frames = list(iter_deterministic_rgb_frames(64, 48, 3, "seed_stream"))
        assert [f.shape for f in frames] == [(48, 64, 3)] * 3
        assert np.array_equal(
            frames[1], np.roll(np.roll(frames[0], 16, axis=1), 8, axis=0)
        )
        assert [hashlib.sha256(f.tobytes()).hexdigest()[:16] for f in frames] == [
            "c61e0b781c9d761d",
            "61af7104637e7eea",
            "8bf33d334ccf2efd",
        ]
        with pytest.raises(ValueError, match="safety limit"):
            next(iter_deterministic_rgb_frames(16384, 16384, 4096, "huge"))

    def test_tokenizer_calibration_hits_exact_counts(self) -> None:
        mat = DagMultimodalMaterializer(tokenizer=_Tokenizer())
        for target in (1, 2, 5, 6, 97):
            text = mat.render_span_text("span", target)
            assert len(_two_tokens_per_long_word(text)) == target
        assert mat.render_span_text("span", 5) is mat.render_span_text("span", 5)

    def test_tokenizer_calibration_unreachable_budget_raises(self) -> None:
        class _Silent:
            _encode_args: dict[str, Any] = {}

            def encode(self, text: str) -> list[int]:
                return []

        with pytest.raises(ValueError, match="could not calibrate span 's' to 7"):
            DagMultimodalMaterializer(tokenizer=_Silent()).render_span_text("s", 7)

    def test_default_uuid_tracks_full_identity(self) -> None:
        seg = DagMediaSegment.model_validate(VID)
        assert default_media_uuid(seg) == default_media_uuid(seg.model_copy())
        assert default_media_uuid(seg) != default_media_uuid(
            DagMediaSegment.model_validate({**VID, "fps": 0.51})
        )


class TestFfmpegPipeline:
    """Video encode/verify through a real subprocess; runs without ffmpeg installed."""

    def test_encode_streams_frames_and_verifies_roundtrip(self, fake_ffmpeg) -> None:
        script, calls = fake_ffmpeg
        seg = DagMediaSegment.model_validate(VID)
        payload = encode_deterministic_mp4(seg, ffmpeg_bin=script)
        assert payload == b"32 16 3 0.50"
        encode_cmd, decode_cmd = calls
        assert encode_cmd[encode_cmd.index("-c:v") + 1] == "libx264"
        assert "yuv4mpegpipe" in decode_cmd

    def test_codec_fallback_and_stderr_tail_in_errors(
        self, fake_ffmpeg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script, calls = fake_ffmpeg
        seg = DagMediaSegment.model_validate(VID)
        monkeypatch.setenv("FAKE_FFMPEG_FAIL_CODECS", "libx264")
        assert encode_deterministic_mp4(seg, ffmpeg_bin=script) == b"32 16 3 0.50"
        assert [c[c.index("-c:v") + 1] for c in calls if "-c:v" in c] == [
            "libx264",
            "mpeg4",
        ]
        monkeypatch.setenv("FAKE_FFMPEG_FAIL_CODECS", "libx264,mpeg4")
        with pytest.raises(RuntimeError, match="(?s)mpeg4 rc=1: .*mpeg4 unavailable"):
            encode_deterministic_mp4(seg, ffmpeg_bin=script)

    @pytest.mark.parametrize(
        "file_text, decode_rc, match",
        [
            param("32 32 3 0.50", "0", "dimension mismatch.*got 32x32", id="geometry"),
            param("32 16 3 24.00", "0", "frame-rate mismatch", id="fps"),
            param("32 16 2 0.50", "0", "frames=2, expected=3", id="frame-count"),
            param("32 16 3 0.50", "1", "(?s)rc=1.*decode_tail_marker", id="rc-with-tail"),
        ],
    )  # fmt: skip
    def test_verify_rejects_mismatches(
        self,
        fake_ffmpeg,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        file_text: str,
        decode_rc: str,
        match: str,
    ) -> None:
        script, _ = fake_ffmpeg
        clip = tmp_path / "clip.mp4"
        clip.write_text(file_text)
        monkeypatch.setenv("FAKE_FFMPEG_DECODE_RC", decode_rc)
        with pytest.raises(RuntimeError, match=match):
            verify_mp4_decodable(
                clip,
                expected_frames=3,
                expected_width=32,
                expected_height=16,
                expected_fps=0.5,
                ffmpeg_bin=script,
            )

    def test_frame_source_error_kills_and_reaps_encoder(self, tmp_path: Path) -> None:
        class _RunningProc:
            def __init__(self) -> None:
                self.stdin, self.stdout = io.BytesIO(), None
                self.stderr = io.BytesIO(b"tail")
                self.returncode: int | None = None
                self.stderr_closed_at_wait: bool | None = None

            def poll(self) -> int | None:
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

            def wait(self) -> int:
                self.stderr_closed_at_wait = self.stderr.closed
                return self.returncode if self.returncode is not None else 0

        def _failing_frames() -> Iterator[np.ndarray]:
            raise ValueError("exceeds safety limit")
            yield np.zeros((2, 2, 3), dtype=np.uint8)

        proc = _RunningProc()
        with (
            patch("subprocess.Popen", return_value=proc),
            pytest.raises(ValueError, match="exceeds safety limit"),
        ):
            _encode_mp4(
                "ffmpeg", "libx264", _failing_frames(), tmp_path / "o.mp4",
                width=2, height=2, fps=24.0,
            )  # fmt: skip
        assert proc.returncode == -9
        assert proc.stderr_closed_at_wait is False
        assert proc.stdin.closed and proc.stderr.closed

    @pytest.mark.ffmpeg
    def test_real_ffmpeg_h264_roundtrip_is_deterministic(self) -> None:
        ff = find_ffmpeg_binary()
        if ff is None:
            pytest.skip("ffmpeg binary not installed")
        seg = DagMediaSegment.model_validate({**VID, "width": 128, "height": 96})
        first = encode_deterministic_mp4(seg, ffmpeg_bin=ff)
        assert first == encode_deterministic_mp4(seg, ffmpeg_bin=ff)
        assert first[4:8] == b"ftyp"

    @pytest.mark.ffmpeg
    @pytest.mark.parametrize("model", ["qwen3_vl", "qwen2_5_vl"])
    def test_real_ffmpeg_encodes_vis_tokens_geometry(self, model: str) -> None:
        ff = find_ffmpeg_binary()
        if ff is None:
            pytest.skip("ffmpeg binary not installed")
        seg = DagMediaSegment.model_validate(
            {**VID_VT, "vis_tokens": 24, "num_frames": 2, "vision_model": model}
        )
        assert encode_deterministic_mp4(seg, ffmpeg_bin=ff)[4:8] == b"ftyp"


class TestLoader:
    def test_shared_media_is_encoded_once_and_identical_across_branches(
        self, tmp_path: Path, fake_ffmpeg
    ) -> None:
        script, calls = fake_ffmpeg
        explicit = {**IMG, "uuid": "author-uuid"}
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "media_segments": [IMG],
                        "prompt_spans": [
                            {"kind": "text", "span_id": "sys", "tokens": 6, "role": "system"},
                            {"kind": "text", "span_id": "ask", "tokens": 5},
                            _media_span("shared_img", 0),
                        ],
                        "spawns": ["eval"],
                    },
                    {
                        "media_segments": [IMG, VID],
                        "prompt_spans": [_media_span("shared_img"), _media_span("shared_vid")],
                        "forks": ["critic"],
                    },
                ],
            },
            {"session_id": "eval", "turns": [{"media_segments": [VID, explicit], "in_text_tokens": 3}]},
            {"session_id": "critic", "turns": [{"messages": [{"role": "user", "content": [
                {"type": "text", "text": "authored"},
                {"type": "media_ref", "media_hash": "shared_img"}]}],
                "media_segments": [IMG]}]},
        ]  # fmt: skip
        loader = DagJsonlLoader(
            _write_dag(tmp_path, rows), tokenizer=_Tokenizer(), emit_media_uuid=True
        )
        with patch.object(dag_jsonl_media, "find_ffmpeg_binary", return_value=script):
            convs = {c.session_id: c for c in loader.load()}

        assert sum("-c:v" in c for c in calls) == 1
        root0, root1 = (t.raw_messages for t in convs["root"].turns)
        assert [m["role"] for m in root0] == ["system", "user"]
        assert len(_two_tokens_per_long_word(root0[0]["content"][0]["text"])) == 6
        assert len(_two_tokens_per_long_word(root0[1]["content"][0]["text"])) == 5
        img_part = root0[1]["content"][1]
        img_uuid = default_media_uuid(DagMediaSegment.model_validate(IMG))
        assert img_part["uuid"] == img_uuid
        assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert root1[0]["content"][0] == img_part
        assert convs["critic"].turns[0].raw_messages[0]["content"][1] == img_part

        vid_part = root1[0]["content"][1]
        eval_parts = convs["eval"].turns[0].raw_messages[0]["content"]
        assert eval_parts[1] == vid_part
        assert base64.b64decode(vid_part["video_url"]["url"].split(",", 1)[1]) == (
            b"32 16 3 0.50"
        )
        assert eval_parts[2]["uuid"] == "author-uuid"
        assert eval_parts[2]["image_url"] == img_part["image_url"]

    def test_vis_tokens_segments_dedupe_across_turns_and_forks(
        self, tmp_path: Path, fake_ffmpeg
    ) -> None:
        script, calls = fake_ffmpeg
        vid = {**VID_VT, "vis_tokens": 24, "num_frames": 2}
        explicit_img = {**IMG_VT, "width": 1024, "height": 576}
        rows = [
            {"session_id": "root", "turns": [
                {"media_segments": [IMG_VT, vid], "in_text_tokens": 3},
                {"media_segments": [explicit_img], "in_text_tokens": 2, "forks": ["critic"]},
            ]},
            {"session_id": "critic", "turns": [{"media_segments": [IMG_VT, vid], "in_text_tokens": 2}]},
        ]  # fmt: skip
        loader = DagJsonlLoader(_write_dag(tmp_path, rows), tokenizer=_Tokenizer())
        with patch.object(dag_jsonl_media, "find_ffmpeg_binary", return_value=script):
            convs = {c.session_id: c for c in loader.load()}

        assert sum("-c:v" in c for c in calls) == 1
        root0, root1 = (t.raw_messages[0]["content"] for t in convs["root"].turns)
        assert convs["critic"].turns[0].raw_messages[0]["content"][1:] == root0[1:]
        assert root1[1] == root0[1]
        png = base64.b64decode(root0[1]["image_url"]["url"].split(",", 1)[1])
        assert PILImage.open(io.BytesIO(png)).size == (1024, 576)
        mp4 = base64.b64decode(root0[2]["video_url"]["url"].split(",", 1)[1])
        assert mp4.split()[:3] == [b"192", b"128", b"2"]

    def test_unreachable_vis_tokens_raise_dag_load_error(self, tmp_path: Path) -> None:
        rows = [{"session_id": "s", "turns": [{"media_segments": [{**IMG_VT, "vis_tokens": 577}]}]}]  # fmt: skip
        with pytest.raises(
            DagLoadError, match="vis_tokens=577 is unreachable for qwen3_vl"
        ):
            DagJsonlLoader(_write_dag(tmp_path, rows)).load()

    def test_uuids_omitted_without_emit_media_uuid(self, tmp_path: Path) -> None:
        rows = [{"session_id": "s", "turns": [{"media_segments": [IMG]}]}]
        part = DagJsonlLoader(_write_dag(tmp_path, rows)).load()[0].turns[0]
        assert "uuid" not in part.raw_messages[0]["content"][0]

    def test_synthesis_errors_wrapped_in_dag_load_error(self, tmp_path: Path) -> None:
        rows = [{"session_id": "s", "turns": [{"media_segments": [VID]}]}]
        with (
            patch.object(dag_jsonl_media, "find_ffmpeg_binary", return_value=None),
            pytest.raises(DagLoadError, match="session 's' turn 0.*ffmpeg not found"),
        ):
            DagJsonlLoader(_write_dag(tmp_path, rows)).load()

    def test_large_inline_payload_logs_one_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"session_id": f"s{i}", "turns": [{"media_segments": [IMG]}]}
            for i in range(3)
        ]
        monkeypatch.setattr(dag_jsonl_media, "INLINE_TOTAL_WARN_BYTES", 10)
        with patch.object(dag_jsonl_media._logger, "warning") as warn:
            DagJsonlLoader(_write_dag(tmp_path, rows)).load()
        warn.assert_called_once()
        assert "across 3 media references" in warn.call_args.args[0]
        assert "largest 'shared_img'" in warn.call_args.args[0]

    def test_small_inline_payload_does_not_warn(self, tmp_path: Path) -> None:
        rows = [{"session_id": "s", "turns": [{"media_segments": [IMG]}]}]
        with patch.object(dag_jsonl_media._logger, "warning") as warn:
            DagJsonlLoader(_write_dag(tmp_path, rows)).load()
        warn.assert_not_called()

    def test_existing_dag_fixtures_load_unchanged(self) -> None:
        fixtures = sorted(
            (Path(__file__).resolve().parents[3] / "fixtures" / "dag").glob(
                "*.dag.jsonl"
            )
        )
        assert len(fixtures) >= 12
        for fx in fixtures:
            loader = DagJsonlLoader(fx)
            assert loader.load()
            assert loader._materializer.inline_media_refs == 0
