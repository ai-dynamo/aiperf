# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multimodal ``dag_jsonl`` deterministic media synthesis & replay."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import aiohttp
import numpy as np
import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    ConversationContextMode,
    CreditPhase,
)
from aiperf.credit.structs import Credit
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.dataset.loader.dag_jsonl_media import (
    DagMultimodalMaterializer,
    LocalMediaHttpServer,
    _build_wire_parts_from_uris,
    _encode_mp4_with_codec,
    compute_mixed_block_hashes,
    default_media_uuid,
    find_ffmpeg_binary,
    iter_deterministic_rgb_frames,
    prepare_materializer_for_conversations,
    staged_frame_filename,
    staged_media_filename,
    synth_deterministic_png_bytes,
    synthesize_deterministic_rgb_frames,
    validate_dag_zero_leakage,
    verify_mp4_decodable,
    verify_png_bytes,
    write_deterministic_h264_mp4,
)
from aiperf.dataset.loader.dag_jsonl_models import (
    DagConversation,
    DagMediaSegment,
    DagPromptSpan,
    DagTurn,
)
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.workers.session_manager import UserSession
from aiperf.workers.worker import _apply_cache_bust


def _write_dag(tmp_path: Path, rows: list[dict], name: str = "mm.dag.jsonl") -> Path:
    out = tmp_path / name
    out.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")
    return out


def _make_y4m_bytes(
    width: int, height: int, num_frames: int, fps_num: int = 12, fps_den: int = 1
) -> bytes:
    header = f"YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} Ip A0:0 C444\n".encode(
        "ascii"
    )
    frame = b"FRAME\n" + (b"\x80" * (width * height * 3))
    return header + (frame * num_frames)


class _FakePopen:
    """Minimal ``subprocess.Popen`` stand-in for streaming encode/decode tests."""

    def __init__(
        self,
        cmd: list[str],
        *,
        stdout_payload: bytes = b"",
        returncode: int = 0,
        written_chunks: list[int] | None = None,
    ) -> None:
        self.cmd = cmd
        self._returncode = returncode
        self._written_chunks = written_chunks
        self.stdin = _ChunkTrackingSink(written_chunks)
        self.stdout = io.BytesIO(stdout_payload)
        self.stderr = io.BytesIO(b"")

    def wait(self) -> int:
        return self._returncode

    def poll(self) -> int:
        return self._returncode

    def kill(self) -> None:
        return None


class _ChunkTrackingSink:
    def __init__(self, chunks: list[int] | None) -> None:
        self._chunks = chunks
        self.closed = False

    def write(self, data: bytes | memoryview) -> int:
        n = len(data)
        if self._chunks is not None:
            self._chunks.append(n)
        return n

    def close(self) -> None:
        self.closed = True


class TestZeroLeakageValidation:
    """Verify structural zero-leakage traces reject forbidden raw keys and URIs."""

    def test_accepts_clean_structural_multimodal_dag_record(self) -> None:
        clean_record = {
            "session_id": "root",
            "turns": [
                {
                    "media_segments": [
                        {
                            "media_hash": "img_ref_a1b2",
                            "kind": "image",
                            "width": 512,
                            "height": 512,
                            "vis_tokens": 256,
                            "framing_tokens": 2,
                        },
                        {
                            "media_hash": "vid_clip_c3d4",
                            "kind": "video",
                            "width": 320,
                            "height": 256,
                            "fps": 24.0,
                            "num_frames": 8,
                            "vis_tokens": 320,
                            "framing_tokens": 32,
                        },
                    ],
                    "prompt_spans": [
                        {
                            "kind": "text",
                            "span_id": "sys_rubric",
                            "tokens": 64,
                            "role": "system",
                        },
                        {
                            "kind": "text",
                            "span_id": "user_intro",
                            "tokens": 32,
                            "role": "user",
                        },
                        {
                            "kind": "media",
                            "span_id": "img_ref_a1b2",
                            "tokens": 258,
                            "media_index": 0,
                            "role": "user",
                        },
                        {
                            "kind": "media",
                            "span_id": "vid_clip_c3d4",
                            "tokens": 352,
                            "media_index": 1,
                            "role": "user",
                        },
                    ],
                    "max_tokens": 64,
                    "forks": ["child-a", "child-b"],
                }
            ],
        }
        validate_dag_zero_leakage(clean_record)

    @pytest.mark.parametrize(
        "bad_payload,expected_match",
        [
            param({"session_id": "s", "messages": []}, "forbidden key 'messages'", id="messages_key"),
            param({"session_id": "s", "raw_text": "hello"}, "forbidden key 'raw_text'", id="raw_text_key"),
            param({"session_id": "s", "uri": "data:image/png;base64,AAAA"}, "inline media URI", id="data_image_uri"),
            param({"session_id": "s", "uri": "data:video/mp4;base64,AAAA"}, "inline media URI", id="data_video_uri"),
            param({"session_id": "s", "blob": "x" * 257}, "oversized raw string", id="oversized_string"),
        ],
    )  # fmt: skip
    def test_rejects_leaked_keys_inline_uris_and_oversized_strings(
        self, bad_payload: dict, expected_match: str
    ) -> None:
        with pytest.raises(ValueError, match=expected_match):
            validate_dag_zero_leakage(bad_payload)


class TestDagMultimodalModels:
    """Validate DagMediaSegment, DagPromptSpan, and DagTurn schema rules."""

    def test_video_requires_even_dimensions_and_positive_fps(self) -> None:
        with pytest.raises(ValidationError, match="even dimensions"):
            DagMediaSegment(
                media_hash="v1", kind="video", width=321, height=240, num_frames=4
            )
        with pytest.raises(ValidationError, match="fps > 0"):
            DagMediaSegment(
                media_hash="v1",
                kind="video",
                width=320,
                height=240,
                fps=0.0,
                num_frames=4,
            )
        with pytest.raises(ValidationError, match="fps >= 0.01"):
            DagMediaSegment(
                media_hash="v1",
                kind="video",
                width=320,
                height=240,
                fps=0.001,
                num_frames=4,
            )

    def test_image_rejects_multi_frame_and_boolean_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="num_frames=1"):
            DagMediaSegment(
                media_hash="i1", kind="image", width=256, height=256, num_frames=4
            )
        with pytest.raises(ValidationError, match="not a boolean"):
            DagMediaSegment.model_validate(
                {"media_hash": "i1", "kind": "image", "width": True, "height": 256}
            )

    def test_turn_mutual_exclusivity_and_media_ref_validation(self) -> None:
        seg = DagMediaSegment(media_hash="img0", kind="image", width=64, height=64)
        span = DagPromptSpan(kind="text", span_id="s0", tokens=16)

        with pytest.raises(ValidationError, match="mutually exclusive"):
            DagTurn(
                messages=[{"role": "user", "content": "hi"}],
                prompt_spans=[span],
            )

        with pytest.raises(ValidationError, match="no .* content part references them"):
            DagTurn(
                messages=[{"role": "user", "content": "hi"}],
                media_segments=[seg],
            )

        with pytest.raises(ValidationError, match="not found in turn media_segments"):
            DagTurn(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "media_ref", "media_hash": "missing"}],
                    }
                ],
                media_segments=[seg],
            )

    def test_structural_tool_role_rejected_while_authored_tool_message_allowed(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="tool_call_id"):
            DagPromptSpan.model_validate(
                {"kind": "text", "span_id": "tool_out", "tokens": 12, "role": "tool"}
            )

        authored = DagTurn(
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "call_abc",
                    "content": "result_payload",
                }
            ]
        )
        assert authored.messages is not None
        assert authored.messages[0]["tool_call_id"] == "call_abc"

    def test_media_prompt_span_requires_user_role(self) -> None:
        for bad_role in ("system", "assistant"):
            with pytest.raises(ValidationError, match="requires role='user'"):
                DagPromptSpan(
                    kind="media",
                    span_id="img0",
                    tokens=16,
                    media_index=0,
                    role=bad_role,  # type: ignore[arg-type]
                )

        sys_text = DagPromptSpan(kind="text", span_id="sys0", tokens=16, role="system")
        asst_text = DagPromptSpan(
            kind="text", span_id="asst0", tokens=16, role="assistant"
        )
        user_media = DagPromptSpan(
            kind="media", span_id="img0", tokens=16, media_index=0, role="user"
        )
        assert sys_text.role == "system"
        assert asst_text.role == "assistant"
        assert user_media.role == "user"

    @pytest.mark.parametrize(
        ("bad_role", "extra_fields"),
        [
            param("system", {}, id="system"),
            param("assistant", {}, id="assistant"),
            param("tool", {"tool_call_id": "call_1"}, id="tool"),
        ],
    )  # fmt: skip
    def test_media_ref_rejected_in_non_user_authored_messages(
        self, bad_role: str, extra_fields: dict[str, Any]
    ) -> None:
        seg = DagMediaSegment(media_hash="img0", kind="image", width=64, height=64)
        msg: dict[str, Any] = {
            "role": bad_role,
            "content": [{"type": "media_ref", "media_hash": "img0"}],
            **extra_fields,
        }
        with pytest.raises(ValidationError, match="requires message role='user'"):
            DagTurn(messages=[msg], media_segments=[seg])

    def test_loader_rejects_non_user_authored_media_ref(self, tmp_path: Path) -> None:
        bad_sys_path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "media_segments": [
                                {
                                    "media_hash": "img0",
                                    "kind": "image",
                                    "width": 64,
                                    "height": 64,
                                }
                            ],
                            "messages": [
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "media_ref", "media_hash": "img0"}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
            name="bad_sys_media_ref.dag.jsonl",
        )
        with pytest.raises(DagLoadError, match="requires message role='user'"):
            DagJsonlLoader(bad_sys_path).load()

    def test_resolve_authored_messages_rejects_non_user_media_ref_before_render(
        self,
    ) -> None:
        seg = DagMediaSegment(media_hash="img0", kind="image", width=64, height=64)
        mat = DagMultimodalMaterializer()
        with (
            patch.object(mat, "render_media_parts") as mock_render,
            pytest.raises(ValueError, match="requires message role='user'"),
        ):
            mat._resolve_authored_messages(
                [
                    {
                        "role": "system",
                        "content": [{"type": "media_ref", "media_hash": "img0"}],
                    }
                ],
                [seg],
            )
        mock_render.assert_not_called()

    def test_loader_accepts_mixed_role_authored_messages_with_user_media_ref(
        self, tmp_path: Path
    ) -> None:
        valid_path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "media_segments": [
                                {
                                    "media_hash": "img0",
                                    "kind": "image",
                                    "width": 64,
                                    "height": 64,
                                }
                            ],
                            "messages": [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": "Rubric."}],
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "media_ref", "media_hash": "img0"}
                                    ],
                                },
                                {"role": "assistant", "content": "Prior answer."},
                            ],
                        }
                    ],
                }
            ],
            name="ok_mixed_authored.dag.jsonl",
        )
        convs = DagJsonlLoader(valid_path).load()
        raw_msgs = convs[0].turns[0].raw_messages
        assert raw_msgs[0] == {
            "role": "system",
            "content": [{"type": "text", "text": "Rubric."}],
        }
        assert raw_msgs[1]["role"] == "user"
        assert raw_msgs[1]["content"][0]["type"] == "image_url"
        assert raw_msgs[2] == {"role": "assistant", "content": "Prior answer."}

    def test_ambiguous_hash_only_media_ref_and_span_require_media_index(self) -> None:
        seg_small = DagMediaSegment(
            media_hash="shared_key", kind="image", width=64, height=64
        )
        seg_large = DagMediaSegment(
            media_hash="shared_key", kind="image", width=128, height=128
        )

        with pytest.raises(ValidationError, match="specify 'media_index' explicitly"):
            DagTurn(
                media_segments=[seg_small, seg_large],
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "media_ref", "media_hash": "shared_key"}],
                    }
                ],
            )

        with pytest.raises(ValidationError, match="specify 'media_index' explicitly"):
            DagTurn(
                media_segments=[seg_small, seg_large],
                prompt_spans=[
                    DagPromptSpan(kind="media", span_id="shared_key", tokens=64)
                ],
            )

        indexed = DagTurn(
            media_segments=[seg_small, seg_large],
            prompt_spans=[
                DagPromptSpan(
                    kind="media", span_id="shared_key", tokens=64, media_index=1
                )
            ],
        )
        assert indexed.prompt_spans[0].media_index == 1

    @pytest.mark.parametrize(
        "probe,expected_dag,expected_multi_turn",
        [
            param({"session_id": "root", "turns": [{"in_text_tokens": 32}]}, True, False, id="positive_in_text_tokens"),
            param({"session_id": "root", "turns": [{"in_text_tokens": 0}]}, False, False, id="zero_in_text_tokens"),
            param({"session_id": "root", "turns": [{"in_text_tokens": True}]}, False, False, id="bool_in_text_tokens"),
            param({"session_id": "s1", "turns": [{"text": "plain multi-turn"}]}, False, True, id="multi_turn_text_only"),
            param({"session_id": "s1", "turns": [{"image_uuids": ["cached-image"]}]}, False, True, id="multi_turn_image_uuids_only"),
            param({"session_id": "s1", "turns": [{"image_uuids": ["cached-image"], "in_text_tokens": 32}]}, False, True, id="multi_turn_image_uuids_with_in_text_tokens"),
            param({"session_id": "s1", "turns": [{"text": "hi", "media_segments": []}]}, False, True, id="multi_turn_empty_media_segments"),
            param({"session_id": "s1", "turns": [{"text": "hi", "prompt_spans": []}]}, False, True, id="multi_turn_empty_prompt_spans"),
            param(
                {"session_id": "s1", "turns": [{"text": "hi", "media_segments": [{"media_hash": "m", "kind": "image", "width": 64, "height": 64}]}]},
                False,
                True,
                id="multi_turn_with_extra_media_segments",
            ),
            param({"session_id": "root", "turns": [{"media_segments": []}]}, False, False, id="empty_media_segments_only"),
            param({"session_id": "root", "turns": [{"prompt_spans": []}]}, False, False, id="empty_prompt_spans_only"),
            param(
                {"session_id": "root", "turns": [{"media_segments": [{"media_hash": "m", "kind": "image", "width": 64, "height": 64}]}]},
                True,
                False,
                id="non_empty_media_segments",
            ),
        ],
    )  # fmt: skip
    def test_can_load_recognizes_positive_in_text_tokens_narrowly(
        self, probe: dict[str, Any], expected_dag: bool, expected_multi_turn: bool
    ) -> None:
        assert DagJsonlLoader.can_load(probe) is expected_dag
        assert MultiTurnDatasetLoader.can_load(probe) is expected_multi_turn


class TestDeterministicSynthesisAndBlockHashes:
    """Verify deterministic PNG/MP4 synthesis, streaming verification, and block hashes."""

    def test_png_synthesis_is_deterministic_compact_and_decodable(self) -> None:
        png_a1 = synth_deterministic_png_bytes(1280, 720, "ref_keyframe_01")
        png_a2 = synth_deterministic_png_bytes(1280, 720, "ref_keyframe_01")
        png_b = synth_deterministic_png_bytes(1280, 720, "ref_keyframe_02")

        assert png_a1 == png_a2
        assert png_a1 != png_b
        assert len(png_a1) < 8192
        verify_png_bytes(png_a1, expected_width=1280, expected_height=720)

    def test_mixed_block_hashes_preserve_prefix_and_order_sensitivity(self) -> None:
        turn0 = [("text", "sys_rubric", 128), ("media", "vid_a", 128)]
        turn1 = [
            ("text", "sys_rubric", 128),
            ("media", "vid_a", 128),
            ("text", "followup", 70),
        ]
        swapped = [("text", "sys_rubric", 128), ("media", "vid_b", 128)]

        h0, tail0 = compute_mixed_block_hashes(turn0, block_size=64)
        h1, tail1 = compute_mixed_block_hashes(turn1, block_size=64)
        hs, _ = compute_mixed_block_hashes(swapped, block_size=64)

        assert len(h0) == 4 and tail0 == 0
        assert len(h1) == 5 and tail1 == 6
        assert h1[: len(h0)] == h0
        assert hs[:2] == h0[:2]
        assert hs[2:] != h0[2:]

    def test_mp4_synthesis_streams_frames_and_preserves_sub_unit_fps(
        self, tmp_path: Path
    ) -> None:
        seg = DagMediaSegment(
            media_hash="vid_sub_fps",
            kind="video",
            width=64,
            height=64,
            fps=0.5,
            num_frames=4,
        )
        y4m_stream = _make_y4m_bytes(64, 64, 4, fps_num=1, fps_den=2)
        encode_cmds: list[list[str]] = []
        chunk_sizes: list[int] = []
        decode_calls = 0

        def _fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
            nonlocal decode_calls
            if "yuv4mpegpipe" in cmd:
                decode_calls += 1
                return _FakePopen(cmd, stdout_payload=y4m_stream)
            encode_cmds.append(list(cmd))
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"\x00" * 256)
            return _FakePopen(cmd, written_chunks=chunk_sizes)

        mat = DagMultimodalMaterializer(
            media_scratch_dir=tmp_path,
            media_base_url="http://127.0.0.1:9000",
            ffmpeg_bin="/usr/bin/ffmpeg",
        )
        with patch("subprocess.Popen", side_effect=_fake_popen):
            paths = mat.stage_media_segment(seg)
            parts = mat.render_media_parts(seg)

        assert len(paths) == 1 and paths[0].is_file()
        assert paths[0].name.endswith("_4f_0.50fps.mp4")
        assert chunk_sizes == [64 * 64 * 3] * 4
        assert encode_cmds[0][encode_cmds[0].index("-r") + 1] == "0.50"
        assert decode_calls == 1
        assert parts[0]["video_url"]["url"].endswith(paths[0].name)

    def test_verify_mp4_decodable_rejects_mismatched_geometry_with_equal_bytes(
        self, tmp_path: Path
    ) -> None:
        mp4_path = tmp_path / "cached.mp4"
        mp4_path.write_bytes(b"\x00" * 256)
        wrong_geom_y4m = _make_y4m_bytes(64, 128, 2, fps_num=24, fps_den=1)

        with (
            patch(
                "subprocess.Popen",
                return_value=_FakePopen(["ffmpeg"], stdout_payload=wrong_geom_y4m),
            ),
            pytest.raises(RuntimeError, match="dimension mismatch.*got 64x128"),
        ):
            verify_mp4_decodable(
                mp4_path,
                expected_frames=4,
                expected_width=64,
                expected_height=64,
                ffmpeg_bin="/usr/bin/ffmpeg",
            )

    def test_staged_filenames_confine_traversal_url_metachars_and_fractional_fps(
        self, tmp_path: Path
    ) -> None:
        scratch = tmp_path / "scratch"
        seg_traversal = DagMediaSegment(
            media_hash="../../outside/img?q=1#frag",
            kind="image",
            width=64,
            height=64,
        )
        seg_plain = DagMediaSegment(
            media_hash="outside_img_q_1_frag",
            kind="image",
            width=64,
            height=64,
        )
        seg_vid_a = DagMediaSegment(
            media_hash="/tmp/vid_rate",
            kind="video",
            width=64,
            height=64,
            fps=23.5,
            num_frames=3,
        )
        seg_vid_b = DagMediaSegment(
            media_hash="/tmp/vid_rate",
            kind="video",
            width=64,
            height=64,
            fps=24.25,
            num_frames=3,
        )

        img_name_a = staged_media_filename(seg_traversal)
        img_name_b = staged_media_filename(seg_plain)
        vid_name_a = staged_media_filename(seg_vid_a)
        vid_name_b = staged_media_filename(seg_vid_b)
        frame_name = staged_frame_filename(seg_vid_a, 0)

        for name in (img_name_a, img_name_b, vid_name_a, vid_name_b, frame_name):
            assert "/" not in name and "\\" not in name and ".." not in name
            assert "?" not in name and "#" not in name and " " not in name
        assert img_name_a != img_name_b
        assert vid_name_a.endswith("_3f_23.50fps.mp4")
        assert vid_name_b.endswith("_3f_24.25fps.mp4")
        assert vid_name_a != vid_name_b

        mat = DagMultimodalMaterializer(
            video_wire_mode="image_frames",
            media_scratch_dir=scratch,
            media_base_url="http://127.0.0.1:8123",
        )
        written_img = mat.stage_media_segment(seg_traversal)
        written_frames = mat.stage_media_segment(seg_vid_a)
        for p in [*written_img, *written_frames]:
            assert p.resolve().parent == scratch.resolve()

        frame_parts = mat.render_media_parts(seg_vid_a)
        assert len(frame_parts) == 3
        for idx, part in enumerate(frame_parts):
            assert part["image_url"]["url"] == (
                f"http://127.0.0.1:8123/{written_frames[idx].name}"
            )

    def test_iter_deterministic_rgb_frames_matches_buffered_output(self) -> None:
        streamed = list(iter_deterministic_rgb_frames(64, 48, 3, "seed_stream"))
        buffered = synthesize_deterministic_rgb_frames(64, 48, 3, "seed_stream")
        assert len(streamed) == len(buffered) == 3
        for s_arr, b_arr in zip(streamed, buffered, strict=True):
            assert s_arr.shape == (48, 64, 3)
            assert s_arr.dtype == np.uint8
            assert 16 <= int(s_arr.min()) < int(s_arr.max()) < 240
            assert np.array_equal(s_arr, b_arr)

        expected_f1 = np.roll(np.roll(streamed[0], shift=16, axis=1), shift=8, axis=0)
        expected_f2 = np.roll(streamed[0], shift=32, axis=1)
        assert np.array_equal(streamed[1], expected_f1)
        assert np.array_equal(streamed[2], expected_f2)

        digests = [hashlib.sha256(arr.tobytes()).hexdigest() for arr in streamed]
        assert digests == [
            "c61e0b781c9d761d03cd524dcff72e07d07531f69505f57f4625f7a98b916728",
            "61af7104637e7eea4088b582c8b1b7d28e9b03a3d2d7daad51c70737c7afbbb7",
            "8bf33d334ccf2efde8f2911f459100bc37bb5ffee25b0ad7e87d0778bfd02a3f",
        ]

    def test_ffmpeg_cleanup_reaps_running_child_before_closing_stderr(
        self, tmp_path: Path
    ) -> None:
        class _RunningFakePopen:
            def __init__(self, stdout_payload: bytes = b"") -> None:
                self._alive = True
                self.killed = False
                self.waited = False
                self.stderr_closed_before_wait = False
                self.stdin = _ChunkTrackingSink(None)
                self.stdout = io.BytesIO(stdout_payload)
                self.stderr = io.BytesIO(b"diagnostic tail\n")

            def poll(self) -> int | None:
                return None if self._alive else -9

            def kill(self) -> None:
                self.killed = True
                self._alive = False

            def wait(self) -> int:
                self.stderr_closed_before_wait = self.stderr.closed
                self.waited = True
                self._alive = False
                return -9

        enc_proc = _RunningFakePopen()

        def _failing_frames() -> Iterator[np.ndarray]:
            raise ValueError("exceeds safety limit")
            yield np.zeros((4, 4, 3), dtype=np.uint8)

        with (
            patch("subprocess.Popen", return_value=enc_proc),
            pytest.raises(ValueError, match="exceeds safety limit"),
        ):
            _encode_mp4_with_codec(
                "/usr/bin/ffmpeg",
                "libx264",
                _failing_frames(),
                tmp_path / "out.mp4",
                width=64,
                height=64,
                fps=24.0,
            )
        assert enc_proc.killed is True
        assert enc_proc.waited is True
        assert enc_proc.stderr_closed_before_wait is False
        assert enc_proc.stderr.closed is True

        mp4_path = tmp_path / "corrupt.mp4"
        mp4_path.write_bytes(b"\x00" * 256)
        bad_marker_y4m = b"YUV4MPEG2 W64 H64 F24:1 Ip A0:0 C444\nBADMARKER\n"
        dec_proc = _RunningFakePopen(stdout_payload=bad_marker_y4m)
        with (
            patch("subprocess.Popen", return_value=dec_proc),
            pytest.raises(RuntimeError, match="Invalid Y4M frame marker"),
        ):
            verify_mp4_decodable(
                mp4_path,
                expected_frames=2,
                expected_width=64,
                expected_height=64,
                ffmpeg_bin="/usr/bin/ffmpeg",
            )
        assert dec_proc.killed is True
        assert dec_proc.waited is True
        assert dec_proc.stderr_closed_before_wait is False
        assert dec_proc.stdout.closed is True
        assert dec_proc.stderr.closed is True

        enc_start_fail = _RunningFakePopen()
        with (
            patch("subprocess.Popen", return_value=enc_start_fail),
            patch(
                "aiperf.dataset.loader.dag_jsonl_media._StderrTailDrain",
                side_effect=RuntimeError("can't start new thread"),
            ),
            pytest.raises(RuntimeError, match="can't start new thread"),
        ):
            _encode_mp4_with_codec(
                "/usr/bin/ffmpeg",
                "libx264",
                iter_deterministic_rgb_frames(16, 16, 2, "drain_fail"),
                tmp_path / "drain_fail.mp4",
                width=16,
                height=16,
                fps=24.0,
            )
        assert enc_start_fail.killed is True
        assert enc_start_fail.waited is True
        assert enc_start_fail.stdin.closed is True
        assert enc_start_fail.stderr.closed is True

        dec_start_fail = _RunningFakePopen(stdout_payload=bad_marker_y4m)
        with (
            patch("subprocess.Popen", return_value=dec_start_fail),
            patch(
                "aiperf.dataset.loader.dag_jsonl_media._StderrTailDrain",
                side_effect=RuntimeError("can't start new thread"),
            ),
            pytest.raises(RuntimeError, match="can't start new thread"),
        ):
            verify_mp4_decodable(
                mp4_path,
                expected_frames=2,
                expected_width=64,
                expected_height=64,
                ffmpeg_bin="/usr/bin/ffmpeg",
            )
        assert dec_start_fail.killed is True
        assert dec_start_fail.waited is True
        assert dec_start_fail.stdout.closed is True
        assert dec_start_fail.stderr.closed is True

    def test_ffmpeg_pipes_drain_large_concurrent_stderr_without_deadlock(
        self, tmp_path: Path
    ) -> None:
        fake_ff = tmp_path / "fake_ffmpeg.py"
        fake_ff.write_text(
            "import pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "if 'yuv4mpegpipe' in args:\n"
            "    sys.stderr.buffer.write(b'W' * 300_000 + b'\\ndecode_tail_marker\\n')\n"
            "    sys.stderr.buffer.flush()\n"
            "    sys.stdout.buffer.write(b'YUV4MPEG2 W16 H16 F12:1 Ip A0:0 C444\\n')\n"
            "    sys.stdout.buffer.write(b'FRAME\\n' + b'\\x80' * (16 * 16 * 3))\n"
            "    sys.stdout.buffer.flush()\n"
            "    sys.exit(1)\n"
            "while sys.stdin.buffer.read(4096):\n"
            "    sys.stderr.buffer.write(b'E' * 65536)\n"
            "    sys.stderr.buffer.flush()\n"
            "sys.stderr.buffer.write(b'\\nencode_tail_marker\\n')\n"
            "sys.stderr.buffer.flush()\n"
            "pathlib.Path(args[-1]).write_bytes(b'\\x00' * 256)\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        real_popen = subprocess.Popen

        def _python_script_popen(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> subprocess.Popen[bytes]:
            rewritten = (
                [sys.executable, *cmd] if cmd and cmd[0] == str(fake_ff) else cmd
            )
            return real_popen(rewritten, *args, **kwargs)

        with patch("subprocess.Popen", side_effect=_python_script_popen):
            rc, stderr_tail = _encode_mp4_with_codec(
                str(fake_ff),
                "libx264",
                iter_deterministic_rgb_frames(32, 32, 8, "verbose_stderr"),
                tmp_path / "enc.mp4",
                width=32,
                height=32,
                fps=12.0,
            )
            assert rc == 1
            assert len(stderr_tail) <= 4096
            assert b"encode_tail_marker" in stderr_tail

            mp4_path = tmp_path / "dec.mp4"
            mp4_path.write_bytes(b"\x00" * 256)
            with pytest.raises(RuntimeError, match="decode_tail_marker"):
                verify_mp4_decodable(
                    mp4_path,
                    expected_frames=2,
                    expected_width=16,
                    expected_height=16,
                    expected_fps=12.0,
                    ffmpeg_bin=str(fake_ff),
                )

    @pytest.mark.ffmpeg
    def test_mp4_synthesis_real_ffmpeg_roundtrip(self, tmp_path: Path) -> None:
        ff = find_ffmpeg_binary()
        if ff is None:
            pytest.skip("ffmpeg binary not installed")
        seg = DagMediaSegment(
            media_hash="vid_real_h264",
            kind="video",
            width=128,
            height=96,
            fps=0.5,
            num_frames=4,
        )
        mp4_a = write_deterministic_h264_mp4(tmp_path / "a.mp4", seg, ffmpeg_bin=ff)
        mp4_b = write_deterministic_h264_mp4(tmp_path / "b.mp4", seg, ffmpeg_bin=ff)
        verify_mp4_decodable(
            mp4_a,
            expected_frames=4,
            expected_width=128,
            expected_height=96,
            expected_fps=0.5,
            ffmpeg_bin=ff,
        )
        assert mp4_a.read_bytes() == mp4_b.read_bytes()


class TestDagLoaderMultimodalDeduplicationAndReplay:
    """Verify cross-turn/branch deduplication, staging readiness, and server lifecycle."""

    def test_media_hash_deduplication_across_fork_and_spawn_branches(
        self, tmp_path: Path
    ) -> None:
        shared_img = {
            "media_hash": "shared_ref_img",
            "kind": "image",
            "width": 128,
            "height": 128,
            "vis_tokens": 16,
        }
        shared_vid = {
            "media_hash": "shared_cand_vid",
            "kind": "video",
            "width": 128,
            "height": 96,
            "fps": 12.0,
            "num_frames": 4,
            "vis_tokens": 48,
        }
        rows = [
            {
                "session_id": "planner-root",
                "turns": [
                    {
                        "media_segments": [shared_img],
                        "prompt_spans": [
                            {
                                "kind": "text",
                                "span_id": "sys_p",
                                "tokens": 24,
                                "role": "system",
                            },
                            {
                                "kind": "text",
                                "span_id": "u_p",
                                "tokens": 16,
                                "role": "user",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_ref_img",
                                "tokens": 18,
                                "media_index": 0,
                                "role": "user",
                            },
                        ],
                        "max_tokens": 32,
                        "spawns": ["eval-a", "eval-b"],
                    },
                    {
                        "media_segments": [shared_img, shared_vid],
                        "prompt_spans": [
                            {
                                "kind": "text",
                                "span_id": "u_replan",
                                "tokens": 20,
                                "role": "user",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_ref_img",
                                "tokens": 18,
                                "media_index": 0,
                                "role": "user",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_cand_vid",
                                "tokens": 50,
                                "media_index": 1,
                                "role": "user",
                            },
                        ],
                        "max_tokens": 48,
                        "forks": ["fork-critic"],
                    },
                ],
            },
            {
                "session_id": "eval-a",
                "turns": [
                    {
                        "media_segments": [shared_img, shared_vid],
                        "prompt_spans": [
                            {
                                "kind": "text",
                                "span_id": "eval_rubric",
                                "tokens": 32,
                                "role": "system",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_ref_img",
                                "tokens": 18,
                                "media_index": 0,
                                "role": "user",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_cand_vid",
                                "tokens": 50,
                                "media_index": 1,
                                "role": "user",
                            },
                        ],
                        "max_tokens": 16,
                    }
                ],
            },
            {
                "session_id": "eval-b",
                "turns": [
                    {
                        "media_segments": [shared_vid, shared_img],
                        "prompt_spans": [
                            {
                                "kind": "text",
                                "span_id": "eval_rubric",
                                "tokens": 32,
                                "role": "system",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_cand_vid",
                                "tokens": 50,
                                "media_index": 0,
                                "role": "user",
                            },
                            {
                                "kind": "media",
                                "span_id": "shared_ref_img",
                                "tokens": 18,
                                "media_index": 1,
                                "role": "user",
                            },
                        ],
                        "max_tokens": 16,
                    }
                ],
            },
            {
                "session_id": "fork-critic",
                "turns": [
                    {
                        "media_segments": [shared_img],
                        "in_text_tokens": 12,
                        "max_tokens": 24,
                    }
                ],
            },
        ]
        for r in rows:
            validate_dag_zero_leakage(r)

        dag_path = _write_dag(tmp_path, rows)
        assert DagJsonlLoader.can_load(rows[0], dag_path) is True

        scratch = tmp_path / "staged_media"
        loader = DagJsonlLoader(
            dag_path,
            media_scratch_dir=scratch,
            media_base_url="http://127.0.0.1:18080",
            video_wire_mode="image_frames",
            emit_media_uuid=True,
        )
        convs = {c.session_id: c for c in loader.load()}
        summary = loader.media_staging_summary
        assert summary is not None
        assert summary.unique_images == 1
        assert summary.unique_videos == 1
        assert summary.total_segment_references == 8
        assert summary.deduplicated_references == 6

        sys_a = convs["eval-a"].turns[0].raw_messages[0]
        sys_b = convs["eval-b"].turns[0].raw_messages[0]
        assert sys_a == sys_b

        root_img_part = convs["planner-root"].turns[0].raw_messages[1]["content"][1]
        eval_a_img_part = convs["eval-a"].turns[0].raw_messages[1]["content"][0]
        fork_img_part = convs["fork-critic"].turns[0].raw_messages[0]["content"][1]
        assert root_img_part == eval_a_img_part == fork_img_part
        expected_uuid = default_media_uuid(DagMediaSegment.model_validate(shared_img))
        assert expected_uuid.startswith("shared_ref_img_")
        assert root_img_part["uuid"] == expected_uuid

        session = UserSession(
            x_correlation_id="corr-1",
            num_turns=1,
            conversation=convs["eval-a"],
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
            turn_list=[convs["eval-a"].turns[0].model_copy(deep=True)],
        )
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="eval-a",
            x_correlation_id="corr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=1000,
            cache_bust_marker="[rid:deadbeef1234]\n\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
            branch_mode=ConversationBranchMode.SPAWN,
        )
        _apply_cache_bust(session, credit, None, media_uuids=True)
        busted_parts = session.turn_list[0].raw_messages[1]["content"]
        img_parts = [p for p in busted_parts if p.get("type") == "image_url"]
        assert img_parts[0]["uuid"].endswith("-deadbeef1234")

    @pytest.mark.asyncio
    async def test_local_media_http_server_serves_staged_png_async(
        self, tmp_path: Path
    ) -> None:
        scratch = tmp_path / "http_staged"
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "http_ref_img",
                                "kind": "image",
                                "width": 128,
                                "height": 128,
                            }
                        ],
                        "in_text_tokens": 8,
                        "max_tokens": 16,
                    }
                ],
            }
        ]
        dag_path = _write_dag(tmp_path, rows, name="http.dag.jsonl")
        with LocalMediaHttpServer(scratch) as media_srv:
            loader = DagJsonlLoader(
                dag_path,
                media_scratch_dir=scratch,
                media_base_url=media_srv.base_url,
            )
            convs = await loader.load_async()
            root_img_part = convs[0].turns[0].raw_messages[0]["content"][1]
            img_url = root_img_part["image_url"]["url"]
            assert img_url.startswith(media_srv.base_url)
            async with (
                aiohttp.ClientSession() as client,
                client.get(img_url, timeout=aiohttp.ClientTimeout(total=5)) as resp,
            ):
                assert resp.status == 200
                fetched_png = await resp.read()
            verify_png_bytes(fetched_png, expected_width=128, expected_height=128)

    def test_local_media_http_server_close_before_start_releases_socket(
        self, tmp_path: Path
    ) -> None:
        srv = LocalMediaHttpServer(tmp_path / "unstarted")
        bound_port = srv.port
        assert bound_port > 0
        srv.close()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_sock:
            probe_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe_sock.bind(("127.0.0.1", bound_port))

    def test_tokenizer_iterative_calibration_and_unreachable_budget(self) -> None:
        class _MultiTokenWordTokenizer:
            def encode(self, text: str) -> list[int]:
                out: list[int] = []
                for idx, word in enumerate(text.split()):
                    out.append(idx * 2)
                    if len(word) > 1:
                        out.append(idx * 2 + 1)
                return out

        tok = _MultiTokenWordTokenizer()
        mat = DagMultimodalMaterializer(tokenizer=tok)
        text_4 = mat.render_span_text("cal_4", 4)
        text_5 = mat.render_span_text("cal_5", 5)
        assert len(tok.encode(text_4)) == 4
        assert len(tok.encode(text_5)) == 5

        class _EvenOnlyTokenizer:
            def encode(self, text: str) -> list[int]:
                return [0, 1] * max(1, len(text.split()))

        bad_mat = DagMultimodalMaterializer(tokenizer=_EvenOnlyTokenizer())
        with pytest.raises(ValueError, match="could not calibrate span 'odd' to 3"):
            bad_mat.render_span_text("odd", 3)

    def test_shared_media_hash_with_distinct_explicit_uuids_stages_all_variants(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "shared_asset",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                                "uuid": "explicit-uuid-1",
                            }
                        ],
                        "in_text_tokens": 4,
                    },
                    {
                        "media_segments": [
                            {
                                "media_hash": "shared_asset",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                                "uuid": "explicit-uuid-2",
                            }
                        ],
                        "in_text_tokens": 4,
                    },
                ],
            }
        ]
        path = _write_dag(tmp_path, rows, name="uuid_variants.dag.jsonl")
        loader = DagJsonlLoader(path)
        convs = loader.load()
        assert loader.media_staging_summary is not None
        assert loader.media_staging_summary.unique_images == 1
        assert loader.media_staging_summary.deduplicated_references == 1

        part0 = convs[0].turns[0].raw_messages[0]["content"][1]
        part1 = convs[0].turns[1].raw_messages[0]["content"][1]
        assert part0["uuid"] == "explicit-uuid-1"
        assert part1["uuid"] == "explicit-uuid-2"
        assert part0["image_url"]["url"] == part1["image_url"]["url"]

    def test_staged_population_survives_lru_capacity_nested_mutation_and_second_load(
        self, tmp_path: Path
    ) -> None:
        turns_batch_1 = [
            {
                "media_segments": [
                    {
                        "media_hash": f"img_batch1_{idx:03d}",
                        "kind": "image",
                        "width": 32,
                        "height": 32,
                    }
                ],
                "in_text_tokens": 4,
            }
            for idx in range(80)
        ]
        path1 = _write_dag(
            tmp_path,
            [{"session_id": "s1", "turns": turns_batch_1}],
            name="batch1.dag.jsonl",
        )
        mat = DagMultimodalMaterializer(max_cache_items=64)
        convs1 = DagJsonlLoader(path1, materializer=mat).load()
        assert len(convs1[0].turns) == 80
        assert mat.is_frozen_for_timed_replay is True

        first_part = convs1[0].turns[0].raw_messages[0]["content"][1]
        original_url = first_part["image_url"]["url"]
        first_part["image_url"]["url"] = "mutated://caller"
        fresh = mat.render_media_parts(
            DagMediaSegment(
                media_hash="img_batch1_000", kind="image", width=32, height=32
            )
        )
        assert fresh[0]["image_url"]["url"] == original_url

        path2 = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "s2",
                    "turns": [
                        {
                            "media_segments": [
                                {
                                    "media_hash": "img_batch2_new",
                                    "kind": "image",
                                    "width": 48,
                                    "height": 48,
                                }
                            ],
                            "in_text_tokens": 4,
                        }
                    ],
                }
            ],
            name="batch2.dag.jsonl",
        )
        convs2 = DagJsonlLoader(path2, materializer=mat).load()
        assert len(convs2) == 1
        assert mat.is_frozen_for_timed_replay is True

    def test_effective_spans_supports_max_session_id_and_max_vis_tokens(
        self, tmp_path: Path
    ) -> None:
        long_sid = "x" * 128
        rows = [
            {
                "session_id": long_sid,
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "max_tok_img",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                                "vis_tokens": 1_048_576,
                                "framing_tokens": 2,
                            }
                        ],
                        "in_text_tokens": 8,
                    }
                ],
            }
        ]
        path = _write_dag(tmp_path, rows, name="limits.dag.jsonl")
        convs = DagJsonlLoader(path).load()
        assert convs[0].session_id == long_sid
        assert len(convs[0].turns[0].raw_messages[0]["content"]) == 2

    def test_staging_oserror_wrapped_in_dag_load_error(self, tmp_path: Path) -> None:
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "disk_err_img",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                            }
                        ]
                    }
                ],
            }
        ]
        path = _write_dag(tmp_path, rows, name="oserr.dag.jsonl")
        with (
            patch(
                "aiperf.dataset.loader.dag_jsonl.prepare_materializer_for_conversations",
                side_effect=OSError("No space left on device"),
            ),
            pytest.raises(
                DagLoadError, match="multimodal staging failed.*No space left"
            ) as exc_info,
        ):
            DagJsonlLoader(path).load()
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_hybrid_messages_media_ref_and_frozen_hotpath_guard(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "session_id": "hybrid-root",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "diagram_01",
                                "kind": "image",
                                "width": 96,
                                "height": 64,
                            }
                        ],
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Inspect this diagram:"},
                                    {"type": "media_ref", "media_hash": "diagram_01"},
                                ],
                            }
                        ],
                        "max_tokens": 32,
                    }
                ],
            }
        ]
        path = _write_dag(tmp_path, rows, name="hybrid.dag.jsonl")
        mat = DagMultimodalMaterializer()
        loader = DagJsonlLoader(path, materializer=mat)
        convs = loader.load()
        content = convs[0].turns[0].raw_messages[0]["content"]
        assert content[0] == {"type": "text", "text": "Inspect this diagram:"}
        uri = content[1]["image_url"]["url"]
        assert uri.startswith("data:image/png;base64,")
        raw_png = base64.b64decode(uri.split(",", 1)[1])
        verify_png_bytes(raw_png, expected_width=96, expected_height=64)

        assert mat.encode_calls_during_timed_replay == 0
        unstaged = DagMediaSegment(
            media_hash="unstaged_late", kind="image", width=64, height=64
        )
        with pytest.raises(RuntimeError, match="frozen timed replay"):
            mat.render_media_parts(unstaged)
        assert mat.encode_calls_during_timed_replay == 1

    def test_synthesized_non_root_system_span_rejected_by_placement_validator(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "prompt_spans": [
                            {
                                "kind": "text",
                                "span_id": "u0",
                                "tokens": 8,
                                "role": "user",
                            }
                        ]
                    },
                    {
                        "prompt_spans": [
                            {
                                "kind": "text",
                                "span_id": "late_sys",
                                "tokens": 8,
                                "role": "system",
                            }
                        ]
                    },
                ],
            }
        ]
        path = _write_dag(tmp_path, rows, name="bad_sys.dag.jsonl")
        with pytest.raises(
            DagLoadError, match="non-root turns may not contain a 'system'"
        ):
            DagJsonlLoader(path).load()

    def test_all_existing_dag_fixtures_load_unchanged(self) -> None:
        fixtures_dir = Path(__file__).resolve().parents[3] / "fixtures" / "dag"
        fixture_files = sorted(fixtures_dir.glob("*.dag.jsonl"))
        assert len(fixture_files) >= 12
        for fx in fixture_files:
            loader = DagJsonlLoader(fx)
            convs = loader.load()
            assert len(convs) >= 1
            assert loader.media_staging_summary is None

    def test_default_media_uuid_distinguishes_geometry_and_preserves_explicit_uuid(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "shared_key",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                            },
                            {
                                "media_hash": "shared_key",
                                "kind": "image",
                                "width": 128,
                                "height": 128,
                            },
                            {
                                "media_hash": "shared_key",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                                "uuid": "explicit-override-uuid",
                            },
                        ],
                        "prompt_spans": [
                            {
                                "kind": "media",
                                "span_id": "s_small",
                                "tokens": 16,
                                "media_index": 0,
                            },
                            {
                                "kind": "media",
                                "span_id": "s_large",
                                "tokens": 64,
                                "media_index": 1,
                            },
                            {
                                "kind": "media",
                                "span_id": "s_explicit",
                                "tokens": 16,
                                "media_index": 2,
                            },
                        ],
                        "forks": ["child"],
                    }
                ],
            },
            {
                "session_id": "child",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "shared_key",
                                "kind": "image",
                                "width": 64,
                                "height": 64,
                            }
                        ],
                        "in_text_tokens": 4,
                    }
                ],
            },
        ]
        path = _write_dag(tmp_path, rows, name="uuid_geom.dag.jsonl")
        convs = {
            c.session_id: c for c in DagJsonlLoader(path, emit_media_uuid=True).load()
        }
        root_parts = convs["root"].turns[0].raw_messages[0]["content"]
        child_img_part = convs["child"].turns[0].raw_messages[0]["content"][1]

        assert root_parts[0]["uuid"] != root_parts[1]["uuid"]
        assert root_parts[0]["image_url"]["url"] != root_parts[1]["image_url"]["url"]
        assert root_parts[2]["uuid"] == "explicit-override-uuid"
        assert root_parts[0]["uuid"] == child_img_part["uuid"]

    def test_implicit_in_text_tokens_staged_with_population_keys_and_wraps_error(
        self, tmp_path: Path
    ) -> None:
        good_rows = [
            {
                "session_id": "alpha",
                "turns": [{"in_text_tokens": 6}, {"in_text_tokens": 8}],
            }
        ]
        good_path = _write_dag(tmp_path, good_rows, name="in_tok_ok.dag.jsonl")
        mat = DagMultimodalMaterializer()
        convs = DagJsonlLoader(good_path, materializer=mat).load()
        assert ("alpha:0:text", 6) in mat._staged_text
        assert ("alpha:1:text", 8) in mat._staged_text
        assert (
            convs[0].turns[0].raw_messages[0]["content"][0]["text"]
            == mat._staged_text[("alpha:0:text", 6)]
        )

        class _EvenOnlyTokenizer:
            def encode(self, text: str) -> list[int]:
                return [0, 1] * max(1, len(text.split()))

        bad_rows = [
            {"session_id": "ok-first", "turns": [{"in_text_tokens": 4}]},
            {"session_id": "bad-second", "turns": [{"in_text_tokens": 3}]},
        ]
        bad_path = _write_dag(tmp_path, bad_rows, name="in_tok_bad.dag.jsonl")
        loader = DagJsonlLoader(bad_path, tokenizer=_EvenOnlyTokenizer())
        with pytest.raises(
            DagLoadError,
            match="multimodal staging failed.*could not calibrate span 'bad-second:0:text'",
        ):
            loader.load()
        assert loader.media_staging_summary is None
        assert loader._conversations == {}

    def test_shared_materializer_stays_frozen_after_failed_load_and_clears_staged_cache(
        self, tmp_path: Path
    ) -> None:
        fresh_mat = DagMultimodalMaterializer()
        bad_conv = DagConversation.model_validate(
            {
                "session_id": "bad",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "v",
                                "kind": "video",
                                "width": 64,
                                "height": 64,
                                "num_frames": 2,
                            }
                        ]
                    }
                ],
            }
        )
        with (
            patch(
                "aiperf.dataset.loader.dag_jsonl_media.find_ffmpeg_binary",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="FFmpeg binary not found"),
        ):
            prepare_materializer_for_conversations([bad_conv], materializer=fresh_mat)
        assert fresh_mat.is_frozen_for_timed_replay is False

        ok_path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "ok",
                    "turns": [
                        {
                            "media_segments": [
                                {
                                    "media_hash": "img_ok",
                                    "kind": "image",
                                    "width": 64,
                                    "height": 64,
                                }
                            ],
                            "in_text_tokens": 4,
                        }
                    ],
                }
            ],
            name="shared_ok.dag.jsonl",
        )
        shared_mat = DagMultimodalMaterializer()
        DagJsonlLoader(ok_path, materializer=shared_mat).load()
        assert shared_mat.is_frozen_for_timed_replay is True
        assert shared_mat._staged_uris
        assert shared_mat._staged_wire_parts
        assert shared_mat._staged_text

        with (
            patch(
                "aiperf.dataset.loader.dag_jsonl_media.find_ffmpeg_binary",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="FFmpeg binary not found"),
        ):
            prepare_materializer_for_conversations([bad_conv], materializer=shared_mat)
        assert shared_mat.is_frozen_for_timed_replay is True

        second_conv = DagConversation.model_validate(
            {
                "session_id": "ok2",
                "turns": [
                    {
                        "media_segments": [
                            {
                                "media_hash": "img_second",
                                "kind": "image",
                                "width": 48,
                                "height": 48,
                            }
                        ],
                        "in_text_tokens": 4,
                    }
                ],
            }
        )
        uncached_probe = DagMediaSegment(
            media_hash="uncached_during_prep", kind="image", width=32, height=32
        )
        observed_frozen_during_prep: list[bool] = []
        orig_compute_uris = shared_mat._compute_segment_uris

        def _checked_compute_uris(seg: DagMediaSegment) -> tuple[str, ...]:
            observed_frozen_during_prep.append(shared_mat.is_frozen_for_timed_replay)
            with pytest.raises(RuntimeError, match="frozen timed replay"):
                shared_mat.render_media_parts(uncached_probe)
            return orig_compute_uris(seg)

        with patch.object(
            shared_mat, "_compute_segment_uris", side_effect=_checked_compute_uris
        ):
            prepare_materializer_for_conversations(
                [second_conv], materializer=shared_mat
            )
        assert observed_frozen_during_prep == [True]
        assert shared_mat.is_frozen_for_timed_replay is True
        assert shared_mat.encode_calls_during_timed_replay == 1

        shared_mat.clear_staged_cache()
        assert shared_mat._staged_uris == {}
        assert shared_mat._staged_wire_parts == {}
        assert shared_mat._staged_filenames == {}
        assert shared_mat._staged_variants == {}
        assert shared_mat._staged_text == {}

    @pytest.mark.asyncio
    async def test_set_media_base_url_serializes_with_async_stage_turns(
        self, tmp_path: Path
    ) -> None:
        turns = [
            DagTurn(
                media_segments=[
                    DagMediaSegment(
                        media_hash=f"img_lock_{idx}",
                        kind="image",
                        width=32,
                        height=32,
                    )
                ],
                in_text_tokens=4,
            )
            for idx in range(2)
        ]
        mat = await asyncio.to_thread(
            lambda: DagMultimodalMaterializer(
                media_scratch_dir=tmp_path,
                media_base_url="http://127.0.0.1:9001",
            )
        )
        real_stage_lock = mat._stage_lock
        seen_lock_threads: set[int] = set()
        meta_lock = threading.Lock()
        setter_attempted_lock = threading.Event()
        entered_wire_pass = threading.Event()
        release_wire_pass = threading.Event()

        class _TrackingRLock:
            def __enter__(self) -> Any:
                with meta_lock:
                    seen_lock_threads.add(threading.get_ident())
                    if len(seen_lock_threads) >= 2:
                        setter_attempted_lock.set()
                return real_stage_lock.__enter__()

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
                return real_stage_lock.__exit__(exc_type, exc, tb)

        mat._stage_lock = _TrackingRLock()  # type: ignore[assignment]
        first_variant = True
        staging_tid: int | None = None

        def _probe_nonblocking_acquire() -> tuple[int, bool]:
            acquired = real_stage_lock.acquire(blocking=False)
            if acquired:
                real_stage_lock.release()
            return threading.get_ident(), acquired

        def _paused_build_wire(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal first_variant, staging_tid
            if first_variant:
                first_variant = False
                staging_tid = threading.get_ident()
                entered_wire_pass.set()
                assert release_wire_pass.wait(timeout=5.0)
            return _build_wire_parts_from_uris(*args, **kwargs)

        with patch(
            "aiperf.dataset.loader.dag_jsonl_media._build_wire_parts_from_uris",
            side_effect=_paused_build_wire,
        ):
            stage_task = asyncio.create_task(mat.async_stage_turns(turns))
            assert await asyncio.to_thread(entered_wire_pass.wait, 5.0) is True
            probe_tid, lock_acquired = await asyncio.to_thread(
                _probe_nonblocking_acquire
            )
            assert probe_tid != staging_tid
            assert lock_acquired is False
            setter_task = asyncio.create_task(
                asyncio.to_thread(mat.set_media_base_url, "http://127.0.0.1:9002/")
            )
            assert await asyncio.to_thread(setter_attempted_lock.wait, 5.0) is True
            release_wire_pass.set()
            summary = await stage_task
            await setter_task

        assert summary.media_base_url == "http://127.0.0.1:9001"
        assert mat.media_base_url == "http://127.0.0.1:9002"
        assert mat._media_cache == {}
        assert len(mat._staged_uris) == 2
        assert all(
            uri.startswith("http://127.0.0.1:9002/")
            for uris in mat._staged_uris.values()
            for uri in uris
        )
        assert len(mat._staged_wire_parts) == 2
        assert all(
            part["image_url"]["url"].startswith("http://127.0.0.1:9002/")
            for parts in mat._staged_wire_parts.values()
            for part in parts
        )

    @pytest.mark.asyncio
    async def test_concurrent_load_async_and_cancellation_serialize_initialization(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {
                "session_id": "root",
                "turns": [
                    {
                        "in_text_tokens": 8,
                        "spawns": ["child-s"],
                    },
                    {"in_text_tokens": 6, "forks": ["child-f"]},
                ],
            },
            {"session_id": "child-s", "turns": [{"in_text_tokens": 4}]},
            {"session_id": "child-f", "turns": [{"in_text_tokens": 4}]},
        ]
        path = _write_dag(tmp_path, rows, name="concurrent.dag.jsonl")
        loader = DagJsonlLoader(path)
        entered = threading.Event()
        second_attempt = threading.Event()
        release = threading.Event()
        parse_calls = 0
        orig_parse = loader._parse_lines
        real_load_lock = loader._load_lock
        seen_threads: set[int] = set()
        meta_lock = threading.Lock()

        class _TrackingLoadLock:
            def __enter__(self) -> Any:
                with meta_lock:
                    seen_threads.add(threading.get_ident())
                    if len(seen_threads) >= 2:
                        second_attempt.set()
                return real_load_lock.__enter__()

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
                return real_load_lock.__exit__(exc_type, exc, tb)

        loader._load_lock = _TrackingLoadLock()  # type: ignore[assignment]

        def _slow_parse() -> None:
            nonlocal parse_calls
            parse_calls += 1
            entered.set()
            assert release.wait(timeout=5.0)
            orig_parse()

        with patch.object(loader, "_parse_lines", side_effect=_slow_parse):
            t1 = asyncio.create_task(loader.load_async())
            assert await asyncio.to_thread(entered.wait, 5.0) is True
            t2 = asyncio.create_task(loader.load_async())
            assert await asyncio.to_thread(second_attempt.wait, 5.0) is True
            t1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t1
            release.set()
            convs = await t2

        assert parse_calls == 1
        by_sid = {c.session_id: c for c in convs}
        assert len(by_sid["root"].branches) == 2
        assert len(by_sid["root"].turns[0].branch_ids) == 1
        assert len(by_sid["root"].turns[1].branch_ids) == 1
        assert len(by_sid["root"].turns[1].prerequisites) == 1

    def test_image_frames_staged_bytes_counts_shared_frame_files_once(
        self, tmp_path: Path
    ) -> None:
        seg_12fps = DagMediaSegment(
            media_hash="shared_vid",
            kind="video",
            width=64,
            height=64,
            fps=12.0,
            num_frames=3,
        )
        seg_24fps = DagMediaSegment(
            media_hash="shared_vid",
            kind="video",
            width=64,
            height=64,
            fps=24.0,
            num_frames=3,
        )
        turns = [
            DagTurn(media_segments=[seg_12fps, seg_24fps], in_text_tokens=4),
            DagTurn(media_segments=[seg_12fps], in_text_tokens=4),
        ]
        mat = DagMultimodalMaterializer(
            video_wire_mode="image_frames",
            media_scratch_dir=tmp_path,
        )
        summary = mat.stage_turns(turns)

        png_files = sorted(tmp_path.glob("*.png"))
        assert len(png_files) == 3
        assert summary.unique_videos == 2
        assert summary.total_segment_references == 3
        assert summary.deduplicated_references == 1
        assert summary.staged_bytes == sum(p.stat().st_size for p in png_files)

    @pytest.mark.asyncio
    async def test_load_async_holds_shared_materializer_lock_through_population(
        self, tmp_path: Path
    ) -> None:
        def _setup() -> tuple[Path, DagMultimodalMaterializer]:
            scratch_dir = tmp_path / "media"
            dag_file = _write_dag(
                tmp_path,
                [
                    {
                        "session_id": "root",
                        "turns": [
                            {
                                "media_segments": [
                                    {
                                        "media_hash": "shared_img",
                                        "kind": "image",
                                        "width": 32,
                                        "height": 32,
                                    }
                                ],
                                "in_text_tokens": 4,
                            }
                        ],
                    }
                ],
                name="shared_populate_lock.dag.jsonl",
            )
            mat = DagMultimodalMaterializer(
                media_scratch_dir=scratch_dir,
                media_base_url="http://127.0.0.1:9001",
            )
            return dag_file, mat

        path, shared_mat = await asyncio.to_thread(_setup)
        real_stage_lock = shared_mat._stage_lock
        seen_lock_threads: set[int] = set()
        meta_lock = threading.Lock()
        setter_attempted_lock = threading.Event()
        entered_populate = threading.Event()
        release_populate = threading.Event()
        populate_tid: int | None = None

        class _TrackingRLock:
            def __enter__(self) -> Any:
                with meta_lock:
                    seen_lock_threads.add(threading.get_ident())
                    if len(seen_lock_threads) >= 2:
                        setter_attempted_lock.set()
                return real_stage_lock.__enter__()

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
                return real_stage_lock.__exit__(exc_type, exc, tb)

        shared_mat._stage_lock = _TrackingRLock()  # type: ignore[assignment]
        loader1 = DagJsonlLoader(path, materializer=shared_mat)
        orig_populate = loader1._populate_conversation

        def _probe_nonblocking_acquire() -> tuple[int, bool]:
            acquired = real_stage_lock.acquire(blocking=False)
            if acquired:
                real_stage_lock.release()
            return threading.get_ident(), acquired

        def _paused_populate(*args: Any, **kwargs: Any) -> None:
            nonlocal populate_tid
            populate_tid = threading.get_ident()
            entered_populate.set()
            assert release_populate.wait(timeout=5.0)
            orig_populate(*args, **kwargs)

        with patch.object(
            loader1, "_populate_conversation", side_effect=_paused_populate
        ):
            load_task = asyncio.create_task(loader1.load_async())
            assert await asyncio.to_thread(entered_populate.wait, 5.0) is True
            probe_tid, lock_acquired = await asyncio.to_thread(
                _probe_nonblocking_acquire
            )
            assert probe_tid != populate_tid
            assert lock_acquired is False
            setter_task = asyncio.create_task(
                asyncio.to_thread(
                    shared_mat.set_media_base_url, "http://127.0.0.1:9002/"
                )
            )
            assert await asyncio.to_thread(setter_attempted_lock.wait, 5.0) is True
            release_populate.set()
            convs1 = await load_task
            await setter_task

        url1 = convs1[0].turns[0].raw_messages[0]["content"][1]["image_url"]["url"]
        assert url1.startswith("http://127.0.0.1:9001/")
        assert shared_mat.media_base_url == "http://127.0.0.1:9002"

        convs2 = await DagJsonlLoader(path, materializer=shared_mat).load_async()
        url2 = convs2[0].turns[0].raw_messages[0]["content"][1]["image_url"]["url"]
        assert url2.startswith("http://127.0.0.1:9002/")

    @pytest.mark.asyncio
    async def test_render_media_parts_new_uuid_retries_atomically_on_base_url_update(
        self, tmp_path: Path
    ) -> None:
        seg_u1 = DagMediaSegment(
            media_hash="img_race",
            kind="image",
            width=32,
            height=32,
            uuid="u1",
        )
        seg_u2 = DagMediaSegment(
            media_hash="img_race",
            kind="image",
            width=32,
            height=32,
            uuid="u2",
        )
        mat = await asyncio.to_thread(
            lambda: DagMultimodalMaterializer(
                media_scratch_dir=tmp_path,
                media_base_url="http://127.0.0.1:9001",
                emit_media_uuid=True,
            )
        )
        await mat.async_stage_turns(
            [DagTurn(media_segments=[seg_u1], in_text_tokens=4)]
        )
        mat.freeze_for_timed_replay(True)

        entered_build = threading.Event()
        release_build = threading.Event()
        u2_calls = 0

        def _paused_first_build(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal u2_calls
            if args[1].uuid == "u2":
                u2_calls += 1
                if u2_calls == 1:
                    entered_build.set()
                    assert release_build.wait(timeout=5.0)
            return _build_wire_parts_from_uris(*args, **kwargs)

        with patch(
            "aiperf.dataset.loader.dag_jsonl_media._build_wire_parts_from_uris",
            side_effect=_paused_first_build,
        ):
            render_task = asyncio.create_task(
                asyncio.to_thread(mat.render_media_parts, seg_u2)
            )
            assert await asyncio.to_thread(entered_build.wait, 5.0) is True
            await asyncio.to_thread(mat.set_media_base_url, "http://127.0.0.1:9002/")
            release_build.set()
            parts = await render_task
            u1_parts = await asyncio.to_thread(mat.render_media_parts, seg_u1)

        assert u2_calls == 2
        assert mat.encode_calls_during_timed_replay == 0
        assert len(parts) == 1
        assert parts[0]["uuid"] == "u2"
        assert parts[0]["image_url"]["url"].startswith("http://127.0.0.1:9002/")
        assert u1_parts[0]["uuid"] == "u1"
        assert u1_parts[0]["image_url"]["url"].startswith("http://127.0.0.1:9002/")
        assert list(mat._media_cache.keys()) == [mat._wire_cache_key(seg_u2)]

    @pytest.mark.asyncio
    async def test_freeze_for_timed_replay_serializes_with_uncached_render_synthesis(
        self, tmp_path: Path
    ) -> None:
        seg_in_flight = DagMediaSegment(
            media_hash="img_in_flight",
            kind="image",
            width=32,
            height=32,
        )
        seg_after_freeze = DagMediaSegment(
            media_hash="img_after_freeze",
            kind="image",
            width=32,
            height=32,
        )
        mat = await asyncio.to_thread(
            lambda: DagMultimodalMaterializer(media_scratch_dir=tmp_path)
        )
        real_stage_lock = mat._stage_lock
        seen_lock_threads: set[int] = set()
        meta_lock = threading.Lock()
        freeze_attempted_lock = threading.Event()
        orig_compute = mat._compute_segment_uris
        entered_compute = threading.Event()
        release_compute = threading.Event()
        freeze_finished = threading.Event()
        compute_tid: int | None = None
        observed_frozen_in_compute: list[bool] = []

        class _TrackingRLock:
            def __enter__(self) -> Any:
                with meta_lock:
                    seen_lock_threads.add(threading.get_ident())
                    if len(seen_lock_threads) >= 2:
                        freeze_attempted_lock.set()
                return real_stage_lock.__enter__()

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
                return real_stage_lock.__exit__(exc_type, exc, tb)

        mat._stage_lock = _TrackingRLock()  # type: ignore[assignment]

        def _probe_nonblocking_acquire() -> tuple[int, bool]:
            acquired = real_stage_lock.acquire(blocking=False)
            if acquired:
                real_stage_lock.release()
            return threading.get_ident(), acquired

        def _paused_compute(seg: DagMediaSegment) -> tuple[str, ...]:
            nonlocal compute_tid
            compute_tid = threading.get_ident()
            entered_compute.set()
            assert release_compute.wait(timeout=5.0)
            observed_frozen_in_compute.append(mat.is_frozen_for_timed_replay)
            return orig_compute(seg)

        def _freeze_worker() -> None:
            mat.freeze_for_timed_replay(True)
            freeze_finished.set()

        with patch.object(mat, "_compute_segment_uris", side_effect=_paused_compute):
            render_task = asyncio.create_task(
                asyncio.to_thread(mat.render_media_parts, seg_in_flight)
            )
            assert await asyncio.to_thread(entered_compute.wait, 5.0) is True
            probe_tid, lock_acquired = await asyncio.to_thread(
                _probe_nonblocking_acquire
            )
            assert probe_tid != compute_tid
            assert lock_acquired is False

            freeze_task = asyncio.create_task(asyncio.to_thread(_freeze_worker))
            assert await asyncio.to_thread(freeze_attempted_lock.wait, 5.0) is True
            assert freeze_finished.is_set() is False
            release_compute.set()
            parts = await render_task
            await freeze_task

            assert len(parts) == 1
            assert observed_frozen_in_compute == [False]
            assert mat.is_frozen_for_timed_replay is True
            assert mat.encode_calls_during_timed_replay == 0

            with pytest.raises(RuntimeError, match="frozen timed replay"):
                await asyncio.to_thread(mat.render_media_parts, seg_after_freeze)
            assert mat.encode_calls_during_timed_replay == 1
            assert observed_frozen_in_compute == [False]

    def test_set_media_base_url_frozen_contract_rebases_scratch_and_rejects_inline(
        self, tmp_path: Path
    ) -> None:
        scratch = tmp_path / "scratch"
        seg_staged = DagMediaSegment(
            media_hash="img_staged_rebase",
            kind="image",
            width=32,
            height=32,
            uuid="u_staged",
        )
        vid_staged = DagMediaSegment(
            media_hash="vid_staged_rebase",
            kind="video",
            width=32,
            height=32,
            fps=12.0,
            num_frames=2,
        )
        seg_new_uuid = DagMediaSegment(
            media_hash="img_staged_rebase",
            kind="image",
            width=32,
            height=32,
            uuid="u_rebased",
        )
        seg_unstaged = DagMediaSegment(
            media_hash="img_never_staged",
            kind="image",
            width=32,
            height=32,
        )
        dag_path = _write_dag(
            tmp_path,
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "media_segments": [
                                seg_staged.model_dump(exclude_none=True),
                                vid_staged.model_dump(exclude_none=True),
                            ],
                            "in_text_tokens": 4,
                        }
                    ],
                }
            ],
            name="rebase_frozen.dag.jsonl",
        )

        # Stage while media_base_url is None (e.g. before HTTP server port bind),
        # freeze via DagJsonlLoader.load(), then rebase onto an HTTP base URL.
        mat = DagMultimodalMaterializer(
            video_wire_mode="image_frames",
            media_scratch_dir=scratch,
            media_base_url=None,
            emit_media_uuid=True,
        )
        DagJsonlLoader(
            dag_path, materializer=mat, video_wire_mode="image_frames"
        ).load()
        assert mat.is_frozen_for_timed_replay is True

        with patch(
            "aiperf.dataset.loader.dag_jsonl_media.synth_deterministic_png_bytes",
            side_effect=AssertionError("unexpected synthesis during URL rebase"),
        ):
            mat.set_media_base_url("http://127.0.0.1:9101/")
            staged_parts = mat.render_media_parts(seg_staged)
            vid_parts = mat.render_media_parts(vid_staged)
            uuid_parts = mat.render_media_parts(seg_new_uuid)
            mat.set_media_base_url("http://127.0.0.1:9102/")
            rebased_parts = mat.render_media_parts(seg_staged)

        assert staged_parts[0]["uuid"] == "u_staged"
        assert staged_parts[0]["image_url"]["url"].startswith("http://127.0.0.1:9101/")
        assert len(vid_parts) == 2
        assert all(
            p["image_url"]["url"].startswith("http://127.0.0.1:9101/")
            for p in vid_parts
        )
        assert uuid_parts[0]["uuid"] == "u_rebased"
        assert uuid_parts[0]["image_url"]["url"].startswith("http://127.0.0.1:9101/")
        assert rebased_parts[0]["image_url"]["url"].startswith("http://127.0.0.1:9102/")
        assert mat.encode_calls_during_timed_replay == 0

        # Reject switching back to inline data URIs while frozen, leaving HTTP URLs intact.
        with pytest.raises(RuntimeError, match="Cannot clear media_base_url to None"):
            mat.set_media_base_url(None)
        assert mat.media_base_url == "http://127.0.0.1:9102"
        assert mat.render_media_parts(seg_staged)[0]["image_url"]["url"].startswith(
            "http://127.0.0.1:9102/"
        )

        # Unstaged segment still hits the frozen guard.
        with pytest.raises(RuntimeError, match="frozen timed replay"):
            mat.render_media_parts(seg_unstaged)
        assert mat.encode_calls_during_timed_replay == 1

        # Frozen inline materializer without media_scratch_dir rejects HTTP base URLs.
        inline_mat = DagMultimodalMaterializer(video_wire_mode="image_frames")
        DagJsonlLoader(
            dag_path, materializer=inline_mat, video_wire_mode="image_frames"
        ).load()
        assert inline_mat.is_frozen_for_timed_replay is True
        with pytest.raises(RuntimeError, match="requires media_scratch_dir"):
            inline_mat.set_media_base_url("http://127.0.0.1:9103")
