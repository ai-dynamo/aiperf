# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multimodal ``dag_jsonl`` deterministic media synthesis & replay."""

from __future__ import annotations

import base64
import subprocess
import urllib.request
from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
from pydantic import ValidationError

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
    compute_mixed_block_hashes,
    find_ffmpeg_binary,
    synth_deterministic_png_bytes,
    validate_dag_zero_leakage,
    verify_mp4_decodable,
    verify_png_bytes,
    write_deterministic_h264_mp4,
)
from aiperf.dataset.loader.dag_jsonl_models import (
    DagMediaSegment,
    DagPromptSpan,
    DagTurn,
)
from aiperf.workers.session_manager import UserSession
from aiperf.workers.worker import _apply_cache_bust


def _write_dag(tmp_path: Path, rows: list[dict], name: str = "mm.dag.jsonl") -> Path:
    out = tmp_path / name
    out.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")
    return out


class TestZeroLeakageValidation:
    """Verify structural zero-leakage traces contain zero raw text or pixels."""

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
            ({"session_id": "s", "messages": []}, "forbidden key 'messages'"),
            ({"session_id": "s", "raw_text": "hello"}, "forbidden key 'raw_text'"),
            (
                {"session_id": "s", "uri": "data:image/png;base64,AAAA"},
                "inline media URI",
            ),
            (
                {"session_id": "s", "uri": "data:video/mp4;base64,AAAA"},
                "inline media URI",
            ),
            ({"session_id": "s", "blob": "x" * 257}, "oversized raw string"),
        ],
    )
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


class TestDeterministicSynthesisAndBlockHashes:
    """Verify deterministic PNG/MP4 synthesis and mixed block hashing."""

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
        # System rubric blocks (first 2 blocks = 128 tokens) match, video blocks diverge
        assert hs[:2] == h0[:2]
        assert hs[2:] != h0[2:]

    def test_mp4_synthesis_with_mocked_ffmpeg_pipeline(self, tmp_path: Path) -> None:
        seg = DagMediaSegment(
            media_hash="vid_mock",
            kind="video",
            width=64,
            height=64,
            fps=12.0,
            num_frames=4,
        )
        expected_raw = b"\x80" * (4 * 64 * 64 * 3)

        def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if (
                "-f" in cmd
                and cmd[cmd.index("-f") + 1] == "rawvideo"
                and cmd[-1] == "-"
            ):
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=expected_raw, stderr=b""
                )
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"\x00" * 256)
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

        out_mp4 = tmp_path / "vid_mock.mp4"
        with patch("subprocess.run", side_effect=_fake_run):
            written = write_deterministic_h264_mp4(
                out_mp4, seg, ffmpeg_bin="/usr/bin/ffmpeg"
            )
            assert written == out_mp4
            assert out_mp4.is_file()

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
            fps=24.0,
            num_frames=6,
        )
        mp4_a = write_deterministic_h264_mp4(tmp_path / "a.mp4", seg, ffmpeg_bin=ff)
        mp4_b = write_deterministic_h264_mp4(tmp_path / "b.mp4", seg, ffmpeg_bin=ff)
        verify_mp4_decodable(
            mp4_a,
            expected_frames=6,
            expected_width=128,
            expected_height=96,
            ffmpeg_bin=ff,
        )
        assert mp4_a.read_bytes() == mp4_b.read_bytes()


class TestDagLoaderMultimodalDeduplicationAndReplay:
    """Verify cross-turn/branch media_hash reuse, HTTP staging, and cache-bust UUIDs."""

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
        with LocalMediaHttpServer(scratch) as media_srv:
            loader = DagJsonlLoader(
                dag_path,
                media_scratch_dir=scratch,
                media_base_url=media_srv.base_url,
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

            # Shared system rubric across SPAWN siblings is byte-identical
            sys_a = convs["eval-a"].turns[0].raw_messages[0]
            sys_b = convs["eval-b"].turns[0].raw_messages[0]
            assert sys_a == sys_b

            # Shared image URL across root, SPAWN siblings, and FORK child matches
            root_img_part = convs["planner-root"].turns[0].raw_messages[1]["content"][1]
            eval_a_img_part = convs["eval-a"].turns[0].raw_messages[1]["content"][0]
            fork_img_part = convs["fork-critic"].turns[0].raw_messages[0]["content"][1]
            assert root_img_part == eval_a_img_part == fork_img_part
            assert root_img_part["uuid"] == "shared_ref_img"

            # Verify live HTTP fetch against LocalMediaHttpServer decodes cleanly
            img_url = root_img_part["image_url"]["url"]
            assert img_url.startswith(media_srv.base_url)
            with urllib.request.urlopen(img_url, timeout=5) as resp:
                fetched_png = resp.read()
            verify_png_bytes(fetched_png, expected_width=128, expected_height=128)

            # Verify seamless integration with worker.py's _apply_cache_bust media_uuids
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
            # First part is the injected text prefix marker; second is the suffixed image part
            img_parts = [p for p in busted_parts if p.get("type") == "image_url"]
            assert img_parts[0]["uuid"].endswith("-deadbeef1234")

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

        # Materializer is frozen after load_dataset(); unstaged synthesis fails loud
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
