# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for visual-token-aware media geometry helpers."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.dataset.generator.multimodal_vistok import (
    QWEN2_5_VL_SPEC,
    QWEN3_VL_SPEC,
    VisionTokenizerSpec,
    compute_image_visual_tokens,
    compute_video_visual_tokens,
    get_vision_tokenizer_spec,
    sample_video_frame_count,
    smart_resize_image,
    smart_resize_video,
)


@pytest.mark.parametrize(
    "sampled_frames,expected_w,expected_h,expected_vis_tokens,expected_capped",
    [
        param(4, 1280, 704, 1760, False, id="720p-4f-uncapped"),
        param(32, 1152, 640, 11520, True, id="720p-32f-capped"),
        param(40, 1056, 576, 11880, True, id="720p-40f-capped"),
        param(120, 608, 320, 11400, True, id="720p-120f-capped"),
    ],
)  # fmt: skip
def test_compute_video_visual_tokens_qwen3_vl_720p_matches_processor(
    sampled_frames: int,
    expected_w: int,
    expected_h: int,
    expected_vis_tokens: int,
    expected_capped: bool,
) -> None:
    grid = compute_video_visual_tokens(
        1280,
        720,
        sampled_frames=sampled_frames,
        model_family="qwen3_vl",
    )
    assert grid.resized_width == expected_w
    assert grid.resized_height == expected_h
    assert grid.vis_tokens == expected_vis_tokens
    assert grid.budget_capped is expected_capped
    assert grid.grid_t == sampled_frames // 2
    assert grid.framing_overhead_tokens == grid.grid_t * 8
    assert grid.total_mm_tokens == expected_vis_tokens + grid.framing_overhead_tokens


def test_compute_image_visual_tokens_qwen3_vl_and_qwen2_5_vl() -> None:
    q3 = compute_image_visual_tokens(1024, 576, model_family="qwen3_vl")
    assert q3.resized_width == 1024
    assert q3.resized_height == 576
    assert q3.grid_thw == (1, 36, 64)
    assert q3.vis_tokens == 18 * 32
    assert q3.framing_overhead_tokens == 2
    assert q3.total_mm_tokens == 578
    assert q3.budget_capped is False

    q25 = compute_image_visual_tokens(560, 560, model_family="qwen2_5_vl")
    assert q25.resized_width == 560
    assert q25.resized_height == 560
    assert q25.vis_tokens == (560 // 28) * (560 // 28)
    assert q25.framing_overhead_tokens == 2


def test_smart_resize_image_upscales_tiny_and_downscales_huge() -> None:
    up_w, up_h, up_capped = smart_resize_image(
        32, 32, factor=32, min_pixels=65_536, max_pixels=16_777_216
    )
    assert up_w * up_h >= 65_536
    assert up_w % 32 == 0 and up_h % 32 == 0
    assert up_capped is False

    dn_w, dn_h, dn_capped = smart_resize_image(
        8192, 4096, factor=32, min_pixels=65_536, max_pixels=1_048_576
    )
    assert dn_w * dn_h <= 1_048_576
    assert dn_w % 32 == 0 and dn_h % 32 == 0
    assert dn_capped is True


def test_sample_video_frame_count_temporal_padding_and_fps() -> None:
    # 5 seconds at 24 fps sampled at 2 fps -> 10 frames (already even)
    assert sample_video_frame_count(120, 24.0, target_fps=2.0, temporal_patch=2) == 10
    # Odd explicit frame count pads up to temporal_patch=2
    assert sample_video_frame_count(120, 24.0, explicit_frames=7, temporal_patch=2) == 8


def test_custom_vision_tokenizer_spec_with_per_frame_cap() -> None:
    custom = VisionTokenizerSpec(
        family="custom_vlm",
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        image_min_pixels=3_136,
        image_max_pixels=12_845_056,
        video_min_pixels=3_136,
        video_max_pixels=19_267_584,
        per_frame_max_pixels=448 * 448,
        image_framing_tokens=2,
        video_group_framing_tokens=0,
        video_fixed_framing_tokens=2,
    )
    grid = compute_video_visual_tokens(1280, 720, sampled_frames=16, spec=custom)
    assert grid.budget_capped is True
    assert grid.resized_width * grid.resized_height <= 448 * 448
    assert grid.framing_overhead_tokens == 2


def test_vision_tokenizer_spec_lookup_and_invalid_inputs() -> None:
    assert get_vision_tokenizer_spec("qwen3_vl") == QWEN3_VL_SPEC
    assert get_vision_tokenizer_spec("qwen3_5_vl") == QWEN3_VL_SPEC
    assert get_vision_tokenizer_spec("qwen2_5_vl") == QWEN2_5_VL_SPEC

    with pytest.raises(ValueError, match="Unsupported vision tokenizer"):
        get_vision_tokenizer_spec("unknown_vlm")

    with pytest.raises(ValueError, match="positive"):
        smart_resize_image(0, 512)

    with pytest.raises(ValueError, match="positive"):
        smart_resize_video(0, 640, 360)
