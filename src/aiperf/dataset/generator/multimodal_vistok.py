# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual-token-aware media geometry calculator for multimodal VLM benchmarks.

Implements exact ViT patch-grid and spatial/temporal token-merge geometry
matching open-source VLM preprocessors (such as HuggingFace ``Qwen3VLProcessor``
and ``Qwen2_5_VLProcessor``) so trace-driven multimodal benchmarks can calibrate
synthetic image and video dimensions to recorded visual token counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VisionTokenizerSpec:
    """Patch, merge, pixel-budget, and framing parameters for a VLM family."""

    family: str
    """Canonical model-family identifier (for example, ``'qwen3_vl'``)."""

    patch_size: int
    """Spatial ViT patch size in pixels along height and width."""

    spatial_merge_size: int
    """Spatial patch merge factor per dimension before LLM projection."""

    temporal_patch_size: int
    """Number of consecutive frames grouped into one temporal ViT patch."""

    image_min_pixels: int
    """Minimum pixel budget (height * width) for single-image smart resize."""

    image_max_pixels: int
    """Maximum pixel budget (height * width) for single-image smart resize."""

    video_min_pixels: int
    """Minimum total pixel budget (frames * height * width) for video resize."""

    video_max_pixels: int
    """Maximum total pixel budget (frames * height * width) for video resize."""

    per_frame_max_pixels: int | None = None
    """Optional additional per-frame pixel cap applied alongside ``video_max_pixels``."""

    image_framing_tokens: int = 2
    """Special boundary tokens added per image (e.g., ``<|vision_start|>``/``<|vision_end|>``)."""

    video_group_framing_tokens: int = 8
    """Framing/timestamp tokens added per merged temporal group ``grid_t``."""

    video_fixed_framing_tokens: int = 0
    """Fixed framing tokens added per video clip when independent of ``grid_t``."""

    @property
    def spatial_factor(self) -> int:
        """Pixel multiple required per spatial axis (``patch_size * spatial_merge_size``)."""
        return self.patch_size * self.spatial_merge_size


QWEN3_VL_SPEC = VisionTokenizerSpec(
    family="qwen3_vl",
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    image_min_pixels=65_536,
    image_max_pixels=16_777_216,
    video_min_pixels=4_096,
    video_max_pixels=25_165_824,
    per_frame_max_pixels=None,
    image_framing_tokens=2,
    video_group_framing_tokens=8,
    video_fixed_framing_tokens=0,
)

QWEN2_5_VL_SPEC = VisionTokenizerSpec(
    family="qwen2_5_vl",
    patch_size=14,
    spatial_merge_size=2,
    temporal_patch_size=2,
    image_min_pixels=3_136,
    image_max_pixels=12_845_056,
    video_min_pixels=3_136,
    video_max_pixels=19_267_584,
    per_frame_max_pixels=None,
    image_framing_tokens=2,
    video_group_framing_tokens=0,
    video_fixed_framing_tokens=2,
)


def get_vision_tokenizer_spec(model_family: str) -> VisionTokenizerSpec:
    """Return the built-in :class:`VisionTokenizerSpec` for ``model_family``."""
    normalized = model_family.strip().lower()
    if normalized in ("qwen3_vl", "qwen3_5_vl", "qwen35_vl"):
        return QWEN3_VL_SPEC
    if normalized in ("qwen2_5_vl", "qwen25_vl", "qwen2_vl"):
        return QWEN2_5_VL_SPEC
    raise ValueError(
        f"Unsupported vision tokenizer model_family {model_family!r}; "
        "expected 'qwen3_vl' or 'qwen2_5_vl' (or pass an explicit VisionTokenizerSpec)"
    )


@dataclass(frozen=True, slots=True)
class VisualTokenGrid:
    """Computed post-resize geometry and merged visual token count."""

    kind: Literal["image", "video"]
    """Media modality (``'image'`` or ``'video'``)."""

    model_family: str
    """Model family identifier used to compute the token grid."""

    src_width: int
    """Source media width in pixels prior to smart resize."""

    src_height: int
    """Source media height in pixels prior to smart resize."""

    frames_in_vit: int
    """Number of temporally padded frames entering the ViT encoder."""

    resized_width: int
    """Post-resize width in pixels aligned to ``spatial_factor``."""

    resized_height: int
    """Post-resize height in pixels aligned to ``spatial_factor``."""

    grid_t: int
    """Merged temporal grid dimension (``frames_in_vit // temporal_patch_size``)."""

    grid_h: int
    """Unmerged spatial patch grid height (``resized_height // patch_size``)."""

    grid_w: int
    """Unmerged spatial patch grid width (``resized_width // patch_size``)."""

    vis_tokens: int
    """Merged visual token count projected into the LLM context."""

    framing_overhead_tokens: int
    """Vision boundary and timestamp framing token overhead."""

    budget_capped: bool
    """True when smart resize downscaled the media to honor ``max_pixels``."""

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        """Return ``(grid_t, grid_h, grid_w)`` matching VLM processor output."""
        return (self.grid_t, self.grid_h, self.grid_w)

    @property
    def total_mm_tokens(self) -> int:
        """Total multimodal prompt tokens (``vis_tokens + framing_overhead_tokens``)."""
        return self.vis_tokens + self.framing_overhead_tokens


def _round_by_factor(value: float, factor: int) -> int:
    return max(factor, int(round(value / factor)) * factor)


def _floor_by_factor(value: float, factor: int) -> int:
    return max(factor, int(math.floor(value / factor)) * factor)


def _ceil_by_factor(value: float, factor: int) -> int:
    return max(factor, int(math.ceil(value / factor)) * factor)


def smart_resize_image(
    width: int,
    height: int,
    *,
    factor: int = 32,
    min_pixels: int = 65_536,
    max_pixels: int = 16_777_216,
) -> tuple[int, int, bool]:
    """Compute aspect-preserving ``(resized_width, resized_height, budget_capped)`` for an image."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Image dimensions must be positive integers, got {width}x{height}"
        )
    if factor <= 0:
        raise ValueError(f"Spatial resize factor must be positive, got {factor}")
    hb = _round_by_factor(height, factor)
    wb = _round_by_factor(width, factor)
    capped = False
    if hb * wb > max_pixels:
        capped = True
        beta = math.sqrt((height * width) / max_pixels)
        hb = _floor_by_factor(height / beta, factor)
        wb = _floor_by_factor(width / beta, factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        hb = _ceil_by_factor(height * beta, factor)
        wb = _ceil_by_factor(width * beta, factor)
    return wb, hb, capped


def smart_resize_video(
    num_frames: int,
    width: int,
    height: int,
    *,
    factor: int = 32,
    temporal_patch: int = 2,
    min_pixels: int = 4_096,
    max_pixels: int = 25_165_824,
    per_frame_max_pixels: int | None = None,
) -> tuple[int, int, int, bool]:
    """Compute ``(padded_frames, resized_width, resized_height, budget_capped)`` for a video."""
    if num_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Video geometry must be positive, got frames={num_frames}, {width}x{height}"
        )
    if factor <= 0 or temporal_patch <= 0:
        raise ValueError(
            f"Resize factor and temporal_patch must be positive, got {factor}, {temporal_patch}"
        )
    remainder = num_frames % temporal_patch
    t_padded = (
        num_frames if remainder == 0 else num_frames + (temporal_patch - remainder)
    )
    hb = _round_by_factor(height, factor)
    wb = _round_by_factor(width, factor)

    effective_budget = max_pixels
    if per_frame_max_pixels is not None:
        effective_budget = min(effective_budget, t_padded * per_frame_max_pixels)

    capped = (t_padded * hb * wb) > effective_budget
    if capped:
        beta = math.sqrt((t_padded * height * width) / effective_budget)
        hb = _floor_by_factor(height / beta, factor)
        wb = _floor_by_factor(width / beta, factor)
    elif (t_padded * hb * wb) < min_pixels:
        beta = math.sqrt(min_pixels / (t_padded * height * width))
        hb = _ceil_by_factor(height * beta, factor)
        wb = _ceil_by_factor(width * beta, factor)
    return t_padded, wb, hb, capped


def sample_video_frame_count(
    src_frames: int,
    src_fps: float,
    *,
    target_fps: float | None = 2.0,
    explicit_frames: int | None = None,
    min_frames: int = 4,
    max_frames: int = 768,
    temporal_patch: int = 2,
) -> int:
    """Compute temporally aligned frame count entering ViT under FPS or explicit sampling."""
    if src_frames <= 0:
        raise ValueError(f"src_frames must be positive, got {src_frames}")
    if temporal_patch <= 0:
        raise ValueError(f"temporal_patch must be positive, got {temporal_patch}")
    if explicit_frames is not None:
        if explicit_frames <= 0:
            raise ValueError(f"explicit_frames must be positive, got {explicit_frames}")
        n_frames = max(temporal_patch, min(explicit_frames, max_frames))
    else:
        fps_eff = target_fps if (target_fps is not None and target_fps > 0) else 2.0
        duration_s = src_frames / src_fps if src_fps > 0 else src_frames / 24.0
        raw = int(math.floor(duration_s * fps_eff))
        n_frames = max(min_frames, min(raw, src_frames, max_frames))
    remainder = n_frames % temporal_patch
    if remainder != 0:
        n_frames += temporal_patch - remainder
    return n_frames


def compute_image_visual_tokens(
    width: int,
    height: int,
    *,
    model_family: str = "qwen3_vl",
    spec: VisionTokenizerSpec | None = None,
    max_pixels: int | None = None,
) -> VisualTokenGrid:
    """Compute the exact post-resize :class:`VisualTokenGrid` for a single image."""
    resolved = spec if spec is not None else get_vision_tokenizer_spec(model_family)
    eff_max_px = max_pixels if max_pixels is not None else resolved.image_max_pixels
    rw, rh, capped = smart_resize_image(
        width,
        height,
        factor=resolved.spatial_factor,
        min_pixels=resolved.image_min_pixels,
        max_pixels=eff_max_px,
    )
    gh = rh // resolved.patch_size
    gw = rw // resolved.patch_size
    merge = resolved.spatial_merge_size
    vis_tok = (gh // merge) * (gw // merge)
    return VisualTokenGrid(
        kind="image",
        model_family=resolved.family,
        src_width=width,
        src_height=height,
        frames_in_vit=1,
        resized_width=rw,
        resized_height=rh,
        grid_t=1,
        grid_h=gh,
        grid_w=gw,
        vis_tokens=vis_tok,
        framing_overhead_tokens=resolved.image_framing_tokens,
        budget_capped=capped,
    )


def compute_video_visual_tokens(
    width: int,
    height: int,
    *,
    src_frames: int = 120,
    src_fps: float = 24.0,
    sampled_frames: int | None = None,
    target_fps: float | None = 2.0,
    model_family: str = "qwen3_vl",
    spec: VisionTokenizerSpec | None = None,
    max_pixels: int | None = None,
) -> VisualTokenGrid:
    """Compute the exact post-resize :class:`VisualTokenGrid` for a video clip."""
    resolved = spec if spec is not None else get_vision_tokenizer_spec(model_family)
    eff_max_px = max_pixels if max_pixels is not None else resolved.video_max_pixels
    n_vit = sample_video_frame_count(
        src_frames,
        src_fps,
        target_fps=target_fps,
        explicit_frames=sampled_frames,
        temporal_patch=resolved.temporal_patch_size,
    )
    t_pad, rw, rh, capped = smart_resize_video(
        n_vit,
        width,
        height,
        factor=resolved.spatial_factor,
        temporal_patch=resolved.temporal_patch_size,
        min_pixels=resolved.video_min_pixels,
        max_pixels=eff_max_px,
        per_frame_max_pixels=resolved.per_frame_max_pixels,
    )
    gt = t_pad // resolved.temporal_patch_size
    gh = rh // resolved.patch_size
    gw = rw // resolved.patch_size
    merge = resolved.spatial_merge_size
    vis_tok = gt * (gh // merge) * (gw // merge)
    framing = resolved.video_fixed_framing_tokens + (
        gt * resolved.video_group_framing_tokens
    )
    return VisualTokenGrid(
        kind="video",
        model_family=resolved.family,
        src_width=width,
        src_height=height,
        frames_in_vit=t_pad,
        resized_width=rw,
        resized_height=rh,
        grid_t=gt,
        grid_h=gh,
        grid_w=gw,
        vis_tokens=vis_tok,
        framing_overhead_tokens=framing,
        budget_capped=capped,
    )
