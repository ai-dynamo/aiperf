# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual-token geometry for ``dag_jsonl`` media segments.

Mirrors the HuggingFace ``transformers`` Qwen-VL ``smart_resize`` rules, with the
pixel budgets from the published ``Qwen3-VL-8B-Instruct`` and
``Qwen2.5-VL-7B-Instruct`` processor configs, so a recorded visual-token count
maps to a synthetic ``(width, height)`` the processor encodes to exactly that
many merged tokens. Video counts assume every frame reaches the vision encoder
(processor frame sampling disabled).
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

VisionModel = Literal["qwen3_vl", "qwen2_5_vl"]

MAX_RESOLVED_SIDE = 4096


@dataclass(frozen=True, slots=True)
class VisionSpec:
    """Patch geometry and ``(min, max)`` pixel budgets of one processor."""

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    image_pixels: tuple[int, int]
    video_pixels: tuple[int, int]
    video_budget_spans_frames: bool
    """Qwen3-VL budgets ``video_pixels`` over all frames; Qwen2.5-VL per frame."""

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


VISION_SPECS: dict[str, VisionSpec] = {
    "qwen3_vl": VisionSpec(16, 2, 2, (65_536, 16_777_216), (4_096, 25_165_824), True),
    "qwen2_5_vl": VisionSpec(14, 2, 2, (3_136, 12_845_056), (3_136, 12_845_056), False),
}  # fmt: skip


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    pixels: tuple[int, int],
    num_frames: int | None = None,
    temporal_factor: int = 2,
) -> tuple[int, int]:
    """Return HF's resized ``(height, width)``; ``num_frames`` selects the Qwen3-VL video variant."""
    min_pixels, max_pixels = pixels
    frames, t_bar = 1, 1
    if num_frames is not None:
        if num_frames < temporal_factor:
            raise ValueError(
                f"video needs num_frames >= {temporal_factor} (got {num_frames})"
            )
        if height < factor or width < factor:
            scale = max(factor / height, factor / width)
            height, width = int(height * scale), int(width * scale)
        frames = num_frames
        t_bar = round(num_frames / temporal_factor) * temporal_factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def visual_tokens(
    kind: Literal["image", "video"],
    width: int,
    height: int,
    *,
    num_frames: int,
    model: VisionModel,
) -> int:
    """Merged visual-token count ``model``'s processor produces for this geometry."""
    spec = VISION_SPECS[model]
    if kind == "image":
        h, w = smart_resize(height, width, factor=spec.factor, pixels=spec.image_pixels)
        grid_t = 1
    else:
        h, w = smart_resize(
            height,
            width,
            factor=spec.factor,
            pixels=spec.video_pixels,
            num_frames=num_frames if spec.video_budget_spans_frames else None,
            temporal_factor=spec.temporal_patch_size,
        )
        grid_t = math.ceil(num_frames / spec.temporal_patch_size)
    return grid_t * (h // spec.factor) * (w // spec.factor)


@lru_cache(maxsize=1024)
def resolve_dimensions(
    kind: Literal["image", "video"],
    vis_tokens: int,
    *,
    num_frames: int,
    model: VisionModel,
) -> tuple[int, int]:
    """Return the ``(width, height)`` nearest 16:9 that encodes to exactly ``vis_tokens``."""
    spec = VISION_SPECS[model]
    grid_t = 1 if kind == "image" else math.ceil(num_frames / spec.temporal_patch_size)
    if vis_tokens % grid_t:
        raise ValueError(
            f"{kind} vis_tokens={vis_tokens} is not divisible by grid_t={grid_t} "
            f"(ceil(num_frames={num_frames} / {spec.temporal_patch_size}))"
        )
    per_frame, max_side = vis_tokens // grid_t, MAX_RESOLVED_SIDE // spec.factor
    candidates = [
        (cols * spec.factor, per_frame // cols * spec.factor)
        for cols in range(1, max_side + 1)
        if per_frame % cols == 0 and per_frame // cols <= max_side
    ]

    def encodes(w: int, h: int) -> bool:
        return (
            visual_tokens(kind, w, h, num_frames=num_frames, model=model) == vis_tokens
        )

    exact = [(w, h) for w, h in candidates if encodes(w, h)]
    if not exact:
        # smart_resize clamped every unresized grid; search its input preimages.
        sides = range(spec.factor, max_side * spec.factor + 1, spec.factor)
        exact = [(w, h) for w in sides for h in sides if encodes(w, h)]
    if not exact:
        raise ValueError(
            f"{kind} vis_tokens={vis_tokens} is unreachable for {model}: no width x "
            f"height (multiples of {spec.factor}, each <= {MAX_RESOLVED_SIDE}) within "
            f"the processor pixel budget encodes to exactly {vis_tokens} tokens"
        )
    return min(
        exact,
        key=lambda wh: (abs(math.log(wh[0] / wh[1] * 9 / 16)), wh[0] * wh[1], -wh[0]),
    )
