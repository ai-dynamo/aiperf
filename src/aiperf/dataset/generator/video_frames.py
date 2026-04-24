# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic video frame generators (moving shapes, grid clock, noise)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def generate_moving_shapes_frames(
    total_frames: int, width: int, height: int
) -> list[Image.Image]:
    """Generate frames with moving geometric shapes."""
    shapes = [
        {
            "type": "circle",
            "color": (255, 0, 0),  # Red circle
            "size": 30,
            "start_x": 0,
            "start_y": height // 2,
            "dx": width / total_frames * 2,  # Move across screen in half duration
            "dy": 0,
        },
        {
            "type": "rectangle",
            "color": (0, 255, 0),  # Green rectangle
            "size": 25,
            "start_x": width // 2,
            "start_y": 0,
            "dx": 0,
            "dy": height / total_frames * 2,  # Move down
        },
        {
            "type": "circle",
            "color": (0, 0, 255),  # Blue circle
            "size": 20,
            "start_x": width,
            "start_y": height,
            "dx": -width / total_frames * 1.5,  # Move diagonally
            "dy": -height / total_frames * 1.5,
        },
    ]

    frames = []
    for frame_num in range(total_frames):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for shape in shapes:
            x = shape["start_x"] + shape["dx"] * frame_num
            y = shape["start_y"] + shape["dy"] * frame_num

            # Wrap around screen edges
            x = x % width
            y = y % height

            size = shape["size"]
            color = shape["color"]

            if shape["type"] == "circle":
                draw.ellipse(
                    [x - size // 2, y - size // 2, x + size // 2, y + size // 2],
                    fill=color,
                )
            elif shape["type"] == "rectangle":
                draw.rectangle(
                    [x - size // 2, y - size // 2, x + size // 2, y + size // 2],
                    fill=color,
                )

        frames.append(img)

    return frames


def generate_grid_clock_frames(
    total_frames: int, width: int, height: int
) -> list[Image.Image]:
    """Generate frames with a grid and clock-like animation."""
    frames = []
    for frame_num in range(total_frames):
        img = Image.new("RGB", (width, height), (32, 32, 32))
        draw = ImageDraw.Draw(img)

        # Draw grid
        grid_size = 32
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(64, 64, 64), width=1)
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(64, 64, 64), width=1)

        # Draw clock hands
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4

        # Frame-based rotation
        angle = (frame_num / total_frames) * 2 * math.pi

        # Hour hand (slower)
        hour_angle = angle / 12
        hour_x = center_x + radius * 0.6 * math.cos(hour_angle - math.pi / 2)
        hour_y = center_y + radius * 0.6 * math.sin(hour_angle - math.pi / 2)
        draw.line([(center_x, center_y), (hour_x, hour_y)], fill=(255, 255, 0), width=3)

        # Minute hand
        min_x = center_x + radius * 0.9 * math.cos(angle - math.pi / 2)
        min_y = center_y + radius * 0.9 * math.sin(angle - math.pi / 2)
        draw.line([(center_x, center_y), (min_x, min_y)], fill=(255, 255, 255), width=2)

        # Clock face circle
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            outline=(128, 128, 128),
            width=2,
        )

        # Center dot
        draw.ellipse(
            [center_x - 3, center_y - 3, center_x + 3, center_y + 3],
            fill=(255, 0, 0),
        )

        # Add frame number in corner
        draw.text((10, 10), f"Frame {frame_num}", fill=(255, 255, 255))

        frames.append(img)

    return frames


def generate_noise_frames(
    total_frames: int, width: int, height: int, noise_rng: np.random.Generator
) -> list[Image.Image]:
    """Generate frames with random noise pixels."""
    return [
        Image.fromarray(noise_rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        for _ in range(total_frames)
    ]
