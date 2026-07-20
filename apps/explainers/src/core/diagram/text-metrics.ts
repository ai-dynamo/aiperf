/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic, scale-aware text metrics for scene layout and rendering.

export const SCENE_TEXT_SCALE = 0.9;
export const DEFAULT_SCENE_FONT_SIZE = 14;
export const CHAR_WIDTH = 6.2;
export const BOLD_CHAR_WIDTH = 6.2;
export const INSET = 8;
export const TITLE_HEIGHT = 22;
export const DETAIL_HEIGHT = 20;
export const SUBTITLE_HEIGHT = 16;
export const CHIP_PAD_X = 24;
export const STEPPER_MIN_CHIP_WIDTH = 72;
export const STEPPER_CHIP_HEIGHT = 26;
export const STEPPER_CHIP_PAD = 24;

export function scaledSceneFontSize(
  value: unknown,
  fallback = DEFAULT_SCENE_FONT_SIZE,
): number {
  const fontSize =
    typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return fontSize * SCENE_TEXT_SCALE;
}

export function estimateTextWidth(
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): number {
  const unit = weight === "bold" ? BOLD_CHAR_WIDTH : CHAR_WIDTH;
  const ratio = fontSize / 11;
  return Math.ceil(text.length * unit * ratio * SCENE_TEXT_SCALE);
}

export function stepperChipWidth(label: string, index: number): number {
  const text = `${index + 1}. ${label}`;
  return Math.max(
    STEPPER_MIN_CHIP_WIDTH,
    estimateTextWidth(text, 11, "bold") + STEPPER_CHIP_PAD,
  );
}
