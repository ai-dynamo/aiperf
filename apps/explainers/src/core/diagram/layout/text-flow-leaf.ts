// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Adapter bridging the existing `wrapTextToWidth` text measurer into the flow
 * engine's `FlowNode["measure"]` contract, so wrapped text can act as a leaf.
 */

import { wrapTextToWidth } from "../text-metrics.js";
import type { FlowConstraint, FlowNode, FlowSize } from "./flow-engine.js";

const DEFAULT_LINE_HEIGHT_RATIO = 1.3;

/**
 * Bridges the `wrapTextToWidth` measurer into the flow engine's
 * `FlowNode["measure"]` contract: given a width constraint, wraps `text` and
 * reports the height that many lines actually need, using the same line-height
 * convention `SceneRenderer` applies at paint time (kept in one place to avoid
 * the measure/paint divergence risk flagged during the wrap-fix effort).
 *
 * Line-height convention (confirmed against `SceneRenderer.tsx`'s text branch):
 * `SceneRenderer` resolves `fontSize = scaledSceneFontSize(node.style?.fontSize)`
 * (already scaled by `SCENE_TEXT_SCALE`), wraps with that SAME scaled font size,
 * and stacks lines at `lineHeight = fontSize * 1.3` on that already-scaled value.
 * So `fontSize` here MUST already be scaled by the caller (Tasks 3-5); this
 * adapter multiplies it by `lineHeightRatio` (default 1.3) exactly as the
 * renderer does — do not pass an unscaled authored font size.
 */
export function textFlowLeaf(
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
  lineHeightRatio: number = DEFAULT_LINE_HEIGHT_RATIO,
): NonNullable<FlowNode["measure"]> {
  return (constraint: FlowConstraint): FlowSize => {
    const lines = wrapTextToWidth(text, constraint.maxWidth, fontSize, weight);
    const lineCount = Math.max(lines.length, 1);
    return {
      width: constraint.maxWidth,
      height: lineCount * fontSize * lineHeightRatio,
    };
  };
}
