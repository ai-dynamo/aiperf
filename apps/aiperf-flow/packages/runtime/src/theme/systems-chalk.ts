// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bundled Systems Chalk theme definition.

import type { FlowThemeIr } from "@aiperf/flow-schema";

import { BUNDLED_ROOT_BASE } from "./registry.js";
import { deepFreeze } from "./types.js";

const bundledSourceMap = {
  source: "runtime:systems_chalk",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
} as const;

/** Non-role shape values owned by Systems Chalk. */
export const SYSTEMS_CHALK_SHAPE = deepFreeze({
  cornerRadiusPx: 12,
} as const);

/** Complete bundled Systems Chalk root theme. */
export const SYSTEMS_CHALK: FlowThemeIr = deepFreeze({
  id: "systems_chalk",
  extends: BUNDLED_ROOT_BASE,
  sourceMap: bundledSourceMap,
  values: {
    "surface.canvas": { kind: "color", value: "#232526" },
    "surface.panel": { kind: "color", value: "#292C2D" },
    "surface.raised": { kind: "color", value: "#303334" },
    "surface.control": { kind: "color", value: "#383C3E" },
    "ink.primary": { kind: "color", value: "#F1F3F2" },
    "ink.muted": { kind: "color", value: "#AEB4B5" },
    "ink.inverse": { kind: "color", value: "#232526" },
    "line.structural": { kind: "color", value: "#D7DADA" },
    "line.guide": { kind: "color", value: "#777D80" },
    "accent.control": { kind: "color", value: "#71D8D0" },
    "accent.execution": { kind: "color", value: "#69C8BA" },
    "accent.compute": { kind: "color", value: "#77B8DE" },
    "accent.attention": { kind: "color", value: "#F0CF58" },
    "accent.success": { kind: "color", value: "#7DCE82" },
    "accent.danger": { kind: "color", value: "#F07972" },
    "accent.focus": { kind: "color", value: "#9BDBF5" },
    "font.display": {
      kind: "font",
      value: ["Nunito Sans", "Segoe UI", "sans-serif"],
    },
    "font.body": {
      kind: "font",
      value: ["Nunito Sans", "Segoe UI", "sans-serif"],
    },
    "font.data": {
      kind: "font",
      value: ["IBM Plex Mono", "Cascadia Code", "monospace"],
    },
    "weight.regular": { kind: "number", value: 400 },
    "weight.label": { kind: "number", value: 500 },
    "weight.emphasis": { kind: "number", value: 600 },
    "size.caption": { kind: "number", value: 11 },
    "size.body": { kind: "number", value: 13 },
    "size.label": { kind: "number", value: 12 },
    "size.title": { kind: "number", value: 18 },
    "stroke.hairline": { kind: "number", value: 1 },
    "stroke.standard": { kind: "number", value: 2 },
    "stroke.emphasis": { kind: "number", value: 3 },
    "stroke.cap": { kind: "enum", value: "round" },
    "stroke.join": { kind: "enum", value: "round" },
    "motion.draw": { kind: "duration", valueMs: 420 },
    "motion.enter": { kind: "duration", valueMs: 240 },
    "motion.emphasis": { kind: "duration", valueMs: 180 },
    "motion.stagger": { kind: "duration", valueMs: 60 },
    "motion.easing": { kind: "enum", value: "ease_out" },
  },
});
