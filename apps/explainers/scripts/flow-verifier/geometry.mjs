/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Default SceneRenderer viewport. */
export const DEFAULT_VIEWPORT = Object.freeze({
  width: 700,
  height: 400,
  margin: 24,
});

/** Snap distance (px) for connector endpoint / dot proximity. */
export const SNAP_PX = 36;

const ARROW_CAPS = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
]);
const ARROW_KINDS = new Set(["line", "path", "arrow", "connector"]);
const DOT_CAPS = new Set(["core.dot", "core.circle"]);
const DOT_KINDS = new Set(["dot", "circle"]);
const BOX_CAPS = new Set(["core.rect", "core.text"]);

export function capabilityOf(node) {
  return String(node?.capabilityId ?? node?.capability ?? "");
}

export function kindOf(node) {
  return String(node?.kind ?? "");
}

export function isArrowLike(node) {
  const cap = capabilityOf(node);
  const kind = kindOf(node);
  return ARROW_CAPS.has(cap) || ARROW_KINDS.has(kind);
}

export function isDotLike(node) {
  const cap = capabilityOf(node);
  const kind = kindOf(node);
  if (DOT_CAPS.has(cap) || DOT_KINDS.has(kind)) return true;
  const r = node?.style?.r;
  return typeof r === "number" && r > 0 && r <= 12;
}

export function isBoxLike(node) {
  const cap = capabilityOf(node);
  if (BOX_CAPS.has(cap)) return true;
  if (isArrowLike(node) || isDotLike(node)) return false;
  return Boolean(node?.geometry || node?.layout);
}

export function walkNodes(roots) {
  const out = [];
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    out.push(node);
    const kids = node.children;
    if (Array.isArray(kids)) {
      for (const child of kids) visit(child);
    }
  };
  for (const root of roots ?? []) visit(root);
  return out;
}

export function nodeIds(roots) {
  return new Set(
    walkNodes(roots)
      .map((n) => n.id)
      .filter((id) => typeof id === "string" && id.length > 0),
  );
}

export function geomOf(node) {
  const g = node?.geometry ?? node?.layout;
  if (!g || typeof g !== "object") return null;
  const x = Number(g.x);
  const y = Number(g.y);
  const w = Number(g.width);
  const h = Number(g.height);
  if (![x, y, w, h].every(Number.isFinite)) return null;
  return { x, y, width: w, height: h };
}

export function boxCenter(geom) {
  return { x: geom.x + geom.width / 2, y: geom.y + geom.height / 2 };
}

export function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

export function pointNearBox(point, geom, snap = SNAP_PX) {
  const cx = Math.min(Math.max(point.x, geom.x), geom.x + geom.width);
  const cy = Math.min(Math.max(point.y, geom.y), geom.y + geom.height);
  return dist(point, { x: cx, y: cy }) <= snap;
}

/**
 * Parse simple SVG path commands into polyline points (M/L/H/V/absolute).
 * Enough for decks-flow authored connectors.
 */
export function pathPoints(pathData) {
  if (typeof pathData !== "string" || pathData.trim() === "") return [];
  const tokens = pathData.match(/[MLHVZmlhvz]|-?\d*\.?\d+(?:e[-+]?\d+)?/gi);
  if (!tokens) return [];
  const points = [];
  let i = 0;
  let x = 0;
  let y = 0;
  let cmd = "M";
  const num = () => Number(tokens[i++]);
  while (i < tokens.length) {
    const t = tokens[i];
    if (/^[MLHVZmlhvz]$/.test(t)) {
      cmd = t;
      i += 1;
      if (cmd === "Z" || cmd === "z") continue;
    }
    if (cmd === "M" || cmd === "L") {
      x = num();
      y = num();
      points.push({ x, y });
      cmd = cmd === "M" ? "L" : cmd;
    } else if (cmd === "m" || cmd === "l") {
      x += num();
      y += num();
      points.push({ x, y });
      cmd = cmd === "m" ? "l" : cmd;
    } else if (cmd === "H") {
      x = num();
      points.push({ x, y });
    } else if (cmd === "h") {
      x += num();
      points.push({ x, y });
    } else if (cmd === "V") {
      y = num();
      points.push({ x, y });
    } else if (cmd === "v") {
      y += num();
      points.push({ x, y });
    } else {
      i += 1;
    }
  }
  return points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
}

export function arrowPathData(node) {
  if (typeof node?.path === "string" && node.path.trim()) return node.path;
  if (typeof node?.d === "string" && node.d.trim()) return node.d;
  const from = node?.from;
  const to = node?.to;
  if (
    from &&
    to &&
    Number.isFinite(from.x) &&
    Number.isFinite(from.y) &&
    Number.isFinite(to.x) &&
    Number.isFinite(to.y) &&
    !(from.x === 0 && from.y === 0 && to.x === 0 && to.y === 0)
  ) {
    return `M${from.x} ${from.y} L${to.x} ${to.y}`;
  }
  return null;
}

export function timelineDurationMs(timeline) {
  let max = 0;
  for (const cue of timeline ?? []) {
    const at = Number(cue.at) || 0;
    const dur = Number(cue.duration) || 0;
    max = Math.max(max, at + dur);
  }
  return max;
}

export function isDrawAction(action) {
  const a = String(action ?? "").toLowerCase();
  return a === "draw" || a === "trace" || a === "reveal-stroke";
}

export function drawProgress(timeline, nodeId, tMs) {
  const cues = (timeline ?? []).filter(
    (c) => c.target === nodeId && isDrawAction(c.action),
  );
  const cue = cues.at(-1);
  if (!cue) return undefined;
  const at = Number(cue.at) || 0;
  const dur = Number(cue.duration) || 0;
  if (tMs <= at) return 0;
  if (dur <= 0) return 1;
  return Math.min(1, Math.max(0, (tMs - at) / dur));
}

export function inViewport(geom, viewport = DEFAULT_VIEWPORT) {
  const { width, height, margin } = viewport;
  return (
    geom.x + geom.width >= -margin &&
    geom.y + geom.height >= -margin &&
    geom.x <= width + margin &&
    geom.y <= height + margin
  );
}
