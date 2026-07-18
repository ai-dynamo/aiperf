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
const MOTION_CAPS = new Set([
  "motion.signal",
  "motion.dot",
  "core.motion",
  "motion.motion-signal",
]);

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

/**
 * Mirror SceneRenderer motion-signal classification so guide strokes are not
 * treated as orphan connectors. Dots are never motion guides.
 */
export function isMotionSignalNode(node) {
  if (isDotLike(node)) return false;
  const cap = capabilityOf(node);
  if (MOTION_CAPS.has(cap)) return true;
  const id = String(node?.id ?? "");
  if (/motion[-_]?sig/i.test(id) || /^motion\d+$/i.test(id)) return true;
  if (/motion/i.test(id) && isArrowLike(node)) return true;
  const label = String(node?.accessibility?.label ?? "").toLowerCase();
  if (label.includes("motion signal")) return true;
  const style = node?.style ?? {};
  const motion = style.motion;
  const role = style.role;
  return (
    motion === true ||
    motion === 1 ||
    motion === "signal" ||
    motion === "dot" ||
    role === "motion" ||
    role === "motion-signal"
  );
}

/** True when the author disabled arrowheads (undirected divider / guide). */
export function markerEndDisabled(node) {
  const markerEnd = node?.style?.markerEnd;
  if (markerEnd === undefined || markerEnd === null) return false;
  if (markerEnd === false || markerEnd === 0) return true;
  if (typeof markerEnd === "string") {
    const token = markerEnd.trim().toLowerCase();
    return token === "none" || token === "false" || token === "0";
  }
  if (typeof markerEnd === "object" && markerEnd !== null) {
    const kind = markerEnd.kind;
    if (typeof kind === "string") {
      const token = kind.trim().toLowerCase();
      return token === "none" || token === "false";
    }
  }
  return false;
}

/** Directed connectors that should snap to boxes (excludes motion + headless). */
export function isDirectedConnector(node) {
  if (!isArrowLike(node) || isMotionSignalNode(node) || markerEndDisabled(node)) {
    return false;
  }
  if (
    node?.style?.arrowhead === false ||
    node?.style?.arrowhead === 0 ||
    node?.style?.arrowhead === "false"
  ) {
    return false;
  }
  const id = String(node?.id ?? "").toLowerCase();
  // Visual dividers / rules are not box-to-box connectors.
  if (/^(split|divider|rule|sep|guide)([-_]|$)/.test(id)) {
    return false;
  }
  const cap = capabilityOf(node);
  // Braces are undirected (mirror SceneRenderer / desugar markerEnd: none).
  if (cap === "core.bracket") {
    return false;
  }
  return true;
}

export function isDotLike(node) {
  const cap = capabilityOf(node);
  const kind = kindOf(node);
  if (DOT_CAPS.has(cap) || DOT_KINDS.has(kind)) return true;
  const r = node?.style?.r;
  return typeof r === "number" && r > 0 && r <= 12;
}

/**
 * Legacy companion dots authored beside a motion path (`s9-motion-sig-dot`).
 * SceneRenderer drops these; verifier flags them as obsolete dead IR.
 */
export function isMotionCompanionDot(node) {
  if (!isDotLike(node)) return false;
  const role = String(node?.style?.role ?? "").toLowerCase();
  if (role === "motion-signal" || role === "motion-dot") return true;
  const id = String(node?.id ?? "");
  return /motion[-_]?sig/i.test(id) && /-dot$/i.test(id);
}

/** Id of the motion path a companion dot is paired with (`…-dot` → stem). */
export function motionCompanionPathId(dotId) {
  const id = String(dotId ?? "");
  const m = /^(.*)-dot$/i.exec(id);
  if (!m || !/motion[-_]?sig/i.test(m[1])) return null;
  return m[1];
}

/** Static legend chips: skip orphan-dot proximity. */
export function isLegendDot(node) {
  if (!isDotLike(node) || isMotionCompanionDot(node)) return false;
  const role = String(node?.style?.role ?? "").toLowerCase();
  const legend = node?.style?.legend;
  if (role === "legend" || role === "legend-chip" || legend === true) return true;
  const id = String(node?.id ?? "").toLowerCase();
  const label = String(node?.accessibility?.label ?? "").toLowerCase();
  return id.includes("legend") || label.includes("legend");
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

/** Edge / corner / center point on a box (SceneRenderer anchor parity). */
export function nodeAnchorPoint(geom, anchor) {
  const center = boxCenter(geom);
  const left = geom.x;
  const right = geom.x + geom.width;
  const top = geom.y;
  const bottom = geom.y + geom.height;
  switch (String(anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: left, y: center.y };
    case "right":
    case "east":
    case "e":
      return { x: right, y: center.y };
    case "top":
    case "north":
    case "n":
      return { x: center.x, y: top };
    case "bottom":
    case "south":
    case "s":
      return { x: center.x, y: bottom };
    case "ne":
      return { x: right, y: top };
    case "nw":
      return { x: left, y: top };
    case "se":
      return { x: right, y: bottom };
    case "sw":
      return { x: left, y: bottom };
    case "center":
    default:
      return center;
  }
}

/**
 * Resolve a connector endpoint to a point. Supports absolute `{x,y}` and
 * node-anchored `{nodeId, anchor}` when `nodesById` is provided.
 */
export function resolveEndpoint(endpoint, nodesById) {
  if (!endpoint || typeof endpoint !== "object") return null;
  const nodeId = endpoint.nodeId;
  if (typeof nodeId === "string" && nodeId.length > 0 && nodesById) {
    const target = nodesById.get(nodeId);
    const g = target ? geomOf(target) : null;
    if (g) return nodeAnchorPoint(g, endpoint.anchor);
  }
  if (Number.isFinite(endpoint.x) && Number.isFinite(endpoint.y)) {
    return { x: endpoint.x, y: endpoint.y };
  }
  return null;
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
 * Parse SVG path commands into polyline points.
 * Supports M/L/H/V (abs/rel) and records endpoints of C/S/Q/T/A so cubic
 * connectors still yield usable start/end for snap checks.
 */
export function pathPoints(pathData) {
  if (typeof pathData !== "string" || pathData.trim() === "") return [];
  const tokens = pathData.match(/[MLHVCSQTAZmlhvcsqtaz]|-?\d*\.?\d+(?:e[-+]?\d+)?/gi);
  if (!tokens) return [];
  const points = [];
  let i = 0;
  let x = 0;
  let y = 0;
  let cmd = "M";
  const num = () => Number(tokens[i++]);
  const push = (px, py) => {
    if (Number.isFinite(px) && Number.isFinite(py)) {
      points.push({ x: px, y: py });
    }
  };
  while (i < tokens.length) {
    const t = tokens[i];
    if (/^[MLHVCSQTAZmlhvcsqtaz]$/.test(t)) {
      cmd = t;
      i += 1;
      if (cmd === "Z" || cmd === "z") continue;
    }
    if (cmd === "M" || cmd === "L") {
      x = num();
      y = num();
      push(x, y);
      cmd = cmd === "M" ? "L" : cmd;
    } else if (cmd === "m" || cmd === "l") {
      x += num();
      y += num();
      push(x, y);
      cmd = cmd === "m" ? "l" : cmd;
    } else if (cmd === "H") {
      x = num();
      push(x, y);
    } else if (cmd === "h") {
      x += num();
      push(x, y);
    } else if (cmd === "V") {
      y = num();
      push(x, y);
    } else if (cmd === "v") {
      y += num();
      push(x, y);
    } else if (cmd === "C") {
      num();
      num();
      num();
      num();
      x = num();
      y = num();
      push(x, y);
    } else if (cmd === "c") {
      num();
      num();
      num();
      num();
      x += num();
      y += num();
      push(x, y);
    } else if (cmd === "S" || cmd === "Q") {
      num();
      num();
      x = num();
      y = num();
      push(x, y);
    } else if (cmd === "s" || cmd === "q") {
      num();
      num();
      x += num();
      y += num();
      push(x, y);
    } else if (cmd === "T") {
      x = num();
      y = num();
      push(x, y);
    } else if (cmd === "t") {
      x += num();
      y += num();
      push(x, y);
    } else if (cmd === "A") {
      num();
      num();
      num();
      num();
      num();
      x = num();
      y = num();
      push(x, y);
    } else if (cmd === "a") {
      num();
      num();
      num();
      num();
      num();
      x += num();
      y += num();
      push(x, y);
    } else {
      i += 1;
    }
  }
  return points;
}

export function arrowPathData(node, nodesById) {
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
  const start = resolveEndpoint(from, nodesById);
  const end = resolveEndpoint(to, nodesById);
  if (start && end) {
    return `M${start.x} ${start.y} L${end.x} ${end.y}`;
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

/** Resolve viewport from scene.viewport with DEFAULT_VIEWPORT fallback. */
export function sceneViewport(scene, override) {
  const authored = scene?.viewport;
  const width =
    typeof authored?.width === "number" && Number.isFinite(authored.width)
      ? authored.width
      : DEFAULT_VIEWPORT.width;
  const height =
    typeof authored?.height === "number" && Number.isFinite(authored.height)
      ? authored.height
      : DEFAULT_VIEWPORT.height;
  const margin =
    typeof override?.margin === "number"
      ? override.margin
      : DEFAULT_VIEWPORT.margin;
  return {
    width: override?.width ?? width,
    height: override?.height ?? height,
    margin,
  };
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
