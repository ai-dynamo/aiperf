/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Snap distance (px) for connector endpoint / dot proximity. */
export const SNAP_PX = 36;

const ARROW_CAPS = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
  "core.elbow",
  "core.route",
  "core.fan-out",
  "core.fan-in",
]);
const ARROW_KINDS = new Set([
  "line",
  "path",
  "arrow",
  "connector",
  "elbow",
  "fan",
]);
const DOT_CAPS = new Set(["core.dot", "core.circle"]);
const DOT_KINDS = new Set(["dot", "circle"]);
const BOX_CAPS = new Set(["core.rect", "core.text"]);
const MOTION_CAPS = new Set([
  "motion.signal",
  "motion.dot",
  "core.motion",
  "core.motion-signal",
  "motion.motion-signal",
]);

/**
 * Returns a node's canonical or authoring-alias capability, mirroring
 * `node-classification.ts` three-tier resolution.
 */
export function capabilityOf(node) {
  if (typeof node?.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node?.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  if (typeof node?.kind === "string" && node.kind.length > 0) {
    return `core.${node.kind}`;
  }
  return "";
}

export function kindOf(node) {
  return String(node?.kind ?? "");
}

export function isArrowLike(node) {
  const cap = capabilityOf(node);
  const kind = kindOf(node);
  return ARROW_CAPS.has(cap) || ARROW_KINDS.has(kind);
}

export function isFanNode(node) {
  const cap = capabilityOf(node);
  return cap === "core.fan-out" || cap === "core.fan-in" || kindOf(node) === "fan";
}

/**
 * Mirror `node-classification.ts` motion-signal classification so guide strokes are not
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

/**
 * Sample canonical snapshot SVG path data for proximity/playback checks.
 *
 * Snapshots provide the resolved `d` string, but not sampled points or DOM path
 * metrics. This parser deliberately interprets only that resolved output; it
 * never reconstructs connector endpoints or routing from authored scene data.
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

/** True when parsed path endpoints coincide within tolerance (zero-length edge). */
export function pathEndpointsCoincident(points, tolerance = 0.5) {
  if (!Array.isArray(points) || points.length < 2) {
    return true;
  }
  const first = points[0];
  const last = points[points.length - 1];
  return Math.hypot(first.x - last.x, first.y - last.y) <= tolerance;
}

function normalizeVector(vector) {
  const length = Math.hypot(vector.x, vector.y);
  if (length < 1e-6) {
    return { x: 1, y: 0 };
  }
  return { x: vector.x / length, y: vector.y / length };
}

function anchorExitDirection(anchor, peer, self) {
  switch (String(anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: -1, y: 0 };
    case "right":
    case "east":
    case "e":
      return { x: 1, y: 0 };
    case "top":
    case "north":
    case "n":
      return { x: 0, y: -1 };
    case "bottom":
    case "south":
    case "s":
      return { x: 0, y: 1 };
    case "ne":
      return normalizeVector({ x: 1, y: -1 });
    case "nw":
      return normalizeVector({ x: -1, y: -1 });
    case "se":
      return normalizeVector({ x: 1, y: 1 });
    case "sw":
      return normalizeVector({ x: -1, y: 1 });
    case "center":
    case "middle":
    case "c":
      if (peer && self) {
        return normalizeVector({ x: peer.x - self.x, y: peer.y - self.y });
      }
      return { x: 1, y: 0 };
    default:
      return { x: 1, y: 0 };
  }
}

// --- Synthetic advanced curved-router matrix ---
// This is the sole intentional geometry mirror in the Node verifier. The
// deck-independent matrix has no scene to resolve and therefore cannot consume
// a canonical snapshot; it independently pressure-tests deterministic routing
// across anchor pairs, obstacle halos, self-loops, lanes, and fallbacks.

const CUBIC_SAMPLE_COUNT = 33;
const CANONICAL_SCALE = 1000;
const MIN_HANDLE = 12;
const MAX_HANDLE = 180;
const CORNER_EPSILON = 0.5;
const CURVATURE_LADDER = [1, 0.5, 0.25, 0.05];
const LANE_OFFSET_LADDER = [1, 0.75, 0.5, 0.25];

/** Default routing options; matches DEFAULT_CURVE_ROUTE_OPTIONS in the TS core. */
export const DEFAULT_CURVE_ROUTE_OPTIONS = Object.freeze({
  clearance: 12,
  curvature: 0.45,
  avoidObstacles: true,
  preferredSide: "auto",
  bundle: false,
  parallelGap: 8,
});

function roundCanonical(value) {
  if (!Number.isFinite(value)) return 0;
  const rounded = Math.round(value * CANONICAL_SCALE) / CANONICAL_SCALE;
  return rounded === 0 ? 0 : rounded;
}

function canonicalPointKey(point) {
  return `${roundCanonical(point.x)},${roundCanonical(point.y)}`;
}

function inflateBounds(bounds, amount) {
  const pad = Number.isFinite(amount) && amount > 0 ? amount : 0;
  return {
    x: bounds.x - pad,
    y: bounds.y - pad,
    width: bounds.width + pad * 2,
    height: bounds.height + pad * 2,
  };
}

function pointInBounds(point, bounds, strict = false) {
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  if (strict) {
    return point.x > left && point.x < right && point.y > top && point.y < bottom;
  }
  return point.x >= left && point.x <= right && point.y >= top && point.y <= bottom;
}

/**
 * Shrink an axis-aligned rectangle so `point` is not in its strict interior.
 * Pushes the nearest edge to the point. Leaves bounds unchanged when the point
 * is already outside the open interior. Used so clearance inflation around a
 * third-party obstacle near a connector endpoint does not treat that endpoint
 * as blocked while still keeping the obstacle for the rest of the path. Mirror
 * of `shrinkBoundsToExcludePoint` in connector-routing-geometry.ts.
 */
function shrinkBoundsToExcludePoint(bounds, point) {
  if (!pointInBounds(point, bounds, true)) return bounds;
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  const distLeft = point.x - left;
  const distRight = right - point.x;
  const distTop = point.y - top;
  const distBottom = bottom - point.y;
  const minDist = Math.min(distLeft, distRight, distTop, distBottom);
  if (minDist === distLeft) {
    return { x: point.x, y: bounds.y, width: right - point.x, height: bounds.height };
  }
  if (minDist === distRight) {
    return { x: bounds.x, y: bounds.y, width: point.x - left, height: bounds.height };
  }
  if (minDist === distTop) {
    return { x: bounds.x, y: point.y, width: bounds.width, height: bottom - point.y };
  }
  return { x: bounds.x, y: bounds.y, width: bounds.width, height: point.y - top };
}

function segmentIntersectsBounds(start, end, bounds, allowBoundary = true) {
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  let t0 = 0;
  let t1 = 1;
  const p = [-dx, dx, -dy, dy];
  const q = [start.x - left, right - start.x, start.y - top, bottom - start.y];
  for (let i = 0; i < 4; i += 1) {
    const pi = p[i];
    const qi = q[i];
    if (pi === 0) {
      if (qi < 0) return false;
    } else {
      const r = qi / pi;
      if (pi < 0) {
        if (r > t1) return false;
        if (r > t0) t0 = r;
      } else {
        if (r < t0) return false;
        if (r < t1) t1 = r;
      }
    }
  }
  if (t0 > t1) return false;
  if (!allowBoundary) return true;
  const mid = (t0 + t1) / 2;
  const midPoint = { x: start.x + dx * mid, y: start.y + dy * mid };
  return pointInBounds(midPoint, bounds, true);
}

function segmentIsVisible(start, end, obstacles) {
  if (!Number.isFinite(start.x) || !Number.isFinite(start.y)) return false;
  if (!Number.isFinite(end.x) || !Number.isFinite(end.y)) return false;
  for (const obstacle of obstacles) {
    if (segmentIntersectsBounds(start, end, obstacle.bounds, true)) return false;
  }
  return true;
}

function simplifyWaypoints(points) {
  const deduped = [];
  for (const point of points) {
    const previous = deduped[deduped.length - 1];
    if (
      previous === undefined ||
      roundCanonical(previous.x) !== roundCanonical(point.x) ||
      roundCanonical(previous.y) !== roundCanonical(point.y)
    ) {
      deduped.push(point);
    }
  }
  if (deduped.length <= 2) return deduped;
  const simplified = [deduped[0]];
  for (let i = 1; i < deduped.length - 1; i += 1) {
    const prev = simplified[simplified.length - 1];
    const curr = deduped[i];
    const next = deduped[i + 1];
    const cross =
      (curr.x - prev.x) * (next.y - prev.y) - (curr.y - prev.y) * (next.x - prev.x);
    if (Math.abs(cross) > 1e-6) simplified.push(curr);
  }
  simplified.push(deduped[deduped.length - 1]);
  return simplified;
}

function cubicPoint(segment, t) {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return {
    x: a * segment.start.x + b * segment.control1.x + c * segment.control2.x + d * segment.end.x,
    y: a * segment.start.y + b * segment.control1.y + c * segment.control2.y + d * segment.end.y,
  };
}

function cubicPenetrations(segment, obstacles) {
  const hits = new Set();
  for (const obstacle of obstacles) {
    for (let i = 0; i < CUBIC_SAMPLE_COUNT; i += 1) {
      const t = i / (CUBIC_SAMPLE_COUNT - 1);
      if (pointInBounds(cubicPoint(segment, t), obstacle.bounds, true)) {
        hits.add(obstacle.id);
        break;
      }
    }
  }
  return [...hits].sort((left, right) => left.localeCompare(right));
}

function routeBounds(points) {
  if (points.length === 0) return { x: 0, y: 0, width: 0, height: 0 };
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  return {
    x: roundCanonical(minX),
    y: roundCanonical(minY),
    width: roundCanonical(maxX - minX),
    height: roundCanonical(maxY - minY),
  };
}

function unit(vector) {
  const length = Math.hypot(vector.x, vector.y);
  if (length < 1e-6) return { x: 1, y: 0 };
  return { x: vector.x / length, y: vector.y / length };
}

function distance2(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function clampHandle(length, curvature) {
  const raw = length * curvature;
  if (!Number.isFinite(raw)) return MIN_HANDLE;
  return Math.min(Math.max(raw, MIN_HANDLE), MAX_HANDLE);
}

function obstacleCorners(obstacles) {
  const seen = new Set();
  const corners = [];
  for (const obstacle of obstacles) {
    const { x, y, width, height } = obstacle.bounds;
    const candidates = [
      { x: x - CORNER_EPSILON, y: y - CORNER_EPSILON },
      { x: x + width + CORNER_EPSILON, y: y - CORNER_EPSILON },
      { x: x - CORNER_EPSILON, y: y + height + CORNER_EPSILON },
      { x: x + width + CORNER_EPSILON, y: y + height + CORNER_EPSILON },
    ];
    for (const candidate of candidates) {
      if (obstacles.some((other) => pointInBounds(candidate, other.bounds, true))) continue;
      const key = canonicalPointKey(candidate);
      if (!seen.has(key)) {
        seen.add(key);
        corners.push({ x: roundCanonical(candidate.x), y: roundCanonical(candidate.y) });
      }
    }
  }
  return corners;
}

function shortestVisiblePath(start, end, obstacles) {
  const startNode = { point: start, key: canonicalPointKey(start) };
  const endNode = { point: end, key: canonicalPointKey(end) };
  const nodes = [startNode];
  const seenKeys = new Set([startNode.key]);
  for (const corner of obstacleCorners(obstacles)) {
    const key = canonicalPointKey(corner);
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      nodes.push({ point: corner, key });
    }
  }
  if (!seenKeys.has(endNode.key)) {
    seenKeys.add(endNode.key);
    nodes.push(endNode);
  }
  nodes.sort((a, b) => a.key.localeCompare(b.key));

  const gScore = new Map();
  const cameFrom = new Map();
  const nodeByKey = new Map();
  for (const node of nodes) nodeByKey.set(node.key, node);
  gScore.set(startNode.key, 0);
  const open = new Set([startNode.key]);
  const heuristic = (node) => distance2(node.point, end);

  while (open.size > 0) {
    let currentKey;
    let bestF = Infinity;
    for (const key of open) {
      const g = gScore.get(key) ?? Infinity;
      const node = nodeByKey.get(key);
      const f = g + heuristic(node);
      if (
        f < bestF - 1e-9 ||
        (Math.abs(f - bestF) <= 1e-9 && (currentKey === undefined || key.localeCompare(currentKey) < 0))
      ) {
        bestF = f;
        currentKey = key;
      }
    }
    if (currentKey === undefined) break;
    if (currentKey === endNode.key) {
      const path = [];
      let cursor = currentKey;
      while (cursor !== undefined) {
        path.push(nodeByKey.get(cursor).point);
        cursor = cameFrom.get(cursor);
      }
      path.reverse();
      return path;
    }
    open.delete(currentKey);
    const current = nodeByKey.get(currentKey);
    const currentG = gScore.get(currentKey) ?? Infinity;
    for (const neighbor of nodes) {
      if (neighbor.key === currentKey) continue;
      if (!segmentIsVisible(current.point, neighbor.point, obstacles)) continue;
      const tentative = currentG + distance2(current.point, neighbor.point);
      const known = gScore.get(neighbor.key) ?? Infinity;
      if (tentative < known - 1e-9) {
        cameFrom.set(neighbor.key, currentKey);
        gScore.set(neighbor.key, tentative);
        open.add(neighbor.key);
      }
    }
  }
  return undefined;
}

function travelTangents(points, fromDir, toDir) {
  const tangents = [];
  for (let i = 0; i < points.length; i += 1) {
    if (i === 0) {
      tangents.push(unit(fromDir));
    } else if (i === points.length - 1) {
      tangents.push(unit({ x: -toDir.x, y: -toDir.y }));
    } else {
      tangents.push(unit({ x: points[i + 1].x - points[i - 1].x, y: points[i + 1].y - points[i - 1].y }));
    }
  }
  return tangents;
}

function smoothPolyline(points, fromDir, toDir, curvature) {
  const segments = [];
  if (points.length < 2) return segments;
  const tangents = travelTangents(points, fromDir, toDir);
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const handle = clampHandle(distance2(a, b), curvature);
    segments.push({
      start: a,
      control1: { x: a.x + tangents[i].x * handle, y: a.y + tangents[i].y * handle },
      control2: { x: b.x - tangents[i + 1].x * handle, y: b.y - tangents[i + 1].y * handle },
      end: b,
    });
  }
  return segments;
}

function endpointExitDirection(anchor, peer, self) {
  return unit(anchorExitDirection(anchor, peer, self));
}

/**
 * True when any leg of a polyline crosses an obstacle's true (uninflated)
 * interior. Used to downgrade `feasible` when the resolved waypoints — found
 * against the clearance-inflated search graph — still cut through real
 * obstacle geometry, e.g. an obstacle dropped from the graph because an
 * endpoint sits inside its true bounds. Mirror of
 * `polylinePenetratesTrueBounds` in connector-routing-search.ts.
 */
function polylinePenetratesTrueBounds(waypoints, obstacles) {
  for (let i = 0; i < waypoints.length - 1; i += 1) {
    const a = waypoints[i];
    const b = waypoints[i + 1];
    for (const obstacle of obstacles) {
      if (segmentIntersectsBounds(a, b, obstacle.bounds, true)) return true;
    }
  }
  return false;
}

/** Mirror of `resolveWaypoints` in connector-routing-search.ts. */
function resolveWaypoints(start, end, obstacles, clearance) {
  // Source/target boxes are excluded upstream. Do not drop third-party
  // obstacles merely because clearance inflation covers an endpoint — that
  // removed them for the *entire* path and let curves cut through far away.
  // Keep the obstacle, but shrink the inflated rect so the endpoint itself is
  // not treated as blocked. Still skip obstacles whose true (uninflated)
  // interior contains an endpoint — that geometry is unavoidable from the
  // endpoint itself; `feasible` below still catches a resulting penetration
  // elsewhere on the path.
  const inflated = obstacles.flatMap((obstacle) => {
    if (pointInBounds(start, obstacle.bounds, true) || pointInBounds(end, obstacle.bounds, true)) {
      return [];
    }
    let bounds = inflateBounds(obstacle.bounds, clearance);
    bounds = shrinkBoundsToExcludePoint(bounds, start);
    bounds = shrinkBoundsToExcludePoint(bounds, end);
    if (bounds.width <= 0 || bounds.height <= 0) return [];
    return [{ id: obstacle.id, bounds }];
  });
  if (inflated.length === 0 || segmentIsVisible(start, end, inflated)) {
    const waypoints = [start, end];
    return { waypoints, feasible: !polylinePenetratesTrueBounds(waypoints, obstacles) };
  }
  const path = shortestVisiblePath(start, end, inflated);
  if (path === undefined || path.length < 2) {
    return { waypoints: [start, end], feasible: false };
  }
  const waypoints = simplifyWaypoints(path);
  return { waypoints, feasible: !polylinePenetratesTrueBounds(waypoints, obstacles) };
}

function formatNumber(value) {
  return String(roundCanonical(value));
}

function segmentsToPathData(start, segments) {
  if (segments.length === 0) return `M${formatNumber(start.x)} ${formatNumber(start.y)}`;
  let d = `M${formatNumber(start.x)} ${formatNumber(start.y)}`;
  for (const segment of segments) {
    d +=
      ` C${formatNumber(segment.control1.x)} ${formatNumber(segment.control1.y)}` +
      ` ${formatNumber(segment.control2.x)} ${formatNumber(segment.control2.y)}` +
      ` ${formatNumber(segment.end.x)} ${formatNumber(segment.end.y)}`;
  }
  return d;
}

function readNumber(record, key) {
  const value = record?.[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function readBoolean(record, key) {
  const value = record?.[key];
  if (typeof value === "boolean") return value;
  if (value === "true") return true;
  if (value === "false") return false;
  return undefined;
}

function readPreferredSide(record, key) {
  const value = record?.[key];
  if (typeof value !== "string") return undefined;
  switch (value.toLowerCase()) {
    case "auto":
      return "auto";
    case "n":
    case "north":
    case "top":
      return "n";
    case "s":
    case "south":
    case "bottom":
      return "s";
    case "e":
    case "east":
    case "right":
      return "e";
    case "w":
    case "west":
    case "left":
      return "w";
    default:
      return undefined;
  }
}

// Retained for `verifyAdvancedCurveRouting` in `ir.mjs`: a deck-independent
// synthetic matrix that asserts router determinism without a resolved snapshot.
export function normalizeCurveRouteOptions(style) {
  const clearanceRaw = readNumber(style, "clearance");
  const curvatureRaw = readNumber(style, "curvature");
  const parallelGapRaw = readNumber(style, "parallelGap") ?? readNumber(style, "laneGap");
  return {
    clearance:
      clearanceRaw !== undefined && clearanceRaw >= 0
        ? clearanceRaw
        : DEFAULT_CURVE_ROUTE_OPTIONS.clearance,
    curvature:
      curvatureRaw !== undefined
        ? Math.min(Math.max(curvatureRaw, 0.05), 0.95)
        : DEFAULT_CURVE_ROUTE_OPTIONS.curvature,
    avoidObstacles: readBoolean(style, "avoidObstacles") ?? DEFAULT_CURVE_ROUTE_OPTIONS.avoidObstacles,
    preferredSide: readPreferredSide(style, "preferredSide") ?? DEFAULT_CURVE_ROUTE_OPTIONS.preferredSide,
    bundle: readBoolean(style, "bundle") ?? DEFAULT_CURVE_ROUTE_OPTIONS.bundle,
    parallelGap:
      parallelGapRaw !== undefined && parallelGapRaw >= 0
        ? parallelGapRaw
        : DEFAULT_CURVE_ROUTE_OPTIONS.parallelGap,
  };
}

function segmentBoundsPoints(start, segments) {
  const points = [start];
  for (const segment of segments) {
    points.push(segment.control1, segment.control2, segment.end);
  }
  return points;
}

/**
 * Obstacles widened by `clearance` for post-smoothing penetration checks, with
 * each inflated rectangle shrunk away from the route's own `start`/`end`.
 * Mirror of `inflateForPenetrationCheck` in connector-routing-search.ts.
 */
function inflateForPenetrationCheck(obstacles, clearance, start, end) {
  return obstacles.flatMap((obstacle) => {
    let bounds = inflateBounds(obstacle.bounds, clearance);
    bounds = shrinkBoundsToExcludePoint(bounds, start);
    bounds = shrinkBoundsToExcludePoint(bounds, end);
    if (bounds.width <= 0 || bounds.height <= 0) return [];
    return [{ id: obstacle.id, bounds }];
  });
}

/** Mirror of `penetratedIds` in connector-routing-search.ts. */
function penetratedIds(segments, obstacles, clearance, start, end) {
  const inflated = inflateForPenetrationCheck(obstacles, clearance, start, end);
  const hits = new Set();
  for (const segment of segments) {
    for (const id of cubicPenetrations(segment, inflated)) hits.add(id);
  }
  return [...hits].sort((a, b) => a.localeCompare(b));
}

function chordNormal(a, b) {
  return unit({ x: -(b.y - a.y), y: b.x - a.x });
}

function applyLaneOffset(points, offset) {
  if (!Number.isFinite(offset) || offset === 0 || points.length < 2) return points;
  const start = points[0];
  const end = points[points.length - 1];
  if (points.length === 2) {
    const normal = chordNormal(start, end);
    const mid = {
      x: (start.x + end.x) / 2 + normal.x * offset,
      y: (start.y + end.y) / 2 + normal.y * offset,
    };
    return [start, mid, end];
  }
  const shifted = [start];
  for (let i = 1; i < points.length - 1; i += 1) {
    const normal = chordNormal(points[i - 1], points[i + 1]);
    shifted.push({ x: points[i].x + normal.x * offset, y: points[i].y + normal.y * offset });
  }
  shifted.push(end);
  return shifted;
}

/** Perimeter loop side order used when the author has no side preference. */
const SELF_LOOP_SIDE_ORDER = ["n", "e", "s", "w"];

/**
 * Perimeter loop candidates for a self-edge. Mirror of `selfLoopCandidates`
 * in connector-routing-search.ts: `preferredSide` moves the matching side to
 * the front of the candidate list instead of filtering, so an obstructed
 * preferred side still falls through to the remaining sides in order.
 */
function selfLoopCandidates(input) {
  const bounds = input.sourceBounds ?? input.targetBounds;
  if (bounds === undefined) return [];
  const gap = input.options.clearance + input.options.parallelGap;
  const start = { x: roundCanonical(input.start.x), y: roundCanonical(input.start.y) };
  const end = { x: roundCanonical(input.end.x), y: roundCanonical(input.end.y) };
  const left = bounds.x - gap;
  const right = bounds.x + bounds.width + gap;
  const top = bounds.y - gap;
  const bottom = bounds.y + bounds.height + gap;
  const bySide = {
    n: [start, { x: start.x, y: top }, { x: end.x, y: top }, end],
    e: [start, { x: right, y: start.y }, { x: right, y: end.y }, end],
    s: [start, { x: start.x, y: bottom }, { x: end.x, y: bottom }, end],
    w: [start, { x: left, y: start.y }, { x: left, y: end.y }, end],
  };
  const preferred = input.options.preferredSide;
  const order =
    preferred === "auto" || !(preferred in bySide)
      ? SELF_LOOP_SIDE_ORDER
      : [preferred, ...SELF_LOOP_SIDE_ORDER.filter((side) => side !== preferred)];
  return order.map((side) => bySide[side]);
}

/**
 * Smooth a polyline down the curvature ladder, keeping the least-penetrating
 * result. Mirror of `renderPolyline` in connector-routing-search.ts: penetration
 * is checked against obstacles inflated by `clearance` so a curve that bows into
 * the buffer zone (without touching the true obstacle rectangle) still triggers
 * a retry instead of being reported as clean.
 */
function renderPolyline(waypoints, fromDir, toDir, obstacles, curvature, clearance) {
  const start = waypoints[0];
  const end = waypoints[waypoints.length - 1];
  let bestSegments = smoothPolyline(waypoints, fromDir, toDir, curvature);
  let bestPenetrations = penetratedIds(bestSegments, obstacles, clearance, start, end);
  if (bestPenetrations.length > 0) {
    for (const factor of CURVATURE_LADDER) {
      const candidate = smoothPolyline(waypoints, fromDir, toDir, Math.max(curvature * factor, 0.02));
      const penetrations = penetratedIds(candidate, obstacles, clearance, start, end);
      if (penetrations.length < bestPenetrations.length) {
        bestSegments = candidate;
        bestPenetrations = penetrations;
      }
      if (penetrations.length === 0) break;
    }
  }
  return { segments: bestSegments, penetrations: bestPenetrations };
}

/** Route one deck-independent synthetic advanced-matrix case. */
export function routeCurve(input) {
  const options = input.options ?? DEFAULT_CURVE_ROUTE_OPTIONS;
  const start = { x: roundCanonical(input.start.x), y: roundCanonical(input.start.y) };
  const end = { x: roundCanonical(input.end.x), y: roundCanonical(input.end.y) };
  const fromDir = endpointExitDirection(input.fromAnchor, end, start);
  const toDir = endpointExitDirection(input.toAnchor, start, end);
  const obstacles = input.obstacles ?? [];

  const isSelfLoop =
    input.sourceId !== undefined &&
    input.sourceId === input.targetId &&
    Math.hypot(end.x - start.x, end.y - start.y) < 1e-3 * (options.clearance + 1) + 4;

  let waypoints = [start, end];
  let feasible = true;
  if (isSelfLoop) {
    const candidates = selfLoopCandidates(input);
    let bestLoop;
    let bestLoopWaypoints;
    for (const candidate of candidates) {
      const rendered = renderPolyline(candidate, fromDir, toDir, obstacles, options.curvature, options.clearance);
      if (rendered.penetrations.length === 0) {
        bestLoop = rendered;
        bestLoopWaypoints = candidate;
        break;
      }
      if (bestLoop === undefined || rendered.penetrations.length < bestLoop.penetrations.length) {
        bestLoop = rendered;
        bestLoopWaypoints = candidate;
      }
    }
    if (bestLoop !== undefined && bestLoopWaypoints !== undefined) {
      return {
        d: segmentsToPathData(start, bestLoop.segments),
        waypoints: bestLoopWaypoints,
        segments: bestLoop.segments,
        bounds: routeBounds(segmentBoundsPoints(start, bestLoop.segments)),
        usedFallback: bestLoop.penetrations.length > 0,
        penetratedObstacleIds: bestLoop.penetrations,
      };
    }
  }

  if (options.avoidObstacles && obstacles.length > 0) {
    const resolved = resolveWaypoints(start, end, obstacles, options.clearance);
    waypoints = resolved.waypoints;
    feasible = resolved.feasible;
  }

  let best = renderPolyline(waypoints, fromDir, toDir, obstacles, options.curvature, options.clearance);
  let bestWaypoints = waypoints;
  const laneOffset = input.laneOffset ?? 0;
  if (laneOffset !== 0) {
    for (const factor of LANE_OFFSET_LADDER) {
      const offsetWaypoints = applyLaneOffset(waypoints, laneOffset * factor);
      const rendered = renderPolyline(
        offsetWaypoints,
        fromDir,
        toDir,
        obstacles,
        options.curvature,
        options.clearance,
      );
      if (rendered.penetrations.length <= best.penetrations.length) {
        best = rendered;
        bestWaypoints = offsetWaypoints;
        break;
      }
    }
  }

  return {
    d: segmentsToPathData(start, best.segments),
    waypoints: bestWaypoints,
    segments: best.segments,
    bounds: routeBounds(segmentBoundsPoints(start, best.segments)),
    usedFallback: !feasible || best.penetrations.length > 0,
    penetratedObstacleIds: best.penetrations,
  };
}

/**
 * Playback sampling remains local because resolved snapshots are static and do
 * not encode cue windows or playhead state.
 */
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

/**
 * Picks the cue whose window most recently began at or before
 * `playbackTimeMs`, falling back to the earliest-authored cue when none has
 * started yet. Mirrors SceneRenderer's `mostRecentlyStartedCue`: authoring
 * the same target with more than one draw/trace cue (e.g. draw early, then
 * confirm later) is a supported idiom, so picking blindly by declaration
 * order (`.at(-1)`) instead of by which window is actually live disagrees
 * with the runtime and discards the earlier cue's live animation window.
 */
function mostRecentlyStartedCue(cues, playbackTimeMs) {
  let started;
  let earliest;
  for (const cue of cues) {
    const atMs = Number(cue.at) || 0;
    if (earliest === undefined || atMs < (Number(earliest.at) || 0)) {
      earliest = cue;
    }
    if (
      atMs <= playbackTimeMs &&
      (started === undefined || atMs >= (Number(started.at) || 0))
    ) {
      started = cue;
    }
  }
  return started ?? earliest;
}

export function drawProgress(timeline, nodeId, tMs) {
  const cues = (timeline ?? []).filter(
    (c) => c.target === nodeId && isDrawAction(c.action),
  );
  const cue = mostRecentlyStartedCue(cues, tMs);
  if (!cue) return undefined;
  const at = Number(cue.at) || 0;
  const dur = Number(cue.duration) || 0;
  if (tMs <= at) return 0;
  if (dur <= 0) return 1;
  return Math.min(1, Math.max(0, (tMs - at) / dur));
}
