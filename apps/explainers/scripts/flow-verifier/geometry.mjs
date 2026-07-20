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

export function isFanNode(node) {
  const cap = capabilityOf(node);
  return cap === "core.fan-out" || cap === "core.fan-in" || kindOf(node) === "fan";
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

function isSoftAnchor(anchor) {
  if (anchor === undefined || anchor === null || String(anchor).length === 0) {
    return true;
  }
  const token = String(anchor).toLowerCase();
  return token === "center" || token === "middle" || token === "c";
}

function facingAnchor(geom, peer) {
  const center = boxCenter(geom);
  const dx = peer.x - center.x;
  const dy = peer.y - center.y;
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? "e" : "w";
  return dy >= 0 ? "s" : "n";
}

function resolveFanEndpoint(endpoint, peer, nodesById) {
  if (!endpoint || typeof endpoint !== "object") return null;
  if (Number.isFinite(endpoint.x) && Number.isFinite(endpoint.y)) {
    return { x: endpoint.x, y: endpoint.y };
  }
  if (typeof endpoint.nodeId !== "string" || endpoint.nodeId.length === 0) {
    return null;
  }
  const target = nodesById?.get(endpoint.nodeId);
  const geom = target ? geomOf(target) : null;
  if (!geom) return null;
  const anchor = isSoftAnchor(endpoint.anchor)
    ? facingAnchor(geom, peer)
    : endpoint.anchor;
  return nodeAnchorPoint(geom, anchor);
}

function centroid(points) {
  if (points.length === 0) return null;
  const sum = points.reduce(
    (total, point) => ({ x: total.x + point.x, y: total.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function compactPoints(points) {
  return points.filter((point, index) => {
    const previous = points[index - 1];
    return (
      previous === undefined ||
      Math.abs(previous.x - point.x) > 0.001 ||
      Math.abs(previous.y - point.y) > 0.001
    );
  });
}

function fanBranchPoints(endpoint, junction, axis, incoming) {
  if (axis === "x") {
    return incoming
      ? [endpoint, { x: junction.x, y: endpoint.y }, junction]
      : [junction, { x: junction.x, y: endpoint.y }, endpoint];
  }
  return incoming
    ? [endpoint, { x: endpoint.x, y: junction.y }, junction]
    : [junction, { x: endpoint.x, y: junction.y }, endpoint];
}

function orthogonalFanPoints(start, end, axis) {
  return axis === "x"
    ? [start, { x: end.x, y: start.y }, end]
    : [start, { x: start.x, y: end.y }, end];
}

function automaticFanJunction(singleton, many, axis) {
  const manyCentroid = centroid(many);
  if (!manyCentroid) return null;
  if (axis === "x") {
    const towardPositive = manyCentroid.x >= singleton.x;
    const corridorEdge = towardPositive
      ? Math.min(...many.map((point) => point.x))
      : Math.max(...many.map((point) => point.x));
    return { x: (singleton.x + corridorEdge) / 2, y: singleton.y };
  }
  const towardPositive = manyCentroid.y >= singleton.y;
  const corridorEdge = towardPositive
    ? Math.min(...many.map((point) => point.y))
    : Math.max(...many.map((point) => point.y));
  return { x: singleton.x, y: (singleton.y + corridorEdge) / 2 };
}

/**
 * Mirror SceneRenderer's endpoint, junction, and trajectory resolution.
 * Returns null when malformed endpoints cannot produce finite connected paths.
 */
export function resolveFanGeometry(node, nodesById) {
  const fanOut = capabilityOf(node) !== "core.fan-in";
  const from = Array.isArray(node?.from) ? node.from : [node?.from];
  const to = Array.isArray(node?.to) ? node.to : [node?.to];
  const singletonEndpoint = (fanOut ? from : to)[0];
  const manyEndpoints = fanOut ? to : from;
  if (!singletonEndpoint || manyEndpoints.some((endpoint) => !endpoint)) {
    return null;
  }

  const origin = { x: 0, y: 0 };
  const roughSingleton = resolveFanEndpoint(
    singletonEndpoint,
    origin,
    nodesById,
  );
  if (!roughSingleton) return null;
  const roughMany = manyEndpoints.map((endpoint) =>
    resolveFanEndpoint(endpoint, roughSingleton, nodesById),
  );
  if (roughMany.some((point) => point === null)) return null;
  const roughManyCentroid = centroid(roughMany);
  if (!roughManyCentroid) return null;
  const singleton = resolveFanEndpoint(
    singletonEndpoint,
    roughManyCentroid,
    nodesById,
  );
  if (!singleton) return null;
  const many = manyEndpoints.map((endpoint) =>
    resolveFanEndpoint(endpoint, singleton, nodesById),
  );
  if (many.some((point) => point === null)) return null;
  const manyCentroid = centroid(many);
  if (!manyCentroid) return null;

  const styledAxis = node?.style?.axis;
  const axis =
    node?.axis === "x" || node?.axis === "y"
      ? node.axis
      : styledAxis === "x" || styledAxis === "y"
        ? styledAxis
        : Math.abs(manyCentroid.x - singleton.x) >=
            Math.abs(manyCentroid.y - singleton.y)
          ? "x"
          : "y";
  const junction =
    node?.junction === undefined
      ? automaticFanJunction(singleton, many, axis)
      : resolveFanEndpoint(
          node.junction,
          fanOut ? manyCentroid : singleton,
          nodesById,
        );
  if (
    !junction ||
    !Number.isFinite(junction.x) ||
    !Number.isFinite(junction.y)
  ) {
    return null;
  }

  const trunk = compactPoints(
    fanOut
      ? orthogonalFanPoints(singleton, junction, axis)
      : orthogonalFanPoints(junction, singleton, axis),
  );
  const branches = many.map((endpoint) =>
    compactPoints(fanBranchPoints(endpoint, junction, axis, !fanOut)),
  );
  const trajectories = branches.map((branch) =>
    compactPoints(
      fanOut
        ? [...trunk, ...branch.slice(1)]
        : [...branch, ...trunk.slice(1)],
    ),
  );
  return { axis, junction, trunk, branches, trajectories };
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

function pathDataFromPoints(points, nodesById) {
  if (!Array.isArray(points) || points.length === 0) return null;
  const resolved = points.map((point) => resolveEndpoint(point, nodesById));
  if (resolved.some((point) => point === null)) return null;
  return resolved
    .slice(1)
    .reduce(
      (path, point) => `${path} L${point.x} ${point.y}`,
      `M${resolved[0].x} ${resolved[0].y}`,
    );
}

function elbowPathData(start, end, via, axis) {
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);
  const preferX = axis === "y" ? false : axis === "x" ? true : dx >= dy;
  if (via) {
    return preferX
      ? `M${start.x} ${start.y} H${via.x} V${via.y} H${end.x} V${end.y}`
      : `M${start.x} ${start.y} V${via.y} H${via.x} V${end.y} H${end.x}`;
  }
  if (preferX) {
    const midX = (start.x + end.x) / 2;
    return `M${start.x} ${start.y} H${midX} V${end.y} H${end.x}`;
  }
  const midY = (start.y + end.y) / 2;
  return `M${start.x} ${start.y} V${midY} H${end.x} V${end.y}`;
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

// --- Deterministic obstacle-aware curved router (Node mirror of the TS core) ---
// Kept byte-behavior-equivalent to
// src/core/diagram/connector-routing-{types,geometry,search}.ts so the verifier
// exercises the same routing the renderer produces.

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

function resolveWaypoints(start, end, obstacles, clearance) {
  const inflated = obstacles
    .map((obstacle) => ({ id: obstacle.id, bounds: inflateBounds(obstacle.bounds, clearance) }))
    .filter(
      (obstacle) =>
        !pointInBounds(start, obstacle.bounds, true) && !pointInBounds(end, obstacle.bounds, true),
    );
  if (inflated.length === 0 || segmentIsVisible(start, end, inflated)) {
    return { waypoints: [start, end], feasible: true };
  }
  const path = shortestVisiblePath(start, end, inflated);
  if (path === undefined || path.length < 2) {
    return { waypoints: [start, end], feasible: false };
  }
  return { waypoints: simplifyWaypoints(path), feasible: true };
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

function penetratedIds(segments, obstacles) {
  const hits = new Set();
  for (const segment of segments) {
    for (const id of cubicPenetrations(segment, obstacles)) hits.add(id);
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
  return [
    [start, { x: start.x, y: top }, { x: end.x, y: top }, end],
    [start, { x: right, y: start.y }, { x: right, y: end.y }, end],
    [start, { x: start.x, y: bottom }, { x: end.x, y: bottom }, end],
    [start, { x: left, y: start.y }, { x: left, y: end.y }, end],
  ];
}

function renderPolyline(waypoints, fromDir, toDir, obstacles, curvature) {
  let bestSegments = smoothPolyline(waypoints, fromDir, toDir, curvature);
  let bestPenetrations = penetratedIds(bestSegments, obstacles);
  if (bestPenetrations.length > 0) {
    for (const factor of CURVATURE_LADDER) {
      const candidate = smoothPolyline(waypoints, fromDir, toDir, Math.max(curvature * factor, 0.02));
      const penetrations = penetratedIds(candidate, obstacles);
      if (penetrations.length < bestPenetrations.length) {
        bestSegments = candidate;
        bestPenetrations = penetrations;
      }
      if (penetrations.length === 0) break;
    }
  }
  return { segments: bestSegments, penetrations: bestPenetrations };
}

/** Node mirror of routeCurve in connector-routing-search.ts. */
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
      const rendered = renderPolyline(candidate, fromDir, toDir, obstacles, options.curvature);
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

  let best = renderPolyline(waypoints, fromDir, toDir, obstacles, options.curvature);
  let bestWaypoints = waypoints;
  const laneOffset = input.laneOffset ?? 0;
  if (laneOffset !== 0) {
    for (const factor of LANE_OFFSET_LADDER) {
      const offsetWaypoints = applyLaneOffset(waypoints, laneOffset * factor);
      const rendered = renderPolyline(offsetWaypoints, fromDir, toDir, obstacles, options.curvature);
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
 * Validate a resolved route: emit an error for unexpected obstacle penetration
 * and a warning for an explicit deterministic fallback.
 */
export function verifyCurveRouteResult(edgeId, result, obstacles) {
  const findings = [];
  const penetrations = penetratedIds(result.segments, obstacles ?? []);
  if (penetrations.length > 0 && !result.usedFallback) {
    findings.push({
      severity: "error",
      code: "CURVE_OBSTACLE_PENETRATION",
      edgeId,
      obstacleIds: penetrations,
    });
  }
  if (result.usedFallback) {
    findings.push({
      severity: "warn",
      code: "CURVE_FALLBACK",
      edgeId,
      obstacleIds: result.penetratedObstacleIds ?? [],
    });
  }
  return findings;
}

function isElbowRoute(node) {
  const capability = capabilityOf(node);
  if (capability === "core.elbow" || capability === "core.route") {
    return true;
  }
  if (node?.kind === "elbow") {
    return true;
  }
  return node?.style?.route === "elbow";
}

function isCurveRoute(node) {
  if (node?.style?.route === "curve") {
    return true;
  }
  return capabilityOf(node) === "core.curve";
}

function endpointAnchor(endpoint) {
  return typeof endpoint?.anchor === "string" && endpoint.anchor.length > 0
    ? endpoint.anchor
    : undefined;
}

export function arrowPathData(node, nodesById) {
  if (typeof node?.d === "string" && node.d.trim()) return node.d;
  if (typeof node?.path === "string" && node.path.trim()) return node.path;
  const pointsPath = pathDataFromPoints(node?.points, nodesById);
  if (pointsPath) return pointsPath;
  const from = node?.from;
  const to = node?.to;
  const start = resolveEndpoint(from, nodesById);
  const end = resolveEndpoint(to, nodesById);
  if (start && end) {
    if (isCurveRoute(node)) {
      return routeCurve({
        edgeId: String(node?.id ?? ""),
        start,
        end,
        fromAnchor: endpointAnchor(from),
        toAnchor: endpointAnchor(to),
        sourceId: typeof from?.nodeId === "string" ? from.nodeId : undefined,
        targetId: typeof to?.nodeId === "string" ? to.nodeId : undefined,
        obstacles: [],
        siblings: [],
        options: normalizeCurveRouteOptions(node?.style),
      }).d;
    }
    if (isElbowRoute(node)) {
      const via = resolveEndpoint(node?.via, nodesById);
      const styledAxis = node?.style?.axis;
      const axis =
        node?.axis === "x" || node?.axis === "y"
          ? node.axis
          : styledAxis === "x" || styledAxis === "y"
            ? styledAxis
            : undefined;
      return elbowPathData(start, end, via, axis);
    }
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
