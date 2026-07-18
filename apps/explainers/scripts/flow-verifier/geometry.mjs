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
    const capability = capabilityOf(node);
    if (
      capability === "core.elbow" ||
      capability === "core.route" ||
      node?.kind === "elbow" ||
      node?.style?.route === "elbow"
    ) {
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
