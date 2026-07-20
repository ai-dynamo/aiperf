/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  SNAP_PX,
  arrowPathData,
  drawProgress,
  geomOf,
  inViewport,
  isArrowLike,
  isBoxLike,
  isDirectedConnector,
  isDotLike,
  isDrawAction,
  isFanNode,
  isLegendDot,
  isMotionCompanionDot,
  isMotionSignalNode,
  nodeIds,
  normalizeCurveRouteOptions,
  pathPoints,
  pointNearBox,
  resolveFanGeometry,
  routeCurve,
  sceneViewport,
  timelineDurationMs,
  walkNodes,
} from "./geometry.mjs";

/**
 * @typedef {{ severity: "error" | "warn"; deck: string; slide: string; code: string; message: string }} Finding
 */

function finding(severity, deck, slide, code, message) {
  return { severity, deck, slide, code, message };
}

/** Timeline cue target ids (`target` and/or stagger `targets`). */
function cueTargets(cue) {
  const out = [];
  if (typeof cue?.target === "string" && cue.target.length > 0) {
    out.push(cue.target);
  }
  if (Array.isArray(cue?.targets)) {
    for (const t of cue.targets) {
      if (typeof t === "string" && t.length > 0) out.push(t);
    }
  }
  return out;
}

function boxGeoms(nodes) {
  const boxes = [];
  for (const node of nodes) {
    if (!isBoxLike(node) || isArrowLike(node) || isDotLike(node)) continue;
    const g = geomOf(node);
    if (!g) continue;
    if (g.width <= 0 || g.height <= 0) continue;
    boxes.push(g);
  }
  return boxes;
}

function endpointAnchored(point, boxes, snap = SNAP_PX) {
  return boxes.some((b) => pointNearBox(point, b, snap));
}

function pointNearPath(point, points, snap = SNAP_PX) {
  if (points.length === 0) return false;
  for (const p of points) {
    if (Math.hypot(point.x - p.x, point.y - p.y) <= snap) return true;
  }
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const len2 = abx * abx + aby * aby;
    if (len2 <= 1e-6) continue;
    let t = ((point.x - a.x) * abx + (point.y - a.y) * aby) / len2;
    t = Math.min(1, Math.max(0, t));
    const cx = a.x + t * abx;
    const cy = a.y + t * aby;
    if (Math.hypot(point.x - cx, point.y - cy) <= snap) return true;
  }
  return false;
}

function fanCardinalityValid(node) {
  if (node?.capability === "core.fan-out") {
    return (
      node.from !== null &&
      typeof node.from === "object" &&
      !Array.isArray(node.from) &&
      Array.isArray(node.to) &&
      node.to.length >= 2
    );
  }
  if (node?.capability === "core.fan-in") {
    return (
      Array.isArray(node.from) &&
      node.from.length >= 2 &&
      node.to !== null &&
      typeof node.to === "object" &&
      !Array.isArray(node.to)
    );
  }
  return false;
}

function pointsNear(left, right, tolerance = 0.001) {
  return (
    Math.abs(left.x - right.x) <= tolerance &&
    Math.abs(left.y - right.y) <= tolerance
  );
}

function fanGeometryConnected(geometry) {
  if (
    geometry.trunk.length < 2 ||
    geometry.branches.some((branch) => branch.length < 2)
  ) {
    return false;
  }
  const trunkTouchesJunction = geometry.trunk.some((point) =>
    pointsNear(point, geometry.junction),
  );
  const branchesTouchJunction = geometry.branches.every((branch) =>
    branch.some((point) => pointsNear(point, geometry.junction)),
  );
  return trunkTouchesJunction && branchesTouchJunction;
}

function hasFanTraceCue(timeline, fanId) {
  return (timeline ?? []).some(
    (cue) =>
      String(cue?.action ?? "").toLowerCase() === "trace" &&
      cueTargets(cue).includes(fanId),
  );
}

function hasStaggeredBranchMotion(timeline, nodesById) {
  return (timeline ?? []).some((cue) => {
    if (String(cue?.action ?? "").toLowerCase() !== "stagger") return false;
    const targets = [...new Set(cueTargets(cue))];
    return (
      targets.length >= 2 &&
      targets.filter((target) => isMotionSignalNode(nodesById.get(target)))
        .length >= 2
    );
  });
}

/**
 * Verify one scene's geometry/timeline contract (shared by slide `render.scene`
 * and `finalCard.scene` — both are `SceneIrLike` and must satisfy the same
 * SceneRenderer invariants).
 *
 * SceneRenderer contract: arrowheads appear only when drawProgress >= 1
 * (or reduced motion / no draw cue). Play layer asserts the live SVG side.
 *
 * @param {string} deck
 * @param {string} slideLabel
 * @param {object} scene
 * @param {{ viewport?: unknown; strictDraw?: boolean }} options
 * @param {Finding[]} findings
 */
function verifySceneIr(deck, slideLabel, scene, options, findings) {
    const roots = scene.roots;
    const timeline = scene.timeline;
    const viewport = sceneViewport(scene, options.viewport);
    if (!Array.isArray(roots) || roots.length === 0) {
      findings.push(
        finding(
          "error",
          deck,
          slideLabel,
          "empty-roots",
          "scene.roots is empty",
        ),
      );
      return;
    }
    if (!Array.isArray(timeline) || timeline.length === 0) {
      findings.push(
        finding(
          "error",
          deck,
          slideLabel,
          "empty-timeline",
          "scene.timeline is empty",
        ),
      );
    }

    const nodes = walkNodes(roots);
    const ids = nodeIds(roots);
    const boxes = boxGeoms(nodes);
    const drawTargets = new Set(
      (timeline ?? [])
        .filter((c) => isDrawAction(c?.action))
        .map((c) => c.target)
        .filter((t) => typeof t === "string" && t.length > 0),
    );

    const nodesById = new Map(
      nodes
        .filter((n) => typeof n?.id === "string" && n.id.length > 0)
        .map((n) => [n.id, n]),
    );
    const staggeredBranchMotion = hasStaggeredBranchMotion(
      timeline,
      nodesById,
    );

    for (const cue of timeline ?? []) {
      const targets = cueTargets(cue);
      if (targets.length === 0) {
        findings.push(
          finding(
            "error",
            deck,
            slideLabel,
            "cue-missing-target",
            `timeline cue ${cue?.id ?? "?"} missing target`,
          ),
        );
        continue;
      }
      for (const target of targets) {
        if (!ids.has(target)) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "cue-unknown-target",
              `timeline target "${target}" not in scene roots`,
            ),
          );
        }
      }
    }

    const pathPolylines = [];
    /** @type {Map<string, {x:number,y:number, node: object}>} */
    const dotCenters = new Map();
    /** @type {string[]} */
    const directedArrowIds = [];

    for (const node of nodes) {
      const id = node.id ?? "?";

      if (isFanNode(node)) {
        const cardinalityValid = fanCardinalityValid(node);
        if (!cardinalityValid) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "fan-invalid-cardinality",
              `fan "${id}" must be fan-out with one source and at least two destinations, or fan-in with at least two sources and one destination`,
            ),
          );
        } else {
          const geometry = resolveFanGeometry(node, nodesById);
          if (!geometry || !fanGeometryConnected(geometry)) {
            findings.push(
              finding(
                "error",
                deck,
                slideLabel,
                "fan-disconnected-junction",
                `fan "${id}" does not resolve to a finite trunk and branches connected at one junction`,
              ),
            );
          } else {
            pathPolylines.push(...geometry.trajectories);
            directedArrowIds.push(id);
          }
        }
        if (!hasFanTraceCue(timeline, id) && !staggeredBranchMotion) {
          findings.push(
            finding(
              "warn",
              deck,
              slideLabel,
              "fan-missing-trace-cue",
              `fan "${id}" has no trace cue and no staggered branch motion substitute`,
            ),
          );
        }
        continue;
      }

      if (isArrowLike(node)) {
        const path = arrowPathData(node, nodesById);
        if (!path) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "arrow-missing-path",
              `arrow/path node "${id}" has no path/d data`,
            ),
          );
          continue;
        }
        const pts = pathPoints(path);
        if (pts.length < 2) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "arrow-degenerate-path",
              `arrow/path node "${id}" path does not yield endpoints`,
            ),
          );
          continue;
        }
        pathPolylines.push(pts);

        // Motion guides and headless dividers are not connectors.
        if (!isDirectedConnector(node)) {
          continue;
        }
        directedArrowIds.push(id);

        const start = pts[0];
        const end = pts[pts.length - 1];
        if (boxes.length > 0) {
          const startOk = endpointAnchored(start, boxes);
          const endOk = endpointAnchored(end, boxes);
          if (!startOk && !endOk) {
            findings.push(
              finding(
                "warn",
                deck,
                slideLabel,
                "floating-arrow",
                `arrow "${id}" endpoints float away from all boxes (snap ${SNAP_PX}px)`,
              ),
            );
          } else if (!startOk || !endOk) {
            findings.push(
              finding(
                "warn",
                deck,
                slideLabel,
                "loose-arrow",
                `arrow "${id}" has one unanchored endpoint`,
              ),
            );
          }
        }
        continue;
      }

      if (isDotLike(node)) {
        const g = geomOf(node);
        if (!g) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "floating-dot",
              `dot "${id}" missing geometry`,
            ),
          );
          continue;
        }
        // Legacy motion companions are dropped by SceneRenderer — flag once
        // so packages/authoring can delete them, then skip proximity checks.
        if (isMotionCompanionDot(node)) {
          findings.push(
            finding(
              "warn",
              deck,
              slideLabel,
              "obsolete-motion-companion",
              `dot "${id}" is a legacy motion companion; remove it (MotionSignal on the path owns the visual)`,
            ),
          );
          continue;
        }
        // Legend chips may float; other dots should sit near a stroke.
        if (!isLegendDot(node)) {
          dotCenters.set(id, {
            x: g.x + g.width / 2,
            y: g.y + g.height / 2,
            node,
          });
        }
        continue;
      }

      if (isBoxLike(node)) {
        const g = geomOf(node);
        if (!g) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "missing-geometry",
              `node "${id}" missing finite geometry`,
            ),
          );
          continue;
        }
        if (g.width <= 0 || g.height <= 0) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "zero-area-box",
              `box "${id}" has non-positive size ${g.width}×${g.height}`,
            ),
          );
          continue;
        }
        if (!inViewport(g, viewport)) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "out-of-viewport",
              `box "${id}" at (${g.x},${g.y}) ${g.width}×${g.height} outside ${viewport.width}×${viewport.height}`,
            ),
          );
        }
      }
    }

    for (const [id, entry] of dotCenters) {
      const center = { x: entry.x, y: entry.y };
      if (pathPolylines.length === 0) continue;
      const near = pathPolylines.some((pts) => pointNearPath(center, pts));
      if (!near) {
        findings.push(
          finding(
            "warn",
            deck,
            slideLabel,
            "orphan-dot",
            `dot "${id}" is not near any path stroke (snap ${SNAP_PX}px)`,
          ),
        );
      }
    }

    // When the scene uses draw reveals, every directed connector should have one
    // so tips are deferred consistently (SceneRenderer drawProgress contract).
    if (drawTargets.size > 0) {
      for (const id of directedArrowIds) {
        if (!drawTargets.has(id)) {
          findings.push(
            finding(
              "warn",
              deck,
              slideLabel,
              "missing-draw-cue",
              `directed arrow "${id}" has no draw/trace/reveal-stroke cue while siblings do`,
            ),
          );
        }
      }
    }

    // Playhead samples across the timeline: mid-draw moments must defer heads.
    if (options.strictDraw) {
      const duration = timelineDurationMs(timeline);
      const seen = new Set();
      const samples = new Set([0, Math.floor(duration / 2), duration]);
      for (const cue of timeline ?? []) {
        if (!isDrawAction(cue?.action)) continue;
        const at = Number(cue.at) || 0;
        const dur = Number(cue.duration) || 0;
        samples.add(at + Math.floor(dur * 0.5));
      }
      for (const t of samples) {
        for (const cue of timeline ?? []) {
          if (!isDrawAction(cue?.action)) continue;
          const progress = drawProgress(timeline, cue.target, t);
          if (
            progress !== undefined &&
            progress > 0 &&
            progress < 1 &&
            !seen.has(cue.target)
          ) {
            seen.add(cue.target);
            findings.push(
              finding(
                "warn",
                deck,
                slideLabel,
                "draw-in-flight",
                `t=${t}ms arrow "${cue.target}" mid-draw; UI must defer arrowhead (drawProgress>=1)`,
              ),
            );
          }
        }
      }
    }
}

const CURVE_ANCHORS = [
  "center",
  "n",
  "s",
  "e",
  "w",
  "ne",
  "nw",
  "se",
  "sw",
];

function pointFinite(point) {
  return (
    point != null &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y)
  );
}

function routeFinite(result) {
  if (/NaN|Infinity/.test(result.d)) return false;
  if (!result.waypoints.every(pointFinite)) return false;
  for (const segment of result.segments) {
    if (
      !pointFinite(segment.start) ||
      !pointFinite(segment.control1) ||
      !pointFinite(segment.control2) ||
      !pointFinite(segment.end)
    ) {
      return false;
    }
  }
  return true;
}

function samePoint(a, b) {
  return Math.abs(a.x - b.x) < 1e-6 && Math.abs(a.y - b.y) < 1e-6;
}

/** Build the deterministic synthetic routing scenarios (deck-independent). */
function buildCurveScenarios() {
  const scenarios = [];
  const middle = { id: "middle", bounds: { x: 170, y: 55, width: 60, height: 90 } };
  for (const fromAnchor of CURVE_ANCHORS) {
    for (const toAnchor of CURVE_ANCHORS) {
      scenarios.push({
        id: `anchors-${fromAnchor}-${toAnchor}`,
        start: { x: 40, y: 100 },
        end: { x: 360, y: 100 },
        fromAnchor,
        toAnchor,
        obstacles: [middle],
      });
    }
  }
  scenarios.push({
    id: "two-obstacles",
    start: { x: 40, y: 100 },
    end: { x: 420, y: 100 },
    fromAnchor: "e",
    toAnchor: "w",
    obstacles: [
      { id: "a", bounds: { x: 140, y: 40, width: 50, height: 120 } },
      { id: "b", bounds: { x: 260, y: 40, width: 50, height: 120 } },
    ],
  });
  scenarios.push({
    id: "same-side",
    start: { x: 60, y: 80 },
    end: { x: 300, y: 80 },
    fromAnchor: "n",
    toAnchor: "n",
    obstacles: [],
  });
  scenarios.push({
    id: "overlapping-bounds",
    start: { x: 100, y: 100 },
    end: { x: 130, y: 110 },
    fromAnchor: "e",
    toAnchor: "w",
    sourceId: "src",
    targetId: "dst",
    sourceBounds: { x: 60, y: 60, width: 80, height: 80 },
    targetBounds: { x: 90, y: 70, width: 80, height: 80 },
    obstacles: [],
  });
  scenarios.push({
    id: "self-loop",
    start: { x: 120, y: 60 },
    end: { x: 160, y: 60 },
    fromAnchor: "n",
    toAnchor: "n",
    sourceId: "self",
    targetId: "self",
    sourceBounds: { x: 100, y: 60, width: 80, height: 60 },
    targetBounds: { x: 100, y: 60, width: 80, height: 60 },
    obstacles: [],
  });
  const parallelBase = {
    start: { x: 40, y: 100 },
    end: { x: 320, y: 100 },
    fromAnchor: "e",
    toAnchor: "w",
    obstacles: [],
  };
  for (const [index, laneOffset] of [-8, 0, 8].entries()) {
    scenarios.push({
      id: `parallel-${index}`,
      ...parallelBase,
      laneOffset,
      group: "parallel",
    });
  }
  for (const index of [0, 1, 2]) {
    scenarios.push({
      id: `bundle-${index}`,
      ...parallelBase,
      laneOffset: 0,
      group: "bundle",
    });
  }
  scenarios.push({
    id: "forced-fallback",
    start: { x: 40, y: 100 },
    end: { x: 360, y: 100 },
    fromAnchor: "e",
    toAnchor: "w",
    // Inflated bounds contain both endpoints, so the router cannot avoid it and
    // must emit a deterministic penetrating fallback with reported ids.
    obstacles: [{ id: "wall", bounds: { x: 0, y: 60, width: 400, height: 80 } }],
    expectFallback: true,
  });
  return scenarios;
}

/**
 * Deck-independent verification of the deterministic curved router: determinism,
 * finite geometry, exact endpoints, no unexpected obstacle penetration, correct
 * fallback reporting, and lane/bundle separation. Returns only errors so a clean
 * run reports zero findings.
 *
 * @returns {Finding[]}
 */
export function verifyAdvancedCurveRouting() {
  const deck = "curve-router";
  const findings = [];
  const scenarios = buildCurveScenarios();
  const byId = new Map();
  for (const scenario of scenarios) {
    const input = {
      edgeId: scenario.id,
      start: scenario.start,
      end: scenario.end,
      fromAnchor: scenario.fromAnchor,
      toAnchor: scenario.toAnchor,
      sourceId: scenario.sourceId,
      targetId: scenario.targetId,
      sourceBounds: scenario.sourceBounds,
      targetBounds: scenario.targetBounds,
      obstacles: scenario.obstacles ?? [],
      siblings: [],
      options: normalizeCurveRouteOptions(scenario.style),
      laneOffset: scenario.laneOffset,
    };
    const first = routeCurve(input);
    const second = routeCurve(input);
    byId.set(scenario.id, first);

    if (first.d !== second.d) {
      findings.push(
        finding("error", deck, scenario.id, "curve-nondeterministic", "repeated routeCurve produced different SVG"),
      );
    }
    if (!routeFinite(first)) {
      findings.push(
        finding("error", deck, scenario.id, "curve-non-finite", `non-finite route geometry: ${first.d}`),
      );
    }
    const expectStart = {
      x: Math.round(scenario.start.x * 1000) / 1000,
      y: Math.round(scenario.start.y * 1000) / 1000,
    };
    const expectEnd = {
      x: Math.round(scenario.end.x * 1000) / 1000,
      y: Math.round(scenario.end.y * 1000) / 1000,
    };
    const gotStart = first.segments[0]?.start ?? first.waypoints[0];
    const gotEnd = first.segments.at(-1)?.end ?? first.waypoints.at(-1);
    if (gotStart === undefined || !samePoint(gotStart, expectStart)) {
      findings.push(
        finding("error", deck, scenario.id, "curve-start-drift", "route does not begin at the authored start"),
      );
    }
    if (gotEnd === undefined || !samePoint(gotEnd, expectEnd)) {
      findings.push(
        finding("error", deck, scenario.id, "curve-end-drift", "route does not end at the authored end"),
      );
    }
    if (first.penetratedObstacleIds.length > 0 && !first.usedFallback) {
      findings.push(
        finding(
          "error",
          deck,
          scenario.id,
          "curve-penetration",
          `route pierces ${first.penetratedObstacleIds.join(", ")} without fallback`,
        ),
      );
    }
    if (scenario.expectFallback) {
      if (!first.usedFallback) {
        findings.push(
          finding("error", deck, scenario.id, "curve-expected-fallback", "expected a deterministic fallback route"),
        );
      }
      if (first.penetratedObstacleIds.length === 0) {
        findings.push(
          finding("error", deck, scenario.id, "curve-fallback-no-ids", "fallback route did not report penetrated obstacles"),
        );
      }
    }
  }

  // Parallel lanes must separate; bundled edges must coincide.
  const parallelDs = new Set(
    ["parallel-0", "parallel-1", "parallel-2"].map((id) => byId.get(id)?.d),
  );
  if (parallelDs.size !== 3) {
    findings.push(
      finding("error", deck, "parallel", "curve-lanes-collapsed", "parallel lanes did not separate into distinct routes"),
    );
  }
  const bundleDs = new Set(
    ["bundle-0", "bundle-1", "bundle-2"].map((id) => byId.get(id)?.d),
  );
  if (bundleDs.size !== 1) {
    findings.push(
      finding("error", deck, "bundle", "curve-bundle-split", "bundled edges did not share one corridor"),
    );
  }

  const anchoredElbow = arrowPathData(
    {
      id: "anchored-elbow",
      kind: "elbow",
      from: { x: 100, y: 100, anchor: "e" },
      to: { x: 200, y: 300, anchor: "w" },
    },
    new Map(),
  );
  if (anchoredElbow !== "M100 100 H150 V300 H200") {
    findings.push(
      finding(
        "error",
        deck,
        "anchored-elbow",
        "elbow-runs-along-component-edge",
        `east-to-west elbow must approach the target horizontally; got ${anchoredElbow}`,
      ),
    );
  }
  return findings;
}

const ROUTING_EXEMPLAR_TITLES = [
  "Complete 9×9 curve matrix",
  "Cardinal curves",
  "Corner and center curves",
  "Same-side links and self-loops",
  "Obstacle avoidance",
  "Parallel lanes",
  "Bundling",
  "Anchor-safe orthogonal routing",
  "Routing controls reference",
];

/** Pin the complete routing cookbook and its 81 ordered anchor pairs. */
function verifyRoutingSdkExamples(pkg, findings) {
  if (pkg?.id !== "flow-sdk-examples") return;
  const slides = Array.isArray(pkg.slides) ? pkg.slides : [];
  if (slides.length !== 19) {
    findings.push(
      finding(
        "error",
        pkg.id,
        "*",
        "routing-exemplar-slide-count",
        `expected 19 slides, got ${slides.length}`,
      ),
    );
  }
  const titles = new Set(slides.map((slide) => slide?.title));
  for (const title of ROUTING_EXEMPLAR_TITLES) {
    if (!titles.has(title)) {
      findings.push(
        finding(
          "error",
          pkg.id,
          "*",
          "routing-exemplar-missing-slide",
          `missing "${title}"`,
        ),
      );
    }
  }
  const matrix = slides.find(
    (slide) => slide?.title === "Complete 9×9 curve matrix",
  );
  const roots = Array.isArray(matrix?.render?.scene?.roots)
    ? matrix.render.scene.roots
    : [];
  const pairs = new Set(
    roots
      .filter((node) => node?.style?.route === "curve")
      .map(
        (node) =>
          `${node?.from?.anchor ?? "center"}:${node?.to?.anchor ?? "center"}`,
      ),
  );
  const anchors = ["center", "n", "s", "e", "w", "ne", "nw", "se", "sw"];
  for (const from of anchors) {
    for (const to of anchors) {
      if (!pairs.has(`${from}:${to}`)) {
        findings.push(
          finding(
            "error",
            pkg.id,
            "curve-matrix",
            "routing-anchor-pair-missing",
            `missing ${from} → ${to}`,
          ),
        );
      }
    }
  }
}

/**
 * Verify one DeckPackage's scenes (static + optional mid-draw contract samples).
 *
 * Checks every slide's `render.scene` plus, when authored, `finalCard.scene`
 * (three decks ship a scene-backed final card that is otherwise unchecked by
 * this static gate — Play exercises it live, but IR must catch authoring
 * regressions before a browser run).
 *
 * @returns {Finding[]}
 */
export function verifyPackageIr(pkg, options = {}) {
  const deck = pkg?.id ?? "unknown";
  const findings = [];
  const slides = Array.isArray(pkg?.slides) ? pkg.slides : [];

  if (slides.length === 0) {
    findings.push(
      finding("error", deck, "*", "empty-slides", "package has no slides"),
    );
    return findings;
  }

  verifyRoutingSdkExamples(pkg, findings);

  slides.forEach((slide, index) => {
    const slideLabel = `${index}:${slide?.id ?? slide?.title ?? "slide"}`;
    const render = slide?.render;
    if (render == null) return;

    const scene = render.scene;
    if (scene == null || typeof scene !== "object") {
      findings.push(
        finding(
          "error",
          deck,
          slideLabel,
          "missing-scene",
          "render present without scene",
        ),
      );
      return;
    }

    verifySceneIr(deck, slideLabel, scene, options, findings);
  });

  const finalCardScene = pkg?.finalCard?.scene;
  if (finalCardScene != null && typeof finalCardScene === "object") {
    verifySceneIr(deck, "finalCard", finalCardScene, options, findings);
  }

  return findings;
}
