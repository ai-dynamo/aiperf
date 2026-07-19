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
  pathPoints,
  pointNearBox,
  resolveFanGeometry,
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
