/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  DEFAULT_VIEWPORT,
  SNAP_PX,
  arrowPathData,
  drawProgress,
  geomOf,
  inViewport,
  isArrowLike,
  isBoxLike,
  isDotLike,
  nodeIds,
  pathPoints,
  pointNearBox,
  walkNodes,
} from "./geometry.mjs";

/**
 * @typedef {{ severity: "error" | "warn"; deck: string; slide: string; code: string; message: string }} Finding
 */

function finding(severity, deck, slide, code, message) {
  return { severity, deck, slide, code, message };
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

function isDrawActionSafe(action) {
  const a = String(action ?? "").toLowerCase();
  return a === "draw" || a === "trace" || a === "reveal-stroke";
}

/**
 * Verify one DeckPackage's scenes (static + optional mid-draw contract samples).
 * @returns {Finding[]}
 */
export function verifyPackageIr(pkg, options = {}) {
  const deck = pkg?.id ?? "unknown";
  const findings = [];
  const viewport = { ...DEFAULT_VIEWPORT, ...(options.viewport ?? {}) };
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

    const roots = scene.roots;
    const timeline = scene.timeline;
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

    for (const cue of timeline ?? []) {
      const target = cue?.target;
      if (typeof target !== "string" || target.length === 0) {
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

    const pathPolylines = [];
    /** @type {Map<string, {x:number,y:number}>} */
    const dotCenters = new Map();

    for (const node of nodes) {
      const id = node.id ?? "?";

      if (isArrowLike(node)) {
        const path = arrowPathData(node);
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
        dotCenters.set(id, {
          x: g.x + g.width / 2,
          y: g.y + g.height / 2,
        });
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

    for (const [id, center] of dotCenters) {
      if (/motion/i.test(id)) continue;
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

    // Optional mid-draw contract samples (one warn per arrow) when --strict-draw.
    if (options.strictDraw) {
      const seen = new Set();
      for (const cue of timeline ?? []) {
        if (!isDrawActionSafe(cue.action)) continue;
        const at = Number(cue.at) || 0;
        const dur = Number(cue.duration) || 0;
        const t = at + Math.floor(dur * 0.5);
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
              `t=${t}ms arrow "${cue.target}" mid-draw; UI must defer arrowhead`,
            ),
          );
        }
      }
    }
  });

  return findings;
}
