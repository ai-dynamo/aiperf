/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! In-memory DeckPackage verification and evaluator-backed diagnostics.

import type {
  DeckPackage,
  RenderNodeIr,
  SceneIr,
  TimelineCueIr,
} from "../schema/index.js";
import { resolveCapabilityId } from "../schema/index.js";
import {
  applyTimelineState,
  evaluateFrame,
  evaluateTimelineState,
  type FrozenCapabilityEvaluatorRegistry,
} from "../runtime/index.js";
import {
  createDevEvaluatorRegistry,
  registeredDevCapabilityIds,
} from "./evaluator-registry.js";
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
  type FanGeometry,
  type Geometry,
  type Point,
  type Viewport,
} from "./verify-geometry.js";

/** One stable finding emitted by static or evaluator-backed verification. */
export type VerificationFinding = Readonly<{
  severity: "error" | "warn";
  deck: string;
  slide: string;
  code: string;
  message: string;
}>;

/** Optional strict draw sampling and viewport override. */
export type VerifyPackageOptions = Readonly<{
  strictDraw?: boolean;
  viewport?: Partial<Viewport>;
}>;

function finding(
  severity: VerificationFinding["severity"],
  deck: string,
  slide: string,
  code: string,
  message: string,
): VerificationFinding {
  return { severity, deck, slide, code, message };
}

/** Timeline cue target ids (`target` and/or stagger `targets`). */
function cueTargets(cue: TimelineCueIr): string[] {
  const out: string[] = [];
  if (typeof cue.target === "string" && cue.target.length > 0) {
    out.push(cue.target);
  }
  if (Array.isArray(cue.targets)) {
    for (const target of cue.targets) {
      if (typeof target === "string" && target.length > 0) out.push(target);
    }
  }
  return out;
}

function boxGeometries(nodes: readonly RenderNodeIr[]): Geometry[] {
  const boxes: Geometry[] = [];
  for (const node of nodes) {
    if (!isBoxLike(node) || isArrowLike(node) || isDotLike(node)) continue;
    const geometry = geomOf(node);
    if (!geometry || geometry.width <= 0 || geometry.height <= 0) continue;
    boxes.push(geometry);
  }
  return boxes;
}

function endpointAnchored(
  point: Point,
  boxes: readonly Geometry[],
  snap = SNAP_PX,
): boolean {
  return boxes.some((box) => pointNearBox(point, box, snap));
}

function pointNearPath(
  point: Point,
  points: readonly Point[],
  snap = SNAP_PX,
): boolean {
  if (points.length === 0) return false;
  for (const candidate of points) {
    if (Math.hypot(point.x - candidate.x, point.y - candidate.y) <= snap) {
      return true;
    }
  }
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= 1e-6) continue;
    let progress =
      ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared;
    progress = Math.min(1, Math.max(0, progress));
    const closestX = start.x + progress * dx;
    const closestY = start.y + progress * dy;
    if (Math.hypot(point.x - closestX, point.y - closestY) <= snap) {
      return true;
    }
  }
  return false;
}

function fanCardinalityValid(node: RenderNodeIr): boolean {
  const value = node as unknown as Readonly<Record<string, unknown>>;
  if (value.capability === "core.fan-out") {
    return (
      value.from !== null &&
      typeof value.from === "object" &&
      !Array.isArray(value.from) &&
      Array.isArray(value.to) &&
      value.to.length >= 2
    );
  }
  if (value.capability === "core.fan-in") {
    return (
      Array.isArray(value.from) &&
      value.from.length >= 2 &&
      value.to !== null &&
      typeof value.to === "object" &&
      !Array.isArray(value.to)
    );
  }
  return false;
}

function pointsNear(left: Point, right: Point, tolerance = 0.001): boolean {
  return (
    Math.abs(left.x - right.x) <= tolerance &&
    Math.abs(left.y - right.y) <= tolerance
  );
}

function fanGeometryConnected(geometry: FanGeometry): boolean {
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

function hasFanTraceCue(
  timeline: readonly TimelineCueIr[],
  fanId: string,
): boolean {
  return timeline.some(
    (cue) =>
      String(cue.action ?? "").toLowerCase() === "trace" &&
      cueTargets(cue).includes(fanId),
  );
}

function hasStaggeredBranchMotion(
  timeline: readonly TimelineCueIr[],
  nodesById: ReadonlyMap<string, RenderNodeIr>,
): boolean {
  return timeline.some((cue) => {
    if (String(cue.action ?? "").toLowerCase() !== "stagger") return false;
    const targets = [...new Set(cueTargets(cue))];
    return (
      targets.length >= 2 &&
      targets.filter((target) => isMotionSignalNode(nodesById.get(target)))
        .length >= 2
    );
  });
}

/**
 * Verifies one in-memory DeckPackage without modifying it.
 *
 * Finding codes and thresholds intentionally mirror the Node flow verifier.
 */
export function verifyPackageIr(
  pkg: DeckPackage,
  options: VerifyPackageOptions = {},
): readonly VerificationFinding[] {
  const deck = pkg.id ?? "unknown";
  const findings: VerificationFinding[] = [];
  const slides = Array.isArray(pkg.slides) ? pkg.slides : [];

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
    const timeline: TimelineCueIr[] = Array.isArray(scene.timeline)
      ? (scene.timeline as TimelineCueIr[])
      : [];
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
    const boxes = boxGeometries(nodes);
    const drawTargets = new Set(
      timeline
        .filter((cue) => isDrawAction(cue.action))
        .map((cue) => cue.target)
        .filter((target) => typeof target === "string" && target.length > 0),
    );
    const nodesById = new Map(
      nodes
        .filter(({ id }) => typeof id === "string" && id.length > 0)
        .map((node) => [node.id, node]),
    );
    const staggeredBranchMotion = hasStaggeredBranchMotion(
      timeline,
      nodesById,
    );

    for (const cue of timeline) {
      const targets = cueTargets(cue);
      if (targets.length === 0) {
        findings.push(
          finding(
            "error",
            deck,
            slideLabel,
            "cue-missing-target",
            `timeline cue ${cue.id ?? "?"} missing target`,
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

    const pathPolylines: Point[][] = [];
    const dotCenters = new Map<string, Point>();
    const directedArrowIds: string[] = [];

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
            pathPolylines.push(
              ...geometry.trajectories.map((points) => [...points]),
            );
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
        const points = pathPoints(path);
        if (points.length < 2) {
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
        pathPolylines.push(points);

        if (!isDirectedConnector(node)) continue;
        directedArrowIds.push(id);

        const start = points[0];
        const end = points[points.length - 1];
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
        const geometry = geomOf(node);
        if (!geometry) {
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
        if (!isLegendDot(node)) {
          dotCenters.set(id, {
            x: geometry.x + geometry.width / 2,
            y: geometry.y + geometry.height / 2,
          });
        }
        continue;
      }

      if (isBoxLike(node)) {
        const geometry = geomOf(node);
        if (!geometry) {
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
        if (geometry.width <= 0 || geometry.height <= 0) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "zero-area-box",
              `box "${id}" has non-positive size ${geometry.width}×${geometry.height}`,
            ),
          );
          continue;
        }
        if (!inViewport(geometry, viewport)) {
          findings.push(
            finding(
              "error",
              deck,
              slideLabel,
              "out-of-viewport",
              `box "${id}" at (${geometry.x},${geometry.y}) ${geometry.width}×${geometry.height} outside ${viewport.width}×${viewport.height}`,
            ),
          );
        }
      }
    }

    for (const [id, center] of dotCenters) {
      if (pathPolylines.length === 0) continue;
      const near = pathPolylines.some((points) =>
        pointNearPath(center, points),
      );
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

    if (options.strictDraw) {
      const duration = timelineDurationMs(timeline);
      const seen = new Set<string>();
      const samples = new Set([0, Math.floor(duration / 2), duration]);
      for (const cue of timeline) {
        if (!isDrawAction(cue.action)) continue;
        const at = Number(cue.at) || 0;
        const cueDuration = Number(cue.duration) || 0;
        samples.add(at + Math.floor(cueDuration * 0.5));
      }
      for (const timeMs of samples) {
        for (const cue of timeline) {
          if (!isDrawAction(cue.action)) continue;
          const progress = drawProgress(timeline, cue.target, timeMs);
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
                `t=${timeMs}ms arrow "${cue.target}" mid-draw; UI must defer arrowhead (drawProgress>=1)`,
              ),
            );
          }
        }
      }
    }
  });

  return findings;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function representativeTimes(timeline: readonly TimelineCueIr[]): number[] {
  // Match runtime timelineDurationMs: ceil so fractional cue ends reach final state.
  const duration = Math.max(0, Math.ceil(timelineDurationMs(timeline)));
  return [...new Set([0, Math.floor(duration / 2), duration])];
}

function isStaggerLikeAction(action: string): boolean {
  return action === "stagger" || action === "enter-children";
}

function directChildIds(node: RenderNodeIr | undefined): readonly string[] {
  if (node === undefined) return [];
  if (node.kind !== "group" && node.kind !== "component") return [];
  return node.children
    .map((child) => child.id)
    .filter((id) => typeof id === "string" && id.length > 0);
}

function resolveStaggerTargets(
  cue: TimelineCueIr,
  nodesById: ReadonlyMap<string, RenderNodeIr>,
): readonly string[] {
  if (Array.isArray(cue.targets) && cue.targets.length > 0) {
    return cue.targets.filter((id) => typeof id === "string" && id.length > 0);
  }
  if (cue.action === "enter-children" && cue.target.length > 0) {
    return directChildIds(nodesById.get(cue.target));
  }
  return [];
}

/**
 * Expands compact stagger / enter-children cues into per-target enter cues with
 * authored step delays. Mirrors SceneRenderer expansion for the evaluator oracle
 * without mutating the package timeline.
 */
function expandTimelineForEvaluation(
  timeline: readonly TimelineCueIr[],
  roots: readonly RenderNodeIr[],
): readonly TimelineCueIr[] {
  const nodesById = new Map(
    walkNodes(roots)
      .filter(({ id }) => typeof id === "string" && id.length > 0)
      .map((node) => [node.id, node]),
  );
  const expanded: TimelineCueIr[] = [];
  for (const cue of timeline) {
    if (!isStaggerLikeAction(String(cue.action))) {
      expanded.push(cue);
      continue;
    }
    const targets = resolveStaggerTargets(cue, nodesById);
    if (targets.length === 0) {
      expanded.push({
        ...cue,
        action: "enter",
        target: cue.target.length > 0 ? cue.target : cue.id,
      });
      continue;
    }
    const step = Math.max(
      0,
      typeof cue.step === "number" && Number.isFinite(cue.step) ? cue.step : 80,
    );
    const at = Number(cue.at) || 0;
    const duration = Number(cue.duration) || 0;
    targets.forEach((targetId, index) => {
      expanded.push({
        id: `${cue.id}__${index}`,
        at: at + index * step,
        duration,
        action: "enter",
        target: targetId,
        sourceMap: cue.sourceMap,
        ...(cue.easing !== undefined ? { easing: cue.easing } : {}),
      });
    });
  }
  return expanded;
}

function collectComponentCapabilityIds(
  roots: readonly RenderNodeIr[],
): readonly string[] {
  const ids: string[] = [];
  for (const node of walkNodes(roots)) {
    if (node.kind === "component") {
      ids.push(resolveCapabilityId(node));
    }
  }
  return ids;
}

function hasAbsoluteConnector(roots: readonly RenderNodeIr[]): boolean {
  return walkNodes(roots).some((node) => {
    if (node.kind !== "connector") return false;
    return (
      typeof node.from.nodeId !== "string" ||
      node.from.nodeId.length === 0 ||
      typeof node.to.nodeId !== "string" ||
      node.to.nodeId.length === 0
    );
  });
}

function oracleSkipReason(
  scene: SceneIr,
  registered: ReadonlySet<string>,
): string | undefined {
  const unsupported = [
    ...new Set(
      collectComponentCapabilityIds(scene.roots).filter(
        (id) => !registered.has(id),
      ),
    ),
  ];
  if (unsupported.length > 0) {
    return `unsupported component capabilities: ${unsupported.join(", ")}`;
  }
  if (hasAbsoluteConnector(scene.roots)) {
    return "foundation evaluator requires node-anchored connectors; absolute endpoints are SceneRenderer-only";
  }
  return undefined;
}

function evaluatorFindings(
  pkg: DeckPackage,
  registry: FrozenCapabilityEvaluatorRegistry,
): VerificationFinding[] {
  const findings: VerificationFinding[] = [];
  const registered = registeredDevCapabilityIds(registry);

  pkg.slides.forEach((slide, index) => {
    if (slide.render === undefined) return;
    const slideLabel = `${index}:${slide.id ?? slide.title ?? "slide"}`;
    const scene = slide.render.scene;
    const skipReason = oracleSkipReason(scene, registered);
    if (skipReason !== undefined) {
      findings.push(
        finding(
          "warn",
          pkg.id,
          slideLabel,
          "evaluator-skipped",
          skipReason,
        ),
      );
      return;
    }

    const expandedTimeline = expandTimelineForEvaluation(
      scene.timeline,
      scene.roots,
    );
    for (const timeMs of representativeTimes(expandedTimeline)) {
      try {
        const frame = evaluateFrame(scene, timeMs, {
          scene: { evaluators: registry },
        });
        const timelineState = evaluateTimelineState(expandedTimeline, timeMs);
        applyTimelineState(frame.displayList.commands, timelineState);
      } catch (error: unknown) {
        findings.push(
          finding(
            "error",
            pkg.id,
            slideLabel,
            "evaluator-failure",
            `t=${timeMs}ms ${errorMessage(error)}`,
          ),
        );
        break;
      }
    }
  });
  return findings;
}

function formatFinding(value: VerificationFinding): string {
  return `[${value.code}] deck=${value.deck} slide=${value.slide} ${value.message}`;
}

/**
 * Runs static and representative-state runtime diagnostics for packages.
 *
 * Each deck and slide is isolated, and all operations consume readonly package
 * data without changing package objects.
 */
export async function runDevDiagnostics(
  packages: readonly DeckPackage[],
): Promise<void> {
  const registry = createDevEvaluatorRegistry();
  for (const pkg of packages) {
    let findings: readonly VerificationFinding[];
    try {
      findings = [...verifyPackageIr(pkg), ...evaluatorFindings(pkg, registry)];
    } catch (error: unknown) {
      findings = [
        finding(
          "error",
          pkg.id,
          "*",
          "diagnostics-failure",
          errorMessage(error),
        ),
      ];
    }
    if (findings.length === 0) continue;

    console.groupCollapsed(
      `Flow diagnostics: ${pkg.id} (${findings.length} finding${findings.length === 1 ? "" : "s"})`,
    );
    try {
      for (const value of findings) {
        const message = formatFinding(value);
        if (value.severity === "error") {
          console.error(message);
        } else if (value.code === "evaluator-skipped") {
          console.info(message);
        } else {
          console.warn(message);
        }
      }
    } finally {
      console.groupEnd();
    }
  }
}
