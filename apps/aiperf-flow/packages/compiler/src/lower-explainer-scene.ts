/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower an embedded slide `@scene` to a DeckPackage `SceneRender`.
//!
//! Accepts:
//! - native cinematic `SceneAst` (rect/connector/timeline/camera)
//! - `embedded-scene-source` with form `native` (parsed via shared
//!   `parseNativeEmbeddedScene`)
//! - decks-flow `package-scene` (`roots` / `timeline` / `camera`)
//!
//! Native paths reuse the document scene lowerer. Package paths normalize the
//! JSON-ish authoring shape into strict `SceneIr`.

import {
  parseNativeEmbeddedScene,
  type EmbeddedSceneSource,
  type PackageSceneAst,
  type SceneAst,
  type ConnectorAst,
  type DocumentAst,
  type LiteralAst,
  type RectAst,
} from "@aiperf/flow-language";
import {
  diagnostic,
  sceneIrSchema,
  type RenderNodeIr,
  type Result,
  type SceneIr,
  type SceneRender,
  type SourceRange,
  type TimelineCueIr,
} from "@aiperf/flow-schema";

import type { LinkedDocument, SceneSymbolTable } from "./link.js";
import { lower } from "./lower.js";

export type LowerExplainerSceneOptions = Readonly<{
  tokens?: ReadonlyMap<string, LiteralAst["value"]>;
  /**
   * Owning slide id for fail-closed empty roots/timeline diagnostics.
   * When set, messages name the slide so authors can locate the bad `@scene`.
   */
  slideId?: string;
  /** Fill-ins when package scenes omit SceneIr identity fields. */
  defaults?: Readonly<{
    id?: string;
    title?: string;
    summary?: string;
    narration?: string;
    fallback?: string;
  }>;
}>;

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

function isSceneAst(value: unknown): value is SceneAst {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "scene" &&
    "id" in value &&
    typeof value.id === "string" &&
    "title" in value &&
    typeof value.title === "string" &&
    "renderDeclarations" in value &&
    Array.isArray(value.renderDeclarations)
  );
}

function isEmbeddedSceneSource(value: unknown): value is EmbeddedSceneSource {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "embedded-scene-source" &&
    "form" in value &&
    "body" in value &&
    typeof value.body === "string"
  );
}

function isPackageSceneAst(value: unknown): value is PackageSceneAst {
  if (typeof value !== "object" || value === null || !("roots" in value)) {
    return false;
  }
  if (!Array.isArray(value.roots)) {
    return false;
  }
  // Explicit package-scene, or PackageSceneIrAst (kind "scene" + preserved roots).
  if ("kind" in value && value.kind === "package-scene") {
    return true;
  }
  return (
    "kind" in value &&
    value.kind === "scene" &&
    "timeline" in value &&
    Array.isArray(value.timeline)
  );
}

function sceneSymbolTable(scene: SceneAst): SceneSymbolTable {
  const nodes = new Map<string, RectAst | ConnectorAst>();
  for (const node of scene.renderDeclarations) {
    if (
      (node.kind === "rect" || node.kind === "connector") &&
      !nodes.has(node.id)
    ) {
      nodes.set(node.id, node);
    }
  }
  return { nodes };
}

function wrapSceneDocument(
  scene: SceneAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): LinkedDocument {
  const document: DocumentAst = {
    kind: "document",
    id: `explainer-scene-${scene.id}`,
    title: scene.title,
    language: { kind: "language", version: 1, sourceMap: scene.sourceMap },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [scene],
    sourceMap: scene.sourceMap,
  };

  return {
    document,
    tokens,
    scenes: new Map([[scene.id, sceneSymbolTable(scene)]]),
    imports: new Map(),
    qualifiedNames: new Map(),
    themes: [],
  };
}

function invalidScene(
  message: string,
  range: SourceRange = unknownRange,
): Result<SceneRender> {
  return {
    ok: false,
    diagnostics: [
      diagnostic("EXPLAINER_SCENE_INVALID", "error", message, range),
    ],
  };
}

function slideLabel(options: LowerExplainerSceneOptions): string {
  return options.slideId !== undefined && options.slideId.length > 0
    ? `Slide "${options.slideId}"`
    : "Embedded @scene";
}

/** Fail-closed diagnostic when scene.roots or scene.timeline is empty. */
function emptySceneField(
  field: "roots" | "timeline",
  options: LowerExplainerSceneOptions,
  range: SourceRange = unknownRange,
  sceneId?: string,
): Result<SceneRender> {
  const owner = slideLabel(options);
  const sceneBit =
    sceneId !== undefined && sceneId.length > 0
      ? ` (scene "${sceneId}")`
      : "";
  if (field === "roots") {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "EXPLAINER_SCENE_ROOTS_REQUIRED",
          "error",
          `${owner}${sceneBit} has an embedded @scene with empty scene.roots.`,
          range,
          "Lower embedded @scene roots into at least one diagram node.",
        ),
      ],
    };
  }
  return {
    ok: false,
    diagnostics: [
      diagnostic(
        "EXPLAINER_TIMELINE_REQUIRED",
        "error",
        `${owner}${sceneBit} has an embedded @scene with empty scene.timeline.`,
        range,
        "Add at least one timeline cue that drives enter, draw, or emphasis motion.",
      ),
    ],
  };
}

function lowerNativeSceneAst(
  scene: SceneAst,
  options: LowerExplainerSceneOptions,
): Result<SceneRender> {
  let lowered;
  try {
    lowered = lower(wrapSceneDocument(scene, options.tokens ?? new Map()));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Scene lowering failed.";
    return invalidScene(message, scene.sourceMap);
  }

  const sceneIr = lowered.scenes[0];
  if (sceneIr === undefined) {
    return invalidScene(
      `Scene "${scene.id}" produced no SceneIr after lowering.`,
      scene.sourceMap,
    );
  }

  if (sceneIr.roots.length === 0) {
    return emptySceneField("roots", options, scene.sourceMap, scene.id);
  }
  if (sceneIr.timeline.length === 0) {
    return emptySceneField("timeline", options, scene.sourceMap, scene.id);
  }

  return validateSceneRender(sceneIr, scene.sourceMap);
}

function validateSceneRender(
  sceneIr: unknown,
  range: SourceRange,
): Result<SceneRender> {
  const parsed = sceneIrSchema.safeParse(sceneIr);
  if (!parsed.success) {
    return {
      ok: false,
      diagnostics: parsed.error.issues.map((issue) => {
        const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
        return diagnostic(
          "EXPLAINER_SCENE_INVALID",
          "error",
          `${path}: ${issue.message}`,
          range,
        );
      }),
    };
  }

  return {
    ok: true,
    value: { kind: "scene", scene: parsed.data },
    diagnostics: [],
  };
}

function capabilityKind(capability: string): RenderNodeIr["kind"] {
  const leaf = capability.includes(".")
    ? capability.slice(capability.lastIndexOf(".") + 1)
    : capability;
  switch (leaf) {
    case "text":
      return "text";
    case "connector":
    case "line":
    case "arrow":
    case "path":
      return "connector";
    case "group":
      return "group";
    default:
      return "rect";
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

function geometryOf(node: Record<string, unknown>): SceneIr["roots"][number]["geometry"] {
  const geometry = asRecord(node.geometry) ?? asRecord(node.layout) ?? {};
  return {
    x: Number(geometry.x ?? 0),
    y: Number(geometry.y ?? 0),
    width: Number(geometry.width ?? 0),
    height: Number(geometry.height ?? 0),
  };
}

function styleOf(node: Record<string, unknown>): Record<string, string | number | boolean> {
  const style = asRecord(node.style) ?? {};
  const out: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(style)) {
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      out[key] = value;
    }
  }
  return out;
}

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** Preserve authored SVG path data (`d` authoring alias → IR `path`). */
function pathOf(node: Record<string, unknown>): string | undefined {
  if (typeof node.path === "string" && node.path.length > 0) {
    return node.path;
  }
  if (typeof node.d === "string" && node.d.length > 0) {
    return node.d;
  }
  return undefined;
}

function pointsOf(
  node: Record<string, unknown>,
): ReadonlyArray<{ x: number; y: number }> | undefined {
  if (!Array.isArray(node.points) || node.points.length === 0) {
    return undefined;
  }
  const points: Array<{ x: number; y: number }> = [];
  for (const point of node.points) {
    const record = asRecord(point);
    if (record === undefined) {
      continue;
    }
    const x = finiteOrUndefined(record.x);
    const y = finiteOrUndefined(record.y);
    if (x === undefined || y === undefined) {
      continue;
    }
    points.push({ x, y });
  }
  return points.length > 0 ? points : undefined;
}

/**
 * Normalize package `from`/`to` endpoints.
 *
 * Decks-flow lines use `{ x, y }`; cinematic connectors use `{ nodeId, anchor? }`.
 * Path-only nodes without endpoints get a zero-point stub so schema stays
 * satisfiable while SceneRenderer prefers authored `path` / `d`.
 */
function connectorEndpointOf(
  value: unknown,
  fallback: "from" | "to",
): {
  nodeId?: string;
  anchor?: string;
  x?: number;
  y?: number;
} {
  const record = asRecord(value);
  if (record === undefined) {
    return { x: 0, y: 0 };
  }
  const x = finiteOrUndefined(record.x);
  const y = finiteOrUndefined(record.y);
  const nodeId =
    typeof record.nodeId === "string" && record.nodeId.length > 0
      ? record.nodeId
      : typeof record.id === "string" && record.id.length > 0
        ? record.id
        : undefined;
  const anchor =
    typeof record.anchor === "string" && record.anchor.length > 0
      ? record.anchor
      : undefined;
  if (x !== undefined && y !== undefined) {
    return {
      x,
      y,
      ...(nodeId !== undefined ? { nodeId } : {}),
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  if (nodeId !== undefined) {
    return {
      nodeId,
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  void fallback;
  return { x: 0, y: 0 };
}

function geometryFromEndpoints(
  from: { x?: number; y?: number },
  to: { x?: number; y?: number },
  fallback: SceneIr["roots"][number]["geometry"],
): SceneIr["roots"][number]["geometry"] {
  if (
    typeof from.x !== "number" ||
    typeof from.y !== "number" ||
    typeof to.x !== "number" ||
    typeof to.y !== "number"
  ) {
    return fallback;
  }
  const x = Math.min(from.x, to.x);
  const y = Math.min(from.y, to.y);
  return {
    x,
    y,
    width: Math.max(Math.abs(to.x - from.x), 0),
    height: Math.max(Math.abs(to.y - from.y), 0),
  };
}

function normalizePackageNode(value: unknown): RenderNodeIr {
  const node = asRecord(value);
  if (node === undefined) {
    throw new Error("package scene root must be an object");
  }
  const id = String(node.id ?? "node");
  const capability =
    typeof node.capabilityId === "string"
      ? node.capabilityId
      : typeof node.capability === "string"
        ? node.capability
        : typeof node.kind === "string"
          ? node.kind.includes(".")
            ? node.kind
            : `core.${node.kind}`
          : "core.rect";
  const children = Array.isArray(node.children)
    ? node.children.map(normalizePackageNode)
    : undefined;
  const kind =
    children !== undefined && children.length > 0
      ? "group"
      : capabilityKind(capability);
  const accessibilityRecord = asRecord(node.accessibility) ?? {};
  const label =
    typeof node.text === "string"
      ? node.text
      : typeof accessibilityRecord.label === "string"
        ? accessibilityRecord.label
        : id;
  const description =
    typeof accessibilityRecord.description === "string" &&
    accessibilityRecord.description.length > 0
      ? accessibilityRecord.description
      : undefined;
  const path = pathOf(node);
  const points = pointsOf(node);
  let geometry = geometryOf(node);
  const base = {
    id,
    capabilityId: capability,
    geometry,
    style: styleOf(node),
    accessibility: {
      label,
      ...(description !== undefined ? { description } : {}),
    },
    fallback: typeof node.fallback === "string" ? node.fallback : label,
    sourceMap: unknownRange,
    ...(path !== undefined ? { path } : {}),
    ...(points !== undefined ? { points } : {}),
  };

  if (kind === "group") {
    return { ...base, kind: "group", children: children ?? [] };
  }
  if (kind === "text") {
    return {
      ...base,
      kind: "text",
      text: typeof node.text === "string" ? node.text : label,
    };
  }
  if (kind === "connector") {
    const from = connectorEndpointOf(node.from, "from");
    const to = connectorEndpointOf(node.to, "to");
    if (
      geometry.width === 0 &&
      geometry.height === 0 &&
      geometry.x === 0 &&
      geometry.y === 0
    ) {
      geometry = geometryFromEndpoints(from, to, geometry);
    }
    return {
      ...base,
      geometry,
      kind: "connector",
      from,
      to,
    };
  }
  return { ...base, kind: "rect" };
}

function normalizePackageTimeline(value: unknown, index: number): TimelineCueIr {
  const cue = asRecord(value) ?? {};
  return {
    id: String(cue.id ?? `cue-${index}`),
    at: Number(cue.at ?? 0),
    duration: Number(cue.duration ?? 0),
    target: String(cue.target ?? ""),
    action: String(cue.action ?? "enter"),
    sourceMap: unknownRange,
  };
}

function lowerPackageScene(
  scene: PackageSceneAst,
  options: LowerExplainerSceneOptions,
): Result<SceneRender> {
  const defaults = options.defaults ?? {};
  const roots = scene.roots.map(normalizePackageNode);
  const timeline = (scene.timeline ?? []).map(normalizePackageTimeline);
  if (roots.length === 0) {
    return emptySceneField("roots", options);
  }
  if (timeline.length === 0) {
    return emptySceneField("timeline", options);
  }
  const id =
    (typeof scene.id === "string" && scene.id.length > 0
      ? scene.id
      : undefined) ??
    defaults.id ??
    "embedded";
  const title =
    (typeof scene.title === "string" && scene.title.length > 0
      ? scene.title
      : undefined) ??
    defaults.title ??
    "Embedded scene";
  const summary =
    (typeof scene.summary === "string" && scene.summary.length > 0
      ? scene.summary
      : undefined) ??
    defaults.summary ??
    title;
  const narration =
    (typeof scene.narration === "string" ? scene.narration : undefined) ??
    defaults.narration ??
    "";
  const fallback =
    (typeof scene.fallback === "string" && scene.fallback.length > 0
      ? scene.fallback
      : undefined) ??
    defaults.fallback ??
    title;
  const accessibilityRecord = asRecord(scene.accessibility) ?? {};
  const accessibilityLabel =
    typeof accessibilityRecord.label === "string" &&
    accessibilityRecord.label.length > 0
      ? accessibilityRecord.label
      : title;
  const viewportRecord = asRecord(scene.viewport);
  const viewport =
    viewportRecord !== undefined
      ? {
          width: Number(viewportRecord.width ?? 700),
          height: Number(viewportRecord.height ?? 400),
        }
      : undefined;
  const sceneIr: SceneIr = {
    id,
    title,
    summary,
    ...(viewport !== undefined ? { viewport } : {}),
    roots,
    camera: [],
    timeline,
    narration,
    interactions: [],
    responsive: [],
    accessibility: {
      label: accessibilityLabel,
      readingOrder: roots.map((node) => node.id),
    },
    fallback,
    sourceMap: unknownRange,
  };
  return validateSceneRender(sceneIr, unknownRange);
}

/**
 * Lowers an embedded slide `@scene` value to `{ kind: "scene", scene }`.
 *
 * Returns diagnostics when the input is not a recognized scene form or the
 * lowered IR fails strict `SceneIr` validation.
 */
export function lowerExplainerScene(
  scene: unknown,
  options: LowerExplainerSceneOptions = {},
): Result<SceneRender> {
  // Prefer preserved package roots/timeline when non-empty (PackageSceneIrAst).
  if (isPackageSceneAst(scene) && scene.roots.length > 0) {
    try {
      return lowerPackageScene(scene, options);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Package scene lowering failed.";
      return invalidScene(message);
    }
  }

  if (isEmbeddedSceneSource(scene)) {
    if (scene.form !== "native") {
      return invalidScene(
        'Expected native embedded-scene-source; package form should be parsed first.',
      );
    }
    const parsed = parseNativeEmbeddedScene(scene.body);
    if (!parsed.ok) {
      return {
        ok: false,
        diagnostics: parsed.diagnostics.map((item) =>
          diagnostic(
            "EXPLAINER_SCENE_INVALID",
            item.severity,
            item.message,
            item.range,
          ),
        ),
      };
    }
    return lowerNativeSceneAst(parsed.value, options);
  }

  if (!isSceneAst(scene)) {
    if (isPackageSceneAst(scene)) {
      // Package-scene with empty roots never enters lowerPackageScene above.
      return emptySceneField("roots", options);
    }
    return invalidScene(
      `${slideLabel(options)}: expected an embedded @scene AST (native SceneAst, package-scene, or native source).`,
    );
  }

  return lowerNativeSceneAst(scene, options);
}
