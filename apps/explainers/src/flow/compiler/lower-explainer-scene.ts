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
//! Native paths reuse the document scene lowerer when only rect/connector
//! nodes are present. Package paths (and native scenes with Task-2 primitives
//! / extended timeline cues) normalize into strict `SceneIr`, desugaring
//! geometry macros and preserving first-class layout / motion / stagger.

import {
  parseNativeEmbeddedScene,
  type ArgumentValueAst,
  type EmbeddedSceneSource,
  type PackageSceneAst,
  type SceneAst,
  type DocumentAst,
  type LiteralAst,
  type PropAssignmentAst,
  type RenderDeclarationAst,
  type ScenePrimitiveAst,
  type TimelineCueEasing,
} from "../language/index.js";
import {
  diagnostic,
  hasErrors,
  sceneIrSchema,
  type CapabilityRegistryManifest,
  type Diagnostic,
  type RenderNodeIr,
  type Result,
  type SceneIr,
  type SceneRender,
  type SourceRange,
  type TimelineCueIr,
} from "../schema/index.js";

import {
  asRecord,
  isSupportedPackageCapability,
} from "./desugar-scene-primitives.js";
import { expandSymbolInvocations } from "./expand-symbols.js";
import { link, type LinkedDocument } from "./link.js";
import { lower } from "./lower.js";
import { collectSymbols } from "./symbols.js";
import { resolveTimelineCueTiming } from "./timeline-timing.js";
import { validate } from "./validate.js";
import { lowerSemanticSceneNode } from "./semantic-scene-node.js";

export type LowerExplainerSceneOptions = Readonly<{
  tokens?: ReadonlyMap<string, LiteralAst["value"]>;
  /** Registry used to validate effective SceneIr capability ids. */
  capabilities?: CapabilityRegistryManifest;
  /** Enables strict capability fail-closed and native narration checks. */
  strict?: boolean;
  /**
   * Owning slide / document source range. Used so embedded `@scene`
   * diagnostics identify the original `.flow` file instead of placeholders.
   */
  sourceRange?: SourceRange;
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

const TIMELINE_EASINGS = new Set<TimelineCueEasing>([
  "linear",
  "ease-in",
  "ease-out",
  "ease-in-out",
]);

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

function wrapSceneDocument(
  scene: SceneAst,
  options: LowerExplainerSceneOptions = {},
): DocumentAst {
  const sourceMap = options.sourceRange ?? scene.sourceMap;
  return {
    kind: "document",
    id: `explainer-scene-${scene.id}`,
    title: scene.title,
    language: { kind: "language", version: 1, sourceMap },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [
      sourceMap === scene.sourceMap ? scene : { ...scene, sourceMap },
    ],
    sourceMap,
  };
}

function validateNativeScene(
  scene: SceneAst,
  options: LowerExplainerSceneOptions,
): Result<LinkedDocument> {
  const document = wrapSceneDocument(scene, options);
  const symbols = collectSymbols(document);
  if (!symbols.ok) {
    return symbols;
  }

  const expanded = expandSymbolInvocations(document, symbols.value);
  if (!expanded.ok) {
    return expanded;
  }

  const linked = link(expanded.value);
  if (!linked.ok) {
    return linked;
  }

  if (options.capabilities === undefined) {
    return linked;
  }
  return validate(linked.value, options.capabilities, options.strict ?? false);
}

function sceneRange(
  scene: PackageSceneAst,
  options: LowerExplainerSceneOptions = {},
): SourceRange {
  const fromScene = scene.sourceMap;
  if (
    fromScene !== undefined &&
    fromScene.source !== "<unknown>" &&
    fromScene.source !== "<embedded-scene>"
  ) {
    return fromScene;
  }
  if (options.sourceRange !== undefined) {
    return options.sourceRange;
  }
  return fromScene ?? unknownRange;
}

function nodeCapabilityId(node: RenderNodeIr): string | undefined {
  if (typeof node.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  return undefined;
}

function collectSceneIrCapabilityIds(
  nodes: readonly RenderNodeIr[],
): readonly string[] {
  const out: string[] = [];
  const visit = (node: RenderNodeIr): void => {
    const capability = nodeCapabilityId(node);
    if (capability !== undefined) {
      out.push(capability);
    }
    if (node.kind === "group" || node.kind === "component") {
      node.children.forEach(visit);
    }
  };
  nodes.forEach(visit);
  return out;
}

/**
 * Fail-closed check over effective lowered SceneIr capability ids.
 *
 * Macros and first-class package selectors recognized by the lowering
 * implementation are authoring vocabulary and are not required in the
 * supplied manifest. Unknown / unlowerable ids fail when `strict` is true.
 */
function validateEffectiveSceneCapabilities(
  sceneIr: SceneIr,
  options: LowerExplainerSceneOptions,
  range: SourceRange,
): readonly Diagnostic[] {
  if (options.capabilities === undefined) {
    return [];
  }

  const available = new Set(
    options.capabilities.capabilities.map(({ id }) => id),
  );
  const strict = options.strict ?? false;
  const diagnostics: Diagnostic[] = [];
  const seen = new Set<string>();

  for (const capability of collectSceneIrCapabilityIds(sceneIr.roots)) {
    if (seen.has(capability)) {
      continue;
    }
    seen.add(capability);
    if (
      available.has(capability) ||
      isSupportedPackageCapability(capability)
    ) {
      continue;
    }
    if (!strict) {
      continue;
    }
    diagnostics.push(
      diagnostic(
        "CAPABILITY_MISSING",
        "error",
        `${slideLabel(options)} uses unknown or unlowerable capability "${capability}".`,
        range,
        `Register "${capability}" in the capability manifest, or use a supported scene primitive / package macro.`,
      ),
    );
  }
  return diagnostics;
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

function resolveArgumentValue(value: ArgumentValueAst): unknown {
  switch (value.kind) {
    case "literal":
      return value.value;
    case "token-reference":
      return `@${value.token}`;
    case "identifier-reference":
      return value.name;
    case "object-literal":
      return Object.fromEntries(
        value.properties.map((property) => [
          property.name,
          resolveArgumentValue(property.value as ArgumentValueAst),
        ]),
      );
  }
}

function propsToRecord(
  props: readonly PropAssignmentAst[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const prop of props) {
    out[prop.name] = resolveArgumentValue(prop.value);
  }
  return out;
}

function renderDeclarationToPackage(
  node: RenderDeclarationAst,
): Record<string, unknown> {
  if (node.kind === "rect") {
    return {
      id: node.id,
      capability: "core.rect",
      layout: { x: node.x, y: node.y, width: node.width, height: node.height },
      style: {
        ...(node.fill.kind === "literal"
          ? { fill: node.fill.value }
          : node.fill.kind === "token-reference"
            ? { fill: `@${node.fill.token}` }
            : {}),
        ...(node.stroke === undefined
          ? {}
          : node.stroke.kind === "literal"
            ? { stroke: node.stroke.value }
            : node.stroke.kind === "token-reference"
              ? { stroke: `@${node.stroke.token}` }
              : {}),
      },
      text: node.label,
      accessibility: {
        label: node.label,
        ...(node.description.trim().length > 0
          ? { description: node.description }
          : {}),
      },
      fallback: node.fallback.text,
    };
  }
  if (node.kind === "connector") {
    return {
      id: node.id,
      capability: "core.connector",
      from: { nodeId: node.from },
      to: { nodeId: node.to },
      style: {
        ...(node.stroke.kind === "literal"
          ? { stroke: node.stroke.value }
          : node.stroke.kind === "token-reference"
            ? { stroke: `@${node.stroke.token}` }
            : {}),
      },
      accessibility: { label: node.label },
      fallback: node.fallback.text,
    };
  }
  if (node.kind === "scene-primitive") {
    return scenePrimitiveToPackage(node);
  }
  // Component invocations fall through as opaque capability-bearing nodes.
  const idProp = node.props.find((prop) => prop.name === "id");
  return {
    id:
      idProp !== undefined
        ? String(resolveArgumentValue(idProp.value))
        : node.name,
    capability:
      node.namespace !== undefined
        ? `${node.namespace}.${node.name}`
        : node.name,
    ...propsToRecord(node.props),
  };
}

function scenePrimitiveToPackage(
  node: ScenePrimitiveAst,
): Record<string, unknown> {
  const props = propsToRecord(node.props);
  return {
    id: node.id,
    capability: node.capability,
    ...props,
    ...(node.children !== undefined
      ? { children: node.children.map(renderDeclarationToPackage) }
      : {}),
    ...(node.fallback !== undefined ? { fallback: node.fallback.text } : {}),
  };
}

function nativeSceneNeedsPackageLower(scene: SceneAst): boolean {
  for (const node of scene.renderDeclarations) {
    if (node.kind === "scene-primitive") {
      return true;
    }
  }
  for (const timeline of scene.timelines) {
    for (const cue of timeline.cues) {
      if (
        cue.action === "stagger" ||
        cue.action === "enter-children" ||
        cue.action === "fade" ||
        cue.action === "exit" ||
        cue.targets !== undefined ||
        cue.step !== undefined ||
        cue.easing !== undefined
      ) {
        return true;
      }
    }
  }
  return false;
}

function nativeSceneToPackageScene(scene: SceneAst): PackageSceneAst {
  const timeline = scene.timelines.flatMap((timelineAst) => {
    const resolvedAt = resolveTimelineCueTiming(timelineAst.cues);
    return timelineAst.cues.map((cue, index) => ({
      id: `${timelineAst.id}-${index}`,
      at: resolvedAt[index]!,
      duration: cue.duration,
      target: cue.target,
      action: cue.action,
      ...(cue.targets !== undefined ? { targets: cue.targets } : {}),
      ...(cue.step !== undefined ? { step: cue.step } : {}),
      ...(cue.easing !== undefined ? { easing: cue.easing } : {}),
    }));
  });
  return {
    kind: "package-scene",
    id: scene.id,
    title: scene.title,
    ...(scene.summary?.text !== undefined
      ? { summary: scene.summary.text }
      : {}),
    ...(scene.narration?.text !== undefined
      ? { narration: scene.narration.text }
      : {}),
    ...(scene.fallback?.text !== undefined
      ? { fallback: scene.fallback.text }
      : {}),
    roots: scene.renderDeclarations.map(renderDeclarationToPackage),
    timeline,
    camera: [],
    accessibility: {
      label: scene.title,
      readingOrder: scene.readingOrder?.references ?? [],
    },
  };
}

function lowerNativeSceneAst(
  scene: SceneAst,
  options: LowerExplainerSceneOptions,
): Result<SceneRender> {
  const validated = validateNativeScene(scene, options);
  if (!validated.ok) {
    return validated;
  }

  const expandedScene = validated.value.document.scenes[0];
  if (expandedScene === undefined) {
    return invalidScene(
      `Scene "${scene.id}" was lost during symbol expansion.`,
      options.sourceRange ?? scene.sourceMap,
    );
  }
  if (nativeSceneNeedsPackageLower(expandedScene)) {
    const loweredPackage = lowerPackageScene(
      {
        ...nativeSceneToPackageScene(expandedScene),
        sourceMap: options.sourceRange ?? expandedScene.sourceMap,
      },
      options,
    );
    return {
      ...loweredPackage,
      diagnostics: [...validated.diagnostics, ...loweredPackage.diagnostics],
    };
  }

  let lowered;
  try {
    lowered = lower(validated.value);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Scene lowering failed.";
    return invalidScene(message, options.sourceRange ?? scene.sourceMap);
  }

  const sceneIr = lowered.scenes[0];
  if (sceneIr === undefined) {
    return invalidScene(
      `Scene "${scene.id}" produced no SceneIr after lowering.`,
      options.sourceRange ?? scene.sourceMap,
    );
  }

  if (sceneIr.roots.length === 0) {
    return emptySceneField(
      "roots",
      options,
      options.sourceRange ?? scene.sourceMap,
      scene.id,
    );
  }
  if (sceneIr.timeline.length === 0) {
    return emptySceneField(
      "timeline",
      options,
      options.sourceRange ?? scene.sourceMap,
      scene.id,
    );
  }

  const range = options.sourceRange ?? expandedScene.sourceMap;
  const rendered = validateSceneRender(sceneIr, options, range);
  return {
    ...rendered,
    diagnostics: [...validated.diagnostics, ...rendered.diagnostics],
  };
}

function validateSceneRender(
  sceneIr: SceneIr,
  options: LowerExplainerSceneOptions,
  range: SourceRange,
): Result<SceneRender> {
  const capabilityDiagnostics = validateEffectiveSceneCapabilities(
    sceneIr,
    options,
    range,
  );
  if (hasErrors(capabilityDiagnostics)) {
    return { ok: false, diagnostics: capabilityDiagnostics };
  }

  const parsed = sceneIrSchema.safeParse(sceneIr);
  if (!parsed.success) {
    return {
      ok: false,
      diagnostics: [
        ...capabilityDiagnostics,
        ...parsed.error.issues.map((issue) => {
          const path =
            issue.path.length === 0 ? "<root>" : issue.path.join(".");
          return diagnostic(
            "EXPLAINER_SCENE_INVALID",
            "error",
            `${path}: ${issue.message}`,
            range,
          );
        }),
      ],
    };
  }

  return {
    ok: true,
    value: { kind: "scene", scene: parsed.data },
    diagnostics: capabilityDiagnostics,
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
    : [];
  const accessibilityRecord = asRecord(node.accessibility) ?? {};
  const label =
    typeof node.text === "string"
      ? node.text
      : typeof node.title === "string"
        ? node.title
        : typeof accessibilityRecord.label === "string"
          ? accessibilityRecord.label
          : id;
  const description =
    typeof accessibilityRecord.description === "string" &&
    accessibilityRecord.description.length > 0
      ? accessibilityRecord.description
      : undefined;
  const fallback =
    typeof node.fallback === "string" ? node.fallback : label;

  return lowerSemanticSceneNode(node, {
    id,
    capability,
    children,
    label,
    fallback,
    sourceMap: unknownRange,
    ...(description !== undefined ? { description } : {}),
  });
}

function findNodeById(
  nodes: readonly RenderNodeIr[],
  id: string,
): RenderNodeIr | undefined {
  for (const node of nodes) {
    if (node.id === id) {
      return node;
    }
    if (node.kind === "group" || node.kind === "component") {
      const nested = findNodeById(node.children, id);
      if (nested !== undefined) {
        return nested;
      }
    }
  }
  return undefined;
}

function normalizeEasing(value: unknown): TimelineCueEasing | undefined {
  return typeof value === "string" &&
    TIMELINE_EASINGS.has(value as TimelineCueEasing)
    ? (value as TimelineCueEasing)
    : undefined;
}

function normalizePackageTimeline(
  value: unknown,
  index: number,
  roots: readonly RenderNodeIr[],
): TimelineCueIr {
  const cue = asRecord(value) ?? {};
  const action = String(cue.action ?? "enter");
  const target = String(cue.target ?? "");
  const easing = normalizeEasing(cue.easing);
  const step =
    typeof cue.step === "number" && Number.isFinite(cue.step)
      ? cue.step
      : undefined;
  let targets: readonly string[] | undefined;
  if (Array.isArray(cue.targets)) {
    targets = cue.targets
      .filter((item): item is string => typeof item === "string" && item.length > 0);
  }

  // Expand enter-children into a compact stagger when child ids are known.
  if (action === "enter-children" && target.length > 0) {
    const group = findNodeById(roots, target);
    if (
      group !== undefined &&
      (group.kind === "group" || group.kind === "component") &&
      group.children.length > 0
    ) {
      return {
        id: String(cue.id ?? `cue-${index}`),
        at: Number(cue.at ?? 0),
        duration: Number(cue.duration ?? 0),
        target,
        action: "stagger",
        targets: group.children.map((child) => child.id),
        step: step ?? 80,
        ...(easing !== undefined ? { easing } : {}),
        sourceMap: unknownRange,
      };
    }
  }

  return {
    id: String(cue.id ?? `cue-${index}`),
    at: Number(cue.at ?? 0),
    duration: Number(cue.duration ?? 0),
    target,
    action,
    ...(targets !== undefined && targets.length > 0 ? { targets } : {}),
    ...(step !== undefined ? { step } : {}),
    ...(easing !== undefined ? { easing } : {}),
    sourceMap: unknownRange,
  };
}

function lowerPackageScene(
  scene: PackageSceneAst,
  options: LowerExplainerSceneOptions,
): Result<SceneRender> {
  const range = sceneRange(scene, options);

  const defaults = options.defaults ?? {};
  const roots = scene.roots.map(normalizePackageNode);
  const timeline = (scene.timeline ?? []).map((cue, index) =>
    normalizePackageTimeline(cue, index, roots),
  );
  if (roots.length === 0) {
    return emptySceneField("roots", options, range);
  }
  if (timeline.length === 0) {
    return emptySceneField("timeline", options, range);
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
  const readingOrder = Array.isArray(accessibilityRecord.readingOrder)
    ? accessibilityRecord.readingOrder.filter(
        (item): item is string => typeof item === "string",
      )
    : roots.map((node) => node.id);
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
      readingOrder:
        readingOrder.length > 0 ? readingOrder : roots.map((node) => node.id),
    },
    fallback,
    sourceMap: range,
  };
  return validateSceneRender(sceneIr, options, range);
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
      return invalidScene(message, options.sourceRange ?? sceneRange(scene, options));
    }
  }

  if (isEmbeddedSceneSource(scene)) {
    if (scene.form !== "native") {
      return invalidScene(
        'Expected native embedded-scene-source; package form should be parsed first.',
        options.sourceRange,
      );
    }
    const sourceName =
      options.sourceRange?.source !== undefined &&
      options.sourceRange.source.length > 0
        ? options.sourceRange.source
        : "<embedded-scene>";
    const parsed = parseNativeEmbeddedScene(scene.body, sourceName);
    if (!parsed.ok) {
      return {
        ok: false,
        diagnostics: parsed.diagnostics.map((item) =>
          diagnostic(
            "EXPLAINER_SCENE_INVALID",
            item.severity,
            item.message,
            options.sourceRange ?? item.range,
          ),
        ),
      };
    }
    const nativeScene =
      options.sourceRange !== undefined &&
      (parsed.value.sourceMap.source === "<embedded-scene>" ||
        parsed.value.sourceMap.source === sourceName)
        ? { ...parsed.value, sourceMap: options.sourceRange }
        : parsed.value;
    return lowerNativeSceneAst(nativeScene, options);
  }

  if (!isSceneAst(scene)) {
    if (isPackageSceneAst(scene)) {
      // Package-scene with empty roots never enters lowerPackageScene above.
      return emptySceneField(
        "roots",
        options,
        options.sourceRange ?? unknownRange,
      );
    }
    return invalidScene(
      `${slideLabel(options)}: expected an embedded @scene AST (native SceneAst, package-scene, or native source).`,
      options.sourceRange,
    );
  }

  return lowerNativeSceneAst(scene, options);
}

/** Re-export for callers / tests that inspect kind mapping. */
export { capabilityKind } from "./desugar-scene-primitives.js";
