/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering from linked `.flow` AST to Flow IR v2.
//!
//! Lowering assumes linking and validation already succeeded: every
//! reference resolves and required accessibility fields are present. It is
//! purely structural (no diagnostics); the caller runs `safeParseFlowIr`
//! against the result to fail closed on any residual schema violation.

import type {
  ArgumentValueAst,
  CameraAst,
  ComponentInvocationAst,
  ConnectorAst,
  InteractionAst,
  LiteralAst,
  RectAst,
  RenderDeclarationAst,
  ResponsiveAst,
  ResponsiveConditionAst,
  SceneAst,
  ThemeAssignmentAst,
  ThemeDeclarationAst,
  TimelineAst,
  ValueAst,
} from "@aiperf/flow-language";
import type {
  CameraKeyframeIr,
  CapabilityRequirement,
  ComponentNodeIr,
  ConnectorNodeIr,
  FlowIr,
  FlowThemeIr,
  GeometryIr,
  InteractionIr,
  JsonValue,
  RectNodeIr,
  RenderNodeIr,
  ResponsiveVariantIr,
  SceneIr,
  StyleValueIr,
  ThemeRole,
  ThemeValueIr,
  TimelineCueIr,
} from "@aiperf/flow-schema";

import type { LinkedDocument, SceneSymbolTable } from "./link.js";

const GEOMETRY_KEYS = ["x", "y", "width", "height"] as const;
type GeometryKey = (typeof GEOMETRY_KEYS)[number];

function isGeometryKey(value: string): value is GeometryKey {
  return (GEOMETRY_KEYS as readonly string[]).includes(value);
}

function lowerStyleValue(
  value: ValueAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): StyleValueIr {
  switch (value.kind) {
    case "literal":
      if (typeof value.value === "number" && !Number.isFinite(value.value)) {
        throw new Error(
          "Internal error: style values must contain only finite numbers.",
        );
      }
      return value.value;
    case "token-reference": {
      const resolved = tokens.get(value.token);
      if (resolved === undefined) {
        throw new Error(
          `Internal error: token "${value.token}" was not resolved during linking.`,
        );
      }
      if (typeof resolved === "number" && !Number.isFinite(resolved)) {
        throw new Error(
          "Internal error: style values must contain only finite numbers.",
        );
      }
      return resolved;
    }
    case "theme-role-reference":
      return { kind: "theme-role", role: value.role as ThemeRole };
  }
}

function resolveArgumentValue(
  value: ArgumentValueAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): JsonValue {
  switch (value.kind) {
    case "literal":
    case "token-reference":
      return lowerStyleValue(value, tokens);
    case "identifier-reference":
      return value.name;
    case "object-literal":
      return Object.fromEntries(
        value.properties.map((property) => [
          property.name,
          resolveArgumentValue(property.value, tokens),
        ]),
      );
  }
}

function rectGeometry(
  nodes: ReadonlyMap<string, RectAst | ConnectorAst>,
  id: string,
): GeometryIr | undefined {
  const node = nodes.get(id);
  return node !== undefined && node.kind === "rect"
    ? { x: node.x, y: node.y, width: node.width, height: node.height }
    : undefined;
}

function boundingBox(boxes: readonly GeometryIr[]): GeometryIr {
  if (boxes.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  const minX = Math.min(...boxes.map((box) => box.x));
  const minY = Math.min(...boxes.map((box) => box.y));
  const maxX = Math.max(...boxes.map((box) => box.x + box.width));
  const maxY = Math.max(...boxes.map((box) => box.y + box.height));
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function lowerRect(
  rect: RectAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
): RectNodeIr {
  const description = rect.description.trim();
  return {
    kind: "rect",
    id: rect.id,
    geometry: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    style: {
      fill: lowerStyleValue(rect.fill, tokens),
      ...(rect.stroke === undefined
        ? {}
        : { stroke: lowerStyleValue(rect.stroke, tokens) }),
    },
    accessibility: {
      label: rect.label,
      ...(description.length > 0 ? { description } : {}),
    },
    fallback: rect.fallback.text,
    sourceMap: rect.sourceMap,
  };
}

function lowerConnector(
  connector: ConnectorAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
  nodes: ReadonlyMap<string, RectAst | ConnectorAst>,
): ConnectorNodeIr {
  const endpointGeometry = [
    rectGeometry(nodes, connector.from),
    rectGeometry(nodes, connector.to),
  ].filter((geometry): geometry is GeometryIr => geometry !== undefined);

  return {
    kind: "connector",
    id: connector.id,
    geometry: boundingBox(endpointGeometry),
    style: { stroke: lowerStyleValue(connector.stroke, tokens) },
    accessibility: { label: connector.label },
    from: { nodeId: connector.from },
    to: { nodeId: connector.to },
    fallback: connector.fallback.text,
    sourceMap: connector.sourceMap,
  };
}

/**
 * Canonical PascalCase stdlib symbol → capability id map.
 */
const SYMBOL_CAPABILITY_IDS: Readonly<Record<string, string>> = {
  Compare: "core.compare",
  FocusContext: "core.focus-context",
  GlyphRun: "core.glyph-run",
  Queue: "viz.queue",
  SegmentStrip: "core.segment-strip",
  SemanticEntity: "core.semantic-entity",
  SemanticMorph: "core.semantic-morph",
  SemanticRelation: "core.semantic-relation",
  SpanMap: "core.span-map",
  StructuredPayload: "core.structured-payload",
  Waterfall: "viz.waterfall",
};

/**
 * Resolves IR `capabilityId` for a component invocation.
 *
 * Precedence:
 * 1. Explicit string `capabilityId` prop
 * 2. Invocation name when it already looks like a capability id (contains `.`)
 * 3. Required canonical capability matching a PascalCase stdlib symbol
 * 4. Original invocation name
 */
function resolveInvocationCapabilityId(
  name: string,
  props: Readonly<Record<string, JsonValue>>,
  requiredCapabilities: ReadonlySet<string>,
): string {
  const fromProp = props.capabilityId;
  if (typeof fromProp === "string" && fromProp.length > 0) {
    return fromProp;
  }
  if (name.includes(".")) {
    return name;
  }
  const canonicalId = SYMBOL_CAPABILITY_IDS[name];
  return canonicalId !== undefined && requiredCapabilities.has(canonicalId)
    ? canonicalId
    : name;
}

function lowerComponentInvocation(
  invocation: ComponentInvocationAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
  requiredCapabilities: ReadonlySet<string>,
  index: number,
): ComponentNodeIr {
  const props: Record<string, JsonValue> = {};
  for (const assignment of invocation.props) {
    props[assignment.name] = resolveArgumentValue(assignment.value, tokens);
  }

  const id =
    typeof props.id === "string"
      ? props.id
      : `${invocation.name.toLowerCase()}-${index}`;
  const label =
    typeof props.label === "string" ? props.label : invocation.name;

  return {
    kind: "component",
    id,
    capabilityId: resolveInvocationCapabilityId(
      invocation.name,
      props,
      requiredCapabilities,
    ),
    props,
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: {},
    accessibility: { label },
    fallback: label,
    children: [],
    sourceMap: invocation.sourceMap,
  };
}

function lowerRenderDeclaration(
  node: RenderDeclarationAst,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
  requiredCapabilities: ReadonlySet<string>,
  nodes: ReadonlyMap<string, RectAst | ConnectorAst>,
  index: number,
): RenderNodeIr {
  if (node.kind === "component-invocation") {
    return lowerComponentInvocation(node, tokens, requiredCapabilities, index);
  }
  return node.kind === "rect"
    ? lowerRect(node, tokens)
    : lowerConnector(node, tokens, nodes);
}

function lowerCamera(
  camera: CameraAst,
  nodes: ReadonlyMap<string, RectAst | ConnectorAst>,
): readonly CameraKeyframeIr[] {
  return camera.keyframes.map((keyframe, index) => {
    const firstTarget = keyframe.targets.references[0];
    const geometry =
      firstTarget === undefined ? undefined : rectGeometry(nodes, firstTarget);
    const center =
      geometry === undefined
        ? { x: 0, y: 0 }
        : { x: geometry.x + geometry.width / 2, y: geometry.y + geometry.height / 2 };
    return {
      id: `${camera.id}-${index}`,
      at: keyframe.time,
      x: center.x,
      y: center.y,
      zoom: keyframe.zoom,
      sourceMap: keyframe.sourceMap,
    };
  });
}

function lowerTimeline(timeline: TimelineAst): readonly TimelineCueIr[] {
  return timeline.cues.map((cue, index) => ({
    id: `${timeline.id}-${index}`,
    at: cue.time,
    duration: cue.duration,
    target: cue.target,
    action: cue.action,
    sourceMap: cue.sourceMap,
  }));
}

function lowerInteraction(interaction: InteractionAst): InteractionIr {
  return {
    id: interaction.id,
    event: interaction.event.name,
    target: interaction.event.target,
    action: interaction.action.name,
    sourceMap: interaction.sourceMap,
  };
}

function conditionText(condition: ResponsiveConditionAst): string {
  return `${condition.property} ${condition.operator} ${condition.value}`;
}

function withGeometry<T extends RenderNodeIr>(
  node: T,
  patch: Partial<GeometryIr>,
): T {
  return { ...node, geometry: { ...node.geometry, ...patch } } as T;
}

function applyResponsiveOverrides(
  nodes: readonly RenderNodeIr[],
  patchesByTarget: ReadonlyMap<string, Partial<GeometryIr>>,
): readonly RenderNodeIr[] {
  return nodes.map((node) => {
    const patch = patchesByTarget.get(node.id);
    const patched = patch === undefined ? node : withGeometry(node, patch);
    return patched.kind === "group" || patched.kind === "component"
      ? { ...patched, children: applyResponsiveOverrides(patched.children, patchesByTarget) }
      : patched;
  });
}

function lowerResponsive(
  responsive: ResponsiveAst,
  roots: readonly RenderNodeIr[],
): ResponsiveVariantIr {
  const patchesByTarget = new Map<string, Partial<GeometryIr>>();
  for (const override of responsive.overrides) {
    if (!isGeometryKey(override.property)) {
      continue;
    }
    const existing = patchesByTarget.get(override.target) ?? {};
    patchesByTarget.set(override.target, {
      ...existing,
      [override.property]: override.value,
    });
  }

  return {
    id: responsive.id,
    condition: conditionText(responsive.condition),
    roots: applyResponsiveOverrides(roots, patchesByTarget),
    sourceMap: responsive.sourceMap,
  };
}

function lowerScene(
  scene: SceneAst,
  symbols: SceneSymbolTable,
  tokens: ReadonlyMap<string, LiteralAst["value"]>,
  requiredCapabilities: ReadonlySet<string>,
): SceneIr {
  const roots = scene.renderDeclarations.map((node, index) =>
    lowerRenderDeclaration(
      node,
      tokens,
      requiredCapabilities,
      symbols.nodes,
      index,
    ),
  );

  return {
    id: scene.id,
    title: scene.title,
    summary: scene.summary?.text ?? "",
    roots,
    camera: scene.cameras.flatMap((camera) => lowerCamera(camera, symbols.nodes)),
    timeline: scene.timelines.flatMap((timeline) => lowerTimeline(timeline)),
    narration: scene.narration?.text ?? "",
    interactions: scene.interactions.map(lowerInteraction),
    responsive: scene.responsiveVariants.map((responsive) =>
      lowerResponsive(responsive, roots),
    ),
    accessibility: {
      label: scene.title,
      readingOrder: scene.readingOrder?.references ?? [],
    },
    fallback: scene.fallback?.text ?? "",
    sourceMap: scene.sourceMap,
  };
}

function lowerThemeAssignment(assignment: ThemeAssignmentAst): ThemeValueIr {
  switch (assignment.valueKind) {
    case "color": {
      if (assignment.value.kind !== "literal" || typeof assignment.value.value !== "string") {
        throw new Error(
          `Internal error: theme role "${assignment.role}" expected a color literal.`,
        );
      }
      return { kind: "color", value: assignment.value.value };
    }
    case "font": {
      if (assignment.value.kind !== "theme-font-literal") {
        throw new Error(
          `Internal error: theme role "${assignment.role}" expected a font stack.`,
        );
      }
      return {
        kind: "font",
        value: Object.freeze([...assignment.value.families]),
      };
    }
    case "number": {
      if (assignment.value.kind !== "literal" || typeof assignment.value.value !== "number") {
        throw new Error(
          `Internal error: theme role "${assignment.role}" expected a numeric literal.`,
        );
      }
      return { kind: "number", value: assignment.value.value };
    }
    case "duration": {
      if (assignment.value.kind !== "literal" || typeof assignment.value.value !== "number") {
        throw new Error(
          `Internal error: theme role "${assignment.role}" expected a duration literal.`,
        );
      }
      return { kind: "duration", valueMs: Math.trunc(assignment.value.value) };
    }
    case "enum": {
      if (assignment.value.kind !== "literal" || typeof assignment.value.value !== "string") {
        throw new Error(
          `Internal error: theme role "${assignment.role}" expected an enum literal.`,
        );
      }
      return { kind: "enum", value: assignment.value.value };
    }
  }
}

function lowerTheme(theme: ThemeDeclarationAst): FlowThemeIr {
  const values: Partial<Record<ThemeRole, ThemeValueIr>> = {};
  for (const assignment of theme.assignments) {
    values[assignment.role as ThemeRole] = lowerThemeAssignment(assignment);
  }
  return {
    id: theme.id,
    extends: theme.extends,
    values,
    sourceMap: theme.sourceMap,
  };
}

/** Lowers a linked document into a Flow IR value, prior to schema validation. */
export function lower(linked: LinkedDocument): FlowIr {
  const capabilities: readonly CapabilityRequirement[] = linked.document.requirements
    .map((requirement) => ({
      id: requirement.capability,
      range: requirement.versionRange,
    }))
    .sort((left, right) => left.id.localeCompare(right.id));

  const tokens: Record<string, string | number | boolean> = {};
  for (const [id, value] of linked.tokens) {
    tokens[id] = value;
  }

  const requiredCapabilities = new Set(capabilities.map(({ id }) => id));
  const scenes = linked.document.scenes.map((scene) => {
    const symbols = linked.scenes.get(scene.id);
    if (symbols === undefined) {
      throw new Error(`Internal error: scene "${scene.id}" was not linked.`);
    }
    return lowerScene(scene, symbols, linked.tokens, requiredCapabilities);
  });

  const themes = (linked.themes ?? [])
    .map(lowerTheme)
    .sort((left, right) => left.id.localeCompare(right.id, "en", { sensitivity: "variant" }));

  return {
    irVersion: 2,
    id: linked.document.id,
    title: linked.document.title,
    capabilities,
    tokens,
    themes,
    ...(linked.useTheme === undefined
      ? {}
      : { defaultTheme: linked.useTheme.themeId }),
    scenes,
    sourceMap: linked.document.sourceMap,
  };
}
