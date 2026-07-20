// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArgumentValueAst,
  CameraAst,
  ComponentInvocationAst,
  ConnectorAst,
  DocumentAst,
  ExplainerAst,
  ExplainerHubAst,
  ForLoopAst,
  FreeformBlockAst,
  ImportDeclarationAst,
  InteractionAst,
  ObjectLiteralAst,
  ParamDeclarationAst,
  RectAst,
  RenderDeclarationAst,
  ResponsiveAst,
  SceneAst,
  ScenePrimitiveAst,
  SlideAst,
  SlotBlockAst,
  SymbolDefinitionAst,
  SymbolBodyStatementAst,
  ThemeAssignmentAst,
  ThemeDeclarationAst,
  TimelineAst,
  TimelineCueAst,
  TypeRefAst,
  ValueAst,
} from "./ast.js";
import type { EmbeddedSceneSource } from "./embedded-scene.js";

const quote = (value: string): string => JSON.stringify(value);
const indent = (level: number): string => "  ".repeat(level);

function assertNever(value: never): never {
  throw new Error(`Unsupported AST node: ${JSON.stringify(value)}`);
}

function formatValue(value: ValueAst): string {
  switch (value.kind) {
    case "literal":
      return typeof value.value === "string"
        ? quote(value.value)
        : String(value.value);
    case "token-reference":
      return `token(${value.token})`;
    case "theme-role-reference":
      return `theme(${value.role})`;
    default:
      return assertNever(value);
  }
}

function formatObjectLiteral(object: ObjectLiteralAst): string {
  const properties = object.properties
    .map((property) => `${property.name}: ${formatArgumentValue(property.value)}`)
    .join(", ");
  return properties.length === 0 ? "{}" : `{ ${properties} }`;
}

function formatArgumentValue(value: ArgumentValueAst): string {
  switch (value.kind) {
    case "literal":
    case "token-reference":
    case "theme-role-reference":
      return formatValue(value);
    case "identifier-reference":
      return value.name;
    case "object-literal":
      return formatObjectLiteral(value);
    case "array-literal":
      return `[${value.items.map(formatArgumentValue).join(", ")}]`;
    case "ref":
      return `ref(${quote(value.target)})`;
    default:
      return assertNever(value);
  }
}

function block(header: string, lines: readonly string[], level: number): string {
  const prefix = indent(level);
  if (lines.length === 0) {
    return `${prefix}${header} {\n${prefix}}`;
  }
  return `${prefix}${header} {\n${lines.join("\n")}\n${prefix}}`;
}

function formatRect(rect: RectAst): string {
  const line = indent(3);
  const style = [
    `${line}fill ${formatValue(rect.fill)}`,
    ...(rect.stroke === undefined
      ? []
      : [`${line}stroke ${formatValue(rect.stroke)}`]),
  ];
  return block(
    `rect ${rect.id}`,
    [
      `${line}x ${rect.x}`,
      `${line}y ${rect.y}`,
      `${line}width ${rect.width}`,
      `${line}height ${rect.height}`,
      ...style,
      `${line}label ${quote(rect.label)}`,
      `${line}role ${quote(rect.role)}`,
      `${line}description ${quote(rect.description)}`,
      `${line}fallback ${quote(rect.fallback.text)}`,
    ],
    2,
  );
}

function formatConnector(connector: ConnectorAst): string {
  const line = indent(3);
  return block(
    `connector ${connector.id}`,
    [
      `${line}from ${connector.from}`,
      `${line}to ${connector.to}`,
      `${line}label ${quote(connector.label)}`,
      `${line}stroke ${formatValue(connector.stroke)}`,
      `${line}fallback ${quote(connector.fallback.text)}`,
    ],
    2,
  );
}

function formatImport(declaration: ImportDeclarationAst): string {
  return `${indent(1)}import ${quote(declaration.path)} as ${declaration.alias}`;
}

function formatThemeAssignment(assignment: ThemeAssignmentAst): string {
  const value =
    assignment.value.kind === "theme-font-literal"
      ? `[${assignment.value.families.map(quote).join(", ")}]`
      : assignment.valueKind === "duration"
        ? `${assignment.value.value}ms`
        : formatValue(assignment.value);
  return `${indent(2)}${assignment.valueKind} ${assignment.role} = ${value}`;
}

function formatTheme(theme: ThemeDeclarationAst): string {
  return block(
    `theme ${theme.id} extends ${theme.extends}`,
    theme.assignments.map(formatThemeAssignment),
    1,
  );
}

function formatType(type: TypeRefAst): string {
  return `${type.name}${type.array === true ? "[]" : ""}`;
}

function formatParam(param: ParamDeclarationAst): string {
  return `${param.name}: ${formatType(param.type)}`;
}

function formatComponentInvocation(
  invocation: ComponentInvocationAst,
  level: number,
): string {
  const name =
    "namespace" in invocation
      ? `${invocation.namespace}.${invocation.name}`
      : invocation.name;
  const props = invocation.props
    .map((prop) => `${prop.name} = ${formatArgumentValue(prop.value)}`)
    .join(", ");
  const header = `${name}(${props})`;
  if (invocation.slots === undefined || invocation.slots.length === 0) {
    return `${indent(level)}${header}`;
  }
  return block(
    header,
    invocation.slots.map((slot) => formatSlot(slot, level + 1)),
    level,
  );
}

function formatSlot(slot: SlotBlockAst, level: number): string {
  const parameter =
    slot.parameter === undefined ? "" : `(${slot.parameter})`;
  return block(
    `${slot.name}${parameter}`,
    slot.body.map((statement) => formatSymbolBodyStatement(statement, level + 1)),
    level,
  );
}

function formatForLoop(loop: ForLoopAst, level: number): string {
  const collection = formatArgumentValue(loop.collection);
  return block(
    `for ${loop.item} in ${collection}`,
    loop.body.map((statement) => formatSymbolBodyStatement(statement, level + 1)),
    level,
  );
}

function formatFreeform(freeform: FreeformBlockAst): string {
  const header = freeform.id === undefined ? "freeform" : `freeform ${freeform.id}`;
  return block(header, freeform.body.map(formatRenderDeclaration), 2);
}

function formatSymbolBodyStatement(
  statement: SymbolBodyStatementAst,
  level: number,
): string {
  switch (statement.kind) {
    case "component-invocation":
      return formatComponentInvocation(statement, level);
    case "slot":
      return formatSlot(statement, level);
    case "for-loop":
      return formatForLoop(statement, level);
    default:
      return assertNever(statement);
  }
}

function formatRenderDeclaration(declaration: RenderDeclarationAst): string {
  switch (declaration.kind) {
    case "rect":
      return formatRect(declaration);
    case "connector":
      return formatConnector(declaration);
    case "scene-primitive":
      return formatScenePrimitive(declaration);
    case "component-invocation":
      return formatComponentInvocation(declaration, 2);
    default:
      return assertNever(declaration);
  }
}

function formatScenePrimitive(node: ScenePrimitiveAst): string {
  const line = indent(3);
  const propLines = node.props.map(
    (prop) => `${line}${prop.name} ${formatArgumentValue(prop.value)}`,
  );
  const childLines =
    node.children === undefined || node.children.length === 0
      ? []
      : [
          `${line}children {`,
          ...node.children.map(
            (child) => `${indent(1)}${formatRenderDeclaration(child).trimStart()}`,
          ),
          `${line}}`,
        ];
  const fallbackLines =
    node.fallback === undefined
      ? []
      : [`${line}fallback ${quote(node.fallback.text)}`];
  return block(
    `${node.primitive} ${node.id}`,
    [...propLines, ...childLines, ...fallbackLines],
    2,
  );
}

function formatTimelineCueTiming(cue: TimelineCueAst): string {
  if (cue.timing.mode === "at") {
    return `at ${cue.timing.ms}`;
  }
  const gap = cue.timing.gap === 0 ? "" : ` +${cue.timing.gap}`;
  return `after ${cue.timing.ref}${gap}`;
}

function formatTimelineCue(cue: TimelineCueAst): string {
  const prefix = `${indent(3)}${formatTimelineCueTiming(cue)} ${cue.action}`;
  if (cue.action === "stagger" && cue.targets !== undefined) {
    const parts = [
      prefix,
      `targets [${cue.targets.join(", ")}]`,
      ...(cue.step === undefined ? [] : [`step ${cue.step}`]),
      `duration ${cue.duration}`,
      ...(cue.easing === undefined ? [] : [`easing ${cue.easing}`]),
    ];
    return parts.join(" ");
  }
  const parts = [
    prefix,
    cue.target,
    `duration ${cue.duration}`,
    ...(cue.step === undefined ? [] : [`step ${cue.step}`]),
    ...(cue.easing === undefined ? [] : [`easing ${cue.easing}`]),
  ];
  return parts.join(" ");
}

function formatSymbol(symbol: SymbolDefinitionAst): string {
  const params = symbol.params.map(formatParam).join(", ");
  const body = symbol.body.map((statement) =>
    formatSymbolBodyStatement(statement, 2),
  );
  return block(`symbol ${symbol.name}(${params})`, body, 1);
}

function formatCamera(camera: CameraAst): string {
  return block(
    `camera ${camera.id}`,
    camera.keyframes.map(
      (keyframe) =>
        `${indent(3)}at ${keyframe.time} frame ${keyframe.targets.references.join(",")} zoom ${keyframe.zoom}`,
    ),
    2,
  );
}

function formatTimeline(timeline: TimelineAst): string {
  return block(
    `timeline ${timeline.id}`,
    timeline.cues.map(formatTimelineCue),
    2,
  );
}

function formatInteraction(interaction: InteractionAst): string {
  return block(
    `interaction ${interaction.id}`,
    [
      `${indent(3)}on ${interaction.event.name} ${interaction.event.target}`,
      `${indent(3)}do ${interaction.action.name} ${interaction.action.target}`,
    ],
    2,
  );
}

function formatResponsive(responsive: ResponsiveAst): string {
  const { condition } = responsive;
  return block(
    `responsive ${responsive.id} when ${condition.property} ${condition.operator} ${condition.value}`,
    responsive.overrides.map(
      (override) =>
        `${indent(3)}set ${override.target}.${override.property} = ${override.value}`,
    ),
    2,
  );
}

function formatScene(scene: SceneAst): string {
  const sections: string[] = [];
  if (scene.summary !== undefined) {
    sections.push(`${indent(2)}summary ${quote(scene.summary.text)}`);
  }
  sections.push(
    ...scene.renderDeclarations.map(formatRenderDeclaration),
    ...(scene.loops ?? []).map((loop) => formatForLoop(loop, 2)),
    ...(scene.freeforms ?? []).map(formatFreeform),
    ...scene.cameras.map(formatCamera),
    ...scene.timelines.map(formatTimeline),
    ...scene.interactions.map(formatInteraction),
    ...scene.responsiveVariants.map(formatResponsive),
  );

  const closingStatements: string[] = [];
  if (scene.narration !== undefined) {
    closingStatements.push(
      `${indent(2)}narrate ${quote(scene.narration.text)}`,
    );
  }
  if (scene.readingOrder !== undefined) {
    closingStatements.push(
      `${indent(2)}reading-order ${scene.readingOrder.references.join(",")}`,
    );
  }
  if (scene.fallback !== undefined) {
    closingStatements.push(
      `${indent(2)}fallback ${quote(scene.fallback.text)}`,
    );
  }
  if (closingStatements.length > 0) {
    sections.push(closingStatements.join("\n"));
  }

  return block(
    `scene ${quote(scene.title)} as ${scene.id}`,
    [sections.join("\n\n")],
    1,
  );
}

function isEmbeddedSceneSource(value: unknown): value is EmbeddedSceneSource {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { kind?: unknown }).kind === "embedded-scene-source" &&
    typeof (value as { body?: unknown }).body === "string"
  );
}

function formatPackageLiteral(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "string") {
    return quote(value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(formatPackageLiteral).join(", ")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return "{}";
    }
    return `{ ${entries
      .map(([key, entry]) => `${key}: ${formatPackageLiteral(entry)}`)
      .join(", ")} }`;
  }
  return quote(String(value));
}

/** Serializes `render: @scene { ... }` / `finalCard: @scene { ... }` bodies. */
function formatEmbeddedScene(scene: unknown, level: number): string {
  const prefix = indent(level);
  const inner = indent(level + 1);

  if (isEmbeddedSceneSource(scene)) {
    const body = scene.body.trim();
    if (body.length === 0) {
      return `${prefix}@scene {\n${prefix}}`;
    }
    return `${prefix}@scene {\n${inner}${body}\n${prefix}}`;
  }

  if (
    typeof scene === "object" &&
    scene !== null &&
    Array.isArray((scene as { roots?: unknown }).roots)
  ) {
    const pkg = scene as {
      roots: readonly unknown[];
      timeline?: readonly unknown[];
      camera?: readonly unknown[];
    };
    const lines = [`${inner}roots: ${formatPackageLiteral(pkg.roots)}`];
    if (pkg.timeline !== undefined && pkg.timeline.length > 0) {
      lines.push(`${inner}timeline: ${formatPackageLiteral(pkg.timeline)}`);
    }
    if (pkg.camera !== undefined && pkg.camera.length > 0) {
      lines.push(`${inner}camera: ${formatPackageLiteral(pkg.camera)}`);
    }
    return `${prefix}@scene {\n${lines.join("\n")}\n${prefix}}`;
  }

  throw new Error(
    `Unsupported embedded scene for formatting: ${JSON.stringify(scene)?.slice(0, 200)}`,
  );
}

function formatHub(hub: ExplainerHubAst, level: number): string {
  const inner = indent(level + 1);
  return block(
    "hub:",
    [
      `${inner}highlight: ${quote(hub.highlight)}`,
      `${inner}title: ${quote(hub.title)}`,
      `${inner}description: ${quote(hub.description)}`,
    ],
    level,
  );
}

function formatSlide(slide: SlideAst, level: number): string {
  const inner = indent(level + 1);
  const lines = [
    `${inner}eyebrow: ${quote(slide.eyebrow)}`,
    `${inner}title: ${quote(slide.title)}`,
    `${inner}lede: ${quote(slide.lede)}`,
    `${inner}narration: ${quote(slide.narration)}`,
  ];
  if (slide.term !== undefined) {
    lines.push(
      `${inner}term: { word: ${quote(slide.term.word)}, meaning: ${quote(slide.term.meaning)} }`,
    );
  }
  if (slide.points.length > 0) {
    lines.push(
      `${inner}points: [${slide.points.map((point) => quote(point)).join(", ")}]`,
    );
  }
  lines.push(`${inner}caption: ${quote(slide.caption)}`);
  if (slide.sceneIr !== undefined) {
    lines.push(
      `${inner}render: ${formatEmbeddedScene(slide.sceneIr, level + 1).trimStart()}`,
    );
  }
  return block(`slide ${quote(slide.title)}`, lines, level);
}

function formatExplainer(explainer: ExplainerAst): string {
  const meta = explainer.metadata;
  const lines = [
    `${indent(1)}id: ${quote(explainer.id)}`,
    `${indent(1)}route: ${quote(meta.route)}`,
    `${indent(1)}topic: ${quote(meta.topic)}`,
    `${indent(1)}storagePrefix: ${quote(meta.storagePrefix)}`,
    `${indent(1)}classPrefix: ${quote(meta.classPrefix)}`,
    `${indent(1)}eyebrowLabel: ${quote(meta.eyebrowLabel)}`,
    `${indent(1)}startGateTitle: ${quote(meta.startGateTitle)}`,
    ...(meta.css === undefined
      ? []
      : [`${indent(1)}css: ${quote(meta.css)}`]),
    formatHub(meta.hub, 1),
    ...explainer.slides.map((slide) => formatSlide(slide, 1)),
    ...(explainer.finalCard === undefined
      ? []
      : [
          `${indent(1)}finalCard: ${formatEmbeddedScene(explainer.finalCard, 1).trimStart()}`,
        ]),
  ];
  return block(`explainer ${quote(explainer.id)}`, lines, 0);
}

export function formatDocument(document: DocumentAst): string {
  // Explainer-form decks are top-level `explainer` blocks (parser routes on the
  // first token). Emitting a `flow { ... }` wrapper would drop slides on reparse.
  if (document.explainers !== undefined && document.explainers.length > 0) {
    return `${document.explainers.map(formatExplainer).join("\n\n")}\n`;
  }

  const declarations = [
    `${indent(1)}language ${document.language.version}`,
    ...(document.imports ?? []).map(formatImport),
    ...document.requirements.map(
      (requirement) =>
        `${indent(1)}require ${requirement.capability} ${quote(requirement.versionRange)}`,
    ),
    ...document.tokens.map(
      (token) =>
        `${indent(1)}token ${token.id} = ${formatValue(token.value)}`,
    ),
    ...document.themes.map(formatTheme),
    ...(document.useTheme === undefined
      ? []
      : [`${indent(1)}use theme ${document.useTheme.themeId}`]),
    ...document.symbols.map(formatSymbol),
    ...document.scenes.map(formatScene),
  ];

  return `flow ${quote(document.title)} as ${document.id} {\n${declarations.join("\n\n")}\n}\n`;
}
