// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArgumentValueAst,
  CameraAst,
  ComponentInvocationAst,
  ConnectorAst,
  DocumentAst,
  ForLoopAst,
  ImportDeclarationAst,
  InteractionAst,
  ObjectLiteralAst,
  ParamDeclarationAst,
  RectAst,
  RenderDeclarationAst,
  ResponsiveAst,
  SceneAst,
  ScenePrimitiveAst,
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
  return block(
    `for ${loop.item} in ${loop.collection}`,
    loop.body.map((statement) => formatSymbolBodyStatement(statement, level + 1)),
    level,
  );
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

function formatTimelineCue(cue: TimelineCueAst): string {
  const prefix = `${indent(3)}at ${cue.time} ${cue.action}`;
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

export function formatDocument(document: DocumentAst): string {
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
