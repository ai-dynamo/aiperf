// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared helpers for explainer slide `render: @scene { ... }` bodies.
 *
 * Two dialects are accepted:
 * - **package** — decks-flow JSON-ish `roots` / `timeline` / `camera` object form
 * - **native** — cinematic Flow scene statements (`rect` / `connector` / `panel` /
 *   `elbow` / `stack` / `signal` / `timeline` / …)
 *
 * Package objects accept arbitrary keys. Known geometry / animation fields that
 * round-trip into SceneAst props (for compiler lowering) include:
 * - panel: `title`, `detail`
 * - header: `title`, `caption`
 * - circle / ellipse: `r` / `rx` / `ry`, `center` `{ x, y }`
 * - stack / grid / rail: `direction`, `cols`, `gap` (also via `style`)
 * - lane / band / swimlane / stepper: structure macros (`title`, `steps`, `labels`, `linked`)
 * - route / elbow / connector: `from`, `to`, `via`, `axis`
 * - fan-out / fan-in: scalar or array `from` / `to`, `axis`, `junction`
 * - motion.signal: `d`, or `from` / `to`
 * - stagger cues: `targets`, `step`, `easing`; actions `stagger` /
 *   `enter-children` / `fade` / `exit` (plus existing `enter` / `draw` / …)
 *
 * Native bodies stay as source so the Chevrotain scene rules in `parseDocument`
 * can lower them without a circular import. Package bodies parse here into a
 * SceneAst that `lowerExplainerScene` accepts (plus preserved roots/timeline).
 */

import type {
  ArgumentValueAst,
  ComponentInvocationAst,
  PropAssignmentAst,
  RectAst,
  RenderDeclarationAst,
  SceneAst,
  TimelineAction,
  TimelineCueAst,
  TimelineCueEasing,
  ValueAst,
} from "./ast.js";
import type { TokenStream } from "./grammar/explainer.js";
import type { SourceRange } from "../schema/index.js";

/** Discriminator for which `@scene` body dialect was authored. */
export type EmbeddedSceneForm = "native" | "package";

/** Captured `@scene` body before dialect-specific parsing/lowering. */
export type EmbeddedSceneSource = Readonly<{
  kind: "embedded-scene-source";
  form: EmbeddedSceneForm;
  body: string;
}>;

/** Parsed decks-flow package scene (`roots` + optional `timeline` / `camera`). */
export type PackageSceneAst = Readonly<{
  kind: "package-scene";
  roots: readonly unknown[];
  timeline: readonly unknown[];
  camera: readonly unknown[];
  /** Optional scene-level identity / a11y fields authored beside roots/timeline. */
  id?: string;
  title?: string;
  summary?: string;
  narration?: string;
  fallback?: string;
  viewport?: unknown;
  accessibility?: unknown;
  /** Owning source range when available (slide / document provenance). */
  sourceMap?: SourceRange;
}>;

/** SceneAst plus preserved package IR fields for compiler shortcut lowering. */
export type PackageSceneIrAst = SceneAst &
  Readonly<{
    roots: readonly unknown[];
    timeline: readonly unknown[];
    camera: readonly unknown[];
    viewport?: unknown;
    accessibility?: unknown;
  }>;

const unknownRange: SourceRange = {
  source: "<embedded-scene>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Token streams that can capture `@scene` bodies must expose the current image. */
export type PeekableTokenStream = TokenStream &
  Readonly<{
    peekImage(): string | undefined;
  }>;

/**
 * Detects which embedded-scene dialect a body uses.
 *
 * Package form leads with `roots:`, `timeline:`, or `camera:` field labels.
 * Anything else is treated as native cinematic scene statements.
 */
export function detectEmbeddedSceneForm(body: string): EmbeddedSceneForm {
  const trimmed = body.trimStart();
  if (/^(roots|timeline|camera)\s*:/.test(trimmed)) {
    return "package";
  }
  return "native";
}

/**
 * Captures an `@scene` body and tags its dialect without fully parsing native
 * scene statements (those reuse `parseDocument` / scene rules later).
 */
export function captureEmbeddedScene(
  tokens: PeekableTokenStream,
): EmbeddedSceneSource {
  const body = captureSceneBody(tokens);
  return {
    kind: "embedded-scene-source",
    form: detectEmbeddedSceneForm(body),
    body,
  };
}

/**
 * Consumes a `{ ... }` block and returns the inner body text (without the
 * outer braces).
 */
export function captureSceneBody(tokens: PeekableTokenStream): string {
  tokens.expect("{");
  const parts: string[] = [];
  let depth = 1;
  while (depth > 0) {
    const image = tokens.peekImage();
    if (image === undefined) {
      throw new Error("Unterminated @scene block");
    }
    if (image === "{") {
      depth += 1;
      parts.push(image);
      tokens.advance();
    } else if (image === "}") {
      depth -= 1;
      if (depth > 0) {
        parts.push(image);
      }
      tokens.advance();
    } else {
      parts.push(image);
      tokens.advance();
    }
  }
  return joinSceneTokens(parts);
}

/**
 * Parses a decks-flow package `@scene` body into roots/timeline/camera arrays.
 */
export function parsePackageSceneBody(body: string): PackageSceneAst {
  return new PackageSceneParser(body).parse();
}

/**
 * Parses an embedded scene source: package form → SceneAst; native form left
 * as source for the shared cinematic parser.
 */
export function parseEmbeddedSceneSource(
  source: EmbeddedSceneSource,
): EmbeddedSceneSource | PackageSceneIrAst {
  if (source.form === "package") {
    return packageSceneToSceneAst(parsePackageSceneBody(source.body));
  }
  return source;
}

/**
 * Converts a parsed package `@scene` body into a SceneAst that
 * `lowerExplainerScene` accepts, preserving roots/timeline for IR-path lowerers.
 */
export function packageSceneToSceneAst(
  packageScene: PackageSceneAst,
  sourceMap: SourceRange = unknownRange,
): PackageSceneIrAst {
  const renderDeclarations = flattenRoots(packageScene.roots, sourceMap);
  const cues = packageScene.timeline
    .map((cue) => timelineCueFromPackage(cue, sourceMap))
    .filter((cue): cue is TimelineCueAst => cue !== undefined);
  const readingOrder = renderDeclarations
    .map((node) => ("id" in node ? node.id : undefined))
    .filter((id): id is string => typeof id === "string" && id.length > 0);
  const label =
    readingOrder[0] !== undefined ? readingOrder[0] : "embedded-scene";
  const sceneId =
    typeof packageScene.id === "string" && packageScene.id.length > 0
      ? packageScene.id
      : "embedded-scene";
  const sceneTitle =
    typeof packageScene.title === "string" && packageScene.title.length > 0
      ? packageScene.title
      : "Embedded scene";
  const summaryText =
    typeof packageScene.summary === "string" && packageScene.summary.length > 0
      ? packageScene.summary
      : "Embedded explainer scene";
  const narrationText =
    typeof packageScene.narration === "string"
      ? packageScene.narration
      : summaryText;
  const fallbackText =
    typeof packageScene.fallback === "string" && packageScene.fallback.length > 0
      ? packageScene.fallback
      : label;

  return {
    kind: "scene",
    id: sceneId,
    title: sceneTitle,
    summary: {
      kind: "summary",
      text: summaryText,
      sourceMap,
    },
    renderDeclarations,
    cameras: [],
    timelines:
      cues.length === 0
        ? []
        : [
            {
              kind: "timeline",
              id: "primary",
              cues,
              sourceMap,
            },
          ],
    interactions: [],
    responsiveVariants: [],
    narration: {
      kind: "narration",
      text: narrationText,
      sourceMap,
    },
    readingOrder: {
      kind: "reading-order",
      references: readingOrder,
      sourceMap,
    },
    fallback: {
      kind: "fallback",
      text: fallbackText,
      sourceMap,
    },
    sourceMap,
    roots: packageScene.roots,
    timeline: packageScene.timeline,
    camera: packageScene.camera,
    ...(packageScene.viewport !== undefined
      ? { viewport: packageScene.viewport }
      : {}),
    ...(packageScene.accessibility !== undefined
      ? { accessibility: packageScene.accessibility }
      : {}),
  };
}

function flattenRoots(
  roots: readonly unknown[],
  sourceMap: SourceRange,
): RenderDeclarationAst[] {
  const out: RenderDeclarationAst[] = [];
  for (const root of roots) {
    collectRenderDeclarations(root, sourceMap, out);
  }
  return out;
}

function collectRenderDeclarations(
  node: unknown,
  sourceMap: SourceRange,
  out: RenderDeclarationAst[],
): void {
  if (typeof node !== "object" || node === null || Array.isArray(node)) {
    return;
  }
  const record = node as Record<string, unknown>;
  const decl = renderDeclarationFromPackage(record, sourceMap);
  if (decl !== undefined) {
    out.push(decl);
  }
  const children = record.children;
  if (Array.isArray(children)) {
    for (const child of children) {
      collectRenderDeclarations(child, sourceMap, out);
    }
  }
}

function renderDeclarationFromPackage(
  node: Record<string, unknown>,
  sourceMap: SourceRange,
): RenderDeclarationAst | undefined {
  const id = typeof node.id === "string" ? node.id : undefined;
  if (id === undefined || id.length === 0) {
    return undefined;
  }
  const capability = capabilityOf(node);
  const layout = geometryOf(node);
  const style =
    typeof node.style === "object" && node.style !== null && !Array.isArray(node.style)
      ? (node.style as Record<string, unknown>)
      : {};
  const accessibility =
    typeof node.accessibility === "object" &&
    node.accessibility !== null &&
    !Array.isArray(node.accessibility)
      ? (node.accessibility as Record<string, unknown>)
      : {};
  const label =
    typeof accessibility.label === "string" && accessibility.label.length > 0
      ? accessibility.label
      : id;
  const description =
    typeof accessibility.description === "string"
      ? accessibility.description
      : "";

  if (capability === "core.rect" || node.kind === "rect") {
    const rect: RectAst = {
      kind: "rect",
      id,
      x: layout.x,
      y: layout.y,
      width: layout.width,
      height: layout.height,
      fill: valueFromUnknown(style.fill ?? "#244a35", sourceMap),
      ...(style.stroke !== undefined
        ? { stroke: valueFromUnknown(style.stroke, sourceMap) }
        : {}),
      label,
      role: "img",
      description,
      fallback: { kind: "fallback", text: label, sourceMap },
      sourceMap,
    };
    return rect;
  }

  const props: PropAssignmentAst[] = [
    {
      kind: "prop-assignment",
      name: "id",
      value: { kind: "literal", value: id, sourceMap },
      sourceMap,
    },
    {
      kind: "prop-assignment",
      name: "capabilityId",
      value: { kind: "literal", value: capability, sourceMap },
      sourceMap,
    },
    {
      kind: "prop-assignment",
      name: "layout",
      value: {
        kind: "object-literal",
        properties: [
          {
            kind: "object-property",
            name: "x",
            value: { kind: "literal", value: layout.x, sourceMap },
            sourceMap,
          },
          {
            kind: "object-property",
            name: "y",
            value: { kind: "literal", value: layout.y, sourceMap },
            sourceMap,
          },
          {
            kind: "object-property",
            name: "width",
            value: { kind: "literal", value: layout.width, sourceMap },
            sourceMap,
          },
          {
            kind: "object-property",
            name: "height",
            value: { kind: "literal", value: layout.height, sourceMap },
            sourceMap,
          },
        ],
        sourceMap,
      },
      sourceMap,
    },
  ];
  if (typeof node.text === "string") {
    props.push({
      kind: "prop-assignment",
      name: "text",
      value: { kind: "literal", value: node.text, sourceMap },
      sourceMap,
    });
  }

  // Known geometry / layout / motion fields (scalars + one-level objects).
  for (const key of KNOWN_PACKAGE_PROP_KEYS) {
    if (!(key in node) || node[key] === undefined) {
      continue;
    }
    props.push({
      kind: "prop-assignment",
      name: key,
      value: argumentFromUnknown(node[key], sourceMap),
      sourceMap,
    });
  }

  for (const [key, value] of Object.entries(style)) {
    props.push({
      kind: "prop-assignment",
      name: key,
      value: argumentFromUnknown(value, sourceMap),
      sourceMap,
    });
  }

  const invocation: ComponentInvocationAst = {
    kind: "component-invocation",
    name: capability.includes(".") ? capability : `core.${capability || "node"}`,
    props,
    sourceMap,
  };
  return invocation;
}

/** Top-level package node fields mirrored onto SceneAst prop assignments. */
const KNOWN_PACKAGE_PROP_KEYS = [
  "title",
  "detail",
  "caption",
  "r",
  "rx",
  "ry",
  "center",
  "from",
  "to",
  "via",
  "axis",
  "junction",
  "direction",
  "cols",
  "gap",
  "d",
  "path",
  "points",
] as const;

function timelineCueFromPackage(
  cue: unknown,
  sourceMap: SourceRange,
): TimelineCueAst | undefined {
  if (typeof cue !== "object" || cue === null || Array.isArray(cue)) {
    return undefined;
  }
  const record = cue as Record<string, unknown>;
  const actionRaw =
    typeof record.action === "string" ? record.action : undefined;
  if (actionRaw === undefined) {
    return undefined;
  }
  const action = normalizeTimelineAction(actionRaw);
  const targets = Array.isArray(record.targets)
    ? record.targets.filter(
        (entry): entry is string =>
          typeof entry === "string" && entry.length > 0,
      )
    : undefined;
  const target =
    typeof record.target === "string"
      ? record.target
      : targets !== undefined && targets.length > 0
        ? ""
        : undefined;
  if (target === undefined) {
    return undefined;
  }
  const at = finiteNumber(record.at ?? record.time, 0);
  const duration = finiteNumber(record.duration, 0);
  const step =
    typeof record.step === "number" && Number.isFinite(record.step)
      ? record.step
      : undefined;
  const easing = normalizeTimelineEasing(record.easing);
  return {
    kind: "timeline-cue",
    time: at,
    duration,
    target,
    action,
    ...(targets !== undefined && targets.length > 0 ? { targets } : {}),
    ...(step !== undefined ? { step } : {}),
    ...(easing !== undefined ? { easing } : {}),
    sourceMap,
  };
}

function normalizeTimelineAction(action: string): TimelineAction {
  switch (action) {
    case "trace":
      return "trace";
    case "enter":
      // Package decks use "enter"; cinematic SceneAst vocabulary uses "reveal".
      return "reveal";
    case "reveal":
    case "draw":
    case "fade":
    case "exit":
    case "emphasis":
    case "emphasize":
    case "pulse":
    case "stagger":
    case "enter-children":
      return action;
    default:
      return "reveal";
  }
}

function normalizeTimelineEasing(value: unknown): TimelineCueEasing | undefined {
  if (
    value === "linear" ||
    value === "ease-in" ||
    value === "ease-out" ||
    value === "ease-in-out"
  ) {
    return value;
  }
  return undefined;
}

function capabilityOf(node: Record<string, unknown>): string {
  if (typeof node.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  if (typeof node.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node.kind === "string" && node.kind.length > 0) {
    return node.kind.includes(".") ? node.kind : `core.${node.kind}`;
  }
  return "core.rect";
}

function geometryOf(node: Record<string, unknown>): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const box =
    (typeof node.layout === "object" && node.layout !== null
      ? (node.layout as Record<string, unknown>)
      : undefined) ??
    (typeof node.geometry === "object" && node.geometry !== null
      ? (node.geometry as Record<string, unknown>)
      : undefined) ??
    {};
  return {
    x: finiteNumber(box.x, 0),
    y: finiteNumber(box.y, 0),
    width: finiteNumber(box.width, 0),
    height: finiteNumber(box.height, 0),
  };
}

function argumentFromUnknown(
  value: unknown,
  sourceMap: SourceRange,
): ArgumentValueAst {
  if (typeof value === "string" && value.startsWith("@theme.")) {
    return {
      kind: "theme-role-reference",
      role: value.slice("@theme.".length),
      sourceMap,
    };
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return { kind: "literal", value, sourceMap };
  }
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    const properties = Object.entries(value as Record<string, unknown>).map(
      ([name, entry]) => ({
        kind: "object-property" as const,
        name,
        value: valueFromUnknown(entry, sourceMap),
        sourceMap,
      }),
    );
    return {
      kind: "object-literal",
      properties,
      sourceMap,
    };
  }
  if (Array.isArray(value)) {
    // Points / path arrays stay as JSON text for the compiler to re-parse.
    return { kind: "literal", value: JSON.stringify(value), sourceMap };
  }
  return { kind: "literal", value: String(value ?? ""), sourceMap };
}

function valueFromUnknown(value: unknown, sourceMap: SourceRange): ValueAst {
  if (typeof value === "string" && value.startsWith("@theme.")) {
    return {
      kind: "theme-role-reference",
      role: value.slice("@theme.".length),
      sourceMap,
    };
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return { kind: "literal", value, sourceMap };
  }
  return { kind: "literal", value: String(value ?? ""), sourceMap };
}

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function joinSceneTokens(parts: readonly string[]): string {
  let out = "";
  for (const part of parts) {
    if (out.length === 0) {
      out = part;
      continue;
    }
    const prev = out[out.length - 1]!;
    const noSpaceBefore = /^[.,\]}:]/.test(part);
    const noSpaceAfter = /[@[{(:]$/.test(prev) || prev === ".";
    if (noSpaceBefore || noSpaceAfter) {
      out += part;
    } else {
      out += ` ${part}`;
    }
  }
  return out.trim();
}

class PackageSceneParser {
  private readonly tokens: string[];
  private index = 0;

  constructor(source: string) {
    this.tokens = tokenizePackageScene(source);
  }

  parse(): PackageSceneAst {
    const roots: unknown[] = [];
    const timeline: unknown[] = [];
    const camera: unknown[] = [];
    let id: string | undefined;
    let title: string | undefined;
    let summary: string | undefined;
    let narration: string | undefined;
    let fallback: string | undefined;
    let viewport: unknown;
    let accessibility: unknown;

    while (!this.done) {
      const key = this.expectIdentifier();
      this.expect(":");
      const value = this.parseValue();
      if (key === "roots") {
        if (!Array.isArray(value)) {
          throw new Error('package @scene field "roots" must be an array');
        }
        roots.push(...value);
      } else if (key === "timeline") {
        if (!Array.isArray(value)) {
          throw new Error('package @scene field "timeline" must be an array');
        }
        timeline.push(...value);
      } else if (key === "camera") {
        if (!Array.isArray(value)) {
          throw new Error('package @scene field "camera" must be an array');
        }
        camera.push(...value);
      } else if (key === "id" && typeof value === "string") {
        id = value;
      } else if (key === "title" && typeof value === "string") {
        title = value;
      } else if (key === "summary" && typeof value === "string") {
        summary = value;
      } else if (key === "narration" && typeof value === "string") {
        narration = value;
      } else if (key === "fallback" && typeof value === "string") {
        fallback = value;
      } else if (key === "viewport") {
        viewport = value;
      } else if (key === "accessibility") {
        accessibility = value;
      } else {
        throw new Error(
          `Unknown package @scene field "${key}" (expected roots, timeline, camera, or scene metadata)`,
        );
      }
      if (this.match(",")) {
        this.advance();
      }
    }

    return {
      kind: "package-scene",
      roots,
      timeline,
      camera,
      ...(id !== undefined ? { id } : {}),
      ...(title !== undefined ? { title } : {}),
      ...(summary !== undefined ? { summary } : {}),
      ...(narration !== undefined ? { narration } : {}),
      ...(fallback !== undefined ? { fallback } : {}),
      ...(viewport !== undefined ? { viewport } : {}),
      ...(accessibility !== undefined ? { accessibility } : {}),
    };
  }

  private get done(): boolean {
    return this.index >= this.tokens.length;
  }

  private peek(): string | undefined {
    return this.tokens[this.index];
  }

  private match(image: string): boolean {
    return this.peek() === image;
  }

  private advance(): void {
    this.index += 1;
  }

  private expect(image: string): void {
    if (!this.match(image)) {
      throw new Error(`Expected "${image}" but got "${this.peek() ?? "<eof>"}"`);
    }
    this.advance();
  }

  private expectIdentifier(): string {
    const current = this.peek();
    if (
      current === undefined ||
      /^(?:[{}\[\]:,@.])$/.test(current) ||
      current.startsWith('"')
    ) {
      throw new Error(`Expected identifier but got "${current ?? "<eof>"}"`);
    }
    this.advance();
    return current;
  }

  private parseValue(): unknown {
    const current = this.peek();
    if (current === undefined) {
      throw new Error("Unexpected end of package @scene body");
    }
    if (current === "{") {
      return this.parseObject();
    }
    if (current === "[") {
      return this.parseArray();
    }
    if (current === "@") {
      return this.parseThemeRef();
    }
    if (current.startsWith('"')) {
      this.advance();
      return JSON.parse(current) as string;
    }
    if (current === "true") {
      this.advance();
      return true;
    }
    if (current === "false") {
      this.advance();
      return false;
    }
    if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(current)) {
      this.advance();
      return Number(current);
    }
    this.advance();
    return current;
  }

  private parseThemeRef(): string {
    this.expect("@");
    const parts: string[] = [this.expectIdentifier()];
    while (this.match(".")) {
      this.advance();
      parts.push(this.expectIdentifier());
    }
    return `@${parts.join(".")}`;
  }

  private parseObject(): Record<string, unknown> {
    this.expect("{");
    const object: Record<string, unknown> = {};
    while (!this.match("}")) {
      if (this.done) {
        throw new Error("Unterminated object in package @scene");
      }
      const key = this.expectIdentifier();
      this.expect(":");
      object[key] = this.parseValue();
      if (this.match(",")) {
        this.advance();
      }
    }
    this.expect("}");
    return object;
  }

  private parseArray(): unknown[] {
    this.expect("[");
    const items: unknown[] = [];
    while (!this.match("]")) {
      if (this.done) {
        throw new Error("Unterminated array in package @scene");
      }
      items.push(this.parseValue());
      if (this.match(",")) {
        this.advance();
      }
    }
    this.expect("]");
    return items;
  }
}

function tokenizePackageScene(source: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i]!;
    if (char === '"') {
      inQuotes = !inQuotes;
      current += char;
      continue;
    }
    if (inQuotes) {
      if (char === "\\" && i + 1 < source.length) {
        current += char + source[i + 1]!;
        i += 1;
        continue;
      }
      current += char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }
    // Keep decimal literals (1.8) and hex colors (#aabbcc) intact.
    if (
      char === "." &&
      /\d$/.test(current) &&
      i + 1 < source.length &&
      /\d/.test(source[i + 1]!)
    ) {
      current += char;
      continue;
    }
    if (char === "#" && current === "") {
      current = char;
      continue;
    }
    if (/[{}\[\]:,@.]/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      tokens.push(char);
      continue;
    }
    current += char;
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}
