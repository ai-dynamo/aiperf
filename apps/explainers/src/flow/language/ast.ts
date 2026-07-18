// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SourceRange } from "../schema/index.js";

export type AstNode<Kind extends string> = Readonly<{
  kind: Kind;
  sourceMap: SourceRange;
}>;

export type LiteralAst = AstNode<"literal"> &
  Readonly<{ value: string | number | boolean }>;

export type TokenReferenceAst = AstNode<"token-reference"> &
  Readonly<{ token: string }>;

export type ThemeRoleReferenceAst = AstNode<"theme-role-reference"> &
  Readonly<{ role: string }>;

export type IdentifierReferenceAst = AstNode<"identifier-reference"> &
  Readonly<{ name: string }>;

export type ReferenceListAst = AstNode<"reference-list"> &
  Readonly<{ references: readonly string[] }>;

export type ReadingOrderAst = AstNode<"reading-order"> &
  Readonly<{ references: readonly string[] }>;

export type ValueAst = LiteralAst | TokenReferenceAst | ThemeRoleReferenceAst;

export type ThemeValueKindAst =
  | "color"
  | "number"
  | "duration"
  | "font"
  | "enum";

export type ThemeFontLiteralAst = AstNode<"theme-font-literal"> &
  Readonly<{ families: readonly string[] }>;

export type ThemeAssignmentAst = AstNode<"theme-assignment"> &
  Readonly<{
    valueKind: ThemeValueKindAst;
    role: string;
    value: LiteralAst | ThemeFontLiteralAst;
  }>;

export type ThemeDeclarationAst = AstNode<"theme-declaration"> &
  Readonly<{
    id: string;
    extends: string;
    assignments: readonly ThemeAssignmentAst[];
  }>;

export type UseThemeAst = AstNode<"use-theme"> &
  Readonly<{ themeId: string }>;

export type LanguageDeclarationAst = AstNode<"language"> &
  Readonly<{ version: number }>;

export type ImportDeclarationAst = AstNode<"import"> &
  Readonly<{ path: string; alias: string }>;

export type RequirementAst = AstNode<"requirement"> &
  Readonly<{ capability: string; versionRange: string }>;

export type TokenDeclarationAst = AstNode<"token"> &
  Readonly<{ id: string; value: LiteralAst }>;

export type SummaryAst = AstNode<"summary"> & Readonly<{ text: string }>;
export type NarrationAst = AstNode<"narration"> & Readonly<{ text: string }>;
export type FallbackAst = AstNode<"fallback"> & Readonly<{ text: string }>;

export type RectAst = AstNode<"rect"> &
  Readonly<{
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    fill: ValueAst;
    stroke?: ValueAst;
    label: string;
    role: string;
    description: string;
    fallback: FallbackAst;
  }>;

export type ConnectorAst = AstNode<"connector"> &
  Readonly<{
    id: string;
    from: string;
    to: string;
    label: string;
    stroke: ValueAst;
    fallback: FallbackAst;
  }>;

/**
 * Native cinematic keyword for a core / layout / motion primitive.
 * Maps 1:1 onto package `capability` ids (see `SCENE_PRIMITIVE_CAPABILITIES`).
 */
export type ScenePrimitiveKeyword =
  | "panel"
  | "header"
  | "circle"
  | "ellipse"
  | "arrow"
  | "elbow"
  | "bracket"
  | "callout"
  | "chip"
  | "note"
  | "divider"
  | "rail"
  | "lane"
  | "band"
  | "swimlane"
  | "stepper"
  | "route"
  | "fan-out"
  | "fan-in"
  | "stack"
  | "grid"
  | "pad"
  | "signal";

/** Capability ids emitted for native scene primitives (Task 1 schema vocabulary). */
export const SCENE_PRIMITIVE_CAPABILITIES = {
  panel: "core.panel",
  header: "core.header",
  circle: "core.circle",
  ellipse: "core.ellipse",
  arrow: "core.arrow",
  elbow: "core.elbow",
  bracket: "core.bracket",
  callout: "core.callout",
  chip: "core.chip",
  note: "core.note",
  divider: "core.divider",
  rail: "layout.rail",
  lane: "core.lane",
  band: "core.band",
  swimlane: "core.swimlane",
  stepper: "core.stepper",
  route: "core.route",
  "fan-out": "core.fan-out",
  "fan-in": "core.fan-in",
  stack: "layout.stack",
  grid: "layout.grid",
  pad: "layout.pad",
  signal: "motion.signal",
} as const satisfies Record<ScenePrimitiveKeyword, string>;

/**
 * Native geometry / layout / motion node. Props mirror package authoring
 * field names (`title`/`detail`, `r`/`center`, `from`/`to`/`via`/`axis`,
 * `direction`/`cols`/`gap`, `d`, …) so the compiler can lower both dialects.
 */
export type ScenePrimitiveAst = AstNode<"scene-primitive"> &
  Readonly<{
    id: string;
    /** Native keyword used in source (`panel`, `stack`, `signal`, …). */
    primitive: ScenePrimitiveKeyword;
    /** Package-form capability id (`core.panel`, `layout.stack`, …). */
    capability: string;
    props: readonly PropAssignmentAst[];
    children?: readonly RenderDeclarationAst[];
    fallback?: FallbackAst;
  }>;

export type NamedTypeRefAst = AstNode<"type-ref"> &
  Readonly<{ name: string; array?: false }>;

export type ArrayTypeRefAst = AstNode<"type-ref"> &
  Readonly<{ name: string; array: true }>;

export type TypeRefAst = NamedTypeRefAst | ArrayTypeRefAst;

/** Compatibility name for consumers using the expanded symbol grammar terminology. */
export type TypeReferenceAst = TypeRefAst;

export type ParamDeclarationAst = AstNode<"param"> &
  Readonly<{ name: string; type: TypeRefAst }>;

/** Compatibility name for a symbol parameter declaration. */
export type ParameterAst = ParamDeclarationAst;

export type ObjectPropertyAst = AstNode<"object-property"> &
  Readonly<{
    name: string;
    value: ValueAst | IdentifierReferenceAst;
  }>;

export type ObjectLiteralAst = AstNode<"object-literal"> &
  Readonly<{ properties: readonly ObjectPropertyAst[] }>;

export type ArgumentValueAst =
  | ValueAst
  | IdentifierReferenceAst
  | ObjectLiteralAst;

export type PropAssignmentAst = AstNode<"prop-assignment"> &
  Readonly<{ name: string; value: ArgumentValueAst }>;

/** Compatibility name for a named component argument. */
export type NamedArgumentAst = PropAssignmentAst;

type ComponentInvocationBaseAst = AstNode<"component-invocation"> &
  Readonly<{
    name: string;
    props: readonly PropAssignmentAst[];
    slots?: readonly SlotBlockAst[];
  }>;

export type UnqualifiedComponentInvocationAst = ComponentInvocationBaseAst &
  Readonly<{ namespace?: undefined }>;

export type QualifiedComponentInvocationAst = ComponentInvocationBaseAst &
  Readonly<{ namespace: string }>;

export type ComponentInvocationAst =
  | UnqualifiedComponentInvocationAst
  | QualifiedComponentInvocationAst;

/** Compatibility name for component invocations in symbol bodies. */
export type ComponentCallAst = ComponentInvocationAst;

export type SlotBlockAst = AstNode<"slot"> &
  Readonly<{
    name: string;
    parameter?: string;
    body: readonly SymbolBodyStatementAst[];
  }>;

export type ForLoopAst = AstNode<"for-loop"> &
  Readonly<{
    item: string;
    collection: string;
    body: readonly SymbolBodyStatementAst[];
  }>;

export type SymbolBodyStatementAst =
  | ComponentInvocationAst
  | SlotBlockAst
  | ForLoopAst;

export type SymbolDefinitionAst = AstNode<"symbol-definition"> &
  Readonly<{
    name: string;
    params: readonly ParamDeclarationAst[];
    body: readonly SymbolBodyStatementAst[];
  }>;

export type RenderDeclarationAst =
  | RectAst
  | ConnectorAst
  | ScenePrimitiveAst
  | ComponentInvocationAst;

export type CameraKeyframeAst = AstNode<"camera-keyframe"> &
  Readonly<{
    time: number;
    targets: ReferenceListAst;
    zoom: number;
  }>;

export type CameraAst = AstNode<"camera"> &
  Readonly<{ id: string; keyframes: readonly CameraKeyframeAst[] }>;

/**
 * Timeline cue actions accepted by native cinematic scenes.
 * Package decks also author `enter` (mapped to `reveal` when building SceneAst).
 */
export type TimelineAction =
  | "reveal"
  | "trace"
  | "enter"
  | "draw"
  | "fade"
  | "exit"
  | "emphasis"
  | "emphasize"
  | "pulse"
  | "stagger"
  | "enter-children";

/** Per-cue easing pass-through (mirrors schema `TimelineCueEasing`). */
export type TimelineCueEasing =
  | "linear"
  | "ease-in"
  | "ease-out"
  | "ease-in-out";

export type TimelineCueAst = AstNode<"timeline-cue"> &
  Readonly<{
    time: number;
    action: TimelineAction;
    /** Primary / group id; may be `""` when `targets` identifies stagger members. */
    target: string;
    duration: number;
    /** Stagger member node ids when `action` is `stagger` / `enter-children`. */
    targets?: readonly string[];
    /** Delay between successive stagger targets. */
    step?: number;
    easing?: TimelineCueEasing;
  }>;

export type TimelineAst = AstNode<"timeline"> &
  Readonly<{ id: string; cues: readonly TimelineCueAst[] }>;

export type InteractionEventAst = AstNode<"interaction-event"> &
  Readonly<{ name: "select"; target: string }>;

export type InteractionActionAst = AstNode<"interaction-action"> &
  Readonly<{ name: "inspect"; target: string }>;

export type InteractionAst = AstNode<"interaction"> &
  Readonly<{
    id: string;
    event: InteractionEventAst;
    action: InteractionActionAst;
  }>;

export type ComparisonOperator = "<" | "<=" | ">" | ">=" | "==" | "!=";

export type ResponsiveConditionAst = AstNode<"responsive-condition"> &
  Readonly<{
    property: string;
    operator: ComparisonOperator;
    value: number;
  }>;

export type ResponsiveOverrideAst = AstNode<"responsive-override"> &
  Readonly<{
    target: string;
    property: string;
    value: number;
  }>;

export type ResponsiveAst = AstNode<"responsive"> &
  Readonly<{
    id: string;
    condition: ResponsiveConditionAst;
    overrides: readonly ResponsiveOverrideAst[];
  }>;

export type SceneAst = AstNode<"scene"> &
  Readonly<{
    title: string;
    id: string;
    summary?: SummaryAst;
    renderDeclarations: readonly RenderDeclarationAst[];
    cameras: readonly CameraAst[];
    timelines: readonly TimelineAst[];
    interactions: readonly InteractionAst[];
    responsiveVariants: readonly ResponsiveAst[];
    narration?: NarrationAst;
    readingOrder?: ReadingOrderAst;
    fallback?: FallbackAst;
  }>;

export type SlideAst = AstNode<"slide"> &
  Readonly<{
    eyebrow: string;
    title: string;
    lede: string;
    narration: string;
    term?: Readonly<{ word: string; meaning: string }>;
    points: readonly string[];
    caption: string;
    /**
     * Embedded diagram from `render: @scene { ... }`.
     * Package form is parsed immediately; native cinematic form is captured as
     * source for the shared scene parser (`rect` / `connector` / `timeline`).
     */
    sceneIr?: unknown;
  }>;

export type ExplainerHubAst = Readonly<{
  title: string;
  highlight: string;
  description: string;
}>;

export type ExplainerAst = AstNode<"explainer"> &
  Readonly<{
    id: string;
    metadata: Readonly<{
      route: string;
      topic: string;
      storagePrefix: string;
      classPrefix: string;
      eyebrowLabel: string;
      startGateTitle: string;
      hub: ExplainerHubAst;
      css?: string | undefined;
    }>;
    slides: readonly SlideAst[];
    /**
     * Optional end-card diagram from `finalCard: @scene { ... }`.
     * Same embedded-scene dialects as slide `render` (`package` / `native`).
     */
    finalCard?: unknown;
  }>;

export type DocumentAst = AstNode<"document"> &
  Readonly<{
    title: string;
    id: string;
    language: LanguageDeclarationAst;
    imports?: readonly ImportDeclarationAst[];
    requirements: readonly RequirementAst[];
    tokens: readonly TokenDeclarationAst[];
    themes: readonly ThemeDeclarationAst[];
    useTheme?: UseThemeAst;
    symbols: readonly SymbolDefinitionAst[];
    scenes: readonly SceneAst[];
    explainers?: readonly ExplainerAst[];
  }>;
