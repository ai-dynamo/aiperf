/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler adapter that drives SDK expansion over a native embedded `@scene`.
//!
//! This bridges the language AST (native component invocations, slots, bounded
//! `for`, semantic `ref()`, and `freeform` blocks) to the transport-neutral SDK
//! expansion engine in `sdk/expand.ts`. It runs after symbol expansion and
//! before scene lowering:
//!
//! ```text
//! parse → expandSymbolInvocations → expandSdkInvocations → (SceneRender)
//! ```
//!
//! Scenes with no `sdk.*` / `aiperf.*` invocation report `status: "not-sdk"`,
//! so the caller keeps using the existing package-form / native lowering path
//! unchanged (deck package-form compatibility is preserved until migration).
//! Scenes that do invoke SDK components expand every invocation to ordinary
//! Scene IR, resolve semantic port references, and emit a validated
//! `SceneRender` alongside the instance / action index (consumed by semantic
//! timeline expansion and reference resolution in later tasks).

import {
  parseNativeEmbeddedScene,
  type ArgumentValueAst,
  type ComponentInvocationAst,
  type DocumentAst,
  type ForLoopAst,
  type FreeformBlockAst,
  type PropAssignmentAst,
  type RectAst,
  type ConnectorAst,
  type RenderDeclarationAst,
  type SceneAst,
  type ScenePrimitiveAst,
  type SlotBlockAst,
  type SymbolBodyStatementAst,
  type TimelineAst,
  type ValueAst,
} from "../language/index.js";
import {
  diagnostic,
  hasErrors,
  sceneIrSchema,
  type Diagnostic,
  type JsonValue,
  type RenderNodeIr,
  type Result,
  type SceneIr,
  type SceneRender,
  type SourceRange,
  type TimelineCueIr,
} from "../schema/index.js";
import type {
  SceneFragment,
  SdkActionName,
  SdkRegistry,
} from "../sdk/index.js";
import { canonicalSdkComponentId } from "../sdk/registry.js";
import {
  buildActionIndex,
  expandSdkInvocation,
  instanceEntryFromFragment,
  resolveFragmentRefs,
  type SdkActionIndex,
  type SdkInstanceEntry,
  type SdkInstanceIndex,
} from "../sdk/expand.js";

import {
  asRecord,
  capabilityKind,
  desugarPackageNode,
  lowerFirstClassPackageNode,
} from "./desugar-scene-primitives.js";
import { expandSymbolInvocations } from "./expand-symbols.js";
import { collectSymbols } from "./symbols.js";
import {
  findUnresolvedAfterRefs,
  resolveTimelineCueTiming,
} from "./timeline-timing.js";

const MAX_FOR_ITERATIONS = 1024;

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Fill-in scene identity used when the native `@scene` omits header fields. */
export type SdkSceneDefaults = Readonly<{
  id?: string;
  title?: string;
  summary?: string;
  narration?: string;
  fallback?: string;
}>;

/** Options controlling native `@scene` SDK expansion. */
export type ExpandSdkOptions = Readonly<{
  registry: SdkRegistry;
  /** Document token bindings used to resolve `@token` prop references. */
  tokens?: ReadonlyMap<string, JsonValue>;
  defaults?: SdkSceneDefaults;
  slideId?: string;
  /** Owning `.flow` source range for diagnostics and generated sourceMaps. */
  sourceRange?: SourceRange;
}>;

/** A fully expanded SDK scene plus its instance / action index. */
export type SdkExpandedScene = Readonly<{
  render: SceneRender;
  instanceIndex: SdkInstanceIndex;
  actionIndex: SdkActionIndex;
}>;

/**
 * Result of attempting SDK expansion on one embedded `@scene`.
 *
 * - `not-sdk`: the scene contains no SDK invocation; the caller should fall
 *   back to the existing package / native lowering path.
 * - `ok`: the scene expanded into a validated `SceneRender`.
 * - `error`: the scene invoked SDK components but expansion failed.
 */
export type SdkExpansionOutcome =
  | Readonly<{ status: "not-sdk" }>
  | Readonly<{ status: "ok"; value: SdkExpandedScene; diagnostics: readonly Diagnostic[] }>
  | Readonly<{ status: "error"; diagnostics: readonly Diagnostic[] }>;

type ExpansionState = {
  readonly registry: SdkRegistry;
  readonly tokens: ReadonlyMap<string, JsonValue>;
  readonly themeTokens: ReadonlyMap<string, JsonValue>;
  readonly sourceRange: SourceRange;
  readonly index: Map<string, SdkInstanceEntry>;
  readonly diagnostics: Diagnostic[];
  autoId: number;
};

type Bindings = ReadonlyMap<string, JsonValue>;

const EMPTY_BINDINGS: Bindings = new Map();

type InvocationResult =
  | Readonly<{ kind: "sdk"; fragment: SceneFragment }>
  | Readonly<{ kind: "not-sdk" }>
  | Readonly<{ kind: "error" }>;

// ---------------------------------------------------------------------------
// Scene-form recognition (mirrors lower-explainer-scene, kept local so this
// adapter never depends on that module's private guards).
// ---------------------------------------------------------------------------

function isSceneAst(value: unknown): value is SceneAst {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    (value as { kind?: unknown }).kind === "scene" &&
    "renderDeclarations" in value &&
    Array.isArray((value as { renderDeclarations?: unknown }).renderDeclarations)
  );
}

function isNativeEmbeddedSource(
  value: unknown,
): value is Readonly<{ kind: "embedded-scene-source"; form: string; body: string }> {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    (value as { kind?: unknown }).kind === "embedded-scene-source" &&
    "form" in value &&
    "body" in value &&
    typeof (value as { body?: unknown }).body === "string"
  );
}

function componentIdOf(invocation: ComponentInvocationAst): string {
  const raw =
    invocation.namespace !== undefined
      ? `${invocation.namespace}.${invocation.name}`
      : invocation.name;
  return canonicalSdkComponentId(raw);
}

// ---------------------------------------------------------------------------
// Pre-scan: does the scene invoke any registered SDK component at all?
// ---------------------------------------------------------------------------

function statementInvokesSdk(
  statement: SymbolBodyStatementAst,
  registry: SdkRegistry,
): boolean {
  if (statement.kind === "component-invocation") {
    if (registry.lookup(componentIdOf(statement)) !== undefined) {
      return true;
    }
    return (statement.slots ?? []).some((slot) =>
      slot.body.some((entry) => statementInvokesSdk(entry, registry)),
    );
  }
  if (statement.kind === "for-loop") {
    return statement.body.some((entry) => statementInvokesSdk(entry, registry));
  }
  if (statement.kind === "slot") {
    return statement.body.some((entry) => statementInvokesSdk(entry, registry));
  }
  return false;
}

function sceneInvokesSdk(scene: SceneAst, registry: SdkRegistry): boolean {
  for (const declaration of scene.renderDeclarations) {
    if (
      declaration.kind === "component-invocation" &&
      statementInvokesSdk(declaration, registry)
    ) {
      return true;
    }
  }
  return (scene.loops ?? []).some((loop) => statementInvokesSdk(loop, registry));
}

// ---------------------------------------------------------------------------
// Value resolution: AST argument → JsonValue.
// ---------------------------------------------------------------------------

function valueAstToJson(
  value: ValueAst,
  tokens: ReadonlyMap<string, JsonValue>,
): JsonValue {
  if (value.kind === "literal") {
    return value.value;
  }
  if (value.kind === "token-reference") {
    const resolved = tokens.get(value.token);
    return resolved !== undefined ? resolved : `@${value.token}`;
  }
  // theme-role-reference: pass the bare role name through; chrome factories
  // accept `@theme.*` role strings while domain factories match bare roles.
  return value.role;
}

function resolveArgumentValue(
  value: ArgumentValueAst,
  bindings: Bindings,
  state: ExpansionState,
): JsonValue {
  switch (value.kind) {
    case "literal":
    case "token-reference":
    case "theme-role-reference":
      return valueAstToJson(value, state.tokens);
    case "identifier-reference": {
      const bound = bindings.get(value.name);
      if (bound !== undefined) {
        return bound;
      }
      state.diagnostics.push(
        diagnostic(
          "SDK_UNKNOWN_BINDING",
          "error",
          `Unknown reference "${value.name}" in SDK component argument.`,
          value.sourceMap,
          "Bind the name via an enclosing `for` loop, or use a literal value.",
        ),
      );
      return null;
    }
    case "array-literal":
      return value.items.map((item) =>
        resolveArgumentValue(item, bindings, state),
      );
    case "ref":
      // Encode as the `{ ref: "instance.port" }` shape SDK endpoint factories
      // recognize; the engine resolves it after all instances register.
      return { ref: value.target };
    case "object-literal": {
      const object: Record<string, JsonValue> = {};
      for (const property of value.properties) {
        object[property.name] = resolveArgumentValue(
          property.value,
          bindings,
          state,
        );
      }
      return object;
    }
  }
}

function resolveProps(
  props: readonly PropAssignmentAst[],
  bindings: Bindings,
  state: ExpansionState,
): Record<string, JsonValue> {
  const record: Record<string, JsonValue> = {};
  for (const prop of props) {
    record[prop.name] = resolveArgumentValue(prop.value, bindings, state);
  }
  return record;
}

// ---------------------------------------------------------------------------
// Invocation / slot / loop expansion.
// ---------------------------------------------------------------------------

function expandComponentInvocation(
  invocation: ComponentInvocationAst,
  bindings: Bindings,
  state: ExpansionState,
): InvocationResult {
  const componentId = componentIdOf(invocation);
  const definition = state.registry.lookup(componentId);
  if (definition === undefined) {
    return { kind: "not-sdk" };
  }

  const props = resolveProps(invocation.props, bindings, state);
  const authoredId = props.id;
  const instanceId =
    typeof authoredId === "string" && authoredId.length > 0
      ? authoredId
      : `${componentId}#${state.autoId++}`;

  if (state.index.has(instanceId)) {
    state.diagnostics.push(
      diagnostic(
        "SDK_DUPLICATE_INSTANCE",
        "error",
        `Duplicate SDK component instance id "${instanceId}".`,
        invocation.sourceMap,
        "Give each SDK component invocation a unique id.",
      ),
    );
    return { kind: "error" };
  }

  const slots = resolveSlots(invocation.slots ?? [], bindings, state);
  const context = {
    instanceId,
    sourceMap: invocation.sourceMap ?? state.sourceRange,
    themeTokens: state.themeTokens,
  };

  const result = expandSdkInvocation(definition, props, slots, context);
  state.diagnostics.push(...result.diagnostics);
  if (!result.ok) {
    return { kind: "error" };
  }

  state.index.set(
    instanceId,
    instanceEntryFromFragment(instanceId, componentId, result.value, context.sourceMap),
  );
  return { kind: "sdk", fragment: result.value };
}

function resolveSlots(
  slots: readonly SlotBlockAst[],
  bindings: Bindings,
  state: ExpansionState,
): Record<string, readonly SceneFragment[]> {
  const record: Record<string, SceneFragment[]> = {};
  for (const slot of slots) {
    record[slot.name] = expandStatements(slot.body, bindings, state);
  }
  return record;
}

function expandStatements(
  statements: readonly SymbolBodyStatementAst[],
  bindings: Bindings,
  state: ExpansionState,
): SceneFragment[] {
  const fragments: SceneFragment[] = [];
  for (const statement of statements) {
    if (statement.kind === "component-invocation") {
      const result = expandComponentInvocation(statement, bindings, state);
      if (result.kind === "sdk") {
        fragments.push(result.fragment);
      } else if (result.kind === "not-sdk") {
        state.diagnostics.push(
          diagnostic(
            "SDK_SLOT_REQUIRES_COMPONENT",
            "error",
            `Slot / loop body entry "${componentIdOf(statement)}" is not a registered SDK component.`,
            statement.sourceMap,
            "Slot and loop bodies accept sdk.* / aiperf.* component invocations only.",
          ),
        );
      }
      continue;
    }
    if (statement.kind === "for-loop") {
      fragments.push(...expandForLoop(statement, bindings, state));
      continue;
    }
    // Nested slot blocks are not a valid slot / loop body element.
    state.diagnostics.push(
      diagnostic(
        "SDK_NESTED_SLOT_UNSUPPORTED",
        "error",
        `Nested slot "${statement.name}" is not allowed inside a slot or loop body.`,
        statement.sourceMap,
        "Author nested structure with component invocations, not bare slots.",
      ),
    );
  }
  return fragments;
}

function resolveCollection(
  loop: ForLoopAst,
  bindings: Bindings,
  state: ExpansionState,
): readonly JsonValue[] | undefined {
  if (loop.collection.kind === "array-literal") {
    return loop.collection.items.map((item) =>
      resolveArgumentValue(item, bindings, state),
    );
  }
  const bound = bindings.get(loop.collection.name);
  if (Array.isArray(bound)) {
    return bound;
  }
  state.diagnostics.push(
    diagnostic(
      "SDK_FOR_COLLECTION_INVALID",
      "error",
      `\`for\` collection "${loop.collection.name}" is not a finite authored array.`,
      loop.sourceMap,
      "Iterate an inline array literal or an array bound by an enclosing loop.",
    ),
  );
  return undefined;
}

function expandForLoop(
  loop: ForLoopAst,
  bindings: Bindings,
  state: ExpansionState,
): SceneFragment[] {
  const items = resolveCollection(loop, bindings, state);
  if (items === undefined) {
    return [];
  }
  if (items.length > MAX_FOR_ITERATIONS) {
    state.diagnostics.push(
      diagnostic(
        "SDK_FOR_LIMIT_EXCEEDED",
        "error",
        `\`for\` loop over ${items.length} items exceeds the maximum of ${MAX_FOR_ITERATIONS}.`,
        loop.sourceMap,
      ),
    );
    return [];
  }

  const fragments: SceneFragment[] = [];
  for (const element of items) {
    const childBindings = new Map(bindings);
    childBindings.set(loop.item, element);
    fragments.push(...expandStatements(loop.body, childBindings, state));
  }
  return fragments;
}

// ---------------------------------------------------------------------------
// Freeform / non-SDK declaration lowering (raw primitives pass through as
// ordinary Scene IR, mirroring the package-form node normalizer).
// ---------------------------------------------------------------------------

function valueAstToStyleScalar(value: ValueAst): string | number | boolean {
  if (value.kind === "literal") {
    return value.value;
  }
  if (value.kind === "token-reference") {
    return `@${value.token}`;
  }
  return value.role;
}

function rectToRecord(node: RectAst): Record<string, unknown> {
  const style: Record<string, unknown> = { fill: valueAstToStyleScalar(node.fill) };
  if (node.stroke !== undefined) {
    style.stroke = valueAstToStyleScalar(node.stroke);
  }
  return {
    id: node.id,
    capability: "core.rect",
    layout: { x: node.x, y: node.y, width: node.width, height: node.height },
    style,
    text: node.label,
    accessibility: {
      label: node.label,
      ...(node.description.trim().length > 0 ? { description: node.description } : {}),
    },
    fallback: node.fallback.text,
  };
}

function connectorToRecord(node: ConnectorAst): Record<string, unknown> {
  return {
    id: node.id,
    capability: "core.connector",
    from: { nodeId: node.from },
    to: { nodeId: node.to },
    style: { stroke: valueAstToStyleScalar(node.stroke) },
    accessibility: { label: node.label },
    fallback: node.fallback.text,
  };
}

function scenePrimitiveToRecord(
  node: ScenePrimitiveAst,
  bindings: Bindings,
  state: ExpansionState,
): Record<string, unknown> {
  return {
    id: node.id,
    capability: node.capability,
    ...resolveProps(node.props, bindings, state),
    ...(node.children !== undefined
      ? { children: node.children.map((child) => declarationToRecord(child, bindings, state)) }
      : {}),
    ...(node.fallback !== undefined ? { fallback: node.fallback.text } : {}),
  };
}

function invocationToOpaqueRecord(
  node: ComponentInvocationAst,
  bindings: Bindings,
  state: ExpansionState,
): Record<string, unknown> {
  const props = resolveProps(node.props, bindings, state);
  const id = typeof props.id === "string" && props.id.length > 0 ? props.id : node.name;
  return {
    id,
    capability: componentIdOf(node),
    ...props,
  };
}

function declarationToRecord(
  declaration: RenderDeclarationAst,
  bindings: Bindings,
  state: ExpansionState,
): Record<string, unknown> {
  switch (declaration.kind) {
    case "rect":
      return rectToRecord(declaration);
    case "connector":
      return connectorToRecord(declaration);
    case "scene-primitive":
      return scenePrimitiveToRecord(declaration, bindings, state);
    case "component-invocation":
      return invocationToOpaqueRecord(declaration, bindings, state);
  }
}

function normalizePackageRecord(value: unknown): RenderNodeIr {
  const node = asRecord(value);
  if (node === undefined) {
    throw new Error("freeform render node must be an object");
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
    ? node.children.map(normalizePackageRecord)
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
  const fallback = typeof node.fallback === "string" ? node.fallback : label;

  const common = {
    id,
    capability,
    children,
    label,
    fallback,
    ...(description !== undefined ? { description } : {}),
  };
  const desugared = desugarPackageNode(node, common);
  if (desugared !== undefined) {
    return desugared;
  }
  return lowerFirstClassPackageNode(node, {
    ...common,
    kind: capabilityKind(capability),
  });
}

function lowerFreeformDeclaration(
  declaration: RenderDeclarationAst,
  bindings: Bindings,
  state: ExpansionState,
): RenderNodeIr {
  return normalizePackageRecord(declarationToRecord(declaration, bindings, state));
}

function lowerFreeformBlock(
  block: FreeformBlockAst,
  state: ExpansionState,
): readonly RenderNodeIr[] {
  const nodes = block.body.map((declaration) =>
    lowerFreeformDeclaration(declaration, EMPTY_BINDINGS, state),
  );
  if (block.id === undefined) {
    return nodes;
  }
  const group: RenderNodeIr = {
    kind: "group",
    id: block.id,
    capabilityId: "core.group",
    geometry: { x: 0, y: 0, width: 0, height: 0 },
    style: {},
    accessibility: { label: block.id },
    fallback: block.id,
    sourceMap: state.sourceRange,
    children: nodes,
  };
  return [group];
}

// ---------------------------------------------------------------------------
// Timeline: authored cues → SceneIr cues, expanding component-instance action
// targets through the action index where they resolve.
// ---------------------------------------------------------------------------

/**
 * Maps authored timeline action words onto the SDK's public action vocabulary.
 *
 * Native cinematic cues author `reveal` / `trace` / `fade` / `exit`; package
 * decks additionally author `enter` / `draw` / `emphasis` / `pulse`. Both
 * dialects normalize onto the `SdkActionName` a component instance publishes
 * so an authored cue can be expanded through the instance's action bindings.
 */
const SDK_TIMELINE_ACTION_ALIASES: Readonly<Record<string, SdkActionName>> = {
  reveal: "enter",
  enter: "enter",
  draw: "draw",
  trace: "trace",
  emphasis: "emphasis",
  emphasize: "emphasis",
  pulse: "pulse",
  stagger: "stagger",
  exit: "exit",
  fade: "fade",
};

/** Default inter-step delay for the `standardReveal` timeline template. */
const DEFAULT_REVEAL_STEP = 240;
/** Default per-cue duration for the `standardReveal` timeline template. */
const DEFAULT_REVEAL_DURATION = 480;

/** Collects every generated / freeform node id in a scene's root subtree. */
function collectNodeIds(
  roots: readonly RenderNodeIr[],
  into: Set<string>,
): Set<string> {
  for (const node of roots) {
    into.add(node.id);
    if (node.kind === "group" || node.kind === "component") {
      collectNodeIds(node.children, into);
    }
  }
  return into;
}

/** Human-readable hint listing the actions an instance publishes. */
function supportedActionsHint(
  instanceActions: ReadonlyMap<SdkActionName, readonly string[]>,
): string {
  const supported = [...instanceActions.keys()];
  return supported.length > 0
    ? `Supported actions for this component instance: ${supported.join(", ")}.`
    : "This component instance exposes no timeline actions.";
}

/** Human-readable hint listing the component instances available for targeting. */
function availableInstancesHint(actionIndex: SdkActionIndex): string {
  const instances = [...actionIndex.keys()];
  return instances.length > 0
    ? `Target a component instance (${instances.join(", ")}) or an existing node id.`
    : "Target an existing scene node id.";
}

/**
 * Fans one authored instance/action cue out to concrete generated node ids.
 *
 * Emits one internal cue per bound node id (stable, index-suffixed ids when an
 * action binds multiple targets). Fails closed — recording a diagnostic and
 * emitting nothing — when the instance is unknown or the action is not one the
 * instance publishes, so refactors that change generated ids never silently
 * drop authored motion.
 */
function pushInstanceActionCues(
  cues: TimelineCueIr[],
  diagnostics: Diagnostic[],
  params: Readonly<{
    instanceId: string;
    action: SdkActionName;
    at: number;
    duration: number;
    baseId: string;
    easing?: TimelineCueIr["easing"];
    actionIndex: SdkActionIndex;
    sourceMap: SourceRange;
  }>,
): void {
  const instanceActions = params.actionIndex.get(params.instanceId);
  if (instanceActions === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_TIMELINE_UNKNOWN_TARGET",
        "error",
        `Timeline targets unknown SDK component instance "${params.instanceId}".`,
        params.sourceMap,
        availableInstancesHint(params.actionIndex),
      ),
    );
    return;
  }
  const bound = instanceActions.get(params.action);
  if (bound === undefined || bound.length === 0) {
    diagnostics.push(
      diagnostic(
        "SDK_TIMELINE_UNSUPPORTED_ACTION",
        "error",
        `SDK component instance "${params.instanceId}" does not support timeline action "${params.action}".`,
        params.sourceMap,
        supportedActionsHint(instanceActions),
      ),
    );
    return;
  }
  bound.forEach((nodeId, boundIndex) => {
    cues.push({
      id: bound.length > 1 ? `${params.baseId}-${boundIndex}` : params.baseId,
      at: params.at,
      duration: params.duration,
      target: nodeId,
      action: params.action,
      ...(params.easing !== undefined ? { easing: params.easing } : {}),
      sourceMap: params.sourceMap,
    });
  });
}

function buildTimeline(
  timelines: readonly TimelineAst[],
  actionIndex: SdkActionIndex,
  nodeIds: ReadonlySet<string>,
  sourceRange: SourceRange,
  diagnostics: Diagnostic[],
): readonly TimelineCueIr[] {
  const cues: TimelineCueIr[] = [];
  for (const timeline of timelines) {
    const resolvedAt = resolveTimelineCueTiming(timeline.cues);
    for (const unresolvedIndex of findUnresolvedAfterRefs(timeline.cues)) {
      const unresolvedCue = timeline.cues[unresolvedIndex]!;
      if (unresolvedCue.timing.mode === "after") {
        diagnostics.push(
          diagnostic(
            "SDK_TIMELINE_UNKNOWN_TARGET",
            "error",
            `Timeline "${timeline.id}" cue "after ${unresolvedCue.timing.ref}" does not match any earlier cue's target in this timeline.`,
            unresolvedCue.sourceMap ?? sourceRange,
            availableInstancesHint(actionIndex),
          ),
        );
      }
    }
    timeline.cues.forEach((cue, index) => {
      const baseId = `${timeline.id}-${index}`;
      const sourceMap = cue.sourceMap ?? sourceRange;
      const at = resolvedAt[index]!;

      // Component-instance target: expand through the instance's public actions.
      if (cue.target.length > 0 && actionIndex.has(cue.target)) {
        const sdkAction = SDK_TIMELINE_ACTION_ALIASES[cue.action];
        if (sdkAction === undefined) {
          diagnostics.push(
            diagnostic(
              "SDK_TIMELINE_UNSUPPORTED_ACTION",
              "error",
              `Timeline cue action "${cue.action}" is not a public SDK action for component instance "${cue.target}".`,
              sourceMap,
              supportedActionsHint(actionIndex.get(cue.target)!),
            ),
          );
          return;
        }
        pushInstanceActionCues(cues, diagnostics, {
          instanceId: cue.target,
          action: sdkAction,
          at,
          duration: cue.duration,
          baseId,
          ...(cue.easing !== undefined ? { easing: cue.easing } : {}),
          actionIndex,
          sourceMap,
        });
        return;
      }

      // A non-stagger target that is neither a component instance nor a real
      // scene node id is a typo or a leaked generated id: fail closed.
      if (cue.target.length > 0 && !nodeIds.has(cue.target)) {
        diagnostics.push(
          diagnostic(
            "SDK_TIMELINE_UNKNOWN_TARGET",
            "error",
            `Timeline cue targets "${cue.target}", which is neither a component instance nor a scene node.`,
            sourceMap,
            availableInstancesHint(actionIndex),
          ),
        );
        return;
      }

      // Freeform / literal node targets (and stagger member lists) pass through.
      cues.push({
        id: baseId,
        at,
        duration: cue.duration,
        target: cue.target,
        action: cue.action,
        ...(cue.targets !== undefined && cue.targets.length > 0
          ? { targets: cue.targets }
          : {}),
        ...(cue.step !== undefined ? { step: cue.step } : {}),
        ...(cue.easing !== undefined ? { easing: cue.easing } : {}),
        sourceMap,
      });
    });
  }
  return cues;
}

/**
 * Semantic inputs for the `standardReveal` timeline template.
 *
 * Names a header, ordered content nodes, edges, and motion overlays by
 * component instance id; the template desugars them into the dominant
 * enter → draw → trace choreography (one call in place of the 8–12 hand cues a
 * scene otherwise repeats) while keeping generated ids private.
 */
export type StandardRevealSpec = Readonly<{
  /** Chrome/header instance revealed first. */
  header?: string;
  /** Content node instances revealed in order after the header. */
  nodes?: readonly string[];
  /** Edge instances drawn once their endpoints have entered. */
  edges?: readonly string[];
  /** Motion overlay instances traced last. */
  motion?: readonly string[];
  /** Base scene time for the first cue (default 0). */
  start?: number;
  /** Delay between successive reveal steps (default 240). */
  step?: number;
  /** Per-cue duration (default 480). */
  duration?: number;
  /** Base id used to seed generated cue ids (default "standard-reveal"). */
  timelineId?: string;
}>;

/**
 * Compiler-side desugar for `sdk.timeline.standardReveal(header, nodes, edges,
 * motion)`.
 *
 * Emits the standard enter → draw → trace cue sequence by referencing each
 * instance's public action bindings, fanning multi-target actions out to stable
 * generated ids. Fails closed with the same diagnostics as authored
 * instance-target cues (unknown instance, unsupported action). The helper is
 * transport-neutral and deterministic; a native authoring surface for it is a
 * grammar concern (see the Task 7 report) and is not wired here.
 */
export function expandStandardReveal(
  spec: StandardRevealSpec,
  actionIndex: SdkActionIndex,
  sourceRange: SourceRange,
): Result<readonly TimelineCueIr[]> {
  const cues: TimelineCueIr[] = [];
  const diagnostics: Diagnostic[] = [];
  const step = spec.step ?? DEFAULT_REVEAL_STEP;
  const duration = spec.duration ?? DEFAULT_REVEAL_DURATION;
  const timelineId = spec.timelineId ?? "standard-reveal";
  let at = spec.start ?? 0;
  let cursor = 0;

  const emit = (instanceId: string, action: SdkActionName): void => {
    pushInstanceActionCues(cues, diagnostics, {
      instanceId,
      action,
      at,
      duration,
      baseId: `${timelineId}-${cursor}`,
      actionIndex,
      sourceMap: sourceRange,
    });
    cursor += 1;
    at += step;
  };

  if (spec.header !== undefined) {
    emit(spec.header, "enter");
  }
  for (const node of spec.nodes ?? []) {
    emit(node, "enter");
  }
  for (const edge of spec.edges ?? []) {
    emit(edge, "draw");
  }
  for (const overlay of spec.motion ?? []) {
    emit(overlay, "trace");
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return { ok: true, value: cues, diagnostics };
}

// ---------------------------------------------------------------------------
// Scene assembly.
// ---------------------------------------------------------------------------

function firstNonEmpty(...values: readonly (string | undefined)[]): string {
  for (const value of values) {
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return "";
}

function assembleScene(
  scene: SceneAst,
  roots: readonly RenderNodeIr[],
  timeline: readonly TimelineCueIr[],
  options: ExpandSdkOptions,
  sourceRange: SourceRange,
): Result<SceneRender> {
  const defaults = options.defaults ?? {};
  const id = firstNonEmpty(scene.id, defaults.id, "embedded");
  const title = firstNonEmpty(scene.title, defaults.title, id, "Embedded scene");
  const summary = firstNonEmpty(scene.summary?.text, defaults.summary, title);
  const narration = scene.narration?.text ?? defaults.narration ?? "";
  const fallback = firstNonEmpty(scene.fallback?.text, defaults.fallback, title);
  const readingOrder =
    scene.readingOrder?.references !== undefined &&
    scene.readingOrder.references.length > 0
      ? scene.readingOrder.references
      : roots.map((root) => root.id);

  const sceneIr: SceneIr = {
    id,
    title,
    summary,
    roots,
    camera: [],
    timeline,
    narration,
    interactions: [],
    responsive: [],
    accessibility: {
      label: firstNonEmpty(scene.title, title),
      readingOrder,
    },
    fallback,
    sourceMap: sourceRange,
  };

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
          sourceRange,
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

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

function wrapSceneDocument(scene: SceneAst, sourceRange: SourceRange): DocumentAst {
  return {
    kind: "document",
    id: `explainer-sdk-scene-${scene.id}`,
    title: scene.title,
    language: { kind: "language", version: 1, sourceMap: sourceRange },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [scene],
    sourceMap: sourceRange,
  };
}

/** Parses / normalizes a raw slide scene value into a native `SceneAst`. */
function toSceneAst(
  rawScene: unknown,
  sourceRange: SourceRange,
): Result<SceneAst> | undefined {
  if (isSceneAst(rawScene)) {
    return { ok: true, value: rawScene, diagnostics: [] };
  }
  if (isNativeEmbeddedSource(rawScene)) {
    if (rawScene.form !== "native") {
      return undefined;
    }
    const sourceName =
      sourceRange.source.length > 0 ? sourceRange.source : "<embedded-scene>";
    const parsed = parseNativeEmbeddedScene(rawScene.body, sourceName);
    if (!parsed.ok) {
      return parsed;
    }
    return { ok: true, value: parsed.value, diagnostics: parsed.diagnostics };
  }
  return undefined;
}

/**
 * Expands SDK component invocations in one embedded `@scene`.
 *
 * Returns `not-sdk` (with the scene left to existing lowering) when the scene
 * invokes no registered SDK component. Otherwise runs symbol expansion, SDK
 * factory expansion, semantic reference resolution, and scene validation,
 * producing a `SceneRender` plus the instance / action index.
 */
export function expandSdkInvocations(
  rawScene: unknown,
  options: ExpandSdkOptions,
): SdkExpansionOutcome {
  const sourceRange = options.sourceRange ?? unknownRange;
  const parsed = toSceneAst(rawScene, sourceRange);
  if (parsed === undefined) {
    return { status: "not-sdk" };
  }
  if (!parsed.ok) {
    // Only surface parse errors when the source clearly targets SDK authoring;
    // otherwise defer to the existing lowering path for a single diagnostic.
    return { status: "not-sdk" };
  }

  if (!sceneInvokesSdk(parsed.value, options.registry)) {
    return { status: "not-sdk" };
  }

  // Run symbol expansion first so legacy macros that emit SDK calls resolve
  // before SDK expansion (parse → symbols → SDK).
  const document = wrapSceneDocument(parsed.value, sourceRange);
  const symbols = collectSymbols(document);
  if (!symbols.ok) {
    return { status: "error", diagnostics: symbols.diagnostics };
  }
  const expandedSymbols = expandSymbolInvocations(document, symbols.value);
  if (!expandedSymbols.ok) {
    return { status: "error", diagnostics: expandedSymbols.diagnostics };
  }
  const scene = expandedSymbols.value.scenes[0];
  if (scene === undefined) {
    return {
      status: "error",
      diagnostics: [
        diagnostic(
          "SDK_SCENE_LOST",
          "error",
          "Embedded @scene was lost during symbol expansion.",
          sourceRange,
        ),
      ],
    };
  }

  const state: ExpansionState = {
    registry: options.registry,
    tokens: options.tokens ?? new Map(),
    themeTokens: options.tokens ?? new Map(),
    sourceRange,
    index: new Map(),
    diagnostics: [],
    autoId: 0,
  };

  const roots: RenderNodeIr[] = [];
  try {
    for (const declaration of scene.renderDeclarations) {
      if (declaration.kind === "component-invocation") {
        const result = expandComponentInvocation(declaration, EMPTY_BINDINGS, state);
        if (result.kind === "sdk") {
          roots.push(...result.fragment.roots);
        } else if (result.kind === "not-sdk") {
          roots.push(lowerFreeformDeclaration(declaration, EMPTY_BINDINGS, state));
        }
        continue;
      }
      roots.push(lowerFreeformDeclaration(declaration, EMPTY_BINDINGS, state));
    }
    for (const loop of scene.loops ?? []) {
      for (const fragment of expandForLoop(loop, EMPTY_BINDINGS, state)) {
        roots.push(...fragment.roots);
      }
    }
    for (const block of scene.freeforms ?? []) {
      roots.push(...lowerFreeformBlock(block, state));
    }
  } catch (error) {
    const cause = error instanceof Error ? error.message : String(error);
    return {
      status: "error",
      diagnostics: [
        ...state.diagnostics,
        diagnostic(
          "SDK_EXPANSION_FAILED",
          "error",
          `SDK scene expansion failed: ${cause}.`,
          sourceRange,
        ),
      ],
    };
  }

  if (hasErrors(state.diagnostics)) {
    return { status: "error", diagnostics: state.diagnostics };
  }

  const instanceIndex: SdkInstanceIndex = state.index;
  const resolved = resolveFragmentRefs(roots, instanceIndex);
  if (!resolved.ok) {
    return {
      status: "error",
      diagnostics: [...state.diagnostics, ...resolved.diagnostics],
    };
  }

  const actionIndex = buildActionIndex(instanceIndex);
  const nodeIds = collectNodeIds(resolved.value, new Set<string>());
  const timeline = buildTimeline(
    scene.timelines,
    actionIndex,
    nodeIds,
    sourceRange,
    state.diagnostics,
  );
  if (hasErrors(state.diagnostics)) {
    return { status: "error", diagnostics: state.diagnostics };
  }
  const render = assembleScene(scene, resolved.value, timeline, options, sourceRange);
  if (!render.ok) {
    return {
      status: "error",
      diagnostics: [...state.diagnostics, ...render.diagnostics],
    };
  }

  return {
    status: "ok",
    value: {
      render: render.value,
      instanceIndex,
      actionIndex,
    },
    diagnostics: [...state.diagnostics, ...render.diagnostics],
  };
}
