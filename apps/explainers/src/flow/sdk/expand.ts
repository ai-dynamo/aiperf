/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SDK expansion engine (transport-neutral core).
//!
//! This module owns the deterministic, browser-safe core of SDK component
//! expansion:
//!
//! - `expandSdkInvocation` runs strict prop validation against a component's
//!   declared descriptor, then invokes its pure factory to produce a
//!   `SceneFragment` (roots + ports + semantic action bindings + provenance).
//! - The instance / port / action index (`SdkInstanceIndex` / `SdkActionIndex`)
//!   records every expanded instance so semantic `ref()` endpoints and
//!   component-instance timeline targets can be resolved after all factories
//!   have run.
//! - `resolveFragmentRefs` rewrites pending semantic-reference endpoints
//!   (`sdk-ref::instance.port`) into concrete `ConnectorEndpointIr` values,
//!   failing closed with source-oriented diagnostics for missing components,
//!   missing ports, and ambiguous references.
//!
//! The AST-facing adapter (native `@scene` walking, slot / bounded-`for`
//! resolution, and DeckPackage lowering) lives in
//! `compiler/expand-sdk.ts`; this module never touches the parser, DOM,
//! React, filesystem, network, or wall clock so factories remain pure.

import {
  validateProps,
  type ComponentPropsSchema,
  type PropDescriptor,
  type PropValueKind,
} from "../compiler/components.js";
import type { ComponentDescriptor } from "../schema/component-descriptor.js";
import {
  diagnostic,
  hasErrors,
  type ConnectorEndpointIr,
  type Diagnostic,
  type JsonValue,
  type PointIr,
  type RenderNodeIr,
  type Result,
  type SourceRange,
} from "../schema/index.js";

import type {
  SceneFragment,
  SdkActionName,
  SdkComponentDefinition,
  SdkExpansionContext,
} from "./types.js";

/**
 * Sentinel prefix marking a `ConnectorEndpointIr.nodeId` as an unresolved
 * semantic `ref("instance.port")`.
 *
 * SDK topology / motion factories encode `ref(...)` endpoints with this prefix
 * (see `sdk/generic/topology.ts`); it is redefined here so ref resolution has
 * no import dependency on any individual factory pack.
 */
export const SDK_PENDING_REF_PREFIX = "sdk-ref::";

/** One expanded SDK component instance's public surface, keyed by instance id. */
export type SdkInstanceEntry = Readonly<{
  instanceId: string;
  componentId: string;
  /** Named semantic ports (`input`, `output`, `worker[0]`, ...). */
  ports: Readonly<Record<string, ConnectorEndpointIr>>;
  /** Public timeline actions mapped to generated node ids. */
  actions: Readonly<Partial<Record<SdkActionName, readonly string[]>>>;
  /** Top-level generated root node ids for this instance. */
  rootIds: readonly string[];
  sourceMap: SourceRange;
}>;

/** Instance index: instance id → its ports, actions, and root ids. */
export type SdkInstanceIndex = ReadonlyMap<string, SdkInstanceEntry>;

/** Action index: instance id → public action → bound generated node ids. */
export type SdkActionIndex = ReadonlyMap<
  string,
  ReadonlyMap<SdkActionName, readonly string[]>
>;

/**
 * Maps a descriptor prop `type` onto the strict runtime kind understood by
 * `validateProps`. Basic scalar kinds validate their JSON value directly;
 * every richer descriptor type (`endpoint`, `object`, `array`, `string[]`,
 * unions, ...) is treated as `json` so `validateProps` still rejects unknown
 * and missing-required props without false-positiving on structured values
 * that only the factory can fully narrow.
 */
function normalizePropKind(type: string): PropValueKind {
  switch (type) {
    case "string":
    case "number":
    case "boolean":
      return type;
    default:
      return "json";
  }
}

/** Projects a component descriptor into a strict `validateProps` schema. */
export function propsSchemaFromDescriptor(
  descriptor: ComponentDescriptor,
): ComponentPropsSchema {
  const props: Record<string, PropDescriptor> = {};
  for (const [name, prop] of Object.entries(descriptor.props)) {
    props[name] = { kind: normalizePropKind(prop.type), required: prop.required };
  }
  return { id: descriptor.id, props };
}

/**
 * Expands a single SDK component invocation.
 *
 * Validates the authored props against the component's declared prop schema,
 * then invokes its deterministic factory. Factories seed every generated node
 * id from `context.instanceId` (`${instanceId}` for the fragment root,
 * `${instanceId}__role` for generated children), so repeated calls with the
 * same instance id and props are byte-stable. Slot fragments are merged by the
 * factory itself; this function passes them through unchanged.
 */
export function expandSdkInvocation(
  definition: SdkComponentDefinition,
  props: Readonly<Record<string, JsonValue>>,
  slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
): Result<SceneFragment> {
  const validated = validateProps(
    props,
    propsSchemaFromDescriptor(definition.descriptor),
    context.sourceMap,
  );
  if (!validated.ok) {
    return { ok: false, diagnostics: validated.diagnostics };
  }

  let result: Result<SceneFragment>;
  try {
    result = definition.factory(props, slots, context);
  } catch (error) {
    const cause = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "SDK_FACTORY_FAILED",
          "error",
          `SDK component "${definition.descriptor.id}" instance "${context.instanceId}" failed during expansion: ${cause}.`,
          context.sourceMap,
        ),
      ],
    };
  }

  if (!result.ok) {
    return result;
  }
  if (result.value.roots.length === 0) {
    return {
      ok: false,
      diagnostics: [
        ...result.diagnostics,
        diagnostic(
          "SDK_FACTORY_EMPTY",
          "error",
          `SDK component "${definition.descriptor.id}" instance "${context.instanceId}" produced no roots.`,
          context.sourceMap,
          "A factory must return at least one root node.",
        ),
      ],
    };
  }

  return {
    ok: true,
    value: result.value,
    diagnostics: [...validated.diagnostics, ...result.diagnostics],
  };
}

/** Builds an instance-index entry from an expanded fragment. */
export function instanceEntryFromFragment(
  instanceId: string,
  componentId: string,
  fragment: SceneFragment,
  sourceMap: SourceRange,
): SdkInstanceEntry {
  return {
    instanceId,
    componentId,
    ports: fragment.ports,
    actions: fragment.actions,
    rootIds: fragment.roots.map((root) => root.id),
    sourceMap,
  };
}

/** Derives a flat action index (instance → action → node ids) from instances. */
export function buildActionIndex(index: SdkInstanceIndex): SdkActionIndex {
  const out = new Map<string, ReadonlyMap<SdkActionName, readonly string[]>>();
  for (const [instanceId, entry] of index) {
    const actions = new Map<SdkActionName, readonly string[]>();
    for (const [action, targets] of Object.entries(entry.actions)) {
      if (targets !== undefined) {
        actions.set(action as SdkActionName, targets);
      }
    }
    out.set(instanceId, actions);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Semantic reference resolution.
// ---------------------------------------------------------------------------

/** Returns the `instance.port` target when an endpoint carries a pending ref. */
function pendingRefTarget(endpoint: ConnectorEndpointIr): string | undefined {
  if (
    typeof endpoint.nodeId === "string" &&
    endpoint.nodeId.startsWith(SDK_PENDING_REF_PREFIX)
  ) {
    return endpoint.nodeId.slice(SDK_PENDING_REF_PREFIX.length);
  }
  return undefined;
}

/**
 * Normalizes dotted indexed port families (`worker.0`) into the bracket form
 * that factories publish (`worker[0]`), so authored `ref("cells.worker.0")`
 * resolves against `ports["worker[0]"]`.
 */
function bracketPortName(port: string): string {
  return port.replace(/\.(\d+)/g, "[$1]");
}

function lookupPort(
  entry: SdkInstanceEntry,
  port: string,
): ConnectorEndpointIr | undefined {
  return entry.ports[port] ?? entry.ports[bracketPortName(port)];
}

function availablePorts(entry: SdkInstanceEntry): string {
  const names = Object.keys(entry.ports);
  return names.length > 0 ? names.join(", ") : "(none)";
}

/**
 * Splits `instance.port` preferring the longest registered instance id that is
 * a proper prefix of `target`. This supports authored ids that contain dots
 * (e.g. `aiperf.controller.output`) without breaking indexed ports
 * (`cells.worker.0` → instance `cells`, port `worker.0`).
 */
function splitInstancePort(
  target: string,
  index: SdkInstanceIndex,
): { instanceId: string; port: string } | undefined {
  let best: { instanceId: string; port: string } | undefined;
  for (const instanceId of index.keys()) {
    const prefix = `${instanceId}.`;
    if (!target.startsWith(prefix) || target.length <= prefix.length) {
      continue;
    }
    if (best === undefined || instanceId.length > best.instanceId.length) {
      best = { instanceId, port: target.slice(prefix.length) };
    }
  }
  if (best !== undefined) {
    return best;
  }
  const separator = target.indexOf(".");
  if (separator <= 0 || separator >= target.length - 1) {
    return undefined;
  }
  return {
    instanceId: target.slice(0, separator),
    port: target.slice(separator + 1),
  };
}

/** Resolves a single `instance.port` reference against the instance index. */
export function resolveRef(
  target: string,
  index: SdkInstanceIndex,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): ConnectorEndpointIr | undefined {
  const split = splitInstancePort(target, index);
  if (split === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_REF_AMBIGUOUS",
        "error",
        `Semantic reference "ref(${target})" must be of the form "instance.port".`,
        sourceMap,
        'Author a reference such as ref("controller.output").',
      ),
    );
    return undefined;
  }
  const { instanceId, port } = split;
  const entry = index.get(instanceId);
  if (entry === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_REF_UNKNOWN_INSTANCE",
        "error",
        `Semantic reference "ref(${target})" names unknown component instance "${instanceId}".`,
        sourceMap,
        `Available instances: ${[...index.keys()].join(", ") || "(none)"}.`,
      ),
    );
    return undefined;
  }
  const endpoint = lookupPort(entry, port);
  if (endpoint === undefined) {
    diagnostics.push(
      diagnostic(
        "SDK_REF_UNKNOWN_PORT",
        "error",
        `Component instance "${instanceId}" (${entry.componentId}) has no port "${port}".`,
        sourceMap,
        `Available ports: ${availablePorts(entry)}.`,
      ),
    );
    return undefined;
  }
  return endpoint;
}

function resolveEndpoint(
  endpoint: ConnectorEndpointIr,
  index: SdkInstanceIndex,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): ConnectorEndpointIr {
  const target = pendingRefTarget(endpoint);
  if (target === undefined) {
    return endpoint;
  }
  const resolved = resolveRef(target, index, sourceMap, diagnostics);
  // On failure a diagnostic has already been recorded; keep the sentinel so
  // downstream schema validation still sees a structurally valid endpoint.
  return resolved ?? endpoint;
}

function resolveFanEndpoints(
  endpoints: ConnectorEndpointIr | readonly ConnectorEndpointIr[],
  index: SdkInstanceIndex,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): ConnectorEndpointIr | readonly ConnectorEndpointIr[] {
  if (Array.isArray(endpoints)) {
    return endpoints.map((endpoint) =>
      resolveEndpoint(endpoint, index, sourceMap, diagnostics),
    );
  }
  return resolveEndpoint(
    endpoints as ConnectorEndpointIr,
    index,
    sourceMap,
    diagnostics,
  );
}

function isConnectorEndpoint(
  point: PointIr | ConnectorEndpointIr,
): point is ConnectorEndpointIr {
  return "nodeId" in point || "anchor" in point;
}

function resolvePolylinePoint(
  point: PointIr | ConnectorEndpointIr,
  index: SdkInstanceIndex,
  sourceMap: SourceRange,
  diagnostics: Diagnostic[],
): PointIr | ConnectorEndpointIr {
  return isConnectorEndpoint(point)
    ? resolveEndpoint(point, index, sourceMap, diagnostics)
    : point;
}

/** Rewrites every pending semantic-ref endpoint in a node subtree. */
export function resolveNodeRefs(
  node: RenderNodeIr,
  index: SdkInstanceIndex,
  diagnostics: Diagnostic[],
): RenderNodeIr {
  const sourceMap = node.sourceMap;
  const points =
    node.points === undefined
      ? undefined
      : node.points.map((point) =>
          resolvePolylinePoint(point, index, sourceMap, diagnostics),
        );

  if (node.kind === "connector") {
    return {
      ...node,
      ...(node.from !== undefined
        ? { from: resolveEndpoint(node.from, index, sourceMap, diagnostics) }
        : {}),
      ...(node.to !== undefined
        ? { to: resolveEndpoint(node.to, index, sourceMap, diagnostics) }
        : {}),
      ...(points !== undefined ? { points } : {}),
    };
  }
  if (node.kind === "fan") {
    const from = resolveFanEndpoints(
      node.from,
      index,
      sourceMap,
      diagnostics,
    );
    const to = resolveFanEndpoints(node.to, index, sourceMap, diagnostics);
    return {
      ...node,
      from,
      to,
      ...(points !== undefined ? { points } : {}),
    };
  }
  if (node.kind === "group" || node.kind === "component") {
    return {
      ...node,
      children: node.children.map((child) =>
        resolveNodeRefs(child, index, diagnostics),
      ),
      ...(points !== undefined ? { points } : {}),
    };
  }
  return points !== undefined ? { ...node, points } : node;
}

/**
 * Resolves every pending semantic reference across a scene's expanded roots.
 *
 * Returns the rewritten roots and any diagnostics for unresolved references;
 * callers fail the compile when `hasErrors` is true.
 */
export function resolveFragmentRefs(
  roots: readonly RenderNodeIr[],
  index: SdkInstanceIndex,
): Result<readonly RenderNodeIr[]> {
  const diagnostics: Diagnostic[] = [];
  const resolved = roots.map((root) =>
    resolveNodeRefs(root, index, diagnostics),
  );
  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }
  return { ok: true, value: resolved, diagnostics };
}
