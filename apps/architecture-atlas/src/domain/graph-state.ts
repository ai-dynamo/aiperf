// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { compressToEncodedURIComponent, decompressFromEncodedURIComponent } from "lz-string";
import { z } from "zod";

import { audienceSchema, type Audience } from "./audience";
import { architectureIdSchema, executionFlavorSchema, type ExecutionFlavor } from "./architecture";

export const GRAPH_STATE_VERSION = 1 as const;
export const GRAPH_STATE_STORAGE_KEY = "aiperf-atlas:graph-state:v1";
const graphTraceModeSchema = z.enum(["none", "upstream", "downstream", "isolate"]);

const coordinateSchema = z
  .object({
    x: z.number().finite(),
    y: z.number().finite(),
  })
  .strict();

const nodePositionSchema = z
  .object({
    nodeId: architectureIdSchema,
    x: z.number().finite(),
    y: z.number().finite(),
  })
  .strict();

const edgeWaypointSchema = z
  .object({
    edgeId: architectureIdSchema,
    points: z.array(coordinateSchema),
  })
  .strict();

export const graphStateSchema = z
  .object({
    version: z.literal(GRAPH_STATE_VERSION),
    sceneId: architectureIdSchema,
    audience: audienceSchema,
    primaryFlavor: executionFlavorSchema,
    compareFlavor: executionFlavorSchema.nullable(),
    expandedNodeIds: z.array(architectureIdSchema),
    focusedEntityId: architectureIdSchema.nullable(),
    traceMode: graphTraceModeSchema,
    nodePositions: z.array(nodePositionSchema),
    edgeWaypoints: z.array(edgeWaypointSchema),
    timelinePosition: z.number().finite().min(0),
  })
  .strict();

export type GraphState = z.infer<typeof graphStateSchema>;
export type NodePositionOverride = z.infer<typeof nodePositionSchema>;
export type EdgeWaypointOverride = z.infer<typeof edgeWaypointSchema>;

export interface CanonicalGraphStateDomain {
  defaultState: GraphState;
  sceneIds: ReadonlySet<string>;
  nodeIds: ReadonlySet<string>;
  edgeIds: ReadonlySet<string>;
  supportedFlavors: ReadonlySet<ExecutionFlavor>;
}

export interface GraphStateRecoveryNotice {
  code: "invalid_url_state" | "stale_url_state" | "invalid_local_state";
  message: string;
  recoverable: true;
}

export interface GraphStateResolution {
  state: GraphState;
  source: "url" | "local" | "canonical";
  notice?: GraphStateRecoveryNotice;
}

export interface GraphStateStorage {
  getItem(key: string): string | null;
  removeItem(key: string): void;
  setItem(key: string, value: string): void;
}

interface LayoutState {
  nodePositions: readonly NodePositionOverride[];
  edgeWaypoints: readonly EdgeWaypointOverride[];
}

export function canonicalGraphState(input: {
  sceneId: string;
  audience: Audience;
  primaryFlavor: ExecutionFlavor;
  compareFlavor?: ExecutionFlavor | null;
  expandedNodeIds?: readonly string[];
  focusedEntityId?: string | null;
  traceMode?: z.infer<typeof graphTraceModeSchema>;
  nodePositions?: readonly NodePositionOverride[];
  edgeWaypoints?: readonly EdgeWaypointOverride[];
  timelinePosition?: number;
}): GraphState {
  return graphStateSchema.parse({
    version: GRAPH_STATE_VERSION,
    sceneId: input.sceneId,
    audience: input.audience,
    primaryFlavor: input.primaryFlavor,
    compareFlavor: input.compareFlavor ?? null,
    expandedNodeIds: sortedUniqueIds(input.expandedNodeIds ?? []),
    focusedEntityId: input.focusedEntityId ?? null,
    traceMode: input.traceMode ?? "none",
    nodePositions: stableNodePositions(input.nodePositions ?? []),
    edgeWaypoints: stableEdgeWaypoints(input.edgeWaypoints ?? []),
    timelinePosition: normalizeTimelinePosition(input.timelinePosition),
  });
}

export function encodeGraphStateForUrl(state: GraphState): string {
  const canonical = graphStateSchema.parse(state);
  return compressToEncodedURIComponent(JSON.stringify(canonical));
}

export function decodeGraphStateFromUrl(
  encoded: string,
  canonical: CanonicalGraphStateDomain,
): Pick<GraphStateResolution, "state" | "notice"> {
  const parsed = parseGraphStatePayload(encoded);
  if (parsed.kind === "invalid") {
    return {
      state: canonical.defaultState,
      notice: {
        code: "invalid_url_state",
        message: "Shared graph state was invalid; restored canonical scene.",
        recoverable: true,
      },
    };
  }
  if (parsed.kind === "stale") {
    return {
      state: canonical.defaultState,
      notice: {
        code: "stale_url_state",
        message: "Shared graph state version is stale; restored canonical scene.",
        recoverable: true,
      },
    };
  }
  const sanitized = sanitizeGraphState(parsed.state, canonical);
  if (!sanitized) {
    return {
      state: canonical.defaultState,
      notice: {
        code: "invalid_url_state",
        message: "Shared graph state was incompatible; restored canonical scene.",
        recoverable: true,
      },
    };
  }
  return { state: sanitized };
}

export function resolveGraphState(input: {
  urlState?: string | null;
  storage: GraphStateStorage;
  canonical: CanonicalGraphStateDomain;
}): GraphStateResolution {
  const encoded = input.urlState?.trim() ?? "";
  if (encoded) {
    const resolved = decodeGraphStateFromUrl(encoded, input.canonical);
    if (!resolved.notice) {
      return { source: "url", state: resolved.state };
    }
    return { source: "canonical", state: resolved.state, notice: resolved.notice };
  }
  return readStoredGraphState(input.storage, input.canonical);
}

export function readStoredGraphState(
  storage: GraphStateStorage,
  canonical: CanonicalGraphStateDomain,
): GraphStateResolution {
  try {
    const raw = storage.getItem(GRAPH_STATE_STORAGE_KEY);
    if (!raw) {
      return { source: "canonical", state: canonical.defaultState };
    }
    const parsed = parseGraphStateObject(JSON.parse(raw));
    if (parsed.kind !== "ok") {
      return {
        source: "canonical",
        state: canonical.defaultState,
        notice: {
          code: "invalid_local_state",
          message: "Stored graph state was invalid; restored canonical scene.",
          recoverable: true,
        },
      };
    }
    const sanitized = sanitizeGraphState(parsed.state, canonical);
    if (!sanitized) {
      return {
        source: "canonical",
        state: canonical.defaultState,
        notice: {
          code: "invalid_local_state",
          message: "Stored graph state was incompatible; restored canonical scene.",
          recoverable: true,
        },
      };
    }
    return { source: "local", state: sanitized };
  } catch {
    return { source: "canonical", state: canonical.defaultState };
  }
}

export function writeStoredGraphState(
  storage: GraphStateStorage,
  state: GraphState,
): void {
  try {
    storage.setItem(
      GRAPH_STATE_STORAGE_KEY,
      JSON.stringify(graphStateSchema.parse(state)),
    );
  } catch {
    // Private browsing modes may deny writes while URL state remains shareable.
  }
}

export function clearStoredGraphState(storage: GraphStateStorage): void {
  try {
    storage.removeItem(GRAPH_STATE_STORAGE_KEY);
  } catch {
    // Ignore unavailable storage implementations.
  }
}

export function mergeLayoutStateWithCanonical(
  canonicalLayout: LayoutState,
  manualLayout: LayoutState,
  canonical: CanonicalGraphStateDomain,
): LayoutState {
  return {
    nodePositions: stableNodePositions([
      ...canonicalLayout.nodePositions,
      ...manualLayout.nodePositions,
    ]).filter(({ nodeId }) => canonical.nodeIds.has(nodeId)),
    edgeWaypoints: stableEdgeWaypoints([
      ...canonicalLayout.edgeWaypoints,
      ...manualLayout.edgeWaypoints,
    ]).filter(({ edgeId }) => canonical.edgeIds.has(edgeId)),
  };
}

export function resetManualLayoutState(state: GraphState): GraphState {
  return {
    ...state,
    nodePositions: [],
    edgeWaypoints: [],
  };
}

function parseGraphStatePayload(
  encoded: string,
): { kind: "ok"; state: GraphState } | { kind: "invalid" } | { kind: "stale" } {
  const decoded = decodePayload(encoded);
  if (!decoded) {
    return { kind: "invalid" };
  }
  return parseGraphStateObject(decoded);
}

function parseGraphStateObject(
  candidate: unknown,
): { kind: "ok"; state: GraphState } | { kind: "invalid" } | { kind: "stale" } {
  const staleVersion = z
    .object({ version: z.number() })
    .passthrough()
    .safeParse(candidate);
  if (
    staleVersion.success &&
    staleVersion.data.version !== GRAPH_STATE_VERSION
  ) {
    return { kind: "stale" };
  }
  const parsed = graphStateSchema.safeParse(candidate);
  if (!parsed.success) {
    return { kind: "invalid" };
  }
  return { kind: "ok", state: parsed.data };
}

function decodePayload(encoded: string): unknown | null {
  const compressed = decompressFromEncodedURIComponent(encoded);
  if (compressed !== null) {
    try {
      return JSON.parse(compressed);
    } catch {
      return null;
    }
  }
  try {
    return JSON.parse(decodeURIComponent(encoded));
  } catch {
    return null;
  }
}

function sanitizeGraphState(
  state: GraphState,
  canonical: CanonicalGraphStateDomain,
): GraphState | null {
  if (!canonical.sceneIds.has(state.sceneId)) {
    return null;
  }
  if (!canonical.supportedFlavors.has(state.primaryFlavor)) {
    return null;
  }
  if (
    state.compareFlavor !== null &&
    !canonical.supportedFlavors.has(state.compareFlavor)
  ) {
    return null;
  }
  if (state.expandedNodeIds.some((nodeId) => !canonical.nodeIds.has(nodeId))) {
    return null;
  }
  if (
    state.focusedEntityId !== null &&
    !canonical.nodeIds.has(state.focusedEntityId) &&
    !canonical.edgeIds.has(state.focusedEntityId)
  ) {
    return null;
  }
  if (
    state.traceMode !== "none" &&
    (state.focusedEntityId === null || !canonical.nodeIds.has(state.focusedEntityId))
  ) {
    return null;
  }
  if (state.nodePositions.some(({ nodeId }) => !canonical.nodeIds.has(nodeId))) {
    return null;
  }
  if (
    state.edgeWaypoints.some(({ edgeId }) => !canonical.edgeIds.has(edgeId))
  ) {
    return null;
  }
  return {
    ...state,
    expandedNodeIds: sortedUniqueIds(state.expandedNodeIds),
    nodePositions: stableNodePositions(state.nodePositions),
    edgeWaypoints: stableEdgeWaypoints(state.edgeWaypoints),
    timelinePosition: normalizeTimelinePosition(state.timelinePosition),
  };
}

function sortedUniqueIds(ids: readonly string[]): string[] {
  return [...new Set(ids)].sort((left, right) => left.localeCompare(right));
}

function stableNodePositions(
  entries: readonly NodePositionOverride[],
): NodePositionOverride[] {
  const byId = new Map<string, NodePositionOverride>();
  for (const entry of entries) {
    byId.set(entry.nodeId, entry);
  }
  return [...byId.values()].sort((left, right) =>
    left.nodeId.localeCompare(right.nodeId),
  );
}

function stableEdgeWaypoints(
  entries: readonly EdgeWaypointOverride[],
): EdgeWaypointOverride[] {
  const byId = new Map<string, EdgeWaypointOverride>();
  for (const entry of entries) {
    byId.set(entry.edgeId, entry);
  }
  return [...byId.values()].sort((left, right) =>
    left.edgeId.localeCompare(right.edgeId),
  );
}

function normalizeTimelinePosition(position: number | undefined): number {
  if (typeof position !== "number" || !Number.isFinite(position)) {
    return 0;
  }
  return Math.max(0, Math.min(1, position));
}
