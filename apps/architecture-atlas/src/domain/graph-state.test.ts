// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import type { ExecutionFlavor } from "./architecture";
import {
  GRAPH_STATE_STORAGE_KEY,
  canonicalGraphState,
  clearStoredGraphState,
  decodeGraphStateFromUrl,
  encodeGraphStateForUrl,
  graphStateSchema,
  mergeLayoutStateWithCanonical,
  readStoredGraphState,
  resetManualLayoutState,
  resolveGraphState,
  writeStoredGraphState,
  type CanonicalGraphStateDomain,
} from "./graph-state";

function canonicalDomain(): CanonicalGraphStateDomain {
  const flavors: ExecutionFlavor[] = ["native_http", "native_grpc", "online_mock"];
  return {
    defaultState: canonicalGraphState({
      audience: "developer",
      primaryFlavor: "native_http",
      sceneId: "scene.runtime-composition",
      timelinePosition: 0,
    }),
    edgeIds: new Set(["edge.runtime.dispatch.metrics"]),
    nodeIds: new Set(["node.runtime-composition", "node.metrics-telemetry"]),
    sceneIds: new Set(["scene.runtime-composition", "scene.metrics-telemetry"]),
    supportedFlavors: new Set(flavors),
  };
}

describe("graph share state", () => {
  it("round-trips URL-safe compressed state", () => {
    const state = canonicalGraphState({
      audience: "maintainer",
      compareFlavor: "native_grpc",
      edgeWaypoints: [
        {
          edgeId: "edge.runtime.dispatch.metrics",
          points: [{ x: 15, y: 30 }],
        },
      ],
      expandedNodeIds: ["node.runtime-composition"],
      focusedEntityId: "node.runtime-composition",
      nodePositions: [{ nodeId: "node.runtime-composition", x: 120, y: 80 }],
      primaryFlavor: "native_http",
      sceneId: "scene.runtime-composition",
      timelinePosition: 42,
    });

    const encoded = encodeGraphStateForUrl(state);
    expect(encoded).toMatch(/^[A-Za-z0-9\-_.!~*'()+$]+$/);

    const decoded = decodeGraphStateFromUrl(encoded, canonicalDomain());
    expect(decoded.state).toEqual(state);
    expect(decoded.state.edgeWaypoints[0]).toEqual({
      edgeId: "edge.runtime.dispatch.metrics",
      points: [{ x: 15, y: 30 }],
    });
    expect(decoded.state.edgeWaypoints[0]).not.toHaveProperty("sourceNodeId");
    expect(decoded.state.edgeWaypoints[0]).not.toHaveProperty("targetNodeId");
    expect(decoded.notice).toBeUndefined();
  });

  it("rejects endpoint semantics in waypoint state", () => {
    const candidate = {
      ...canonicalGraphState({
        audience: "developer",
        primaryFlavor: "native_http",
        sceneId: "scene.runtime-composition",
      }),
      edgeWaypoints: [
        {
          edgeId: "edge.runtime.dispatch.metrics",
          points: [{ x: 1, y: 2 }],
          sourceNodeId: "node.runtime-composition",
          targetNodeId: "node.metrics-telemetry",
        },
      ],
    };

    expect(graphStateSchema.safeParse(candidate).success).toBe(false);
  });

  it("rejects waypoint edge IDs absent from the canonical graph", () => {
    const state = canonicalGraphState({
      audience: "developer",
      edgeWaypoints: [
        {
          edgeId: "edge.unknown",
          points: [{ x: 1, y: 2 }],
        },
      ],
      primaryFlavor: "native_http",
      sceneId: "scene.runtime-composition",
    });

    const resolved = resolveGraphState({
      canonical: canonicalDomain(),
      storage: {
        getItem: () => null,
        removeItem: () => undefined,
        setItem: () => undefined,
      },
      urlState: encodeGraphStateForUrl(state),
    });

    expect(resolved.source).toBe("canonical");
    expect(resolved.notice?.code).toBe("invalid_url_state");
  });

  it("prefers valid URL state over local state", () => {
    const canonical = canonicalDomain();
    const localState = canonicalGraphState({
      audience: "executive",
      primaryFlavor: "online_mock",
      sceneId: "scene.metrics-telemetry",
      timelinePosition: 12,
    });
    const urlState = canonicalGraphState({
      audience: "maintainer",
      primaryFlavor: "native_grpc",
      sceneId: "scene.runtime-composition",
      timelinePosition: 77,
    });
    const storage = {
      getItem: (key: string) =>
        key === GRAPH_STATE_STORAGE_KEY ? JSON.stringify(localState) : null,
      setItem: () => undefined,
      removeItem: () => undefined,
    };

    const resolved = resolveGraphState({
      canonical,
      storage,
      urlState: encodeGraphStateForUrl(urlState),
    });
    expect(resolved.source).toBe("url");
    expect(resolved.state).toEqual(urlState);
  });

  it("normalizes timeline position to flow timeline range", () => {
    const canonical = canonicalDomain();
    const encoded = encodeURIComponent(
      JSON.stringify({
        version: 1,
        sceneId: "scene.runtime-composition",
        audience: "developer",
        primaryFlavor: "native_http",
        compareFlavor: null,
        expandedNodeIds: [],
        focusedEntityId: null,
        traceMode: "none",
        nodePositions: [],
        edgeWaypoints: [],
        timelinePosition: 7.5,
      }),
    );

    const resolved = resolveGraphState({
      canonical,
      storage: { getItem: () => null, removeItem: () => undefined, setItem: () => undefined },
      urlState: encoded,
    });
    expect(resolved.source).toBe("url");
    expect(resolved.state.timelinePosition).toBe(1);
  });

  it("recovers canonical defaults with stale URL notice", () => {
    const canonical = canonicalDomain();
    const staleEncoded = encodeURIComponent(
      JSON.stringify({ version: 0, sceneId: "scene.runtime-composition" }),
    );

    const resolved = resolveGraphState({
      canonical,
      storage: { getItem: () => null, removeItem: () => undefined, setItem: () => undefined },
      urlState: staleEncoded,
    });
    expect(resolved.source).toBe("canonical");
    expect(resolved.state).toEqual(canonical.defaultState);
    expect(resolved.notice?.code).toBe("stale_url_state");
  });

  it("recovers canonical defaults with invalid URL notice", () => {
    const canonical = canonicalDomain();
    const resolved = resolveGraphState({
      canonical,
      storage: { getItem: () => null, removeItem: () => undefined, setItem: () => undefined },
      urlState: "not-valid-state",
    });
    expect(resolved.source).toBe("canonical");
    expect(resolved.state).toEqual(canonical.defaultState);
    expect(resolved.notice?.code).toBe("invalid_url_state");
  });

  it("recovers canonical defaults with invalid local state notice", () => {
    const canonical = canonicalDomain();
    const incompatibleLocalState = canonicalGraphState({
      audience: "developer",
      nodePositions: [{ nodeId: "node.unknown", x: 1, y: 2 }],
      primaryFlavor: "native_http",
      sceneId: "scene.runtime-composition",
      timelinePosition: 0.4,
    });
    const resolved = resolveGraphState({
      canonical,
      storage: {
        getItem: () => JSON.stringify(incompatibleLocalState),
        removeItem: () => undefined,
        setItem: () => undefined,
      },
      urlState: null,
    });

    expect(resolved.source).toBe("canonical");
    expect(resolved.state).toEqual(canonical.defaultState);
    expect(resolved.notice?.code).toBe("invalid_local_state");
  });

  it("rejects semantic graph content from URL state", () => {
    const canonical = canonicalDomain();
    const encoded = encodeURIComponent(
      JSON.stringify({
        version: 1,
        sceneId: "scene.runtime-composition",
        audience: "developer",
        primaryFlavor: "native_http",
        compareFlavor: null,
        expandedNodeIds: [],
        focusedEntityId: null,
        traceMode: "none",
        nodePositions: [],
        edgeWaypoints: [],
        timelinePosition: 0,
        graphNodes: [{ id: "node.injected", title: "forbidden semantic content" }],
      }),
    );

    const resolved = resolveGraphState({
      canonical,
      storage: { getItem: () => null, removeItem: () => undefined, setItem: () => undefined },
      urlState: encoded,
    });
    expect(resolved.source).toBe("canonical");
    expect(resolved.notice?.code).toBe("invalid_url_state");
  });
});

describe("layout state helpers", () => {
  it("merges canonical layout with manual overrides and preserves semantics", () => {
    const merged = mergeLayoutStateWithCanonical(
      {
        edgeWaypoints: [
          {
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 10, y: 10 }],
          },
        ],
        nodePositions: [
          { nodeId: "node.runtime-composition", x: 20, y: 30 },
          { nodeId: "node.metrics-telemetry", x: 40, y: 50 },
        ],
      },
      {
        edgeWaypoints: [
          {
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 70, y: 80 }],
          },
        ],
        nodePositions: [{ nodeId: "node.metrics-telemetry", x: 100, y: 120 }],
      },
      canonicalDomain(),
    );

    expect(merged.nodePositions).toEqual([
      { nodeId: "node.metrics-telemetry", x: 100, y: 120 },
      { nodeId: "node.runtime-composition", x: 20, y: 30 },
    ]);
    expect(merged.edgeWaypoints).toEqual([
      {
        edgeId: "edge.runtime.dispatch.metrics",
        points: [{ x: 70, y: 80 }],
      },
    ]);
  });

  it("resets manual layout overrides to canonical-only state", () => {
    const reset = resetManualLayoutState(
      canonicalGraphState({
        audience: "developer",
        edgeWaypoints: [
          {
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 7, y: 11 }],
          },
        ],
        expandedNodeIds: ["node.runtime-composition"],
        focusedEntityId: "node.runtime-composition",
        nodePositions: [{ nodeId: "node.runtime-composition", x: 12, y: 16 }],
        primaryFlavor: "native_http",
        sceneId: "scene.runtime-composition",
        timelinePosition: 3,
      }),
    );

    expect(reset.nodePositions).toEqual([]);
    expect(reset.edgeWaypoints).toEqual([]);
    expect(reset.expandedNodeIds).toEqual(["node.runtime-composition"]);
    expect(reset.sceneId).toBe("scene.runtime-composition");
  });
});

describe("local storage helpers", () => {
  it("uses stable storage helpers for read/write/clear", () => {
    const memory = new Map<string, string>();
    const storage = {
      getItem: (key: string) => memory.get(key) ?? null,
      removeItem: (key: string) => void memory.delete(key),
      setItem: (key: string, value: string) => void memory.set(key, value),
    };
    const state = canonicalGraphState({
      audience: "developer",
      primaryFlavor: "native_http",
      sceneId: "scene.runtime-composition",
      timelinePosition: 9,
    });

    writeStoredGraphState(storage, state);
    const readBack = readStoredGraphState(storage, canonicalDomain());
    expect(readBack.state).toEqual(state);
    expect(readBack.source).toBe("local");

    clearStoredGraphState(storage);
    const afterClear = readStoredGraphState(storage, canonicalDomain());
    expect(afterClear.source).toBe("canonical");
  });
});
