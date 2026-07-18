// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";

import { architectureCatalog } from "../../content";
import {
  AtlasLayoutService,
  FitAfterLayoutScheduler,
  LayoutWorkerAdapter,
  LAYOUT_PROTOCOL_VERSION,
  LayoutHierarchySchema,
  LayoutRequestSchema,
  LayoutWorkerRequestSchema,
  LayoutWorkerResponseSchema,
  buildLayoutRequest,
  composeBandLayouts,
  deterministicFallbackLayout,
  type LayoutResult,
  type LayoutWorkerPort,
} from "./layout";

function crashableWorker() {
  const listeners = new Map<string, Set<(event: Event) => void>>();
  const posted: Array<{ requestId: number }> = [];
  const terminate = vi.fn();
  const worker = {
    addEventListener: (type: string, listener: (event: Event) => void) => {
      const registered = listeners.get(type) ?? new Set();
      registered.add(listener);
      listeners.set(type, registered);
    },
    postMessage: (message: { requestId: number }) => posted.push(message),
    removeEventListener: (type: string, listener: (event: Event) => void) => {
      listeners.get(type)?.delete(listener);
    },
    terminate,
  } as unknown as LayoutWorkerPort;
  return {
    emit: (type: string, event: Event) => {
      for (const listener of listeners.get(type) ?? []) {
        listener(event);
      }
    },
    eventTypes: () => [...listeners.keys()],
    posted,
    terminate,
    worker,
  };
}

function selectComponents(ids: readonly string[]) {
  return architectureCatalog.components.filter((component) =>
    ids.includes(component.id),
  );
}

describe("semantic atlas layout", () => {
  it("parses a valid versioned layout request", () => {
    const request = buildLayoutRequest(
      selectComponents(["component.clock-seam", "component.scheduling"]),
      [],
      "ownership",
      {
        hierarchy: [
          {
            id: "component.scheduling",
            parentId: "component.clock-seam",
          },
        ],
        partialRelayout: {
          expandedSubgraphs: [
            {
              nodeIds: ["component.clock-seam", "component.scheduling"],
              rootId: "component.clock-seam",
            },
          ],
          manualPositions: [
            { id: "component.clock-seam", x: 120, y: 240 },
          ],
          relayoutNodeIds: ["component.scheduling"],
        },
      },
    );

    expect(request.version).toBe(LAYOUT_PROTOCOL_VERSION);
    expect(LayoutRequestSchema.parse(request)).toEqual(request);
  });

  it("rejects unknown layout protocol fields", () => {
    const request = buildLayoutRequest([], [], "ownership");

    expect(() =>
      LayoutHierarchySchema.parse([
        { id: "child", parentId: "parent", unexpected: true },
      ]),
    ).toThrow();
    expect(() =>
      LayoutRequestSchema.parse({ ...request, unexpected: true }),
    ).toThrow();
    expect(() =>
      LayoutRequestSchema.parse({
        ...request,
        bands: [{ id: "ownership.rust", label: "Rust", order: 0, extra: true }],
      }),
    ).toThrow();
  });

  it("rejects a stale layout protocol version", () => {
    const request = buildLayoutRequest([], [], "ownership");

    expect(() =>
      LayoutRequestSchema.parse({ ...request, version: 0 }),
    ).toThrow();
    expect(() =>
      LayoutWorkerRequestSchema.parse({
        request,
        requestId: 1,
        version: 0,
      }),
    ).toThrow();
  });

  it("rejects malformed hierarchy and partial relayout state", () => {
    const request = buildLayoutRequest(
      selectComponents(["component.clock-seam", "component.scheduling"]),
      [],
      "ownership",
    );

    expect(() =>
      LayoutRequestSchema.parse({
        ...request,
        nodes: request.nodes.map((node) =>
          node.id === "component.scheduling"
            ? { ...node, parentId: "component.missing" }
            : node,
        ),
      }),
    ).toThrow();
    expect(() =>
      LayoutRequestSchema.parse({
        ...request,
        partialRelayout: {
          expandedSubgraphs: [
            {
              nodeIds: ["component.scheduling"],
              rootId: "component.clock-seam",
            },
          ],
          manualPositions: [],
          relayoutNodeIds: ["component.scheduling"],
        },
      }),
    ).toThrow();
  });

  it("strictly parses worker request and response envelopes", () => {
    const request = buildLayoutRequest([], [], "ownership");
    const requestEnvelope = {
      request,
      requestId: 7,
      version: LAYOUT_PROTOCOL_VERSION,
    };
    const responseEnvelope = {
      requestId: 7,
      result: deterministicFallbackLayout(request, "test"),
      version: LAYOUT_PROTOCOL_VERSION,
    };

    expect(LayoutWorkerRequestSchema.parse(requestEnvelope)).toEqual(
      requestEnvelope,
    );
    expect(LayoutWorkerResponseSchema.parse(responseEnvelope)).toEqual(
      responseEnvelope,
    );
    expect(() =>
      LayoutWorkerRequestSchema.parse({ ...requestEnvelope, extra: true }),
    ).toThrow();
    expect(() =>
      LayoutWorkerResponseSchema.parse({
        error: "bad",
        requestId: 7,
        result: responseEnvelope.result,
        version: LAYOUT_PROTOCOL_VERSION,
      }),
    ).toThrow();
  });

  it("assigns explicit ownership bands", () => {
    const request = buildLayoutRequest(
      architectureCatalog.components,
      architectureCatalog.edges,
      "ownership",
    );

    expect(request.bands.map(({ id }) => id)).toContain("ownership.rust");
    expect(
      request.nodes.find(({ id }) => id === "component.clock-seam")?.bandId,
    ).toBe("ownership.rust");
  });

  it("assigns each component to its typed primary lifecycle band", () => {
    const request = buildLayoutRequest(
      architectureCatalog.components,
      architectureCatalog.edges,
      "lifecycle",
    );

    expect(
      request.nodes.find(({ id }) => id === "component.native-metrics")?.bandId,
    ).toBe("lifecycle.measurement");
    expect(
      new Set(request.nodes.map(({ id }) => id)).size,
    ).toBe(architectureCatalog.components.length);
  });

  it("produces deterministic grouped fallback geometry", () => {
    const request = buildLayoutRequest(
      architectureCatalog.components.slice(0, 5),
      architectureCatalog.edges,
      "ownership",
    );

    const first = deterministicFallbackLayout(request, "worker unavailable");
    const second = deterministicFallbackLayout(request, "worker unavailable");

    expect(first).toEqual(second);
    expect(first.degraded).toBe(true);
    expect(first.bands.every(({ width, height }) => width > 0 && height > 0)).toBe(
      true,
    );
    expect(new Set(first.bands.map(({ y }) => y)).size).toBe(
      first.bands.length,
    );
    expect(new Set(first.bands.map(({ x }) => x)).size).toBe(1);
  });

  it("renders lifecycle groups as ordered non-overlapping lanes", () => {
    const request = buildLayoutRequest(
      architectureCatalog.components,
      architectureCatalog.edges,
      "lifecycle",
    );
    const result = deterministicFallbackLayout(request, "test");

    expect(new Set(result.bands.map(({ x }) => x)).size).toBe(
      result.bands.length,
    );
    expect(new Set(result.bands.map(({ y }) => y)).size).toBe(1);
  });

  it("includes hierarchical expansion and partial relayout in layout key", () => {
    const components = architectureCatalog.components.slice(0, 3);
    const [root, child, other] = components;
    const base = buildLayoutRequest(components, architectureCatalog.edges, "ownership");
    const partial = buildLayoutRequest(components, architectureCatalog.edges, "ownership", {
      hierarchy: [{ id: child.id, parentId: root.id }],
      partialRelayout: {
        expandedSubgraphs: [{ nodeIds: [root.id, child.id], rootId: root.id }],
        manualPositions: [
          { id: child.id, x: 140, y: 220 },
          { id: other.id, x: 340, y: 460 },
        ],
        relayoutNodeIds: [child.id],
      },
    });

    expect(partial.key).not.toBe(base.key);
    expect(partial.partialRelayout?.expandedSubgraphs).toEqual([
      { nodeIds: [root.id, child.id], rootId: root.id },
    ]);
    expect(
      partial.nodes.find(({ id }) => id === child.id)?.parentId,
    ).toBe(root.id);
  });

  it("preserves unaffected manual positions during partial relayout merge", () => {
    const request = buildLayoutRequest(
      selectComponents([
        "component.clock-seam",
        "component.scheduling",
        "component.python-frontend",
      ]),
      [],
      "ownership",
      {
        partialRelayout: {
          expandedSubgraphs: [],
          manualPositions: [{ id: "component.clock-seam", x: 777, y: 555 }],
          relayoutNodeIds: ["component.scheduling"],
        },
      },
    );
    const merged = composeBandLayouts(request, [
      {
        bandId: "ownership.rust",
        height: 280,
        positions: [
          { id: "component.clock-seam", x: 4, y: 8 },
          { id: "component.scheduling", x: 40, y: 52 },
        ],
        width: 320,
      },
      {
        bandId: "ownership.python",
        height: 240,
        positions: [{ id: "component.python-frontend", x: 16, y: 24 }],
        width: 260,
      },
    ]);
    const preserved = merged.positions.find(
      ({ id }) => id === "component.clock-seam",
    );
    const relaidOut = merged.positions.find(
      ({ id }) => id === "component.scheduling",
    );

    expect(preserved).toMatchObject({ x: 777, y: 555 });
    expect(relaidOut).not.toMatchObject({ x: 777, y: 555 });
    expect(merged.partialRelayout).toEqual({
      preservedManualNodeIds: ["component.clock-seam"],
      relaidOutNodeIds: ["component.scheduling"],
    });
  });

  it("keeps manual positions in deterministic fallback for unaffected nodes", () => {
    const request = buildLayoutRequest(
      selectComponents([
        "component.clock-seam",
        "component.scheduling",
        "component.python-frontend",
      ]),
      [],
      "ownership",
      {
        partialRelayout: {
          expandedSubgraphs: [],
          manualPositions: [
            { id: "component.clock-seam", x: 600, y: 610 },
            { id: "component.scheduling", x: 22, y: 44 },
          ],
          relayoutNodeIds: ["component.scheduling"],
        },
      },
    );
    const result = deterministicFallbackLayout(request, "fallback");

    expect(
      result.positions.find(({ id }) => id === "component.clock-seam"),
    ).toMatchObject({ x: 600, y: 610 });
    expect(
      result.positions.find(({ id }) => id === "component.scheduling"),
    ).not.toMatchObject({ x: 22, y: 44 });
    expect(result.partialRelayout).toEqual({
      preservedManualNodeIds: ["component.clock-seam"],
      relaidOutNodeIds: ["component.scheduling"],
    });
  });
});

describe("layout worker adapter", () => {
  it("correlates worker responses without running ELK on the caller", async () => {
    let receive:
      | ((event: MessageEvent<unknown>) => void)
      | undefined;
    const result: LayoutResult = {
      bands: [],
      degraded: false,
      positions: [],
    };
    const worker: LayoutWorkerPort = {
      addEventListener: (type, listener) => {
        if (type === "message") {
          receive = listener as (event: MessageEvent<unknown>) => void;
        }
      },
      postMessage: (message) => {
        queueMicrotask(() =>
          receive?.(
            new MessageEvent("message", {
              data: {
                requestId: message.requestId,
                result,
                version: LAYOUT_PROTOCOL_VERSION,
              },
            }),
          ),
        );
      },
      removeEventListener: () => undefined,
      terminate: () => undefined,
    };
    const adapter = new LayoutWorkerAdapter(worker);
    const request = buildLayoutRequest([], [], "ownership");

    await expect(adapter.layout(request)).resolves.toEqual(result);
    adapter.dispose();
  });

  it("falls back on worker error and replaces the poisoned worker", async () => {
    const firstWorker = crashableWorker();
    const replacementWorker = crashableWorker();
    const request = buildLayoutRequest(
      architectureCatalog.components.slice(0, 2),
      [],
      "ownership",
    );
    const adapter = new LayoutWorkerAdapter(
      firstWorker.worker,
      () => replacementWorker.worker,
    );
    const service = new AtlasLayoutService(adapter);
    const first = service.layout(request);

    await Promise.resolve();
    expect(firstWorker.eventTypes()).toContain("error");
    firstWorker.emit(
      "error",
      new ErrorEvent("error", { message: "ELK worker crashed" }),
    );

    await expect(first).resolves.toMatchObject({
      degraded: true,
      reason: "ELK worker crashed",
    });
    expect(firstWorker.terminate).toHaveBeenCalledOnce();

    const retried = service.layout(request);
    await Promise.resolve();
    expect(replacementWorker.posted).toHaveLength(1);
    replacementWorker.emit(
      "message",
      new MessageEvent("message", {
        data: {
          requestId: replacementWorker.posted[0]?.requestId,
          result: deterministicFallbackLayout(request, "recovered"),
          version: LAYOUT_PROTOCOL_VERSION,
        },
      }),
    );
    await expect(retried).resolves.toMatchObject({ reason: "recovered" });
  });

  it("rejects all pending work and resets after message decode failure", async () => {
    const firstWorker = crashableWorker();
    const replacementWorker = crashableWorker();
    const adapter = new LayoutWorkerAdapter(
      firstWorker.worker,
      () => replacementWorker.worker,
    );
    const request = buildLayoutRequest([], [], "ownership");
    const first = adapter.layout(request);
    const second = adapter.layout(request);

    expect(firstWorker.eventTypes()).toContain("messageerror");
    firstWorker.emit("messageerror", new MessageEvent("messageerror"));

    await expect(first).rejects.toThrow(/decode/i);
    await expect(second).rejects.toThrow(/decode/i);
    expect(firstWorker.terminate).toHaveBeenCalledOnce();

    const retried = adapter.layout(request);
    expect(replacementWorker.posted).toHaveLength(1);
    replacementWorker.emit(
      "message",
      new MessageEvent("message", {
        data: {
          requestId: replacementWorker.posted[0]?.requestId,
          result: { bands: [], degraded: false, positions: [] },
          version: LAYOUT_PROTOCOL_VERSION,
        },
      }),
    );
    await expect(retried).resolves.toMatchObject({ degraded: false });
  });

  it("evicts failed cache entries and retries after fallback", async () => {
    const request = buildLayoutRequest(
      architectureCatalog.components.slice(0, 2),
      [],
      "ownership",
    );
    let attempts = 0;
    const service = new AtlasLayoutService({
      layout: async () => {
        attempts += 1;
        if (attempts === 1) {
          throw new Error("worker crashed");
        }
        return {
          ...deterministicFallbackLayout(request, "unused"),
          degraded: false,
          reason: undefined,
        };
      },
    });

    await expect(service.layout(request)).resolves.toMatchObject({
      degraded: true,
      reason: "worker crashed",
    });
    await expect(service.layout(request)).resolves.toMatchObject({
      degraded: false,
    });
    expect(attempts).toBe(2);
  });
});

describe("fit-after-layout scheduling", () => {
  it("fits only the newest committed positioned node set", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextId = 0;
    const scheduler = new FitAfterLayoutScheduler(
      (callback) => {
        nextId += 1;
        callbacks.set(nextId, callback);
        return nextId;
      },
      (id) => callbacks.delete(id),
    );
    const fit = vi.fn();

    scheduler.schedule(["stale"], fit);
    scheduler.schedule(["current-a", "current-b"], fit);
    for (const callback of callbacks.values()) {
      callback(0);
    }

    expect(fit).toHaveBeenCalledOnce();
    expect(fit).toHaveBeenCalledWith(["current-a", "current-b"]);
  });

  it("cancels stale fit work while a reset layout is unresolved", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    const scheduler = new FitAfterLayoutScheduler(
      (callback) => {
        callbacks.set(1, callback);
        return 1;
      },
      (id) => callbacks.delete(id),
    );
    const fit = vi.fn();

    scheduler.schedule(["stale-before-reset"], fit);
    scheduler.cancel();
    for (const callback of callbacks.values()) {
      callback(0);
    }

    expect(fit).not.toHaveBeenCalled();
  });
});
