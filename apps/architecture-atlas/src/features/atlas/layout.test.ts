// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";

import { architectureCatalog } from "../../content";
import {
  AtlasLayoutService,
  FitAfterLayoutScheduler,
  LayoutWorkerAdapter,
  buildLayoutRequest,
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

describe("semantic atlas layout", () => {
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
              data: { requestId: message.requestId, result },
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
