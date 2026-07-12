// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildLayoutRequest,
  deterministicFallbackLayout,
  type LayoutRequest,
  type LayoutResult,
} from "./layout-protocol";

export * from "./layout-protocol";

interface LayoutExecutor {
  layout(request: LayoutRequest): Promise<LayoutResult>;
}

interface WorkerRequest {
  request: LayoutRequest;
  requestId: number;
}

interface WorkerResponse {
  error?: string;
  requestId: number;
  result?: LayoutResult;
}

export interface LayoutWorkerPort {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  postMessage(message: WorkerRequest): void;
  removeEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  terminate(): void;
}

export class LayoutWorkerAdapter implements LayoutExecutor {
  private readonly pending = new Map<
    number,
    {
      reject(error: Error): void;
      resolve(result: LayoutResult): void;
    }
  >();
  private requestId = 0;

  private readonly receive = (event: MessageEvent<unknown>) => {
    const response = event.data as WorkerResponse;
    const pending = this.pending.get(response.requestId);
    if (!pending) {
      return;
    }
    this.pending.delete(response.requestId);
    if (response.error || !response.result) {
      pending.reject(new Error(response.error ?? "layout worker returned no result"));
    } else {
      pending.resolve(response.result);
    }
  };

  constructor(private readonly worker: LayoutWorkerPort) {
    worker.addEventListener("message", this.receive);
  }

  layout(request: LayoutRequest): Promise<LayoutResult> {
    this.requestId += 1;
    const requestId = this.requestId;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { reject, resolve });
      try {
        this.worker.postMessage({ request, requestId });
      } catch (error) {
        this.pending.delete(requestId);
        reject(error);
      }
    });
  }

  dispose(): void {
    this.worker.removeEventListener("message", this.receive);
    this.worker.terminate();
    for (const { reject } of this.pending.values()) {
      reject(new Error("layout worker disposed"));
    }
    this.pending.clear();
  }
}

export class AtlasLayoutService {
  private readonly cache = new Map<string, Promise<LayoutResult>>();

  constructor(private readonly executor: LayoutExecutor) {}

  layout(request: LayoutRequest): Promise<LayoutResult> {
    const cached = this.cache.get(request.key);
    if (cached) {
      return cached;
    }
    const pending = Promise.resolve()
      .then(() => this.executor.layout(request))
      .catch((error: unknown) => {
        this.cache.delete(request.key);
        return deterministicFallbackLayout(
          request,
          error instanceof Error ? error.message : String(error),
        );
      });
    this.cache.set(request.key, pending);
    return pending;
  }
}

export class FitAfterLayoutScheduler {
  private pendingFrame?: number;

  constructor(
    private readonly requestFrame: (callback: FrameRequestCallback) => number,
    private readonly cancelFrame: (id: number) => void,
  ) {}

  schedule(nodeIds: readonly string[], fit: (ids: string[]) => void): void {
    this.cancel();
    this.pendingFrame = this.requestFrame(() => {
      this.pendingFrame = undefined;
      fit([...nodeIds]);
    });
  }

  cancel(): void {
    if (this.pendingFrame !== undefined) {
      this.cancelFrame(this.pendingFrame);
      this.pendingFrame = undefined;
    }
  }

  dispose(): void {
    this.cancel();
  }
}

let defaultService: AtlasLayoutService | undefined;

function defaultLayoutService(): AtlasLayoutService {
  if (defaultService) {
    return defaultService;
  }
  const executor: LayoutExecutor =
    typeof Worker === "undefined"
      ? {
          layout: async () => {
            throw new Error("layout worker unavailable");
          },
        }
      : new LayoutWorkerAdapter(
          new Worker(new URL("./layout.worker.ts", import.meta.url), {
            type: "module",
          }),
        );
  defaultService = new AtlasLayoutService(executor);
  return defaultService;
}

export function layoutAtlas(request: LayoutRequest): Promise<LayoutResult> {
  return defaultLayoutService().layout(request);
}

export { buildLayoutRequest };
