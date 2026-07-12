// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  LAYOUT_PROTOCOL_VERSION,
  LayoutWorkerResponseSchema,
  buildLayoutRequest,
  deterministicFallbackLayout,
  type LayoutRequest,
  type LayoutResult,
  type LayoutWorkerRequest,
} from "./layout-protocol";

export * from "./layout-protocol";

interface LayoutExecutor {
  layout(request: LayoutRequest): Promise<LayoutResult>;
}

export interface LayoutWorkerPort {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  addEventListener(
    type: "error",
    listener: (event: ErrorEvent) => void,
  ): void;
  addEventListener(
    type: "messageerror",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  postMessage(message: LayoutWorkerRequest): void;
  removeEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
  removeEventListener(
    type: "error",
    listener: (event: ErrorEvent) => void,
  ): void;
  removeEventListener(
    type: "messageerror",
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
  private worker?: LayoutWorkerPort;
  private disposed = false;

  private readonly receive = (event: MessageEvent<unknown>) => {
    const parsed = LayoutWorkerResponseSchema.safeParse(event.data);
    if (!parsed.success) {
      this.poison("layout worker response validation failure");
      return;
    }
    const response = parsed.data;
    const pending = this.pending.get(response.requestId);
    if (!pending) {
      return;
    }
    this.pending.delete(response.requestId);
    if ("error" in response) {
      pending.reject(new Error(response.error));
    } else {
      pending.resolve(response.result);
    }
  };
  private readonly receiveError = (event: ErrorEvent) => {
    this.poison(event.message || "layout worker crashed");
  };
  private readonly receiveMessageError = () => {
    this.poison("layout worker message decode failure");
  };

  constructor(
    worker: LayoutWorkerPort,
    private readonly replaceWorker: () => LayoutWorkerPort = () => {
      throw new Error("layout worker cannot be replaced");
    },
  ) {
    this.worker = worker;
    this.attach(worker);
  }

  private attach(worker: LayoutWorkerPort): void {
    worker.addEventListener("message", this.receive);
    worker.addEventListener("error", this.receiveError);
    worker.addEventListener("messageerror", this.receiveMessageError);
  }

  private detach(worker: LayoutWorkerPort): void {
    worker.removeEventListener("message", this.receive);
    worker.removeEventListener("error", this.receiveError);
    worker.removeEventListener("messageerror", this.receiveMessageError);
  }

  private poison(message: string): void {
    const worker = this.worker;
    this.worker = undefined;
    if (worker) {
      this.detach(worker);
      worker.terminate();
    }
    const error = new Error(message);
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const request of pending) {
      request.reject(error);
    }
  }

  private activeWorker(): LayoutWorkerPort {
    if (this.disposed) {
      throw new Error("layout worker adapter is disposed");
    }
    if (!this.worker) {
      this.worker = this.replaceWorker();
      this.attach(this.worker);
    }
    return this.worker;
  }

  layout(request: LayoutRequest): Promise<LayoutResult> {
    this.requestId += 1;
    const requestId = this.requestId;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { reject, resolve });
      try {
        this.activeWorker().postMessage({
          request,
          requestId,
          version: LAYOUT_PROTOCOL_VERSION,
        });
      } catch (error) {
        this.pending.delete(requestId);
        reject(error);
      }
    });
  }

  dispose(): void {
    this.disposed = true;
    this.poison("layout worker disposed");
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
      : (() => {
          const createWorker = () =>
            new Worker(new URL("./layout.worker.ts", import.meta.url), {
              type: "module",
            }) as LayoutWorkerPort;
          return new LayoutWorkerAdapter(
            createWorker(),
            createWorker,
          );
        })();
  defaultService = new AtlasLayoutService(executor);
  return defaultService;
}

export function layoutAtlas(request: LayoutRequest): Promise<LayoutResult> {
  return defaultLayoutService().layout(request);
}

export { buildLayoutRequest };
