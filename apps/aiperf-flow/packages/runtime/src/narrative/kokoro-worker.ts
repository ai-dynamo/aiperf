// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedicated-worker Kokoro model loading and speech inference.

export type KokoroDevice = "webgpu" | "wasm";
export type KokoroDtype = "fp32" | "q8";

/** Commands accepted by the Kokoro inference worker. */
export type KokoroWorkerCommand =
  | Readonly<{
      type: "initialize";
      modelId: string;
      device: KokoroDevice;
      dtype: KokoroDtype;
    }>
  | Readonly<{
      type: "synthesize";
      requestId: string;
      cueId: string;
      text: string;
      voiceId: string | null;
      rate: number;
    }>
  | Readonly<{ type: "cancel"; requestId: string; cueId: string }>;

export type KokoroWorkerVoice = Readonly<{
  id: string;
  name: string;
  language: string;
  default: boolean;
}>;

/** Events emitted by the Kokoro inference worker. */
export type KokoroWorkerMessage =
  | Readonly<{ type: "progress"; progress: number; file: string }>
  | Readonly<{ type: "ready"; voices: readonly KokoroWorkerVoice[] }>
  | Readonly<{
      type: "audio";
      requestId: string;
      cueId: string;
      wav: ArrayBuffer;
    }>
  | Readonly<{
      type: "error";
      message: string;
      requestId?: string;
      cueId?: string;
    }>;

type KokoroProgress = Readonly<{
  status: string;
  file?: string;
  progress?: number;
}>;

/** Narrow model surface used to keep worker orchestration independently testable. */
export type KokoroModel = Readonly<{
  voices: Readonly<Record<string, Readonly<{ name: string; language: string }>>>;
  generate(
    text: string,
    options: Readonly<{ voice?: string; speed: number }>,
  ): Promise<Readonly<{ toWav(): ArrayBuffer }>>;
}>;

export type KokoroModelLoader = (
  modelId: string,
  options: Readonly<{
    device: KokoroDevice;
    dtype: KokoroDtype;
    progress_callback(progress: KokoroProgress): void;
  }>,
) => Promise<KokoroModel>;

export type KokoroWorkerRuntimeOptions = Readonly<{
  load: KokoroModelLoader;
  postMessage(message: KokoroWorkerMessage, transfer?: Transferable[]): void;
}>;

/** Creates the stateful command handler hosted by the module worker. */
export function createKokoroWorkerRuntime(
  options: KokoroWorkerRuntimeOptions,
): Readonly<{ handle(command: KokoroWorkerCommand): Promise<void> }> {
  let model: KokoroModel | null = null;
  const cancelledRequestIds = new Set<string>();

  const handle = async (command: KokoroWorkerCommand): Promise<void> => {
    try {
      if (command.type === "initialize") {
        model = await options.load(command.modelId, {
          device: command.device,
          dtype: command.dtype,
          progress_callback: (progress) => {
            if (
              progress.status === "progress" &&
              typeof progress.progress === "number"
            ) {
              options.postMessage({
                type: "progress",
                progress: progress.progress,
                file: progress.file ?? "",
              });
            }
          },
        });
        options.postMessage({
          type: "ready",
          voices: Object.entries(model.voices).map(([id, voice]) =>
            Object.freeze({
              id,
              name: voice.name,
              language: voice.language,
              default: id === "af_heart",
            }),
          ),
        });
        return;
      }

      if (command.type === "cancel") {
        cancelledRequestIds.add(command.requestId);
        return;
      }

      if (model === null) {
        throw new Error("Kokoro model is not initialized");
      }

      const audio = await model.generate(command.text, {
        ...(command.voiceId === null ? {} : { voice: command.voiceId }),
        speed: command.rate,
      });
      if (cancelledRequestIds.has(command.requestId)) {
        cancelledRequestIds.delete(command.requestId);
        return;
      }
      const wav = audio.toWav();
      options.postMessage(
        {
          type: "audio",
          requestId: command.requestId,
          cueId: command.cueId,
          wav,
        },
        [wav],
      );
    } catch (error) {
      options.postMessage({
        type: "error",
        message: error instanceof Error ? error.message : String(error),
        ...(command.type === "synthesize"
          ? { requestId: command.requestId, cueId: command.cueId }
          : {}),
      });
    }
  };

  return Object.freeze({ handle });
}

type WorkerScope = Readonly<{
  document?: unknown;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<KokoroWorkerCommand>) => void,
  ): void;
  postMessage(message: KokoroWorkerMessage, transfer?: Transferable[]): void;
}>;

const candidateScope = globalThis as unknown as Partial<WorkerScope>;
if (
  candidateScope.document === undefined &&
  typeof candidateScope.addEventListener === "function" &&
  typeof candidateScope.postMessage === "function"
) {
  const scope = candidateScope as WorkerScope;
  const runtime = createKokoroWorkerRuntime({
    load: async (modelId, options) => {
      const { KokoroTTS } = await import("kokoro-js");
      return (await KokoroTTS.from_pretrained(modelId, {
        device: options.device,
        dtype: options.dtype,
        progress_callback: options.progress_callback,
      })) as KokoroModel;
    },
    postMessage: (message, transfer) => scope.postMessage(message, transfer),
  });
  scope.addEventListener("message", (event) => {
    void runtime.handle(event.data);
  });
}
