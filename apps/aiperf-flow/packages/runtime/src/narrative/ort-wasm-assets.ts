// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vite-emitted ONNX Runtime asset URLs for the Kokoro worker.

import ortWasmUrl from "../../../../node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.wasm?url";
import ortMjsUrl from "../../../../node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.mjs?url";

import type { KokoroWasmPaths } from "./kokoro-worker.js";

/** Absolute-ready root-relative URLs for the ORT jsep wasm glue pair. */
export const BUNDLED_ORT_WASM_PATHS: KokoroWasmPaths = Object.freeze({
  wasm: ortWasmUrl,
  mjs: ortMjsUrl,
});
