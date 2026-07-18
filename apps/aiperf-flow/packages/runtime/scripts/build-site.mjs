/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { build } from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const runtimeRoot = path.dirname(fileURLToPath(new URL("..", import.meta.url)));
const flowRoot = path.dirname(path.dirname(runtimeRoot));
const transformersDist = path.join(
  flowRoot,
  "node_modules/@huggingface/transformers/dist",
);
const ortWasmName = "ort-wasm-simd-threaded.jsep.wasm";
const ortMjsName = "ort-wasm-simd-threaded.jsep.mjs";
const ortOutDir = path.join(runtimeRoot, "dist/narrative");

await mkdir(ortOutDir, { recursive: true });
await copyFile(
  path.join(transformersDist, ortWasmName),
  path.join(ortOutDir, ortWasmName),
);
await copyFile(
  path.join(transformersDist, ortMjsName),
  path.join(ortOutDir, ortMjsName),
);

const ortAssetModule = `
import type { KokoroWasmPaths } from "./kokoro-worker.js";
export const BUNDLED_ORT_WASM_PATHS = Object.freeze({
  wasm: ${JSON.stringify(`./${ortWasmName}`)},
  mjs: ${JSON.stringify(`./${ortMjsName}`)},
});
`;

await build({
  entryPoints: ["src/site.tsx"],
  bundle: true,
  minify: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  outfile: "dist/site.js",
  sourcemap: true,
});

await build({
  entryPoints: ["src/narrative/kokoro-worker.ts"],
  bundle: true,
  minify: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  outfile: "dist/narrative/kokoro-worker.js",
  sourcemap: true,
  plugins: [
    {
      name: "ort-wasm-assets-stub",
      setup(buildApi) {
        buildApi.onResolve(
          { filter: /^\.\/ort-wasm-assets\.js$/ },
          (args) => ({
            path: path.join(args.resolveDir, "ort-wasm-assets.stub"),
            namespace: "ort-assets",
          }),
        );
        buildApi.onLoad({ filter: /.*/, namespace: "ort-assets" }, () => ({
          contents: ortAssetModule,
          loader: "ts",
          resolveDir: path.join(runtimeRoot, "src/narrative"),
        }));
      },
    },
  ],
});

await copyFile(
  "dist/narrative/kokoro-worker.js",
  "dist/kokoro-worker.js",
);
await copyFile(
  "dist/narrative/kokoro-worker.js.map",
  "dist/kokoro-worker.js.map",
);
