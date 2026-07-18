/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { build } from "esbuild";
import { copyFile } from "node:fs/promises";

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
});

await copyFile(
  "dist/narrative/kokoro-worker.js",
  "dist/kokoro-worker.js",
);
await copyFile(
  "dist/narrative/kokoro-worker.js.map",
  "dist/kokoro-worker.js.map",
);
