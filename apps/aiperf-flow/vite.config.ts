/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  // elkjs is CommonJS with no ESM build; pin the exact entry the layout engine imports so Vite
  // always pre-bundles it. Without this, adding the dep to a running dev server leaves it
  // un-optimized and the browser import fails, blanking every diagram that uses the engine.
  optimizeDeps: {
    include: ["elkjs/lib/elk.bundled.js"],
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/vitest.setup.ts"],
  },
});
