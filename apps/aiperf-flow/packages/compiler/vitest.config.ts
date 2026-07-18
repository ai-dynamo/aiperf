// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

const packageRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@aiperf/flow-schema": path.resolve(packageRoot, "../schema/src/index.ts"),
      "@aiperf/flow-language": path.resolve(
        packageRoot,
        "../language/src/index.ts",
      ),
    },
  },
  test: {
    restoreMocks: true,
  },
});
