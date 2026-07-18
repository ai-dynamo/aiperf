import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: [path.join(rootDir, "preview/vitest.setup.ts")],
  },
  resolve: {
    alias: {
      "@aiperf/flow-schema": path.join(
        rootDir,
        "packages/schema/src/index.ts",
      ),
      "@aiperf/flow-runtime": path.join(
        rootDir,
        "packages/runtime/src/index.ts",
      ),
    },
  },
});
