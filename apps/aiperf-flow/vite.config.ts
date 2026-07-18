import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.join(rootDir, "preview"),
  plugins: [react()],
  assetsInclude: ["**/*.wasm"],
  worker: {
    format: "es",
  },
  resolve: {
    alias: {
      "@aiperf/flow-schema": path.join(rootDir, "packages/schema/src/index.ts"),
      "@aiperf/flow-runtime": path.join(rootDir, "packages/runtime/src/index.ts"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5188,
    strictPort: true,
    open: false,
    fs: {
      allow: [rootDir],
    },
  },
});
