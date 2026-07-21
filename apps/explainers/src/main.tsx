/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { loadDeckPackages } from "./core/load-deck-flows";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

if (import.meta.env.DEV) {
  // Opt-in, non-blocking: compiles every deck in the background for dev
  // diagnostics only. The app itself never waits on this — each route lazily
  // compiles just its own deck via `load-deck-flows.ts`.
  void import("./flow/dev-tools/index.js").then(
    async ({ runDevDiagnostics }) => {
      try {
        await runDevDiagnostics(await loadDeckPackages());
      } catch (error: unknown) {
        console.warn("Flow developer diagnostics failed to run", error);
      }
    },
    (error: unknown) => {
      console.warn("Flow developer diagnostics failed to load", error);
    },
  );
}
