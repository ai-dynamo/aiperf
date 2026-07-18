/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { compiledDeckPackages } from "./core/load-deck-flows";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

if (import.meta.env.DEV) {
  void import("./flow/dev-tools/index.js").then(
    async ({ runDevDiagnostics }) => {
      try {
        await runDevDiagnostics(compiledDeckPackages());
      } catch (error: unknown) {
        console.warn("Flow developer diagnostics failed to run", error);
      }
    },
    (error: unknown) => {
      console.warn("Flow developer diagnostics failed to load", error);
    },
  );
}
