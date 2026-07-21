/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import { DECK_MANIFEST } from "./core/deck-registry";
import { ThemeProvider } from "./core/ui";
import { Hub } from "./routes/Hub";
import { DeckRoute } from "./routes/DeckRoute";

export function App() {
  return (
    <ThemeProvider>
      <HashRouter>
        <Routes>
          <Route path="/" element={<Hub />} />
          {DECK_MANIFEST.map((entry) => (
            <Route key={entry.id} path={entry.route} element={<DeckRoute />} />
          ))}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HashRouter>
    </ThemeProvider>
  );
}
