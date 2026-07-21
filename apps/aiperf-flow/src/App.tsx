/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { BrowserRouter, Route, Routes } from "react-router-dom";
import { DeckRoute } from "./deck/DeckRoute.js";

export function App(): React.JSX.Element {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/:deckId" element={<DeckRoute />} />
      </Routes>
    </BrowserRouter>
  );
}
