/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { StoryPage } from "./StoryPage.js";
import { AtlasPage } from "./AtlasPage.js";
import { AbilitiesPage } from "./AbilitiesPage.js";

type CellularPageId = "story" | "atlas" | "abilities";

const PAGES: ReadonlyArray<PageTabDefinition<CellularPageId>> = [
  { id: "story", label: "Story" },
  { id: "atlas", label: "Recipe atlas" },
  { id: "abilities", label: "Abilities" },
];

/**
 * Ports `docs/canvases/cellular-architecture.canvas.tsx` (a real Cursor Canvas) onto aiperf-flow's
 * component vocabulary. The single continuous storyboard becomes three composed pages switched via
 * `PageTabs`: the 20-page `useStepSimulator` story walkthrough, the interactive recipe/route atlas
 * explorer, and the ability matrix.
 */
export function CellularArchitectureDeck(): React.JSX.Element {
  const [page, setPage] = useState<CellularPageId>("story");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Cellular Architecture" />
      <div className="border-b border-stroke-secondary bg-surface-page px-8 py-3">
        <PageTabs pages={PAGES} current={page} onChange={setPage} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          {page === "story" && <StoryPage />}
          {page === "atlas" && <AtlasPage />}
          {page === "abilities" && <AbilitiesPage />}
        </div>
      </div>
    </div>
  );
}
