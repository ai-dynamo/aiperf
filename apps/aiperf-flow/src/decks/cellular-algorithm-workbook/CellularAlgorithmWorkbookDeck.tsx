/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `docs/canvases/cellular-algorithm-workbook.canvas.tsx` (a large, real Cursor canvas) onto
//! aiperf-flow's component vocabulary. The source's three host "modes" (workbook / compose /
//! decisions) become three `PageTabs` pages, each an independent component; the shared
//! source-grounded algorithm data + pure routing logic live in `data.ts`.

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { inkClassName } from "../../theme/tokens.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { ALGORITHMS } from "./data.js";
import { WorkbookPage } from "./WorkbookPage.js";
import { ComposePage } from "./ComposePage.js";
import { DecisionsPage } from "./DecisionsPage.js";

type WorkbookPageId = "workbook" | "compose" | "decisions";

const PAGES: ReadonlyArray<PageTabDefinition<WorkbookPageId>> = [
  { id: "workbook", label: "Workbook" },
  { id: "compose", label: "Compose" },
  { id: "decisions", label: "Decisions" },
];

/** Top-level deck component for the cellular algorithm workbook. */
export function CellularAlgorithmWorkbookDeck(): React.JSX.Element {
  const [page, setPage] = useState<WorkbookPageId>("workbook");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Cellular Algorithm Workbook" />
      <div className="border-b border-stroke-secondary bg-surface-page px-8 py-3">
        <div className="mb-2">
          <Eyebrow>Cellular execution · algorithm workbook</Eyebrow>
          <h1 className={`text-2xl font-semibold ${inkClassName("primary")}`}>Reason from gate to artifact</h1>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Study how a run is admitted, partitioned, executed, captured, and merged across cells.
            {" "}
            {ALGORITHMS.length} source-grounded algorithms.
          </p>
        </div>
        <PageTabs pages={PAGES} current={page} onChange={setPage} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          {page === "workbook" && <WorkbookPage />}
          {page === "compose" && <ComposePage />}
          {page === "decisions" && <DecisionsPage />}
        </div>
      </div>
    </div>
  );
}
