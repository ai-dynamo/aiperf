/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { OverviewPage } from "./OverviewPage.js";
import { LaunchPage } from "./LaunchPage.js";
import { SeamsPage } from "./SeamsPage.js";
import { LoopPage } from "./LoopPage.js";
import { DispatchPage } from "./DispatchPage.js";
import { ParityPage } from "./ParityPage.js";
import { EnginePage } from "./EnginePage.js";
import { DetailToggle, type Level } from "./shared.js";

type DynosimOfflinePageId = "overview" | "launch" | "seams" | "loop" | "dispatch" | "parity" | "engine";

const PAGES: ReadonlyArray<PageTabDefinition<DynosimOfflinePageId>> = [
  { id: "overview", label: "Overview" },
  { id: "launch", label: "Launch" },
  { id: "seams", label: "Architecture" },
  { id: "loop", label: "Loop" },
  { id: "dispatch", label: "Dispatch" },
  { id: "parity", label: "Parity" },
  { id: "engine", label: "Engine" },
];

/**
 * Ports `docs/canvases/dynosim-offline-flow.canvas.tsx` (a real, hand-authored Cursor Canvas)
 * onto aiperf-flow's component vocabulary: seven in-deck pages switched via `PageTabs`, grounded
 * in `aiperf-cli profile -> load/yaml -> aiperf --execute -> RunnerApplication ->
 * offline_execution -> dynosim.rs -> graph/runtime.rs -> parity`. A shared executive / developer
 * / maintainer detail toggle (ported from the source canvas) controls how much implementation
 * detail each page's captions surface.
 */
export function DynosimOfflineFlowDeck(): React.JSX.Element {
  const [page, setPage] = useState<DynosimOfflinePageId>("overview");
  const [level, setLevel] = useState<Level>("developer");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Dynosim Offline" actions={<DetailToggle level={level} onChange={setLevel} />} />
      <div className="border-b border-stroke-secondary bg-surface-page py-3">
        <div className="mx-auto max-w-6xl px-10">
          <PageTabs pages={PAGES} current={page} onChange={setPage} />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          {page === "overview" && <OverviewPage level={level} />}
          {page === "launch" && <LaunchPage level={level} />}
          {page === "seams" && <SeamsPage level={level} />}
          {page === "loop" && <LoopPage level={level} />}
          {page === "dispatch" && <DispatchPage level={level} />}
          {page === "parity" && <ParityPage level={level} />}
          {page === "engine" && <EnginePage level={level} />}
        </div>
      </div>
    </div>
  );
}
