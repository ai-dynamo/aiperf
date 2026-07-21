/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { OverviewPage } from "./OverviewPage.js";
import { InternalsPage } from "./InternalsPage.js";

type OfflineCosimulationPageId = "overview" | "internals";

const PAGES: ReadonlyArray<PageTabDefinition<OfflineCosimulationPageId>> = [
  { id: "overview", label: "Overview" },
  { id: "internals", label: "Internals" },
];

/**
 * Ports `offline-cosimulation.canvas.tsx` (a real, hand-authored Cursor Canvas) onto
 * aiperf-flow's component vocabulary: two in-deck pages switched via `PageTabs`. Socket-free
 * Dynamo execution through AIPerf's native measurement path — AIPerf owns orchestration, clock,
 * and measurement while a passive steppable engine steps in-process behind the `dynosim` feature.
 */
export function OfflineCosimulationDeck(): React.JSX.Element {
  const [page, setPage] = useState<OfflineCosimulationPageId>("overview");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar
        section="Offline Co-simulation"
        actions={
          <span className="rounded-md border border-stroke-secondary px-2 py-0.5 text-[11px] font-medium text-ink-tertiary shadow-sm">
            dynosim feature
          </span>
        }
      />
      <div className="border-b border-stroke-secondary bg-surface-page py-3">
        <div className="mx-auto max-w-6xl px-10">
          <div className="mb-2">
            <div className="text-sm font-semibold text-ink-primary">Offline co-simulation</div>
            <div className="text-xs text-ink-tertiary">
              Socket-free Dynamo execution through AIPerf&apos;s native measurement path
            </div>
          </div>
          <PageTabs pages={PAGES} current={page} onChange={setPage} />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          {page === "overview" && <OverviewPage />}
          {page === "internals" && <InternalsPage />}
        </div>
      </div>
    </div>
  );
}
