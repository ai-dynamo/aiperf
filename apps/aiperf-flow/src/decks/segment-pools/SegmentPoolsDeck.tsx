/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { OverviewPage } from "./OverviewPage.js";
import { PoolPage } from "./PoolPage.js";
import { PayloadsPage } from "./PayloadsPage.js";
import { BodyPlanPage } from "./BodyPlanPage.js";
import { PrefixPage } from "./PrefixPage.js";
import { DispatchPage } from "./DispatchPage.js";

type SegmentPoolsPageId =
  | "overview"
  | "pool"
  | "payloads"
  | "bodyplan"
  | "prefix"
  | "dispatch";

const PAGES: ReadonlyArray<PageTabDefinition<SegmentPoolsPageId>> = [
  { id: "overview", label: "Overview" },
  { id: "pool", label: "Pool" },
  { id: "payloads", label: "Payloads" },
  { id: "bodyplan", label: "BodyPlan" },
  { id: "prefix", label: "Prefix" },
  { id: "dispatch", label: "Dispatch" },
];

/**
 * Ports `docs/canvases/segment-pools-and-body-plans.canvas.tsx` (a real, hand-authored Cursor
 * Canvas) onto aiperf-flow's component vocabulary: six in-deck pages switched via `PageTabs`,
 * each independently built from React Flow diagrams, stateful simulators, and prose primitives.
 */
export function SegmentPoolsDeck(): React.JSX.Element {
  const [page, setPage] = useState<SegmentPoolsPageId>("overview");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <div className="border-b border-stroke-secondary bg-surface-page px-8 py-4">
        <PageTabs pages={PAGES} current={page} onChange={setPage} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          {page === "overview" && <OverviewPage />}
          {page === "pool" && <PoolPage />}
          {page === "payloads" && <PayloadsPage />}
          {page === "bodyplan" && <BodyPlanPage />}
          {page === "prefix" && <PrefixPage />}
          {page === "dispatch" && <DispatchPage />}
        </div>
      </div>
    </div>
  );
}
