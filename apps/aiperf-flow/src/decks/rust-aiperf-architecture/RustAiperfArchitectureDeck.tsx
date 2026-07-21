/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { inkClassName } from "../../theme/tokens.js";
import { SystemPage } from "./SystemPage.js";
import { ProcessesPage } from "./ProcessesPage.js";
import { RuntimePage } from "./RuntimePage.js";
import { ProtocolPage } from "./ProtocolPage.js";
import { ScheduledPage } from "./ScheduledPage.js";
import { GraphPage } from "./GraphPage.js";
import { EndpointsPage } from "./EndpointsPage.js";
import { MetricsPage } from "./MetricsPage.js";
import { CellularPage } from "./CellularPage.js";
import { FeaturesPage } from "./FeaturesPage.js";
import { SeamsPage } from "./SeamsPage.js";

type RustAiperfPageId =
  | "system"
  | "processes"
  | "runtime"
  | "protocol"
  | "scheduled"
  | "graph"
  | "endpoints"
  | "metrics"
  | "cellular"
  | "features"
  | "seams";

interface PageMeta extends PageTabDefinition<RustAiperfPageId> {
  hint: string;
}

const PAGES: ReadonlyArray<PageMeta> = [
  { id: "system", label: "1 · System", hint: "product landscape" },
  { id: "processes", label: "2 · Processes", hint: "crates and boundaries" },
  { id: "runtime", label: "3 · Runtime", hint: "one request end-to-end" },
  { id: "protocol", label: "4 · Protocol", hint: "one child lifecycle" },
  { id: "scheduled", label: "5 · Scheduled", hint: "paced workload path" },
  { id: "graph", label: "6 · Graph", hint: "trace replay path" },
  { id: "endpoints", label: "7 · Endpoints", hint: "dialect preparation" },
  { id: "metrics", label: "8 · Metrics", hint: "measurement and exports" },
  { id: "cellular", label: "9 · Cellular", hint: "multi-process scale" },
  { id: "features", label: "10 · Builds", hint: "feature composition" },
  { id: "seams", label: "11 · Seams", hint: "extension internals" },
];

/**
 * Ports the Rust AIPerf architecture Cursor canvas onto aiperf-flow's component vocabulary:
 * eleven in-deck pages switched via `PageTabs`, each an intro paragraph, one React Flow node/edge
 * diagram, a Grid of `Callout` cards, and an evidence row of source anchors. Four zoom levels,
 * from product boundaries to the hot-path seams, grounded in the current workspace and feature graph.
 */
export function RustAiperfArchitectureDeck(): React.JSX.Element {
  const [page, setPage] = useState<RustAiperfPageId>("system");
  const currentHint = PAGES.find((p) => p.id === page)?.hint;

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Rust AIPerf Architecture" />
      <div className="border-b border-stroke-secondary bg-surface-page py-4">
        <div className="mx-auto max-w-6xl px-10">
          <h1 className="text-xl font-bold">Rust AIPerf architecture</h1>
          <p className={`mt-1 max-w-4xl text-sm ${inkClassName("secondary")}`}>
            Four zoom levels, from product boundaries to the hot-path seams. Grounded in the current workspace code
            and Cargo feature graph.
          </p>
          <div className="mt-3">
            <PageTabs pages={PAGES} current={page} onChange={setPage} />
          </div>
          {currentHint !== undefined && (
            <div className={`mt-2 text-xs uppercase tracking-wide ${inkClassName("tertiary")}`}>{currentHint}</div>
          )}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          {page === "system" && <SystemPage />}
          {page === "processes" && <ProcessesPage />}
          {page === "runtime" && <RuntimePage />}
          {page === "protocol" && <ProtocolPage />}
          {page === "scheduled" && <ScheduledPage />}
          {page === "graph" && <GraphPage />}
          {page === "endpoints" && <EndpointsPage />}
          {page === "metrics" && <MetricsPage />}
          {page === "cellular" && <CellularPage />}
          {page === "features" && <FeaturesPage />}
          {page === "seams" && <SeamsPage />}
          <div className={`mt-8 border-t border-stroke-secondary pt-3 text-xs ${inkClassName("tertiary")}`}>
            Reading convention: solid edges are primary paths; dashed edges are optional, delegated, or feature-gated.
            Source buttons open the implementation files used to anchor each view.
          </div>
        </div>
      </div>
    </div>
  );
}
