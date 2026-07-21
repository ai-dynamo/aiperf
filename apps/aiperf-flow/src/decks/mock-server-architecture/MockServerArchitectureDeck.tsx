/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Top-level "Mock Foundry" deck: ports the hand-authored Cursor canvas
//! `docs/canvases/mock-server-architecture.canvas.tsx` onto aiperf-flow's component vocabulary.
//! The canvas's ten chapters become ten `PageTabs` pages, each synthesizing that chapter's slice
//! of the audited 64-page catalog into an interactive React Flow signature walkthrough plus a
//! verbatim source/proof/invariant table.

import { useState } from "react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { CHAPTERS, type ChapterId } from "./catalog.js";
import { OrientationPage } from "./OrientationPage.js";
import { IngressPage } from "./IngressPage.js";
import { LlmProtocolsPage } from "./LlmProtocolsPage.js";
import { SpecializedPage } from "./SpecializedPage.js";
import { GrpcPage } from "./GrpcPage.js";
import { TimingPage } from "./TimingPage.js";
import { SchedulerPage } from "./SchedulerPage.js";
import { SemanticsPage } from "./SemanticsPage.js";
import { ObservabilityPage } from "./ObservabilityPage.js";
import { ProofPage } from "./ProofPage.js";

const PAGES: ReadonlyArray<PageTabDefinition<ChapterId>> = CHAPTERS.map((chapter) => ({
  id: chapter.id,
  label: chapter.short,
}));

const CHAPTER_COMPONENT: Record<ChapterId, () => React.JSX.Element> = {
  orientation: OrientationPage,
  ingress: IngressPage,
  llm: LlmProtocolsPage,
  specialized: SpecializedPage,
  grpc: GrpcPage,
  timing: TimingPage,
  scheduler: SchedulerPage,
  semantics: SemanticsPage,
  observability: ObservabilityPage,
  proof: ProofPage,
};

/**
 * The Mock Foundry deck — one continuous cutaway of `aiperf-mock-server` across ten chapter tabs.
 * Exported top-level component for routing integration.
 */
export function MockServerArchitectureDeck(): React.JSX.Element {
  const [page, setPage] = useState<ChapterId>("orientation");
  const Active = CHAPTER_COMPONENT[page];

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Mock Foundry — mock-server architecture" />
      <div className="border-b border-stroke-secondary bg-surface-page px-8 py-3">
        <PageTabs pages={PAGES} current={page} onChange={setPage} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <Active />
        </div>
      </div>
    </div>
  );
}
