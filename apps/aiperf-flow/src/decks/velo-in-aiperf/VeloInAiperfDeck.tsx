/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { PageTabs, type PageTabDefinition } from "../../shell/PageTabs.js";
import { TopBar } from "../../shell/TopBar.js";
import { IndexPage } from "./IndexPage.js";
import { RadarPage } from "./RadarPage.js";
import { XrayPage } from "./XrayPage.js";
import { GatePage } from "./GatePage.js";
import { PressPage } from "./PressPage.js";
import { ScopePage } from "./ScopePage.js";
import { CourierPage } from "./CourierPage.js";
import { MergePage } from "./MergePage.js";
import { PhaserPage } from "./PhaserPage.js";
import { DatasetPage } from "./DatasetPage.js";
import { TreePage } from "./TreePage.js";

/** Union of every in-deck page: the constellation index plus the ten mechanism instruments. */
export type VeloPageId =
  | "index"
  | "radar"
  | "xray"
  | "gate"
  | "press"
  | "scope"
  | "courier"
  | "merge"
  | "phaser"
  | "dataset"
  | "tree";

const PAGES: ReadonlyArray<PageTabDefinition<VeloPageId>> = [
  { id: "index", label: "Index" },
  { id: "radar", label: "R · Radar" },
  { id: "xray", label: "X · X-ray" },
  { id: "gate", label: "G · Gate" },
  { id: "press", label: "P · Press" },
  { id: "scope", label: "H · Scope" },
  { id: "courier", label: "C · Courier" },
  { id: "merge", label: "M · Merge" },
  { id: "phaser", label: "Φ · Phaser" },
  { id: "dataset", label: "D · Dataset" },
  { id: "tree", label: "T · Tree" },
];

/**
 * Ports `docs/canvases/velo-in-aiperf.canvas.tsx` ("Velo mechanisms") onto aiperf-flow's
 * component vocabulary: a constellation index plus ten interactive instruments describing how
 * cellular identity, synchronization, distribution, and reduction cross AIPerf's Velo transport
 * plane. Each mechanism is one in-deck page switched via `PageTabs`; diagram content is authored
 * as real `@xyflow/react` node/edge graphs, interactivity via `useStepSimulator`/`useState`.
 */
export function VeloInAiperfDeck(): React.JSX.Element {
  const [page, setPage] = useState<VeloPageId>("index");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Velo Mechanisms" />
      <div className="border-b border-stroke-secondary bg-surface-page py-3">
        <div className="mx-auto max-w-6xl px-10">
          <PageTabs pages={PAGES} current={page} onChange={setPage} />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl bg-surface-page px-10 py-8">
          <ReactFlowProvider>
            {page === "index" && <IndexPage onSelect={setPage} />}
            {page === "radar" && <RadarPage />}
            {page === "xray" && <XrayPage />}
            {page === "gate" && <GatePage />}
            {page === "press" && <PressPage />}
            {page === "scope" && <ScopePage />}
            {page === "courier" && <CourierPage />}
            {page === "merge" && <MergePage />}
            {page === "phaser" && <PhaserPage />}
            {page === "dataset" && <DatasetPage />}
            {page === "tree" && <TreePage />}
          </ReactFlowProvider>
        </div>
      </div>
    </div>
  );
}
