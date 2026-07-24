/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `docs/canvases/rust-architecture-internals.canvas.tsx` (a real, hand-authored Cursor
//! Canvas) onto aiperf-flow's component vocabulary. Unlike `SegmentPoolsDeck`, the source is a
//! single scrolling view (no page-id union / PageTabs), so this composes its thirteen sections
//! in order down one page. The global detail level is lifted here and passed to each section;
//! per-section view toggles are local `useState` inside their own files.

import { useState } from "react";
import { TopBar } from "../../shell/TopBar.js";
import { Divider } from "../../layout/Divider.js";
import type { Detail } from "./parts.js";
import { HeroSection } from "./HeroSection.js";
import { ProcessBoundarySection } from "./ProcessBoundarySection.js";
import { ExecutionLifecycleSection } from "./ExecutionLifecycleSection.js";
import { CompositionConstellationSection } from "./CompositionConstellationSection.js";
import { SeamTheaterSection } from "./SeamTheaterSection.js";
import { TransportDeepDiveSection } from "./TransportDeepDiveSection.js";
import { WorkloadForkSection } from "./WorkloadForkSection.js";
import { GraphDeepDiveSection } from "./GraphDeepDiveSection.js";
import { WorkerTopologySection } from "./WorkerTopologySection.js";
import { CellularDeepDiveSection } from "./CellularDeepDiveSection.js";
import { MeasurementRiverSection } from "./MeasurementRiverSection.js";
import { MetricsDeepDiveSection } from "./MetricsDeepDiveSection.js";
import { MechanismsSection } from "./MechanismsSection.js";

/**
 * Single-view deck: "Inside Rust AIPerf". Thirteen composed sections tracing one native run
 * from authored Config v2 through worker-local transport, measurement, and final artifacts.
 */
export function RustArchitectureInternalsDeck(): React.JSX.Element {
  const [detail, setDetail] = useState<Detail>("engineering");

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Rust Architecture Internals" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto min-h-full max-w-6xl 2xl:max-w-[1728px] bg-surface-page px-10 py-8">
          <HeroSection detail={detail} onDetailChange={setDetail} />
          <Divider className="my-9" />
          <ProcessBoundarySection detail={detail} />
          <Divider className="my-9" />
          <ExecutionLifecycleSection detail={detail} />
          <Divider className="my-9" />
          <CompositionConstellationSection detail={detail} />
          <Divider className="my-9" />
          <SeamTheaterSection detail={detail} />
          <Divider className="my-9" />
          <TransportDeepDiveSection detail={detail} />
          <Divider className="my-9" />
          <WorkloadForkSection detail={detail} />
          <Divider className="my-9" />
          <GraphDeepDiveSection detail={detail} />
          <Divider className="my-9" />
          <WorkerTopologySection detail={detail} />
          <Divider className="my-9" />
          <CellularDeepDiveSection detail={detail} />
          <Divider className="my-9" />
          <MeasurementRiverSection detail={detail} />
          <Divider className="my-9" />
          <MetricsDeepDiveSection detail={detail} />
          <Divider className="my-9" />
          <MechanismsSection detail={detail} />
        </div>
      </div>
    </div>
  );
}

export default RustArchitectureInternalsDeck;
