/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { BrowserRouter, Route, Routes } from "react-router-dom";
import { DeckRoute } from "./deck/DeckRoute.js";
import { AiperfGraphEngineDeck } from "./decks/aiperf-graph-engine/AiperfGraphEngineDeck.js";
import { AiperfMetricsAccumulatorDeck } from "./decks/aiperf-metrics-accumulator/AiperfMetricsAccumulatorDeck.js";
import { CanvasRepoLayoutDeck } from "./decks/canvas-repo-layout/CanvasRepoLayoutDeck.js";
import { CellularAlgorithmWorkbookDeck } from "./decks/cellular-algorithm-workbook/CellularAlgorithmWorkbookDeck.js";
import { CellularArchitectureDeck } from "./decks/cellular-architecture/CellularArchitectureDeck.js";
import { ClaudeCodeSubagentStepperDeck } from "./decks/claude-code-subagent-stepper/ClaudeCodeSubagentStepperDeck.js";
import { DynosimOfflineFlowDeck } from "./decks/dynosim-offline-flow/DynosimOfflineFlowDeck.js";
import { GraphFanInDeck } from "./decks/graph-fan-in/GraphFanInDeck.js";
import { GraphStepEmitStrategyDeck } from "./decks/graph-step-emit-strategy/GraphStepEmitStrategyDeck.js";
import { GraphSubsystemOverviewDeck } from "./decks/graph-subsystem-overview/GraphSubsystemOverviewDeck.js";
import { MockerClockInversionDeck } from "./decks/mocker-clock-inversion/MockerClockInversionDeck.js";
import { MockServerArchitectureDeck } from "./decks/mock-server-architecture/MockServerArchitectureDeck.js";
import { OfflineCosimulationDeck } from "./decks/offline-cosimulation/OfflineCosimulationDeck.js";
import { RustAiperfArchitectureDeck } from "./decks/rust-aiperf-architecture/RustAiperfArchitectureDeck.js";
import { RustArchitectureInternalsDeck } from "./decks/rust-architecture-internals/RustArchitectureInternalsDeck.js";
import { RustPortFlowDeck } from "./decks/rust-port-flow/RustPortFlowDeck.js";
import { RustPortWhyDeck } from "./decks/rust-port-why/RustPortWhyDeck.js";
import { SegmentPoolsDeck } from "./decks/segment-pools/SegmentPoolsDeck.js";
import { SlurmArchitectureDeck } from "./decks/slurm-architecture/SlurmArchitectureDeck.js";
import { SlurmExplainedStepByStepDeck } from "./decks/slurm-explained-step-by-step/SlurmExplainedStepByStepDeck.js";
import { StepDispatchEmitSystemDeck } from "./decks/step-dispatch-emit-system/StepDispatchEmitSystemDeck.js";
import { UpcomingAsyncDataflowDeck } from "./decks/upcoming-async-dataflow/UpcomingAsyncDataflowDeck.js";
import { VeloInAiperfDeck } from "./decks/velo-in-aiperf/VeloInAiperfDeck.js";
import { WekaIngestPipelineDeck } from "./decks/weka-ingest-pipeline/WekaIngestPipelineDeck.js";
import { WekaRuntimeStepperDeck } from "./decks/weka-runtime-stepper/WekaRuntimeStepperDeck.js";
import { WekaSegmentStoreDeck } from "./decks/weka-segment-store/WekaSegmentStoreDeck.js";
import { WekaTimingCausalityDeck } from "./decks/weka-timing-causality/WekaTimingCausalityDeck.js";
import { WekaTimingTransformsDeck } from "./decks/weka-timing-transforms/WekaTimingTransformsDeck.js";
import { WekaTimingTransformsInteractiveDeck } from "./decks/weka-timing-transforms-interactive/WekaTimingTransformsInteractiveDeck.js";
import { WekaTrieBuildDeck } from "./decks/weka-trie-build/WekaTrieBuildDeck.js";
import { Home } from "./routes/Home.js";

export function App(): React.JSX.Element {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/segment-pools" element={<SegmentPoolsDeck />} />
        <Route path="/aiperf-graph-engine" element={<AiperfGraphEngineDeck />} />
        <Route path="/aiperf-metrics-accumulator" element={<AiperfMetricsAccumulatorDeck />} />
        <Route path="/canvas-repo-layout" element={<CanvasRepoLayoutDeck />} />
        <Route path="/cellular-algorithm-workbook" element={<CellularAlgorithmWorkbookDeck />} />
        <Route path="/cellular-architecture" element={<CellularArchitectureDeck />} />
        <Route path="/claude-code-subagent-stepper" element={<ClaudeCodeSubagentStepperDeck />} />
        <Route path="/dynosim-offline-flow" element={<DynosimOfflineFlowDeck />} />
        <Route path="/graph-fan-in" element={<GraphFanInDeck />} />
        <Route path="/graph-step-emit-strategy" element={<GraphStepEmitStrategyDeck />} />
        <Route path="/graph-subsystem-overview" element={<GraphSubsystemOverviewDeck />} />
        <Route path="/mocker-clock-inversion" element={<MockerClockInversionDeck />} />
        <Route path="/mock-server-architecture" element={<MockServerArchitectureDeck />} />
        <Route path="/offline-cosimulation" element={<OfflineCosimulationDeck />} />
        <Route path="/rust-aiperf-architecture" element={<RustAiperfArchitectureDeck />} />
        <Route path="/rust-architecture-internals" element={<RustArchitectureInternalsDeck />} />
        <Route path="/rust-port-flow" element={<RustPortFlowDeck />} />
        <Route path="/rust-port-why" element={<RustPortWhyDeck />} />
        <Route path="/slurm-architecture" element={<SlurmArchitectureDeck />} />
        <Route path="/slurm-explained-step-by-step" element={<SlurmExplainedStepByStepDeck />} />
        <Route path="/step-dispatch-emit-system" element={<StepDispatchEmitSystemDeck />} />
        <Route path="/upcoming-async-dataflow" element={<UpcomingAsyncDataflowDeck />} />
        <Route path="/velo-in-aiperf" element={<VeloInAiperfDeck />} />
        <Route path="/weka-ingest-pipeline" element={<WekaIngestPipelineDeck />} />
        <Route path="/weka-runtime-stepper" element={<WekaRuntimeStepperDeck />} />
        <Route path="/weka-segment-store" element={<WekaSegmentStoreDeck />} />
        <Route path="/weka-timing-causality" element={<WekaTimingCausalityDeck />} />
        <Route path="/weka-timing-transforms" element={<WekaTimingTransformsDeck />} />
        <Route path="/weka-timing-transforms-interactive" element={<WekaTimingTransformsInteractiveDeck />} />
        <Route path="/weka-trie-build" element={<WekaTrieBuildDeck />} />
        <Route path="/:deckId" element={<DeckRoute />} />
      </Routes>
    </BrowserRouter>
  );
}
