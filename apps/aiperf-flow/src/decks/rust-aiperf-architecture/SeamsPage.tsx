/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, RoundNode, MiniArrow } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of AIPerf's extension seams: a frozen registry at the center, ringed by
// the trait/impl substitution points that stay open around one single-run core. Ported from SeamsView.

/** SeamsView: compile-time composition and execution-path substitution around one run core. */
export function SeamsPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Extension internals">
        The architecture stays open in two directions: compile-time product composition at startup, and transport/clock
        substitution on the execution path. Cellular mode scales around the same single-run core.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · REGISTRY SEAMS",
          title: "Where does it stay open?",
          body: "AIPerfRegistry freezes once per image; traits substitute around one run core.",
        }}
        spokes={[
          {
            accent: "purple",
            badge: 1,
            title: "Extension registration",
            diagram: (
              <Diagram>
                <NodeChip>AIPerfExtension</NodeChip>
                <MiniArrow />
                <NodeChip accent>AIPerfRegistry</NodeChip>
              </Diagram>
            ),
            children: "AIPerfExtension registers transactionally into AIPerfRegistry, frozen once per executable image.",
          },
          {
            accent: "green",
            badge: 2,
            title: "Datasets & endpoints",
            diagram: (
              <Diagram>
                <NodeChip accent>datasets</NodeChip>
                <MiniArrow />
                <NodeChip>endpoints</NodeChip>
              </Diagram>
            ),
            children: "Loaders + samplers and endpoint body/response factories resolve from the frozen registry.",
          },
          {
            accent: "blue",
            badge: 3,
            title: "RequestSink<R>",
            diagram: (
              <Diagram>
                <NodeChip accent>RequestSink&lt;R&gt;</NodeChip>
                <MiniArrow />
                <NodeChip>HTTP · gRPC</NodeChip>
              </Diagram>
            ),
            children: "Transport-neutral orchestration drives a transport-native R: HTTP/SSE Hyper, gRPC Tonic, DynoSim.",
          },
          {
            accent: "cyan",
            badge: 4,
            title: "Clock injection",
            diagram: (
              <Diagram>
                <NodeChip accent>Clock</NodeChip>
                <MiniArrow />
                <NodeChip>Real | Sim</NodeChip>
              </Diagram>
            ),
            children: "The executor substitutes RealClock or SimClock without touching the workload/graph orchestration.",
          },
          {
            accent: "yellow",
            badge: 5,
            title: "RequestObserver",
            diagram: (
              <Diagram>
                <NodeChip>HTTP · gRPC</NodeChip>
                <MiniArrow />
                <NodeChip accent>RequestObserver</NodeChip>
              </Diagram>
            ),
            children: "Every transport folds into one RequestObserver event stream for measurement; registries stay independent.",
          },
          {
            accent: "orange",
            badge: 6,
            title: "Cellular scaling",
            diagram: (
              <Diagram>
                <RoundNode>0</RoundNode>
                <RoundNode accent>1</RoundNode>
                <RoundNode>N</RoundNode>
                <MiniArrow />
                <NodeChip>controller merge</NodeChip>
              </Diagram>
            ),
            children: "The controller slices budgets to cell 0..N (ordinary execute paths); every cell ships its terminal partition directly to the controller merge.",
          },
          {
            accent: "red",
            badge: 7,
            title: "Controller commit",
            diagram: (
              <Diagram>
                <NodeChip>cells</NodeChip>
                <MiniArrow />
                <NodeChip accent>final report</NodeChip>
              </Diagram>
            ),
            children: "Cross-process cells use the opt-in Velo feature; hierarchy requests are refused before startup and the controller commits the final report.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Extension registry", path: "rust/aiperf/src/extensions/mod.rs" },
          { label: "Cell controller", path: "rust/aiperf/src/runner_protocol/cellular_controller.rs" },
          { label: "Observer implementation", path: "rust/loadgen-core/src/observer.rs" },
        ]}
      />
    </div>
  );
}
