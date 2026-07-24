/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineCanvas, ZoomStage } from "../../../interactive/index.js";
import { OVERVIEW_ID, buildZoomTree } from "../stage.js";
import { runtimeStage, runtimeSteps } from "./runtime.js";

/** A minimal deck-shaped harness: ZoomStage over just this stage, each level rendered as a canvas
 *  whose node clicks drill down — mirrors how RustPortFlowDeck wires PipelineCanvas to ctx.drill. */
function Harness(): React.JSX.Element {
  return (
    <ZoomStage tree={buildZoomTree([runtimeStage])} rootId={OVERVIEW_ID}>
      {(ctx) => (
        <div>
          <p>level:{ctx.level}</p>
          <PipelineCanvas
            nodes={ctx.node.nodes}
            edges={ctx.node.edges}
            onNodeClick={(id) => ctx.drill(id)}
          />
        </div>
      )}
    </ZoomStage>
  );
}

describe("runtimeStage", () => {
  it("declares the spine-1 metadata the overview lays out by", () => {
    expect(runtimeStage.id).toBe("runtime");
    expect(runtimeStage.order).toBe(1);
    expect(runtimeStage.label).toBe("Runtime & self-exec");
    expect(runtimeStage.tone).toBe("blue");
  });

  it("wires the level-1 self-exec spine with real protocol-v2 type names", () => {
    const titles = runtimeStage.subgraph!.nodes.map((n) => (n.data as { title: string }).title);
    expect(titles).toEqual(
      expect.arrayContaining([
        "aiperf-cli",
        "EnvelopeV2",
        "aiperf --execute",
        "Coordinator",
        "RunTerminalV2",
        "Three orthogonal seams",
      ]),
    );
    // The two drillable nodes are exactly the two leaf keys.
    expect(runtimeStage.subgraph!.children).toEqual(["runtime_selfexec", "runtime_seams"]);
    expect(Object.keys(runtimeStage.leaves ?? {})).toEqual(["runtime_selfexec", "runtime_seams"]);
  });

  it("cites real, verified rust file:line source anchors", () => {
    const byLabel = new Map((runtimeStage.evidence ?? []).map((e) => [e.label, e.path]));
    expect(byLabel.get("struct EnvelopeV2")).toBe("runtime/src/engine/protocol_v2.rs:115");
    expect(byLabel.get("struct RunTerminalV2")).toBe("runtime/src/engine/protocol_v2.rs:1058");
    expect(byLabel.get('EXECUTE_FLAG "--execute"')).toBe("cli/src/execute_mode.rs:49");
    expect(byLabel.get("Coordinator::handle (composition root)")).toBe(
      "runtime/src/engine/coordinator.rs:114",
    );
    // The three orthogonal seams each have a trait anchor.
    expect(byLabel.get("trait Clock — Time seam")).toBe("runtime/src/clock/clock.rs:12");
    expect(byLabel.get("trait WorkerSink — Transport seam")).toBe(
      "runtime/src/engine/turn_execution.rs:74",
    );
    expect(byLabel.get("trait Workload — Workload seam")).toBe("runtime/src/scheduled.rs:1115");
  });

  it("supplies a play fragment whose steps traverse real level-1 node ids in order", () => {
    const nodeIds = new Set(runtimeStage.subgraph!.nodes.map((n) => n.id));
    for (const step of runtimeSteps) {
      expect(nodeIds.has(step.nodeId)).toBe(true);
      expect(step.caption.length).toBeGreaterThan(0);
    }
    expect(runtimeSteps[0]!.nodeId).toBe("cli");
    expect(runtimeSteps.at(-1)!.nodeId).toBe("terminal");
    expect(runtimeSteps.map((s) => s.caption).join(" ")).toContain("OperationV2::Execute");
  });

  it("renders the level-1 canvas naming EnvelopeV2 and the Coordinator composition root", () => {
    render(<Harness />);
    // Overview shows the single stage card; drill into it.
    fireEvent.click(screen.getByText("Runtime & self-exec"));
    expect(screen.getByText("level:1")).toBeInTheDocument();
    expect(screen.getByText("EnvelopeV2")).toBeInTheDocument();
    expect(screen.getByText("Coordinator")).toBeInTheDocument();
    expect(screen.getByText("RunTerminalV2")).toBeInTheDocument();
  });

  it("drills level-2 into the three orthogonal seams (Clock / WorkerSink / Workload)", () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("Runtime & self-exec"));
    fireEvent.click(screen.getByText("Three orthogonal seams"));
    expect(screen.getByText("level:2")).toBeInTheDocument();
    expect(screen.getByText(/trait Clock/)).toBeInTheDocument();
    expect(screen.getByText(/trait WorkerSink/)).toBeInTheDocument();
    expect(screen.getByText(/trait Workload/)).toBeInTheDocument();
  });

  it("drills level-2 into the self-exec stdio handshake (run_once ⇄ --execute child)", () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("Runtime & self-exec"));
    fireEvent.click(screen.getByText("aiperf --execute"));
    expect(screen.getByText("level:2")).toBeInTheDocument();
    expect(screen.getByText("execute::run_once")).toBeInTheDocument();
    expect(screen.getByText("aiperf --execute child")).toBeInTheDocument();
  });
});
