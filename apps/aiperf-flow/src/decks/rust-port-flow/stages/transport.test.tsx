/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Import the implementation via its explicit `.tsx` path: a stub `transport.ts` still exists in
// this directory until the integration agent removes it, and `bundler` resolution of a
// `./transport.js` specifier would otherwise pick the `.ts` stub over this file.
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineCanvas } from "../../../interactive/index.js";
import { buildZoomTree } from "../stage.js";
import { transportStage, transportFlowSteps } from "./transport.tsx";

const LEAF_IDS = ["transport-http", "transport-grpc", "transport-dry-run", "transport-dynosim"];

describe("transportStage (Stage 6 — Transport seam)", () => {
  it("is the order-6 yellow Transport seam stage", () => {
    expect(transportStage.id).toBe("transport");
    expect(transportStage.order).toBe(6);
    expect(transportStage.label).toBe("Transport seam");
    expect(transportStage.tone).toBe("yellow");
  });

  it("wires each level-1 target node to a drillable leaf of the same id", () => {
    const sub = transportStage.subgraph;
    expect(sub).toBeDefined();
    // The four targets are declared drillable children...
    expect(sub!.children).toEqual(LEAF_IDS);
    // ...each child id is a real level-1 node id (so a click can drill it)...
    const nodeIds = new Set(sub!.nodes.map((n) => n.id));
    for (const leafId of LEAF_IDS) {
      expect(nodeIds.has(leafId)).toBe(true);
    }
    // ...and each has a registered level-2 subgraph.
    for (const leafId of LEAF_IDS) {
      expect(transportStage.leaves?.[leafId]).toBeDefined();
    }
  });

  it("cites the verified two-trait seam + four-target source anchors", () => {
    const paths = (transportStage.evidence ?? []).map((e) => e.path);
    expect(paths).toContain("runtime/src/engine/turn_execution.rs:74"); // trait WorkerSink
    expect(paths).toContain("runtime/src/engine/turn_execution.rs:136"); // trait ExecutionSinkBuilder
    expect(paths).toContain("runtime/src/transport/http/sink.rs:164"); // TransportSink
    expect(paths).toContain("runtime/src/transport/grpc/sink.rs:102"); // GrpcTransportSink
    expect(paths).toContain("runtime/src/dynosim.rs:594"); // SteppableEngine
  });

  it("renders the level-1 seam: Dispatcher, the two-trait boundary, and all four sink targets", () => {
    const sub = transportStage.subgraph!;
    render(<PipelineCanvas nodes={sub.nodes} edges={sub.edges} height={360} />);
    expect(screen.getByText("Rc<dyn Dispatcher>")).toBeInTheDocument();
    expect(screen.getByText("Two-trait seam")).toBeInTheDocument();
    expect(screen.getByText("WorkerSink + ExecutionSinkBuilder")).toBeInTheDocument();
    expect(screen.getByText("TransportSink")).toBeInTheDocument();
    expect(screen.getByText("GrpcTransportSink")).toBeInTheDocument();
    expect(screen.getByText("DryRunTransportFactoryV2")).toBeInTheDocument();
    expect(screen.getByText("SteppableEngine")).toBeInTheDocument();
  });

  it("renders the HTTP leaf (level-2) with its real builder + streaming sink impls", () => {
    const leaf = transportStage.leaves!["transport-http"]!;
    expect(leaf.label).toMatch(/hyper/i);
    render(<PipelineCanvas nodes={leaf.nodes} edges={leaf.edges} height={240} />);
    expect(screen.getByText("HttpSinkBuilder")).toBeInTheDocument();
    expect(screen.getAllByText("ExecutionSinkBuilder").length).toBeGreaterThan(0);
    expect(screen.getByText("TransportSink")).toBeInTheDocument();
    expect(screen.getByText("SSE token stream")).toBeInTheDocument();
  });

  it("builds into the shared ZoomTree with the transport node + its four leaves as keys", () => {
    const tree = buildZoomTree([transportStage]);
    expect(tree.transport).toBeDefined();
    expect(tree.transport?.children).toEqual(LEAF_IDS);
    for (const leafId of LEAF_IDS) {
      expect(tree[leafId]).toBeDefined();
    }
  });

  it("supplies a FlowStep fragment whose active ids match seam node ids and captions name real types", () => {
    const validIds = new Set([
      ...transportStage.subgraph!.nodes.map((n) => n.id),
      ...Object.values(transportStage.leaves!).flatMap((l) => l.nodes.map((n) => n.id)),
      ...LEAF_IDS,
    ]);
    for (const step of transportFlowSteps) {
      expect(validIds.has(step.nodeId)).toBe(true);
    }
    const seamCaption = transportFlowSteps.find((s) => s.nodeId === "transport__seam")?.caption ?? "";
    expect(seamCaption).toMatch(/ExecutionSinkBuilder/);
    expect(seamCaption).toMatch(/WorkerSink/);
    const httpCaption = transportFlowSteps.find((s) => s.nodeId === "transport-http")?.caption ?? "";
    expect(httpCaption).toMatch(/TransportSink/);
    expect(httpCaption).toMatch(/TTFT/);
  });
});
