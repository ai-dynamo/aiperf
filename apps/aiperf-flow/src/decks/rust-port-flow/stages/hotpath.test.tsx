/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { hotPathStage, hotPathSteps, HotPathCards } from "./hotpath.js";
import { buildZoomTree } from "../stage.js";

describe("hotPathStage — StageDef", () => {
  it("is spine stage 7, red-toned, keyed 'hotpath'", () => {
    expect(hotPathStage.id).toBe("hotpath");
    expect(hotPathStage.order).toBe(7);
    expect(hotPathStage.tone).toBe("red");
    expect(hotPathStage.label).toBe("Request hot-path");
  });

  it("lays out the six-step hot-path spine as level-1 card nodes", () => {
    const ids = hotPathStage.subgraph!.nodes.map((n) => n.id);
    expect(ids).toEqual([
      "hp-workload",
      "hotpath.admission",
      "hotpath.dispatch",
      "hp-sink",
      "hp-reduce",
      "hp-measure",
    ]);
    // Real type names ride on the node data, not placeholder copy.
    const titles = hotPathStage.subgraph!.nodes.map((n) => (n.data as { title: string }).title);
    expect(titles).toContain("RequestRateWorkload");
    expect(titles).toContain("reduce_parsed_response");
    expect(titles).toContain("measure_dispatch");
    // Every node is a registered `card` type.
    expect(hotPathStage.subgraph!.nodes.every((n) => n.type === "card")).toBe(true);
  });

  it("wires the spine edges in traversal order with a TTFT hand-off", () => {
    const edges = hotPathStage.subgraph!.edges;
    expect(edges.map((e) => [e.source, e.target])).toEqual([
      ["hp-workload", "hotpath.admission"],
      ["hotpath.admission", "hotpath.dispatch"],
      ["hotpath.dispatch", "hp-sink"],
      ["hp-sink", "hp-reduce"],
      ["hp-reduce", "hp-measure"],
    ]);
    expect(edges.every((e) => e.type === "flow")).toBe(true);
    expect(edges.find((e) => e.source === "hp-sink")!.label).toBe("TTFT: first token");
  });

  it("exposes exactly the two drillable children that back its level-2 leaves", () => {
    expect(hotPathStage.subgraph!.children).toEqual(["hotpath.admission", "hotpath.dispatch"]);
    expect(Object.keys(hotPathStage.leaves!)).toEqual(["hotpath.admission", "hotpath.dispatch"]);
    // The admission leaf resolves both timing gates by their real struct names.
    const admissionTitles = hotPathStage.leaves!["hotpath.admission"]!.nodes.map(
      (n) => (n.data as { title: string }).title,
    );
    expect(admissionTitles).toEqual(["SlotPool", "StopChecker"]);
    // The dispatch leaf surfaces the once-only TTFT latch.
    const dispatchTitles = hotPathStage.leaves!["hotpath.dispatch"]!.nodes.map(
      (n) => (n.data as { title: string }).title,
    );
    expect(dispatchTitles).toContain("first_token_released");
  });

  it("cites only verified rust/ source anchors", () => {
    const paths = hotPathStage.evidence!.map((e) => e.path);
    expect(paths).toContain("runtime/src/request_rate.rs:140");
    expect(paths).toContain("runtime/src/timing/slots.rs:105");
    expect(paths).toContain("runtime/src/timing/stop.rs:164");
    expect(paths).toContain("runtime/src/transport/core/dispatch.rs:332");
    expect(paths).toContain("runtime/src/transport/reduce.rs:55");
    expect(paths).toContain("runtime/src/transport/measure.rs:92");
    // Every anchor is a real file:line, never a bare filename or a spec reference.
    expect(hotPathStage.evidence!.every((e) => /\.rs:\d+$/.test(e.path))).toBe(true);
  });

  it("plugs into buildZoomTree so each drillable node id is a navigable tree key", () => {
    const tree = buildZoomTree([hotPathStage]);
    expect(tree["hotpath"]!.children).toEqual(["hotpath.admission", "hotpath.dispatch"]);
    // Both leaf ids must exist as their own tree nodes for ZoomStage.drill to accept them.
    expect(tree["hotpath.admission"]).toBeDefined();
    expect(tree["hotpath.dispatch"]).toBeDefined();
    expect(tree["hotpath.admission"]!.label).toBe("Admission gate");
  });
});

describe("hotPathSteps — play fragment", () => {
  it("is one step per spine node, in order, carrying real type-named captions", () => {
    expect(hotPathSteps.map((s) => s.nodeId)).toEqual([
      "hp-workload",
      "hotpath.admission",
      "hotpath.dispatch",
      "hp-sink",
      "hp-reduce",
      "hp-measure",
    ]);
    // Steps reference exactly the spine node ids.
    const nodeIds = new Set(hotPathStage.subgraph!.nodes.map((n) => n.id));
    expect(hotPathSteps.every((s) => nodeIds.has(s.nodeId))).toBe(true);
    // TTFT is named where the dispatcher hands off.
    const dispatchStep = hotPathSteps.find((s) => s.nodeId === "hotpath.dispatch")!;
    expect(dispatchStep.caption).toMatch(/on_first_token once with TTFT/);
  });
});

describe("HotPathCards — explainer grid", () => {
  it("renders callouts naming the real hot-path types", () => {
    render(<HotPathCards />);
    expect(screen.getByText("Workload issues the schedule")).toBeInTheDocument();
    expect(screen.getByText("TTFT = first token observation")).toBeInTheDocument();
    // Real code identifiers, not placeholder prose.
    expect(screen.getAllByText("RequestRateWorkload").length).toBeGreaterThan(0);
    expect(screen.getAllByText("SlotPool").length).toBeGreaterThan(0);
    expect(screen.getAllByText("StopChecker").length).toBeGreaterThan(0);
    expect(screen.getAllByText("dispatch_collect").length).toBeGreaterThan(0);
    expect(screen.getAllByText("reduce_parsed_response").length).toBeGreaterThan(0);
    expect(screen.getAllByText("measure_dispatch").length).toBeGreaterThan(0);
  });
});
