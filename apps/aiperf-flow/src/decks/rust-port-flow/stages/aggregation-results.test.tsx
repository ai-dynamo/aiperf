/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineCanvas } from "../../../interactive/index.js";
import { buildZoomTree } from "../stage.js";
import { aggregationResultsStage, aggregationResultsSteps } from "./aggregation-results.js";

describe("aggregationResultsStage (rust-port-flow stage 8)", () => {
  it("keeps the spine slot: id/order/label/tone for the registry", () => {
    expect(aggregationResultsStage.id).toBe("aggregation");
    expect(aggregationResultsStage.order).toBe(8);
    expect(aggregationResultsStage.label).toBe("Aggregation → results");
    expect(aggregationResultsStage.tone).toBe("green");
  });

  it("renders its level-1 subgraph naming the real aggregation types", () => {
    const sub = aggregationResultsStage.subgraph!;
    expect(sub).toBeDefined();
    render(<PipelineCanvas nodes={sub.nodes} edges={sub.edges} />);
    expect(screen.getByText("NativeMetricsObserver")).toBeInTheDocument();
    expect(screen.getByText("NaN-sparse column store")).toBeInTheDocument();
    expect(screen.getByText("NativeReporter → NativeReport")).toBeInTheDocument();
    expect(screen.getByText("ExporterRegistry")).toBeInTheDocument();
    expect(screen.getByText("RunTerminalV2")).toBeInTheDocument();
  });

  it("splits EXACT folds from the t-digest SKETCH (not DDSketch)", () => {
    const sub = aggregationResultsStage.subgraph!;
    render(<PipelineCanvas nodes={sub.nodes} edges={sub.edges} />);
    expect(screen.getByText("Exact record fold")).toBeInTheDocument();
    expect(screen.getByText("t-digest sketch")).toBeInTheDocument();
    // The sketch node cites the real type and rules out DDSketch.
    expect(screen.getByText(/cellular::sketch::TDigest/)).toBeInTheDocument();
    expect(screen.getByText(/NOT DDSketch/)).toBeInTheDocument();
  });

  it("exposes the exact-vs-t-digest-merge leaf as a clickable drill child", () => {
    // The overview→stage→leaf wiring: the comparison node id is a declared child AND a leaf key.
    expect(aggregationResultsStage.subgraph!.children).toContain("aggExactVsSketch");
    expect(aggregationResultsStage.leaves).toBeDefined();
    const leaf = aggregationResultsStage.leaves!["aggExactVsSketch"]!;
    expect(leaf.label).toBe("Exact folds vs t-digest merge");

    const tree = buildZoomTree([aggregationResultsStage]);
    // buildZoomTree registers the leaf as its own navigable node.
    expect(tree["aggExactVsSketch"]).toBeDefined();

    render(<PipelineCanvas nodes={leaf.nodes} edges={leaf.edges} />);
    expect(screen.getByText("Exact: merge_records_in_global_order")).toBeInTheDocument();
    expect(screen.getByText("t-digest: TDigest::merge")).toBeInTheDocument();
    expect(screen.getByText(/Percentiles \+ stddev become streaming estimates/)).toBeInTheDocument();
  });

  it("cites verified real source anchors, not spec markdown", () => {
    const paths = aggregationResultsStage.evidence!.map((e) => e.path);
    expect(paths).toContain("runtime/src/metrics.rs:203");
    expect(paths).toContain("runtime/src/cellular/mod.rs:33");
    expect(paths).toContain("runtime/src/metrics_core/report.rs:1031");
    expect(paths).toContain("runtime/src/metrics_core/report.rs:1079");
    expect(paths).toContain("runtime/src/export/mod.rs:208");
    expect(paths).toContain("runtime/src/export/mod.rs:258");
    expect(paths).toContain("runtime/src/engine/coordinator.rs:334");
    // Every anchor is a real file:line pin.
    for (const p of paths) {
      expect(p).toMatch(/^runtime\/src\/.+\.rs:\d+$/);
    }
  });

  it("provides a level-1 FlowStep fragment over the real subgraph node ids", () => {
    const nodeIds = new Set(aggregationResultsStage.subgraph!.nodes.map((n) => n.id));
    expect(aggregationResultsSteps.length).toBe(nodeIds.size);
    for (const step of aggregationResultsSteps) {
      expect(nodeIds.has(step.nodeId)).toBe(true);
      expect(step.caption.length).toBeGreaterThan(0);
    }
    // The first hop names the worker-local observer; the last emits the terminal envelope.
    expect(aggregationResultsSteps[0]!.nodeId).toBe("agg-observer");
    expect(aggregationResultsSteps[0]!.caption).toMatch(/NativeMetricsObserver/);
    expect(aggregationResultsSteps.at(-1)!.nodeId).toBe("agg-terminal");
    expect(aggregationResultsSteps.at(-1)!.caption).toMatch(/RunTerminalV2 carrying its report_path/);
  });
});
