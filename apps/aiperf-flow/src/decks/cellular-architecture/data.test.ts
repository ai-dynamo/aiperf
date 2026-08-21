/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  NODES,
  EDGES,
  STORY_STEPS,
  ABILITIES,
  storyVisibility,
  deriveRoute,
  statusLabel,
} from "./data.js";

describe("cellular atlas catalog", () => {
  it("has exactly 20 ordered story pages, page 20 is the full atlas", () => {
    expect(STORY_STEPS).toHaveLength(20);
    STORY_STEPS.forEach((step, index) => {
      expect(step.page).toBe(index + 1);
      expect(step.invariant).toBeTruthy();
      expect(step.symbol).toBeTruthy();
      expect(step.path).toBeTruthy();
      expect(step.proof).toBeTruthy();
      expect(step.change).toBeTruthy();
    });
    expect(STORY_STEPS[19].fullAtlas).toBe(true);
    expect(STORY_STEPS[0].title).toBe("One run. Many cells. One report.");
  });

  it("every edge references known nodes", () => {
    const ids = new Set(NODES.map((node) => node.id));
    for (const edge of EDGES) {
      expect(ids.has(edge.from)).toBe(true);
      expect(ids.has(edge.to)).toBe(true);
    }
  });

  it("every added node/edge on a story page exists and edges have visible endpoints", () => {
    const nodeIds = new Set(NODES.map((node) => node.id));
    const edgeIds = new Set(EDGES.map((edge) => edge.id));
    STORY_STEPS.forEach((step) => {
      step.addedNodeIds.forEach((id) => expect(nodeIds.has(id)).toBe(true));
      step.addedEdgeIds.forEach((id) => expect(edgeIds.has(id)).toBe(true));
      if (step.page < 20) {
        const vis = storyVisibility(step.page);
        EDGES.filter((edge) => vis.edgeIds.has(edge.id)).forEach((edge) => {
          expect(vis.nodeIds.has(edge.from)).toBe(true);
          expect(vis.nodeIds.has(edge.to)).toBe(true);
        });
      }
    });
  });

  it("preserves the ability matrix statuses verbatim", () => {
    const byDimension = new Map(ABILITIES.map((a) => [a.dimension, a]));
    expect(byDimension.get("DynoSim transport")?.status).toBe("Rejected");
    expect(byDimension.get("Sketch")?.status).toBe("Approximation");
    expect(byDimension.get("External sink")?.status).toBe("Planned");
    expect(byDimension.get("Work unit")?.status).toBe("Built");
    expect(byDimension.get("Native Kubernetes roles")?.status).toBe("Partial");
  });

  it("maps status to its uppercase label", () => {
    expect(statusLabel("built")).toBe("BUILT");
    expect(statusLabel("partial")).toBe("PARTIAL");
    expect(statusLabel("planned")).toBe("PLANNED");
    expect(statusLabel("rejected")).toBe("REJECTED");
  });
});

describe("deriveRoute", () => {
  it("T1 sketch synchronized yields bounded sketch fidelity to the controller", () => {
    const route = deriveRoute("t1", "scheduled", "sketch", "synchronized");
    expect(route.percentiles).toBe("Approximate · t-digest");
    expect(route.topology).toBe("Cells → controller");
    expect(route.nodeIds.has("tag-sketch")).toBe(true);
    expect(route.nodeIds.has("report")).toBe(true);
    expect(route.warning).toBeUndefined();
  });

  it("refuses every hierarchy request before controller startup", () => {
    const route = deriveRoute("t2", "scheduled", "retain", "synchronized");
    expect(route.topology).toBe("Hierarchy request → refusal");
    expect(route.nodeIds.has("aggregator")).toBe(false);
    expect(route.nodeIds.has("cell")).toBe(false);
    expect(route.warning).toBe("Hierarchical aggregation is unavailable and refused before controller startup.");
  });

  it("T3 routes to the planned external sink with no authoritative report", () => {
    const route = deriveRoute("t3", "scheduled", "sketch", "synchronized");
    expect(route.topology).toBe("Cells → external ingest (planned)");
    expect(route.nodeIds.has("external-sink")).toBe(true);
    expect(route.nodeIds.has("report")).toBe(false);
    expect(route.warning).toContain("T3 no-central-merge external streaming remains planned");
  });

  it("graph + retain warns about deterministic-per-topology concatenation", () => {
    const route = deriveRoute("t0", "graph", "retain", "synchronized");
    expect(route.memory).toBe("O(records)");
    expect(route.warning).toContain("Graph retain concatenates by cell and renumbers densely");
  });
});
