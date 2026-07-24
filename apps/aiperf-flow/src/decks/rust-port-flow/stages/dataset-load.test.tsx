/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { datasetStage, datasetFlowSteps } from "./dataset-load.js";
import { PipelineCanvas } from "../../../interactive/index.js";

describe("dataset-load stage", () => {
  it("keeps the stub's spine identity (id/order/label/tone)", () => {
    expect(datasetStage.id).toBe("dataset");
    expect(datasetStage.order).toBe(2);
    expect(datasetStage.label).toBe("Dataset loading");
    expect(datasetStage.tone).toBe("green");
  });

  it("wires a level-1 load pipeline naming the real content-lowering types", () => {
    const titles = datasetStage.subgraph!.nodes.map((n) => (n.data as { title: string }).title);
    expect(titles).toContain("DatasetLoader::load");
    expect(titles).toContain("SegmentPool");
    expect(titles).toContain("intern(parent, payload)");
    expect(titles).toContain("InMemorySegmentStore");
    expect(titles).toContain("Turn");
  });

  it("exposes the two level-2 drill targets from the intern node", () => {
    expect(datasetStage.subgraph!.children).toEqual(["domains", "hashing"]);
    const domainEdge = datasetStage.subgraph!.edges.find((e) => e.target === "domains");
    const hashingEdge = datasetStage.subgraph!.edges.find((e) => e.target === "hashing");
    expect(domainEdge?.source).toBe("intern");
    expect(hashingEdge?.source).toBe("intern");
  });

  it("enumerates the six disjoint BLAKE3 SegmentDomains in the domains leaf", () => {
    const domains = datasetStage.leaves!.domains;
    const kinds = domains.nodes.map((n) => (n.data as { title: string }).title);
    expect(kinds).toEqual([
      "message",
      "text-only",
      "raw",
      "token-ids",
      "media",
      "trace-hash-ids",
    ]);
  });

  it("shows prefix-folded hashing → dedup → dense Handle in the hashing leaf", () => {
    const hashing = datasetStage.leaves!.hashing;
    const titles = hashing.nodes.map((n) => (n.data as { title: string }).title);
    expect(titles).toContain("payload_id(parent, payload)");
    expect(titles).toContain("push_interned → ids map");
    expect(titles).toContain("Handle(u32)");
    // The fold edge runs parent-hash → payload_id → dedup → dense handle.
    expect(hashing.edges.map((e) => `${e.source}->${e.target}`)).toEqual([
      "hash-parent->hash-payload",
      "hash-payload->hash-dedup",
      "hash-dedup->hash-handle",
    ]);
  });

  it("pins verified rust/runtime/src/dataset source anchors", () => {
    const paths = datasetStage.evidence!.map((e) => e.path);
    expect(paths).toContain("runtime/src/dataset/segment.rs:238"); // SegmentStore trait
    expect(paths).toContain("runtime/src/dataset/segment.rs:319"); // intern + dedup
    expect(paths).toContain("runtime/src/dataset/segment.rs:514"); // freeze → InMemorySegmentStore
    expect(paths).toContain("runtime/src/dataset/segment.rs:28"); // dense Handle(u32)
    expect(paths).toContain("runtime/src/dataset/model.rs:282"); // Turn.body: [Handle]
  });

  it("provides a play fragment traversing the pipeline with real-type captions", () => {
    const nodeIds = new Set(datasetStage.subgraph!.nodes.map((n) => n.id));
    for (const step of datasetFlowSteps) {
      expect(nodeIds.has(step.nodeId)).toBe(true);
    }
    const captions = datasetFlowSteps.map((s) => s.caption).join("\n");
    expect(captions).toMatch(/prefix-folded BLAKE3/);
    expect(captions).toMatch(/InMemorySegmentStore/);
    expect(captions).toMatch(/SmallVec<\[Handle; 1\]>/);
  });

  it("renders the level-1 subgraph as a real React Flow canvas", () => {
    render(<PipelineCanvas nodes={datasetStage.subgraph!.nodes} edges={datasetStage.subgraph!.edges} />);
    expect(screen.getByText("SegmentPool")).toBeInTheDocument();
    expect(screen.getByText("InMemorySegmentStore")).toBeInTheDocument();
  });
});
