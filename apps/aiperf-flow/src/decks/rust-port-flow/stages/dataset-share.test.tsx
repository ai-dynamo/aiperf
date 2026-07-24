/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Node } from "@xyflow/react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { datasetShareStage, datasetShareSteps } from "./dataset-share.js";
import { PipelineCanvas } from "../../../interactive/index.js";

/** Find a node in a table by id and return its `data` for content assertions. */
function nodeData(nodes: Node[], id: string): Record<string, unknown> {
  const node = nodes.find((n) => n.id === id);
  expect(node, `node ${id} exists`).toBeDefined();
  return node!.data as Record<string, unknown>;
}

describe("datasetShareStage (rust-port-flow stage 3)", () => {
  it("keeps the stable stage identity the deck registry and overview depend on", () => {
    expect(datasetShareStage.id).toBe("sharing");
    expect(datasetShareStage.order).toBe(3);
    expect(datasetShareStage.label).toBe("Sharing the dataset");
    expect(datasetShareStage.tone).toBe("cyan");
  });

  it("level-1 subgraph names the real zero-copy sharing types", () => {
    const nodes = datasetShareStage.subgraph!.nodes;
    expect(nodeData(nodes, "share-store").title).toBe("InMemorySegmentStore");
    expect(nodeData(nodes, "share-store").detail).toMatch(/bytes live exactly once/);
    expect(nodeData(nodes, "share-arc").title).toBe("Arc<dyn SegmentStore>");
    expect(nodeData(nodes, "share-arc").detail).toMatch(/zero-copy, never a byte copy/);
    // Turns carry Handles, not bytes.
    expect(nodeData(nodes, "share-turns").detail).toMatch(/dense Handle \(u32\), not inline bytes/);
    // The frozen store is shared across worker threads (Send + Sync).
    expect(nodeData(nodes, "share-wn").detail).toMatch(/Send \+ Sync/);
  });

  it("renders the subgraph through PipelineCanvas with real type names", () => {
    render(<PipelineCanvas nodes={datasetShareStage.subgraph!.nodes} edges={datasetShareStage.subgraph!.edges} />);
    expect(screen.getAllByText("InMemorySegmentStore").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Arc<dyn SegmentStore>").length).toBeGreaterThan(0);
    expect(screen.getAllByText("content_server").length).toBeGreaterThan(0);
  });

  it("the content_server node drills into a leaf that contrasts a media sidecar with text sharing", () => {
    // The drillable node id is advertised as a child AND is the leaf key.
    expect(datasetShareStage.subgraph!.children).toContain("sharing-content-server");
    const leaf = datasetShareStage.leaves!["sharing-content-server"];
    expect(leaf, "content-server leaf exists").toBeDefined();
    expect(leaf!.label).toMatch(/media sidecar/);
    const leafNodes = leaf!.nodes;
    // The media sidecar's real types.
    expect(nodeData(leafNodes, "cs-pub").title).toBe("ContentServerMediaPublisher");
    expect(nodeData(leafNodes, "cs-server").title).toBe("ContentServerRuntime");
    expect(nodeData(leafNodes, "cs-cfg").title).toBe("ContentServerSidecar");
    // The explicit contrast: TEXT sharing vs MEDIA delivery are disjoint.
    expect(nodeData(leafNodes, "cs-text").detail).toMatch(/No HTTP, no bytes copied/);
    expect(nodeData(leafNodes, "cs-media").detail).toMatch(/entirely separate from text/);
  });

  it("renders the content_server leaf with the media-publisher type", () => {
    const leaf = datasetShareStage.leaves!["sharing-content-server"]!;
    render(<PipelineCanvas nodes={leaf.nodes} edges={leaf.edges} />);
    expect(screen.getAllByText("ContentServerMediaPublisher").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Dataset TEXT sharing").length).toBeGreaterThan(0);
    expect(screen.getAllByText("MEDIA delivery").length).toBeGreaterThan(0);
  });

  it("pins verified rust source anchors, including content_server file:line", () => {
    const byPath = new Map(datasetShareStage.evidence!.map((e) => [e.path, e.label]));
    expect(byPath.has("runtime/src/dataset/segment.rs:238")).toBe(true); // SegmentStore trait
    expect(byPath.has("runtime/src/dataset/segment.rs:514")).toBe(true); // freeze
    expect(byPath.has("runtime/src/dataset/segment.rs:533")).toBe(true); // InMemorySegmentStore
    expect(byPath.has("runtime/src/dataset/segment.rs:28")).toBe(true); // Handle
    expect(byPath.has("runtime/src/dataset/dataset.rs:48")).toBe(true); // Arc<dyn SegmentStore>
    expect(byPath.has("runtime/src/dataset/model.rs:216")).toBe(true); // Turn
    expect(byPath.has("runtime/src/content_server/mod.rs:4")).toBe(true); // content_server sidecar
    expect(byPath.has("cli/src/model/telemetry.rs:229")).toBe(true); // ContentServerSidecar
  });

  it("supplies play steps that traverse the sharing path and name the sidecar as separate", () => {
    const ids = datasetShareSteps.map((s) => s.nodeId);
    expect(ids).toEqual(["share-store", "share-arc", "share-turns", "share-w0", "sharing-content-server"]);
    // Every step's nodeId must reference a real level-1 subgraph node.
    const subgraphIds = new Set(datasetShareStage.subgraph!.nodes.map((n) => n.id));
    for (const id of ids) {
      expect(subgraphIds.has(id)).toBe(true);
    }
    const last = datasetShareSteps[datasetShareSteps.length - 1]!;
    expect(last.caption).toMatch(/SEPARATE run-owned media sidecar/);
  });
});
