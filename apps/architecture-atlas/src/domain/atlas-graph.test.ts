// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { architectureCatalog } from "../content";
import {
  deriveAtlasGraph,
  deriveCrateDependents,
  dependencyNeighborhood,
  searchCrates,
} from "./atlas-graph";

describe("atlas graph derivation", () => {
  it("searches audience copy, crates, and contracts", () => {
    const bySummary = deriveAtlasGraph(architectureCatalog, {
      query: "without sockets",
    });
    const byCrate = deriveAtlasGraph(architectureCatalog, {
      query: "aiperf-clock",
    });
    const byContract = deriveAtlasGraph(architectureCatalog, {
      query: "HttpEndpointBinding",
    });

    expect(bySummary.components.map(({ id }) => id)).toContain(
      "component.dynamo-offline",
    );
    expect(byCrate.components.map(({ id }) => id)).toContain(
      "component.clock-seam",
    );
    expect(byContract.components.map(({ id }) => id)).toContain(
      "component.http-transport",
    );
  });

  it("applies mode, status, and ownership filters without dangling edges", () => {
    const result = deriveAtlasGraph(architectureCatalog, {
      modes: ["online_grpc"],
      owners: ["rust"],
      statuses: ["built"],
    });
    const ids = new Set(result.components.map(({ id }) => id));

    expect(result.components.length).toBeGreaterThan(0);
    expect(result.components.every(({ owner }) => owner === "rust")).toBe(true);
    expect(
      result.edges.every(
        ({ from, to, status }) =>
          ids.has(from) && ids.has(to) && status === "built",
      ),
    ).toBe(true);
  });

  it("finds transitive upstream and downstream dependencies", () => {
    const graph = deriveAtlasGraph(architectureCatalog, {});
    const neighborhood = dependencyNeighborhood(
      "component.rust-runtime",
      graph.edges,
    );

    expect(neighborhood.upstream).toContain("component.rust-runner");
    expect(neighborhood.upstream).toContain("component.python-frontend");
    expect(neighborhood.downstream).toContain("component.inference-target");
    expect(neighborhood.related.has("component.rust-runtime")).toBe(true);
  });

  it("searches the crate directory across responsibility and contracts", () => {
    expect(
      searchCrates(architectureCatalog.crates, "virtual time").map(
        ({ packageName }) => packageName,
      ),
    ).toContain("aiperf-clock");
    expect(
      searchCrates(architectureCatalog.crates, "GraphSink").map(
        ({ packageName }) => packageName,
      ),
    ).toContain("aiperf-graph");
  });

  it("preserves dependency kind in reverse dependent relationships", () => {
    const dependents = deriveCrateDependents(
      architectureCatalog.crates,
      "crate.aiperf-rng",
    );

    expect(dependents).toContainEqual(
      expect.objectContaining({
        kind: "dev",
        crate: expect.objectContaining({ packageName: "aiperf-extensions" }),
      }),
    );
    expect(dependents).toContainEqual(
      expect.objectContaining({
        kind: "normal",
        crate: expect.objectContaining({ packageName: "aiperf-mock-rs" }),
      }),
    );
  });
});
