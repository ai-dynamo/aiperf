// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { architectureCatalog } from "../content";
import {
  architectureCatalogSchema,
  type ArchitectureCatalog,
} from "./architecture";
import { validateArchitectureCatalog } from "./integrity";

const repositoryRoot = pathToFileURL(
  `${resolve(dirname(fileURLToPath(import.meta.url)), "../../../../")}/`,
);

function minimalCatalog(): ArchitectureCatalog {
  return architectureCatalogSchema.parse({
    schemaVersion: 2,
    components: [
      {
        id: "component.python",
        kind: "component",
        owner: "python",
        status: "built",
        title: {
          executive: "Configuration front door",
          developer: "Python configuration boundary",
          maintainer: "Config-v2 Python projection",
        },
        summary: {
          executive: "Owns the product controls and presentation boundary.",
          developer: "Authors one strict run request and launches the native runner.",
          maintainer:
            "Projects protocol-v2 input without protocol-v1 fallback or resolved state.",
        },
        evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 12 } }],
        modes: ["online_http"],
        contracts: ["protocol-v2"],
      },
    ],
    edges: [],
    risks: [],
    lifecycleStages: [],
    views: [
      {
        id: "view.ownership",
        kind: "view",
        route: "/",
        title: {
          executive: "Ownership map",
          developer: "System ownership",
          maintainer: "Product ownership boundary",
        },
        summary: {
          executive: "Shows who owns each product decision.",
          developer: "Connects authoring, execution, and presentation.",
          maintainer: "Pins ownership claims to implementation evidence.",
        },
        componentIds: ["component.python"],
        edgeIds: [],
        riskIds: [],
      },
    ],
    crates: [],
    pairSupport: [],
  });
}

describe("architecture catalog schema", () => {
  it("rejects missing audience copy", () => {
    const catalog = minimalCatalog();
    const component = {
      ...catalog.components[0],
      title: {
        executive: "Configuration front door",
        developer: "Python configuration boundary",
      },
    };

    expect(() =>
      architectureCatalogSchema.parse({ ...catalog, components: [component] }),
    ).toThrow();
  });

  it("rejects duplicate entity IDs", async () => {
    const catalog = minimalCatalog();
    catalog.components.push({ ...catalog.components[0] });

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/duplicate.*component\.python/i);
  });

  it("rejects edges with missing targets", async () => {
    const catalog = minimalCatalog();
    catalog.edges.push({
      id: "edge.missing",
      kind: "message",
      from: "component.python",
      to: "component.runner",
      label: "Launches one run",
      protocol: "JSONL protocol v2",
      status: "built",
      evidence: [{ path: "AGENTS.md" }],
    });

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/component\.runner/);
  });

  it.each([
    {
      evidence: { path: "not-a-real-source.rs" },
      expected: /not-a-real-source\.rs/,
    },
    {
      evidence: { path: "AGENTS.md", lines: { start: 12, end: 2 } },
      expected: /line range/i,
    },
    {
      evidence: { path: "AGENTS.md", lines: { start: 1, end: 100000 } },
      expected: /exceeds/i,
    },
  ])("rejects invalid evidence: $evidence", async ({ evidence, expected }) => {
    const catalog = minimalCatalog();
    catalog.components[0].evidence = [evidence];

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(expected);
  });

  it("accepts the complete source-grounded catalog", async () => {
    await expect(
      validateArchitectureCatalog(architectureCatalog, repositoryRoot),
    ).resolves.toBeUndefined();
  });
});
