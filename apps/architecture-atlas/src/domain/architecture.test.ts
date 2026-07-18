// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { architectureCatalog } from "../content";
import {
  architectureCatalogSchema,
  crateReferenceSchema,
  workloadSchema,
  type ArchitectureCatalog,
  type ArchitectureRisk,
  type LifecycleStage,
} from "./architecture";
import {
  validateArchitectureCatalog,
  validateWorkspaceCrates,
} from "./integrity";

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
        lifecycleBand: "authoring",
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
  it("models telemetry_watch as an executable workload", () => {
    expect(workloadSchema.options).toContain("telemetry_watch");
    expect(
      architectureCatalog.pairSupport.some(
        ({ mode, workload, status }) =>
          mode === "online_http" &&
          workload === "telemetry_watch" &&
          status === "built",
      ),
    ).toBe(true);
  });

  it("requires one explicit primary lifecycle band per component", () => {
    expect(
      architectureCatalog.components.every(
        (component) => "lifecycleBand" in component,
      ),
    ).toBe(true);
  });

  it("places durable telemetry history in the measurement lifecycle", () => {
    expect(
      architectureCatalog.components.find(
        ({ id }) => id === "component.telemetry-archive",
      )?.lifecycleBand,
    ).toBe("measurement");
  });

  it("models Cargo dependency kinds explicitly", () => {
    const source = architectureCatalog.crates.find(
      ({ packageName }) => packageName === "aiperf-extensions",
    );
    expect(source).toBeDefined();
    const parsed = crateReferenceSchema.parse(source);
    expect(parsed.dependencies).toContainEqual({
      crateId: "crate.aiperf-rng",
      kind: "dev",
    });
  });

  it("rejects Cargo dependency kind mismatches", () => {
    const packages = architectureCatalog.crates.map((crate) => {
      return {
        name: crate.packageName,
        manifest_path: fileURLToPath(
          new URL(`${crate.path}/Cargo.toml`, repositoryRoot),
        ),
        dependencies: crate.dependencies.map(({ crateId, kind }) => ({
          kind:
            crate.packageName === "aiperf-extensions" &&
            crateId === "crate.aiperf-rng"
              ? "normal" as const
              : kind,
          name: crateId.replace(/^crate\./u, ""),
          path: fileURLToPath(
            new URL(
              `crates/${crateId.replace(/^crate\./u, "")}`,
              repositoryRoot,
            ),
          ),
        })),
      };
    });

    expect(() =>
      validateWorkspaceCrates(
        architectureCatalog,
        packages,
        repositoryRoot,
      ),
    ).toThrow(/dependency.*kind|workspace dependency mismatch/iu);
  });

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
      collection: "risks" as const,
      entity: {
        id: "risk.missing",
        kind: "risk" as const,
        status: "unbuilt" as const,
        severity: "medium" as const,
        title: {
          executive: "Missing ownership risk",
          developer: "Missing component reference",
          maintainer: "Dangling risk.componentIds value",
        },
        summary: {
          executive: "Shows risk that cannot be connected to accountable ownership.",
          developer: "References a component absent from the architecture catalog.",
          maintainer: "Integrity validation must reject component.missing from risk.componentIds.",
        },
        componentIds: ["component.missing"],
        evidence: [{ path: "AGENTS.md" }],
      },
    },
    {
      collection: "lifecycleStages" as const,
      entity: {
        id: "lifecycle.missing",
        kind: "lifecycle" as const,
        order: 1,
        title: {
          executive: "Missing lifecycle owner",
          developer: "Missing lifecycle component",
          maintainer: "Dangling lifecycleStage.componentIds value",
        },
        summary: {
          executive: "Shows a run stage without an accountable system owner.",
          developer: "References a component absent from the architecture catalog.",
          maintainer:
            "Integrity validation must reject component.missing from lifecycleStage.componentIds.",
        },
        componentIds: ["component.missing"],
        evidence: [{ path: "AGENTS.md" }],
      },
    },
  ])("rejects missing component references in $collection", async ({
    collection,
    entity,
  }) => {
    const catalog = minimalCatalog();
    if (collection === "risks") {
      catalog.risks.push(entity as ArchitectureRisk);
    } else {
      catalog.lifecycleStages.push(entity as LifecycleStage);
    }

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/component\.missing/);
  });

  it("rejects duplicate architecture view routes", async () => {
    const catalog = minimalCatalog();
    catalog.views.push({
      ...catalog.views[0],
      id: "view.duplicate",
    });

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/duplicate.*route.*\//i);
  });

  it("rejects duplicate mode and workload pairs", async () => {
    const catalog = minimalCatalog();
    const pair = {
      id: "pair.first",
      mode: "online_http" as const,
      workload: "scheduled" as const,
      status: "built" as const,
      notes: {
        executive: "Primary production request path.",
        developer: "Scheduled requests execute over native HTTP.",
        maintainer: "RunnerApplication registers online_http plus scheduled.",
      },
      evidence: [{ path: "AGENTS.md" }],
    };
    catalog.pairSupport.push(pair, { ...pair, id: "pair.second" });

    await expect(
      validateArchitectureCatalog(catalog, repositoryRoot),
    ).rejects.toThrow(/duplicate.*online_http.*scheduled/i);
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
