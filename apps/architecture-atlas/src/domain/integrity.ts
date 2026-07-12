// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import {
  architectureCatalogSchema,
  type ArchitectureCatalog,
  type AudienceCopy,
  type EvidenceReference,
} from "./architecture";

const requiredRoutes = [
  "/",
  "/journey",
  "/execution",
  "/data-plane",
  "/observability",
  "/parity",
  "/atlas",
] as const;

export interface WorkspacePackage {
  name: string;
  manifest_path: string;
  dependencies?: Array<{
    name: string;
    path?: string | null;
  }>;
}

function normalizedCopy(value: string): string {
  return value.toLocaleLowerCase().replaceAll(/[^a-z0-9]/g, "");
}

function assertDistinctCopy(id: string, field: string, copy: AudienceCopy): void {
  const variants = Object.values(copy).map(normalizedCopy);
  if (new Set(variants).size !== variants.length) {
    throw new Error(`${id}.${field} audience variants must be meaningfully distinct`);
  }
}

function assertUniqueIds(
  groups: ReadonlyArray<readonly { id: string }[]>,
): void {
  const ids = new Set<string>();
  for (const entity of groups.flat()) {
    if (ids.has(entity.id)) {
      throw new Error(`duplicate architecture ID: ${entity.id}`);
    }
    ids.add(entity.id);
  }
}

async function validateEvidence(
  evidence: EvidenceReference,
  repositoryRoot: URL,
): Promise<void> {
  const sourceUrl = new URL(evidence.path, repositoryRoot);
  try {
    const sourceStat = await stat(sourceUrl);
    if (!sourceStat.isFile()) {
      throw new Error("not a file");
    }
  } catch {
    throw new Error(`evidence file does not exist: ${evidence.path}`);
  }

  if (!evidence.lines) {
    return;
  }
  if (evidence.lines.start > evidence.lines.end) {
    throw new Error(`invalid evidence line range for ${evidence.path}`);
  }
  const source = await readFile(sourceUrl, "utf8");
  const lineCount = source.split(/\r?\n/u).length;
  if (evidence.lines.end > lineCount) {
    throw new Error(
      `evidence line range for ${evidence.path} exceeds ${lineCount} lines`,
    );
  }
}

function allEvidence(catalog: ArchitectureCatalog): EvidenceReference[] {
  return [
    ...catalog.components.flatMap(({ evidence }) => evidence),
    ...catalog.edges.flatMap(({ evidence }) => evidence),
    ...catalog.risks.flatMap(({ evidence }) => evidence),
    ...catalog.lifecycleStages.flatMap(({ evidence }) => evidence),
    ...catalog.crates.flatMap(({ evidence }) => evidence),
    ...catalog.pairSupport.flatMap(({ evidence }) => evidence),
  ];
}

export async function validateArchitectureCatalog(
  input: unknown,
  repositoryRoot: URL,
): Promise<void> {
  const catalog = architectureCatalogSchema.parse(input);
  assertUniqueIds([
    catalog.components,
    catalog.edges,
    catalog.risks,
    catalog.lifecycleStages,
    catalog.views,
    catalog.crates,
    catalog.pairSupport,
  ]);

  const componentIds = new Set(catalog.components.map(({ id }) => id));
  const edgeIds = new Set(catalog.edges.map(({ id }) => id));
  const riskIds = new Set(catalog.risks.map(({ id }) => id));
  const crateIds = new Set(catalog.crates.map(({ id }) => id));

  for (const edge of catalog.edges) {
    for (const endpoint of [edge.from, edge.to]) {
      if (!componentIds.has(endpoint)) {
        throw new Error(`${edge.id} references missing component ${endpoint}`);
      }
    }
  }
  for (const component of catalog.components) {
    for (const crateId of component.crateIds) {
      if (!crateIds.has(crateId)) {
        throw new Error(`${component.id} references missing crate ${crateId}`);
      }
    }
  }
  for (const crateReference of catalog.crates) {
    for (const dependencyId of crateReference.dependencyCrateIds) {
      if (!crateIds.has(dependencyId)) {
        throw new Error(
          `${crateReference.id} references missing crate ${dependencyId}`,
        );
      }
    }
  }
  for (const view of catalog.views) {
    for (const componentId of view.componentIds) {
      if (!componentIds.has(componentId)) {
        throw new Error(`${view.id} references missing component ${componentId}`);
      }
    }
    for (const edgeId of view.edgeIds) {
      if (!edgeIds.has(edgeId)) {
        throw new Error(`${view.id} references missing edge ${edgeId}`);
      }
    }
    for (const riskId of view.riskIds) {
      if (!riskIds.has(riskId)) {
        throw new Error(`${view.id} references missing risk ${riskId}`);
      }
    }
  }

  for (const entity of [
    ...catalog.components,
    ...catalog.risks,
    ...catalog.lifecycleStages,
    ...catalog.views,
    ...catalog.crates,
  ]) {
    assertDistinctCopy(entity.id, "title", entity.title);
    assertDistinctCopy(entity.id, "summary", entity.summary);
  }
  for (const crateReference of catalog.crates) {
    assertDistinctCopy(
      crateReference.id,
      "responsibility",
      crateReference.responsibility,
    );
  }
  for (const pair of catalog.pairSupport) {
    assertDistinctCopy(pair.id, "notes", pair.notes);
  }

  await Promise.all(
    allEvidence(catalog).map((evidence) =>
      validateEvidence(evidence, repositoryRoot),
    ),
  );
  await Promise.all(
    catalog.crates.flatMap((crateReference) =>
      crateReference.keySourcePaths.map((path) =>
        validateEvidence({ path }, repositoryRoot),
      ),
    ),
  );

  const viewRoutes = new Set(catalog.views.map(({ route }) => route));
  for (const route of requiredRoutes) {
    if (!viewRoutes.has(route)) {
      throw new Error(`architecture view coverage is missing route ${route}`);
    }
  }
}

export function validateWorkspaceCrates(
  catalog: ArchitectureCatalog,
  packages: readonly WorkspacePackage[],
  repositoryRoot: URL,
): void {
  const rootPath = fileURLToPath(repositoryRoot).replace(/\/$/u, "");
  const catalogPackages = new Set(
    catalog.crates.map(({ packageName }) => packageName),
  );
  for (const crateReference of catalog.crates) {
    const workspacePackage = packages.find(
      ({ name }) => name === crateReference.packageName,
    );
    const manifestPath = workspacePackage?.manifest_path;
    const expected = `${rootPath}/${crateReference.path}/Cargo.toml`;
    if (manifestPath !== expected) {
      throw new Error(
        `${crateReference.id} Cargo identity mismatch: expected ${expected}, got ${manifestPath ?? "missing"}`,
      );
    }
    if (workspacePackage?.dependencies) {
      const actualDependencies = [
        ...new Set(
          workspacePackage.dependencies
            .filter(
              ({ name, path }) =>
                path?.startsWith(`${rootPath}/crates/`) &&
                catalogPackages.has(name),
            )
            .map(({ name }) => `crate.${name}`),
        ),
      ].sort();
      const catalogDependencies = [...crateReference.dependencyCrateIds].sort();
      if (
        actualDependencies.length !== catalogDependencies.length ||
        actualDependencies.some(
          (dependency, index) => dependency !== catalogDependencies[index],
        )
      ) {
        throw new Error(
          `${crateReference.id} workspace dependency mismatch: expected ${actualDependencies.join(", ")}, got ${catalogDependencies.join(", ")}`,
        );
      }
    }
  }
  const missing = packages
    .filter(({ manifest_path }) => manifest_path.startsWith(`${rootPath}/crates/`))
    .map(({ name }) => name)
    .filter((name) => !catalogPackages.has(name));
  if (missing.length > 0) {
    throw new Error(`workspace crates missing from catalog: ${missing.join(", ")}`);
  }
}
