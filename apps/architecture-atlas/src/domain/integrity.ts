// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import {
  architectureCatalogSchema,
  type ArchitectureCatalog,
  type AudienceCopy,
  type CargoDependencyKind,
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
    kind?: CargoDependencyKind | null;
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

function assertUniqueValues(
  values: readonly string[],
  label: string,
): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      throw new Error(`duplicate ${label}: ${value}`);
    }
    seen.add(value);
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
    ...catalog.graphNodes.flatMap(({ evidence }) => evidence),
    ...catalog.graphEdges.flatMap(({ evidence }) => evidence),
  ];
}

function assertGraphHierarchy(nodes: ArchitectureCatalog["graphNodes"]): void {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const visiting = new Set<string>();
  const visited = new Set<string>();

  const visit = (nodeId: string): void => {
    if (visited.has(nodeId)) {
      return;
    }
    if (visiting.has(nodeId)) {
      throw new Error(`graph hierarchy cycle detected at ${nodeId}`);
    }
    visiting.add(nodeId);
    const node = byId.get(nodeId);
    if (!node) {
      throw new Error(`graph hierarchy references missing node ${nodeId}`);
    }
    for (const childId of node.childIds) {
      const child = byId.get(childId);
      if (!child) {
        throw new Error(`${node.id} references missing child ${childId}`);
      }
      if (child.parentId !== node.id) {
        throw new Error(
          `${node.id} child ${childId} must reference parent ${node.id}`,
        );
      }
      visit(childId);
    }
    visiting.delete(nodeId);
    visited.add(nodeId);
  };

  for (const node of nodes) {
    if (node.parentId && !byId.has(node.parentId)) {
      throw new Error(`${node.id} references missing parent ${node.parentId}`);
    }
    if (node.parentId && !byId.get(node.parentId)?.childIds.includes(node.id)) {
      throw new Error(
        `${node.id} declares parent ${node.parentId}, but that parent does not declare the child`,
      );
    }
  }
  for (const node of nodes) {
    visit(node.id);
  }
}

function isDesignEvidencePath(path: string): boolean {
  return path.startsWith("specs/") || path.startsWith("docs/superpowers/specs/");
}

function assertImplementationEvidence(catalog: ArchitectureCatalog): void {
  for (const entity of [...catalog.graphNodes, ...catalog.graphEdges]) {
    for (const reference of entity.evidence) {
      const designPath = isDesignEvidencePath(reference.path);
      if (designPath && reference.role !== "design") {
        if (reference.role === "source") {
          throw new Error(
            `${entity.id} design evidence ${reference.path} cannot be declared as source`,
          );
        }
        throw new Error(
          `${entity.id} design evidence ${reference.path} requires role "design"`,
        );
      }
      if (reference.role !== "design" && reference.role !== "source") {
        throw new Error(`${entity.id} graph evidence requires an explicit role`);
      }
      if (reference.role === "source" && !reference.lines) {
        throw new Error(
          `${entity.id} source evidence ${reference.path} requires a line range`,
        );
      }
    }

    const hasSource = entity.evidence.some(
      (reference) =>
        reference.role === "source" &&
        reference.lines !== undefined &&
        !isDesignEvidencePath(reference.path),
    );
    const hasDesign = entity.evidence.some(
      (reference) =>
        reference.role === "design" && isDesignEvidencePath(reference.path),
    );
    if (entity.status.state === "built" && !hasSource) {
      throw new Error(`${entity.id} is built but has only design evidence`);
    }
    if (entity.status.state === "planned" && !hasDesign) {
      throw new Error(`${entity.id} is planned but has no design evidence`);
    }
    if (
      entity.flavors.includes("dynamo_online") &&
      entity.status.delivery === "runner_pair" &&
      entity.status.state === "built"
    ) {
      throw new Error(
        `${entity.id} must remain planned for dedicated dynamo_online runner integration`,
      );
    }
  }
}

function assertGraphReferences(catalog: ArchitectureCatalog): void {
  const nodeIds = new Set(catalog.graphNodes.map(({ id }) => id));
  const edgeIds = new Set(catalog.graphEdges.map(({ id }) => id));
  const edgesById = new Map(catalog.graphEdges.map((edge) => [edge.id, edge]));
  const portsByNode = new Map(
    catalog.graphNodes.map((node) => [
      node.id,
      new Map(node.seamPorts.map((port) => [port.id, port.channel])),
    ]),
  );

  for (const edge of catalog.graphEdges) {
    const endpoints = [edge.source, edge.target];
    for (const endpoint of endpoints) {
      if (!nodeIds.has(endpoint.nodeId)) {
        throw new Error(`${edge.id} references missing node ${endpoint.nodeId}`);
      }
      const nodePorts = portsByNode.get(endpoint.nodeId);
      const portChannel = nodePorts?.get(endpoint.portId);
      if (!portChannel) {
        throw new Error(
          `${edge.id} references missing port ${endpoint.portId} on ${endpoint.nodeId}`,
        );
      }
      if (portChannel !== edge.channel) {
        throw new Error(
          `${edge.id} channel ${edge.channel} does not match port ${endpoint.portId} channel ${portChannel}`,
        );
      }
    }
  }

  for (const scene of catalog.graphScenes) {
    for (const nodeId of scene.nodeIds) {
      if (!nodeIds.has(nodeId)) {
        throw new Error(`${scene.id} references missing scene node ${nodeId}`);
      }
    }
    for (const edgeId of scene.edgeIds) {
      if (!edgeIds.has(edgeId)) {
        throw new Error(`${scene.id} references missing scene edge ${edgeId}`);
      }
      const edge = edgesById.get(edgeId);
      if (
        edge &&
        (!scene.nodeIds.includes(edge.source.nodeId) ||
          !scene.nodeIds.includes(edge.target.nodeId))
      ) {
        throw new Error(
          `${scene.id} edge ${edgeId} has an endpoint outside the scene`,
        );
      }
    }
  }
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
    catalog.graphNodes,
    catalog.graphEdges,
    catalog.graphScenes,
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
  for (const risk of catalog.risks) {
    for (const componentId of risk.componentIds) {
      if (!componentIds.has(componentId)) {
        throw new Error(`${risk.id} references missing component ${componentId}`);
      }
    }
  }
  for (const stage of catalog.lifecycleStages) {
    for (const componentId of stage.componentIds) {
      if (!componentIds.has(componentId)) {
        throw new Error(`${stage.id} references missing component ${componentId}`);
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
    for (const dependency of crateReference.dependencies) {
      if (!crateIds.has(dependency.crateId)) {
        throw new Error(
          `${crateReference.id} references missing crate ${dependency.crateId}`,
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
  assertUniqueValues(
    catalog.views.map(({ route }) => route),
    "architecture view route",
  );
  assertUniqueValues(
    catalog.pairSupport.map(({ mode, workload }) => `${mode} + ${workload}`),
    "mode/workload pair",
  );
  assertGraphHierarchy(catalog.graphNodes);
  assertGraphReferences(catalog);
  assertImplementationEvidence(catalog);

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
      const actualDependencies = workspacePackage.dependencies
        .filter(
          ({ name, path }) =>
            path?.startsWith(`${rootPath}/crates/`) &&
            catalogPackages.has(name),
        )
        .map(({ kind, name }) => `crate.${name}:${kind ?? "normal"}`)
        .sort();
      const catalogDependencies = crateReference.dependencies
        .map(({ crateId, kind }) => `${crateId}:${kind}`)
        .sort();
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
