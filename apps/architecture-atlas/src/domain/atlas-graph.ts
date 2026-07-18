// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureCatalog,
  ArchitectureComponent,
  ArchitectureEdge,
  ArchitectureStatus,
  CargoDependencyKind,
  CrateReference,
  ExecutionMode,
  Ownership,
} from "./architecture";

export interface AtlasFilters {
  query?: string;
  modes?: readonly ExecutionMode[];
  statuses?: readonly ArchitectureStatus[];
  owners?: readonly Ownership[];
}

export interface DerivedAtlasGraph {
  components: ArchitectureComponent[];
  edges: ArchitectureEdge[];
}

export interface DependencyNeighborhood {
  upstream: string[];
  downstream: string[];
  related: Set<string>;
}

export interface CrateDependent {
  crate: CrateReference;
  kind: CargoDependencyKind;
}

function includesQuery(values: readonly string[], query: string): boolean {
  return values.some((value) => value.toLocaleLowerCase().includes(query));
}

export function deriveAtlasGraph(
  catalog: ArchitectureCatalog,
  filters: AtlasFilters,
): DerivedAtlasGraph {
  const query = filters.query?.trim().toLocaleLowerCase() ?? "";
  const crates = new Map(catalog.crates.map((crate) => [crate.id, crate]));
  const components = catalog.components.filter((component) => {
    const crateTerms = component.crateIds.flatMap((id) => {
      const crate = crates.get(id);
      return crate ? [crate.packageName, crate.summary.developer] : [];
    });
    return (
      (!filters.modes?.length ||
        filters.modes.some((mode) => component.modes.includes(mode))) &&
      (!filters.statuses?.length ||
        filters.statuses.includes(component.status)) &&
      (!filters.owners?.length || filters.owners.includes(component.owner)) &&
      (!query ||
        includesQuery(
          [
            ...Object.values(component.title),
            ...Object.values(component.summary),
            ...component.contracts,
            ...crateTerms,
          ],
          query,
        ))
    );
  });
  const ids = new Set(components.map(({ id }) => id));
  const edges = catalog.edges.filter(
    (edge) =>
      ids.has(edge.from) &&
      ids.has(edge.to) &&
      (!filters.statuses?.length || filters.statuses.includes(edge.status)),
  );
  return { components, edges };
}

function trace(
  start: string,
  edges: readonly ArchitectureEdge[],
  direction: "upstream" | "downstream",
): string[] {
  const found = new Set<string>();
  const pending = [start];
  while (pending.length > 0) {
    const current = pending.shift();
    if (!current) {
      continue;
    }
    for (const edge of edges) {
      const next =
        direction === "upstream" && edge.to === current
          ? edge.from
          : direction === "downstream" && edge.from === current
            ? edge.to
            : undefined;
      if (next && next !== start && !found.has(next)) {
        found.add(next);
        pending.push(next);
      }
    }
  }
  return [...found].sort();
}

export function dependencyNeighborhood(
  componentId: string,
  edges: readonly ArchitectureEdge[],
): DependencyNeighborhood {
  const upstream = trace(componentId, edges, "upstream");
  const downstream = trace(componentId, edges, "downstream");
  return {
    upstream,
    downstream,
    related: new Set([componentId, ...upstream, ...downstream]),
  };
}

export function searchCrates(
  crates: readonly CrateReference[],
  query: string,
): CrateReference[] {
  const normalized = query.trim().toLocaleLowerCase();
  if (!normalized) {
    return [...crates];
  }
  return crates.filter((crate) =>
    includesQuery(
      [
        crate.packageName,
        crate.path,
        ...Object.values(crate.title),
        ...Object.values(crate.summary),
        ...Object.values(crate.responsibility),
        ...crate.contracts,
        ...crate.keySourcePaths,
        ...crate.parityScars,
      ],
      normalized,
    ),
  );
}

export function deriveCrateDependents(
  crates: readonly CrateReference[],
  targetCrateId: string,
): CrateDependent[] {
  return crates
    .flatMap((crate) =>
      crate.dependencies
        .filter(({ crateId }) => crateId === targetCrateId)
        .map(({ kind }) => ({ crate, kind })),
    )
    .sort(
      (left, right) =>
        left.kind.localeCompare(right.kind) ||
        left.crate.packageName.localeCompare(right.crate.packageName),
    );
}
