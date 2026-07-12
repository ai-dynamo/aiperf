// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureComponent,
  ArchitectureEdge,
  LifecycleBand,
  Ownership,
} from "../../domain/architecture";

export type LayoutPerspective = "ownership" | "lifecycle";

export interface LayoutRequest {
  bands: LayoutBandDefinition[];
  edges: Array<{ from: string; id: string; to: string }>;
  key: string;
  nodes: Array<{ bandId: string; id: string }>;
  perspective: LayoutPerspective;
}

export interface LayoutBandDefinition {
  id: string;
  label: string;
  order: number;
}

export interface RelativeBandLayout {
  bandId: string;
  height: number;
  positions: Array<{ id: string; x: number; y: number }>;
  width: number;
}

export interface LayoutBand extends LayoutBandDefinition {
  height: number;
  width: number;
  x: number;
  y: number;
}

export interface LayoutPosition {
  bandId: string;
  id: string;
  x: number;
  y: number;
}

export interface LayoutResult {
  bands: LayoutBand[];
  degraded: boolean;
  positions: LayoutPosition[];
  reason?: string;
}

const ownershipOrder: Ownership[] = ["python", "rust", "external", "legacy"];
const ownershipLabels: Record<Ownership, string> = {
  python: "Product control",
  rust: "Run execution",
  external: "External peers",
  legacy: "Retained semantics",
};
const lifecycleOrder: LifecycleBand[] = [
  "authoring",
  "validation",
  "execution",
  "measurement",
  "presentation",
];
const lifecycleLabels: Record<LifecycleBand, string> = {
  authoring: "Authoring",
  validation: "Validation and preparation",
  execution: "Execution",
  measurement: "Measurement",
  presentation: "Presentation",
};

const NODE_WIDTH = 248;
const NODE_HEIGHT = 112;
const NODE_GAP = 28;
const BAND_PADDING = 32;
const BAND_GAP = 52;

export function buildLayoutRequest(
  components: readonly ArchitectureComponent[],
  edges: readonly ArchitectureEdge[],
  perspective: LayoutPerspective,
): LayoutRequest {
  const definitions: LayoutBandDefinition[] =
    perspective === "ownership"
      ? ownershipOrder.map((owner, order) => ({
          id: `ownership.${owner}`,
          label: ownershipLabels[owner],
          order,
        }))
      : lifecycleOrder.map((band, order) => ({
          id: `lifecycle.${band}`,
          label: lifecycleLabels[band],
          order,
        }));
  const nodes = [...components]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((component) => ({
      bandId:
        perspective === "ownership"
          ? `ownership.${component.owner}`
          : `lifecycle.${component.lifecycleBand}`,
      id: component.id,
    }));
  const visibleBandIds = new Set(nodes.map(({ bandId }) => bandId));
  const bands = definitions.filter(({ id }) => visibleBandIds.has(id));
  const nodeIds = new Set(nodes.map(({ id }) => id));
  const visibleEdges = [...edges]
    .filter(({ from, to }) => nodeIds.has(from) && nodeIds.has(to))
    .sort((left, right) => left.id.localeCompare(right.id))
    .map(({ from, id, to }) => ({ from, id, to }));
  return {
    bands,
    edges: visibleEdges,
    key: [
      perspective,
      nodes.map(({ bandId, id }) => `${id}@${bandId}`).join(","),
      visibleEdges.map(({ id }) => id).join(","),
    ].join("|"),
    nodes,
    perspective,
  };
}

export function composeBandLayouts(
  request: LayoutRequest,
  relativeLayouts: readonly RelativeBandLayout[],
): LayoutResult {
  const layouts = new Map(
    relativeLayouts.map((layout) => [layout.bandId, layout]),
  );
  const bands: LayoutBand[] = [];
  const positions: LayoutPosition[] = [];
  let offset = 0;
  for (const definition of request.bands) {
    const layout = layouts.get(definition.id);
    if (!layout) {
      continue;
    }
    const x = request.perspective === "lifecycle" ? offset : 0;
    const y = request.perspective === "ownership" ? offset : 0;
    const width = layout.width + BAND_PADDING * 2;
    const height = layout.height + BAND_PADDING * 2;
    bands.push({ ...definition, height, width, x, y });
    positions.push(
      ...layout.positions.map((position) => ({
        ...position,
        bandId: definition.id,
        x: x + BAND_PADDING + position.x,
        y: y + BAND_PADDING + position.y,
      })),
    );
    offset +=
      (request.perspective === "ownership" ? height : width) + BAND_GAP;
  }
  return { bands, degraded: false, positions };
}

function fallbackBandLayout(
  request: LayoutRequest,
  band: LayoutBandDefinition,
): RelativeBandLayout {
  const nodes = request.nodes.filter(({ bandId }) => bandId === band.id);
  const columns =
    request.perspective === "ownership"
      ? Math.min(4, Math.max(1, nodes.length))
      : 1;
  const rows = Math.ceil(nodes.length / columns);
  return {
    bandId: band.id,
    height: rows * NODE_HEIGHT + Math.max(0, rows - 1) * NODE_GAP,
    positions: nodes.map(({ id }, index) => ({
      id,
      x: (index % columns) * (NODE_WIDTH + NODE_GAP),
      y: Math.floor(index / columns) * (NODE_HEIGHT + NODE_GAP),
    })),
    width: columns * NODE_WIDTH + Math.max(0, columns - 1) * NODE_GAP,
  };
}

export function deterministicFallbackLayout(
  request: LayoutRequest,
  reason: string,
): LayoutResult {
  return {
    ...composeBandLayouts(
      request,
      request.bands.map((band) => fallbackBandLayout(request, band)),
    ),
    degraded: true,
    reason,
  };
}
