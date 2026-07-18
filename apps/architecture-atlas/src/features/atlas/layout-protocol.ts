// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ArchitectureComponent,
  ArchitectureEdge,
  LifecycleBand,
  Ownership,
} from "../../domain/architecture";
import { z } from "zod";

export type LayoutPerspective = "ownership" | "lifecycle";
export const LAYOUT_PROTOCOL_VERSION = 1 as const;

export interface LayoutHierarchyParent {
  id: string;
  parentId: string;
}

export interface LayoutExpandedSubgraph {
  nodeIds: string[];
  rootId: string;
}

export interface LayoutManualPosition {
  id: string;
  x: number;
  y: number;
}

export interface LayoutPartialRelayout {
  expandedSubgraphs: LayoutExpandedSubgraph[];
  manualPositions: LayoutManualPosition[];
  relayoutNodeIds: string[];
}

export interface BuildLayoutRequestOptions {
  hierarchy?: LayoutHierarchyParent[];
  partialRelayout?: LayoutPartialRelayout;
}

export interface LayoutRequest {
  bands: LayoutBandDefinition[];
  edges: Array<{ from: string; id: string; to: string }>;
  key: string;
  nodes: Array<{ bandId: string; id: string; parentId?: string }>;
  partialRelayout?: LayoutPartialRelayout;
  perspective: LayoutPerspective;
  version: typeof LAYOUT_PROTOCOL_VERSION;
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
  partialRelayout?: {
    preservedManualNodeIds: string[];
    relaidOutNodeIds: string[];
  };
  positions: LayoutPosition[];
  reason?: string;
}

const IdentifierSchema = z.string().min(1);
const FiniteNumberSchema = z.number().finite();

export const LayoutHierarchyParentSchema = z
  .object({
    id: IdentifierSchema,
    parentId: IdentifierSchema,
  })
  .strict();

export const LayoutHierarchySchema = z.array(LayoutHierarchyParentSchema);

export const LayoutExpandedSubgraphSchema = z
  .object({
    nodeIds: z.array(IdentifierSchema).min(1),
    rootId: IdentifierSchema,
  })
  .strict()
  .refine(({ nodeIds, rootId }) => nodeIds.includes(rootId), {
    message: "expanded subgraph rootId must be included in nodeIds",
    path: ["rootId"],
  });

export const LayoutPartialRelayoutSchema = z
  .object({
    expandedSubgraphs: z.array(LayoutExpandedSubgraphSchema),
    manualPositions: z.array(
      z
        .object({
          id: IdentifierSchema,
          x: FiniteNumberSchema,
          y: FiniteNumberSchema,
        })
        .strict(),
    ),
    relayoutNodeIds: z.array(IdentifierSchema),
  })
  .strict();

const LayoutBandDefinitionSchema = z
  .object({
    id: IdentifierSchema,
    label: z.string(),
    order: z.number().int().nonnegative(),
  })
  .strict();

const LayoutNodeSchema = z
  .object({
    bandId: IdentifierSchema,
    id: IdentifierSchema,
    parentId: IdentifierSchema.optional(),
  })
  .strict();

const LayoutEdgeSchema = z
  .object({
    from: IdentifierSchema,
    id: IdentifierSchema,
    to: IdentifierSchema,
  })
  .strict();

export const LayoutRequestSchema = z
  .object({
    bands: z.array(LayoutBandDefinitionSchema),
    edges: z.array(LayoutEdgeSchema),
    key: z.string(),
    nodes: z.array(LayoutNodeSchema),
    partialRelayout: LayoutPartialRelayoutSchema.optional(),
    perspective: z.enum(["ownership", "lifecycle"]),
    version: z.literal(LAYOUT_PROTOCOL_VERSION),
  })
  .strict()
  .superRefine((request, context) => {
    const nodes = new Map(request.nodes.map((node) => [node.id, node]));
    for (const [index, node] of request.nodes.entries()) {
      if (!node.parentId) {
        continue;
      }
      const parent = nodes.get(node.parentId);
      if (!parent || parent.id === node.id || parent.bandId !== node.bandId) {
        context.addIssue({
          code: "custom",
          message: "hierarchy parent must be a different node in the same band",
          path: ["nodes", index, "parentId"],
        });
        continue;
      }
      const visited = new Set([node.id]);
      let ancestor = parent;
      while (ancestor.parentId) {
        if (visited.has(ancestor.id)) {
          context.addIssue({
            code: "custom",
            message: "hierarchy must not contain cycles",
            path: ["nodes", index, "parentId"],
          });
          break;
        }
        visited.add(ancestor.id);
        const next = nodes.get(ancestor.parentId);
        if (!next) {
          break;
        }
        ancestor = next;
      }
    }
    const partial = request.partialRelayout;
    if (!partial) {
      return;
    }
    const nodeIds = new Set(nodes.keys());
    const references = [
      ...partial.relayoutNodeIds,
      ...partial.manualPositions.map(({ id }) => id),
      ...partial.expandedSubgraphs.flatMap(({ nodeIds: members, rootId }) => [
        rootId,
        ...members,
      ]),
    ];
    if (references.some((id) => !nodeIds.has(id))) {
      context.addIssue({
        code: "custom",
        message: "partial relayout references an unknown node",
        path: ["partialRelayout"],
      });
    }
  });

const LayoutBandSchema = LayoutBandDefinitionSchema.extend({
  height: FiniteNumberSchema.nonnegative(),
  width: FiniteNumberSchema.nonnegative(),
  x: FiniteNumberSchema,
  y: FiniteNumberSchema,
}).strict();

const LayoutPositionSchema = z
  .object({
    bandId: IdentifierSchema,
    id: IdentifierSchema,
    x: FiniteNumberSchema,
    y: FiniteNumberSchema,
  })
  .strict();

export const LayoutResultSchema = z
  .object({
    bands: z.array(LayoutBandSchema),
    degraded: z.boolean(),
    partialRelayout: z
      .object({
        preservedManualNodeIds: z.array(IdentifierSchema),
        relaidOutNodeIds: z.array(IdentifierSchema),
      })
      .strict()
      .optional(),
    positions: z.array(LayoutPositionSchema),
    reason: z.string().optional(),
  })
  .strict();

const WorkerEnvelopeBase = {
  requestId: z.number().int().nonnegative(),
  version: z.literal(LAYOUT_PROTOCOL_VERSION),
};

export const LayoutWorkerRequestSchema = z
  .object({
    ...WorkerEnvelopeBase,
    request: LayoutRequestSchema,
  })
  .strict();

export const LayoutWorkerResponseSchema = z.union([
  z
    .object({
      ...WorkerEnvelopeBase,
      result: LayoutResultSchema,
    })
    .strict(),
  z
    .object({
      ...WorkerEnvelopeBase,
      error: z.string().min(1),
    })
    .strict(),
]);

export type LayoutWorkerRequest = z.infer<typeof LayoutWorkerRequestSchema>;
export type LayoutWorkerResponse = z.infer<typeof LayoutWorkerResponseSchema>;

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
  options: BuildLayoutRequestOptions = {},
): LayoutRequest {
  const hierarchyByNode = new Map(
    (options.hierarchy ?? []).map(({ id, parentId }) => [id, parentId]),
  );
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
      parentId: hierarchyByNode.get(component.id),
    }));
  const visibleBandIds = new Set(nodes.map(({ bandId }) => bandId));
  const bands = definitions.filter(({ id }) => visibleBandIds.has(id));
  const nodeIds = new Set(nodes.map(({ id }) => id));
  const visibleEdges = [...edges]
    .filter(({ from, to }) => nodeIds.has(from) && nodeIds.has(to))
    .sort((left, right) => left.id.localeCompare(right.id))
    .map(({ from, id, to }) => ({ from, id, to }));
  const partialRelayout = normalizePartialRelayout(options.partialRelayout, nodes);
  const hierarchyKey = nodes
    .filter((node) => typeof node.parentId === "string")
    .map((node) => `${node.id}>${node.parentId}`)
    .join(",");
  return {
    bands,
    edges: visibleEdges,
    key: [
      `v${LAYOUT_PROTOCOL_VERSION}`,
      perspective,
      nodes.map(({ bandId, id }) => `${id}@${bandId}`).join(","),
      hierarchyKey,
      visibleEdges.map(({ id }) => id).join(","),
      serializePartialRelayoutKey(partialRelayout),
    ].join("|"),
    nodes,
    partialRelayout,
    perspective,
    version: LAYOUT_PROTOCOL_VERSION,
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
  const partial = applyPartialRelayoutOverrides(positions, request.partialRelayout);
  return {
    bands,
    degraded: false,
    partialRelayout: partial.summary,
    positions: partial.positions,
  };
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

function normalizePartialRelayout(
  partialRelayout: LayoutPartialRelayout | undefined,
  nodes: ReadonlyArray<{ id: string }>,
): LayoutPartialRelayout | undefined {
  if (!partialRelayout) {
    return undefined;
  }
  const nodeIds = new Set(nodes.map(({ id }) => id));
  const relayoutNodeIds = [...new Set(partialRelayout.relayoutNodeIds)]
    .filter((id) => nodeIds.has(id))
    .sort((left, right) => left.localeCompare(right));
  const manualPositions = [...partialRelayout.manualPositions]
    .filter(({ id }) => nodeIds.has(id))
    .sort((left, right) => left.id.localeCompare(right.id));
  const expandedSubgraphs = partialRelayout.expandedSubgraphs
    .map(({ nodeIds: members, rootId }) => ({
      nodeIds: [...new Set(members)]
        .filter((id) => nodeIds.has(id))
        .sort((left, right) => left.localeCompare(right)),
      rootId,
    }))
    .filter(
      ({ nodeIds: members, rootId }) =>
        members.length > 0 && members.includes(rootId) && nodeIds.has(rootId),
    )
    .sort((left, right) => left.rootId.localeCompare(right.rootId));
  return {
    expandedSubgraphs,
    manualPositions,
    relayoutNodeIds,
  };
}

function serializePartialRelayoutKey(
  partialRelayout: LayoutPartialRelayout | undefined,
): string {
  if (!partialRelayout) {
    return "";
  }
  const expanded = partialRelayout.expandedSubgraphs
    .map(({ nodeIds, rootId }) => `${rootId}:${nodeIds.join("+")}`)
    .join(",");
  const manual = partialRelayout.manualPositions
    .map(({ id, x, y }) => `${id}:${x}:${y}`)
    .join(",");
  return `${partialRelayout.relayoutNodeIds.join(",")}|${expanded}|${manual}`;
}

function applyPartialRelayoutOverrides(
  positions: LayoutPosition[],
  partialRelayout: LayoutPartialRelayout | undefined,
): {
  positions: LayoutPosition[];
  summary?: { preservedManualNodeIds: string[]; relaidOutNodeIds: string[] };
} {
  if (!partialRelayout) {
    return { positions };
  }
  const relayoutNodeIds = new Set(partialRelayout.relayoutNodeIds);
  const manualById = new Map(
    partialRelayout.manualPositions.map((position) => [position.id, position]),
  );
  const visibleNodeIds = new Set(positions.map(({ id }) => id));
  const preservedManualNodeIds: string[] = [];
  const relaidOutNodeIds = [...relayoutNodeIds]
    .filter((id) => visibleNodeIds.has(id))
    .sort((left, right) => left.localeCompare(right));
  const overriddenPositions = positions.map((position) => {
    const manual = manualById.get(position.id);
    if (!manual || relayoutNodeIds.has(position.id)) {
      return position;
    }
    preservedManualNodeIds.push(position.id);
    return {
      ...position,
      x: manual.x,
      y: manual.y,
    };
  });
  return {
    positions: overriddenPositions,
    summary: {
      preservedManualNodeIds: preservedManualNodeIds.sort((a, b) =>
        a.localeCompare(b),
      ),
      relaidOutNodeIds,
    },
  };
}
