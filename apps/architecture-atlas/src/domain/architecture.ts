// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from "zod";

const idPattern = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
const repositoryPathPattern = /^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$)).+$/;

export const architectureIdSchema = z.string().regex(idPattern);
export const ownershipSchema = z.enum([
  "python",
  "rust",
  "external",
  "legacy",
]);
export const architectureStatusSchema = z.enum([
  "built",
  "feature-gated",
  "runtime-conditional",
  "compatibility-only",
  "legacy-parallel",
  "unbuilt",
]);
export const executionModeSchema = z.enum([
  "online_http",
  "online_grpc",
  "dynamo_offline",
  "online_mock",
]);
export const lifecycleBandSchema = z.enum([
  "authoring",
  "validation",
  "execution",
  "measurement",
  "presentation",
]);
export const cargoDependencyKindSchema = z.enum(["normal", "build", "dev"]);
export const workloadSchema = z.enum([
  "scheduled",
  "graph",
  "static_accuracy",
  "agentic",
  "evaluation",
  "telemetry_watch",
]);
export const tierSchema = z.union([
  z.literal(0),
  z.literal(1),
  z.literal(2),
  z.literal(3),
]);
export const audienceLevelSchema = z.enum([
  "executive",
  "developer",
  "maintainer",
]);
export const flowChannelSchema = z.enum([
  "control",
  "request_data",
  "token",
  "telemetry",
  "report_result",
]);
export const executionFlavorSchema = z.enum([
  "native_http",
  "native_grpc",
  "online_mock",
  "dynamo_offline",
  "dynamo_online",
]);
export const implementationStateSchema = z.enum(["built", "planned"]);
export const implementationDeliverySchema = z.enum([
  "unconditional",
  "feature_gated",
  "runtime_conditional",
  "runner_pair",
  "library_seam",
  "compatibility_only",
  "legacy_parallel",
]);

export const audienceCopySchema = z
  .object({
    executive: z.string().trim().min(8),
    developer: z.string().trim().min(8),
    maintainer: z.string().trim().min(8),
  })
  .strict();

export const lineRangeSchema = z
  .object({
    start: z.number().int().positive(),
    end: z.number().int().positive(),
  })
  .strict();

export const evidenceReferenceSchema = z
  .object({
    path: z.string().regex(repositoryPathPattern),
    lines: lineRangeSchema.optional(),
    symbol: z.string().trim().min(1).optional(),
    role: z.enum(["source", "design"]).optional(),
  })
  .strict();

const implementationStatusSchema = z
  .object({
    state: implementationStateSchema,
    delivery: implementationDeliverySchema,
  })
  .strict();

const audienceTopologySchema = z
  .object({
    visibility: z.array(audienceLevelSchema).min(1),
    autoExpandDepth: z
      .object({
        executive: tierSchema,
        developer: tierSchema,
        maintainer: tierSchema,
      })
      .strict(),
  })
  .strict();

const sceneAudienceSchema = z
  .object({
    visibility: z.array(audienceLevelSchema).min(1),
    defaultDepth: z
      .object({
        executive: tierSchema,
        developer: tierSchema,
        maintainer: tierSchema,
      })
      .strict(),
  })
  .strict();

const seamPortSchema = z
  .object({
    id: architectureIdSchema,
    name: z.string().trim().min(1),
    channel: flowChannelSchema,
  })
  .strict();

export const graphNodeSchema = z
  .object({
    id: architectureIdSchema,
    tier: tierSchema,
    parentId: architectureIdSchema.nullable(),
    childIds: z.array(architectureIdSchema),
    owner: ownershipSchema,
    status: implementationStatusSchema,
    flavors: z.array(executionFlavorSchema).min(1),
    title: audienceCopySchema,
    summary: audienceCopySchema,
    evidence: z.array(evidenceReferenceSchema).min(1),
    seamPorts: z.array(seamPortSchema).min(1),
    audience: audienceTopologySchema,
    footnotes: z.array(audienceCopySchema).default([]),
  })
  .strict();

export const graphEdgeSchema = z
  .object({
    id: architectureIdSchema,
    source: z
      .object({
        nodeId: architectureIdSchema,
        portId: architectureIdSchema,
      })
      .strict(),
    target: z
      .object({
        nodeId: architectureIdSchema,
        portId: architectureIdSchema,
      })
      .strict(),
    channel: flowChannelSchema,
    status: implementationStatusSchema,
    flavors: z.array(executionFlavorSchema).min(1),
    protocol: z.string().trim().min(1),
    evidence: z.array(evidenceReferenceSchema).min(1),
    footnotes: z.array(audienceCopySchema).default([]),
  })
  .strict();

export const graphSceneSchema = z
  .object({
    id: architectureIdSchema,
    title: z.string().trim().min(1),
    rustScene: z.boolean(),
    nodeIds: z.array(architectureIdSchema).min(1),
    edgeIds: z.array(architectureIdSchema),
    audience: sceneAudienceSchema,
  })
  .strict();

const copyEntitySchema = z.object({
  id: architectureIdSchema,
  title: audienceCopySchema,
  summary: audienceCopySchema,
});

export const componentSchema = copyEntitySchema
  .extend({
    kind: z.literal("component"),
    owner: ownershipSchema,
    lifecycleBand: lifecycleBandSchema,
    status: architectureStatusSchema,
    evidence: z.array(evidenceReferenceSchema).min(1),
    modes: z.array(executionModeSchema),
    contracts: z.array(z.string().trim().min(1)),
    crateIds: z.array(architectureIdSchema).default([]),
  })
  .strict();

const edgeBaseSchema = z.object({
  id: architectureIdSchema,
  from: architectureIdSchema,
  to: architectureIdSchema,
  label: z.string().trim().min(1),
  status: architectureStatusSchema,
  evidence: z.array(evidenceReferenceSchema).min(1),
});

export const architectureEdgeSchema = z.discriminatedUnion("kind", [
  edgeBaseSchema
    .extend({
      kind: z.literal("message"),
      protocol: z.string().trim().min(1),
    })
    .strict(),
  edgeBaseSchema
    .extend({
      kind: z.literal("dependency"),
      contract: z.string().trim().min(1),
    })
    .strict(),
  edgeBaseSchema
    .extend({
      kind: z.literal("control"),
      control: z.string().trim().min(1),
    })
    .strict(),
]);

export const riskSchema = copyEntitySchema
  .extend({
    kind: z.literal("risk"),
    status: architectureStatusSchema,
    severity: z.enum(["low", "medium", "high"]),
    componentIds: z.array(architectureIdSchema).min(1),
    evidence: z.array(evidenceReferenceSchema).min(1),
  })
  .strict();

export const lifecycleStageSchema = copyEntitySchema
  .extend({
    kind: z.literal("lifecycle"),
    order: z.number().int().nonnegative(),
    componentIds: z.array(architectureIdSchema).min(1),
    evidence: z.array(evidenceReferenceSchema).min(1),
  })
  .strict();

export const architectureViewSchema = copyEntitySchema
  .extend({
    kind: z.literal("view"),
    route: z.string().startsWith("/"),
    componentIds: z.array(architectureIdSchema).min(1),
    edgeIds: z.array(architectureIdSchema),
    riskIds: z.array(architectureIdSchema),
  })
  .strict();

export const pairSupportSchema = z
  .object({
    id: architectureIdSchema,
    mode: executionModeSchema,
    workload: workloadSchema,
    status: architectureStatusSchema,
    notes: audienceCopySchema,
    evidence: z.array(evidenceReferenceSchema).min(1),
  })
  .strict();

export const crateReferenceSchema = copyEntitySchema
  .extend({
    kind: z.literal("crate"),
    packageName: z.string().regex(/^[a-z][a-z0-9-]+$/),
    path: z.string().regex(/^crates\/[a-z0-9-]+$/),
    status: architectureStatusSchema,
    responsibility: audienceCopySchema,
    keySourcePaths: z.array(z.string().regex(repositoryPathPattern)).min(1),
    dependencies: z.array(
      z
        .object({
          crateId: architectureIdSchema,
          kind: cargoDependencyKindSchema,
        })
        .strict(),
    ),
    contracts: z.array(z.string().trim().min(1)),
    modes: z.array(executionModeSchema),
    parityScars: z.array(z.string().trim().min(1)),
    evidence: z.array(evidenceReferenceSchema).min(1),
  })
  .strict();

export const architectureCatalogSchema = z
  .object({
    schemaVersion: z.literal(2),
    components: z.array(componentSchema).min(1),
    edges: z.array(architectureEdgeSchema),
    risks: z.array(riskSchema),
    lifecycleStages: z.array(lifecycleStageSchema),
    views: z.array(architectureViewSchema).min(1),
    crates: z.array(crateReferenceSchema),
    pairSupport: z.array(pairSupportSchema),
    graphNodes: z.array(graphNodeSchema).default([]),
    graphEdges: z.array(graphEdgeSchema).default([]),
    graphScenes: z.array(graphSceneSchema).default([]),
  })
  .strict();

export type Ownership = z.infer<typeof ownershipSchema>;
export type ArchitectureStatus = z.infer<typeof architectureStatusSchema>;
export type ExecutionMode = z.infer<typeof executionModeSchema>;
export type LifecycleBand = z.infer<typeof lifecycleBandSchema>;
export type CargoDependencyKind = z.infer<typeof cargoDependencyKindSchema>;
export type Workload = z.infer<typeof workloadSchema>;
export type AudienceCopy = z.infer<typeof audienceCopySchema>;
export type EvidenceReference = z.infer<typeof evidenceReferenceSchema>;
export type ArchitectureComponent = z.infer<typeof componentSchema>;
export type ArchitectureEdge = z.infer<typeof architectureEdgeSchema>;
export type ArchitectureRisk = z.infer<typeof riskSchema>;
export type LifecycleStage = z.infer<typeof lifecycleStageSchema>;
export type ArchitectureView = z.infer<typeof architectureViewSchema>;
export type PairSupport = z.infer<typeof pairSupportSchema>;
export type CrateReference = z.infer<typeof crateReferenceSchema>;
export type ArchitectureCatalog = z.infer<typeof architectureCatalogSchema>;
export type Tier = z.infer<typeof tierSchema>;
export type AudienceLevel = z.infer<typeof audienceLevelSchema>;
export type FlowChannel = z.infer<typeof flowChannelSchema>;
export type ExecutionFlavor = z.infer<typeof executionFlavorSchema>;
export type ImplementationState = z.infer<typeof implementationStateSchema>;
export type ImplementationDelivery = z.infer<typeof implementationDeliverySchema>;
export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;
export type GraphScene = z.infer<typeof graphSceneSchema>;
