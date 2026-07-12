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
