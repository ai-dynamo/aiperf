/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { diagnostic, type Result } from "./diagnostic.js";
import type { SourceRange } from "./source.js";

export const CAPABILITY_KINDS = [
  "primitive",
  "layout",
  "effect",
  "transform",
  "action",
  "asset-loader",
  "exporter",
] as const;

export type CapabilityKind = (typeof CAPABILITY_KINDS)[number];

export type AccessibilityContract = Readonly<{
  requiresLabel: boolean;
  keyboardOperable: boolean;
  screenReaderFallback: boolean;
}>;

export type CapabilityCostModel = Readonly<{
  base: number;
  perNode: number;
}>;

export type CapabilityDescriptor = Readonly<{
  id: string;
  version: string;
  kind: CapabilityKind;
  description: string;
  nodeKinds: readonly string[];
  deterministic: boolean;
  accessibility: AccessibilityContract;
  fallback: string;
  cost: CapabilityCostModel;
}>;

export type CapabilityRegistryManifest = Readonly<{
  capabilities: readonly CapabilityDescriptor[];
}>;

const manifestRange: SourceRange = {
  source: "<capability-manifest>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Creates a deterministic manifest or diagnoses duplicate capability IDs. */
export function createCapabilityManifest(
  descriptors: readonly CapabilityDescriptor[],
): Result<CapabilityRegistryManifest> {
  const capabilities = [...descriptors].sort(({ id: left }, { id: right }) =>
    left.localeCompare(right),
  );
  const duplicate = capabilities.find(
    ({ id }, index) => index > 0 && capabilities[index - 1]?.id === id,
  );

  if (duplicate !== undefined) {
    return {
      ok: false,
      diagnostics: [
        diagnostic(
          "CAPABILITY_DUPLICATE",
          "error",
          `Duplicate capability ID "${duplicate.id}".`,
          manifestRange,
          "Use a unique capability ID.",
        ),
      ],
    };
  }

  return { ok: true, value: { capabilities }, diagnostics: [] };
}

const accessibleVisual: AccessibilityContract = {
  requiresLabel: true,
  keyboardOperable: false,
  screenReaderFallback: true,
};

const interactive: AccessibilityContract = {
  requiresLabel: true,
  keyboardOperable: true,
  screenReaderFallback: true,
};

const standardCost: CapabilityCostModel = { base: 1, perNode: 1 };

const foundationDescriptors: readonly CapabilityDescriptor[] = [
  {
    id: "core.group",
    version: "1.0.0",
    kind: "primitive",
    description: "Groups ordered render nodes.",
    nodeKinds: ["group"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: standardCost,
  },
  {
    id: "core.rect",
    version: "1.0.0",
    kind: "primitive",
    description: "Renders a rectangle.",
    nodeKinds: ["rect"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.rect",
    cost: standardCost,
  },
  {
    id: "core.text",
    version: "1.0.0",
    kind: "primitive",
    description: "Renders plain text.",
    nodeKinds: ["text"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.text",
    cost: standardCost,
  },
  {
    id: "core.connector",
    version: "1.0.0",
    kind: "primitive",
    description: "Connects two render nodes.",
    nodeKinds: ["connector"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.connector",
    cost: standardCost,
  },
  {
    id: "core.camera",
    version: "1.0.0",
    kind: "transform",
    description: "Applies deterministic camera keyframes.",
    nodeKinds: ["camera-keyframe"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: standardCost,
  },
  {
    id: "core.timeline",
    version: "1.0.0",
    kind: "effect",
    description: "Sequences finite timeline cues.",
    nodeKinds: ["timeline-cue"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: standardCost,
  },
  {
    id: "core.inspect",
    version: "1.0.0",
    kind: "action",
    description: "Exposes accessible node inspection.",
    nodeKinds: ["interaction"],
    deterministic: true,
    accessibility: interactive,
    fallback: "core.text",
    cost: standardCost,
  },
];

const foundationManifest = createCapabilityManifest(foundationDescriptors);

if (!foundationManifest.ok) {
  throw new Error(foundationManifest.diagnostics[0]?.message);
}

export const FOUNDATION_CAPABILITIES: CapabilityRegistryManifest =
  foundationManifest.value;

const leafCost: CapabilityCostModel = { base: 2, perNode: 0 };
const hybridCost: CapabilityCostModel = { base: 3, perNode: 1 };

/** P0 hybrid stdlib components and their narrow deterministic leaves. */
const p0Descriptors: readonly CapabilityDescriptor[] = [
  {
    id: "core.glyph-run",
    version: "1.0.0",
    kind: "primitive",
    description: "Measured glyph run with stable grapheme span ids.",
    nodeKinds: ["component"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.text",
    cost: hybridCost,
  },
  {
    id: "core.span-map",
    version: "1.0.0",
    kind: "layout",
    description: "Maps source spans onto target spans with coverage checks.",
    nodeKinds: ["component"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: hybridCost,
  },
  {
    id: "core.semantic-morph",
    version: "1.0.0",
    kind: "effect",
    description: "Correspondence table for semantic morphs across timeline beats.",
    nodeKinds: ["component"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: hybridCost,
  },
  {
    id: "core.segment-strip",
    version: "1.0.0",
    kind: "layout",
    description: "Ordered segment strip with nested packing and continuation.",
    nodeKinds: ["component"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: hybridCost,
  },
  {
    id: "viz.queue",
    version: "1.0.0",
    kind: "layout",
    description: "Queue visualization driven by a deterministic policy simulation.",
    nodeKinds: ["component"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: hybridCost,
  },
  {
    id: "viz.waterfall",
    version: "1.0.0",
    kind: "layout",
    description: "Nested interval waterfall across ordered lifecycle lanes.",
    nodeKinds: ["component"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: hybridCost,
  },
  {
    id: "leaf.glyph-measure",
    version: "1.0.0",
    kind: "transform",
    description: "Grapheme boundary measurement leaf for glyph runs.",
    nodeKinds: ["leaf"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.text",
    cost: leafCost,
  },
  {
    id: "leaf.span-interval",
    version: "1.0.0",
    kind: "transform",
    description: "Span overlap index and coverage projection leaf.",
    nodeKinds: ["leaf"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: leafCost,
  },
  {
    id: "core.segment-strip.layout",
    version: "1.0.0",
    kind: "layout",
    description: "Deterministic nested strip packing layout leaf.",
    nodeKinds: ["leaf"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: leafCost,
  },
  {
    id: "viz.queue.policy",
    version: "1.0.0",
    kind: "transform",
    description: "FIFO and priority queue policy simulation leaf.",
    nodeKinds: ["leaf"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: leafCost,
  },
  {
    id: "viz.waterfall.nest-layout",
    version: "1.0.0",
    kind: "layout",
    description: "Nested interval lane layout leaf for waterfalls.",
    nodeKinds: ["leaf"],
    deterministic: true,
    accessibility: accessibleVisual,
    fallback: "core.group",
    cost: leafCost,
  },
];

const p0Manifest = createCapabilityManifest([
  ...foundationDescriptors,
  ...p0Descriptors,
]);

if (!p0Manifest.ok) {
  throw new Error(p0Manifest.diagnostics[0]?.message);
}

/** Foundation plus P0 hybrid/leaf capability registry. */
export const P0_CAPABILITIES: CapabilityRegistryManifest = p0Manifest.value;

/** P0-only capability ids for compile-time require checks. */
export const P0_CAPABILITY_IDS: readonly string[] = p0Descriptors.map(
  ({ id }) => id,
);
