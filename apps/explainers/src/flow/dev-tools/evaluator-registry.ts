/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dev-tools CapabilityEvaluator registry over vendored contribution leaves.

import type { ComponentNodeIr } from "../schema/index.js";
import { contributeGlyphRun } from "../runtime/evaluate/contributions/glyph-run.js";
import { contributeQueue } from "../runtime/evaluate/contributions/queue.js";
import { contributeSegmentStrip } from "../runtime/evaluate/contributions/segment-strip.js";
import { contributeSemanticMorph } from "../runtime/evaluate/contributions/semantic-morph.js";
import { contributeSpanMap } from "../runtime/evaluate/contributions/span-map.js";
import { contributeWaterfall } from "../runtime/evaluate/contributions/waterfall.js";
import {
  CapabilityEvaluatorRegistry,
  type CapabilityContribution,
  type CapabilityEvaluator,
  type FrozenCapabilityEvaluatorRegistry,
} from "../runtime/evaluate/registry.js";
import type { DrawCommand, HitRegion } from "../runtime/display-list.js";
import type {
  SemanticEntityProjection,
  SemanticRelationProjection,
} from "../runtime/evaluate/types.js";

type UnknownRecord = Readonly<Record<string, unknown>>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function stringProp(props: UnknownRecord, key: string): string | undefined {
  const value = props[key];
  return typeof value === "string" ? value : undefined;
}

function numberProp(props: UnknownRecord, key: string): number | undefined {
  const value = props[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function arrayProp<T>(props: UnknownRecord, key: string): readonly T[] {
  const value = props[key];
  return Array.isArray(value) ? (value as readonly T[]) : [];
}

function fromContribution(products: {
  commands: readonly DrawCommand[];
  hitRegions?: readonly HitRegion[];
  semanticEntities?: readonly SemanticEntityProjection[];
  semanticRelations?: readonly SemanticRelationProjection[];
}): CapabilityContribution {
  const entities = products.semanticEntities ?? [];
  const relations = products.semanticRelations ?? [];
  return {
    display: {
      commands: products.commands,
      hitRegions: products.hitRegions ?? [],
    },
    semantic: {
      entities,
      relations,
      readingOrder: [
        ...entities.map(({ id }) => id),
        ...relations.map(({ id }) => id),
      ],
    },
  };
}

function glyphRunEvaluator(): CapabilityEvaluator {
  return {
    capabilityId: "core.glyph-run",
    evaluate(node: ComponentNodeIr): CapabilityContribution {
      const props = record(node.props);
      const style = record(node.style);
      const text = stringProp(props, "text") ?? "";
      const fontFamily =
        stringProp(record(props.font), "family") ??
        (typeof style.fontFamily === "string" ? style.fontFamily : "sans-serif");
      const fontSize =
        numberProp(record(props.font), "sizePx") ??
        (typeof style.fontSize === "number" ? style.fontSize : 16);
      const contribution = contributeGlyphRun({
        id: node.id,
        text,
        bounds: node.geometry,
        origin: {
          x: node.geometry.x,
          y: node.geometry.y + node.geometry.height,
        },
        font: {
          family: fontFamily,
          sizePx: fontSize,
          ...(numberProp(record(props.font), "weight") !== undefined
            ? { weight: numberProp(record(props.font), "weight") }
            : typeof style.fontWeight === "number"
              ? { weight: style.fontWeight }
              : {}),
        },
        ...(typeof style.fill === "string"
          ? { fill: style.fill }
          : stringProp(props, "fill") !== undefined
            ? { fill: stringProp(props, "fill") }
            : {}),
        ...(stringProp(props, "locale") !== undefined
          ? { locale: stringProp(props, "locale") }
          : {}),
      });
      return fromContribution(contribution);
    },
  };
}

function segmentStripEvaluator(): CapabilityEvaluator {
  return {
    capabilityId: "core.segment-strip",
    evaluate(node: ComponentNodeIr): CapabilityContribution {
      const props = record(node.props);
      const contribution = contributeSegmentStrip({
        id: node.id,
        segments: arrayProp(props, "segments"),
        layout: record(props.layout) as never,
        ...(props.style !== undefined ? { style: props.style as never } : {}),
      });
      return fromContribution(contribution);
    },
  };
}

function queueEvaluator(): CapabilityEvaluator {
  return {
    capabilityId: "viz.queue",
    evaluate(node: ComponentNodeIr, context): CapabilityContribution {
      const props = record(node.props);
      const contribution = contributeQueue({
        id: node.id,
        arrivals: arrayProp(props, "arrivals"),
        policy: (props.policy ?? "fifo") as never,
        atMs: context.atMs,
        bounds: node.geometry,
        ...(numberProp(props, "capacity") !== undefined
          ? { capacity: numberProp(props, "capacity") }
          : {}),
        ...(numberProp(props, "chipWidth") !== undefined
          ? { chipWidth: numberProp(props, "chipWidth") }
          : {}),
        ...(numberProp(props, "padding") !== undefined
          ? { padding: numberProp(props, "padding") }
          : {}),
        ...(numberProp(props, "gap") !== undefined
          ? { gap: numberProp(props, "gap") }
          : {}),
      });
      return fromContribution(contribution);
    },
  };
}

function waterfallEvaluator(): CapabilityEvaluator {
  return {
    capabilityId: "viz.waterfall",
    evaluate(node: ComponentNodeIr, context): CapabilityContribution {
      const props = record(node.props);
      const contribution = contributeWaterfall({
        id: node.id,
        events: arrayProp(props, "events"),
        layout: record(props.layout) as never,
        atMs: context.atMs,
        ...(props.reducedMotion === true ? { reducedMotion: true } : {}),
        ...(props.style !== undefined ? { style: props.style as never } : {}),
      });
      return fromContribution(contribution);
    },
  };
}

function spanMapEvaluator(): CapabilityEvaluator {
  return {
    capabilityId: "core.span-map",
    evaluate(node: ComponentNodeIr): CapabilityContribution {
      const props = record(node.props);
      const contribution = contributeSpanMap({
        id: node.id,
        source: node.sourceMap,
        spans: arrayProp(props, "spans"),
        edges: arrayProp(props, "edges"),
        ...(props.requireCover !== undefined
          ? { requireCover: props.requireCover as never }
          : {}),
      });
      return fromContribution(contribution);
    },
  };
}

function semanticMorphEvaluator(): CapabilityEvaluator {
  return {
    capabilityId: "core.semantic-morph",
    evaluate(node: ComponentNodeIr, context): CapabilityContribution {
      const props = record(node.props);
      const contribution = contributeSemanticMorph({
        id: node.id,
        atMs: context.atMs,
        startMs: numberProp(props, "startMs") ?? 0,
        durationMs: numberProp(props, "durationMs") ?? 0,
        sources: arrayProp(props, "sources"),
        targets: arrayProp(props, "targets"),
        correspondences: arrayProp(props, "correspondences"),
        ...(props.reducedMotion === true ? { reducedMotion: true } : {}),
        ...(typeof props.reducedMotionPolicy === "string"
          ? { reducedMotionPolicy: props.reducedMotionPolicy as never }
          : {}),
        ...(typeof props.fill === "string" ? { fill: props.fill } : {}),
      });
      return fromContribution(contribution);
    },
  };
}

/** Builds the frozen evaluator registry for browser developer diagnostics. */
export function createDevEvaluatorRegistry(): FrozenCapabilityEvaluatorRegistry {
  return new CapabilityEvaluatorRegistry([
    glyphRunEvaluator(),
    segmentStripEvaluator(),
    queueEvaluator(),
    waterfallEvaluator(),
    spanMapEvaluator(),
    semanticMorphEvaluator(),
  ]).freeze();
}

/** Capability ids registered for the developer evaluator oracle. */
export function registeredDevCapabilityIds(
  registry: FrozenCapabilityEvaluatorRegistry = createDevEvaluatorRegistry(),
): ReadonlySet<string> {
  return new Set(registry.capabilityIds());
}
