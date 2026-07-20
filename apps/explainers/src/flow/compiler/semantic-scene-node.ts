/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct lowering from package records to native semantic Scene IR.

import type {
  JsonValue,
  RenderNodeIr,
  SourceRange,
} from "../schema/index.js";
import {
  capabilityKind,
  lowerFirstClassPackageNode,
} from "./package-node-lower.js";

const STRUCTURAL_KEYS = new Set([
  "id",
  "kind",
  "capability",
  "capabilityId",
  "geometry",
  "layout",
  "style",
  "accessibility",
  "fallback",
  "sourceMap",
  "children",
  "path",
  "points",
  "from",
  "to",
  "via",
  "axis",
  "junction",
  "edgeRef",
  "text",
]);

export type SemanticNodeLoweringCommon = Readonly<{
  id: string;
  capability: string;
  children: readonly RenderNodeIr[];
  label: string;
  description?: string;
  fallback: string;
  sourceMap: SourceRange;
}>;

function jsonValue(value: unknown): JsonValue | undefined {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  if (Array.isArray(value)) {
    const entries: JsonValue[] = [];
    for (const entry of value) {
      const converted = jsonValue(entry);
      if (converted === undefined) {
        return undefined;
      }
      entries.push(converted);
    }
    return entries;
  }
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  const record: Record<string, JsonValue> = {};
  for (const [key, entry] of Object.entries(value)) {
    const converted = jsonValue(entry);
    if (converted !== undefined) {
      record[key] = converted;
    }
  }
  return record;
}

/** Retain capability-authored values while excluding shared IR structure. */
function semanticProps(
  node: Readonly<Record<string, unknown>>,
): Readonly<Record<string, JsonValue>> {
  const props: Record<string, JsonValue> = {};
  for (const [key, value] of Object.entries(node)) {
    if (STRUCTURAL_KEYS.has(key)) {
      continue;
    }
    const converted = jsonValue(value);
    if (converted !== undefined) {
      props[key] = converted;
    }
  }
  return props;
}

/**
 * Lower one normalized package node without generating visual-only children.
 */
export function lowerSemanticSceneNode(
  node: Readonly<Record<string, unknown>>,
  common: SemanticNodeLoweringCommon,
): RenderNodeIr {
  const lowered = lowerFirstClassPackageNode(
    node as Record<string, unknown>,
    {
      id: common.id,
      capability: common.capability,
      kind: capabilityKind(common.capability),
      children: common.children,
      label: common.label,
      fallback: common.fallback,
      ...(common.description !== undefined
        ? { description: common.description }
        : {}),
    },
  );
  const props = semanticProps(node);
  const edgeRef =
    typeof node.edgeRef === "string" && node.edgeRef.length > 0
      ? node.edgeRef
      : undefined;
  if (lowered.kind === "connector" && edgeRef !== undefined) {
    const { from: _from, to: _to, ...edgeBoundSignal } = lowered;
    return {
      ...edgeBoundSignal,
      sourceMap: common.sourceMap,
      edgeRef,
      ...(Object.keys(props).length > 0 ? { props } : {}),
    } as RenderNodeIr;
  }
  return {
    ...lowered,
    sourceMap: common.sourceMap,
    ...(Object.keys(props).length > 0 ? { props } : {}),
  } as RenderNodeIr;
}

