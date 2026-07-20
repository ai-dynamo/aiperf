/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared node capability and motion-signal classification for resolver, renderer, and verifier.

import type { SceneNodeLike } from "./scene-types.js";

const DOT_CAPABILITIES = new Set(["core.dot"]);
const DOT_KINDS = new Set(["dot"]);

const ARROW_CAPABILITIES = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
  "core.elbow",
  "core.route",
  "core.fan-out",
  "core.fan-in",
]);

const ARROW_KINDS = new Set([
  "line",
  "path",
  "arrow",
  "connector",
  "elbow",
  "fan",
]);

/** Capabilities that classify a connector-like node as an undirected motion guide. */
const MOTION_SIGNAL_CAPABILITIES = new Set([
  "motion.signal",
  "motion.dot",
  "core.motion",
  "core.motion-signal",
  "motion.motion-signal",
]);

/**
 * Returns a node's canonical or authoring-alias capability.
 *
 * Mirrors resolver three-tier resolution: `capabilityId`, then `capability`,
 * then a `core.${kind}` fallback so kind-only nodes still classify correctly.
 */
export function capabilityOf(node: SceneNodeLike): string {
  if (typeof node.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  return typeof node.kind === "string" && node.kind.length > 0
    ? `core.${node.kind}`
    : "";
}

/** Whether a node describes a connector or path stroke. */
export function isArrowLike(node: SceneNodeLike): boolean {
  const capability = capabilityOf(node);
  const kind = node.kind ?? "";
  return ARROW_CAPABILITIES.has(capability) || ARROW_KINDS.has(kind);
}

/** Small filled dots only — circles/ellipses render as shapes, not motion dots. */
export function isDotLike(node: SceneNodeLike, capability = ""): boolean {
  const cap = capability.length > 0 ? capability : capabilityOf(node);
  if (DOT_CAPABILITIES.has(cap)) {
    return true;
  }
  if (typeof node.kind === "string" && DOT_KINDS.has(node.kind)) {
    return true;
  }
  // Never promote rect / panel chrome: authors use `r` as corner radius.
  if (
    cap === "core.rect" ||
    cap === "core.panel" ||
    cap === "core.header" ||
    cap === "core.circle" ||
    cap === "core.ellipse" ||
    node.kind === "rect" ||
    node.kind === "circle" ||
    node.kind === "ellipse"
  ) {
    return false;
  }
  // Legacy bare nodes with `style.r` and no capability → small motion/dot mark.
  if (cap.length > 0) {
    return false;
  }
  const radius = node.style?.r;
  return typeof radius === "number" && Number.isFinite(radius) && radius > 0;
}

/**
 * Traveling MentalModel-style motion guides (often authored as `motion-sig` paths).
 *
 * Shared by canonical resolver direction policy, SceneRenderer paint, and verifier
 * orphan checks so authored foundation nodes cannot diverge across layers.
 */
export function isMotionSignalNode(
  node: SceneNodeLike,
  capability = "",
): boolean {
  const cap = capability.length > 0 ? capability : capabilityOf(node);
  if (isDotLike(node, cap)) {
    return false;
  }
  if (MOTION_SIGNAL_CAPABILITIES.has(cap)) {
    return true;
  }
  if (/motion[-_]?sig/i.test(node.id)) {
    return true;
  }
  if (/^motion\d+$/i.test(node.id)) {
    return true;
  }
  if (/motion/i.test(node.id) && isArrowLike(node)) {
    return true;
  }
  const label = (node.accessibility?.label ?? "").toLowerCase();
  if (label.includes("motion signal")) {
    return true;
  }
  const motion = node.style?.motion;
  const role = node.style?.role;
  return (
    motion === true ||
    motion === 1 ||
    motion === "signal" ||
    motion === "dot" ||
    role === "motion" ||
    role === "motion-signal"
  );
}
