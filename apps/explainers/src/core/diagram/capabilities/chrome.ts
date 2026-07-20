/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure visual geometry for native semantic chrome capabilities.

import type {
  SceneGeometryLike,
  SceneNodeLike,
} from "../SceneRenderer.js";

export type SemanticTextPart = Readonly<{
  id: string;
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize: number;
  fontWeight?: string;
  anchor: "start" | "middle" | "end";
  tone?: "primary" | "secondary";
}>;

export type SemanticBoxPart = Readonly<{
  id: string;
  geometry: SceneGeometryLike;
  radius: number;
}>;

export type SemanticChrome = Readonly<{
  rootBox?: SemanticBoxPart;
  boxes: readonly SemanticBoxPart[];
  texts: readonly SemanticTextPart[];
}>;

const INSET = 8;
const TITLE_HEIGHT = 22;
const DETAIL_HEIGHT = 20;
const STEPPER_HEIGHT = 26;
const STEPPER_MIN_WIDTH = 72;
const STEPPER_CHAR_WIDTH = 6.2;
const STEPPER_PAD = 24;

function stringProp(node: SceneNodeLike, key: string): string | undefined {
  const value = node.props?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function stringArrayProp(
  node: SceneNodeLike,
  key: string,
): readonly string[] {
  const value = node.props?.[key];
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is string =>
          typeof entry === "string" && entry.length > 0,
      )
    : [];
}

function gapOf(node: SceneNodeLike, fallback = 12): number {
  const value = node.style?.gap ?? node.props?.gap;
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, value)
    : fallback;
}

function stepWidth(label: string, index: number): number {
  return Math.max(
    STEPPER_MIN_WIDTH,
    Math.ceil(`${index + 1}. ${label}`.length * STEPPER_CHAR_WIDTH) +
      STEPPER_PAD,
  );
}

/** Whether a semantic node carries renderer-owned visual content. */
export function hasNativeSemanticChrome(node: SceneNodeLike): boolean {
  return (
    node.props !== undefined &&
    [
      "core.panel",
      "core.header",
      "core.chip",
      "core.note",
      "core.lane",
      "core.band",
      "core.stepper",
    ].includes(node.capabilityId ?? node.capability ?? "")
  );
}

/** Resolve renderer-owned boxes and text without generating Scene IR children. */
export function resolveSemanticChrome(
  node: SceneNodeLike,
  geometry: SceneGeometryLike,
): SemanticChrome {
  const capability = node.capabilityId ?? node.capability ?? "";
  const title =
    stringProp(node, "title") ??
    stringProp(node, "label") ??
    stringProp(node, "text");
  const detail =
    stringProp(node, "detail") ?? stringProp(node, "caption");
  const subtitle = stringProp(node, "subtitle");
  if (capability === "core.stepper") {
    const steps = stringArrayProp(node, "steps");
    const gap = gapOf(node);
    let cursorX = geometry.x;
    const boxes: SemanticBoxPart[] = [];
    const texts: SemanticTextPart[] = [];
    steps.forEach((step, index) => {
      const width = stepWidth(step, index);
      boxes.push({
        id: `${node.id}__step-${index}`,
        geometry: {
          x: cursorX,
          y: geometry.y,
          width,
          height: STEPPER_HEIGHT,
        },
        radius: 4,
      });
      texts.push({
        id: `${node.id}__step-${index}-label`,
        text: `${index + 1}. ${step}`,
        x: cursorX,
        y: geometry.y,
        width,
        height: STEPPER_HEIGHT,
        fontSize: 11,
        fontWeight: "bold",
        anchor: "middle",
      });
      cursorX += width + gap;
    });
    return { boxes, texts };
  }

  const texts: SemanticTextPart[] = [];
  if (title !== undefined) {
    const centered =
      capability === "core.panel" ||
      capability === "core.chip" ||
      capability === "core.note";
    texts.push({
      id: `${node.id}__title`,
      text: title,
      x: centered ? geometry.x : geometry.x + INSET,
      y:
        capability === "core.chip"
          ? geometry.y
          : geometry.y + INSET + (centered ? 2 : 0),
      width: centered ? geometry.width : Math.max(geometry.width - INSET * 2, 0),
      height: capability === "core.chip" ? geometry.height : TITLE_HEIGHT,
      fontSize: capability === "core.header" ? 13 : capability === "core.chip" ? 11 : 14,
      fontWeight: "bold",
      anchor: centered ? "middle" : "start",
    });
  }
  if (detail !== undefined) {
    texts.push({
      id: `${node.id}__detail`,
      text: detail,
      x: geometry.x + INSET,
      y: geometry.y + INSET + TITLE_HEIGHT + 4,
      width: Math.max(geometry.width - INSET * 2, 0),
      height: DETAIL_HEIGHT,
      fontSize: 11.5,
      anchor: capability === "core.panel" ? "middle" : "start",
      tone: "secondary",
    });
  }
  if (subtitle !== undefined) {
    texts.push({
      id: `${node.id}__subtitle`,
      text: subtitle,
      x: geometry.x + INSET,
      y: geometry.y + INSET + TITLE_HEIGHT + DETAIL_HEIGHT + 6,
      width: Math.max(geometry.width - INSET * 2, 0),
      height: 16,
      fontSize: 10,
      anchor: "middle",
      tone: "secondary",
    });
  }
  return {
    rootBox: {
      id: `${node.id}__chrome`,
      geometry,
      radius:
        capability === "core.chip"
          ? Math.max(geometry.height / 2, 4)
          : capability === "core.band"
            ? 6
            : 8,
    },
    boxes: [],
    texts,
  };
}

