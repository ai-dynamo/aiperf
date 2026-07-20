/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure visual geometry for native semantic chrome capabilities.

import type {
  SceneGeometryLike,
  SceneNodeLike,
} from "../scene-types.js";
import {
  DETAIL_HEIGHT,
  INSET,
  STEPPER_CHIP_HEIGHT,
  TITLE_HEIGHT,
  stepperChipWidth,
} from "../text-metrics.js";

export type SemanticTextPart = Readonly<{
  id: string;
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize: number;
  fontWeight?: string;
  fontFamily?: string;
  fontStyle?: string;
  whiteSpace?: string;
  anchor: "start" | "middle" | "end";
  tone?: "primary" | "secondary";
  inkRole?: string;
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

function presentationOf(node: SceneNodeLike): string | undefined {
  return stringProp(node, "presentation");
}

function radiusOf(node: SceneNodeLike, fallback: number): number {
  return typeof node.style?.radius === "number" &&
    Number.isFinite(node.style.radius)
    ? node.style.radius
    : fallback;
}

function inkRoleOf(node: SceneNodeLike): string | undefined {
  return stringProp(node, "inkRole");
}

/** Whether a semantic node carries renderer-owned visual content. */
export function hasNativeSemanticChrome(node: SceneNodeLike): boolean {
  const capability = node.capabilityId ?? node.capability ?? "";
  if (node.props === undefined) {
    return false;
  }
  const presentation = presentationOf(node);
  if (
    presentation === "code-block" ||
    presentation === "quote" ||
    presentation === "avatar" ||
    presentation === "icon-label"
  ) {
    return true;
  }
  return (
    capability.startsWith("diagram.") ||
    [
      "core.panel",
      "core.header",
      "core.chip",
      "core.note",
      "core.lane",
      "core.band",
      "core.stepper",
      "layout.frame",
    ].includes(capability)
  );
}

/** Resolve renderer-owned boxes and text without generating Scene IR children. */
export function resolveSemanticChrome(
  node: SceneNodeLike,
  geometry: SceneGeometryLike,
): SemanticChrome {
  const capability = node.capabilityId ?? node.capability ?? "";
  const presentation = presentationOf(node);
  const inkRole = inkRoleOf(node);

  if (presentation === "code-block" || presentation === "quote") {
    const text = stringProp(node, "text") ?? "";
    return {
      rootBox: {
        id: `${node.id}__chrome`,
        geometry,
        radius: radiusOf(node, 6),
      },
      boxes: [],
      texts: [
        {
          id: `${node.id}__text`,
          text,
          x: geometry.x + 12,
          y: geometry.y + 10,
          width: Math.max(geometry.width - 24, 0),
          height: Math.max(geometry.height - 20, 0),
          fontSize: 12,
          fontFamily: presentation === "code-block" ? "monospace" : undefined,
          fontStyle: presentation === "quote" ? "italic" : undefined,
          whiteSpace: presentation === "code-block" ? "pre" : undefined,
          anchor: "start",
          ...(inkRole !== undefined ? { inkRole } : {}),
        },
      ],
    };
  }

  if (presentation === "avatar") {
    return {
      rootBox: {
        id: `${node.id}__chrome`,
        geometry,
        radius: radiusOf(node, Math.max(geometry.width, geometry.height) / 2),
      },
      boxes: [],
      texts: [],
    };
  }

  if (presentation === "icon-label") {
    const label = stringProp(node, "label") ?? stringProp(node, "text") ?? "";
    return {
      rootBox: {
        id: `${node.id}__chrome`,
        geometry,
        radius: radiusOf(node, 8),
      },
      boxes: [],
      texts: [
        {
          id: `${node.id}__label`,
          text: label,
          x: geometry.x + 40,
          y: geometry.y + 8,
          width: Math.max(geometry.width - 48, 0),
          height: Math.max(geometry.height - 16, 0),
          fontSize: 12,
          anchor: "start",
          ...(inkRole !== undefined ? { inkRole } : {}),
        },
      ],
    };
  }

  const title =
    stringProp(node, "title") ??
    stringProp(node, "label") ??
    stringProp(node, "text");
  const detail =
    stringProp(node, "detail") ?? stringProp(node, "caption");
  const subtitle = stringProp(node, "subtitle");
  const isDiagram = capability.startsWith("diagram.");
  const isDiagramBoundary = capability === "diagram.boundary";
  if (capability === "core.stepper") {
    const steps = stringArrayProp(node, "steps");
    const gap = gapOf(node);
    let cursorX = geometry.x;
    const boxes: SemanticBoxPart[] = [];
    const texts: SemanticTextPart[] = [];
    steps.forEach((step, index) => {
      const width = stepperChipWidth(step, index);
      boxes.push({
        id: `${node.id}__step-${index}`,
        geometry: {
          x: cursorX,
          y: geometry.y,
          width,
          height: STEPPER_CHIP_HEIGHT,
        },
        radius: 4,
      });
      texts.push({
        id: `${node.id}__step-${index}-label`,
        text: `${index + 1}. ${step}`,
        x: cursorX,
        y: geometry.y,
        width,
        height: STEPPER_CHIP_HEIGHT,
        fontSize: 11,
        fontWeight: "bold",
        anchor: "middle",
        ...(inkRole !== undefined ? { inkRole } : {}),
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
      x: centered
        ? geometry.x
        : geometry.x + (isDiagramBoundary ? 12 : isDiagram ? 46 : INSET),
      y:
        capability === "core.chip"
          ? geometry.y
          : isDiagramBoundary
            ? geometry.y + 8
            : isDiagram
              ? geometry.y + (detail === undefined ? 20 : 12)
              : geometry.y + INSET + (centered ? 2 : 0),
      width: centered
        ? geometry.width
        : Math.max(
            geometry.width -
              (isDiagramBoundary ? 24 : isDiagram ? 56 : INSET * 2),
            0,
          ),
      height: capability === "core.chip" ? geometry.height : TITLE_HEIGHT,
      fontSize:
        capability === "core.header"
          ? 13
          : capability === "core.chip"
            ? 11
            : isDiagramBoundary
              ? 12
              : isDiagram
                ? 13
                : 14,
      fontWeight: "bold",
      anchor: centered ? "middle" : "start",
      ...(inkRole !== undefined ? { inkRole } : {}),
    });
  }
  if (detail !== undefined) {
    texts.push({
      id: `${node.id}__detail`,
      text: detail,
      x: geometry.x + (isDiagram ? 46 : INSET),
      y: isDiagram
        ? geometry.y + 38
        : geometry.y + INSET + TITLE_HEIGHT + 4,
      width: Math.max(geometry.width - (isDiagram ? 56 : INSET * 2), 0),
      height: DETAIL_HEIGHT,
      fontSize: isDiagram ? 10 : 11.5,
      anchor: capability === "core.panel" ? "middle" : "start",
      tone: "secondary",
      ...(inkRole !== undefined ? { inkRole } : {}),
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
      ...(inkRole !== undefined ? { inkRole } : {}),
    });
  }
  return {
    rootBox: {
      id: `${node.id}__chrome`,
      geometry,
      radius: radiusOf(
        node,
        capability === "core.chip"
          ? Math.max(geometry.height / 2, 4)
          : capability === "core.band"
            ? 6
            : 8,
      ),
    },
    boxes: [],
    texts,
  };
}
