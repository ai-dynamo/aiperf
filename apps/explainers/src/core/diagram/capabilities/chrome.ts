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
  SUBTITLE_HEIGHT,
  TITLE_HEIGHT,
  stepperChipWidth,
} from "../text-metrics.js";
import { managedLayoutOptions } from "./layout.js";

export type SemanticGeneratedRole =
  | "chrome"
  | "title"
  | "detail"
  | "subtitle"
  | "caption"
  | "label"
  | "step";

export type SemanticTextPart = Readonly<{
  id: string;
  role: SemanticGeneratedRole;
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
  role: SemanticGeneratedRole;
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

function capabilityOf(node: SceneNodeLike): string {
  return node.capabilityId ?? node.capability ?? "";
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

function generatedTextRole(
  node: SceneNodeLike,
  role: "title" | "detail" | "subtitle",
): SemanticGeneratedRole {
  const capability = node.capabilityId ?? node.capability;
  if (capability === "core.chip") return "label";
  if (capability === "core.note") return "caption";
  if (role === "detail" && capability === "core.header") return "caption";
  return role;
}

function generatedTextId(
  node: SceneNodeLike,
  role: "title" | "detail" | "subtitle",
): string {
  return `${node.id}__${generatedTextRole(node, role)}`;
}

/** Whether a semantic node carries renderer-owned visual content. */
export function hasNativeSemanticChrome(node: SceneNodeLike): boolean {
  const capability = node.capabilityId ?? node.capability ?? "";
  if (node.props === undefined) {
    return false;
  }
  // Layout-managed steppers expand into chip (or other) children that own
  // paint. Compatibility `core.step` children still share chrome ownership.
  if (
    capability === "core.stepper" &&
    Array.isArray(node.children) &&
    node.children.length > 0 &&
    node.children.some((child) => capabilityOf(child) !== "core.step")
  ) {
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
        role: "chrome",
        geometry,
        radius: radiusOf(node, 0),
      },
      boxes: [],
      texts: [
        {
          id: `${node.id}__text`,
          role: "label",
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
        role: "chrome",
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
        role: "chrome",
        geometry,
        radius: radiusOf(node, 0),
      },
      boxes: [],
      texts: [
        {
          id: `${node.id}__label`,
          role: "label",
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

  if (capability === "core.stepper") {
    const steps = stringArrayProp(node, "steps");
    const children = node.children ?? [];
    if (
      steps.length === 0 ||
      (children.length > 0 &&
        children.some((child) => capabilityOf(child) !== "core.step"))
    ) {
      return { boxes: [], texts: [] };
    }
    const gap = gapOf(node);
    let cursorX = geometry.x;
    const boxes: SemanticBoxPart[] = [];
    const texts: SemanticTextPart[] = [];
    steps.forEach((step, index) => {
      const stepId = `${node.id}-step-${index}`;
      const width = stepperChipWidth(step, index);
      boxes.push({
        id: stepId,
        role: "step",
        geometry: {
          x: cursorX,
          y: geometry.y,
          width,
          height: STEPPER_CHIP_HEIGHT,
        },
        radius: 0,
      });
      texts.push({
        id: `${stepId}__label`,
        role: "label",
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

  const title =
    stringProp(node, "title") ??
    stringProp(node, "label") ??
    stringProp(node, "text");
  const detail =
    stringProp(node, "detail") ?? stringProp(node, "caption");
  const subtitle = stringProp(node, "subtitle");
  const isDiagram = capability.startsWith("diagram.");
  const isDiagramBoundary = capability === "diagram.boundary";

  // Frames place managed content at `padding`; chrome must share that x-origin.
  const framePadding =
    capability === "layout.frame"
      ? managedLayoutOptions(node).padding
      : undefined;
  const chromeInsetX =
    framePadding !== undefined
      ? framePadding
      : isDiagramBoundary
        ? 12
        : isDiagram
          ? 46
          : INSET;
  const chromeInsetXEnd =
    framePadding !== undefined
      ? framePadding
      : isDiagramBoundary
        ? 12
        : isDiagram
          ? 10
          : INSET;

  const texts: SemanticTextPart[] = [];
  if (title !== undefined) {
    const centered =
      capability === "core.panel" ||
      capability === "core.chip" ||
      capability === "core.note";
    texts.push({
      id: generatedTextId(node, "title"),
      role: generatedTextRole(node, "title"),
      text: title,
      x: centered ? geometry.x : geometry.x + chromeInsetX,
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
        : Math.max(geometry.width - chromeInsetX - chromeInsetXEnd, 0),
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
    const detailInsetX = isDiagram ? 46 : chromeInsetX;
    const detailInsetXEnd = isDiagram ? 10 : chromeInsetXEnd;
    texts.push({
      id: generatedTextId(node, "detail"),
      role: generatedTextRole(node, "detail"),
      text: detail,
      x: geometry.x + detailInsetX,
      y: isDiagram
        ? geometry.y + 38
        : geometry.y + INSET + TITLE_HEIGHT + 4,
      width: Math.max(geometry.width - detailInsetX - detailInsetXEnd, 0),
      height: DETAIL_HEIGHT,
      fontSize: isDiagram ? 10 : 11.5,
      anchor: capability === "core.panel" ? "middle" : "start",
      tone: "secondary",
      ...(inkRole !== undefined ? { inkRole } : {}),
    });
  }
  if (subtitle !== undefined) {
    texts.push({
      id: generatedTextId(node, "subtitle"),
      role: generatedTextRole(node, "subtitle"),
      text: subtitle,
      x: geometry.x + chromeInsetX,
      y: geometry.y + INSET + TITLE_HEIGHT + DETAIL_HEIGHT + 6,
      width: Math.max(geometry.width - chromeInsetX - chromeInsetXEnd, 0),
      height: SUBTITLE_HEIGHT,
      fontSize: 10,
      anchor: "middle",
      tone: "secondary",
      ...(inkRole !== undefined ? { inkRole } : {}),
    });
  }
  return {
    rootBox: {
      id: `${node.id}__chrome`,
      role: "chrome",
      geometry,
      radius: radiusOf(
        node,
        capability === "core.chip"
          ? Math.max(geometry.height / 2, 4)
          : capability === "core.band"
            ? 0
            : 0,
      ),
    },
    boxes: [],
    texts,
  };
}
