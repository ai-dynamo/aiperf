// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure, framework-agnostic two-pass flow-layout engine.
 *
 * This module intentionally has no React/DOM/SVG dependency. It measures a
 * tree of {@link FlowNode}s bottom-up (leaves via their own `measure`,
 * containers by summing/maxing children along their axis) and then positions
 * every node top-down from the root's resolved box, mirroring the semantics of
 * real flow/flexbox layout. It is deliberately independent of the existing
 * `core/diagram/capabilities/layout.ts` module.
 */

/** Width (and optional height) budget handed to a node during measurement. */
export type FlowConstraint = { maxWidth: number; maxHeight?: number };

/** A resolved intrinsic size for a node. */
export type FlowSize = { width: number; height: number };

/**
 * A layout node. Leaf nodes supply a `measure` callback that sizes themselves
 * against a width constraint (e.g. text wrapping); container nodes omit
 * `measure` and are sized from their children by the engine.
 */
export type FlowNode = {
  id: string;
  // Leaf nodes measure themselves given a width constraint (e.g. text
  // wrapping to that width); container nodes' measure is derived from
  // children by the engine and this field is omitted for containers.
  measure?: (constraint: FlowConstraint) => FlowSize;
  direction?: "row" | "column"; // containers only; default "row"
  gap?: number;
  justify?: "start" | "center" | "end" | "space-between";
  align?: "start" | "center" | "end" | "stretch";
  padding?: number;
  children?: readonly FlowNode[];
  // A fixed dimension always wins over measured/derived size on that axis.
  fixedWidth?: number;
  fixedHeight?: number;
};

/** A positioned, sized box for a node, relative to the root's origin. */
export type FlowBox = { x: number; y: number; width: number; height: number };

/**
 * Measure a node bottom-up. Leaves defer to their own `measure`; containers
 * derive their size from measured children along `direction`. `fixedWidth`/
 * `fixedHeight` always override the measured/derived value on that axis.
 */
function measureNode(node: FlowNode, constraint: FlowConstraint): FlowSize {
  if (node.measure) {
    const measured = node.measure(constraint);
    return {
      width: node.fixedWidth ?? measured.width,
      height: node.fixedHeight ?? measured.height,
    };
  }
  const direction = node.direction ?? "row";
  const gap = node.gap ?? 0;
  const padding = node.padding ?? 0;
  const children = node.children ?? [];
  const childConstraint: FlowConstraint = {
    maxWidth: Math.max(constraint.maxWidth - padding * 2, 0),
  };
  const sizes = children.map((child) => measureNode(child, childConstraint));
  const mainTotal =
    sizes.reduce(
      (sum, size) => sum + (direction === "row" ? size.width : size.height),
      0,
    ) + gap * Math.max(children.length - 1, 0);
  const crossMax = sizes.reduce(
    (max, size) => Math.max(max, direction === "row" ? size.height : size.width),
    0,
  );
  const width =
    node.fixedWidth ??
    (direction === "row" ? mainTotal + padding * 2 : crossMax + padding * 2);
  const height =
    node.fixedHeight ??
    (direction === "column" ? mainTotal + padding * 2 : crossMax + padding * 2);
  return { width, height };
}

/**
 * Position a node top-down. Given a resolved box, place children along the
 * main axis honoring `gap`/`justify`, align the cross axis with `align`, and
 * recurse into container children. Records every node's box in `out`.
 */
function positionNode(
  node: FlowNode,
  box: FlowBox,
  out: Map<string, FlowBox>,
): void {
  out.set(node.id, box);
  const children = node.children;
  if (children === undefined || children.length === 0) {
    return;
  }
  const direction = node.direction ?? "row";
  const gap = node.gap ?? 0;
  const padding = node.padding ?? 0;
  const justify = node.justify ?? "start";
  const align = node.align ?? "start";
  const contentBox: FlowBox = {
    x: box.x + padding,
    y: box.y + padding,
    width: box.width - padding * 2,
    height: box.height - padding * 2,
  };
  const childConstraint: FlowConstraint = { maxWidth: contentBox.width };
  const sizes = children.map((child) => measureNode(child, childConstraint));
  const mainTotal =
    sizes.reduce(
      (sum, size) => sum + (direction === "row" ? size.width : size.height),
      0,
    ) + gap * Math.max(children.length - 1, 0);
  const contentMain = direction === "row" ? contentBox.width : contentBox.height;
  const freeMain = Math.max(contentMain - mainTotal, 0);
  const extraGap =
    justify === "space-between" && children.length > 1
      ? freeMain / (children.length - 1)
      : 0;
  let cursor =
    (direction === "row" ? contentBox.x : contentBox.y) +
    (justify === "center" ? freeMain / 2 : justify === "end" ? freeMain : 0);
  children.forEach((child, index) => {
    const size = sizes[index]!;
    const mainSize = direction === "row" ? size.width : size.height;
    const crossSize = direction === "row" ? size.height : size.width;
    const contentCross =
      direction === "row" ? contentBox.height : contentBox.width;
    const crossOffset =
      align === "center"
        ? (contentCross - crossSize) / 2
        : align === "end"
          ? contentCross - crossSize
          : 0;
    const resolvedCrossSize = align === "stretch" ? contentCross : crossSize;
    const childBox: FlowBox =
      direction === "row"
        ? {
            x: cursor,
            y: contentBox.y + (align === "stretch" ? 0 : crossOffset),
            width: mainSize,
            height: resolvedCrossSize,
          }
        : {
            x: contentBox.x + (align === "stretch" ? 0 : crossOffset),
            y: cursor,
            width: resolvedCrossSize,
            height: mainSize,
          };
    positionNode(child, childBox, out);
    cursor += mainSize + gap + extraGap;
  });
}

/**
 * Two-pass layout: measures every node bottom-up (leaves via their own
 * `measure`, containers by summing/maxing children along their axis),
 * then positions every node top-down from the root's resolved box.
 * Returns every node's box in coordinates relative to the root's origin
 * (caller translates into absolute scene coordinates).
 */
export function layoutFlow(
  root: FlowNode,
  constraint: FlowConstraint,
): ReadonlyMap<string, FlowBox> {
  const rootSize = measureNode(root, constraint);
  const out = new Map<string, FlowBox>();
  positionNode(root, { x: 0, y: 0, ...rootSize }, out);
  return out;
}
