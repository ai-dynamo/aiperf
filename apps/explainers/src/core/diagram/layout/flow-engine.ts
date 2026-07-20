// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure, framework-agnostic flow-layout engine with full CSS-flexbox-equivalent
 * semantics: direction (incl. reverse), wrap (multi-line), justify-content,
 * align-items, align-content, gap, padding, per-child margin, per-child
 * grow/shrink/basis, and min/max size constraints.
 *
 * This module intentionally has no React/DOM/SVG dependency and is
 * deliberately independent of the existing `core/diagram/capabilities/layout.ts`
 * module.
 *
 * Correctness-by-construction: every node's intrinsic size (and, for wrapping
 * containers, its line grouping) is computed exactly ONCE, in {@link measure},
 * and cached by node id. {@link layoutFlow}'s position pass never re-invokes a
 * leaf's `measure` callback or recomputes a container's line grouping from
 * scratch — it only reads the cache and does arithmetic (grow/shrink
 * distribution, alignment) on the cached numbers. This is what makes the
 * engine immune to the "position pass measures a different width than the
 * measure pass did" divergence class of bug: there is only ever one
 * measurement per node, full stop.
 */

export type FlowConstraint = { maxWidth: number; maxHeight?: number };
export type FlowSize = { width: number; height: number };
export type FlowBox = { x: number; y: number; width: number; height: number };

export type FlowMarginBox = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};
export type FlowMargin = number | Partial<FlowMarginBox>;

export type FlowDirection = "row" | "column" | "row-reverse" | "column-reverse";
export type FlowJustify =
  | "start"
  | "center"
  | "end"
  | "space-between"
  | "space-around"
  | "space-evenly";
export type FlowAlign = "start" | "center" | "end" | "stretch";
export type FlowAlignContent = FlowJustify | "stretch";

export type FlowNode = {
  id: string;
  /** Leaves measure themselves against a width constraint (e.g. text wrap). */
  measure?: (constraint: FlowConstraint) => FlowSize;
  direction?: FlowDirection; // containers only; default "row"
  wrap?: "nowrap" | "wrap"; // default "nowrap"
  gap?: number; // shorthand for both rowGap and columnGap
  rowGap?: number;
  columnGap?: number;
  justify?: FlowJustify; // main-axis distribution within a line; default "start"
  align?: FlowAlign; // cross-axis alignment within a line; default "start"
  alignContent?: FlowAlignContent; // cross-axis distribution of lines (wrap only); default "start"
  padding?: number;
  children?: readonly FlowNode[];
  // A fixed dimension always wins over measured/derived/grown size on that axis.
  fixedWidth?: number;
  fixedHeight?: number;
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
  margin?: FlowMargin;
  // Flex distribution along the main axis, applied at position time only —
  // never triggers a re-measure (see module doc).
  grow?: number; // flex-grow; default 0
  shrink?: number; // flex-shrink; default 1
  basis?: number; // flex-basis on the main axis; overrides the measured main size before grow/shrink is applied
};

function resolveMargin(margin: FlowMargin | undefined): FlowMarginBox {
  if (margin === undefined) {
    return { top: 0, right: 0, bottom: 0, left: 0 };
  }
  if (typeof margin === "number") {
    return { top: margin, right: margin, bottom: margin, left: margin };
  }
  return {
    top: margin.top ?? 0,
    right: margin.right ?? 0,
    bottom: margin.bottom ?? 0,
    left: margin.left ?? 0,
  };
}

function clampSize(node: FlowNode, size: FlowSize): FlowSize {
  let width = size.width;
  let height = size.height;
  if (node.minWidth !== undefined) width = Math.max(width, node.minWidth);
  if (node.maxWidth !== undefined) width = Math.min(width, node.maxWidth);
  if (node.minHeight !== undefined) height = Math.max(height, node.minHeight);
  if (node.maxHeight !== undefined) height = Math.min(height, node.maxHeight);
  width = node.fixedWidth ?? width;
  height = node.fixedHeight ?? height;
  return { width, height };
}

function isRowLike(direction: FlowDirection): boolean {
  return direction === "row" || direction === "row-reverse";
}

function marginMain(margin: FlowMarginBox, rowLike: boolean): number {
  return rowLike ? margin.left + margin.right : margin.top + margin.bottom;
}

function marginCross(margin: FlowMarginBox, rowLike: boolean): number {
  return rowLike ? margin.top + margin.bottom : margin.left + margin.right;
}

/** One computed line of a (possibly wrapping) container, in measurement units. */
type FlowLine = {
  childIndexes: number[];
  mainTotal: number; // sum of child main-sizes + their main-margins + inter-child gap
  crossSize: number; // max of child cross-sizes + their cross-margins
};

type MeasuredNode = {
  size: FlowSize;
  childSizes?: FlowSize[]; // one per child, in child order, cache-only (never re-measured)
  childMargins?: FlowMarginBox[];
  lines?: FlowLine[]; // container only, when applicable
};

/** Per-`layoutFlow`-call cache: every node's measurement is computed once. */
class FlowCache {
  private readonly nodes = new Map<string, MeasuredNode>();

  get(id: string): MeasuredNode | undefined {
    return this.nodes.get(id);
  }

  set(id: string, value: MeasuredNode): void {
    this.nodes.set(id, value);
  }
}

/**
 * Greedily group `mains` (each child's main-axis size including its own
 * main-axis margin) into lines whose running total (plus inter-child gap)
 * does not exceed `mainBudget`. A single child wider than the budget still
 * gets its own line (never dropped, never split). When `wrap` is `"nowrap"`
 * every child is packed into one line regardless of budget (matching
 * flexbox's default, where overflow is allowed rather than silently
 * wrapping).
 */
function groupIntoLines(
  mains: readonly number[],
  crosses: readonly number[],
  gap: number,
  mainBudget: number,
  wrap: "nowrap" | "wrap",
): FlowLine[] {
  if (mains.length === 0) {
    return [];
  }
  if (wrap === "nowrap") {
    const mainTotal =
      mains.reduce((sum, value) => sum + value, 0) + gap * Math.max(mains.length - 1, 0);
    const crossSize = crosses.reduce((max, value) => Math.max(max, value), 0);
    return [{ childIndexes: mains.map((_, index) => index), mainTotal, crossSize }];
  }
  const lines: FlowLine[] = [];
  let currentIndexes: number[] = [];
  let currentMain = 0;
  mains.forEach((main, index) => {
    const withChild = currentIndexes.length === 0 ? main : currentMain + gap + main;
    if (currentIndexes.length > 0 && withChild > mainBudget) {
      lines.push({
        childIndexes: currentIndexes,
        mainTotal: currentMain,
        crossSize: currentIndexes.reduce((max, i) => Math.max(max, crosses[i]!), 0),
      });
      currentIndexes = [index];
      currentMain = main;
    } else {
      currentIndexes.push(index);
      currentMain = withChild;
    }
  });
  if (currentIndexes.length > 0) {
    lines.push({
      childIndexes: currentIndexes,
      mainTotal: currentMain,
      crossSize: currentIndexes.reduce((max, i) => Math.max(max, crosses[i]!), 0),
    });
  }
  return lines;
}

/**
 * Measure `node` against `constraint`, caching the result (and, for
 * containers, every child's size/margin and computed line grouping) in
 * `cache` so the position pass never needs to re-measure anything.
 */
function measure(node: FlowNode, constraint: FlowConstraint, cache: FlowCache): FlowSize {
  const cached = cache.get(node.id);
  if (cached !== undefined) {
    return cached.size;
  }

  if (node.measure) {
    const raw = node.measure(constraint);
    const size = clampSize(node, raw);
    cache.set(node.id, { size });
    return size;
  }

  const direction = node.direction ?? "row";
  const rowLike = isRowLike(direction);
  const wrap = node.wrap ?? "nowrap";
  const padding = node.padding ?? 0;
  const gapBoth = node.gap ?? 0;
  const rowGap = node.rowGap ?? gapBoth;
  const columnGap = node.columnGap ?? gapBoth;
  const mainGap = rowLike ? columnGap : rowGap;
  const crossGap = rowLike ? rowGap : columnGap;
  const children = node.children ?? [];

  // Children measure against THIS node's own effective size, not the raw
  // incoming constraint — if the author fixed/capped this container's size,
  // that authored intent (not an ancestor's unrelated constraint) is what
  // children actually have to fit into.
  const effectiveMaxWidth = node.fixedWidth ?? node.maxWidth ?? constraint.maxWidth;
  const effectiveMaxHeight = node.fixedHeight ?? node.maxHeight ?? constraint.maxHeight;
  const childConstraint: FlowConstraint = {
    maxWidth: Math.max(effectiveMaxWidth - padding * 2, 0),
    maxHeight:
      effectiveMaxHeight !== undefined
        ? Math.max(effectiveMaxHeight - padding * 2, 0)
        : undefined,
  };

  const childMargins = children.map((child) => resolveMargin(child.margin));
  const childSizes = children.map((child) => {
    const rawSize = measure(child, childConstraint, cache);
    // basis overrides the measured main size before grow/shrink (position-time only)
    if (child.basis === undefined) {
      return rawSize;
    }
    return rowLike
      ? { width: child.basis, height: rawSize.height }
      : { width: rawSize.width, height: child.basis };
  });

  const mains = childSizes.map(
    (size, i) => (rowLike ? size.width : size.height) + marginMain(childMargins[i]!, rowLike),
  );
  const crosses = childSizes.map(
    (size, i) => (rowLike ? size.height : size.width) + marginCross(childMargins[i]!, rowLike),
  );
  const mainBudget = rowLike ? childConstraint.maxWidth : (childConstraint.maxHeight ?? Infinity);
  const lines = groupIntoLines(mains, crosses, mainGap, mainBudget, wrap);

  const contentMain = lines.reduce((max, line) => Math.max(max, line.mainTotal), 0);
  const contentCross =
    lines.reduce((sum, line) => sum + line.crossSize, 0) + crossGap * Math.max(lines.length - 1, 0);

  const width = rowLike ? contentMain + padding * 2 : contentCross + padding * 2;
  const height = rowLike ? contentCross + padding * 2 : contentMain + padding * 2;
  const size = clampSize(node, { width, height });

  cache.set(node.id, { size, childSizes, childMargins, lines });
  return size;
}

/** Distributes `freeSpace` across a line's children per `justify`. */
function distributeJustify(
  justify: FlowJustify,
  freeSpace: number,
  count: number,
): { leading: number; between: number } {
  if (count <= 1) {
    const leading = justify === "center" ? freeSpace / 2 : justify === "end" ? freeSpace : 0;
    return { leading, between: 0 };
  }
  switch (justify) {
    case "center":
      return { leading: freeSpace / 2, between: 0 };
    case "end":
      return { leading: freeSpace, between: 0 };
    case "space-between":
      return { leading: 0, between: freeSpace / (count - 1) };
    case "space-around": {
      const each = freeSpace / count;
      return { leading: each / 2, between: each };
    }
    case "space-evenly": {
      const each = freeSpace / (count + 1);
      return { leading: each, between: each };
    }
    default:
      return { leading: 0, between: 0 };
  }
}

/** Distributes `freeCross` across lines per `alignContent` (wrap only). */
function distributeAlignContent(
  alignContent: FlowAlignContent,
  freeCross: number,
  lineCount: number,
): { leading: number; between: number; stretchEach: number } {
  if (alignContent === "stretch") {
    return { leading: 0, between: 0, stretchEach: freeCross / Math.max(lineCount, 1) };
  }
  if (lineCount <= 1) {
    const leading = alignContent === "center" ? freeCross / 2 : alignContent === "end" ? freeCross : 0;
    return { leading, between: 0, stretchEach: 0 };
  }
  switch (alignContent) {
    case "center":
      return { leading: freeCross / 2, between: 0, stretchEach: 0 };
    case "end":
      return { leading: freeCross, between: 0, stretchEach: 0 };
    case "space-between":
      return { leading: 0, between: freeCross / (lineCount - 1), stretchEach: 0 };
    case "space-around": {
      const each = freeCross / lineCount;
      return { leading: each / 2, between: each, stretchEach: 0 };
    }
    case "space-evenly": {
      const each = freeCross / (lineCount + 1);
      return { leading: each, between: each, stretchEach: 0 };
    }
    default:
      return { leading: 0, between: 0, stretchEach: 0 };
  }
}

function positionContainer(
  node: FlowNode,
  box: FlowBox,
  cache: FlowCache,
  out: Map<string, FlowBox>,
): void {
  const measured = cache.get(node.id);
  const children = node.children ?? [];
  if (measured === undefined || measured.lines === undefined || children.length === 0) {
    return;
  }
  const direction = node.direction ?? "row";
  const rowLike = isRowLike(direction);
  const reverse = direction === "row-reverse" || direction === "column-reverse";
  const wrap = node.wrap ?? "nowrap";
  const padding = node.padding ?? 0;
  const gapBoth = node.gap ?? 0;
  const rowGap = node.rowGap ?? gapBoth;
  const columnGap = node.columnGap ?? gapBoth;
  const mainGap = rowLike ? columnGap : rowGap;
  const crossGap = rowLike ? rowGap : columnGap;
  const justify = node.justify ?? "start";
  const align = node.align ?? "start";
  const alignContent = node.alignContent ?? "start";

  const childSizes = measured.childSizes!;
  const childMargins = measured.childMargins!;
  const lines = measured.lines;

  const contentBox: FlowBox = {
    x: box.x + padding,
    y: box.y + padding,
    width: box.width - padding * 2,
    height: box.height - padding * 2,
  };
  const contentMainTotal = rowLike ? contentBox.width : contentBox.height;
  const contentCrossTotal = rowLike ? contentBox.height : contentBox.width;

  const linesCrossTotal =
    lines.reduce((sum, line) => sum + line.crossSize, 0) + crossGap * Math.max(lines.length - 1, 0);
  const freeCross = Math.max(contentCrossTotal - linesCrossTotal, 0);
  const contentDist =
    wrap === "wrap"
      ? distributeAlignContent(alignContent, freeCross, lines.length)
      : { leading: 0, between: 0, stretchEach: 0 };

  let crossCursor = (rowLike ? contentBox.y : contentBox.x) + contentDist.leading;

  // `row-reverse`/`column-reverse` only reverses the MAIN axis (child order
  // within a line, handled below via `orderedIndexes`) — line stacking order
  // along the cross axis is unaffected by direction; only `wrap-reverse`
  // (not implemented — no current composite needs it) would reverse that.
  lines.forEach((line) => {
    // A single implied line (the nowrap default, by far the common case)
    // always occupies the FULL cross-axis content space — align-content
    // only has meaning across multiple lines. With more than one line, a
    // line's cross size is its own content unless alignContent stretches it.
    const lineCrossSize =
      lines.length <= 1
        ? contentCrossTotal
        : alignContent === "stretch"
          ? line.crossSize + contentDist.stretchEach
          : line.crossSize;

    // Per-child main size = measured main + margin; flex-grow/shrink adjusts
    // this arithmetically against the line's own free space, never
    // re-measuring anything (see module doc: the cache is the sole source
    // of truth for every child's size — this loop only redistributes it).
    const orderedIndexes = reverse ? [...line.childIndexes].reverse() : line.childIndexes;
    const childMains = orderedIndexes.map((i) => {
      const size = childSizes[i]!;
      const margin = childMargins[i]!;
      return (rowLike ? size.width : size.height) + marginMain(margin, rowLike);
    });
    const baseMainTotal =
      childMains.reduce((sum, value) => sum + value, 0) + mainGap * Math.max(childMains.length - 1, 0);
    const freeMain = contentMainTotal - baseMainTotal;

    let adjustedMains = childMains;
    if (freeMain > 0) {
      const totalGrow = orderedIndexes.reduce((sum, i) => sum + (children[i]!.grow ?? 0), 0);
      if (totalGrow > 0) {
        adjustedMains = childMains.map((main, idx) => {
          const grow = children[orderedIndexes[idx]!]!.grow ?? 0;
          return main + (freeMain * grow) / totalGrow;
        });
      }
    } else if (freeMain < 0) {
      const totalShrinkWeight = orderedIndexes.reduce((sum, i, idx) => {
        const shrink = children[i]!.shrink ?? 1;
        return sum + shrink * childMains[idx]!;
      }, 0);
      if (totalShrinkWeight > 0) {
        adjustedMains = childMains.map((main, idx) => {
          const shrink = children[orderedIndexes[idx]!]!.shrink ?? 1;
          const weight = (shrink * main) / totalShrinkWeight;
          return Math.max(main + freeMain * weight, 0);
        });
      }
    }

    const remainingFree = Math.max(
      contentMainTotal -
        (adjustedMains.reduce((sum, value) => sum + value, 0) +
          mainGap * Math.max(adjustedMains.length - 1, 0)),
      0,
    );
    const justifyDist = distributeJustify(justify, remainingFree, orderedIndexes.length);

    let mainCursor = (rowLike ? contentBox.x : contentBox.y) + justifyDist.leading;

    orderedIndexes.forEach((childIndex, idx) => {
      const child = children[childIndex]!;
      const size = childSizes[childIndex]!;
      const margin = childMargins[childIndex]!;
      const mainSizeWithMargin = adjustedMains[idx]!;
      const marginMainStart = rowLike ? margin.left : margin.top;
      const marginMainEnd = rowLike ? margin.right : margin.bottom;
      const mainSize = Math.max(mainSizeWithMargin - marginMainStart - marginMainEnd, 0);

      const crossSize = rowLike ? size.height : size.width;
      const marginCrossStart = rowLike ? margin.top : margin.left;
      const marginCrossEnd = rowLike ? margin.bottom : margin.right;
      const resolvedCrossSize =
        align === "stretch"
          ? Math.max(lineCrossSize - marginCrossStart - marginCrossEnd, 0)
          : crossSize;
      const crossOffset =
        align === "center"
          ? Math.max((lineCrossSize - marginCrossStart - marginCrossEnd - resolvedCrossSize) / 2, 0)
          : align === "end"
            ? Math.max(lineCrossSize - marginCrossStart - marginCrossEnd - resolvedCrossSize, 0)
            : 0;

      const childBox: FlowBox = rowLike
        ? {
            x: mainCursor + marginMainStart,
            y: crossCursor + marginCrossStart + crossOffset,
            width: mainSize,
            height: resolvedCrossSize,
          }
        : {
            x: crossCursor + marginCrossStart + crossOffset,
            y: mainCursor + marginMainStart,
            width: resolvedCrossSize,
            height: mainSize,
          };

      out.set(child.id, childBox);
      if (child.children !== undefined && child.children.length > 0) {
        positionContainer(child, childBox, cache, out);
      }

      mainCursor += mainSizeWithMargin + mainGap + justifyDist.between;
    });

    crossCursor += lineCrossSize + crossGap + contentDist.between;
  });
}

/**
 * Layout: {@link measure} sizes every node bottom-up exactly once (leaves via
 * their own `measure`, containers by grouping children into lines and
 * summing/maxing along `direction`), then the position pass places every node
 * top-down from the root's resolved box, reading exclusively from the
 * measurement cache (no re-measurement, ever — see module doc).
 *
 * Returns every node's box in coordinates relative to the root's origin
 * (caller translates into absolute scene coordinates).
 */
export function layoutFlow(
  root: FlowNode,
  constraint: FlowConstraint,
): ReadonlyMap<string, FlowBox> {
  const cache = new FlowCache();
  const rootSize = measure(root, constraint, cache);
  const out = new Map<string, FlowBox>();
  const rootBox: FlowBox = { x: 0, y: 0, ...rootSize };
  out.set(root.id, rootBox);
  if (root.children !== undefined && root.children.length > 0) {
    positionContainer(root, rootBox, cache, out);
  }
  return out;
}
