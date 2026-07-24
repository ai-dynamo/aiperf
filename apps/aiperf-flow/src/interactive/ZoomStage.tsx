/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Domain-agnostic semantic-zoom container over a `ZoomTree`. Manages a drill *path* (parametric
//! depth — not hard-coded to any number of levels), an active node id, a breadcrumb trail, and a
//! `motion` `layout`/`layoutId` shared-element expand/collapse. Backdrop-click and `Esc` pop a
//! level; Left/Right arrows move between siblings at the current level; `prefers-reduced-motion`
//! degrades the transition to instant. Content is supplied by the caller via a render-prop
//! (`children`), so the container itself knows nothing about what a level draws.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LayoutGroup, motion, useReducedMotion } from "motion/react";
import type { ZoomTree, ZoomTreeNode } from "./types.js";
import { Row } from "../layout/Row.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../theme/tokens.js";

/** Context handed to the `ZoomStage` render-prop for the currently active level. */
export interface ZoomStageContext<T = unknown> {
  /** Ids from the root to the active node (length − 1 is the depth). */
  path: readonly string[];
  /** Current depth: 0 at the root/overview, growing by one per drill. */
  level: number;
  /** Id of the deepest (active) node in `path`. */
  activeId: string;
  /** The active node's `ZoomTree` entry. */
  node: ZoomTreeNode<T>;
  /** Breadcrumb trail: one `{ id, label }` per path entry, root first. */
  breadcrumb: ReadonlyArray<{ id: string; label: string }>;
  /** Sibling ids of the active node (children of its parent; `[activeId]` at the root). */
  siblings: readonly string[];
  /** Push a child id onto the path. No-op unless `childId` is one of `node.children`. */
  drill: (childId: string) => void;
  /** Pop one level. No-op at the root. */
  pop: () => void;
  /** Replace the active node with a sibling (the arrow-key nav target). No-op if not a sibling. */
  goToSibling: (siblingId: string) => void;
  /** Whether the user prefers reduced motion (content may skip its own animations too). */
  reducedMotion: boolean;
}

export interface ZoomStageProps<T = unknown> {
  /** The zoom tree to navigate. */
  tree: ZoomTree<T>;
  /** Id of the node shown at depth 0. */
  rootId: string;
  /** Renders the content for the active level. */
  children: (ctx: ZoomStageContext<T>) => React.ReactNode;
  /** Notified whenever the active path changes (e.g. to sync a URL or a play head). */
  onNavigate?: (path: readonly string[]) => void;
  className?: string;
}

/** Siblings of the deepest node = children of its parent, or `[activeId]` when it is the root. */
function siblingsOf<T>(tree: ZoomTree<T>, path: readonly string[]): string[] {
  const activeId = path[path.length - 1]!;
  if (path.length < 2) {
    return [activeId];
  }
  const parent = tree[path[path.length - 2]!];
  const children = parent?.children ?? [];
  return children.length > 0 ? children : [activeId];
}

/**
 * Semantic-zoom stage. Renders a breadcrumb bar, an optional pop-a-level backdrop (below depth 0),
 * and the caller's render-prop content inside a `motion` shared-element wrapper keyed by the
 * active node id.
 */
export function ZoomStage<T = unknown>({
  tree,
  rootId,
  children,
  onNavigate,
  className,
}: ZoomStageProps<T>): React.JSX.Element {
  const [path, setPath] = useState<string[]>(() => [rootId]);
  const prefersReduced = useReducedMotion() ?? false;

  // Keep the latest path in a ref so the window key listener stays a stable, single subscription
  // yet always acts on current state (no stale closure, no re-subscribe churn per navigation).
  const pathRef = useRef(path);
  pathRef.current = path;

  const setPathAnd = useCallback(
    (nextPath: string[]) => {
      setPath(nextPath);
      onNavigate?.(nextPath);
    },
    [onNavigate],
  );

  const drill = useCallback(
    (childId: string) => {
      const current = pathRef.current;
      const active = tree[current[current.length - 1]!];
      if (active?.children?.includes(childId) && tree[childId]) {
        setPathAnd([...current, childId]);
      }
    },
    [tree, setPathAnd],
  );

  const pop = useCallback(() => {
    const current = pathRef.current;
    if (current.length > 1) {
      setPathAnd(current.slice(0, -1));
    }
  }, [setPathAnd]);

  const goToDepth = useCallback(
    (depth: number) => {
      const current = pathRef.current;
      if (depth >= 0 && depth < current.length - 1) {
        setPathAnd(current.slice(0, depth + 1));
      }
    },
    [setPathAnd],
  );

  const goToSibling = useCallback(
    (siblingId: string) => {
      const current = pathRef.current;
      const sibs = siblingsOf(tree, current);
      if (sibs.includes(siblingId) && tree[siblingId]) {
        setPathAnd([...current.slice(0, -1), siblingId]);
      }
    },
    [tree, setPathAnd],
  );

  // Esc pops a level; Left/Right cycle among siblings at the current level.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const current = pathRef.current;
      if (event.key === "Escape") {
        if (current.length > 1) {
          event.preventDefault();
          setPathAnd(current.slice(0, -1));
        }
        return;
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        const sibs = siblingsOf(tree, current);
        if (sibs.length < 2) {
          return;
        }
        const activeId = current[current.length - 1]!;
        const idx = sibs.indexOf(activeId);
        const nextIdx =
          event.key === "ArrowRight"
            ? (idx + 1) % sibs.length
            : (idx - 1 + sibs.length) % sibs.length;
        event.preventDefault();
        setPathAnd([...current.slice(0, -1), sibs[nextIdx]!]);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [tree, setPathAnd]);

  const activeId = path[path.length - 1]!;
  const node = tree[activeId];
  const level = path.length - 1;

  const breadcrumb = useMemo(
    () => path.map((id) => ({ id, label: tree[id]?.label ?? id })),
    [path, tree],
  );
  const siblings = useMemo(() => siblingsOf(tree, path), [tree, path]);

  if (!node) {
    // Defensive: an unknown active id can only happen if a caller mutates the tree out from under
    // an open path. Surface it honestly rather than crashing the whole deck.
    return (
      <div className={className}>
        <p className={`text-sm ${inkClassName("tertiary")}`}>Unknown zoom node: {activeId}</p>
      </div>
    );
  }

  const ctx: ZoomStageContext<T> = {
    path,
    level,
    activeId,
    node,
    breadcrumb,
    siblings,
    drill,
    pop,
    goToSibling,
    reducedMotion: prefersReduced,
  };

  return (
    <div className={className}>
      <nav aria-label="Zoom breadcrumb" className="mb-3">
        <Row gap={6} align="center" wrap>
          {breadcrumb.map((crumb, index) => {
            const isLast = index === breadcrumb.length - 1;
            return (
              <Row key={crumb.id} gap={6} align="center">
                {index > 0 && <span className={`text-xs ${inkClassName("quaternary")}`}>/</span>}
                <button
                  type="button"
                  disabled={isLast}
                  onClick={() => goToDepth(index)}
                  aria-current={isLast ? "page" : undefined}
                  className={
                    isLast
                      ? `rounded-md border px-2 py-0.5 text-xs font-semibold shadow-sm ${surfaceClassName("elevated")} ${strokeClassName("primary")} ${inkClassName("primary")}`
                      : `rounded-md px-2 py-0.5 text-xs font-medium transition-colors hover:text-accent-primary ${inkClassName("secondary")}`
                  }
                >
                  {crumb.label}
                </button>
              </Row>
            );
          })}
        </Row>
      </nav>

      <div className="relative">
        {level > 0 && (
          <motion.button
            type="button"
            aria-label="Back one level"
            onClick={pop}
            className="absolute inset-0 z-0 cursor-zoom-out bg-transparent"
            initial={prefersReduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: prefersReduced ? 0 : 0.15 }}
          />
        )}
        <LayoutGroup>
          <motion.div
            key={activeId}
            layout={!prefersReduced}
            layoutId={`zoomstage-${activeId}`}
            className="relative z-10"
            transition={{ duration: prefersReduced ? 0 : 0.28, ease: "easeInOut" }}
          >
            {children(ctx)}
          </motion.div>
        </LayoutGroup>
      </div>
    </div>
  );
}
