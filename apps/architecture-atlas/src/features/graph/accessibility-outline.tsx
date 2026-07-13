// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState, type KeyboardEvent } from "react";

import type { AudienceLevel, GraphEdge, GraphNode } from "../../domain/architecture";

export interface AccessibilityOutlineProps {
  audience: AudienceLevel;
  visibleNodes: readonly GraphNode[];
  visibleEdges: readonly GraphEdge[];
  expandedNodeIds: readonly string[];
  onSelectEntity(entityId: string): void;
  onExpandNode(nodeId: string): void;
  onCollapseNode(nodeId: string): void;
  onIsolateEntity(entityId: string): void;
  onInspectEntity(entityId: string): void;
  defaultCollapsed?: boolean;
}

function keyAction(
  event: KeyboardEvent<HTMLLIElement>,
  actions: {
    select(): void;
    expand?(): void;
    collapse?(): void;
    isolate(): void;
    inspect(): void;
  },
) {
  switch (event.key) {
    case "Enter":
    case " ":
      event.preventDefault();
      actions.select();
      break;
    case "ArrowRight":
      if (!actions.expand) {
        return;
      }
      event.preventDefault();
      actions.expand();
      break;
    case "ArrowLeft":
      if (!actions.collapse) {
        return;
      }
      event.preventDefault();
      actions.collapse();
      break;
    case "i":
    case "I":
      event.preventDefault();
      actions.isolate();
      break;
    case "x":
    case "X":
      event.preventDefault();
      actions.inspect();
      break;
    default:
      break;
  }
}

export function AccessibilityOutline({
  audience,
  visibleNodes,
  visibleEdges,
  expandedNodeIds,
  onSelectEntity,
  onExpandNode,
  onCollapseNode,
  onIsolateEntity,
  onInspectEntity,
  defaultCollapsed = true,
}: AccessibilityOutlineProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const nodeById = useMemo(
    () => new Map(visibleNodes.map((node) => [node.id, node])),
    [visibleNodes],
  );
  const outlineLevel = (node: GraphNode): number => {
    const visited = new Set([node.id]);
    let level = 1;
    let parentId = node.parentId;
    while (parentId) {
      const parent = nodeById.get(parentId);
      if (!parent || visited.has(parent.id)) {
        break;
      }
      visited.add(parent.id);
      level += 1;
      parentId = parent.parentId;
    }
    return level;
  };

  return (
    <section aria-label="Graph accessibility outline" className="graph-outline-overlay">
      <button
        aria-expanded={!collapsed}
        className="graph-outline-toggle"
        onClick={() => setCollapsed((current) => !current)}
        type="button"
      >
        {collapsed ? "Show graph accessibility outline" : "Hide graph accessibility outline"}
      </button>
      {!collapsed ? (
        <ul aria-label="Visible graph outline" className="graph-outline-tree" role="tree">
          {visibleNodes.map((node) => {
            const title = node.title[audience];
            const isExpanded = expandedNodeIds.includes(node.id);
            return (
              <li
                aria-expanded={node.childIds.length > 0 ? isExpanded : undefined}
                aria-label={`Node ${title}`}
                aria-level={outlineLevel(node)}
                key={node.id}
                onKeyDown={(event) =>
                  keyAction(event, {
                    select: () => onSelectEntity(node.id),
                    expand: () => onExpandNode(node.id),
                    collapse: () => onCollapseNode(node.id),
                    isolate: () => onIsolateEntity(node.id),
                    inspect: () => onInspectEntity(node.id),
                  })
                }
                role="treeitem"
                tabIndex={0}
              >
                <button
                  aria-label={`Select node ${title}`}
                  data-graph-entity-id={node.id}
                  data-graph-entity-trigger="true"
                  onClick={() => onSelectEntity(node.id)}
                  type="button"
                >
                  {title}
                </button>
                <span>{isExpanded ? "Expanded" : "Collapsed"}</span>
                <button onClick={() => onExpandNode(node.id)} type="button">
                  Expand
                </button>
                <button onClick={() => onCollapseNode(node.id)} type="button">
                  Collapse
                </button>
                <button onClick={() => onIsolateEntity(node.id)} type="button">
                  Isolate
                </button>
                <button onClick={() => onInspectEntity(node.id)} type="button">
                  Inspect
                </button>
              </li>
            );
          })}
          {visibleEdges.map((edge) => {
            const source = nodeById.get(edge.source.nodeId)?.title[audience] ?? edge.source.nodeId;
            const target = nodeById.get(edge.target.nodeId)?.title[audience] ?? edge.target.nodeId;
            const edgeLabel = `${source} -> ${target} via ${edge.protocol}`;
            return (
              <li
                aria-label={`Edge ${edgeLabel}`}
                aria-level={1}
                key={edge.id}
                onKeyDown={(event) =>
                  keyAction(event, {
                    select: () => onSelectEntity(edge.id),
                    isolate: () => onIsolateEntity(edge.id),
                    inspect: () => onInspectEntity(edge.id),
                  })
                }
                role="treeitem"
                tabIndex={0}
              >
                <button
                  aria-label={`Select edge ${edgeLabel}`}
                  data-graph-entity-id={edge.id}
                  data-graph-entity-trigger="true"
                  onClick={() => onSelectEntity(edge.id)}
                  type="button"
                >
                  {edgeLabel}
                </button>
                <button onClick={() => onIsolateEntity(edge.id)} type="button">
                  Isolate
                </button>
                <button onClick={() => onInspectEntity(edge.id)} type="button">
                  Inspect
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </section>
  );
}
