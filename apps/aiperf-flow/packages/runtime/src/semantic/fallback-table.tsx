// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";

import type {
  SemanticEntityProjection,
  SemanticProjection,
  SemanticRelationProjection,
} from "../evaluate/types.js";

export type SemanticFallbackTableProps = Readonly<{
  projection: SemanticProjection;
  caption?: string | undefined;
}>;

type SemanticRow =
  | Readonly<{ type: "entity"; value: SemanticEntityProjection }>
  | Readonly<{ type: "relation"; value: SemanticRelationProjection }>;

function orderedRows(projection: SemanticProjection): readonly SemanticRow[] {
  const entities = new Map(
    projection.entities.map((entity) => [entity.id, entity]),
  );
  const relations = new Map(
    projection.relations.map((relation) => [relation.id, relation]),
  );
  const seen = new Set<string>();
  const rows: SemanticRow[] = [];

  const append = (id: string): void => {
    if (seen.has(id)) {
      return;
    }
    const entity = entities.get(id);
    if (entity !== undefined) {
      seen.add(id);
      rows.push({ type: "entity", value: entity });
      return;
    }
    const relation = relations.get(id);
    if (relation !== undefined) {
      seen.add(id);
      rows.push({ type: "relation", value: relation });
    }
  };

  projection.readingOrder.forEach(append);
  projection.entities.forEach((entity) => append(entity.id));
  projection.relations.forEach((relation) => append(relation.id));
  return rows;
}

function sourceText(
  source: SemanticEntityProjection["source"],
): string | null {
  return source === undefined
    ? null
    : `${source.source}:${source.startOffset}-${source.endOffset}`;
}

/**
 * Accessible tabular alternative for a backend-neutral semantic projection.
 */
export function SemanticFallbackTable({
  projection,
  caption = `${projection.sceneId} semantic alternative`,
}: SemanticFallbackTableProps): ReactNode {
  const entityLabels = new Map(
    projection.entities.map((entity) => [entity.id, entity.label]),
  );

  return (
    <table
      className="aiperf-flow__semantic-fallback-table"
      data-scene-id={projection.sceneId}
    >
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th scope="col">Type</th>
          <th scope="col">Item</th>
          <th scope="col">Description</th>
          <th scope="col">Role</th>
          <th scope="col">Relationship</th>
          <th scope="col">Evidence</th>
          <th scope="col">Source</th>
        </tr>
      </thead>
      <tbody>
        {orderedRows(projection).map((row) => {
          if (row.type === "entity") {
            const value = row.value;
            const source = sourceText(value.source);
            return (
              <tr
                data-semantic-id={value.id}
                data-semantic-type="entity"
                key={`entity:${value.id}`}
              >
                <td>Entity</td>
                <th scope="row">{value.label}</th>
                <td>{value.description ?? "—"}</td>
                <td>{value.role ?? value.kind ?? "—"}</td>
                <td>—</td>
                <td>{value.evidenceIds?.join(", ") ?? "—"}</td>
                <td>{source ?? "—"}</td>
              </tr>
            );
          }

          const value = row.value;
          const source = sourceText(value.source);
          const from = entityLabels.get(value.fromId) ?? value.fromId;
          const to = entityLabels.get(value.toId) ?? value.toId;
          return (
            <tr
              data-from-id={value.fromId}
              data-semantic-id={value.id}
              data-semantic-type="relation"
              data-to-id={value.toId}
              key={`relation:${value.id}`}
            >
              <td>Relation</td>
              <th scope="row">{value.label ?? value.id}</th>
              <td>—</td>
              <td>{value.role ?? "—"}</td>
              <td>{`${from} → ${to}`}</td>
              <td>—</td>
              <td>{source ?? "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
