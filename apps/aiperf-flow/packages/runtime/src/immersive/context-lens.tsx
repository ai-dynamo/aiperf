// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type KeyboardEvent,
  type ReactNode,
  useId,
} from "react";

import type {
  SemanticEntityProjection,
  SemanticProjection,
  SemanticRelationProjection,
} from "../evaluate/types.js";

/** Edge-attached inspector for one selected semantic entity. */
export type ContextLensProps = Readonly<{
  projection: SemanticProjection;
  entityId: string;
  onClose(): void;
  onFocusWorld(entityId: string): void;
  onOpenTwin(entityId: string): void;
}>;

type RelatedEndpoint = Readonly<{
  relation: SemanticRelationProjection;
  relatedId: string;
  relatedLabel: string;
  direction: "from" | "to";
}>;

function entityById(
  projection: SemanticProjection,
  id: string,
): SemanticEntityProjection | undefined {
  return projection.entities.find((entity) => entity.id === id);
}

function entityLabel(
  projection: SemanticProjection,
  id: string,
): string {
  return entityById(projection, id)?.label ?? id;
}

function relatedEndpoints(
  projection: SemanticProjection,
  entityId: string,
): readonly RelatedEndpoint[] {
  const related: RelatedEndpoint[] = [];
  for (const relation of projection.relations) {
    if (relation.fromId === entityId) {
      related.push({
        relation,
        relatedId: relation.toId,
        relatedLabel: entityLabel(projection, relation.toId),
        direction: "to",
      });
      continue;
    }
    if (relation.toId === entityId) {
      related.push({
        relation,
        relatedId: relation.fromId,
        relatedLabel: entityLabel(projection, relation.fromId),
        direction: "from",
      });
    }
  }
  return related;
}

function evidenceIds(
  entity: SemanticEntityProjection,
): readonly string[] | undefined {
  const ids = entity.evidenceIds;
  if (ids === undefined || ids.length === 0) {
    return undefined;
  }
  return ids;
}

function sourceText(
  source: SemanticEntityProjection["source"],
): string | null {
  return source === undefined
    ? null
    : `${source.source}:${source.startOffset}-${source.endOffset}`;
}

/**
 * Context Lens: projects relations and evidence for one semantic entity ID
 * without reinterpreting Flow IR. Optional evidence is never invented; when
 * absent the lens reports that none is attached.
 */
export function ContextLens({
  projection,
  entityId,
  onClose,
  onFocusWorld,
  onOpenTwin,
}: ContextLensProps): ReactNode {
  const titleId = useId();
  const entity = entityById(projection, entityId);
  if (entity === undefined) {
    return null;
  }

  const roleOrKind = entity.role ?? entity.kind;
  const related = relatedEndpoints(projection, entity.id);
  const evidence = evidenceIds(entity);
  const source = sourceText(entity.source);

  function onKeyDown(event: KeyboardEvent<HTMLElement>): void {
    if (event.key !== "Escape") {
      return;
    }
    event.preventDefault();
    onClose();
  }

  return (
    <aside
      aria-labelledby={titleId}
      className="aiperf-flow__context-lens"
      data-entity-id={entity.id}
      data-scene-id={projection.sceneId}
      onKeyDown={onKeyDown}
      role="region"
      tabIndex={-1}
    >
      <h2 className="aiperf-flow__context-lens-title" id={titleId}>
        Context Lens
      </h2>

      <strong>{entity.label}</strong>
      {roleOrKind === undefined ? null : (
        <p className="aiperf-flow__context-lens-role" data-role={roleOrKind}>
          {roleOrKind}
        </p>
      )}
      {entity.description === undefined ? null : (
        <p className="aiperf-flow__context-lens-description">
          {entity.description}
        </p>
      )}

      <section
        aria-label="Relations"
        className="aiperf-flow__context-lens-relations"
      >
        {related.length === 0 ? (
          <p>No related entities</p>
        ) : (
          <ul>
            {related.map(({ relation, relatedId, relatedLabel, direction }) => (
              <li
                data-direction={direction}
                data-from={relation.fromId}
                data-relation-id={relation.id}
                data-related-id={relatedId}
                data-to={relation.toId}
                key={relation.id}
              >
                {relation.label ??
                  (direction === "to"
                    ? `${entity.label} → ${relatedLabel}`
                    : `${relatedLabel} → ${entity.label}`)}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section
        aria-label="Evidence"
        className="aiperf-flow__context-lens-evidence"
      >
        {evidence === undefined ? (
          <p>No evidence is attached</p>
        ) : (
          <ul>
            {evidence.map((id) => (
              <li data-evidence-id={id} key={id}>
                {id}
              </li>
            ))}
          </ul>
        )}
      </section>

      {source === null ? null : (
        <p
          className="aiperf-flow__context-lens-source"
          data-source={entity.source?.source}
        >
          {source}
        </p>
      )}

      <div className="aiperf-flow__context-lens-actions">
        <button
          onClick={() => onFocusWorld(entity.id)}
          type="button"
        >
          Focus World
        </button>
        <button
          onClick={() => onOpenTwin(entity.id)}
          type="button"
        >
          Open semantic twin
        </button>
        <button onClick={onClose} type="button">
          Close
        </button>
      </div>
    </aside>
  );
}
