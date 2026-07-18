// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";

import type {
  SemanticEntityProjection,
  SemanticProjection,
} from "../evaluate/types.js";

export type SemanticTwinProps = Readonly<{
  projection: SemanticProjection;
  focusedEntityId: string | null;
  selectedEntityId: string | null;
  onFocus(entityId: string): void;
  onActivate(entityId: string): void;
  compact?: boolean | undefined;
}>;

function entityById(
  projection: SemanticProjection,
  id: string,
): SemanticEntityProjection | undefined {
  return projection.entities.find((entity) => entity.id === id);
}

function orderedEntities(
  projection: SemanticProjection,
): readonly SemanticEntityProjection[] {
  const seen = new Set<string>();
  const ordered: SemanticEntityProjection[] = [];

  for (const id of projection.readingOrder) {
    const entity = entityById(projection, id);
    if (entity === undefined || seen.has(entity.id)) {
      continue;
    }
    seen.add(entity.id);
    ordered.push(entity);
  }

  for (const entity of projection.entities) {
    if (seen.has(entity.id)) {
      continue;
    }
    seen.add(entity.id);
    ordered.push(entity);
  }

  return ordered;
}

/**
 * Always-mounted semantic HTML twin. Renders landmarks, reading-order entities,
 * relations, and transcript linkage without interpreting Flow IR.
 */
export function SemanticTwin({
  projection,
  focusedEntityId,
  selectedEntityId,
  onFocus,
  onActivate,
  compact = false,
}: SemanticTwinProps): ReactNode {
  const entities = orderedEntities(projection);

  return (
    <section
      aria-label="Semantic outline"
      className={
        compact
          ? "aiperf-flow__semantic-twin aiperf-flow__semantic-twin--compact"
          : "aiperf-flow__semantic-twin"
      }
      data-compact={compact ? "true" : "false"}
      data-scene-id={projection.sceneId}
    >
      <ol aria-label="Entities" className="aiperf-flow__semantic-entities">
        {entities.map((entity) => {
          const selected = selectedEntityId === entity.id;
          const focused = focusedEntityId === entity.id;
          return (
            <li key={entity.id}>
              <button
                aria-current={selected ? "true" : undefined}
                aria-describedby={
                  entity.description === undefined
                    ? undefined
                    : `flow-semantic-desc-${entity.id}`
                }
                aria-label={entity.label}
                aria-selected={selected ? "true" : "false"}
                data-entity-id={entity.id}
                data-evidence-ids={entity.evidenceIds?.join(" ") ?? undefined}
                data-focused={focused ? "true" : "false"}
                data-kind={entity.kind ?? entity.role}
                data-selected={selected ? "true" : "false"}
                onClick={() => onActivate(entity.id)}
                onFocus={() => onFocus(entity.id)}
                tabIndex={focused ? 0 : -1}
                type="button"
              >
                {entity.label}
              </button>
              {entity.description === undefined ? null : (
                <p id={`flow-semantic-desc-${entity.id}`}>{entity.description}</p>
              )}
            </li>
          );
        })}
      </ol>

      <ul aria-label="Relations" className="aiperf-flow__semantic-relations">
        {projection.relations.map((relation) => (
          <li
            data-from={relation.fromId}
            data-kind={relation.role}
            data-relation-id={relation.id}
            data-to={relation.toId}
            key={relation.id}
          >
            {relation.label ?? `${relation.fromId} → ${relation.toId}`}
          </li>
        ))}
      </ul>

      {(projection.captions ?? []).length > 0 ? (
        <div
          aria-live="polite"
          data-transcript-cue={projection.transcriptCueId}
          role="status"
        >
          {(projection.captions ?? []).map((caption) => (
            <p key={caption}>{caption}</p>
          ))}
        </div>
      ) : null}
    </section>
  );
}
