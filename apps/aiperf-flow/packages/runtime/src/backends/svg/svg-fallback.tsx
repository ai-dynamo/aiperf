// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";

import type { Bounds, DisplayList, DrawCommand } from "../../display-list.js";
import type {
  EvaluatedScene,
  SemanticEntityProjection,
} from "../../evaluate/types.js";

export type SvgFallbackProps = Readonly<{
  scene: EvaluatedScene;
  displayList: DisplayList;
  selectedEntityIds?: readonly string[];
  focusedEntityId?: string | null;
  onFocusEntity?(entityId: string): void;
  onSelectEntity?(entityId: string): void;
}>;

function renderCommand(command: DrawCommand): ReactNode {
  switch (command.kind) {
    case "path":
      return (
        <path
          d={command.path}
          data-draw-command-id={command.id}
          fill={command.fill ?? "none"}
          key={command.id}
          stroke={command.stroke ?? "none"}
          strokeWidth={command.strokeWidth}
        />
      );
    case "text":
      return (
        <text
          data-draw-command-id={command.id}
          fill={command.fill ?? "currentColor"}
          fontFamily={command.font.family}
          fontSize={command.font.sizePx}
          fontWeight={command.font.weight}
          key={command.id}
          x={command.origin.x}
          y={command.origin.y}
        >
          {command.text}
        </text>
      );
    case "image":
      return (
        <image
          data-draw-command-id={command.id}
          height={command.destination.height}
          href={command.assetId}
          key={command.id}
          opacity={command.opacity}
          width={command.destination.width}
          x={command.destination.x}
          y={command.destination.y}
        />
      );
    case "clip": {
      const clipId = `flow-svg-clip-${command.id}`;
      return (
        <g
          clipPath={`url(#${clipId})`}
          data-draw-command-id={command.id}
          key={command.id}
        >
          <defs>
            <clipPath id={clipId}>
              <path d={command.path} />
            </clipPath>
          </defs>
          {command.children.map(renderCommand)}
        </g>
      );
    }
    case "layer":
      return (
        <g
          data-draw-command-id={command.id}
          key={command.id}
          opacity={command.opacity}
          style={{ mixBlendMode: command.blendMode }}
        >
          {command.children.map(renderCommand)}
        </g>
      );
    case "group":
      return (
        <g data-draw-command-id={command.id} key={command.id}>
          {command.children.map(renderCommand)}
        </g>
      );
  }
}

function sceneDescription(scene: EvaluatedScene): string {
  const entities = new Map(
    scene.semantic.entities.map((entity) => [entity.id, entity]),
  );
  return scene.semantic.readingOrder
    .map((id) => entities.get(id))
    .filter(
      (entity): entity is SemanticEntityProjection => entity !== undefined,
    )
    .map(({ label, description }) =>
      description === undefined ? label : `${label}. ${description}`,
    )
    .join(" ");
}

function semanticTarget(
  entity: SemanticEntityProjection,
  bounds: Bounds,
  selected: boolean,
  focused: boolean,
  onFocusEntity: SvgFallbackProps["onFocusEntity"],
  onSelectEntity: SvgFallbackProps["onSelectEntity"],
): ReactNode {
  return (
    <g
      aria-label={entity.label}
      aria-selected={selected ? "true" : "false"}
      data-entity-id={entity.id}
      data-focus-target={entity.id}
      data-focused={focused ? "true" : undefined}
      data-selected={selected ? "true" : undefined}
      id={`flow-svg-${entity.id}`}
      key={entity.id}
      onClick={() => onSelectEntity?.(entity.id)}
      onFocus={() => onFocusEntity?.(entity.id)}
      role="group"
      tabIndex={focused ? 0 : -1}
    >
      <desc>{entity.description ?? entity.label}</desc>
      <rect
        fill="transparent"
        height={bounds.height}
        pointerEvents="all"
        width={bounds.width}
        x={bounds.x}
        y={bounds.y}
      />
    </g>
  );
}

/** Projects evaluated contracts into simplified SVG without reading Flow IR. */
export function SvgFallback({
  scene,
  displayList,
  selectedEntityIds = [],
  focusedEntityId = null,
  onFocusEntity,
  onSelectEntity,
}: SvgFallbackProps): ReactNode {
  const entities = new Map(
    scene.semantic.entities.map((entity) => [entity.id, entity]),
  );
  const selected = new Set(selectedEntityIds);
  const description = sceneDescription(scene);
  const { paintBounds } = displayList;

  return (
    <svg
      aria-label={description || scene.semantic.sceneId}
      className="aiperf-flow__svg-fallback"
      role="img"
      viewBox={`${paintBounds.x} ${paintBounds.y} ${paintBounds.width} ${paintBounds.height}`}
    >
      <desc>{description || scene.semantic.sceneId}</desc>
      {displayList.commands.map(renderCommand)}
      {displayList.hitRegions.flatMap((region) => {
        const entity = entities.get(region.semanticId);
        return entity === undefined
          ? []
          : [
              semanticTarget(
                entity,
                region.bounds,
                selected.has(entity.id),
                focusedEntityId === entity.id,
                onFocusEntity,
                onSelectEntity,
              ),
            ];
      })}
      {displayList.commands.length === 0 ? (
        <text role="note" x={paintBounds.x} y={paintBounds.y + 16}>
          {description || scene.semantic.sceneId}
        </text>
      ) : null}
    </svg>
  );
}
