// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import { renderCanvasDisplayList } from "../../src/backends/canvas/canvas-renderer.js";
import { SvgFallback } from "../../src/backends/svg/svg-fallback.js";
import { SemanticTwin } from "../../src/semantic/semantic-twin.js";
import {
  CONFORMANCE_ENTITY,
  CONFORMANCE_SELECTION,
  displayListFixture,
  evaluatedSceneFixture,
} from "../fixtures/evaluated-scene.js";

type SemanticSnapshot = Readonly<{
  entityIds: readonly string[];
  labels: readonly string[];
  focusTargets: readonly string[];
  selectedEntityIds: readonly string[];
}>;

type CanvasCall = Readonly<{
  name: string;
  arguments: readonly unknown[];
}>;

function recordingCanvasContext(): Readonly<{
  context: CanvasRenderingContext2D;
  calls: readonly CanvasCall[];
}> {
  const calls: CanvasCall[] = [];
  const context = new Proxy(
    {},
    {
      get(_target, property) {
        if (property === "canvas") {
          return { width: 640, height: 360 };
        }
        if (property === "measureText") {
          return (text: string) => ({ width: text.length * 8 });
        }
        return (...arguments_: readonly unknown[]) => {
          calls.push({ name: String(property), arguments: arguments_ });
        };
      },
      set(_target, property, value) {
        calls.push({ name: `set:${String(property)}`, arguments: [value] });
        return true;
      },
    },
  ) as CanvasRenderingContext2D;

  return { context, calls };
}

function domSnapshot(container: HTMLElement): SemanticSnapshot {
  const entities = [
    ...container.querySelectorAll<HTMLElement>("[data-entity-id]"),
  ];
  return {
    entityIds: entities.map((entity) => entity.dataset.entityId ?? ""),
    labels: entities.map((entity) => entity.getAttribute("aria-label") ?? ""),
    focusTargets: entities
      .filter((entity) => entity.tabIndex === 0)
      .map((entity) => entity.dataset.entityId ?? ""),
    selectedEntityIds: entities
      .filter((entity) => entity.getAttribute("aria-selected") === "true")
      .map((entity) => entity.dataset.entityId ?? ""),
  };
}

afterEach(cleanup);

describe("backend semantic conformance", () => {
  test("preserves entity, label, focus, and selection semantics", () => {
    const recorder = recordingCanvasContext();
    const canvasOutput = renderCanvasDisplayList(
      displayListFixture,
      recorder.context,
      { devicePixelRatio: 1, quality: "reference" },
    );

    const semanticTwin = render(
      <SemanticTwin
        focusedEntityId={CONFORMANCE_SELECTION.focusedEntityId}
        onActivate={() => undefined}
        onFocus={() => undefined}
        projection={evaluatedSceneFixture.semantic}
        selectedEntityId={CONFORMANCE_ENTITY.id}
      />,
    );
    const svgFallback = render(
      <SvgFallback
        displayList={displayListFixture}
        focusedEntityId={CONFORMANCE_SELECTION.focusedEntityId}
        scene={evaluatedSceneFixture}
        selectedEntityIds={CONFORMANCE_SELECTION.selectedEntityIds}
      />,
    );

    const expected: SemanticSnapshot = {
      entityIds: [CONFORMANCE_ENTITY.id],
      labels: [CONFORMANCE_ENTITY.label],
      focusTargets: [CONFORMANCE_ENTITY.focusTarget],
      selectedEntityIds: [CONFORMANCE_ENTITY.id],
    };
    const canvasSnapshot: SemanticSnapshot = {
      entityIds: canvasOutput.hitRegions.map(({ entityId }) => entityId),
      labels: canvasOutput.hitRegions.map(({ label }) => label),
      focusTargets: canvasOutput.hitRegions
        .filter(({ focusable }) => focusable)
        .map(({ entityId }) => entityId),
      selectedEntityIds: canvasOutput.hitRegions
        .filter(({ selected }) => selected)
        .map(({ entityId }) => entityId),
    };

    expect(recorder.calls.length).toBeGreaterThan(0);
    expect(canvasSnapshot).toEqual(expected);
    expect(domSnapshot(semanticTwin.container)).toEqual(expected);
    expect(domSnapshot(svgFallback.container)).toEqual(expected);
  });
});
