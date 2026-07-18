// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import type { SemanticProjection } from "../../src/evaluate/types.js";
import { SemanticTwin } from "../../src/semantic/semantic-twin.js";

afterEach(cleanup);

const projection: SemanticProjection = {
  sceneId: "lifecycle",
  readingOrder: ["observe", "arrive", "admit"],
  entities: [
    {
      id: "arrive",
      label: "Arrive",
      description: "Request enters the runtime",
      kind: "phase",
    },
    {
      id: "admit",
      label: "Admit",
      description: "Worker admits the request",
      kind: "phase",
    },
    {
      id: "observe",
      label: "Observe",
      description: "Observer records metrics",
      kind: "phase",
      evidenceIds: ["ev-1"],
    },
  ],
  relations: [
    {
      id: "r0",
      fromId: "arrive",
      toId: "admit",
      role: "next",
      label: "then admit",
    },
    {
      id: "r1",
      fromId: "admit",
      toId: "observe",
      role: "next",
      label: "then observe",
    },
  ],
  transcriptCueId: "cue-admit",
  captions: ["Admission completes before observation."],
};

describe("SemanticTwin", () => {
  test("renders entities in reading order independent of visual draw order", () => {
    render(
      <SemanticTwin
        focusedEntityId={null}
        onActivate={() => {}}
        onFocus={() => {}}
        projection={projection}
        selectedEntityId={null}
      />,
    );

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    const entityButtons = within(twin).getAllByRole("button", {
      name: /Arrive|Admit|Observe/,
    });

    expect(entityButtons.map((node) => node.getAttribute("data-entity-id"))).toEqual([
      "observe",
      "arrive",
      "admit",
    ]);
  });

  test("keeps the twin mounted without display:none or aria-hidden", () => {
    const { container } = render(
      <SemanticTwin
        compact
        focusedEntityId={null}
        onActivate={() => {}}
        onFocus={() => {}}
        projection={projection}
        selectedEntityId={null}
      />,
    );

    const root = container.querySelector(".aiperf-flow__semantic-twin");
    expect(root).not.toBeNull();
    expect(root?.getAttribute("aria-hidden")).not.toBe("true");
    expect(getComputedStyle(root as Element).display).not.toBe("none");
  });

  test("links transcript cue and captions into the twin", () => {
    render(
      <SemanticTwin
        focusedEntityId="admit"
        onActivate={() => {}}
        onFocus={() => {}}
        projection={projection}
        selectedEntityId="admit"
      />,
    );

    expect(screen.getByRole("status").getAttribute("data-transcript-cue")).toBe(
      "cue-admit",
    );
    expect(screen.getByText("Admission completes before observation.")).toBeTruthy();
  });

  test("exposes relations for traversal and evidence ids on entities", () => {
    render(
      <SemanticTwin
        focusedEntityId="observe"
        onActivate={() => {}}
        onFocus={() => {}}
        projection={projection}
        selectedEntityId="observe"
      />,
    );

    expect(screen.getByRole("list", { name: "Relations" })).toBeTruthy();
    expect(screen.getByText("then observe")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Observe" }).getAttribute("data-evidence-ids"),
    ).toBe("ev-1");
  });

  test("synchronizes focus and activation callbacks for keyboard selection", () => {
    const onFocus = vi.fn();
    const onActivate = vi.fn();

    render(
      <SemanticTwin
        focusedEntityId="arrive"
        onActivate={onActivate}
        onFocus={onFocus}
        projection={projection}
        selectedEntityId="arrive"
      />,
    );

    const admit = screen.getByRole("button", { name: "Admit" });
    fireEvent.focus(admit);
    fireEvent.click(admit);

    expect(onFocus).toHaveBeenCalledWith("admit");
    expect(onActivate).toHaveBeenCalledWith("admit");
  });

  test("marks the selected entity for visual/semantic parity", () => {
    render(
      <SemanticTwin
        focusedEntityId="admit"
        onActivate={() => {}}
        onFocus={() => {}}
        projection={projection}
        selectedEntityId="admit"
      />,
    );

    const admit = screen.getByRole("button", { name: "Admit" });
    expect(admit.getAttribute("aria-current")).toBe("true");
    expect(admit.getAttribute("data-selected")).toBe("true");
    expect(admit.getAttribute("data-focused")).toBe("true");
  });

  test("mounts the accessible table alternative for tabular semantics", () => {
    const tabularProjection: SemanticProjection = {
      ...projection,
      entities: projection.entities.map((entity) =>
        entity.id === "observe" ? { ...entity, role: "table" } : entity,
      ),
    };

    render(
      <SemanticTwin
        focusedEntityId={null}
        onActivate={() => {}}
        onFocus={() => {}}
        projection={tabularProjection}
        selectedEntityId={null}
      />,
    );

    const twin = screen.getByRole("region", { name: "Semantic outline" });
    const table = within(twin).getByRole("table", {
      name: "lifecycle semantic alternative",
    });
    expect(table.getAttribute("aria-hidden")).not.toBe("true");
    expect(getComputedStyle(table).display).not.toBe("none");
  });
});
