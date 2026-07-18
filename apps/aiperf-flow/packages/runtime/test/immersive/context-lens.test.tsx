// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { ContextLens } from "../../src/immersive/context-lens.js";
import type { SemanticProjection } from "../../src/evaluate/types.js";

afterEach(cleanup);

// Backend-neutral projection for one request-lifecycle scene. Only "client"
// carries a role, description, evidence, and source reference; "worker" is bare
// so the lens must report absent evidence honestly.
function projection(): SemanticProjection {
  return {
    sceneId: "request-lifecycle",
    entities: [
      {
        id: "client",
        label: "Client",
        role: "actor",
        description: "Issues the chat completion request.",
        evidenceIds: ["turn-0-request", "turn-0-usage"],
        source: { source: "request-lifecycle.flow", startOffset: 12, endOffset: 48 },
      },
      { id: "router", label: "Router", kind: "service" },
      { id: "worker", label: "Worker", role: "sink" },
    ],
    relations: [
      { id: "client-router", fromId: "client", toId: "router", label: "dispatches" },
      { id: "router-worker", fromId: "router", toId: "worker" },
    ],
    readingOrder: ["client", "router", "worker"],
  };
}

function lens(): HTMLElement {
  return screen.getByRole("region", { name: "Context Lens" });
}

function renderLens(
  entityId: string,
  handlers: Partial<{
    onClose(): void;
    onFocusWorld(entityId: string): void;
    onOpenTwin(entityId: string): void;
  }> = {},
): void {
  render(
    <ContextLens
      entityId={entityId}
      onClose={handlers.onClose ?? vi.fn()}
      onFocusWorld={handlers.onFocusWorld ?? vi.fn()}
      onOpenTwin={handlers.onOpenTwin ?? vi.fn()}
      projection={projection()}
    />,
  );
}

describe("ContextLens entity summary", () => {
  test("projects the selected entity label, role, and description", () => {
    renderLens("client");

    const region = lens();
    expect(region.getAttribute("data-entity-id")).toBe("client");
    expect(region.getAttribute("data-scene-id")).toBe("request-lifecycle");
    expect(within(region).getByText("Client")).not.toBeNull();
    expect(within(region).getByText("actor")).not.toBeNull();
    expect(
      within(region).getByText("Issues the chat completion request."),
    ).not.toBeNull();
  });

  test("falls back to the entity kind when no role is authored", () => {
    renderLens("router");

    expect(within(lens()).getByText("service")).not.toBeNull();
  });

  test("renders nothing for an entity absent from the projection", () => {
    const { container } = render(
      <ContextLens
        entityId="phantom"
        onClose={vi.fn()}
        onFocusWorld={vi.fn()}
        onOpenTwin={vi.fn()}
        projection={projection()}
      />,
    );

    expect(container.firstChild).toBeNull();
    expect(screen.queryByRole("region", { name: "Context Lens" })).toBeNull();
  });
});

describe("ContextLens relations", () => {
  test("projects both outgoing and incoming relations with authored labels", () => {
    renderLens("router");

    const relations = within(lens()).getByRole("region", { name: "Relations" });
    const items = within(relations).getAllByRole("listitem");
    expect(items).toHaveLength(2);

    const incoming = relations.querySelector('[data-relation-id="client-router"]');
    expect(incoming?.getAttribute("data-direction")).toBe("from");
    expect(incoming?.textContent).toContain("dispatches");

    const outgoing = relations.querySelector('[data-relation-id="router-worker"]');
    expect(outgoing?.getAttribute("data-direction")).toBe("to");
    // No authored label, so the lens synthesizes a directional summary.
    expect(outgoing?.textContent).toContain("Router → Worker");
  });

  test("reports when no related entities exist", () => {
    render(
      <ContextLens
        entityId="lonely"
        onClose={vi.fn()}
        onFocusWorld={vi.fn()}
        onOpenTwin={vi.fn()}
        projection={{
          sceneId: "isolated",
          entities: [{ id: "lonely", label: "Lonely" }],
          relations: [],
          readingOrder: ["lonely"],
        }}
      />,
    );

    const relations = within(lens()).getByRole("region", { name: "Relations" });
    expect(within(relations).getByText("No related entities")).not.toBeNull();
    expect(within(relations).queryAllByRole("listitem")).toHaveLength(0);
  });
});

describe("ContextLens evidence", () => {
  test("lists attached evidence identifiers", () => {
    renderLens("client");

    const evidence = within(lens()).getByRole("region", { name: "Evidence" });
    const ids = within(evidence)
      .getAllByRole("listitem")
      .map((item) => item.getAttribute("data-evidence-id"));
    expect(ids).toEqual(["turn-0-request", "turn-0-usage"]);
  });

  test("reports absent evidence without inventing any", () => {
    renderLens("worker");

    const evidence = within(lens()).getByRole("region", { name: "Evidence" });
    expect(within(evidence).getByText("No evidence is attached")).not.toBeNull();
    expect(within(evidence).queryAllByRole("listitem")).toHaveLength(0);
  });

  test("renders an authored source reference as source:start-end", () => {
    renderLens("client");

    const source = lens().querySelector(".aiperf-flow__context-lens-source");
    expect(source?.textContent).toBe("request-lifecycle.flow:12-48");
  });

  test("omits the source reference when none is authored", () => {
    renderLens("worker");

    expect(
      lens().querySelector(".aiperf-flow__context-lens-source"),
    ).toBeNull();
  });
});

describe("ContextLens actions", () => {
  test("Focus World and Open semantic twin forward the entity identity", () => {
    const onFocusWorld = vi.fn();
    const onOpenTwin = vi.fn();
    renderLens("client", { onFocusWorld, onOpenTwin });

    fireEvent.click(screen.getByRole("button", { name: "Focus World" }));
    fireEvent.click(screen.getByRole("button", { name: "Open semantic twin" }));

    expect(onFocusWorld).toHaveBeenCalledTimes(1);
    expect(onFocusWorld).toHaveBeenCalledWith("client");
    expect(onOpenTwin).toHaveBeenCalledTimes(1);
    expect(onOpenTwin).toHaveBeenCalledWith("client");
  });

  test("Close button and Escape both request dismissal", () => {
    const onClose = vi.fn();
    renderLens("client", { onClose });

    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.keyDown(lens(), { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
