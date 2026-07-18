// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import type { SemanticProjection } from "../../src/evaluate/types.js";
import { SemanticFallbackTable } from "../../src/semantic/fallback-table.js";

afterEach(cleanup);

const projection: SemanticProjection = {
  sceneId: "request-lifecycle",
  readingOrder: ["observe", "flows-to", "queue", "observe"],
  entities: [
    {
      id: "queue",
      label: "Request queue",
      description: "Requests waiting for admission",
      role: "table",
      evidenceIds: ["trace-17", "metric-queue-depth"],
    },
    {
      id: "observe",
      label: "Metrics observer",
      role: "chart",
    },
    {
      id: "worker",
      label: "Model worker",
      description: "Processes admitted requests",
    },
  ],
  relations: [
    {
      id: "flows-to",
      fromId: "queue",
      toId: "worker",
      label: "Queue feeds worker",
      role: "data-flow",
    },
    {
      id: "records",
      fromId: "worker",
      toId: "observe",
      label: "Observer records worker output",
    },
  ],
};

describe("SemanticFallbackTable", () => {
  test("renders entities and relations in semantic reading order", () => {
    render(<SemanticFallbackTable projection={projection} />);

    const table = screen.getByRole("table", {
      name: "request-lifecycle semantic alternative",
    });
    const rows = within(table).getAllByRole("row").slice(1);

    expect(rows.map((row) => row.getAttribute("data-semantic-id"))).toEqual([
      "observe",
      "flows-to",
      "queue",
      "worker",
      "records",
    ]);
    expect(rows.map((row) => row.getAttribute("data-semantic-type"))).toEqual([
      "entity",
      "relation",
      "entity",
      "entity",
      "relation",
    ]);
  });

  test("preserves descriptions, evidence, roles, and relation endpoints", () => {
    render(
      <SemanticFallbackTable
        caption="Request lifecycle data"
        projection={projection}
      />,
    );

    const table = screen.getByRole("table", {
      name: "Request lifecycle data",
    });
    const queueRow = within(table)
      .getByText("Request queue")
      .closest("tr");
    const relationRow = within(table)
      .getByText("Queue feeds worker")
      .closest("tr");

    expect(queueRow).not.toBeNull();
    expect(within(queueRow as HTMLElement).getByText("Requests waiting for admission")).toBeTruthy();
    expect(within(queueRow as HTMLElement).getByText("trace-17, metric-queue-depth")).toBeTruthy();
    expect(within(queueRow as HTMLElement).getByText("table")).toBeTruthy();

    expect(relationRow).not.toBeNull();
    expect(within(relationRow as HTMLElement).getByText("Request queue → Model worker")).toBeTruthy();
    expect(within(relationRow as HTMLElement).getByText("data-flow")).toBeTruthy();
  });

  test("falls back to stable identifiers when relation endpoints are absent", () => {
    const incomplete: SemanticProjection = {
      sceneId: "partial",
      readingOrder: ["unknown-link"],
      entities: [],
      relations: [
        {
          id: "unknown-link",
          fromId: "missing-source",
          toId: "missing-target",
        },
      ],
    };

    render(<SemanticFallbackTable projection={incomplete} />);

    expect(screen.getByText("missing-source → missing-target")).toBeTruthy();
    expect(screen.getByText("unknown-link")).toBeTruthy();
  });
});
