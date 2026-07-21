/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { WorkbookPage } from "./WorkbookPage.js";
import { ALGORITHMS } from "./data.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <WorkbookPage />
    </ReactFlowProvider>,
  );
}

describe("WorkbookPage", () => {
  it("renders the index with the total algorithm count and the first algorithm sheet", () => {
    renderPage();
    expect(screen.getByText(`of ${ALGORITHMS.length}`)).toBeInTheDocument();
    const first = ALGORITHMS[0];
    // Title appears both in the index list and the sheet heading.
    expect(screen.getAllByText(first.title).length).toBeGreaterThan(0);
    expect(screen.getByText(first.summary)).toBeInTheDocument();
  });

  it("shows the first trace frame and its synchronized pseudocode, then steps forward", () => {
    renderPage();
    const first = ALGORITHMS[0];
    expect(screen.getByText(`Frame 1/${first.frames.length}`)).toBeInTheDocument();
    // Active pseudocode line id is surfaced as an active pill.
    expect(screen.getAllByText(first.frames[0].activeLineId).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(screen.getByText(`Frame 2/${first.frames.length}`)).toBeInTheDocument();
  });

  it("filters the index by search term", () => {
    renderPage();
    const search = screen.getByPlaceholderText("Source, invariant, failure…");
    fireEvent.change(search, { target: { value: "aggregator placement" } });
    // A far smaller indexed count than the full set.
    expect(screen.queryByText(`${ALGORITHMS.length} indexed`)).not.toBeInTheDocument();
  });

  it("selecting a different algorithm from the index swaps the sheet", () => {
    renderPage();
    const target = ALGORITHMS.find((a) => a.chapter === "artifacts");
    expect(target).toBeDefined();
    if (!target) return;
    // Click the index row (buttons carry aria-pressed).
    const rows = screen.getAllByRole("button", { pressed: false });
    const row = rows.find((el) => within(el).queryByText(target.title));
    expect(row).toBeDefined();
    fireEvent.click(row!);
    expect(screen.getAllByText(target.title).length).toBeGreaterThan(0);
    expect(screen.getByText(target.summary)).toBeInTheDocument();
  });

  it("renders the source contract path for the selected algorithm", () => {
    renderPage();
    const first = ALGORITHMS[0];
    expect(
      screen.getByText(
        `${first.source.path}:${first.source.startLine}-${first.source.endLine}`,
      ),
    ).toBeInTheDocument();
  });
});
