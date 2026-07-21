/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PayloadsPage } from "./PayloadsPage.js";

describe("PayloadsPage", () => {
  it("renders the heading and all six domain selector options", () => {
    render(<PayloadsPage />);
    expect(screen.getByText("Payload — six disjoint hash domains")).toBeInTheDocument();

    for (const name of ["Message", "Text", "Raw", "TokenIds", "Media", "TraceHashIds"]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("defaults to the Message domain with its recipe rows visible", () => {
    render(<PayloadsPage />);
    expect(screen.getByRole("button", { name: "Message" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Payload::Message")).toBeInTheDocument();
    expect(screen.getByText("role, wire: Bytes, tokens: Box<[u32]>")).toBeInTheDocument();
    expect(screen.getByText(`"message\\0"`)).toBeInTheDocument();
    expect(screen.getByText("each token (u32 LE)")).toBeInTheDocument();
  });

  it("switches the displayed recipe when a different domain is clicked", () => {
    render(<PayloadsPage />);

    fireEvent.click(screen.getByRole("button", { name: "TraceHashIds" }));

    expect(screen.getByRole("button", { name: "TraceHashIds" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Message" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Payload::TraceHashIds")).toBeInTheDocument();
    expect(screen.getByText("hash_ids: Box<[i64]>, block_size: usize")).toBeInTheDocument();
    expect(screen.getByText(`"trace-hash-ids\\0"`)).toBeInTheDocument();
    expect(screen.getByText("each hash id (i64 sequence)")).toBeInTheDocument();
    expect(screen.queryByText("Payload::Message")).not.toBeInTheDocument();
  });

  it("always shows the shared version constant and parent-id framing rows", () => {
    render(<PayloadsPage />);

    for (const label of ["HASH_VERSION", "domain prefix", "parent id"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(screen.getAllByText(`b"aiperf-dataset-segment-v1\\0"`).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Media" }));
    expect(screen.getAllByText(`b"aiperf-dataset-segment-v1\\0"`).length).toBeGreaterThan(0);
  });

  it("renders the why-parent-by-id callout", () => {
    render(<PayloadsPage />);
    const heading = screen.getByText("Why parent-by-id, not parent-by-index");
    expect(within(heading.closest("div")!.parentElement!).getByText(/content hash/)).toBeInTheDocument();
  });
});
