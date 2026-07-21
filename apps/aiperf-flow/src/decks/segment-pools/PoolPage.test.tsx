/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PoolPage, simulatePoolInterning } from "./PoolPage.js";

describe("simulatePoolInterning", () => {
  it("starts empty at upTo=0", () => {
    const result = simulatePoolInterning(
      [{ id: "a", conversation: 1, role: "system", content: "hi" }],
      0,
    );
    expect(result.arena).toHaveLength(0);
    expect(result.dedup).toBe(0);
  });

  it("dedupes a row whose content+parent identity was already interned", () => {
    const rows = [
      { id: "c1s", conversation: 1 as const, role: "system" as const, content: "You are helpful." },
      { id: "c2s", conversation: 2 as const, role: "system" as const, content: "You are helpful." },
    ];
    const result = simulatePoolInterning(rows, 2);
    expect(result.arena).toHaveLength(1);
    expect(result.dedup).toBe(1);
    expect(result.resolved.c1s.handle).toBe(result.resolved.c2s.handle);
    expect(result.resolved.c1s.deduped).toBe(false);
    expect(result.resolved.c2s.deduped).toBe(true);
  });

  it("does not dedupe identical content under different parents", () => {
    const rows = [
      { id: "root1", conversation: 1 as const, role: "system" as const, content: "same" },
      { id: "root2", conversation: 2 as const, role: "system" as const, content: "different-parent" },
      { id: "child1", conversation: 1 as const, role: "user" as const, content: "hello", parent: "root1" },
      { id: "child2", conversation: 2 as const, role: "user" as const, content: "hello", parent: "root2" },
    ];
    const result = simulatePoolInterning(rows, 4);
    expect(result.dedup).toBe(0);
    expect(result.arena).toHaveLength(4);
  });
});

describe("PoolPage", () => {
  it("starts with an empty arena", () => {
    render(<PoolPage />);
    expect(screen.getByText("0/6 steps")).toBeInTheDocument();
  });

  it("clicking 'Intern next' advances the arena state", () => {
    render(<PoolPage />);
    fireEvent.click(screen.getByRole("button", { name: "Intern next" }));
    expect(screen.getByText("1/6 steps")).toBeInTheDocument();
    expect(screen.getAllByText("You are a helpful assistant.").length).toBeGreaterThan(0);
  });

  it("'Reset' returns the arena to empty after stepping forward", () => {
    render(<PoolPage />);
    fireEvent.click(screen.getByRole("button", { name: "Intern next" }));
    fireEvent.click(screen.getByRole("button", { name: "Intern next" }));
    expect(screen.getByText("2/6 steps")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("0/6 steps")).toBeInTheDocument();
  });

  it("shows a dedup indicator once a duplicate content string is interned", () => {
    render(<PoolPage />);
    // Rows: c1s, c1u, c1a, c2s (dedup of c1s), c2u (dedup of c1u), c2a
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByRole("button", { name: "Intern next" }));
    }
    expect(screen.getByText("4/6 steps")).toBeInTheDocument();
    expect(screen.getByText("deduped → reused handle")).toBeInTheDocument();
    expect(screen.getByText("Dedup hit")).toBeInTheDocument();
  });

  it("'Run all' advances through every intern call", () => {
    render(<PoolPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run all" }));
    expect(screen.getByText("6/6 steps")).toBeInTheDocument();
  });
});
