/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Table } from "./Table.js";

const columns = [
  { key: "service", label: "Service" },
  { key: "status", label: "Status" },
  { key: "rps", label: "RPS", align: "end" as const },
];

const rows = [
  { service: "api-gateway", status: "Steady", rps: "3.2k" },
  { service: "workers", status: "Hot", rps: "8.1k", tone: "warning" as const },
];

describe("Table", () => {
  it("renders a semantic table with header labels and row cells", () => {
    render(<Table columns={columns} rows={rows} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader")).toHaveLength(3);
    expect(screen.getByText("Service")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("RPS")).toBeInTheDocument();
    expect(screen.getByText("api-gateway")).toBeInTheDocument();
    expect(screen.getByText("workers")).toBeInTheDocument();
    expect(screen.getByText("8.1k")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(3); // 1 header + 2 body rows
  });

  it("applies per-column alignment to header and body cells", () => {
    render(<Table columns={columns} rows={rows} />);
    const rpsHeader = screen.getByText("RPS").closest("th");
    expect(rpsHeader?.className).toContain("text-end");
    const rpsCell = screen.getByText("8.1k").closest("td");
    expect(rpsCell?.className).toContain("text-end");
  });

  it("defaults column alignment to start when unspecified", () => {
    render(<Table columns={columns} rows={rows} />);
    const serviceHeader = screen.getByText("Service").closest("th");
    expect(serviceHeader?.className).toContain("text-start");
  });

  it("applies a subtle tint class to rows with a tone", () => {
    render(<Table columns={columns} rows={rows} />);
    const hotRow = screen.getByText("Hot").closest("tr");
    expect(hotRow?.className).toContain("bg-category-yellow");
  });

  it("leaves neutral/untoned rows without a tint class", () => {
    render(<Table columns={columns} rows={rows} />);
    const steadyRow = screen.getByText("Steady").closest("tr");
    expect(steadyRow?.className).not.toContain("bg-category");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <Table columns={columns} rows={rows} className="extra-table-class" />,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-table-class");
    expect(root.className).toContain("border-collapse");
  });
});
