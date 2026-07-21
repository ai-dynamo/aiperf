/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Row } from "./Row.js";

describe("Row", () => {
  it("renders children in a horizontal flex row", () => {
    const { container } = render(
      <Row>
        <span>one</span>
        <span>two</span>
      </Row>,
    );
    expect(screen.getByText("one")).toBeInTheDocument();
    expect(screen.getByText("two")).toBeInTheDocument();
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("flex");
    expect(root.className).toContain("flex-row");
  });

  it("applies a default gap via inline style", () => {
    const { container } = render(<Row>content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gap).toBe("16px");
  });

  it("applies a caller-supplied gap in pixels", () => {
    const { container } = render(<Row gap={4}>content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gap).toBe("4px");
  });

  it("maps align to the matching items-* class", () => {
    const { container } = render(<Row align="center">content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("items-center");
  });

  it("maps justify to the matching justify-* class", () => {
    const { container } = render(<Row justify="space-between">content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("justify-between");
  });

  it("omits flex-wrap by default", () => {
    const { container } = render(<Row>content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).not.toContain("flex-wrap");
  });

  it("applies flex-wrap when wrap is true", () => {
    const { container } = render(<Row wrap>content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("flex-wrap");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(<Row className="extra-row-class">content</Row>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-row-class");
    expect(root.className).toContain("flex-row");
  });
});
