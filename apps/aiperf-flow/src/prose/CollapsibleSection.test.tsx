/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CollapsibleSection } from "./CollapsibleSection.js";

describe("CollapsibleSection", () => {
  it("renders the title", () => {
    render(<CollapsibleSection title="Details">Body text</CollapsibleSection>);
    expect(screen.getByText("Details")).toBeInTheDocument();
  });

  it("hides children by default", () => {
    render(<CollapsibleSection title="Details">Body text</CollapsibleSection>);
    expect(screen.queryByText("Body text")).not.toBeInTheDocument();
  });

  it("shows children when defaultOpen is true", () => {
    render(
      <CollapsibleSection title="Details" defaultOpen>
        Body text
      </CollapsibleSection>,
    );
    expect(screen.getByText("Body text")).toBeInTheDocument();
  });

  it("toggles children visibility on header click", () => {
    render(<CollapsibleSection title="Details">Body text</CollapsibleSection>);
    const toggle = screen.getByRole("button", { name: "Details" });
    fireEvent.click(toggle);
    expect(screen.getByText("Body text")).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(screen.queryByText("Body text")).not.toBeInTheDocument();
  });

  it("reflects open state via aria-expanded", () => {
    render(<CollapsibleSection title="Details">Body text</CollapsibleSection>);
    const toggle = screen.getByRole("button", { name: "Details" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
  });

  it("flips the chevron rotation class based on open state", () => {
    render(<CollapsibleSection title="Details">Body text</CollapsibleSection>);
    const toggle = screen.getByRole("button", { name: "Details" });
    const chevron = toggle.querySelector("svg");
    expect(chevron).not.toBeNull();
    expect(chevron?.getAttribute("class")).not.toContain("rotate-90");
    fireEvent.click(toggle);
    expect(chevron?.getAttribute("class")).toContain("rotate-90");
  });

  it("has no rounded corners or box shadow on the root", () => {
    const { container } = render(<CollapsibleSection title="Details">Body text</CollapsibleSection>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("rounded-none");
    expect(root.className).not.toContain("shadow");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <CollapsibleSection title="Details" className="extra-class">
        Body text
      </CollapsibleSection>,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-class");
    expect(root.className).toContain("rounded-none");
  });
});
