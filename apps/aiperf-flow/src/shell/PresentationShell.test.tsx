/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PresentationShell } from "./PresentationShell.js";
import type { SlideDefinition } from "../deck/types.js";

const slides: SlideDefinition[] = [
  {
    id: "s1",
    eyebrow: "One",
    title: "Slide One",
    lede: "Lede one.",
    narration: "Narration one.",
    caption: "Caption one.",
    nodes: [],
    edges: [],
  },
  {
    id: "s2",
    eyebrow: "Two",
    title: "Slide Two",
    lede: "Lede two.",
    narration: "Narration two.",
    caption: "Caption two.",
    nodes: [],
    edges: [],
  },
];

describe("PresentationShell", () => {
  it("shows the progress label for the current slide", () => {
    render(
      <PresentationShell slides={slides} slideIndex={0} onSlideIndexChange={vi.fn()}>
        <div>content</div>
      </PresentationShell>,
    );
    expect(screen.getByText("1 / 2")).toBeInTheDocument();
  });

  it("advances to the next slide when Next is clicked", () => {
    const onSlideIndexChange = vi.fn();
    render(
      <PresentationShell slides={slides} slideIndex={0} onSlideIndexChange={onSlideIndexChange}>
        <div>content</div>
      </PresentationShell>,
    );
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(onSlideIndexChange).toHaveBeenCalledWith(1);
  });

  it("disables Back on the first slide and Next on the last", () => {
    const { rerender } = render(
      <PresentationShell slides={slides} slideIndex={0} onSlideIndexChange={vi.fn()}>
        <div>content</div>
      </PresentationShell>,
    );
    expect(screen.getByRole("button", { name: /back/i })).toBeDisabled();

    rerender(
      <PresentationShell slides={slides} slideIndex={1} onSlideIndexChange={vi.fn()}>
        <div>content</div>
      </PresentationShell>,
    );
    expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
  });

  it("shows narration in the subtitles panel", () => {
    render(
      <PresentationShell slides={slides} slideIndex={0} onSlideIndexChange={vi.fn()}>
        <div>content</div>
      </PresentationShell>,
    );
    expect(screen.getByText("Narration one.")).toBeInTheDocument();
  });

  it("toggles speaker notes visibility", () => {
    render(
      <PresentationShell slides={slides} slideIndex={0} onSlideIndexChange={vi.fn()}>
        <div>content</div>
      </PresentationShell>,
    );
    expect(screen.queryByText("Caption one.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /speaker notes/i }));
    expect(screen.getByText("Caption one.")).toBeInTheDocument();
  });
});
