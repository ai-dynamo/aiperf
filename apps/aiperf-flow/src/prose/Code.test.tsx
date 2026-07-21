/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Code } from "./Code.js";

describe("Code", () => {
  it("renders block content in a pre/code element by default", () => {
    const { container } = render(<Code>{`line one\nline two`}</Code>);
    const pre = container.querySelector("pre");
    expect(pre).not.toBeNull();
    const code = pre?.querySelector("code");
    expect(code).not.toBeNull();
    expect(code?.textContent).toBe("line one\nline two");
  });

  it("applies monospace and surface/ink/stroke tokens to block code", () => {
    const { container } = render(<Code>{"const x = 1;"}</Code>);
    const pre = container.querySelector("pre") as HTMLElement;
    expect(pre.className).toContain("font-mono");
    expect(pre.className).toContain("rounded-none");
    expect(pre.className).toContain("border-stroke-secondary");
    expect(pre.className).toContain("bg-surface-panel");
  });

  it("renders inline content as a span, not a pre block", () => {
    const { container } = render(<Code inline>foo</Code>);
    expect(container.querySelector("pre")).toBeNull();
    const span = container.querySelector("span");
    expect(span).not.toBeNull();
    expect(span?.textContent).toBe("foo");
  });

  it("applies monospace and a subtle background tint to inline code", () => {
    const { container } = render(<Code inline>foo</Code>);
    const span = container.querySelector("span") as HTMLElement;
    expect(span.className).toContain("font-mono");
    expect(span.className).toContain("bg-surface-panel");
  });

  it("merges a caller-supplied className onto the block root", () => {
    const { container } = render(<Code className="extra-code-class">{"x"}</Code>);
    const pre = container.querySelector("pre") as HTMLElement;
    expect(pre.className).toContain("extra-code-class");
  });

  it("merges a caller-supplied className onto the inline root", () => {
    const { container } = render(
      <Code inline className="extra-inline-class">
        x
      </Code>,
    );
    const span = container.querySelector("span") as HTMLElement;
    expect(span.className).toContain("extra-inline-class");
  });
});
