/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WekaTimingTransformsDeck } from "./WekaTimingTransformsDeck.js";

describe("WekaTimingTransformsDeck", () => {
  it("renders the title, pill, and framing copy", () => {
    render(<WekaTimingTransformsDeck />);
    expect(screen.getByText("Weka timing transforms")).toBeInTheDocument();
    expect(screen.getByText("warped clock")).toBeInTheDocument();
    expect(screen.getAllByText(/idle-gap warp/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/t\* snapshot chop/).length).toBeGreaterThan(0);
  });

  it("renders the idle-gap warp section heading and framing text", () => {
    render(<WekaTimingTransformsDeck />);
    expect(
      screen.getByText("Idle-gap warp — collapse dead air, preserve shape"),
    ).toBeInTheDocument();
    expect(screen.getByText(/active intervals/)).toBeInTheDocument();
    expect(screen.getByText("Active intervals A–D · idle 26s -> 5s")).toBeInTheDocument();
    expect(screen.getByText("cap = 5s")).toBeInTheDocument();
  });

  it("renders the raw and warped rows with the A-D interval nodes and their durations", () => {
    render(<WekaTimingTransformsDeck />);
    expect(screen.getByText("raw")).toBeInTheDocument();
    expect(screen.getByText("warped")).toBeInTheDocument();
    expect(screen.getAllByText("A").length).toBe(2);
    expect(screen.getAllByText("B").length).toBe(2);
    expect(screen.getAllByText("C").length).toBe(2);
    expect(screen.getAllByText("D").length).toBe(2);
    expect(screen.getByText("30s – 32s")).toBeInTheDocument();
    expect(screen.getByText("9s – 11s")).toBeInTheDocument();
    expect(screen.getByText("idle 26s > cap")).toBeInTheDocument();
    expect(screen.getByText("cap 5s")).toBeInTheDocument();
  });

  it("renders the two idle-gap warp callouts", () => {
    render(<WekaTimingTransformsDeck />);
    expect(screen.getByText("Why active-interval, not start-to-start")).toBeInTheDocument();
    expect(screen.getByText("api_time is never warped")).toBeInTheDocument();
  });

  it("renders the t* snapshot chop section heading and framing text", () => {
    render(<WekaTimingTransformsDeck />);
    expect(
      screen.getByText("t* snapshot chop — resume from the live frontier"),
    ).toBeInTheDocument();
    expect(screen.getByText(/chop_trie_at_tstar/)).toBeInTheDocument();
    expect(screen.getByText("Before and after the chop at t*")).toBeInTheDocument();
    expect(screen.getByText("t* = 10s")).toBeInTheDocument();
  });

  it("renders the before/after chop rows with n0-n5 nodes and the START re-root", () => {
    render(<WekaTimingTransformsDeck />);
    expect(screen.getByText("before")).toBeInTheDocument();
    expect(screen.getByText("after")).toBeInTheDocument();
    expect(screen.getAllByText("n3").length).toBe(2);
    expect(screen.getAllByText("n4").length).toBe(2);
    expect(screen.getAllByText("n5").length).toBe(2);
    expect(screen.getByText("n0")).toBeInTheDocument();
    expect(screen.getByText("n1")).toBeInTheDocument();
    expect(screen.getByText("n2")).toBeInTheDocument();
    expect(screen.getByText("START")).toBeInTheDocument();
    expect(screen.getByText("min_start_delay = arrival − t*")).toBeInTheDocument();
  });

  it("renders the prompt-path-kept-whole callout", () => {
    render(<WekaTimingTransformsDeck />);
    expect(screen.getByText("Prompt path is kept whole")).toBeInTheDocument();
    expect(screen.getByText(/prompt_segment_ids/)).toBeInTheDocument();
  });
});
