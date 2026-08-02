/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { WekaSegmentStoreDeck } from "./WekaSegmentStoreDeck.js";

describe("WekaSegmentStoreDeck", () => {
  it("renders the title and WEKA_UNIFIED_STORE badge", () => {
    render(<WekaSegmentStoreDeck />);
    expect(
      screen.getByText("Weka content plane and the unified segment store"),
    ).toBeInTheDocument();
    expect(screen.getByText("WEKA_UNIFIED_STORE")).toBeInTheDocument();
  });

  it("renders the segment pool prefix-sharing explanation and chip paths", () => {
    render(<WekaSegmentStoreDeck />);
    expect(screen.getByText("Content-addressed segment pool")).toBeInTheDocument();
    expect(screen.getByText("Two turns sharing a prefix")).toBeInTheDocument();
    expect(screen.getByText("r2 prompt")).toBeInTheDocument();
    expect(screen.getByText("r4 prompt")).toBeInTheDocument();
    expect(screen.getByText("SegmentPool")).toBeInTheDocument();
    // s0..s2 shared between both paths plus the pool row -> at least 3 occurrences of s0
    expect(screen.getAllByText("s0").length).toBeGreaterThanOrEqual(3);
    expect(screen.getAllByText("s4").length).toBeGreaterThanOrEqual(1);
  });

  it("renders the 'what a node carries' callout with real field names", () => {
    render(<WekaSegmentStoreDeck />);
    expect(screen.getByText("What a node carries")).toBeInTheDocument();
    expect(screen.getByText("metadata.trie.prompt_segment_ids")).toBeInTheDocument();
    expect(screen.getByText("response_id")).toBeInTheDocument();
  });

  it("renders the unified store on-disk layout diagram nodes", () => {
    render(<WekaSegmentStoreDeck />);
    expect(screen.getByText("Unified store — one directory, four files")).toBeInTheDocument();
    expect(screen.getByText("On-disk layout")).toBeInTheDocument();
    expect(screen.getByText("mmap ACCESS_READ")).toBeInTheDocument();
    expect(screen.getByText("aiperf_graph_segments_<benchmark_id>/")).toBeInTheDocument();
    expect(screen.getByText("content.idx")).toBeInTheDocument();
    expect(screen.getByText("hex map (A1) | packed 'Q' handles (A2)")).toBeInTheDocument();
    expect(screen.getByText("content.blob")).toBeInTheDocument();
    expect(screen.getByText("nodes.idx")).toBeInTheDocument();
    expect(screen.getByText("nodes.blob")).toBeInTheDocument();
  });

  it("renders the A1 vs A2 format-detection branch cards", () => {
    render(<WekaSegmentStoreDeck />);
    expect(
      screen.getByText("A1 vs A2 — detected from the first byte of content.idx"),
    ).toBeInTheDocument();
    expect(screen.getByText("A1 hex composition")).toBeInTheDocument();
    expect(screen.getByText("_interned = False")).toBeInTheDocument();
    expect(screen.getByText("A2 packed int handles")).toBeInTheDocument();
    expect(screen.getByText("_interned = True")).toBeInTheDocument();
    expect(
      screen.getByText("Worker branches on the store-level _interned flag"),
    ).toBeInTheDocument();
  });

  it("renders the Phase A scope callout", () => {
    render(<WekaSegmentStoreDeck />);
    expect(screen.getByText("Phase A scope")).toBeInTheDocument();
    expect(screen.getByText(/WEKA_SEGMENT_TRIE_IR=False/)).toBeInTheDocument();
  });
});
