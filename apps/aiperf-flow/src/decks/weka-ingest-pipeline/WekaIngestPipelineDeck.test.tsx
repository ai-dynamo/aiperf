/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { WekaIngestPipelineDeck } from "./WekaIngestPipelineDeck.js";

describe("WekaIngestPipelineDeck", () => {
  it("renders the heading, pill, and framing copy", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Weka ingest, build, and runtime pipeline")).toBeInTheDocument();
    expect(screen.getByText("segment-trie IR")).toBeInTheDocument();
    expect(screen.getAllByText(/ParsedGraph \+ SegmentPool/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/build plane/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/schedule plane/).length).toBeGreaterThan(0);
  });

  it("renders the four top-level stats", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Input forms")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("Coupled contracts")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Emitted IR")).toBeInTheDocument();
    expect(screen.getByText("LlmNode-only")).toBeInTheDocument();
    expect(screen.getByText("Worker materialize")).toBeInTheDocument();
    expect(screen.getByText("stateless")).toBeInTheDocument();
  });

  it("renders the three input-form nodes", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Local .json")).toBeInTheDocument();
    expect(screen.getByText("single Weka trace")).toBeInTheDocument();
    expect(screen.getByText("Local dir")).toBeInTheDocument();
    expect(screen.getByText("parallel dir parser")).toBeInTheDocument();
    expect(screen.getByText("HF org/name (weka)")).toBeInTheDocument();
    expect(screen.getByText("streaming rows")).toBeInTheDocument();
  });

  it("renders the parse/build-plane pipeline nodes", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("workload_detect")).toBeInTheDocument();
    expect(screen.getByText("~4KiB signature sniff")).toBeInTheDocument();
    expect(screen.getAllByText("parse_graph_workload").length).toBeGreaterThan(0);
    expect(screen.getByText("shared ingest seam")).toBeInTheDocument();
    expect(screen.getByText("from_weka_trace")).toBeInTheDocument();
    expect(screen.getByText("seed · tokenizer · corpus · idle cap")).toBeInTheDocument();
    expect(screen.getByText("build_trie_graph")).toBeInTheDocument();
    expect(screen.getByText("-> ParsedGraph + SegmentPool")).toBeInTheDocument();
    expect(screen.getByText("DatasetManager build plane")).toBeInTheDocument();
    expect(screen.getByText("_configure_graph_workload")).toBeInTheDocument();
  });

  it("renders the store and schedule-plane nodes", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("segment store")).toBeInTheDocument();
    expect(screen.getByText("content plane (dedup)")).toBeInTheDocument();
    expect(screen.getByText("graph delta store")).toBeInTheDocument();
    expect(screen.getByText("addressing plane (envelopes)")).toBeInTheDocument();
    expect(screen.getByText("TimingManager reparse")).toBeInTheDocument();
    expect(screen.getByText("scheduling_only=True")).toBeInTheDocument();
    expect(screen.getByText("GraphIRReplayStrategy")).toBeInTheDocument();
    expect(screen.getByText("trie_node_ordinals")).toBeInTheDocument();
    expect(screen.getByText("worker materialize")).toBeInTheDocument();
    expect(screen.getByText("(trace_id, ordinal, variant)")).toBeInTheDocument();
  });

  it("renders the legend entries", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("build plane (content + addressing)")).toBeInTheDocument();
    expect(screen.getByText("parse / schedule plane")).toBeInTheDocument();
    expect(screen.getByText("input forms")).toBeInTheDocument();
    expect(screen.getByText("ordinal agreement")).toBeInTheDocument();
  });

  it("renders the two coupled-contract callouts and the trie caveat", () => {
    render(
      <ReactFlowProvider>
        <WekaIngestPipelineDeck />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Graph / topology contract")).toBeInTheDocument();
    expect(screen.getByText(/dense node ordinals/)).toBeInTheDocument();
    expect(screen.getByText("Payload contract")).toBeInTheDocument();
    expect(screen.getByText(/never from predecessor channel values/)).toBeInTheDocument();
    expect(screen.getByText("Trie caveat")).toBeInTheDocument();
    expect(screen.getByText(/graph_meta.msgpack/)).toBeInTheDocument();
  });
});
