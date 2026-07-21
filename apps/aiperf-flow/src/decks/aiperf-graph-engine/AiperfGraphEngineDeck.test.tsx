/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AiperfGraphEngineDeck } from "./AiperfGraphEngineDeck.js";

describe("AiperfGraphEngineDeck", () => {
  it("renders the title and framing copy", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("AIPerf v2 — Async-Dataflow Graph Engine")).toBeInTheDocument();
    expect(
      screen.getByText(/directed dataflow graph/),
    ).toBeInTheDocument();
    expect(screen.getAllByText("LlmNode").length).toBeGreaterThan(0);
  });

  it("renders the stat tiles", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("Node kinds")).toBeInTheDocument();
    expect(screen.getByText("14")).toBeInTheDocument();
    expect(screen.getAllByText("Channel types").length).toBeGreaterThan(0);
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getAllByText("Reducers").length).toBeGreaterThan(0);
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("Planes (build · schedule)")).toBeInTheDocument();
  });

  it("renders the 'what v2 means' callout", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText('What "v2" means here')).toBeInTheDocument();
    expect(screen.getAllByText("dag_jsonl", { exact: false }).length).toBeGreaterThan(0);
    expect(screen.getAllByText("TraceExecutor").length).toBeGreaterThan(0);
  });

  it("renders the Build -> Schedule -> Execute pipeline stages", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("1 · Build IR")).toBeInTheDocument();
    expect(screen.getByText("DatasetManager")).toBeInTheDocument();
    expect(screen.getByText("2 · Schedule")).toBeInTheDocument();
    expect(screen.getByText("TimingManager")).toBeInTheDocument();
    expect(screen.getByText("3 · Execute")).toBeInTheDocument();
    expect(screen.getAllByText("TraceExecutor").length).toBeGreaterThan(0);
    expect(screen.getByText("src/aiperf/dataset/loader/graph/parser.py")).toBeInTheDocument();
  });

  it("renders the dataflow graph nodes", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("START")).toBeInTheDocument();
    expect(screen.getByText("bootstrap")).toBeInTheDocument();
    // "planner" is also selected by default, so it appears once in the graph and
    // once in the side detail panel.
    expect(screen.getAllByText("planner").length).toBe(2);
    expect(screen.getByText("spawn critic")).toBeInTheDocument();
    expect(screen.getByText("join critic")).toBeInTheDocument();
    expect(screen.getByText("compact ctx")).toBeInTheDocument();
    expect(screen.getByText("final answer")).toBeInTheDocument();
    expect(screen.getByText("END")).toBeInTheDocument();
  });

  it("shows the default-selected node detail and updates on click", () => {
    render(<AiperfGraphEngineDeck />);
    // "planner" is selected by default.
    expect(
      screen.getAllByText(/Dispatch one LLM call → one credit, one request, one record/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("src/aiperf/graph/dispatch/llm.py")).toBeInTheDocument();

    fireEvent.click(screen.getByText("spawn critic"));
    expect(
      screen.getAllByText(/Detach a sub-agent child trace with fresh context/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("src/aiperf/graph/dispatch/spawn.py")).toBeInTheDocument();
  });

  it("renders the legend", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("Wire I/O (credit · request · record)")).toBeInTheDocument();
    expect(screen.getByText("Content / replay (no wire traffic)")).toBeInTheDocument();
    expect(screen.getByText("Control flow (spawn · await · loop · barrier)")).toBeInTheDocument();
    expect(screen.getByText("START / END")).toBeInTheDocument();
  });

  it("renders runtime collaborators", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getAllByText("TraceExecutor").length).toBeGreaterThan(0);
    expect(screen.getByText("Scheduler")).toBeInTheDocument();
    expect(screen.getByText("VersionedChannelStore")).toBeInTheDocument();
    expect(screen.getAllByText("_TraceContext").length).toBeGreaterThan(0);
    expect(screen.getByText("CreditIssuer adapter")).toBeInTheDocument();
    expect(screen.getByText("Dispatch table")).toBeInTheDocument();
  });

  it("renders the node kind taxonomy table with all 14 kinds", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("Node kind")).toBeInTheDocument();
    expect(screen.getAllByText("SpawnNode").length).toBeGreaterThan(0);
    expect(screen.getAllByText("AwaitNode").length).toBeGreaterThan(0);
    expect(screen.getByText("SubgraphNode")).toBeInTheDocument();
    expect(screen.getByText("LoopNode")).toBeInTheDocument();
    expect(screen.getByText("BarrierNode")).toBeInTheDocument();
    expect(screen.getByText("ReplayNode")).toBeInTheDocument();
    expect(screen.getAllByText("ToolCallNode").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ToolResultNode").length).toBeGreaterThan(0);
    expect(screen.getAllByText("CompactNode").length).toBeGreaterThan(0);
    expect(screen.getAllByText("BootstrapNode").length).toBeGreaterThan(0);
    expect(screen.getByText("DelayNode")).toBeInTheDocument();
    expect(screen.getByText("dispatch/edges.py")).toBeInTheDocument();
  });

  it("renders channel types and reducers", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getAllByText("Channel types").length).toBeGreaterThan(0);
    expect(screen.getAllByText("tool_call_stream", { exact: false }).length).toBeGreaterThan(0);
    expect(screen.getByText("text")).toBeInTheDocument();
    expect(screen.getAllByText("Reducers").length).toBeGreaterThan(0);
    expect(screen.getByText("overwrite")).toBeInTheDocument();
    expect(screen.getByText("last write wins (default)")).toBeInTheDocument();
    expect(screen.getByText("add_messages")).toBeInTheDocument();
    expect(screen.getByText("stream_append")).toBeInTheDocument();
    expect(screen.getByText("stream_passthrough")).toBeInTheDocument();
  });

  it("renders the reference doc links", () => {
    render(<AiperfGraphEngineDeck />);
    expect(screen.getByText("docs/benchmark-modes/dag.md")).toBeInTheDocument();
    expect(screen.getByText("docs/reference/weka-graph-structural-handoff.md")).toBeInTheDocument();
  });
});
