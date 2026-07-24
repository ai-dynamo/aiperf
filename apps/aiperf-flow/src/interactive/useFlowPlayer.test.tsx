/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useFlowPlayer } from "./useFlowPlayer.js";
import type { FlowStep } from "./types.js";

const STEPS: FlowStep[] = [
  { nodeId: "scheduler", caption: "RequestRateWorkload issues the request" },
  { nodeId: "admission", caption: "SlotPool + StopChecker admit it" },
  { nodeId: "dispatch", caption: "Dispatcher routes to the WorkerSink" },
];

function Harness(): React.JSX.Element {
  // autoPlayMs is large so no timer fires during a synchronous test.
  const player = useFlowPlayer(STEPS, { autoPlayMs: 100000 });
  return (
    <div>
      <p>node: {player.activeNodeId}</p>
      <p>caption: {player.caption}</p>
      <p>index: {player.index}</p>
      <p>isLast: {String(player.isLast)}</p>
      <button type="button" onClick={player.next}>
        next
      </button>
      <button type="button" onClick={player.back}>
        back
      </button>
      <button type="button" onClick={() => player.scrubTo(2)}>
        scrub-end
      </button>
      <button type="button" onClick={player.reset}>
        reset
      </button>
    </div>
  );
}

describe("useFlowPlayer", () => {
  it("starts on the first step", () => {
    render(<Harness />);
    expect(screen.getByText("node: scheduler")).toBeInTheDocument();
    expect(screen.getByText("caption: RequestRateWorkload issues the request")).toBeInTheDocument();
    expect(screen.getByText("index: 0")).toBeInTheDocument();
  });

  it("advances and retreats through the steps", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "next" }));
    expect(screen.getByText("node: admission")).toBeInTheDocument();
    expect(screen.getByText("caption: SlotPool + StopChecker admit it")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "back" }));
    expect(screen.getByText("node: scheduler")).toBeInTheDocument();
  });

  it("scrubs directly to a later step and reports isLast at the end", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "scrub-end" }));
    expect(screen.getByText("node: dispatch")).toBeInTheDocument();
    expect(screen.getByText("index: 2")).toBeInTheDocument();
    expect(screen.getByText("isLast: true")).toBeInTheDocument();
  });

  it("clamps next at the last step (no overrun)", () => {
    render(<Harness />);
    for (let i = 0; i < 6; i++) {
      fireEvent.click(screen.getByRole("button", { name: "next" }));
    }
    expect(screen.getByText("index: 2")).toBeInTheDocument();
    expect(screen.getByText("node: dispatch")).toBeInTheDocument();
  });

  it("resets to the first step", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "scrub-end" }));
    fireEvent.click(screen.getByRole("button", { name: "reset" }));
    expect(screen.getByText("index: 0")).toBeInTheDocument();
    expect(screen.getByText("node: scheduler")).toBeInTheDocument();
  });
});
