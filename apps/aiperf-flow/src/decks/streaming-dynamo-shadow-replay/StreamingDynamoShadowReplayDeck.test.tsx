/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "../../test/router.js";
import { describe, expect, it } from "vitest";
import { StreamingDynamoShadowReplayDeck } from "./StreamingDynamoShadowReplayDeck.js";

const TAB_LABELS = [
  "1 · Overview",
  "2 · Source",
  "3 · Session",
  "4 · Pipeline & Results",
  "5 · Shadow Replay",
];

describe("StreamingDynamoShadowReplayDeck", () => {
  it("renders the deck header and all five tab labels", () => {
    render(<StreamingDynamoShadowReplayDeck />);
    expect(screen.getByText("Streaming Dynamo shadow replay")).toBeInTheDocument();
    for (const label of TAB_LABELS) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("shows the Overview page by default", () => {
    render(<StreamingDynamoShadowReplayDeck />);
    expect(screen.getByText("End-to-end streaming shadow replay")).toBeInTheDocument();
  });

  it("switches to the Source page", () => {
    render(<StreamingDynamoShadowReplayDeck />);
    fireEvent.click(screen.getByRole("button", { name: "2 · Source" }));
    expect(screen.getByText("Acquire and decode trace files")).toBeInTheDocument();
  });

  it("switches to the Session page", () => {
    render(<StreamingDynamoShadowReplayDeck />);
    fireEvent.click(screen.getByRole("button", { name: "3 · Session" }));
    expect(screen.getByText("Join fragments into conversations")).toBeInTheDocument();
  });

  it("switches to the Pipeline & Results page", () => {
    render(<StreamingDynamoShadowReplayDeck />);
    fireEvent.click(screen.getByRole("button", { name: "4 · Pipeline & Results" }));
    expect(screen.getByText("Deliver actions, compact results, export")).toBeInTheDocument();
  });

  it("switches to the Shadow Replay page", () => {
    render(<StreamingDynamoShadowReplayDeck />);
    fireEvent.click(screen.getByRole("button", { name: "5 · Shadow Replay" }));
    expect(screen.getByText("Re-execute recorded requests against a live endpoint")).toBeInTheDocument();
  });
});
