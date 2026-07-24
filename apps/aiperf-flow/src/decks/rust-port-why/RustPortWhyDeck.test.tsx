/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { RustPortWhyDeck } from "./RustPortWhyDeck.js";
import { DECKS } from "../../routes/Home.js";

function renderDeck() {
  return render(
    <MemoryRouter initialEntries={["/rust-port-why"]}>
      <RustPortWhyDeck />
    </MemoryRouter>,
  );
}

describe("RustPortWhyDeck (executive overview)", () => {
  it("leads with the honest one-sentence thesis naming the GIL control plane and Dynamo reuse", () => {
    renderDeck();
    expect(
      screen.getByText(/exists only to work around Python's GIL/i),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/Dynamo/).length).toBeGreaterThan(0);
  });

  it("states the two primary reasons as the hub-and-spoke drivers", () => {
    renderDeck();
    // Reason 1 — strategic shared core. (HubSpoke renders each spoke twice: ring + narrow fallback.)
    expect(screen.getAllByText(/Shared core with Dynamo/i).length).toBeGreaterThan(0);
    // Reason 2 — collapse the accidental multiprocess/ZMQ plane.
    expect(screen.getAllByText(/one process, not ten services/i).length).toBeGreaterThan(0);
  });

  it("reports the RECENT, honest perf numbers (parity at real concurrency; ~3x at the ceiling)", () => {
    renderDeck();
    // Parity at server-bound 250 concurrency.
    expect(screen.getAllByText(/statistically indistinguishable/i).length).toBeGreaterThan(0);
    // Ceiling throughput against a fast target.
    expect(screen.getAllByText(/13,746/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/4,500/).length).toBeGreaterThan(0);
  });

  it("keeps integrity: the client is NOT the bottleneck on normal LLM runs", () => {
    renderDeck();
    expect(
      screen.getByText(/the inference server is the bottleneck, not the client/i),
    ).toBeInTheDocument();
  });

  it("does not overclaim: the single-node connection ceiling is an OS/TCP fact, not a language win", () => {
    renderDeck();
    expect(
      screen.getByText(/OS\/TCP ephemeral-port fact, not a language advantage/i),
    ).toBeInTheDocument();
  });

  it("answers 'why now' — accessibility was the original Python reason, now eroded by AI-assisted development", () => {
    renderDeck();
    expect(screen.getByText(/Why Rust, why now/i)).toBeInTheDocument();
    expect(screen.getAllByText(/spec-driven/i).length).toBeGreaterThan(0);
  });

  it("uses the port itself as PROOF the language barrier is already broken", () => {
    renderDeck();
    expect(screen.getByText(/language barrier is already broken/i)).toBeInTheDocument();
    expect(
      screen.getByText(/the port existing at all is the evidence the barrier is gone/i),
    ).toBeInTheDocument();
  });

  it("preempts the free-threaded (no-GIL) Python objection honestly", () => {
    renderDeck();
    expect(screen.getByText(/no-GIL\) Python make this moot/i)).toBeInTheDocument();
    // The killer honest point: free-threading gives parallelism but not memory safety.
    expect(screen.getByText(/data races/i)).toBeInTheDocument();
  });

  it("pitches fast startup as a quick edit-run-measure iteration win", () => {
    renderDeck();
    // Rendered twice by HubSpoke (ring + narrow fallback).
    expect(screen.getAllByText(/Cold start in a blink/i).length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(/edit-run-measure iteration is tight/i).length,
    ).toBeGreaterThan(0);
  });

  it("is registered on Home's deck listing", () => {
    expect(DECKS.some((deck) => deck.path === "/rust-port-why")).toBe(true);
  });
});
