// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import type { FlowIr, SceneIr } from "@aiperf/flow-schema";
import { cleanup, fireEvent, render, screen, within, act } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { FlowApp } from "../src/app.js";
import type {
  NarratorBackend,
  NarratorUtterance,
  NarratorVoice,
} from "../src/narrative/narrator.js";
import type { Clock } from "../src/player.js";

const sourceMap = {
  source: "narrative-app.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

afterEach(cleanup);

class FakeNarratorBackend implements NarratorBackend {
  readonly available = true;
  readonly spoken: NarratorUtterance[] = [];
  readonly operations: string[] = [];
  readonly completions = new Map<string, () => void>();
  readonly #voices: readonly NarratorVoice[] = [
    { id: "voice-a", name: "Voice A", language: "en-US", default: true },
  ];

  voices(): readonly NarratorVoice[] {
    return this.#voices;
  }

  speak(utterance: NarratorUtterance, onComplete?: () => void): void {
    this.spoken.push(utterance);
    this.operations.push(`speak:${utterance.cueId}`);
    if (onComplete !== undefined) {
      this.completions.set(utterance.cueId, onComplete);
    }
  }

  pause(): void {
    this.operations.push("pause");
  }

  resume(): void {
    this.operations.push("resume");
  }

  cancel(): void {
    this.operations.push("cancel");
  }
}

class VirtualClock implements Clock {
  #nowNs: bigint;
  #nextHandle = 1;
  readonly #callbacks = new Map<number, () => void>();

  constructor(nowNs = 0n) {
    this.#nowNs = nowNs;
  }

  nowNs(): bigint {
    return this.#nowNs;
  }

  requestFrame(callback: () => void): number {
    const handle = this.#nextHandle;
    this.#nextHandle += 1;
    this.#callbacks.set(handle, callback);
    return handle;
  }

  cancelFrame(handle: number): void {
    this.#callbacks.delete(handle);
  }

  advanceMs(elapsedMs: number): void {
    this.#nowNs += BigInt(elapsedMs) * 1_000_000n;
    const callbacks = [...this.#callbacks.values()];
    this.#callbacks.clear();
    callbacks.forEach((callback) => callback());
  }
}

function foundationRoots(): SceneIr["roots"] {
  return [
    {
      kind: "rect",
      id: "cli",
      geometry: { x: 40, y: 100, width: 160, height: 72 },
      style: { fill: "#315b8a" },
      accessibility: {
        label: "CLI",
        description: "Command-line process",
      },
      fallback: "CLI",
      sourceMap,
    },
  ];
}

function legacyScene(): SceneIr {
  return {
    id: "execution",
    title: "Execution boundary",
    summary: "The CLI starts a runtime.",
    roots: foundationRoots(),
    camera: [],
    timeline: [
      {
        id: "reveal-cli",
        at: 0,
        duration: 400,
        action: "reveal",
        target: "cli",
        sourceMap,
      },
    ],
    narration: "The CLI starts a fresh runtime and dispatches work.",
    interactions: [],
    responsive: [],
    accessibility: {
      label: "Execution boundary diagram",
      readingOrder: ["cli"],
    },
    fallback: "CLI starts Runtime.",
    sourceMap,
  };
}

function timedScene(): SceneIr {
  return {
    ...legacyScene(),
    id: "timed-execution",
    title: "Timed execution",
    narration: "Legacy full transcript remains for the transcript panel.",
    narrativeTrack: {
      language: "en-US",
      voice: "nvidia-narrator",
      cues: [
        {
          id: "dispatch",
          startMs: 0,
          endMs: 300,
          spokenText: "The runtime dispatches work.",
          subtitleText: "Runtime dispatches work.",
        },
        {
          id: "observe",
          startMs: 300,
          endMs: 700,
          spokenText: "The observer records every request.",
          subtitleText: "Observer records requests.",
        },
      ],
    },
    timeline: [
      {
        id: "reveal-cli",
        at: 0,
        duration: 700,
        action: "reveal",
        target: "cli",
        sourceMap,
      },
    ],
  };
}

function flowWith(scenes: readonly SceneIr[]): FlowIr {
  return {
    irVersion: 2,
    id: "narrative-flow",
    title: "Narrative flow",
    capabilities: [],
    tokens: {},
    themes: [],
    scenes,
    sourceMap,
  } as unknown as FlowIr;
}

function runCommand(label: string | RegExp): void {
  fireEvent.click(screen.getByRole("button", { name: "Open commands" }));
  fireEvent.click(
    screen.getByRole("option", {
      name: typeof label === "string" ? new RegExp(label, "i") : label,
    }),
  );
}

describe("FlowApp narrative synchronization", () => {
  test("synthesizes one subtitle cue from legacy scene.narration", () => {
    const backend = new FakeNarratorBackend();
    render(
      <FlowApp
        flow={flowWith([legacyScene()])}
        forceSvgFallback
        narratorBackend={backend}
      />,
    );

    const subtitles = screen.getByRole("region", { name: "Subtitles" });
    expect(subtitles.getAttribute("data-cue-id")).toBe("execution:narration");
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).toContain("The CLI starts a fresh runtime and dispatches work.");

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual([
      "execution:narration",
    ]);
  });

  test("advances timed narrativeTrack subtitles and narration with the player", () => {
    const backend = new FakeNarratorBackend();
    const clock = new VirtualClock();
    render(
      <FlowApp
        clock={clock}
        flow={flowWith([timedScene()])}
        forceSvgFallback
        narratorBackend={backend}
      />,
    );

    expect(screen.getByRole("region", { name: "Subtitles" }).getAttribute("data-cue-id")).toBe(
      "dispatch",
    );
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).toContain("Runtime dispatches work.");

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual(["dispatch"]);

    act(() => {
      clock.advanceMs(300);
    });
    expect(screen.getByRole("region", { name: "Subtitles" }).getAttribute("data-cue-id")).toBe(
      "observe",
    );
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).toContain("Observer records requests.");
    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual([
      "dispatch",
      "observe",
    ]);
  });

  test("advances after voice completion and stops on the final scene", async () => {
    const backend = new FakeNarratorBackend();
    const second = {
      ...legacyScene(),
      id: "second",
      title: "Second scene",
      narration: "The final scene narration.",
    };
    render(
      <FlowApp
        flow={flowWith([legacyScene(), second])}
        forceSvgFallback
        narratorBackend={backend}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => {
      backend.completions.get("execution:narration")?.();
    });

    await screen.findByRole("heading", { name: "Second scene" });
    await vi.waitFor(() =>
      expect(backend.completions.has("second:narration")).toBe(true),
    );

    act(() => {
      backend.completions.get("second:narration")?.();
    });

    expect(screen.getByRole("heading", { name: "Second scene" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();
  });

  test("keeps the full transcript panel while showing only the active subtitle cue", () => {
    render(
      <FlowApp
        flow={flowWith([timedScene()])}
        forceSvgFallback
        narratorBackend={new FakeNarratorBackend()}
      />,
    );

    const transcript = screen.getByRole("region", {
      name: "Narration transcript",
    });
    expect(within(transcript).getByText(
      "Legacy full transcript remains for the transcript panel.",
    )).toBeTruthy();
    expect(
      document.querySelector(".aiperf-flow__subtitle-cue")?.textContent,
    ).not.toContain("Legacy full transcript remains for the transcript panel.");
  });

  test("captions toggle hides the cue while mute cancels audible narration", () => {
    const backend = new FakeNarratorBackend();
    render(
      <FlowApp
        flow={flowWith([timedScene()])}
        forceSvgFallback
        narratorBackend={backend}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    runCommand("Mute narration");
    expect(backend.operations).toContain("cancel");
    runCommand("Unmute narration");

    fireEvent.click(screen.getByRole("button", { name: "Turn subtitles off" }));
    expect(document.querySelector(".aiperf-flow__subtitle-cue")).toBeNull();
    expect(
      within(screen.getByRole("region", { name: "Subtitles" })).getByRole(
        "status",
      ).textContent,
    ).toBe("");
  });

  test("pause, restart, navigation, and unmount keep narrator lifecycle aligned", () => {
    const backend = new FakeNarratorBackend();
    const clock = new VirtualClock();
    const { unmount } = render(
      <FlowApp
        clock={clock}
        flow={flowWith([timedScene(), legacyScene()])}
        forceSvgFallback
        narratorBackend={backend}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => {
      clock.advanceMs(150);
    });
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    expect(backend.operations.at(-1)).toBe("pause");

    runCommand("Restart");
    expect(backend.operations).toContain("cancel");
    expect(screen.getByText("0 ms")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    fireEvent.click(screen.getByRole("button", { name: "Next scene" }));
    expect(screen.getByRole("heading", { name: "Execution boundary" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Subtitles" }).getAttribute("data-cue-id")).toBe(
      "execution:narration",
    );

    const cancelCountBeforeUnmount = backend.operations.filter(
      (operation) => operation === "cancel",
    ).length;
    unmount();
    expect(
      backend.operations.filter((operation) => operation === "cancel").length,
    ).toBeGreaterThan(cancelCountBeforeUnmount);
  });

  test("exploration pauses and resumes narration at the exact beat", () => {
    const backend = new FakeNarratorBackend();
    const clock = new VirtualClock();
    render(
      <FlowApp
        clock={clock}
        flow={flowWith([timedScene()])}
        forceSvgFallback
        narratorBackend={backend}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => {
      clock.advanceMs(300);
    });
    expect(screen.getByText("300 ms")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Explore" }));
    expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();
    expect(backend.operations).toContain("pause");
    expect(screen.getByText("300 ms")).toBeTruthy();
    expect(screen.getByRole("region", { name: "Subtitles" }).getAttribute("data-cue-id")).toBe(
      "observe",
    );

    fireEvent.click(screen.getByRole("button", { name: "Resume lesson" }));
    expect(backend.operations.at(-1)).toBe("resume");
    expect(screen.getByText("300 ms")).toBeTruthy();
    expect(screen.getByRole("region", { name: "Subtitles" }).getAttribute("data-cue-id")).toBe(
      "observe",
    );
  });

  test("preserves Canvas/SVG stage and semantic twin while narration mounts", () => {
    const { container } = render(
      <FlowApp
        flow={flowWith([legacyScene()])}
        forceSvgFallback
        narratorBackend={new FakeNarratorBackend()}
      />,
    );

    expect(screen.getByRole("region", { name: "Scene field" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Semantic outline" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Subtitles" })).toBeTruthy();
    expect(container.querySelector("svg.aiperf-flow__svg-fallback")).not.toBeNull();
  });
});
