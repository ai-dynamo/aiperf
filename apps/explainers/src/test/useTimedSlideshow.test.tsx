import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as narration from "../core/narration";
import { useTimedSlideshow } from "../core/useTimedSlideshow";

describe("useTimedSlideshow", () => {
  beforeEach(() => {
    vi.spyOn(narration, "stopNarration").mockImplementation(() => {});
    vi.spyOn(narration, "speakNarration").mockReturnValue(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stops narration and clears word index when paused", async () => {
    const { result, rerender } = renderHook(
      (props: { playing: boolean; index: number }) =>
        useTimedSlideshow({
          index: props.index,
          playing: props.playing,
          narrationEnabled: false,
          narrations: ["one two three"],
          onAdvance: vi.fn(),
        }),
      { initialProps: { playing: true, index: 0 } },
    );

    expect(result.current.activeWordIndex).toBe(0);

    rerender({ playing: false, index: 0 });

    await waitFor(() => {
      expect(result.current.activeWordIndex).toBe(-1);
    });
    expect(narration.stopNarration).toHaveBeenCalled();
  });

  it("restarts narration when slide index changes", () => {
    const speak = vi.spyOn(narration, "speakNarration").mockReturnValue(() => {});
    const { rerender } = renderHook(
      (props: { index: number }) =>
        useTimedSlideshow({
          index: props.index,
          playing: true,
          narrationEnabled: false,
          narrations: ["first slide", "second slide"],
          onAdvance: vi.fn(),
        }),
      { initialProps: { index: 0 } },
    );

    expect(speak).toHaveBeenCalledTimes(1);
    expect(speak.mock.calls[0][0]).toBe("first slide");

    rerender({ index: 1 });

    expect(speak).toHaveBeenCalledTimes(2);
    expect(speak.mock.calls[1][0]).toBe("second slide");
  });

  it("restarts narration when restartKey changes on the same slide", () => {
    const speak = vi.spyOn(narration, "speakNarration").mockReturnValue(() => {});
    const { rerender } = renderHook(
      (props: { restartKey: number }) =>
        useTimedSlideshow({
          index: 0,
          playing: true,
          narrationEnabled: false,
          narrations: ["hello world"],
          restartKey: props.restartKey,
          onAdvance: vi.fn(),
        }),
      { initialProps: { restartKey: 0 } },
    );

    expect(speak).toHaveBeenCalledTimes(1);
    rerender({ restartKey: 1 });
    expect(speak).toHaveBeenCalledTimes(2);
    expect(speak.mock.calls[1][0]).toBe("hello world");
  });
});
