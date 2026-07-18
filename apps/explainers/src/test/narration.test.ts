import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { estimateNarrationMs, speakNarration, splitWords, stopNarration } from "../core/narration";

describe("narration timing", () => {
  it("estimates duration from word count with a minimum floor", () => {
    expect(estimateNarrationMs("hello")).toBeGreaterThanOrEqual(2500);
    const longer = estimateNarrationMs(
      "AIPerf ships as one native aiperf binary with a hidden execution child.",
    );
    expect(longer).toBeGreaterThan(estimateNarrationMs("short line"));
  });

  it("splits words for subtitles", () => {
    expect(splitWords("  one   two three ")).toEqual(["one", "two", "three"]);
  });
});

describe("speakNarration fallback", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    stopNarration();
  });

  it("advances via timer when speech is disabled", () => {
    const onComplete = vi.fn();
    const onWord = vi.fn();

    speakNarration("one two three four", {
      useSpeech: false,
      onWord,
      onComplete,
    });

    expect(onWord).toHaveBeenCalledWith(0);
    vi.runAllTimers();
    expect(onComplete).toHaveBeenCalled();
  });
});
