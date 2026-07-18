// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  createBrowserSpeechSynthesisBackend,
  NarratorController,
  type NarratorBackend,
  type NarratorUtterance,
  type NarratorVoice,
} from "../../src/narrative/narrator.js";

class FakeBackend implements NarratorBackend {
  readonly available = true;
  readonly spoken: NarratorUtterance[] = [];
  readonly operations: string[] = [];
  readonly #voices: readonly NarratorVoice[] = [
    { id: "voice-a", name: "Voice A", language: "en-US", default: true },
    { id: "voice-b", name: "Voice B", language: "en-GB", default: false },
  ];

  voices(): readonly NarratorVoice[] {
    return this.#voices;
  }

  speak(utterance: NarratorUtterance): void {
    this.spoken.push(utterance);
    this.operations.push(`speak:${utterance.cueId}`);
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

const cues = [
  { id: "establish", atMs: 0, text: "Establish the request path." },
  { id: "dispatch", atMs: 300, text: "Dispatch work to the runtime." },
  { id: "observe", atMs: 800, text: "Observe the response." },
] as const;

describe("NarratorController", () => {
  test("dispatches cue ids at canonical integer scene times", () => {
    const backend = new FakeBackend();
    const narrator = new NarratorController(cues, backend);

    narrator.play(0.9);
    narrator.sync(299.9);
    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual(["establish"]);

    narrator.sync(300.8);
    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual([
      "establish",
      "dispatch",
    ]);
    expect(narrator.snapshot()).toMatchObject({
      timeMs: 300,
      activeCueId: "dispatch",
      status: "playing",
    });
  });

  test("pause-to-explore resumes the active cue without replay or skip", () => {
    const backend = new FakeBackend();
    const narrator = new NarratorController(cues, backend);

    narrator.play(0);
    narrator.sync(300);
    narrator.pause(300);
    narrator.resume(300);
    narrator.sync(799);
    narrator.sync(800);

    expect(backend.operations).toEqual([
      "cancel",
      "speak:establish",
      "cancel",
      "speak:dispatch",
      "pause",
      "resume",
      "cancel",
      "speak:observe",
    ]);
  });

  test("seek cancels speech and starts only a cue exactly at the target beat", () => {
    const backend = new FakeBackend();
    const narrator = new NarratorController(cues, backend);

    narrator.play(0);
    narrator.seek(300);

    expect(backend.operations).toEqual([
      "cancel",
      "speak:establish",
      "cancel",
      "cancel",
      "speak:dispatch",
    ]);
    expect(narrator.snapshot().timeMs).toBe(300);
  });

  test("mute consumes crossed cues without delayed replay after unmute", () => {
    const backend = new FakeBackend();
    const narrator = new NarratorController(cues, backend);

    narrator.play(0);
    narrator.setMuted(true);
    narrator.sync(800);
    narrator.setMuted(false);
    narrator.sync(800);

    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual(["establish"]);
    expect(backend.operations).toContain("cancel");
  });

  test("rate and voice selection apply to subsequent utterances", () => {
    const backend = new FakeBackend();
    const narrator = new NarratorController(cues, backend);

    narrator.setRate(1.25);
    narrator.selectVoice("voice-b");
    narrator.play(0);

    expect(backend.spoken[0]).toEqual({
      cueId: "establish",
      text: "Establish the request path.",
      rate: 1.25,
      voiceId: "voice-b",
    });
    expect(narrator.voices()).toEqual(backend.voices());
  });

  test("stop resets narration so an explicit play replays from zero", () => {
    const backend = new FakeBackend();
    const narrator = new NarratorController(cues, backend);

    narrator.play(0);
    narrator.stop();
    expect(narrator.snapshot()).toMatchObject({
      timeMs: 0,
      activeCueId: null,
      status: "stopped",
    });

    narrator.play();
    expect(backend.spoken.map(({ cueId }) => cueId)).toEqual([
      "establish",
      "establish",
    ]);
  });

  test("rejects duplicate cue ids and invalid rates", () => {
    const backend = new FakeBackend();

    expect(
      () =>
        new NarratorController(
          [
            { id: "same", atMs: 0, text: "First." },
            { id: "same", atMs: 1, text: "Second." },
          ],
          backend,
        ),
    ).toThrow(/duplicate narration cue id/i);
    const narrator = new NarratorController(cues, backend);
    expect(() => narrator.setRate(0)).toThrow(RangeError);
  });
});

describe("browser speech synthesis backend", () => {
  test("returns null when speech synthesis is unavailable", () => {
    expect(createBrowserSpeechSynthesisBackend(null)).toBeNull();
  });

  test("speaks with the selected browser voice through injected platform APIs", () => {
    const spoken: FakeUtterance[] = [];
    const browserVoice = {
      voiceURI: "browser-voice",
      name: "Browser Voice",
      lang: "en-US",
      default: true,
    };
    class FakeUtterance {
      rate = 1;
      voice: typeof browserVoice | null = null;

      constructor(readonly text: string) {}
    }
    const backend = createBrowserSpeechSynthesisBackend({
      synthesis: {
        getVoices: () => [browserVoice],
        speak: (utterance) => {
          spoken.push(utterance as FakeUtterance);
        },
        pause: () => undefined,
        resume: () => undefined,
        cancel: () => undefined,
      },
      Utterance: FakeUtterance,
    });

    expect(backend).not.toBeNull();
    backend?.speak({
      cueId: "intro",
      text: "Audible preview.",
      rate: 1.1,
      voiceId: "browser-voice",
    });

    expect(spoken[0]).toMatchObject({
      text: "Audible preview.",
      rate: 1.1,
      voice: browserVoice,
    });
    expect(backend?.voices()).toEqual([
      {
        id: "browser-voice",
        name: "Browser Voice",
        language: "en-US",
        default: true,
      },
    ]);
  });
});
