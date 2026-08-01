# Narrated autoplay

Decks that speak their narration and advance themselves when each step finishes.
Ported from `apps/explainers/src/core`.

Narration is synthesized live in the browser through the Web Speech API. There
are no audio assets and no TTS build step: a step's narration is a plain string,
and everything temporal is derived at runtime.

## Layout

| File | Role |
| --- | --- |
| `narration.ts` | The speech engine. No React, no dependencies. Verbatim copy from explainers. |
| `useNarratedPlayback.ts` | Speaks the current step, reports word position, calls `onAdvance` at the end. |
| `useNarratedDeck.ts` | Playback state machine: start gate, play/pause, `restartKey`, persisted settings, keyboard nav. |
| `useSpeechVoices.ts` | Enumerates English voices as the engine loads them. |

UI lives in `src/shell/`: `StartGate`, `PlaybackControls`, `Subtitles`, `VoicePicker`.

## Using it

```tsx
const narrated = useNarratedDeck({ narrations, storagePrefix: `deck:${id}` });

<PresentationShell
  slides={slides}
  slideIndex={narrated.index}
  onSlideIndexChange={narrated.goTo}
  narrated={narrated}
  title={title}
>
  <Slide key={`${slide.id}-${narrated.restartKey}`} slide={slide} />
</PresentationShell>
```

The `narrated` prop is optional. Without it `PresentationShell` stays a manual
click-through deck.

For a bespoke deck already using `useStepSimulator`, use `useNarratedDeck`
instead: it owns the index itself and advances on narration completion rather
than on a fixed `autoPlayMs` timer.

## Constraints worth knowing before you change this

**Browsers block speech until a user gesture.** `unlockSpeech()` must be called
from inside a click handler. That is the only reason `StartGate` exists — remove
it and the first step will silently fail to speak.

**`onboundary` is unreliable.** Many engines never fire it. `speakNarration`
therefore always runs estimated word timers alongside it, and lets real boundary
events override the estimate when they arrive. Deleting the estimated path
leaves subtitles stuck on the first word on those engines.

**`onend` is not guaranteed either.** A hard `setTimeout` backs it up, so a
wedged engine cannot stall a deck permanently. `onerror` deliberately ignores
`interrupted` and `canceled` so that a user-initiated pause does not schedule a
spurious advance.

**There is no seek within an utterance.** Pause is implemented as cancel, and
resume restarts the current step's narration from its beginning.

**`restartKey` is the sync primitive.** It bumps on step change, revisit, and
play. Pass it as a React `key` (or restart dependency) to any animation that
should restart with the voice. Voice and visuals are not cue-locked to each
other; both are driven off the same `(playing, restartKey, speed)` triple.

**Voice quality varies wildly** by OS and browser, which is why `VoicePicker`
exists and why voices are filtered to `en`.

## If deterministic timing is ever needed

Pre-render narration to audio files and swap `speakNarration`'s internals for an
`HTMLAudioElement` driven by `timeupdate`. Its signature — `onWord`,
`onComplete`, returns a cancel function — is already the right seam, and nothing
above it would change.
