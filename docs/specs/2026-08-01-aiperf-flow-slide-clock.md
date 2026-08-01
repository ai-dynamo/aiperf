<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Slide clock: continuous virtual time for `apps/aiperf-flow`

## Why

Every subsystem these decks explain has time as its subject — the idle-gap warp, the
sweep line, phase windows, ITL between tokens. The app draws time as a *coordinate*:
`sweep`, `timeline`, and `slices` each flatten a whole run onto a static axis. The
reader is handed a finished picture and asked to reconstruct the dynamics.

That costs three specific claims the current charts cannot make:

- **The sweep-line identity is asserted, not shown.** The Gantt sits above the curve
  and the caption says one produces the other. Nothing connects a bar's edge to the
  step it causes.
- **The warp's payoff is a number, not an experience.** "122 seconds replay in 35" is
  read, not felt, because both blocks are drawn complete and simultaneous.
- **Idle and hung are indistinguishable.** Both are flat. In these systems that
  distinction matters enormously and no static chart carries it.

The app already has a step-based player (`src/interactive/useFlowPlayer.ts` over
`src/state/useStepSimulator.ts`), but it walks a discrete `FlowStep[]` on a fixed
`autoPlayMs` timer. There is no `t`, and nothing shares a time base across a slide.

This spec introduces one: a continuous, narration-driven virtual clock, deliberately
mirroring the seam the Rust runtime is organized around (`aiperf_runtime::clock::Clock`,
`RealClock`/`SimClock`). The explainer for a clock-driven system becomes clock-driven.

## Approach (decided)

Four decisions, taken before design:

1. **Continuous slide clock**, not an extended step player and not standalone
   simulators. A slide exposes a real `t`; charts become functions of `t`.
2. **Narration is the master clock**, with authored cue points and a linear
   auto-fit default.
3. **Scope is the clock plus a retrofit** of the three existing time-axis charts
   (`sweep`, `timeline`, `slices`). No new node types.
4. **`t` reaches charts through an external store**, never through `node.data`.

Decision 4 is a hard constraint, not a preference. `src/deck/Slide.tsx` already
memoizes its node and edge arrays on the reveal set, with a comment recording why:
fresh node identities on every render made React Flow re-render its edges and restart
their CSS dash animation. Threading `t` through `node.data` would recreate every node
object 60 times a second — the same pathology at 60× the rate.

## Module: `src/clock/`

A new top-level module, parallel to `src/nodes/`, `src/edges/`, and `src/interactive/`
rather than inside any of them: it is a time source, not a renderer.

```
src/clock/
  clockStore.ts          framework-free store; no React, no DOM
  cues.ts                pure cue resolution
  SlideClockProvider.tsx owns the single rAF loop
  useSlideClock.ts       useSyncExternalStore subscription
  index.ts               public surface
```

### `clockStore.ts`

```ts
export type ClockSnapshot = { t: number; playing: boolean };

export interface ClockStore {
  subscribe: (onChange: () => void) => () => void;
  getSnapshot: () => ClockSnapshot;
  /** Set virtual time directly. The provider's only write path. */
  setTime: (t: number, playing: boolean) => void;
}

export function createClockStore(initial: ClockSnapshot): ClockStore;
```

No React import. `getSnapshot` returns a stable object identity while the value is
unchanged, which `useSyncExternalStore` requires to avoid an infinite render loop.

### `cues.ts`

```ts
/** "By word `atWord` of the narration, virtual time should be `t`." */
export type Cue = { atWord: number; t: number };

export type ClockSpec = {
  /** Virtual-time bounds of the slide's data, e.g. `[0, 122]`. */
  span: [number, number];
  /** Ascending by `atWord`. Omitted → linear fit across `span`. */
  cues?: Cue[];
};

/**
 * Map continuous word progress onto virtual time. Piecewise-linear between cues,
 * clamped outside the first and last.
 */
export function resolveTime(
  wordProgress: number,
  wordCount: number,
  spec: ClockSpec,
): number;
```

Cues are anchored in **word positions, not milliseconds**. Narration duration varies
with voice, browser, and the user's speed setting; word count does not. A cue table
authored once holds for every playback.

The warp slide is the motivating case: 87 of its 122 virtual seconds are dead air. A
linear fit spends ~70% of the slide creeping across a region where nothing happens.
Cues cross it in the handful of words the narration actually spends on it.

### `SlideClockProvider.tsx`

Owns the one `requestAnimationFrame` loop and is the sole writer to the store.

**Progress estimation.** `useNarratedDeck` exposes `activeWordIndex`. That signal is
discrete — it advances one word at a time, which would make a playhead stutter. The
provider smooths it:

```
wordProgress = activeWordIndex + clamp(msSinceWordChange / avgWordMs, 0, 1)
```

The integer part snaps back to truth on every word change, so error cannot
accumulate; the fractional part interpolates between them. `avgWordMs` is derived
from the step's own elapsed time and word count, so it adapts to the actual voice
rather than assuming a rate.

Word count must come from `splitWords(narration).length` in
`src/audio/narration.ts` — the same function `speakNarration` indexes against, so
`activeWordIndex` and `wordCount` cannot disagree about what a word is.

**`activeWordIndex` is not monotonic, and the clock must be.** `speakNarration`
drives word events from *two* sources at once: `driveEstimatedWords()` schedules a
timer per word at the estimated rate, and `utterance.onboundary` reports the real
position. Both call the same `onWord`. When actual speech runs slower than the
estimate, a boundary event reports a *lower* index than an already-fired timer and
`activeWordIndex` jumps backwards. Mapped straight through, virtual time would jump
backwards with it and a self-drawing curve would visibly un-draw.

The provider therefore clamps `t` monotonically within a step: `t = max(t, resolved)`,
reset on `restartKey`. Backwards narration corrections are absorbed as a pause in the
playhead rather than a rewind.

**Silent mode needs no separate path.** `speakNarration` calls `driveEstimatedWords()`
in its non-speech branch too, so `activeWordIndex` advances on estimated timers
whether or not speech is enabled, and whether or not the browser supports
`onboundary`. One progress source covers all three cases. `estimateNarrationMs(text,
speed)` is available for a duration-based sanity bound if one is ever needed;
`formatStepDuration` is *not* usable for this — it returns a display string (`"12s"`).

**Reduced motion.** When `useReducedMotion()` is true the provider pins `t` to
`span[1]` and never starts the loop. Charts then render exactly what they render
today — fully drawn. There is no separate static rendering path to build or keep in
sync, because "static" is definitionally "the clock at the end of its span."

**Pause and scrub.** `t` freezes when narration is paused. Revisiting a slide resets
via the existing `restartKey`.

### `useSlideClock.ts`

```ts
/** Current virtual time, or `undefined` when the slide authored no clock. */
export function useSlideClock(): number | undefined;
```

`useSyncExternalStore` against the context's store. Returns `undefined` when no
provider is present, which is the signal a chart uses to render its static form.

The hook exposes `t` only, deliberately, even though `ClockSnapshot` also carries
`playing`. No chart in this scope changes its rendering based on whether the clock is
running — a paused clock is just a `t` that stops advancing. `playing` exists for the
provider's own bookkeeping and for future playback chrome; exposing it now would
invite charts to branch on it and lose the property that a chart is a pure function of
`t`, which is what makes rendering at a fixed `t` a complete test.

## Integration points

### `src/deck/types.ts`

`SlideDefinition` gains one optional field:

```ts
/** Opt in to the slide clock. Absent → no provider, no loop, charts fully drawn. */
clock?: ClockSpec;
```

### `src/deck/Slide.tsx`

Wraps `SlideCanvas` in `SlideClockProvider` when `slide.clock` is present, passing
the narration handle through. The node and edge arrays and their existing
memoization are untouched.

### Chart behavior at `t`

| Chart | With a clock | Without |
|---|---|---|
| `sweep` | Playhead at `x(t)`. Step path clipped to `t` so the curve draws itself. Event ticks past `t` hidden. Bars active at `t` filled, not-yet-started bars outlined. | Unchanged: full curve, all ticks, all bars filled. |
| `timeline` | Two playheads over one progress — the raw head crawls through dead air while the warped head jumps. Bars light while active. | Unchanged: both blocks fully drawn. |
| `slices` | Buckets fill as the playhead crosses them; the active bucket is highlighted; the clipped trailing region resolves at the end. | Unchanged: all buckets drawn, trailing slice starred. |

`intervals`, `blocks`, and `ragged` stay static. `intervals` is a straightforward
follow-on (light rows as `t` passes, badge on completion) and is deliberately not in
this scope.

Geometry additions live in the existing `*Layout.ts` modules beside the coordinates
they extend, keeping the components free of arithmetic and the geometry unit-testable.

`sweepMath.ts` needs no change. Its `stepPathD(pts, x, y, tMin, tMax)` already
produces a correctly clipped curve when given
`stepPathD(points.filter((p) => p.t <= t), x, y, 0, t)`: the final segment holds the
last value out to `t`, which is exactly a curve drawn up to now. Verified against the
existing implementation rather than assumed.

### Scope boundary

This spec does **not** cover: new node types (`stage`, `firing`, `store`, `cells`),
per-token pulse rendering, interactive simulator pages, or migrating
`src/interactive/`'s step player onto the clock. Those are separate designs that
depend on this seam existing first.

## Back-compat

The `clock` field is optional and absent everywhere on introduction. All 52 existing
slides across the four registered decks (`async-dataflow-engine` 13,
`python-graph-workload` 27, `native-diagram-vocabulary` 8, `metrics-plane` 4) render
byte-identically until one opts in.
The regression surface is the `Slide.tsx` provider wrapper and nothing else.

## Testing

The clock is injectable, which is the whole point — the same reason `SimClock` exists
in the runtime. A scene at a given `t` is exactly assertable, with no timers, no
`requestAnimationFrame`, and no speech synthesis in the test path.

- **`cues.ts`** — table-driven: linear fallback with no cues, interpolation between
  cues, clamping before the first and after the last, a cue table that crosses dead
  air faster than linear.
- **`clockStore.ts`** — subscribe/notify/unsubscribe, and snapshot identity stability
  while the value is unchanged (the `useSyncExternalStore` contract).
- **Charts** — render at fixed `t` values and assert the DOM. At minimum, per chart:
  nothing drawn past the playhead, everything drawn at `t = span[1]`, and the
  no-clock render identical to the current output.
- **Node identity stability** — one integration test asserting the React Flow node
  array keeps its reference across a `t` change. This is the regression the whole
  delivery decision exists to prevent, so it gets an explicit test rather than a
  comment.
- **Reduced motion** — with `useReducedMotion()` true, a clocked slide renders
  identically to the same slide with no clock authored. jsdom provides no
  `window.matchMedia`, so `useReducedMotion()` returns null under test and this case
  needs an explicit `vi.mock("motion/react")` (or a `matchMedia` polyfill in
  `vitest.setup.ts`) rather than relying on the environment.
- **Monotonicity** — feeding the provider a decreasing `activeWordIndex` must not
  decrease `t`. This is a real code path, not a hypothetical: see the two competing
  word-event sources above.

`requestAnimationFrame` is available in jsdom and `useSyncExternalStore` works there
(React 19.2.7), both confirmed by probe. Hook tests follow the existing
`vi.useFakeTimers()` + `renderHook` + `act` convention from `useReveal.test.ts`.

Browser verification uses the existing Playwright suite (`npm run test:browser`,
`e2e/decks.spec.ts`), which asserts no clipped nodes, correct framing after the reveal
cascade, and no console errors — the checks jsdom cannot make.

## Risks

- **60 fps leaf re-renders.** Three SVG charts per slide is well within budget. If a
  future slide is heavier, `useSlideClock` can take a selector so a chart re-renders
  only when its quantized value changes (`slices` needs only the active bucket
  index). Not built until something needs it.
- **`onboundary` support varies by browser.** The silent-mode estimator is the
  fallback and already exists; a browser without boundary events degrades to
  duration-estimated progress rather than to a frozen playhead.
- **Cue tables drift from narration edits.** A cue past the end of a narration clamps
  rather than breaking. Deck-structure tests can later assert `atWord <= wordCount`;
  noted, not built here.

## Source anchors

Existing code this design builds on or constrains:

- `apps/aiperf-flow/src/deck/Slide.tsx` — reveal-driven `fitView`, and the node/edge
  memoization whose comment motivates the external-store delivery decision.
- `apps/aiperf-flow/src/deck/types.ts` — `SlideDefinition`, gaining `clock`.
- `apps/aiperf-flow/src/audio/useNarratedDeck.ts` — `activeWordIndex`, `restartKey`,
  `narrationEnabled`, the progress source.
- `apps/aiperf-flow/src/audio/narration.ts` — `formatStepDuration`, the silent-mode
  fallback.
- `apps/aiperf-flow/src/nodes/{Sweep,Timeline,Slices}.tsx` and their
  `{sweep,timeline,slices}Layout.ts` — the retrofit targets.
- `apps/aiperf-flow/src/interactive/RequestParticle.tsx` — the existing
  `useReducedMotion` precedent.
- `rust/runtime/src/clock/` — the `Clock`/`RealClock`/`SimClock` seam this mirrors.
