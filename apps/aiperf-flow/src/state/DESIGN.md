# `src/state`

`useStepSimulator<T>` is a small, in-memory hook for step-through diagram
walkthroughs: Play/Pause/Step/Reset over a fixed `steps: readonly T[]` array,
with `index`/`current`/`isFirst`/`isLast`/`isPlaying` derived state and
`next`/`back`/`reset`/`togglePlay` actions. `next`/`back` clamp at the array
bounds instead of throwing, and autoplay (`togglePlay` + `opts.autoPlayMs`,
default 1000ms) stops itself on reaching the last step rather than looping.

This mirrors the ergonomics of Cursor Canvas's `useCanvasState` (see
`~/.cursor/skills-cursor/canvas/sdk/hooks.d.ts`) but intentionally drops its
disk-backed persistence to a `.canvas.data.json` sidecar file. That feature
exists because a Cursor canvas is an IDE-embedded artifact that must survive
editor reloads and restarts across sessions. `aiperf-flow` decks are
presentation decks rendered for the duration of a slide deck session — plain
`React.useState`-equivalent, process-lifetime state is the right scope here,
and adding file persistence would be scope creep with no host integration to
back it.
