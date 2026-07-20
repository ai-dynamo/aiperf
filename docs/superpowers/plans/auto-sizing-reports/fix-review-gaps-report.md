# Auto-Sizing Final Review Gap Fixes

## Status

All four Important findings from `final-review.md` are fixed.

## Changes

- `resolvePanelLayout` now sizes `props.text` as the title fallback used by
  semantic chrome.
- `core.header` now has an intrinsic resolver for `title` and `caption`.
- `core.text` now sizes from `node.text`, authored `style.fontSize`, and shared
  scale-aware text metrics.
- Chip, panel/note, header, and text resolvers preserve authored geometry when
  `overflow: "hidden"` or `clip: true` is set.
- Layout regressions cover SDK note-shaped props, header chrome, text labels,
  and clipped chips.
- The rail renderer assertion now accounts for the rail root's world-space
  offset, matching canonical scene resolution.

## TDD and verification

The new layout tests were run before implementation and failed in all four
targeted cases. After implementation, the required command passed:

```text
npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts src/core/diagram/text-metrics.test.ts src/core/diagram/SceneRenderer.sdk-primitives.test.tsx

Test Files  3 passed (3)
Tests       31 passed (31)
```

IDE diagnostics reported no errors in the edited layout source or tests.
