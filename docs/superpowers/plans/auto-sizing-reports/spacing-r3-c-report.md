# Round-3 Fixer C Report — Avatar Intrinsic Minimums

## Scope completed

- Added `resolveAvatarLayout` in `layout.ts` and wired `core.group` presentation `"avatar"` through `resolvePresentationLayout`.
- Authored width/height remain minimums; resolver expands to a square side of `max(authored.width, authored.height, 40)` where `40 = 8 + 24 + 8` matches catalog icon placement (`inset 8`, glyph up to `24×24`).
- `clip: true` and `overflow: hidden` preserve authored bounds via existing `clipsOverflow` guard.
- No changes to `chrome.ts` or `generic/chrome.ts` — avatar chrome is circle-only (no label band to measure); catalog default `48×48` already exceeds the floor.

## TDD and verification

- Added regression tests: default-size passthrough, undersized grow to 40, rectangular grow to square, hero passthrough, clipped exception.
- Confirmed undersized test failed before implementation (`30×30` stayed identity-sized).
- Passed avatar tests; full file run: **30/31** (`aligns frame chrome text with managed padding` fails pre-existing — out of scope).
- Command: `npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts`

No commit. No `.flow` or `index.css` edits.
