# Spacing Round 2 Fix E — Footer CSS Trim

**Date:** 2026-07-20  
**Scope:** `apps/explainers/src/index.css` only (Fixer E)  
**Deck checked:** `/#/cellular-internals` slide 1 (fixture + baseline from Fix B)

## Changes

| Selector | Before | After |
|----------|--------|-------|
| `.ex-stage-footer` gap | `clamp(8px, 1.2vh, 12px)` | `clamp(6px, 1vh, 10px)` |
| `.ex-stage-footer` pad-bottom | `max(64px, 6vh)` | `max(60px, 5.5vh)` |
| `.ex-subtitles` padding | `7px 10px` | `6px 10px` |
| `.ex-subtitles__label` margin-bottom | `6px` | `5px` |
| `@media (max-height: 760px)` footer gap / pad-bottom | `6px` / `64px` | `5px` / `62px` |
| `@media (max-height: 760px)` subtitles block | `6px` | `5px` |
| `@media (max-width: 720px)` footer gap / pad-bottom | `7px` / `64px` | `6px` / `60px` |

**Unchanged:** hero top padding, lede visibility/`text-wrap: pretty`, footer grid structure.

## Verification (Playwright + built CSS)

Live `/#/cellular-internals` preview blocked by unrelated `sdk-diagram-catalog.flow` compile errors at registry load. Clearance measured via `artifacts/clearance-fixture.html` wired to `dist/assets/index-*.css` (cellular slide 1 copy + subtitles + bottom nav).

| Viewport | Lede→nav gap | Footer pad-bottom | Hero pad-top | Lede clipped | Pass (≥8px) |
|----------|--------------|-------------------|--------------|--------------|-------------|
| 1440×900 | **8.0px** | 60px | 49.5px | no | yes |
| 1280×720 | **8.0px** | 62px | 44px | no | yes |
| 390×844 | **8.0px** | 60px | 58px | no | yes |

Fix B baseline gaps were 12 / 10 / 12 px at the same viewports with 64px footer floor. This pass trims **4px** pad on tall + mobile, **2px** on short-height (720), plus ~2–3px subtitle/gap chrome — reclaiming ~6–8px vertical stage room for the diagram grid row without dropping below the 8px lede/nav floor.

## Notes

- Short-height breakpoint keeps a **62px** floor (not 60px) so 720p laptops retain minimum clearance.
- When deck registry compile is restored, re-run live preview check on `/#/cellular-internals` to confirm fixture parity.
