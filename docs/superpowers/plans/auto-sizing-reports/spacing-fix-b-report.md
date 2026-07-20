# Spacing Fix B — Stage/Footer CSS Rebalance

**Date:** 2026-07-20  
**Scope:** `apps/explainers/src/index.css` only (Fixer B)  
**Deck checked:** `/#/cellular-internals` slide 1

## Changes

| Selector | Before | After | Δ (typical) |
|----------|--------|-------|-------------|
| `.ex-stage-hero` top padding | `clamp(54px, 7vh, 76px)` | `clamp(48px, 5.5vh, 66px)` | −6–10px |
| `.ex-shell--present .ex-stage-hero` top | `clamp(34px, 4vh, 48px)` | `clamp(30px, 3.2vh, 42px)` | −4–6px |
| `@media (max-height: 760px)` hero top | `50px` / present `30px` | `44px` / present `26px` | −6px / −4px |
| `@media (max-width: 720px)` hero top | `72px` | `58px` | −14px |
| `.ex-shell--chrome-hidden` hero top | `26px` | `22px` | −4px |
| `.ex-subtitles` padding | `9px 10px` | `7px 10px` | −4px block |
| `.ex-subtitles__label` margin-bottom | `8px` | `6px` | −2px |
| `.ex-subtitles__text` font-size | `clamp(16px, 1.7vw, 24px)` | `clamp(15px, 1.55vw, 22px)` | −1–2px |
| Mobile subtitles text | `clamp(15px, 4.5vw, 19px)` | `clamp(14px, 4.2vw, 18px)` | −1px |
| Short-height subtitles block | `7px` | `6px` | −1px |

**Unchanged (per constraints):** `.ex-stage-footer` bottom padding remains `max(64px, 6vh)` (64px floor on short/mobile). Lede `text-wrap: pretty` and footer grid structure untouched.

## Verification (Playwright, `vite preview` + `@playwright/test`)

Route: `http://127.0.0.1:4174/#/cellular-internals` (default shell, slide 1).

| Viewport | Lede→nav gap | Footer pad-bottom | Hero pad-top | SVG height | Lede clipped |
|----------|--------------|-------------------|--------------|------------|--------------|
| 1440×900 | **12px** | 64px | 49.5px | 633px | no |
| 1280×720 | **10px** | 64px | 44px | 513px | no |
| 390×844 | **12px** | 64px | 58px | 547px | no |

All viewports meet the ≥10px lede/bottom-nav clearance target; footer retains 64px nav clearance; lede text fully visible (`scrollHeight === clientHeight`).

## Notes

- Hero top padding still clears progress chrome (`top: 54px` + 9px bar); minimum desktop top is 48px with diagram centered below chrome gradient.
- Present-mode and chrome-hidden variants scaled proportionally.
- Full `npm run build` blocked by unrelated TS errors in layout/resolver files; verification used `npx vite build` (CSS bundled successfully).
