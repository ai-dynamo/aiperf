# Round 5 — Root residual closeout

**Date:** 2026-07-20  
**Scope:** Items still listed open after Round 4 final (IR residuals, constant drift, temp probes).

## Status

| Item | Before | After |
|------|--------|-------|
| Full-repo IR (`flow-verifier:ir`) | Reported residual routing/motion noise | **0 error(s), 0 warn(s)** (already clear; reconfirmed) |
| Per-deck overlaps / escapes | 0 (R4) | **0** |
| Duplicate `INSET`/`TITLE_HEIGHT` in `generic/chrome.ts` | Local copies | **Import from `text-metrics.ts`** |
| Temp probes (`_probe-*.ts`, `verify_curves_tmp.mjs`) | Present | **Deleted** |
| Vitest (explainers) | — | **327/327** |
| `assert:sdk-authoring --strict` | — | **OK** |

## Code change

`apps/explainers/src/flow/sdk/generic/chrome.ts` — replace local `INSET`/`TITLE_HEIGHT` with imports from `../../../core/diagram/text-metrics.js`.

No deck or resolver behavior changes in this round.
