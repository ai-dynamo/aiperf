# Spacing Fix R2-C — Product Deck Cramped Panels

**Date:** 2026-07-20  
**Scope:** Seven product explainer `.flow` decks (Domain 3)

## Problem

Scout Round 2 §4: systematic **56–70 px** two-band `sdk.Panel` geometry across architecture/product decks fights resolver slack and clips title/detail bands in the Node verifier.

## Height rules applied

| Authored height | New height | Context |
|-----------------|------------|---------|
| 56–58 | **72** | Two-band panels |
| 60–64 | **72** | Two-band panels |
| 65–70 | **78** | Two-band panels |
| 56 (pipeline stage) | **68** | `sdk.Pipeline` child stages only (`s4-n0`–`s4-n2`) |

Three-band cards: none in scope were below 88 px (`s2-card` already 92).

## Y nudges (overlap after grow)

| Deck | IDs | Adjustment |
|------|-----|------------|
| `flow-sdk-examples.flow` | `s4-b`, `s4-c` | y 150→160, 220→240 |
| `flow-sdk-examples.flow` | `s6-d`, `s6-e` | y 190→200, 260→280 |
| `aiperf-vs-locust.flow` | `s1-t1`, `s1-t2`, `s1-t3` | y 150→158 (clear taller `s1-locust`) |

## Changes by deck

### `segment-pools.flow` (18 nodes)

| IDs | Before → After |
|-----|----------------|
| `composer`, `freeze`, `hash-recipe`, `h0`–`h3`, `h23`, `in-memory-store`, `literals`, `messages`, `overrides`, `raw`, `token`, `wires` | 70→**78** |
| `body-plan`, `dataset`, `materializer` | 60→**72** |

### `rust-architecture.flow` (10 nodes)

| IDs | Before → After |
|-----|----------------|
| `registry`, `loaders`, `samplers`, `endpoints`, `transports`, `coord`, `w0`–`w2` | 70→**78** |
| `mock` | 58→**72** |

### `rust-architecture-atlas.flow` (68 nodes)

All two-band hub/satellite panels: **70→78** (52 nodes) or **60→72** (16 nodes: `real`, `mock`, `dyno`, `ext`, `reg`, `ds`, `ep`, `tr`, `wl`, `ex`).

### `tstar-warmup.flow` (47 nodes)

| Pattern | Count | Before → After |
|---------|-------|----------------|
| Two-band panels | 39 | 70→**78** |
| Compact row panels | 8 | 60→**72** (`s14-*`, `s16-start`, `s20-*`) |

Representative IDs: `s2-ord-a/b`, `s6-n0`–`s6-n3`, `s16-n0`–`s16-n3`, `s19-lo/gate/full`.

### `dynosim.flow` (2 nodes)

| IDs | Before → After |
|-----|----------------|
| `s5-drive`, `s6-drive` | 70→**78** |

### `aiperf-vs-locust.flow` (31 nodes + 3 y-nudges)

| Pattern | Before → After |
|---------|----------------|
| Slide 2 execute/wait row | 70→**78** |
| Slide 3 workers | 60→**72** |
| Slide 8 lifecycle AND row | 60→**72** |
| Comparison cards | 65→**78** |
| `s10-pool` (nested, width-first geometry) | 60→**72** |

### `flow-sdk-examples.flow` (22 nodes + 4 y-nudges)

| IDs | Before → After |
|-----|----------------|
| `s4-a`–`s4-c` | 56→**72** |
| `s4-n0`–`s4-n2` (pipeline) | 56→**68** |
| `s4-src` | 70→**78** |
| `rx3-center`, `rx4-*`, `rx5-src/dst` | 56–58→**72** |
| `rx6-*`, `rx7-*`, `s9-json/csv` | 60–64→**72** |
| `s6-c`–`s6-e` | 56→**72** |

## Unchanged (intentional)

- Single-band nodes (`sdk.Label`, `sdk.Note`, `sdk.Header`, thread chips at 50 px).
- Panels already ≥80 px (tstar domain cards, dynosim comparison row at 80).
- IDs, narration, timelines, slide structure preserved.

## Verification

```bash
npm --prefix apps/explainers run flow-verifier -- --ir-only \
  --deck segment-pools --deck rust-architecture --deck rust-architecture-atlas \
  --deck tstar-warmup --deck dynosim --deck aiperf-vs-locust --deck flow-sdk-examples
```

Spot-check: `segment-pools` hash-recipe slide, `aiperf-vs-locust` slide 2 execute row, `flow-sdk-examples` slide 4 pipeline + stagger column.

## Out of scope

No TypeScript, CSS, resolver, or commit changes per Fixer C charter.
