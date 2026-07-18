/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

export const CSS = `
.deck-algorithm-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.deck-algorithm-hero { max-width: 920px; }
.deck-algorithm-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.deck-algorithm-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.deck-algorithm-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.deck-algorithm-lede { font-size: 17px; line-height: 1.6; }
.deck-algorithm-points { display: grid; gap: 12px; }
.deck-algorithm-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.deck-algorithm-slide { animation: deck-algorithm-enter 420ms ease-out both; }
.deck-algorithm-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: deck-algorithm-draw 1.15s ease-out 240ms forwards; }
.deck-algorithm-live > g:not(.deck-algorithm-motion) > rect { animation: deck-algorithm-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
@keyframes deck-algorithm-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes deck-algorithm-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes deck-algorithm-draw { to { stroke-dashoffset: 0; } }
@media (max-width: 900px) { .deck-algorithm-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .deck-algorithm-slide, .deck-algorithm-live > g:not(.deck-algorithm-motion) > rect, .deck-algorithm-live path[marker-end] { animation: none; }
  .deck-algorithm-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .deck-algorithm-motion { display: none; }
}
`;
