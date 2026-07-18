export const CSS = `
.deck-segment-pools-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.deck-segment-pools-hero { max-width: 920px; }
.deck-segment-pools-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.deck-segment-pools-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.deck-segment-pools-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.deck-segment-pools-lede { font-size: 17px; line-height: 1.6; }
.deck-segment-pools-points { display: grid; gap: 12px; }
.deck-segment-pools-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.deck-segment-pools-slide { animation: deck-segment-pools-enter 420ms ease-out both; }
.deck-segment-pools-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: deck-segment-pools-draw 1.15s ease-out 240ms forwards; }
.deck-segment-pools-live > g:not(.deck-segment-pools-motion) > rect { animation: deck-segment-pools-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
@keyframes deck-segment-pools-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes deck-segment-pools-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes deck-segment-pools-draw { to { stroke-dashoffset: 0; } }
@media (max-width: 900px) { .deck-segment-pools-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .deck-segment-pools-slide, .deck-segment-pools-live > g:not(.deck-segment-pools-motion) > rect, .deck-segment-pools-live path[marker-end] { animation: none; }
  .deck-segment-pools-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .deck-segment-pools-motion { display: none; }
}
`;
