export const CSS = `
.deck-velo-deep-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.deck-velo-deep-hero { max-width: 920px; }
.deck-velo-deep-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.deck-velo-deep-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.deck-velo-deep-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.deck-velo-deep-lede { font-size: 17px; line-height: 1.6; }
.deck-velo-deep-points { display: grid; gap: 12px; }
.deck-velo-deep-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.deck-velo-deep-slide { animation: deck-velo-deep-enter 420ms ease-out both; }
.deck-velo-deep-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: deck-velo-deep-draw 1.15s ease-out 240ms forwards; }
.deck-velo-deep-live > g:not(.deck-velo-deep-motion) > rect { animation: deck-velo-deep-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
@keyframes deck-velo-deep-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes deck-velo-deep-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes deck-velo-deep-draw { to { stroke-dashoffset: 0; } }
@media (max-width: 900px) { .deck-velo-deep-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .deck-velo-deep-slide, .deck-velo-deep-live > g:not(.deck-velo-deep-motion) > rect, .deck-velo-deep-live path[marker-end] { animation: none; }
  .deck-velo-deep-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .deck-velo-deep-motion { display: none; }
}
`;
