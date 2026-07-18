export const CSS = `
.deck-cellular-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.deck-cellular-hero { max-width: 920px; }
.deck-cellular-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.deck-cellular-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.deck-cellular-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.deck-cellular-lede { font-size: 17px; line-height: 1.6; }
.deck-cellular-points { display: grid; gap: 12px; }
.deck-cellular-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.deck-cellular-slide { animation: deck-cellular-enter 420ms ease-out both; }
.deck-cellular-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: deck-cellular-draw 1.15s ease-out 240ms forwards; }
.deck-cellular-live > g:not(.deck-cellular-motion) > rect { animation: deck-cellular-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
@keyframes deck-cellular-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes deck-cellular-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes deck-cellular-draw { to { stroke-dashoffset: 0; } }
@media (max-width: 900px) { .deck-cellular-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .deck-cellular-slide, .deck-cellular-live > g:not(.deck-cellular-motion) > rect, .deck-cellular-live path[marker-end] { animation: none; }
  .deck-cellular-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .deck-cellular-motion { display: none; }
}
`;
