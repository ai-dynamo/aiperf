export const CSS = `
.deck-dynosim-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.deck-dynosim-hero { max-width: 920px; }
.deck-dynosim-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.deck-dynosim-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.deck-dynosim-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.deck-dynosim-lede { font-size: 17px; line-height: 1.6; }
.deck-dynosim-points { display: grid; gap: 12px; }
.deck-dynosim-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.deck-dynosim-slide { animation: deck-dynosim-enter 420ms ease-out both; }
.deck-dynosim-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: deck-dynosim-draw 1.15s ease-out 240ms forwards; }
.deck-dynosim-live > g:not(.deck-dynosim-motion) > rect { animation: deck-dynosim-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
.deck-dynosim-pulse { opacity: 0; animation: deck-dynosim-pulse 2.2s ease-out both infinite; }
@keyframes deck-dynosim-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes deck-dynosim-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes deck-dynosim-draw { to { stroke-dashoffset: 0; } }
@keyframes deck-dynosim-pulse { 0% { opacity: 0; } 4% { opacity: .72; } 12% { opacity: .72; } 21%, 100% { opacity: 0; } }
@media (max-width: 900px) { .deck-dynosim-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .deck-dynosim-slide, .deck-dynosim-live > g:not(.deck-dynosim-motion) > rect, .deck-dynosim-live path[marker-end], .deck-dynosim-pulse { animation: none; }
  .deck-dynosim-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .deck-dynosim-motion { display: none; }
}
`;
