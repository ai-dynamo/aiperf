export const CSS = `
.deck-rust-atlas-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.deck-rust-atlas-hero { max-width: 920px; }
.deck-rust-atlas-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.deck-rust-atlas-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.deck-rust-atlas-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.deck-rust-atlas-lede { font-size: 17px; line-height: 1.6; }
.deck-rust-atlas-points { display: grid; gap: 12px; }
.deck-rust-atlas-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.deck-rust-atlas-slide { animation: deck-rust-atlas-enter 420ms ease-out both; }
.deck-rust-atlas-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: deck-rust-atlas-draw 1.15s ease-out 240ms forwards; }
.deck-rust-atlas-live > g:not(.deck-rust-atlas-motion) > rect { animation: deck-rust-atlas-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
@keyframes deck-rust-atlas-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes deck-rust-atlas-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes deck-rust-atlas-draw { to { stroke-dashoffset: 0; } }
@media (max-width: 900px) { .deck-rust-atlas-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .deck-rust-atlas-slide, .deck-rust-atlas-live > g:not(.deck-rust-atlas-motion) > rect, .deck-rust-atlas-live path[marker-end] { animation: none; }
  .deck-rust-atlas-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .deck-rust-atlas-motion { display: none; }
}
`;
