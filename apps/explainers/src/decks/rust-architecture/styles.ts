export const CSS = `

.rust-arch-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.rust-arch-hero { max-width: 920px; }
.rust-arch-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.rust-arch-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.rust-arch-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.rust-arch-lede { font-size: 17px; line-height: 1.6; }
.rust-arch-points { display: grid; gap: 12px; }
.rust-arch-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.rust-arch-slide { animation: rust-arch-enter 420ms ease-out both; }
.rust-arch-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: rust-arch-draw 1.15s ease-out 240ms forwards; }
.rust-arch-live > g:not(.rust-arch-motion) > rect { animation: rust-arch-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
.rust-arch-box-pulse { opacity: 0; animation: rust-arch-box-pulse 2.2s ease-out both infinite; }
@keyframes rust-arch-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes rust-arch-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes rust-arch-draw { to { stroke-dashoffset: 0; } }
@keyframes rust-arch-box-pulse { 0% { opacity: 0; } 4% { opacity: .72; } 12% { opacity: .72; } 21%, 100% { opacity: 0; } }
@media (max-width: 900px) { .rust-arch-details { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
  .rust-arch-slide, .rust-arch-live > g:not(.rust-arch-motion) > rect, .rust-arch-live path[marker-end], .rust-arch-box-pulse { animation: none; }
  .rust-arch-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .rust-arch-motion { display: none; }
}
`;
