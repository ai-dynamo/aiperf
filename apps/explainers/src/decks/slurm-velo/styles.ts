export const CSS = `

.slurm101-page { min-height: 100%; font-size: 16px; line-height: 1.55; }
.slurm101-hero { max-width: 920px; }
.slurm101-rail { display: flex; flex-wrap: wrap; gap: 6px; }
.slurm101-stage { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
.slurm101-details { display: grid; grid-template-columns: minmax(260px, .75fr) minmax(360px, 1.25fr); gap: 14px; align-items: start; }
.slurm101-lede { font-size: 17px; line-height: 1.6; }
.slurm101-points { display: grid; gap: 12px; }
.slurm101-point { display: grid; grid-template-columns: 16px 1fr; gap: 10px; font-size: 16px; line-height: 1.55; }
.slurm101-slide { animation: slurm101-enter 420ms ease-out both; }
.slurm101-live path[marker-end] { stroke-dasharray: 520; stroke-dashoffset: 520; animation: slurm101-draw 1.15s ease-out 240ms forwards; }
.slurm101-live > g:not(.slurm101-motion) > rect { animation: slurm101-node 520ms ease-out both; transform-box: fill-box; transform-origin: center; }
.slurm101-box-pulse { opacity: 0; animation-name: slurm101-box-pulse; animation-timing-function: ease-out; animation-fill-mode: both; }
@keyframes slurm101-enter { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slurm101-node { from { opacity: .3; transform: scale(.985); } to { opacity: 1; transform: scale(1); } }
@keyframes slurm101-draw { to { stroke-dashoffset: 0; } }
@keyframes slurm101-box-pulse { 0% { opacity: 0; } 4% { opacity: .72; } 12% { opacity: .72; } 21%, 100% { opacity: 0; } }
@media (max-width: 900px) {
  .slurm101-details { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .slurm101-slide, .slurm101-live > g:not(.slurm101-motion) > rect, .slurm101-live path[marker-end], .slurm101-box-pulse { animation: none; }
  .slurm101-live path[marker-end] { stroke-dasharray: none; stroke-dashoffset: 0; }
  .slurm101-motion { display: none; }
}
`;
