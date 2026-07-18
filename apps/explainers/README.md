# AIPerf Explainers

Unified narrated slideshow SPA for AIPerf architecture topics.

## Develop

```bash
cd apps/explainers
npm install
npm run dev
```

Routes (hash router):

- `/#/` — hub
- `/#/rust-architecture`
- `/#/slurm-velo`
- `/#/dynosim`

## Build

```bash
npm run build
npm run preview
```

## Deploy to GitHub Pages

```bash
npm run deploy:pages
```

Legacy subfolder URLs redirect into hash routes:

- `rust-architecture-explainer/` → `#/rust-architecture`
- `slurm-explainer/` → `#/slurm-velo`
