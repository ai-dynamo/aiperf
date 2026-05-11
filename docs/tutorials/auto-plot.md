<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0 -->

# Auto-Plot: Generate Plots Automatically After `aiperf profile`

`aiperf profile` can invoke `aiperf plot` against the artifact directory the moment a benchmark finishes, so you do not have to remember to run a second command. This tutorial covers how to turn it on, when it is on by default, how to make plotting failures fatal, and how to disable it.

For the full plot catalogue, theming, dashboard mode, and `~/.aiperf/plot_config.yaml` customization, see [Visualization and Plotting with AIPerf](plot.md). Auto-plot is a thin post-run hook around the same engine — this page only covers the integration.

## What auto-plot does

When `--auto-plot` resolves to true, `aiperf profile` registers a post-run callback. After the benchmark exits with code 0, the callback calls `aiperf plot` with `paths=[<artifact_dir>]` (PNG mode, default theme, default `~/.aiperf/plot_config.yaml`). Output lands under `<artifact_dir>/plots/`. The callback runs in-process; no extra subprocess.

Auto-plot is fired once per successful run:

- **Single run:** after the orchestrator returns 0.
- **Multi-run / sweep / search recipe:** once at the end, against the top-level `<artifact_dir>` (which contains the per-variation subdirectories). The plot command auto-detects multi-run mode from the directory layout — no extra wiring needed.

Single-run: the callback only fires on exit 0. Multi-run / sweep: the callback fires when at least one trial succeeded, even on a partial-failure path — the surviving per-run JSONL/CSV/JSON are on disk and downstream hooks can still consume them. Only a run where zero trials succeeded skips the callback entirely.

## Quick start

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --url http://vllm.internal:8000 \
    --concurrency 16 \
    --request-count 200 \
    --auto-plot
```

After the run, the artifact tree looks like:

```text
artifacts/
└── Llama-3.1-8B-Instruct-concurrency16/
    ├── profile_export.jsonl
    ├── profile_export_aiperf.json
    ├── profile_export_aiperf.csv
    └── plots/
        ├── ttft_over_time.png
        ├── ttft_timeline.png
        ├── timeslices_ttft.png
        ├── timeslices_itl.png
        ├── gpu_utilization_and_throughput_over_time.png
        └── aiperf_plot.log
```

The exact set of PNGs depends on the data captured (single-run vs. multi-run, GPU telemetry, timeslice data); see [the plot catalogue](plot.md#visualization-modes) for the full list.

## When auto-plot is on by default

The resolution rule (implemented in [`resolve_auto_plot`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/config/flags/_converter_optionals.py)):

| User passed `--auto-plot` / `--no-auto-plot`? | Active search recipe `auto_plot_default`? | Result |
|---|---|---|
| `--auto-plot` | (any) | **on** |
| `--no-auto-plot` | (any) | **off** |
| (nothing) | recipe sets `auto_plot_default = True` | **on** |
| (nothing) | recipe leaves it unset / sets `False` | **off** |
| (nothing) | no recipe at all | **off** |

In short: explicit CLI flag wins; otherwise, the active recipe decides; otherwise, off. Plain `aiperf profile` (no recipe, no flag) does **not** auto-plot — you opt in.

## Per-recipe defaults

Built-in search recipes that ship with `auto_plot_default = True` (their output is a curve worth visualizing immediately):

- `concurrency-ramp`
- `prefill-ttft-curve`
- `decode-itl-curve`

All other built-in recipes (`max-throughput-ttft-sla`, `max-throughput-itl-sla`, `max-concurrency-under-sla`, `max-goodput-under-slo`, `pareto-sweep`) leave it unset, which falls back to **off**. Those recipes search for an optimum rather than producing a curve, so the default plots are less useful.

External plugin recipes that don't define the attribute also fall back to off — the read site uses `getattr(recipe, "auto_plot_default", False)`.

You can always override either way with an explicit `--auto-plot` or `--no-auto-plot`.

## `--plot-required`: warn vs. strict

By default, an auto-plot failure is **non-fatal**: the run is considered successful, the artifacts are intact, and you get a `WARNING` log line:

```text
WARNING aiperf.plot.auto_plot - auto-plot failed (run artifacts intact at
  artifacts/my-run); see artifacts/my-run/plots/aiperf_plot.log for details.
  Re-run `aiperf plot artifacts/my-run` manually if needed.
```

Pass `--plot-required` to flip this: any plotting exception is re-raised, `aiperf profile` exits non-zero, and your CI catches it.

```bash
aiperf profile ... --auto-plot --plot-required
```

Recommended:

- Local interactive runs: `--auto-plot` (warn-only is fine — you can re-run plotting by hand).
- CI / automation: `--auto-plot --plot-required` (so plot-tooling regressions fail loudly).

`--plot-required` only matters when auto-plot is on. Setting it without `--auto-plot` (and without a recipe that defaults to on) is a no-op.

## What gets plotted

Auto-plot calls the same `aiperf plot` engine as the standalone command, with one fixed configuration:

- **Mode:** PNG.
- **Theme:** light (default).
- **Config:** `~/.aiperf/plot_config.yaml` (auto-created on first run). When the AIPerf YAML contains a `plot:` envelope, that envelope overrides `~/.aiperf/plot_config.yaml` for this run — see [Auto-plot with the `plot:` envelope](#auto-plot-with-the-plot-envelope) below.
- **Output dir:** `<artifact_dir>/plots/`.

If you want a custom theme, the dashboard, a different config file, or a different output location, **do not** rely on auto-plot — re-run `aiperf plot` by hand once the benchmark finishes:

```bash
aiperf plot artifacts/my-run --theme dark --output ./presentation-plots
aiperf plot artifacts/my-run --dashboard
```

See the [Customization Options](plot.md#customization-options) and [Interactive Dashboard Mode](plot.md#interactive-dashboard-mode) sections of the plot tutorial for the available knobs.

## Auto-plot with the `plot:` envelope

If the AIPerf YAML you pass to `aiperf profile -f config.yaml` contains a `plot:` section (inline dict or a bare-string path to a plot-config file), two things happen automatically:

1. **Auto-plot flips on.** `artifacts.auto_plot` is set to `True` by a model validator (`AIPerfConfig._plot_implies_auto_plot` in `src/aiperf/config/config.py`) whenever a `plot:` envelope is present and the user did not explicitly set `auto_plot`. An explicit `auto_plot: false` still wins; you get an info-level breadcrumb in the log so the silence isn't a surprise.
2. **The envelope overrides `~/.aiperf/plot_config.yaml`.** The auto-plot callback materializes the resolved envelope to `<artifact_dir>/.aiperf-plot-config.yaml` and passes that path to the plot engine for this run. Your user-level `~/.aiperf/plot_config.yaml` is ignored for this benchmark.

The materialized `.aiperf-plot-config.yaml` becomes a run artifact. Re-running `aiperf plot <artifact_dir>` later picks it up via the existing `--config` priority chain, so the run's plots reproduce without you having to keep the original AIPerf YAML or your user-level plot config around.

```yaml
benchmark:
  artifacts:
    dir: ./artifacts/my-run
plot: ./plots/baseline.yaml   # auto_plot flips on; this envelope wins over ~/.aiperf/plot_config.yaml
```

See [`src/aiperf/config/plot.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/config/plot.py) for the envelope schema and the allowed inline-vs-path forms.

## Disabling auto-plot

To suppress auto-plot, even for a recipe that defaults to on:

```bash
aiperf profile --search-recipe prefill-ttft-curve ... --no-auto-plot
```

`--no-auto-plot` is the cyclopts negative form of `--auto-plot`; both flags share `auto_plot` on `CLIConfig`. Passing either flag marks the field as user-set, so the explicit value wins over the recipe default.

## YAML config form

The same fields live on the v2 `artifacts` block when you drive the CLI with `aiperf profile -f config.yaml`:

```yaml
benchmark:
  artifacts:
    dir: ./artifacts/my-run
    auto_plot: true
    plot_required: false
```

`benchmark.artifacts.auto_plot` is a plain bool here (not the tri-state CLI input) — the CLI-to-YAML converter resolves the tri-state against the active recipe before writing the YAML-shape config. CLI flags overlay on top of YAML in the usual way: `aiperf profile -f config.yaml --no-auto-plot` wins.

## Troubleshooting

### Auto-plot fired but no PNGs appeared

Check `<artifact_dir>/plots/aiperf_plot.log` first. Common causes:

- **Custom export filenames:** if you set `--profile-export-prefix` or `--profile-export-file`, the plot command may not find the expected `profile_export.jsonl` / `profile_export_aiperf.json`. See the [plot tutorial's quick-start warning](plot.md#quick-start). Re-run without the custom prefix, or rename the files to match defaults.
- **Multi-run profile incompatibility:** `aiperf plot` does not recurse into `profile_runs/<trial_n>/` subdirectories. Auto-plot points at the top-level artifact dir, so runs that combine `--auto-plot` with `--num-profile-runs > 1` will not produce single-run PNGs for individual trials. Sweep aggregates still plot correctly.

### Auto-plot raised under `--plot-required`

`--plot-required` re-raises whatever the plot pipeline raised; the traceback is the source of truth. The most common failure modes:

- **Missing plotting dependencies** (the run completed but `aiperf plot` itself fails to import a backend). Reinstall with the plotting extras and try again.
- **No data to plot** (e.g., zero successful requests). `aiperf plot` raises rather than emit empty PNGs; the run results are still in `<artifact_dir>` for inspection.

In warn mode, the same exceptions become a single `WARNING` line and the command exits 0.

### `--plot-required` set but I never see strict-mode failures

If auto-plot resolves to **off** (no flag, no recipe default), `--plot-required` is dormant — there is nothing to make fatal. Either pass `--auto-plot`, or use a recipe that defaults it on (see the [per-recipe defaults](#per-recipe-defaults) above).

## Related

- [Visualization and Plotting with AIPerf](plot.md) — the full `aiperf plot` reference.
- [Search Recipes](../sweeping/search-recipes.md) — recipes that drive `auto_plot_default`.
- [Adaptive Search](adaptive-search.md) — Bayesian-optimization outer loop; pairs nicely with `--auto-plot` for the curve-emitting recipes.
- [`docs/cli-options.md`](../cli-options.md) — autogenerated flag reference, including `--auto-plot` and `--plot-required`.
