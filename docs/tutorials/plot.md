---
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Visualization and Plotting with AIPerf
---

# Visualization and Plotting with AIPerf

Generate PNG visualizations from AIPerf profiling data with automatic mode detection, NVIDIA brand styling, and support for multi-run comparisons and single-run analysis.

## Overview

The `aiperf plot` command automatically detects whether to generate multi-run comparison plots or single-run time series analysis based on your directory structure, including nested `profile_runs/run_000N` directories from multi-run profiles. It integrates GPU telemetry and timeslice data when available. Aggregate summary directories may not be directly plottable; point `aiperf plot` at the run root or concrete per-run directories that contain profile exports.

**Key Features:**
- Automatic mode detection (multi-run comparison vs single-run analysis)
- GPU telemetry integration (power, utilization, memory, temperature)
- Timeslice support (performance evolution across time windows)
- Configurable plots via `~/.aiperf/plot_config.yaml`

> [!NOTE]
> **Multi-Run Profile Discovery**: When `--num-profile-runs > 1` produces `profile_runs/` subdirectories (e.g., `artifacts/my_run/profile_runs/run_0001/` for no-sweep multi-run and `trial_0001/` for sweep multi-run), the plot command auto-discovers them across no-sweep, REPEATED, INDEPENDENT, and adaptive Bayesian-optimization layouts. To plot a specific cell directly, you may also pass `<base>/profile_runs/` explicitly.

## Quick Start

> [!WARNING]
> **Custom export filenames not supported:** The plot command expects default export filenames (`profile_export.jsonl`, `profile_export_aiperf.json`). If you ran `aiperf profile` with `--profile-export-file` or a custom `--profile-export-prefix`, the output files will have different names and will not be detected by `aiperf plot`. To use the plot command, re-run profiling without custom export file options, or rename the files to match the default names.

Analyze a single profiling run:

```bash
aiperf plot <single_run_name>
```

**Sample Output (Successful Run):**

```text
INFO     Loading single-run data from: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/
INFO     Detected mode: SINGLE_RUN
INFO     Generating 5 time series plots
INFO     Creating plot: ttft_over_time.png
INFO     Creating plot: ttft_timeline.png
INFO     Creating plot: timeslices_ttft.png
INFO     Creating plot: timeslices_itl.png
INFO     Creating plot: gpu_utilization_and_throughput_over_time.png
INFO     Successfully generated 5 plots
INFO     Plots saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/plots/
```

Compare multiple runs in a directory:

```bash
aiperf plot <run_directory>
```

**Sample Output (Successful Run):**

```text
INFO     Loading multi-run data from: artifacts/sweep_qwen/
INFO     Detected mode: MULTI_RUN
INFO     Found 3 runs to compare
INFO     Generating 4 comparison plots
INFO     Creating plot: pareto_curve_throughput_per_gpu_vs_latency.png
INFO     Creating plot: pareto_curve_throughput_per_gpu_vs_interactivity.png
INFO     Creating plot: ttft_vs_throughput.png
INFO     Creating plot: latency_throughput_uncertainty.png
INFO     Successfully generated 4 plots
INFO     Plots saved to: artifacts/sweep_qwen/plots/
```

Other common invocations:

```bash
# Compare all runs across multiple directories
aiperf plot <dir1> <dir2> <dir3>

# Compare specific runs
aiperf plot <run1> <run2> <run3>

# Specify custom output location
aiperf plot <path> --output <output_directory>
```

Launch interactive dashboard for exploration:

```bash
aiperf plot <path> --dashboard
```

**Sample Output (Successful Run):**

```text
INFO     Loading data from: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/
INFO     Starting interactive dashboard
INFO     Dash is running on http://localhost:8050/

 * Serving Flask app 'aiperf.plot.dashboard'
 * Debug mode: off
INFO     Dashboard ready at http://localhost:8050/
INFO     Press Ctrl+C to quit
```

Use dark theme:

```bash
aiperf plot <path> --theme dark
```

**Sample Output (Successful Run):**

```text
INFO     Loading data from: artifacts/sweep_qwen/
INFO     Detected mode: MULTI_RUN
INFO     Using dark theme
INFO     Found 3 runs to compare
INFO     Generating 4 comparison plots
INFO     Successfully generated 4 plots
INFO     Plots saved to: artifacts/sweep_qwen/plots/
```

**Output directory logic:**
- If `--output` specified: uses that path
- Otherwise: `<first_input_path>/plots/`
- Default (no paths): `./artifacts/plots/`

**Customize plots**: Edit `~/.aiperf/plot_config.yaml` (auto-created on first run) to enable/disable plots or customize visualizations. See [Plot Configuration](#plot-configuration-yaml) for details.

## Visualization Modes

The plot command automatically detects visualization mode based on directory structure:

### Multi-Run Comparison Mode

Compares metrics across multiple profiling runs to identify optimal configurations.

**Auto-detected when:**
- Directory contains multiple run subdirectories, OR
- Multiple paths specified as arguments

**Example:**
```text
artifacts/sweep_qwen/
├── Qwen3-0.6B-concurrency1/
├── Qwen3-0.6B-concurrency2/
└── Qwen3-0.6B-concurrency4/
```

**Default plots (4):**
1. **TTFT vs Throughput** - Time to first token vs request throughput
2. **Token Throughput per GPU vs Latency** - GPU efficiency vs latency (requires GPU telemetry)
3. **Token Throughput per GPU vs Interactivity** - GPU efficiency vs TTFT (requires GPU telemetry)
4. **Latency vs Throughput (Joint Uncertainty)** - latency vs throughput-per-GPU with 95% confidence ellipses

> [!TIP]
> Use [Experiment Classification](#experiment-classification) to assign semantic colors (grey for baselines, green for treatments) for clearer visual distinction.

#### Example Visualizations

![TTFT vs Throughput](../diagrams/plot-examples/multi-run/ttft-vs-throughput.png)

Shows how time to first token varies with request throughput across concurrency levels. **Potentially useful for finding the sweet spot between responsiveness and capacity**: ideal configurations maintain low TTFT even at high throughput. If TTFT increases sharply at certain throughput levels, this may indicate a prefill bottleneck (batch scheduler contention or compute limitations).

![Pareto Curve: Throughput per GPU vs Latency](../diagrams/plot-examples/multi-run/pareto-curve-throughput-per-gpu-vs-latency.png)

Highlights optimal configurations on the Pareto frontier that maximize GPU efficiency while minimizing latency. **Points on the frontier are optimal; points below are suboptimal** configurations. Potentially useful for choosing GPU count and batch sizes to maximize hardware ROI. A steep curve may indicate opportunities to improve latency with minimal throughput loss, while a flat curve can suggest you're near the efficiency limit.

![Pareto Curve: Throughput per GPU vs Interactivity](../diagrams/plot-examples/multi-run/pareto-curve-throughput-per-gpu-vs-interactivity.png)

Shows the trade-off between GPU efficiency and interactivity (TTFT). **Potentially useful for determining max concurrency before user experience degrades**: flat regions show where adding concurrency maintains interactivity, while steep sections may indicate diminishing returns. The "knee" of the curve can help identify where throughput gains start to significantly hurt responsiveness.

### Single-Run Analysis Mode

Analyzes performance over time for a single profiling run.

**Auto-detected when:**
- Directory contains `profile_export.jsonl` directly

**Example:**
```text
artifacts/single_run/
└── profile_export.jsonl
```

**Default plots (5, enabled in shipped `single_run_defaults`):**
1. **TTFT Over Time** (`ttft_over_time`) - Time to first token per request
2. **TTFT Timeline** (`ttft_timeline`) - Per-request TTFT plotted against request start time
3. **TTFT Across Timeslices** (`timeslices_ttft`) - TTFT statistics per time window
4. **ITL Across Timeslices** (`timeslices_itl`) - Inter-token latency statistics per time window
5. **GPU Utilization and Throughput Over Time** (`gpu_utilization_and_throughput_over_time`) - Correlated GPU usage and token rate (requires GPU telemetry)

**Commented-out by default** (uncomment in `~/.aiperf/plot_config.yaml` to enable):
- **Inter-Token Latency Over Time** (`itl_over_time`) - ITL per request
- **Request Latency Over Time** (`latency_over_time`) - End-to-end latency progression
- **Dispersed Throughput Over Time** (`dispersed_throughput_over_time`) - Continuous token generation rate

**Additional plots (when data available):**
- Timeslice plots (when `--slice-duration` used during profiling)
- GPU telemetry plots (when `--gpu-telemetry` used during profiling)

#### Example Visualizations

![TTFT Over Time](../diagrams/plot-examples/single-run/time-series/ttft-over-time.png)

Time to first token for each request, revealing prefill latency patterns and potential warm-up effects. **Initial spikes may indicate cold start; stable later values show steady-state performance**. Potentially useful for determining necessary warmup period or identifying warmup configuration issues. Unexpected spikes during steady-state can suggest resource contention, garbage collection pauses, or batch scheduler interference.

![Inter-Token Latency Over Time](../diagrams/plot-examples/single-run/time-series/itl-over-time.png)

Inter-token latency per request, showing generation performance consistency. **Consistent ITL may indicate stable generation; variance can suggest batch scheduling issues**. Potentially useful for identifying decode-phase bottlenecks separate from prefill issues. If ITL increases over time, this may indicate KV cache memory pressure or growing batch sizes causing decode slowdown.

![Request Latency Over Time](../diagrams/plot-examples/single-run/time-series/latency-over-time.png)

End-to-end latency progression throughout the run. **Overall system health check**: ramp-up at the start is normal, but sustained increases may indicate performance degradation. Potentially useful for identifying if your system maintains performance or degrades over time. Sudden jumps may correlate with other requests completing or starting, potentially revealing batch scheduling patterns.

![Request Timeline: TTFT](../diagrams/plot-examples/single-run/time-series/ttft-timeline.png)

Individual requests plotted as lines spanning their duration from start to end. **Visualizes request scheduling and concurrency patterns**: overlapping lines show concurrent execution, while gaps may indicate scheduling delays. Dense packing can suggest efficient utilization; sparse patterns may suggest underutilized capacity or rate limiting effects.

### Dispersed Throughput

The **Dispersed Throughput Over Time** plot uses an event-based approach for accurate token generation rate visualization. Unlike binning methods that create artificial spikes, this distributes tokens evenly across their actual generation time:
- **Prefill phase** (request_start → TTFT): 0 tok/sec
- **Generation phase** (TTFT → request_end): constant rate = output_tokens / (request_end - TTFT)

This provides smooth, continuous representation that correlates better with server metrics like GPU utilization.

![Dispersed Throughput Over Time](../diagrams/plot-examples/single-run/time-series/dispersed-throughput-over-time.png)

**Smooth ramps may show healthy scaling; drops can indicate bottlenecks**. Potentially useful for correlating with GPU metrics to identify whether bottlenecks are GPU-bound, memory-bound, or CPU-bound. A plateau may indicate you've reached max sustainable throughput for your configuration. Sudden drops can potentially correlate with resource exhaustion or scheduler saturation.

## Customization Options

### Plot Configuration YAML

Customize which plots are generated and how they appear by editing `~/.aiperf/plot_config.yaml`.

#### Enable/Disable Plots

**Multi-run plots:**
```yaml
visualization:
  multi_run_defaults:
    - pareto_curve_throughput_per_gpu_vs_latency
    - pareto_curve_throughput_per_gpu_vs_interactivity
    - ttft_vs_throughput
```

**Single-run plots:**
```yaml
visualization:
  single_run_defaults:
    - ttft_over_time
    - itl_over_time
    - dispersed_throughput_over_time
    # ... add or remove plots
```

#### Customize Plot Grouping

Multi-run comparison plots group runs to create colored lines/series. Customize the `groups:` field in plot presets:

**Group by model** (useful for comparing different models):
```yaml
multi_run_plots:
  ttft_vs_throughput:
    groups: [model]
```

**Group by directory** (useful for hierarchical experiments):
```yaml
multi_run_plots:
  ttft_vs_throughput:
    groups: [experiment_group]
```

**Group by run name** (default - each run is separate):
```yaml
multi_run_plots:
  ttft_vs_throughput:
    groups: [run_name]
```

> [!NOTE]
> When experiment classification is enabled, all multi-run plots automatically group by `experiment_group` to preserve treatment variants with semantic colors.

> [!TIP]
> See the CONFIGURATION GUIDE section in `~/.aiperf/plot_config.yaml` for detailed customization options.

### Experiment Classification

Classify runs as "baseline" or "treatment" for semantic color assignment in multi-run comparisons.

**Configuration** (`~/.aiperf/plot_config.yaml`):
```yaml
experiment_classification:
  baselines:
    - "*baseline*"     # Glob patterns
    - "*_agg_*"
  treatments:
    - "*treatment*"
    - "*_disagg_*"
  default: treatment   # Fallback when no match
```

**Result:**
- **Baselines**: Grey shades, listed first in legend
- **Treatments**: NVIDIA green shades, listed after baselines
- **Use case**: Clear visual distinction for A/B testing

> [!WARNING]
> When enabled, **all multi-run plots automatically group by experiment_group** (directory name) to preserve individual treatment variants with semantic baseline/treatment colors.

**Pattern notes**: Uses glob syntax (`*` = wildcard), case-sensitive, first match wins.

#### Example

**Directory structure:**
```text
artifacts/
├── baseline_moderate_io_isl100_osl200_streaming/           # Grey
│   ├── concurrency_1/
│   └── concurrency_2/
├── treatment_large_context_isl500_osl50_streaming/         # Green
│   ├── concurrency_1/
│   └── concurrency_2/
└── treatment_long_generation_isl50_osl500_streaming/       # Blue
    ├── concurrency_1/
    └── concurrency_2/
```

**Result**: 3 lines in plots (1 baseline + 2 treatments, each with semantic colors)

**Advanced**: Use `group_extraction_pattern` to aggregate variants:
```yaml
group_extraction_pattern: "^(treatment_\d+)"  # Groups treatment_1_varA + treatment_1_varB → "treatment_1"
```

> [!TIP]
> See `src/aiperf/plot/default_plot_config.yaml` for all configuration options.

![Pareto Curve with Experiment Classification](../diagrams/plot-examples/multi-run/config-experiment-classification/pareto-curve-throughput-per-gpu-vs-interactivity.png)

![TTFT vs Throughput with Experiment Classification](../diagrams/plot-examples/multi-run/config-experiment-classification/ttft-vs-throughput.png)

### Theme Options

```bash
# Light theme (default)
aiperf plot <path>

# Dark theme (for presentations)
aiperf plot <path> --theme dark
```

The dark theme uses a dark background optimized for presentations while maintaining NVIDIA brand colors.

#### Multi-Run Dark Theme

![TTFT vs Throughput (Dark)](../diagrams/plot-examples/multi-run/theme-dark-mode/ttft-vs-throughput.png)

![Pareto Curve: Latency (Dark)](../diagrams/plot-examples/multi-run/theme-dark-mode/pareto-curve-throughput-per-gpu-vs-latency.png)

![Pareto Curve: Interactivity (Dark)](../diagrams/plot-examples/multi-run/theme-dark-mode/pareto-curve-throughput-per-gpu-vs-interactivity.png)

#### Single-Run Dark Theme

![GPU Utilization (Dark)](../diagrams/plot-examples/single-run/time-series/theme-dark-mode/gpu-utilization-and-throughput-over-time.png)

![ITL Over Time (Dark)](../diagrams/plot-examples/single-run/time-series/theme-dark-mode/itl-over-time.png)

![ITL Across Timeslices (Dark)](../diagrams/plot-examples/single-run/time-series/theme-dark-mode/timeslices-itl.png)

## Interactive Dashboard Mode

Launch an interactive localhost-hosted dashboard for real-time exploration of profiling data with dynamic metric selection, filtering, and visualization customization.

```bash
# Launch dashboard with default settings (localhost:8050)
aiperf plot --dashboard

# Specify custom port
aiperf plot --dashboard --port 9000

# Launch with dark theme
aiperf plot --dashboard --theme dark

# Specify data paths
aiperf plot path/to/runs --dashboard
```

**Key Features:**
- **Dynamic metric switching**: Toggle between avg, p50, p90, p95, p99 statistics in real-time
- **Run filtering**: Select which runs to display via checkboxes
- **Log scale toggles**: Per-plot X/Y axis log scale controls
- **Config viewer**: Click on data points to view full run configuration
- **Custom plots**: Add new plots with custom axis selections
- **Plot management**: Hide/show plots dynamically
- **Export**: Download visible plots as PNG bundle

The dashboard automatically detects visualization mode (multi-run comparison or single-run analysis) and displays appropriate tabs and controls. Press Ctrl+C in the terminal to stop the server.

> [!TIP]
> The dashboard binds to `127.0.0.1` by default and requires no authentication. For remote access, either bind on all interfaces with `aiperf plot --dashboard --host 0.0.0.0 --port 9000` (only on trusted networks) or use SSH port forwarding: `ssh -L 8050:localhost:8050 user@remote-host`

> [!NOTE]
> Dashboard mode and PNG mode are separate. To generate both static PNGs and launch the dashboard, run the commands separately.

## Advanced Features

### GPU Telemetry Integration

**Multi-run plots** (when telemetry available):
- Token Throughput per GPU vs Latency
- Token Throughput per GPU vs Interactivity

**Single-run plots** (time series):
```text
- GPU Utilization Over Time
- GPU Memory Usage Over Time
```

![GPU Utilization and Throughput Over Time](../diagrams/plot-examples/single-run/time-series/gpu-utilization-and-throughput-over-time.png)

**Correlates compute resources with token generation performance**. High GPU utilization with low throughput may suggest compute-bound workloads (consider optimizing model/batch size). Low utilization with low throughput can indicate bottlenecks elsewhere (KV cache, memory bandwidth, CPU scheduling). Potentially useful for targeting >80% GPU utilization for efficient hardware usage.

> [!TIP]
> See the [GPU Telemetry Tutorial](gpu-telemetry.md) for setup and detailed analysis.

### Timeslice Integration

When timeslice data is available (via `--slice-duration` during profiling), plots show performance evolution across time windows.

**Generated timeslice plots:**
```text
- TTFT Across Timeslices
- ITL Across Timeslices
- Throughput Across Timeslices
- Latency Across Timeslices
```

**Timeslices enable easy outlier identification and bucketing analysis**. Each time window (bucket) shows avg/p50/p95 statistics, making it simple to spot which periods have outlier performance. Slice 0 often shows cold-start overhead, while later slices may reveal degradation. Flat bars across slices may indicate stable performance; increasing trends can suggest resource exhaustion. Potentially useful for quickly isolating performance issues to specific phases (warmup, steady-state, or degradation).

![TTFT Across Timeslices](../diagrams/plot-examples/single-run/timeslices/timeslices-ttft.png)

![ITL Across Timeslices](../diagrams/plot-examples/single-run/timeslices/timeslices-itl.png)

![Latency Across Timeslices](../diagrams/plot-examples/single-run/timeslices/timeslices-latency.png)

> [!TIP]
> See the [Timeslices Tutorial](timeslices.md) for configuration and analysis.

## Output Files

Plots are saved as PNG files in the output directory:

```
plots/
├── ttft_vs_throughput.png
├── pareto_curve_throughput_per_gpu_vs_latency.png
├── pareto_curve_throughput_per_gpu_vs_interactivity.png
├── latency_throughput_uncertainty.png
├── ttft_over_time.png (single-run)
├── dispersed_throughput_over_time.png (single-run)
├── gpu_utilization_and_throughput_over_time.png (if GPU telemetry)
└── timeslices_*.png (if timeslice data available)
```

## Best Practices

> [!TIP]
> **Consistent Configurations**: When comparing runs, vary only one parameter (e.g., concurrency) while keeping others constant. This isolates the impact of that specific parameter.

> [!TIP]
> **Use Experiment Classification**: Configure [experiment classification](#experiment-classification) to distinguish baselines from treatments with semantic colors.

> [!TIP]
> **Include Warmup**: Use `--warmup-request-count` to ensure steady state before measurement, reducing noise in visualizations.

> [!WARNING]
> **Directory Structure**: Ensure consistent naming - runs to compare must be in subdirectories of a common parent.

> [!NOTE]
> **GPU Metrics**: GPU telemetry plots only appear when telemetry data is available. Ensure DCGM is running during profiling. See [GPU Telemetry Tutorial](gpu-telemetry.md).

## Troubleshooting

### No Plots Generated

**Solutions**:
- Verify input directory contains valid `profile_export.jsonl` files
- If you used `--profile-export-file` or `--profile-export-prefix` during profiling, the output files have non-default names and will not be detected by the plot command. Re-run without custom export file options, or rename files to match the defaults (`profile_export.jsonl`, `profile_export_aiperf.json`)
- Check output directory is writable
- Review console output for error messages

### Missing GPU Plots

**Solutions**:
- Verify `gpu_telemetry_export.jsonl` exists and contains data
- Ensure DCGM exporter was running during profiling
- Check telemetry data is present in profile exports

### Incorrect Mode Detection

**Solutions**:
- Check directory structure:
  - Multi-run: parent directory with multiple run subdirectories
  - Single-run: directory with `profile_export.jsonl` directly inside
- Ensure all run directories contain valid `profile_export.jsonl` files

## Related Documentation

- [Working with Profile Exports](working-with-profile-exports.md) - Understanding profiling data format
- [GPU Telemetry](gpu-telemetry.md) - Collecting GPU metrics
- [Timeslices](timeslices.md) - Time-windowed performance analysis
- [Request Rate and Concurrency](request-rate-concurrency.md) - Load generation strategies
