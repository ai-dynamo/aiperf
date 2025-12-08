<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Visualization and Plotting with AIPerf

Generate PNG visualizations from AIPerf profiling data with automatic mode detection, NVIDIA brand styling, and support for multi-run comparisons and single-run analysis.

## Overview

The `aiperf plot` command automatically detects whether to generate multi-run comparison plots or single-run time series analysis based on your directory structure. It integrates GPU telemetry and timeslice data when available.

**Key Features:**
- Automatic mode detection (multi-run comparison vs single-run analysis)
- GPU telemetry integration (power, utilization, memory, temperature)
- Timeslice support (performance evolution across time windows)
- Configurable plots via `~/.aiperf/plot_config.yaml`

## Quick Start

```bash
# Analyze a single profiling run
aiperf plot <single_run_name>

# Compare multiple runs in a directory
aiperf plot <run_directory>

# Compare all runs across multiple directories
aiperf plot <dir1> <dir2> <dir3>

# Compare specific runs
aiperf plot <run1> <run2> <run3>

# Specify custom output location
aiperf plot <path> --output <output_directory>

# Use dark theme
aiperf plot <path> --theme dark
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
```
artifacts/sweep_qwen/
├── Qwen3-0.6B-concurrency1/
├── Qwen3-0.6B-concurrency2/
└── Qwen3-0.6B-concurrency4/
```

**Default plots (3):**
1. **TTFT vs Throughput** - Time to first token vs request throughput
2. **Token Throughput per GPU vs Latency** - GPU efficiency vs latency (requires GPU telemetry)
3. **Token Throughput per GPU vs Interactivity** - GPU efficiency vs TTFT (requires GPU telemetry)

> [!TIP]
> Use [Experiment Classification](#experiment-classification) to assign semantic colors (grey for baselines, green for treatments) for clearer visual distinction.

#### Example Visualizations

![TTFT vs Throughput](../diagrams/plot_examples/multi_run/ttft_vs_throughput.png)

Shows how time to first token varies with request throughput across concurrency levels, helping identify configurations that balance responsiveness with load.

![Pareto Curve: Throughput per GPU vs Latency](../diagrams/plot_examples/multi_run/pareto_curve_throughput_per_gpu_vs_latency.png)

Highlights optimal configurations on the Pareto frontier that maximize GPU efficiency while minimizing latency.

![Pareto Curve: Throughput per GPU vs Interactivity](../diagrams/plot_examples/multi_run/pareto_curve_throughput_per_gpu_vs_interactivity.png)

Shows the trade-off between GPU efficiency and interactivity (TTFT).

### Single-Run Analysis Mode

Analyzes performance over time for a single profiling run.

**Auto-detected when:**
- Directory contains `profile_export.jsonl` directly

**Example:**
```
artifacts/single_run/
└── profile_export.jsonl
```

**Default plots (4+):**
1. **TTFT Over Time** - Time to first token per request
2. **Inter-Token Latency Over Time** - ITL per request
3. **Request Latency Over Time** - End-to-end latency progression
4. **Dispersed Throughput Over Time** - Continuous token generation rate

**Additional plots (when data available):**
- Timeslice plots (when `--slice-duration` used during profiling)
- GPU telemetry plots (when `--gpu-telemetry` used during profiling)

#### Example Visualizations

![TTFT Over Time](../diagrams/plot_examples/single_run/time_series/ttft_over_time.png)

Time to first token for each request, revealing prefill latency patterns and potential warm-up effects.

![Inter-Token Latency Over Time](../diagrams/plot_examples/single_run/time_series/itl_over_time.png)

Inter-token latency per request, showing generation performance consistency.

![Request Latency Over Time](../diagrams/plot_examples/single_run/time_series/latency_over_time.png)

End-to-end latency progression throughout the run.

### Dispersed Throughput

The **Dispersed Throughput Over Time** plot uses an event-based approach for accurate token generation rate visualization. Unlike binning methods that create artificial spikes, this distributes tokens evenly across their actual generation time:
- **Prefill phase** (request_start → TTFT): 0 tok/sec
- **Generation phase** (TTFT → request_end): constant rate = output_tokens / (request_end - TTFT)

This provides smooth, continuous representation that correlates better with server metrics like GPU utilization.

![Dispersed Throughput Over Time](../diagrams/plot_examples/single_run/dispersed_throughput_over_time.png)

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

> [!IMPORTANT]
> When enabled, **all multi-run plots automatically group by experiment_group** (directory name) to preserve individual treatment variants with semantic baseline/treatment colors.

**Pattern notes**: Uses glob syntax (`*` = wildcard), case-sensitive, first match wins.

#### Example

**Directory structure:**
```
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

![Pareto Curve with Experiment Classification](../diagrams/plot_examples/multi_run/config_experiment_classification/pareto_curve_throughput_per_gpu_vs_interactivity.png)

![TTFT vs Throughput with Experiment Classification](../diagrams/plot_examples/multi_run/config_experiment_classification/ttft_vs_throughput.png)

### Theme Options

```bash
# Light theme (default)
aiperf plot <path>

# Dark theme (for presentations)
aiperf plot <path> --theme dark
```

The dark theme uses a dark background optimized for presentations while maintaining NVIDIA brand colors.

#### Multi-Run Dark Theme

![TTFT vs Throughput (Dark)](../diagrams/plot_examples/multi_run/theme_dark_mode/ttft_vs_throughput.png)

![Pareto Curve: Latency (Dark)](../diagrams/plot_examples/multi_run/theme_dark_mode/pareto_curve_throughput_per_gpu_vs_latency.png)

![Pareto Curve: Interactivity (Dark)](../diagrams/plot_examples/multi_run/theme_dark_mode/pareto_curve_throughput_per_gpu_vs_interactivity.png)

#### Single-Run Dark Theme

![GPU Utilization (Dark)](../diagrams/plot_examples/single_run/time_series/theme_dark_mode/gpu_utilization_and_throughput_over_time.png)

![ITL Over Time (Dark)](../diagrams/plot_examples/single_run/time_series/theme_dark_mode/itl_over_time.png)

![ITL Across Timeslices (Dark)](../diagrams/plot_examples/single_run/time_series/theme_dark_mode/timeslices_itl.png)

## Advanced Features

### GPU Telemetry Integration

When GPU telemetry is collected (via `--gpu-telemetry` flag during profiling), plots automatically include GPU metrics.

**Multi-run plots** (when telemetry available):
- Token Throughput per GPU vs Latency
- Token Throughput per GPU vs Interactivity

**Single-run plots** (time series):
- GPU Utilization Over Time
- GPU Memory Usage Over Time

![GPU Utilization and Throughput Over Time](../diagrams/plot_examples/single_run/time_series/gpu_utilization_and_throughput_over_time.png)

> [!TIP]
> See the [GPU Telemetry Tutorial](gpu-telemetry.md) for setup and detailed analysis.

### Timeslice Integration

When timeslice data is available (via `--slice-duration` during profiling), plots show performance evolution across time windows.

**Generated timeslice plots:**
- TTFT Across Timeslices
- ITL Across Timeslices
- Throughput Across Timeslices
- Latency Across Timeslices

These help identify warm-up effects, performance degradation, and steady-state behavior.

![TTFT Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_ttft.png)

![ITL Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_itl.png)

![Latency Across Timeslices](../diagrams/plot_examples/single_run/timeslices/timeslices_latency.png)

> [!TIP]
> See the [Timeslices Tutorial](timeslices.md) for configuration and analysis.

## Output Files

Plots are saved as PNG files in the output directory:

```
plots/
├── ttft_vs_throughput.png
├── pareto_curve_throughput_per_gpu_vs_latency.png
├── pareto_curve_throughput_per_gpu_vs_interactivity.png
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
