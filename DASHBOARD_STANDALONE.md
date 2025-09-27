<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Dashboard UI - Standalone Mode

This guide explains how to run the AIPerf Dashboard UI in standalone mode for testing mouse and keyboard interactions. The standalone mode uses **real AIPerf modules** with minimal ZMQ mocking for a more authentic testing experience.

## Quick Start

```bash
# Make sure you're in the aiperf project directory
cd /path/to/aiperf

# Activate the virtual environment
source .venv/bin/activate

# Run the standalone dashboard
python run_dashboard_standalone.py
```

## Features

- **🎯 Authentic UI Testing**: Uses **real AIPerf modules** including genuine ZMQ infrastructure
- **🎬 Real Demo Data**: Generates realistic metrics using actual `MetricResult` objects
- **⌨️ Full Keyboard Support**: All original keyboard shortcuts work perfectly
- **🖱️ Mouse Interactions**: Double-click to maximize panels, scrolling, etc.
- **🎨 Complete Theme**: Uses the full AIPerf theme and styling
- **🔧 Authentic Infrastructure**: Real ZMQ communication, configs, and service components
- **📦 Proper Loading**: Uses `module_loader` to register all implementations correctly

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Overview mode |
| `2` | Maximize Progress panel |
| `3` | Maximize Metrics panel |
| `4` | Maximize Workers panel |
| `5` | Maximize Logs panel |
| `ESC` | Restore view (un-maximize) |
| `L` | Toggle logs visibility |
| `Ctrl+C` | Quit application |
| `Ctrl+S` | Save screenshot (hidden) |

## Mouse Interactions

- **Double-click** any panel to maximize/restore it
- **Scroll** in log viewer to navigate through messages
- **Click** to focus different elements

## Epic Demo Data 🎬

The standalone mode showcases a **full profiling workflow simulation**:

### 📊 Complete Profiling Phases
- **Warmup Phase** (15s): Real `RequestsStats` with `CreditPhase.WARMUP`, smooth progress 0→100 over 12s
- **Profiling Phase** (45s): Real `RequestsStats` with `CreditPhase.PROFILING`, 0→1000 over 40s, high throughput
- **Records Phase** (20s): Real `RecordsStats` for data processing, 0→1000 over 15s, selective workers
- **Completed Phase** (10s): Final cleanup and reset for continuous demo cycle

### ⚡ Dynamic Worker Management
- **8 Demo Workers**: `worker-01` through `worker-08` with realistic behaviors
- **Real WorkerStats**: Uses authentic `WorkerTaskStats` and `ProcessingStats` models
- **Status Updates**: Workers transition between IDLE/ACTIVE based on current phase
- **Individual Stats**: Task completion, processing records, error tracking per worker
- **Realistic Patterns**: Workers start/stop in waves, higher activity during profiling

### 📈 REAL AIPerf Metrics Dashboard
- **8 Authentic Metrics**: Uses actual AIPerf metric tags and display properties
- **TTFT** (`ttft`): Time to First Token - 120ms warmup → 45ms profiling → 35ms records
- **Request Latency** (`request_latency`): End-to-end latency - 180ms → 85ms → 65ms
- **Token Throughput** (`output_token_throughput`): 250 → 1250 → 800 tokens/sec
- **Request Throughput** (`request_throughput`): 15 → 45 → 25 requests/sec
- **Output Sequence Length** (`output_sequence_length`): Token counts per request
- **Per-User Throughput** (`output_token_throughput_per_user`): Individual user rates
- **Request Count** (`request_count`): Total processed requests with proper display order
- **Error Count** (`error_request_count`): Phase-appropriate error simulation

### 📝 Smart Logging
- **Phase-Specific Messages**: Different log types for warmup, profiling, records, completion
- **Realistic Content**: Worker updates, batch processing, performance milestones
- **Varied Log Levels**: INFO, DEBUG, SUCCESS, WARNING with appropriate weighting
- **Contextual Timing**: Messages match the current profiling phase activity

## Troubleshooting

### Import Errors
If you see import errors, make sure:
1. You're in the correct project directory
2. The virtual environment is activated
3. All required dependencies are installed

### UI Not Responding
- Press `ESC` to restore view if panels seem stuck
- Use `Ctrl+C` to quit and restart

### Performance Issues
The standalone mode is lightweight, but if you experience lag:
- The demo data generation updates every second
- UI redraws are optimized for performance
- Try resizing the terminal window

### Progress Tracking
The demo includes debug output to track progress:
- Every 5 seconds: Shows current phase progress (e.g., "🔥 Warmup Progress: 41/100 (5s)")
- Phase transitions: Displays when moving between phases with timestamps
- Smooth progression: Fixed issue where progress could get stuck at low values

## Development Notes

- **Fully Real AIPerf**: Uses actual ServiceConfig, SystemController, UI components, **and ZMQ modules**
- **Proper Module Loading**: Uses `module_loader.ensure_modules_loaded()` to register all implementations
- **Minimal Mocking**: Only LogConsumer is mocked to avoid complex queue setup
- **Real ZMQ Communication**: Uses genuine ZeroMQ communication infrastructure
- **Authentic Data**: Demo data uses real `MetricResult` objects with proper validation
- **Valid Configuration**: Creates minimal but valid UserConfig and ServiceConfig objects
- **Cross-platform**: Works on macOS, Linux, and Windows

## What's Tested

This standalone mode allows you to test:
- ✅ Panel maximize/minimize behavior
- ✅ Keyboard navigation and shortcuts
- ✅ Mouse interactions and scrolling
- ✅ Theme and color rendering
- ✅ Layout responsiveness
- ✅ Real-time data updates
- ✅ Log viewer functionality

## What's Not Included

- ❌ Actual profiling workloads
- ❌ Full service lifecycle management (services don't actually start)
- ❌ Worker coordination and distributed processing
- ❌ Real endpoint connections
- ❌ Complex log queue setup (LogConsumer is simplified)

Perfect for UI development, testing user interactions, and validating the dashboard's look and feel with **authentic AIPerf infrastructure**!
