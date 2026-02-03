<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Documentation

Welcome to AIPerf, a comprehensive benchmarking tool for measuring the performance of generative AI models served by any inference solution.

## What is AIPerf?

AIPerf is a production-grade, Python-based benchmarking tool that provides:

- **Accurate Performance Metrics**: Measure throughput, latency, token statistics, and resource utilization
- **Scalable Architecture**: Multi-process design handles high request rates with distributed workers
- **Flexible Load Patterns**: Support for concurrency, request-rate, trace replay, and user-centric timing modes
- **Extensible Plugin System**: Customize endpoints, datasets, timing strategies, and exporters
- **Comprehensive Analysis**: Real-time dashboard, detailed exports, and time-sliced metrics

## Quick Links

### Getting Started
- **[Installation & First Benchmark](getting-started.md)** - Install AIPerf and run your first benchmark
- **[Tutorial](tutorial.md)** - Step-by-step guide with practical examples
- **[Examples Gallery](examples.md)** - Real-world benchmark scenarios

### Core Concepts
- **[Architecture Overview](architecture.md)** - System design and components
- **[Metrics Reference](metrics_reference.md)** - Complete guide to all available metrics
- **[CLI Options](cli_options.md)** - Command-line reference

### Configuration & Tuning
- **[Environment Variables](environment_variables.md)** - Configuration via environment
- **[Performance Tuning](performance-tuning.md)** - Optimize AIPerf for your workload
- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking

### Features & Tutorials

#### Load Control & Timing
- [Request Rate with Max Concurrency](tutorials/request-rate-concurrency.md) - Dual control of rate and concurrency
- [Arrival Patterns](tutorials/arrival-patterns.md) - Traffic pattern simulation
- [Prefill Concurrency](tutorials/prefill-concurrency.md) - Memory-safe long-context benchmarking
- [Gradual Ramping](tutorials/ramping.md) - Smooth load increases
- [Warmup Phase](tutorials/warmup.md) - Eliminate cold-start effects
- [User-Centric Timing](tutorials/user-centric-timing.md) - Per-user rate limiting
- [Request Cancellation](tutorials/request-cancellation.md) - Test timeout behavior
- [Multi-URL Load Balancing](tutorials/multi-url-load-balancing.md) - Distribute across servers

#### Workloads & Data
- [Trace Benchmarking](tutorials/trace-benchmarking.md) - Deterministic workload replay
- [Custom Prompt Benchmarking](tutorials/custom-prompt-benchmarking.md) - Use your own prompts
- [Fixed Schedule](tutorials/fixed-schedule.md) - Precise timestamp-based execution
- [Time-based Benchmarking](tutorials/time-based-benchmarking.md) - Duration-based testing
- [Sequence Distributions](tutorials/sequence-distributions.md) - Mixed ISL/OSL workloads
- [Reproducibility](reproducibility.md) - Deterministic dataset generation
- [Benchmark Datasets](benchmark_datasets.md) - Available datasets

#### Analysis & Monitoring
- [Timeslice Metrics](tutorials/timeslices.md) - Time-windowed metric analysis
- [Goodput](tutorials/goodput.md) - SLO-compliant throughput
- [HTTP Trace Metrics](tutorials/http-trace-metrics.md) - Detailed HTTP lifecycle timing
- [Profile Exports](tutorials/working-with-profile-exports.md) - Parse and analyze results
- [Visualization & Plotting](tutorials/plot.md) - Generate performance charts
- [GPU Telemetry](tutorials/gpu-telemetry.md) - GPU metrics via DCGM
- [Server Metrics](server_metrics/server-metrics.md) - Prometheus-compatible server metrics

#### Endpoints & APIs
- [UI Types](tutorials/ui-types.md) - Dashboard, simple, or headless modes
- [Vision/Multimodal](tutorials/vision.md) - Benchmark image-capable models
- [Embeddings](tutorials/embeddings.md) - Benchmark embedding endpoints
- [Rankings](tutorials/rankings.md) - Benchmark ranking/reranking endpoints
- [Template Endpoint](tutorials/template-endpoint.md) - Create custom endpoint adapters

### Advanced Topics
- **[Plugin System](plugins/plugin-system.md)** - Extend AIPerf with custom plugins
- **[Creating Your First Plugin](plugins/creating-your-first-plugin.md)** - Plugin development guide
- **[Development Patterns](dev/patterns.md)** - Code patterns for contributors
- **[Deployment Guide](deployment.md)** - Docker, Kubernetes, multi-node setups

### Reference
- **[FAQ](faq.md)** - Frequently asked questions
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Glossary](glossary.md)** - Definitions of key terms
- **[Versioning & Upgrades](versioning-upgrade.md)** - Version compatibility and upgrades

### Migration & Comparison
- **[Migrating from GenAI-Perf](migrating.md)** - Migration guide for GenAI-Perf users
- **[Feature Comparison](genai-perf-feature-comparison.md)** - CLI feature comparison matrix

## Document Organization

```
docs/
├── index.md                    # This file - documentation hub
├── getting-started.md          # Installation and first benchmark
├── tutorial.md                 # Main tutorial
├── architecture.md             # System architecture
├── metrics_reference.md        # Complete metrics guide
├── cli_options.md             # Command-line reference
├── environment_variables.md    # Environment configuration
├── best-practices.md          # Benchmarking best practices
├── performance-tuning.md      # Performance optimization
├── deployment.md              # Deployment guide
├── troubleshooting.md         # Problem solving
├── faq.md                     # Frequently asked questions
├── glossary.md                # Term definitions
├── examples.md                # Real-world examples
├── benchmark_datasets.md      # Available datasets
├── reproducibility.md         # Deterministic benchmarking
├── migrating.md              # Migration from GenAI-Perf
│
├── tutorials/                 # Feature-specific tutorials
│   ├── request-rate-concurrency.md
│   ├── arrival-patterns.md
│   ├── prefill-concurrency.md
│   ├── goodput.md
│   ├── gpu-telemetry.md
│   └── ... (31 tutorials)
│
├── plugins/                   # Plugin system docs
│   ├── plugin-system.md
│   └── creating-your-first-plugin.md
│
├── server_metrics/            # Server metrics docs
│   ├── server-metrics.md
│   └── server_metrics_reference.md
│
└── dev/                       # Developer documentation
    └── patterns.md
```

## Need Help?

- **Issues**: Report bugs at [GitHub Issues](https://github.com/ai-dynamo/aiperf/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/ai-dynamo/aiperf/discussions)
- **Discord**: Join our community at [Discord](https://discord.gg/D92uqZRjCZ)
- **Documentation**: Browse comprehensive docs at [DeepWiki](https://deepwiki.com/ai-dynamo/aiperf)

## Contributing

Interested in contributing? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

AIPerf is licensed under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.
