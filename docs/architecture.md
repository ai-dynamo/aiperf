<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Architecture of AIPerf

AIPerf is a distributed benchmarking tool for measuring AI inference performance. It generates load against inference endpoints, collects detailed performance metrics, and provides comprehensive analysis of throughput, latency, and resource utilization.

## Architecture Overview

AIPerf is designed as a modular, extensible benchmarking framework that separates concerns across three architectural planes. The system scales horizontally by adding more workers while maintaining centralized orchestration.

![AIPerf High-Level Architecture](diagrams/high-level-architecture-diagram.png)

### Three-Plane Architecture

| Plane | Components | Purpose |
|-------|-----------|---------|
| **Control Plane** | SystemController, Timing Manager, Dataset Manager, Worker Manager | Decides what, when, and how many requests to send |
| **Data Plane** | Workers, Inference Server | Executes the actual I/O and request/response cycle |
| **Analytic Plane** | Record Processors, Records Manager, GPU Telemetry Manager, Server Metrics Manager | Processes benchmark results and collects GPU telemetry and server metrics for comprehensive analysis |

### Request Lifecycle

1. **Initialization**: Dataset Manager loads data, Timing Manager prepares schedule
2. **Execution**: Workers receive credits, access data, send requests to inference server
3. **Collection**: Workers capture response timing and content
4. **Processing**: Record Processors compute metrics in parallel
5. **Aggregation**: Records Manager collects and exports results

## Core Components

### System Controller

The System Controller is the central orchestrator that manages the lifecycle and coordination of all major modules involved in a benchmarking run.

**Key Responsibilities:**
- Registering and initializing core components
- Orchestrating the start, execution, and shutdown of benchmarking tasks
- Handling configuration, resource allocation, and inter-module communication
- Monitoring the overall progress and health of the benchmarking process
- Managing error handling, cleanup, and graceful termination of all modules

### Dataset Manager

The Dataset Manager handles all aspects of input data management during benchmarking runs.

**Key Responsibilities:**
- Loading datasets from various sources (JSONL, CSV, synthetic generators, trace replay formats)
- Parsing and validating input data to ensure it matches the expected format
- Writing dataset to memory-mapped files, enabling workers to access data directly without message passing
- Supporting custom dataset types, such as MoonCake traces, for advanced benchmarking scenarios
- Managing the lifecycle of datasets, including initialization, iteration, and cleanup

### Timing Manager

The Timing Manager controls and coordinates the timing of requests during benchmarking runs through a credit-based system.

**Key Responsibilities:**
- Scheduling when each request should be sent based on the selected timing mode (fixed schedule, request-rate, or user-centric rate)
- Managing precise timing to accurately reproduce real-world or synthetic load patterns
- Supporting advanced timing scenarios, such as replaying traces with specific inter-arrival times or simulating bursty traffic
- Ensuring that requests are dispatched to workers at the correct intervals for reliable measurement

### Worker Manager

The Worker Manager orchestrates and manages the pool of worker processes that execute benchmarking tasks.

**Key Responsibilities:**
- Coordinating with the system controller to spawn and shut down workers that send requests to the inference server
- Monitoring worker status, progress, and resource usage
- Handling worker lifecycle events, such as startup, shutdown, and error recovery
- Managing worker pool size based on benchmarking requirements

### Workers

Workers execute individual benchmarking tasks. Each worker operates as a process that sends requests to the inference server, collects responses, and records performance metrics.

**Key Responsibilities:**
- Receiving timing credits from the Credit Router
- Accessing dataset via memory-mapped files
- Formatting the data for the target endpoint
- Sending requests to the target endpoint according to the specified schedule
- Recording request and response timestamps
- Maintaining conversation context for multi-turn conversations
- Reporting results to the record processors for aggregation and analysis

### Record Processor

The Record Processor processes and interprets the responses received from the inference server during benchmarking.

**Key Responsibilities:**
- Parsing raw inference results to extract relevant metrics (latency, output tokens, correctness)
- Handling different response formats from various model endpoints (OpenAI, vLLM, Triton, custom APIs)
- Validating and normalizing results to ensure consistency across benchmarking runs
- Computing metrics derived from individual requests (TTFT, TPOT, E2E latency, throughput)
- Supporting error detection and handling for malformed or unexpected responses
- Scales horizontally to handle high-volume metric computation

### Records Manager

The Records Manager handles the collection, organization, and storage of benchmarking records and results.

**Key Responsibilities:**
- Aggregating data from the records processors (inference results, timing information, metrics)
- Storing records in memory and/or exporting them to files (CSV, JSON) for later analysis
- Providing interfaces for querying, filtering, and summarizing benchmarking results
- Supporting the generation of reports and artifacts for performance evaluation
- Managing the final export of aggregated performance summaries and per-request details

### GPU Telemetry Manager

The GPU Telemetry Manager collects GPU metrics from DCGM (Data Center GPU Manager) Exporter endpoints during benchmarking runs.

**Key Responsibilities:**
- Collecting GPU metrics from DCGM Exporter endpoints (power usage, energy consumption, utilization, memory usage, temperature, XID errors, power violations)
- Auto-discovering DCGM endpoints (default: `http://localhost:9400/metrics`)
- Supporting custom DCGM endpoints via `--gpu-telemetry` flag
- Exporting GPU telemetry alongside benchmark results

### Server Metrics Manager

The Server Metrics Manager collects metrics from Prometheus-compatible endpoints during benchmarking runs.

**Key Responsibilities:**
- Collecting metrics from Prometheus-compatible endpoints (inference server application metrics, system metrics, custom metrics)
- Auto-discovering metrics endpoints from configured inference server URLs (`--url`)
- Supporting custom Prometheus endpoints via `--server-metrics` flag
- Parsing any metrics exposed in Prometheus format (gauges, counters, histograms)
- Typical metrics collected: inference server KV cache usage, request counts, latencies, batch sizes, model-specific metrics, and server resource metrics
- Exporting server metrics alongside benchmark results

## Key Mechanisms

### Worker Execution

Workers execute the benchmark workload by sending requests to the inference server and collecting measurements. They are on the critical path for measurement accuracy and performance.

**Request Execution Loop:**
1. Receive timing credit from Credit Router
2. Access dataset entry via memory-mapped files
3. Format request for target endpoint (OpenAI, vLLM, etc.)
4. Send HTTP request to inference server
5. Collect response timing (TTFT, TPOT, E2E latency) and token counts
6. Report raw results to Record Processor

**State Management:**
- Maintains conversation context for multi-turn conversations
- Tracks token counts for throughput calculations
- Stateless between workers

**Performance Optimizations:**
- Async I/O for non-blocking HTTP calls
- Connection pooling for HTTP reuse
- Minimal processing (offload to Record Processors)

### Credit System & Request Timing

The Timing Manager uses a **credit-based flow control system** to precisely control when requests are sent to the inference server. This mechanism is fundamental to AIPerf's ability to reproduce specific load patterns and prevent overwhelming the server.

**How Credits Work:**
- Each credit grants permission to send exactly one request
- The Timing Manager issues credits according to the configured timing mode:
  - **Fixed schedule mode**: Replays conversation traces at precise timestamps from dataset metadata
  - **Request-rate mode**: Issues credits at a specific rate with configurable arrival patterns (constant, Poisson, gamma, concurrency burst)
  - **User-centric rate mode**: Each session acts as a separate user with calculated gaps between turns

**Flow Control Benefits:**
- Prevents overwhelming the inference server
- Enables precise reproduction of load patterns
- Provides natural backpressure when workers or server slow down
- Allows accurate measurement without artificial delays

**Credit Distribution:**
- Credits are routed to workers via ROUTER/DEALER pattern through a credit router
- Router selects workers based on sticky sessions (multi-turn conversations) or least-loaded worker selection
- No coordination required between workers
- Scales to large numbers of workers without bottlenecks
- Efficient message routing minimizes overhead

### Data Flow & Messaging

This section describes the end-to-end message flow during a benchmark run, showing how data moves between components through the ZMQ message bus.

![Data Flow](diagrams/data-flow-diagram.png)

**Key Data Structures:**
- **Timing Credit**: Grants permission to send one request
- **Dataset Entry**: Prompt and conversation context
- **Raw Result**: Request timing, tokens, response text
- **Processed Record**: Computed metrics (latency, throughput, etc.)
- **Aggregated Results**: Final performance summary and per-request details

**Message Flow:**
1. Credit Router routes credits to workers via ROUTER/DEALER pattern
2. Workers access dataset entries via memory-mapped files
3. Workers send requests to Inference Server (external HTTP)
4. Workers push raw results to Record Processors
5. Record Processors publish processed records to Records Manager
6. Records Manager aggregates and exports final results

**Flow Control:**
- Credit-based system prevents overwhelming the inference server
- Router controls which worker receives each credit based on load and sticky sessions
- Backpressure automatically applies when components slow down
- Asynchronous messaging ensures workers spend maximum time on I/O

### Metric Computation

Record Processors transform raw timing data into meaningful performance metrics. This computation happens in parallel across multiple processor instances for scalability.

**Core Metrics Computed:**
- **TTFT (Time to First Token)**: Time from request start to first token received
- **TPOT (Time per Output Token)**: Average time between successive output tokens
- **E2E Latency**: Total time from request start to completion
- **Input/Output Token Counts**: For throughput calculations
- **Request Success/Failure**: Error detection and categorization

**Processing Pipeline:**
1. Receive raw result from Worker (timestamps, tokens, response text)
2. Parse response format (OpenAI, vLLM, etc.)
3. Extract timing information and token counts
4. Compute derived metrics (TTFT, TPOT, throughput)
5. Validate and normalize results
6. Publish processed record to Records Manager

**Scalability:**
- Multiple Record Processor instances run in parallel
- Each processor handles a stream of raw results independently
- No coordination required between processors
- Scales linearly with the number of workers

**Error Handling:**
- Detects malformed responses
- Categorizes errors (timeout, invalid format, server error)
- Includes error details in processed records
- Enables debugging and analysis of failure modes

## Communication Architecture

All components communicate via a **ZeroMQ (ZMQ) message bus**, designed for low-latency, high-throughput message passing.

### Why ZMQ?

AIPerf uses ZMQ to maintain **measurement accuracy** by decoupling orchestration logic from execution:

- **Low-overhead messaging**: Credits are routed directly to workers via ROUTER/DEALER pattern
- **Asynchronous by design**: No blocking calls between services, ensuring workers spend maximum time on I/O and timing
- **Efficient transport**: ZMQ is designed for low-overhead inter-process communication
- **Scalability**: Supports distributed workers across multiple nodes without code changes

### Communication Patterns

AIPerf uses **ZMQ proxies** for message routing between services and workers:

- Services publish strongly-typed messages to specific topics (Pub/Sub pattern)
- Services subscribe to relevant message types
- Router/Dealer patterns for credit distribution to workers
- Request/Reply patterns for synchronous operations
- Asynchronous, decoupled communication (no shared mutable state)

### State Management

**Stateless design** for scalability:
- **Workers**: No shared state between workers; each maintains only local conversation context for multi-turn requests
- **Services**: All service state is ephemeral and can be reconstructed from configuration
- **Coordination**: Credit distribution happens through the message bus; dataset access via memory-mapped files
- **Results**: Only aggregated results are persistent (exported to files)

## Design Principles

### Separation of Concerns
- Control plane handles orchestration, scheduling, and data management
- Workers focus solely on request execution and basic data collection
- Record processors handle compute-intensive metric calculations
- Clean interfaces between components via message bus

### Scalability
- Workers scale horizontally for load generation
- Record processors scale horizontally for metric computation
- Credit-based flow control prevents overwhelming the system
- Async I/O throughout for maximum efficiency

### Extensibility
- Plugin system for datasets, endpoints, transports, metrics
- Decorator-based factory registration for easy additions
- Support for custom formats and protocols
- Modular architecture allows component replacement

## Deployment Modes

AIPerf supports distributed execution with two deployment models:

- **Multiprocess Mode**: Each service runs as a separate process on a single node (default for single-node deployments)
- **Kubernetes Mode**: Services and workers run as separate pods in a Kubernetes cluster (for multi-node deployments) *(not yet implemented)*

## External Dependencies

AIPerf integrates with external systems:

- **Inference Server**: The target system being benchmarked (vLLM, Dynamo, SGLang, etc.)
- **DCGM Exporter**: Optional GPU telemetry source for GPU Telemetry Manager (exposes GPU metrics in Prometheus format)
- **Prometheus-compatible endpoints**: Optional server/application metrics source for Server Metrics Manager (inference servers like vLLM expose metrics in Prometheus format at their /metrics endpoint)
