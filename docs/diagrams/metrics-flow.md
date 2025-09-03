<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

```mermaid
flowchart TD
    %% Input Data
    A["Parsed Inference Results<br/><em>(Load&nbsp;Balanced)</em>"]

    %% Stage 1: Distributed Record Processing
    A --> B1["MetricRecordProcessor<br/><em>(Distributed&nbsp;instance&nbsp;1)</em>"]
    A --> B2["MetricRecordProcessor<br/><em>(Distributed&nbsp;instance&nbsp;2)</em>"]
    A --> B3["MetricRecordProcessor<br/><em>(Distributed&nbsp;instance&nbsp;N)</em>"]

    %% RECORD Metric Path
    B1 --> C1["RECORD: RequestLatencyMetric<br/>parse_record() → 125ms<br/><em>(Individual&nbsp;value&nbsp;per&nbsp;request)</em>"]
    B2 --> C2["RECORD: RequestLatencyMetric<br/>parse_record() → 87ms<br/><em>(Individual&nbsp;value&nbsp;per&nbsp;request)</em>"]
    B3 --> C3["RECORD: RequestLatencyMetric<br/>parse_record() → 203ms<br/><em>(Individual&nbsp;value&nbsp;per&nbsp;request)</em>"]

    %% AGGREGATE Metric Path
    B1 --> D1["AGGREGATE: TotalRequestsMetric<br/>parse_record() → +1<br/><em>(Individual&nbsp;contribution)</em>"]
    B2 --> D2["AGGREGATE: TotalRequestsMetric<br/>parse_record() → +1<br/><em>(Individual&nbsp;contribution)</em>"]
    B3 --> D3["AGGREGATE: TotalRequestsMetric<br/>parse_record() → +1<br/><em>(Individual&nbsp;contribution)</em>"]

    %% MetricRecordDict Collection
    C1 --> E1["MetricRecordDict<br/><em>(Per-record&nbsp;results)</em>"]
    D1 --> E1
    C2 --> E2["MetricRecordDict<br/><em>(Per-record&nbsp;results)</em>"]
    D2 --> E2
    C3 --> E3["MetricRecordDict<br/><em>(Per-record&nbsp;results)</em>"]
    D3 --> E3

    %% Stage 2: Centralized Results Processing
    E1 --> G["RecordsManager → MetricResultsProcessor<br/><em>(Single&nbsp;centralized&nbsp;instance)</em>"]
    E2 --> G
    E3 --> G

    %% RECORD Processing in Central
    G --> H1["RECORD Collection<br/>append(125ms)<br/>append(87ms)<br/>append(203ms)<br/><em>(Collect&nbsp;all&nbsp;individual&nbsp;values)</em>"]

    %% AGGREGATE Processing in Central
    G --> H2["AGGREGATE Accumulation<br/>aggregate_value(+1) → total=1<br/>aggregate_value(+1) → total=2<br/>aggregate_value(+1) → total=3<br/><em>(Accumulate&nbsp;across&nbsp;processors)</em>"]

    H1 --> L["MetricResultsDict<br/><em>(Full&nbsp;profile&nbsp;run&nbsp;results)</em>"]
    H2 --> L

    %% Stage 4: Summarize Function Processing
    L --> I2["Summarize Function<br/>summarize()<br/><em>(Process&nbsp;all&nbsp;collected&nbsp;results)</em>"]

    %% Three outputs from Summarize Function
    I2 --> J1["RECORD Statistics<br/>p50=125ms, p95=203ms<br/>mean=138ms, std=58ms<br/>min=87ms, max=203ms<br/><em>(Full&nbsp;statistical&nbsp;analysis)</em>"]

    I2 --> J2["AGGREGATE Results<br/>final_value=3<br/>count=1<br/><em>(Single&nbsp;accumulated&nbsp;total)</em>"]

    I2 --> J3["DERIVED: ThroughputMetric<br/>derive_value(results)<br/>= total_requests / duration<br/>= 3 / 5.2s = 0.58 req/s<br/><em>(Computed&nbsp;from&nbsp;other&nbsp;metrics)</em>"]

    %% Final Output
    J1 --> K["MetricResult List<br/><em>Complete&nbsp;performance&nbsp;analysis</em>"]
    J2 --> K
    J3 --> K

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000000,font-weight:bold
    classDef distributed fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000000,font-weight:bold
    classDef recordMetric fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000000,font-weight:bold
    classDef aggregateMetric fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000000,font-weight:bold
    classDef derivedMetric fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000000,font-weight:bold
    classDef transport fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000000,font-weight:bold
    classDef central fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000000,font-weight:bold
    classDef collection fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#000000,font-weight:bold
    classDef statistics fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000000,font-weight:bold
    classDef output fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000000,font-weight:bold

    %% Apply styles
    class A input
    class B1,B2,B3 distributed
    class C1,C2,C3,J1,H1 recordMetric
    class D1,D2,D3,J2,H2 aggregateMetric
    class J3 derivedMetric
    class I2 central
    class I1,G statistics
    class E1,E2,E3,F,L transport
    class K output
```
