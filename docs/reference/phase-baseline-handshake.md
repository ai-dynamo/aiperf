# Phase Baseline Handshake

The phase baseline handshake captures point-in-time baseline readings at phase boundaries without coupling `TimingManager` to specific collectors. `PhaseRunner` pauses at each boundary through `PhaseGateClient`; `SystemController` fans the request out through `BaselineCoordinator`; registered baseline collectors scrape once and ACK; then the gate releases and the benchmark continues.

## Component map

```mermaid
flowchart LR
    subgraph TimingManager
        PR[PhaseRunner]
        PGC[PhaseGateClient]
        PR -->|before_phase / after_phase| PGC
    end

    subgraph Controller
        SC[SystemController]
        BC[BaselineCoordinator]
        SC --> BC
    end

    subgraph Collectors[Baseline collector services]
        GTM[GPUTelemetryManager]
        SMM[ServerMetricsManager]
        BCM[BaselineCollectorMixin]
        GTM --> BCM
        SMM --> BCM
    end

    PGC -->|PhaseStartGateCommand / PhaseEndGateCommand| SC
    BC -->|PhaseBaselineRequestMessage| BCM
    BCM -->|PhaseBaselineAckMessage| SC
    SC -->|PhaseGateGrantedResponse| PGC
```

## Per-phase message sequence

```mermaid
sequenceDiagram
    autonumber
    participant Runner as PhaseRunner
    participant Gate as PhaseGateClient
    participant SC as SystemController
    participant Coord as BaselineCoordinator
    participant Collector as Baseline collectors

    Runner->>Gate: before_phase(phase_id, phase_name)
    Gate->>SC: PhaseStartGateCommand
    SC->>Coord: gate_phase(kind=START)
    Coord->>Collector: PhaseBaselineRequestMessage(kind=START)
    Collector->>Collector: collect_baseline(START, phase_id, phase_name)
    Collector-->>SC: PhaseBaselineAckMessage(success=True)
    SC->>Coord: handle_ack(ack)
    Coord-->>SC: all registered collectors acked
    SC-->>Gate: PhaseGateGrantedResponse
    Gate-->>Runner: start gate released

    Runner->>Runner: issue credits and wait for returns

    Runner->>Gate: after_phase(phase_id, phase_name)
    Gate->>SC: PhaseEndGateCommand
    SC->>Coord: gate_phase(kind=END)
    Coord->>Collector: PhaseBaselineRequestMessage(kind=END)
    Collector->>Collector: collect_baseline(END, phase_id, phase_name)
    Collector-->>SC: PhaseBaselineAckMessage(success=True)
    SC->>Coord: handle_ack(ack)
    Coord-->>SC: all registered collectors acked
    SC-->>Gate: PhaseGateGrantedResponse
    Gate-->>Runner: end gate released
```

## Credit ordering at phase boundaries

```mermaid
sequenceDiagram
    autonumber
    participant Runner as PhaseRunner
    participant Gate as PhaseGateClient
    participant Controller as SystemController
    participant Coord as BaselineCoordinator
    participant Collectors as Baseline collectors
    participant Issuer as CreditIssuer
    participant Workers as Workers

    Runner->>Gate: before_phase(phase_id, phase_name)
    Gate->>Controller: PhaseStartGateCommand
    Controller->>Coord: gate_phase(kind=START)
    Coord->>Collectors: PhaseBaselineRequestMessage(kind=START)
    Collectors-->>Controller: PhaseBaselineAckMessage(success=True)
    Controller-->>Gate: PhaseGateGrantedResponse
    Gate-->>Runner: START gate released
    Runner->>Issuer: start strategy.execute_phase()
    Issuer->>Workers: publish credits for this phase
    Workers-->>Runner: return credit results
    Runner->>Runner: wait for sends complete, then returns drain
    Runner->>Gate: after_phase(phase_id, phase_name)
    Gate->>Controller: PhaseEndGateCommand
    Controller->>Coord: gate_phase(kind=END)
    Coord->>Collectors: PhaseBaselineRequestMessage(kind=END)
    Collectors-->>Controller: PhaseBaselineAckMessage(success=True)
    Controller-->>Gate: PhaseGateGrantedResponse
    Gate-->>Runner: END gate released
    Runner->>Runner: phase transition may complete
```

## TimingManager phase flow

```mermaid
flowchart TD
    A[PhaseRunner starts phase] --> B[Generate phase_id and phase_name]
    B --> C[PhaseGateClient.before_phase]
    C --> D[SystemController handles PHASE_START_GATE]
    D --> E[BaselineCoordinator broadcasts START request]
    E --> F[Collectors scrape START baseline]
    F --> G[Collectors publish START ACKs]
    G --> H[SystemController returns PhaseGateGrantedResponse]
    H --> I[PhaseRunner issues credits]
    I --> J[PhaseRunner waits for phase completion and returns]
    J --> K[PhaseGateClient.after_phase]
    K --> L[SystemController handles PHASE_END_GATE]
    L --> M[BaselineCoordinator broadcasts END request]
    M --> N[Collectors scrape END baseline]
    N --> O[Collectors publish END ACKs]
    O --> P[SystemController returns PhaseGateGrantedResponse]
    P --> Q[PhaseRunner completes phase transition]
```
