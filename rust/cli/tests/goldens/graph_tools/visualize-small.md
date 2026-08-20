## Graph topology

```mermaid
flowchart LR
  start(["START"])
  n0["n00000000"]:::llm
  n1["n00000001"]:::llm
  n2["n00000003"]:::llm
  n3["n00000002"]:::llm
  n4["n00000004"]:::llm
  end_node(["END"])
  start -->|"schedule completion"| n0
  n0 -->|"schedule completion"| n1
  n0 -->|"schedule completion"| n2
  n1 -->|"schedule completion"| n3
  n2 -->|"schedule completion"| n4
  n3 -->|"schedule completion"| end_node
  n4 -->|"schedule completion"| end_node
  classDef llm fill:#76b900,color:#000,stroke:#333
  classDef tool fill:#5b8def,color:#fff,stroke:#333
  classDef invalid fill:#d9534f,color:#fff,stroke:#333
```

This is the selected trace's resolved Graph-IR topology.

## Resolved plan

- Source: $REPO/tests/fixtures/dag/small.dag.jsonl
- Format: dag_jsonl
- Trace: root
- Driver: static_graph

## Illustrative readiness waves

| Wave | Nodes ready | Trigger |
|---:|---|---|
| 0 | n00000000 | START |
| 1 | n00000001, n00000003 | completed: n00000000 |
| 2 | n00000002, n00000004 | completed: n00000001, n00000003 |
