# AIP-904: Support Configurable Per-Conversation Session Affinity Header

**Linear:** [AIP-904](https://linear.app/nvidia/issue/AIP-904/support-configurable-per-conversation-session-affinity-header) — Assignee: Elias Bermudez — Status: Todo — Team: AIPerf
**Branch:** `dbermudez/aip-904-support-configurable-per-conversation-session-affinity`

---

## Request (from Michael Shin, CCluster team)

The CCluster team works with Shopify to run their models in production. They added session affinity to their infrastructure: requests with the same session header are consistently routed to the same Kubernetes pod. They want to use aiperf to benchmark and demonstrate that behavior under concurrent multi-turn load.

aiperf already generates a stable per-conversation `X-Correlation-ID` and sends it on every request. It also supports static custom headers via `--header`. What is needed is a way to send that same per-conversation identifier under a **configurable header name** (e.g. `X-Session-ID`).

---

## Expected Behavior

- Each multi-turn conversation gets a unique session header value.
- All turns in the same conversation reuse that same value.
- Concurrent conversations receive different values.
- Existing static `--header` behavior continues to work for auth and fixed headers.

---

## Motivation

- Models Shopify's production session-affinity traffic directly.
- Validates that each session is consistently routed to the same backend pod.
- Makes aiperf more useful for Shopify's production benchmarking workflows.

---

## Relevant Background in the Codebase

aiperf already has two related mechanisms:

1. **`X-Correlation-ID`** — a stable per-conversation identifier already generated and sent on every request. The value is consistent across turns in the same conversation.
2. **`--header`** — CLI flag for static custom headers (auth tokens, fixed values). Documented in `docs/cli-options.md`.

The new feature bridges these two: reuse the existing per-conversation identifier but send it under a user-specified header name instead of (or in addition to) `X-Correlation-ID`.

---

## CLI Interface

```bash
aiperf benchmark \
  --session-header X-Session-ID \
  ...
```

When `--session-header` is set, the per-conversation UUID is sent under that name **instead of** `X-Correlation-ID`. Without `--session-header`, behavior is identical to today.

---

## Key Files to Investigate

| File | Relevance |
|---|---|
| `src/aiperf/common/config/input_config.py` | `--header` and `--extra-inputs` CLI parsing |
| `src/aiperf/workers/inference_client.py` | Where per-request HTTP headers are assembled and sent |
| `src/aiperf/workers/endpoints/openai_chat.py` | Endpoint that currently sends `X-Correlation-ID` |
| `src/aiperf/common/models/model_endpoint_info.py` | `EndpointInfo` — holds static header config |
| `src/aiperf/dataset/loader/models.py` | Request/session data models |
| `docs/cli-options.md` | Auto-generated CLI reference (update after adding flag) |
| `docs/environment-variables.md` | Auto-generated env var docs (update if adding `AIPERF_*` var) |

---

## Implementation

Four files changed:

| File | Change |
|---|---|
| `src/aiperf/common/config/input_config.py` | `session_header: str \| None` field with `--session-header` CLI alias |
| `src/aiperf/common/models/model_endpoint_info.py` | `session_header: str \| None` on `EndpointInfo`; wired in `from_user_config` |
| `src/aiperf/transports/base_transports.py` | `build_headers` uses configured name or falls back to `"X-Correlation-ID"` |
| `tests/unit/transports/test_base_transport.py` | Unit tests for rename, fallback, uniqueness, and static header isolation |

The value sent is always the raw per-conversation UUID (no template support; can be extended later).

---

## Testing

- Unit test: single conversation sends consistent header value across turns.
- Unit test: two concurrent conversations send different header values.
- Unit test: `--header` static headers are unaffected.
- Test location: `tests/unit/workers/` or `tests/unit/common/config/`.

---

## Slack Thread

Original request: https://nvidia.slack.com/archives/C01KB8WDXJP/p1778861002945479
