<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Session Test Helpers

Comprehensive helper functions for testing multi-turn sessions, sticky routing, and request cancellation.

## Quick Reference

| Helper | Purpose | Lines Saved |
|--------|---------|-------------|
| `group_records_by_session()` | Group JSONL by session ID | 5-8 |
| `assert_sticky_routing()` | Validate sticky routing | 15-20 |
| `assert_jsonl_turns_sequential()` | Validate sequential turns | 12-15 |
| `count_cancelled_requests()` | Count cancellations | 3-5 |
| `validate_cancellation_errors()` | Validate error details | 5-8 |
| `assert_sessions_complete()` | Check all turns present | 10-15 |
| `get_session_worker_mapping()` | Get session→worker map | 5-10 |

## Import Statement

```python
from tests.component_integration.conftest import (
    group_records_by_session,
    assert_sticky_routing,
    assert_jsonl_turns_sequential,
    count_cancelled_requests,
    validate_cancellation_errors,
    assert_sessions_complete,
    get_session_worker_mapping,
)
```

## Common Usage Patterns

### Pattern 1: Basic Multi-Turn Session Test

```python
def test_multi_turn_sessions(self, cli: AIPerfCLI):
    result = cli.run_sync(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --num-sessions 10 \
            --session-turns-mean 5 \
            --session-turns-stddev 0 \
            ...
        """
    )

    # Validate sticky routing and turn sequences
    assert_sticky_routing(result.jsonl)
    assert_jsonl_turns_sequential(result.jsonl)

    # Verify session completeness
    assert_sessions_complete(result.jsonl, expected_turns=5)
```

### Pattern 2: Cancellation Test

```python
def test_request_cancellation(self, cli: AIPerfCLI):
    result = cli.run_sync(
        f"""
        aiperf profile \
            --request-cancellation-rate 30 \
            ...
        """
    )

    # Validate cancellation errors
    validate_cancellation_errors(result.jsonl)

    # Check cancellation rate
    cancelled_count = count_cancelled_requests(result.jsonl)
    total_count = len(result.jsonl)
    cancellation_rate = cancelled_count / total_count
    assert 0.20 < cancellation_rate < 0.40  # ~30% ±10%
```

### Pattern 3: Load Distribution Analysis

```python
def test_worker_distribution(self, cli: AIPerfCLI):
    result = cli.run_sync(
        f"""
        aiperf profile \
            --num-sessions 100 \
            --workers-max 5 \
            ...
        """
    )

    # Analyze worker distribution
    worker_map = get_session_worker_mapping(result.jsonl)
    workers_used = set(worker_map.values())

    assert len(workers_used) == 5, "All workers should be used"

    # Verify sticky routing
    assert_sticky_routing(result.jsonl)
```

### Pattern 4: Advanced Session Analysis

```python
def test_session_detailed_analysis(self, cli: AIPerfCLI):
    result = cli.run_sync(...)

    # Group sessions for detailed analysis
    sessions = group_records_by_session(result.jsonl)

    # Analyze each session
    for session_id, records in sessions.items():
        # Check turn count
        assert len(records) == expected_turns

        # Check all turns went to same worker
        worker_ids = {r.metadata.worker_id for r in records}
        assert len(worker_ids) == 1

        # Check turn indices are sequential
        turns = [r.metadata.turn_index for r in records]
        assert turns == list(range(len(records)))

    # Or use convenience helpers
    assert_sticky_routing(result.jsonl)
    assert_jsonl_turns_sequential(result.jsonl)
```

## Detailed API

### `group_records_by_session(jsonl_records) -> dict[str, list]`

Groups JSONL records by session ID (x_correlation_id). Records within each session are sorted by turn_index.

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects from result.jsonl

**Returns:**
- Dictionary mapping x_correlation_id to list of records

**Example:**
```python
sessions = group_records_by_session(result.jsonl)
for session_id, records in sessions.items():
    print(f"Session {session_id}: {len(records)} turns")
```

---

### `assert_sticky_routing(jsonl_records) -> dict[str, list]`

Validates that all turns within each session went to the same worker.

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects

**Returns:**
- Dictionary mapping x_correlation_id to list of records (if validation passes)

**Raises:**
- AssertionError: If any session violated sticky routing

**Example:**
```python
sessions = assert_sticky_routing(result.jsonl)
assert len(sessions) == expected_num_sessions
```

**Alias:** `validate_sticky_routing` (backward compatibility)

---

### `assert_jsonl_turns_sequential(jsonl_records) -> None`

Validates that turn indices are sequential (0, 1, 2, ...) within each session.

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects

**Raises:**
- AssertionError: If any session has non-sequential turns

**Example:**
```python
assert_jsonl_turns_sequential(result.jsonl)
```

---

### `count_cancelled_requests(jsonl_records) -> int`

Counts the number of cancelled requests.

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects

**Returns:**
- Number of cancelled requests

**Example:**
```python
cancelled_count = count_cancelled_requests(result.jsonl)
assert cancelled_count > 0, "Expected some cancellations"
```

---

### `validate_cancellation_errors(jsonl_records) -> None`

Validates that all cancelled requests have proper error details:
- error is not None
- error.code == 499
- error.type == "RequestCancellationError"

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects

**Raises:**
- AssertionError: If any cancelled request has invalid error details

**Example:**
```python
validate_cancellation_errors(result.jsonl)
```

---

### `assert_sessions_complete(jsonl_records, expected_turns: int) -> None`

Validates that all sessions have the expected number of turns.

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects
- `expected_turns`: Expected number of turns per session

**Raises:**
- AssertionError: If any session doesn't have expected turns

**Example:**
```python
assert_sessions_complete(result.jsonl, expected_turns=5)
```

---

### `get_session_worker_mapping(jsonl_records) -> dict[str, str]`

Returns a mapping of session IDs to their assigned worker IDs.

**Args:**
- `jsonl_records`: List of AIPerfMetricRecord objects

**Returns:**
- Dictionary mapping x_correlation_id to worker_id

**Example:**
```python
worker_map = get_session_worker_mapping(result.jsonl)
workers_used = set(worker_map.values())
assert len(workers_used) == expected_num_workers
```

---

### `assert_turns_sequential(sessions: dict[str, list]) -> None`

Validates turn indices are sequential within pre-grouped sessions.

**Args:**
- `sessions`: Dictionary from group_records_by_session()

**Raises:**
- AssertionError: If any session has non-sequential turns

**Example:**
```python
sessions = group_records_by_session(result.jsonl)
assert_turns_sequential(sessions)
```

**Note:** Use `assert_jsonl_turns_sequential()` if you don't already have grouped sessions.

## Testing Guidelines

1. **Always validate sticky routing** in multi-turn tests
2. **Always validate turn sequences** to catch missing turns
3. **Use cancellation helpers** in any test with `--request-cancellation-rate`
4. **Analyze worker distribution** in load balancing tests
5. **Group sessions first** if you need multiple analyses on the same data

## Error Messages

Helpers provide detailed, actionable error messages:

```
AssertionError: Session abc123 violated sticky routing:
workers={'worker_1', 'worker_2'}, turns=[0, 1, 2, 3]
```

```
AssertionError: Session xyz789 has non-sequential turns:
got [0, 1, 3], expected [0, 1, 2]
```

```
AssertionError: Cancelled request req_456 has wrong error code:
500 (expected 499)
```

## See Also

- [HELPER_USAGE.md](./HELPER_USAGE.md) - Comprehensive usage examples
- [/SESSION_HELPERS_SUMMARY.md](../../../SESSION_HELPERS_SUMMARY.md) - Implementation summary
- [/tests/component_integration/conftest.py](../conftest.py) - Source code
