# MTP Multi-Turn ISL Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `InferenceResultParser` so that when `--use-server-token-count` is active, each subsequent turn's reported ISL is corrected by the delta accumulated from prior MTP assistant outputs.

**Architecture:** `InferenceResultParser` already owns the tokenizer and processes every record. We add a `_session_isl_corrections: dict[str, int]` dict keyed by `x_correlation_id`. At OSL time for turn N, we tokenize the output text, compute `delta = server_completion_tokens - re_encoded_count`, and accumulate it. At ISL time for turn N+1, we add the accumulated correction to `server_prompt_tokens`. State is cleaned up when `is_final_turn=True`.

**Tech Stack:** Python 3.11+, pytest, `asyncio.to_thread` (already used in the file), existing `_parse_output_and_reasoning_texts` and `_compute_token_count` helpers.

## Global Constraints

- No changes to `Turn` model, `Worker`, `session_manager.py`, or wire format.
- `use_server_token_count=False` path is unaffected — all new code is inside the server-count branch.
- `disable_tokenization=True` skips delta computation (can't tokenize).
- All new code follows existing patterns: `Field(description=...)`, async/await, `lambda` for expensive logs.
- Run `pre-commit run --all-files` before committing.

---

## File Map

| File | Action |
|---|---|
| `src/aiperf/records/inference_result_parser.py` | Add `_session_isl_corrections`, update `_compute_server_token_counts` signature and body, update call site in `process_valid_record` |
| `tests/unit/records/test_inference_result_parser.py` | Add `TestMTPMultiTurnISLCorrection` class; update one existing direct call to `_compute_server_token_counts` |

---

### Task 1: Implement ISL correction in `InferenceResultParser`

**Files:**
- Modify: `src/aiperf/records/inference_result_parser.py:44-70` (`__init__`)
- Modify: `src/aiperf/records/inference_result_parser.py:308-310` (`process_valid_record` call site)
- Modify: `src/aiperf/records/inference_result_parser.py:489-535` (`_compute_server_token_counts`)
- Test: `tests/unit/records/test_inference_result_parser.py`

**Interfaces:**
- Consumes: `RecordContext.x_correlation_id` (str, always set), `RecordContext.is_final_turn` (bool, defaults True), `RecordContext.turn_index` (int)
- Produces: `_session_isl_corrections: dict[str, int]` (internal state); corrected `TokenCounts.input` when a prior-turn delta exists

- [ ] **Step 1: Write the failing tests**

Add a new class at the bottom of `tests/unit/records/test_inference_result_parser.py`, and update the one existing direct call to `_compute_server_token_counts`:

```python
# ── Update existing direct call (around line 405) ──────────────────────────
# The method will require a request_record argument after our change.
# Change:
#   token_counts = await setup_inference_parser._compute_server_token_counts(responses)
# To:
#   token_counts = await setup_inference_parser._compute_server_token_counts(
#       responses, request_record
#   )


# ── New test class ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestMTPMultiTurnISLCorrection:
    """ISL correction accumulates per-session when use_server_token_count=True."""

    def _make_record(
        self,
        sample_turn,
        *,
        x_correlation_id: str = "session-abc",
        turn_index: int = 0,
        is_final_turn: bool = False,
    ) -> RequestRecord:
        request_info = create_test_request_info(
            turns=[sample_turn],
            turn_index=turn_index,
        )
        request_info.x_correlation_id = x_correlation_id
        request_info.is_final_turn = is_final_turn
        return RequestRecord(
            request_info=request_info,
            model_name="test-model",
        )

    def _mock_tokenizer(self, encode_len: int):
        tok = MagicMock()
        tok.encode.side_effect = lambda _text: list(range(encode_len))
        return tok

    async def test_isl_corrected_on_subsequent_turn(
        self, server_token_parser, sample_turn
    ):
        """Turn N+1 ISL is increased by the delta from turn N's MTP discrepancy."""
        parser = server_token_parser

        # Turn 0: server says 1024 output, client re-encodes to 949 → delta = 75
        rec0 = self._make_record(sample_turn, turn_index=0, is_final_turn=False)
        setup_parser_responses(
            parser, [make_parsed_response(text="output", prompt_tokens=100, completion_tokens=1024)]
        )
        parser.get_tokenizer = AsyncMock(return_value=self._mock_tokenizer(949))
        await parser.process_valid_record(rec0)

        # Turn 1: server says prompt_tokens=2000, expect 2000 + 75 = 2075
        rec1 = self._make_record(sample_turn, turn_index=1, is_final_turn=True)
        setup_parser_responses(
            parser, [make_parsed_response(text="out1", prompt_tokens=2000, completion_tokens=100)]
        )
        result = await parser.process_valid_record(rec1)

        assert result.token_counts.input == 2075

    async def test_no_correction_for_single_turn(
        self, server_token_parser, sample_turn
    ):
        """A record with is_final_turn=True from the start has no prior correction."""
        parser = server_token_parser
        rec = self._make_record(sample_turn, turn_index=0, is_final_turn=True)
        setup_parser_responses(
            parser, [make_parsed_response(text="out", prompt_tokens=500, completion_tokens=1024)]
        )
        result = await parser.process_valid_record(rec)

        assert result.token_counts.input == 500

    async def test_zero_delta_when_counts_agree(
        self, server_token_parser, sample_turn
    ):
        """When server and re-encoded counts match (MTP off), ISL is unchanged."""
        parser = server_token_parser

        rec0 = self._make_record(sample_turn, turn_index=0, is_final_turn=False)
        setup_parser_responses(
            parser, [make_parsed_response(text="out", prompt_tokens=100, completion_tokens=1000)]
        )
        parser.get_tokenizer = AsyncMock(return_value=self._mock_tokenizer(1000))
        await parser.process_valid_record(rec0)

        rec1 = self._make_record(sample_turn, turn_index=1, is_final_turn=True)
        setup_parser_responses(
            parser, [make_parsed_response(text="out1", prompt_tokens=2000, completion_tokens=100)]
        )
        result = await parser.process_valid_record(rec1)

        assert result.token_counts.input == 2000

    async def test_session_state_cleared_after_final_turn(
        self, server_token_parser, sample_turn
    ):
        """_session_isl_corrections entry is removed after processing is_final_turn=True."""
        parser = server_token_parser
        sid = "session-cleanup"

        rec0 = self._make_record(sample_turn, x_correlation_id=sid, turn_index=0, is_final_turn=False)
        setup_parser_responses(
            parser, [make_parsed_response(text="out", prompt_tokens=100, completion_tokens=1024)]
        )
        parser.get_tokenizer = AsyncMock(return_value=self._mock_tokenizer(949))
        await parser.process_valid_record(rec0)
        assert sid in parser._session_isl_corrections

        rec1 = self._make_record(sample_turn, x_correlation_id=sid, turn_index=1, is_final_turn=True)
        setup_parser_responses(
            parser, [make_parsed_response(text="out1", prompt_tokens=2000, completion_tokens=100)]
        )
        await parser.process_valid_record(rec1)
        assert sid not in parser._session_isl_corrections

    async def test_concurrent_sessions_corrected_independently(
        self, server_token_parser, sample_turn
    ):
        """Two interleaved sessions each accumulate their own corrections."""
        parser = server_token_parser

        # Session A: delta = 75 (1024 - 949)
        rec_a0 = self._make_record(sample_turn, x_correlation_id="sid-a", turn_index=0, is_final_turn=False)
        setup_parser_responses(
            parser, [make_parsed_response(text="out-a", prompt_tokens=100, completion_tokens=1024)]
        )
        parser.get_tokenizer = AsyncMock(return_value=self._mock_tokenizer(949))
        await parser.process_valid_record(rec_a0)

        # Session B: delta = 50 (800 - 750)
        rec_b0 = self._make_record(sample_turn, x_correlation_id="sid-b", turn_index=0, is_final_turn=False)
        setup_parser_responses(
            parser, [make_parsed_response(text="out-b", prompt_tokens=200, completion_tokens=800)]
        )
        parser.get_tokenizer = AsyncMock(return_value=self._mock_tokenizer(750))
        await parser.process_valid_record(rec_b0)

        # Session A turn 1: corrected by 75 only
        rec_a1 = self._make_record(sample_turn, x_correlation_id="sid-a", turn_index=1, is_final_turn=True)
        setup_parser_responses(
            parser, [make_parsed_response(text="out", prompt_tokens=3000, completion_tokens=50)]
        )
        result_a = await parser.process_valid_record(rec_a1)
        assert result_a.token_counts.input == 3075

        # Session B turn 1: corrected by 50 only
        rec_b1 = self._make_record(sample_turn, x_correlation_id="sid-b", turn_index=1, is_final_turn=True)
        setup_parser_responses(
            parser, [make_parsed_response(text="out", prompt_tokens=4000, completion_tokens=50)]
        )
        result_b = await parser.process_valid_record(rec_b1)
        assert result_b.token_counts.input == 4050
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/records/test_inference_result_parser.py::TestMTPMultiTurnISLCorrection -v
```

Expected: all 5 new tests FAIL (method signature mismatch and missing logic).

- [ ] **Step 3: Update the existing direct call to `_compute_server_token_counts`**

In `tests/unit/records/test_inference_result_parser.py` around line 405, change:

```python
token_counts = await setup_inference_parser._compute_server_token_counts(
    responses
)
```

to:

```python
token_counts = await setup_inference_parser._compute_server_token_counts(
    responses, request_record
)
```

The `request_record` fixture is already defined in the same file. Add it to the test method's parameter list if it isn't there yet:

```python
async def test_output_excludes_reasoning_tokens(
    self,
    setup_inference_parser,
    request_record,       # add this
    completion_tokens,
    reasoning_tokens,
    expected_output,
):
```

- [ ] **Step 4: Implement the changes in `inference_result_parser.py`**

**4a. Add `_session_isl_corrections` to `__init__`** (after `self.disable_tokenization` assignment, around line 64):

```python
self._session_isl_corrections: dict[str, int] = {}
```

**4b. Update `_compute_server_token_counts` signature** (line 489):

```python
async def _compute_server_token_counts(
    self, responses: list[ParsedResponse], request_record: RequestRecord
) -> TokenCounts:
```

**4c. Replace the body of `_compute_server_token_counts`** with the corrected version.
The existing logic stays intact; new code is added after the `token_counts` object is built and before the warning check:

```python
async def _compute_server_token_counts(
    self, responses: list[ParsedResponse], request_record: RequestRecord
) -> TokenCounts:
    """Compute token counts using server-provided usage fields.

    Walks `responses` ONCE to find the last chunk with usage and reads
    all token counts from that single Usage. This guarantees the input,
    reasoning, and output counts are mutually consistent (all from the
    same chunk), and it avoids three redundant walks of the same list.

    When ``use_server_token_count`` is active and the record belongs to a
    multi-turn session, accumulates a per-session ISL correction so that
    subsequent turns' reported ISL reflects the server-reported output
    token count rather than the re-tokenized wire count. This corrects
    the MTP tokenization artifact where ``completion_tokens`` and the
    re-encoded text token count diverge.

    Args:
        responses: List of parsed responses from the server
        request_record: The originating request record, used to read
            session identity and turn lifecycle fields.

    Returns:
        TokenCounts populated with server-reported values. All fields
        are None if no chunk had usage at all.
    """
    usage = find_last_non_empty_usage(responses)
    if usage is None:
        input_token_count = None
        reasoning_token_count = None
        output_token_count = None
    else:
        reasoning_token_count = usage.reasoning_tokens
        output_token_count = self._server_output_minus_reasoning(
            usage.completion_tokens, reasoning_token_count
        )
        input_token_count = usage.prompt_tokens

    # Apply ISL correction accumulated from prior turns in this session.
    ctx = request_record.request_info
    session_id = ctx.x_correlation_id if ctx is not None else None
    is_final_turn = ctx.is_final_turn if ctx is not None else True

    if session_id and input_token_count is not None:
        correction = self._session_isl_corrections.get(session_id, 0)
        if correction:
            input_token_count += correction

    # Accumulate delta for future turns in this session (skip on the final
    # turn — there is no next turn to correct, and skipping avoids wasted
    # tokenization).
    if (
        session_id
        and not is_final_turn
        and output_token_count is not None
        and not self.disable_tokenization
        and request_record.model_name is not None
    ):
        try:
            output_texts, _ = self._parse_output_and_reasoning_texts(responses)
            if output_texts:
                tokenizer = await self.get_tokenizer(request_record.model_name)
                re_encoded = await self._compute_token_count(tokenizer, output_texts) or 0
                delta = output_token_count - re_encoded
                if delta:
                    self._session_isl_corrections[session_id] = (
                        self._session_isl_corrections.get(session_id, 0) + delta
                    )
        except Exception as exc:
            self.warning(
                lambda exc=exc: (
                    f"Failed to compute ISL correction delta for session "
                    f"'{session_id}': {exc!r}. ISL for subsequent turns may be undercounted."
                )
            )

    # Release session state after the final turn to prevent unbounded growth.
    if session_id and is_final_turn:
        self._session_isl_corrections.pop(session_id, None)

    token_counts = TokenCounts(
        input=input_token_count,
        reasoning=reasoning_token_count,
        output=output_token_count,
    )

    # Warn if server provided no usage information
    if (
        token_counts.input is None
        and token_counts.output is None
        and token_counts.reasoning is None
    ):
        self.warning(
            "Server did not provide token usage information. Token count metrics will be unavailable. "
            "Verify that your API endpoint supports usage reporting (stream_options are automatically configured for OpenAI-compatible endpoints)."
        )

    return token_counts
```

**4d. Update the call site in `process_valid_record`** (line 310):

```python
# Before:
token_counts = await self._compute_server_token_counts(resp)

# After:
token_counts = await self._compute_server_token_counts(resp, request_record)
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
uv run pytest tests/unit/records/test_inference_result_parser.py::TestMTPMultiTurnISLCorrection -v
```

Expected: all 5 new tests PASS.

- [ ] **Step 6: Run the full existing test file to ensure nothing regressed**

```bash
uv run pytest tests/unit/records/test_inference_result_parser.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Run the full unit suite**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: all tests PASS.

- [ ] **Step 8: Run pre-commit**

```bash
pre-commit run --all-files
```

Fix any issues raised before committing.

- [ ] **Step 9: Commit**

```bash
git add src/aiperf/records/inference_result_parser.py \
        tests/unit/records/test_inference_result_parser.py
git commit -s -m "fix(records): correct multi-turn ISL when use-server-token-count is active

When MTP with synthetic rejection sampling is enabled, vLLM's
completion_tokens diverges from the re-encoded text token count (e.g.
1024 vs 949). With --use-server-token-count, the current turn's OSL is
already correct. This fix accumulates the per-turn delta in
InferenceResultParser._session_isl_corrections (keyed by
x_correlation_id) at OSL processing time, and adds it to
usage.prompt_tokens when reporting ISL for subsequent turns.

No changes to the Turn model, Worker, or wire format.

Fixes AIP-1013."
```
