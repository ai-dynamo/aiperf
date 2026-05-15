# `extra_body` Support in Additional Formats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire per-turn `Turn.extra_body` into every endpoint formatter that builds a JSON request body (except `raw_endpoint` and `template_endpoint`) and into the `mooncake_trace`, `single_turn`, and `multi_turn` dataset loaders, matching `openai_chat`'s existing top-level shallow-merge convention.

**Architecture:** Two parallel surfaces. (1) Loaders learn to read `extra_body` from input rows and stamp it on `Turn`. (2) Each formatter merges `extra_body` last (after formatter defaults, after `model_endpoint.endpoint.extra`) so user-provided keys win. Inheritance reuse: `BaseRankingsEndpoint` covers Cohere / NIM / HF-TEI rankings; `EmbeddingsEndpoint._build_payload` covers OpenAI + NIM embeddings; `ChatEmbeddingsEndpoint` inherits from `ChatEndpoint` and gets it for free.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, `uv`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-05-09-extra-body-additional-formats-design.md`

---

## File Structure

**Models (Pydantic schema):**
- Modify: `src/aiperf/dataset/loader/models.py` — add `extra_body` to `MooncakeTrace` and `SingleTurn`. (`MultiTurn.turns: list[SingleTurn]` inherits at no extra cost.)

**Loaders:**
- Modify: `src/aiperf/dataset/loader/mooncake_trace.py` — propagate `extra_body` in `_build_turn` across all three branches.
- Modify: `src/aiperf/dataset/loader/single_turn.py` — pass `extra_body` to `Turn(...)`.
- Modify: `src/aiperf/dataset/loader/multi_turn.py` — pass `extra_body` to `Turn(...)`.

**Endpoint formatters (9 actual edits cover 13 formatters via inheritance):**
- Modify: `src/aiperf/endpoints/base_rankings_endpoint.py` — covers `cohere_rankings`, `nim_rankings`, `hf_tei_rankings`.
- Modify: `src/aiperf/endpoints/openai_embeddings.py::_build_payload` — covers `EmbeddingsEndpoint`, `NIMEmbeddingsEndpoint`.
- Modify: `src/aiperf/endpoints/openai_completions.py`.
- Modify: `src/aiperf/endpoints/openai_responses.py` (multi-turn; uses `_latest_turn_attr`).
- Modify: `src/aiperf/endpoints/openai_image_generation.py`.
- Modify: `src/aiperf/endpoints/openai_video_generation.py`.
- Modify: `src/aiperf/endpoints/nim_image_retrieval.py`.
- Modify: `src/aiperf/endpoints/huggingface_generate.py`.
- Modify: `src/aiperf/endpoints/solido_rag.py`.

**Tests:**
- Modify: `tests/unit/dataset/loader/test_mooncake_payload_mode.py` (add `extra_body` cases).
- Modify: `tests/unit/dataset/loader/test_mooncake_trace_messages.py` (add messages-mode `extra_body` case).
- Modify: `tests/unit/dataset/loader/test_single_turn.py` (add `extra_body` case).
- Modify: `tests/unit/dataset/loader/test_multi_turn.py` (add per-inner-turn `extra_body` case).
- Modify: `tests/unit/endpoints/test_cohere_rankings_endpoint.py` (covers base-class merge).
- Modify: `tests/unit/endpoints/test_completions_endpoint.py`.
- Modify: `tests/unit/endpoints/test_responses_endpoint.py` (multi-turn case).
- Modify: `tests/unit/endpoints/test_embeddings_endpoint.py` (also tests NIM via base).
- Modify: `tests/unit/endpoints/test_image_generation_endpoint.py`.
- Modify: `tests/unit/endpoints/test_video_generation_endpoint.py`.
- Modify: `tests/unit/endpoints/test_nim_image_retrieval_endpoint.py`.
- Modify: `tests/unit/endpoints/test_huggingface_generate.py`.
- Modify: `tests/unit/endpoints/test_solido_rag.py`.

**Docs:**
- Modify: `docs/dev/patterns.md` — add `extra_body` merge note.
- Modify: mooncake / single-turn / multi-turn schema reference docs (located via grep in Task 15).

---

## Task 1: Add `extra_body` field to `MooncakeTrace` and `SingleTurn` models

**Files:**
- Modify: `src/aiperf/dataset/loader/models.py`
- Modify: `tests/unit/dataset/loader/test_mooncake_payload_mode.py`
- Modify: `tests/unit/dataset/loader/test_single_turn.py`

- [ ] **Step 1: Write failing tests**

In `tests/unit/dataset/loader/test_mooncake_payload_mode.py`, append to the existing `TestMooncakeTracePayloadMode` class (or add a new top-level test):

```python
def test_mooncake_trace_accepts_extra_body():
    t = MooncakeTrace(
        text_input="Hello",
        extra_body={"vendor_top_k": 5, "ignore_eos": True},
    )
    assert t.extra_body == {"vendor_top_k": 5, "ignore_eos": True}


def test_mooncake_trace_extra_body_defaults_to_none():
    t = MooncakeTrace(text_input="Hello")
    assert t.extra_body is None
```

In `tests/unit/dataset/loader/test_single_turn.py`, after the existing `test_create_with_text_only`:

```python
def test_single_turn_accepts_extra_body(self):
    data = SingleTurn(
        text="What is deep learning?",
        extra_body={"top_p": 0.9, "seed": 42},
    )
    assert data.extra_body == {"top_p": 0.9, "seed": 42}


def test_single_turn_extra_body_defaults_to_none(self):
    data = SingleTurn(text="What is deep learning?")
    assert data.extra_body is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/dataset/loader/test_mooncake_payload_mode.py::test_mooncake_trace_accepts_extra_body tests/unit/dataset/loader/test_single_turn.py::TestSingleTurn::test_single_turn_accepts_extra_body -v
```

Expected: FAIL with `AttributeError` / `ValidationError` (extra fields forbidden) on both.

- [ ] **Step 3: Add `extra_body` to `MooncakeTrace` model**

In `src/aiperf/dataset/loader/models.py`, in the `MooncakeTrace` class, after the `session_id` field (line ~273) and BEFORE the first `@model_validator`:

```python
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-turn extra fields shallow-merged into the request body at "
            "dispatch time, matching the OpenAI SDK's extra_body convention. "
            "Keys override formatter defaults on collision."
        ),
    )
```

- [ ] **Step 4: Add `extra_body` to `SingleTurn` model**

In the same file, in the `SingleTurn` class, after the `output_length` field (line ~84) and BEFORE the first `@model_validator`:

```python
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-turn extra fields shallow-merged into the request body at "
            "dispatch time, matching the OpenAI SDK's extra_body convention. "
            "Keys override formatter defaults on collision."
        ),
    )
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS. The two new tests pass; nothing else regresses.

- [ ] **Step 6: Format and commit**

```bash
ruff format src/aiperf/dataset/loader/models.py tests/unit/dataset/loader/test_mooncake_payload_mode.py tests/unit/dataset/loader/test_single_turn.py
ruff check --fix src/aiperf/dataset/loader/models.py tests/unit/dataset/loader/test_mooncake_payload_mode.py tests/unit/dataset/loader/test_single_turn.py
git add src/aiperf/dataset/loader/models.py tests/unit/dataset/loader/test_mooncake_payload_mode.py tests/unit/dataset/loader/test_single_turn.py
git commit -s -m "feat(dataset): add extra_body field to MooncakeTrace and SingleTurn models"
```

---

## Task 2: Wire mooncake loader to propagate `extra_body` to `Turn`

**Files:**
- Modify: `src/aiperf/dataset/loader/mooncake_trace.py:112-128` (`_build_turn`)
- Modify: `tests/unit/dataset/loader/test_mooncake_payload_mode.py`
- Modify: `tests/unit/dataset/loader/test_mooncake_trace_messages.py`

- [ ] **Step 1: Write failing test for payload-mode propagation**

Append to `tests/unit/dataset/loader/test_mooncake_payload_mode.py` inside `TestMooncakeTraceLoaderPayload`:

```python
    def test_extra_body_propagates_to_turn_in_payload_mode(
        self,
        tmp_path: Path,
        default_user_config: UserConfig,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "timestamp": 0,
                        "payload": {"prompt": "p", "max_tokens": 40},
                        "extra_body": {"vendor_x": 1, "stream": False},
                    }
                )
            )
            f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        turn = conversations[0].turns[0]
        assert turn.extra_body == {"vendor_x": 1, "stream": False}
```

- [ ] **Step 2: Write failing test for messages-mode and text-input-mode**

In `tests/unit/dataset/loader/test_mooncake_trace_messages.py`, find an appropriate test class (or create `TestMooncakeTraceExtraBody`) and add:

```python
class TestMooncakeTraceExtraBody:
    def test_extra_body_propagates_to_turn_in_messages_mode(
        self,
        tmp_path: Path,
        default_user_config,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "extra_body": {"top_k": 7},
                    }
                )
            )
            f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        turn = conversations[0].turns[0]
        assert turn.extra_body == {"top_k": 7}

    def test_extra_body_propagates_to_turn_in_text_input_mode(
        self,
        tmp_path: Path,
        default_user_config,
        mock_prompt_generator,
    ):
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "text_input": "hi",
                        "extra_body": {"min_tokens": 50},
                    }
                )
            )
            f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            user_config=default_user_config,
            prompt_generator=mock_prompt_generator,
        )
        conversations = loader.convert_to_conversations(loader.load_dataset())
        turn = conversations[0].turns[0]
        assert turn.extra_body == {"min_tokens": 50}
```

(If `test_mooncake_trace_messages.py` doesn't already import the same fixtures as `test_mooncake_payload_mode.py`, copy `default_user_config` and `mock_prompt_generator` fixtures from there into the file — or move them to `tests/unit/dataset/loader/conftest.py` if both files use them.)

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/unit/dataset/loader/test_mooncake_payload_mode.py::TestMooncakeTraceLoaderPayload::test_extra_body_propagates_to_turn_in_payload_mode tests/unit/dataset/loader/test_mooncake_trace_messages.py::TestMooncakeTraceExtraBody -v
```

Expected: FAIL — `turn.extra_body` is `None` because the loader doesn't propagate it yet.

- [ ] **Step 4: Wire `_build_turn` to propagate `extra_body`**

In `src/aiperf/dataset/loader/mooncake_trace.py`, replace the `_build_turn` method (lines 112-128) with:

```python
    def _build_turn(self, trace: MooncakeTrace, prompt: str) -> Turn:
        if trace.payload is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=trace.delay,
                max_tokens=trace.output_length,
                raw_payload=trace.payload,
                extra_body=trace.extra_body,
            )
        if trace.messages is not None:
            return Turn(
                timestamp=trace.timestamp,
                delay=trace.delay,
                max_tokens=trace.output_length,
                raw_messages=trace.messages,
                raw_tools=trace.tools,
                extra_body=trace.extra_body,
            )
        turn = super()._build_turn(trace, prompt)
        if trace.extra_body is not None:
            turn.extra_body = trace.extra_body
        return turn
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS — three new tests pass, prior tests still pass.

- [ ] **Step 6: Format and commit**

```bash
ruff format src/aiperf/dataset/loader/mooncake_trace.py tests/unit/dataset/loader/test_mooncake_payload_mode.py tests/unit/dataset/loader/test_mooncake_trace_messages.py
ruff check --fix src/aiperf/dataset/loader/mooncake_trace.py tests/unit/dataset/loader/test_mooncake_payload_mode.py tests/unit/dataset/loader/test_mooncake_trace_messages.py
git add src/aiperf/dataset/loader/mooncake_trace.py tests/unit/dataset/loader/test_mooncake_payload_mode.py tests/unit/dataset/loader/test_mooncake_trace_messages.py
git commit -s -m "feat(dataset): propagate extra_body to Turn in mooncake_trace loader"
```

---

## Task 3: Wire `single_turn` loader to propagate `extra_body` to `Turn`

**Files:**
- Modify: `src/aiperf/dataset/loader/single_turn.py:140-151`
- Modify: `tests/unit/dataset/loader/test_single_turn.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/dataset/loader/test_single_turn.py` (outside the `TestSingleTurn` class — at module top level, since the existing file mixes class- and function-style tests):

```python
def test_single_turn_loader_propagates_extra_body_to_turn(tmp_path):
    path = tmp_path / "single.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "Hello",
                "extra_body": {"vendor_a": 1, "vendor_b": "x"},
            }
        )
        + "\n"
    )
    loader = SingleTurnDatasetLoader(filename=path)
    conversations = loader.convert_to_conversations(loader.load_dataset())
    turn = conversations[0].turns[0]
    assert turn.extra_body == {"vendor_a": 1, "vendor_b": "x"}
```

(If the file already imports `json` and `SingleTurnDatasetLoader` near the top, no new imports needed. If not, the imports `import json` and `from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader` are already present.)

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/dataset/loader/test_single_turn.py::test_single_turn_loader_propagates_extra_body_to_turn -v
```

Expected: FAIL — `turn.extra_body` is `None`.

- [ ] **Step 3: Add `extra_body` to `Turn(...)` constructor**

In `src/aiperf/dataset/loader/single_turn.py`, in `convert_to_conversations`, change the `Turn(...)` constructor (lines ~141-150) to:

```python
                conversation.turns.append(
                    Turn(
                        texts=media[MediaType.TEXT],
                        images=media[MediaType.IMAGE],
                        audios=media[MediaType.AUDIO],
                        videos=media[MediaType.VIDEO],
                        timestamp=single_turn.timestamp,
                        delay=single_turn.delay,
                        role=single_turn.role,
                        max_tokens=single_turn.output_length,
                        extra_body=single_turn.extra_body,
                    )
                )
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/dataset/loader/single_turn.py tests/unit/dataset/loader/test_single_turn.py
ruff check --fix src/aiperf/dataset/loader/single_turn.py tests/unit/dataset/loader/test_single_turn.py
git add src/aiperf/dataset/loader/single_turn.py tests/unit/dataset/loader/test_single_turn.py
git commit -s -m "feat(dataset): propagate extra_body to Turn in single_turn loader"
```

---

## Task 4: Wire `multi_turn` loader to propagate `extra_body` per inner turn

**Files:**
- Modify: `src/aiperf/dataset/loader/multi_turn.py:162-173`
- Modify: `tests/unit/dataset/loader/test_multi_turn.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/dataset/loader/test_multi_turn.py` at module level:

```python
def test_multi_turn_loader_propagates_per_inner_turn_extra_body(tmp_path):
    path = tmp_path / "multi.jsonl"
    path.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "turns": [
                    {"text": "Hello", "extra_body": {"vendor_a": 1}},
                    {"text": "Hi", "extra_body": {"vendor_b": 2}},
                    {"text": "Bye"},  # no extra_body
                ],
            }
        )
        + "\n"
    )
    loader = MultiTurnDatasetLoader(filename=path)
    conversations = loader.convert_to_conversations(loader.load_dataset())
    turns = conversations[0].turns
    assert turns[0].extra_body == {"vendor_a": 1}
    assert turns[1].extra_body == {"vendor_b": 2}
    assert turns[2].extra_body is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/dataset/loader/test_multi_turn.py::test_multi_turn_loader_propagates_per_inner_turn_extra_body -v
```

Expected: FAIL — all three turn `extra_body` values are `None`.

- [ ] **Step 3: Add `extra_body` to `Turn(...)` constructor**

In `src/aiperf/dataset/loader/multi_turn.py`, in `convert_to_conversations`, change the `Turn(...)` constructor (lines ~163-172) to:

```python
                    conversation.turns.append(
                        Turn(
                            texts=media[MediaType.TEXT],
                            images=media[MediaType.IMAGE],
                            audios=media[MediaType.AUDIO],
                            videos=media[MediaType.VIDEO],
                            timestamp=single_turn.timestamp,
                            delay=single_turn.delay,
                            role=single_turn.role,
                            max_tokens=single_turn.output_length,
                            extra_body=single_turn.extra_body,
                        )
                    )
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/dataset/loader/multi_turn.py tests/unit/dataset/loader/test_multi_turn.py
ruff check --fix src/aiperf/dataset/loader/multi_turn.py tests/unit/dataset/loader/test_multi_turn.py
git add src/aiperf/dataset/loader/multi_turn.py tests/unit/dataset/loader/test_multi_turn.py
git commit -s -m "feat(dataset): propagate extra_body to Turn in multi_turn loader"
```

---

## Task 5: `BaseRankingsEndpoint` merges `extra_body` (covers Cohere / NIM / HF-TEI rankings)

**Files:**
- Modify: `src/aiperf/endpoints/base_rankings_endpoint.py:30-98`
- Modify: `tests/unit/endpoints/test_cohere_rankings_endpoint.py`

- [ ] **Step 1: Write failing test**

In `tests/unit/endpoints/test_cohere_rankings_endpoint.py`, append a method to the existing `TestCohereRankingsEndpoint` class (which already has `model_endpoint`, `converter`, and `basic_turn` fixtures):

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        turn = Turn(
            texts=[
                Text(name="query", contents=["q"]),
                Text(name="passages", contents=["p1", "p2"]),
            ],
            extra_body={"vendor_top_k": 5, "model": "override-model"},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_top_k"] == 5
        # extra_body wins over formatter-built `model` key.
        assert payload["model"] == "override-model"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_cohere_rankings_endpoint.py::TestCohereRankingsEndpoint::test_extra_body_shallow_merges_into_payload -v
```

Expected: FAIL — `vendor_top_k` not in payload.

- [ ] **Step 3: Merge `extra_body` in base rankings formatter**

In `src/aiperf/endpoints/base_rankings_endpoint.py`, in `BaseRankingsEndpoint.format_payload`, replace the block at lines ~93-98:

```python
        payload = self.build_payload(query_text, passage_texts, model_name)
        if extra:
            payload.update(extra)

        self.trace(lambda: f"Formatted rankings payload: {payload}")
        return payload
```

with:

```python
        payload = self.build_payload(query_text, passage_texts, model_name)
        if extra:
            payload.update(extra)

        if turn.extra_body:
            payload.update(turn.extra_body)

        self.trace(lambda: f"Formatted rankings payload: {payload}")
        return payload
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/endpoints/base_rankings_endpoint.py tests/unit/endpoints/test_cohere_rankings_endpoint.py
ruff check --fix src/aiperf/endpoints/base_rankings_endpoint.py tests/unit/endpoints/test_cohere_rankings_endpoint.py
git add src/aiperf/endpoints/base_rankings_endpoint.py tests/unit/endpoints/test_cohere_rankings_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in BaseRankingsEndpoint"
```

---

## Task 6: `openai_completions` merges `extra_body`

**Files:**
- Modify: `src/aiperf/endpoints/openai_completions.py:43-69`
- Modify: `tests/unit/endpoints/test_completions_endpoint.py`

- [ ] **Step 1: Write failing test**

Open `tests/unit/endpoints/test_completions_endpoint.py` and identify the existing test class / fixture pattern (mirrors the pattern in `test_cohere_rankings_endpoint.py`). Add a test method to the primary test class:

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        turn = Turn(
            texts=[Text(contents=["hello"])],
            extra_body={"vendor_top_k": 5, "stream": True},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_top_k"] == 5
        # extra_body wins over formatter-built `stream` key (default streaming=False).
        assert payload["stream"] is True
```

If the test file's existing fixtures use different names (`endpoint`, `request_info_basic`, etc.), adapt the new test to those exact names. If `Text` and `Turn` are already imported at top, drop the local import.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_completions_endpoint.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 3: Merge `extra_body` in completions formatter**

In `src/aiperf/endpoints/openai_completions.py::format_payload`, after the existing `if extra: payload.update(extra)` block (line 53) and BEFORE the `stream_options` block (line 55), insert:

```python
        if turn.extra_body:
            payload.update(turn.extra_body)
```

The full block becomes:

```python
        if extra:
            payload.update(extra)

        if turn.extra_body:
            payload.update(turn.extra_body)

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/endpoints/openai_completions.py tests/unit/endpoints/test_completions_endpoint.py
ruff check --fix src/aiperf/endpoints/openai_completions.py tests/unit/endpoints/test_completions_endpoint.py
git add src/aiperf/endpoints/openai_completions.py tests/unit/endpoints/test_completions_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in openai_completions"
```

---

## Task 7: `openai_responses` merges `extra_body` (multi-turn, latest-wins)

**Files:**
- Modify: `src/aiperf/endpoints/openai_responses.py:191-221`
- Modify: `tests/unit/endpoints/test_responses_endpoint.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/endpoints/test_responses_endpoint.py` (using the same `responses_endpoint` fixture pattern as `test_message_merging_fixes.py`):

```python
class TestResponsesExtraBody:
    def test_extra_body_shallow_merges_into_payload(self, responses_endpoint):
        from aiperf.common.models import Text, Turn
        turn = Turn(
            role="user",
            texts=[Text(contents=["hello"])],
            extra_body={"vendor_top_k": 5},
        )
        request_info = create_request_info(
            model_endpoint=responses_endpoint.model_endpoint,
            turns=[turn],
        )
        payload = responses_endpoint.format_payload(request_info)
        assert payload["vendor_top_k"] == 5

    def test_extra_body_inherits_from_parent_turn(self, responses_endpoint):
        from aiperf.common.models import Text, Turn
        parent = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            extra_body={"vendor_x": 1},
        )
        child = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=responses_endpoint.model_endpoint,
            turns=[parent, child],
        )
        payload = responses_endpoint.format_payload(request_info)
        # _latest_turn_attr walks from end and picks parent's value when child has None.
        assert payload["vendor_x"] == 1
```

If `responses_endpoint` fixture is not already in this file, add it (mirror the `chat_endpoint` fixture from `test_message_merging_fixes.py`):

```python
import pytest
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def responses_endpoint():
    ep_info = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/endpoints/test_responses_endpoint.py::TestResponsesExtraBody -v
```

Expected: FAIL — `vendor_top_k` / `vendor_x` not in payload.

- [ ] **Step 3: Merge `extra_body` in responses formatter**

In `src/aiperf/endpoints/openai_responses.py::format_payload`, after the existing `_latest_turn_attr` calls (line ~192) add an `extra_body` walk; then merge after `model_endpoint.endpoint.extra` (after line 207). Final state:

```python
        max_tokens = self._latest_turn_attr(turns, "max_tokens")
        model_name = self._latest_turn_attr(turns, "model")
        extra_body = self._latest_turn_attr(turns, "extra_body")

        payload: dict[str, Any] = {
            "input": input_items,
            "model": model_name or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if request_info.system_message:
            payload["instructions"] = request_info.system_message

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if extra_body:
            payload.update(extra_body)

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/endpoints/openai_responses.py tests/unit/endpoints/test_responses_endpoint.py
ruff check --fix src/aiperf/endpoints/openai_responses.py tests/unit/endpoints/test_responses_endpoint.py
git add src/aiperf/endpoints/openai_responses.py tests/unit/endpoints/test_responses_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in openai_responses (latest-wins)"
```

---

## Task 8: `openai_embeddings._build_payload` merges `extra_body` (covers OpenAI + NIM embeddings)

**Files:**
- Modify: `src/aiperf/endpoints/openai_embeddings.py:66-88`
- Modify: `tests/unit/endpoints/test_embeddings_endpoint.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/endpoints/test_embeddings_endpoint.py` inside the primary test class:

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        turn = Turn(
            texts=[Text(contents=["hello"])],
            extra_body={"vendor_top_k": 5, "input_type": "query"},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_top_k"] == 5
        assert payload["input_type"] == "query"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_embeddings_endpoint.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 3: Merge `extra_body` in `_build_payload`**

In `src/aiperf/endpoints/openai_embeddings.py::_build_payload`, after the existing `if model_endpoint.endpoint.extra: payload.update(...)` block (line ~85) and before `self.trace(...)`:

```python
        if turn.extra_body:
            payload.update(turn.extra_body)
```

The full block becomes:

```python
        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if turn.extra_body:
            payload.update(turn.extra_body)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload
```

(`turn` is already a parameter of `_build_payload`, so no signature change is needed. `NIMEmbeddingsEndpoint` calls this same `_build_payload` and inherits the merge automatically.)

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/endpoints/openai_embeddings.py tests/unit/endpoints/test_embeddings_endpoint.py
ruff check --fix src/aiperf/endpoints/openai_embeddings.py tests/unit/endpoints/test_embeddings_endpoint.py
git add src/aiperf/endpoints/openai_embeddings.py tests/unit/endpoints/test_embeddings_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in embeddings (OpenAI + NIM)"
```

---

## Task 9: `openai_image_generation` merges `extra_body`

**Files:**
- Modify: `src/aiperf/endpoints/openai_image_generation.py`
- Modify: `tests/unit/endpoints/test_image_generation_endpoint.py`

- [ ] **Step 1: Read the formatter**

Read `src/aiperf/endpoints/openai_image_generation.py::format_payload` to identify (a) which turn variable is used (likely `turn = request_info.turns[0]`), and (b) the exact location of the existing `if model_endpoint.endpoint.extra: payload.update(...)` block.

- [ ] **Step 2: Write failing test**

Append to `tests/unit/endpoints/test_image_generation_endpoint.py` inside the primary test class:

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        turn = Turn(
            texts=[Text(contents=["a cat"])],
            extra_body={"vendor_quality": "hd"},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_quality"] == "hd"
```

(Adapt the fixture names to whatever the file already uses.)

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_image_generation_endpoint.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 4: Merge `extra_body` in image-generation formatter**

In `src/aiperf/endpoints/openai_image_generation.py::format_payload`, immediately AFTER the existing `if model_endpoint.endpoint.extra: payload.update(model_endpoint.endpoint.extra)` block and BEFORE `self.trace(...)`, add:

```python
        if turn.extra_body:
            payload.update(turn.extra_body)
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
ruff format src/aiperf/endpoints/openai_image_generation.py tests/unit/endpoints/test_image_generation_endpoint.py
ruff check --fix src/aiperf/endpoints/openai_image_generation.py tests/unit/endpoints/test_image_generation_endpoint.py
git add src/aiperf/endpoints/openai_image_generation.py tests/unit/endpoints/test_image_generation_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in openai_image_generation"
```

---

## Task 10: `openai_video_generation` merges `extra_body`

**Files:**
- Modify: `src/aiperf/endpoints/openai_video_generation.py`
- Modify: `tests/unit/endpoints/test_video_generation_endpoint.py`

- [ ] **Step 1: Read the formatter**

Read `src/aiperf/endpoints/openai_video_generation.py::format_payload` for the same reasons as Task 9, Step 1.

- [ ] **Step 2: Write failing test**

Append to `tests/unit/endpoints/test_video_generation_endpoint.py` inside the primary test class:

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        turn = Turn(
            texts=[Text(contents=["a sunset"])],
            extra_body={"vendor_fps": 24},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_fps"] == 24
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_video_generation_endpoint.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 4: Merge `extra_body` in video-generation formatter**

In `src/aiperf/endpoints/openai_video_generation.py::format_payload`, immediately AFTER the existing `if model_endpoint.endpoint.extra: payload.update(model_endpoint.endpoint.extra)` block and BEFORE `self.trace(...)`, add:

```python
        if turn.extra_body:
            payload.update(turn.extra_body)
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
ruff format src/aiperf/endpoints/openai_video_generation.py tests/unit/endpoints/test_video_generation_endpoint.py
ruff check --fix src/aiperf/endpoints/openai_video_generation.py tests/unit/endpoints/test_video_generation_endpoint.py
git add src/aiperf/endpoints/openai_video_generation.py tests/unit/endpoints/test_video_generation_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in openai_video_generation"
```

---

## Task 11: `nim_image_retrieval` merges `extra_body`

**Files:**
- Modify: `src/aiperf/endpoints/nim_image_retrieval.py`
- Modify: `tests/unit/endpoints/test_nim_image_retrieval_endpoint.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/endpoints/test_nim_image_retrieval_endpoint.py` inside the primary test class:

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        # Use whatever turn shape the existing tests in this file use as a
        # baseline, but add `extra_body={...}`.
        turn = Turn(
            texts=[Text(contents=["a cat"])],
            extra_body={"vendor_top_k": 3},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_top_k"] == 3
```

(Match whatever turn-construction shape the file already uses for happy-path tests — the image-retrieval endpoint may require an image rather than a text.)

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_nim_image_retrieval_endpoint.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 3: Merge `extra_body` in image-retrieval formatter**

In `src/aiperf/endpoints/nim_image_retrieval.py::format_payload`, immediately AFTER the existing `if model_endpoint.endpoint.extra: payload.update(model_endpoint.endpoint.extra)` block and BEFORE `self.trace(...)`, add:

```python
        if turn.extra_body:
            payload.update(turn.extra_body)
```

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/endpoints/nim_image_retrieval.py tests/unit/endpoints/test_nim_image_retrieval_endpoint.py
ruff check --fix src/aiperf/endpoints/nim_image_retrieval.py tests/unit/endpoints/test_nim_image_retrieval_endpoint.py
git add src/aiperf/endpoints/nim_image_retrieval.py tests/unit/endpoints/test_nim_image_retrieval_endpoint.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in nim_image_retrieval"
```

---

## Task 12: `huggingface_generate` merges `extra_body` (top-level, not into `parameters`)

**Files:**
- Modify: `src/aiperf/endpoints/huggingface_generate.py:38-44`
- Modify: `tests/unit/endpoints/test_huggingface_generate.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/endpoints/test_huggingface_generate.py` inside the primary test class:

```python
    def test_extra_body_shallow_merges_at_top_level(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        turn = Turn(
            texts=[Text(contents=["hello"])],
            extra_body={"vendor_x": 1, "stream": True},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        # extra_body lands at the TOP level (not inside `parameters`),
        # matching OpenAI SDK extra_body semantics.
        assert payload["vendor_x"] == 1
        assert payload["stream"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_huggingface_generate.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 3: Merge `extra_body` at top level in TGI formatter**

In `src/aiperf/endpoints/huggingface_generate.py::format_payload`, replace the block at lines ~38-44:

```python
        payload: dict[str, Any] = {
            "inputs": inputs,
            "parameters": parameters,
        }

        self.trace(lambda: f"Formatted TGI payload: {payload}")
        return payload
```

with:

```python
        payload: dict[str, Any] = {
            "inputs": inputs,
            "parameters": parameters,
        }

        if turn.extra_body:
            payload.update(turn.extra_body)

        self.trace(lambda: f"Formatted TGI payload: {payload}")
        return payload
```

(Top-level merge intentionally — matches OpenAI SDK `extra_body=` semantics. Users who want to override `parameters` keys must do so as `extra_body={"parameters": {...}}`, which replaces the whole sub-dict — documented limitation.)

- [ ] **Step 4: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/endpoints/huggingface_generate.py tests/unit/endpoints/test_huggingface_generate.py
ruff check --fix src/aiperf/endpoints/huggingface_generate.py tests/unit/endpoints/test_huggingface_generate.py
git add src/aiperf/endpoints/huggingface_generate.py tests/unit/endpoints/test_huggingface_generate.py
git commit -s -m "feat(endpoints): merge per-turn extra_body at top level in huggingface_generate"
```

---

## Task 13: `solido_rag` merges `extra_body`

**Files:**
- Modify: `src/aiperf/endpoints/solido_rag.py`
- Modify: `tests/unit/endpoints/test_solido_rag.py`

- [ ] **Step 1: Read the formatter**

Read `src/aiperf/endpoints/solido_rag.py::format_payload` and locate (a) the turn variable name and (b) the existing `if model_endpoint.endpoint.extra: payload.update(...)` block.

- [ ] **Step 2: Write failing test**

Append to `tests/unit/endpoints/test_solido_rag.py` inside the primary test class:

```python
    def test_extra_body_shallow_merges_into_payload(self, converter, model_endpoint):
        from aiperf.common.models import Text, Turn
        # Match the existing happy-path turn shape this test file uses.
        turn = Turn(
            texts=[Text(contents=["query"])],
            extra_body={"vendor_top_k": 5},
        )
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["vendor_top_k"] == 5
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/endpoints/test_solido_rag.py -k extra_body_shallow_merges -v
```

Expected: FAIL.

- [ ] **Step 4: Merge `extra_body` in solido_rag formatter**

In `src/aiperf/endpoints/solido_rag.py::format_payload`, immediately AFTER the existing `if model_endpoint.endpoint.extra: payload.update(model_endpoint.endpoint.extra)` block and BEFORE `self.trace(...)`, add:

```python
        if turn.extra_body:
            payload.update(turn.extra_body)
```

- [ ] **Step 5: Run unit tests**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
ruff format src/aiperf/endpoints/solido_rag.py tests/unit/endpoints/test_solido_rag.py
ruff check --fix src/aiperf/endpoints/solido_rag.py tests/unit/endpoints/test_solido_rag.py
git add src/aiperf/endpoints/solido_rag.py tests/unit/endpoints/test_solido_rag.py
git commit -s -m "feat(endpoints): merge per-turn extra_body in solido_rag"
```

---

## Task 14: Documentation updates

**Files:**
- Modify: `docs/dev/patterns.md`
- Modify: mooncake / single-turn / multi-turn schema reference docs (located in Step 1)

- [ ] **Step 1: Locate the schema reference docs**

```bash
grep -rln "input_length\|MooncakeTrace\|mooncake.*trace" docs/ | grep -vE "(plan|spec|index)" | head
grep -rln "single_turn\|SingleTurn" docs/ | grep -vE "(plan|spec|index)" | head
grep -rln "multi_turn\|MultiTurn" docs/ | grep -vE "(plan|spec|index)" | head
```

Identify the user-facing markdown that documents each loader's accepted JSON schema (likely under `docs/benchmark-modes/`, `docs/tutorials/`, or `docs/reference/`). Note the exact paths.

- [ ] **Step 2: Update `docs/dev/patterns.md`**

In the endpoint section of `docs/dev/patterns.md`, add (or extend an existing block on `extra_body`):

```markdown
### Per-turn `extra_body`

Every endpoint formatter that builds a JSON request body — `openai_chat`,
`openai_completions`, `openai_responses`, `openai_embeddings` (and `nim_embeddings`),
`openai_image_generation`, `openai_video_generation`, `nim_image_retrieval`,
`huggingface_generate`, `solido_rag`, `chat_embeddings`, and the rankings family
(`cohere_rankings`, `nim_rankings`, `hf_tei_rankings`) — shallow-merges
`Turn.extra_body` into the wire body at the very end of payload construction,
AFTER `model_endpoint.endpoint.extra`. User-supplied keys win on collision,
matching the OpenAI SDK's `extra_body=` keyword.

For multi-turn endpoints (`openai_chat`, `openai_responses`), `extra_body` is
walked from the last turn backwards (`_latest_turn_attr`), so FORK-mode DAG
children inherit a parent's `extra_body` if they don't redeclare it.

`raw_endpoint` and `template_endpoint` intentionally skip this merge — they
ship the user-authored body verbatim ("what you author is what gets sent").
```

- [ ] **Step 3: Update each schema reference doc**

For each file located in Step 1, add `extra_body` to the documented JSON schema. Example sentence to add to the recognized-fields table or list:

```markdown
| `extra_body` | dict (optional) | Vendor extras shallow-merged into the top of the request body at dispatch (matches OpenAI SDK `extra_body=` semantics). |
```

For the mooncake doc, this goes alongside `text_input` / `messages` / `payload` / `output_length`.
For the single-turn / multi-turn docs, alongside `text` / `images` / `output_length`.

- [ ] **Step 4: Verify the docs index**

If any new doc files were created (none expected for this task — only edits), `tools/check_docs_index.py` enforces them in CI; running `make generate-all-docs` covers auto-generated CLI/env-var docs (no change here).

```bash
make generate-all-docs
```

Expected: no diff, since this change adds no CLI options or env vars.

- [ ] **Step 5: Commit docs**

```bash
git add docs/
git commit -s -m "docs: document per-turn extra_body merging across endpoint formatters and loaders"
```

---

## Task 15: End-of-branch verification

- [ ] **Step 1: Run unit suite**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS.

- [ ] **Step 2: Run pre-commit on staged files**

```bash
pre-commit run --files $(git diff --name-only origin/main...HEAD)
```

Expected: PASS. Address any formatter / lint hits and amend the relevant commit (or add a fix commit).

- [ ] **Step 3: Confirm acceptance criteria from spec**

Walk the spec's acceptance criteria one by one and confirm each is satisfied:

1. Each of the 13 listed formatters merges `extra_body` (verified by 9 test cases — three formatters covered by `BaseRankingsEndpoint`, two by `EmbeddingsEndpoint._build_payload`, one by `ChatEmbeddingsEndpoint` inheritance from `ChatEndpoint`).
2. Mooncake / single-turn / multi-turn JSONL inputs propagate `extra_body` to `Turn`.
3. `raw_endpoint`, `template_endpoint`, and `openai_chat` are unchanged (verify with `git diff origin/main...HEAD -- src/aiperf/endpoints/raw_endpoint.py src/aiperf/endpoints/template_endpoint.py src/aiperf/endpoints/openai_chat.py` — should be empty).
4. New unit tests cover default / merge / override cases.
5. Docs updated in `docs/dev/patterns.md` and the loader reference docs.
6. Unit tests + pre-commit pass.
