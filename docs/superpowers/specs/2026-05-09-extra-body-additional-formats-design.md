# `extra_body` support in additional endpoint formats and dataset loaders

**Date:** 2026-05-09
**Author:** acasagrande@nvidia.com
**Status:** Proposed

## Background

`Turn.extra_body` (`src/aiperf/common/models/dataset_models.py:199`) carries a per-turn dict that is shallow-merged into the wire request body at dispatch time, matching the OpenAI SDK's `extra_body=` keyword. Today only two pieces of the system honor it:

- `openai_chat.py` formatter merges it into the request payload (`src/aiperf/endpoints/openai_chat.py:70,93-94`).
- `dag_jsonl.py` loader populates it on `Turn` (`src/aiperf/dataset/loader/dag_jsonl.py:259`).

Every other endpoint formatter ignores it, and every loader except `dag_jsonl` drops it on the floor. A user who wants to set per-turn vendor extras (`top_k`, `chat_template_kwargs`, `nvext`, etc.) for a non-chat endpoint, or who wants their mooncake / single-turn / multi-turn input file to carry `extra_body`, has no way to do it today.

## Goals

1. Wire per-turn `extra_body` merging into every endpoint formatter that builds a JSON request body.
2. Let the `mooncake_trace`, `single_turn`, and `multi_turn` loaders read `extra_body` from input rows and propagate it onto `Turn`.
3. Preserve the existing convention: shallow `dict.update`, user-provided keys win on collision, and `extra_body` is applied after CLI-level `model_endpoint.endpoint.extra`.

## Non-goals

- No CLI surface change. No new flag for synthetic loaders (`random_pool`, `inputs_json`, HF/public-dataset loaders).
- No deep-merge semantics. Nested keys (`stream_options`, `chat_template_kwargs`) replace whole, not merge piecewise.
- No change to `raw_endpoint` (replays verbatim) or `template_endpoint` (user owns the body) — these intentionally bypass formatter merging.
- No change to `openai_chat` (already correct).

## Design

### 1. Endpoint formatter changes

Add a single 2-line block to each formatter listed below, placed just before the formatter's final return / trace log, and AFTER the existing `model_endpoint.endpoint.extra` merge so per-turn extras win:

```python
if extra_body:
    payload.update(extra_body)
```

The `extra_body` value is sourced differently depending on whether the formatter walks multiple turns:

- **Multi-turn (chat-style)** — use `self._latest_turn_attr(turns, "extra_body")`, the same helper `openai_chat` uses for `model` / `tools` / `max_tokens` / `extra_body`. This walks turns from the end and returns the first non-None value, so FORK-mode DAG children whose final turn does not redeclare extras still inherit the parent's intent.
- **Single-turn** — read `turn.extra_body` directly from whichever turn the formatter consumes (typically `request_info.turns[0]` or `[-1]`).

| Formatter | File | Mode | Source |
|---|---|---|---|
| Chat embeddings | `chat_embeddings.py` | Multi-turn | `_latest_turn_attr` |
| Completions | `openai_completions.py` | Single-turn | `turn.extra_body` |
| Responses API | `openai_responses.py` | Single-turn | `turn.extra_body` |
| Embeddings | `openai_embeddings.py` | Single-turn | `turn.extra_body` |
| Image generation | `openai_image_generation.py` | Single-turn | `turn.extra_body` |
| Video generation | `openai_video_generation.py` | Single-turn | `turn.extra_body` |
| NIM embeddings | `nim_embeddings.py` | Single-turn | `turn.extra_body` |
| NIM image retrieval | `nim_image_retrieval.py` | Single-turn | `turn.extra_body` |
| NIM rankings | `nim_rankings.py` | Single-turn | `turn.extra_body` |
| Cohere rankings | `cohere_rankings.py` | Single-turn | `turn.extra_body` |
| HF TEI rankings | `hf_tei_rankings.py` | Single-turn | `turn.extra_body` |
| HF generate | `huggingface_generate.py` | Single-turn | `turn.extra_body` |
| Solido RAG | `solido_rag.py` | Single-turn | `turn.extra_body` |

**Skipped formatters (intentional):**

- `raw_endpoint.py` — replays the trace's pre-built body verbatim (replay mode).
- `template_endpoint.py` — user owns the entire body via Jinja template.
- `openai_chat.py` — already implements the same merge.

**Merge order, end-to-end:**

```
formatter-built defaults
  → model_endpoint.endpoint.extra      (CLI / config-level)
  → extra_body                         (per-turn, latest-wins-across-turns)
```

Each step is `dict.update`, so each later step's keys clobber earlier ones on collision. This matches `openai_chat.py:90-94` exactly.

### 2. Dataset loader changes

#### `mooncake_trace`

Add an optional field to the `MooncakeTrace` Pydantic model in `src/aiperf/dataset/loader/models.py`:

```python
extra_body: dict[str, Any] | None = Field(
    default=None,
    description=(
        "Per-turn extra fields shallow-merged into the request body at "
        "dispatch time, matching the OpenAI SDK's extra_body convention."
    ),
)
```

In `src/aiperf/dataset/loader/mooncake_trace.py::_build_turn`, propagate `trace.extra_body` to the `Turn(...)` constructor in all three branches:

- `payload is not None` branch — add `extra_body=trace.extra_body`.
- `messages is not None` branch — add `extra_body=trace.extra_body`.
- Fall-through branch (currently `return super()._build_turn(trace, prompt)`) — call `super()` to get the base Turn, then set `turn.extra_body = trace.extra_body` if non-None before returning. This avoids duplicating the base loader's media-conversion logic.

#### `single_turn` and `multi_turn`

Add the same `extra_body` field to the `SingleTurn` Pydantic model in `src/aiperf/dataset/loader/models.py`. Because `MultiTurn.turns: list[SingleTurn]`, `MultiTurn` inherits the field at no extra cost.

In `src/aiperf/dataset/loader/single_turn.py::convert_to_conversations`, add `extra_body=single_turn.extra_body` to the `Turn(...)` constructor.

In `src/aiperf/dataset/loader/multi_turn.py::convert_to_conversations`, add `extra_body=single_turn.extra_body` (the loop variable, an inner `SingleTurn`) to the `Turn(...)` constructor.

### 3. Tests

#### Endpoint formatter tests

For each modified formatter, add a unit test asserting:

1. **Default**: When `turn.extra_body` is None, the payload is unchanged from current behavior. (Regression guard.)
2. **Merge**: When `turn.extra_body = {"top_k": 5, "vendor_field": "x"}`, the payload contains both keys with those values.
3. **Override on collision**: When `turn.extra_body = {"stream": False}` against a formatter that already sets `"stream": True`, the final payload has `"stream": False`. (Confirms last-write-wins semantics.)

For `chat_embeddings` only, add a fourth test: multi-turn `_latest_turn_attr` walk — earlier turn carries `extra_body={"a": 1}`, final turn carries `None`, expect `{"a": 1}` in payload (parent intent inherited).

Place tests in `tests/unit/endpoints/test_<formatter>.py` (create if missing). Use `pytest.param(..., id="...")` ids matching the three scenarios.

#### Loader tests

In `tests/unit/dataset/loader/`:

- `test_mooncake_trace.py` — add three parametrized cases (payload-mode line with `extra_body`, messages-mode line with `extra_body`, plain text-input line with `extra_body`); assert the resulting `Turn.extra_body` matches the input.
- `test_single_turn.py` — add a case where the JSONL line has `extra_body`, assert the resulting `Turn.extra_body` matches.
- `test_multi_turn.py` — add a case where one inner turn carries `extra_body` and another does not; assert per-turn propagation is exact.

#### Integration smoke

Existing `tests/unit/endpoints/test_message_merging_fixes.py` already exercises chat `extra_body` propagation through the full RequestInfo → payload path; no change needed beyond confirming it still passes.

### 4. Documentation updates

| File | Change |
|---|---|
| `docs/dev/patterns.md` | Endpoint section: one paragraph noting per-turn `extra_body` is shallow-merged into the wire body across all formatters except `raw_endpoint` and `template_endpoint`, applied AFTER the CLI-level `model_endpoint.endpoint.extra`. |
| `docs/cli-options.md` | No change — auto-generated; no new CLI flag. |
| `docs/environment-variables.md` | No change. |
| Mooncake schema docs (wherever the loader's JSON schema is shown — likely a tutorial or `docs/benchmark-modes/`) | Add `extra_body` to the recognized field list with one-line description. |
| Single-turn / multi-turn schema docs (same surface) | Add `extra_body` to recognized fields. |

The `dag_jsonl` docs already document `extra_body`; no change there.

## Risks and trade-offs

- **Surface area**: 13 formatter files + 3 loader files + 2 model fields + ~20 unit tests. All edits are mechanical and follow `openai_chat`'s template, so the cognitive load per file is low.
- **Last-write-wins on collision**: A user who sets `extra_body={"stream": False}` against a streaming-enabled run silently disables streaming for that turn. This is the documented OpenAI SDK semantic, but worth flagging in the patterns doc.
- **`raw_endpoint` and `template_endpoint` carve-out**: Surprising for users who expect uniform behavior. Mitigation: explicit one-liner in the patterns doc explaining why these two skip the merge ("what you author is what gets sent").
- **Synthetic loaders untouched**: `random_pool`, `inputs_json`, HF/public-dataset loaders still ignore `extra_body`. Out of scope for this change; revisit when there's a concrete user request.

## Out of scope

- Synthetic-loader CLI surface for static `extra_body`.
- Deep-merge semantics for nested `extra_body` keys.
- Per-formatter validation of `extra_body` keys against vendor schemas.
- Non-mooncake / non-single-turn / non-multi-turn loaders.

## Acceptance criteria

1. Each of the 13 listed formatters merges `extra_body` into its payload, with `extra_body` keys winning over both formatter defaults and `model_endpoint.endpoint.extra`.
2. Mooncake, single-turn, and multi-turn JSONL inputs containing `extra_body` produce `Turn` objects whose `extra_body` matches the input verbatim.
3. `raw_endpoint`, `template_endpoint`, and `openai_chat` are unchanged.
4. New unit tests cover default / merge / override cases for every modified formatter and loader.
5. Documentation reflects the new behavior.
6. `pre-commit run --all-files` and `uv run pytest tests/unit/ -n auto` pass.
