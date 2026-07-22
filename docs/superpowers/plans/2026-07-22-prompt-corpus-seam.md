<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompt Corpus Clean Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Rust-style `prompts.corpus` / `--prompt-corpus` seam so `sonnet` and `coding` are selected by one shared factory and actually drive synthetic + trace synthesis.

**Architecture:** Rename authored field to `prompts.corpus` on all dataset types; add `resolve_prompt_generator()` in `dataset/generator/corpus.py`; wire Base/Custom/Public composers through it; hard-cut flat `prompt_corpus`.

**Tech Stack:** Python 3.11+, Pydantic config (`BaseConfig`), pytest, existing `PromptGenerator` / `CodingContentGenerator`.

**Spec:** `docs/superpowers/specs/2026-07-22-prompt-corpus-seam-design.md`

## Global Constraints

- Corpus values: `sonnet` | `coding` only (no `random`)
- Authored shape: always `prompts.corpus` (CLI flag remains `--prompt-corpus`)
- Defaults: keep loader `default_prompt_corpus` from `plugins.yaml`
- Hard cut: no top-level `prompt_corpus` alias
- `Field(description=...)` on every new Pydantic field; `X | Y` not `Optional`
- Dependencies via `uv`; tests via `uv run pytest`
- DCO sign-off on commits: `git commit -s`
- Docs required for CLI + new `docs/reference/prompt-corpus.md` + `docs/index.yml`
- Known limitation: `CodingContentGenerator` has no prefix-prompt API; prefix prompts stay sonnet-only

## File Structure

| File | Responsibility |
|---|---|
| `src/aiperf/dataset/generator/corpus.py` | **Create** — `resolve_prompt_generator()` |
| `tests/unit/dataset/generator/test_corpus_resolver.py` | **Create** — factory unit tests |
| `src/aiperf/config/dataset/content.py` | Rename `PromptConfig.prompt_corpus` → `corpus`; add `PromptSelectionConfig` |
| `src/aiperf/config/dataset/config.py` | File/Public: remove flat `prompt_corpus`, add `prompts: PromptSelectionConfig \| None` |
| `src/aiperf/config/loader/helpers.py` | `get_prompt_corpus()` reads only `prompts.corpus` |
| `src/aiperf/config/flags/_converter_dataset.py` | Project CLI into `prompts.corpus` for all types |
| `src/aiperf/dataset/composer/base.py` | Build `prompt_generator` via factory |
| `src/aiperf/dataset/composer/custom.py` | Trace injection via factory |
| `src/aiperf/dataset/composer/public.py` | Trace injection via factory |
| `src/aiperf/dataset/mmap_cache.py` | Key from `prompts.corpus` |
| `docs/reference/prompt-corpus.md` | **Create** — user-facing seam doc |
| `docs/index.yml` | Register reference doc |
| Existing unit tests listed per task | Update expectations |

---

### Task 1: Shared corpus factory

**Files:**
- Create: `src/aiperf/dataset/generator/corpus.py`
- Create: `tests/unit/dataset/generator/test_corpus_resolver.py`

**Interfaces:**
- Produces: `resolve_prompt_generator(*, corpus, default_corpus, tokenizer, prompts=None, prefix_prompts=None) -> PromptGenerator | CodingContentGenerator`

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import PromptCorpus
from aiperf.config.dataset.content import PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.corpus import resolve_prompt_generator
from aiperf.dataset.generator.prompt import PromptGenerator


@pytest.mark.parametrize(
    ("corpus", "default_corpus", "expected_type"),
    [
        param(PromptCorpus.CODING, None, CodingContentGenerator, id="explicit-coding"),
        param(PromptCorpus.SONNET, PromptCorpus.CODING, PromptGenerator, id="explicit-sonnet-wins"),
        param(None, PromptCorpus.CODING, CodingContentGenerator, id="default-coding"),
        param(None, None, PromptGenerator, id="fallback-sonnet"),
        param(None, "coding", CodingContentGenerator, id="default-str-coding"),
    ],
)  # fmt: skip
def test_resolve_prompt_generator_selection(
    mock_tokenizer, corpus, default_corpus, expected_type
):
    gen = resolve_prompt_generator(
        corpus=corpus,
        default_corpus=default_corpus,
        tokenizer=mock_tokenizer,
        prompts=PromptConfig(),
    )
    assert isinstance(gen, expected_type)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dataset/generator/test_corpus_resolver.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `aiperf.dataset.generator.corpus`

- [ ] **Step 3: Write minimal implementation**

Create `src/aiperf/dataset/generator/corpus.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt-corpus selector for synthetic and trace synthesis."""

from __future__ import annotations

from aiperf.common.enums import PromptCorpus
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator


def resolve_prompt_generator(
    *,
    corpus: PromptCorpus | str | None,
    default_corpus: PromptCorpus | str | None,
    tokenizer: Tokenizer,
    prompts: PromptConfig | None = None,
    prefix_prompts: PrefixPromptConfig | None = None,
) -> PromptGenerator | CodingContentGenerator:
    """Pick sonnet vs coding generator from authored corpus + loader default.

    Resolution: explicit ``corpus`` -> ``default_corpus`` -> ``PromptCorpus.SONNET``.
    """
    resolved = corpus or default_corpus or PromptCorpus.SONNET
    if resolved == PromptCorpus.CODING or resolved == "coding":
        return CodingContentGenerator(
            config=prompts or PromptConfig(),
            tokenizer=tokenizer,
        )
    return PromptGenerator(
        prompts=prompts,
        prefix_prompts=prefix_prompts,
        tokenizer=tokenizer,
    )
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `uv run pytest tests/unit/dataset/generator/test_corpus_resolver.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/generator/corpus.py tests/unit/dataset/generator/test_corpus_resolver.py
git commit -s -m "$(cat <<'EOF'
feat(dataset): add shared prompt-corpus resolver

Centralize sonnet/coding generator selection so composers share one seam.
EOF
)"
```

---

### Task 2: Authored `prompts.corpus` config shape

**Files:**
- Modify: `src/aiperf/config/dataset/content.py` — rename field; add `PromptSelectionConfig`
- Modify: `src/aiperf/config/dataset/config.py` — File/Public `prompts`; drop flat `prompt_corpus`
- Modify: `src/aiperf/config/dataset/__init__.py` (export `PromptSelectionConfig` if re-exported)
- Modify: `src/aiperf/config/loader/helpers.py` — simplify `get_prompt_corpus`
- Modify: `tests/unit/config/test_dataset_content_config_defaults.py`
- Create/modify: `tests/unit/config/test_prompt_corpus_authored_shape.py` (hard-cut + nested read)

**Interfaces:**
- Consumes: none from Task 1
- Produces: `PromptConfig.corpus`, `PromptSelectionConfig.corpus`, `FileDataset.prompts`, `PublicDataset.prompts`, `get_prompt_corpus() -> PromptCorpus | None`

- [ ] **Step 1: Write failing tests**

Add to a new file `tests/unit/config/test_prompt_corpus_authored_shape.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiperf.common.enums import PromptCorpus
from aiperf.config.dataset import FileDataset, PublicDataset, SyntheticDataset
from aiperf.config.dataset.content import PromptConfig, PromptSelectionConfig
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import CustomDatasetType, PublicDatasetType
from tests.unit.conftest import make_run_from_cli


def test_prompt_config_uses_corpus_not_prompt_corpus():
    cfg = PromptConfig(corpus=PromptCorpus.CODING)
    assert cfg.corpus == PromptCorpus.CODING
    assert not hasattr(cfg, "prompt_corpus") or "prompt_corpus" not in PromptConfig.model_fields


def test_file_dataset_rejects_flat_prompt_corpus():
    with pytest.raises(Exception) as exc:
        FileDataset.model_validate(
            {
                "type": "file",
                "path": "x.jsonl",
                "format": "weka_trace",
                "prompt_corpus": "coding",
            }
        )
    assert "prompt_corpus" in str(exc.value).lower() or "extra" in str(exc.value).lower()


def test_file_dataset_accepts_prompts_corpus():
    ds = FileDataset.model_validate(
        {
            "type": "file",
            "path": "x.jsonl",
            "format": "weka_trace",
            "prompts": {"corpus": "coding"},
        }
    )
    assert ds.prompts is not None
    assert ds.prompts.corpus == PromptCorpus.CODING


def test_get_prompt_corpus_reads_prompts_corpus_only(tmp_path):
    p = tmp_path / "t.jsonl"
    p.touch()
    cli = CLIConfig.model_construct(
        model_names=["m"],
        input_file=str(p),
        custom_dataset_type=CustomDatasetType.WEKA_TRACE,
    )
    run = make_run_from_cli(cli)
    ds = run.cfg.get_default_dataset()
    # After Task 3 converter may set this; for this task set authored shape directly:
    ds.prompts = PromptSelectionConfig(corpus=PromptCorpus.SONNET)
    assert run.cfg.get_prompt_corpus() == PromptCorpus.SONNET
```

Also update `tests/unit/config/test_dataset_content_config_defaults.py`:

```python
assert config.corpus is None
```

(replace `assert config.prompt_corpus is None`)

- [ ] **Step 2: Run tests to verify fail**

Run: `uv run pytest tests/unit/config/test_prompt_corpus_authored_shape.py tests/unit/config/test_dataset_content_config_defaults.py::test_prompt_config_defaults -v`

Expected: FAIL on missing `corpus` / still having `prompt_corpus`

- [ ] **Step 3: Implement config changes**

In `content.py`:

1. Rename `PromptConfig.prompt_corpus` → `corpus` (keep same description, mention `prompts.corpus`).
2. Add:

```python
class PromptSelectionConfig(BaseConfig):
    """Slim prompts block for file/public datasets (corpus selection only)."""

    model_config = ConfigDict(extra="forbid")

    corpus: Annotated[
        PromptCorpus | None,
        Field(
            default=None,
            description="Source corpus for synthesized prompt text. "
            "'sonnet' uses Shakespeare sonnets. "
            "'coding' uses realistic coding content. "
            "When unset, the active dataset loader's default applies. "
            "Honored only where content is synthesized (synthetic + hash/trace "
            "replay); verbatim loaders ignore it.",
        ),
    ]
```

In `config.py` `FileDataset` and `PublicDataset`:

- Delete `prompt_corpus` fields.
- Add:

```python
prompts: Annotated[
    PromptSelectionConfig | None,
    Field(
        default=None,
        description="Prompt synthesis selection for this dataset. "
        "Author ``prompts.corpus`` to choose sonnet vs coding when content "
        "is synthesized (trace hash_id reconstruction). Verbatim formats ignore it.",
    ),
]
```

Import `PromptSelectionConfig` from content; export from `config.py` / package `__init__` if peers export `PromptConfig`.

In `helpers.py`:

```python
def get_prompt_corpus(self) -> PromptCorpus | None:
    """Resolve the active dataset's authored ``prompts.corpus``."""
    dataset = self.get_default_dataset()
    prompts = getattr(dataset, "prompts", None)
    if prompts is None:
        return None
    return getattr(prompts, "corpus", None)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/config/test_prompt_corpus_authored_shape.py tests/unit/config/test_dataset_content_config_defaults.py -v`

Expected: PASS for Task 2 tests. Other suites may fail until Task 3–5 — that is OK if you only run these files.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/dataset/content.py src/aiperf/config/dataset/config.py src/aiperf/config/dataset/__init__.py src/aiperf/config/loader/helpers.py tests/unit/config/test_prompt_corpus_authored_shape.py tests/unit/config/test_dataset_content_config_defaults.py
git commit -s -m "$(cat <<'EOF'
refactor(config): author prompt corpus as prompts.corpus

Align file/public/synthetic datasets on one nested field and drop the flat prompt_corpus.
EOF
)"
```

---

### Task 3: CLI converter projects `prompts.corpus`

**Files:**
- Modify: `src/aiperf/config/flags/_converter_dataset.py`
- Modify: `tests/unit/config/test_trace_flag_routing.py`

**Interfaces:**
- Consumes: `PromptSelectionConfig` / `prompts.corpus` shape from Task 2
- Produces: `build_dataset()` dict with `prompts: {corpus: ...}` for file/public; synthetic `prompts.corpus`

- [ ] **Step 1: Update failing routing tests**

In `test_trace_flag_routing.py`, change assertions to nested shape and update the module docstring to say corpus is re-attached under `prompts` (cache-bust may still be top-level):

```python
def test_routes_onto_file_trace(self, trace_jsonl: Path) -> None:
    out = build_dataset(_file_cli(trace_jsonl, prompt_corpus="coding"))
    assert out["type"] == "file"
    assert out.get("prompts", {}).get("corpus") == "coding"
    ds = convert_cli_to_aiperf(
        _file_cli(trace_jsonl, prompt_corpus="coding")
    ).benchmark.datasets[0]
    assert ds.prompts is not None
    assert ds.prompts.corpus == "coding"


def test_routes_onto_public_weka_hf(self) -> None:
    out = build_dataset(_public_cli(prompt_corpus="coding"))
    assert out["type"] == "public"
    assert out.get("prompts", {}).get("corpus") == "coding"
    ds = convert_cli_to_aiperf(
        _public_cli(prompt_corpus="coding")
    ).benchmark.datasets[0]
    assert ds.prompts is not None
    assert ds.prompts.corpus == "coding"
```

Add a synthetic routing case in the same file or a sibling test:

```python
def test_routes_onto_synthetic_prompts_corpus() -> None:
    cli = CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        prompt_corpus="coding",
        prompt_input_tokens_mean=16,
    )
    out = build_dataset(cli)
    assert out["type"] == "synthetic"
    assert out.get("prompts", {}).get("corpus") == "coding"
```

- [ ] **Step 2: Run tests — expect fail**

Run: `uv run pytest tests/unit/config/test_trace_flag_routing.py::TestPromptCorpusRouting -v`

Expected: FAIL asserting nested `prompts.corpus`

- [ ] **Step 3: Fix converter**

In `_build_prompts`:

```python
if "prompt_corpus" in s and cli.prompt_corpus is not None:
    prompts["corpus"] = cli.prompt_corpus
```

In `_apply_corpus_and_cache_bust` (rename docstring; keep cache_bust top-level):

```python
def _apply_corpus_and_cache_bust(d: dict[str, Any], cli: CLIConfig) -> None:
    """Route ``--prompt-corpus`` / ``--cache-bust`` onto FILE/PUBLIC datasets.

    Corpus is re-attached as ``prompts.corpus`` after the synthetic prompts
    subtable is stripped. Cache-bust remains a top-level ``cache_bust`` field.
    """
    from aiperf.common.enums import DatasetType

    if d.get("type") not in (DatasetType.FILE, DatasetType.PUBLIC):
        return
    s = cli.model_fields_set
    if "prompt_corpus" in s and cli.prompt_corpus is not None:
        prompts = d.get("prompts")
        if not isinstance(prompts, dict):
            prompts = {}
            d["prompts"] = prompts
        prompts["corpus"] = cli.prompt_corpus
    if "cache_bust" in s:
        d["cache_bust"] = {"target": cli.cache_bust}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/config/test_trace_flag_routing.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/flags/_converter_dataset.py tests/unit/config/test_trace_flag_routing.py
git commit -s -m "$(cat <<'EOF'
fix(cli): project --prompt-corpus into prompts.corpus

File and public datasets keep the flag after the synthetic prompts strip.
EOF
)"
```

---

### Task 4: Wire composers through the factory

**Files:**
- Modify: `src/aiperf/dataset/composer/base.py`
- Modify: `src/aiperf/dataset/composer/custom.py`
- Modify: `src/aiperf/dataset/composer/public.py`
- Modify: `tests/unit/dataset/composer/test_coding_corpus_injection.py`
- Create: `tests/unit/dataset/composer/test_synthetic_prompt_corpus.py`

**Interfaces:**
- Consumes: `resolve_prompt_generator` (Task 1), `get_prompt_corpus()` (Task 2)

- [ ] **Step 1: Write failing composer tests**

`tests/unit/dataset/composer/test_synthetic_prompt_corpus.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from tests.unit.conftest import make_run_from_cli


def test_synthetic_default_uses_sonnet_generator(mock_tokenizer):
    cli = CLIConfig(
        model_names=["m"],
        endpoint_type="chat",
        prompt_input_tokens_mean=16,
    )
    run = make_run_from_cli(cli)
    composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
    assert isinstance(composer.prompt_generator, PromptGenerator)
    assert not isinstance(composer.prompt_generator, CodingContentGenerator)


def test_synthetic_prompt_corpus_coding_uses_coding_generator(mock_tokenizer):
    cli = CLIConfig(
        model_names=["m"],
        endpoint_type="chat",
        prompt_input_tokens_mean=16,
        prompt_corpus="coding",
    )
    run = make_run_from_cli(cli)
    composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
    assert isinstance(composer.prompt_generator, CodingContentGenerator)
```

Update `test_coding_corpus_injection.py` override test to set nested prompts:

```python
from aiperf.config.dataset.content import PromptSelectionConfig

def test_explicit_prompt_corpus_overrides_loader_default(self, weka_run, mock_tokenizer):
    weka_run.cfg.get_default_dataset().prompts = PromptSelectionConfig(
        corpus=PromptCorpus.SONNET
    )
    assert weka_run.cfg.get_prompt_corpus() == PromptCorpus.SONNET
    composer = CustomDatasetComposer(run=weka_run, tokenizer=mock_tokenizer)
    composer.create_dataset()
    loader_gen = composer.loader.prompt_generator
    assert isinstance(loader_gen, PromptGenerator)
    assert not isinstance(loader_gen, CodingContentGenerator)
```

Remove the outdated PORT-TODO comment.

- [ ] **Step 2: Run tests — expect synthetic coding fail**

Run: `uv run pytest tests/unit/dataset/composer/test_synthetic_prompt_corpus.py tests/unit/dataset/composer/test_coding_corpus_injection.py -v`

Expected: `test_synthetic_prompt_corpus_coding_uses_coding_generator` FAIL (still `PromptGenerator`)

- [ ] **Step 3: Wire composers**

In `base.py`, replace direct `PromptGenerator(...)` construction:

```python
from aiperf.dataset.generator.corpus import resolve_prompt_generator

# inside __init__, when tokenizer is present:
self.prompt_generator = resolve_prompt_generator(
    corpus=self.run.cfg.get_prompt_corpus(),
    default_corpus=None,
    tokenizer=tokenizer,
    prompts=self._synthetic_prompts,
    prefix_prompts=compensated_prefix_prompts,
) if tokenizer else None
```

Widen the annotation if needed:
`self.prompt_generator: PromptGenerator | CodingContentGenerator | None`

In `custom.py`, replace `_select_trace_prompt_generator` body:

```python
def _select_trace_prompt_generator(self, loader_metadata: Any) -> Any:
    from aiperf.dataset.generator.corpus import resolve_prompt_generator

    return resolve_prompt_generator(
        corpus=self.run.cfg.get_prompt_corpus(),
        default_corpus=loader_metadata.default_prompt_corpus,
        tokenizer=self.prompt_generator.tokenizer,
        prompts=self._synthetic_prompts,
        prefix_prompts=None,
    )
```

Keep the docstring intent (why coding vs sonnet matters for hash_ids).

In `public.py` `_inject_trace_kwargs`, replace the manual if/else with the same `resolve_prompt_generator(...)` call (tokenizer from `self.prompt_generator.tokenizer`).

- [ ] **Step 4: Run composer tests**

Run: `uv run pytest tests/unit/dataset/composer/test_synthetic_prompt_corpus.py tests/unit/dataset/composer/test_coding_corpus_injection.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/dataset/composer/base.py src/aiperf/dataset/composer/custom.py src/aiperf/dataset/composer/public.py tests/unit/dataset/composer/test_synthetic_prompt_corpus.py tests/unit/dataset/composer/test_coding_corpus_injection.py
git commit -s -m "$(cat <<'EOF'
feat(dataset): honor prompts.corpus in synthetic and trace composers

Route generator construction through the shared corpus resolver.
EOF
)"
```

---

### Task 5: mmap cache key + leftover references

**Files:**
- Modify: `src/aiperf/dataset/mmap_cache.py`
- Modify: `tests/unit/dataset/test_mmap_cache.py`
- Grep + fix any remaining `prompt_corpus` attribute access on datasets (exclude CLI flag name `prompt_corpus` on `CLIConfig`, plugins `default_prompt_corpus`, and RNG namespace strings)

- [ ] **Step 1: Write/adjust failing mmap assertion**

In `test_mmap_cache.py`, keep asserting the cache payload includes the authored corpus, but read via nested prompts. Update setup that assigned `dataset.prompt_corpus` to set `dataset.prompts = PromptSelectionConfig(corpus=...)`.

Payload key may stay `"prompt_corpus"` for hash stability **or** rename to `"corpus"` — prefer `"corpus"` and update the assert:

```python
assert "corpus" in payload
assert payload["corpus"] == "coding"  # or whatever the test sets
```

- [ ] **Step 2: Run test — expect fail**

Run: `uv run pytest tests/unit/dataset/test_mmap_cache.py -k corpus -v`

- [ ] **Step 3: Fix mmap key extraction**

```python
_prompts = getattr(dataset, "prompts", None)
"corpus": getattr(_prompts, "corpus", None),
```

Also update the comment above that mentions `prompt_corpus`.

- [ ] **Step 4: Repo-wide leftover check**

Run: `rg -n "\\.prompt_corpus|prompt_corpus=" src/aiperf tests --glob '*.py'`

Fix any remaining dataset-field usages. Leave:
- `CLIConfig.prompt_corpus`
- `default_prompt_corpus` plugin metadata
- RNG `"dataset.prompt.corpus"`
- test names that say “prompt corpus” in prose

Regenerate config schema if the repo expects it:

Run: `make generate-config-schema` (or whatever the Makefile target is; pre-commit `generate-config-schema` must pass)

- [ ] **Step 5: Broader unit confirmation**

Run: `uv run pytest tests/unit/config/test_trace_flag_routing.py tests/unit/dataset/composer/test_coding_corpus_injection.py tests/unit/dataset/composer/test_synthetic_prompt_corpus.py tests/unit/dataset/generator/test_corpus_resolver.py tests/unit/dataset/test_mmap_cache.py tests/unit/config/test_prompt_corpus_authored_shape.py -n auto`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/dataset/mmap_cache.py tests/unit/dataset/test_mmap_cache.py src/aiperf/config/schema/
git commit -s -m "$(cat <<'EOF'
fix(dataset): key mmap cache on prompts.corpus

Keep sonnet vs coding reconstructions from sharing a stale mmap entry.
EOF
)"
```

---

### Task 6: Documentation

**Files:**
- Create: `docs/reference/prompt-corpus.md`
- Modify: `docs/index.yml` (add under Reference section near isl docs)
- Modify: `src/aiperf/config/flags/cli_config.py` — Field description if needed to say `prompts.corpus`
- Run: `make generate-cli-docs` (updates `docs/cli-options.md`)

- [ ] **Step 1: Write `docs/reference/prompt-corpus.md`**

```markdown
<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompt corpus selection

AIPerf synthesizes prompt text from a named corpus when the dataset does not
already carry verbatim content. Author the corpus as ``prompts.corpus`` in YAML
or pass ``--prompt-corpus`` on the CLI.

## Values

| Value | Content |
|-------|---------|
| `sonnet` | Shakespeare sonnets (default for synthetic and most loaders) |
| `coding` | Procedural coding / tool-use content |

## When it applies

Honored only where content is **synthesized**:

- synthetic datasets
- count / hash-id trace loaders (e.g. `mooncake_trace`, `bailian_trace`, `weka_trace`)
- public trace datasets that reconstruct from hash ids (e.g. SemiAnalysis weka HF)

Verbatim formats (`single_turn`, `multi_turn`, `baseten_trace`, …) ignore it.

## Defaults

When omitted, the active loader's ``default_prompt_corpus`` from the plugin
registry applies. Agentic coding loaders such as ``weka_trace`` default to
``coding``; most others default to ``sonnet``. Synthetic with no authored
corpus uses ``sonnet``.

## YAML shape

```yaml
datasets:
  - type: synthetic
    prompts:
      isl: 128
      corpus: coding

  - type: file
    format: weka_trace
    path: ./traces/
    prompts:
      corpus: coding
```

Do not author a top-level ``prompt_corpus`` field; use ``prompts.corpus``.
```

- [ ] **Step 2: Register in `docs/index.yml`**

Under the Reference section, add:

```yaml
    - page: Prompt Corpus Selection
      path: reference/prompt-corpus.md
```

(near `reference/isl-tokenization.md`)

- [ ] **Step 3: Tighten CLI Field description**

In `cli_config.py` `prompt_corpus` Field description, append that it projects to `prompts.corpus` and is honored only for synthesized content.

- [ ] **Step 4: Regenerate CLI docs + validate index**

```bash
make generate-cli-docs
python tools/check_docs_index.py
```

- [ ] **Step 5: Commit**

```bash
git add docs/reference/prompt-corpus.md docs/index.yml docs/cli-options.md src/aiperf/config/flags/cli_config.py
git commit -s -m "$(cat <<'EOF'
docs: document prompts.corpus seam

Describe sonnet/coding selection, defaults, and the YAML/CLI authored shape.
EOF
)"
```

---

### Task 7: Final verification

**Files:** none new

- [ ] **Step 1: Format / lint**

```bash
ruff format . && ruff check --fix .
```

- [ ] **Step 2: Unit tests for touched areas**

```bash
uv run pytest tests/unit/dataset/generator/test_corpus_resolver.py tests/unit/dataset/composer/test_synthetic_prompt_corpus.py tests/unit/dataset/composer/test_coding_corpus_injection.py tests/unit/config/test_prompt_corpus_authored_shape.py tests/unit/config/test_trace_flag_routing.py tests/unit/config/test_dataset_content_config_defaults.py tests/unit/dataset/test_mmap_cache.py -n auto
```

Expected: all PASS

- [ ] **Step 3: Pre-commit on changed files**

```bash
pre-commit run --all-files
```

Fix any schema / CLI doc / license hook fallout; commit hook-driven regenerations if needed with `git commit -s -m "chore: refresh generated artifacts after prompt-corpus seam"`

- [ ] **Step 4: Done checklist vs spec**

Confirm each spec requirement has a task deliverable:

| Spec item | Task |
|---|---|
| `prompts.corpus` authored shape | 2 |
| CLI projection | 3 |
| Shared factory | 1 |
| Synthetic honors coding | 4 |
| Trace honors + defaults | 4 |
| Hard cut flat field | 2 |
| mmap key | 5 |
| Docs | 6 |
| No `random` | — out of scope |

---

## Self-review (plan vs spec)

1. **Spec coverage:** All locked decisions and consumer paths have tasks. `random` / ISL repair explicitly out of scope.
2. **Placeholders:** None — concrete code and commands in each step.
3. **Type consistency:** `PromptSelectionConfig.corpus`, `PromptConfig.corpus`, `resolve_prompt_generator(...)`, `get_prompt_corpus()` used consistently across tasks.
