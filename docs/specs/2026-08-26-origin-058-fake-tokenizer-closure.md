# Origin #58 FakeTokenizer closure

## Purpose

Close origin/main `9b60a3d479` without inventing a Rust API for a Python-only
test double.

## Built

The upstream object is already in shared ancestry, but the semantic tree had
lost its Python `FakeTokenizer.num_prompt_special_tokens()` method. This
closure restores its exact zero return. Native tokenizers expose the
corresponding product concept through
`TextTokenizer::num_special_tokens_to_add()`: the trait default is zero and
the concrete Hugging Face implementation overrides it only when its tokenizer
post-processor adds bare-input tokens.  Native test fixtures inherit the zero
default.

## Non-goals

No Rust `FakeTokenizer`, Python interoperability layer, product configuration,
or synthetic native method is introduced. Native code must not mirror Python
test-harness structure when the actual product seam is already represented by
a Rust trait default.

## Source anchors

- `tests/harness/fake_tokenizer.py`
- `rust/runtime/src/dataset/tokenizer.rs`
- `artifacts/archives/origin-main-findings/commit-058-9b60a3d479.md`
