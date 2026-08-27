<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataset and prompt generation parity audit

Domain: synthetic prompt generation, sequence-length sampling, prefix reuse / cache
targeting, dataset loading (file, public, HuggingFace, trace), multi-turn
conversation construction.

**Python baseline (reference):** `/mnt/4tb/aiperf-parity-py-main/src/aiperf/`, git
rev `bc359bf8fd` (`origin/main`). All Python `path:line` citations below are
rooted at that worktree.

**Rust (new implementation):** `rust/` in this working tree.

Baseline-correction note: an earlier revision of this report compared against
`src/aiperf/` in this working tree, which sits on the feature branch
`ajc/native-rust-runtime-plugins` (4345 commits ahead of `origin/main`, 132
locally modified Python files including 21 under `dataset/`). That branch's local
merge reverted `PromptGenerator.generate`'s signature while keeping the new call
site, so the synthetic text path could not execute there. Every finding has been
re-derived against `bc359bf8fd`; branch artifacts are listed under
[Withdrawn after baseline correction](#withdrawn-after-baseline-correction).

`docs/dev/python-rust-parity-gaps.md` (dated 2026-07-17) contains no
dataset/prompt-generation entries overlapping any finding here, so all surviving
findings are marked NEW.

## Summary

The special-token ISL question is **resolved as parity**, not a gap: `origin/main`
Python subtracts `tokenizer.num_special_tokens_to_add(pair=False)` from the
synthetic ISL mean on exactly the same code path Rust does, with the same gating.
The earlier P0 was an artifact of comparing against a branch whose synthetic path
could not run.

What survives is narrower and centres on *exactness of the emitted prompt* rather
than the ISL budget. Python guarantees, via a decode/re-encode repair loop, that a
`sonnet` prompt's text re-encodes to exactly the requested length; Rust's corpus
generator has no such loop, and separately reports the composer's intended token
count rather than the wire count, so the resulting drift is invisible in metrics.
Python also subtracts chat-template and cache-bust overhead from the ISL budget
under `--apply-chat-template`; Rust has no counterpart, so that flag shifts wire
ISL in Rust and does not in Python. The hash-id block path still diverges twice:
Python derives block content from `(trace_id, hash_id)` while Rust derives it from
first-touch order, and Python's "prefix-only" hash-id layout is rejected outright
by Rust. Finally, a prefix pool configured with `length` but no `pool_size` raises
in Python and silently emits no prefix in Rust.

Counts: 2 P1, 4 P2, 1 P3 surviving. 5 findings withdrawn, 3 changed.

## Findings

### 1. `sonnet` prompts lose Python's exact-length guarantee, and Rust reports the intended count rather than the wire count

**Severity:** P1
**Status:** NEW (CHANGED — merges and re-scopes the former P1 #3)

Two mechanisms compound, and neither is observable from AIPerf's own output.

**(a) The repair loop.** Python's `PromptGenerator.generate_prompt` always routes
through `_decode_to_exact_len`, which decodes, re-encodes, and trims or tops up
until the *re-encoded* count equals the target:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:543:548
        tokens = self._sample_tokens(num_tokens)
        target = num_tokens
        if prefix_tokens:
            tokens = list(prefix_tokens) + tokens
            target += len(prefix_tokens)
        return self._decode_to_exact_len(tokens, target, _max_retries=_max_retries)
```

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:501:514
        for _ in range(_max_retries):
            text = self.tokenizer.decode(tokens)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) == target:
                return text
            if len(tokens) > target:
                tokens = tokens[:target]
            else:
                tokens = tokens + self._sample_topup_tokens(target - len(tokens))
        return self.tokenizer.decode(tokens)
```

Rust's `CorpusPromptGenerator` — which serves **both** `sonnet` and `coding` —
decodes once with no repair:

```415:426:rust/runtime/src/dataset/prompt.rs
    fn generate(
        &mut self,
        num_tokens: usize,
        hash_ids: &[i64],
        block_size: usize,
    ) -> Result<GeneratedPrompt> {
        let tokens = self.build_token_ids(num_tokens, hash_ids, block_size)?;
        Ok(GeneratedPrompt {
            text: self.tokenizer.decode(&tokens)?,
            tokens,
        })
    }
```

`generate_with_prefix` (`rust/runtime/src/dataset/prompt.rs:428-447`) likewise
concatenates and decodes without repair. Rust's `random` corpus *does* repair
(`repair_exact_text_tokens`, `rust/runtime/src/dataset/prompt.rs:940-959`), and
Python's `coding` generator does *not*
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/coding_content.py:785:791`
— a bare `decode`). So the divergence is scoped precisely to the **default
`sonnet` corpus**: Python exact, Rust unrepaired.

**(b) The reported number.** Python's reported ISL is always recomputed from the
exact wire bytes; the docstring is explicit that composer-side counts are never
read:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/records/inference_result_parser.py:365:379
        Source of truth is ``request_info.payload_bytes`` -- the exact JSON the
        endpoint sent on the wire -- decoded once and walked by
        ``extract_payload_inputs`` into tokenisable ``texts`` ... ``turns`` are
        never read here: on the payload-bytes fast path they are a content-free
        stub, and the canonical body is the bytes.
```

The `origin/main` synthetic composer sets no `turn.input_tokens` at all
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/synthetic.py:251:262`
appends contents and returns) — there is no composer-side count to prefer.

Rust records the pre-decode id count at composition
(`rust/runtime/src/dataset/loader/synthetic.rs:351-360`) and, by default, trusts
it verbatim:

```639:648:rust/runtime/src/engine/execute/ramp_adaptive.rs
pub(crate) fn select_input_token_counter(
    tokenizer: Arc<dyn TextTokenizer>,
    apply_chat_template: bool,
) -> Arc<dyn InputTokenCounter> {
    if apply_chat_template {
        Arc::new(EndpointInputTokenCounter::new(tokenizer, true))
    } else {
        Arc::new(AuthoredInputTokenCounter)
    }
}
```

Rust does have Python's wire re-tokenization
(`EndpointInputTokenCounter`, `rust/runtime/src/multiturn.rs:190-234`), but it is
only selected under `--apply-chat-template`. The stated justification — "the wire
body is exactly those pre-tokenized segments, so the authored count is already
exact" — holds for the repairing `random` generator and is exactly what mechanism
(a) breaks for `sonnet`/`coding`.

**Observable user impact:** at `--isl 550` on the default `sonnet` corpus, Python
puts exactly 550 body tokens on the wire and reports 550. Rust puts *approximately*
550 on the wire (BPE re-merge at the sampled slice boundary is the normal case) and
reports exactly 550 regardless. `input_sequence_length`, `total_isl`,
`input_token_throughput`, `total_token_throughput`, and
`usage_prompt_tokens_diff_pct` all inherit the difference, and Rust's ISL cannot be
used to detect the drift because it is the intended value by construction. A
second-order term stacks on multi-message payloads: Python joins the extracted
texts with a single space before encoding
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/records/inference_result_parser.py`,
`add_text_count` analogue) while Rust sums per-segment counts, so the seam is
charged on one side only.

**Confidence:** High for both mechanisms, read directly off both code paths. The
*magnitude* of (a) is tokenizer-dependent and unmeasured here.

### 2. Prefix pool with `length` but no `pool_size`: Python raises, Rust silently emits no prefix

**Severity:** P1
**Status:** NEW

`PrefixPromptConfig`'s only cross-field validator enforces mutual exclusivity
against the shared-system/user-context mode; it does **not** require `pool_size`
and `length` together:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/config/dataset/content.py:438:449
    @model_validator(mode="after")
    def _validate_prefix_exclusivity(self) -> Self:
        pool_group = (self.pool_size, self.length)
        system_group = (self.shared_system_length, self.user_context_length)
        has_pool = any(v is not None for v in pool_group)
        has_system = any(v is not None for v in system_group)
        if has_pool and has_system:
            raise ValueError(
                "pool_size/length and shared_system_length/user_context_length "
                "are mutually exclusive"
            )
        return self
```

With `length` set and `pool_size` absent, `prefix_prompt_enabled` is **True**
(it tests only `length > 0`):

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/base.py:562:573
    @property
    def prefix_prompt_enabled(self) -> bool:
        prefix_length = (
            self._synthetic_prefix_prompts.length
            if self._synthetic_prefix_prompts is not None
            else None
        )
        return (
            self.prompt_generator is not None
            and prefix_length is not None
            and prefix_length > 0
        )
```

…so `with_prefix` is set
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/synthetic.py:225`),
but `initialize_prefix_pool` returned early on `pool_size <= 0`:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:380:383
        length = self.prefix_prompts.length or 0
        pool_size = self.prefix_prompts.pool_size or 0
        if pool_size <= 0:
            return
```

…leaving `_prefix_prompt_tokens` empty, so the draw raises:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:748:755
    def _random_prefix_index(self) -> int:
        """Pick a pool slot, raising when the pool was never initialized."""
        if not self._prefix_prompt_tokens:
            raise InvalidStateError(
                "Attempted to sample a prefix prompt but the prefix prompts pool is empty. "
                "Please ensure that initialize_prefix_pool() has been called."
            )
        return self._prefix_rng.choice(range(len(self._prefix_prompt_tokens)))
```

Rust returns an empty pool for any missing half of the pair, and an empty pool is
simply skipped at use time:

```704:724:rust/runtime/src/dataset/loader/synthetic.rs
fn build_prefix_pool(
    prefixes: &SyntheticPrefixConfig,
    generator: &mut dyn PromptGenerator,
    requires_raw_token_ids: bool,
) -> Result<Vec<GeneratedPrompt>> {
    let (Some(size), Some(tokens)) = (prefixes.pool_size, prefixes.prefix_tokens) else {
        return Ok(Vec::new());
    };
```

```343:345:rust/runtime/src/dataset/loader/synthetic.rs
                            let selected_prefix = (turn_index == 0 && !prefix_pool.is_empty())
                                .then(|| prefix_rng.choice(&prefix_pool).ok().cloned())
                                .flatten();
```

Rust's config validation does not close the gap either — the only prefix-pool
check in `rust/runtime/src/config/validate.rs:236-244` is the system-prompt
exclusivity guard.

**Observable user impact:** `--prefix-prompt-length 200` without
`--num-prefix-prompts` fails loudly in Python at dataset composition and, in Rust,
produces a complete benchmark with zero prefix sharing, no warning, and no
indication that the requested cache-targeting configuration was discarded. The
user's KV-cache hit-rate measurement is silently invalid.

**Confidence:** High. Both control-flow paths are read directly; neither side has
a compensating validator.

### 3. `--apply-chat-template` shifts wire ISL in Rust and not in Python

**Severity:** P2
**Status:** NEW (CHANGED — this is the residual of the withdrawn P0 #1)

Python decomposes chat-template overhead into a per-request fixed cost and a
per-message wrap, and subtracts them from the synthetic ISL draw so the *wire*
length still matches `--isl`:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/base.py:294:320
    @property
    def first_turn_isl_adjustment(self) -> int:
        return (
            self._chat_template_per_request_fixed_tokens
            + self._chat_template_per_msg_wrap_tokens
            + self._first_turn_cache_bust_marker_tokens
        )
    # ...
    @property
    def subsequent_turn_isl_adjustment(self) -> int:
        return self._chat_template_per_msg_wrap_tokens
```

Both components are zero unless the user opted in
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/base.py:258:270`),
and the subtraction is applied per turn with a prefix-aware floor:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/synthetic.py:229:242
        adjustment = (
            self.first_turn_isl_adjustment
            if is_first
            else self.subsequent_turn_isl_adjustment
        )
        floor = 0 if with_prefix else 1
        if adjustment > 0:
            isl = max(floor, isl - adjustment)
        isl = max(floor, isl)
```

Rust has no counterpart. `input_token_subtraction` is the *only* ISL subtraction
in the synthetic path (`rust/runtime/src/dataset/loader/synthetic.rs:281`), and it
carries only the special-token count (finding: parity, see
[Checked and consistent](#checked-and-consistent)). Grepping
`rust/runtime/src/` finds no chat-template or cache-bust marker token-cost
estimate feeding any budget.

**Observable user impact:** `--isl 550 --apply-chat-template` puts ~550 tokens on
the wire in Python (the template wrapping is compensated) and 550 + template
overhead in Rust. The overhead is a small per-message constant (role header + EOT,
plus a one-off generation-prompt suffix), so the divergence is a handful of tokens
per request — but it grows with turn count and is exactly the quantity Python went
out of its way to hold constant. Note also that Python's first-turn adjustment is
larger than its subsequent-turn adjustment; Rust has no per-turn structure to
diverge from. P2 rather than P1 because it requires an explicit opt-in flag and the
magnitude is small and bounded.

**Confidence:** High for both sides. The precise token counts depend on
`_estimate_chat_template_overheads`'s probe and were not measured.

### 4. "Prefix-only" hash-id layout is accepted and tail-padded by Python, rejected by Rust

**Severity:** P2
**Status:** NEW (STILL VALID — re-cited; severity lowered from P1 because the
Rust refusal is loud and the `block_size` amplifier of the former #4 is withdrawn)

Python supports three layouts, including hash ids that cover only the cached
prefix, padding the un-hashed tail from the corpus:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:597:600
        - **Prefix only** (``M*block_size < num_tokens``): real
          captured traces (e.g. weka kv-cache-tester) where ``hash_ids`` lists
          only the cached prefix and the un-hashed tail represents fresh
          tokens. The tail is padded with sampled (uncached) tokens.
```

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:674:679
        # Prefix-only: pad the un-hashed tail with sampled (uncached) tokens.
        tail = num_tokens - len(final_prompt)
        if tail > 0:
            final_prompt.extend(self._sample_tokens(tail))

        return final_prompt
```

Rust admits only exact-tile and last-block-partial; a final block larger than
`block_size` is an error:

```475:483:rust/runtime/src/dataset/prompt.rs
            let final_size = num_tokens
                .checked_sub(prefix)
                .filter(|size| *size <= block_size);
            let final_size = final_size.ok_or_else(|| {
                DatasetError::Validation(format!(
                    "input length {num_tokens}, {} hash ids, and block size {block_size} are incompatible",
                    hash_ids.len()
                ))
            })?;
```

The trace loaders call straight in with `?`
(`rust/runtime/src/dataset/loader/trace.rs:324-328` mooncake, `:488` bailian), so
there is no softening.

**Observable user impact:** a captured mooncake- or bailian-shaped trace whose
`hash_ids` describe only the cached prefix — the documented weka kv-cache-tester
shape — loads and replays in Python and fails dataset construction in Rust. The
refusal is loud, which caps severity, but it is undocumented and removes a working
input shape.

**Confidence:** High.

### 5. Hash-id block content is keyed by `(trace_id, hash_id)` in Python and by first-touch order in Rust

**Severity:** P2
**Status:** NEW (STILL VALID — re-cited to baseline; severity lowered from P1
because within-process cache structure is preserved and the cross-composer
consequence remains untraced)

Python reseeds a hash-scoped RNG before sampling each new block:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:643:659
        for index, hash_id in enumerate(hash_ids):
            current_block_size = final_block_size if index == m - 1 else block_size
            cached = self._cache.get(hash_id)
            if cached is None:
                # Reseed per-(trace_id, hash_id) so the same hash_id in a
                # different trace file (different trace_id scope) produces
                # different tokens. ...
                self._hash_id_corpus_rng.reseed_for_hash_id(hash_id)
                cached = sample_tokens_from_corpus(
                    self._tokenized_corpus,
                    current_block_size,
                    self._hash_id_corpus_rng,
                    self.tokenizer.block_separation_token_id,
                )
                self._cache[hash_id] = cached
```

Rust draws the block from the generator's single sequential corpus stream at the
moment the hash id is first seen, with no reseed:

```491:501:rust/runtime/src/dataset/prompt.rs
                if !self.blocks.contains_key(hash_id) {
                    let mut block = Vec::with_capacity(size);
                    if let Some(separator) = self.block_separator {
                        block.push(separator);
                        block.extend(self.sample_tokens(size.saturating_sub(1))?);
                    } else {
                        block.extend(self.sample_tokens(size)?);
                    }
                    self.blocks.insert(*hash_id, block);
                }
```

Rust *has* the hash-scoped generator (`rust/runtime/src/rng/hash_id.rs`,
`reseed_for_hash_id`); `CorpusPromptGenerator` does not use it.

**Observable user impact:** in Python the token content of block `H` is a pure
function of `(trace_id, H)`, identical regardless of which rows are composed, in
what order, or by which process. In Rust it depends on how many other distinct
hash ids were materialized before `H` in that generator's walk. Within one
composition pass memoization still guarantees `H` always expands to the same
tokens, so single-process cache-hit structure is preserved. The exposure is any
configuration where more than one generator composes overlapping hash ids, or
where row ordering/filtering changes (session subsetting, or a `--cells N` split
where each cell composes a different subset): blocks Python would make
token-identical across composers become different tokens, and cross-composer
prefix-cache hits vanish.

**Confidence:** High for the order-dependence. The cross-cell consequence is a
reasoned inference, not a traced cellular execution — see Unverified.

### 6. Prefix-pool prefixes are exact-length in Python and unrepaired in Rust

**Severity:** P2
**Status:** NEW

Python's `_create_prefix_tokens` length-corrects each pooled prefix through the
same decode/re-encode contract as the body, so a pool entry is exactly `length`
tokens *as text*:

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:415:426
        if self._corpus == PromptCorpus.RANDOM:
            generator = getattr(self, "_preseed_rng", None) or self._corpus_rng
            n = len(self._allowed_tokens)
            tokens = [
                self._allowed_tokens[int(i)]
                for i in generator.integers(0, n, size=length)
            ]
        else:
            tokens = self._sample_tokens(length)

        text = self._decode_to_exact_len(tokens, length)
        return self.tokenizer.encode(text, add_special_tokens=False)
```

Python then prepends those ids to the body and repairs the *combined* sequence to
`len(prefix) + num_tokens`
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/generator/prompt.py:545:548`),
which the docstring notes avoids charging BPE for the seam (AIP-1118). Rust's
`generate_prefix` / `generate_with_prefix` on `CorpusPromptGenerator` do neither
(`rust/runtime/src/dataset/prompt.rs:428-447`).

**Observable user impact:** the shared prefix is still *token-identical* across
prompts in Rust — the pool entry is materialized once and its ids are reused
verbatim — so cache targeting works. What differs is the achieved total length:
Python emits exactly `prefix_len + isl` wire tokens, Rust emits that count of
composed ids whose decoded text may re-encode to a different number, and the seam
between prefix and body is uncorrected. This is the prefix-path instance of
finding 1. Python's `coding` generator shares Rust's behavior here
(`.../coding_content.py:800:816` samples the pool without repair), so the
divergence is again scoped to `sonnet`.

**Confidence:** High.

### 7. Turn-delay draw: Python truncates the mean and clamps to >= 1 ms, Rust keeps the fractional mean and ceils with no floor

**Severity:** P3
**Status:** NEW (STILL VALID — re-cited to baseline; severity lowered from P2)

```/mnt/4tb/aiperf-parity-py-main/src/aiperf/dataset/composer/synthetic.py:180:185
        if not is_first and self._turn_delay_mean > 0:
            delay = self._delay_sampler_rng.sample_positive_normal_integer(
                int(self._turn_delay_mean),
                self._turn_delay_stddev,
            )
            turn.delay = delay * self._turn_delay_ratio
```

`sample_positive_normal_integer` returns `max(1, round(mean))` for `stddev <= 0`
and `max(1, ceil(sample))` otherwise
(`/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/random_generator.py:313:315`).

```224:232:rust/runtime/src/dataset/loader/synthetic.rs
                if turn_index > 0 && shape.turn_delay_ms.expected_value() > 0.0 {
                    let delay = shape
                        .turn_delay_ms
                        .sample(&mut delay_rng)
                        .map_err(|error| DatasetError::Validation(error.to_string()))?
                        .ceil()
                        * shape.turn_delay_ratio;
                    turn.delay_ms = Some(delay);
                }
```

**Observable user impact:** `turn_delay: {mean: 10.7, stddev: 0}` yields 10 ms in
Python (`int(10.7)` then `round`) and 11 ms in Rust (`ceil(10.7)`). Python
guarantees at least 1 ms of think time per non-first turn; Rust does not.
`turn_delay_ratio` is a post-draw multiplier on both sides, so its semantics
agree. Bounded to <= 1 ms per turn boundary, hence P3.

**Confidence:** High.

## Withdrawn after baseline correction

- **Former P0 #1 — "Rust subtracts tokenizer special tokens from every synthetic
  ISL; Python subtracts only chat-template/cache-bust overhead."** INVALID on all
  three arms, each verified separately (the withdrawal is not blanket; see
  [Checked and consistent](#checked-and-consistent) for the per-arm derivation,
  including the VLLM-vs-SGLANG style split on the range-ratio arm and the proof
  that Rust's `input_token_subtraction` is never consumed on the
  sequence-distribution arm). The residual chat-template difference is now
  finding 3.
- **Former P0 #2 — "Python's synthetic text path raises `NameError` at HEAD,
  making the prefix-prompt pool dead code."** INVALID, branch artifact. On
  `origin/main` `generate()` declares the kwargs
  (`.../generator/prompt.py:428:436`), `CodingContentGenerator.generate` does too
  (`.../generator/coding_content.py:762:771`), and the caller defines both locals
  (`.../composer/synthetic.py:225` and `:246`). The prefix pool is live.
- **Former P1 #4 — "`bailian_trace` default `block_size` is 512 in Python and 16
  in Rust."** INVALID. `origin/main` carries the default in plugin metadata, not
  the shared `InputTokensDefaults.BLOCK_SIZE`:
  `/mnt/4tb/aiperf-parity-py-main/src/aiperf/plugin/plugins.yaml:539:541` declares
  `default_block_size: 16` for `bailian_trace`, matching
  `rust/runtime/src/dataset/loader/trace.rs:475`. Mooncake is 512 on both
  (`plugins.yaml:559:561`, `trace.rs:219`).
- **Former P2 #7 — "prefix pool `pool_size` without `length` is accepted by Python
  and rejected by Rust."** INVALID in that direction. Python's
  `prefix_prompt_enabled` requires `length > 0`
  (`.../composer/base.py:562:573`), so `pool_size` alone silently disables the
  prefix; Rust's `build_prefix_pool` returns an empty pool for the same input
  (`rust/runtime/src/dataset/loader/synthetic.rs:709:711`). Consistent. The
  *reverse* asymmetry is real and is now finding 2.
- **Former P2 #9 — "zero-stddev exactness: Python rounds the mean, Rust's
  distribution path ceils it."** INVALID as a reachable divergence. Python's
  non-`seq_distribution` path truncates to `int(prompts.isl.expected_value)`
  before sampling (`.../composer/base.py:454:459`), so a fractional authored mean
  never reaches the rounding step, and
  `RandomGenerator::sample_positive_normal_integer` already uses
  `mean.round_ties_even()` to match Python's banker's rounding
  (`rust/runtime/src/rng/generator.rs:382`). A fractional `isl.mean` would take
  `SamplingDistribution::sample_int`'s `ceil`
  (`rust/runtime/src/rng/dist.rs:72:74`) against Python's truncation, but `--isl`
  is integer-typed on both surfaces, so no authored config reaches it. Noted in
  Unverified instead.
- **Former P2 #10 — "requested length longer than the corpus: Python warns and
  returns a short prompt, Rust silently repeats the corpus."** CHANGED to a
  warning-only difference, not counted as a finding. `origin/main` Python does
  wrap once and warn (`.../generator/prompt.py:732:743`), but
  `_decode_to_exact_len`'s top-up path then restores the requested length, so both
  sides emit an exact-length prompt containing repeated corpus text. Only the
  warning differs; Rust emits none.
- **Former low-severity note — "Rust accepts `--prompt-corpus random`, Python's
  enum has only `sonnet` and `coding`."** INVALID, branch artifact. `origin/main`
  declares `RANDOM = "random"`
  (`/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/enums/enums.py`, `PromptCorpus`)
  and `resolve_prompt_generator` dispatches it to a vLLM-style `PromptGenerator`
  (`.../generator/corpus.py:35:46`). No silent `sonnet` fallback.

## Checked and consistent

**Special-token ISL subtraction (the re-derived P0).** Both sides subtract
`tokenizer.num_special_tokens_to_add(pair=False)` from the synthetic ISL budget,
on the same three-way gate:

- *Plain `isl` mean (no `sequence_distribution`, no `random_range_ratio`).*
  Python subtracts from the mean before the draw:
  `isl_mean = max(0, isl_mean - self._num_special_tokens)`
  (`.../composer/base.py:469:470`), where `_num_special_tokens` is
  `tokenizer.num_prompt_special_tokens()` (`.../composer/base.py:182:184`), itself
  a direct pass-through to HF `num_special_tokens_to_add(pair=False)`
  (`.../common/tokenizer.py:897:904`). Rust subtracts from the sampled value:
  `sampled.saturating_sub(prompt.input_token_subtraction).max(1)`
  (`rust/runtime/src/dataset/loader/synthetic.rs:280:282`), fed from
  `tokenizer.num_special_tokens_to_add()`
  (`rust/runtime/src/engine/execute/dataset_build.rs:536`), which reads the
  `tokenizers` post-processor's `added_tokens(false)`
  (`rust/runtime/src/dataset/tokenizer.rs:745:749`) — the same quantity HF's fast
  tokenizer delegates to. Subtracting an integer before vs. after the `ceil` is
  arithmetically identical, so the draws agree.
- *`sequence_distribution` authored (paired ISL/OSL entries).* Two distinct
  distribution families live in
  `/mnt/4tb/aiperf-parity-py-main/src/aiperf/common/models/sequence_distribution.py`
  and `get_sequence_distribution` dispatches between them
  (`.../config/dataset/content.py:317:353`). The paired-entry branch builds
  `SequenceLengthDistribution(pairs)` (`content.py:317:337`), whose constructor
  takes only `pairs` (`sequence_distribution.py:140`) — the
  `num_special_tokens=` forward at `content.py:349` belongs to the *other*
  branch. So Python applies no special-token subtraction here.

  Rust sets `prompts.input_token_subtraction` on this path
  (`dataset_build.rs:530:537` fires because `sequence_distribution` implies
  `random_range_ratio.is_none()`), but the field is **never consumed**: its only
  read is `rust/runtime/src/dataset/loader/synthetic.rs:281`, inside the
  `None =>` arm of `match paired_lengths`, and an authored
  `sequence_distribution` populates `compose.sequence_length_distribution`
  (`dataset_build.rs:480:484`) so `paired_lengths` is always `Some`
  (`synthetic.rs:252:258` maps a `Some` distribution to a `Some` pair). Setting
  the field is inert here. Both sides: no subtraction, no double-count.
- *`random_range_ratio` authored.* This is the family the
  `num_special_tokens` plumbing serves, and the behavior is **style-dependent on
  both sides** via `random_corpus_style`. `_CLASS_FOR_MODE`
  (`sequence_distribution.py:1107:1110`) keys `RangeRatioDistribution` /
  `SGLangRangeRatioDistribution` off `RandomCorpusStyle`:
  - *VLLM* folds the count into the window —
    `adjusted = max(0, self.isl_mean - self.num_special_tokens)`
    (`VLLMRatioConfig.compute_input_bounds`, `sequence_distribution.py:712`) — and
    `adjust_sampled_isl` is an explicit no-op (`:897:908`). Rust matches:
    `bounds((input_mean - special_tokens).max(0), ratio.input, 0)`
    (`rust/runtime/src/dataset/random_range.rs:157:159`) with
    `adjust_input` returning `input` unchanged (`:216`).
  - *SGLANG* derives the window from the raw mean and shifts each drawn length
    afterwards — `max(1, isl - self._config.num_special_tokens)`
    (`SGLangRangeRatioDistribution.adjust_sampled_isl`,
    `sequence_distribution.py:1083`) — because subtracting from the mean first
    would rescale the lower bound rather than shift the distribution. Rust
    matches: SGLang bounds are `[max(1, floor(mean*r)).min(mean), mean]` from the
    raw mean (`random_range.rs:161:174`) and
    `adjust_input` is `(input - self.special_tokens).max(1)` (`:217`).
  - The apply-once discipline also matches: Python stores preseeded values
    already-adjusted and applies `adjust_sampled_isl` only on the live-draw path
    (`sequence_distribution.py:893`, `:1093:1099`, with the docstring at
    `:905:907` warning that doing both would subtract twice); Rust applies
    `adjust_input` once in `preseed` (`random_range.rs:241`) and once in the
    fallback live draw (`:318`).

  Both: style-correct, applied once, not double-charged.
- *Per turn, first vs. subsequent.* The subtraction is unconditional per draw on
  both sides — Python's is baked into the cached
  `_get_turn_sequence_lengths` result, Rust's into each `sample_int`. Neither
  varies it by turn index.
- *With the prefix path active.* Prefixes are additive on both sides and are
  deliberately excluded from the subtraction: Python states it explicitly
  ("Prefix prompts are additive: total wire ISL is body + prefix ... The prefix is
  NOT subtracted from the ISL budget", `.../composer/base.py:288:292`), and Rust
  prepends `prefix_tokens` ahead of a full `input_tokens` body
  (`rust/runtime/src/dataset/prompt.rs:428:447`).

**Prefix-reuse comparison (re-examined with Python's path live).**

- *Token-identity of the shared prefix.* Both materialize each pool entry once and
  reuse its ids verbatim: Python caches `_prefix_prompt_tokens`
  (`.../generator/prompt.py:393:395`) and prepends ids, not strings
  (`:545:547`); Rust caches `GeneratedPrompt.tokens`
  (`rust/runtime/src/dataset/loader/synthetic.rs:704:724`) and prepends ids
  (`rust/runtime/src/dataset/prompt.rs:433:434`). Both avoid the string-join BPE
  seam. Consistent.
- *Whether the prefix counts toward requested ISL.* Additive on both sides, as
  above. `--isl 550` with a 200-token prefix targets 750 wire tokens on both.
  Consistent.
- *Per-prompt vs. per-session application.* Both draw a fresh pool entry per
  prompt (inside the batch loop) and both gate on the first turn only: Python
  `with_prefix = is_first and self.prefix_prompt_enabled`
  (`.../composer/synthetic.py:225`) with the draw inside
  `for _ in range(self._prompt_batch_size)` (`:251:260`); Rust
  `turn_index == 0 && !prefix_pool.is_empty()` inside
  `for _ in 0..prompt.batch_size`
  (`rust/runtime/src/dataset/loader/synthetic.rs:342:345`). Consistent.
- *Pool sampling.* Uniform choice with replacement from a dedicated prefix RNG
  stream on both sides (`.../generator/prompt.py:755` `self._prefix_rng.choice`;
  `rust/runtime/src/dataset/loader/synthetic.rs:344` `prefix_rng.choice`).
  Consistent in distribution; the concrete sequence differs because the RNG
  algorithms differ, which is out of scope for this domain.

**Other items verified against the baseline.**

- Default synthetic ISL 550, stddev 0 on both sides
  (`.../config/dataset/defaults.py:44:45`,
  `rust/runtime/src/config/model/dataset.rs:156-162`).
- Mooncake `block_size` default 512 and bailian 16 on both sides
  (`.../plugin/plugins.yaml:539:541`, `:559:561`;
  `rust/runtime/src/dataset/loader/trace.rs:219`, `:475`).
- Default `prompts.batch_size` 1 on both sides
  (`.../config/dataset/content.py`, `rust/runtime/src/config/model/dataset.rs:146-148`).
- Default `turn_delay_ratio` 1.0, applied as a post-draw multiplier on both sides
  (`.../composer/synthetic.py:185`,
  `rust/runtime/src/dataset/loader/synthetic.rs:230`).
- `turns` expected value must be at least 1, enforced as a validation error rather
  than silently floored (`.../composer/synthetic.py:54` via `max(1, ...)` on the
  cached mean plus config validation;
  `rust/runtime/src/dataset/loader/synthetic.rs:556-560`).
- The prefix-pool mode and the shared-system/user-context mode are mutually
  exclusive on both sides, enforced as an error
  (`.../config/dataset/content.py:438:449`,
  `rust/runtime/src/dataset/loader/synthetic.rs:668-675`).
- A hash id re-requested at a different block size is a hard error on both sides
  rather than a silent re-materialization
  (`.../generator/prompt.py:660:670`, `rust/runtime/src/dataset/prompt.rs:501-507`).
- Both hash-id paths prepend a block-separation token when the tokenizer exposes
  one (`.../generator/prompt.py:653:658` via `sample_tokens_from_corpus`,
  `rust/runtime/src/dataset/prompt.rs:493-495`).
- Client tokenization stays authoritative for ISL unless `use_server_token_count`
  is set, on both sides
  (`.../records/inference_result_parser.py`, `rust/runtime/src/metrics.rs:571-598`).
- Rust *does* possess Python's wire-payload re-tokenization, including the
  chat-template branch (`EndpointInputTokenCounter`,
  `rust/runtime/src/multiturn.rs:190-234`); the divergence in finding 1 is which
  counter is selected, not a missing capability.
- `--prompt-corpus random` exists on both sides, dispatched to a vLLM-style
  generator (`.../generator/corpus.py:35:46`,
  `rust/runtime/src/dataset/prompt.rs` `RandomPromptGenerator`), and both repair
  the round trip on that corpus.

## Unverified / needs runtime check

Resolved since the previous revision: the special-token subtraction question
(now parity, see Checked and consistent), the chat-template overhead comparison
(now finding 3), the prefix-reuse comparison (now parity plus finding 6), and the
prefix-pool validation question (reversed into finding 2).

Still open:

- **Magnitude of finding 1.** How far an unrepaired `sonnet` prompt's decoded text
  drifts from the requested token count is tokenizer-dependent and was not
  measured. Needs: a run with a pinned HF tokenizer at a fixed `--isl` on both
  sides, comparing per-record `input_sequence_length` and the server's own
  `usage.prompt_tokens` against the request.
- **Cross-cell consequence of finding 5.** Whether a `--cells N` mooncake replay
  actually composes hash-id blocks in more than one generator (and therefore loses
  cross-cell prefix identity) depends on the cellular dataset-distribution path,
  which was not traced. Needs: reading the controller/cell dataset push in
  `rust/runtime/src/cellular/` plus an `--cells 2` mooncake run comparing the
  materialized text of a hash id present in both cells.
- **HuggingFace field inference.** Rust's `hf` loader infers prompt/completion
  fields from a sample row and resolves subset/split through the dataset-viewer
  `/info`. Whether Python's public-dataset loaders infer the same fields for the
  same repo, and whether subset/split defaults agree, was not compared. Needs: a
  side-by-side load of one `--hf-dataset` repo and a diff of the first N composed
  prompts.
- **`sequence_distribution` per-entry draw.** Python builds `SequenceLengthPair`s
  carrying per-entry ISL/OSL stddev
  (`.../config/dataset/content.py:323:337`) and Rust has a
  `PROBABILITY_SUM_ABS_TOLERANCE` against 100.0
  (`rust/runtime/src/rng/dist.rs:724-726`), consistent at a glance, but the
  per-entry draw order, stddev application, and index-selection tie behavior were
  not compared in detail.
- **Fractional authored `isl.mean`.** Python truncates
  (`int(prompts.isl.expected_value)`, `.../composer/base.py:454:459`) where Rust
  ceils (`SamplingDistribution::sample_int`,
  `rust/runtime/src/rng/dist.rs:72:74`). Whether any Config-v2 surface admits a
  non-integer `dataset.prompts.isl.mean` was not confirmed; if one does, this is a
  silent off-by-one.
- **Multi-turn history accumulation.** Whether message-array growth, role
  assignment, and response echoing accumulate identically across turns was not
  compared. Needs: a two-turn synthetic run on both sides with raw payload
  capture.
- **Batched embeddings/rankings input sampling.** Rust's rankings path draws
  passage count, passage length, and query length from three separate streams
  (`rust/runtime/src/dataset/loader/synthetic.rs:464-518`); the draw order and
  clamping in `.../composer/synthetic_rankings.py` were not read.
