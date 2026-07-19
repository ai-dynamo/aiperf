# Synthetic Dataset Generator explainer — design

## Goal

Add a new `apps/explainers` deck that explains how the synthetic dataset
generator combines text tokens and media into a single `Turn`: the two
top-level branches (token-native no-decode vs. text generation), how prompt
and media segments are chained through a shared `parent` handle, how they
land in `Turn.content` as `ContentGroup`s, and how `Turn::dispatch_body`
picks the final wire representation.

## Source of truth

- `rust/runtime/src/dataset/loader/synthetic.rs` — `SyntheticComposer::compose`,
  `append_media_batch`.
- `rust/runtime/src/dataset/generator/mod.rs` — `SyntheticDatasetConfig`,
  `SyntheticPromptConfig`, `SyntheticPrefixConfig`, `SyntheticMediaGenerator`.
- `rust/runtime/src/dataset/model.rs` — `Turn::dispatch_body`, `ContentGroup`.

## Deck metadata

- `id: "synthetic-dataset-generator"`
- `route: "/synthetic-dataset-generator"`
- `topic: "dataset"` (sibling of `segment-pools` in the hub)
- Source file: `apps/explainers/decks-flow/synthetic-dataset-generator.flow`
- Registered in `EXPECTED_DECK_ROUTES` in
  `apps/explainers/src/core/deck-registry.ts`, following the existing
  `segment-pools` / `tstar-warmup` registration pattern exactly (same tuple
  shape, same route-bookmark stability requirement).

## Slide plan (9 slides)

1. **Overview** — `SyntheticDatasetConfig` shape fields (entries, turns,
   prompts, prefixes, images, audio, video); frame the deck's question: text
   and media both end up as `Handle`s chained into one `Turn`.
2. **Two branches at the top** — the composer forks on
   `config.requires_raw_token_ids`: token-native (no-decode) vs. standard
   text generation. Show the fork as the first decision point.
3. **No-decode branch** — `generator.generate_token_ids` →
   `segments.intern_token_ids` directly; `turn.input_tokens` set from the
   token count; `Turn::dispatch_body(None, Some(handle), &[])`. No text is
   ever constructed.
4. **Standard text branch, prefix reuse path** — `PrefixReuse::prompt_tokens`
   assembles shared-prefix + unique-suffix token ids, decodes once, interns
   via `segments.intern_text`, and is pushed as a `ContentGroup { kind: Text
   }`.
5. **Standard text branch, generated-prompt path** — `PromptGenerator::generate`,
   optional prefix-pool prefix interned as its own parent segment (visible in
   the content-addressed chain) ahead of the full text segment, then
   tokenized and interned as the child.
6. **Media batches join the same chain** — `append_media_batch` for
   image/audio/video: each `GeneratedMedia` value is interned via
   `segments.intern_media` with `parent` threaded forward from whatever came
   before it (text segment → image segment → ...); each media batch becomes
   its own `ContentGroup` (`image_url` / `input_audio` / `video_url`).
7. **Parent-chaining detail** — why `parent` threads through every intern
   call (content-addressed prefix chain / dedup key), and how a mixed
   text+image+audio+video turn ends up as multiple `ContentGroup`s sharing
   one parent lineage.
8. **`Turn::dispatch_body` precedence** — raw payload > raw token ids >
   `messages` (the `ContentGroup`s); why text+media turns route through
   `messages` while the no-decode branch bypasses it entirely. Cite the
   precedence unit test in `model.rs`.
9. **Recap** — one summary diagram of the full pipeline: shape config →
   prompt/media generators → segment-pool interning with parent chain →
   `ContentGroup` assembly → `dispatch_body` → wire `Turn`. File:line
   evidence citations.

## Style

Match `segment-pools.flow` conventions: SDK primitives (`Header`, `Stepper`,
`Panel`, `Edge`, `Callout`), each slide with `eyebrow` / `title` / `lede` /
`narration` / `term` / `points` / `caption` / `render` scene block, and a hub
entry with `highlight` / `title` / `description`.

## Verification

- `npm run build` in `apps/explainers`
- Existing fail-closed import-boundary scan
- `deckByRoute` / `validateDeckRegistry` checks that already gate the
  registry (no new test infra needed — same gate every existing deck passes
  through)

## Out of scope

- The rankings composer (`SyntheticRankingsComposer`) — different shape,
  no media combination, not part of this deck's question.
- Any change to `synthetic.rs`/`model.rs` runtime behavior — this is a
  documentation-only addition.
