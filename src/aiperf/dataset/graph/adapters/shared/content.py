# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared corpus-backed content synthesis for the recorded-trace adapters.

The dynamo adapter consumes it, and the trie core's content callbacks are built
on it.

Weka traces carry ``hash_ids`` + token counts (``in``/``out``), NOT real text.
The block-decode callback is a plain ``corpus[start:start+bs]`` window seeded
per ``(trace_id,
hash_id)`` (NO block-separation token, full-size blocks). Partial tails use
sha256-keyed corpus offsets so two processes produce identical bytes. The result
is byte-deterministic given ``(trace, root_seed, tokenizer)``.

:class:`CorpusContentSynthesizer` owns a corpus-backed content generator
(:class:`~aiperf.dataset.generator.coding_content.CodingContentGenerator` or
:class:`~aiperf.dataset.generator.prompt.PromptGenerator`) and exposes the
block-decode / partial-tail / decode-to-text callbacks the trie builders
(:mod:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering`) consume to emit
message-unit prompt segments. This module carries no
conversation-reconstruction state of its own.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

from aiperf.common import random_generator as rng

if TYPE_CHECKING:
    import numpy as np

    # The shm-backed int32 token array a parallel-parse worker attaches, or a
    # plain token-id list; both expose the ``__getitem__``/``__len__`` surface
    # the block-decode and partial-tail callbacks read.
    SharedCorpus = np.ndarray | list[int]
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import PromptConfig
from aiperf.dataset.generator.prompt import PromptGenerator

# Last-resort seed for offline weka synthesis when no ambient RNG manager exists
# and no explicit run seed was threaded in (offline tooling calling
# ``parse_graph`` directly, which never runs ``bootstrap``). A fixed value
# keeps offline content byte-deterministic across runs and processes.
_DEFAULT_OFFLINE_SEED = 42


@contextlib.contextmanager
def _seeded_global_rng(root_seed: int | None) -> Iterator[None]:
    """Pin the global RNG ``root_seed`` for the duration, then restore it.

    Weka content synthesis derives its corpus pool + per-hash reseed key from the
    GLOBAL ``rng`` manager's ``root_seed`` (see
    :meth:`CorpusContentSynthesizer._build_generator`). In the main benchmark path
    ``bootstrap`` seeds that manager with the run ``--random-seed`` before the
    dataset is parsed, so the global state is already correct. But the directory
    parse fans the per-file build across a multiprocessing pool; a ``fork``
    worker inherits the seeded ``_manager``, while a forkserver/``spawn`` worker
    starts with ``_manager is None`` and ``rng.derive`` raises
    ``InvalidStateError`` (no seed inherited). Threading the run seed in and
    pinning it here makes the synthesized bytes identical to the in-process
    build at ANY seed, regardless of the pool start method.

    ``root_seed is None`` defers to the ambient manager when one exists (the
    bootstrap-seeded in-process path + the unit-test auto-fixture seed).
    Production offline routes -- any direct ``parse_graph`` call from tooling
    with no run config -- resolve a concrete seed via
    :func:`resolve_effective_root_seed` BEFORE reaching this pin, so they pass a
    non-``None`` ``root_seed`` and never take the ``None`` branch. The pin
    survives only as a last resort: a caller that passes ``root_seed=None`` with
    no ambient manager would otherwise trip the ``rng.derive`` in
    :meth:`CorpusContentSynthesizer._build_generator` (``InvalidStateError``), so
    :data:`_DEFAULT_OFFLINE_SEED` is pinned for the duration to keep offline
    content deterministic. Either way the prior manager (including an unset
    ``None``) is restored on exit so a shared in-process caller (tests, the
    serial parse path) is not left with a mutated global.
    """
    if root_seed is None:
        if rng._manager is not None:
            yield
            return
        prior = rng._manager
        rng.init(_DEFAULT_OFFLINE_SEED)
        try:
            yield
        finally:
            rng._manager = prior
        return
    prior = rng._manager
    rng.reset()
    rng.init(root_seed)
    try:
        yield
    finally:
        rng._manager = prior


def resolve_effective_root_seed(root_seed: int | None) -> int:
    """Resolve the concrete content seed threaded into every recorded-trace parse route (dynamo).

    Explicit run seed wins; otherwise the ambient bootstrap-seeded manager's
    root seed; otherwise a fresh OS-entropy seed generated ONCE here. The
    resolved ``int`` is threaded (via the resolved parse kwargs) into the
    serial path and every pool worker, so one run's synthesized content is
    internally consistent at any parallel threshold while distinct unseeded
    runs differ — the unseeded fallback is deliberately NOT a hardcoded
    constant. No other process re-derives this seed: the DatasetManager parses
    once and the TimingManager ingests the sidecar it writes rather than
    re-parsing, so a per-process random fallback cannot diverge content within a
    run.
    """
    if root_seed is not None:
        return root_seed
    ambient = rng.root_seed()
    if ambient is not None:
        return ambient
    return secrets.randbits(64)


# Weka content corpus selector. A local string literal (only this module needs
# it, so no shared enum). ``"coding"`` matches the weka loader's
# ``default_prompt_corpus: coding`` -- the corpus the recorded weka workloads
# were captured against.
PromptCorpus = Literal["coding", "sonnet"]
PROMPT_CORPUS_CODING: PromptCorpus = "coding"


# ---------------------------------------------------------------------------
# dag-v3 integration: corpus-backed synthesizer exposing decode/sample callbacks.
# ---------------------------------------------------------------------------


class CorpusContentSynthesizer:
    """Owns a corpus-backed generator and exposes recorded-trace content callbacks.

    Determinism: given the same ``(trace, root_seed, tokenizer, corpus)`` the
    decoded block tokens are byte-identical to agentx's ``WekaTraceLoader`` wire
    payload. Block tokens are seeded per ``(trace_id, hash_id)`` via the
    generator's ``_hash_id_corpus_rng``; partial tails via ``sha256(seed_string)``
    corpus offsets.

    Corpus selection MUST match agentx's live weka path for byte-exact payload
    parity. agentx's ``CustomDatasetComposer`` honors the weka loader's
    ``default_prompt_corpus: coding`` (``plugins.yaml``) and injects a
    :class:`~aiperf.dataset.generator.coding_content.CodingContentGenerator`
    (a procedurally-built coding-text tool pool, RNG-keyed
    ``dataset.coding_content.corpus``) rather than the Shakespeare
    :class:`~aiperf.dataset.generator.prompt.PromptGenerator`
    (``dataset.prompt.corpus``). dag-v3 therefore defaults to ``"coding"`` so the
    graph synthesis draws from the SAME corpus and reseed key; the wire
    bytes then match agentx turn-for-turn (a plain ``PromptGenerator`` produces
    Shakespeare text with matching token COUNTS but entirely different BYTES,
    which is why a Shakespeare-backed synthesizer diverges from the live agentx
    A/B). ``"sonnet"`` selects the Shakespeare path for callers that want it.

    Both backends expose the same ``_tokenized_corpus`` / ``_corpus_size`` /
    ``_cache`` / ``_hash_id_corpus_rng`` / ``tokenizer`` surface the block-decode
    and partial-tail callbacks read, so the content path is corpus-agnostic.
    """

    def __init__(
        self,
        tokenizer_name: str,
        *,
        prompt_corpus: PromptCorpus = PROMPT_CORPUS_CODING,
        root_seed: int | None = None,
        shared_corpus: SharedCorpus | None = None,
    ) -> None:
        """Build the corpus-backed generator (or adopt a pre-built corpus).

        ``shared_corpus`` supplies a pre-built token array (the parallel
        parser's parent-built shared-memory block): the generator is then
        constructed WITHOUT paying its own corpus build and reads the supplied
        array directly. The array MUST be byte-identical to what this
        synthesizer's ``(tokenizer, prompt_corpus, root_seed)`` would build
        (the parent builds it with the SAME key), so decoded content is
        byte-identical to a self-built synthesizer.
        """
        self._tokenizer_name = tokenizer_name
        self._prompt_corpus = prompt_corpus
        self._root_seed = root_seed
        with _seeded_global_rng(root_seed):
            self._pg, self._hash_id_corpus_rng = self._build_generator(
                tokenizer_name, prompt_corpus, shared_corpus=shared_corpus
            )
        self._block_size = 64
        # First ``_corpus_size`` seen by :meth:`_decode_block_tokens_offset_cached`;
        # asserted per call so a corpus rebind that would invalidate the cached
        # offsets fails loud instead of silently reproducing wrong bytes.
        self._offset_cache_corpus_size: int | None = None

    @staticmethod
    def reset_worker_cache() -> None:
        """Drop the process-level synthesizer cache (test isolation)."""
        _WORKER_SYNTH_CACHE.clear()

    def corpus_tokens(self) -> list[int]:
        """Return the generator's built corpus token ids (``_tokenized_corpus``).

        Used by the shared worker pool
        (:mod:`~aiperf.dataset.graph.adapters.shared.pool`) to hoist the
        deterministic corpus into a single ``shared_memory`` block built ONCE in
        the parent process; every worker then attaches that block (CoW) instead
        of rebuilding the ~600K-token coding pool per worker -- the per-worker
        rebuild is what ballooned RSS into tens of GB on the real corpus.
        """
        return list(self._pg._tokenized_corpus)

    def attach_shared_corpus(self, corpus: SharedCorpus) -> None:
        """Point the generator's corpus at a pre-built (shared) token array.

        ``corpus`` must be the byte-identical token sequence
        :meth:`corpus_tokens` would have produced for this synthesizer's
        ``(tokenizer, prompt_corpus, root_seed)`` -- the parent builds it once
        with the SAME seed, so determinism (and the byte-exact serial/parallel
        parity contract) is preserved. Replaces ``_tokenized_corpus`` /
        ``_corpus_size`` in place; the block-decode and partial-tail callbacks
        read those attributes, so no other wiring changes. Prefer constructing
        with ``shared_corpus=`` instead, which skips the private corpus build
        entirely; this rebind is for an already-built synthesizer.
        """
        self._pg._tokenized_corpus = corpus
        self._pg._corpus_size = len(corpus)

    @staticmethod
    def _build_generator(
        tokenizer_name: str,
        prompt_corpus: PromptCorpus,
        shared_corpus: SharedCorpus | None = None,
    ) -> tuple[PromptGenerator, HashIdRandomGenerator]:
        """Build the corpus backend + its per-``(trace_id, hash_id)`` reseed RNG.

        Returns ``(generator, hash_id_corpus_rng)``. For ``"coding"`` the
        ``CodingContentGenerator`` already builds its own ``_hash_id_corpus_rng``
        in ``__init__`` (off ``rng.derive("dataset.coding_content.corpus")``),
        so it is used directly -- matching agentx's live weka path exactly. For
        ``"sonnet"`` the plain ``PromptGenerator`` does NOT build one (only
        ``CodingContentGenerator`` does), so it is derived here from the
        generator's ``_corpus_rng`` (``rng.derive("dataset.prompt.corpus")``).
        ``from_base_rng`` reads ``base_rng.seed`` without consuming RNG state, so
        the reseed key is stable.

        ``shared_corpus`` short-circuits the generator's own corpus build: the
        corpus-build step (``_build_tool_pool`` / ``_initialize_corpus``) is
        replaced with a direct adoption of the supplied array. Byte parity is
        preserved because the block-decode path depends only on the corpus
        CONTENT (supplied identical by contract) and on ``_hash_id_corpus_rng``,
        whose seed derives from the corpus-rng SEED (read without consuming
        state) -- never on RNG state the skipped build would have advanced
        (the build consumes only the independent ``_template_rng`` stream).

        Both the coding corpus POOL text (``_template_rng.shuffle`` block order in
        ``CodingContentGenerator._build_tool_pool``) and the per-``(trace_id,
        hash_id)`` reseed key (``_hash_id_corpus_rng.seed`` derived off
        ``dataset.coding_content.corpus``) read the GLOBAL RNG ``root_seed`` via
        ``rng.derive``. The caller (:meth:`__init__`) wraps this in
        :func:`_seeded_global_rng` so an explicit ``root_seed`` is honored even
        when the global manager is unset (e.g. a ``spawn``-started parallel parse
        worker that did NOT inherit the parent's seeded ``_manager``).
        """
        # In a forkserver worker the configured tokenizer is preloaded into
        # the helper heap and CoW-shared (see aiperf.dataset._tokenizer_preload);
        # prefer it to avoid each worker re-reading the tokenizer from disk. The
        # preload and the on-demand fallback both key on the SAME (name, trust,
        # revision) triple, so a hit and a fallback build produce identical
        # content.
        from aiperf.dataset._tokenizer_preload import (
            _env_revision,
            _env_trust_remote_code,
            get_preloaded,
        )

        trust_remote_code = _env_trust_remote_code()
        revision = _env_revision()

        tokenizer = get_preloaded(
            tokenizer_name, trust_remote_code=trust_remote_code, revision=revision
        )
        if tokenizer is None:
            # A transient load failure (network blip, HF down, auth) must NOT
            # silently degrade to the builtin tokenizer: the run would then
            # produce builtin-sized content while claiming the REQUESTED
            # name/revision. Fail loudly instead so a transient failure never
            # yields mislabeled content. resolve_alias=True mirrors the
            # forkserver preload (aiperf.dataset._tokenizer_preload._preload)
            # so a hit and a miss resolve the SAME tokenizer.
            tokenizer = Tokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
                resolve_alias=True,
            )

        if prompt_corpus == PROMPT_CORPUS_CODING:
            # Local import: the coding generator pulls in the procedural code
            # builders, kept off the import path for the SONNET/topology paths.
            from aiperf.dataset.generator.coding_content import (
                CodingContentGenerator,
            )

            if shared_corpus is None:
                ccg = CodingContentGenerator(config=PromptConfig(), tokenizer=tokenizer)
            else:

                class _SharedCorpusCodingContentGenerator(CodingContentGenerator):
                    """Adopts the supplied corpus instead of building the pool."""

                    def _build_tool_pool(self) -> None:
                        self._tool_pool = shared_corpus

                ccg = _SharedCorpusCodingContentGenerator(
                    config=PromptConfig(), tokenizer=tokenizer
                )
            return ccg, ccg._hash_id_corpus_rng

        # prefix-prompt pool disabled (``prefix_prompts=None``): weka synthesis
        # never draws from it, and a pool would only consume the independent
        # prefix RNG -- the block/tail RNGs are separate, so output is unchanged.
        if shared_corpus is None:
            pg = PromptGenerator(
                prompts=PromptConfig(),
                prefix_prompts=None,
                tokenizer=tokenizer,
            )
        else:

            class _SharedCorpusPromptGenerator(PromptGenerator):
                """Adopts the supplied corpus instead of tokenizing the file."""

                def _initialize_corpus(self) -> None:
                    self._tokenized_corpus = shared_corpus
                    self._corpus_size = len(shared_corpus)

            pg = _SharedCorpusPromptGenerator(
                prompts=PromptConfig(),
                prefix_prompts=None,
                tokenizer=tokenizer,
            )
        return pg, HashIdRandomGenerator.from_base_rng(pg._corpus_rng)

    # --- agentx-faithful content callbacks --------------------------------

    def _decode_block_tokens(
        self,
        hash_ids: list[int],
        *,
        block_size: int | None = None,
        cache: dict[int, list[int]] | None = None,
        trace_id: str | None = None,
    ) -> list[int]:
        """Concatenate per-hash-id token blocks (mirrors agentx exactly).

        Plain ``corpus[start:start+bs]`` window seeded per ``(trace_id,
        hash_id)``; NO block-separation token, full-size blocks.

        ``trace_id`` scopes the hash-id namespace, for corpora that declare
        ``hash_id_scope: "local"`` (ids are a per-trace namespace): a caller
        passes the trace's own id -- hash_id 0 in trace A and trace B
        then decode to DIFFERENT bytes, preventing manufactured cross-trace
        KV-cache sharing at the server. A ``trace_id`` caller MUST supply its
        own ``cache``: the shared ``pg._cache`` is keyed by bare hash id and
        would leak one trace's bytes into another. ``None`` keeps the GLOBAL
        namespace (the linear mooncake path, and dynamo's recorded chained
        sequence hashes, which are content-global identity by construction).

        ``block_size`` / ``cache`` also serve NON-weka trie block sizes (dynamo
        records 16-token blocks): the defaults keep the global single-namespace
        path byte-exact, while a caller decoding at a different size MUST also
        supply its own cache -- mixing block sizes in the shared cache would
        return wrong-sized blocks across adapters.
        """
        pg = self._pg
        r = self._hash_id_corpus_rng
        bs = self._block_size if block_size is None else block_size
        corpus = pg._tokenized_corpus
        corpus_size = pg._corpus_size
        if cache is None and trace_id is not None:
            raise ValueError(
                "trace-scoped block decode requires a per-trace cache; the "
                "shared cache is keyed by bare hash id and would leak bytes "
                "across traces"
            )
        cache = pg._cache if cache is None else cache
        tokens: list[int] = []
        for h in hash_ids:
            cached = cache.get(h)
            if cached is None:
                # Preserve the bare call shape on the default path: injected
                # test doubles (and any external RNG duck-type) predate the
                # trace_id kwarg.
                if trace_id is None:
                    r.reseed_for_hash_id(h)
                else:
                    r.reseed_for_hash_id(h, trace_id=trace_id)
                start = r.randrange(corpus_size)
                end = start + bs
                cached = list(corpus[start:end])
                if end > corpus_size:
                    cached.extend(corpus[: end - corpus_size])
                cache[h] = cached
            tokens.extend(cached)
        return tokens

    def _decode_block_tokens_offset_cached(
        self,
        hash_ids: list[int],
        *,
        block_size: int,
        offset_cache: dict[int, int],
    ) -> list[int]:
        """Memory-lean twin of :meth:`_decode_block_tokens` (GLOBAL namespace only).

        Byte-for-byte identical output to :meth:`_decode_block_tokens` for the
        same ``(hash_ids, block_size)`` and RNG object, but caches one ``int``
        corpus offset per hash id instead of the decoded ``list[int]`` block --
        ``~4-8 B`` vs ``~block_size`` ints per entry, the decode-cache tier's
        single largest on-peak shave at corpus scale. The
        ``tests/unit/dataset/graph/adapters/test_dynamo_trie_lowering.py``
        differential test pins the equality across miss/hit paths and
        list/``np.int32`` corpus backings.

        The equality rests on :meth:`~aiperf.common.hash_id_random_generator.HashIdRandomGenerator.reseed_for_hash_id`
        being a FULL reseed: ``start`` is a pure function of ``(seed, scope,
        hash_id, corpus_size)``, so the miss path here issues the IDENTICAL
        ``reseed_for_hash_id(h)`` + ``randrange(corpus_size)`` call pair (same
        RNG object, same order -> same RNG trajectory) that :meth:`_decode_block_tokens`
        issues on its first occurrence of ``h``. Both paths then read the same
        ``corpus[start:start+bs]`` window (wraparound included). Repeat
        (hit-path) decodes re-slice from the cached offset rather than reissuing
        RNG calls, so neither method touches the RNG on a repeat -- the
        trajectories stay aligned.

        **Corpus-immutability assumption.** A cached offset reproduces the
        original bytes ONLY while ``_tokenized_corpus`` / ``_corpus_size`` are
        unchanged, or rebound to a BYTE-IDENTICAL array (the
        :meth:`attach_shared_corpus` contract -- note :func:`get_or_build_synthesizer`
        rebinds on a cache HIT). A byte-identical rebind keeps ``_corpus_size``
        equal, so as a cheap tripwire the first-seen ``corpus_size`` is snapshotted
        and asserted per call: a contract-violating rebind to a DIFFERENT-sized
        corpus fails loud rather than silently emitting wrong bytes. (A same-size
        byte-different rebind is out of scope -- the size check is a guard, not a
        content hash; the contract above forbids it.)
        """
        pg = self._pg
        r = self._hash_id_corpus_rng
        bs = block_size
        corpus = pg._tokenized_corpus
        corpus_size = pg._corpus_size
        snapshot = self._offset_cache_corpus_size
        if snapshot is None:
            self._offset_cache_corpus_size = corpus_size
        elif snapshot != corpus_size:
            raise RuntimeError(
                f"corpus size changed under the offset cache ({snapshot} -> "
                f"{corpus_size}): cached offsets no longer reproduce the original "
                "bytes (see the corpus-immutability assumption on "
                "_decode_block_tokens_offset_cached)."
            )
        tokens: list[int] = []
        for h in hash_ids:
            # ``get`` (not truthiness): offset 0 is a legal cached start.
            start = offset_cache.get(h)
            if start is None:
                r.reseed_for_hash_id(h)
                start = r.randrange(corpus_size)
                offset_cache[h] = start
            end = start + bs
            if end > corpus_size:
                # Wraparound: verbatim two-step shape from _decode_block_tokens.
                block = list(corpus[start:end])
                block.extend(corpus[: end - corpus_size])
                tokens.extend(block)
            else:
                # DELIBERATE shape deviation from _decode_block_tokens (waived
                # per the Task-3 amendment): extend straight from the slice,
                # skipping its redundant ``list()`` copy -- the hit path runs
                # once per covered block, and that copy alone cost ~6.5% of a
                # full corpus-scale parse. Byte-identity holds: extend-from-
                # slice appends exactly the values ``list(corpus[start:end])``
                # would (for an np backing, iterating the view yields the same
                # np scalars ``list(view)`` yields), and the window is the same.
                tokens.extend(corpus[start:end])
        return tokens

    def _sample_partial_tail_tokens(self, n_tokens: int, seed: str) -> list[int]:
        """Deterministic per-seed partial-block tokens sized to ``n_tokens``.

        sha256-keyed corpus offset (PYTHONHASHSEED-independent, cross-process
        stable). Mirrors agentx ``HashIdsPromptSynthesisMixin``.
        """
        if n_tokens <= 0:
            return []
        pg = self._pg
        corpus_size = pg._corpus_size
        digest = hashlib.sha256(seed.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % max(corpus_size - n_tokens, 1)
        return list(pg._tokenized_corpus[offset : offset + n_tokens])

    def _decode_tokens_to_text(self, tokens: list[int]) -> str:
        return self._pg.tokenizer.decode(tokens)


# Process-level synthesizer cache keyed by (tokenizer, corpus, root_seed). In a
# parallel-parse worker this is populated once (by the shared pool's
# _init_worker, which also attaches the shared-memory corpus) and reused across
# every trace the worker handles -- so the ~600K-token coding pool is built ONCE
# per process and the corpus array is shared across workers, instead of rebuilt
# per trace (the rebuild-per-trace path ballooned RSS into tens of GB).
_WORKER_SYNTH_CACHE: dict[tuple[str, str, int | None], CorpusContentSynthesizer] = {}


def get_or_build_synthesizer(
    tokenizer_name: str,
    *,
    prompt_corpus: str,
    root_seed: int | None,
    shared_corpus: SharedCorpus | None = None,
) -> CorpusContentSynthesizer:
    """Return a cached synthesizer for the key, building one on first miss.

    The block cache is content-addressed by ``hash_id`` (see
    :meth:`CorpusContentSynthesizer._decode_block_tokens`), so reusing one
    synthesizer across traces is byte-identical to building a fresh one per
    trace -- the serial path (no cache populated) and the parallel path (cache
    populated in ``_init_worker``) produce the same wire bytes.

    ``shared_corpus`` (the parallel parser's parent-built shm array) is adopted
    instead of built: a cache MISS constructs the synthesizer directly on it
    (skipping the private corpus build entirely); a cache HIT rebinds the
    existing synthesizer's corpus to it (releasing any privately-built copy).
    Byte parity is unaffected -- the array is byte-identical to what the key
    would build.
    """
    key = (tokenizer_name, prompt_corpus, root_seed)
    synth = _WORKER_SYNTH_CACHE.get(key)
    if synth is None:
        synth = CorpusContentSynthesizer(
            tokenizer_name=tokenizer_name,
            prompt_corpus=prompt_corpus,  # type: ignore[arg-type]
            root_seed=root_seed,
            shared_corpus=shared_corpus,
        )
        _WORKER_SYNTH_CACHE[key] = synth
    elif shared_corpus is not None:
        synth.attach_shared_corpus(shared_corpus)
    return synth


__all__ = [
    "PROMPT_CORPUS_CODING",
    "PromptCorpus",
    "CorpusContentSynthesizer",
]
