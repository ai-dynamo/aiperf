# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hash-id scope: per-trace vs cross-trace content identity in the weka trie path.

Weka traces declare ``hash_id_scope`` as either ``"local"`` or ``"global"``:

* ``"local"``: hash ids are a PER-TRACE namespace, so hash_id 0 in trace A and
  hash_id 0 in trace B are different logical blocks and must synthesize
  different bytes. Deciding content from the hash id alone would manufacture
  cross-trace KV-cache prefix sharing at the server that the recorded workload
  never had.
* ``"global"``: ONE hash namespace is shared across every trace in the corpus,
  so equal hash ids must synthesize byte-identical blocks -- and, because
  segment ids are content-addressed, identical pool segment ids -- reproducing
  the recorded cross-trace KV-cache sharing. Response/tail content stays
  trace-scoped in both modes: node ids recur across traces and are not part of
  the recorded hash namespace.

Any other scope value is rejected with :class:`WekaHashScopeError`.

These tests drive the REAL synthesizer callbacks (builtin tokenizer), not the
stub callbacks the topology tests inject, because the scoping lives in the
production callback layer.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.adapters.weka.trace import (
    WekaHashScopeError,
    from_weka_trace,
)

_SEED = 1234


def _trace_dict(trace_id: str, scope: str = "local") -> dict:
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": scope,
        "requests": [
            {
                "type": "n",
                "t": 0.0,
                "api_time": 1.0,
                "model": "m",
                "hash_ids": [0, 1],
                "in": 128,
                "out": 8,
            },
            {
                "type": "n",
                "t": 5.0,
                "api_time": 1.0,
                "model": "m",
                "hash_ids": [0, 1, 2],
                "in": 192,
                "out": 8,
            },
        ],
    }


def _parse(
    tmp_path: Path, trace_id: str, scope: str = "local"
) -> tuple[list[str], list[str], str]:
    """Parse one trace; return (r_0 prompt path, r_0 prompt contents, r_0 response content)."""
    p = tmp_path / f"{trace_id}-{scope}.json"
    p.write_bytes(orjson.dumps(_trace_dict(trace_id, scope)))
    parsed = from_weka_trace(
        str(p), content_root_seed=_SEED, content_tokenizer="builtin"
    )
    pool = parsed.segment_pool
    assert pool is not None
    node = parsed.graph.nodes[f"{trace_id}:0"]
    path = node.metadata["trie"]["prompt_segment_ids"]
    contents = [m["content"] for m in pool.materialize(path)]
    response = next(
        s
        for s in pool.by_id.values()
        if s.role == "assistant" and s.parent_id == path[-1]
    )
    return path, contents, response.content


def test_local_scope_traces_synthesize_distinct_content(tmp_path: Path) -> None:
    """Same local hash ids in two different traces => different bytes + segment ids."""
    path_a, contents_a, response_a = _parse(tmp_path, "trace-A")
    path_b, contents_b, response_b = _parse(tmp_path, "trace-B")

    assert contents_a != contents_b, (
        "hash_id_scope='local' requires per-trace content identity; identical "
        "bytes across traces manufacture cross-trace KV-cache sharing"
    )
    assert path_a != path_b, (
        "content-addressed segment ids must not collide across local-scope traces"
    )
    assert response_a != response_b, (
        "response segments seeded only by node_id collide across traces"
    )


def test_global_scope_traces_synthesize_identical_content(tmp_path: Path) -> None:
    """Same global hash ids in two different traces => identical bytes + segment ids.

    This is the cross-trace KV-cache sharing contract: under
    ``hash_id_scope='global'`` equal hash ids are the SAME logical block, so
    two traces sharing a hash prefix must materialize byte-identical prompt
    content -- and, segment ids being content-addressed, the SAME pool entries.
    Responses stay trace-scoped: they are per-request output, not part of the
    recorded hash namespace.
    """
    path_a, contents_a, response_a = _parse(tmp_path, "trace-A", scope="global")
    path_b, contents_b, response_b = _parse(tmp_path, "trace-B", scope="global")

    assert contents_a == contents_b, (
        "hash_id_scope='global' requires one cross-trace hash namespace; "
        "differing bytes for equal hash ids drop the recorded sharing"
    )
    assert path_a == path_b, (
        "content-addressed segment ids must dedup across global-scope traces"
    )
    assert response_a != response_b, (
        "response tails must stay trace-scoped under global scope; node ids "
        "recur in every trace and are not part of the hash namespace"
    )


def test_global_scope_differs_from_local_scope_bytes(tmp_path: Path) -> None:
    """Scope selects the decode namespace: same trace, different scope => different bytes."""
    _path_l, contents_local, _resp_l = _parse(tmp_path, "trace-A", scope="local")
    _path_g, contents_global, _resp_g = _parse(tmp_path, "trace-A", scope="global")

    assert contents_local != contents_global, (
        "global scope must decode from the bare hash-id namespace, not the "
        "trace-scoped one"
    )


@pytest.mark.parametrize("scope", ["local", "global"])
def test_same_trace_reparse_is_byte_identical(tmp_path: Path, scope: str) -> None:
    """Scoped synthesis stays deterministic: same trace => same bytes."""
    path_1, contents_1, response_1 = _parse(tmp_path, "trace-A", scope=scope)
    path_2, contents_2, response_2 = _parse(tmp_path, "trace-A", scope=scope)

    assert path_1 == path_2
    assert contents_1 == contents_2
    assert response_1 == response_2


@pytest.mark.parametrize("scope", ["local", "global"])
def test_token_counts_unaffected_by_hash_scope(tmp_path: Path, scope: str) -> None:
    """Scoping changes bytes, never geometry: covered-count ISL holds per trace."""
    from aiperf.dataset.graph.adapters.shared.content import (
        get_or_build_synthesizer,
    )

    synth = get_or_build_synthesizer("builtin", prompt_corpus="coding", root_seed=_SEED)
    encode = synth._pg.tokenizer.encode
    for trace_id in ("trace-A", "trace-B"):
        _path, contents, _resp = _parse(tmp_path, trace_id, scope=scope)
        assert sum(len(encode(c)) for c in contents) == 128


def test_unrecognized_hash_scope_rejected(tmp_path: Path) -> None:
    """A scope outside {'local', 'global'} raises WekaHashScopeError with the cause."""
    p = tmp_path / "bad-scope.json"
    p.write_bytes(orjson.dumps(_trace_dict("trace-A", scope="cluster")))

    with pytest.raises(WekaHashScopeError, match="hash_id_scope='cluster'"):
        from_weka_trace(str(p), content_root_seed=_SEED, content_tokenizer="builtin")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
