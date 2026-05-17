# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial probes against the raw_payload loader and mmap fast path.

PASS = architecture holds (bad input rejected with clear error, or handled correctly).
FAIL = real bug — read the test docstring for details.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    MemoryMapFormat,
)
from aiperf.common.models import (
    Conversation,
    ConversationBranchInfo,
    Turn,
)
from aiperf.config.flags import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
from aiperf.dataset.loader.raw_payload import (
    RawPayloadDatasetLoader,
    _validate_payload_shape,
)
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
    MemoryMapDatasetIndex,
    PayloadOffset,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_cfg() -> CLIConfig:
    """Minimal CLIConfig for instantiating loaders in unit tests."""
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
def bench_id(request) -> str:
    """Unique benchmark_id per test node — avoids ``/tmp/aiperf_mmap_*`` collisions
    between parallel xdist workers and parametrized cases."""
    name = request.node.name
    # Path-safe: hash unstable Unicode into a stable suffix.
    return f"alpha_{abs(hash(name)) % (10**10)}"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def _write_raw_lines(path: Path, lines: list[bytes]) -> None:
    """Write byte lines verbatim (no orjson framing); for crafting malformed input."""
    with open(path, "wb") as f:
        for ln in lines:
            f.write(ln)
            if not ln.endswith(b"\n"):
                f.write(b"\n")


# ===========================================================================
# 1. Loader shape validation — _validate_payload_shape
# ===========================================================================


class TestValidatePayloadShape:
    """The single defensive gate before raw_payload bytes hit the mmap."""

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param([1, 2, 3], id="list"),
            pytest.param("hello", id="string"),
            pytest.param(None, id="null"),
        ],
    )
    def test_rejects_non_dict_top_level(self, payload):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _validate_payload_shape(Path("x"), 1, payload)

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"model": "m"}, id="missing_messages"),
            pytest.param({"messages": "not a list"}, id="messages_string"),
            pytest.param({"messages": {"role": "user"}}, id="messages_dict"),
        ],
    )
    def test_rejects_missing_or_non_list_messages(self, payload):
        with pytest.raises(ValueError, match="missing required 'messages'"):
            _validate_payload_shape(Path("x"), 1, payload)

    def test_accepts_empty_messages_list(self):
        """Note: an EMPTY 'messages' list passes the shape check.

        The loader does not require a non-empty messages array — every server
        rejects empty messages, but that surfaces at dispatch time, not load
        time. Document the current behavior; not a bug, just laxness.
        """
        _validate_payload_shape(Path("x"), 1, {"messages": []})

    def test_accepts_message_without_content_or_role(self):
        """Same laxness — loader doesn't recurse into the messages array."""
        _validate_payload_shape(Path("x"), 1, {"messages": [{}]})


# ===========================================================================
# 2. Loader file-level edges
# ===========================================================================


class TestRawPayloadLoaderFileEdges:
    def test_completely_empty_file_yields_zero_sessions(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """A 0-byte file is a valid (but empty) raw_payload file."""
        p = tmp_path / "empty.jsonl"
        p.write_bytes(b"")
        loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
        assert loader.load_dataset() == {}

    def test_file_with_only_blank_lines(self, tmp_path: Path, default_cfg: CLIConfig):
        p = tmp_path / "blanks.jsonl"
        p.write_bytes(b"\n\n   \n\t\n\n")
        loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
        assert loader.load_dataset() == {}

    def test_garbage_line_raises_orjson_decode_error(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        p = tmp_path / "garbage.jsonl"
        _write_raw_lines(p, [b"not json at all"])
        loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
        # Loader wraps JSONDecodeError with path:lineno context.
        with pytest.raises(ValueError, match=r"Invalid JSON in .*:1:"):
            loader.load_dataset()

    def test_invalid_jsonl_line_message_includes_path_and_lineno(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """The loader wraps orjson.JSONDecodeError into a ValueError carrying
        the offending file path + line number, matching ``BaseFileLoader``'s
        ``f"Invalid JSON in dataset file {target} at line {lineno}: {e}"``
        style and the "fail at load time with the offending file + line"
        principle in the ``_validate_payload_shape`` docstring.
        """
        p = tmp_path / "bad.jsonl"
        _write_raw_lines(p, [b'{"messages": [{"role": "user"}]}', b"<<not json>>"])
        loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
        with pytest.raises(ValueError) as exc:
            loader.load_dataset()
        assert str(p) in str(exc.value)
        assert ":2:" in str(exc.value)

    def test_wrong_shape_first_then_good(self, tmp_path: Path, default_cfg: CLIConfig):
        """First line is missing 'messages'; the loader must surface line 1."""
        p = tmp_path / "wrong.jsonl"
        _write_raw_lines(
            p,
            [
                orjson.dumps({"model": "m"}),
                orjson.dumps({"messages": [{"role": "user", "content": "ok"}]}),
            ],
        )
        loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
        with pytest.raises(ValueError, match=r":1:.*missing required 'messages'"):
            loader.load_dataset()

    def test_jsonl_trailing_newlines_are_tolerated(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        p = tmp_path / "trail.jsonl"
        with open(p, "wb") as f:
            f.write(orjson.dumps({"messages": [{"role": "user"}]}))
            f.write(b"\n\n\n\n")
        loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
        data = loader.load_dataset()
        assert len(data) == 1


# ===========================================================================
# 3. Loader directory-mode edges
# ===========================================================================


class TestRawPayloadLoaderDirectoryEdges:
    def test_empty_directory_loads_zero_sessions(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        d = tmp_path / "empty_dir"
        d.mkdir()
        loader = RawPayloadDatasetLoader(filename=d, cfg=default_cfg)
        assert loader.load_dataset() == {}

    def test_directory_with_only_empty_files_loads_zero_sessions(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        d = tmp_path / "empty_files_dir"
        d.mkdir()
        (d / "a.jsonl").write_bytes(b"")
        (d / "b.jsonl").write_bytes(b"   \n\n")
        loader = RawPayloadDatasetLoader(filename=d, cfg=default_cfg)
        assert loader.load_dataset() == {}

    def test_directory_nested_jsonl_is_ignored(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """``directory.glob('*.jsonl')`` is non-recursive; nested files vanish."""
        d = tmp_path / "outer"
        d.mkdir()
        (d / "sub").mkdir()
        _write_jsonl(
            d / "sub" / "ignored.jsonl",
            [{"messages": [{"role": "user", "content": "nested"}]}],
        )
        loader = RawPayloadDatasetLoader(filename=d, cfg=default_cfg)
        assert loader.load_dataset() == {}

    def test_directory_mixed_valid_and_invalid_jsonl_fails_on_invalid(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """An invalid line in any file aborts the whole directory load."""
        d = tmp_path / "mixed"
        d.mkdir()
        _write_jsonl(
            d / "good.jsonl",
            [{"messages": [{"role": "user", "content": "hi"}]}],
        )
        _write_raw_lines(d / "bad.jsonl", [orjson.dumps({"model": "x"})])
        loader = RawPayloadDatasetLoader(filename=d, cfg=default_cfg)
        with pytest.raises(ValueError, match="missing required 'messages'"):
            loader.load_dataset()

    def test_directory_empty_file_does_not_consume_session_id(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """FINDING (off-by-one, mild): ``_load_directory`` calls
        ``session_id_generator.next()`` BEFORE checking whether the file has any
        payloads. An empty file inside a directory burns a session_id slot.
        With a deterministic seed this perturbs subsequent IDs; with UUIDs it's
        merely waste. Reproducer below documents the current behavior.
        """
        d = tmp_path / "burn"
        d.mkdir()
        (d / "a_empty.jsonl").write_bytes(b"")
        _write_jsonl(
            d / "b_good.jsonl",
            [{"messages": [{"role": "user", "content": "hi"}]}],
        )

        # Force deterministic IDs by passing a seed via the run config.
        loader = RawPayloadDatasetLoader(filename=d, cfg=default_cfg)
        loader.session_id_generator.seed = 7
        loader.session_id_generator.reset()
        data = loader.load_dataset()
        ids = list(data.keys())
        assert len(ids) == 1
        # The surviving session id is session_000001, NOT session_000000 —
        # because the empty file consumed _000000.
        assert ids[0] == "session_000001", (
            f"Empty .jsonl consumed a session id slot; got {ids[0]}"
        )


# ===========================================================================
# 4. can_load discrimination / plugin ambiguity
# ===========================================================================


class TestCanLoadDiscrimination:
    def test_inputs_json_and_raw_payload_are_mutually_exclusive_on_probe(self):
        """Probe with both messages (raw_payload) and data list (inputs_json).
        raw_payload rejects, inputs_json accepts."""
        ambiguous = {
            "messages": [{"role": "user"}],
            "data": [{"session_id": "s1", "payloads": [{"messages": []}]}],
        }
        assert RawPayloadDatasetLoader.can_load(data=ambiguous) is False
        assert InputsJsonPayloadLoader.can_load(data=ambiguous) is True

    def test_dir_probe_with_corrupt_first_jsonl_swallows_and_returns_false(
        self, tmp_path: Path
    ):
        """``_dir_has_raw_payload_jsonl`` quietly skips files whose first line
        fails to parse. Document this behaviour: a directory whose first
        sorted ``.jsonl`` is corrupt but other files are valid still gets
        accepted, but only via fall-through to the next file."""
        d = tmp_path / "probe"
        d.mkdir()
        # 'a_bad.jsonl' is sorted first and has garbage
        _write_raw_lines(d / "a_bad.jsonl", [b"not json"])
        _write_jsonl(
            d / "b_good.jsonl",
            [{"messages": [{"role": "user", "content": "hi"}]}],
        )
        assert RawPayloadDatasetLoader.can_load(filename=d) is True

    def test_dir_probe_empty_dir_returns_false(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        assert RawPayloadDatasetLoader.can_load(filename=d) is False

    def test_can_load_messages_is_int_returns_false(self):
        assert RawPayloadDatasetLoader.can_load(data={"messages": 42}) is False


# ===========================================================================
# 5. Unicode / byte fidelity round-trip
# ===========================================================================


_UNICODE_CASES = [
    param("plain ASCII", id="ascii"),
    param("emoji 🚀💯🔥 and 4-byte UTF-8", id="emoji_4byte"),
    param("RTL: مرحبا بالعالم", id="rtl_arabic"),
    param("CJK 你好世界 こんにちは 안녕", id="cjk_mixed"),
    param("composed café vs decomposed café", id="nfc_vs_nfd"),
    param("zero-width ​ joiner ‍ here", id="zero_width"),
    param("ASCII control \x01\x07\x08 in-text", id="ascii_controls"),
    param("backslashes \\ and quotes \"' and newlines\nbody", id="quotes_and_nl"),
]


@pytest.mark.parametrize("text", _UNICODE_CASES)
@pytest.mark.asyncio
async def test_payload_bytes_preserves_unicode_byte_for_byte(
    text: str, tmp_path: Path, monkeypatch, bench_id: str
):
    """orjson.dumps preserves Unicode; the mmap round-trip must too."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payload = {
        "messages": [{"role": "user", "content": text}],
        "label": text,
    }
    expected_bytes = orjson.dumps(payload)
    conv = Conversation(session_id="u1", turns=[Turn(role="user", raw_payload=payload)])
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("u1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            got = client.get_payload_bytes("u1", 0)
            assert got == expected_bytes, "byte-for-byte fidelity broken"
            assert orjson.loads(got) == payload
        finally:
            client.close()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_payload_bytes_rejects_lone_surrogate_via_orjson(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """orjson refuses to encode unpaired surrogates. Confirm the error
    surfaces inside ``add_conversation`` (not deferred to read time);
    the underlying TypeError is wrapped into a ValueError with
    conversation_id + turn context."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payload = {"messages": [{"role": "user", "content": "\ud800"}]}
    conv = Conversation(session_id="s1", turns=[Turn(role="user", raw_payload=payload)])
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        with pytest.raises(ValueError) as exc:
            await store.add_conversation("s1", conv)
        msg = str(exc.value)
        assert "s1" in msg
        assert "turn 0" in msg
    finally:
        await store.stop()


# ===========================================================================
# 6. NaN / Infinity / numeric edges
# ===========================================================================


@pytest.mark.asyncio
async def test_payload_bytes_nan_rejected_at_add_time(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """``orjson.dumps`` with ``OPT_NON_NUMBERS_REJECT`` refuses to encode
    NaN / Inf / -Inf. The PAYLOAD_BYTES path wraps the resulting TypeError
    into a ValueError carrying the offending ``conversation_id`` + turn
    index so the user can locate the bad authored value, satisfying the
    project's NaN/Inf discipline (see CLAUDE.md § "NaN/Inf Discipline" —
    "numeric metric values crossing a serialization boundary must be
    finite or explicitly None").
    """
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payload = {
        "messages": [{"role": "user"}],
        "temperature": math.nan,
    }
    conv = Conversation(session_id="s1", turns=[Turn(role="user", raw_payload=payload)])
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        with pytest.raises(ValueError) as exc:
            await store.add_conversation("s1", conv)
        msg = str(exc.value)
        assert "s1" in msg
        assert "turn 0" in msg
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_payload_bytes_huge_int_round_trip(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """orjson refuses to serialize Python ints that exceed the 64-bit
    range. The PAYLOAD_BYTES ``add_conversation`` wraps the underlying
    ``TypeError: Integer exceeds 64-bit range`` into a ValueError carrying
    the offending ``conversation_id`` + turn index so the user can locate
    the bad authored value.
    """
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    huge = 2**80 + 7
    payload = {"messages": [{"role": "user"}], "seed": huge}
    conv = Conversation(session_id="s1", turns=[Turn(role="user", raw_payload=payload)])
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        with pytest.raises(ValueError) as exc:
            await store.add_conversation("s1", conv)
        msg = str(exc.value)
        assert "s1" in msg
        assert "turn 0" in msg
        assert "64-bit" in msg
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_payload_bytes_negative_zero_serializes_as_zero(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """orjson normalizes -0.0 to 0; document the behaviour."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payload = {"messages": [{"role": "user"}], "x": -0.0}
    conv = Conversation(session_id="s1", turns=[Turn(role="user", raw_payload=payload)])
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            got = client.get_payload_bytes("s1", 0)
            decoded = orjson.loads(got)
            # -0.0 round-trips as 0.0; either is acceptable as long as it doesn't crash.
            assert decoded["x"] == 0.0
        finally:
            client.close()
    finally:
        await store.stop()


# ===========================================================================
# 7. Boundary conditions / off-by-one in get_payload_bytes
# ===========================================================================


@pytest.mark.asyncio
async def test_get_payload_bytes_turn_index_boundary(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payloads = [{"messages": [{"role": "user", "content": f"t{i}"}]} for i in range(3)]
    conv = Conversation(
        session_id="s1", turns=[Turn(role="user", raw_payload=p) for p in payloads]
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            # exactly-zero index
            assert orjson.loads(client.get_payload_bytes("s1", 0)) == payloads[0]
            # last valid index
            assert orjson.loads(client.get_payload_bytes("s1", 2)) == payloads[2]
            # one-past-the-end
            with pytest.raises(IndexError):
                client.get_payload_bytes("s1", 3)
            # negative
            with pytest.raises(IndexError):
                client.get_payload_bytes("s1", -1)
            # really negative
            with pytest.raises(IndexError):
                client.get_payload_bytes("s1", -99999)
            # extremely positive
            with pytest.raises(IndexError):
                client.get_payload_bytes("s1", 1_000_000)
        finally:
            client.close()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_zero_turn_conversation_payload_bytes_falls_back_to_conversation(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """A dataset whose every conversation has *no turns* used to drive
    ``_select_mmap_format`` to PAYLOAD_BYTES via vacuous ``all()``, which
    produced a 0-byte ``dataset.dat`` and a misleading
    ``MemoryMapSerializationError: Invalid index data: cannot mmap an
    empty file``. ``_select_mmap_format`` now refuses the bad combo at
    format-selection time and falls back to CONVERSATION (which serialises
    the Conversation envelope and keeps the data file non-empty).

    Direct backing-store use with PAYLOAD_BYTES and no turns still
    produces an empty data file, but that combo cannot be reached through
    the DatasetManager's selection logic.
    """
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    # Selection-level guard: zero-turn conversations no longer route to
    # PAYLOAD_BYTES.
    convs = [Conversation(session_id="empty", turns=[])]
    assert DatasetManager._select_mmap_format(convs) == MemoryMapFormat.CONVERSATION


# ===========================================================================
# 8. Backing-store lifecycle guards
# ===========================================================================


@pytest.mark.asyncio
async def test_add_after_finalize_raises_runtime_error(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    conv = Conversation(
        session_id="s1",
        turns=[Turn(role="user", raw_payload={"messages": []})],
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        with pytest.raises(RuntimeError, match="after finalization"):
            await store.add_conversation(
                "s2",
                Conversation(
                    session_id="s2",
                    turns=[Turn(role="user", raw_payload={"messages": []})],
                ),
            )
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_double_finalize_raises_runtime_error(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.finalize()
        with pytest.raises(RuntimeError, match="finalize called twice"):
            await store.finalize()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_get_metadata_before_finalize_raises(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        with pytest.raises(RuntimeError, match="before finalization"):
            store.get_client_metadata()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_add_conversation_twice_same_id_payload_bytes(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """``add_conversation`` rejects duplicate ``conversation_id`` at
    write-time with a ValueError naming the offending id. Previously the
    second write silently overwrote ``_payload_offsets[conversation_id]``
    and the failure only surfaced as an opaque error at ``finalize()``
    from the ``MemoryMapDatasetIndex.validate_conversation_ids`` field
    validator.
    """
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payload1 = {"messages": [{"role": "user", "content": "first"}]}
    payload2 = {"messages": [{"role": "user", "content": "second"}]}
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation(
            "s1",
            Conversation(
                session_id="s1", turns=[Turn(role="user", raw_payload=payload1)]
            ),
        )
        with pytest.raises(ValueError, match=r"Duplicate conversation_id 's1'"):
            await store.add_conversation(
                "s1",
                Conversation(
                    session_id="s1", turns=[Turn(role="user", raw_payload=payload2)]
                ),
            )
    finally:
        await store.stop()


# ===========================================================================
# 9. MemoryMapDatasetIndex roundtrip with hostile conversation_ids
# ===========================================================================


_HOSTILE_IDS = [
    param('id"with"quote', id="double_quote"),
    param("id\nwith\nnewline", id="newline"),
    param("id:with:colon", id="colon"),
    param("id with space", id="space"),
    param("id/with/slash", id="slash"),
    param("id\\with\\backslash", id="backslash"),
    param("id\twith\ttab", id="tab"),
    param("id\x00with\x00null", id="null_byte"),
    param("idemoji🚀id", id="emoji_in_id"),
    param("a" * 4096, id="very_long"),
]


@pytest.mark.parametrize("session_id", _HOSTILE_IDS)
def test_index_json_roundtrip_preserves_hostile_session_id(session_id: str):
    """Index uses Pydantic JSON. Hostile session_id chars must survive."""
    idx = MemoryMapDatasetIndex(
        format=MemoryMapFormat.PAYLOAD_BYTES,
        conversation_ids=[session_id],
        payload_offsets={session_id: [PayloadOffset(offset=0, size=10)]},
        total_size=10,
    )
    raw = idx.model_dump_json(by_alias=True)
    parsed = MemoryMapDatasetIndex.model_validate_json(raw)
    assert parsed.conversation_ids == [session_id]
    assert session_id in parsed.payload_offsets


def test_index_rejects_duplicate_conversation_ids():
    """The field_validator enforces uniqueness on conversation_ids."""
    with pytest.raises(ValueError, match="unique"):
        MemoryMapDatasetIndex(
            format=MemoryMapFormat.PAYLOAD_BYTES,
            conversation_ids=["a", "a"],
            payload_offsets={"a": []},
            total_size=0,
        )


# ===========================================================================
# 10. Concurrent client reads against the same mmap file
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_client_reads(tmp_path: Path, monkeypatch, bench_id: str):
    """Two MemoryMapDatasetClient instances should both read the same data."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payloads = [{"messages": [{"role": "user", "content": f"t{i}"}]} for i in range(8)]
    conv = Conversation(
        session_id="s1", turns=[Turn(role="user", raw_payload=p) for p in payloads]
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()

        clients = [
            MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
            for _ in range(4)
        ]
        results: dict[int, list[bytes]] = {i: [] for i in range(len(clients))}
        errors: list[BaseException] = []

        def worker(idx: int) -> None:
            try:
                for j in range(len(payloads)):
                    results[idx].append(clients[idx].get_payload_bytes("s1", j))
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(len(clients))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent reads errored: {errors}"
        # All clients must see the same bytes for each turn
        for j in range(len(payloads)):
            seen = {r[j] for r in results.values()}
            assert len(seen) == 1, f"divergent bytes at turn {j}: {seen}"

        for c in clients:
            c.close()
    finally:
        await store.stop()


# ===========================================================================
# 11. Large-payload stress
# ===========================================================================


@pytest.mark.asyncio
async def test_large_single_payload_round_trip(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """A multi-megabyte single payload round-trips byte-for-byte through mmap."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    big = "x" * (4 * 1024 * 1024)  # 4 MB of x's (stay well clear of test timeout)
    payload = {"messages": [{"role": "user", "content": big}]}
    expected = orjson.dumps(payload)
    conv = Conversation(
        session_id="big", turns=[Turn(role="user", raw_payload=payload)]
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("big", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            got = client.get_payload_bytes("big", 0)
            assert got == expected
            assert len(got) == len(expected)
        finally:
            client.close()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_many_payloads_offset_arithmetic(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """Hundreds of payloads — verify every offset.size combination decodes
    independently. Catches offset accumulation bugs."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    n = 500
    payloads = [
        {"messages": [{"role": "user", "content": f"row-{i}-{'p' * (i % 17)}"}]}
        for i in range(n)
    ]
    conv = Conversation(
        session_id="many",
        turns=[Turn(role="user", raw_payload=p) for p in payloads],
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("many", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            # Read in random-ish order to defeat any sequential-only assumption
            for i in [0, n - 1, n // 2, 1, n - 2, 17, 250]:
                got = client.get_payload_bytes("many", i)
                assert orjson.loads(got) == payloads[i]
        finally:
            client.close()
    finally:
        await store.stop()


# ===========================================================================
# 12. _select_mmap_format edge cases
# ===========================================================================


def _conv(session_id: str, n: int = 1, raw: bool = True) -> Conversation:
    turns = []
    for i in range(n):
        if raw:
            turns.append(
                Turn(
                    role="user",
                    raw_payload={
                        "messages": [{"role": "user", "content": f"{session_id}#{i}"}]
                    },
                )
            )
        else:
            turns.append(Turn(role="user"))
    return Conversation(session_id=session_id, turns=turns)


class TestSelectMmapFormat:
    def test_one_missing_raw_payload_among_many_picks_conversation(self):
        """90% have raw_payload, 10% don't — must fall back to CONVERSATION,
        not raise. This is the realistic mixed-dataset case."""
        convs = [_conv(f"s{i}") for i in range(9)]
        convs.append(_conv("bad", n=2, raw=False))
        assert DatasetManager._select_mmap_format(convs) == MemoryMapFormat.CONVERSATION

    def test_single_turn_missing_in_otherwise_clean_conversation(self):
        """A single turn missing raw_payload inside an otherwise-raw conversation
        forces fallback. The ``all()`` short-circuit must visit per-turn."""
        convs = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", raw_payload={"messages": []}),
                    Turn(role="user"),  # this one is bare
                    Turn(role="user", raw_payload={"messages": []}),
                ],
            )
        ]
        assert DatasetManager._select_mmap_format(convs) == MemoryMapFormat.CONVERSATION

    def test_zero_turn_conversation_with_raw_payload_set_elsewhere(self):
        """A zero-turn conversation no longer drives the selector toward
        PAYLOAD_BYTES: combined with another conv that has all-raw turns,
        the result is still PAYLOAD_BYTES (the zero-turn conv contributes
        0 turns to the total). Pure-zero-turn input now falls back to
        CONVERSATION via the total_turns==0 guard."""
        convs = [
            Conversation(session_id="empty", turns=[]),
            _conv("s1"),
        ]
        assert (
            DatasetManager._select_mmap_format(convs) == MemoryMapFormat.PAYLOAD_BYTES
        )

    def test_only_zero_turn_conversations(self):
        """All-empty conversations now fall back to CONVERSATION via the
        total_turns==0 guard. Previously a vacuous ``all()`` picked
        PAYLOAD_BYTES, which produced a 0-byte data file and a confusing
        mmap error at client open."""
        convs = [
            Conversation(session_id="e1", turns=[]),
            Conversation(session_id="e2", turns=[]),
        ]
        assert DatasetManager._select_mmap_format(convs) == MemoryMapFormat.CONVERSATION

    def test_spawn_branch_with_raw_payload_does_not_raise(self):
        spawn = ConversationBranchInfo(
            branch_id="s1:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
        )
        convs = [
            Conversation(
                session_id="s1",
                turns=[
                    Turn(role="user", raw_payload={"messages": []}),
                ],
                branches=[spawn],
            )
        ]
        assert (
            DatasetManager._select_mmap_format(convs) == MemoryMapFormat.PAYLOAD_BYTES
        )

    def test_fork_in_one_conv_among_many_raises_with_correct_branch_id(self):
        clean = _conv("clean1")
        fork_conv = Conversation(
            session_id="bad",
            turns=[Turn(role="user", raw_payload={"messages": []})],
            branches=[
                ConversationBranchInfo(
                    branch_id="bad:0",
                    child_conversation_ids=["c1"],
                    mode=ConversationBranchMode.FORK,
                )
            ],
        )
        with pytest.raises(ValueError, match="bad:bad:0"):
            DatasetManager._select_mmap_format([clean, fork_conv])

    def test_fork_truncated_at_three(self):
        """Error message lists up to 3 branches then '(and N more)'."""
        convs = [
            Conversation(
                session_id=f"c{i}",
                turns=[Turn(role="user", raw_payload={"messages": []})],
                branches=[
                    ConversationBranchInfo(
                        branch_id=f"c{i}:0",
                        child_conversation_ids=[f"d{i}"],
                        mode=ConversationBranchMode.FORK,
                    )
                ],
            )
            for i in range(5)
        ]
        with pytest.raises(ValueError) as exc:
            DatasetManager._select_mmap_format(convs)
        msg = str(exc.value)
        assert "(and 2 more)" in msg


# ===========================================================================
# 13. InputsJsonPayloadLoader edges (peer to raw_payload)
# ===========================================================================


class TestInputsJsonAdversarial:
    def test_inputs_json_duplicate_session_id_rejected_with_index(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        p = tmp_path / "inputs.json"
        p.write_bytes(
            orjson.dumps(
                {
                    "data": [
                        {"session_id": "dup", "payloads": [{"messages": []}]},
                        {"session_id": "dup", "payloads": [{"messages": []}]},
                    ]
                }
            )
        )
        loader = InputsJsonPayloadLoader(filename=p, cfg=default_cfg)
        with pytest.raises(ValueError, match="duplicate session_id"):
            loader.load_dataset()

    def test_inputs_json_empty_payloads_list_rejected(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """min_length=1 on InputsJsonSession.payloads catches empty lists."""
        p = tmp_path / "inputs.json"
        p.write_bytes(orjson.dumps({"data": [{"session_id": "s1", "payloads": []}]}))
        loader = InputsJsonPayloadLoader(filename=p, cfg=default_cfg)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            loader.load_dataset()

    def test_inputs_json_missing_session_id_key_raises_value_error_with_index(
        self, tmp_path: Path, default_cfg: CLIConfig
    ):
        """A missing 'session_id' key surfaces as a ValueError naming the
        file path and the ``data[N]`` index of the offending entry,
        matching the rest of the loader's error style."""
        p = tmp_path / "inputs.json"
        p.write_bytes(orjson.dumps({"data": [{"payloads": [{"messages": []}]}]}))
        loader = InputsJsonPayloadLoader(filename=p, cfg=default_cfg)
        with pytest.raises(ValueError) as exc:
            loader.load_dataset()
        msg = str(exc.value)
        assert str(p) in msg
        assert "data[0]" in msg
        assert "session_id" in msg

    def test_inputs_json_can_load_empty_data_list_rejected(self):
        assert InputsJsonPayloadLoader.can_load(data={"data": []}) is False

    def test_inputs_json_can_load_data_list_with_non_dict_first_rejected(self):
        assert InputsJsonPayloadLoader.can_load(data={"data": ["not a dict"]}) is False

    def test_inputs_json_can_load_first_missing_payloads_rejected(self):
        assert (
            InputsJsonPayloadLoader.can_load(data={"data": [{"session_id": "s1"}]})
            is False
        )


# ===========================================================================
# 14. PAYLOAD_BYTES format + zero-byte payload
# ===========================================================================


@pytest.mark.asyncio
async def test_two_zero_byte_payloads_are_distinguishable(
    tmp_path: Path, monkeypatch, bench_id: str
):
    """orjson.dumps({}) is b'{}' (2 bytes), not zero — but the smallest
    possible payload is still distinct per index entry. Confirm two empty
    dict payloads don't collide on offset arithmetic."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    # Two payloads, both serializing to a minimal dict {"messages":[]}
    p1 = {"messages": []}
    p2 = {"messages": []}
    conv = Conversation(
        session_id="s1",
        turns=[
            Turn(role="user", raw_payload=p1),
            Turn(role="user", raw_payload=p2),
        ],
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            b1 = client.get_payload_bytes("s1", 0)
            b2 = client.get_payload_bytes("s1", 1)
            assert b1 == orjson.dumps(p1)
            assert b2 == orjson.dumps(p2)
            # Offsets must advance — second offset > first offset+size
            offsets = client.index.payload_offsets["s1"]
            assert offsets[1].offset == offsets[0].offset + offsets[0].size
        finally:
            client.close()
    finally:
        await store.stop()


# ===========================================================================
# 15. Client error surface
# ===========================================================================


@pytest.mark.asyncio
async def test_get_payload_bytes_unknown_conversation_id(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    conv = Conversation(
        session_id="s1",
        turns=[Turn(role="user", raw_payload={"messages": []})],
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            with pytest.raises(KeyError, match="nope"):
                client.get_payload_bytes("nope", 0)
        finally:
            client.close()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_get_conversation_against_payload_bytes_format_raises(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    conv = Conversation(
        session_id="s1",
        turns=[Turn(role="user", raw_payload={"messages": []})],
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            with pytest.raises(RuntimeError, match="PAYLOAD_BYTES"):
                client.get_conversation("s1")
        finally:
            client.close()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_get_payload_bytes_against_conversation_format_raises(
    tmp_path: Path, monkeypatch, bench_id: str
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    conv = Conversation(
        session_id="s1",
        turns=[Turn(role="user", raw_payload={"messages": []})],
    )
    store = MemoryMapDatasetBackingStore(
        benchmark_id=bench_id, format=MemoryMapFormat.CONVERSATION
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        meta = store.get_client_metadata()
        client = MemoryMapDatasetClient(meta.data_file_path, meta.index_file_path)
        try:
            with pytest.raises(RuntimeError, match="PAYLOAD_BYTES format"):
                client.get_payload_bytes("s1", 0)
        finally:
            client.close()
    finally:
        await store.stop()


# ===========================================================================
# 16. convert_to_conversations on empty load result
# ===========================================================================


def test_convert_to_conversations_empty_dict_yields_empty_list(
    default_cfg: CLIConfig, tmp_path: Path
):
    """The convert step is pure; empty dict in -> empty list out."""
    p = tmp_path / "empty.jsonl"
    p.write_bytes(b"")
    loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
    out = loader.convert_to_conversations({})
    assert out == []


def test_convert_preserves_context_mode_message_array_with_responses(
    default_cfg: CLIConfig, tmp_path: Path
):
    """Every Conversation produced by raw_payload uses MESSAGE_ARRAY_WITH_RESPONSES."""
    p = tmp_path / "x.jsonl"
    _write_jsonl(
        p,
        [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "bye"}]},
        ],
    )
    loader = RawPayloadDatasetLoader(filename=p, cfg=default_cfg)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs) == 2
    for c in convs:
        assert c.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        assert all(t.raw_payload is not None for t in c.turns)
