from __future__ import annotations

import array as _array
import contextlib
import mmap
from dataclasses import dataclass
from io import BufferedWriter
from pathlib import Path

import aiofiles
import orjson

from aiperf.common.constants import IS_MACOS, IS_WINDOWS
from aiperf.common.exceptions import (
    MemoryMapFileOperationError,
    MemoryMapSerializationError,
)

_UNIFIED_DIR_PREFIX = "aiperf_graph_segments_"
_CONTENT_BLOB = "content.blob"
_CONTENT_IDX = "content.idx"
_NODES_BLOB = "nodes.blob"
_NODES_IDX = "nodes.idx"

_HEAD = b'{"messages":['

_IDX_TYPECODE = "Q"  # 64-bit unsigned; offsets/sizes fit comfortably


def _unified_dir(base_path: Path, benchmark_id: str) -> Path:
    return Path(base_path) / f"{_UNIFIED_DIR_PREFIX}{benchmark_id}"


def _encode_inner_key(node_ordinal: int, phase_variant: str) -> str:
    """The store's (ordinal, variant) inner key, '<ordinal>:<variant>' --
    self-consistent between the writer and GraphSegmentUnifiedClient."""
    return f"{node_ordinal}:{phase_variant}"


@dataclass(slots=True)
class NodeEnvelope:
    """A single per-(trace, node) manifest envelope to be persisted."""

    node_ordinal: int
    """Zero-based ordinal of the node within its trace."""

    phase_variant: str
    """Phase variant this envelope belongs to (e.g. ``"profiling"``)."""

    envelope_bytes: bytes
    """Pre-serialized ``orjson.dumps`` envelope blob to store verbatim."""


@dataclass(slots=True, frozen=True)
class GraphStoreBuildStats:
    """One store-build memory snapshot, computed at :meth:`finalize` entry.

    A cheap (``O(traces) + O(1)``) derivation from the buffers the store already
    holds -- the measurement baseline that makes pool/envelope-size regressions
    visible in the build log instead of surfacing as mystery RSS.
    """

    segment_count: int
    """Number of unique interned content segments (``len(self._spans)``)."""

    content_bytes: int
    """Total bytes of the interned content blob.

    Post-spill this is the running ``_content_bytes_written`` counter (the
    encoded content is streamed straight to ``content.blob`` at ``put`` time and
    never accumulated in RAM), which equals the finalized ``content.blob`` size
    and the sum of every span's length."""

    node_manifest_count: int
    """Total per-node manifest envelopes indexed across all traces."""

    manifest_bytes: int
    """Total bytes appended to the manifest region (``len(self._nodes_buf)``)."""

    trace_count: int
    """Number of distinct traces carrying at least one manifest."""

    peak_rss_mib: float | None
    """Process peak resident set size in MiB at finalize entry, or ``None`` where
    unavailable (Windows). ``RUSAGE_SELF`` only -- excludes weka pool workers."""


def _peak_rss_mib() -> float | None:
    """Process peak resident set size in MiB, or ``None`` where unavailable.

    Windows has no ``resource`` module, so return ``None`` there (the import is
    lazy, after the early return, so this module still imports on Windows).
    ``ru_maxrss`` units are platform-specific: macOS (Darwin) reports BYTES,
    Linux reports KiB -- convert each to MiB with the right divisor.
    """
    if IS_WINDOWS:
        return None
    import resource

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if IS_MACOS:
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


class GraphSegmentUnifiedBackingStore:
    """Build-time writer for the unified store: an interned (A2) content pool
    (blob + packed span index; the hex->handle map lives only in this writer)
    AND a per-node manifest region (concatenated envelope blobs + a
    (trace, ordinal:variant) offset index).

    Content spills incrementally: :meth:`put_segment` streams each segment's
    wire blob straight to an open ``content.blob`` write handle and advances a
    running ``_content_bytes_written`` counter, so the encoded content is never
    accumulated in a RAM ``bytearray`` and :meth:`finalize` no longer
    materializes a ``bytes(self._content_buf)`` transient double. Only the
    ``_ids`` map (dedup + handle resolution) and the ``_spans`` list
    (``content.idx`` is written from it) stay resident; both are released at
    finalize END. The handle is plain buffered binary file I/O with NO
    event-loop coupling, so the store may be constructed on one thread and
    drained / finalized inside ``asyncio.run`` on a worker thread (see
    ``GraphStoreBuilder``).

    ``_nodes_buf`` is deliberately NOT spilled: the manifest region fills in the
    drain window (below the parse peak) and is small (~0.2-0.3 GB even at 1M
    nodes), so streaming it would add a second write handle for no meaningful
    peak-RAM win. It stays a RAM ``bytearray`` flushed once at finalize.

    A store that errors before finalize would otherwise leave a half-written
    ``content.blob`` on disk (the incremental spill writes as it goes);
    :meth:`abort` closes the handle and unlinks the store files so a later
    :class:`GraphSegmentUnifiedClient` open never trips on a partial blob."""

    def __init__(self, base_path: Path | str, benchmark_id: str) -> None:
        d = _unified_dir(Path(base_path), benchmark_id)
        d.mkdir(parents=True, exist_ok=True)
        self._dir = d
        self._content_blob_path = d / _CONTENT_BLOB
        self._content_idx_path = d / _CONTENT_IDX
        self._nodes_blob_path = d / _NODES_BLOB
        self._nodes_idx_path = d / _NODES_IDX
        # content region: put_segment streams each blob to _content_blob_file and
        # advances _content_bytes_written; the encoded content never sits in RAM.
        self._ids: dict[str, int] = {}
        self._spans: list[tuple[int, int]] = []
        self._content_blob_file: BufferedWriter | None = open(  # noqa: SIM115
            self._content_blob_path, "wb"
        )
        self._content_bytes_written = 0
        self._content_buf = bytearray()  # vestigial: never appended to post-spill
        # nodes region: trace_id -> inner_key -> (offset, size)
        self._nodes_buf = bytearray()
        self._node_offsets: dict[str, dict[str, list[int]]] = {}
        self._finalized = False
        self._build_stats: GraphStoreBuildStats | None = None

    @property
    def data_dir(self) -> Path:
        """The store directory (``aiperf_graph_segments_<benchmark_id>``)."""
        return self._dir

    def put_segment(
        self, segment_id: str, role: str, content: str, wire_json: str | None = None
    ) -> int:
        """Intern one segment's wire blob; return its dense insertion-index handle.

        When ``wire_json`` is provided (a raw-authored dag message) the persisted
        blob is ``wire_json.encode()`` verbatim -- key order and extra keys preserved
        byte-for-byte. Otherwise the blob is the derived ``{"role", "content"}`` dict
        (the existing normalized behavior for token/text segments).
        """
        if self._finalized:
            raise RuntimeError("Cannot put_segment after finalize")
        existing = self._ids.get(segment_id)
        if existing is not None:
            return existing
        b = (
            wire_json.encode()
            if wire_json is not None
            else orjson.dumps({"role": role, "content": content})
        )
        # Spill: write the blob straight to disk and advance the running byte
        # counter. off is the pre-write counter -- identical to the old
        # len(self._content_buf) -- so content.blob bytes and content.idx spans
        # are byte-identical to the accumulate-then-flush build.
        off = self._content_bytes_written
        assert self._content_blob_file is not None
        self._content_blob_file.write(b)
        self._content_bytes_written += len(b)
        handle = len(self._spans)
        self._ids[segment_id] = handle
        self._spans.append((off, len(b)))
        return handle

    def segment_handle(self, segment_id: str) -> int | None:
        if self._finalized:
            raise RuntimeError(
                "Cannot segment_handle after finalize: the writer's id map is "
                "released once finalize flushes the store; read handles via "
                "GraphSegmentUnifiedClient instead."
            )
        return self._ids.get(segment_id)

    def add_node_manifest(
        self,
        trace_id: str,
        node_ordinal: int,
        phase_variant: str,
        envelope_bytes: bytes,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add_node_manifest after finalize")
        off = len(self._nodes_buf)
        self._nodes_buf += envelope_bytes
        key = _encode_inner_key(node_ordinal, phase_variant)
        self._node_offsets.setdefault(trace_id, {})[key] = [off, len(envelope_bytes)]

    def add_node_manifest_interned(
        self,
        trace_id: str,
        node_ordinal: int,
        phase_variant: str,
        handles: list[int],
        dispatch_overrides: dict,
        stream: bool,
        *,
        items: list[dict] | None = None,
        capture: bool = False,
        extra_headers: dict[str, str] | None = None,
        endpoint_extra_applied: bool = False,
    ) -> None:
        """Write one node's manifest envelope.

        ``items``/``capture`` are the dynamic-content additions:
        ``items`` is the ordered assembly program for slot-carrying
        nodes (``{"h": handle}`` / ``{"s": {"src": ordinal}}`` /
        ``{"m": {"role", "parts"}}``), ``capture`` marks producer nodes whose
        responses the worker pools. ``extra_headers`` carries per-node HTTP
        headers (dynamo ``x-dynamo-*`` session identity) the worker attaches to
        the request HEADERS, never the body. ``endpoint_extra_applied`` marks a
        node whose adapter already folded the run's ``--extra-inputs`` into
        ``dispatch_overrides`` at parse, so the worker must NOT re-merge
        ``endpoint.extra`` (the adapter-owned values win). All four are OMITTED
        when unset, so envelopes for header-less / flag-less corpora (weka,
        static native, dynamo) stay byte-identical.
        """
        envelope: dict = {
            "handles": list(handles),
            "dispatch_overrides": dispatch_overrides,
            "stream": stream,
        }
        if items is not None:
            envelope["items"] = items
        if capture:
            envelope["capture"] = True
        if extra_headers:
            envelope["extra_headers"] = dict(extra_headers)
        if endpoint_extra_applied:
            envelope["endpoint_extra_applied"] = True
        self.add_node_manifest(
            trace_id, node_ordinal, phase_variant, orjson.dumps(envelope)
        )

    def _compute_build_stats(self) -> GraphStoreBuildStats:
        """Snapshot the store's build-memory footprint from its live buffers.

        ``O(traces) + O(1)``: two ``len()`` reads, one ``len()`` over the trace
        map, and one ``len()`` per trace's inner map. No pass over content and no
        per-trace maxima -- the buffers already hold the running totals.
        """
        return GraphStoreBuildStats(
            segment_count=len(self._spans),
            content_bytes=self._content_bytes_written,
            node_manifest_count=sum(len(v) for v in self._node_offsets.values()),
            manifest_bytes=len(self._nodes_buf),
            trace_count=len(self._node_offsets),
            peak_rss_mib=_peak_rss_mib(),
        )

    @property
    def build_stats(self) -> GraphStoreBuildStats | None:
        """The build-memory snapshot, or ``None`` until :meth:`finalize` runs.

        ``manifest_bytes`` and the counters reflect APPENDED (write-side) totals:
        a duplicate ``(trace, ordinal, variant)`` write orphans the earlier blob
        in ``_nodes_buf`` while ``node_manifest_count`` tracks live index entries,
        so a future count/bytes divergence reads as that duplicate-write bug.
        ``content_bytes`` now consumes the running ``_content_bytes_written``
        counter the incremental spill maintains (the seam this docstring used to
        name as future work). ``manifest_bytes`` stays a ``len(self._nodes_buf)``
        read because the manifest region is not spilled. The counter is
        monotonic, so the finalize-ENTRY snapshot still measures the full content
        footprint even though the blob was streamed to disk during the write.
        """
        return self._build_stats

    def abort(self) -> None:
        """Best-effort teardown for a store that errored before finalize.

        The incremental spill streams ``content.blob`` as it goes, so an aborted
        build would otherwise leave a partial blob for a later
        :class:`GraphSegmentUnifiedClient` open to trip on. Close the write
        handle and unlink the four store files. Idempotent and non-raising: safe
        to call twice, and safe after a SUCCESSFUL finalize (the ``_finalized``
        guard skips the unlink so a complete store is never deleted). The
        ``GraphStoreBuilder`` drain paths call this, then remove the store dir,
        before re-raising the drain error.
        """
        handle = self._content_blob_file
        self._content_blob_file = None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        if self._finalized:
            # A fully finalized store is complete on disk -- never unlink it.
            return
        for path in (
            self._content_blob_path,
            self._content_idx_path,
            self._nodes_blob_path,
            self._nodes_idx_path,
        ):
            with contextlib.suppress(OSError):
                path.unlink()

    async def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("finalize called twice")
        # Snapshot BEFORE any remaining write so a mid-write failure still leaves
        # the full-footprint measure here (the running counter is monotonic).
        self._build_stats = self._compute_build_stats()
        # content.blob was streamed to disk incrementally by put_segment; flush
        # and close the handle instead of writing bytes(self._content_buf) -- the
        # finalize transient double is gone.
        if self._content_blob_file is not None:
            self._content_blob_file.flush()
            self._content_blob_file.close()
            self._content_blob_file = None
        async with aiofiles.open(self._content_idx_path, "wb") as f:
            flat = _array.array(_IDX_TYPECODE)
            for off, size in self._spans:
                flat.append(off)
                flat.append(size)
            await f.write(flat.tobytes())
        async with aiofiles.open(self._nodes_blob_path, "wb") as f:
            await f.write(bytes(self._nodes_buf))
        async with aiofiles.open(self._nodes_idx_path, "wb") as f:
            await f.write(orjson.dumps(self._node_offsets))
        self._finalized = True
        # Release the write-side accumulation state now that every file is
        # flushed. The store object lives on through the structural merge /
        # sidecar / prefix-cache tail with zero readers of these buffers
        # (build_stats was snapshotted at entry above). Post-spill the
        # load-bearing clears are _ids (dedup map) and _spans (content.idx
        # source), which are what still pin RAM through the post-finalize tail;
        # _content_buf is already empty (content spilled to disk at put time) so
        # its clear is a formality. The put/add/handle methods guard on
        # _finalized, so nothing touches them post-release.
        self._ids = {}
        self._spans = []
        self._content_buf = bytearray()
        self._nodes_buf = bytearray()
        self._node_offsets = {}


class GraphSegmentUnifiedClient:
    """Worker-side reader for the unified store. Presents BOTH faces --
    per-node envelope addressing and interned content -- so worker_materialize
    is handed this ONE object for both. A2-strict: only the interned packed
    ``content.idx`` is accepted; a legacy JSON (A1) index fails loud."""

    def __init__(self, base_path: Path | str, benchmark_id: str) -> None:
        d = _unified_dir(Path(base_path), benchmark_id)
        self._data_dir = d
        self._content_blob_path = d / _CONTENT_BLOB
        self._content_idx_path = d / _CONTENT_IDX
        self._nodes_blob_path = d / _NODES_BLOB
        self._nodes_idx_path = d / _NODES_IDX
        self._content_mm: mmap.mmap | None = None
        self._content_mv: memoryview | None = None
        self._spans: list[list[int]] = []
        self._nodes_mm: mmap.mmap | None = None
        self._nodes_mv: memoryview | None = None
        self._node_offsets: dict[str, dict[str, list[int]]] = {}
        self._opened = False

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def _read_or_raise(self, path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as e:
            raise MemoryMapFileOperationError(f"missing {path}") from e

    def _load_content_idx(self) -> None:
        raw = self._read_or_raise(self._content_idx_path)
        # A2 (packed) is raw 'Q' bytes; a legacy A1 (hex) index is a JSON object
        # beginning with b'{' and is no longer readable.
        if raw[:1] == b"{":
            raise ValueError(
                f"legacy non-interned unified store (pre-v3) at "
                f"{self._content_idx_path}; re-parse required"
            )
        flat = _array.array(_IDX_TYPECODE)
        flat.frombytes(raw)
        self._spans = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]

    def _load_idx(self, path: Path) -> dict:
        try:
            data = path.read_bytes()
        except OSError as e:
            raise MemoryMapFileOperationError(f"missing {path}") from e
        try:
            return orjson.loads(data)
        except orjson.JSONDecodeError as e:
            raise MemoryMapSerializationError(str(e)) from e

    def _map_blob(self, path: Path) -> tuple[mmap.mmap | None, memoryview | None]:
        if not path.exists():
            raise MemoryMapFileOperationError(f"missing {path}")
        if path.stat().st_size == 0:
            return None, None
        fh = path.open("rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        fh.close()
        return mm, memoryview(mm)

    def _validate_content_spans(self) -> None:
        """Reject spans past the end of ``content.blob`` (mirrors the nodes-region
        check in :meth:`get_node_envelope`) -- Python slice clamping would
        otherwise return silently truncated bytes from a stale/partial store."""
        content_len = 0 if self._content_mv is None else len(self._content_mv)
        for handle, (off, size) in enumerate(self._spans):
            if off + size > content_len:
                raise MemoryMapSerializationError(
                    f"content handle {handle}: offset {off}+{size} exceeds "
                    f"content.blob ({content_len})"
                )

    def open(self) -> GraphSegmentUnifiedClient:
        try:
            self._load_content_idx()
            self._node_offsets = self._load_idx(self._nodes_idx_path)
            self._content_mm, self._content_mv = self._map_blob(self._content_blob_path)
            self._nodes_mm, self._nodes_mv = self._map_blob(self._nodes_blob_path)
            self._validate_content_spans()
        except BaseException:
            self.close()
            raise
        self._opened = True
        return self

    def _require_opened(self, operation: str) -> None:
        """Reject reads on an unopened client with an actionable error."""
        if not self._opened:
            raise RuntimeError(
                f"GraphSegmentUnifiedClient.{operation}: unified store at "
                f"{self._data_dir} is not opened; call open() (or use the "
                "client as a context manager) before reading."
            )

    # --- addressing face (per-node envelope reads) ---
    def get_node_envelope(
        self, trace_id: str, node_ordinal: int, phase_variant: str = "profiling"
    ) -> bytes | None:
        self._require_opened("get_node_envelope")
        trace_offsets = self._node_offsets.get(trace_id)
        if trace_offsets is None:
            return None
        info = trace_offsets.get(_encode_inner_key(node_ordinal, phase_variant))
        if info is None:
            return None
        off, size = info
        end = off + size
        nodes_len = 0 if self._nodes_mv is None else len(self._nodes_mv)
        if self._nodes_mv is None or end > nodes_len:
            raise MemoryMapSerializationError(
                f"{trace_id!r} node {node_ordinal} ({phase_variant!r}): "
                f"offset {off}+{size} exceeds nodes.blob ({nodes_len})"
            )
        return bytes(self._nodes_mv[off:end])

    # --- interned (A2) int-handle face ---
    def materialize_handles(self, handles: list[int]) -> list[dict[str, str]]:
        self._require_opened("materialize_handles")
        out: list[dict[str, str]] = []
        for h in handles:
            if h < 0 or h >= len(self._spans):
                raise MemoryMapSerializationError(f"unknown handle {h!r}")
            assert self._content_mv is not None
            off, size = self._spans[h]
            out.append(orjson.loads(self._content_mv[off : off + size]))
        return out

    def build_request_body_handles(
        self, handles: list[int], overrides_inner: bytes
    ) -> bytes:
        self._require_opened("build_request_body_handles")
        parts: list[bytes | memoryview] = [_HEAD]
        first = True
        for h in handles:
            if h < 0 or h >= len(self._spans):
                raise MemoryMapSerializationError(f"unknown handle {h!r}")
            assert self._content_mv is not None
            off, size = self._spans[h]
            if not first:
                parts.append(b",")
            parts.append(self._content_mv[off : off + size])
            first = False
        parts.append(b"]}" if not overrides_inner else b"]," + overrides_inner + b"}")
        return b"".join(parts)

    def close(self) -> None:
        for mv_attr, mm_attr in (
            ("_content_mv", "_content_mm"),
            ("_nodes_mv", "_nodes_mm"),
        ):
            mv = getattr(self, mv_attr)
            mm = getattr(self, mm_attr)
            if mv is not None:
                mv.release()
                setattr(self, mv_attr, None)
            if mm is not None:
                mm.close()
                setattr(self, mm_attr, None)
        self._opened = False

    def __enter__(self) -> GraphSegmentUnifiedClient:
        return self.open() if not self._opened else self

    def __exit__(self, *exc) -> None:
        self.close()
