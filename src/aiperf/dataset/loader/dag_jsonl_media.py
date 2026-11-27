# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic media and text for structural ``dag_jsonl`` turns.

Images render as PNG and videos as H.264 (``mpeg4`` fallback) MP4, keyed by
``DagMediaSegment.cache_key`` and sent as inline ``data:`` URIs. Text spans are
built from neutral single-token words and, when a tokenizer is available,
calibrated to the exact requested token count.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import inspect
import io
import math
import random
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
from PIL import Image as PILImage

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.dataset.loader.dag_jsonl_models import DagMediaSegment, DagTurn

_logger = AIPerfLogger(__name__)

SINGLE_TOKEN_NEUTRAL_WORDS: tuple[str, ...] = (
    "scene", "frame", "camera", "motion", "subject", "light", "shadow", "object",
    "surface", "table", "ground", "sky", "background", "foreground", "left", "right",
    "center", "steady", "pan", "zoom", "track", "focus", "detail", "texture", "color",
    "balance", "transition", "interval", "sequence", "action", "start", "middle",
    "finish", "rule", "constraint", "goal", "check", "score", "verify", "align",
    "consistent", "smooth", "clear", "dynamic", "angle", "timing", "anchor", "state",
    "context", "step", "plan", "adjust", "render", "output", "window", "view", "path",
    "field", "group", "layer", "order", "scale", "depth", "sharp",
)  # fmt: skip

MAX_VIDEO_RAW_BYTES: int = 8 * 1024**3
INLINE_SEGMENT_WARN_BYTES: int = 4 * 1024**2
INLINE_TOTAL_WARN_BYTES: int = 32 * 1024**2
_MAX_CALIBRATION_PROBES: int = 48
_STDERR_TAIL_MAX_BYTES: int = 4096
_MIB: float = 1024.0**2


def _seed64(*parts: object) -> int:
    joined = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def synth_deterministic_png_bytes(width: int, height: int, seed_key: str) -> bytes:
    """Encode a ``width x height`` RGB PNG of 4x3 colour blocks seeded by ``seed_key``."""
    rng = np.random.default_rng(_seed64("png", seed_key, width, height))
    blocks = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    rows = np.arange(height) * 3 // height
    cols = np.arange(width) * 4 // width
    buf = io.BytesIO()
    PILImage.fromarray(blocks[rows[:, None], cols[None, :]]).save(
        buf, format="PNG", compress_level=6
    )
    return buf.getvalue()


def iter_deterministic_rgb_frames(
    width: int, height: int, num_frames: int, seed_key: str
) -> Iterator[np.ndarray]:
    """Yield ``num_frames`` deterministic ``(height, width, 3)`` uint8 frames one at a time."""
    total_raw_bytes = width * height * num_frames * 3
    if total_raw_bytes > MAX_VIDEO_RAW_BYTES:
        raise ValueError(
            f"Video segment '{seed_key}' raw RGB stream ({total_raw_bytes} bytes) "
            f"exceeds safety limit ({MAX_VIDEO_RAW_BYTES} bytes)"
        )
    rng = np.random.default_rng(_seed64("mp4_rgb", seed_key, width, height, num_frames))
    tile = 16
    grid = (max(2, math.ceil(height / tile)), max(2, math.ceil(width / tile)), 3)
    base = rng.integers(16, 240, size=grid, dtype=np.uint8)
    frame0 = np.repeat(np.repeat(base, tile, axis=0), tile, axis=1)[:height, :width]
    for idx in range(num_frames):
        frame = np.roll(frame0, shift=(idx * 16) % width, axis=1)
        if idx % 2 == 1:
            frame = np.roll(frame, shift=(idx * 8) % height, axis=0)
        yield np.ascontiguousarray(frame)


def find_ffmpeg_binary() -> str | None:
    """Locate ``ffmpeg`` on ``PATH``, falling back to the optional ``imageio_ffmpeg`` binary."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    with contextlib.suppress(Exception):
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    return None


class _StderrTailDrain:
    """Drain a subprocess ``stderr`` pipe on a daemon thread, keeping only its tail.

    Without a concurrent reader, a chatty ffmpeg fills the pipe buffer and
    deadlocks against our blocking stdin writes / stdout reads.
    """

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._tail = bytearray()
        self._thread = threading.Thread(
            target=self._drain, name="aiperf-ffmpeg-stderr", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            while chunk := self._stream.read(4096):
                self._tail.extend(chunk)
                del self._tail[:-_STDERR_TAIL_MAX_BYTES]

    def tail(self, timeout: float = 5.0) -> str:
        self._thread.join(timeout)
        return bytes(self._tail).decode("utf-8", "ignore").strip()[-200:]


@contextlib.contextmanager
def _run_ffmpeg(
    cmd: list[str], *, feed_stdin: bool
) -> Iterator[tuple[subprocess.Popen[bytes], _StderrTailDrain]]:
    """Run ``cmd`` with a drained ``stderr``; on exit kill it if still running, then reap it."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if feed_stdin else subprocess.DEVNULL,
        stdout=subprocess.DEVNULL if feed_stdin else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield proc, _StderrTailDrain(proc.stderr)  # type: ignore[arg-type]
    finally:
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        if proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()
        proc.wait()
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


def _encode_mp4(
    ffmpeg_bin: str,
    codec: str,
    frames: Iterable[np.ndarray],
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> tuple[int, str]:
    """Stream raw RGB ``frames`` into ffmpeg's stdin; return ``(returncode, stderr tail)``."""
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{fps:.2f}", "-i", "-",
        "-an", "-c:v", codec, "-pix_fmt", "yuv420p",
        *(["-preset", "ultrafast", "-crf", "23"] if codec == "libx264" else []),
        "-movflags", "+faststart", str(out_path),
    ]  # fmt: skip
    with _run_ffmpeg(cmd, feed_stdin=True) as (proc, stderr):
        assert proc.stdin is not None
        with contextlib.suppress(OSError):
            for frame in frames:
                proc.stdin.write(memoryview(frame).cast("B"))
            proc.stdin.close()
        return proc.wait(), stderr.tail()


def _parse_y4m_header(line: bytes) -> tuple[int, int, float | None]:
    fields = {tok[:1]: tok[1:] for tok in line.decode("ascii", "replace").split()[1:]}
    try:
        width, height = int(fields["W"]), int(fields["H"])
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"Missing frame geometry in Y4M header: {line[:80]!r}"
        ) from e
    num, _, den = fields.get("F", "").partition(":")
    if num.isdigit() and den.isdigit() and int(den) > 0:
        return width, height, int(num) / int(den)
    return width, height, None


def _count_y4m_frames(stream: IO[bytes], frame_bytes: int) -> int:
    count = 0
    while marker := stream.readline(1024):
        if not (
            marker == b"FRAME\n"
            or (marker.startswith(b"FRAME ") and marker.endswith(b"\n"))
        ):
            raise RuntimeError(
                f"Invalid Y4M frame marker at frame {count}: {marker[:40]!r}"
            )
        remaining = frame_bytes
        while remaining:
            chunk = stream.read(min(65536, remaining))
            if not chunk:
                raise RuntimeError(f"Truncated Y4M frame payload at frame {count}")
            remaining -= len(chunk)
        count += 1
    return count


def verify_mp4_decodable(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    ffmpeg_bin: str,
) -> None:
    """Stream-decode ``path`` to YUV4MPEG2 and check geometry, frame rate and frame count."""
    cmd = [
        ffmpeg_bin, "-v", "error", "-i", str(path),
        "-f", "yuv4mpegpipe", "-pix_fmt", "yuv444p", "-strict", "-1", "-",
    ]  # fmt: skip
    with _run_ffmpeg(cmd, feed_stdin=False) as (proc, stderr):
        assert proc.stdout is not None
        header = proc.stdout.readline(1024)
        if not header.startswith(b"YUV4MPEG2 "):
            proc.kill()
            proc.wait()
            raise RuntimeError(
                f"ffmpeg produced no Y4M stream for {path}: {stderr.tail()}"
            )
        width, height, fps = _parse_y4m_header(header)
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"Decoded MP4 dimension mismatch for {path}: got {width}x{height}, "
                f"expected {expected_width}x{expected_height}"
            )
        fps_tolerance = max(0.02, expected_fps * 0.005)
        if fps is not None and abs(fps - expected_fps) > fps_tolerance:
            raise RuntimeError(
                f"Decoded MP4 frame-rate mismatch for {path}: got {fps:.2f}fps, "
                f"expected {expected_fps:.2f}fps"
            )
        frames = _count_y4m_frames(proc.stdout, width * height * 3)
        rc = proc.wait()
        tail = stderr.tail()
    if rc != 0 or frames != expected_frames:
        raise RuntimeError(
            f"ffmpeg decode verification failed for {path} (rc={rc}, "
            f"frames={frames}, expected={expected_frames}): {tail}"
        )


def encode_deterministic_mp4(
    segment: DagMediaSegment, *, ffmpeg_bin: str | None = None
) -> bytes:
    """Stream-encode ``segment`` to a decode-verified ``yuv420p`` MP4 and return its bytes."""
    ff = ffmpeg_bin or find_ffmpeg_binary()
    if ff is None:
        raise RuntimeError(
            f"ffmpeg not found while encoding video segment '{segment.media_hash}'; "
            "install ffmpeg (e.g. 'apt install ffmpeg' or 'brew install ffmpeg')"
        )
    width, height, num_frames = segment.width, segment.height, segment.num_frames
    fps = segment.cache_key[4]
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="aiperf_dag_mp4_") as tmp:
        out_path = Path(tmp) / "clip.mp4"
        for codec in ("libx264", "mpeg4"):
            frames = iter_deterministic_rgb_frames(
                width, height, num_frames, segment.media_hash
            )
            rc, tail = _encode_mp4(
                ff, codec, frames, out_path, width=width, height=height, fps=fps
            )
            if rc != 0 or not out_path.is_file():
                errors.append(f"{codec} rc={rc}: {tail}")
                continue
            try:
                verify_mp4_decodable(
                    out_path,
                    expected_frames=num_frames,
                    expected_width=width,
                    expected_height=height,
                    expected_fps=fps,
                    ffmpeg_bin=ff,
                )
            except RuntimeError as e:
                errors.append(f"{codec}: {e}")
                continue
            return out_path.read_bytes()
    raise RuntimeError(
        f"Failed to encode a decodable MP4 for '{segment.media_hash}' "
        f"({width}x{height}, {num_frames} frames): {'; '.join(errors)}"
    )


def default_media_uuid(segment: DagMediaSegment) -> str:
    """Derive a stable cache ``uuid`` from the full ``segment.cache_key``."""
    identity = "\x1f".join(str(p) for p in segment.cache_key)
    return f"dag-{segment.kind}-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _token_counter(tokenizer: Any | None) -> Callable[[str], int] | None:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    # AIPerf's Tokenizer already disables special tokens via ``_encode_args``.
    if not hasattr(tokenizer, "_encode_args"):
        with contextlib.suppress(TypeError, ValueError):
            if "add_special_tokens" in inspect.signature(encode).parameters:
                encode = functools.partial(encode, add_special_tokens=False)
    return lambda text: len(encode(text))


class DagMultimodalMaterializer:
    """Render structural ``DagTurn`` content into OpenAI chat messages.

    Every unique ``cache_key`` is encoded once and each reference reuses the
    same ``data:`` URI, so identical segments across turns, FORK children and
    SPAWN sub-agents send byte-identical media.
    """

    def __init__(
        self,
        *,
        tokenizer: Any | None = None,
        emit_media_uuid: bool = False,
        ffmpeg_bin: str | None = None,
    ) -> None:
        self.emit_media_uuid = emit_media_uuid
        self.ffmpeg_bin = ffmpeg_bin
        self._count_tokens = _token_counter(tokenizer)
        self._uris: dict[tuple[Any, ...], str] = {}
        self._texts: dict[tuple[str, int], str] = {}
        self.inline_media_refs = 0
        self.inline_media_bytes = 0
        self._largest_inline: tuple[int, str] = (0, "")

    def materialize_turn_messages(
        self, turn: DagTurn, *, fallback_key: str
    ) -> list[dict[str, Any]]:
        """Return the OpenAI ``messages`` for ``turn``, synthesizing structural content."""
        if turn.messages is not None:
            if not turn.media_segments:
                return list(turn.messages)
            return [self._resolve_media_refs(m, turn) for m in turn.messages]
        messages: list[dict[str, Any]] = []
        for role, part in self._iter_span_parts(turn, fallback_key):
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].append(part)
            else:
                messages.append({"role": role, "content": [part]})
        return messages

    def _resolve_media_refs(self, msg: dict[str, Any], turn: DagTurn) -> dict[str, Any]:
        content = msg.get("content")
        if not isinstance(content, list):
            return msg
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "media_ref":
                idx = part.get("media_index")
                if idx is None:
                    idx = next(
                        i
                        for i, seg in enumerate(turn.media_segments)
                        if seg.media_hash == part["media_hash"]
                    )
                part = self.render_media_part(turn.media_segments[idx])
            parts.append(part)
        return {**msg, "content": parts}

    def _iter_span_parts(
        self, turn: DagTurn, fallback_key: str
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        segments = turn.media_segments
        if not turn.prompt_spans:
            if turn.in_text_tokens:
                yield (
                    "user",
                    self._text_part(f"{fallback_key}:text", turn.in_text_tokens),
                )
            for seg in segments:
                yield "user", self.render_media_part(seg)
            return
        first_by_hash: dict[str, int] = {}
        for idx, seg in enumerate(segments):
            first_by_hash.setdefault(seg.media_hash, idx)
        next_idx = 0
        for span in turn.prompt_spans:
            if span.kind == "text":
                yield span.role, self._text_part(span.span_id, span.tokens or 1)
                continue
            if span.media_index is not None:
                idx = span.media_index
            else:
                idx = first_by_hash.get(span.span_id, next_idx)
            next_idx = idx + 1
            yield span.role, self.render_media_part(segments[idx])

    def _text_part(self, span_id: str, num_tokens: int) -> dict[str, Any]:
        return {"type": "text", "text": self.render_span_text(span_id, num_tokens)}

    def render_span_text(self, span_id: str, num_tokens: int) -> str:
        """Return deterministic text for ``span_id``, calibrated to ``num_tokens`` when a tokenizer is set."""
        key = (span_id, num_tokens)
        if key not in self._texts:
            self._texts[key] = self._synthesize_text(span_id, num_tokens)
        return self._texts[key]

    def _synthesize_text(self, span_id: str, target: int) -> str:
        rng = random.Random(_seed64("span_tok", span_id))
        words: list[str] = []

        def text_of(num_words: int) -> str:
            while len(words) < num_words:
                words.append(rng.choice(SINGLE_TOKEN_NEUTRAL_WORDS))
            return " " + " ".join(words[:num_words])

        if self._count_tokens is None:
            return text_of(target)
        return self._calibrate(span_id, target, text_of, self._count_tokens)

    @staticmethod
    def _calibrate(
        span_id: str,
        target: int,
        text_of: Callable[[int], str],
        count: Callable[[str], int],
    ) -> str:
        # Grow (then bisect) the word count bracketing ``target``; a word is not
        # always exactly one token, so finish by padding with 1-token " a".
        lo_k, lo_text, lo_n = 0, "", 0
        hi_k: int | None = None
        max_words = max(target * 8, target + 64)
        k = target
        for _ in range(_MAX_CALIBRATION_PROBES):
            text = text_of(k)
            n = count(text)
            if n == target:
                return text
            if n < target:
                lo_k, lo_text, lo_n = k, text, n
            else:
                hi_k = k
            if hi_k is None:
                k = min(max_words, max(k + 1, math.ceil(k * target / max(1, n))))
            else:
                k = (lo_k + hi_k) // 2
            if k in (lo_k, hi_k):
                break
        for pad in range(1, min(16, target - lo_n) + 1):
            text = lo_text + " a" * pad
            if count(text) == target:
                return text
        raise ValueError(
            f"Tokenizer could not calibrate span '{span_id}' to {target} tokens "
            f"(closest below: {lo_n})"
        )

    def render_media_part(self, segment: DagMediaSegment) -> dict[str, Any]:
        """Return a fresh ``image_url`` / ``video_url`` content part for ``segment``."""
        uri = self._uris.get(segment.cache_key)
        if uri is None:
            uri = self._uris[segment.cache_key] = self._encode_data_uri(segment)
        self.inline_media_refs += 1
        self.inline_media_bytes += len(uri)
        if len(uri) > self._largest_inline[0]:
            self._largest_inline = (len(uri), segment.media_hash)
        url_key = "image_url" if segment.kind == "image" else "video_url"
        part: dict[str, Any] = {"type": url_key, url_key: {"url": uri}}
        uuid = segment.uuid or (
            default_media_uuid(segment) if self.emit_media_uuid else None
        )
        if uuid is not None:
            part["uuid"] = uuid
        return part

    def _encode_data_uri(self, segment: DagMediaSegment) -> str:
        if segment.kind == "image":
            mime = "image/png"
            payload = synth_deterministic_png_bytes(
                segment.width, segment.height, segment.media_hash
            )
        else:
            mime = "video/mp4"
            payload = encode_deterministic_mp4(segment, ffmpeg_bin=self.ffmpeg_bin)
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    def warn_if_inline_payload_large(self) -> None:
        """Log one warning when inline ``data:`` URIs exceed the per-segment or total threshold."""
        largest, media_hash = self._largest_inline
        if (
            largest <= INLINE_SEGMENT_WARN_BYTES
            and self.inline_media_bytes <= INLINE_TOTAL_WARN_BYTES
        ):
            return
        _logger.warning(
            f"dag_jsonl inline media is large: {self.inline_media_bytes / _MIB:.1f} MiB "
            f"of data: URIs across {self.inline_media_refs} media references "
            f"(largest '{media_hash}': {largest / _MIB:.1f} MiB). Every turn that "
            "references a segment repeats its data: URI on the wire; reduce "
            "width/height/num_frames to shrink requests."
        )
