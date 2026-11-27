# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multimodal media synthesis and untimed staging for ``dag_jsonl``.

Provides content-addressed PNG and H.264 MP4 generation keyed by
``(kind, media_hash, width, height, fps, num_frames)``, untimed pre-staging
across ``FORK`` and ``SPAWN`` branches, optional local HTTP static serving,
mixed text+media block-hash computation, and heuristic structural trace checks.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import hashlib
import http.server
import inspect
import io
import math
import os
import random
import re
import shutil
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from aiperf.dataset.loader.dag_jsonl_models import (
        DagMediaSegment,
        DagTurn,
    )

SINGLE_TOKEN_NEUTRAL_WORDS: tuple[str, ...] = (
    "scene",
    "frame",
    "camera",
    "motion",
    "subject",
    "light",
    "shadow",
    "object",
    "surface",
    "table",
    "ground",
    "sky",
    "background",
    "foreground",
    "left",
    "right",
    "center",
    "steady",
    "pan",
    "zoom",
    "track",
    "focus",
    "detail",
    "texture",
    "color",
    "balance",
    "transition",
    "interval",
    "sequence",
    "action",
    "start",
    "middle",
    "finish",
    "rule",
    "constraint",
    "goal",
    "check",
    "score",
    "verify",
    "align",
    "consistent",
    "smooth",
    "clear",
    "dynamic",
    "angle",
    "timing",
    "anchor",
    "state",
    "context",
    "step",
    "plan",
    "adjust",
    "render",
    "output",
    "window",
    "view",
    "path",
    "field",
    "group",
    "layer",
    "order",
    "scale",
    "depth",
    "sharp",
)

ZERO_LEAKAGE_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "user_prompt",
        "messages",
        "text",
        "raw_text",
        "content",
        "pixel_values",
        "raw_bytes",
        "image_bytes",
        "video_bytes",
        "base64",
        "file_name",
    }
)

MAX_STREAMED_VIDEO_RAW_BYTES: int = 8 * 1024 * 1024 * 1024
MAX_BUFFERED_VIDEO_RAW_BYTES: int = 256 * 1024 * 1024
_UNSAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _seed64(*parts: object) -> int:
    joined = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def validate_dag_zero_leakage(obj: Any, *, path: str = "$") -> None:
    """Heuristic structural lint rejecting raw-content keys, inline URIs, and long strings.

    Checks that ``obj`` omits common raw-payload keys (such as ``messages`` and
    ``raw_text``), inline ``data:`` / ``file://`` media URIs, and string values
    exceeding 256 characters. This helper is a structural guardrail, not a
    semantic guarantee that opaque fields contain no confidential text; trace
    authors must supply opaque identifiers for ``session_id``, ``span_id``, and
    ``media_hash`` and omit source data.
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ZERO_LEAKAGE_FORBIDDEN_KEYS:
                raise ValueError(
                    f"Zero-leakage violation: forbidden key {key!r} at {path}"
                )
            validate_dag_zero_leakage(val, path=f"{path}.{key}")
        return
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            validate_dag_zero_leakage(item, path=f"{path}[{idx}]")
        return
    if isinstance(obj, str):
        if obj.startswith(("data:image/", "data:video/", "file://")):
            raise ValueError(
                f"Zero-leakage violation: inline media URI found at {path}"
            )
        if len(obj) > 256:
            raise ValueError(
                f"Zero-leakage violation: oversized raw string ({len(obj)} chars) at {path}"
            )


def compute_mixed_block_hashes(
    segments: Iterable[tuple[str, str, int]],
    *,
    block_size: int = 64,
) -> tuple[list[str], int]:
    """Compute chained ``block_size``-token hashes over mixed ``(kind, key, count)`` spans."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    pseudo_ids: list[int] = []
    unpack_u32 = struct.Struct("<8I").unpack
    for kind, key, count in segments:
        if count <= 0:
            continue
        base_seed = f"{kind}:{key}:".encode()
        n_batches = (count + 7) // 8
        produced = 0
        for batch_idx in range(n_batches):
            digest = hashlib.sha256(base_seed + str(batch_idx).encode("ascii")).digest()
            vals = unpack_u32(digest)
            take = min(8, count - produced)
            pseudo_ids.extend(vals[:take])
            produced += take

    n_blocks = len(pseudo_ids) // block_size
    pack_block = struct.Struct(f"<{block_size}I").pack
    prev = b""
    hashes: list[str] = []
    for idx in range(n_blocks):
        chunk = pseudo_ids[idx * block_size : (idx + 1) * block_size]
        prev = hashlib.sha256(prev + pack_block(*chunk)).digest()[:8]
        hashes.append(prev.hex())
    tail_tokens = len(pseudo_ids) - n_blocks * block_size
    return hashes, tail_tokens


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def synth_deterministic_png_bytes(
    width: int,
    height: int,
    seed_key: str,
    *,
    cols: int = 4,
    rows: int = 3,
) -> bytes:
    """Encode a deterministic RGB PNG of exact ``width x height`` keyed by ``seed_key``."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"PNG dimensions must be positive integers, got {width}x{height}"
        )
    rng = random.Random(_seed64("png", seed_key, width, height))
    colours = [
        [bytes(rng.randrange(256) for _ in range(3)) for _ in range(cols)]
        for _ in range(rows)
    ]
    xs = [width * i // cols for i in range(cols + 1)]
    ys = [height * j // rows for j in range(rows + 1)]
    zero_row = b"\x02" + bytes(width * 3)
    compressor = zlib.compressobj(6)
    idat_parts: list[bytes] = []
    for row_idx in range(rows):
        row_h = ys[row_idx + 1] - ys[row_idx]
        if row_h <= 0:
            continue
        band = b"".join(
            colours[row_idx][col_idx] * (xs[col_idx + 1] - xs[col_idx])
            for col_idx in range(cols)
        )
        idat_parts.append(compressor.compress(b"\x00" + band))
        for _ in range(row_h - 1):
            idat_parts.append(compressor.compress(zero_row))
    idat_parts.append(compressor.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"".join(idat_parts))
        + _png_chunk(b"IEND", b"")
    )


def verify_png_bytes(
    png_bytes: bytes,
    *,
    expected_width: int,
    expected_height: int,
) -> None:
    """Verify ``png_bytes`` decodes cleanly to ``(expected_width, expected_height)`` RGB."""
    if len(png_bytes) < 33 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Generated PNG payload lacks a valid PNG file signature")
    ihdr_w, ihdr_h = struct.unpack(">II", png_bytes[16:24])
    if (ihdr_w, ihdr_h) != (expected_width, expected_height):
        raise RuntimeError(
            f"PNG IHDR dimension mismatch: got {ihdr_w}x{ihdr_h}, "
            f"expected {expected_width}x{expected_height}"
        )
    with PILImage.open(io.BytesIO(png_bytes)) as img:
        img.load()
        if img.size != (expected_width, expected_height) or img.mode != "RGB":
            raise RuntimeError(
                f"Decoded PNG mismatch: got size={img.size} mode={img.mode}, "
                f"expected ({expected_width}, {expected_height}) RGB"
            )


def iter_deterministic_rgb_frames(
    width: int,
    height: int,
    num_frames: int,
    seed_key: str,
) -> Iterator[np.ndarray]:
    """Yield ``num_frames`` deterministic ``(height, width, 3)`` uint8 RGB arrays one at a time."""
    if width <= 0 or height <= 0 or num_frames <= 0:
        raise ValueError(
            f"Video frame geometry must be positive, got {width}x{height} @ {num_frames}f"
        )
    total_raw_bytes = width * height * num_frames * 3
    if total_raw_bytes > MAX_STREAMED_VIDEO_RAW_BYTES:
        raise ValueError(
            f"Video segment '{seed_key}' raw RGB stream ({total_raw_bytes} bytes) "
            f"exceeds safety limit ({MAX_STREAMED_VIDEO_RAW_BYTES} bytes)"
        )
    rng = np.random.default_rng(_seed64("mp4_rgb", seed_key, width, height, num_frames))
    tile = 16
    gh = max(2, math.ceil(height / tile))
    gw = max(2, math.ceil(width / tile))
    base = rng.integers(16, 240, size=(gh, gw, 3), dtype=np.uint8)
    frame0 = np.repeat(np.repeat(base, tile, axis=0), tile, axis=1)[:height, :width, :]
    for frame_idx in range(num_frames):
        shifted = np.roll(frame0, shift=(frame_idx * 16) % max(1, width), axis=1)
        if frame_idx % 2 == 1:
            shifted = np.roll(shifted, shift=(frame_idx * 8) % max(1, height), axis=0)
        yield np.ascontiguousarray(shifted)


def synthesize_deterministic_rgb_frames(
    width: int,
    height: int,
    num_frames: int,
    seed_key: str,
) -> list[np.ndarray]:
    """Generate a materialized list of ``num_frames`` deterministic RGB arrays for small clips."""
    total_raw_bytes = width * height * num_frames * 3
    if total_raw_bytes > MAX_BUFFERED_VIDEO_RAW_BYTES:
        raise ValueError(
            f"Buffered RGB synthesis for '{seed_key}' ({total_raw_bytes} bytes) exceeds "
            f"{MAX_BUFFERED_VIDEO_RAW_BYTES} bytes; use iter_deterministic_rgb_frames instead"
        )
    return list(iter_deterministic_rgb_frames(width, height, num_frames, seed_key))


def find_ffmpeg_binary() -> str | None:
    """Locate system ``ffmpeg`` or optional ``imageio_ffmpeg`` binary if installed."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    with contextlib.suppress(Exception):
        import imageio_ffmpeg  # type: ignore[import-not-found]

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return str(exe)
    return None


def _effective_video_fps(fps: float | None) -> float:
    """Return normalized two-decimal video FPS, defaulting non-positive inputs to 24.00."""
    if fps is None or fps <= 0.0:
        return 24.0
    eff = round(float(fps), 2)
    if eff <= 0.0:
        raise ValueError(
            f"Video FPS {fps} rounds to 0.00 at two-decimal precision; "
            "supply fps >= 0.01"
        )
    return eff


def _parse_y4m_header(header_line: bytes) -> tuple[int, int, float | None]:
    if not header_line.startswith(b"YUV4MPEG2 "):
        raise RuntimeError(f"Unexpected FFmpeg Y4M stream header: {header_line[:80]!r}")
    width = 0
    height = 0
    fps: float | None = None
    for token in header_line.decode("ascii", "replace").split()[1:]:
        if token.startswith("W") and len(token) > 1:
            width = int(token[1:])
        elif token.startswith("H") and len(token) > 1:
            height = int(token[1:])
        elif token.startswith("F") and ":" in token:
            num_s, den_s = token[1:].split(":", 1)
            den = int(den_s)
            if den > 0:
                fps = int(num_s) / den
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Missing frame geometry in FFmpeg Y4M header: {header_line[:80]!r}"
        )
    return width, height, fps


def _discard_exact_bytes(stream: Any, num_bytes: int, chunk_size: int = 65536) -> bool:
    remaining = num_bytes
    while remaining > 0:
        chunk = stream.read(min(chunk_size, remaining))
        if not chunk:
            return False
        remaining -= len(chunk)
    return True


def _consume_y4m_frames(stream: Any, frame_bytes: int, path: Path) -> int:
    decoded_frames = 0
    while True:
        marker = stream.readline(1024)
        if not marker:
            return decoded_frames
        if not marker.endswith(b"\n") or not (
            marker == b"FRAME\n" or marker.startswith(b"FRAME ")
        ):
            raise RuntimeError(
                f"Invalid Y4M frame marker at frame {decoded_frames} for {path}: {marker[:40]!r}"
            )
        if not _discard_exact_bytes(stream, frame_bytes):
            raise RuntimeError(
                f"Truncated Y4M frame payload at frame {decoded_frames} for {path}"
            )
        decoded_frames += 1


def _terminate_proc(proc: subprocess.Popen[bytes]) -> bytes:
    with contextlib.suppress(OSError):
        proc.kill()
    stderr_data = b""
    if proc.stderr is not None:
        with contextlib.suppress(OSError):
            stderr_data = proc.stderr.read()
    proc.wait()
    return stderr_data


def _verify_y4m_stream(
    proc: subprocess.Popen[bytes],
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float | None,
) -> tuple[int, int, bytes]:
    assert proc.stdout is not None and proc.stderr is not None
    header_line = proc.stdout.readline(1024)
    if not header_line or not header_line.endswith(b"\n"):
        stderr_msg = _terminate_proc(proc).decode("utf-8", "ignore")[:200]
        raise RuntimeError(
            f"FFmpeg decode produced no valid Y4M header for {path}: {stderr_msg}"
        )
    actual_w, actual_h, actual_fps = _parse_y4m_header(header_line.rstrip(b"\r\n"))
    if (actual_w, actual_h) != (expected_width, expected_height):
        _terminate_proc(proc)
        raise RuntimeError(
            f"Decoded MP4 dimension mismatch for {path}: got {actual_w}x{actual_h}, "
            f"expected {expected_width}x{expected_height}"
        )
    if expected_fps is not None and actual_fps is not None:
        eff_fps = _effective_video_fps(expected_fps)
        if abs(actual_fps - eff_fps) > max(0.02, eff_fps * 0.005):
            _terminate_proc(proc)
            raise RuntimeError(
                f"Decoded MP4 frame-rate mismatch for {path}: got {actual_fps:.2f}fps, "
                f"expected {eff_fps:.2f}fps"
            )
    decoded_frames = _consume_y4m_frames(proc.stdout, actual_w * actual_h * 3, path)
    stderr_bytes = proc.stderr.read()
    rc = proc.wait()
    return rc, decoded_frames, stderr_bytes


def verify_mp4_decodable(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float | None = None,
    ffmpeg_bin: str | None = None,
) -> None:
    """Stream-decode ``path`` via ``ffmpeg`` Y4M output to verify geometry, FPS, and frame count."""
    if not path.is_file() or path.stat().st_size < 128:
        raise RuntimeError(f"Generated MP4 file is missing or truncated: {path}")
    ff = ffmpeg_bin or find_ffmpeg_binary()
    if ff is None:
        raise RuntimeError(
            "Cannot verify MP4 decodability because 'ffmpeg' binary was not found"
        )
    cmd = [
        ff,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "yuv4mpegpipe",
        "-pix_fmt",
        "yuv444p",
        "-strict",
        "-1",
        "-",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    try:
        rc, decoded_frames, stderr_bytes = _verify_y4m_stream(
            proc,
            path,
            expected_frames=expected_frames,
            expected_width=expected_width,
            expected_height=expected_height,
            expected_fps=expected_fps,
        )
    finally:
        proc.stdout.close()
        proc.stderr.close()
        if proc.poll() is None:
            _terminate_proc(proc)

    if rc != 0 or decoded_frames != expected_frames:
        stderr_snippet = stderr_bytes.decode("utf-8", "ignore")[:200]
        raise RuntimeError(
            f"FFmpeg decode verification failed for {path} (rc={rc}, "
            f"frames={decoded_frames} vs expected={expected_frames}): {stderr_snippet}"
        )


def _encode_mp4_with_codec(
    ff: str,
    codec: str,
    frames: Iterable[np.ndarray],
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> tuple[int, bytes]:
    cmd = [
        ff,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.2f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if codec == "libx264":
        cmd.extend(["-preset", "ultrafast", "-crf", "23"])
    cmd.extend(["-movflags", "+faststart", str(out_path)])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for frame in frames:
            try:
                proc.stdin.write(memoryview(frame).cast("B"))
            except (BrokenPipeError, OSError):
                break
        with contextlib.suppress(BrokenPipeError, OSError):
            proc.stdin.close()
        stderr_bytes = proc.stderr.read()
        rc = proc.wait()
        return rc, stderr_bytes
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        proc.stderr.close()
        if proc.poll() is None:
            _terminate_proc(proc)


def write_deterministic_h264_mp4(
    path: Path,
    segment: DagMediaSegment,
    *,
    ffmpeg_bin: str | None = None,
) -> Path:
    """Stream-encode a deterministic ``yuv420p`` MP4 for ``segment`` and verify decodability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h, n_frames = segment.width, segment.height, segment.num_frames
    if w % 2 != 0 or h % 2 != 0:
        raise ValueError(f"yuv420p MP4 requires even dimensions, got {w}x{h}")
    fps = _effective_video_fps(segment.fps)
    ff = ffmpeg_bin or find_ffmpeg_binary()
    if ff is None:
        raise RuntimeError(
            f"FFmpeg binary not found while encoding video segment '{segment.media_hash}'. "
            "Install ffmpeg (e.g. 'apt install ffmpeg' or 'brew install ffmpeg') or use "
            "video_wire_mode='image_frames'."
        )
    if path.is_file() and path.stat().st_size >= 128:
        verify_mp4_decodable(
            path,
            expected_frames=n_frames,
            expected_width=w,
            expected_height=h,
            expected_fps=fps,
            ffmpeg_bin=ff,
        )
        return path

    tmp_path = path.with_name(
        f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
    )
    errors: list[str] = []
    for codec in ("libx264", "mpeg4"):
        frame_iter = iter_deterministic_rgb_frames(w, h, n_frames, segment.media_hash)
        rc, stderr_bytes = _encode_mp4_with_codec(
            ff, codec, frame_iter, tmp_path, width=w, height=h, fps=fps
        )
        if rc == 0 and tmp_path.is_file():
            try:
                verify_mp4_decodable(
                    tmp_path,
                    expected_frames=n_frames,
                    expected_width=w,
                    expected_height=h,
                    expected_fps=fps,
                    ffmpeg_bin=ff,
                )
                os.replace(tmp_path, path)
                return path
            except RuntimeError as exc:
                errors.append(f"{codec} verify failed: {exc}")
        else:
            stderr_msg = stderr_bytes.decode("utf-8", "ignore")[:160]
            errors.append(f"{codec} rc={rc}: {stderr_msg}")
        tmp_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Failed to encode decodable MP4 for '{segment.media_hash}' ({w}x{h}, {n_frames}f): "
        f"{'; '.join(errors)}"
    )


def _safe_media_stem(media_hash: str, identity_token: str) -> str:
    cleaned = _UNSAFE_SLUG_RE.sub("_", media_hash).strip("._-")[:32]
    slug = cleaned or "media"
    digest = hashlib.sha256(identity_token.encode("utf-8")).hexdigest()[:16]
    return f"{slug}_{digest}"


def staged_media_filename(segment: DagMediaSegment) -> str:
    """Return a filesystem- and URL-safe content-addressed basename for ``segment``."""
    if segment.kind == "image":
        identity = (
            f"image\x1f{segment.media_hash}\x1f{segment.width}\x1f{segment.height}"
        )
        stem = _safe_media_stem(segment.media_hash, identity)
        return f"{stem}_{segment.width}x{segment.height}.png"
    eff_fps = _effective_video_fps(segment.fps)
    identity = (
        f"video\x1f{segment.media_hash}\x1f{segment.width}\x1f{segment.height}"
        f"\x1f{eff_fps:.2f}\x1f{segment.num_frames}"
    )
    stem = _safe_media_stem(segment.media_hash, identity)
    return (
        f"{stem}_{segment.width}x{segment.height}_"
        f"{segment.num_frames}f_{eff_fps:.2f}fps.mp4"
    )


def staged_frame_filename(segment: DagMediaSegment, frame_idx: int) -> str:
    """Return a filesystem- and URL-safe basename for one PNG frame of ``segment``."""
    identity = (
        f"video_frame\x1f{segment.media_hash}\x1f{segment.width}\x1f{segment.height}"
    )
    stem = _safe_media_stem(segment.media_hash, identity)
    return f"{stem}_f{frame_idx:03d}_{segment.width}x{segment.height}.png"


class _QuietStaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


class _ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LocalMediaHttpServer:
    """Background-thread static HTTP server for untimed pre-staged DAG media files."""

    def __init__(
        self,
        directory: Path | str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        handler = functools.partial(_QuietStaticHandler, directory=str(self.directory))
        self._httpd = _ReusableThreadingHTTPServer((host, port), handler)
        self.host = host
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="aiperf-dag-media-http",
            daemon=True,
        )
        self._started = False
        self._closed = False

    @property
    def base_url(self) -> str:
        """Base HTTP URL serving staged media files (e.g. ``http://127.0.0.1:51234``)."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> LocalMediaHttpServer:
        """Start serving pre-staged files on the background daemon thread."""
        if self._closed:
            raise RuntimeError("Cannot start a closed LocalMediaHttpServer")
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def close(self) -> None:
        """Shut down the HTTP server (if started) and release its bound socket."""
        if self._closed:
            return
        self._closed = True
        if self._started and self._thread.is_alive():
            self._httpd.shutdown()
        self._httpd.server_close()
        if self._started:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> LocalMediaHttpServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class MediaStagingSummary:
    """Statistics produced by untimed pre-staging of DAG multimodal segments."""

    unique_images: int
    """Number of distinct content-addressed synthetic PNG images materialized."""

    unique_videos: int
    """Number of distinct content-addressed synthetic video clips materialized."""

    total_segment_references: int
    """Total ``DagMediaSegment`` references encountered across all turns/branches."""

    deduplicated_references: int
    """Number of repeated media-identity references served from the pre-staged cache."""

    staged_bytes: int
    """Total on-disk bytes written across unique pre-staged media files."""

    staging_wall_s: float
    """Untimed wall-clock duration in seconds spent staging media before replay."""

    media_scratch_dir: str | None
    """Scratch directory holding pre-staged PNG/MP4 files, when used."""

    media_base_url: str | None
    """HTTP base URL used for URL-mode media parts, or ``None`` for inline data URIs."""


@dataclass(frozen=True, slots=True)
class _DagRenderSpan:
    kind: Literal["text", "media"]
    span_id: str
    tokens: int
    media_index: int | None
    role: str


def _clone_media_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for part in parts:
        item = dict(part)
        if isinstance(item.get("image_url"), dict):
            item["image_url"] = dict(item["image_url"])
        if isinstance(item.get("video_url"), dict):
            item["video_url"] = dict(item["video_url"])
        cloned.append(item)
    return cloned


def _build_wire_parts_from_uris(
    uris: tuple[str, ...],
    segment: DagMediaSegment,
    *,
    video_wire_mode: Literal["video_url", "image_frames"],
    emit_media_uuid: bool,
) -> list[dict[str, Any]]:
    base_uuid = segment.uuid or (segment.media_hash if emit_media_uuid else None)
    if segment.kind == "image":
        part: dict[str, Any] = {"type": "image_url", "image_url": {"url": uris[0]}}
        if base_uuid is not None:
            part["uuid"] = base_uuid
        return [part]
    if video_wire_mode == "image_frames":
        parts: list[dict[str, Any]] = []
        for idx, uri in enumerate(uris):
            frame_part: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": uri},
            }
            if base_uuid is not None:
                frame_part["uuid"] = f"{base_uuid}:f{idx:03d}"
            parts.append(frame_part)
        return parts
    video_part: dict[str, Any] = {"type": "video_url", "video_url": {"url": uris[0]}}
    if base_uuid is not None:
        video_part["uuid"] = base_uuid
    return [video_part]


def _file_stat_token(
    path: Path, sig: tuple[Any, ...]
) -> tuple[str, int, int, tuple[Any, ...]] | None:
    if not path.is_file():
        return None
    st = path.stat()
    return (str(path.resolve()), st.st_mtime_ns, st.st_size, sig)


class DagMultimodalMaterializer:
    """Content-addressed synthesizer and untimed stager for ``dag_jsonl`` turns."""

    def __init__(
        self,
        *,
        video_wire_mode: Literal["video_url", "image_frames"] = "video_url",
        media_scratch_dir: Path | str | None = None,
        media_base_url: str | None = None,
        emit_media_uuid: bool = False,
        tokenizer: Any | None = None,
        ffmpeg_bin: str | None = None,
        max_cache_items: int = 4096,
    ) -> None:
        self.video_wire_mode = video_wire_mode
        self.media_scratch_dir = (
            Path(media_scratch_dir) if media_scratch_dir is not None else None
        )
        if self.media_scratch_dir is not None:
            self.media_scratch_dir.mkdir(parents=True, exist_ok=True)
        self.media_base_url = media_base_url.rstrip("/") if media_base_url else None
        self.emit_media_uuid = emit_media_uuid
        self.tokenizer = tokenizer
        self.ffmpeg_bin = ffmpeg_bin
        self._max_cache = max(64, max_cache_items)
        self._media_cache: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = (
            OrderedDict()
        )
        self._staged_wire_parts: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._staged_uris: dict[tuple[Any, ...], tuple[str, ...]] = {}
        self._verified_files: set[tuple[str, int, int, tuple[Any, ...]]] = set()
        self._text_cache: OrderedDict[tuple[str, int], str] = OrderedDict()
        self._staged_text: dict[tuple[str, int], str] = {}
        self._frozen_for_timed_replay = False
        self.encode_calls_during_timed_replay = 0

    @property
    def is_frozen_for_timed_replay(self) -> bool:
        """Whether uncached media synthesis is currently forbidden."""
        return self._frozen_for_timed_replay

    def set_media_base_url(self, base_url: str | None) -> None:
        """Update the HTTP base URL for staged media and invalidate cached wire dicts."""
        self.media_base_url = base_url.rstrip("/") if base_url else None
        self._media_cache.clear()
        self._staged_wire_parts.clear()
        self._staged_uris.clear()

    def freeze_for_timed_replay(self, frozen: bool = True) -> None:
        """Forbid any uncached media synthesis while timed benchmarking is active."""
        self._frozen_for_timed_replay = frozen

    def _guard_not_frozen(self, context: str) -> None:
        if self._frozen_for_timed_replay:
            self.encode_calls_during_timed_replay += 1
            raise RuntimeError(
                f"Media synthesis invoked during frozen timed replay ({context}); "
                "all unique media_hash entries must be pre-staged prior to replay."
            )

    def render_span_text(self, span_id: str, num_tokens: int) -> str:
        """Synthesize deterministic text calibrated to ``num_tokens`` for ``span_id``."""
        target = max(1, int(num_tokens))
        cache_key = (span_id, target)
        if cache_key in self._staged_text:
            return self._staged_text[cache_key]
        if cache_key in self._text_cache:
            self._text_cache.move_to_end(cache_key)
            return self._text_cache[cache_key]

        rng = random.Random(_seed64("span_tok", span_id))
        words = [rng.choice(SINGLE_TOKEN_NEUTRAL_WORDS) for _ in range(target)]
        text = " " + " ".join(words)

        if self.tokenizer is not None:
            text = self._calibrate_with_tokenizer(
                span_id, text, words, target=target, rng=rng
            )

        self._text_cache[cache_key] = text
        if len(self._text_cache) > self._max_cache * 4:
            self._text_cache.popitem(last=False)
        return text

    def _resolve_encode_fn(self) -> Any | None:
        encode_fn = getattr(self.tokenizer, "encode", None)
        if encode_fn is None or hasattr(self.tokenizer, "_encode_args"):
            return encode_fn
        try:
            params = inspect.signature(encode_fn).parameters
        except (TypeError, ValueError):
            return encode_fn
        if "add_special_tokens" in params:
            return functools.partial(encode_fn, add_special_tokens=False)
        return encode_fn

    @staticmethod
    def _count_encoded_tokens(encode_fn: Any, candidate: str) -> int:
        encoded = encode_fn(candidate)
        ids = encoded.ids if hasattr(encoded, "ids") else encoded
        return len(ids)

    def _fine_tune_calibrated_text(
        self,
        encode_fn: Any,
        word_pool: list[str],
        *,
        lo_cand: str | None,
        lo_cnt: int,
        target: int,
    ) -> str | None:
        if lo_cand is None:
            for short_cand in (word_pool[0], "a", "x", "1"):
                if self._count_encoded_tokens(encode_fn, short_cand) == target:
                    return short_cand
            return None
        gap = target - lo_cnt
        if 0 < gap <= 16:
            for pad in range(1, gap + 1):
                padded = lo_cand + (" a" * pad)
                if self._count_encoded_tokens(encode_fn, padded) == target:
                    return padded
        return None

    def _calibrate_with_tokenizer(
        self,
        span_id: str,
        text: str,
        words: list[str],
        *,
        target: int,
        rng: random.Random,
    ) -> str:
        encode_fn = self._resolve_encode_fn()
        if encode_fn is None:
            return text
        actual = self._count_encoded_tokens(encode_fn, text)
        if actual == target:
            return text

        word_pool = list(words)
        max_words = min(4_194_304, max(target * 8, target + 64))
        best_count = actual

        def _probe_words(k: int) -> tuple[str, int]:
            nonlocal best_count
            while len(word_pool) < k:
                word_pool.append(rng.choice(SINGLE_TOKEN_NEUTRAL_WORDS))
            cand = " " + " ".join(word_pool[:k])
            cnt = self._count_encoded_tokens(encode_fn, cand)
            if abs(cnt - target) < abs(best_count - target):
                best_count = cnt
            return cand, cnt

        lo_k, lo_cand, lo_cnt, hi_k = (
            self._grow_word_budget(
                _probe_words,
                len(word_pool),
                text,
                actual,
                target=target,
                max_words=max_words,
            )
            if actual < target
            else self._shrink_word_budget(
                _probe_words, len(word_pool), actual, target=target
            )
        )
        if lo_cnt == target and lo_cand is not None:
            return lo_cand

        lo_cand, lo_cnt, exact = self._binary_search_word_budget(
            _probe_words,
            lo_k,
            lo_cand,
            lo_cnt,
            hi_k=hi_k,
            target=target,
        )
        if exact is not None:
            return exact

        tuned = self._fine_tune_calibrated_text(
            encode_fn, word_pool, lo_cand=lo_cand, lo_cnt=lo_cnt, target=target
        )
        if tuned is not None:
            return tuned
        raise ValueError(
            f"Tokenizer could not calibrate span '{span_id}' to {target} tokens "
            f"(closest={best_count} tokens)"
        )

    @staticmethod
    def _binary_search_word_budget(
        probe_fn: Any,
        lo_k: int | None,
        lo_cand: str | None,
        lo_cnt: int,
        *,
        hi_k: int | None,
        target: int,
    ) -> tuple[str | None, int, str | None]:
        if lo_k is None or hi_k is None:
            return lo_cand, lo_cnt, None
        while hi_k - lo_k > 1:
            mid = (lo_k + hi_k) // 2
            cand, cnt = probe_fn(mid)
            if cnt == target:
                return lo_cand, lo_cnt, cand
            if cnt < target:
                lo_k, lo_cand, lo_cnt = mid, cand, cnt
            else:
                hi_k = mid
        return lo_cand, lo_cnt, None

    @staticmethod
    def _grow_word_budget(
        probe_fn: Any,
        init_k: int,
        init_cand: str,
        init_cnt: int,
        *,
        target: int,
        max_words: int,
    ) -> tuple[int | None, str | None, int, int | None]:
        k, cnt = init_k, init_cnt
        lo_k: int | None = k
        lo_cand: str | None = init_cand
        lo_cnt = cnt
        for _ in range(20):
            if k >= max_words:
                break
            est = int(math.ceil(k * target / max(1, cnt)))
            next_k = min(max_words, max(k + 1, est if est > k else k * 2))
            cand, cnt = probe_fn(next_k)
            if cnt == target:
                return next_k, cand, cnt, next_k
            if cnt > target:
                return lo_k, lo_cand, lo_cnt, next_k
            lo_k, lo_cand, lo_cnt, k = next_k, cand, cnt, next_k
        return lo_k, lo_cand, lo_cnt, None

    @staticmethod
    def _shrink_word_budget(
        probe_fn: Any,
        init_k: int,
        init_cnt: int,
        *,
        target: int,
    ) -> tuple[int | None, str | None, int, int | None]:
        k, cnt = init_k, init_cnt
        hi_k: int | None = k
        for _ in range(20):
            if k <= 1:
                break
            est = (k * target) // max(1, cnt)
            next_k = max(1, min(k - 1, est if est < k else k // 2))
            cand, cnt = probe_fn(next_k)
            if cnt == target:
                return next_k, cand, cnt, next_k
            if cnt < target:
                return next_k, cand, cnt, hi_k
            hi_k, k = next_k, next_k
        return None, None, 0, hi_k

    def stage_media_segment(self, segment: DagMediaSegment) -> list[Path]:
        """Materialize and decode-verify ``segment`` on disk in ``media_scratch_dir``."""
        self._guard_not_frozen(segment.media_hash)
        if self.media_scratch_dir is None:
            raise RuntimeError(
                "stage_media_segment requires media_scratch_dir to be configured"
            )
        if segment.kind == "image":
            out_path = self.media_scratch_dir / staged_media_filename(segment)
            self._ensure_verified_png(
                out_path, segment.width, segment.height, segment.media_hash
            )
            return [out_path]

        if self.video_wire_mode == "image_frames":
            written: list[Path] = []
            for idx in range(segment.num_frames):
                fp = self.media_scratch_dir / staged_frame_filename(segment, idx)
                self._ensure_verified_png(
                    fp, segment.width, segment.height, f"{segment.media_hash}:f{idx}"
                )
                written.append(fp)
            return written

        mp4_path = self.media_scratch_dir / staged_media_filename(segment)
        self._ensure_verified_mp4(mp4_path, segment)
        return [mp4_path]

    def _ensure_verified_png(
        self, out_path: Path, width: int, height: int, seed_key: str
    ) -> None:
        sig = ("png", width, height)
        tok = _file_stat_token(out_path, sig)
        if tok is not None and tok in self._verified_files:
            return
        self._write_or_verify_png(out_path, width, height, seed_key)
        new_tok = _file_stat_token(out_path, sig)
        if new_tok is not None:
            self._verified_files.add(new_tok)

    def _ensure_verified_mp4(self, mp4_path: Path, segment: DagMediaSegment) -> Path:
        eff_fps = _effective_video_fps(segment.fps)
        sig = ("mp4", segment.width, segment.height, eff_fps, segment.num_frames)
        tok = _file_stat_token(mp4_path, sig)
        if tok is not None and tok in self._verified_files:
            return mp4_path
        written = write_deterministic_h264_mp4(
            mp4_path, segment, ffmpeg_bin=self.ffmpeg_bin
        )
        new_tok = _file_stat_token(written, sig)
        if new_tok is not None:
            self._verified_files.add(new_tok)
        return written

    @staticmethod
    def _write_or_verify_png(
        out_path: Path, width: int, height: int, seed_key: str
    ) -> None:
        if not out_path.is_file():
            raw = synth_deterministic_png_bytes(width, height, seed_key)
            verify_png_bytes(raw, expected_width=width, expected_height=height)
            tmp = out_path.with_name(
                f".{out_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp.write_bytes(raw)
            os.replace(tmp, out_path)
            return
        verify_png_bytes(
            out_path.read_bytes(), expected_width=width, expected_height=height
        )

    def _uri_cache_key(self, segment: DagMediaSegment) -> tuple[Any, ...]:
        return (
            *segment.cache_key,
            self.video_wire_mode,
            self.media_base_url or "inline",
        )

    def _wire_cache_key(self, segment: DagMediaSegment) -> tuple[Any, ...]:
        return (
            *segment.cache_key,
            segment.uuid,
            self.emit_media_uuid,
            self.video_wire_mode,
            self.media_base_url or "inline",
        )

    def _put_media_cache(
        self, wire_key: tuple[Any, ...], parts: list[dict[str, Any]]
    ) -> None:
        self._media_cache[wire_key] = parts
        self._media_cache.move_to_end(wire_key)
        if len(self._media_cache) > self._max_cache:
            self._media_cache.popitem(last=False)

    def render_media_parts(self, segment: DagMediaSegment) -> list[dict[str, Any]]:
        """Render OpenAI ``image_url`` / ``video_url`` content parts for ``segment``."""
        wire_key = self._wire_cache_key(segment)
        if wire_key in self._staged_wire_parts:
            return _clone_media_parts(self._staged_wire_parts[wire_key])
        if wire_key in self._media_cache:
            self._media_cache.move_to_end(wire_key)
            return _clone_media_parts(self._media_cache[wire_key])

        uri_key = self._uri_cache_key(segment)
        if uri_key in self._staged_uris:
            parts = _build_wire_parts_from_uris(
                self._staged_uris[uri_key],
                segment,
                video_wire_mode=self.video_wire_mode,
                emit_media_uuid=self.emit_media_uuid,
            )
            self._put_media_cache(wire_key, parts)
            return _clone_media_parts(parts)

        self._guard_not_frozen(segment.media_hash)
        uris = self._compute_segment_uris(segment)
        parts = _build_wire_parts_from_uris(
            uris,
            segment,
            video_wire_mode=self.video_wire_mode,
            emit_media_uuid=self.emit_media_uuid,
        )
        self._put_media_cache(wire_key, parts)
        return _clone_media_parts(parts)

    def _compute_segment_uris(self, segment: DagMediaSegment) -> tuple[str, ...]:
        if segment.kind == "image":
            return (
                self._resolve_image_uri(
                    segment,
                    seed_key=segment.media_hash,
                    filename=staged_media_filename(segment),
                ),
            )
        if self.video_wire_mode == "image_frames":
            return tuple(
                self._resolve_image_uri(
                    segment,
                    seed_key=f"{segment.media_hash}:f{idx}",
                    filename=staged_frame_filename(segment, idx),
                )
                for idx in range(segment.num_frames)
            )
        return (self._resolve_video_mp4_uri(segment),)

    def _resolve_image_uri(
        self, segment: DagMediaSegment, *, seed_key: str, filename: str
    ) -> str:
        if self.media_scratch_dir is not None:
            out_path = self.media_scratch_dir / filename
            self._ensure_verified_png(out_path, segment.width, segment.height, seed_key)
            if self.media_base_url is not None:
                return f"{self.media_base_url}/{filename}"
            b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        raw = synth_deterministic_png_bytes(segment.width, segment.height, seed_key)
        verify_png_bytes(
            raw, expected_width=segment.width, expected_height=segment.height
        )
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    def _resolve_video_mp4_uri(self, segment: DagMediaSegment) -> str:
        fname = staged_media_filename(segment)
        if self.media_scratch_dir is not None:
            mp4_path = self._ensure_verified_mp4(
                self.media_scratch_dir / fname, segment
            )
            if self.media_base_url is not None:
                return f"{self.media_base_url}/{mp4_path.name}"
            b64 = base64.b64encode(mp4_path.read_bytes()).decode("ascii")
            return f"data:video/mp4;base64,{b64}"
        with tempfile.TemporaryDirectory(prefix="aiperf_dag_mp4_") as td:
            mp4_path = write_deterministic_h264_mp4(
                Path(td) / fname, segment, ffmpeg_bin=self.ffmpeg_bin
            )
            b64 = base64.b64encode(mp4_path.read_bytes()).decode("ascii")
            return f"data:video/mp4;base64,{b64}"

    def stage_turns(self, turns: Iterable[DagTurn]) -> MediaStagingSummary:
        """Untimed pre-generation pass across all ``turns`` in a DAG dataset."""
        t0 = time.perf_counter()
        unique_segments: OrderedDict[tuple[Any, ...], DagMediaSegment] = OrderedDict()
        wire_variants: OrderedDict[tuple[Any, ...], DagMediaSegment] = OrderedDict()
        total_refs = 0
        for turn in turns:
            for span in turn.prompt_spans:
                if span.kind == "text":
                    rendered = self.render_span_text(span.span_id, span.tokens)
                    self._staged_text[(span.span_id, max(1, int(span.tokens)))] = (
                        rendered
                    )
            for seg in turn.media_segments:
                total_refs += 1
                unique_segments.setdefault(seg.cache_key, seg)
                wire_variants.setdefault((*seg.cache_key, seg.uuid), seg)

        unique_images = 0
        unique_videos = 0
        staged_bytes = 0
        for seg in unique_segments.values():
            if self.media_scratch_dir is not None:
                paths = self.stage_media_segment(seg)
                staged_bytes += sum(p.stat().st_size for p in paths if p.is_file())
            uri_key = self._uri_cache_key(seg)
            if uri_key not in self._staged_uris:
                self._staged_uris[uri_key] = self._compute_segment_uris(seg)
            if seg.kind == "image":
                unique_images += 1
            else:
                unique_videos += 1

        for variant in wire_variants.values():
            uris = self._staged_uris[self._uri_cache_key(variant)]
            parts = _build_wire_parts_from_uris(
                uris,
                variant,
                video_wire_mode=self.video_wire_mode,
                emit_media_uuid=self.emit_media_uuid,
            )
            wire_key = self._wire_cache_key(variant)
            self._staged_wire_parts[wire_key] = parts
            self._put_media_cache(wire_key, parts)

        return MediaStagingSummary(
            unique_images=unique_images,
            unique_videos=unique_videos,
            total_segment_references=total_refs,
            deduplicated_references=max(0, total_refs - len(unique_segments)),
            staged_bytes=staged_bytes,
            staging_wall_s=round(time.perf_counter() - t0, 4),
            media_scratch_dir=(
                str(self.media_scratch_dir)
                if self.media_scratch_dir is not None
                else None
            ),
            media_base_url=self.media_base_url,
        )

    async def async_stage_turns(self, turns: Iterable[DagTurn]) -> MediaStagingSummary:
        """Async wrapper running ``stage_turns`` off the asyncio event loop."""
        return await asyncio.to_thread(self.stage_turns, list(turns))

    def materialize_turn_messages(
        self,
        turn: DagTurn,
        *,
        fallback_key: str = "turn",
    ) -> list[dict[str, Any]]:
        """Materialize OpenAI-compatible ``messages`` for ``turn``."""
        if turn.messages is not None:
            if not turn.media_segments:
                return list(turn.messages)
            return self._resolve_authored_messages(turn.messages, turn.media_segments)
        spans = self._effective_spans(turn, fallback_key=fallback_key)
        return self._messages_from_spans(spans, turn.media_segments)

    def _resolve_authored_messages(
        self,
        messages: list[dict[str, Any]],
        segments: list[DagMediaSegment],
    ) -> list[dict[str, Any]]:
        by_hash: dict[str, DagMediaSegment] = {}
        for seg in segments:
            by_hash.setdefault(seg.media_hash, seg)
        resolved: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                resolved.append(dict(msg))
                continue
            new_parts: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "media_ref":
                    seg = (
                        segments[part["media_index"]]
                        if part.get("media_index") is not None
                        else by_hash[str(part["media_hash"])]
                    )
                    new_parts.extend(self.render_media_parts(seg))
                else:
                    new_parts.append(part)
            resolved.append({**msg, "content": new_parts})
        return resolved

    @staticmethod
    def _effective_spans(turn: DagTurn, *, fallback_key: str) -> list[_DagRenderSpan]:
        if turn.prompt_spans:
            return [
                _DagRenderSpan(
                    kind=s.kind,
                    span_id=s.span_id,
                    tokens=s.tokens,
                    media_index=s.media_index,
                    role=s.role,
                )
                for s in turn.prompt_spans
            ]
        spans: list[_DagRenderSpan] = []
        if turn.in_text_tokens and turn.in_text_tokens > 0:
            spans.append(
                _DagRenderSpan(
                    kind="text",
                    span_id=f"{fallback_key}:text",
                    tokens=turn.in_text_tokens,
                    media_index=None,
                    role="user",
                )
            )
        for idx, seg in enumerate(turn.media_segments):
            spans.append(
                _DagRenderSpan(
                    kind="media",
                    span_id=seg.media_hash,
                    tokens=0,
                    media_index=idx,
                    role="user",
                )
            )
        return spans

    def _messages_from_spans(
        self,
        spans: list[_DagRenderSpan],
        segments: list[DagMediaSegment],
    ) -> list[dict[str, Any]]:
        by_hash: dict[str, int] = {}
        for idx, seg in enumerate(segments):
            by_hash.setdefault(seg.media_hash, idx)
        messages: list[dict[str, Any]] = []
        fallback_media_idx = 0
        for span in spans:
            if span.kind == "text":
                parts = [
                    {
                        "type": "text",
                        "text": self.render_span_text(span.span_id, span.tokens),
                    }
                ]
            else:
                if span.media_index is not None:
                    idx = span.media_index
                elif span.span_id in by_hash:
                    idx = by_hash[span.span_id]
                else:
                    idx = fallback_media_idx
                fallback_media_idx = idx + 1
                parts = self.render_media_parts(segments[idx])
            if messages and messages[-1]["role"] == span.role:
                messages[-1]["content"].extend(parts)
            else:
                messages.append({"role": span.role, "content": list(parts)})
        return messages


def _has_multimodal_or_structural_turns(
    conversations: Iterable[Any],
) -> bool:
    for conv in conversations:
        for turn in conv.turns:
            if (
                turn.media_segments
                or turn.prompt_spans
                or turn.in_text_tokens is not None
            ):
                return True
    return False


def prepare_materializer_for_conversations(
    conversations: list[Any],
    *,
    materializer: DagMultimodalMaterializer | None = None,
    media_scratch_dir: Path | None = None,
    media_base_url: str | None = None,
    video_wire_mode: Literal["video_url", "image_frames"] = "video_url",
    emit_media_uuid: bool = False,
    tokenizer: Any | None = None,
) -> tuple[DagMultimodalMaterializer | None, MediaStagingSummary | None]:
    """Run untimed pre-staging across ``conversations`` when multimodal turns exist."""
    if not _has_multimodal_or_structural_turns(conversations):
        return None, None

    all_turns = [turn for conv in conversations for turn in conv.turns]
    if materializer is not None:
        was_frozen = materializer.is_frozen_for_timed_replay
        materializer.freeze_for_timed_replay(False)
        try:
            summary = materializer.stage_turns(all_turns)
        except Exception:
            materializer.freeze_for_timed_replay(was_frozen)
            raise
        materializer.freeze_for_timed_replay(True)
        return materializer, summary

    if media_base_url is not None and media_scratch_dir is None:
        raise ValueError(
            "media_base_url requires media_scratch_dir so staged media files "
            "persist on disk for HTTP serving"
        )

    if media_scratch_dir is not None:
        mat = DagMultimodalMaterializer(
            video_wire_mode=video_wire_mode,
            media_scratch_dir=media_scratch_dir,
            media_base_url=media_base_url,
            emit_media_uuid=emit_media_uuid,
            tokenizer=tokenizer,
        )
        summary = mat.stage_turns(all_turns)
        mat.freeze_for_timed_replay(True)
        return mat, summary

    with tempfile.TemporaryDirectory(prefix="aiperf_dag_media_") as temp_dir:
        mat = DagMultimodalMaterializer(
            video_wire_mode=video_wire_mode,
            media_scratch_dir=Path(temp_dir),
            media_base_url=None,
            emit_media_uuid=emit_media_uuid,
            tokenizer=tokenizer,
        )
        summary = mat.stage_turns(all_turns)
        mat.media_scratch_dir = None
        mat.freeze_for_timed_replay(True)
        return mat, summary


async def async_prepare_materializer_for_conversations(
    conversations: list[Any],
    *,
    materializer: DagMultimodalMaterializer | None = None,
    media_scratch_dir: Path | None = None,
    media_base_url: str | None = None,
    video_wire_mode: Literal["video_url", "image_frames"] = "video_url",
    emit_media_uuid: bool = False,
    tokenizer: Any | None = None,
) -> tuple[DagMultimodalMaterializer | None, MediaStagingSummary | None]:
    """Async wrapper running ``prepare_materializer_for_conversations`` off the event loop."""
    return await asyncio.to_thread(
        prepare_materializer_for_conversations,
        conversations,
        materializer=materializer,
        media_scratch_dir=media_scratch_dir,
        media_base_url=media_base_url,
        video_wire_mode=video_wire_mode,
        emit_media_uuid=emit_media_uuid,
        tokenizer=tokenizer,
    )
