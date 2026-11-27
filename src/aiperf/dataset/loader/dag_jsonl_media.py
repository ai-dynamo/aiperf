# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multimodal media synthesis and untimed staging for ``dag_jsonl``.

Provides content-addressed PNG and H.264 MP4 generation keyed by
``(kind, media_hash, width, height, fps, num_frames)``, untimed pre-staging
across ``FORK`` and ``SPAWN`` branches, optional local HTTP static serving,
mixed text+media block-hash computation, and zero-leakage trace validation.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import http.server
import io
import math
import os
import random
import shutil
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from aiperf.dataset.loader.dag_jsonl_models import (
        DagMediaSegment,
        DagPromptSpan,
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


def _seed64(*parts: object) -> int:
    joined = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def validate_dag_zero_leakage(obj: Any, *, path: str = "$") -> None:
    """Recursively verify that a structural DAG JSON object holds zero raw data."""
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


def synthesize_deterministic_rgb_frames(
    width: int,
    height: int,
    num_frames: int,
    seed_key: str,
) -> list[np.ndarray]:
    """Generate ``num_frames`` deterministic ``(height, width, 3)`` uint8 RGB arrays."""
    if width <= 0 or height <= 0 or num_frames <= 0:
        raise ValueError(
            f"Video frame geometry must be positive, got {width}x{height} @ {num_frames}f"
        )
    rng = np.random.default_rng(_seed64("mp4_rgb", seed_key, width, height, num_frames))
    tile = 16
    gh = max(2, math.ceil(height / tile))
    gw = max(2, math.ceil(width / tile))
    base = rng.integers(16, 240, size=(gh, gw, 3), dtype=np.uint8)
    frame0 = np.repeat(np.repeat(base, tile, axis=0), tile, axis=1)[:height, :width, :]
    frames: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        shifted = np.roll(frame0, shift=(frame_idx * 16) % max(1, width), axis=1)
        if frame_idx % 2 == 1:
            shifted = np.roll(shifted, shift=(frame_idx * 8) % max(1, height), axis=0)
        frames.append(np.ascontiguousarray(shifted))
    return frames


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


def verify_mp4_decodable(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    ffmpeg_bin: str | None = None,
) -> None:
    """Decode ``path`` via ``ffmpeg`` rawvideo output to verify frame count and size."""
    if not path.is_file() or path.stat().st_size < 128:
        raise RuntimeError(f"Generated MP4 file is missing or truncated: {path}")
    ff = ffmpeg_bin or find_ffmpeg_binary()
    if ff is None:
        raise RuntimeError(
            "Cannot verify MP4 decodability because 'ffmpeg' binary was not found"
        )
    proc = subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    expected_bytes = expected_frames * expected_width * expected_height * 3
    if proc.returncode != 0 or len(proc.stdout) != expected_bytes:
        stderr_snippet = proc.stderr.decode("utf-8", "ignore")[:200]
        raise RuntimeError(
            f"FFmpeg decode verification failed for {path} (rc={proc.returncode}, "
            f"bytes={len(proc.stdout)} vs expected={expected_bytes}): {stderr_snippet}"
        )


def _encode_mp4_with_codec(
    ff: str,
    codec: str,
    raw_payload: bytes,
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> subprocess.CompletedProcess[bytes]:
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
    return subprocess.run(cmd, input=raw_payload, capture_output=True, check=False)


def write_deterministic_h264_mp4(
    path: Path,
    segment: DagMediaSegment,
    *,
    ffmpeg_bin: str | None = None,
) -> Path:
    """Encode a deterministic ``yuv420p`` MP4 for ``segment`` and verify decodability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h, n_frames = segment.width, segment.height, segment.num_frames
    if w % 2 != 0 or h % 2 != 0:
        raise ValueError(f"yuv420p MP4 requires even dimensions, got {w}x{h}")
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
            ffmpeg_bin=ff,
        )
        return path

    fps = max(1.0, float(segment.fps or 24.0))
    frames = synthesize_deterministic_rgb_frames(w, h, n_frames, segment.media_hash)
    raw_payload = b"".join(frame.tobytes() for frame in frames)
    tmp_path = path.with_name(
        f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
    )
    errors: list[str] = []
    for codec in ("libx264", "mpeg4"):
        proc = _encode_mp4_with_codec(
            ff, codec, raw_payload, tmp_path, width=w, height=h, fps=fps
        )
        if proc.returncode == 0 and tmp_path.is_file():
            try:
                verify_mp4_decodable(
                    tmp_path,
                    expected_frames=n_frames,
                    expected_width=w,
                    expected_height=h,
                    ffmpeg_bin=ff,
                )
                os.replace(tmp_path, path)
                return path
            except RuntimeError as exc:
                errors.append(f"{codec} verify failed: {exc}")
        else:
            stderr_msg = proc.stderr.decode("utf-8", "ignore")[:160]
            errors.append(f"{codec} rc={proc.returncode}: {stderr_msg}")
        tmp_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Failed to encode decodable MP4 for '{segment.media_hash}' ({w}x{h}, {n_frames}f): "
        f"{'; '.join(errors)}"
    )


def staged_media_filename(segment: DagMediaSegment) -> str:
    """Return canonical content-addressed filename for a staged ``DagMediaSegment``."""
    if segment.kind == "image":
        return f"{segment.media_hash}_{segment.width}x{segment.height}.png"
    fps_tag = int(round(float(segment.fps))) if segment.fps > 0 else 24
    return (
        f"{segment.media_hash}_{segment.width}x{segment.height}_"
        f"{segment.num_frames}f_{fps_tag}fps.mp4"
    )


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

    @property
    def base_url(self) -> str:
        """Base HTTP URL serving staged media files (e.g. ``http://127.0.0.1:51234``)."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> LocalMediaHttpServer:
        """Start serving pre-staged files on the background daemon thread."""
        self._thread.start()
        return self

    def close(self) -> None:
        """Shut down the HTTP server and join its background thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
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
    """Number of repeated ``media_hash`` references served from the pre-staged cache."""

    staged_bytes: int
    """Total on-disk bytes written across unique pre-staged media files."""

    staging_wall_s: float
    """Untimed wall-clock duration in seconds spent staging media before replay."""

    media_scratch_dir: str | None
    """Scratch directory holding pre-staged PNG/MP4 files, when used."""

    media_base_url: str | None
    """HTTP base URL used for URL-mode media parts, or ``None`` for inline data URIs."""


def _attach_part_uuid(
    part: dict[str, Any], segment: DagMediaSegment, *, emit_media_uuid: bool
) -> dict[str, Any]:
    part_uuid = segment.uuid or (segment.media_hash if emit_media_uuid else None)
    if part_uuid is None:
        return part
    return {**part, "uuid": part_uuid}


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
        self._text_cache: OrderedDict[tuple[str, int], str] = OrderedDict()
        self._frozen_for_timed_replay = False
        self.encode_calls_during_timed_replay = 0

    def set_media_base_url(self, base_url: str | None) -> None:
        """Update the HTTP base URL for staged media and invalidate cached wire dicts."""
        self.media_base_url = base_url.rstrip("/") if base_url else None
        self._media_cache.clear()

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
        if cache_key in self._text_cache:
            self._text_cache.move_to_end(cache_key)
            return self._text_cache[cache_key]

        rng = random.Random(_seed64("span_tok", span_id))
        words = [rng.choice(SINGLE_TOKEN_NEUTRAL_WORDS) for _ in range(target)]
        text = " " + " ".join(words)

        if self.tokenizer is not None:
            text = self._calibrate_with_tokenizer(text, words, target, rng)

        self._text_cache[cache_key] = text
        if len(self._text_cache) > self._max_cache * 4:
            self._text_cache.popitem(last=False)
        return text

    def _calibrate_with_tokenizer(
        self,
        text: str,
        words: list[str],
        target: int,
        rng: random.Random,
    ) -> str:
        encode_fn = getattr(self.tokenizer, "encode", None)
        if encode_fn is None:
            return text
        encoded = encode_fn(text)
        ids = encoded.ids if hasattr(encoded, "ids") else encoded
        actual = len(ids)
        if actual > target:
            words = words[: max(1, len(words) - (actual - target))]
            return " " + " ".join(words)
        if actual < target:
            words.extend(
                rng.choice(SINGLE_TOKEN_NEUTRAL_WORDS) for _ in range(target - actual)
            )
            return " " + " ".join(words)
        return text

    def stage_media_segment(self, segment: DagMediaSegment) -> list[Path]:
        """Materialize and decode-verify ``segment`` on disk in ``media_scratch_dir``."""
        self._guard_not_frozen(segment.media_hash)
        if self.media_scratch_dir is None:
            raise RuntimeError(
                "stage_media_segment requires media_scratch_dir to be configured"
            )
        if segment.kind == "image":
            out_path = self.media_scratch_dir / staged_media_filename(segment)
            self._write_or_verify_png(
                out_path, segment.width, segment.height, segment.media_hash
            )
            return [out_path]

        if self.video_wire_mode == "image_frames":
            written: list[Path] = []
            for idx in range(segment.num_frames):
                frame_name = (
                    f"{segment.media_hash}_f{idx:03d}_"
                    f"{segment.width}x{segment.height}.png"
                )
                fp = self.media_scratch_dir / frame_name
                self._write_or_verify_png(
                    fp, segment.width, segment.height, f"{segment.media_hash}:f{idx}"
                )
                written.append(fp)
            return written

        mp4_path = self.media_scratch_dir / staged_media_filename(segment)
        write_deterministic_h264_mp4(mp4_path, segment, ffmpeg_bin=self.ffmpeg_bin)
        return [mp4_path]

    @staticmethod
    def _write_or_verify_png(
        out_path: Path, width: int, height: int, seed_key: str
    ) -> None:
        if not out_path.is_file():
            raw = synth_deterministic_png_bytes(width, height, seed_key)
            verify_png_bytes(raw, expected_width=width, expected_height=height)
            tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, out_path)
            return
        verify_png_bytes(
            out_path.read_bytes(), expected_width=width, expected_height=height
        )

    def render_media_parts(self, segment: DagMediaSegment) -> list[dict[str, Any]]:
        """Render OpenAI ``image_url`` / ``video_url`` content parts for ``segment``."""
        wire_key = (
            *segment.cache_key,
            segment.uuid,
            self.emit_media_uuid,
            self.video_wire_mode,
            self.media_base_url or "inline",
        )
        if wire_key in self._media_cache:
            self._media_cache.move_to_end(wire_key)
            return [dict(p) for p in self._media_cache[wire_key]]

        self._guard_not_frozen(segment.media_hash)
        parts = self._build_uncached_media_parts(segment)
        self._media_cache[wire_key] = parts
        if len(self._media_cache) > self._max_cache:
            self._media_cache.popitem(last=False)
        return [dict(p) for p in parts]

    def _build_uncached_media_parts(
        self, segment: DagMediaSegment
    ) -> list[dict[str, Any]]:
        if segment.kind == "image":
            uri = self._resolve_image_uri(
                segment,
                seed_key=segment.media_hash,
                filename=staged_media_filename(segment),
            )
            part = {"type": "image_url", "image_url": {"url": uri}}
            return [
                _attach_part_uuid(part, segment, emit_media_uuid=self.emit_media_uuid)
            ]

        if self.video_wire_mode == "image_frames":
            return self._render_video_as_frame_parts(segment)

        uri = self._resolve_video_mp4_uri(segment)
        part = {"type": "video_url", "video_url": {"url": uri}}
        return [_attach_part_uuid(part, segment, emit_media_uuid=self.emit_media_uuid)]

    def _resolve_image_uri(
        self, segment: DagMediaSegment, *, seed_key: str, filename: str
    ) -> str:
        if self.media_scratch_dir is not None:
            out_path = self.media_scratch_dir / filename
            self._write_or_verify_png(out_path, segment.width, segment.height, seed_key)
            if self.media_base_url is not None:
                return f"{self.media_base_url}/{filename}"
            b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        raw = synth_deterministic_png_bytes(segment.width, segment.height, seed_key)
        verify_png_bytes(
            raw, expected_width=segment.width, expected_height=segment.height
        )
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    def _render_video_as_frame_parts(
        self, segment: DagMediaSegment
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        base_uuid = segment.uuid or (
            segment.media_hash if self.emit_media_uuid else None
        )
        for idx in range(segment.num_frames):
            fname = (
                f"{segment.media_hash}_f{idx:03d}_{segment.width}x{segment.height}.png"
            )
            uri = self._resolve_image_uri(
                segment, seed_key=f"{segment.media_hash}:f{idx}", filename=fname
            )
            part: dict[str, Any] = {"type": "image_url", "image_url": {"url": uri}}
            if base_uuid is not None:
                part["uuid"] = f"{base_uuid}:f{idx:03d}"
            parts.append(part)
        return parts

    def _resolve_video_mp4_uri(self, segment: DagMediaSegment) -> str:
        fname = staged_media_filename(segment)
        if self.media_scratch_dir is not None:
            mp4_path = write_deterministic_h264_mp4(
                self.media_scratch_dir / fname, segment, ffmpeg_bin=self.ffmpeg_bin
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
        total_refs = 0
        for turn in turns:
            for span in turn.prompt_spans:
                if span.kind == "text":
                    self.render_span_text(span.span_id, span.tokens)
            for seg in turn.media_segments:
                total_refs += 1
                unique_segments.setdefault(seg.cache_key, seg)

        unique_images = 0
        unique_videos = 0
        staged_bytes = 0
        for seg in unique_segments.values():
            if self.media_scratch_dir is not None:
                paths = self.stage_media_segment(seg)
                staged_bytes += sum(p.stat().st_size for p in paths if p.is_file())
            self.render_media_parts(seg)
            if seg.kind == "image":
                unique_images += 1
            else:
                unique_videos += 1

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
    def _effective_spans(turn: DagTurn, *, fallback_key: str) -> list[DagPromptSpan]:
        if turn.prompt_spans:
            return list(turn.prompt_spans)
        from aiperf.dataset.loader.dag_jsonl_models import DagPromptSpan

        spans: list[DagPromptSpan] = []
        if turn.in_text_tokens and turn.in_text_tokens > 0:
            spans.append(
                DagPromptSpan(
                    kind="text",
                    span_id=f"{fallback_key}:text",
                    tokens=turn.in_text_tokens,
                    role="user",
                )
            )
        for idx, seg in enumerate(turn.media_segments):
            est_tokens = (seg.vis_tokens or 256) + seg.framing_tokens
            spans.append(
                DagPromptSpan(
                    kind="media",
                    span_id=seg.media_hash,
                    tokens=max(1, est_tokens),
                    media_index=idx,
                    role="user",
                )
            )
        return spans

    def _messages_from_spans(
        self,
        spans: list[DagPromptSpan],
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
        summary = materializer.stage_turns(all_turns)
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
