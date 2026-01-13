# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for integration test utilities."""

from tests.integration.utils import _check_mp4_fragmentation


class TestMP4FragmentationDetection:
    """Tests for MP4 fragmentation detection."""

    def test_detects_fragmented_mp4(self):
        """Test that fragmented MP4 (with moof box) is detected."""
        # Simulate MP4 header with moof (movie fragment) box
        # Format: [size: 4 bytes][type: 4 bytes][data]
        fragmented_mp4 = (
            b"\x00\x00\x00\x20"  # ftyp box size (32 bytes)
            b"ftyp"  # ftyp box type
            b"isom\x00\x00\x02\x00"  # major brand
            b"isomiso2mp41"  # compatible brands
            b"\x00\x00\x00\x08"  # moof box size (8 bytes)
            b"moof"  # moof box type (indicates fragmentation)
            + b"\x00"
            * 1000  # padding
        )
        assert _check_mp4_fragmentation(fragmented_mp4) is True

    def test_detects_non_fragmented_mp4(self):
        """Test that non-fragmented MP4 (without moof box) is detected."""
        # Simulate MP4 header without moof box
        non_fragmented_mp4 = (
            b"\x00\x00\x00\x20"  # ftyp box size
            b"ftyp"  # ftyp box type
            b"isom\x00\x00\x02\x00"  # major brand
            b"isomiso2mp41"  # compatible brands
            b"\x00\x00\x00\x08"  # moov box size
            b"moov" + b"\x00" * 1000  # moov box type (not moof)  # padding
        )
        assert _check_mp4_fragmentation(non_fragmented_mp4) is False

    def test_handles_short_file(self):
        """Test handling of very short file."""
        short_mp4 = b"ftypisom"
        assert _check_mp4_fragmentation(short_mp4) is False

    def test_handles_empty_file(self):
        """Test handling of empty file."""
        assert _check_mp4_fragmentation(b"") is False
