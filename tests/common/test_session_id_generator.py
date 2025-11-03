# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SessionIDGenerator class."""

from aiperf.common.session_id_generator import SessionIDGenerator


class TestSessionIDGenerator:
    """Tests for the SessionIDGenerator class."""

    def test_deterministic_mode_sequential_ids(self):
        """Test that providing a seed generates sequential IDs."""
        gen = SessionIDGenerator(seed=42)

        id1 = gen.next()
        id2 = gen.next()
        id3 = gen.next()

        assert id1 == "session_000000"
        assert id2 == "session_000001"
        assert id3 == "session_000002"

    def test_deterministic_mode_custom_prefix(self):
        """Test that custom prefix works with seed."""
        gen = SessionIDGenerator(seed=42, prefix="conv")

        id1 = gen.next()
        id2 = gen.next()

        assert id1 == "conv_000000"
        assert id2 == "conv_000001"

    def test_non_deterministic_mode_uuid_format(self):
        """Test that no seed generates UUIDs."""
        gen = SessionIDGenerator(seed=None)

        id1 = gen.next()
        id2 = gen.next()

        # UUIDs should be different
        assert id1 != id2
        # UUIDs should contain hyphens (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        assert "-" in id1
        assert "-" in id2
        # UUIDs should be 36 characters long
        assert len(id1) == 36
        assert len(id2) == 36

    def test_reset_counter(self):
        """Test that reset() resets the counter with seed."""
        gen = SessionIDGenerator(seed=42)

        id1 = gen.next()
        id2 = gen.next()
        gen.reset()
        id3 = gen.next()

        assert id1 == "session_000000"
        assert id2 == "session_000001"
        assert id3 == "session_000000"  # Reset back to 0

    def test_get_counter(self):
        """Test that get_counter() returns the current counter value."""
        gen = SessionIDGenerator(seed=42)

        assert gen.get_counter() == 0
        gen.next()
        assert gen.get_counter() == 1
        gen.next()
        assert gen.get_counter() == 2
        gen.reset()
        assert gen.get_counter() == 0

    def test_seed_value_doesnt_affect_sequence(self):
        """Test that different seeds produce same sequence (only affects determinism flag)."""
        gen1 = SessionIDGenerator(seed=42)
        gen2 = SessionIDGenerator(seed=99)

        # Both should produce same sequence starting from 0
        assert gen1.next() == "session_000000"
        assert gen2.next() == "session_000000"
        assert gen1.next() == "session_000001"
        assert gen2.next() == "session_000001"


class TestReproducibility:
    """Tests for reproducibility with SessionIDGenerator."""

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same IDs."""
        # First run
        gen1 = SessionIDGenerator(seed=42, prefix="run")
        ids_run1 = [gen1.next() for _ in range(5)]

        # Second run
        gen2 = SessionIDGenerator(seed=42, prefix="run")
        ids_run2 = [gen2.next() for _ in range(5)]

        # Should be identical
        assert ids_run1 == ids_run2
        assert ids_run1 == [
            "run_000000",
            "run_000001",
            "run_000002",
            "run_000003",
            "run_000004",
        ]

    def test_non_deterministic_mode_produces_different_ids(self):
        """Test that no seed produces different IDs on each run."""
        # First run
        gen1 = SessionIDGenerator(seed=None)
        ids_run1 = [gen1.next() for _ in range(5)]

        # Second run
        gen2 = SessionIDGenerator(seed=None)
        ids_run2 = [gen2.next() for _ in range(5)]

        # Should be different (with extremely high probability)
        assert ids_run1 != ids_run2

    def test_deterministic_mode_uniqueness_within_run(self):
        """Test that all IDs within a run are unique."""
        gen = SessionIDGenerator(seed=42)
        ids = [gen.next() for _ in range(100)]

        # All IDs should be unique
        assert len(ids) == len(set(ids))

    def test_large_counter_values(self):
        """Test that large counter values work correctly."""
        gen = SessionIDGenerator(seed=42)

        # Generate a large number of IDs
        for _ in range(1000):
            gen.next()

        id1000 = gen.next()
        assert id1000 == "session_001000"
