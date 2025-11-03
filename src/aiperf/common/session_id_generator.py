# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Session ID Generator for AIPerf conversations.

Provides deterministic session ID generation for reproducible benchmarking.
When initialized with a seed, generates sequential IDs like "session_000000",
"session_000001", etc. Without a seed, generates random UUIDs.

Usage:
    from aiperf.common.session_id_generator import SessionIDGenerator

    # Each component creates its own generator
    generator = SessionIDGenerator(seed=user_config.input.random_seed)

    # Get next ID
    session_id = generator.next()  # "session_000000" if seed provided, UUID if not

Each loader/composer creates its own generator - no need to pass them around.
"""

import uuid


class SessionIDGenerator:
    """Generates unique session IDs for conversations.

    Supports two modes based on seed:
    - With seed (int): Sequential counter-based IDs (e.g., "session_000000")
    - Without seed (None): Random UUIDs (e.g., "a1b2c3d4-...")

    Attributes:
        seed: Random seed value. If None, uses UUIDs. If int, uses sequential IDs.
        prefix: String prefix for deterministic IDs (default: "session").
    """

    def __init__(self, seed: int | None = None, prefix: str = "session"):
        """Initialize the session ID generator.

        Args:
            seed: Random seed for reproducibility. If None, uses UUIDs.
                  If provided, uses sequential counter-based IDs.
            prefix: Prefix for deterministic IDs (e.g., "session" -> "session_000000").

        Example:
            # Deterministic (with seed)
            gen = SessionIDGenerator(seed=42)
            id1 = gen.next()  # "session_000000"

            # Non-deterministic (no seed)
            gen = SessionIDGenerator(seed=None)
            id1 = gen.next()  # "a1b2c3d4-5678-..."
        """
        self.seed = seed
        self.prefix = prefix
        self._counter = 0

    def next(self) -> str:
        """Generate and return the next session ID.

        Returns:
            A unique session ID string. Format depends on whether seed was provided:
            - With seed: "{prefix}_{counter:06d}" (e.g., "session_000000")
            - Without seed: UUID4 string (e.g., "a1b2c3d4-...")
        """
        if self.seed is not None:
            session_id = f"{self.prefix}_{self._counter:06d}"
            self._counter += 1
            return session_id
        else:
            return str(uuid.uuid4())

    def reset(self):
        """Reset the internal counter to 0.

        Only affects deterministic mode (when seed is provided).
        Used primarily for testing.
        """
        self._counter = 0

    def get_counter(self) -> int:
        """Get the current counter value.

        Returns:
            The number of session IDs generated (only meaningful when seed is provided).
        """
        return self._counter
