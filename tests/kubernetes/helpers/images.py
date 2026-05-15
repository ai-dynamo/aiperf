# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docker image management for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.cluster import _run_streaming

logger = AIPerfLogger(__name__)


@dataclass
class ImageConfig:
    """Configuration for a Docker image."""

    name: str
    """Image name without tag."""

    tag: str = "latest"
    """Image tag."""

    dockerfile: str | None = None
    """Dockerfile path relative to project root, or None for default."""

    build_context: Path | None = None
    """Build context directory, or None for project root."""

    @property
    def full_name(self) -> str:
        """Get full image name with tag."""
        return f"{self.name}:{self.tag}"


@dataclass
class ImageManager:
    """Manages Docker images for testing."""

    project_root: Path
    """Path to the project root directory."""

    images: dict[str, ImageConfig] = field(default_factory=dict)
    """Image configurations keyed by logical name."""

    _built: set[str] = field(default_factory=set, init=False)
    """Set of image keys that have been built."""

    def __post_init__(self) -> None:
        """Initialize default images."""
        if not self.images:
            self.images = {
                "aiperf": ImageConfig(
                    name="aiperf",
                    tag="local",
                    dockerfile=None,
                    build_context=self.project_root,
                ),
                "mock-server": ImageConfig(
                    name="aiperf-mock-server",
                    tag="latest",
                    dockerfile="dev/deploy/Dockerfile.mock-server",
                    build_context=self.project_root,
                ),
            }

    async def image_exists(self, image_key: str) -> bool:
        """Check if an image exists locally.

        Args:
            image_key: Key in the images dict.

        Returns:
            True if image exists.
        """
        if image_key not in self.images:
            raise ValueError(f"Unknown image: {image_key}")

        config = self.images[image_key]

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            config.full_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0

    async def build(self, image_key: str, force: bool = False) -> None:
        """Build a Docker image.

        Args:
            image_key: Key in the images dict.
            force: Rebuild even if image exists.
        """
        if image_key not in self.images:
            raise ValueError(f"Unknown image: {image_key}")

        config = self.images[image_key]

        if not force and await self.image_exists(image_key):
            logger.info(f"Image already exists: {config.full_name}")
            return

        logger.info(f"Building image: {config.full_name}")

        cmd = ["docker", "build", "-t", config.full_name]

        if config.dockerfile:
            cmd.extend(["-f", str(self.project_root / config.dockerfile)])

        build_context = config.build_context or self.project_root
        cmd.append(str(build_context))

        await _run_streaming(cmd, "DOCKER", f"Failed to build {config.full_name}")

        self._built.add(image_key)
        logger.info(f"Built image: {config.full_name}")

    async def build_all(self, force: bool = False) -> None:
        """Build all configured images.

        Args:
            force: Rebuild even if images exist.
        """
        for image_key in self.images:
            await self.build(image_key, force=force)

    def get_image_name(self, image_key: str) -> str:
        """Get full image name for a key.

        Args:
            image_key: Key in the images dict.

        Returns:
            Full image name with tag.
        """
        if image_key not in self.images:
            raise ValueError(f"Unknown image: {image_key}")

        return self.images[image_key].full_name

    async def get_image_id(self, image_key: str) -> str | None:
        """Get image ID for a key.

        Args:
            image_key: Key in the images dict.

        Returns:
            Image ID or None if not found.
        """
        if image_key not in self.images:
            raise ValueError(f"Unknown image: {image_key}")

        config = self.images[image_key]

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            config.full_name,
            "--format",
            "{{.Id}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            return None

        return stdout.decode().strip()

    async def tag(self, image_key: str, new_tag: str) -> str:
        """Tag an existing image with a new tag.

        Args:
            image_key: Key in the images dict.
            new_tag: New tag to apply.

        Returns:
            New full image name.
        """
        if image_key not in self.images:
            raise ValueError(f"Unknown image: {image_key}")

        config = self.images[image_key]
        new_name = f"{config.name}:{new_tag}"

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "tag",
            config.full_name,
            new_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to tag image: {stderr.decode()}")

        return new_name

    async def remove(self, image_key: str, force: bool = False) -> None:
        """Remove a Docker image.

        Args:
            image_key: Key in the images dict.
            force: Force removal.
        """
        if image_key not in self.images:
            raise ValueError(f"Unknown image: {image_key}")

        config = self.images[image_key]

        cmd = ["docker", "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(config.full_name)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        self._built.discard(image_key)
