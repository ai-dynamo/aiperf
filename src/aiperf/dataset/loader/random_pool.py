# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import ValidationError

from aiperf.common import random_generator as rng
from aiperf.common.enums import MediaType
from aiperf.common.models import Audio, Conversation, Image, Media, Text, Turn, Video
from aiperf.config.dataset.config import FileDataset
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import RandomPool
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

logger = logging.getLogger(__name__)

# Type aliases
Filename: TypeAlias = str


class RandomPoolDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads data from a single file or a directory.

    Each line in the file represents single-turn conversation data,
    and files create individual pools for random sampling:
      - Single file: All lines form one single pool (to be randomly sampled from)
      - Directory: Each file becomes a separate pool, then pools are randomly sampled
                   and merged into conversations later.

    The random pool custom dataset
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch size > 1)
      - supports named fields for each modality (e.g. text_field_a, text_field_b, etc.)
      - DOES NOT support multi-turn or its features (e.g. delay, sessions, etc.)

    Batching and named pools are mutually exclusive: batch sizes other than 1 are
    rejected outright for directory input (multiple named pools), since batching
    flattens every pool into one anonymous pool per modality. Directory input is
    caught at config time by ``_reject_file_dataset_incompatible``; the pool shapes
    only visible after parsing (inline multi-key ``records:``, entries embedding
    named media objects) are caught here by ``_reject_batching_with_named_pools``.
    A batch size on a modality absent from the pool does not count as batching --
    see ``_batching_requested``.

    Note on batching and associations:
    When entries have paired data across modalities (e.g. {"image": "cat.png", "text": "describe
    this cat"}), enabling batch sizes > 1 flattens each modality into an independent pool and
    samples from them separately. This intentionally breaks per-entry associations — the name
    "random_pool" implies independent sampling. Use the single_turn dataset type instead if exact
    request structure with preserved pairings is required.

    Example:

    1. Single file
    ```jsonl
    {"text": "Who are you?", "image": "/path/to/image1.png"}
    {"text": "Explain what is the meaning of life.", "image": "/path/to/image2.png"}
    ...
    ```
    The file will form a single pool of text and image data that will be used
    to generate conversations.

    2. Directory

    Directory will be useful if user wants to
      - create multiple pools of different modalities separately (e.g. text, image)
      - specify different field names for the same modality.

    data/queries.jsonl
    ```jsonl
    {"texts": [{"name": "query", "contents": ["Who are you?"]}]}
    {"texts": [{"name": "query", "contents": ["What is the meaning of life?"]}]}
    ...
    ```

    data/passages.jsonl
    ```jsonl
    {"texts": [{"name": "passage", "contents": ["I am a cat."]}]}
    {"texts": [{"name": "passage", "contents": ["I am a dog."]}]}
    ...
    ```

    The loader will create two separate pools for each file: queries and passages.
    Each pool is a text dataset with a different field name (e.g. query, passage),
    and loader will later sample from these two pools to create conversations.
    """

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        inline_records: list[dict[str, Any]]
        | dict[str, list[dict[str, Any]]]
        | None = None,
        run: BenchmarkRun | None = None,
        num_conversations: int = 1,
        **kwargs,
    ):
        super().__init__(
            filename=filename, inline_records=inline_records, run=run, **kwargs
        )
        self._rng = rng.derive("dataset.loader.random_pool")
        self.num_conversations = (
            num_conversations if num_conversations is not None else 100
        )
        # Per-modality batch sizes come from different places depending on the
        # dataset type.  SyntheticDataset carries them nested under
        # prompts/images/audio/video sub-configs.  FileDataset (the --input-file
        # path) stores them as flat fields populated by the CLI converter.
        dataset = self.run.cfg.get_default_dataset()
        if isinstance(dataset, FileDataset):
            self.batch_size_text = (
                dataset.prompt_batch_size
                if dataset.prompt_batch_size is not None
                else 1
            )
            self.batch_size_image = (
                dataset.image_batch_size if dataset.image_batch_size is not None else 1
            )
            self.batch_size_audio = (
                dataset.audio_batch_size if dataset.audio_batch_size is not None else 1
            )
            self.batch_size_video = (
                dataset.video_batch_size if dataset.video_batch_size is not None else 1
            )
        else:
            prompts = getattr(dataset, "prompts", None)
            images = getattr(dataset, "images", None)
            audio = getattr(dataset, "audio", None)
            video = getattr(dataset, "video", None)
            self.batch_size_image = getattr(images, "batch_size", 1) if images else 1
            self.batch_size_text = getattr(prompts, "batch_size", 1) if prompts else 1
            self.batch_size_audio = getattr(audio, "batch_size", 1) if audio else 1
            self.batch_size_video = getattr(video, "batch_size", 1) if video else 1

    @staticmethod
    def _validate_path(path: Path) -> int:
        """Validate all files and directories recursively against the RandomPool model.

        Args:
            path: The path to the file or directory to validate.

        Returns:
            int: Count of files with at least one valid line.

        Raises:
            ValidationError: If any file contains invalid data.
        """
        valid_count = 0

        if path.is_dir():
            # if path is a directory, recursively call this function for each child
            # if any child fails validation, it will exit early with an exception
            for file in path.iterdir():
                valid_count += RandomPoolDatasetLoader._validate_path(file)

        elif path.is_file():
            # if path is a file, validate the first non-empty line against the RandomPool model
            # if the line is valid, increment the valid count and break the loop,
            # otherwise a ValidationError will be raised and the function will exit early
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    RandomPool.model_validate_json(line)
                    valid_count += 1
                    break

        return valid_count

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        RandomPool is the only loader that supports directory inputs.
        For structural detection, RandomPool format is ambiguous with SingleTurn
        (both have modality fields), so explicit 'type' field or directory path is required.

        Returns:
            True only if filename is a directory with at least one valid file.
            False otherwise (including for regular files without explicit type).
        """

        if data is not None and data.get("type") == CustomDatasetType.RANDOM_POOL:
            try:
                RandomPool.model_validate(data)
                return True
            except ValidationError:
                return False

        if filename is not None:
            try:
                path = Path(filename) if isinstance(filename, str) else filename
                # Only match directories - files are ambiguous with SingleTurn
                if path.is_dir():
                    valid_count = cls._validate_path(path)
                    return valid_count > 0
                return False
            except ValidationError:
                return False

        # RandomPool schema is very similar to SingleTurn, so we can't reliably
        # distinguish without an explicit type field or directory path
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for RandomPool."""
        return DatasetSamplingStrategy.SHUFFLE

    def load_dataset(self) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from a file, directory, or inline records.

        Returns a dictionary mapping pool name to a list of RandomPool objects.
          - File mode (single file): one pool keyed by file basename.
          - File mode (directory): one pool per file, keyed by file basename.
          - Inline mode (flat list): one pool keyed by ``"<inline>"``.
          - Inline mode (dict-of-lists): one pool per dict key.
        """
        if self.inline_records is not None:
            if isinstance(self.inline_records, dict):
                return {
                    pool_name: [
                        RandomPool.model_validate(r)
                        for r in self._iter_record_dicts(source=pool_name)
                    ]
                    for pool_name in self.inline_records
                }
            return {
                "<inline>": [
                    RandomPool.model_validate(r) for r in self._iter_record_dicts()
                ]
            }

        path = Path(self.filename)
        if path.is_file():
            dataset_pool = self._load_dataset_from_file(path)
            return {path.name: dataset_pool}
        return self._load_dataset_from_dir(path)

    def _load_dataset_from_file(self, file_path: Path) -> list[RandomPool]:
        """Load random pool data from a single file.

        Args:
            file_path: The path to the file containing the data.

        Returns:
            A list of RandomPool objects.
        """
        dataset_pool: list[RandomPool] = []

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                random_pool_data = RandomPool.model_validate_json(line)
                dataset_pool.append(random_pool_data)

        return dataset_pool

    def _load_dataset_from_dir(
        self, dir_path: Path
    ) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from all files in a directory.

        Args:
            dir_path: The path to the directory containing the files.

        Returns:
            A dictionary mapping filename to list of RandomPool objects.
        """
        data: dict[Filename, list[RandomPool]] = defaultdict(list)

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file():
                dataset_pool = self._load_dataset_from_file(file_path)
                data[file_path.name].extend(dataset_pool)

        return data

    def convert_to_conversations(
        self, data: dict[Filename, list[RandomPool]]
    ) -> list[Conversation]:
        """Convert random pool data to conversation objects.

        When any batch size deviates from 1, uses flat pool sampling to form a single
        turn with multiple items per conversation. Otherwise, each RandomPool entry
        becomes a single-turn conversation.

        Sampling is always done with replacement, so duplicates within a single request
        are possible when batch_size exceeds pool size.

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.

        Returns:
            A list of conversations.
        """
        logger.info(
            "Sampling random_pool dataset entries with replacement. "
            "Duplicates within a single request are possible when batch_size exceeds pool size."
        )
        if self._batching_requested(data):
            self._reject_batching_with_named_pools(data)
            return self._convert_to_conversations_batched(data)

        conversations = [
            Conversation(session_id=self.session_id_generator.next())
            for _ in range(self.num_conversations)
        ]

        # F x N (F: num of files, N: num of conversations)
        sampled_dataset: dict[Filename, list[Turn]] = {}

        # Randomly sample (with replacement) from each dataset pool
        for filename, dataset_pool in data.items():
            samples = self._rng.choices(dataset_pool, k=self.num_conversations)
            turns: list[Turn] = []
            for sample in samples:
                media = self.convert_to_media_objects(sample, name=Path(filename).stem)
                turns.append(
                    Turn(
                        texts=media[MediaType.TEXT],
                        images=media[MediaType.IMAGE],
                        audios=media[MediaType.AUDIO],
                        videos=media[MediaType.VIDEO],
                    )
                )
            sampled_dataset[filename] = turns

        # Merge turns for each conversation
        for i, batched_turns in enumerate(zip(*sampled_dataset.values(), strict=False)):
            turn = self._merge_turns(batched_turns)
            conversations[i].turns.append(turn)

        return conversations

    def _batching_requested(self, data: dict[Filename, list[RandomPool]]) -> bool:
        """Return True if a modality actually present in the pool has batch size != 1.

        A batch size on a modality absent from the pool produces the same (empty)
        output on either path, so it must not select the flattened one:
        ``--image-batch-size 0`` against a text-only pool means "disable image
        inputs entirely" (as the flag's own description says), not "flatten every
        named text pool into one anonymous pool".
        """
        for batch_size, singular, plural in (
            (self.batch_size_text, "text", "texts"),
            (self.batch_size_image, "image", "images"),
            (self.batch_size_audio, "audio", "audios"),
            (self.batch_size_video, "video", "videos"),
        ):
            if batch_size == 1:
                continue
            if any(
                getattr(entry, singular) is not None or getattr(entry, plural)
                for pool in data.values()
                for entry in pool
            ):
                return True
        return False

    @staticmethod
    def _reject_batching_with_named_pools(
        data: dict[Filename, list[RandomPool]],
    ) -> None:
        """Reject batch sizes other than 1 when batching would discard pool identity.

        Two distinct ways this happens, both from the same root cause:
        ``_build_flat_pool`` unwraps embedded ``Text``/``Image``/``Audio``/``Video``
        objects down to bare content strings, and ``_convert_to_conversations_batched``
        rebuilds them as ``X(name="", contents=...)`` -- discarding any authored
        ``name`` and, for images, any authored ``uuids`` (vLLM cache-reuse IDs).

        1. Multiple named pools in ``data`` -- one per key. Directory input (multiple
           files, e.g. ``queries.jsonl`` -> ``query``, ``passages.jsonl`` -> ``passage``)
           and inline YAML ``records:`` with multiple top-level keys (e.g.
           ``records: {queries: [...], passages: [...]}``) both land here; either
           way, each key is a separately named pool that name-sensitive endpoints
           (e.g. rankings, which routes on the ``query``/``queries`` and ``passages``
           field names) depend on. Flattening across pools merges them into one
           anonymous pool per modality.

        2. A single pool whose entries embed named ``Text``/``Image``/``Audio``/
           ``Video`` objects, or ``Image`` objects carrying ``uuids``. This is
           reachable from a single file (or single-key inline YAML ``records``)
           and is not caught by pool count alone.

        Raises:
            ValueError: If either condition applies.
        """
        if len(data) > 1:
            names = ", ".join(sorted(data))
            raise ValueError(
                f"random_pool batch sizes other than 1 are not supported for named "
                f"pools (found {len(data)}: {names}). Drop the batch-size flags, or "
                "use a single unnamed pool -- one file, or a flat inline records: list."
            )
        if RandomPoolDatasetLoader._pool_entries_carry_metadata(data):
            raise ValueError(
                "random_pool batch sizes other than 1 are not supported when pool "
                "entries carry named Text/Image/Audio/Video objects or image cache "
                "uuids. Drop the batch-size flags, or strip name/uuids from the "
                "entries if only their contents matter."
            )

    @staticmethod
    def _pool_entries_carry_metadata(data: dict[Filename, list[RandomPool]]) -> bool:
        """Return True if any pool entry authors a Media name or (image) uuids.

        Checks the plural list fields (``texts``/``images``/``audios``/``videos`` --
        the only ones that can hold ``Media`` objects rather than bare strings)
        across every ``RandomPool`` entry in every pool.
        """
        for pool in data.values():
            for entry in pool:
                for items in (entry.texts, entry.images, entry.audios, entry.videos):
                    if not items:
                        continue
                    for item in items:
                        if isinstance(item, Media) and (
                            item.name or getattr(item, "uuids", None)
                        ):
                            return True
        return False

    def _build_flat_pool(
        self,
        data: dict[Filename, list[RandomPool]],
        singular: str,
        plural: str,
    ) -> list[str]:
        """Collect all strings for a given modality from all pool entries into a flat list.

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.
            singular: The field name for a single item (e.g. "image").
            plural: The field name for a list of items (e.g. "images").

        Returns:
            A flat list of strings for the given modality.
        """
        pool: list[str] = []
        for items in data.values():
            for item in items:
                value = getattr(item, singular)
                if value is not None:
                    pool.append(value)
                values = getattr(item, plural)
                if values is not None:
                    for v in values:
                        if isinstance(v, str):
                            pool.append(v)
                        else:
                            pool.extend(v.contents)
        return pool

    def _convert_to_conversations_batched(
        self, data: dict[Filename, list[RandomPool]]
    ) -> list[Conversation]:
        """Convert pool data to conversations using flat pool batch sampling.

        Builds a flat pool per modality from all pool entries. For each conversation,
        samples batch_size_image images, batch_size_text texts, batch_size_audio audios,
        and batch_size_video videos (all with replacement) from their respective pools.
        Modalities absent from the pool are omitted; modalities present but whose batch
        size is 0 are suppressed; modalities present at batch size 1 (the default) are
        still sampled so no data is silently dropped when only one batch size > 1.

        Note: per-entry associations (e.g. a paired text+image) are not preserved —
        each modality is sampled independently from its flat pool.

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.

        Returns:
            A list of conversations, each with one turn containing a batch of items.
        """
        image_pool = self._build_flat_pool(data, "image", "images")
        text_pool = self._build_flat_pool(data, "text", "texts")
        audio_pool = self._build_flat_pool(data, "audio", "audios")
        video_pool = self._build_flat_pool(data, "video", "videos")

        if not (
            (image_pool and self.batch_size_image > 0)
            or (text_pool and self.batch_size_text > 0)
            or (audio_pool and self.batch_size_audio > 0)
            or (video_pool and self.batch_size_video > 0)
        ):
            # Raise rather than warn: this loader runs inside the DatasetManager
            # subprocess, whose stdlib-logger output only reaches the log file, so a
            # warning here never surfaces on the console. The run would then fail
            # with a generic "check the server URL/endpoint/response format" error
            # for what is purely a local config mistake.
            raise ValueError(
                f"random_pool batch sizes produce turns with no content: every "
                f"modality is absent from the pool or has batch size 0 (text="
                f"{self.batch_size_text}, image={self.batch_size_image}, audio="
                f"{self.batch_size_audio}, video={self.batch_size_video}). Set a "
                "batch size above 0 for a modality that is present in the pool."
            )

        conversations = []
        for _ in range(self.num_conversations):
            images: list[Image] = []
            if image_pool and self.batch_size_image > 0:
                sampled = self._rng.choices(image_pool, k=self.batch_size_image)
                processed = [
                    self._handle_media_content(p, MediaType.IMAGE) for p in sampled
                ]
                images = [Image(name="", contents=processed)]

            texts: list[Text] = []
            if text_pool and self.batch_size_text > 0:
                sampled_texts = self._rng.choices(text_pool, k=self.batch_size_text)
                texts = [Text(name="", contents=sampled_texts)]

            audios: list[Audio] = []
            if audio_pool and self.batch_size_audio > 0:
                sampled_audios = self._rng.choices(audio_pool, k=self.batch_size_audio)
                processed_audios = [
                    self._handle_media_content(a, MediaType.AUDIO)
                    for a in sampled_audios
                ]
                audios = [Audio(name="", contents=processed_audios)]

            videos: list[Video] = []
            if video_pool and self.batch_size_video > 0:
                sampled_videos = self._rng.choices(video_pool, k=self.batch_size_video)
                processed_videos = [
                    self._handle_media_content(v, MediaType.VIDEO)
                    for v in sampled_videos
                ]
                videos = [Video(name="", contents=processed_videos)]

            turn = Turn(texts=texts, images=images, audios=audios, videos=videos)
            conv = Conversation(session_id=self.session_id_generator.next())
            conv.turns.append(turn)
            conversations.append(conv)

        return conversations

    def _merge_turns(self, turns: list[Turn]) -> Turn:
        """Merge turns into a single turn.

        Args:
            turns: A list of turns.

        Returns:
            A single turn.
        """
        merged_turn = Turn(
            texts=[text for turn in turns for text in turn.texts],
            images=[image for turn in turns for image in turn.images],
            audios=[audio for turn in turns for audio in turn.audios],
            videos=[video for turn in turns for video in turn.videos],
        )
        return merged_turn
