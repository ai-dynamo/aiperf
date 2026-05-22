# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import PublicDataset
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, PublicDatasetType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class PublicDatasetComposer(BaseDatasetComposer):
    """Composer for public benchmark datasets loaded from remote sources.

    Instantiates the appropriate public dataset loader using plugin metadata,
    loads the dataset, and finalizes all turns with model name and max_tokens.
    """

    def __init__(self, *, run: BenchmarkRun, tokenizer: Tokenizer | None, **kwargs):
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        dataset = run.cfg.get_default_dataset()
        if not isinstance(dataset, PublicDataset):
            raise ValueError("PublicDatasetComposer requires a public dataset.")
        self._public_dataset = dataset

    def create_dataset(self) -> list[Conversation]:
        raise NotImplementedError("Use create_dataset_async() for public datasets")

    async def create_dataset_async(self) -> list[Conversation]:
        """Load and finalize a public benchmark dataset.

        Returns:
            list[Conversation]: Finalized conversations ready for benchmarking.
        """
        dataset_type: PublicDatasetType = self._public_dataset.dataset

        LoaderClass = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, dataset_type)

        loader_kwargs = self._build_loader_kwargs(dataset_type, LoaderClass)
        loader = LoaderClass(
            run=self.run,
            tokenizer=self.tokenizer,
            **loader_kwargs,
        )

        data = await loader.load_dataset()
        conversations = await loader.convert_to_conversations(data)

        for conversation in conversations:
            for turn in conversation.turns:
                self._finalize_turn(turn)

        self._finalize_conversations(conversations)
        return conversations

    def _build_loader_kwargs(
        self, dataset_type: PublicDatasetType, loader_class: type
    ) -> dict[str, Any]:
        """Build loader constructor kwargs from plugin metadata.

        Reads HF-specific fields from the plugin metadata and returns only the
        non-None values so that non-HF loaders (e.g. ShareGPT) receive no
        unexpected kwargs.

        Args:
            dataset_type: The public dataset plugin name.
            loader_class: The loader class about to be instantiated. Used to
                validate that opt-in metadata fields (e.g. ``multi_turn``) are
                actually declared on the constructor before forwarding them,
                so users learn about misconfigurations instead of silently
                getting the default behavior.

        Returns:
            dict of kwargs to pass to the loader constructor.
        """
        loader_metadata = plugins.get_public_dataset_loader_metadata(dataset_type)
        kwargs: dict[str, Any] = {}

        if loader_metadata.hf_dataset_name is not None:
            kwargs["hf_dataset_name"] = loader_metadata.hf_dataset_name
            kwargs["hf_split"] = loader_metadata.hf_split
            cli_subset = self._public_dataset.hf_subset
            subset = cli_subset if cli_subset is not None else loader_metadata.hf_subset
            if subset is not None:
                kwargs["hf_subset"] = subset

        optional_fields = {
            "prompt_column": loader_metadata.prompt_column,
            "image_column": loader_metadata.image_column,
            "video_column": loader_metadata.video_column,
            "audio_column": loader_metadata.audio_column,
            "category": loader_metadata.category,
            "prompt_template": loader_metadata.prompt_template,
        }
        kwargs.update({k: v for k, v in optional_fields.items() if v is not None})

        if loader_metadata.conversation_column is not None:
            kwargs["conversation_column"] = loader_metadata.conversation_column
            kwargs["message_content_key"] = loader_metadata.message_content_key

        if loader_metadata.multi_turn:
            if not self._loader_accepts_kwarg(loader_class, "multi_turn"):
                raise ValueError(
                    f"Loader {loader_class.__name__} for dataset {dataset_type!r} "
                    "does not support the 'multi_turn' metadata flag. Remove "
                    "'multi_turn: true' from this loader's plugin metadata, or "
                    "use a loader that declares 'multi_turn' on its constructor "
                    "(e.g. HFConversationDatasetLoader, SpeedBenchLoader, "
                    "SpecBenchLoader)."
                )
            kwargs["multi_turn"] = True

        if loader_metadata.streaming:
            kwargs["streaming"] = loader_metadata.streaming

        return kwargs

    @staticmethod
    def _loader_accepts_kwarg(loader_class: type, name: str) -> bool:
        """Return True when ``name`` is an explicitly declared parameter of
        ``loader_class.__init__`` (or any class in its MRO before ``BaseMixin``).

        ``**kwargs`` does not count as acceptance — the silent-swallow chain
        through ``BaseMixin.__init__`` is exactly what this check exists to
        catch.
        """
        for klass in loader_class.__mro__:
            if klass.__name__ == "BaseMixin":
                break
            try:
                params = inspect.signature(klass.__init__).parameters
            except (TypeError, ValueError):
                continue
            param = params.get(name)
            if param is not None and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False
