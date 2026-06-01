# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.enums import ConversationContextMode, DatasetFormat
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str
from aiperf.config.dataset import FileDataset
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.dataset.loader.base_loader import BaseLoader
from aiperf.dataset.utils import check_file_exists
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, PluginType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class CustomDatasetComposer(BaseDatasetComposer):
    def __init__(self, *, run: BenchmarkRun, tokenizer: Tokenizer | None, **kwargs):
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        dataset = run.cfg.get_default_dataset()
        if not isinstance(dataset, FileDataset):
            raise ValueError("CustomDatasetComposer requires a file-based dataset.")
        self._file_dataset: FileDataset = dataset
        self._file_path: str | None = (
            str(dataset.path) if dataset.path is not None else None
        )
        self._inline_records = dataset.records
        self.loader: BaseLoader | None = None

    def create_dataset(self) -> list[Conversation]:
        """Create conversations from a file, directory, or inline records.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        # TODO: (future) for K8s, we need to transfer file data from SC (across node)
        is_inline = self._inline_records is not None
        if not is_inline:
            check_file_exists(Path(self._file_path))

        # Honor an explicit ``FileDataset.format`` (set via ``--custom-dataset-type``)
        # before falling back to structural inference. ``format`` defaults to
        # ``SINGLE_TURN``, so we use ``model_fields_set`` to distinguish "user
        # picked single_turn" from "default applied". This is required for
        # ``random_pool`` on JSONL files, whose schema overlaps with
        # ``single_turn``: structural inference always picks single_turn,
        # silently dropping the random-with-replacement sampling semantics.
        explicit_format = self._explicit_format()
        if explicit_format is not None:
            dataset_type = explicit_format
            self.info(f"Using explicit dataset format: {dataset_type}")
        elif is_inline:
            # Inline mode has no file to peek at for structural inference, so
            # we trust the (defaulted-or-set) FileDataset.format directly.
            dataset_type = self._format_to_loader_type(self._file_dataset.format)
            self.info(f"Using inline dataset format: {dataset_type}")
        else:
            dataset_type = self._infer_dataset_type(self._file_path)
            self.info(f"Auto-detected dataset type: {dataset_type}")

        # Validate synthesis options are only used with mooncake_trace
        self._validate_synthesis_config(dataset_type)

        self._create_loader_instance(dataset_type)
        dataset = self.loader.load_dataset()
        conversations = self.loader.convert_to_conversations(dataset)

        # Finalize all turns with metadata (custom datasets need this)
        for conversation in conversations:
            for turn in conversation.turns:
                self._finalize_turn(turn)

        # Finalize conversation-level context prompts
        self._finalize_conversations(conversations)
        return conversations

    def get_default_context_mode(self) -> ConversationContextMode | None:
        """Delegate to the loader's format-specific default, if a loader was created."""
        if self.loader is not None:
            return self.loader.get_default_context_mode()
        return None

    def _explicit_format(self) -> CustomDatasetType | None:
        """Return the user-selected loader type from ``FileDataset.format``.

        Returns ``None`` when the user did not explicitly set ``format`` (so
        structural inference should run). The CLI converter only emits
        ``format`` when ``--custom-dataset-type`` was provided, so
        ``model_fields_set`` membership is a reliable user-set signal.
        """
        if "format" not in self._file_dataset.model_fields_set:
            return None
        return self._format_to_loader_type(self._file_dataset.format)

    @staticmethod
    def _format_to_loader_type(fmt: DatasetFormat) -> CustomDatasetType:
        """Map a DatasetFormat enum value to its CustomDatasetType.

        Both enums mirror the custom_dataset_loader plugin registry and share
        identical string values, so a direct value-based conversion works.
        """
        return CustomDatasetType(fmt.value)

    def _infer_dataset_type(self, file_path: str) -> CustomDatasetType:
        """Infer the custom dataset type from the input file.

        Queries all registered loaders to check if they can handle the data format.

        Args:
            file_path: Path to the JSONL file or directory

        Returns:
            CustomDatasetType if successfully inferred

        Raises:
            ValueError: If no loader can handle the data format
        """
        try:
            path = Path(file_path)

            # If it's a directory, use path-based detection only
            if path.is_dir():
                return self._infer_type(data=None, filename=file_path)

            # For files, read first non-empty line and use both content and path detection
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    try:
                        data = load_json_str(line)
                    except orjson.JSONDecodeError:
                        # Non-JSON file (e.g. CSV) — fall back to filename-based detection
                        return self._infer_type(data=None, filename=file_path)
                    return self._infer_type(data=data, filename=file_path)

        except ValueError as e:
            self.exception(
                f"Error inferring dataset type from file: {file_path}: {e!r}"
            )
            raise

    def _infer_type(
        self, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> CustomDatasetType:
        """Infer the dataset type from data and/or filename.

        First checks for explicit 'type' field in the data, then falls back to
        structural detection by querying registered loaders via the factory.

        Args:
            data: Optional dictionary representing a single line from the JSONL file.
                  None indicates path-based detection only (e.g., for directories).
            filename: Optional path to the input file/directory for path-based detection

        Returns:
            CustomDatasetType if successfully inferred

        Raises:
            ValueError: If the type field is invalid or no loader can handle the data format
        """
        # Check for explicit type field first (most efficient).
        # Skip values that aren't known dataset types (e.g. Bailian's "type": "text"
        # is a request type, not a dataset type) and fall through to structural detection.
        if data is not None and data.get("type") in CustomDatasetType:
            explicit_type = CustomDatasetType(data["type"])
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, explicit_type
            )
            if not LoaderClass.can_load(data, filename):
                raise ValueError(
                    f"Explicit type field {explicit_type} specified, but loader {LoaderClass.__name__} "
                    "cannot handle the data format. Please specify --custom-dataset-type explicitly."
                )
            self.info(f"Using explicit type field: {explicit_type}")
            return explicit_type

        detected_type = None
        for entry, LoaderClass in plugins.iter_all(PluginType.CUSTOM_DATASET_LOADER):
            if LoaderClass.can_load(data, filename):
                self.info(
                    f"Loader {LoaderClass.__name__} can handle the input file data format."
                )
                dataset_type = CustomDatasetType(entry.name)
                if detected_type is not None:
                    raise ValueError(
                        f"Multiple loaders can handle the data format: {detected_type} and {dataset_type}. "
                        "Please specify --custom-dataset-type explicitly."
                    )
                detected_type = dataset_type

        if detected_type is None:
            raise ValueError(
                "No loader can handle the data format. Please specify --custom-dataset-type explicitly."
            )

        return detected_type

    def _should_synthesize(self) -> bool:
        """Whether the user has set any trace-synthesis options on this dataset.

        Reads the ``FileDataset.synthesis`` block; any of ``speedup_ratio``,
        ``prefix_len_multiplier``, ``prefix_root_multiplier``, or related
        knobs differing from their identity defaults flips this on.
        """
        s = self._file_dataset.synthesis
        if s is None:
            return False
        return (
            s.speedup_ratio != 1.0
            or s.prefix_len_multiplier != 1.0
            or s.prefix_root_multiplier != 1
            or s.prompt_len_multiplier != 1.0
            or s.output_len_multiplier != 1.0
        )

    def _validate_synthesis_config(self, dataset_type: CustomDatasetType) -> None:
        """Validate that synthesis options are only used with trace datasets.

        Args:
            dataset_type: The determined dataset type.

        Raises:
            ValueError: If synthesis options are set but dataset type is not a trace format.
        """
        s = self._file_dataset.synthesis
        if (
            s is not None
            and (
                self._should_synthesize()
                or s.max_isl is not None
                or s.max_osl is not None
            )
            and not plugins.is_trace_dataset(dataset_type)
        ):
            raise ValueError(
                f"Synthesis options (--synthesis-speedup-ratio, --synthesis-prefix-len-multiplier, "
                f"--synthesis-prefix-root-multiplier, --synthesis-prompt-len-multiplier, "
                f"--synthesis-output-len-multiplier, --synthesis-max-isl, --synthesis-max-osl) "
                f"are only supported with trace datasets, "
                f"but got {dataset_type}. "
                f"Either remove synthesis options or use a trace dataset type."
            )

    def _create_loader_instance(self, dataset_type: CustomDatasetType) -> None:
        """Initializes the dataset loader based on the custom dataset type.

        Args:
            dataset_type: The type of custom dataset to create.
        """
        kwargs: dict[str, Any] = {}
        loader_metadata = plugins.get_dataset_loader_metadata(dataset_type)
        if loader_metadata.is_trace:
            if self.prompt_generator is None:
                raise ValueError(
                    "Trace datasets require a tokenizer for prompt synthesis. "
                    "Ensure the endpoint supports tokenization or provide a --tokenizer."
                )
            kwargs["prompt_generator"] = self.prompt_generator

            if loader_metadata.default_block_size is not None:
                kwargs["default_block_size"] = loader_metadata.default_block_size

        elif dataset_type == CustomDatasetType.RANDOM_POOL:
            # ``FileDataset.entries`` is the pool size for random_pool (the
            # converter populates it from
            # ``input.conversation.num_dataset_entries`` / ``num`` /
            # ``loadgen.request_count``
            # in priority order). Pass it through so the loader produces
            # ``entries`` distinct sampled conversations rather than the loader's
            # default of 100. Leave None when not set so the loader applies its
            # own default.
            kwargs["num_conversations"] = self._file_dataset.entries

        LoaderClass = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, dataset_type)
        if self._inline_records is not None:
            self.loader = LoaderClass(
                inline_records=self._inline_records,
                run=self.run,
                **kwargs,
            )
        else:
            self.loader = LoaderClass(
                filename=self._file_path,
                run=self.run,
                **kwargs,
            )
