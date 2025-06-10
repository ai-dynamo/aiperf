#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, TypeAlias

Filename: TypeAlias = str
TextData: TypeAlias = list[str]
ImageData: TypeAlias = list[str]
AudioData: TypeAlias = list[str]
InputData: TypeAlias = TextData | ImageData | AudioData
OptionalData: TypeAlias = dict[str, Any]
PayloadMetadata: TypeAlias = dict[str, Any]
DataRowField: TypeAlias = InputData | OptionalData | PayloadMetadata
DataRowDict: TypeAlias = dict[str, DataRowField]
GenericDatasetDict: TypeAlias = dict[Filename, list[DataRowDict]]


@dataclass
class DataRow:
    texts: TextData = field(default_factory=list)
    images: ImageData = field(default_factory=list)
    audios: AudioData = field(default_factory=list)
    optional_data: dict[str, Any] = field(default_factory=dict)
    payload_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> DataRowDict:
        """
        Converts the DataRow object to a dictionary.
        """
        datarow_dict: DataRowDict = {}

        if self.texts:
            datarow_dict["texts"] = self.texts
        if self.images:
            datarow_dict["images"] = self.images
        if self.audios:
            datarow_dict["audios"] = self.audios
        if self.optional_data:
            datarow_dict["optional_data"] = self.optional_data
        if self.payload_metadata:
            datarow_dict["payload_metadata"] = self.payload_metadata
        return datarow_dict


@dataclass
class FileData:
    rows: list[DataRow]

    def to_list(self) -> list[DataRowDict]:
        """
        Converts the FileData object to a list.
        Output format example for two payloads from a file:
        [
            {
                'texts': ['text1', 'text2'],
                'images': ['image1', 'image2'],
                'audios': ['audio1', 'audio2'],
                'optional_data': {},
                'payload_metadata': {},
            },
            {
                'texts': ['text3', 'text4'],
                'images': ['image3', 'image4'],
                'audios': ['audio3', 'audio4'],
                'optional_data': {},
                'payload_metadata': {},
            },
        ]
        """
        return [row.to_dict() for row in self.rows]


@dataclass
class GenericDataset:
    files_data: dict[str, FileData]

    def to_dict(self) -> GenericDatasetDict:
        """
        Converts the entire DataStructure object to a dictionary.
        Output format example for one payload from two files:
        {
            'file_0': [
                {
                    'texts': ['text1', 'text2'],
                    'images': ['image1', 'image2'],
                    'audios': ['audio1', 'audio2'],
                    'optional_data': {},
                    'payload_metadata': {},
                },
            ],
            'file_1': [
                {
                    'texts': ['text1', 'text2'],
                    'images': ['image1', 'image2'],
                    'audios': ['audio1', 'audio2'],
                    'optional_data': {},
                    'payload_metadata': {},
                },
            ],
        }
        """
        return {
            filename: file_data.to_list()
            for filename, file_data in self.files_data.items()
        }
