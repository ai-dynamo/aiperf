#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import io
from typing import Any, Dict
from pydantic import (
    BaseModel,
    SerializationInfo,
    model_serializer,
)

import ruamel.yaml
from ruamel.yaml import YAML, RoundTripRepresenter, safe_dump
from ruamel.yaml.comments import CommentedMap
from enum import Enum

ruamel.yaml.representer.RoundTripRepresenter.ignore_aliases = lambda self, data: True


class BaseConfig(BaseModel):
    @model_serializer(mode="wrap")
    def serialize(self, handler, info: SerializationInfo):
        context = info.context or {}
        if context.get("verbose", False):
            print(f"{self.__class__.__name__} Verbose: {context['verbose']}")
        data = handler(self, info)
        # Could add/remove fields or adjust based on context here
        return data

    def serialize_to_yaml(self, indent: int = 2) -> str:
        """
        Serialize a Pydantic model to a YAML string.

        Args:
            indent: The per-level indentation to use.
        """
        # Dump model to dict with context (flags propagate recursively)
        context = {
            "verbose": getattr(self, "verbose", False),
        }

        data = self.model_dump(context=context)

        # Attach comments recursively
        commented_data = self._attach_comments(
            data=data,
            model=self,
            context=context,
            indent=indent,
        )

        # Dump to YAML
        yaml = YAML(pure=True)
        yaml.indent(mapping=indent, sequence=indent, offset=indent)
        yaml.default_flow_style = False

        # Register for all Enum subclasses
        # yaml.representer.add_multi_representer(Enum, self._enum_representer)

        # Control vertical spacing based on verbose flag
        if context.get("verbose", False):
            # If verbose is True, output will include comments and spaces
            yaml.compact(seq_seq=False, seq_map=False)
        else:
            # If verbose is False, output will be compact without spaces
            yaml.compact(seq_seq=True, seq_map=True)

        stream = io.StringIO()
        yaml.dump(commented_data, stream)
        return stream.getvalue()

    def _attach_comments(
        self,
        data: Any,
        model: BaseModel,
        context: dict,
        indent: int = 2,
        indent_level: int = 0,
    ) -> Any:
        """
        Recursively convert dicts to ruamel.yaml CommentedMap and attach comments from
        Pydantic field descriptions, or based on context (e.g., verbose flag).

        Args:
            data: The raw data to convert to a CommentedMap.
            model: The Pydantic model that contains the field descriptions.
            context: The Pydantic serializer context which contains the serializer flags.
            indent: The per-level indentation to use for the comments.
            indent_level: The current level of indentation. The actual indentation is
                `indent * indent_level`.

        Returns:
            The data with comments attached.
        """
        if isinstance(data, dict):
            # Create a CommentedMap to store the commented data. This is a special type of
            # dict provided by the ruamel.yaml library that preserves the order of the keys and
            # allows for comments to be attached to the keys.
            commented_map = CommentedMap()

            for field_name, value in data.items():
                field = model.model_fields.get(field_name)

                if self._do_not_add_field_to_template(field):
                    continue

                value = self._preprocess_value(value)

                if self._is_a_nested_config(field, value):
                    # Recursively process nested models
                    commented_map[field_name] = self._attach_comments(
                        value,
                        getattr(model, field_name),
                        context=context,
                        indent=indent,
                        indent_level=indent_level + 1,
                    )
                else:
                    # Attach the value to the commented map
                    commented_map[field_name] = value

                # Attach comment if verbose and description exists
                if context.get("verbose") and field and field.description:
                    # Set the comment before the key, with the specified indentation
                    commented_map.yaml_set_comment_before_after_key(
                        field_name,
                        before=field.description,
                        indent=indent * indent_level,
                    )

            return commented_map

    def _do_not_add_field_to_template(self, field: Any) -> bool:
        return (
            field
            and field.json_schema_extra
            and not field.json_schema_extra.get("add_to_template")
        )

    def _is_a_nested_config(self, field: Any, value: Any) -> bool:
        return (
            isinstance(value, dict)
            and field
            and issubclass(field.annotation, BaseModel)
        )

    def _preprocess_value(self, value: Any) -> Any:
        """
        Preprocess the value before serialization.
        """

        if isinstance(value, list):
            return ", ".join(map(str, value))
        elif isinstance(value, Enum):
            return str(value.value).lower()
        return value
