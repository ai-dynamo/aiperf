# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from enum import Enum
from pathlib import PosixPath
from textwrap import indent
from typing import Any, Dict

from aiperf.common.models.config.config_field import ConfigField


class BaseConfig:
    """
    A base class that holds a collection of configuration fields (ConfigField).

    The __getattr__ and __setattr__ methods are custom to allow
    the user to access the value of the ConfigField directly.

    Examples:
      field_a.field_b = 5 sets the value of field_b to 5
      field_a.field_b returns the value of field_b

      field_a.get_field("field_b") returns the ConfigField object of field_b
    """

    def __init__(self):
        self._fields = {}
        self._children = {}

        # This exists just to make looking up values when debugging easier
        self._values = {}

    ###########################################################################
    # Top-Level Methods
    ###########################################################################
    def get_field(self, name) -> ConfigField:
        if name not in self._fields:
            raise ValueError(f"{name} not found in ConfigFields")

        return self._fields[name]

    def any_field_set_by_user(self) -> bool:
        return any(field.is_set_by_user for field in self._fields.values())

    def check_required_fields_are_set(self) -> None:
        for name, field in self._fields.items():
            if field.required and not field.is_set_by_user:
                raise ValueError(f"Required field {name} is not set")

        for child in self._children.values():
            child.check_required_fields_are_set()

    def model_dump(self) -> Dict[str, Any]:
        config_dict = {}
        for key, value in self._values.items():
            if hasattr(value, "model_dump"):
                config_dict[key] = value.model_dump()
            else:
                config_dict[key] = self._get_legal_json_value(value)

        return config_dict

    def _get_legal_json_value(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.name.lower()
        elif isinstance(value, PosixPath):
            return str(value)
        elif hasattr(value, "__dict__"):
            return value.__dict__
        elif isinstance(value, dict):
            config_dict = {}
            for k, v in value.items():
                config_dict[k] = self._get_legal_json_value(v)

            return config_dict
        elif (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, bool)
            or isinstance(value, list)
        ):
            return value
        elif value is None:
            return ""
        else:
            raise ValueError(f"Value {value} is not a legal JSON value")

    ###########################################################################
    # Template Creation Methods
    ###########################################################################
    def create_template(self, header: str, level: int = 1, verbose=False) -> str:
        indention = "  " * level

        template = self._add_header_to_template(header, indention)
        template += self._add_fields_to_template(indention, verbose)
        template += "\n"
        template += self._add_children_to_template(level, verbose)

        return template

    def _add_header_to_template(self, header: str, indention: str) -> str:
        template = ""
        if header:
            template = indent(f"{header}:\n", indention)
        return template

    def _add_fields_to_template(self, indention: str, verbose: bool) -> str:
        template = ""
        for name, field in self._fields.items():
            template_comment = self._get_template_comment(field, verbose)
            template += self._create_template_from_comment(template_comment, indention)
            template += self._add_field_to_template(field, name, indention)

            if verbose and field.verbose_template_comment:
                template += "\n"

        return template

    def _add_children_to_template(self, level: int, verbose: bool) -> str:
        template = ""
        for name, child in self._children.items():
            template += child.create_template(
                header=name, level=level + 1, verbose=verbose
            )

        return template

    def _get_template_comment(self, field: ConfigField, verbose: bool) -> str:
        if verbose and field.verbose_template_comment:
            return field.verbose_template_comment
        else:
            return field.template_comment if field.template_comment else ""

    def _create_template_from_comment(self, comment: str, indention: str) -> str:
        template = ""
        if comment:
            comment_lines = comment.split("\n")
            for comment_line in comment_lines:
                template += indent(f"  # {comment_line}\n", indention)

        return template

    def _add_field_to_template(
        self, field: ConfigField, name: str, indention: str
    ) -> str:
        template = ""
        if field.add_to_template:
            json_value = self._get_legal_json_value(self.__getattr__(name))
            if isinstance(json_value, list):
                json_value = ", ".join(map(str, json_value))

            template = indent(
                f"  {name}: {json_value}\n",
                indention,
            )
        return template

    ###########################################################################
    # Dunder Methods
    ###########################################################################
    def __setattr__(self, name, value):
        # This prevents recursion failure in __init__
        if name == "_fields" or name == "_values" or name == "_children":
            self.__dict__[name] = value
        else:
            if isinstance(value, ConfigField):
                self._fields[name] = value
            elif name in self._fields:
                self._fields[name].value = value
            else:
                self._children[name] = value
                self._values[name] = value
                return

            if self._fields[name].is_set_by_user:
                self._values[name] = self._fields[name].value
            else:
                self._values[name] = self._fields[name].default

    def __getattr__(self, name):
        if name == "_fields" or name == "_values":
            return self.__dict__[name]
        elif name in self._children:
            return self._children[name]
        else:
            if not name in self._fields:
                raise AttributeError(f"{name} not found in ConfigFields")

            if self._fields[name].is_set_by_user:
                return self._fields[name].value
            else:
                return self._fields[name].default

    def __deepcopy__(self, memo):
        # new_copy = BaseConfig()
        cls = self.__class__
        new_copy = cls.__new__(cls)
        new_copy.__init__()
        memo[id(self)] = new_copy

        for key, value in self._fields.items():
            new_value = copy.deepcopy(value, memo)
            new_copy.__setattr__(key, new_value)

        for key, value in self._children.items():
            new_value = copy.deepcopy(value, memo)
            new_copy.__setattr__(key, new_value)

        return new_copy

    def __delitem__(self, key):
        if key in self._fields:
            del self._fields[key]
            del self._values[key]
        else:
            del self._children[key]
        return
