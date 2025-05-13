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

from enum import Enum
from typing import Any, Dict, Optional


class ConfigField:
    """
    A class to represent a configuration field.
    Only the default value is required, all other values are optional.
    Bounds and choices are checked when the value is set.

    default: The default value of the configuration field.
    required: Whether a value is required to be set by the user.
    add_to_template: Whether the field should be added to the template.
    template_comment: A comment to add to the template.
    value: The value of the configuration field.
    bounds: A dictionary with upper and lower bounds.
    choices: A list of choices or an Enum.
    is_set_by_user: Whether the value was set by the user.
    """

    def __init__(
        self,
        default: Any,
        required: bool = False,
        add_to_template: bool = True,
        template_comment: Optional[str] = None,
        verbose_template_comment: Optional[str] = None,
        value: Optional[Any] = None,
        bounds: Optional[Dict[str, Any]] = None,
        choices: Optional[Any] = None,
    ):
        self.default = default
        self.required = required
        self.add_to_template = add_to_template
        self.template_comment = template_comment
        self.verbose_template_comment = verbose_template_comment
        self._set_bounds(bounds)
        self.choices = choices
        self.is_set_by_user = False
        self.value = value

    def _set_bounds(self, bounds: Optional[Dict[str, Any]] = None) -> None:
        if not bounds:
            self.bounds = None
        else:
            self.bounds = {}
            for key, value in bounds.items():
                if key not in ["upper", "max", "lower", "min"]:
                    raise ValueError(
                        f"User Config: {key} is not a valid key for bounds. "
                        f"Valid keys are 'upper', 'max, 'lower', or 'min'."
                    )
                if key in ["upper", "max"]:
                    self.bounds["upper"] = value
                elif key in ["lower", "min"]:
                    self.bounds["lower"] = value

    def _check_bounds(self) -> None:
        if isinstance(self.value, list):
            for item in self.value:
                self._check_item_bounds(item)
        else:
            self._check_item_bounds(self.value)

    def _check_item_bounds(self, item: Any) -> None:
        if isinstance(item, (int, float)):
            if self.bounds and "upper" in self.bounds:
                if self.bounds["upper"] < item:
                    raise ValueError(
                        f"User Config: {item} exceeds upper bounds ({self.bounds})"
                    )
            if self.bounds and "lower" in self.bounds:
                if self.bounds["lower"] > item:
                    raise ValueError(
                        f"User Config: {item} exceeds lower bounds ({self.bounds})"
                    )
        else:
            if self.bounds:
                raise ValueError(
                    f"User Config: {item} is not a valid type for bounds checking (int/float)"
                )

    def _check_choices(self) -> None:
        if not self.choices or not self.value:
            return

        value_list = (
            list(self.value.keys()) if type(self.value) is dict else [self.value]
        )

        if isinstance(self.choices, list):
            for value in value_list:
                if value not in self.choices:
                    raise ValueError(
                        f"User Config: {value} not in list of choices: f{self.choices}"
                    )
        elif issubclass(self.choices, Enum):
            for value in value_list:
                if not isinstance(value, Enum):
                    raise ValueError(
                        f"User Config: {value} is not in list of choices: f{self.choices}"
                    )
                if value.name not in [e.name for e in self.choices]:
                    raise ValueError(
                        f"User Config: {value.name} not in list of choices: f{self.choices}"
                    )

    def __setattr__(self, name, value):
        self.__dict__[name] = value

        if name == "value" and value is not None and value != "":
            self.is_set_by_user = True
            self._check_bounds()
            self._check_choices()

    def __eq__(self, other):
        if not isinstance(other, ConfigField):
            return False
        return self.__dict__ == other.__dict__
