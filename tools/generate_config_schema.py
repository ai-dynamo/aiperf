# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate JSON Schema for AIPerf YAML configuration files.

This generator creates JSON schemas that enable:
- IDE autocompletion (VSCode, IntelliJ, etc.)
- Configuration validation in editors
- Documentation generation
- API client generation

The generated schema includes all discriminated unions, nested models,
and comprehensive descriptions from the Pydantic models.

Schema Generation Strategy:
    The config models use Pydantic @model_validator(mode="before") pre-validators
    to normalize simplified input forms. This generator enhances the schema to
    reflect these normalizations so IDEs can autocomplete simplified forms:

    1. Singular→Plural aliases (model→models, url→urls, etc.)
    2. Distribution types accepting int/float (isl: 512 → FixedDistribution)
    3. models accepting string/list[str] (models: "llama" → full ModelsAdvanced)
    4. Config objects accepting string URLs (server_metrics: "http://..." → full config)

Usage:
    python -m tools.generate_config_schema
    python -m tools.generate_config_schema --check
    python -m tools.generate_config_schema --verbose

Output:
    src/aiperf/config/schema/aiperf-config.schema.json
"""

from __future__ import annotations

import json
from pathlib import Path

from tools._core import (
    GeneratedFile,
    Generator,
    GeneratorResult,
    SchemaGenerationError,
    main,
    print_step,
)

# =============================================================================
# Paths
# =============================================================================

CONFIG_DIR = Path("src/aiperf/config")
SCHEMA_DIR = CONFIG_DIR / "schema"
SCHEMA_FILE = SCHEMA_DIR / "aiperf-config.schema.json"

# Schema metadata
SCHEMA_VERSION = "2.0.0"
SCHEMA_ID = "https://nvidia.github.io/aiperf/schemas/aiperf-config.schema.json"


# =============================================================================
# Pre-validator Schema Enhancements
# =============================================================================
# These mappings mirror the @model_validator(mode="before") pre-validators in the
# Pydantic models. The pre-validators normalize user input at runtime; these
# mappings ensure the JSON Schema also accepts the simplified forms for IDE support.

# Singular → Plural aliases handled by pre-validators
# Format: model_name → {singular_field: (plural_field, container_type)}
# container_type: "list" wraps in list, "dict" wraps with key "default"
SINGULAR_TO_PLURAL_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    # AIPerfConfig.normalize_before_validation handles: model, dataset
    # Note: 'phases' uses a different normalization pattern (type key detection, not singular/plural)
    "AIPerfConfig": {
        "model": ("models", "list"),
        "dataset": ("datasets", "dict"),
    },
    # EndpointConfig.normalize_before_validation handles: url
    "EndpointConfig": {
        "url": ("urls", "list"),
    },
    # ServerMetricsConfig.normalize_before_validation handles: url
    "ServerMetricsConfig": {
        "url": ("urls", "list"),
    },
    # GpuTelemetryConfig.normalize_before_validation handles: url
    "GpuTelemetryConfig": {
        "url": ("urls", "list"),
    },
}

# Models that accept simplified forms via pre-validators
# These models can accept string or other simplified input that gets normalized
SIMPLIFIED_INPUT_MODELS: dict[str, dict] = {
    # FixedDistribution.coerce_scalar: int/float → {type: fixed, value: X}
    "FixedDistribution": {
        "accepts": ["integer", "number"],
        "description": "Can be a number (interpreted as fixed value) or a distribution object with type field.",
    },
    # RampConfig via RampSpec: int/float/string → {duration, strategy}
    # _normalize_ramp accepts: 60, 30.0, "30s", "5m", "2h" → {duration: X}
    "RampConfig": {
        "accepts": ["integer", "number", "string"],
        "description": "Can be a duration (number in seconds or string like '30s', '5m', '2h') or {duration, strategy} object.",
    },
    # ServerMetricsConfig.normalize_before_validation: string URL → full config
    "ServerMetricsConfig": {
        "accepts": ["string"],
        "description": "Can be a URL string or full config object.",
    },
    # GpuTelemetryConfig.normalize_before_validation: string URL → full config
    "GpuTelemetryConfig": {
        "accepts": ["string"],
        "description": "Can be a URL string or full config object.",
    },
}

# Fields that accept duration strings via DurationSpec type alias
# These fields use BeforeValidator(_normalize_duration) which parses:
# - Numbers: 30, 5.5 (interpreted as seconds)
# - Strings: "30s", "5m", "2h", "30 sec", "5 min"
DURATION_STRING_FIELDS: set[str] = {
    "duration",  # Phase stop condition: run for this duration
    "grace_period",  # Phase: wait time after stop condition before force-stopping
    "slice_duration",  # Artifacts: time slice duration for trend analysis
}

# Jinja2 template pattern for schema validation
# Matches strings containing {{ ... }} syntax
# Examples: "{{ concurrency }}", "{{ variables.target * 2 }}"
JINJA2_TEMPLATE_PATTERN = r".*\{\{.*\}\}.*"

# Environment variable pattern for schema validation
# Matches strings containing ${VAR} or ${VAR:default} syntax
# Examples: "${CONCURRENCY}", "${RATE:100}", "${ISL:512}"
ENV_VAR_PATTERN = r".*\$\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\}.*"

# Fields to EXCLUDE from Jinja2 template support
# These are fields where Jinja2 doesn't make sense (e.g., discriminator fields, enums)
JINJA2_EXCLUDED_FIELDS: set[str] = {
    "type",  # Discriminator field for unions (phases, rate, concurrency phase types)
}

# $ref types that accept numeric input via pre-validators
# These are the only $ref types that should get Jinja2 support
JINJA2_NUMERIC_REF_TYPES: set[str] = {
    "FixedDistribution",  # Accepts int/float via coerce_scalar
    "RampConfig",  # Accepts int/float/string via _normalize_ramp
}

# Fields that support sweep lists (scalar field accepts list for auto-expansion).
# Mirrors MAGIC_LIST_FIELDS in aiperf.config.sweep - sweep expansion strips the
# list before Pydantic validation, so the models stay as scalar types.
SWEEP_LIST_FIELDS: set[str] = {
    "level",
    "concurrency",
    "rate",
    "count",
    "requests",
    "time",
    "mean",
}


# =============================================================================
# Schema Generation
# =============================================================================


class ConfigSchemaGenerator(Generator):
    """Generator for AIPerf config JSON Schema."""

    name = "Config Schema Generator"
    description = "Generate JSON Schema for AIPerf YAML configuration files"

    def generate(self) -> GeneratorResult:
        """Generate the JSON schema from AIPerfConfig Pydantic model."""
        # Import config to get the models
        try:
            from aiperf.config import AIPerfConfig
        except ImportError as e:
            raise SchemaGenerationError(
                "Failed to import config models",
                {
                    "error": str(e),
                    "hint": "Run: uv pip install -e .",
                },
            ) from e

        if self.verbose:
            print_step("Generating JSON schema from AIPerfConfig model")

        # Generate the base schema from Pydantic
        try:
            schema = AIPerfConfig.model_json_schema(mode="serialization")
        except Exception as e:
            raise SchemaGenerationError(
                "Failed to generate JSON schema",
                {"error": str(e)},
            ) from e

        if self.verbose:
            print_step(
                f"Generated schema with {len(schema.get('$defs', {}))} definitions"
            )

        # Enhance schema with metadata
        enhanced_schema = self._enhance_schema(schema)

        # Fix discriminated unions to use proper JSON Schema discriminators
        discriminator_count = self._fix_discriminated_unions(enhanced_schema)
        if self.verbose and discriminator_count > 0:
            print_step(f"Fixed {discriminator_count} discriminated unions")

        # Add singular field aliases (model → models, url → urls, etc.)
        alias_count = self._add_singular_field_aliases(enhanced_schema)
        if self.verbose and alias_count > 0:
            print_step(f"Added {alias_count} singular field aliases")

        # Add pre-validator simplified form support (FixedDistribution accepts int/float, etc.)
        prevalidator_count = self._add_prevalidator_simplified_forms(enhanced_schema)
        if self.verbose and prevalidator_count > 0:
            print_step(f"Added {prevalidator_count} pre-validator simplified forms")

        # Add duration string support (duration: "30s", grace_period: "5m", etc.)
        duration_count = self._add_duration_string_support(enhanced_schema)
        if self.verbose and duration_count > 0:
            print_step(f"Added duration string support to {duration_count} fields")

        # Add models field simplified forms (string/list[str] → ModelsAdvanced)
        self._add_models_simplified_forms(enhanced_schema)
        if self.verbose:
            print_step("Added models field simplified forms")

        # Add phases field shorthand form (single PhaseConfig → dict)
        self._add_phases_shorthand_form(enhanced_schema)
        if self.verbose:
            print_step("Added phases field shorthand form support")

        # Add Jinja2 template support to numeric fields
        jinja2_count = self._add_jinja2_template_support(enhanced_schema)
        if self.verbose and jinja2_count > 0:
            print_step(f"Added Jinja2 template support to {jinja2_count} fields")

        # Add sweep list support to sweepable numeric fields
        sweep_count = self._add_sweep_list_support(enhanced_schema)
        if self.verbose and sweep_count > 0:
            print_step(f"Added sweep list support to {sweep_count} fields")

        # Serialize with proper formatting
        content = json.dumps(enhanced_schema, indent=2, ensure_ascii=False) + "\n"

        if self.verbose:
            print_step(f"Schema size: {len(content):,} bytes")

        return GeneratorResult(
            files=[GeneratedFile(SCHEMA_FILE, content)],
            summary=f"Generated config schema v{SCHEMA_VERSION}",
        )

    def _enhance_schema(self, schema: dict) -> dict:
        """Enhance the Pydantic schema with additional metadata."""
        # Add JSON Schema draft version and ID
        enhanced = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": SCHEMA_ID,
            **schema,
        }

        # Update title and description
        enhanced["title"] = "AIPerf Configuration Schema"
        enhanced["description"] = (
            "JSON Schema for AIPerf YAML configuration files. "
            "This schema enables IDE autocompletion and validation for AIPerf benchmark configurations. "
            f"Schema version: {SCHEMA_VERSION}"
        )

        # Add schema metadata
        enhanced["$comment"] = (
            "Auto-generated by tools/generate_config_schema.py. Do not edit manually."
        )

        # Add YAML language server compatibility
        # This helps VSCode YAML extension recognize the schema
        if "x-taplo" not in enhanced:
            enhanced["x-taplo"] = {
                "initKeys": ["models", "endpoint", "datasets", "phases"]
            }

        return enhanced

    def _fix_discriminated_unions(self, schema: dict) -> int:
        """
        Fix discriminated unions to work with JSON Schema validators.

        Pydantic generates `oneOf` arrays for discriminated unions, but JSON Schema
        validators require exactly one match. When types have optional discriminator
        fields with defaults, data can match multiple schemas causing validation errors.

        This method:
        1. Converts `oneOf` to `anyOf` for discriminated unions (more lenient)
        2. Adds discriminator mappings for IDEs that support them (VSCode YAML, etc.)

        Args:
            schema: The JSON schema to modify (in place).

        Returns:
            Number of discriminated unions fixed.
        """
        defs = schema.get("$defs", {})
        fixed_count = 0

        def get_discriminator_value(ref: str) -> tuple[str, str] | None:
            """Get the discriminator field name and const value from a $ref."""
            # Extract def name from "#/$defs/LoadPhase" -> "LoadPhase"
            if not ref.startswith("#/$defs/"):
                return None

            def_name = ref[len("#/$defs/") :]
            def_schema = defs.get(def_name, {})
            properties = def_schema.get("properties", {})

            # Look for a property with a const value (discriminator field)
            for prop_name, prop_schema in properties.items():
                if "const" in prop_schema:
                    return prop_name, prop_schema["const"]

            return None

        def fix_one_of(parent: dict) -> bool:
            """Fix a oneOf array to use anyOf with discriminator if applicable."""
            one_of = parent.get("oneOf")
            if not isinstance(one_of, list) or len(one_of) < 2:
                return False

            # Check if all refs have the same discriminator field with different values
            discriminator_field = None
            mapping: dict[str, str] = {}

            for item in one_of:
                ref = item.get("$ref")
                if not ref:
                    # Not all items are $refs, can't add discriminator
                    return False

                result = get_discriminator_value(ref)
                if not result:
                    return False

                field_name, const_value = result

                if discriminator_field is None:
                    discriminator_field = field_name
                elif discriminator_field != field_name:
                    # Different discriminator fields, can't use single discriminator
                    return False

                mapping[const_value] = ref

            if not mapping or not discriminator_field:
                return False

            # Convert oneOf to anyOf (more lenient for validators)
            parent["anyOf"] = parent.pop("oneOf")

            # Add discriminator for IDEs that support it (VSCode YAML, etc.)
            parent["discriminator"] = {
                "propertyName": discriminator_field,
                "mapping": mapping,
            }
            return True

        def walk_schema(obj: dict | list, path: str = "") -> None:
            """Recursively walk schema and fix oneOf arrays."""
            nonlocal fixed_count

            if isinstance(obj, dict):
                # Check for oneOf at this level
                if "oneOf" in obj and fix_one_of(obj):
                    fixed_count += 1

                # Recurse into nested structures
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        walk_schema(value, f"{path}.{key}")

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        walk_schema(item, f"{path}[{i}]")

        # Walk all definitions
        for def_name, def_schema in defs.items():
            walk_schema(def_schema, f"$defs.{def_name}")

        # Walk top-level properties
        if "properties" in schema:
            walk_schema(schema["properties"], "properties")

        return fixed_count

    def _add_singular_field_aliases(self, schema: dict) -> int:
        """
        Add singular field aliases to the schema for user convenience.

        For fields that support singular shorthand (like model → models),
        add the singular form as an alternative property name in the schema.
        This allows IDEs to autocomplete both `model:` and `models:`.

        The runtime model validators (mode="before") handle the actual
        conversion from singular to plural form.

        Args:
            schema: The JSON schema to modify (in place).

        Returns:
            Number of aliases added.
        """
        defs = schema.get("$defs", {})
        alias_count = 0

        for model_name, aliases in SINGULAR_TO_PLURAL_ALIASES.items():
            # Find the model definition
            model_schema = defs.get(model_name)
            if not model_schema:
                # Check if it's the root schema (AIPerfConfig)
                if model_name == "AIPerfConfig":
                    model_schema = schema
                else:
                    continue

            properties = model_schema.get("properties", {})
            required = model_schema.get("required", [])

            for singular, (plural, _container_type) in aliases.items():
                if plural not in properties:
                    continue

                # Get the plural field's schema
                plural_schema = properties[plural]

                # Create the singular alias with same schema but marked as deprecated-ish
                # Use the items schema for list types, or the additionalProperties for dict types
                singular_schema = self._create_singular_schema(plural_schema, plural)

                if singular_schema:
                    properties[singular] = singular_schema
                    alias_count += 1

                    # Remove plural from required since singular is now an alternative
                    if plural in required:
                        required.remove(plural)

                        # Add anyOf constraint: at least one of singular or plural required
                        # This ensures schema validation catches missing url/urls
                        any_of_required = model_schema.get("anyOf", [])
                        any_of_required.extend(
                            [
                                {"required": [singular]},
                                {"required": [plural]},
                            ]
                        )
                        model_schema["anyOf"] = any_of_required

        return alias_count

    def _create_singular_schema(
        self, plural_schema: dict, plural_name: str
    ) -> dict | None:
        """
        Create a schema for a singular field alias.

        The singular schema allows either:
        - A single item (for list fields)
        - A single object (for dict fields)

        The runtime normalizer will convert these to their plural form.

        Args:
            plural_schema: The schema for the plural field.
            plural_name: The plural field name (e.g., "models").

        Returns:
            Schema for the singular field, or None if not applicable.
        """
        # Extract description and add note about singular form
        base_description = plural_schema.get("description", "")
        singular_description = (
            f"Shorthand for '{plural_name}'. "
            f"Accepts a single value which will be converted to a list/dict. "
            f"{base_description}"
        )

        # For array types, allow either a single item or an array
        if plural_schema.get("type") == "array":
            items_schema = plural_schema.get("items", {})
            return {
                "description": singular_description,
                "oneOf": [
                    items_schema,  # Single item
                    plural_schema,  # Array (for flexibility)
                ],
                "x-singular-alias-of": plural_name,
            }

        # For object/dict types with additionalProperties, allow a single value
        if (
            plural_schema.get("type") == "object"
            and "additionalProperties" in plural_schema
        ):
            value_schema = plural_schema.get("additionalProperties", {})
            return {
                "description": singular_description,
                "oneOf": [
                    value_schema,  # Single value (will be wrapped with key "default")
                    plural_schema,  # Full dict (for flexibility)
                ],
                "x-singular-alias-of": plural_name,
            }

        # For $ref types (like ModelsConfig), allow the ref or simplified form
        if "$ref" in plural_schema:
            return {
                "description": singular_description,
                "oneOf": [
                    {"type": "string"},  # Single model name
                    {"type": "array", "items": {"type": "string"}},  # List of names
                    plural_schema,  # Full ModelsConfig
                ],
                "x-singular-alias-of": plural_name,
            }

        # For anyOf types, extract and allow singular
        if "anyOf" in plural_schema:
            # Find the array variant if present
            for variant in plural_schema["anyOf"]:
                if variant.get("type") == "array":
                    items_schema = variant.get("items", {})
                    return {
                        "description": singular_description,
                        "oneOf": [
                            items_schema,
                            plural_schema,
                        ],
                        "x-singular-alias-of": plural_name,
                    }

        return None

    def _add_prevalidator_simplified_forms(self, schema: dict) -> int:
        """
        Add simplified form support for models with pre-validators.

        Many Pydantic models use @model_validator(mode="before") to accept
        simplified input forms. This method enhances the schema to reflect
        these normalizations for IDE autocompletion.

        Currently handles:
        - FixedDistribution: accepts int/float in addition to {mean, stddev} object
        - ServerMetricsConfig: accepts string URL in addition to full config
        - GpuTelemetryConfig: accepts string URL in addition to full config

        Args:
            schema: The JSON schema to modify (in place).

        Returns:
            Number of definitions enhanced.
        """
        defs = schema.get("$defs", {})
        enhanced_count = 0

        for model_name, config in SIMPLIFIED_INPUT_MODELS.items():
            if model_name not in defs:
                continue

            model_schema = defs[model_name]
            accepts = config["accepts"]
            description = config.get("description", "")

            # Create alternatives for the simplified forms
            alternatives = []

            # Add the original schema
            original_schema = {k: v for k, v in model_schema.items()}
            alternatives.append(original_schema)

            # Add simplified type alternatives
            for type_name in accepts:
                alternatives.append({"type": type_name})

            # Replace the model schema with anyOf allowing all forms
            # Preserve the title and description
            defs[model_name] = {
                "title": model_schema.get("title", model_name),
                "description": f"{model_schema.get('description', '')} {description}".strip(),
                "anyOf": alternatives,
            }
            enhanced_count += 1

        return enhanced_count

    def _add_duration_string_support(self, schema: dict) -> int:
        """
        Add string type support to duration fields.

        Fields like `duration` and `grace_period` use DurationSpec type which
        accepts strings like "30s", "5m", "2h" via BeforeValidator. This method
        enhances the schema to allow string input for these fields.

        Args:
            schema: The JSON schema to modify (in place).

        Returns:
            Number of fields enhanced.
        """
        enhanced_count = 0

        def enhance_duration_field(field_schema: dict) -> bool:
            """Add string type to a duration field's anyOf."""
            if "anyOf" not in field_schema:
                return False

            any_of = field_schema["anyOf"]

            # Check if already has string type
            has_string = any(
                item.get("type") == "string"
                for item in any_of
                if isinstance(item, dict)
            )
            if has_string:
                return False

            # Check if this looks like a duration field (has number type)
            has_number = any(
                item.get("type") == "number"
                for item in any_of
                if isinstance(item, dict)
            )
            if not has_number:
                return False

            # Add string type for duration strings like "30s", "5m", "2h"
            any_of.insert(
                0,
                {
                    "type": "string",
                    "pattern": r"^\d+(?:\.\d+)?\s*(?:s|sec|m|min|h|hr|hour)?$",
                    "description": "Duration string (e.g., '30s', '5m', '2h').",
                },
            )

            # Update description if present
            if "description" in field_schema:
                field_schema["description"] = (
                    f"{field_schema['description']} "
                    "Accepts number (seconds) or string like '30s', '5m', '2h'."
                )

            return True

        def walk_properties(obj: dict, path: str = "") -> None:
            """Recursively walk schema and enhance duration fields."""
            nonlocal enhanced_count

            if not isinstance(obj, dict):
                return

            # Check properties at this level
            properties = obj.get("properties", {})
            for field_name, field_schema in properties.items():
                if (
                    field_name in DURATION_STRING_FIELDS
                    and isinstance(field_schema, dict)
                    and enhance_duration_field(field_schema)
                ):
                    enhanced_count += 1

                # Recurse into nested properties
                if isinstance(field_schema, dict):
                    walk_properties(field_schema, f"{path}.{field_name}")

            # Check $defs for nested models
            for def_name, def_schema in obj.get("$defs", {}).items():
                if isinstance(def_schema, dict):
                    walk_properties(def_schema, f"$defs.{def_name}")

            # Check items in arrays
            items = obj.get("items")
            if isinstance(items, dict):
                walk_properties(items, f"{path}[]")

            # Check anyOf/oneOf variants
            for key in ("anyOf", "oneOf"):
                variants = obj.get(key, [])
                for i, variant in enumerate(variants):
                    if isinstance(variant, dict):
                        walk_properties(variant, f"{path}.{key}[{i}]")

            # Check additionalProperties
            additional = obj.get("additionalProperties")
            if isinstance(additional, dict):
                walk_properties(additional, f"{path}.*")

        walk_properties(schema)
        return enhanced_count

    def _add_models_simplified_forms(self, schema: dict) -> None:
        """
        Add simplified form support for the 'models' field in AIPerfConfig.

        The AIPerfConfig.normalize_before_validation pre-validator accepts:
        - A string: "llama" → ModelsAdvanced with single item
        - A list of strings: ["llama", "mistral"] → ModelsAdvanced with items
        - A ModelsAdvanced object: passed through as-is

        This method updates the schema to reflect these options.

        Args:
            schema: The JSON schema to modify (in place).
        """
        # Get the properties from the root schema (AIPerfConfig)
        properties = schema.get("properties", {})
        if "models" not in properties:
            return

        models_schema = properties["models"]

        # Get the original $ref or schema
        original_ref = models_schema.get("$ref")
        original_description = models_schema.get("description", "")

        if not original_ref:
            # Already modified or unexpected structure
            return

        # Create a new schema that accepts string, list[str], or ModelsAdvanced
        enhanced_description = (
            f"{original_description} "
            "Accepts a model name string, list of model names, or full ModelsAdvanced config."
        ).strip()

        properties["models"] = {
            "description": enhanced_description,
            "oneOf": [
                # Single model name string
                {
                    "type": "string",
                    "description": "Single model name (normalized to ModelsAdvanced).",
                },
                # List of model name strings
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "List of model names (normalized to ModelsAdvanced).",
                },
                # Full ModelsAdvanced config
                {"$ref": original_ref},
            ],
        }

    def _add_phases_shorthand_form(self, schema: dict) -> None:
        """
        Add shorthand form support for the 'phases' field in AIPerfConfig.

        The AIPerfConfig.normalize_before_validation pre-validator accepts:
        - A single PhaseConfig with 'type' key: {type: concurrency, ...}
          → Normalized to {default: {type: concurrency, ...}}
        - A dict of named phases: {name: {type: concurrency, ...}}
          → Passed through as-is

        This method updates the schema to accept both forms.

        Args:
            schema: The JSON schema to modify (in place).
        """
        # Get the properties from the root schema (AIPerfConfig)
        properties = schema.get("properties", {})
        if "phases" not in properties:
            return

        phases_schema = properties["phases"]

        # Get the current schema (should be object with additionalProperties)
        original_description = phases_schema.get("description", "")

        # The PhaseConfig is a discriminated union, get its definition
        # We need to reference it for the shorthand form
        defs = schema.get("$defs", {})
        phase_config_ref = "#/$defs/PhaseConfig"

        # Check if PhaseConfig exists
        if "PhaseConfig" not in defs:
            return

        # Create a new schema that accepts either:
        # 1. A single PhaseConfig (shorthand) - object with 'type' key
        # 2. A dict of named PhaseConfigs (original form)
        enhanced_description = (
            f"{original_description} "
            "Shorthand: use a single phase config directly (with 'type' key) "
            "which gets wrapped as {{default: config}}."
        ).strip()

        properties["phases"] = {
            "description": enhanced_description,
            "oneOf": [
                # Shorthand: single PhaseConfig with type discriminator
                {
                    "$ref": phase_config_ref,
                    "description": "Single phase config (normalized to {default: config}).",
                },
                # Original: dict of named phases
                phases_schema,
            ],
        }

    def _add_jinja2_template_support(self, schema: dict) -> int:
        """
        Add Jinja2 and env var string support to ALL numeric fields in the schema.

        Automatically processes all fields with numeric types (integer, number)
        or $ref to numeric types (like FixedDistribution), and modifies their schema
        to also accept strings matching:
        - Jinja2 template pattern: {{ ... }}
        - Environment variable pattern: ${VAR} or ${VAR:default}

        This enables IDE validation to accept configurations like:
            concurrency: "{{ variables.target_concurrency }}"
            isl: "${ISL:512}"
            slos:
              ttft: "{{ max_ttft }}"

        The runtime loader handles substitution via render_jinja2_templates()
        and substitute_env_vars().

        Args:
            schema: The JSON schema to modify (in place).

        Returns:
            Number of fields modified.
        """
        modified_count = 0

        # Template variants to add to numeric fields
        template_variants = [
            {
                "type": "string",
                "pattern": JINJA2_TEMPLATE_PATTERN,
                "description": "Jinja2 template (e.g., '{{ variable }}').",
            },
            {
                "type": "string",
                "pattern": ENV_VAR_PATTERN,
                "description": "Environment variable (e.g., '${VAR}' or '${VAR:default}').",
            },
        ]

        def has_template_variant(variants: list) -> bool:
            """Check if template variants already present in a list of schemas."""
            patterns = {JINJA2_TEMPLATE_PATTERN, ENV_VAR_PATTERN}
            return any(
                item.get("type") == "string" and item.get("pattern") in patterns
                for item in variants
            )

        def is_numeric_ref(ref: str) -> bool:
            """Check if a $ref points to a known numeric-accepting type."""
            # Extract type name from "#/$defs/FixedDistribution" -> "FixedDistribution"
            if ref.startswith("#/$defs/"):
                type_name = ref[len("#/$defs/") :]
                return type_name in JINJA2_NUMERIC_REF_TYPES
            return False

        def add_template_variants(field_schema: dict) -> dict | None:
            """
            Add Jinja2 and env var template string variants to a field schema.

            Returns modified schema or None if not applicable.
            """
            # Case 1: Simple type field (type: integer or type: number)
            if field_schema.get("type") in ("integer", "number"):
                return {
                    "oneOf": [
                        field_schema,  # Original type
                        *template_variants,  # Template strings
                    ],
                }

            # Case 2: Direct $ref to a known numeric-accepting type (e.g., FixedDistribution)
            if (
                "$ref" in field_schema
                and "oneOf" not in field_schema
                and "anyOf" not in field_schema
                and is_numeric_ref(field_schema["$ref"])
            ):
                return {
                    "oneOf": [
                        field_schema,  # Original $ref
                        *template_variants,  # Template strings
                    ],
                }

            # Case 3: Already has oneOf - only add if contains numeric types
            if "oneOf" in field_schema:
                one_of = field_schema["oneOf"]
                has_numeric = any(
                    item.get("type") in ("integer", "number")
                    or ("$ref" in item and is_numeric_ref(item["$ref"]))
                    for item in one_of
                )
                if has_numeric and not has_template_variant(one_of):
                    one_of.extend(template_variants)
                    return field_schema
                return None

            # Case 4: anyOf (e.g., nullable fields, or $ref with null)
            if "anyOf" in field_schema:
                any_of = field_schema["anyOf"]
                # Check if any variant is numeric or a known numeric $ref
                has_numeric = any(
                    item.get("type") in ("integer", "number")
                    or ("$ref" in item and is_numeric_ref(item["$ref"]))
                    for item in any_of
                )
                if not has_numeric:
                    return None

                if not has_template_variant(any_of):
                    any_of.extend(template_variants)
                    return field_schema
                return None

            return None

        def process_additional_properties(field_schema: dict) -> bool:
            """
            Add template support to additionalProperties of a dict field.

            For fields like slos: dict[str, float], modify the additionalProperties
            to accept either the original type or template strings.

            Returns True if modified.
            """
            # Handle anyOf containing object with additionalProperties
            if "anyOf" in field_schema:
                for variant in field_schema["anyOf"]:
                    if (
                        variant.get("type") == "object"
                        and "additionalProperties" in variant
                    ):
                        add_props = variant["additionalProperties"]
                        if isinstance(add_props, dict) and add_props.get("type") in (
                            "number",
                            "integer",
                        ):
                            # Wrap additionalProperties in oneOf
                            variant["additionalProperties"] = {
                                "oneOf": [
                                    add_props,
                                    *template_variants,
                                ],
                            }
                            return True

            # Handle direct object with additionalProperties
            if (
                field_schema.get("type") == "object"
                and "additionalProperties" in field_schema
            ):
                add_props = field_schema["additionalProperties"]
                if isinstance(add_props, dict) and add_props.get("type") in (
                    "number",
                    "integer",
                ):
                    field_schema["additionalProperties"] = {
                        "oneOf": [
                            add_props,
                            *template_variants,
                        ],
                    }
                    return True

            return False

        def process_properties(properties: dict) -> None:
            """Process properties dict and add Jinja2 support to all numeric fields."""
            nonlocal modified_count

            for prop_name, prop_schema in list(properties.items()):
                if not isinstance(prop_schema, dict):
                    continue

                # Skip excluded fields (discriminators, enums, etc.)
                if prop_name in JINJA2_EXCLUDED_FIELDS:
                    continue

                # Try additionalProperties first (for dict[str, numeric] fields like slos)
                if process_additional_properties(prop_schema):
                    prop_schema["x-jinja2-supported"] = True
                    modified_count += 1
                    continue

                # Try to add template variants to any numeric field
                modified_schema = add_template_variants(prop_schema)
                if modified_schema:
                    # Preserve metadata (description, title, default)
                    metadata = {
                        k: v
                        for k, v in prop_schema.items()
                        if k in ("description", "title", "default", "x-sweep-field")
                    }
                    # Merge metadata into modified schema
                    if "oneOf" in modified_schema and "oneOf" not in prop_schema:
                        # New oneOf was created, add metadata at top level
                        properties[prop_name] = {
                            **metadata,
                            **modified_schema,
                            "x-jinja2-supported": True,
                        }
                    else:
                        # Modified in place (anyOf/oneOf was updated)
                        prop_schema["x-jinja2-supported"] = True
                    modified_count += 1

        # Process all definitions
        for def_schema in schema.get("$defs", {}).values():
            if "properties" in def_schema:
                process_properties(def_schema["properties"])

        # Process top-level properties
        if "properties" in schema:
            process_properties(schema["properties"])

        return modified_count

    def _add_sweep_list_support(self, schema: dict) -> int:
        """
        Add list type support to sweepable numeric fields.

        Fields in SWEEP_LIST_FIELDS (e.g., concurrency, rate, requests) accept
        a list of numeric values for automatic sweep expansion. The sweep layer
        strips the list before Pydantic validation, so models stay as scalars.

        This makes the JSON Schema honest about what the YAML actually accepts:
            concurrency: 8          # single value
            concurrency: [8, 16, 32]  # sweep over values

        Args:
            schema: The JSON schema to modify (in place).

        Returns:
            Number of fields modified.
        """
        modified_count = 0

        def add_sweep_array(field_schema: dict, field_name: str) -> dict | None:
            """Add array variant to a sweepable field's schema."""
            # Determine the item type from the field's current schema
            item_type = None

            if field_schema.get("type") in ("integer", "number"):
                item_type = field_schema["type"]
            elif "oneOf" in field_schema:
                for variant in field_schema["oneOf"]:
                    if variant.get("type") in ("integer", "number"):
                        item_type = variant["type"]
                        break
            elif "anyOf" in field_schema:
                for variant in field_schema["anyOf"]:
                    if variant.get("type") in ("integer", "number"):
                        item_type = variant["type"]
                        break

            if not item_type:
                return None

            # Check if array variant already present
            for key in ("oneOf", "anyOf"):
                for variant in field_schema.get(key, []):
                    if variant.get("type") == "array":
                        return None

            array_variant = {
                "type": "array",
                "items": {"type": item_type},
                "minItems": 2,
                "description": f"List of values to sweep over for '{field_name}'.",
            }

            # Append to existing oneOf/anyOf, or wrap in oneOf
            if "oneOf" in field_schema:
                field_schema["oneOf"].append(array_variant)
                return field_schema
            elif "anyOf" in field_schema:
                field_schema["anyOf"].append(array_variant)
                return field_schema
            else:
                return {
                    "oneOf": [
                        field_schema,
                        array_variant,
                    ],
                }

        def process_properties(properties: dict) -> None:
            nonlocal modified_count

            for prop_name, prop_schema in list(properties.items()):
                if not isinstance(prop_schema, dict):
                    continue
                if prop_name not in SWEEP_LIST_FIELDS:
                    continue

                modified = add_sweep_array(prop_schema, prop_name)
                if modified is not None:
                    # Preserve metadata
                    if modified is not prop_schema:
                        metadata = {
                            k: v
                            for k, v in prop_schema.items()
                            if k
                            in ("description", "title", "default", "x-jinja2-supported")
                        }
                        properties[prop_name] = {**metadata, **modified}
                    prop_schema = properties[prop_name]
                    prop_schema["x-sweep-field"] = True
                    modified_count += 1

        # Process all definitions
        for def_schema in schema.get("$defs", {}).values():
            if "properties" in def_schema:
                process_properties(def_schema["properties"])

        # Process top-level properties
        if "properties" in schema:
            process_properties(schema["properties"])

        return modified_count


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    main(ConfigSchemaGenerator)
