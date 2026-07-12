// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Descriptor-generated Arrow schema v1 authorities.
//!
//! Exact checked-in descriptor bytes own field order, names, nullability,
//! nested child layout, enum vocabularies, and fingerprints. Rust code only
//! interprets that closed descriptor language; generated Arrow values are not
//! an independent schema authority.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use serde_json::Value;

use crate::{
    CanonicalDescriptor, DescriptorError, Digest, LogicalField, LogicalSchema, LogicalType, TableId,
};

const SCHEMA_VERSION: &str = "1.0";
const SCHEMA_FINGERPRINT_DOMAIN: &[u8] = b"aiperf.archive.arrow-schema.v1";

/// Canonical attempts-table descriptor.
pub const ATTEMPTS_ARROW_SCHEMA_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "attempts-arrow-schema-v1",
    include_bytes!("../descriptors/attempts-arrow-schema-v1.json"),
);

/// Canonical family-metadata-table descriptor.
pub const FAMILIES_ARROW_SCHEMA_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "families-arrow-schema-v1",
    include_bytes!("../descriptors/families-arrow-schema-v1.json"),
);

/// Canonical structured-samples-table descriptor.
pub const SAMPLES_ARROW_SCHEMA_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "samples-arrow-schema-v1",
    include_bytes!("../descriptors/samples-arrow-schema-v1.json"),
);

/// Canonical lifecycle-marker-table descriptor.
pub const MARKERS_ARROW_SCHEMA_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "markers-arrow-schema-v1",
    include_bytes!("../descriptors/markers-arrow-schema-v1.json"),
);

/// Canonical loss-range-table descriptor.
pub const LOSSES_ARROW_SCHEMA_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "losses-arrow-schema-v1",
    include_bytes!("../descriptors/losses-arrow-schema-v1.json"),
);

/// Canonical raw-reference-table descriptor.
pub const RAW_REFERENCES_ARROW_SCHEMA_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "raw-references-arrow-schema-v1",
    include_bytes!("../descriptors/raw-references-arrow-schema-v1.json"),
);

/// Every primary archive table descriptor in numeric table order.
pub const ALL_ARROW_SCHEMA_DESCRIPTORS_V1: &[(TableId, CanonicalDescriptor)] = &[
    (TableId::Attempts, ATTEMPTS_ARROW_SCHEMA_V1),
    (TableId::Families, FAMILIES_ARROW_SCHEMA_V1),
    (TableId::Samples, SAMPLES_ARROW_SCHEMA_V1),
    (TableId::Markers, MARKERS_ARROW_SCHEMA_V1),
    (TableId::Losses, LOSSES_ARROW_SCHEMA_V1),
    (TableId::RawReferences, RAW_REFERENCES_ARROW_SCHEMA_V1),
];

/// One generated Arrow schema bound to its exact checked-in descriptor.
#[derive(Clone, Debug)]
pub struct ArchiveTableSchemaV1 {
    table: TableId,
    table_name: &'static str,
    descriptor: CanonicalDescriptor,
    fingerprint: Digest,
    schema: SchemaRef,
    logical_schema: LogicalSchema,
}

impl ArchiveTableSchemaV1 {
    /// Returns the closed table ID.
    #[must_use]
    pub const fn table(&self) -> TableId {
        self.table
    }

    /// Returns the frozen table name used in metadata and object paths.
    #[must_use]
    pub const fn table_name(&self) -> &'static str {
        self.table_name
    }

    /// Returns the exact checked-in descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> CanonicalDescriptor {
        self.descriptor
    }

    /// Returns the Arrow-schema fingerprint over exact descriptor bytes.
    #[must_use]
    pub const fn fingerprint(&self) -> Digest {
        self.fingerprint
    }

    /// Returns the generated Arrow schema including mandatory metadata.
    #[must_use]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Returns the descriptor-equivalent logical-row schema.
    #[must_use]
    pub const fn logical_schema(&self) -> &LogicalSchema {
        &self.logical_schema
    }
}

/// Complete six-table schema registry generated as one validated unit.
#[derive(Clone, Debug)]
pub struct ArchiveSchemasV1 {
    tables: BTreeMap<TableId, ArchiveTableSchemaV1>,
}

impl ArchiveSchemasV1 {
    /// Parses and validates every checked-in descriptor.
    pub fn load() -> Result<Self, SchemaError> {
        let mut tables = BTreeMap::new();
        for (table, descriptor) in ALL_ARROW_SCHEMA_DESCRIPTORS_V1 {
            let schema = load_table(*table, *descriptor)?;
            if tables.insert(*table, schema).is_some() {
                return Err(SchemaError::DuplicateTable(*table));
            }
        }
        Ok(Self { tables })
    }

    /// Returns one table schema.
    pub fn table(&self, table: TableId) -> Result<&ArchiveTableSchemaV1, SchemaError> {
        self.tables
            .get(&table)
            .ok_or(SchemaError::MissingTable(table))
    }

    /// Iterates all schemas in numeric table order.
    pub fn iter(&self) -> impl Iterator<Item = &ArchiveTableSchemaV1> {
        self.tables.values()
    }
}

/// Returns the frozen lower-snake-case table name.
#[must_use]
pub const fn table_name(table: TableId) -> &'static str {
    match table {
        TableId::Attempts => "attempts",
        TableId::Families => "families",
        TableId::Samples => "samples",
        TableId::Markers => "markers",
        TableId::Losses => "losses",
        TableId::RawReferences => "raw_references",
    }
}

/// Decodes one frozen table name.
pub fn table_id(name: &str) -> Result<TableId, SchemaError> {
    match name {
        "attempts" => Ok(TableId::Attempts),
        "families" => Ok(TableId::Families),
        "samples" => Ok(TableId::Samples),
        "markers" => Ok(TableId::Markers),
        "losses" => Ok(TableId::Losses),
        "raw_references" => Ok(TableId::RawReferences),
        _ => Err(SchemaError::UnknownTableName(name.to_string())),
    }
}

/// Computes the normative schema fingerprint over exact descriptor bytes.
#[must_use]
pub fn arrow_schema_fingerprint(exact_descriptor_bytes: &[u8]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SCHEMA_FINGERPRINT_DOMAIN);
    hasher.update(&[0]);
    hasher.update(exact_descriptor_bytes);
    Digest::from_bytes(*hasher.finalize().as_bytes())
}

fn load_table(
    expected_table: TableId,
    descriptor: CanonicalDescriptor,
) -> Result<ArchiveTableSchemaV1, SchemaError> {
    descriptor.validate().map_err(SchemaError::Descriptor)?;
    let value: Value = serde_json::from_slice(descriptor.bytes()).map_err(SchemaError::Json)?;
    let object = value
        .as_object()
        .ok_or(SchemaError::InvalidDescriptor("root must be an object"))?;
    let authored_table = object
        .get("table")
        .and_then(Value::as_str)
        .ok_or(SchemaError::InvalidDescriptor("table must be a string"))?;
    let table = table_id(authored_table)?;
    if table != expected_table {
        return Err(SchemaError::TableMismatch {
            expected: expected_table,
            actual: table,
        });
    }
    if object.get("version").and_then(Value::as_str) != Some(SCHEMA_VERSION) {
        return Err(SchemaError::InvalidDescriptor(
            "schema version must be exactly 1.0",
        ));
    }
    let aliases = object
        .get("aliases")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let authored_fields = object
        .get("fields")
        .and_then(Value::as_array)
        .ok_or(SchemaError::InvalidDescriptor("fields must be an array"))?;
    let mut seen = BTreeSet::new();
    let mut fields = Vec::with_capacity(authored_fields.len());
    for field in authored_fields {
        let parsed = parse_field(field, &aliases, &mut Vec::new())?;
        if !seen.insert(parsed.name().to_string()) {
            return Err(SchemaError::DuplicateField(parsed.name().to_string()));
        }
        fields.push(parsed);
    }
    let fingerprint = arrow_schema_fingerprint(descriptor.bytes());
    let table_name = table_name(table);
    let metadata = HashMap::from([
        (
            "aiperf.archive.schema_fingerprint".to_string(),
            fingerprint.to_hex(),
        ),
        (
            "aiperf.archive.schema_version".to_string(),
            SCHEMA_VERSION.to_string(),
        ),
        ("aiperf.archive.table".to_string(), table_name.to_string()),
    ]);
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    let logical_schema = LogicalSchema {
        table,
        fingerprint,
        fields: schema
            .fields()
            .iter()
            .map(|field| logical_field(field))
            .collect::<Result<_, _>>()?,
    };
    Ok(ArchiveTableSchemaV1 {
        table,
        table_name,
        descriptor,
        fingerprint,
        schema,
        logical_schema,
    })
}

fn parse_field(
    value: &Value,
    aliases: &serde_json::Map<String, Value>,
    alias_stack: &mut Vec<String>,
) -> Result<Field, SchemaError> {
    let items =
        value
            .as_array()
            .filter(|items| items.len() == 3)
            .ok_or(SchemaError::InvalidDescriptor(
                "field must be [name, nullable, type]",
            ))?;
    let name =
        items[0]
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or(SchemaError::InvalidDescriptor(
                "field name must be non-empty UTF-8",
            ))?;
    let nullable = items[1].as_bool().ok_or(SchemaError::InvalidDescriptor(
        "field nullability must be boolean",
    ))?;
    let data_type = parse_type(&items[2], aliases, alias_stack)?;
    Ok(Field::new(name, data_type, nullable))
}

fn parse_type(
    value: &Value,
    aliases: &serde_json::Map<String, Value>,
    alias_stack: &mut Vec<String>,
) -> Result<DataType, SchemaError> {
    if let Some(name) = value.as_str() {
        return match name {
            "bool" => Ok(DataType::Boolean),
            "i64" => Ok(DataType::Int64),
            "u16" => Ok(DataType::UInt16),
            "u64" => Ok(DataType::UInt64),
            "utf8" => Ok(DataType::Utf8),
            "uuid" => Ok(DataType::FixedSizeBinary(16)),
            "digest" => Ok(DataType::FixedSizeBinary(32)),
            "epoch_ns" => Ok(DataType::Decimal128(38, 0)),
            "string_map" => Ok(string_map_type()),
            "archive_number" => Ok(archive_number_type()),
            "source_timestamp" => Ok(timestamp_type()),
            "created_timestamp" => Ok(timestamp_type()),
            "exemplar" => Ok(exemplar_type()),
            alias => {
                let definition = aliases
                    .get(alias)
                    .ok_or_else(|| SchemaError::UnknownType(alias.to_string()))?;
                if alias_stack.iter().any(|entry| entry == alias) {
                    return Err(SchemaError::AliasCycle(alias.to_string()));
                }
                alias_stack.push(alias.to_string());
                let parsed = parse_type(definition, aliases, alias_stack);
                alias_stack.pop();
                parsed
            }
        };
    }
    let object = value.as_object().filter(|object| object.len() == 1).ok_or(
        SchemaError::InvalidDescriptor("complex type must have exactly one discriminator"),
    )?;
    if let Some(values) = object.get("enum8") {
        let values = values
            .as_array()
            .ok_or(SchemaError::InvalidDescriptor("enum8 must be an array"))?;
        if values.is_empty() || values.len() > 128 {
            return Err(SchemaError::InvalidDescriptor(
                "enum8 requires one through 128 values",
            ));
        }
        let mut seen = BTreeSet::new();
        for value in values {
            let value = value.as_str().filter(|value| !value.is_empty()).ok_or(
                SchemaError::InvalidDescriptor("enum8 values must be non-empty strings"),
            )?;
            if !seen.insert(value) {
                return Err(SchemaError::DuplicateEnumValue(value.to_string()));
            }
        }
        return Ok(enum8_type());
    }
    if let Some(element) = object.get("list") {
        let element_type = parse_type(element, aliases, alias_stack)?;
        return Ok(DataType::List(Arc::new(Field::new(
            "item",
            element_type,
            false,
        ))));
    }
    if let Some(fields) = object.get("struct") {
        let fields = fields
            .as_array()
            .ok_or(SchemaError::InvalidDescriptor("struct must be an array"))?;
        let mut parsed = Vec::with_capacity(fields.len());
        let mut seen = BTreeSet::new();
        for field in fields {
            let field = parse_field(field, aliases, alias_stack)?;
            if !seen.insert(field.name().to_string()) {
                return Err(SchemaError::DuplicateField(field.name().to_string()));
            }
            parsed.push(field);
        }
        return Ok(DataType::Struct(Fields::from(parsed)));
    }
    Err(SchemaError::InvalidDescriptor(
        "unknown complex type discriminator",
    ))
}

fn enum8_type() -> DataType {
    DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
}

fn string_map_type() -> DataType {
    let entries = DataType::Struct(Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
    ]));
    DataType::Map(Arc::new(Field::new("entries", entries, false)), true)
}

fn archive_number_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("kind", enum8_type(), false),
        Field::new("source_lexeme", DataType::Utf8, true),
        Field::new("finite_value", DataType::Float64, true),
        Field::new("exact_u64", DataType::UInt64, true),
        Field::new("f64_status", enum8_type(), false),
    ]))
}

fn timestamp_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("lexeme", DataType::Utf8, true),
        Field::new("normalized_unix_ns", DataType::Decimal128(38, 0), true),
        Field::new("status", enum8_type(), false),
    ]))
}

fn exemplar_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("labels", string_map_type(), false),
        Field::new("value", archive_number_type(), false),
        Field::new("timestamp", timestamp_type(), false),
    ]))
}

fn logical_field(field: &Field) -> Result<LogicalField, SchemaError> {
    Ok(LogicalField {
        nullable: field.is_nullable(),
        logical_type: logical_type(field.data_type())?,
    })
}

fn logical_type(data_type: &DataType) -> Result<LogicalType, SchemaError> {
    match data_type {
        DataType::Boolean => Ok(LogicalType::Bool),
        DataType::Int8 => Ok(LogicalType::I8),
        DataType::Int16 => Ok(LogicalType::I16),
        DataType::Int32 => Ok(LogicalType::I32),
        DataType::Int64 => Ok(LogicalType::I64),
        DataType::UInt8 => Ok(LogicalType::U8),
        DataType::UInt16 => Ok(LogicalType::U16),
        DataType::UInt32 => Ok(LogicalType::U32),
        DataType::UInt64 => Ok(LogicalType::U64),
        DataType::Float64 => Ok(LogicalType::Float64),
        DataType::Decimal128(38, 0) => Ok(LogicalType::Decimal128),
        DataType::Utf8 => Ok(LogicalType::Utf8),
        DataType::Binary => Ok(LogicalType::Binary),
        DataType::FixedSizeBinary(width) => usize::try_from(*width)
            .map(LogicalType::FixedBinary)
            .map_err(|_| SchemaError::UnsupportedArrowType(data_type.clone())),
        DataType::Dictionary(index, value)
            if index.as_ref() == &DataType::Int8 && value.as_ref() == &DataType::Utf8 =>
        {
            Ok(LogicalType::Enum)
        }
        DataType::Struct(fields) => Ok(LogicalType::Struct(
            fields
                .iter()
                .map(|field| logical_field(field))
                .collect::<Result<_, _>>()?,
        )),
        DataType::List(element) => Ok(LogicalType::List(Box::new(logical_field(element)?))),
        DataType::Map(entries, true) if entries.name() == "entries" => Ok(LogicalType::StringMap),
        _ => Err(SchemaError::UnsupportedArrowType(data_type.clone())),
    }
}

/// Invalid checked-in schema descriptor or unsupported generated Arrow type.
#[derive(Debug)]
pub enum SchemaError {
    /// A canonical descriptor failed validation.
    Descriptor(DescriptorError),
    /// Descriptor bytes were not valid serde JSON after canonical validation.
    Json(serde_json::Error),
    /// Descriptor shape disagrees with the closed v1 language.
    InvalidDescriptor(&'static str),
    /// Two descriptors claimed one table.
    DuplicateTable(TableId),
    /// A required table descriptor is absent.
    MissingTable(TableId),
    /// Authored table name is not a v1 table.
    UnknownTableName(String),
    /// Registered and authored table identities disagree.
    TableMismatch {
        /// Registered table.
        expected: TableId,
        /// Authored descriptor table.
        actual: TableId,
    },
    /// One struct or table repeats a field name.
    DuplicateField(String),
    /// An enum repeats one logical string.
    DuplicateEnumValue(String),
    /// A type token is neither built in nor a declared alias.
    UnknownType(String),
    /// Aliases recursively refer to one another.
    AliasCycle(String),
    /// A generated Arrow type has no v1 logical-row encoding.
    UnsupportedArrowType(DataType),
}

impl Display for SchemaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(error) => {
                write!(formatter, "invalid Arrow schema descriptor: {error}")
            }
            Self::Json(error) => write!(formatter, "invalid Arrow schema JSON: {error}"),
            Self::InvalidDescriptor(message) => {
                write!(formatter, "invalid Arrow schema descriptor: {message}")
            }
            Self::DuplicateTable(table) => write!(formatter, "duplicate schema for {table:?}"),
            Self::MissingTable(table) => write!(formatter, "missing schema for {table:?}"),
            Self::UnknownTableName(name) => write!(formatter, "unknown archive table {name:?}"),
            Self::TableMismatch { expected, actual } => write!(
                formatter,
                "schema table mismatch: expected {expected:?}, found {actual:?}"
            ),
            Self::DuplicateField(name) => write!(formatter, "duplicate schema field {name:?}"),
            Self::DuplicateEnumValue(value) => {
                write!(formatter, "duplicate Enum8 value {value:?}")
            }
            Self::UnknownType(name) => write!(formatter, "unknown schema type {name:?}"),
            Self::AliasCycle(name) => write!(formatter, "schema alias cycle at {name:?}"),
            Self::UnsupportedArrowType(data_type) => {
                write!(formatter, "unsupported Arrow type {data_type:?}")
            }
        }
    }
}

impl std::error::Error for SchemaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_six_descriptor_generated_schemas_have_exact_metadata_and_field_counts() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let expected = [
            (TableId::Attempts, 35),
            (TableId::Families, 21),
            (TableId::Samples, 25),
            (TableId::Markers, 22),
            (TableId::Losses, 27),
            (TableId::RawReferences, 11),
        ];
        for (table, field_count) in expected {
            let table_schema = schemas.table(table).unwrap();
            assert_eq!(table_schema.schema().fields().len(), field_count);
            assert_eq!(
                table_schema.schema().metadata()["aiperf.archive.table"],
                table_name(table)
            );
            assert_eq!(
                table_schema.schema().metadata()["aiperf.archive.schema_version"],
                "1.0"
            );
            assert_eq!(
                table_schema.schema().metadata()["aiperf.archive.schema_fingerprint"],
                table_schema.fingerprint().to_hex()
            );
        }
        assert_eq!(schemas.iter().count(), 6);
    }

    #[test]
    fn nested_aliases_generate_exact_number_timestamp_map_and_payload_layouts() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let samples = schemas.table(TableId::Samples).unwrap().schema();
        assert_eq!(
            samples.field_with_name("archive_id").unwrap().data_type(),
            &DataType::FixedSizeBinary(16)
        );
        assert!(matches!(
            samples.field_with_name("labels").unwrap().data_type(),
            DataType::Map(entries, true)
                if matches!(entries.data_type(), DataType::Struct(fields)
                    if fields[0].name() == "key" && fields[1].name() == "value")
        ));
        let DataType::Struct(payload) = samples.field_with_name("payload").unwrap().data_type()
        else {
            panic!("payload must be a struct")
        };
        assert_eq!(
            payload
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            [
                "scalar",
                "counter",
                "stateset",
                "info",
                "histogram",
                "summary"
            ]
        );
        assert!(payload.iter().all(|field| field.is_nullable()));
    }

    #[test]
    fn logical_schema_is_derived_from_the_same_arrow_descriptor() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        for schema in schemas.iter() {
            assert_eq!(schema.logical_schema().table, schema.table());
            assert_eq!(schema.logical_schema().fingerprint, schema.fingerprint());
            assert_eq!(
                schema.logical_schema().fields.len(),
                schema.schema().fields().len()
            );
        }
    }
}
