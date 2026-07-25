// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit newtypes for native and display metric values, plus metric definitions.

use crate::metrics_core::units::{MetricValueType, Unit};
use std::collections::HashMap;
use std::sync::LazyLock;

// Re-export the O(1) exhaustive per-tag lookup so the unified `definitions`
// facade exposes both the tag-keyed and the id-keyed lookup from one path.
pub use crate::metrics_core::catalog::metric_definition;

/// A value in a metric's native (math/SLA) unit.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Native(pub f64);

/// A value in a metric's human display unit.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Display(pub f64);

impl Native {
    pub const fn new(v: f64) -> Self {
        Self(v)
    }
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl Display {
    pub const fn new(v: f64) -> Self {
        Self(v)
    }
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl std::ops::Add for Native {
    type Output = Native;
    fn add(self, o: Native) -> Native {
        Native(self.0 + o.0)
    }
}

impl std::ops::Sub for Native {
    type Output = Native;
    fn sub(self, o: Native) -> Native {
        Native(self.0 - o.0)
    }
}

/// Grouping bucket a metric definition belongs to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionGroup {
    /// The default (ungrouped) bucket.
    Default,
    /// A named group bucket.
    Named(&'static str),
}

/// A static description of a metric: identity, units, display, and SLA policy.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Definition {
    /// Stable identifier for the metric.
    pub id: &'static str,
    /// Full human-readable header.
    pub header: &'static str,
    /// Optional shortened header.
    pub short_header: Option<&'static str>,
    /// Whether the short header omits the unit suffix.
    pub short_header_hide_unit: bool,
    /// Native (math/SLA) unit of the metric.
    pub unit: Unit,
    /// Optional preferred display unit.
    pub display_unit: Option<Unit>,
    /// Optional ordering hint for display.
    pub display_order: Option<u32>,
    /// Grouping bucket.
    pub group: DefinitionGroup,
    /// Whether larger values are better for SLA comparisons.
    pub larger_is_better: bool,
    /// Shape of values emitted by the metric.
    pub value_type: MetricValueType,
    /// Alternate identifiers for the metric.
    pub aliases: &'static [&'static str],
    /// Version this metric was deprecated since, if any.
    pub deprecated_since: Option<&'static str>,
}

impl Definition {
    /// Returns the display unit, falling back to the native unit when unset.
    pub fn effective_display_unit(&self) -> Unit {
        self.display_unit.unwrap_or(self.unit)
    }

    /// Returns whether `value` satisfies `threshold` given the metric's direction.
    pub fn passes_threshold(&self, value: Native, threshold: Native) -> bool {
        if self.larger_is_better {
            value >= threshold
        } else {
            value <= threshold
        }
    }

    /// Formats a raw value according to the metric's value type.
    pub fn format_value(&self, value: f64) -> String {
        match self.value_type {
            MetricValueType::Int => format!("{}", value.round() as i64),
            _ => format!("{value:.2}"),
        }
    }
}

/// Registry of every [`Definition`] keyed by its public, namespaced id.
///
/// Built once (config/render-time only; never touched on a per-record path) from
/// the static metric `CATALOG`. Metric ids are namespaced `aiperf.<tag>` at
/// map-build time; each alias is inserted verbatim (metrics declare none today,
/// but the loop is wired). A later task registers analyzer definitions here —
/// their `def.id` is already fully namespaced (e.g. `analyzer.isl`), so they are
/// inserted under their own key with no extra prefix. See the marked insertion
/// point below.
static REGISTRY: LazyLock<HashMap<String, &'static Definition>> = LazyLock::new(|| {
    let mut map: HashMap<String, &'static Definition> = HashMap::new();
    for spec in crate::metrics_core::catalog::CATALOG.iter() {
        // Metric ids are the bare tag key (e.g. "request_latency"); namespace them.
        map.insert(format!("aiperf.{}", spec.def.id), &spec.def);
        // Aliases are inserted verbatim (full string). Metrics have none yet.
        for alias in spec.def.aliases {
            map.insert((*alias).to_string(), &spec.def);
        }
    }

    // Task 9 insertion point: register analyzer base-concept definitions here.
    // Their `def.id` is already fully namespaced (`analyzer.*`), so insert each
    // under `def.id` directly (plus any aliases), without the `aiperf.` prefix.
    for def in crate::dataset::analysis::ANALYZER_DEFINITIONS {
        map.insert(def.id.to_string(), def);
        for alias in def.aliases {
            map.insert((*alias).to_string(), def);
        }
    }

    map
});

/// Looks up a [`Definition`] by its exact public id (namespaced id or alias).
///
/// Config/render-time only — never called on a per-record or per-token path.
pub fn definition(id: &str) -> Option<&'static Definition> {
    REGISTRY.get(id).copied()
}

/// Resolves a concrete output name to its [`Definition`].
///
/// First tries an exact [`definition`] lookup. Otherwise, if `name` matches the
/// parameterized `turn<N>_<suffix>` shape (N = digits), it maps to the analyzer
/// base concept `analyzer.per_turn_<suffix>`. That base def is registered by a
/// later task, so the parameterized branch returns `None` until then — the rule
/// is implemented now regardless. Config/render-time only.
pub fn resolve(name: &str) -> Option<&'static Definition> {
    if let Some(def) = definition(name) {
        return Some(def);
    }
    if let Some(rest) = name.strip_prefix("turn") {
        if let Some(us) = rest.find('_') {
            let (digits, suffix) = (&rest[..us], &rest[us + 1..]);
            if !digits.is_empty()
                && digits.bytes().all(|b| b.is_ascii_digit())
                && !suffix.is_empty()
            {
                return definition(&format!("analyzer.per_turn_{suffix}"));
            }
        }
    }
    None
}

/// Returns every registered [`Definition`] (unordered).
pub fn all_definitions() -> Vec<&'static Definition> {
    REGISTRY.values().copied().collect()
}

/// Returns every registered public id, sorted.
pub fn ids_sorted() -> Vec<String> {
    let mut ids: Vec<String> = REGISTRY.keys().cloned().collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_orders_and_adds_within_scale() {
        assert!(Native::new(2.0) > Native::new(1.0));
        assert_eq!((Native::new(1.0) + Native::new(2.0)).get(), 3.0);
    }

    fn def_fixture() -> Definition {
        Definition {
            id: "test.metric",
            header: "Test Metric",
            short_header: None,
            short_header_hide_unit: false,
            unit: Unit::Count,
            display_unit: None,
            display_order: None,
            group: DefinitionGroup::Default,
            larger_is_better: false,
            value_type: MetricValueType::Float,
            aliases: &[],
            deprecated_since: None,
        }
    }

    #[test]
    fn passes_threshold_respects_direction() {
        let smaller = Definition {
            larger_is_better: false,
            ..def_fixture()
        };
        let larger = Definition {
            larger_is_better: true,
            ..def_fixture()
        };
        assert!(smaller.passes_threshold(Native::new(5.0), Native::new(10.0)));
        assert!(!smaller.passes_threshold(Native::new(15.0), Native::new(10.0)));
        assert!(larger.passes_threshold(Native::new(15.0), Native::new(10.0)));
        assert!(!larger.passes_threshold(Native::new(5.0), Native::new(10.0)));
    }

    #[test]
    fn effective_display_unit_falls_back_to_native() {
        let d = def_fixture();
        assert_eq!(d.effective_display_unit(), d.unit);
    }

    #[test]
    fn definition_lookup_by_namespaced_id_and_alias() {
        use crate::metrics_core::MetricTag;
        let d = definition("aiperf.request_latency").expect("known id");
        assert_eq!(d.header, metric_definition(MetricTag::RequestLatency).header);
        assert!(definition("does.not.exist").is_none());
    }

    #[test]
    fn facade_path_resolves() {
        // Confirms the unified facade re-export compiles and points at the same data.
        assert!(crate::definitions::definition("aiperf.request_latency").is_some());
    }

    /// Pins the full public definition-id set (metric `aiperf.*` ids plus
    /// `analyzer.*` ids) as an accepted `insta` snapshot. This is the id
    /// contract: any addition, removal, or rename of a registered id changes
    /// the snapshot and fails review.
    ///
    /// CI LOCK: run the suite with `INSTA_UPDATE=no` so drift FAILS the test
    /// rather than silently rewriting the accepted `.snap`. The committed
    /// `.snap` is the source of truth; regenerate deliberately with
    /// `cargo insta review`/`accept` only when an id change is intended.
    ///
    /// SPDX NOTE: the generated `.snap` carries NO SPDX header — insta's
    /// snapshot parser does not tolerate leading comment lines before its
    /// `---` metadata block (it treats the file as empty and the test fails).
    /// The license attribution lives here, on the generating test, instead.
    /// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
    /// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
    #[test]
    fn definition_ids_snapshot() {
        insta::assert_json_snapshot!(crate::metrics_core::definition::ids_sorted());
    }
}
