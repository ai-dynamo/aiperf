// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit newtypes for native and display metric values, plus metric definitions.

use crate::metrics_core::units::{MetricValueType, Unit};

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
}
