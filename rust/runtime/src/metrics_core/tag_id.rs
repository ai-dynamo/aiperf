// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open metric identity.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_metric_tag_registers_without_touching_the_enum() {
        let mut registry = MetricTagRegistry::builtin();
        let id = registry
            .register("plugin.metric")
            .expect("register a new metric tag");

        assert_eq!(id.as_str(), "plugin.metric");
        assert_eq!(MetricTagId::resolve_in(&registry, "plugin.metric"), Some(id));
    }

    #[test]
    fn duplicate_metric_tag_registration_is_rejected() {
        let mut registry = MetricTagRegistry::builtin();

        assert!(registry.register("request_count").is_err());
    }
}
