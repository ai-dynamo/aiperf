// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trait-factory registry for native accuracy benchmarks and graders.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::{
    AIME_SYSTEM_PROMPT, AccuracyBenchmark, AccuracyError, Aime24Benchmark, Aime25Benchmark,
    AimeBenchmark, BenchmarkConfig, BenchmarkProblem, BigBenchBenchmark, ChatMessage,
    CodeExecutionGrader, DatasetSource, ExactMatchGrader, ExpressionGrader, GpqaDiamondBenchmark,
    GpqaGrader, Grader, Gsm8kBenchmark, Gsm8kGrader, HellaSwagBenchmark, LatexGrader,
    LcbCodeGenerationBenchmark, Math500Benchmark, MathGrader, MmluBenchmark, MmluProBenchmark,
    MmluProGrader, MultipleChoiceGrader,
};

/// Static defaults and aliases for one benchmark plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkMetadata {
    /// Canonical CLI/report name.
    pub name: &'static str,
    /// Alternate accepted spellings.
    pub aliases: &'static [&'static str],
    /// Default registered grader.
    pub default_grader: &'static str,
    /// Default few-shot count.
    pub default_n_shots: usize,
    /// Default chain-of-thought policy.
    pub default_enable_cot: bool,
    /// Default system prompt, if the reference injects one.
    pub default_system_prompt: Option<&'static str>,
}

/// Resolved benchmark plugin and its defaults.
pub struct RegisteredBenchmark {
    /// Plugin metadata.
    pub metadata: &'static BenchmarkMetadata,
    /// Fresh concrete trait implementation.
    pub benchmark: Box<dyn AccuracyBenchmark>,
}

impl RegisteredBenchmark {
    /// Validates row-independent benchmark settings before remote acquisition.
    pub fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        self.benchmark.validate_config(config)
    }

    /// Materializes problems and injects the user/default system prompt uniformly.
    pub fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
        system_prompt: Option<&str>,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        let mut problems = self.benchmark.load_problems(source, config)?;
        if let Some(prompt) = system_prompt
            .or(self.metadata.default_system_prompt)
            .filter(|prompt| !prompt.is_empty())
        {
            for problem in &mut problems {
                problem.messages.insert(0, ChatMessage::system(prompt));
            }
        }
        Ok(problems)
    }
}

type BenchmarkFactory = fn() -> Box<dyn AccuracyBenchmark>;
type GraderFactory = fn() -> Rc<dyn Grader>;

#[derive(Clone, Copy)]
struct BenchmarkRegistration {
    metadata: &'static BenchmarkMetadata,
    factory: BenchmarkFactory,
}

#[derive(Clone, Copy)]
struct GraderRegistration {
    name: &'static str,
    aliases: &'static [&'static str],
    factory: GraderFactory,
}

/// Extensible benchmark registry. Names resolve to factories, so each run gets
/// independent concrete state and runtime code never switches on benchmark kinds.
pub struct AccuracyRegistry {
    benchmarks: BTreeMap<&'static str, BenchmarkRegistration>,
    canonical_names: Vec<&'static str>,
    graders: BTreeMap<&'static str, GraderRegistration>,
    canonical_grader_names: Vec<&'static str>,
}

impl AccuracyRegistry {
    /// Builds the complete native in-tree benchmark registry.
    pub fn builtin() -> Self {
        let mut registry = Self {
            benchmarks: BTreeMap::new(),
            canonical_names: Vec::new(),
            graders: BTreeMap::new(),
            canonical_grader_names: Vec::new(),
        };
        for registration in BUILTIN_BENCHMARKS {
            registry.register(*registration);
        }
        registry.canonical_names.sort_unstable();
        for registration in BUILTIN_GRADERS {
            registry.register_grader(*registration);
        }
        registry.canonical_grader_names.sort_unstable();
        registry
    }

    fn register_grader(&mut self, registration: GraderRegistration) {
        self.canonical_grader_names.push(registration.name);
        self.graders.insert(registration.name, registration);
        for alias in registration.aliases {
            self.graders.insert(alias, registration);
        }
    }

    fn register(&mut self, registration: BenchmarkRegistration) {
        self.canonical_names.push(registration.metadata.name);
        self.benchmarks
            .insert(registration.metadata.name, registration);
        for alias in registration.metadata.aliases {
            self.benchmarks.insert(alias, registration);
        }
    }

    /// Resolves a canonical name or alias.
    pub fn benchmark(&self, name: &str) -> Result<RegisteredBenchmark, AccuracyError> {
        let normalized = name.trim().to_ascii_lowercase();
        let registration = self.benchmarks.get(normalized.as_str()).ok_or_else(|| {
            AccuracyError::UnknownBenchmark {
                name: name.to_string(),
                available: self
                    .canonical_names
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            }
        })?;
        Ok(RegisteredBenchmark {
            metadata: registration.metadata,
            benchmark: (registration.factory)(),
        })
    }

    /// Canonical benchmark names in deterministic order.
    pub fn benchmark_names(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.canonical_names.iter().copied()
    }

    /// Resolves a grader name or alias to a fresh implementation.
    pub fn grader(&self, name: &str) -> Result<Rc<dyn Grader>, AccuracyError> {
        let normalized = name.trim().to_ascii_lowercase();
        let registration =
            self.graders
                .get(normalized.as_str())
                .ok_or_else(|| AccuracyError::UnknownGrader {
                    name: name.to_string(),
                    available: self
                        .canonical_grader_names
                        .iter()
                        .map(|name| (*name).to_string())
                        .collect(),
                })?;
        let grader = (registration.factory)();
        grader.check_available()?;
        Ok(grader)
    }

    /// Canonical grader names in deterministic order.
    pub fn grader_names(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.canonical_grader_names.iter().copied()
    }
}

impl Default for AccuracyRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

macro_rules! registration {
    ($metadata:ident, $type:ty) => {
        BenchmarkRegistration {
            metadata: &$metadata,
            factory: || Box::new(<$type>::default()),
        }
    };
}

static MMLU_PRO: BenchmarkMetadata = BenchmarkMetadata {
    name: "mmlu-pro",
    aliases: &["mmlu_pro"],
    default_grader: "mmlu-pro",
    default_n_shots: 5,
    default_enable_cot: true,
    default_system_prompt: None,
};
static MMLU: BenchmarkMetadata = BenchmarkMetadata {
    name: "mmlu",
    aliases: &[],
    default_grader: "multiple-choice",
    default_n_shots: 5,
    default_enable_cot: false,
    default_system_prompt: None,
};
static AIME: BenchmarkMetadata = BenchmarkMetadata {
    name: "aime",
    aliases: &[],
    default_grader: "math",
    default_n_shots: 8,
    default_enable_cot: true,
    default_system_prompt: Some(AIME_SYSTEM_PROMPT),
};
static HELLASWAG: BenchmarkMetadata = BenchmarkMetadata {
    name: "hellaswag",
    aliases: &[],
    default_grader: "exact-match",
    default_n_shots: 10,
    default_enable_cot: false,
    default_system_prompt: None,
};
static BIGBENCH: BenchmarkMetadata = BenchmarkMetadata {
    name: "bigbench",
    aliases: &["bbh", "bigbench-hard"],
    default_grader: "exact-match",
    default_n_shots: 3,
    default_enable_cot: true,
    default_system_prompt: None,
};
static AIME24: BenchmarkMetadata = BenchmarkMetadata {
    name: "aime24",
    aliases: &["aime-2024"],
    default_grader: "expression",
    default_n_shots: 0,
    default_enable_cot: false,
    default_system_prompt: None,
};
static AIME25: BenchmarkMetadata = BenchmarkMetadata {
    name: "aime25",
    aliases: &["aime-2025"],
    default_grader: "expression",
    default_n_shots: 0,
    default_enable_cot: false,
    default_system_prompt: None,
};
static MATH_500: BenchmarkMetadata = BenchmarkMetadata {
    name: "math-500",
    aliases: &["math_500"],
    default_grader: "latex",
    default_n_shots: 0,
    default_enable_cot: false,
    default_system_prompt: None,
};
static GSM8K: BenchmarkMetadata = BenchmarkMetadata {
    name: "gsm8k",
    aliases: &[],
    default_grader: "gsm8k",
    default_n_shots: 0,
    default_enable_cot: false,
    default_system_prompt: None,
};
static GPQA: BenchmarkMetadata = BenchmarkMetadata {
    name: "gpqa-diamond",
    aliases: &["gpqa_diamond"],
    default_grader: "gpqa",
    default_n_shots: 0,
    default_enable_cot: false,
    default_system_prompt: None,
};
static LCB: BenchmarkMetadata = BenchmarkMetadata {
    name: "lcb-codegeneration",
    aliases: &["lcb_codegeneration", "lcb:codegeneration"],
    default_grader: "code-execution",
    default_n_shots: 0,
    default_enable_cot: false,
    default_system_prompt: None,
};

static BUILTIN_BENCHMARKS: &[BenchmarkRegistration] = &[
    registration!(MMLU_PRO, MmluProBenchmark),
    registration!(MMLU, MmluBenchmark),
    registration!(AIME, AimeBenchmark),
    registration!(HELLASWAG, HellaSwagBenchmark),
    registration!(BIGBENCH, BigBenchBenchmark),
    registration!(AIME24, Aime24Benchmark),
    registration!(AIME25, Aime25Benchmark),
    registration!(MATH_500, Math500Benchmark),
    registration!(GSM8K, Gsm8kBenchmark),
    registration!(GPQA, GpqaDiamondBenchmark),
    registration!(LCB, LcbCodeGenerationBenchmark),
];

macro_rules! grader_registration {
    ($name:literal, [$($alias:literal),* $(,)?], $type:ty) => {
        GraderRegistration {
            name: $name,
            aliases: &[$($alias),*],
            factory: || Rc::new(<$type>::default()),
        }
    };
}

static BUILTIN_GRADERS: &[GraderRegistration] = &[
    grader_registration!("mmlu-pro", ["mmlu_pro"], MmluProGrader),
    grader_registration!("exact-match", ["exact_match"], ExactMatchGrader),
    grader_registration!("multiple-choice", ["multiple_choice"], MultipleChoiceGrader),
    grader_registration!("math", [], MathGrader),
    grader_registration!("expression", ["lighteval_expr"], ExpressionGrader),
    grader_registration!("latex", ["lighteval_latex"], LatexGrader),
    grader_registration!("gpqa", ["lighteval_gpqa"], GpqaGrader),
    grader_registration!("gsm8k", ["lighteval_gsm8k"], Gsm8kGrader),
    grader_registration!("code-execution", ["code_execution"], CodeExecutionGrader),
];

#[cfg(test)]
mod tests {
    use super::AccuracyRegistry;

    #[test]
    fn resolves_aliases_without_runtime_kind_switches() {
        let registry = AccuracyRegistry::builtin();
        let resolved = registry.benchmark("mmlu_pro").unwrap();
        assert_eq!(resolved.metadata.name, "mmlu-pro");
        assert_eq!(resolved.benchmark.name(), "mmlu-pro");
        assert_eq!(registry.benchmark_names().len(), 11);
        assert_eq!(
            registry.grader("multiple_choice").unwrap().name(),
            "multiple-choice"
        );
        assert_eq!(registry.grader_names().len(), 9);
    }
}
