// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trait-factory registry for native accuracy benchmarks and graders.

use std::collections::{BTreeMap, BTreeSet};
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

/// Constructor for fresh benchmark implementation state.
pub type BenchmarkFactory = fn() -> Box<dyn AccuracyBenchmark>;
/// Constructor for fresh grader implementation state.
pub type GraderFactory = fn() -> Rc<dyn Grader>;

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
#[derive(Clone)]
pub struct AccuracyRegistry {
    benchmarks: BTreeMap<String, BenchmarkRegistration>,
    canonical_names: Vec<&'static str>,
    graders: BTreeMap<String, GraderRegistration>,
    canonical_grader_names: Vec<&'static str>,
}

impl AccuracyRegistry {
    /// Creates an empty benchmark and grader registry.
    pub fn new() -> Self {
        Self {
            benchmarks: BTreeMap::new(),
            canonical_names: Vec::new(),
            graders: BTreeMap::new(),
            canonical_grader_names: Vec::new(),
        }
    }

    /// Builds the complete native in-tree benchmark registry.
    pub fn builtin() -> Self {
        let mut registry = Self::new();
        for registration in BUILTIN_BENCHMARKS {
            registry
                .register_benchmark_factory(registration.metadata, registration.factory)
                .expect("built-in benchmark registrations are valid and unique");
        }
        for registration in BUILTIN_GRADERS {
            registry
                .register_grader_factory(
                    registration.name,
                    registration.aliases,
                    registration.factory,
                )
                .expect("built-in grader registrations are valid and unique");
        }
        registry
    }

    /// Register a `Default` benchmark implementation and its static metadata.
    pub fn register_benchmark<B>(
        &mut self,
        metadata: &'static BenchmarkMetadata,
    ) -> Result<(), AccuracyError>
    where
        B: AccuracyBenchmark + Default + 'static,
    {
        self.register_benchmark_factory(metadata, default_benchmark_factory::<B>)
    }

    /// Register a benchmark factory and its static metadata.
    pub fn register_benchmark_factory(
        &mut self,
        metadata: &'static BenchmarkMetadata,
        factory: BenchmarkFactory,
    ) -> Result<(), AccuracyError> {
        let names = registration_names("benchmark", metadata.name, metadata.aliases)?;
        ensure_available("benchmark", &self.benchmarks, &names)?;
        let implementation_name = factory().name();
        ensure_implementation_name("benchmark", metadata.name, implementation_name)?;
        let registration = BenchmarkRegistration { metadata, factory };
        for name in names {
            self.benchmarks.insert(name, registration);
        }
        self.canonical_names.push(metadata.name);
        self.canonical_names.sort_unstable();
        Ok(())
    }

    /// Register a `Default` grader implementation under a canonical name and aliases.
    pub fn register_grader<G>(
        &mut self,
        name: &'static str,
        aliases: &'static [&'static str],
    ) -> Result<(), AccuracyError>
    where
        G: Grader + Default + 'static,
    {
        self.register_grader_factory(name, aliases, default_grader_factory::<G>)
    }

    /// Register a grader factory under a canonical name and aliases.
    pub fn register_grader_factory(
        &mut self,
        name: &'static str,
        aliases: &'static [&'static str],
        factory: GraderFactory,
    ) -> Result<(), AccuracyError> {
        let names = registration_names("grader", name, aliases)?;
        ensure_available("grader", &self.graders, &names)?;
        let implementation_name = factory().name();
        ensure_implementation_name("grader", name, implementation_name)?;
        let registration = GraderRegistration {
            name,
            aliases,
            factory,
        };
        for name in names {
            self.graders.insert(name, registration);
        }
        self.canonical_grader_names.push(name);
        self.canonical_grader_names.sort_unstable();
        Ok(())
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

fn default_benchmark_factory<B>() -> Box<dyn AccuracyBenchmark>
where
    B: AccuracyBenchmark + Default + 'static,
{
    Box::new(B::default())
}

fn default_grader_factory<G>() -> Rc<dyn Grader>
where
    G: Grader + Default + 'static,
{
    Rc::new(G::default())
}

fn registration_names(
    category: &'static str,
    canonical: &'static str,
    aliases: &'static [&'static str],
) -> Result<Vec<String>, AccuracyError> {
    let mut names = Vec::with_capacity(aliases.len() + 1);
    let mut unique = BTreeSet::new();
    for authored in std::iter::once(canonical).chain(aliases.iter().copied()) {
        let normalized = authored.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err(AccuracyError::InvalidRegistration {
                category,
                message: "names and aliases cannot be empty".into(),
            });
        }
        if !unique.insert(normalized.clone()) {
            return Err(AccuracyError::InvalidRegistration {
                category,
                message: format!("name or alias {authored:?} is repeated"),
            });
        }
        names.push(normalized);
    }
    Ok(names)
}

fn ensure_available<T>(
    category: &'static str,
    registrations: &BTreeMap<String, T>,
    names: &[String],
) -> Result<(), AccuracyError> {
    if let Some(name) = names
        .iter()
        .find(|name| registrations.contains_key(name.as_str()))
    {
        return Err(AccuracyError::DuplicateRegistration {
            category,
            name: name.clone(),
        });
    }
    Ok(())
}

fn ensure_implementation_name(
    category: &'static str,
    registered: &'static str,
    implementation: &'static str,
) -> Result<(), AccuracyError> {
    if registered
        .trim()
        .eq_ignore_ascii_case(implementation.trim())
    {
        Ok(())
    } else {
        Err(AccuracyError::InvalidRegistration {
            category,
            message: format!(
                "registered name {registered:?} does not match implementation name {implementation:?}"
            ),
        })
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
    use crate::{
        AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, DatasetSource, Grader,
    };
    use aiperf_metrics::GradingResult;

    use super::{AccuracyRegistry, BenchmarkMetadata};

    #[derive(Default)]
    struct FixtureBenchmark;

    impl AccuracyBenchmark for FixtureBenchmark {
        fn name(&self) -> &'static str {
            "fixture"
        }

        fn load_problems(
            &self,
            _source: &dyn DatasetSource,
            _config: &BenchmarkConfig,
        ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
            Ok(Vec::new())
        }
    }

    #[derive(Default)]
    struct FixtureGrader;

    #[async_trait::async_trait(?Send)]
    impl Grader for FixtureGrader {
        fn name(&self) -> &'static str {
            "fixture-grader"
        }

        async fn grade(
            &self,
            response_text: &str,
            ground_truth: &str,
        ) -> Result<GradingResult, AccuracyError> {
            Ok(GradingResult::from_score(
                f64::from(response_text == ground_truth),
                false,
                ground_truth,
            ))
        }
    }

    static FIXTURE: BenchmarkMetadata = BenchmarkMetadata {
        name: "fixture",
        aliases: &["fixture_alias"],
        default_grader: "fixture-grader",
        default_n_shots: 0,
        default_enable_cot: false,
        default_system_prompt: None,
    };

    static CONFLICT: BenchmarkMetadata = BenchmarkMetadata {
        name: "other-fixture",
        aliases: &["mmlu"],
        default_grader: "exact-match",
        default_n_shots: 0,
        default_enable_cot: false,
        default_system_prompt: None,
    };

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

    #[test]
    fn external_factories_register_and_conflicts_are_rejected() {
        let mut registry = AccuracyRegistry::builtin();
        registry
            .register_benchmark::<FixtureBenchmark>(&FIXTURE)
            .unwrap();
        registry
            .register_grader::<FixtureGrader>("fixture-grader", &["fixture_grader"])
            .unwrap();

        assert_eq!(
            registry
                .benchmark("fixture_alias")
                .unwrap()
                .benchmark
                .name(),
            "fixture"
        );
        assert_eq!(
            registry.grader("fixture_grader").unwrap().name(),
            "fixture-grader"
        );
        assert_eq!(registry.benchmark_names().len(), 12);
        assert_eq!(registry.grader_names().len(), 10);

        let error = registry
            .register_benchmark::<FixtureBenchmark>(&CONFLICT)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate benchmark registration \"mmlu\"")
        );
        assert!(registry.benchmark("other-fixture").is_err());
    }
}
