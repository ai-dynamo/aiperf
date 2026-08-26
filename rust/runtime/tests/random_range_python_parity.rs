// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RNG and token-vector parity against the Python/NumPy reference.

use std::sync::Arc;

use aiperf_runtime::dataset::{
    CorpusPromptGeneratorFactory, PromptGeneratorFactory, RandomCorpusStyle, RandomRangePlan,
    RandomRangeRatio, TextTokenizer,
};
use aiperf_runtime::rng::RngRoot;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    provenance: Provenance,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Provenance {
    draw_order: String,
    token_formula: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    style: RandomCorpusStyle,
    algorithm: String,
    seed: u64,
    entries: usize,
    input_mean: i64,
    output_mean: i64,
    input_ratio: f64,
    output_ratio: f64,
    special_tokens: i64,
    input_bounds: (i64, i64),
    output_bounds: (i64, i64),
    inputs: Vec<i64>,
    outputs: Vec<i64>,
    offsets: Vec<usize>,
    token_pool: Vec<u32>,
    requests: Vec<ReferenceRequest>,
}

#[derive(Deserialize)]
struct ReferenceRequest {
    token_ids: Vec<u32>,
}

struct VectorTokenizer {
    allowed: Arc<[u32]>,
}

impl TextTokenizer for VectorTokenizer {
    fn encode(&self, _text: &str) -> aiperf_runtime::dataset::Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn decode(&self, token_ids: &[u32]) -> aiperf_runtime::dataset::Result<String> {
        Ok(token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    fn vocab_size(&self) -> Option<u32> {
        Some(16)
    }

    fn allowed_random_token_ids(&self) -> Option<Arc<[u32]>> {
        Some(Arc::clone(&self.allowed))
    }

    fn name(&self) -> &str {
        "python-vector-tokenizer"
    }
}

#[test]
fn python_numpy_vectors_match_native_lengths_offsets_and_tokens() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/random_range_python_vectors.json")).unwrap();
    assert_eq!(
        fixture.provenance.draw_order,
        "all_inputs_then_all_outputs_then_all_offsets"
    );
    assert_eq!(
        fixture.provenance.token_formula,
        "pool[(offset + request_index + token_index) % len(pool)]"
    );
    for case in fixture.cases {
        match case.style {
            RandomCorpusStyle::Vllm => assert_eq!(
                case.algorithm, "numpy.random.default_rng/PCG64",
                "{}",
                case.name
            ),
            RandomCorpusStyle::Sglang => assert_eq!(
                case.algorithm, "numpy.random.RandomState/MT19937",
                "{}",
                case.name
            ),
        }
        let plan = RandomRangePlan::new(
            case.style,
            case.input_mean,
            case.output_mean,
            RandomRangeRatio::new(case.input_ratio, case.output_ratio).unwrap(),
            case.special_tokens,
        )
        .unwrap();
        assert_eq!(plan.input_bounds(), case.input_bounds, "{}", case.name);
        assert_eq!(plan.output_bounds(), case.output_bounds, "{}", case.name);
        let seeded = plan.preseed(case.entries, case.seed, 16).unwrap();
        assert_eq!(seeded.inputs(), case.inputs, "{}", case.name);
        assert_eq!(seeded.outputs(), case.outputs, "{}", case.name);
        assert_eq!(seeded.offsets(), case.offsets, "{}", case.name);

        let tokenizer = VectorTokenizer {
            allowed: Arc::from(case.token_pool),
        };
        let factory =
            CorpusPromptGeneratorFactory::random_reference(case.style, Arc::from(seeded.offsets()));
        let mut generator = factory
            .create(&tokenizer, RngRoot::new(Some(case.seed)))
            .unwrap();
        assert_eq!(case.requests.len(), case.entries, "{}", case.name);
        for (request_index, reference) in case.requests.iter().enumerate() {
            let token_ids = generator
                .generate_token_ids(case.inputs[request_index] as usize, &[], 1)
                .unwrap();
            assert_eq!(
                token_ids, reference.token_ids,
                "{} request {request_index}",
                case.name
            );
        }
    }
}
