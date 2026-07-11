// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepEval-aligned HellaSwag sentence-completion prompts.
//!
//! Ported from `src/aiperf/accuracy/benchmarks/hellaswag.py:1-328` and
//! the prompt shape pinned by `tests/unit/accuracy/test_hellaswag_benchmark.py`.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::common::{
    finish_selection, generation, integer, item_id, metadata, normalized_task, problem,
    required_string, string_array,
};
use crate::{
    AccuracyBenchmark, AccuracyError, BenchmarkConfig, BenchmarkProblem, ChatMessage,
    DatasetSource, DatasetSplit,
};

/// DeepEval's default output-confinement instruction.
pub const HELLASWAG_CONFINEMENT: &str = "Output 'A', 'B', 'C', or 'D'. Full answer not needed.";
/// Maximum few-shot count accepted by DeepEval.
pub const HELLASWAG_MAX_N_SHOTS: usize = 15;

// DeepEval task enum order at revision
// `625814c0c7f3fe88abd2dd7cf96944b2b4d9ed68`, from
// `deepeval/benchmarks/hellaswag/task.py:1-196`. Keeping enum names as
// well as values preserves its two accepted selector forms.
const HELLASWAG_TASKS: &[(&str, &str)] = &[
    ("APPLYING_SUNSCREEN", "Applying sunscreen"),
    ("TRIMMING_BRANCHES_OR_HEDGES", "Trimming branches or hedges"),
    ("DISC_DOG", "Disc dog"),
    ("WAKEBOARDING", "Wakeboarding"),
    ("SKATEBOARDING", "Skateboarding"),
    ("WATERSKIING", "Waterskiing"),
    ("WASHING_HANDS", "Washing hands"),
    ("SAILING", "Sailing"),
    ("PLAYING_CONGAS", "Playing congas"),
    ("BALLET", "Ballet"),
    ("ROOF_SHINGLE_REMOVAL", "Roof shingle removal"),
    ("HAND_CAR_WASH", "Hand car wash"),
    ("KITE_FLYING", "Kite flying"),
    ("PLAYING_POOL", "Playing pool"),
    ("PLAYING_LACROSSE", "Playing lacrosse"),
    ("LAYUP_DRILL_IN_BASKETBALL", "Layup drill in basketball"),
    ("HOME_AND_GARDEN", "Home and Garden"),
    ("PLAYING_BEACH_VOLLEYBALL", "Playing beach volleyball"),
    ("CALF_ROPING", "Calf roping"),
    ("SCUBA_DIVING", "Scuba diving"),
    ("MIXING_DRINKS", "Mixing drinks"),
    ("PUTTING_ON_SHOES", "Putting on shoes"),
    ("MAKING_A_LEMONADE", "Making a lemonade"),
    ("UNCATEGORIZED", "Uncategorized"),
    ("ZUMBA", "Zumba"),
    ("PLAYING_BADMINTON", "Playing badminton"),
    ("PLAYING_BAGPIPES", "Playing bagpipes"),
    ("FOOD_AND_ENTERTAINING", "Food and Entertaining"),
    ("PERSONAL_CARE_AND_STYLE", "Personal Care and Style"),
    ("CRICKET", "Cricket"),
    ("SHOVELING_SNOW", "Shoveling snow"),
    ("PING_PONG", "Ping-pong"),
    ("HOLIDAYS_AND_TRADITIONS", "Holidays and Traditions"),
    ("ICE_FISHING", "Ice fishing"),
    ("BEACH_SOCCER", "Beach soccer"),
    ("TABLE_SOCCER", "Table soccer"),
    ("SWIMMING", "Swimming"),
    ("BATON_TWIRLING", "Baton twirling"),
    ("JAVELIN_THROW", "Javelin throw"),
    ("SHOT_PUT", "Shot put"),
    ("DOING_CRUNCHES", "Doing crunches"),
    ("POLISHING_SHOES", "Polishing shoes"),
    ("TRAVEL", "Travel"),
    ("USING_UNEVEN_BARS", "Using uneven bars"),
    ("PLAYING_HARMONICA", "Playing harmonica"),
    ("RELATIONSHIPS", "Relationships"),
    ("HIGH_JUMP", "High jump"),
    ("MAKING_A_SANDWICH", "Making a sandwich"),
    ("POWERBOCKING", "Powerbocking"),
    ("REMOVING_ICE_FROM_CAR", "Removing ice from car"),
    ("SHAVING", "Shaving"),
    ("SHARPENING_KNIVES", "Sharpening knives"),
    ("WELDING", "Welding"),
    ("USING_PARALLEL_BARS", "Using parallel bars"),
    ("HOME_CATEGORIES", "Home,Categories"),
    ("ROCK_CLIMBING", "Rock climbing"),
    ("SNOW_TUBING", "Snow tubing"),
    ("WASHING_FACE", "Washing face"),
    ("ASSEMBLING_BICYCLE", "Assembling bicycle"),
    (
        "TENNIS_SERVE_WITH_BALL_BOUNCING",
        "Tennis serve with ball bouncing",
    ),
    ("SHUFFLEBOARD", "Shuffleboard"),
    ("DODGEBALL", "Dodgeball"),
    ("CAPOEIRA", "Capoeira"),
    ("PAINTBALL", "Paintball"),
    ("DOING_A_POWERBOMB", "Doing a powerbomb"),
    ("DOING_MOTOCROSS", "Doing motocross"),
    ("PLAYING_ICE_HOCKEY", "Playing ice hockey"),
    ("PHILOSOPHY_AND_RELIGION", "Philosophy and Religion"),
    ("ARCHERY", "Archery"),
    ("CARS_AND_OTHER_VEHICLES", "Cars & Other Vehicles"),
    ("RUNNING_A_MARATHON", "Running a marathon"),
    ("THROWING_DARTS", "Throwing darts"),
    ("PAINTING_FURNITURE", "Painting furniture"),
    ("HAVING_AN_ICE_CREAM", "Having an ice cream"),
    ("SLACKLINING", "Slacklining"),
    ("CAMEL_RIDE", "Camel ride"),
    ("ARM_WRESTLING", "Arm wrestling"),
    ("HULA_HOOP", "Hula hoop"),
    ("SURFING", "Surfing"),
    ("PLAYING_PIANO", "Playing piano"),
    ("GARGLING_MOUTHWASH", "Gargling mouthwash"),
    ("PLAYING_ACCORDION", "Playing accordion"),
    ("HORSEBACK_RIDING", "Horseback riding"),
    ("PUTTING_IN_CONTACT_LENSES", "Putting in contact lenses"),
    ("PLAYING_SAXOPHONE", "Playing saxophone"),
    ("FUTSAL", "Futsal"),
    ("LONG_JUMP", "Long jump"),
    ("LONGBOARDING", "Longboarding"),
    ("POLE_VAULT", "Pole vault"),
    ("BUILDING_SANDCASTLES", "Building sandcastles"),
    ("PLATFORM_DIVING", "Platform diving"),
    ("PAINTING", "Painting"),
    ("SPINNING", "Spinning"),
    ("CARVING_JACK_O_LANTERNS", "Carving jack-o-lanterns"),
    ("BRAIDING_HAIR", "Braiding hair"),
    ("YOUTH", "Youth"),
    ("PLAYING_VIOLIN", "Playing violin"),
    ("CANOEING", "Canoeing"),
    ("CHEERLEADING", "Cheerleading"),
    ("PETS_AND_ANIMALS", "Pets and Animals"),
    ("KAYAKING", "Kayaking"),
    ("CLEANING_SHOES", "Cleaning shoes"),
    ("KNITTING", "Knitting"),
    ("BAKING_COOKIES", "Baking cookies"),
    ("DOING_FENCING", "Doing fencing"),
    ("PLAYING_GUITARRA", "Playing guitarra"),
    ("USING_THE_ROWING_MACHINE", "Using the rowing machine"),
    ("GETTING_A_HAIRCUT", "Getting a haircut"),
    ("MOOPING_FLOOR", "Mooping floor"),
    ("RIVER_TUBING", "River tubing"),
    ("CLEANING_SINK", "Cleaning sink"),
    ("GROOMING_DOG", "Grooming dog"),
    ("DISCUS_THROW", "Discus throw"),
    ("CLEANING_WINDOWS", "Cleaning windows"),
    ("FINANCE_AND_BUSINESS", "Finance and Business"),
    ("HANGING_WALLPAPER", "Hanging wallpaper"),
    ("ROPE_SKIPPING", "Rope skipping"),
    ("WINDSURFING", "Windsurfing"),
    ("KNEELING", "Kneeling"),
    ("GETTING_A_PIERCING", "Getting a piercing"),
    ("ROCK_PAPER_SCISSORS", "Rock-paper-scissors"),
    ("SPORTS_AND_FITNESS", "Sports and Fitness"),
    ("BREAKDANCING", "Breakdancing"),
    ("WALKING_THE_DOG", "Walking the dog"),
    ("PLAYING_DRUMS", "Playing drums"),
    ("PLAYING_WATER_POLO", "Playing water polo"),
    ("BMX", "BMX"),
    ("SMOKING_A_CIGARETTE", "Smoking a cigarette"),
    ("BLOWING_LEAVES", "Blowing leaves"),
    ("BULLFIGHTING", "Bullfighting"),
    ("DRINKING_COFFEE", "Drinking coffee"),
    ("BATHING_DOG", "Bathing dog"),
    ("TANGO", "Tango"),
    ("WRAPPING_PRESENTS", "Wrapping presents"),
    ("PLASTERING", "Plastering"),
    ("PLAYING_BLACKJACK", "Playing blackjack"),
    ("FUN_SLIDING_DOWN", "Fun sliding down"),
    ("WORK_WORLD", "Work World"),
    ("TRIPLE_JUMP", "Triple jump"),
    ("TUMBLING", "Tumbling"),
    ("SKIING", "Skiing"),
    ("DOING_KICKBOXING", "Doing kickboxing"),
    ("BLOW_DRYING_HAIR", "Blow-drying hair"),
    ("DRUM_CORPS", "Drum corps"),
    ("SMOKING_HOOKAH", "Smoking hookah"),
    ("MOWING_THE_LAWN", "Mowing the lawn"),
    ("VOLLEYBALL", "Volleyball"),
    ("LAYING_TILE", "Laying tile"),
    ("STARTING_A_CAMPFIRE", "Starting a campfire"),
    ("SUMO", "Sumo"),
    ("HURLING", "Hurling"),
    ("PLAYING_KICKBALL", "Playing kickball"),
    ("MAKING_A_CAKE", "Making a cake"),
    ("FIXING_THE_ROOF", "Fixing the roof"),
    ("PLAYING_POLO", "Playing polo"),
    ("REMOVING_CURLERS", "Removing curlers"),
    ("ELLIPTICAL_TRAINER", "Elliptical trainer"),
    ("HEALTH", "Health"),
    ("SPREAD_MULCH", "Spread mulch"),
    ("CHOPPING_WOOD", "Chopping wood"),
    ("BRUSHING_TEETH", "Brushing teeth"),
    ("USING_THE_POMMEL_HORSE", "Using the pommel horse"),
    ("SNATCH", "Snatch"),
    ("CLIPPING_CAT_CLAWS", "Clipping cat claws"),
    ("PUTTING_ON_MAKEUP", "Putting on makeup"),
    ("HAND_WASHING_CLOTHES", "Hand washing clothes"),
    ("HITTING_A_PINATA", "Hitting a pinata"),
    ("TAI_CHI", "Tai chi"),
    ("GETTING_A_TATTOO", "Getting a tattoo"),
    ("DRINKING_BEER", "Drinking beer"),
    ("SHAVING_LEGS", "Shaving legs"),
    ("DOING_KARATE", "Doing karate"),
    ("PLAYING_RUBIK_CUBE", "Playing rubik cube"),
    ("FAMILY_LIFE", "Family Life"),
    ("ROLLERBLADING", "Rollerblading"),
    (
        "EDUCATION_AND_COMMUNICATIONS",
        "Education and Communications",
    ),
    ("FIXING_BICYCLE", "Fixing bicycle"),
    ("BEER_PONG", "Beer pong"),
    ("IRONING_CLOTHES", "Ironing clothes"),
    ("CUTTING_THE_GRASS", "Cutting the grass"),
    ("RAKING_LEAVES", "Raking leaves"),
    ("PLAYING_SQUASH", "Playing squash"),
    ("HOPSCOTCH", "Hopscotch"),
    ("INSTALLING_CARPET", "Installing carpet"),
    ("POLISHING_FURNITURE", "Polishing furniture"),
    (
        "DECORATING_THE_CHRISTMAS_TREE",
        "Decorating the Christmas tree",
    ),
    ("PREPARING_SALAD", "Preparing salad"),
    ("PREPARING_PASTA", "Preparing pasta"),
    ("VACUUMING_FLOOR", "Vacuuming floor"),
    ("CLEAN_AND_JERK", "Clean and jerk"),
    ("COMPUTERS_AND_ELECTRONICS", "Computers and Electronics"),
    ("CROQUET", "Croquet"),
];

#[derive(Debug, Clone)]
struct Row {
    index: usize,
    activity: String,
    context: String,
    endings: Vec<String>,
    label: Option<usize>,
    source: Value,
}

/// Native HellaSwag benchmark.
#[derive(Debug, Clone, Copy, Default)]
pub struct HellaSwagBenchmark;

impl AccuracyBenchmark for HellaSwagBenchmark {
    fn name(&self) -> &'static str {
        "hellaswag"
    }

    fn validate_config(&self, config: &BenchmarkConfig) -> Result<(), AccuracyError> {
        if config.n_shots > HELLASWAG_MAX_N_SHOTS {
            return Err(AccuracyError::UnsupportedConfiguration(format!(
                "hellaswag accepts at most {HELLASWAG_MAX_N_SHOTS} shots, got {}",
                config.n_shots
            )));
        }
        resolve_tasks(&config.tasks).map(|_| ())
    }

    fn load_problems(
        &self,
        source: &dyn DatasetSource,
        config: &BenchmarkConfig,
    ) -> Result<Vec<BenchmarkProblem>, AccuracyError> {
        self.validate_config(config)?;
        let train = parse_rows(source.load_rows(DatasetSplit::Train)?)?;
        let validation = parse_rows(source.load_rows(DatasetSplit::Validation)?)?;
        let selected = resolve_tasks(&config.tasks)?;

        let mut seen = BTreeSet::new();
        let shots = train
            .iter()
            .filter(|row| seen.insert(row.activity.clone()))
            .take(config.n_shots)
            .collect::<Vec<_>>();

        let mut problems = Vec::new();
        let mut occurrences = BTreeMap::<String, usize>::new();
        for task in selected {
            let occurrence = occurrences.entry(task.clone()).or_default();
            for row in validation.iter().filter(|row| row.activity == task) {
                let Some(label) = row.label else {
                    continue;
                };
                if label >= 4 {
                    return Err(super::common::invalid_row(
                        row.index,
                        format!("HellaSwag label {label} is outside 0..=3"),
                    ));
                }
                let mut prompt = format!(
                    "The following are multiple choice questions (with answers) are sentence completion problems about {}.\n\n",
                    row.activity
                );
                for shot in &shots {
                    let Some(shot_label) = shot.label else {
                        continue;
                    };
                    prompt.push_str(&format_question(shot, Some(shot_label))?);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&format_question(row, None)?);
                prompt.push_str("\n\n");
                prompt.push_str(HELLASWAG_CONFINEMENT);
                let base_id = item_id(&row.source, row.index, &["ind", "id", "question_id"]);
                let unique_id = if *occurrence == 0 {
                    base_id
                } else {
                    format!("{base_id}:repeat:{}", *occurrence)
                };
                problems.push(problem(
                    self.name(),
                    unique_id,
                    normalized_task("hellaswag", &row.activity),
                    vec![ChatMessage::user(prompt)],
                    char::from(b'A' + label as u8).to_string(),
                    generation(config.max_tokens.unwrap_or(5), Vec::new()),
                    metadata([
                        ("activity_label", json!(row.activity)),
                        ("generation_size", json!(5)),
                    ]),
                ));
            }
            *occurrence += 1;
        }
        finish_selection(self.name(), config, problems)
    }
}

fn parse_rows(rows: Vec<Value>) -> Result<Vec<Row>, AccuracyError> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let label = match row.get("label") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) if value.is_empty() => None,
                _ => Some(integer(row, "label", index)? as usize),
            };
            let endings = string_array(row, "endings", index)?;
            if endings.len() != 4 {
                return Err(super::common::invalid_row(
                    index,
                    format!("HellaSwag expects four endings, found {}", endings.len()),
                ));
            }
            Ok(Row {
                index,
                activity: required_string(row, "activity_label", index)?,
                context: required_string(row, "ctx", index)?,
                endings,
                label,
                source: row.clone(),
            })
        })
        .collect()
}

fn resolve_tasks(requested: &[String]) -> Result<Vec<String>, AccuracyError> {
    if requested.is_empty() {
        return Ok(HELLASWAG_TASKS
            .iter()
            .map(|(_, value)| (*value).to_string())
            .collect());
    }
    let all_count = requested
        .iter()
        .filter(|task| task.eq_ignore_ascii_case("all"))
        .count();
    if all_count > 0 {
        if requested.len() != 1 {
            return Err(AccuracyError::UnsupportedConfiguration(
                "hellaswag task 'all' cannot be mixed with other activity labels".to_string(),
            ));
        }
        return Ok(HELLASWAG_TASKS
            .iter()
            .map(|(_, value)| (*value).to_string())
            .collect());
    }
    let mut selected = Vec::with_capacity(requested.len());
    for task in requested {
        let enum_name = task.to_ascii_uppercase();
        let resolved = HELLASWAG_TASKS
            .iter()
            .find(|(name, value)| value.eq_ignore_ascii_case(task) || *name == enum_name.as_str());
        let Some((_, resolved)) = resolved else {
            let mut available = HELLASWAG_TASKS
                .iter()
                .map(|(_, value)| (*value).to_string())
                .collect::<Vec<_>>();
            available.sort();
            return Err(AccuracyError::UnknownTask {
                task: task.clone(),
                available,
            });
        };
        selected.push((*resolved).to_string());
    }
    Ok(selected)
}

fn format_question(row: &Row, answer: Option<usize>) -> Result<String, AccuracyError> {
    if answer.is_some_and(|answer| answer >= row.endings.len()) {
        return Err(super::common::invalid_row(
            row.index,
            "few-shot answer is outside ending range".to_string(),
        ));
    }
    let mut output = row.context.clone();
    for (index, ending) in row.endings.iter().enumerate() {
        output.push('\n');
        output.push(char::from(b'A' + index as u8));
        output.push_str(". ");
        output.push_str(ending);
    }
    output.push_str("\nAnswer:");
    if let Some(answer) = answer {
        output.push(' ');
        output.push(char::from(b'A' + answer as u8));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::InMemoryDatasetSource;

    #[test]
    fn renders_reference_question_and_unique_label_shots() {
        let row = |activity: &str, context: &str, label: usize| json!({"activity_label":activity,"ctx":context,"endings":["a","b","c","d"],"label":label});
        let source = InMemoryDatasetSource::from_splits([
            (DatasetSplit::Train, vec![row("Sailing", "shot", 1)]),
            (DatasetSplit::Validation, vec![row("Sailing", "query", 2)]),
        ]);
        let problems = HellaSwagBenchmark
            .load_problems(
                &source,
                &BenchmarkConfig {
                    tasks: vec!["sailing".to_string()],
                    n_shots: 1,
                    enable_cot: false,
                    max_problems: None,
                    max_tokens: None,
                },
            )
            .unwrap();
        let prompt = &problems[0].messages[0].content;
        assert!(prompt.contains("shot\nA. a\nB. b\nC. c\nD. d\nAnswer: B"));
        assert!(prompt.ends_with(HELLASWAG_CONFINEMENT));
        assert_eq!(problems[0].ground_truth, "C");
    }

    #[test]
    fn task_validation_uses_pinned_deepeval_enum_before_rows() {
        let all = resolve_tasks(&[]).unwrap();
        assert_eq!(all.first().map(String::as_str), Some("Applying sunscreen"));
        assert_eq!(all.last().map(String::as_str), Some("Croquet"));
        assert!(all.len() > 100);
        assert_eq!(
            resolve_tasks(&["HOME_CATEGORIES".to_string()]).unwrap(),
            ["Home,Categories"]
        );
        assert_eq!(
            resolve_tasks(&["applying sunscreen".to_string()]).unwrap(),
            ["Applying sunscreen"]
        );
        assert!(
            HellaSwagBenchmark
                .validate_config(&BenchmarkConfig {
                    tasks: vec!["NOT_A_REAL_TASK".to_string()],
                    n_shots: 0,
                    enable_cot: false,
                    max_problems: None,
                    max_tokens: None,
                })
                .is_err()
        );
    }
}
