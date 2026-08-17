# AIPerf runtime strict clippy baseline inventory

Command: `env -u RUSTC_WRAPPER cargo clippy -p aiperf-runtime --all-targets --features engine -- -D warnings`

Workspace note: the repository workspace manifest is at `rust/Cargo.toml`; the command was run from `rust/` because the outer repository directory has no Cargo.toml. Full raw output was captured at `/tmp/aiperf-runtime-clippy.txt` during collection.

The command exited non-zero with 149 compilation errors (the diagnostic stream contains 157 source-location records; repeated records are retained exactly as emitted).

## graph/recording

- `runtime/src/graph/recorded/agent_recording/tests.rs:229:42` — digits grouped inconsistently by underscores
- `runtime/src/graph/recorded/trie/timing.rs:21:5` — variant `StartToStart` is never constructed
- `runtime/src/graph/driver.rs:429:5` — this function has too many arguments (8/7)
- `runtime/src/graph/driver.rs:469:1` — this `impl` can be derived
- `runtime/src/graph/executor.rs:200:35` — this expression creates a reference which is immediately dereferenced by the compiler
- `runtime/src/graph/executor.rs:525:35` — this expression creates a reference which is immediately dereferenced by the compiler
- `runtime/src/graph/executor.rs:544:35` — this expression creates a reference which is immediately dereferenced by the compiler
- `runtime/src/graph/recorded/content.rs:195:5` — methods called `new` usually return `Self`
- `runtime/src/graph/recorded/trie/messages.rs:122:1` — this function has too many arguments (8/7)
- `runtime/src/graph/recorded/trie/mod.rs:82:1` — this function has too many arguments (8/7)
- `runtime/src/graph/recorded/weka/mod.rs:256:13` — unnecessary `if let` since only the `Some` variant of the iterator element is used
- `runtime/src/graph/snapshot.rs:413:9` — this `if` statement can be collapsed
- `runtime/src/graph/tools/dispatch.rs:487:1` — items after a test module
- `runtime/src/graph/workload.rs:1154:31` — manual implementation of `.is_multiple_of()`

## execution

- `runtime/src/scheduled.rs:513:5` — field `discard_local_measurement` is never read
- `runtime/src/eval/execution/compose_project.rs:78:19` — associated items `reserve` and `state` are never used
- `runtime/src/eval/execution/docker_process.rs:3191:4` — function `prepare_verifier_files` is never used
- `runtime/src/eval/execution/task_environment.rs:15:19` — struct `ServiceHandle` is never constructed
- `runtime/src/eval/execution/task_environment.rs:20:19` — associated items `new` and `service` are never used
- `runtime/src/eval/execution/task_environment.rs:37:16` — field `network_lease` is never read
- `runtime/src/eval/execution/task_environment.rs:52:8` — method `service` is never used
- `runtime/src/eval/execution/task_environment.rs:85:19` — struct `DockerfileEnvironmentLease` is never constructed
- `runtime/src/eval/execution/task_environment.rs:93:19` — associated function `new` is never used
- `runtime/src/export/genai_perf.rs:15:5` — doc list item without indentation
- `runtime/src/export/genai_perf.rs:16:5` — doc list item without indentation
- `runtime/src/export/genai_perf.rs:17:5` — doc list item without indentation
- `runtime/src/export/genai_perf.rs:18:5` — doc list item without indentation
- `runtime/src/export/genai_perf.rs:19:5` — doc list item without indentation
- `runtime/src/export/parquet/units.rs:155:5` — consider using `sort_by_key`
- `runtime/src/export/parquet.rs:452:24` — this `if` statement can be collapsed
- `runtime/src/fixed_schedule.rs:190:9` — this `if` statement can be collapsed
- `runtime/src/multiturn.rs:1235:21` — this expression creates a reference which is immediately dereferenced by the compiler
- `runtime/src/multiturn.rs:1254:32` — unnecessary closure used with `bool::then`
- `runtime/src/engine/cellular_cell.rs:912:1` — large size difference between variants
- `runtime/src/engine/cellular_controller.rs:1248:9` — this `if` statement can be collapsed
- `runtime/src/engine/cellular_controller.rs:1397:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1398:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1399:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1400:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1401:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1402:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1403:5` — doc list item without indentation
- `runtime/src/engine/cellular_controller.rs:1404:5` — doc list item without indentation
- `runtime/src/engine/control_hooks.rs:270:6` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/engine/coordinator.rs:450:5` — unnecessary closure used with `bool::then`
- `runtime/src/engine/dataset_input.rs:520:23` — this lifetime isn't used in the function definition
- `runtime/src/engine/dry_run.rs:775:5` — this function has too many arguments (12/7)
- `runtime/src/engine/execute/compose_sidecars.rs:940:9` — this `if` statement can be collapsed
- `runtime/src/engine/execute/dataset_build.rs:433:1` — this function has too many arguments (8/7)
- `runtime/src/engine/execute/dataset_build.rs:523:1` — this function has too many arguments (8/7)
- `runtime/src/engine/execute/entrypoints.rs:28:1` — this function has too many arguments (8/7)
- `runtime/src/engine/execute/entrypoints.rs:223:1` — this function has too many arguments (10/7)
- `runtime/src/engine/execute/sharding.rs:191:1` — large size difference between variants
- `runtime/src/engine/graph_execution.rs:768:9` — doc list item without indentation
- `runtime/src/engine/graph_execution.rs:769:9` — doc list item without indentation
- `runtime/src/engine/graph_execution.rs:891:51` — this expression creates a reference which is immediately dereferenced by the compiler
- `runtime/src/engine/graph_execution.rs:1012:13` — this `if` statement can be collapsed
- `runtime/src/engine/graph_execution.rs:1020:9` — this `if` statement can be collapsed
- `runtime/src/engine/graph_input.rs:825:25` — unnecessary closure used to substitute value for `Option::None`
- `runtime/src/engine/graph_input.rs:1192:32` — called `map_or(None, ..)` on an `Option` value
- `runtime/src/engine/graph_input.rs:1328:32` — called `map_or(None, ..)` on an `Option` value
- `runtime/src/engine/graph_phase_runtime.rs:259:5` — clamp-like pattern without using clamp function
- `runtime/src/engine/graph_phase_runtime.rs:2393:12` — manual implementation of `Option::map`
- `runtime/src/engine/protocol_v2.rs:416:13` — this `if` statement can be collapsed
- `runtime/src/engine/turn_execution.rs:365:1` — large size difference between variants
- `runtime/src/engine/turn_execution.rs:408:1` — large size difference between variants
- `runtime/src/engine/turn_execution.rs:1250:28` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/adaptive_core/actuator.rs:163:5` — methods called `new` usually return `Self`
- `runtime/src/adaptive_core/actuator.rs:177:5` — methods called `new` usually return `Self`
- `runtime/src/clock/mod.rs:6:1` — module has the same name as its containing module
- `runtime/src/config/resolve.rs:1383:27` — unnecessary closure used with `bool::then`
- `runtime/src/config/resolve.rs:1438:19` — struct update has no effect, all the fields in the struct have already been specified
- `runtime/src/config/validate.rs:117:5` — this `if` statement can be collapsed
- `runtime/src/endpoints/mod.rs:19:1` — module has the same name as its containing module
- `runtime/src/eval/execution/artifacts.rs:547:9` — this `if` statement can be collapsed
- `runtime/src/eval/execution/compose_project.rs:275:43` — this expression creates a reference which is immediately dereferenced by the compiler
- `runtime/src/eval/execution/docker_process.rs:775:23` — try not to call a closure in the expression where it is declared
- `runtime/src/eval/execution/docker_process.rs:937:5` — this function has too many arguments (8/7)
- `runtime/src/eval/execution/docker_process.rs:1658:5` — returning the result of a `let` binding from a block
- `runtime/src/eval/execution/docker_process.rs:1663:1` — this function has too many arguments (8/7)
- `runtime/src/eval/execution/docker_process.rs:1686:1` — this function has too many arguments (9/7)
- `runtime/src/eval/execution/docker_process.rs:3108:1` — this function has too many arguments (8/7)
- `runtime/src/eval/execution/docker_process.rs:3155:1` — this function has too many arguments (8/7)
- `runtime/src/eval/execution/docker_process.rs:3308:1` — this function has too many arguments (9/7)
- `runtime/src/eval/execution/docker_process.rs:3815:5` — this block may be rewritten with the `?` operator
- `runtime/src/eval/execution/docker_runtime.rs:36:1` — this function has too many arguments (11/7)
- `runtime/src/metrics_core/definition.rs:169:5` — this `if` statement can be collapsed
- `runtime/src/metrics_core/sweepline/mod.rs:464:13` — it looks like you're manually copying between slices
- `runtime/src/rng/compat/numpy_generator.rs:118:29` — casting to the same type is unnecessary (`u64` -> `u64`)
- `runtime/src/rng/configured.rs:58:1` — large size difference between variants
- `runtime/src/transport/http/sink.rs:208:15` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/run.rs:234:21` — function `run_paced_with_backend` is never used
- `runtime/src/run.rs:695:21` — function `run_user_centric_adaptive_with_backend` is never used
- `runtime/src/scheduled.rs:513:5` — field `discard_local_measurement` is never read
- `runtime/src/transport/http/sink.rs:942:12` — struct `RecordingObserver` is never constructed
- `runtime/src/test_util.rs:40:8` — function `chat_dispatch_table` is never used
- `runtime/src/export/genai_perf.rs:15:5` — doc list item without indentation
- `runtime/src/export/server_metrics/tests.rs:290:5` — field assignment outside of initializer for an instance created with Default::default()
- `runtime/src/export/server_metrics/tests.rs:330:5` — field assignment outside of initializer for an instance created with Default::default()
- `runtime/src/metrics.rs:1209:9` — field assignment outside of initializer for an instance created with Default::default()
- `runtime/src/engine/graph_input.rs:1820:14` — called `.err().expect()` on a `Result` value
- `runtime/src/eval/execution/multi_step.rs:93:1` — items after a test module
- `runtime/src/rng/compat/numpy_generator.rs:263:38` — float has excessive precision
- `runtime/src/rng/compat/numpy_generator.rs:264:38` — float has excessive precision
- `runtime/src/rng/compat/numpy_generator.rs:265:38` — float has excessive precision
- `runtime/src/rng/compat/numpy_generator.rs:278:38` — float has excessive precision
- `runtime/src/config/phase_validate.rs:344:49` — useless use of `vec!`
- `runtime/src/config/phase_validate.rs:359:49` — useless use of `vec!`

## datasets

- `runtime/src/dataset/analysis/prefix_cache.rs:195:17` — this `if` statement can be collapsed
- `runtime/src/dataset/analysis.rs:539:9` — this `if` statement can be collapsed
- `runtime/src/dataset/mod.rs:16:1` — module has the same name as its containing module
- `runtime/src/dataset/dataset.rs:1175:25` — manual implementation of `.is_multiple_of()`
- `runtime/src/dataset/loader/public.rs:304:28` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/dataset/loader/public.rs:306:27` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/dataset/loader/public.rs:673:33` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/dataset/loader/public.rs:675:31` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/dataset/dataset.rs:1175:25` — manual implementation of `.is_multiple_of()`
- `runtime/src/dataset/fetch.rs:271:9` — field assignment outside of initializer for an instance created with Default::default()
- `runtime/src/dataset/fetch.rs:279:9` — field assignment outside of initializer for an instance created with Default::default()
- `runtime/src/dataset/request.rs:3634:9` — for loop over a single element
- `runtime/src/dataset/loader/baseten.rs:1384:17` — length comparison to one

## agent replay/AgentX

- `runtime/src/agentic_replay.rs:354:23` — this `MutexGuard` is held across an await point
- `runtime/src/agentic_replay.rs:1028:43` — this `map_or` can be simplified
- `runtime/src/agentx/chains.rs:263:9` — this `if` statement can be collapsed
- `runtime/src/agentx/chains.rs:358:13` — writing `&mut Vec` instead of `&mut [_]` involves a new object where a slice will do
- `runtime/src/agentx/chains.rs:489:9` — this `if` statement can be collapsed
- `runtime/src/agentx/chains.rs:490:13` — this `if` statement can be collapsed
- `runtime/src/agentx/chains.rs:537:5` — this `if` statement can be collapsed
- `runtime/src/agentx/chains.rs:538:9` — this `if` statement can be collapsed
- `runtime/src/agentx/chains.rs:631:14` — the loop variable `i` is used to index `cands`
- `runtime/src/agentx/chains.rs:803:6` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/agentx/loader.rs:149:5` — this `if` statement can be collapsed
- `runtime/src/agentx/loader.rs:203:5` — this `if` statement can be collapsed
- `runtime/src/agentx/loader.rs:236:1` — this function has too many arguments (8/7)
- `runtime/src/agentx/loader.rs:743:13` — this `if` statement can be collapsed
- `runtime/src/agentx/plan.rs:208:6` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/agentx/plan.rs:209:18` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/agentx/replay.rs:137:6` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/agentx/replay.rs:127:1` — this function has too many arguments (9/7)
- `runtime/src/agentx/scenario.rs:715:30` — this method chain can be written more clearly with `if .. else ..`
- `runtime/src/agentx/selection.rs:45:9` — this `if` statement can be collapsed
- `runtime/src/agentx/selection.rs:53:9` — this `if` statement can be collapsed
- `runtime/src/agentx/session_tree.rs:47:15` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/agentx/session_tree.rs:68:52` — very complex type used. Consider factoring parts into `type` definitions
- `runtime/src/agentx/session_tree.rs:89:9` — this `if` statement can be collapsed
- `runtime/src/agentx/session_tree.rs:145:9` — this `if` statement can be collapsed
- `runtime/src/agentx/session_tree.rs:190:13` — this `if` statement can be collapsed
- `runtime/src/agentx/synth.rs:514:21` — this `repeat().take()` can be written more concisely
- `runtime/src/agentx/trajectory_source.rs:144:9` — this `if` statement can be collapsed
- `runtime/src/agentx/weka_dataset.rs:78:61` — you seem to use `.enumerate()` and immediately discard the index
- `runtime/src/agentx/loader.rs:971:9` — doc list item without indentation
- `runtime/src/agentx/loader.rs:972:9` — doc list item without indentation
- `runtime/src/agentx/loader.rs:973:9` — doc list item without indentation
- `runtime/src/agentx/switch.rs:262:17` — length comparison to one

## engine tests

- `runtime/src/engine/workers_characterization.rs:100:33` — useless conversion to the same type: `std::net::TcpListener`
- `runtime/src/engine/workers_characterization.rs:1795:33` — useless conversion to the same type: `std::net::TcpListener`
- `runtime/src/engine/workers_characterization.rs:1866:20` — manual implementation of `.is_multiple_of()`
