// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native sbatch job-script generation: `aiperf slurm generate`.
//!
//! Emits the batch script that submits a native cellular run under SLURM. The
//! script requests `cells + 1` tasks — one controller (rank 0) plus `cells`
//! load-generating cell tasks — exports the launcher selection every task reads
//! ([`super::run`] consumes it), and ends with the single `srun aiperf slurm run
//! --config <abs-path>` line that starts every task with the identical command.
//!
//! Generation is local and offline: it reads no SLURM environment, contacts no
//! controller, and only requires the referenced Config v2 file to exist so a
//! submitted script cannot fail on a path typo minutes into an allocation.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;

/// The `slurm generate` subcommand token routed to this module.
pub const GENERATE_SUBCOMMAND: &str = "generate";

/// Velo bootstrap port the controller binds when the script does not override it.
const DEFAULT_CONTROLLER_PORT: u16 = 9500;

#[derive(Debug, Parser)]
#[command(
    name = "aiperf slurm generate",
    about = "Generate an sbatch job script for a native cellular AIPerf benchmark",
    disable_help_subcommand = true
)]
struct GenerateCli {
    /// Path to the AIPerf Config v2 YAML file.
    #[arg(long)]
    config: PathBuf,
    /// Number of load-generating cells (controller is an extra task).
    #[arg(long)]
    cells: u32,
    /// SLURM job name.
    #[arg(long, default_value = "aiperf")]
    job_name: String,
    /// SLURM partition.
    #[arg(long)]
    partition: Option<String>,
    /// SLURM account.
    #[arg(long)]
    account: Option<String>,
    /// Job time limit (HH:MM:SS).
    #[arg(long, default_value = "01:00:00")]
    time: String,
    /// Node count (default: cells + 1).
    #[arg(long)]
    nodes: Option<u32>,
    /// Tasks per node.
    #[arg(long, default_value_t = 1)]
    ntasks_per_node: u32,
    /// GPUs per node (optional).
    #[arg(long)]
    gpus_per_node: Option<u32>,
    /// Velo bootstrap port for the controller (AIPERF_CONTROLLER_PORT).
    #[arg(long, default_value_t = DEFAULT_CONTROLLER_PORT)]
    controller_port: u16,
    /// Write the script to this file instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

/// Resolved job-script inputs, separated from clap so the builder is testable.
struct ScriptRequest<'a> {
    config: &'a Path,
    cells: u32,
    job_name: &'a str,
    partition: Option<&'a str>,
    account: Option<&'a str>,
    time: &'a str,
    nodes: Option<u32>,
    ntasks_per_node: u32,
    gpus_per_node: Option<u32>,
    controller_port: u16,
}

/// Run `aiperf slurm generate` with the arguments following `generate`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let cli = GenerateCli::try_parse_from(std::iter::once("aiperf slurm generate".to_owned()).chain(args.iter().cloned()))
        .map_err(|error| anyhow::anyhow!("{}", error.render().ansi()))?;
    let script = build_sbatch_script(&ScriptRequest {
        config: &cli.config,
        cells: cli.cells,
        job_name: &cli.job_name,
        partition: cli.partition.as_deref(),
        account: cli.account.as_deref(),
        time: &cli.time,
        nodes: cli.nodes,
        ntasks_per_node: cli.ntasks_per_node,
        gpus_per_node: cli.gpus_per_node,
        controller_port: cli.controller_port,
    })?;

    match cli.output.as_deref() {
        Some(path) => std::fs::write(path, &script)
            .map_err(|error| anyhow::anyhow!("failed to write {}: {error}", path.display()))?,
        None => {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout.write_all(script.as_bytes())?;
            stdout.flush()?;
        }
    }
    Ok(0)
}

/// Build the sbatch job-script text for a native cellular AIPerf run.
///
/// `cells` counts only the load-generating tasks; the controller is an extra
/// task, so `--ntasks` is `cells + 1` and `--nodes` defaults to the same value
/// (one task per node) when not overridden. The `srun` line carries the config's
/// absolute path because the submitted script's working directory is the
/// allocation's, not the caller's.
fn build_sbatch_script(request: &ScriptRequest<'_>) -> anyhow::Result<String> {
    anyhow::ensure!(
        request.cells >= 1,
        "--cells must be >= 1 (got {})",
        request.cells
    );
    let config = std::fs::canonicalize(request.config).map_err(|_| {
        anyhow::anyhow!("config file does not exist: {}", request.config.display())
    })?;

    let ntasks = u64::from(request.cells) + 1;
    let nodes = request.nodes.map_or(ntasks, u64::from);

    let mut script = String::new();
    script.push_str("#!/bin/bash\n");
    script.push_str(&format!("#SBATCH --job-name={}\n", request.job_name));
    script.push_str(&format!("#SBATCH --nodes={nodes}\n"));
    script.push_str(&format!("#SBATCH --ntasks={ntasks}\n"));
    script.push_str(&format!(
        "#SBATCH --ntasks-per-node={}\n",
        request.ntasks_per_node
    ));
    script.push_str(&format!("#SBATCH --time={}\n", request.time));
    if let Some(partition) = request.partition {
        script.push_str(&format!("#SBATCH --partition={partition}\n"));
    }
    if let Some(account) = request.account {
        script.push_str(&format!("#SBATCH --account={account}\n"));
    }
    if let Some(gpus_per_node) = request.gpus_per_node {
        script.push_str(&format!("#SBATCH --gpus-per-node={gpus_per_node}\n"));
    }

    script.push('\n');
    script.push_str("export AIPERF_CELL_LAUNCHER=slurm\n");
    script.push_str(&format!(
        "export AIPERF_CONTROLLER_PORT={}\n",
        request.controller_port
    ));
    script.push('\n');
    script.push_str(&format!(
        "srun aiperf slurm run --config {}\n",
        config.display()
    ));
    Ok(script)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(config: &'a Path, cells: u32) -> ScriptRequest<'a> {
        ScriptRequest {
            config,
            cells,
            job_name: "aiperf",
            partition: None,
            account: None,
            time: "01:00:00",
            nodes: None,
            ntasks_per_node: 1,
            gpus_per_node: None,
            controller_port: DEFAULT_CONTROLLER_PORT,
        }
    }

    fn config_fixture(directory: &tempfile::TempDir) -> PathBuf {
        let path = directory.path().join("benchmark.yaml");
        std::fs::write(&path, "benchmark: {}\n").expect("write config fixture");
        path
    }

    #[test]
    fn default_script_is_exact() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let config = config_fixture(&directory);
        let absolute = std::fs::canonicalize(&config).expect("canonical config");

        let script = build_sbatch_script(&request(&config, 4)).expect("script");

        assert_eq!(
            script,
            format!(
                "#!/bin/bash\n\
                 #SBATCH --job-name=aiperf\n\
                 #SBATCH --nodes=5\n\
                 #SBATCH --ntasks=5\n\
                 #SBATCH --ntasks-per-node=1\n\
                 #SBATCH --time=01:00:00\n\
                 \n\
                 export AIPERF_CELL_LAUNCHER=slurm\n\
                 export AIPERF_CONTROLLER_PORT=9500\n\
                 \n\
                 srun aiperf slurm run --config {}\n",
                absolute.display()
            )
        );
    }

    #[test]
    fn optional_directives_and_node_override_apply() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let config = config_fixture(&directory);
        let script = build_sbatch_script(&ScriptRequest {
            job_name: "myrun",
            partition: Some("batch"),
            account: Some("proj123"),
            time: "02:30:00",
            nodes: Some(2),
            gpus_per_node: Some(8),
            controller_port: 9700,
            ..request(&config, 4)
        })
        .expect("script");

        assert!(script.contains("#SBATCH --job-name=myrun\n"));
        assert!(script.contains("#SBATCH --partition=batch\n"));
        assert!(script.contains("#SBATCH --account=proj123\n"));
        assert!(script.contains("#SBATCH --gpus-per-node=8\n"));
        assert!(script.contains("#SBATCH --time=02:30:00\n"));
        // The node override never changes the task count: the controller is
        // always one task beyond the requested cells.
        assert!(script.contains("#SBATCH --nodes=2\n"));
        assert!(script.contains("#SBATCH --ntasks=5\n"));
        assert!(script.contains("export AIPERF_CONTROLLER_PORT=9700\n"));
    }

    #[test]
    fn invalid_inputs_are_refused() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let config = config_fixture(&directory);

        let zero_cells = build_sbatch_script(&request(&config, 0)).expect_err("cells refused");
        assert!(
            zero_cells.to_string().contains("--cells must be >= 1"),
            "{zero_cells}"
        );

        let missing = directory.path().join("nope.yaml");
        let error = build_sbatch_script(&request(&missing, 1)).expect_err("missing config refused");
        assert!(
            error.to_string().contains("config file does not exist"),
            "{error}"
        );
    }
}
