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
//! Generation is local and offline: it reads no SLURM environment and contacts no
//! controller. It does require the referenced Config v2 file to exist, and it mints
//! the run's cellular bootstrap material into a private run directory, so a submitted
//! script cannot fail on a path typo — or on absent role material — minutes into an
//! allocation.

use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use aiperf_runtime::engine::cellular_bootstrap::{
    CellularRole, DeploymentMaterial, mint_deployment_material,
};
use clap::Parser;

/// The `slurm generate` subcommand token routed to this module.
pub const GENERATE_SUBCOMMAND: &str = "generate";

/// Velo bootstrap port the controller binds when the script does not override it.
const DEFAULT_CONTROLLER_PORT: u16 = 9500;

/// Run-directory child holding one run's private bootstrap bundles.
const BOOTSTRAP_DIRECTORY: &str = "bootstrap";

/// Bundle filename the rank-0 controller task reads.
const CONTROLLER_BUNDLE: &str = "controller.bin";

/// Bundle filename cell `cell_id` reads, derived by rank inside `aiperf slurm run`.
fn cell_bundle_name(cell_id: u32) -> String {
    format!("cell-{cell_id}.bin")
}

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
    /// Run directory holding this run's private bootstrap material
    /// (default: `<job-name>-slurm-run` beside the resolved config).
    #[arg(long)]
    run_dir: Option<PathBuf>,
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
    run_dir: Option<&'a Path>,
}

/// The generated script text plus the run directory its exports name.
#[derive(Debug)]
struct GeneratedScript {
    text: String,
    run_dir: PathBuf,
}

/// Run `aiperf slurm generate` with the arguments following `generate`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let cli = GenerateCli::try_parse_from(
        std::iter::once("aiperf slurm generate".to_owned()).chain(args.iter().cloned()),
    )
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
        run_dir: cli.run_dir.as_deref(),
    })?;

    // Mint before emitting: a refused run directory must not hand back a script
    // whose exports name material that was never written.
    provision_material(&script.run_dir, cli.cells)?;

    match cli.output.as_deref() {
        Some(path) => std::fs::write(path, &script.text)
            .map_err(|error| anyhow::anyhow!("failed to write {}: {error}", path.display()))?,
        None => {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout.write_all(script.text.as_bytes())?;
            stdout.flush()?;
        }
    }
    Ok(0)
}

/// Mint one run's cellular material and place each bundle in its private run directory.
///
/// The `bootstrap` directory is created `0700` with create-new semantics, so a second
/// generation into the same run directory is refused rather than truncating material an
/// already-submitted allocation still depends on. Each bundle is written no-follow,
/// create-new, `0600` — the same private-file contract the Kubernetes submission path
/// uses — because `aiperf slurm run` refuses any bootstrap mount that is not exactly that.
fn provision_material(run_dir: &Path, cells: u32) -> anyhow::Result<DeploymentMaterial> {
    let roles = (0..cells).map(CellularRole::Cell).collect::<Vec<_>>();
    // Mint first: a rejected roster must not leave an empty private directory behind.
    let material = mint_deployment_material(&roles)
        .map_err(|error| anyhow::anyhow!("failed to mint cellular bootstrap material: {error}"))?;

    std::fs::create_dir_all(run_dir).map_err(|error| {
        anyhow::anyhow!("failed to create run directory {}: {error}", run_dir.display())
    })?;
    let directory = run_dir.join(BOOTSTRAP_DIRECTORY);
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&directory)
        .map_err(|error| {
            anyhow::anyhow!(
                "refusing to replace existing bootstrap material in {}: {error} \
                 (remove the directory or pass a different --run-dir)",
                directory.display()
            )
        })?;

    write_private_bundle(&directory.join(CONTROLLER_BUNDLE), &material.controller)?;
    for (role, bytes) in &material.roles {
        let CellularRole::Cell(cell_id) = role else {
            anyhow::bail!("minted material carries an unexpected role {role:?}");
        };
        write_private_bundle(&directory.join(cell_bundle_name(*cell_id)), bytes)?;
    }
    tracing::debug!(
        directory = %directory.display(),
        cells,
        "minted SLURM cellular bootstrap material"
    );
    Ok(material)
}

/// Write one bundle as a no-follow, create-new, `0600` regular file.
fn write_private_bundle(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    let write = || -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        file.write_all(contents)?;
        file.sync_all()
    };
    write().map_err(|error| {
        anyhow::anyhow!(
            "failed to write bootstrap bundle {}: {error}",
            path.display()
        )
    })
}

/// Build the sbatch job-script text for a native cellular AIPerf run.
///
/// `cells` counts only the load-generating tasks; the controller is an extra
/// task, so `--ntasks` is `cells + 1` and `--nodes` defaults to the same value
/// (one task per node) when not overridden. The `srun` line and the bootstrap
/// exports carry absolute paths because the submitted script's working directory
/// is the allocation's, not the caller's.
///
/// The script exports the controller's bundle path and the *directory* holding the
/// cell bundles: `srun` gives every task one environment, so each cell resolves its
/// own bundle from its rank inside the binary rather than from shell expansion.
fn build_sbatch_script(request: &ScriptRequest<'_>) -> anyhow::Result<GeneratedScript> {
    anyhow::ensure!(
        request.cells >= 1,
        "--cells must be >= 1 (got {})",
        request.cells
    );
    let config = std::fs::canonicalize(request.config)
        .map_err(|_| anyhow::anyhow!("config file does not exist: {}", request.config.display()))?;
    let run_dir = resolve_run_dir(request, &config)?;
    let bootstrap = run_dir.join(BOOTSTRAP_DIRECTORY);

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
    script.push_str(&format!(
        "export AIPERF_CONTROLLER_BOOTSTRAP_FILE={}\n",
        bootstrap.join(CONTROLLER_BUNDLE).display()
    ));
    script.push_str(&format!(
        "export AIPERF_ROLE_BOOTSTRAP_DIR={}\n",
        bootstrap.display()
    ));
    script.push('\n');
    script.push_str(&format!(
        "srun aiperf slurm run --config {}\n",
        config.display()
    ));
    Ok(GeneratedScript {
        text: script,
        run_dir,
    })
}

/// Resolve the absolute run directory: the explicit `--run-dir`, else
/// `<job-name>-slurm-run` beside the resolved config.
fn resolve_run_dir(request: &ScriptRequest<'_>, config: &Path) -> anyhow::Result<PathBuf> {
    match request.run_dir {
        Some(path) => std::path::absolute(path).map_err(|error| {
            anyhow::anyhow!("failed to resolve --run-dir {}: {error}", path.display())
        }),
        None => {
            let parent = config.parent().ok_or_else(|| {
                anyhow::anyhow!(
                    "config path {} has no parent directory for the default --run-dir",
                    config.display()
                )
            })?;
            Ok(parent.join(format!("{}-slurm-run", request.job_name)))
        }
    }
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
            run_dir: None,
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
        let run_dir = absolute
            .parent()
            .expect("config parent")
            .join("aiperf-slurm-run");

        let script = build_sbatch_script(&request(&config, 4)).expect("script");

        assert_eq!(script.run_dir, run_dir);
        assert_eq!(
            script.text,
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
                 export AIPERF_CONTROLLER_BOOTSTRAP_FILE={run}/bootstrap/controller.bin\n\
                 export AIPERF_ROLE_BOOTSTRAP_DIR={run}/bootstrap\n\
                 \n\
                 srun aiperf slurm run --config {config}\n",
                run = run_dir.display(),
                config = absolute.display()
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
        .expect("script")
        .text;

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
        // The default run directory is named for the job, beside the config.
        assert!(script.contains("/myrun-slurm-run/bootstrap\n"), "{script}");
    }

    #[test]
    fn generate_mints_per_rank_material() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("fixture directory");
        let config = config_fixture(&directory);
        let run_dir = directory.path().join("run");

        let material = provision_material(&run_dir, 3).expect("minted material");

        let bootstrap = run_dir.join(BOOTSTRAP_DIRECTORY);
        assert_eq!(
            std::fs::metadata(&bootstrap)
                .expect("bootstrap directory")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        let controller = bootstrap.join(CONTROLLER_BUNDLE);
        assert_eq!(
            std::fs::read(&controller).expect("controller bundle"),
            material.controller
        );
        assert_eq!(
            std::fs::metadata(&controller)
                .expect("controller metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        for cell in 0..3 {
            let path = bootstrap.join(cell_bundle_name(cell));
            let bytes = std::fs::read(&path).expect("cell bundle");
            assert_eq!(
                &bytes,
                material
                    .roles
                    .get(&CellularRole::Cell(cell))
                    .expect("minted cell bundle"),
                "cell {cell} bundle must be the one minted for its role"
            );
            // Role tag: kind 1 (cell), tier 0, little-endian cell id.
            assert_eq!(bytes[9], 1, "cell {cell} bundle is not cell material");
            assert_eq!(
                u32::from_le_bytes(bytes[14..18].try_into().expect("role id bytes")),
                cell
            );
            assert_eq!(
                std::fs::metadata(&path)
                    .expect("cell metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        // The generated script names exactly the provisioned paths.
        let script = build_sbatch_script(&ScriptRequest {
            run_dir: Some(&run_dir),
            ..request(&config, 3)
        })
        .expect("script");
        assert!(
            script
                .text
                .contains(&format!("export AIPERF_ROLE_BOOTSTRAP_DIR={}\n", bootstrap.display())),
            "{}",
            script.text
        );
    }

    #[test]
    fn generate_refuses_to_overwrite_existing_material() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let run_dir = directory.path().join("run");

        provision_material(&run_dir, 2).expect("first mint");
        let bootstrap = run_dir.join(BOOTSTRAP_DIRECTORY);
        let before = std::fs::read(bootstrap.join(CONTROLLER_BUNDLE)).expect("first controller");

        let error = provision_material(&run_dir, 2).expect_err("second mint must be refused");
        assert!(
            error.to_string().contains("bootstrap material"),
            "{error}"
        );
        assert_eq!(
            std::fs::read(bootstrap.join(CONTROLLER_BUNDLE)).expect("controller after refusal"),
            before,
            "the first run's material must survive byte-identically"
        );
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
