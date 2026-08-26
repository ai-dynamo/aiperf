// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-shot acquisition of process-private cellular security material.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
#[cfg(unix)]
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
#[cfg(test)]
use std::thread::sleep;

use anyhow::{Context, Result, bail, ensure};
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::TryRngCore;
use velo::PeerInfo;

use crate::engine::cellular_registration::{
    CellRegistrationCredential, CellSecurityContext, RoleVerifyingKey,
};

/// Fixed descriptor inherited by a same-host cell process.
pub(crate) const CELL_SECURITY_FD: i32 = 3;
/// Environment variable containing only [`CELL_SECURITY_FD`], never key bytes.
pub(crate) const CELL_SECURITY_FD_ENV: &str = "AIPERF_CELL_SECURITY_FD";
/// Deployment-mounted controller material visible only to the controller role.
pub(crate) const CONTROLLER_BOOTSTRAP_FILE_ENV: &str = "AIPERF_CONTROLLER_BOOTSTRAP_FILE";
/// Deployment-mounted role material visible only to the process it authorizes.
pub(crate) const ROLE_BOOTSTRAP_FILE_ENV: &str = "AIPERF_ROLE_BOOTSTRAP_FILE";

const MATERIAL_MAGIC: &[u8; 8] = b"AIPRFSEC";
const MATERIAL_VERSION: u8 = 1;
const FIXED_MATERIAL_BYTES: usize = 8 + 1 + 9 + 32 + 32 + 32 + 4;
const ROSTER_ENTRY_BYTES: usize = 9 + 32;
const MAX_ROLE_COUNT: usize = 1024;
const MAX_MATERIAL_BYTES: usize = FIXED_MATERIAL_BYTES + MAX_ROLE_COUNT * ROSTER_ENTRY_BYTES;

/// One process identity in a cellular run.
#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum CellularRole {
    /// A benchmark-executing cell.
    Cell(u32),
    /// A fold-only reduction-tree aggregator.
    Aggregator { tier: u32, id: u32 },
}

/// Controller acquisition source.
pub(crate) enum ControllerSecuritySource {
    /// Mint all process identities in memory for same-host children.
    LocalMint,
    /// Read one controller-only private binary file.
    DeploymentFile(PathBuf),
}

/// Worker acquisition source.
pub(crate) enum RoleSecuritySource {
    /// Read once from a launcher-owned inherited pipe.
    #[cfg(unix)]
    InheritedFd(OwnedFd),
    /// Read one role-specific private binary file.
    DeploymentFile(PathBuf),
}

/// Controller security plus one-shot same-host worker material, when applicable.
pub(crate) struct PreparedControllerSecurity {
    pub(crate) context: Arc<CellSecurityContext>,
    pub(crate) local_roles: Option<LocalRoleProvisioner>,
}

/// Non-cloneable, controller-owned role material drained exactly once per role.
pub(crate) struct LocalRoleProvisioner {
    materials: BTreeMap<CellularRole, Vec<u8>>,
}

impl LocalRoleProvisioner {
    /// Drain one role's fixed binary material.
    pub(crate) fn take(&mut self, role: CellularRole) -> Result<Vec<u8>> {
        self.materials
            .remove(&role)
            .ok_or_else(|| anyhow::anyhow!("local role material is unavailable for {role:?}"))
    }
}

struct DecodedMaterial {
    run_nonce: [u8; 32],
    signing_key: SigningKey,
    controller_verifier: VerifyingKey,
    roster: Box<[RoleVerifyingKey]>,
}

static PROCESS_CELL_SECURITY: OnceLock<Arc<CellSecurityContext>> = OnceLock::new();
static PROCESS_CELL_SECURITY_STATE: AtomicU8 = AtomicU8::new(SECURITY_VACANT);
const SECURITY_VACANT: u8 = 0;
const SECURITY_ACQUIRING: u8 = 1;
const SECURITY_INSTALLED: u8 = 2;
#[cfg(test)]
static TEST_ROLE_SOURCE_READS: AtomicUsize = AtomicUsize::new(0);

struct ProcessSecurityClaim {
    is_finished: bool,
}

impl ProcessSecurityClaim {
    fn acquire() -> Result<Self> {
        PROCESS_CELL_SECURITY_STATE
            .compare_exchange(
                SECURITY_VACANT,
                SECURITY_ACQUIRING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| {
                anyhow::anyhow!("cell security context is already installed or being acquired")
            })?;
        Ok(Self { is_finished: false })
    }

    fn install(mut self, context: Arc<CellSecurityContext>) -> Result<()> {
        if PROCESS_CELL_SECURITY.set(context).is_err() {
            PROCESS_CELL_SECURITY_STATE.store(SECURITY_INSTALLED, Ordering::Release);
            self.is_finished = true;
            bail!("cell security context installation invariant was violated");
        }
        PROCESS_CELL_SECURITY_STATE.store(SECURITY_INSTALLED, Ordering::Release);
        self.is_finished = true;
        Ok(())
    }
}

impl Drop for ProcessSecurityClaim {
    fn drop(&mut self) {
        if !self.is_finished {
            PROCESS_CELL_SECURITY_STATE.store(SECURITY_VACANT, Ordering::Release);
        }
    }
}

/// Prepare the controller context before any listener is bound.
pub(crate) fn prepare_controller_security(
    source: ControllerSecuritySource,
    roles: &[CellularRole],
) -> Result<PreparedControllerSecurity> {
    ensure_mintable_roster(roles)?;

    match source {
        ControllerSecuritySource::LocalMint => mint_local_security(roles),
        ControllerSecuritySource::DeploymentFile(path) => {
            let bytes = read_private_deployment_file(&path, "controller")?;
            let material = decode_material(&bytes, None)?;
            ensure!(
                material
                    .roster
                    .iter()
                    .map(|entry| entry.role)
                    .eq(roles.iter().copied()),
                "controller deployment material has the wrong roster"
            );
            Ok(PreparedControllerSecurity {
                context: Arc::new(CellSecurityContext::controller(
                    material.run_nonce,
                    material.signing_key,
                    material.roster,
                )?),
                local_roles: None,
            })
        }
    }
}

/// Refuse a roster that `decode_material` would only reject later, at process start.
fn ensure_mintable_roster(roles: &[CellularRole]) -> Result<()> {
    ensure!(!roles.is_empty(), "controller security roster is empty");
    ensure!(
        roles.len() <= MAX_ROLE_COUNT,
        "controller security roster is oversized"
    );
    let unique = roles.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == roles.len(),
        "controller security roster has duplicate roles"
    );
    Ok(())
}

/// One run's freshly minted private material and the public roster naming it.
struct MintedRun {
    run_nonce: [u8; 32],
    controller_signer: SigningKey,
    roster: Box<[RoleVerifyingKey]>,
    role_materials: BTreeMap<CellularRole, Vec<u8>>,
}

/// Mint one run nonce, the controller key, and one key per role, in caller order.
///
/// Every minting path routes through this single block. A second copy of the
/// key-generation sequence could drift into reusing a nonce or a key across runs,
/// so same-host and out-of-band deployment provisioning share exactly this one.
fn mint_run_material(roles: &[CellularRole]) -> Result<MintedRun> {
    let run_nonce = random_bytes("run nonce")?;
    let controller_signer = SigningKey::from_bytes(&random_bytes("controller key")?);
    let controller_verifier = controller_signer.verifying_key();
    let mut role_signers = Vec::with_capacity(roles.len());
    let mut roster = Vec::with_capacity(roles.len());
    for role in roles.iter().copied() {
        let signer = SigningKey::from_bytes(&random_bytes("role key")?);
        roster.push(RoleVerifyingKey {
            role,
            verifier: signer.verifying_key(),
        });
        role_signers.push((role, signer));
    }
    let roster = roster.into_boxed_slice();
    let mut role_materials = BTreeMap::new();
    for (role, signer) in role_signers {
        role_materials.insert(
            role,
            encode_material(Some(role), run_nonce, &signer, controller_verifier, &roster)?,
        );
    }
    Ok(MintedRun {
        run_nonce,
        controller_signer,
        roster,
        role_materials,
    })
}

/// One run's complete deployment material: the controller bundle plus one bundle per role.
///
/// Every bundle shares one run nonce and one roster; the controller bundle carries the
/// signing key whose verifier the roster names. The bytes are opaque outside this module —
/// no private-key type crosses this boundary.
#[derive(Debug)]
pub struct DeploymentMaterial {
    /// Controller bundle, role field absent.
    pub controller: Vec<u8>,
    /// One bundle per requested role, keyed by role.
    pub roles: BTreeMap<CellularRole, Vec<u8>>,
}

/// Mint one run's material for a deployment that provisions every process out of band.
///
/// Unlike the same-host path, no context is installed in this process: the caller
/// receives opaque bytes to place in each process's private mount. The roster is
/// refused up front when it is empty, oversized, or has duplicate roles.
pub fn mint_deployment_material(roles: &[CellularRole]) -> Result<DeploymentMaterial> {
    ensure_mintable_roster(roles)?;
    let minted = mint_run_material(roles)?;
    let controller = encode_material(
        None,
        minted.run_nonce,
        &minted.controller_signer,
        minted.controller_signer.verifying_key(),
        &minted.roster,
    )?;
    Ok(DeploymentMaterial {
        controller,
        roles: minted.role_materials,
    })
}

fn mint_local_security(roles: &[CellularRole]) -> Result<PreparedControllerSecurity> {
    let minted = mint_run_material(roles)?;
    Ok(PreparedControllerSecurity {
        context: Arc::new(CellSecurityContext::controller(
            minted.run_nonce,
            minted.controller_signer,
            minted.roster,
        )?),
        local_roles: Some(LocalRoleProvisioner {
            materials: minted.role_materials,
        }),
    })
}

/// Acquire and install this worker's context exactly once.
pub(crate) fn acquire_process_cell_security(
    expected: CellularRole,
) -> Result<Arc<CellSecurityContext>> {
    let claim = ProcessSecurityClaim::acquire()?;
    let context = acquire_role_security(role_source_from_environment()?, expected)?;
    claim.install(Arc::clone(&context))?;
    Ok(context)
}

fn acquire_role_security(
    source: RoleSecuritySource,
    expected: CellularRole,
) -> Result<Arc<CellSecurityContext>> {
    let bytes = match source {
        #[cfg(unix)]
        RoleSecuritySource::InheritedFd(fd) => read_bounded(File::from(fd), "inherited role")?,
        RoleSecuritySource::DeploymentFile(path) => {
            read_private_deployment_file(&path, "deployment role")?
        }
    };
    let material = decode_material(&bytes, Some(expected))?;
    Ok(Arc::new(CellSecurityContext::worker(
        material.run_nonce,
        expected,
        material.signing_key,
        material.controller_verifier,
    )?))
}

#[cfg(test)]
fn install_process_cell_security(context: Arc<CellSecurityContext>) -> Result<()> {
    ProcessSecurityClaim::acquire()?.install(context)
}

/// Borrow the process-owned context without rereading any source.
pub(crate) fn process_cell_security() -> Result<&'static Arc<CellSecurityContext>> {
    PROCESS_CELL_SECURITY
        .get()
        .ok_or_else(|| anyhow::anyhow!("cell security context is not installed"))
}

fn role_source_from_environment() -> Result<RoleSecuritySource> {
    #[cfg(test)]
    {
        TEST_ROLE_SOURCE_READS.fetch_add(1, Ordering::Relaxed);
        if std::env::var_os("AIPERF_TEST_DELAY_ROLE_SOURCE").is_some() {
            sleep(std::time::Duration::from_millis(100));
        }
    }
    if let Some(value) = std::env::var_os(CELL_SECURITY_FD_ENV) {
        ensure!(
            value == std::ffi::OsStr::new("3"),
            "inherited role source has an invalid descriptor"
        );
        #[cfg(unix)]
        {
            // SAFETY: fcntl only queries whether the fixed descriptor is valid.
            let descriptor_flags = unsafe { libc::fcntl(CELL_SECURITY_FD, libc::F_GETFD) };
            ensure!(
                descriptor_flags >= 0,
                "inherited role source descriptor is unavailable"
            );
            // SAFETY: the local launcher transfers ownership of this fixed descriptor
            // to the child. The process-wide claim prevents a second owner.
            let fd = unsafe { OwnedFd::from_raw_fd(CELL_SECURITY_FD) };
            return Ok(RoleSecuritySource::InheritedFd(fd));
        }
        #[cfg(not(unix))]
        bail!("inherited role source is unsupported on this platform");
    }
    let path = std::env::var_os(ROLE_BOOTSTRAP_FILE_ENV)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("cell has no role security source"))?;
    Ok(RoleSecuritySource::DeploymentFile(PathBuf::from(path)))
}

fn read_private_deployment_file(path: &Path, source_class: &str) -> Result<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("{source_class} security file is unavailable"))?;
    let metadata = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("{source_class} security file is unavailable"))?;
    ensure!(
        metadata.is_file(),
        "{source_class} security file is not regular"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o7777 == 0o600,
            "{source_class} security file is not private"
        );
    }
    ensure!(
        metadata.len() <= MAX_MATERIAL_BYTES as u64,
        "{source_class} security file is oversized"
    );
    read_bounded(file, source_class)
}

fn read_bounded(mut reader: impl Read, source_class: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take((MAX_MATERIAL_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| anyhow::anyhow!("{source_class} security source is unreadable"))?;
    ensure!(
        bytes.len() <= MAX_MATERIAL_BYTES,
        "{source_class} security source is oversized"
    );
    Ok(bytes)
}

fn encode_material(
    role: Option<CellularRole>,
    run_nonce: [u8; 32],
    signer: &SigningKey,
    controller_verifier: VerifyingKey,
    roster: &[RoleVerifyingKey],
) -> Result<Vec<u8>> {
    ensure!(
        roster.len() <= MAX_ROLE_COUNT,
        "security roster is oversized"
    );
    let mut bytes = Vec::with_capacity(FIXED_MATERIAL_BYTES + roster.len() * ROSTER_ENTRY_BYTES);
    bytes.extend_from_slice(MATERIAL_MAGIC);
    bytes.push(MATERIAL_VERSION);
    bytes.extend_from_slice(&encode_role(role));
    bytes.extend_from_slice(&run_nonce);
    bytes.extend_from_slice(&signer.to_bytes());
    bytes.extend_from_slice(controller_verifier.as_bytes());
    bytes.extend_from_slice(&(roster.len() as u32).to_le_bytes());
    for entry in roster {
        bytes.extend_from_slice(&encode_role(Some(entry.role)));
        bytes.extend_from_slice(entry.verifier.as_bytes());
    }
    Ok(bytes)
}

fn decode_material(bytes: &[u8], expected: Option<CellularRole>) -> Result<DecodedMaterial> {
    ensure!(
        bytes.len() <= MAX_MATERIAL_BYTES,
        "security material is oversized"
    );
    ensure!(
        bytes.len() >= FIXED_MATERIAL_BYTES,
        "security material is truncated"
    );
    ensure!(
        &bytes[..8] == MATERIAL_MAGIC,
        "security material has invalid magic"
    );
    ensure!(
        bytes[8] == MATERIAL_VERSION,
        "security material has invalid version"
    );
    let role = decode_role(&bytes[9..18])?;
    ensure!(role == expected, "security material has the wrong role");
    let run_nonce = fixed_array(&bytes[18..50], "run nonce")?;
    let signing_bytes = fixed_array(&bytes[50..82], "signing key")?;
    let controller_bytes = fixed_array(&bytes[82..114], "controller verifier")?;
    let roster_count = u32::from_le_bytes(fixed_array(&bytes[114..118], "roster count")?) as usize;
    ensure!(
        roster_count <= MAX_ROLE_COUNT,
        "security material roster is oversized"
    );
    let expected_len = FIXED_MATERIAL_BYTES
        .checked_add(roster_count.saturating_mul(ROSTER_ENTRY_BYTES))
        .ok_or_else(|| anyhow::anyhow!("security material roster is oversized"))?;
    ensure!(
        bytes.len() >= expected_len,
        "security material is truncated"
    );
    ensure!(
        bytes.len() == expected_len,
        "security material has trailing bytes"
    );
    let signing_key = SigningKey::from_bytes(&signing_bytes);
    let controller_verifier = VerifyingKey::from_bytes(&controller_bytes)
        .map_err(|_| anyhow::anyhow!("security material has an invalid controller verifier"))?;
    let mut seen = BTreeSet::new();
    let mut roster = Vec::with_capacity(roster_count);
    for chunk in bytes[FIXED_MATERIAL_BYTES..].chunks_exact(ROSTER_ENTRY_BYTES) {
        let roster_role = decode_role(&chunk[..9])?
            .ok_or_else(|| anyhow::anyhow!("security material roster contains controller role"))?;
        ensure!(
            seen.insert(roster_role),
            "security material roster has duplicate roles"
        );
        let verifier = VerifyingKey::from_bytes(&fixed_array(&chunk[9..], "role verifier")?)
            .map_err(|_| anyhow::anyhow!("security material has an invalid role verifier"))?;
        roster.push(RoleVerifyingKey {
            role: roster_role,
            verifier,
        });
    }
    ensure!(!roster.is_empty(), "security material roster is empty");
    match role {
        None => ensure!(
            signing_key.verifying_key() == controller_verifier,
            "controller security material has a key mismatch"
        ),
        Some(worker_role) => {
            let worker_verifier = roster
                .iter()
                .find(|entry| entry.role == worker_role)
                .map(|entry| entry.verifier)
                .ok_or_else(|| anyhow::anyhow!("role security material has a run mismatch"))?;
            ensure!(
                worker_verifier == signing_key.verifying_key(),
                "role security material has a run mismatch"
            );
        }
    }
    Ok(DecodedMaterial {
        run_nonce,
        signing_key,
        controller_verifier,
        roster: roster.into_boxed_slice(),
    })
}

fn encode_role(role: Option<CellularRole>) -> [u8; 9] {
    let mut bytes = [0_u8; 9];
    match role {
        None => {}
        Some(CellularRole::Cell(id)) => {
            bytes[0] = 1;
            bytes[5..].copy_from_slice(&id.to_le_bytes());
        }
        Some(CellularRole::Aggregator { tier, id }) => {
            bytes[0] = 2;
            bytes[1..5].copy_from_slice(&tier.to_le_bytes());
            bytes[5..].copy_from_slice(&id.to_le_bytes());
        }
    }
    bytes
}

fn decode_role(bytes: &[u8]) -> Result<Option<CellularRole>> {
    let tier = u32::from_le_bytes(fixed_array(&bytes[1..5], "role tier")?);
    let id = u32::from_le_bytes(fixed_array(&bytes[5..9], "role id")?);
    match bytes[0] {
        0 if tier == 0 && id == 0 => Ok(None),
        1 if tier == 0 => Ok(Some(CellularRole::Cell(id))),
        2 => Ok(Some(CellularRole::Aggregator { tier, id })),
        _ => bail!("security material has an invalid role"),
    }
}

fn fixed_array<const N: usize>(bytes: &[u8], class: &str) -> Result<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("security material has an invalid {class}"))
}

fn random_bytes(class: &str) -> Result<[u8; 32]> {
    let mut bytes = [0_u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut bytes)
        .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular {class}"))?;
    Ok(bytes)
}

/// Connect a fresh Velo instance using only the installed process context.
pub(crate) async fn connect_authenticated_controller(
    velo: &velo::Velo,
    coordinate: &str,
    cell_id: u32,
    context: &Arc<CellSecurityContext>,
) -> Result<(PeerInfo, CellRegistrationCredential)> {
    ensure!(
        context.role() == Some(CellularRole::Cell(cell_id)),
        "installed security context has the wrong role"
    );
    let credential = context.registration_credential()?;
    let peer = crate::cellular::transport::connect::connect_controller(velo, coordinate)
        .await
        .context("connecting to controller")?;
    Ok((peer, credential))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a roster exactly as `encode_material` writes it, so tests compare
    /// wire bytes rather than a structural equality the roster type does not define.
    fn roster_bytes(roster: &[RoleVerifyingKey]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for entry in roster {
            bytes.extend_from_slice(&encode_role(Some(entry.role)));
            bytes.extend_from_slice(entry.verifier.as_bytes());
        }
        bytes
    }

    #[test]
    fn deployment_material_shares_one_nonce_and_roster() {
        let roles = [
            CellularRole::Cell(0),
            CellularRole::Cell(1),
            CellularRole::Cell(2),
        ];
        let material = mint_deployment_material(&roles).expect("mint deployment material");
        let controller = decode_material(&material.controller, None).expect("controller bundle");
        let expected_roster = roster_bytes(&controller.roster);

        assert!(
            controller
                .roster
                .iter()
                .map(|entry| entry.role)
                .eq(roles.iter().copied()),
            "roster must keep the caller's role order"
        );
        assert_eq!(material.roles.len(), roles.len());
        for role in roles {
            let bundle = material.roles.get(&role).expect("role bundle");
            let decoded = decode_material(bundle, Some(role)).expect("role bundle decodes");
            assert_eq!(decoded.run_nonce, controller.run_nonce);
            assert_eq!(roster_bytes(&decoded.roster), expected_roster);
        }
    }

    #[test]
    fn deployment_controller_bundle_holds_the_roster_authority() {
        let roles = [CellularRole::Cell(0), CellularRole::Cell(1)];
        let material = mint_deployment_material(&roles).expect("mint deployment material");
        let controller = decode_material(&material.controller, None).expect("controller bundle");
        let cell = decode_material(
            material
                .roles
                .get(&CellularRole::Cell(0))
                .expect("role bundle"),
            Some(CellularRole::Cell(0)),
        )
        .expect("role bundle decodes");

        assert_eq!(
            controller.signing_key.verifying_key(),
            controller.controller_verifier
        );
        assert_eq!(
            controller.signing_key.verifying_key(),
            cell.controller_verifier,
            "every role bundle must name the controller bundle's own key"
        );
        assert!(
            decode_material(&material.controller, Some(CellularRole::Cell(0))).is_err(),
            "the controller bundle carries no role"
        );
    }

    #[test]
    fn deployment_role_bundle_matches_its_roster_entry() {
        let roles = [
            CellularRole::Cell(0),
            CellularRole::Aggregator { tier: 1, id: 0 },
        ];
        let material = mint_deployment_material(&roles).expect("mint deployment material");
        for role in roles {
            let bundle = material.roles.get(&role).expect("role bundle");
            let decoded = decode_material(bundle, Some(role)).expect("role bundle decodes");
            let entry = decoded
                .roster
                .iter()
                .find(|entry| entry.role == role)
                .expect("roster entry");
            assert_eq!(entry.verifier, decoded.signing_key.verifying_key());
            for other in roles.iter().copied().filter(|other| *other != role) {
                assert!(
                    decode_material(bundle, Some(other)).is_err(),
                    "a role bundle must not authorize {other:?}"
                );
            }
        }
    }

    #[test]
    fn deployment_material_rejects_empty_and_duplicate_roles() {
        assert!(mint_deployment_material(&[]).is_err());
        assert!(
            mint_deployment_material(&[CellularRole::Cell(0), CellularRole::Cell(0)]).is_err(),
            "duplicate roles must fail before any key is minted"
        );
    }

    fn prepared_one_cell() -> PreparedControllerSecurity {
        prepare_controller_security(
            ControllerSecuritySource::LocalMint,
            &[CellularRole::Cell(0)],
        )
        .expect("prepare")
    }

    #[test]
    fn local_bootstrap_material_is_taken_only_once_per_role() {
        let mut prepared = prepared_one_cell();
        let roles = prepared.local_roles.as_mut().expect("local roles");
        assert!(roles.take(CellularRole::Cell(0)).is_ok());
        assert!(roles.take(CellularRole::Cell(0)).is_err());
    }

    #[test]
    fn binary_role_material_rejects_trailing_or_wrong_role_without_secret_diagnostics() {
        use base64::Engine as _;

        let mut prepared = prepared_one_cell();
        let mut bytes = prepared
            .local_roles
            .as_mut()
            .expect("local roles")
            .take(CellularRole::Cell(0))
            .expect("role bytes");
        let secret = base64::engine::general_purpose::STANDARD.encode(&bytes[50..82]);
        bytes.push(0);
        let trailing = decode_material(&bytes, Some(CellularRole::Cell(0)))
            .err()
            .expect("trailing bytes must fail");
        assert!(!format!("{trailing:#}").contains(&secret));

        bytes.pop();
        let wrong_role = decode_material(&bytes, Some(CellularRole::Cell(1)))
            .err()
            .expect("wrong role must fail");
        assert!(!format!("{wrong_role:#}").contains(&secret));
    }

    #[test]
    fn binary_role_material_rejects_truncated_duplicate_and_run_mismatch() {
        let mut prepared = prepare_controller_security(
            ControllerSecuritySource::LocalMint,
            &[CellularRole::Cell(0), CellularRole::Cell(1)],
        )
        .expect("prepare");
        let bytes = prepared
            .local_roles
            .as_mut()
            .expect("local roles")
            .take(CellularRole::Cell(0))
            .expect("role bytes");
        assert!(decode_material(&bytes[..bytes.len() - 1], Some(CellularRole::Cell(0))).is_err());

        let mut duplicate = bytes.clone();
        let second_entry = FIXED_MATERIAL_BYTES + ROSTER_ENTRY_BYTES;
        let first_entry =
            bytes[FIXED_MATERIAL_BYTES..FIXED_MATERIAL_BYTES + ROSTER_ENTRY_BYTES].to_vec();
        duplicate[second_entry..second_entry + ROSTER_ENTRY_BYTES].copy_from_slice(&first_entry);
        assert!(decode_material(&duplicate, Some(CellularRole::Cell(0))).is_err());

        let mut wrong_run = bytes.clone();
        wrong_run[50..82].copy_from_slice(&SigningKey::from_bytes(&[0x44; 32]).to_bytes());
        let error = decode_material(&wrong_run, Some(CellularRole::Cell(0)))
            .err()
            .expect("wrong run must fail");
        assert!(format!("{error:#}").contains("run mismatch"));
    }

    #[test]
    fn binary_role_material_rejects_invalid_header_and_bounds() {
        let mut prepared = prepared_one_cell();
        let bytes = prepared
            .local_roles
            .as_mut()
            .expect("local roles")
            .take(CellularRole::Cell(0))
            .expect("role bytes");

        let mut invalid_magic = bytes.clone();
        invalid_magic[0] ^= 0xff;
        assert!(decode_material(&invalid_magic, Some(CellularRole::Cell(0))).is_err());

        let mut invalid_version = bytes.clone();
        invalid_version[8] = MATERIAL_VERSION + 1;
        assert!(decode_material(&invalid_version, Some(CellularRole::Cell(0))).is_err());

        let mut oversized_roster = bytes.clone();
        oversized_roster[114..118].copy_from_slice(&((MAX_ROLE_COUNT as u32) + 1).to_le_bytes());
        assert!(decode_material(&oversized_roster, Some(CellularRole::Cell(0))).is_err());

        let oversized_input = vec![0_u8; MAX_MATERIAL_BYTES + 1];
        assert!(decode_material(&oversized_input, Some(CellularRole::Cell(0))).is_err());

        let mut empty_roster = bytes[..FIXED_MATERIAL_BYTES].to_vec();
        empty_roster[114..118].copy_from_slice(&0_u32.to_le_bytes());
        assert!(decode_material(&empty_roster, Some(CellularRole::Cell(0))).is_err());
    }

    #[test]
    fn process_security_state_is_installed_once_and_reused_by_fresh_channels() {
        use std::process::Command;

        const CHILD_ENV: &str = "AIPERF_TEST_PROCESS_SECURITY_CHILD";
        const TEST_NAME: &str = "engine::cellular_bootstrap::tests::process_security_state_is_installed_once_and_reused_by_fresh_channels";
        if std::env::var_os(CHILD_ENV).is_none() {
            let status = Command::new(std::env::current_exe().expect("current test executable"))
                .args(["--exact", TEST_NAME, "--nocapture"])
                .env(CHILD_ENV, "1")
                .status()
                .expect("run isolated process-security test");
            assert!(status.success(), "isolated process-security test failed");
            return;
        }

        let mut prepared = prepared_one_cell();
        let bytes = prepared
            .local_roles
            .as_mut()
            .expect("local roles")
            .take(CellularRole::Cell(0))
            .expect("role bytes");
        let material = decode_material(&bytes, Some(CellularRole::Cell(0))).expect("decode");
        let context = Arc::new(
            CellSecurityContext::worker(
                material.run_nonce,
                CellularRole::Cell(0),
                material.signing_key,
                material.controller_verifier,
            )
            .expect("context"),
        );

        install_process_cell_security(Arc::clone(&context)).expect("first install");
        let fetch = process_cell_security().expect("fetch context");
        let phase = process_cell_security().expect("phase context");
        let ship = process_cell_security().expect("ship context");
        assert!(Arc::ptr_eq(&context, fetch));
        assert!(Arc::ptr_eq(fetch, phase));
        assert!(Arc::ptr_eq(phase, ship));
        assert!(install_process_cell_security(context).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_process_acquisition_reads_the_role_source_once() {
        use std::os::unix::fs::PermissionsExt;
        use std::process::Command;
        use std::sync::Barrier;

        const CHILD_ENV: &str = "AIPERF_TEST_CONCURRENT_SECURITY_CHILD";
        const TEST_NAME: &str = "engine::cellular_bootstrap::tests::concurrent_process_acquisition_reads_the_role_source_once";
        if std::env::var_os(CHILD_ENV).is_some() {
            let mut prepared = prepared_one_cell();
            let bytes = prepared
                .local_roles
                .as_mut()
                .expect("local roles")
                .take(CellularRole::Cell(0))
                .expect("role bytes");
            let directory = tempfile::tempdir().expect("temporary directory");
            let path = directory.path().join("cell-0.bin");
            std::fs::write(&path, bytes).expect("write material");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .expect("permissions");
            // SAFETY: this exact-test subprocess has not started worker threads yet.
            unsafe {
                std::env::remove_var(CELL_SECURITY_FD_ENV);
                std::env::set_var("AIPERF_TEST_DELAY_ROLE_SOURCE", "1");
            }
            // A failed acquisition releases the claim so a corrected source can retry.
            unsafe { std::env::set_var(ROLE_BOOTSTRAP_FILE_ENV, directory.path()) };
            assert!(acquire_process_cell_security(CellularRole::Cell(0)).is_err());
            unsafe { std::env::set_var(ROLE_BOOTSTRAP_FILE_ENV, &path) };
            TEST_ROLE_SOURCE_READS.store(0, Ordering::Relaxed);
            let barrier = Arc::new(Barrier::new(3));
            let mut handles = Vec::new();
            for _ in 0..2 {
                let barrier = Arc::clone(&barrier);
                handles.push(std::thread::spawn(move || {
                    barrier.wait();
                    acquire_process_cell_security(CellularRole::Cell(0))
                }));
            }
            barrier.wait();
            let results = handles
                .into_iter()
                .map(|handle| handle.join().expect("acquisition thread"))
                .collect::<Vec<_>>();
            assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
            assert_eq!(
                TEST_ROLE_SOURCE_READS.load(Ordering::Relaxed),
                1,
                "a concurrent loser must fail before rereading role material"
            );
            return;
        }

        let status = Command::new(std::env::current_exe().expect("current test executable"))
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_ENV, "1")
            .status()
            .expect("run isolated acquisition test");
        assert!(status.success(), "isolated acquisition test failed");
    }

    #[cfg(unix)]
    #[test]
    fn inherited_role_source_refuses_closed_fixed_descriptor() {
        use std::process::Command;

        const CHILD_ENV: &str = "AIPERF_TEST_CLOSED_SECURITY_FD_CHILD";
        const TEST_NAME: &str = "engine::cellular_bootstrap::tests::inherited_role_source_refuses_closed_fixed_descriptor";
        if std::env::var_os(CHILD_ENV).is_some() {
            // SAFETY: closing the fixed descriptor immediately before acquisition
            // guarantees the loader and test harness cannot reopen it in between.
            unsafe { libc::close(CELL_SECURITY_FD) };
            let error = match role_source_from_environment() {
                Ok(_) => panic!("closed inherited descriptor must be refused"),
                Err(error) => error,
            };
            assert!(format!("{error:#}").contains("unavailable"));
            return;
        }

        let mut command = Command::new(std::env::current_exe().expect("current test executable"));
        command
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_ENV, "1")
            .env(CELL_SECURITY_FD_ENV, "3")
            .env_remove(ROLE_BOOTSTRAP_FILE_ENV);
        let status = command.status().expect("run isolated descriptor test");
        assert!(status.success(), "isolated descriptor test failed");
    }

    #[cfg(unix)]
    #[test]
    fn inherited_fd_role_source_is_consumed_into_one_opaque_context() {
        use std::io::Write;
        use std::os::unix::net::UnixStream;

        let mut prepared = prepared_one_cell();
        let bytes = prepared
            .local_roles
            .as_mut()
            .expect("local roles")
            .take(CellularRole::Cell(0))
            .expect("role bytes");
        let (read, mut write) = UnixStream::pair().expect("security pipe");
        write.write_all(&bytes).expect("write role material");
        drop(write);

        let context = acquire_role_security(
            RoleSecuritySource::InheritedFd(read.into()),
            CellularRole::Cell(0),
        )
        .expect("acquire role");
        assert_eq!(context.role(), Some(CellularRole::Cell(0)));
    }

    #[cfg(unix)]
    #[test]
    fn deployment_role_source_reads_one_private_binary_file() {
        use std::os::unix::fs::PermissionsExt;

        let mut prepared = prepared_one_cell();
        let bytes = prepared
            .local_roles
            .as_mut()
            .expect("local roles")
            .take(CellularRole::Cell(0))
            .expect("role bytes");
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("cell-0.bin");
        std::fs::write(&path, bytes).expect("write material");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("permissions");

        let context = acquire_role_security(
            RoleSecuritySource::DeploymentFile(path),
            CellularRole::Cell(0),
        )
        .expect("acquire role");
        assert_eq!(context.role(), Some(CellularRole::Cell(0)));
    }

    #[cfg(unix)]
    #[test]
    fn deployment_controller_source_owns_only_controller_key_and_public_roster() {
        use std::os::unix::fs::PermissionsExt;

        let run_nonce = [0x31; 32];
        let controller = SigningKey::from_bytes(&[0x41; 32]);
        let worker = SigningKey::from_bytes(&[0x51; 32]);
        let roster = [RoleVerifyingKey {
            role: CellularRole::Cell(0),
            verifier: worker.verifying_key(),
        }];
        let bytes = encode_material(
            None,
            run_nonce,
            &controller,
            controller.verifying_key(),
            &roster,
        )
        .expect("encode controller material");
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("controller.bin");
        std::fs::write(&path, bytes).expect("write material");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("permissions");

        let prepared = prepare_controller_security(
            ControllerSecuritySource::DeploymentFile(path),
            &[CellularRole::Cell(0)],
        )
        .expect("prepare controller");
        assert!(prepared.context.role().is_none());
        assert!(prepared.local_roles.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn deployment_file_rejects_public_permissions_and_symlink() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("role.bin");
        std::fs::write(&path, b"material").expect("write");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("permissions");
        assert!(read_private_deployment_file(&path, "deployment role").is_err());

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("permissions");
        let link = directory.path().join("role-link.bin");
        symlink(&path, &link).expect("symlink");
        assert!(read_private_deployment_file(&link, "deployment role").is_err());

        assert!(read_private_deployment_file(directory.path(), "deployment role").is_err());

        let oversized = directory.path().join("oversized.bin");
        std::fs::write(&oversized, vec![0_u8; MAX_MATERIAL_BYTES + 1]).expect("write oversized");
        std::fs::set_permissions(&oversized, std::fs::Permissions::from_mode(0o600))
            .expect("permissions");
        assert!(read_private_deployment_file(&oversized, "deployment role").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn deployment_file_requires_exact_private_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("role.bin");
        std::fs::write(&path, b"material").expect("write");
        for mode in [0o400, 0o700] {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode))
                .expect("permissions");
            assert!(
                read_private_deployment_file(&path, "deployment role").is_err(),
                "mode {mode:o} must be refused"
            );
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .expect("permissions");
        assert!(read_private_deployment_file(&path, "deployment role").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn deployment_file_rejects_fifo_without_blocking() {
        use std::ffi::CString;
        use std::sync::mpsc::RecvTimeoutError;
        use std::time::Duration;

        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("role.fifo");
        let path_bytes = CString::new(path.as_os_str().as_encoded_bytes()).expect("fifo path");
        // SAFETY: the path is a live, NUL-free temporary path and mode is valid.
        assert_eq!(unsafe { libc::mkfifo(path_bytes.as_ptr(), 0o600) }, 0);

        let read_path = path.clone();
        let (send, receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let _ = send.send(read_private_deployment_file(&read_path, "deployment role"));
        });
        let result = match receive.recv_timeout(Duration::from_millis(250)) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                let _writer = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .expect("unblock legacy FIFO reader");
                let _ = receive.recv_timeout(Duration::from_secs(1));
                reader.join().expect("reader thread");
                panic!("deployment FIFO open blocked before rejecting its file type");
            }
            Err(RecvTimeoutError::Disconnected) => panic!("reader disconnected"),
        };
        reader.join().expect("reader thread");
        assert!(result.is_err());
    }
}
