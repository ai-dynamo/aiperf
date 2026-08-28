// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "streaming", feature = "streaming-crypto"))]

//! Sensitive-state envelope, key-source, and target-policy contracts.

use std::collections::BTreeMap;
use std::io::Write;

use aiperf_runtime::streaming::checkpoint::{
    CheckpointEpoch, CheckpointGeneration, CheckpointParticipantId, StreamRunIdentity,
};
use aiperf_runtime::streaming::checkpoint_backend::{
    CheckpointBackendPlacement, CheckpointRetention, StreamingCheckpointBackendDescriptor,
};
use aiperf_runtime::streaming::failure::SensitiveStateFailureCode;
use aiperf_runtime::streaming::identity::{ContentDigest, LogicalReplayRunId};
use aiperf_runtime::streaming::policy::{
    SensitiveStateKeyId, SessionTargetPolicy, fold_target_reply, validate_target_policy,
};
use aiperf_runtime::streaming::sensitive_state::{
    MAX_SENSITIVE_MATERIAL_BYTES, NativeSensitiveStateKeyResolver, RefusingSensitiveStateKeyResolver,
    SensitiveStateContext, StreamingSensitiveStateKeyResolver, decrypt_sensitive, encrypt_sensitive,
    read_bounded, read_private_key_file, acquire_process_sensitive_state,
};
use zeroize::Zeroizing;

const KEY_ID: &str = "primary";

fn key_id() -> SensitiveStateKeyId {
    SensitiveStateKeyId::new(KEY_ID)
}

fn resolver_with(entries: &[(&str, u8)]) -> NativeSensitiveStateKeyResolver {
    let mut keys = BTreeMap::new();
    for (selector, fill) in entries {
        keys.insert(
            SensitiveStateKeyId::new(*selector),
            Zeroizing::new([*fill; 32]),
        );
    }
    NativeSensitiveStateKeyResolver::new(keys)
}

fn digest(fill: u8) -> ContentDigest {
    ContentDigest::from_bytes([fill; 32])
}

fn context() -> SensitiveStateContext {
    SensitiveStateContext {
        run: StreamRunIdentity::new(LogicalReplayRunId::from_bytes([7; 32])),
        generation: CheckpointGeneration::new(CheckpointEpoch::new(3), digest(9)),
        participant: CheckpointParticipantId::new("aiperf.stream.session"),
        schema_id: "aiperf.stream.session.transcript".to_string(),
        schema_version: 2,
        policy_digest: digest(11),
    }
}

fn backend(protects_sensitive_state: bool) -> StreamingCheckpointBackendDescriptor {
    StreamingCheckpointBackendDescriptor {
        id: "test",
        description: "test backend",
        is_durable: true,
        has_leased_readers: true,
        has_atomic_generations: true,
        has_result_segments: true,
        protects_sensitive_state,
        retention: CheckpointRetention::GenerationReachability,
        placement: CheckpointBackendPlacement::ControllerLocal,
        supports_virtual_clock: true,
    }
}

#[test]
fn target_state_requires_authenticated_external_key_and_bound_aad() {
    let resolver = resolver_with(&[(KEY_ID, 0x5a)]);
    let context = context();
    let plaintext = b"the model said something a customer typed".to_vec();

    let envelope = encrypt_sensitive(&resolver, &key_id(), &context, &plaintext)
        .expect("seal under resolved key");
    let opened = decrypt_sensitive(&resolver, &context, &envelope).expect("round trip");
    assert_eq!(opened.as_slice(), plaintext.as_slice());

    // The same envelope under a different generation is not openable.
    let mut other = context.clone();
    other.generation = CheckpointGeneration::new(CheckpointEpoch::new(5), digest(9));
    assert_eq!(
        decrypt_sensitive(&resolver, &other, &envelope)
            .expect_err("generation is bound")
            .failure(),
        SensitiveStateFailureCode::Authentication
    );

    let rendered = format!("{envelope:?}");
    assert!(!rendered.contains("customer"), "envelope Debug leaks content");
}

#[test]
fn wrong_key_and_single_bit_tamper_both_fail_authentication() {
    let resolver = resolver_with(&[(KEY_ID, 0x5a), ("other", 0x17)]);
    let context = context();
    let envelope =
        encrypt_sensitive(&resolver, &key_id(), &context, b"protected").expect("seal");

    let mut wrong_key = envelope.clone();
    wrong_key.key_id = SensitiveStateKeyId::new("other");
    assert_eq!(
        decrypt_sensitive(&resolver, &context, &wrong_key)
            .expect_err("wrong key")
            .failure(),
        SensitiveStateFailureCode::Authentication
    );

    let mut body_bit = envelope.clone();
    body_bit.ciphertext[0] ^= 0x01;
    assert_eq!(
        decrypt_sensitive(&resolver, &context, &body_bit)
            .expect_err("ciphertext tamper")
            .failure(),
        SensitiveStateFailureCode::Authentication
    );

    let mut tag_bit = envelope.clone();
    let last = tag_bit.ciphertext.len() - 1;
    tag_bit.ciphertext[last] ^= 0x01;
    assert_eq!(
        decrypt_sensitive(&resolver, &context, &tag_bit)
            .expect_err("tag tamper")
            .failure(),
        SensitiveStateFailureCode::Authentication
    );
}

#[test]
fn nonces_are_unique_across_repeated_seals_of_identical_plaintext() {
    let resolver = resolver_with(&[(KEY_ID, 0x5a)]);
    let context = context();
    let mut nonces = std::collections::BTreeSet::new();
    let mut ciphertexts = std::collections::BTreeSet::new();
    for _ in 0..1024 {
        let envelope =
            encrypt_sensitive(&resolver, &key_id(), &context, b"identical").expect("seal");
        assert!(nonces.insert(envelope.nonce));
        assert!(ciphertexts.insert(envelope.ciphertext));
    }
}

#[test]
fn every_aad_field_is_load_bearing() {
    let resolver = resolver_with(&[(KEY_ID, 0x5a), ("alternate", 0x33)]);
    let base = context();
    let envelope = encrypt_sensitive(&resolver, &key_id(), &base, b"bound state").expect("seal");

    let mutations: Vec<(&str, SensitiveStateContext)> = vec![
        ("run", {
            let mut mutated = base.clone();
            mutated.run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([8; 32]));
            mutated
        }),
        ("generation epoch", {
            let mut mutated = base.clone();
            mutated.generation = CheckpointGeneration::new(CheckpointEpoch::new(4), digest(9));
            mutated
        }),
        ("generation digest", {
            let mut mutated = base.clone();
            mutated.generation = CheckpointGeneration::new(CheckpointEpoch::new(3), digest(10));
            mutated
        }),
        ("participant", {
            let mut mutated = base.clone();
            mutated.participant = CheckpointParticipantId::new("aiperf.stream.other");
            mutated
        }),
        ("schema id", {
            let mut mutated = base.clone();
            mutated.schema_id = "aiperf.stream.session.other".to_string();
            mutated
        }),
        ("schema version", {
            let mut mutated = base.clone();
            mutated.schema_version = 3;
            mutated
        }),
        ("policy digest", {
            let mut mutated = base.clone();
            mutated.policy_digest = digest(12);
            mutated
        }),
    ];
    for (field, mutated) in mutations {
        let outcome = decrypt_sensitive(&resolver, &mutated, &envelope);
        let error = outcome
            .map(|_| ())
            .expect_err(&format!("{field} binding is decorative"));
        assert_eq!(error.failure(), SensitiveStateFailureCode::Authentication);
    }
}

#[test]
fn key_id_is_bound_by_associated_data() {
    let resolver = resolver_with(&[(KEY_ID, 0x5a), ("alternate", 0x5a)]);
    let base = context();
    let envelope = encrypt_sensitive(&resolver, &key_id(), &base, b"bound state").expect("seal");

    // Two selectors resolve to identical bytes, so only the key-id binding in
    // the associated data can distinguish them.
    let mut relabeled = envelope.clone();
    relabeled.key_id = SensitiveStateKeyId::new("alternate");
    assert_eq!(
        decrypt_sensitive(&resolver, &base, &relabeled)
            .expect_err("key id is bound")
            .failure(),
        SensitiveStateFailureCode::Authentication
    );
}

#[cfg(unix)]
#[test]
fn key_file_must_be_regular_exact_0600_and_not_a_symlink() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temp dir");
    let material = format!("{KEY_ID} {}\n", "5a".repeat(32));

    let private = root.path().join("private.key");
    std::fs::write(&private, &material).expect("write key");
    std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let read = read_private_key_file(&private).expect("exact 0600 regular file resolves");
    assert_eq!(read.as_slice(), material.as_bytes());

    let group_readable = root.path().join("group.key");
    std::fs::write(&group_readable, &material).expect("write key");
    std::fs::set_permissions(&group_readable, std::fs::Permissions::from_mode(0o640))
        .expect("chmod");
    assert_eq!(
        read_private_key_file(&group_readable)
            .expect_err("0640 refused")
            .failure(),
        SensitiveStateFailureCode::KeyNotPrivate
    );

    let link = root.path().join("link.key");
    std::os::unix::fs::symlink(&private, &link).expect("symlink");
    assert_eq!(
        read_private_key_file(&link).expect_err("symlink refused").failure(),
        SensitiveStateFailureCode::KeyNotPrivate
    );

    assert_eq!(
        read_private_key_file(root.path())
            .expect_err("directory refused")
            .failure(),
        SensitiveStateFailureCode::KeyNotPrivate
    );

    let fifo = root.path().join("fifo.key");
    let c_path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).expect("path");
    // SAFETY: `c_path` is a valid NUL-terminated path inside the temp dir.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    assert_eq!(
        read_private_key_file(&fifo).expect_err("fifo refused").failure(),
        SensitiveStateFailureCode::KeyNotPrivate
    );
}

#[cfg(unix)]
#[test]
fn oversized_key_source_is_refused_without_full_read() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temp dir");
    let oversized = root.path().join("big.key");
    let mut file = std::fs::File::create(&oversized).expect("create");
    file.write_all(&vec![b'a'; MAX_SENSITIVE_MATERIAL_BYTES + 1])
        .expect("write");
    drop(file);
    std::fs::set_permissions(&oversized, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    assert_eq!(
        read_private_key_file(&oversized)
            .expect_err("oversized file refused by metadata")
            .failure(),
        SensitiveStateFailureCode::KeyMalformed
    );

    // The stream path has no metadata to consult, so it must refuse on the
    // read-`MAX + 1` bound instead.
    let stream = std::io::Cursor::new(vec![b'a'; MAX_SENSITIVE_MATERIAL_BYTES + 1]);
    assert_eq!(
        read_bounded(stream)
            .expect_err("oversized stream refused by bound")
            .failure(),
        SensitiveStateFailureCode::KeyMalformed
    );
}

#[test]
fn parsed_material_resolves_only_declared_selectors() {
    let mut material = Zeroizing::new(
        format!("{KEY_ID} {}\nsecondary {}\n", "5a".repeat(32), "17".repeat(32)).into_bytes(),
    );
    let resolver = NativeSensitiveStateKeyResolver::parse(&mut material).expect("parse");
    // The parse buffer is wiped, so the key bytes do not survive in it.
    assert!(material.iter().all(|byte| *byte == 0));

    assert_eq!(
        resolver.resolve(&key_id()).expect("declared").key.as_slice(),
        &[0x5a; 32]
    );
    assert_eq!(
        resolver
            .resolve(&SensitiveStateKeyId::new("absent"))
            .expect_err("undeclared")
            .failure(),
        SensitiveStateFailureCode::KeyUnavailable
    );
}

#[test]
fn default_process_resolver_refuses_every_selector() {
    let resolver = RefusingSensitiveStateKeyResolver;
    assert_eq!(
        resolver.resolve(&key_id()).expect_err("refusing").failure(),
        SensitiveStateFailureCode::KeyUnavailable
    );
    assert_eq!(
        encrypt_sensitive(&resolver, &key_id(), &context(), b"state")
            .expect_err("closed-loop state cannot be sealed without material")
            .failure(),
        SensitiveStateFailureCode::KeyUnavailable
    );
}

#[test]
fn process_material_is_claimed_once_and_a_second_claim_is_refused() {
    // The claim is process-wide, so this is the only test that touches it: a
    // second acquisition must be a typed refusal rather than a second read of a
    // one-shot pipe, which would come back empty and look like missing material.
    let first = acquire_process_sensitive_state().expect("first claim installs a resolver");
    assert_eq!(
        first
            .resolve(&key_id())
            .expect_err("no source is configured in this process")
            .failure(),
        SensitiveStateFailureCode::KeyUnavailable
    );
    assert_eq!(
        acquire_process_sensitive_state()
            .expect_err("second claim refused")
            .failure(),
        SensitiveStateFailureCode::KeyUnavailable
    );
}

#[test]
fn target_closed_loop_requires_protecting_backend_or_checkpoint_none() {
    let generation = CheckpointGeneration::new(CheckpointEpoch::new(3), digest(9));

    // Protecting backend plus a key id.
    assert!(
        validate_target_policy(
            SessionTargetPolicy::TargetClosedLoop,
            Some(&backend(true)),
            Some(&generation),
            Some(&key_id()),
        )
        .is_ok()
    );

    // No backend and no resume claim.
    assert!(
        validate_target_policy(SessionTargetPolicy::TargetClosedLoop, None, None, None).is_ok()
    );

    // A durable backend without the flag is refused.
    assert!(
        validate_target_policy(
            SessionTargetPolicy::TargetClosedLoop,
            Some(&backend(false)),
            None,
            Some(&key_id()),
        )
        .is_err()
    );

    // A protecting backend with no key id has nothing to seal with.
    assert!(
        validate_target_policy(
            SessionTargetPolicy::TargetClosedLoop,
            Some(&backend(true)),
            None,
            None,
        )
        .is_err()
    );

    // Checkpoint-`none` with a resume claim is a contradiction.
    assert!(
        validate_target_policy(
            SessionTargetPolicy::TargetClosedLoop,
            None,
            Some(&generation),
            None,
        )
        .is_err()
    );

    // `recorded_inputs` retains nothing sensitive, so every shape is admissible.
    assert!(
        validate_target_policy(
            SessionTargetPolicy::RecordedInputs,
            Some(&backend(false)),
            Some(&generation),
            None,
        )
        .is_ok()
    );
}

#[test]
fn recorded_inputs_does_not_mutate_later_requests() {
    let recorded = b"recorded target content".to_vec();
    let observed = b"what the model actually said".to_vec();

    let fold = fold_target_reply(
        SessionTargetPolicy::RecordedInputs,
        &recorded,
        observed.clone(),
    );
    assert!(fold.divergence().is_divergent, "divergence is still reported");
    assert_eq!(fold.divergence().recorded_len, recorded.len());
    assert_eq!(fold.divergence().observed_len, observed.len());
    // Nothing observed is retained, so the next request is byte-identical to
    // the recording regardless of what the endpoint replied.
    assert_eq!(fold.retained(), None);
}

#[test]
fn target_closed_loop_retains_the_actual_reply_and_survives_restart() {
    let recorded = b"recorded target content".to_vec();
    let observed = b"what the model actually said".to_vec();

    let fold = fold_target_reply(
        SessionTargetPolicy::TargetClosedLoop,
        &recorded,
        observed.clone(),
    );
    assert!(fold.divergence().is_divergent);
    let retained = fold.retained().expect("closed loop retains the reply");
    assert_eq!(retained, observed.as_slice());

    // A checkpoint/restore round trip through a protecting backend reproduces
    // the same subsequent request byte-for-byte.
    let resolver = resolver_with(&[(KEY_ID, 0x5a)]);
    let context = context();
    let envelope =
        encrypt_sensitive(&resolver, &key_id(), &context, retained).expect("seal retained state");
    let restored = decrypt_sensitive(&resolver, &context, &envelope).expect("restore");
    assert_eq!(restored.as_slice(), observed.as_slice());
}

#[test]
fn key_material_is_absent_from_process_visible_surfaces() {
    let resolver = resolver_with(&[(KEY_ID, 0x5a)]);
    let context = context();
    let key = resolver.resolve(&key_id()).expect("resolve");
    let envelope = encrypt_sensitive(&resolver, &key_id(), &context, b"secret").expect("seal");

    // 0x5a repeated is what the key bytes would render as in any Debug that
    // walked them; the id is the only key-related value that may appear.
    for rendered in [
        format!("{resolver:?}"),
        format!("{key:?}"),
        format!("{envelope:?}"),
    ] {
        assert!(rendered.contains(KEY_ID), "key id must remain visible");
        assert!(!rendered.contains("90, 90"), "key bytes leaked: {rendered}");
        assert!(!rendered.contains("secret"), "plaintext leaked: {rendered}");
    }

    let serialized = serde_json::to_string(&envelope).expect("serialize envelope");
    assert!(serialized.contains(KEY_ID));
    assert!(!serialized.contains("secret"));

    let key_id_json = serde_json::to_string(&key_id()).expect("serialize key id");
    assert_eq!(key_id_json, format!("\"{KEY_ID}\""));
}
