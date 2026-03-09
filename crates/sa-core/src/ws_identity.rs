//! Mutual WS identity helpers.
//!
//! This module implements the explicit frontend-first handshake requested by the
//! user:
//! - the frontend sends a proof first,
//! - the backend verifies it,
//! - the backend then returns its own proof,
//! - the frontend verifies the backend before using the connection.
//!
//! The proof material combines:
//! - a 5-second UTC time bucket,
//! - a local machine fingerprint derived from hostname + MAC,
//! - a direction-specific label (`sa-cli` vs `sa`),
//! - protocol metadata,
//! - and nonces to bind one handshake to one exact connection.

use crate::ws_protocol::{ClientHello, HelloReject, ServerHello};
use anyhow::{Context as _, bail};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Stable protocol identifier carried in every handshake packet.
pub const WS_PROTOCOL_ID: &str = "sa-ws/v1";

/// Hash algorithm label advertised in handshake packets.
pub const WS_HASH_ALGO: &str = "sha256";

/// Width of one time bucket in seconds.
pub const WS_TIME_STEP_SECS: u64 = 5;

/// Accepted clock skew in time buckets.
pub const WS_ALLOWED_SKEW_BUCKETS: u64 = 1;

/// Mandatory handshake timeout in seconds.
pub const WS_HANDSHAKE_TIMEOUT_SECS: u64 = 5;

/// Expected frontend name.
pub const EXPECTED_CLIENT_NAME: &str = "sa-cli";

/// Expected backend name.
pub const EXPECTED_SERVER_NAME: &str = "sa";

/// Versioned label for the machine fingerprint material.
const MACHINE_FINGERPRINT_LABEL: &str = "sa-machine-fingerprint/v1";

/// Versioned label for the client proof.
const CLIENT_PROOF_LABEL: &str = "sa-cli-proof/v1";

/// Versioned label for the server proof.
const SERVER_PROOF_LABEL: &str = "sa-server-proof/v1";

/// Local identity material reused across handshakes.
#[derive(Debug, Clone)]
pub struct LocalIdentity {
    /// Full machine fingerprint hash used in proof generation.
    pub machine_fingerprint: String,
    /// Short hint exposed on the wire for easier debugging.
    pub machine_hint: String,
}

/// Build the local machine fingerprint.
pub fn load_local_identity() -> anyhow::Result<LocalIdentity> {
    let hostname = hostname::get()
        .context("Failed to read local hostname for WS identity")?
        .to_string_lossy()
        .trim()
        .to_lowercase();

    let mac = mac_address::get_mac_address()
        .context("Failed to read local MAC address for WS identity")?
        .map(|addr| {
            addr.bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join(":")
        })
        .unwrap_or_else(|| "no-mac".to_string());

    let material = format!("{MACHINE_FINGERPRINT_LABEL}|host={hostname}|mac={mac}");
    let machine_fingerprint = sha256_hex(&material);
    let machine_hint = machine_fingerprint.chars().take(12).collect::<String>();

    Ok(LocalIdentity {
        machine_fingerprint,
        machine_hint,
    })
}

/// Return the current UTC time bucket.
pub fn current_time_bucket() -> anyhow::Result<i64> {
    current_time_bucket_at(SystemTime::now())
}

/// Return the UTC time bucket for an explicit timestamp.
pub fn current_time_bucket_at(now: SystemTime) -> anyhow::Result<i64> {
    let unix_secs = now
        .duration_since(UNIX_EPOCH)
        .context("System clock is before UNIX_EPOCH")?
        .as_secs();
    Ok((unix_secs / WS_TIME_STEP_SECS) as i64)
}

/// Build the mandatory client-first hello packet.
pub fn build_client_hello(
    local: &LocalIdentity,
    client_version: &str,
) -> anyhow::Result<ClientHello> {
    let time_bucket = current_time_bucket()?;
    let client_nonce = uuid::Uuid::new_v4().to_string();
    let proof = build_client_proof(
        &local.machine_fingerprint,
        client_version,
        time_bucket,
        &client_nonce,
    );

    Ok(ClientHello {
        protocol: WS_PROTOCOL_ID.to_string(),
        hash_algo: WS_HASH_ALGO.to_string(),
        time_step_secs: WS_TIME_STEP_SECS,
        allowed_skew_buckets: WS_ALLOWED_SKEW_BUCKETS,
        client_name: EXPECTED_CLIENT_NAME.to_string(),
        client_version: client_version.to_string(),
        time_bucket,
        machine_hint: local.machine_hint.clone(),
        client_nonce,
        proof,
    })
}

/// Verify the client's mandatory hello packet.
pub fn verify_client_hello(
    hello: &ClientHello,
    local: &LocalIdentity,
    now: SystemTime,
) -> anyhow::Result<()> {
    verify_common_fields(
        &hello.protocol,
        &hello.hash_algo,
        hello.time_step_secs,
        hello.allowed_skew_buckets,
    )?;

    if hello.client_name != EXPECTED_CLIENT_NAME {
        bail!(
            "Unexpected client name: expected `{}`, got `{}`",
            EXPECTED_CLIENT_NAME,
            hello.client_name
        );
    }

    if hello.machine_hint != local.machine_hint {
        bail!(
            "Machine fingerprint hint mismatch: expected `{}`, got `{}`",
            local.machine_hint,
            hello.machine_hint
        );
    }

    let now_bucket = current_time_bucket_at(now)?;
    if !is_time_bucket_acceptable(hello.time_bucket, now_bucket) {
        bail!(
            "Client hello time bucket {} is outside the accepted window around {}",
            hello.time_bucket,
            now_bucket
        );
    }

    let expected = build_client_proof(
        &local.machine_fingerprint,
        &hello.client_version,
        hello.time_bucket,
        &hello.client_nonce,
    );
    if hello.proof != expected {
        bail!("Client hello proof mismatch")
    }

    Ok(())
}

/// Build the backend proof returned after the client proof was verified.
pub fn build_server_hello(
    local: &LocalIdentity,
    server_version: &str,
    client_nonce: &str,
) -> anyhow::Result<ServerHello> {
    let time_bucket = current_time_bucket()?;
    let server_nonce = uuid::Uuid::new_v4().to_string();
    let proof = build_server_proof(
        &local.machine_fingerprint,
        server_version,
        time_bucket,
        client_nonce,
        &server_nonce,
    );

    Ok(ServerHello {
        protocol: WS_PROTOCOL_ID.to_string(),
        hash_algo: WS_HASH_ALGO.to_string(),
        time_step_secs: WS_TIME_STEP_SECS,
        allowed_skew_buckets: WS_ALLOWED_SKEW_BUCKETS,
        server_name: EXPECTED_SERVER_NAME.to_string(),
        server_version: server_version.to_string(),
        time_bucket,
        machine_hint: local.machine_hint.clone(),
        client_nonce: client_nonce.to_string(),
        server_nonce,
        proof,
    })
}

/// Verify the backend proof on the client side.
pub fn verify_server_hello(
    hello: &ServerHello,
    local: &LocalIdentity,
    expected_client_nonce: &str,
    now: SystemTime,
) -> anyhow::Result<()> {
    verify_common_fields(
        &hello.protocol,
        &hello.hash_algo,
        hello.time_step_secs,
        hello.allowed_skew_buckets,
    )?;

    if hello.server_name != EXPECTED_SERVER_NAME {
        bail!(
            "Unexpected server name: expected `{}`, got `{}`",
            EXPECTED_SERVER_NAME,
            hello.server_name
        );
    }

    if hello.machine_hint != local.machine_hint {
        bail!(
            "Machine fingerprint hint mismatch: expected `{}`, got `{}`",
            local.machine_hint,
            hello.machine_hint
        );
    }

    if hello.client_nonce != expected_client_nonce {
        bail!(
            "Server hello echoed unexpected client nonce: expected `{}`, got `{}`",
            expected_client_nonce,
            hello.client_nonce
        );
    }

    let now_bucket = current_time_bucket_at(now)?;
    if !is_time_bucket_acceptable(hello.time_bucket, now_bucket) {
        bail!(
            "Server hello time bucket {} is outside the accepted window around {}",
            hello.time_bucket,
            now_bucket
        );
    }

    let expected = build_server_proof(
        &local.machine_fingerprint,
        &hello.server_version,
        hello.time_bucket,
        &hello.client_nonce,
        &hello.server_nonce,
    );
    if hello.proof != expected {
        bail!("Server hello proof mismatch")
    }

    Ok(())
}

/// Build a machine-readable rejection payload.
pub fn build_hello_reject(reason: impl Into<String>, server_version: &str) -> HelloReject {
    HelloReject {
        protocol: WS_PROTOCOL_ID.to_string(),
        server_name: EXPECTED_SERVER_NAME.to_string(),
        server_version: server_version.to_string(),
        reason: reason.into(),
    }
}

/// Return `true` if a received bucket is still inside the accepted clock skew.
pub fn is_time_bucket_acceptable(received: i64, now_bucket: i64) -> bool {
    received.abs_diff(now_bucket) <= WS_ALLOWED_SKEW_BUCKETS
}

/// Validate the shared handshake metadata fields.
fn verify_common_fields(
    protocol: &str,
    hash_algo: &str,
    time_step_secs: u64,
    allowed_skew_buckets: u64,
) -> anyhow::Result<()> {
    if protocol != WS_PROTOCOL_ID {
        bail!(
            "Unexpected protocol id: expected `{}`, got `{}`",
            WS_PROTOCOL_ID,
            protocol
        );
    }
    if hash_algo != WS_HASH_ALGO {
        bail!(
            "Unexpected hash algorithm: expected `{}`, got `{}`",
            WS_HASH_ALGO,
            hash_algo
        );
    }
    if time_step_secs != WS_TIME_STEP_SECS {
        bail!(
            "Unexpected time step: expected `{}`, got `{}`",
            WS_TIME_STEP_SECS,
            time_step_secs
        );
    }
    if allowed_skew_buckets != WS_ALLOWED_SKEW_BUCKETS {
        bail!(
            "Unexpected allowed skew: expected `{}`, got `{}`",
            WS_ALLOWED_SKEW_BUCKETS,
            allowed_skew_buckets
        );
    }
    Ok(())
}

/// Build the client-side proof.
fn build_client_proof(
    machine_fingerprint: &str,
    client_version: &str,
    time_bucket: i64,
    client_nonce: &str,
) -> String {
    sha256_hex(&format!(
        "{CLIENT_PROOF_LABEL}|{WS_PROTOCOL_ID}|{EXPECTED_CLIENT_NAME}|{client_version}|{time_bucket}|{machine_fingerprint}|{client_nonce}"
    ))
}

/// Build the server-side proof.
fn build_server_proof(
    machine_fingerprint: &str,
    server_version: &str,
    time_bucket: i64,
    client_nonce: &str,
    server_nonce: &str,
) -> String {
    sha256_hex(&format!(
        "{SERVER_PROOF_LABEL}|{WS_PROTOCOL_ID}|{EXPECTED_SERVER_NAME}|{server_version}|{time_bucket}|{machine_fingerprint}|{client_nonce}|{server_nonce}"
    ))
}

/// Hash one UTF-8 string with SHA-256 and return lower-case hex.
fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fake_identity() -> LocalIdentity {
        LocalIdentity {
            machine_fingerprint: "abc123machinefingerprint".to_string(),
            machine_hint: "abc123machin".to_string(),
        }
    }

    #[test]
    fn client_hello_verifies_inside_time_window() {
        let identity = fake_identity();
        let now = UNIX_EPOCH + Duration::from_secs(50);
        let time_bucket = current_time_bucket_at(now).expect("bucket");
        let hello = ClientHello {
            protocol: WS_PROTOCOL_ID.to_string(),
            hash_algo: WS_HASH_ALGO.to_string(),
            time_step_secs: WS_TIME_STEP_SECS,
            allowed_skew_buckets: WS_ALLOWED_SKEW_BUCKETS,
            client_name: EXPECTED_CLIENT_NAME.to_string(),
            client_version: "0.1.0".to_string(),
            time_bucket,
            machine_hint: identity.machine_hint.clone(),
            client_nonce: "nonce-a".to_string(),
            proof: build_client_proof(
                &identity.machine_fingerprint,
                "0.1.0",
                time_bucket,
                "nonce-a",
            ),
        };

        verify_client_hello(&hello, &identity, now).expect("client hello should verify");
    }

    #[test]
    fn client_hello_rejects_wrong_bucket() {
        let identity = fake_identity();
        let now = UNIX_EPOCH + Duration::from_secs(50);
        let now_bucket = current_time_bucket_at(now).expect("bucket");
        let hello = ClientHello {
            protocol: WS_PROTOCOL_ID.to_string(),
            hash_algo: WS_HASH_ALGO.to_string(),
            time_step_secs: WS_TIME_STEP_SECS,
            allowed_skew_buckets: WS_ALLOWED_SKEW_BUCKETS,
            client_name: EXPECTED_CLIENT_NAME.to_string(),
            client_version: "0.1.0".to_string(),
            time_bucket: now_bucket + 2,
            machine_hint: identity.machine_hint.clone(),
            client_nonce: "nonce-a".to_string(),
            proof: build_client_proof(
                &identity.machine_fingerprint,
                "0.1.0",
                now_bucket + 2,
                "nonce-a",
            ),
        };

        let err = verify_client_hello(&hello, &identity, now).expect_err("bucket should fail");
        assert!(err.to_string().contains("outside the accepted window"));
    }

    #[test]
    fn server_hello_verifies_when_client_nonce_matches() {
        let identity = fake_identity();
        let now = UNIX_EPOCH + Duration::from_secs(50);
        let time_bucket = current_time_bucket_at(now).expect("bucket");
        let hello = ServerHello {
            protocol: WS_PROTOCOL_ID.to_string(),
            hash_algo: WS_HASH_ALGO.to_string(),
            time_step_secs: WS_TIME_STEP_SECS,
            allowed_skew_buckets: WS_ALLOWED_SKEW_BUCKETS,
            server_name: EXPECTED_SERVER_NAME.to_string(),
            server_version: "0.1.0".to_string(),
            time_bucket,
            machine_hint: identity.machine_hint.clone(),
            client_nonce: "nonce-a".to_string(),
            server_nonce: "nonce-b".to_string(),
            proof: build_server_proof(
                &identity.machine_fingerprint,
                "0.1.0",
                time_bucket,
                "nonce-a",
                "nonce-b",
            ),
        };

        verify_server_hello(&hello, &identity, "nonce-a", now).expect("server hello should verify");
    }
}
