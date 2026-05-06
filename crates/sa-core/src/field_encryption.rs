//! Field-level encryption for sensitive data in the memory system (D9).
//!
//! Provides AES-256-GCM encryption of individual text fields so that
//! sensitive values (API keys, credentials, PII that must be retained
//! in recoverable form) are never written to disk in plaintext.
//!
//! ## Key management
//!
//! The master key is a 32-byte slice.  Callers can load it from an
//! environment variable (`SA_FIELD_KEY`, hex-encoded) or from the
//! configuration file.  If no key is configured, encryption is a no-op
//! and fields are stored in plaintext (the module still compiles, but
//! `is_active()` returns `false`).
//!
//! ## Wire format
//!
//! Encrypted values are stored as:
//!
//! ```text
//! base64( nonce[12] || ciphertext[plaintext.len() + 16] )
//! ```
//!
//! Nonce is randomly generated per encryption call (96-bit), giving a
//! unique ciphertext even for identical plaintexts.

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use anyhow::Context as _;
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for field-level encryption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEncryptionConfig {
    /// Whether field encryption is enabled.
    /// When `false`, `FieldEncryptor::encrypt()` is a no-op.
    #[serde(default)]
    pub enabled: bool,

    /// Hex-encoded AES-256 key (64 hex chars).
    /// If empty and `enabled == true`, the module will try the
    /// `SA_FIELD_KEY` environment variable.
    #[serde(default)]
    pub master_key_hex: String,
}

impl Default for FieldEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            master_key_hex: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldEncryptor
// ---------------------------------------------------------------------------

/// Holds the symmetric key and provides `encrypt()` / `decrypt()`.
pub struct FieldEncryptor {
    cipher: Option<Aes256Gcm>,
}

impl FieldEncryptor {
    /// Create an encryptor from a config.
    ///
    /// Returns an inactive encryptor (`is_active() == false`) when:
    /// - `config.enabled` is `false`, OR
    /// - the key cannot be resolved (both `master_key_hex` and `SA_FIELD_KEY` are empty/malformed).
    pub fn from_config(config: &FieldEncryptionConfig) -> anyhow::Result<Self> {
        if !config.enabled {
            return Ok(Self { cipher: None });
        }

        let key_hex = if config.master_key_hex.is_empty() {
            std::env::var("SA_FIELD_KEY").unwrap_or_default()
        } else {
            config.master_key_hex.clone()
        };

        if key_hex.is_empty() {
            tracing::warn!("field encryption enabled but no key configured; disabling");
            return Ok(Self { cipher: None });
        }

        let key_bytes = hex::decode(&key_hex)
            .with_context(|| "master_key_hex must be valid hex")?;

        if key_bytes.len() != 32 {
            anyhow::bail!(
                "master_key must be exactly 32 bytes (64 hex chars), got {} bytes",
                key_bytes.len()
            );
        }

        let key = aes_gcm::Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);

        tracing::info!("field encryption activated");
        Ok(Self {
            cipher: Some(cipher),
        })
    }

    /// Whether encryption is active.
    pub fn is_active(&self) -> bool {
        self.cipher.is_some()
    }

    /// Encrypt a plaintext string.
    ///
    /// Returns a base64-encoded string: `nonce(12) || ciphertext`.
    /// If encryption is inactive, the plaintext is returned unchanged.
    pub fn encrypt(&self, plaintext: &str) -> anyhow::Result<String> {
        let Some(ref cipher) = self.cipher else {
            return Ok(plaintext.to_string());
        };

        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| anyhow::anyhow!("AES-GCM encryption failed: {e}"))?;

        // Pack: nonce (12) + ciphertext (len)
        let mut packed = Vec::with_capacity(12 + ciphertext.len());
        packed.extend_from_slice(&nonce_bytes);
        packed.extend_from_slice(&ciphertext);

        Ok(base64::engine::general_purpose::STANDARD.encode(&packed))
    }

    /// Decrypt a base64-encoded ciphertext.
    ///
    /// Returns the original plaintext string.
    /// If encryption is inactive, the input is returned unchanged
    /// (assumed to be plaintext).
    pub fn decrypt(&self, encrypted: &str) -> anyhow::Result<String> {
        let Some(ref cipher) = self.cipher else {
            return Ok(encrypted.to_string());
        };

        let packed = base64::engine::general_purpose::STANDARD
            .decode(encrypted)
            .with_context(|| "encrypted field is not valid base64")?;

        if packed.len() < 12 + 16 {
            anyhow::bail!("encrypted field too short (need at least nonce 12 + tag 16)");
        }

        let nonce = Nonce::from_slice(&packed[..12]);
        let ciphertext = &packed[12..];

        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("AES-GCM decryption failed (wrong key?): {e}"))?;

        String::from_utf8(plaintext).with_context(|| "decrypted field is not valid UTF-8")
    }

    /// Encrypt a value, wrapping the result in a marker so decryption can
    /// be applied selectively during reads.
    ///
    /// Produces `ENC:<base64>` when active, or the plaintext when inactive.
    pub fn encrypt_field(&self, field_value: &str) -> anyhow::Result<String> {
        if !self.is_active() || field_value.is_empty() {
            return Ok(field_value.to_string());
        }
        Ok(format!("ENC:{}", self.encrypt(field_value)?))
    }

    /// Decrypt a value that may be prefixed with `ENC:`.
    ///
    /// Plaintext values without `ENC:` prefix are returned unchanged.
    pub fn decrypt_field(&self, field_value: &str) -> anyhow::Result<String> {
        if let Some(enc) = field_value.strip_prefix("ENC:") {
            self.decrypt(enc)
        } else {
            Ok(field_value.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(key_hex: &str) -> FieldEncryptionConfig {
        FieldEncryptionConfig {
            enabled: true,
            master_key_hex: key_hex.to_string(),
        }
    }

    fn sample_key() -> String {
        // Deterministic 32-byte key for tests.
        "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899".to_string()
    }

    #[test]
    fn inactive_encryptor_roundtrips_plaintext() {
        let enc = FieldEncryptor::from_config(&FieldEncryptionConfig::default())
            .expect("default config should build");
        assert!(!enc.is_active());

        let ct = enc.encrypt("hello world").unwrap();
        assert_eq!(ct, "hello world"); // no-op

        let pt = enc.decrypt("hello world").unwrap();
        assert_eq!(pt, "hello world");
    }

    #[test]
    fn active_encryptor_roundtrips() {
        let enc = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("key config should build");
        assert!(enc.is_active());

        let plain = "this is a secret";
        let ct = enc.encrypt(plain).unwrap();
        assert_ne!(ct, plain);

        let pt = enc.decrypt(&ct).unwrap();
        assert_eq!(pt, plain);
    }

    #[test]
    fn encrypt_field_marker_roundtrips() {
        let enc = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("key config should build");

        let ef = enc.encrypt_field("secret-value").unwrap();
        assert!(ef.starts_with("ENC:"));

        let pt = enc.decrypt_field(&ef).unwrap();
        assert_eq!(pt, "secret-value");
    }

    #[test]
    fn decrypt_field_no_marker_passes_through() {
        let enc = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("key config should build");

        let pt = enc.decrypt_field("plaintext without prefix").unwrap();
        assert_eq!(pt, "plaintext without prefix");
    }

    #[test]
    fn encrypt_field_empty_passes_through() {
        let enc = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("key config should build");

        let ef = enc.encrypt_field("").unwrap();
        assert_eq!(ef, "");
    }

    #[test]
    fn nondeterministic_nonce_yields_different_ciphertexts() {
        let enc = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("key config should build");

        let ct1 = enc.encrypt("same plaintext").unwrap();
        let ct2 = enc.encrypt("same plaintext").unwrap();
        assert_ne!(ct1, ct2, "same plaintext should produce different ciphertexts");
    }

    #[test]
    fn tampered_ciphertext_fails_decryption() {
        let enc = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("key config should build");

        let ct = enc.encrypt("tamper me").unwrap();
        // Flip a byte in the packed representation
        let mut packed = base64::engine::general_purpose::STANDARD
            .decode(&ct)
            .unwrap();
        packed[15] ^= 0x01;
        let tampered = base64::engine::general_purpose::STANDARD.encode(&packed);

        let result = enc.decrypt(&tampered);
        assert!(result.is_err(), "tampered ciphertext should fail decryption");
    }

    #[test]
    fn wrong_key_fails_decryption() {
        let enc1 = FieldEncryptor::from_config(&test_config(&sample_key()))
            .expect("enc1 should build");
        let other_key = "ffeeddccbbaa00112233445566778899ffeeddccbbaa00112233445566778899".to_string();
        let enc2 = FieldEncryptor::from_config(&test_config(&other_key))
            .expect("enc2 should build");

        let ct = enc1.encrypt("cross-decrypt").unwrap();
        let result = enc2.decrypt(&ct);
        assert!(result.is_err(), "decrypt with wrong key should fail");
    }

    #[test]
    fn bad_key_length_rejected() {
        let bad_config = test_config("aabb");
        let result = FieldEncryptor::from_config(&bad_config);
        assert!(result.is_err());
    }
}
