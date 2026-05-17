//! Write filter pipeline: PII detection + scope enforcement for memory writes.
//!
//! This module sits between the agent/session write path and `write_daily_memory`,
//! providing:
//! - PII detection and sanitization (delegates to `pii_detector`)
//! - Automatic scope downgrade when PII is detected
//! - Write filter result for downstream consumers

use crate::config::PrivacyConfig;
use crate::memory_scope::MemoryScope;
use crate::pii_detector::{PiiSpan, SanitizePolicy, detect_pii, sanitize_text};

/// Result of the write filter pipeline.
#[derive(Debug, Clone)]
pub struct WriteFilterResult {
    /// Whether the write should proceed (false = blocked by PII policy).
    pub allowed: bool,
    /// Recommended scope after PII inspection.
    pub scope: MemoryScope,
    /// PII detections found in the content.
    pub pii_detections: Vec<PiiSpan>,
    /// Content after sanitization (may equal original if no PII).
    pub sanitized_content: String,
    /// Human-readable reason if write was modified or blocked.
    pub reason: Option<String>,
}

/// Apply the write filter pipeline to content before memory write.
///
/// Pipeline:
/// 1. If PII detection is disabled, pass through unchanged.
/// 2. Detect PII in the content.
/// 3. If PII found:
///    - Sanitize the content according to the configured policy
///    - Force scope to at least `Session` (PII should not persist at User/Shared)
/// 4. Return the filter result with sanitized content and recommended scope.
pub fn apply_write_filter(
    content: &str,
    policy: &PrivacyConfig,
    original_scope: MemoryScope,
) -> WriteFilterResult {
    // If PII detection is disabled, pass through.
    if !policy.pii_detection {
        return WriteFilterResult {
            allowed: true,
            scope: original_scope,
            pii_detections: Vec::new(),
            sanitized_content: content.to_string(),
            reason: None,
        };
    }

    let result = detect_pii(content);

    if !result.has_pii {
        return WriteFilterResult {
            allowed: true,
            scope: original_scope,
            pii_detections: Vec::new(),
            sanitized_content: content.to_string(),
            reason: None,
        };
    }

    // PII detected — sanitize and downgrade scope.
    let sanitized = sanitize_text(content, &result.detections, policy.sanitize_policy);

    // Force scope downgrade: PII content should not persist at User/Shared scope
    // unless the sanitize policy is Hash (reversible, not plaintext).
    let safe_scope = match policy.sanitize_policy {
        SanitizePolicy::Hash => {
            // Hash preserves the structure but not the plaintext — safe at original scope
            original_scope
        }
        SanitizePolicy::Redact | SanitizePolicy::None => {
            // Redact or raw PII — downgrade to Session to prevent persistence
            if original_scope.level() > MemoryScope::Session.level() {
                MemoryScope::Session
            } else {
                original_scope
            }
        }
    };

    let scope_change = safe_scope != original_scope;
    let reason = if scope_change {
        Some(format!(
            "PII detected ({} items), scope downgraded {}→{} with {:?} policy",
            result.detections.len(),
            original_scope,
            safe_scope,
            policy.sanitize_policy,
        ))
    } else {
        Some(format!(
            "PII detected ({} items), sanitized with {:?} policy",
            result.detections.len(),
            policy.sanitize_policy,
        ))
    };

    WriteFilterResult {
        allowed: true,
        scope: safe_scope,
        pii_detections: result.detections,
        sanitized_content: sanitized,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_privacy_config() -> PrivacyConfig {
        PrivacyConfig::default()
    }

    fn privacy_disabled() -> PrivacyConfig {
        PrivacyConfig {
            pii_detection: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_no_pii_passes_through() {
        let config = default_privacy_config();
        let result = apply_write_filter("今天天气不错，适合出去走走", &config, MemoryScope::User);
        assert!(result.allowed);
        assert_eq!(result.scope, MemoryScope::User);
        assert!(result.pii_detections.is_empty());
        assert!(result.reason.is_none());
    }

    #[test]
    fn test_pii_detected_redacts_and_downgrades() {
        let config = default_privacy_config();
        let content = "我的身份证号是 110101199003077758，需要登记";
        let result = apply_write_filter(content, &config, MemoryScope::User);

        assert!(result.allowed);
        // Redact policy should downgrade to Session
        assert_eq!(result.scope, MemoryScope::Session);
        assert!(!result.pii_detections.is_empty());
        assert!(result.reason.is_some());
        // Sanitized content should not contain the raw ID card
        assert!(!result.sanitized_content.contains("110101199003077758"));
    }

    #[test]
    fn test_pii_with_hash_policy_preserves_scope() {
        let config = PrivacyConfig {
            sanitize_policy: SanitizePolicy::Hash,
            ..Default::default()
        };
        let content = "我的邮箱 test@example.com";
        let result = apply_write_filter(content, &config, MemoryScope::User);

        assert!(result.allowed);
        // Hash policy should preserve original scope
        assert_eq!(result.scope, MemoryScope::User);
        assert!(!result.pii_detections.is_empty());
    }

    #[test]
    fn test_pii_disabled_passes_through() {
        let config = privacy_disabled();
        let content = "身份证号 110101199003077758";
        let result = apply_write_filter(content, &config, MemoryScope::User);

        assert!(result.allowed);
        assert_eq!(result.scope, MemoryScope::User);
        assert!(result.pii_detections.is_empty());
        // Content should be unchanged
        assert!(result.sanitized_content.contains("110101199003077758"));
    }

    #[test]
    fn test_session_scope_not_further_downgraded() {
        let config = default_privacy_config();
        let content = "手机号 13800138000";
        let result = apply_write_filter(content, &config, MemoryScope::Session);

        assert!(result.allowed);
        assert_eq!(result.scope, MemoryScope::Session);
    }
}
