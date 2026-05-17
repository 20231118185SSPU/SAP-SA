//! Memory scope system: controls cross-scope access to memory entries.
//!
//! Scopes define the visibility boundary for memory records:
//! - `Session` — transient, only visible within the current conversation
//! - `User`    — persistent, visible across all sessions for the user
//! - `Shared`  — cross-user, requires explicit promotion
//! - `Deleted` — soft-deleted, hidden from search, recoverable

use serde::{Deserialize, Serialize};
use std::fmt;

/// Visibility scope for a memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryScope {
    /// Only visible within the current conversation session.
    /// Automatically promoted to `User` if importance > 0.7.
    Session,

    /// Visible across all sessions for the same user (default).
    #[default]
    User,

    /// Cross-user shared memory (team/project level).
    /// Requires explicit user authorization to promote from `User`.
    Shared,

    /// Soft-deleted entries — hidden from search, recoverable via tombstone.
    Deleted,
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryScope::Session => write!(f, "session"),
            MemoryScope::User => write!(f, "user"),
            MemoryScope::Shared => write!(f, "shared"),
            MemoryScope::Deleted => write!(f, "deleted"),
        }
    }
}

impl MemoryScope {
    /// Parse from YAML front matter string (case-insensitive, defaults to User).
    pub fn from_str_scope(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "session" => MemoryScope::Session,
            "shared" => MemoryScope::Shared,
            "deleted" => MemoryScope::Deleted,
            _ => MemoryScope::User,
        }
    }

    /// Numeric privilege level (higher = broader access).
    pub fn level(&self) -> u8 {
        match self {
            MemoryScope::Session => 0,
            MemoryScope::User => 1,
            MemoryScope::Shared => 2,
            MemoryScope::Deleted => 255,
        }
    }

    /// Check if `self` can access memory at `target` scope.
    ///
    /// Rules:
    /// - Session scope cannot access User or Shared memory
    /// - User scope can access User and Session memory, not Shared
    /// - Shared scope can access everything
    pub fn can_access(&self, target: MemoryScope) -> bool {
        self.level() >= target.level()
    }

    /// Whether cross-scope promotion requires user confirmation.
    pub fn requires_promotion(from: MemoryScope, to: MemoryScope) -> bool {
        to.level() > from.level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_user() {
        let s = MemoryScope::default();
        assert_eq!(s, MemoryScope::User);
    }

    #[test]
    fn test_from_str() {
        assert_eq!(MemoryScope::from_str_scope("session"), MemoryScope::Session);
        assert_eq!(MemoryScope::from_str_scope("user"), MemoryScope::User);
        assert_eq!(MemoryScope::from_str_scope("shared"), MemoryScope::Shared);
        assert_eq!(MemoryScope::from_str_scope("unknown"), MemoryScope::User);
        assert_eq!(MemoryScope::from_str_scope(""), MemoryScope::User);
    }

    #[test]
    fn test_access_rules() {
        let session = MemoryScope::Session;
        let user = MemoryScope::User;
        let shared = MemoryScope::Shared;

        // Session can only see session
        assert!(session.can_access(MemoryScope::Session));
        assert!(!session.can_access(MemoryScope::User));
        assert!(!session.can_access(MemoryScope::Shared));

        // User can see session + user, not shared
        assert!(user.can_access(MemoryScope::Session));
        assert!(user.can_access(MemoryScope::User));
        assert!(!user.can_access(MemoryScope::Shared));

        // Shared can see everything
        assert!(shared.can_access(MemoryScope::Session));
        assert!(shared.can_access(MemoryScope::User));
        assert!(shared.can_access(MemoryScope::Shared));
    }

    #[test]
    fn test_promotion_requires_confirmation() {
        assert!(MemoryScope::requires_promotion(
            MemoryScope::Session,
            MemoryScope::User
        ));
        assert!(MemoryScope::requires_promotion(
            MemoryScope::User,
            MemoryScope::Shared
        ));
        assert!(!MemoryScope::requires_promotion(
            MemoryScope::User,
            MemoryScope::Session
        ));
        assert!(!MemoryScope::requires_promotion(
            MemoryScope::Shared,
            MemoryScope::User
        ));
    }

    #[test]
    fn test_serialize() {
        let s = MemoryScope::Session;
        let yaml = serde_yaml::to_string(&s).unwrap();
        assert!(yaml.contains("session"));
    }
}
