//! Memory Metabolism — automatic lifecycle management for memories.
//!
//! Design goals:
//! - **Auto-archive**: memories that meet certain criteria are automatically archived
//! - **Smart decay**: importance decays over time without access
//! - **Promotion**: frequently accessed memories get promoted
//! - **Deletion**: deprecated memories are eventually hard-deleted
//!
//! Memory lifecycle stages:
//! 1. **Working**: active in hot buffer
//! 2. **Short-term**: recent daily memories
//! 3. **Long-term**: topic memories, core memories
//! 4. **Archive**: cold storage, rarely accessed
//! 5. **Deprecated**: soft-deleted, awaiting hard-delete

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::cold_store::ColdStorageConfig;

/// Memory metabolism configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetabolismConfig {
    /// Enable memory metabolism.
    pub enabled: bool,
    /// Days without access before memory is considered for archival.
    pub archive_after_days: u32,
    /// Minimum importance score to avoid archival.
    pub min_importance_threshold: f64,
    /// Days to keep deprecated memories before hard-delete.
    pub deprecated_retention_days: u32,
    /// Maximum number of memories to archive per run.
    pub max_archive_per_run: usize,
    /// Whether to auto-promote frequently accessed memories.
    pub enable_promotion: bool,
    /// Access count threshold for promotion.
    pub promotion_access_threshold: u32,
}

impl Default for MetabolismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            archive_after_days: 30,
            min_importance_threshold: 0.3,
            deprecated_retention_days: 90,
            max_archive_per_run: 10,
            enable_promotion: true,
            promotion_access_threshold: 10,
        }
    }
}

/// Memory lifecycle stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryStage {
    /// Active in working memory.
    Working,
    /// Recent daily memories.
    ShortTerm,
    /// Topic memories, core memories.
    LongTerm,
    /// Cold storage, rarely accessed.
    Archive,
    /// Soft-deleted, awaiting hard-delete.
    Deprecated,
}

/// Memory entry with lifecycle metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLifecycleEntry {
    /// Path to the memory file.
    pub path: PathBuf,
    /// Current lifecycle stage.
    pub stage: MemoryStage,
    /// Current importance score (0.0-1.0).
    pub importance: f64,
    /// Number of times accessed.
    pub access_count: u32,
    /// Last access timestamp (seconds since epoch).
    pub last_access: Option<u64>,
    /// Creation timestamp (seconds since epoch).
    pub created_at: u64,
    /// Last modified timestamp (seconds since epoch).
    pub modified_at: u64,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl MemoryLifecycleEntry {
    /// Create a new entry from file metadata.
    pub fn from_file(path: PathBuf, stage: MemoryStage) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            path,
            stage,
            importance: 0.5,
            access_count: 0,
            last_access: None,
            created_at: now,
            modified_at: now,
            tags: Vec::new(),
        }
    }

    /// Calculate age in days.
    pub fn age_days(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        (now - self.created_at) / 86400
    }

    /// Calculate days since last access.
    pub fn days_since_access(&self) -> Option<u64> {
        let last = self.last_access?;
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Some((now - last) / 86400)
    }

    /// Check if entry should be archived.
    pub fn should_archive(&self, config: &MetabolismConfig) -> bool {
        if self.stage == MemoryStage::Archive || self.stage == MemoryStage::Deprecated {
            return false;
        }

        // Check importance threshold
        if self.importance >= config.min_importance_threshold {
            return false;
        }

        // Check access recency
        if let Some(days) = self.days_since_access() {
            days >= config.archive_after_days as u64
        } else {
            // Never accessed, check age
            self.age_days() >= config.archive_after_days as u64
        }
    }

    /// Check if entry should be promoted.
    pub fn should_promote(&self, config: &MetabolismConfig) -> bool {
        if !config.enable_promotion {
            return false;
        }

        // Already at highest stage
        if self.stage == MemoryStage::LongTerm || self.stage == MemoryStage::Archive {
            return false;
        }

        // Check access count
        self.access_count >= config.promotion_access_threshold
    }

    /// Check if deprecated entry should be hard-deleted.
    pub fn should_hard_delete(&self, config: &MetabolismConfig) -> bool {
        if self.stage != MemoryStage::Deprecated {
            return false;
        }

        if let Some(days) = self.days_since_access() {
            days >= config.deprecated_retention_days as u64
        } else {
            self.age_days() >= config.deprecated_retention_days as u64
        }
    }

    /// Apply decay to importance score.
    pub fn apply_decay(&mut self, decay_rate: f64, min_importance: f64) {
        let days = self.days_since_access().unwrap_or(0) as f64;
        let decay_factor = (-decay_rate * days).exp();
        self.importance = (self.importance * decay_factor).max(min_importance);
    }
}

/// Memory metabolism manager.
pub struct MemoryMetabolism {
    config: MetabolismConfig,
    workspace_root: PathBuf,
}

impl MemoryMetabolism {
    /// Create a new metabolism manager.
    pub fn new(config: MetabolismConfig, workspace_root: PathBuf) -> Self {
        Self {
            config,
            workspace_root,
        }
    }

    /// Process memory lifecycle: archive, promote, delete.
    pub fn process_lifecycle(
        &self,
        entries: &mut Vec<MemoryLifecycleEntry>,
    ) -> MetabolismResult {
        let mut result = MetabolismResult::default();

        if !self.config.enabled {
            return result;
        }

        // 1. Archive entries that meet criteria
        let mut to_archive = Vec::new();
        for entry in entries.iter() {
            if entry.should_archive(&self.config) {
                to_archive.push(entry.path.clone());
            }
        }

        for path in to_archive.iter().take(self.config.max_archive_per_run) {
            if let Some(entry) = entries.iter_mut().find(|e| e.path == *path) {
                entry.stage = MemoryStage::Archive;
                result.archived += 1;
            }
        }

        // 2. Promote frequently accessed entries
        let mut to_promote = Vec::new();
        for entry in entries.iter() {
            if entry.should_promote(&self.config) {
                to_promote.push(entry.path.clone());
            }
        }

        for path in to_promote {
            if let Some(entry) = entries.iter_mut().find(|e| e.path == *path) {
                match entry.stage {
                    MemoryStage::Working => {
                        entry.stage = MemoryStage::ShortTerm;
                        result.promoted += 1;
                    }
                    MemoryStage::ShortTerm => {
                        entry.stage = MemoryStage::LongTerm;
                        result.promoted += 1;
                    }
                    _ => {}
                }
            }
        }

        // 3. Hard-delete deprecated entries
        let mut to_delete = Vec::new();
        for entry in entries.iter() {
            if entry.should_hard_delete(&self.config) {
                to_delete.push(entry.path.clone());
            }
        }

        for path in to_delete {
            if let Some(idx) = entries.iter().position(|e| e.path == path) {
                entries.remove(idx);
                result.deleted += 1;
            }
        }

        // 4. Apply decay to all entries
        for entry in entries.iter_mut() {
            entry.apply_decay(0.01, 0.1);
        }

        result
    }

    /// Generate a metabolism report.
    pub fn generate_report(&self, entries: &[MemoryLifecycleEntry]) -> String {
        let mut report = String::new();
        report.push_str("## Memory Metabolism Report\n\n");

        // Stage distribution
        let working_count = entries.iter().filter(|e| e.stage == MemoryStage::Working).count();
        let short_term_count = entries.iter().filter(|e| e.stage == MemoryStage::ShortTerm).count();
        let long_term_count = entries.iter().filter(|e| e.stage == MemoryStage::LongTerm).count();
        let archive_count = entries.iter().filter(|e| e.stage == MemoryStage::Archive).count();
        let deprecated_count = entries.iter().filter(|e| e.stage == MemoryStage::Deprecated).count();

        report.push_str("### Stage Distribution\n");
        report.push_str(&format!("- Working: {}\n", working_count));
        report.push_str(&format!("- Short-term: {}\n", short_term_count));
        report.push_str(&format!("- Long-term: {}\n", long_term_count));
        report.push_str(&format!("- Archive: {}\n", archive_count));
        report.push_str(&format!("- Deprecated: {}\n", deprecated_count));

        // Importance distribution
        let avg_importance = if entries.is_empty() {
            0.0
        } else {
            entries.iter().map(|e| e.importance).sum::<f64>() / entries.len() as f64
        };
        report.push_str(&format!("\n### Importance\n"));
        report.push_str(&format!("- Average: {:.2}\n", avg_importance));

        // Candidates for action
        let archive_candidates = entries.iter().filter(|e| e.should_archive(&self.config)).count();
        let promote_candidates = entries.iter().filter(|e| e.should_promote(&self.config)).count();
        let delete_candidates = entries.iter().filter(|e| e.should_hard_delete(&self.config)).count();

        report.push_str("\n### Action Candidates\n");
        report.push_str(&format!("- Archive: {}\n", archive_candidates));
        report.push_str(&format!("- Promote: {}\n", promote_candidates));
        report.push_str(&format!("- Hard-delete: {}\n", delete_candidates));

        report
    }
}

/// Result of a metabolism processing run.
#[derive(Debug, Default)]
pub struct MetabolismResult {
    /// Number of entries archived.
    pub archived: usize,
    /// Number of entries promoted.
    pub promoted: usize,
    /// Number of entries hard-deleted.
    pub deleted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(stage: MemoryStage, importance: f64, access_count: u32) -> MemoryLifecycleEntry {
        MemoryLifecycleEntry {
            path: PathBuf::from("test.md"),
            stage,
            importance,
            access_count,
            last_access: Some(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
            created_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            modified_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            tags: Vec::new(),
        }
    }

    #[test]
    fn test_should_archive_low_importance() {
        let config = MetabolismConfig {
            archive_after_days: 30,
            min_importance_threshold: 0.3,
            ..Default::default()
        };

        let entry = create_test_entry(MemoryStage::ShortTerm, 0.1, 0);
        assert!(entry.should_archive(&config));
    }

    #[test]
    fn test_should_not_archive_high_importance() {
        let config = MetabolismConfig {
            archive_after_days: 30,
            min_importance_threshold: 0.3,
            ..Default::default()
        };

        let entry = create_test_entry(MemoryStage::ShortTerm, 0.8, 0);
        assert!(!entry.should_archive(&config));
    }

    #[test]
    fn test_should_promote_high_access() {
        let config = MetabolismConfig {
            promotion_access_threshold: 10,
            enable_promotion: true,
            ..Default::default()
        };

        let entry = create_test_entry(MemoryStage::Working, 0.5, 15);
        assert!(entry.should_promote(&config));
    }

    #[test]
    fn test_should_not_promote_low_access() {
        let config = MetabolismConfig {
            promotion_access_threshold: 10,
            enable_promotion: true,
            ..Default::default()
        };

        let entry = create_test_entry(MemoryStage::Working, 0.5, 3);
        assert!(!entry.should_promote(&config));
    }

    #[test]
    fn test_lifecycle_processing() {
        let config = MetabolismConfig::default();
        let metabolism = MemoryMetabolism::new(config, PathBuf::from("."));

        let mut entries = vec![
            create_test_entry(MemoryStage::ShortTerm, 0.1, 0),
            create_test_entry(MemoryStage::Working, 0.5, 15),
        ];

        let result = metabolism.process_lifecycle(&mut entries);
        assert_eq!(result.archived, 1);
        assert_eq!(result.promoted, 1);
    }
}
