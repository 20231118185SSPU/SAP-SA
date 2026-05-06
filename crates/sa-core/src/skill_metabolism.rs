//! Skill Metabolism — automatic lifecycle management for skills.
//!
//! Design goals:
//! - **Auto-elimination**: remove low-frequency skills
//! - **Auto-merge**: combine similar skills
//! - **Usage tracking**: monitor skill usage patterns
//! - **Smart promotion**: promote frequently used skills
//!
//! Skill lifecycle:
//! 1. **Discovery**: skill found in skill directories
//! 2. **Active**: skill is available for use
//! 3. **Low-frequency**: skill is rarely used
//! 4. **Deprecated**: skill is marked for removal
//! 5. **Merged**: skill content merged into another skill

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Skill metabolism configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SkillMetabolismConfig {
    /// Enable skill metabolism.
    pub enabled: bool,
    /// Minimum usage count to remain active.
    pub min_usage_threshold: u32,
    /// Days without use before considering elimination.
    pub inactive_days_threshold: u32,
    /// Similarity threshold for merging (0.0-1.0).
    pub merge_similarity_threshold: f64,
    /// Maximum number of skills to eliminate per run.
    pub max_eliminations_per_run: usize,
    /// Whether to auto-merge similar skills.
    pub enable_auto_merge: bool,
    /// Whether to auto-eliminate low-frequency skills.
    pub enable_auto_eliminate: bool,
}

impl Default for SkillMetabolismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_usage_threshold: 5,
            inactive_days_threshold: 90,
            merge_similarity_threshold: 0.8,
            max_eliminations_per_run: 5,
            enable_auto_merge: true,
            enable_auto_eliminate: true,
        }
    }
}

/// Skill lifecycle stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SkillStage {
    /// Newly discovered.
    Discovered,
    /// Active and available.
    Active,
    /// Rarely used.
    LowFrequency,
    /// Marked for removal.
    Deprecated,
    /// Merged into another skill.
    Merged,
}

/// Skill usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillUsageStats {
    /// Total number of times used.
    pub usage_count: u32,
    /// Number of successful uses.
    pub success_count: u32,
    /// Number of failed uses.
    pub failure_count: u32,
    /// Last used timestamp (seconds since epoch).
    pub last_used: Option<u64>,
    /// Average execution time in milliseconds.
    pub avg_execution_time_ms: f64,
}

impl Default for SkillUsageStats {
    fn default() -> Self {
        Self {
            usage_count: 0,
            success_count: 0,
            failure_count: 0,
            last_used: None,
            avg_execution_time_ms: 0.0,
        }
    }
}

/// Skill entry with lifecycle metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLifecycleEntry {
    /// Skill name.
    pub name: String,
    /// Path to SKILL.md file.
    pub path: PathBuf,
    /// Current lifecycle stage.
    pub stage: SkillStage,
    /// Skill description.
    pub description: String,
    /// Usage statistics.
    pub usage: SkillUsageStats,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Skill version.
    pub version: String,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last modified timestamp.
    pub modified_at: u64,
}

impl SkillLifecycleEntry {
    /// Create a new entry from a skill file.
    pub fn from_file(path: PathBuf, name: String, description: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            name,
            path,
            stage: SkillStage::Discovered,
            description,
            usage: SkillUsageStats::default(),
            tags: Vec::new(),
            version: "1.0.0".to_string(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Record a skill usage.
    pub fn record_usage(&mut self, success: bool, execution_time_ms: f64) {
        self.usage.usage_count += 1;
        if success {
            self.usage.success_count += 1;
        } else {
            self.usage.failure_count += 1;
        }

        self.usage.last_used = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        // Update average execution time
        let total_time = self.usage.avg_execution_time_ms
            * (self.usage.usage_count - 1) as f64;
        self.usage.avg_execution_time_ms =
            (total_time + execution_time_ms) / self.usage.usage_count as f64;
    }

    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.usage.usage_count == 0 {
            return 0.0;
        }
        self.usage.success_count as f64 / self.usage.usage_count as f64
    }

    /// Calculate days since last use.
    pub fn days_since_last_use(&self) -> Option<u64> {
        let last = self.usage.last_used?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Some((now - last) / 86400)
    }

    /// Check if skill should be eliminated.
    pub fn should_eliminate(&self, config: &SkillMetabolismConfig) -> bool {
        if !config.enable_auto_eliminate {
            return false;
        }

        // Already deprecated or merged
        if self.stage == SkillStage::Deprecated || self.stage == SkillStage::Merged {
            return false;
        }

        // Check usage threshold
        if self.usage.usage_count < config.min_usage_threshold {
            // Check if inactive long enough
            if let Some(days) = self.days_since_last_use() {
                return days >= config.inactive_days_threshold as u64;
            }
            // Never used, check age
            let age = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                - self.created_at;
            return age >= config.inactive_days_threshold as u64 * 86400;
        }

        false
    }

    /// Check if skill should be promoted.
    pub fn should_promote(&self) -> bool {
        match self.stage {
            SkillStage::Discovered => self.usage.usage_count >= 3,
            SkillStage::LowFrequency => self.usage.usage_count >= 10,
            _ => false,
        }
    }

    /// Calculate similarity with another skill.
    pub fn similarity(&self, other: &SkillLifecycleEntry) -> f64 {
        // Simple similarity based on description and tags
        let desc_similarity = self.text_similarity(&self.description, &other.description);
        let tag_similarity = self.tag_similarity(&other.tags);

        (desc_similarity * 0.7 + tag_similarity * 0.3).min(1.0)
    }

    /// Calculate text similarity (simple word overlap).
    fn text_similarity(&self, text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Calculate tag similarity.
    fn tag_similarity(&self, other_tags: &[String]) -> f64 {
        if self.tags.is_empty() && other_tags.is_empty() {
            return 1.0;
        }

        let tags1: std::collections::HashSet<&String> = self.tags.iter().collect();
        let tags2: std::collections::HashSet<&String> = other_tags.iter().collect();

        let intersection = tags1.intersection(&tags2).count();
        let union = tags1.union(&tags2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Skill metabolism manager.
pub struct SkillMetabolism {
    config: SkillMetabolismConfig,
    workspace_root: PathBuf,
}

impl SkillMetabolism {
    /// Create a new metabolism manager.
    pub fn new(config: SkillMetabolismConfig, workspace_root: PathBuf) -> Self {
        Self {
            config,
            workspace_root,
        }
    }

    /// Process skill lifecycle: eliminate, merge, promote.
    pub fn process_lifecycle(
        &self,
        entries: &mut Vec<SkillLifecycleEntry>,
    ) -> SkillMetabolismResult {
        let mut result = SkillMetabolismResult::default();

        if !self.config.enabled {
            return result;
        }

        // 1. Eliminate low-frequency skills
        let mut to_eliminate = Vec::new();
        for entry in entries.iter() {
            if entry.should_eliminate(&self.config) {
                to_eliminate.push(entry.name.clone());
            }
        }

        for name in to_eliminate.iter().take(self.config.max_eliminations_per_run) {
            if let Some(entry) = entries.iter_mut().find(|e| e.name == *name) {
                entry.stage = SkillStage::Deprecated;
                result.eliminated += 1;
            }
        }

        // 2. Merge similar skills
        if self.config.enable_auto_merge {
            let merge_candidates = self.find_merge_candidates(entries);
            for (name1, name2) in merge_candidates {
                if let Some(idx) = entries.iter().position(|e| e.name == name2) {
                    entries.remove(idx);
                    result.merged += 1;
                }
            }
        }

        // 3. Promote frequently used skills
        for entry in entries.iter_mut() {
            if entry.should_promote() {
                match entry.stage {
                    SkillStage::Discovered => {
                        entry.stage = SkillStage::Active;
                        result.promoted += 1;
                    }
                    SkillStage::LowFrequency => {
                        entry.stage = SkillStage::Active;
                        result.promoted += 1;
                    }
                    _ => {}
                }
            }
        }

        // 4. Update stages based on usage
        for entry in entries.iter_mut() {
            if entry.stage == SkillStage::Active
                && entry.usage.usage_count < self.config.min_usage_threshold
            {
                if let Some(days) = entry.days_since_last_use() {
                    if days >= 30 {
                        entry.stage = SkillStage::LowFrequency;
                    }
                }
            }
        }

        result
    }

    /// Find pairs of skills that should be merged.
    fn find_merge_candidates(&self, entries: &[SkillLifecycleEntry]) -> Vec<(String, String)> {
        let mut candidates = Vec::new();
        let mut processed = std::collections::HashSet::new();

        for i in 0..entries.len() {
            if processed.contains(&entries[i].name) {
                continue;
            }

            for j in i + 1..entries.len() {
                if processed.contains(&entries[j].name) {
                    continue;
                }

                let similarity = entries[i].similarity(&entries[j]);
                if similarity >= self.config.merge_similarity_threshold {
                    candidates.push((entries[i].name.clone(), entries[j].name.clone()));
                    processed.insert(entries[j].name.clone());
                    break;
                }
            }
        }

        candidates
    }

    /// Generate a metabolism report.
    pub fn generate_report(&self, entries: &[SkillLifecycleEntry]) -> String {
        let mut report = String::new();
        report.push_str("## Skill Metabolism Report\n\n");

        // Stage distribution
        let discovered_count = entries.iter().filter(|e| e.stage == SkillStage::Discovered).count();
        let active_count = entries.iter().filter(|e| e.stage == SkillStage::Active).count();
        let low_freq_count = entries.iter().filter(|e| e.stage == SkillStage::LowFrequency).count();
        let deprecated_count = entries.iter().filter(|e| e.stage == SkillStage::Deprecated).count();
        let merged_count = entries.iter().filter(|e| e.stage == SkillStage::Merged).count();

        report.push_str("### Stage Distribution\n");
        report.push_str(&format!("- Discovered: {}\n", discovered_count));
        report.push_str(&format!("- Active: {}\n", active_count));
        report.push_str(&format!("- Low-frequency: {}\n", low_freq_count));
        report.push_str(&format!("- Deprecated: {}\n", deprecated_count));
        report.push_str(&format!("- Merged: {}\n", merged_count));

        // Usage statistics
        let total_usage: u32 = entries.iter().map(|e| e.usage.usage_count).sum();
        let avg_usage = if entries.is_empty() {
            0.0
        } else {
            total_usage as f64 / entries.len() as f64
        };
        report.push_str(&format!("\n### Usage Statistics\n"));
        report.push_str(&format!("- Total usage count: {}\n", total_usage));
        report.push_str(&format!("- Average usage per skill: {:.1}\n", avg_usage));

        // Candidates for action
        let eliminate_candidates = entries.iter().filter(|e| e.should_eliminate(&self.config)).count();
        let promote_candidates = entries.iter().filter(|e| e.should_promote()).count();

        report.push_str("\n### Action Candidates\n");
        report.push_str(&format!("- Eliminate: {}\n", eliminate_candidates));
        report.push_str(&format!("- Promote: {}\n", promote_candidates));

        // Top used skills
        let mut top_skills: Vec<&SkillLifecycleEntry> = entries.iter().collect();
        top_skills.sort_by(|a, b| b.usage.usage_count.cmp(&a.usage.usage_count));
        top_skills.truncate(5);

        if !top_skills.is_empty() {
            report.push_str("\n### Top Used Skills\n");
            for skill in top_skills {
                report.push_str(&format!(
                    "- {}: {} uses ({:.1}% success)\n",
                    skill.name,
                    skill.usage.usage_count,
                    skill.success_rate() * 100.0
                ));
            }
        }

        report
    }
}

/// Result of a skill metabolism processing run.
#[derive(Debug, Default)]
pub struct SkillMetabolismResult {
    /// Number of skills eliminated.
    pub eliminated: usize,
    /// Number of skills merged.
    pub merged: usize,
    /// Number of skills promoted.
    pub promoted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_skill(name: &str, usage_count: u32) -> SkillLifecycleEntry {
        SkillLifecycleEntry {
            name: name.to_string(),
            path: PathBuf::from(format!("skills/{}/SKILL.md", name)),
            stage: SkillStage::Active,
            description: format!("Test skill {}", name),
            usage: SkillUsageStats {
                usage_count,
                success_count: usage_count,
                failure_count: 0,
                last_used: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                ),
                avg_execution_time_ms: 100.0,
            },
            tags: vec!["test".to_string()],
            version: "1.0.0".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            modified_at: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    #[test]
    fn test_should_eliminate_low_usage() {
        let config = SkillMetabolismConfig {
            min_usage_threshold: 5,
            inactive_days_threshold: 90,
            enable_auto_eliminate: true,
            ..Default::default()
        };

        let mut entry = create_test_skill("test", 1);
        entry.usage.last_used = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - 100 * 86400,
        );

        assert!(entry.should_eliminate(&config));
    }

    #[test]
    fn test_should_not_eliminate_high_usage() {
        let config = SkillMetabolismConfig {
            min_usage_threshold: 5,
            enable_auto_eliminate: true,
            ..Default::default()
        };

        let entry = create_test_skill("test", 10);
        assert!(!entry.should_eliminate(&config));
    }

    #[test]
    fn test_should_promote_discovered() {
        let mut entry = create_test_skill("test", 3);
        entry.stage = SkillStage::Discovered;
        assert!(entry.should_promote());
    }

    #[test]
    fn test_similarity() {
        let entry1 = create_test_skill("test1", 0);
        let entry2 = create_test_skill("test2", 0);

        let mut entry1 = entry1;
        entry1.description = "memory management system".to_string();
        entry1.tags = vec!["memory".to_string(), "system".to_string()];

        let mut entry2 = entry2;
        entry2.description = "memory management module".to_string();
        entry2.tags = vec!["memory".to_string(), "module".to_string()];

        let similarity = entry1.similarity(&entry2);
        assert!(similarity > 0.5);
    }

    #[test]
    fn test_lifecycle_processing() {
        let config = SkillMetabolismConfig::default();
        let metabolism = SkillMetabolism::new(config, PathBuf::from("."));

        let mut entries = vec![
            create_test_skill("active", 10),
            create_test_skill("low_freq", 1),
        ];

        let result = metabolism.process_lifecycle(&mut entries);
        // active should stay active, low_freq might be promoted if it has enough usage
        assert!(result.promoted <= 1);
    }
}
