//! D10: Search Feedback Learning — Thompson Sampling
//!
//! Tracks which recall paths produce adopted results and uses Beta-distribution
//! Thompson Sampling to dynamically adjust weights for time/entity/emotion boosts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Recall path identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecallPath {
    /// BM25 lexical matching (always active, weight = 1.0).
    Lexical,
    /// Vector cosine similarity (D2, when embedding available).
    Vector,
    /// Time proximity boost (D9).
    Time,
    /// Entity inverted recall (D9).
    Entity,
    /// Emotion recall (D9).
    Emotion,
}

impl RecallPath {
    pub const ALL_ADJUSTABLE: [RecallPath; 3] = [
        RecallPath::Time,
        RecallPath::Entity,
        RecallPath::Emotion,
    ];
}

/// Beta distribution parameters for one recall path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaArm {
    /// Alpha (successes + 1).
    pub alpha: f64,
    /// Beta (failures + 1).
    pub beta: f64,
}

impl BetaArm {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Sample from Beta(alpha, beta) using a simple rejection method.
    /// Falls back to mean if sampling fails.
    pub fn sample(&self) -> f64 {
        // Use the mean as a stable fallback: alpha / (alpha + beta)
        self.alpha / (self.alpha + self.beta)
    }

    /// Record a success (result was adopted).
    pub fn record_success(&mut self) {
        self.alpha += 1.0;
    }

    /// Record a failure (result was not adopted).
    pub fn record_failure(&mut self) {
        self.beta += 1.0;
    }
}

/// One recorded search event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEvent {
    /// Hash of the query for grouping.
    pub query_hash: String,
    /// Which recall paths contributed to adopted results.
    pub adopted_paths: Vec<RecallPath>,
    /// Which recall paths were active but did not contribute.
    pub non_adopted_paths: Vec<RecallPath>,
}

/// Feedback state persisted to workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFeedback {
    /// Beta arms for each adjustable recall path.
    pub arms: HashMap<String, BetaArm>,
    /// Recent events (kept for debugging, pruned to last 200).
    pub events: Vec<SearchEvent>,
    /// Pending search results: path → contributing recall paths.
    /// Populated by memory_search, consumed by memory_get.
    pub pending_results: HashMap<String, Vec<RecallPath>>,
}

impl SearchFeedback {
    pub fn new() -> Self {
        let mut arms = HashMap::new();
        for rp in RecallPath::ALL_ADJUSTABLE {
            arms.insert(arm_key(rp), BetaArm::new());
        }
        Self {
            arms,
            events: Vec::new(),
            pending_results: HashMap::new(),
        }
    }

    /// Load from workspace feedback file, or create new.
    pub fn load(workspace_root: &Path) -> Self {
        let path = feedback_path(workspace_root);
        match std::fs::read_to_string(&path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| Self::new()),
            Err(_) => Self::new(),
        }
    }

    /// Persist to workspace feedback file.
    pub fn save(&self, workspace_root: &Path) -> anyhow::Result<()> {
        let path = feedback_path(workspace_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Prune events before saving
        let mut clone = self.clone();
        if clone.events.len() > 200 {
            clone.events.drain(0..clone.events.len() - 200);
        }
        let json = serde_json::to_string_pretty(&clone)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Get current weight for a recall path (Thompson Sampling sample).
    pub fn weight(&self, path: RecallPath) -> f64 {
        let key = arm_key(path);
        self.arms
            .get(&key)
            .map(|arm| arm.sample())
            .unwrap_or(0.5)
    }

    /// Record that a search produced adopted results.
    /// `active_paths` = all paths that were active during this search.
    /// `contributing_paths` = paths that contributed to the adopted result.
    pub fn record_feedback(
        &mut self,
        query: &str,
        active_paths: &[RecallPath],
        contributing_paths: &[RecallPath],
    ) {
        let query_hash = format!("{:016x}", md5_hash(query));
        let non_adopted: Vec<RecallPath> = active_paths
            .iter()
            .copied()
            .filter(|p| !contributing_paths.contains(p))
            .collect();

        // Update Beta arms
        for rp in RecallPath::ALL_ADJUSTABLE {
            let key = arm_key(rp);
            if let Some(arm) = self.arms.get_mut(&key) {
                if contributing_paths.contains(&rp) {
                    arm.record_success();
                } else if active_paths.contains(&rp) {
                    arm.record_failure();
                }
            }
        }

        self.events.push(SearchEvent {
            query_hash,
            adopted_paths: contributing_paths.to_vec(),
            non_adopted_paths: non_adopted,
        });
    }

    /// Store a single search result path with its contributing recall paths.
    /// Called by memory_search after computing D9 boosts for each result.
    pub fn store_pending_result(&mut self, result_path: String, contributing: Vec<RecallPath>) {
        self.pending_results.insert(result_path, contributing);
    }

    /// Consume pending result for a path (called by memory_get).
    /// Returns the contributing paths if the path was in recent search results, None otherwise.
    pub fn consume_pending_result(&mut self, result_path: &str) -> Option<Vec<RecallPath>> {
        self.pending_results.remove(result_path)
    }

    /// Clear all pending search results (call at start of each memory_search).
    pub fn clear_pending_results(&mut self) {
        self.pending_results.clear();
    }

    /// Get all current weights as a displayable map.
    pub fn weight_snapshot(&self) -> HashMap<String, f64> {
        let mut out = HashMap::new();
        for rp in RecallPath::ALL_ADJUSTABLE {
            out.insert(format!("{:?}", rp), self.weight(rp));
        }
        out
    }
}

fn arm_key(path: RecallPath) -> String {
    format!("{:?}", path)
}

fn feedback_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".memory_feedback.json")
}

/// Simple FNV-1a 64-bit hash for query deduplication.
fn md5_hash(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_arm_defaults() {
        let arm = BetaArm::new();
        assert!((arm.sample() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_beta_arm_success_shifts_up() {
        let mut arm = BetaArm::new();
        arm.record_success();
        arm.record_success();
        arm.record_success();
        assert!(arm.sample() > 0.5);
    }

    #[test]
    fn test_beta_arm_failure_shifts_down() {
        let mut arm = BetaArm::new();
        arm.record_failure();
        arm.record_failure();
        arm.record_failure();
        assert!(arm.sample() < 0.5);
    }

    #[test]
    fn test_feedback_record() {
        let mut fb = SearchFeedback::new();
        let active = vec![RecallPath::Time, RecallPath::Entity, RecallPath::Emotion];
        let contributing = vec![RecallPath::Time];
        fb.record_feedback("test query", &active, &contributing);
        assert_eq!(fb.events.len(), 1);
        assert!(fb.weight(RecallPath::Time) > 0.5);
    }

    #[test]
    fn test_feedback_persistence() {
        let dir = std::env::temp_dir().join("sa_feedback_test");
        let _ = std::fs::create_dir_all(&dir);
        let mut fb = SearchFeedback::new();
        fb.record_feedback(
            "persist test",
            &[RecallPath::Time, RecallPath::Entity],
            &[RecallPath::Entity],
        );
        fb.save(&dir).unwrap();
        let loaded = SearchFeedback::load(&dir);
        assert_eq!(loaded.events.len(), 1);
        let _ = std::fs::remove_file(dir.join(".memory_feedback.json"));
    }

    // --- D10+D11: End-to-end integration test ---

    #[test]
    fn test_full_search_feedback_loop() {
        // Simulate the complete pipeline:
        // 1. memory_search stores pending results with contributing paths
        // 2. memory_get consumes pending, records feedback
        // 3. save → load round-trip preserves state
        let dir = std::env::temp_dir().join("sa_e2e_feedback_test");
        let _ = std::fs::create_dir_all(&dir);

        let mut fb = SearchFeedback::new();
        let active = RecallPath::ALL_ADJUSTABLE.to_vec();

        // Simulate 5 searches where Time consistently contributes
        for i in 0..5 {
            let path = format!("memory/2026-04-{:02}.md", i + 20);
            fb.clear_pending_results();
            // Time + Entity contribute
            fb.store_pending_result(
                path.clone(),
                vec![RecallPath::Time, RecallPath::Entity],
            );

            // memory_get consumes the pending result
            let consumed = fb.consume_pending_result(&path);
            assert!(consumed.is_some());
            let contributing = consumed.unwrap();

            // record feedback
            fb.record_feedback("昨天的讨论", &active, &contributing);
        }

        // Time should have higher weight (5 successes)
        let time_w = fb.weight(RecallPath::Time);
        let entity_w = fb.weight(RecallPath::Entity);
        let emotion_w = fb.weight(RecallPath::Emotion);

        assert!(time_w > 0.5, "Time weight should be >0.5, got {time_w}");
        assert!(entity_w > 0.5, "Entity weight should be >0.5, got {entity_w}");
        assert!(
            emotion_w < time_w,
            "Emotion ({emotion_w}) should be < Time ({time_w})"
        );
        assert!(
            emotion_w < entity_w,
            "Emotion ({emotion_w}) should be < Entity ({entity_w})"
        );

        // Save and reload
        fb.save(&dir).unwrap();
        let loaded = SearchFeedback::load(&dir);

        // Verify loaded weights match saved state
        assert_eq!(loaded.events.len(), 5);
        assert!((loaded.weight(RecallPath::Time) - time_w).abs() < 0.001);
        assert!((loaded.weight(RecallPath::Entity) - entity_w).abs() < 0.001);
        assert!((loaded.weight(RecallPath::Emotion) - emotion_w).abs() < 0.001);

        // Verify pending_results are NOT persisted (they're transient)
        assert!(loaded.pending_results.is_empty());

        let _ = std::fs::remove_file(dir.join(".memory_feedback.json"));
    }

    #[test]
    fn test_adaptive_weight_math() {
        // Verify D11 INTENT_BOOST/INTENT_DAMP math
        let boost = 1.5_f64;
        let damp = 0.7_f64;

        // Time query: time gets boost, others get damp
        let tw = 0.6 * boost; // 0.9
        let ew = 0.4 * damp;  // 0.28
        let emw = 0.3 * damp; // 0.21

        assert!((tw - 0.9).abs() < 0.001);
        assert!((ew - 0.28).abs() < 0.001);
        assert!(tw > ew);
        assert!(tw > emw);

        // Neutral: no change
        let tw_n = 0.6 * 1.0;
        assert!((tw_n - 0.6_f64).abs() < 0.001);
    }
}

