//! Noise Assessment Module — measures context window noise to optimize agent efficiency.
//!
//! Design goals:
//! - **Noise metrics**: measure repeated content, stale information, irrelevant tool outputs
//! - **Integration points**: hook into working_memory, compaction, and prompt injection
//! - **Actionable feedback**: provide noise ratio and suggestions for reduction
//!
//! Noise sources:
//! 1. **Duplication**: same or similar content appearing multiple times
//! 2. **Stale content**: old messages with low importance and no recent access
//! 3. **Tool output bloat**: verbose tool outputs that don't contribute to understanding
//! 4. **Irrelevant context**: content that doesn't match current task focus

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::working_memory::WorkingMemoryEntry;

/// Noise assessment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseConfig {
    /// Enable noise assessment.
    pub enabled: bool,
    /// Threshold for "high noise" warning (0.0-1.0).
    pub high_noise_threshold: f64,
    /// Maximum age (in days) before content is considered stale.
    pub stale_after_days: u32,
    /// Minimum importance score to not be considered noise.
    pub min_importance: f64,
    /// Similarity threshold for duplicate detection (0.0-1.0).
    pub duplicate_similarity: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            high_noise_threshold: 0.4,
            stale_after_days: 7,
            min_importance: 0.3,
            duplicate_similarity: 0.8,
        }
    }
}

/// Noise assessment result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseAssessment {
    /// Total tokens/chars analyzed.
    pub total_chars: usize,
    /// Characters identified as noise.
    pub noise_chars: usize,
    /// Overall noise ratio (0.0-1.0).
    pub noise_ratio: f64,
    /// Breakdown by noise source.
    pub breakdown: NoiseBreakdown,
    /// Suggestions for noise reduction.
    pub suggestions: Vec<String>,
}

/// Breakdown of noise by source type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseBreakdown {
    /// Duplicate content chars.
    pub duplicate_chars: usize,
    /// Stale content chars.
    pub stale_chars: usize,
    /// Low-importance content chars.
    pub low_importance_chars: usize,
    /// Tool output bloat chars.
    pub tool_bloat_chars: usize,
}

/// Noise assessor for working memory and conversation context.
pub struct NoiseAssessor {
    config: NoiseConfig,
}

impl NoiseAssessor {
    /// Create a new noise assessor with the given configuration.
    pub fn new(config: NoiseConfig) -> Self {
        Self { config }
    }

    /// Assess noise in a collection of working memory entries.
    pub fn assess_working_memory(&self, entries: &[WorkingMemoryEntry]) -> NoiseAssessment {
        if !self.config.enabled || entries.is_empty() {
            return NoiseAssessment {
                total_chars: 0,
                noise_chars: 0,
                noise_ratio: 0.0,
                breakdown: NoiseBreakdown {
                    duplicate_chars: 0,
                    stale_chars: 0,
                    low_importance_chars: 0,
                    tool_bloat_chars: 0,
                },
                suggestions: Vec::new(),
            };
        }

        let total_chars: usize = entries.iter().map(|e| e.char_count).sum();
        let mut breakdown = NoiseBreakdown {
            duplicate_chars: 0,
            stale_chars: 0,
            low_importance_chars: 0,
            tool_bloat_chars: 0,
        };
        let mut suggestions = Vec::new();

        // 1. Detect duplicates using simple text similarity
        let duplicate_chars = self.detect_duplicates(entries);
        breakdown.duplicate_chars = duplicate_chars;

        // 2. Detect stale content
        let stale_chars = self.detect_stale(entries);
        breakdown.stale_chars = stale_chars;

        // 3. Detect low-importance content
        let low_importance_chars = self.detect_low_importance(entries);
        breakdown.low_importance_chars = low_importance_chars;

        // 4. Detect tool output bloat
        let tool_bloat_chars = self.detect_tool_bloat(entries);
        breakdown.tool_bloat_chars = tool_bloat_chars;

        // Calculate total noise (avoid double-counting)
        let noise_chars = (duplicate_chars + stale_chars + low_importance_chars + tool_bloat_chars)
            .min(total_chars);
        let noise_ratio = if total_chars > 0 {
            noise_chars as f64 / total_chars as f64
        } else {
            0.0
        };

        // Generate suggestions
        if noise_ratio > self.config.high_noise_threshold {
            suggestions.push(format!(
                "当前上下文噪音占比 {:.1}%，超过阈值 {:.1}%",
                noise_ratio * 100.0,
                self.config.high_noise_threshold * 100.0
            ));
        }

        if duplicate_chars > 0 {
            suggestions.push(format!(
                "检测到 {} 字符的重复内容，建议去重",
                duplicate_chars
            ));
        }

        if stale_chars > 0 {
            suggestions.push(format!(
                "检测到 {} 字符的过期内容（超过 {} 天），建议清理",
                stale_chars, self.config.stale_after_days
            ));
        }

        if low_importance_chars > 0 {
            suggestions.push(format!(
                "检测到 {} 字符的低重要性内容，考虑移除",
                low_importance_chars
            ));
        }

        if tool_bloat_chars > 0 {
            suggestions.push(format!(
                "检测到 {} 字符的冗长工具输出，考虑精简",
                tool_bloat_chars
            ));
        }

        NoiseAssessment {
            total_chars,
            noise_chars,
            noise_ratio,
            breakdown,
            suggestions,
        }
    }

    /// Detect duplicate content using simple n-gram similarity.
    fn detect_duplicates(&self, entries: &[WorkingMemoryEntry]) -> usize {
        let mut seen_texts: HashMap<String, usize> = HashMap::new();
        let mut duplicate_chars = 0;

        for entry in entries {
            let text = entry
                .message
                .text_content()
                .unwrap_or_default()
                .to_lowercase();

            // Use first 200 chars as fingerprint
            let fingerprint: String = text.chars().take(200).collect();

            let count = seen_texts.entry(fingerprint).or_insert(0);
            *count += 1;

            // If we've seen this content before, count as duplicate
            if *count > 1 {
                duplicate_chars += entry.char_count;
            }
        }

        duplicate_chars
    }

    /// Detect stale content based on importance and access patterns.
    fn detect_stale(&self, entries: &[WorkingMemoryEntry]) -> usize {
        let mut stale_chars = 0;

        for entry in entries {
            // Low importance + not consolidation candidate = potentially stale
            if entry.importance < self.config.min_importance && !entry.is_consolidation_candidate {
                stale_chars += entry.char_count;
            }
        }

        stale_chars
    }

    /// Detect low-importance content.
    fn detect_low_importance(&self, entries: &[WorkingMemoryEntry]) -> usize {
        entries
            .iter()
            .filter(|e| e.importance < self.config.min_importance)
            .map(|e| e.char_count)
            .sum()
    }

    /// Detect verbose tool outputs.
    fn detect_tool_bloat(&self, entries: &[WorkingMemoryEntry]) -> usize {
        let mut tool_bloat = 0;

        for entry in entries {
            let text = entry.message.text_content().unwrap_or_default();

            // Tool outputs are typically verbose if they:
            // 1. Start with common tool prefixes
            // 2. Are very long (>500 chars)
            // 3. Have low importance
            let is_likely_tool_output = text.starts_with("```")
                || text.starts_with("File:")
                || text.starts_with("Directory:")
                || text.starts_with("Error:")
                || text.starts_with("Result:");

            if is_likely_tool_output && text.len() > 500 && entry.importance < 0.5 {
                tool_bloat += entry.char_count;
            }
        }

        tool_bloat
    }

    /// Generate a noise report string for prompt injection.
    pub fn format_noise_report(&self, assessment: &NoiseAssessment) -> String {
        if !self.config.enabled || assessment.total_chars == 0 {
            return String::new();
        }

        let mut report = String::new();
        report.push_str("## 上下文噪音评估\n\n");
        report.push_str(&format!("- 总字符数: {}\n", assessment.total_chars));
        report.push_str(&format!(
            "- 噪音字符数: {} ({:.1}%)\n",
            assessment.noise_chars,
            assessment.noise_ratio * 100.0
        ));
        report.push_str(&format!(
            "- 重复内容: {}\n",
            assessment.breakdown.duplicate_chars
        ));
        report.push_str(&format!(
            "- 过期内容: {}\n",
            assessment.breakdown.stale_chars
        ));
        report.push_str(&format!(
            "- 低重要性: {}\n",
            assessment.breakdown.low_importance_chars
        ));
        report.push_str(&format!(
            "- 工具输出冗余: {}\n",
            assessment.breakdown.tool_bloat_chars
        ));

        if !assessment.suggestions.is_empty() {
            report.push_str("\n### 建议\n");
            for suggestion in &assessment.suggestions {
                report.push_str(&format!("- {}\n", suggestion));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatMessage;
    use crate::working_memory::WorkingMemoryConfig;

    fn create_test_entry(text: &str, importance: f64) -> WorkingMemoryEntry {
        let message = ChatMessage::text("user", text);
        let mut entry = WorkingMemoryEntry::new(message, &WorkingMemoryConfig::default());
        entry.importance = importance;
        entry
    }

    #[test]
    fn test_empty_assessment() {
        let assessor = NoiseAssessor::new(NoiseConfig::default());
        let assessment = assessor.assess_working_memory(&[]);
        assert_eq!(assessment.noise_ratio, 0.0);
        assert!(assessment.suggestions.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let assessor = NoiseAssessor::new(NoiseConfig::default());
        let entries = vec![
            create_test_entry("This is a test message", 0.5),
            create_test_entry("This is a test message", 0.5),
            create_test_entry("Different content", 0.5),
        ];
        let assessment = assessor.assess_working_memory(&entries);
        assert!(assessment.breakdown.duplicate_chars > 0);
    }

    #[test]
    fn test_low_importance_detection() {
        let config = NoiseConfig {
            min_importance: 0.3,
            ..Default::default()
        };
        let assessor = NoiseAssessor::new(config);
        let entries = vec![
            create_test_entry("Important message", 0.8),
            create_test_entry("Not important", 0.1),
        ];
        let assessment = assessor.assess_working_memory(&entries);
        assert!(assessment.breakdown.low_importance_chars > 0);
    }

    #[test]
    fn test_noise_report_format() {
        let assessor = NoiseAssessor::new(NoiseConfig::default());
        let assessment = NoiseAssessment {
            total_chars: 1000,
            noise_chars: 300,
            noise_ratio: 0.3,
            breakdown: NoiseBreakdown {
                duplicate_chars: 100,
                stale_chars: 100,
                low_importance_chars: 50,
                tool_bloat_chars: 50,
            },
            suggestions: vec!["Test suggestion".to_string()],
        };
        let report = assessor.format_noise_report(&assessment);
        assert!(report.contains("30.0%"));
        assert!(report.contains("Test suggestion"));
    }
}
