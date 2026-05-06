//! Timeline Retrieval — time-based memory search and filtering.
//!
//! Design goals:
//! - **Time-based search**: find memories by date range
//! - **Temporal context**: understand when events happened
//! - **Timeline visualization**: generate timeline summaries
//! - **Integration**: work with existing memory search systems
//!
//! Timeline features:
//! 1. Date-range queries
//! 2. Relative time queries ("last week", "last month")
//! 3. Timeline generation for memory files
//! 4. Temporal relevance scoring

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Timeline retrieval configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TimelineConfig {
    /// Enable timeline retrieval.
    pub enabled: bool,
    /// Maximum number of results per query.
    pub max_results: usize,
    /// Default date format for parsing.
    pub date_format: String,
    /// Whether to include time-of-day in results.
    pub include_time: bool,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_results: 50,
            date_format: "%Y-%m-%d".to_string(),
            include_time: false,
        }
    }
}

/// Time range for queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeRange {
    /// Specific date range.
    Range {
        start: String,
        end: String,
    },
    /// Relative time (e.g., "last 7 days").
    Relative {
        days: u32,
    },
    /// Single date.
    Single {
        date: String,
    },
    /// Last N entries.
    Last {
        count: usize,
    },
}

/// Timeline entry with temporal metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Date string (YYYY-MM-DD).
    pub date: String,
    /// Optional time string (HH:MM:SS).
    pub time: Option<String>,
    /// File path.
    pub path: PathBuf,
    /// Content preview.
    pub preview: String,
    /// Tags associated with this entry.
    pub tags: Vec<String>,
    /// Importance score.
    pub importance: f64,
}

/// Timeline query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineResult {
    /// Matching entries.
    pub entries: Vec<TimelineEntry>,
    /// Total entries found.
    pub total: usize,
    /// Date range covered.
    pub date_range: Option<(String, String)>,
    /// Query execution time in milliseconds.
    pub query_time_ms: u64,
}

/// Timeline retrieval engine.
pub struct TimelineRetrieval {
    config: TimelineConfig,
    workspace_root: PathBuf,
}

impl TimelineRetrieval {
    /// Create a new timeline retrieval engine.
    pub fn new(config: TimelineConfig, workspace_root: PathBuf) -> Self {
        Self {
            config,
            workspace_root,
        }
    }

    /// Search memories by time range.
    pub fn search_by_range(&self, range: &TimeRange) -> TimelineResult {
        let start_time = std::time::Instant::now();

        if !self.config.enabled {
            return TimelineResult {
                entries: Vec::new(),
                total: 0,
                date_range: None,
                query_time_ms: 0,
            };
        }

        let mut entries = Vec::new();
        let memory_dir = self.workspace_root.join("memory");

        // Scan memory directory for date-based files
        if memory_dir.exists() {
            self.scan_directory(&memory_dir, &mut entries);
        }

        // Scan topics directory
        let topics_dir = memory_dir.join("topics");
        if topics_dir.exists() {
            self.scan_directory(&topics_dir, &mut entries);
        }

        // Filter by time range
        let filtered = self.filter_by_range(&entries, range);

        // Sort by date (newest first)
        let mut sorted = filtered;
        sorted.sort_by(|a, b| b.date.cmp(&a.date));

        // Limit results
        sorted.truncate(self.config.max_results);

        let date_range = if !sorted.is_empty() {
            let min_date = sorted.iter().map(|e| e.date.as_str()).min().unwrap_or("");
            let max_date = sorted.iter().map(|e| e.date.as_str()).max().unwrap_or("");
            Some((min_date.to_string(), max_date.to_string()))
        } else {
            None
        };

        let total = sorted.len();

        TimelineResult {
            entries: sorted,
            total,
            date_range,
            query_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }

    /// Scan a directory for timeline entries.
    fn scan_directory(&self, dir: &Path, entries: &mut Vec<TimelineEntry>) {
        if let Ok(read_dir) = std::fs::read_dir(dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "md" {
                            if let Some(timeline_entry) = self.parse_file_entry(&path) {
                                entries.push(timeline_entry);
                            }
                        }
                    }
                } else if path.is_dir() {
                    // Recursively scan subdirectories
                    self.scan_directory(&path, entries);
                }
            }
        }
    }

    /// Parse a file into a timeline entry.
    fn parse_file_entry(&self, path: &Path) -> Option<TimelineEntry> {
        let filename = path.file_stem()?.to_str()?;
        let content = std::fs::read_to_string(path).ok()?;

        // Try to extract date from filename (YYYY-MM-DD.md)
        let date = if filename.len() >= 10 && filename.chars().nth(4) == Some('-') {
            filename[..10].to_string()
        } else {
            // Try to extract from content (YAML front matter)
            self.extract_date_from_content(&content)?
        };

        // Extract preview (first non-empty line)
        let preview = content
            .lines()
            .find(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .unwrap_or("")
            .chars()
            .take(200)
            .collect::<String>();

        // Extract tags from YAML front matter
        let tags = self.extract_tags_from_content(&content);

        // Calculate importance based on content
        let importance = self.calculate_importance(&content);

        Some(TimelineEntry {
            date,
            time: None,
            path: path.to_path_buf(),
            preview,
            tags,
            importance,
        })
    }

    /// Extract date from YAML front matter.
    fn extract_date_from_content(&self, content: &str) -> Option<String> {
        let mut in_front_matter = false;
        for line in content.lines() {
            if line.trim() == "---" {
                in_front_matter = !in_front_matter;
                continue;
            }
            if in_front_matter && line.starts_with("date:") {
                let date_str = line[5..].trim().trim_matches('"').trim_matches('\'');
                return Some(date_str.to_string());
            }
        }
        None
    }

    /// Extract tags from YAML front matter.
    fn extract_tags_from_content(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();
        let mut in_front_matter = false;
        let mut in_tags = false;

        for line in content.lines() {
            if line.trim() == "---" {
                in_front_matter = !in_front_matter;
                in_tags = false;
                continue;
            }
            if in_front_matter {
                if line.starts_with("tags:") {
                    in_tags = true;
                    // Check if tags are on the same line
                    let tag_str = line[5..].trim();
                    if tag_str.starts_with('[') {
                        // Inline tags: tags: [tag1, tag2]
                        for tag in tag_str.trim_matches('[').trim_matches(']').split(',') {
                            let tag = tag.trim().trim_matches('"').trim_matches('\'');
                            if !tag.is_empty() {
                                tags.push(tag.to_string());
                            }
                        }
                        in_tags = false;
                    }
                    continue;
                }
                if in_tags {
                    let trimmed = line.trim();
                    if trimmed.starts_with("- ") {
                        let tag = trimmed[2..].trim().trim_matches('"').trim_matches('\'');
                        tags.push(tag.to_string());
                    } else {
                        in_tags = false;
                    }
                }
            }
        }

        tags
    }

    /// Calculate importance based on content.
    fn calculate_importance(&self, content: &str) -> f64 {
        let mut importance = 0.5;

        // More content = more important
        let word_count = content.split_whitespace().count();
        if word_count > 100 {
            importance += 0.1;
        }
        if word_count > 500 {
            importance += 0.1;
        }

        // Headers indicate structure
        let header_count = content.lines().filter(|l| l.starts_with('#')).count();
        importance += (header_count as f64 * 0.02).min(0.2);

        // Lists indicate action items
        let list_count = content.lines().filter(|l| l.trim().starts_with("- ")).count();
        importance += (list_count as f64 * 0.01).min(0.1);

        importance.min(1.0)
    }

    /// Filter entries by time range.
    fn filter_by_range(&self, entries: &[TimelineEntry], range: &TimeRange) -> Vec<TimelineEntry> {
        match range {
            TimeRange::Range { start, end } => entries
                .iter()
                .filter(|e| e.date >= *start && e.date <= *end)
                .cloned()
                .collect(),
            TimeRange::Relative { days } => {
                let cutoff = chrono::Utc::now() - chrono::Duration::days(*days as i64);
                let cutoff_str = cutoff.format("%Y-%m-%d").to_string();
                entries
                    .iter()
                    .filter(|e| e.date >= cutoff_str)
                    .cloned()
                    .collect()
            }
            TimeRange::Single { date } => entries
                .iter()
                .filter(|e| e.date == *date)
                .cloned()
                .collect(),
            TimeRange::Last { count } => {
                let mut sorted = entries.to_vec();
                sorted.sort_by(|a, b| b.date.cmp(&a.date));
                sorted.truncate(*count);
                sorted
            }
        }
    }

    /// Generate a timeline summary.
    pub fn generate_timeline_summary(&self, result: &TimelineResult) -> String {
        let mut summary = String::new();
        summary.push_str("## Timeline Summary\n\n");

        if result.entries.is_empty() {
            summary.push_str("No entries found for the specified time range.\n");
            return summary;
        }

        summary.push_str(&format!("Found {} entries", result.total));
        if let Some((start, end)) = &result.date_range {
            summary.push_str(&format!(" from {} to {}", start, end));
        }
        summary.push_str(".\n\n");

        // Group by date
        let mut by_date: BTreeMap<String, Vec<&TimelineEntry>> = BTreeMap::new();
        for entry in &result.entries {
            by_date.entry(entry.date.clone()).or_default().push(entry);
        }

        for (date, entries) in by_date.iter().rev() {
            summary.push_str(&format!("### {}\n", date));
            for entry in entries {
                summary.push_str(&format!("- {}\n", entry.preview));
                if !entry.tags.is_empty() {
                    summary.push_str(&format!("  Tags: {}\n", entry.tags.join(", ")));
                }
            }
            summary.push('\n');
        }

        summary
    }

    /// Find related memories by temporal proximity.
    pub fn find_related_by_time(
        &self,
        target_date: &str,
        entries: &[TimelineEntry],
        max_results: usize,
    ) -> Vec<TimelineEntry> {
        let mut scored: Vec<(f64, &TimelineEntry)> = entries
            .iter()
            .map(|entry| {
                let days_diff = self.date_difference_days(target_date, &entry.date);
                let temporal_score = 1.0 / (1.0 + days_diff as f64);
                let combined_score = temporal_score * 0.7 + entry.importance * 0.3;
                (combined_score, entry)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(max_results)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Calculate difference in days between two dates.
    fn date_difference_days(&self, date1: &str, date2: &str) -> i64 {
        if let (Ok(d1), Ok(d2)) = (
            chrono::NaiveDate::parse_from_str(date1, "%Y-%m-%d"),
            chrono::NaiveDate::parse_from_str(date2, "%Y-%m-%d"),
        ) {
            (d1 - d2).num_days().abs()
        } else {
            i64::MAX
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_relative() {
        let range = TimeRange::Relative { days: 7 };
        if let TimeRange::Relative { days } = range {
            assert_eq!(days, 7);
        }
    }

    #[test]
    fn test_timeline_entry_creation() {
        let entry = TimelineEntry {
            date: "2024-01-15".to_string(),
            time: Some("14:30:00".to_string()),
            path: PathBuf::from("memory/2024-01-15.md"),
            preview: "Test preview".to_string(),
            tags: vec!["test".to_string()],
            importance: 0.8,
        };
        assert_eq!(entry.date, "2024-01-15");
        assert_eq!(entry.tags.len(), 1);
    }

    #[test]
    fn test_extract_tags() {
        let content = r#"---
title: Test
tags:
  - tag1
  - tag2
---
Content here
"#;
        let config = TimelineConfig::default();
        let retrieval = TimelineRetrieval::new(config, PathBuf::from("."));
        let tags = retrieval.extract_tags_from_content(content);
        assert_eq!(tags, vec!["tag1", "tag2"]);
    }

    #[test]
    fn test_calculate_importance() {
        let config = TimelineConfig::default();
        let retrieval = TimelineRetrieval::new(config, PathBuf::from("."));

        let short_content = "Short content";
        let long_content = "Word ".repeat(200);
        let structured_content = "# Header\n- Item 1\n- Item 2\n- Item 3";

        assert!(retrieval.calculate_importance(&long_content) > retrieval.calculate_importance(short_content));
        assert!(retrieval.calculate_importance(structured_content) > 0.5);
    }
}
