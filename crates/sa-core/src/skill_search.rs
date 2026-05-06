//! Local skill search engine inspired by GenericAgent's environment-aware
//! skill retrieval. Uses simple keyword matching (similar to BM25 relevance)
//! to find skills matching a natural-language query.
//!
//! The index is built from the `CommandRegistry` (aliased as `SkillRegistry`)
//! and refreshed lazily.

use crate::commands::{CommandListItem, CommandRegistry};
use serde::Serialize;
use std::sync::Arc;

/// A lightweight inverted-index style skill database for keyword search.
#[derive(Debug, Clone)]
pub struct SkillIndex {
    entries: Vec<SkillIndexEntry>,
}

#[derive(Debug, Clone)]
struct SkillIndexEntry {
    name: String,
    description: String,
    when_to_use: Option<String>,
    /// Pre-computed lowercase tokens for fast matching.
    tokens: Vec<String>,
}

/// A single search hit returned by `SkillIndex::search`.
#[derive(Debug, Clone, Serialize)]
pub struct SkillSearchResult {
    /// Skill name.
    pub name: String,
    /// Skill description.
    pub description: String,
    /// Human-readable explanation of why this skill matched.
    pub match_reason: String,
}

impl SkillIndex {
    /// Build the index from a `CommandRegistry` snapshot.
    pub fn build(skills: &CommandRegistry) -> Self {
        let entries = skills
            .list()
            .into_iter()
            .map(|item| {
                let desc_lower = item.description.to_lowercase();
                let tokens = tokenize(&item.name)
                    .into_iter()
                    .chain(tokenize(&desc_lower))
                    .collect();
                SkillIndexEntry {
                    name: item.name,
                    description: item.description,
                    when_to_use: item.when_to_use,
                    tokens,
                }
            })
            .collect();
        Self { entries }
    }

    /// Build the index from an `Arc<CommandRegistry>`.
    pub fn from_arc(skills: &Arc<CommandRegistry>) -> Self {
        Self::build(skills.as_ref())
    }

    /// Search skills by query string. Returns up to `limit` results sorted by
    /// relevance (number of matching keyword tokens, descending).
    pub fn search(&self, query: &str, limit: usize) -> Vec<SkillSearchResult> {
        let query_tokens: Vec<String> = tokenize(&query.to_lowercase());
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, &SkillIndexEntry)> = self
            .entries
            .iter()
            .map(|entry| {
                let score = query_tokens
                    .iter()
                    .filter(|qt| entry.tokens.iter().any(|et| et.contains(qt.as_str())))
                    .count();
                (score, entry)
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(limit);

        scored
            .into_iter()
            .map(|(score, entry)| {
                let match_reason = if score >= query_tokens.len() {
                    format!("高度匹配查询「{}」", query)
                } else if score > 1 {
                    format!("匹配 {score} 个关键词")
                } else {
                    format!("部分匹配「{}」", query)
                };
                SkillSearchResult {
                    name: entry.name.clone(),
                    description: entry.description.clone(),
                    match_reason,
                }
            })
            .collect()
    }
}

/// Simple tokenizer: split on whitespace and common punctuation, lowercase,
/// and filter out empty/single-char tokens.
fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| c.is_whitespace() || c == '-' || c == '_' || c == '/' || c == ',')
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|s| s.len() > 1)
        .collect()
}

/// Collect basic environment information for periodic prompt injection.
pub fn detect_environment_info() -> String {
    let os = std::env::consts::OS;
    let family = std::env::consts::FAMILY;
    let shell = std::env::var("SHELL")
        .or_else(|_| std::env::var("COMSPEC"))
        .unwrap_or_else(|_| "unknown".to_string());

    // Detect available runtimes.
    let mut runtimes = Vec::new();
    for (name, cmd) in [
        ("Node.js", "node"),
        ("Python", "python3"),
        ("Python (alt)", "python"),
        ("Ruby", "ruby"),
        ("Go", "go"),
    ] {
        if which::which(cmd).is_ok() {
            runtimes.push(name.to_string());
        }
    }

    format!(
        "OS: {os} ({family})\nShell: {shell}\n可用运行时: {}",
        if runtimes.is_empty() {
            "未检测到".to_string()
        } else {
            runtimes.join(", ")
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello-World, foo_bar/baz");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
        assert!(tokens.contains(&"baz".to_string()));
    }

    #[test]
    fn test_empty_skill_index() {
        let registry = CommandRegistry::default();
        let index = SkillIndex::build(&registry);
        let results = index.search("anything", 5);
        assert!(results.is_empty());
    }
}
