//! Memory Pointer System — MEMORY.md only stores pointers, actual content in sub-files.
//!
//! Design goals:
//! - **Pointer-based**: MEMORY.md contains markdown links to actual memory files
//! - **Lazy loading**: only load content when pointer is accessed
//! - **Hierarchical**: support nested pointer structures
//! - **Backward compatible**: still support direct content in MEMORY.md
//!
//! Pointer format:
//! ```markdown
//! # MEMORY.md (Pointer Layer)
//!
//! ## 核心记忆
//! - [人格定义](memory/core/persona.md)
//! - [用户偏好](memory/core/preferences.md)
//!
//! ## 长期记忆
//! - [技术决策](memory/long-term/decisions.md)
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Memory pointer entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPointer {
    /// Display label for the pointer.
    pub label: String,
    /// Relative path to the target file.
    pub target_path: PathBuf,
    /// Category/group for organization.
    pub category: String,
    /// Priority level (higher = more important).
    pub priority: u32,
    /// Whether this pointer is active.
    pub active: bool,
}

/// Memory pointer collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPointerCollection {
    /// All pointers in the collection.
    pub pointers: Vec<MemoryPointer>,
    /// Source file path.
    pub source_path: PathBuf,
}

impl MemoryPointerCollection {
    /// Create a new empty collection.
    pub fn new(source_path: PathBuf) -> Self {
        Self {
            pointers: Vec::new(),
            source_path,
        }
    }

    /// Parse MEMORY.md content and extract pointers.
    pub fn parse_from_content(content: &str, source_path: PathBuf) -> Self {
        let mut collection = Self::new(source_path);
        let mut current_category = String::from("uncategorized");

        for line in content.lines() {
            let trimmed = line.trim();

            // Detect category headers (## xxx)
            if trimmed.starts_with("## ") {
                current_category = trimmed[3..].trim().to_string();
                continue;
            }

            // Detect pointer links: - [label](path)
            if let Some(pointer) = Self::parse_pointer_line(trimmed, &current_category) {
                collection.pointers.push(pointer);
            }
        }

        collection
    }

    /// Parse a single line for a pointer link.
    fn parse_pointer_line(line: &str, category: &str) -> Option<MemoryPointer> {
        // Pattern: - [label](path)
        let line = line.trim();
        if !line.starts_with("- [") {
            return None;
        }

        let after_dash = &line[2..];
        let bracket_start = after_dash.find('[')?;
        let bracket_end = after_dash.find(']')?;
        let paren_start = after_dash.find('(')?;
        let paren_end = after_dash.find(')')?;

        if bracket_start != 0 || bracket_end <= bracket_start || paren_start != bracket_end + 1 {
            return None;
        }

        let label = after_dash[bracket_start + 1..bracket_end].trim().to_string();
        let path_str = after_dash[paren_start + 1..paren_end].trim();

        // Skip non-local links (http, mailto, etc.)
        if path_str.starts_with("http") || path_str.starts_with("mailto") {
            return None;
        }

        Some(MemoryPointer {
            label,
            target_path: PathBuf::from(path_str),
            category: category.to_string(),
            priority: Self::estimate_priority(category),
            active: true,
        })
    }

    /// Estimate priority based on category.
    fn estimate_priority(category: &str) -> u32 {
        match category.to_lowercase().as_str() {
            "核心记忆" | "core memory" | "persona" | "identity" => 100,
            "长期记忆" | "long-term memory" => 80,
            "短期记忆" | "short-term memory" => 60,
            "工作记忆" | "working memory" => 40,
            _ => 50,
        }
    }

    /// Get all pointers for a specific category.
    pub fn pointers_by_category(&self, category: &str) -> Vec<&MemoryPointer> {
        self.pointers
            .iter()
            .filter(|p| p.category == category && p.active)
            .collect()
    }

    /// Get all active pointers sorted by priority.
    pub fn active_pointers(&self) -> Vec<&MemoryPointer> {
        let mut pointers: Vec<&MemoryPointer> = self
            .pointers
            .iter()
            .filter(|p| p.active)
            .collect();
        pointers.sort_by(|a, b| b.priority.cmp(&a.priority));
        pointers
    }

    /// Resolve a pointer to its absolute path.
    pub fn resolve_path(&self, pointer: &MemoryPointer, workspace_root: &Path) -> PathBuf {
        let relative = &pointer.target_path;
        if relative.is_absolute() {
            relative.clone()
        } else {
            // Resolve relative to the source file's directory
            let source_dir = self
                .source_path
                .parent()
                .unwrap_or(workspace_root);
            source_dir.join(relative)
        }
    }

    /// Check if a pointer target exists.
    pub fn target_exists(&self, pointer: &MemoryPointer, workspace_root: &Path) -> bool {
        let path = self.resolve_path(pointer, workspace_root);
        path.exists()
    }

    /// Generate a summary of all pointers for prompt injection.
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        let mut current_category = String::new();

        for pointer in self.active_pointers() {
            if pointer.category != current_category {
                if !current_category.is_empty() {
                    summary.push('\n');
                }
                summary.push_str(&format!("## {}\n", pointer.category));
                current_category = pointer.category.clone();
            }
            summary.push_str(&format!("- [{}]\n", pointer.label));
        }

        summary
    }
}

/// Memory pointer manager for handling pointer-based memory access.
pub struct MemoryPointerManager {
    workspace_root: PathBuf,
}

impl MemoryPointerManager {
    /// Create a new pointer manager.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    /// Load and parse MEMORY.md for pointers.
    pub fn load_pointers(&self) -> anyhow::Result<MemoryPointerCollection> {
        let memory_path = self.workspace_root.join("MEMORY.md");
        if !memory_path.exists() {
            // Try alternative path
            let alt_path = self.workspace_root.join("memory.md");
            if !alt_path.exists() {
                return Ok(MemoryPointerCollection::new(memory_path));
            }
            return self.load_from_path(&alt_path);
        }
        self.load_from_path(&memory_path)
    }

    /// Load pointers from a specific file.
    fn load_from_path(&self, path: &Path) -> anyhow::Result<MemoryPointerCollection> {
        let content = std::fs::read_to_string(path)?;
        Ok(MemoryPointerCollection::parse_from_content(&content, path.to_path_buf()))
    }

    /// Resolve a pointer and read its content.
    pub fn resolve_and_read(&self, pointer: &MemoryPointer) -> anyhow::Result<String> {
        let path = self.resolve_pointer_path(pointer);
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "Pointer target not found: {}",
                path.display()
            ));
        }
        Ok(std::fs::read_to_string(path)?)
    }

    /// Resolve a pointer's absolute path.
    pub fn resolve_pointer_path(&self, pointer: &MemoryPointer) -> PathBuf {
        let relative = &pointer.target_path;
        if relative.is_absolute() {
            relative.clone()
        } else {
            self.workspace_root.join(relative)
        }
    }

    /// Check if MEMORY.md is pointer-based (contains markdown links).
    pub fn is_pointer_based(&self) -> bool {
        let memory_path = self.workspace_root.join("MEMORY.md");
        if !memory_path.exists() {
            return false;
        }

        if let Ok(content) = std::fs::read_to_string(&memory_path) {
            // Check for markdown link pattern
            content.contains("- [") && content.contains("](")
        } else {
            false
        }
    }

    /// Generate a pointer-based MEMORY.md template.
    pub fn generate_template(&self) -> String {
        r#"# MEMORY.md (Pointer Layer)

## 核心记忆
- [人格定义](memory/core/persona.md)
- [用户偏好](memory/core/preferences.md)
- [项目上下文](memory/core/project.md)

## 长期记忆
- [技术决策](memory/long-term/decisions.md)
- [架构演进](memory/long-term/architecture.md)
- [经验教训](memory/long-term/lessons.md)

## 短期记忆
- [当前任务](memory/short-term/current-task.md)
- [待解决](memory/short-term/pending.md)

## 工作记忆
- [会话笔记](memory/working/session-notes.md)
"#
        .to_string()
    }

    /// Create directory structure for pointer-based memory.
    pub fn create_structure(&self) -> anyhow::Result<()> {
        let dirs = [
            "memory/core",
            "memory/long-term",
            "memory/short-term",
            "memory/working",
        ];

        for dir in &dirs {
            let path = self.workspace_root.join(dir);
            std::fs::create_dir_all(path)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pointer_line() {
        let line = "- [人格定义](memory/core/persona.md)";
        let pointer = MemoryPointerCollection::parse_pointer_line(line, "核心记忆");
        assert!(pointer.is_some());
        let pointer = pointer.unwrap();
        assert_eq!(pointer.label, "人格定义");
        assert_eq!(pointer.target_path, PathBuf::from("memory/core/persona.md"));
        assert_eq!(pointer.category, "核心记忆");
    }

    #[test]
    fn test_parse_non_pointer_line() {
        let line = "This is just regular text";
        let pointer = MemoryPointerCollection::parse_pointer_line(line, "test");
        assert!(pointer.is_none());
    }

    #[test]
    fn test_parse_http_link() {
        let line = "- [External](http://example.com)";
        let pointer = MemoryPointerCollection::parse_pointer_line(line, "test");
        assert!(pointer.is_none());
    }

    #[test]
    fn test_parse_content() {
        let content = r#"# MEMORY.md

## 核心记忆
- [人格定义](memory/core/persona.md)
- [用户偏好](memory/core/preferences.md)

## 长期记忆
- [技术决策](memory/long-term/decisions.md)
"#;
        let collection = MemoryPointerCollection::parse_from_content(
            content,
            PathBuf::from("MEMORY.md"),
        );
        assert_eq!(collection.pointers.len(), 3);
        assert_eq!(collection.pointers_by_category("核心记忆").len(), 2);
        assert_eq!(collection.pointers_by_category("长期记忆").len(), 1);
    }

    #[test]
    fn test_priority_estimation() {
        assert_eq!(
            MemoryPointerCollection::estimate_priority("核心记忆"),
            100
        );
        assert_eq!(
            MemoryPointerCollection::estimate_priority("长期记忆"),
            80
        );
        assert_eq!(
            MemoryPointerCollection::estimate_priority("短期记忆"),
            60
        );
        assert_eq!(
            MemoryPointerCollection::estimate_priority("其他"),
            50
        );
    }

    #[test]
    fn test_active_pointers_sorted() {
        let content = r#"# MEMORY.md

## 短期记忆
- [当前任务](memory/short-term/current.md)

## 核心记忆
- [人格](memory/core/persona.md)
"#;
        let collection = MemoryPointerCollection::parse_from_content(
            content,
            PathBuf::from("MEMORY.md"),
        );
        let active = collection.active_pointers();
        assert_eq!(active.len(), 2);
        // Core memory should be first (higher priority)
        assert_eq!(active[0].label, "人格");
        assert_eq!(active[1].label, "当前任务");
    }
}
