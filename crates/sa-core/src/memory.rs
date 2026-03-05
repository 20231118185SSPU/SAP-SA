//! Markdown-based memory loading and retrieval for StudyAdministrator (SA).
//!
//! The user asked that memory should work more like OpenClaw:
//! - Stable memory files live in the workspace (`MEMORY.md` / `memory.md`).
//! - Daily memory lives under `memory/*.md`.
//! - The model should not receive a synthetic JSONL summary blob each turn.
//! - Instead, it should use dedicated tools to search and read memory on demand.
//!
//! This module therefore provides three pieces:
//! 1. A small prompt block that injects only root memory files (`MEMORY.md`
//!    and/or `memory.md`) when they exist.
//! 2. `MemorySearch`: lexical search over `MEMORY.md`, `memory.md`, and
//!    `memory/**/*.md`.
//! 3. `MemoryGet`: bounded file/line reads restricted to those same memory
//!    files.
//!
//! Important design notes:
//! - We intentionally keep the search implementation simple and inspectable.
//!   It is not embedding-based semantic search.
//! - Security matters more than convenience:
//!   - only files inside the workspace are allowed
//!   - only `MEMORY.md`, `memory.md`, and `memory/**/*.md` are readable
//!   - path traversal and symlink escapes are rejected

use anyhow::Context as _;
use serde::Serialize;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

/// Primary long-term memory filename used by OpenClaw-style workspaces.
pub const PRIMARY_MEMORY_FILE: &str = "MEMORY.md";

/// Alternate lowercase memory filename accepted for compatibility.
pub const ALT_MEMORY_FILE: &str = "memory.md";

/// Directory that stores daily / rolling Markdown memory notes.
pub const MEMORY_DIR: &str = "memory";

/// Per-file prompt injection cap for root memory files.
const MAX_PROMPT_MEMORY_CHARS_PER_FILE: usize = 20_000;

/// Total prompt injection cap across all root memory files.
const MAX_PROMPT_MEMORY_TOTAL_CHARS: usize = 40_000;

/// Default maximum number of search hits returned by `MemorySearch`.
const DEFAULT_MEMORY_SEARCH_RESULTS: usize = 5;

/// Hard maximum number of search hits returned by `MemorySearch`.
const MAX_MEMORY_SEARCH_RESULTS: usize = 10;

/// Default number of lines returned by `MemoryGet` when a caller requests a
/// starting line but omits an explicit line count.
const DEFAULT_MEMORY_GET_LINES: usize = 40;

/// Hard cap for `MemoryGet`.
const MAX_MEMORY_GET_LINES: usize = 200;

/// One memory search hit returned to the model.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySearchResult {
    /// Workspace-relative path.
    pub path: String,
    /// 1-based starting line number.
    pub start_line: usize,
    /// 1-based ending line number.
    pub end_line: usize,
    /// Best-effort lexical relevance score.
    pub score: f64,
    /// Snippet text shown to the model.
    pub snippet: String,
}

/// Bounded file read returned by `MemoryGet`.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryReadResult {
    /// Workspace-relative path.
    pub path: String,
    /// 1-based starting line actually returned.
    pub from_line: usize,
    /// 1-based ending line actually returned.
    pub to_line: usize,
    /// Text content for the selected line range.
    pub text: String,
}

/// Build the prompt block that injects root memory files, mirroring the
/// OpenClaw idea that stable curated memory may be included in prompt context.
///
/// Daily memory files under `memory/` are intentionally excluded here so they
/// remain on-demand via tools.
pub async fn build_prompt_block(workspace_root: &Path) -> anyhow::Result<String> {
    let workspace_root = canonical_workspace_root(workspace_root)?;
    let mut remaining = MAX_PROMPT_MEMORY_TOTAL_CHARS;
    let mut loaded = 0usize;
    let mut out = String::from("## Memory Context\n\n");

    for candidate in candidate_prompt_memory_paths(&workspace_root) {
        if remaining == 0 {
            break;
        }
        let Ok(meta) = tokio::fs::metadata(&candidate).await else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }

        let raw = tokio::fs::read_to_string(&candidate)
            .await
            .with_context(|| {
                format!("Failed to read memory prompt file: {}", candidate.display())
            })?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        let capped = trim_chars(trimmed, MAX_PROMPT_MEMORY_CHARS_PER_FILE.min(remaining));
        if capped.is_empty() {
            break;
        }

        loaded += 1;
        remaining = remaining.saturating_sub(capped.chars().count());
        let display = workspace_relative_display(&workspace_root, &candidate);
        out.push_str(&format!("### `{display}`\n\n```text\n{capped}\n```\n\n"));
    }

    if loaded == 0 {
        out.push_str("- (no `MEMORY.md` / `memory.md` was loaded)\n");
    }

    Ok(out)
}

/// Return `true` when a path points at one of the dedicated memory files that
/// should be handled by `MemorySearch` / `MemoryGet` instead of generic
/// `Agents.md` reference preloading.
pub fn is_memory_reference(raw: &str) -> bool {
    let normalized = normalize_rel_path(raw);
    let lower = normalized.to_ascii_lowercase();

    lower == "memory.md" || (lower.starts_with("memory/") && lower.ends_with(".md"))
}

/// Search workspace memory files and return the most relevant snippets.
pub async fn search_markdown_memory(
    workspace_root: &Path,
    query: &str,
    max_results: Option<usize>,
    min_score: Option<f64>,
) -> anyhow::Result<Vec<MemorySearchResult>> {
    let workspace_root = canonical_workspace_root(workspace_root)?;
    let query = query.trim();
    if query.is_empty() {
        anyhow::bail!("MemorySearch query must not be empty");
    }

    let max_results = max_results
        .unwrap_or(DEFAULT_MEMORY_SEARCH_RESULTS)
        .clamp(1, MAX_MEMORY_SEARCH_RESULTS);
    let min_score = min_score.unwrap_or(0.1_f64).max(0.0);
    let query_lower = query.to_ascii_lowercase();
    let tokens = tokenize(query);

    let mut hits = Vec::<MemorySearchResult>::new();
    for file_path in indexed_memory_files(&workspace_root)? {
        let display = workspace_relative_display(&workspace_root, &file_path);
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .with_context(|| format!("Failed to read memory file: {}", file_path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            continue;
        }

        for (start, end) in line_windows(lines.len()) {
            let snippet = lines[start..end].join("\n");
            let score = score_snippet(&snippet, &query_lower, &tokens);
            if score < min_score {
                continue;
            }

            hits.push(MemorySearchResult {
                path: display.clone(),
                start_line: start + 1,
                end_line: end,
                score,
                snippet: trim_chars(snippet.trim(), 700),
            });
        }
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });
    hits.dedup_by(|a, b| a.path == b.path && a.start_line == b.start_line);
    hits.truncate(max_results);

    Ok(hits)
}

/// Read a bounded line range from one allowed memory file.
pub async fn read_markdown_memory(
    workspace_root: &Path,
    rel_path: &str,
    from: Option<usize>,
    lines: Option<usize>,
) -> anyhow::Result<MemoryReadResult> {
    let workspace_root = canonical_workspace_root(workspace_root)?;
    let path = resolve_allowed_memory_path(&workspace_root, rel_path)?;
    let display = workspace_relative_display(&workspace_root, &path);

    let raw = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("Failed to read memory file: {}", path.display()))?;
    let file_lines: Vec<&str> = raw.lines().collect();

    if file_lines.is_empty() {
        return Ok(MemoryReadResult {
            path: display,
            from_line: 1,
            to_line: 0,
            text: String::new(),
        });
    }

    let start = from.unwrap_or(1).max(1);
    let count = lines
        .unwrap_or_else(|| {
            if from.is_some() {
                DEFAULT_MEMORY_GET_LINES
            } else {
                file_lines.len()
            }
        })
        .clamp(1, MAX_MEMORY_GET_LINES);

    let start_idx = start.saturating_sub(1).min(file_lines.len());
    let end_idx = start_idx.saturating_add(count).min(file_lines.len());
    let text = file_lines[start_idx..end_idx].join("\n");

    Ok(MemoryReadResult {
        path: display,
        from_line: start_idx + 1,
        to_line: end_idx,
        text,
    })
}

/// List the root memory files that may be injected into the prompt.
fn candidate_prompt_memory_paths(workspace_root: &Path) -> Vec<PathBuf> {
    vec![
        workspace_root.join(PRIMARY_MEMORY_FILE),
        workspace_root.join(ALT_MEMORY_FILE),
    ]
}

/// Return every Markdown memory file that `MemorySearch` is allowed to inspect.
fn indexed_memory_files(workspace_root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::<PathBuf>::new();

    for candidate in candidate_prompt_memory_paths(workspace_root) {
        if candidate.is_file() {
            files.push(std::fs::canonicalize(&candidate).with_context(|| {
                format!(
                    "Failed to canonicalize memory file: {}",
                    candidate.display()
                )
            })?);
        }
    }

    let memory_dir = workspace_root.join(MEMORY_DIR);
    if memory_dir.is_dir() {
        for entry in walkdir::WalkDir::new(&memory_dir).follow_links(false) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            if path
                .extension()
                .and_then(|value| value.to_str())
                .is_none_or(|value| !value.eq_ignore_ascii_case("md"))
            {
                continue;
            }

            let canonical = std::fs::canonicalize(path).with_context(|| {
                format!("Failed to canonicalize memory file: {}", path.display())
            })?;
            if !canonical.starts_with(workspace_root) {
                anyhow::bail!(
                    "Memory file escapes workspace root: {}",
                    canonical.display()
                );
            }

            files.push(canonical);
        }
    }

    files.sort();
    files.dedup();
    Ok(files)
}

/// Resolve a user-provided memory path and ensure it stays inside the
/// workspace's allowed memory locations.
fn resolve_allowed_memory_path(workspace_root: &Path, rel_path: &str) -> anyhow::Result<PathBuf> {
    let normalized = normalize_rel_path(rel_path);
    if !is_memory_reference(&normalized) {
        anyhow::bail!(
            "MemoryGet only allows `MEMORY.md`, `memory.md`, or `memory/*.md`: {}",
            rel_path
        );
    }

    let candidate = workspace_root.join(normalized);
    let canonical = std::fs::canonicalize(&candidate)
        .with_context(|| format!("Memory file not found: {}", candidate.display()))?;
    if !canonical.starts_with(workspace_root) {
        anyhow::bail!(
            "Memory path escapes workspace root: {}",
            canonical.display()
        );
    }
    if !canonical.is_file() {
        anyhow::bail!("MemoryGet requires a file path: {}", canonical.display());
    }

    Ok(canonical)
}

/// Canonicalize the workspace root once so later path checks are stable.
fn canonical_workspace_root(workspace_root: &Path) -> anyhow::Result<PathBuf> {
    std::fs::canonicalize(workspace_root).with_context(|| {
        format!(
            "Failed to canonicalize workspace root for memory operations: {}",
            workspace_root.display()
        )
    })
}

/// Produce a workspace-relative display path with forward slashes.
fn workspace_relative_display(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Normalize user-facing relative paths into a stable slash form.
fn normalize_rel_path(raw: &str) -> String {
    raw.trim()
        .trim_start_matches("./")
        .trim_start_matches(".\\")
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_string()
}

/// Split a search query into lowercase lexical tokens.
fn tokenize(query: &str) -> Vec<String> {
    let mut tokens: Vec<String> = query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect();

    if tokens.is_empty() {
        tokens.push(query.trim().to_ascii_lowercase());
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

/// Generate overlapping line windows so search hits stay compact.
fn line_windows(total_lines: usize) -> Vec<(usize, usize)> {
    if total_lines == 0 {
        return Vec::new();
    }

    let window = 8usize;
    let step = 4usize;
    let mut out = Vec::new();
    let mut start = 0usize;

    while start < total_lines {
        let end = start.saturating_add(window).min(total_lines);
        out.push((start, end));
        if end == total_lines {
            break;
        }
        start = start.saturating_add(step);
    }

    out
}

/// Score one snippet against the query using simple lexical overlap.
fn score_snippet(snippet: &str, query_lower: &str, tokens: &[String]) -> f64 {
    let lower = snippet.to_ascii_lowercase();
    let mut score = 0.0_f64;

    if lower.contains(query_lower) {
        score += 3.0;
    }

    for token in tokens {
        let matches = lower.matches(token).count();
        if matches > 0 {
            score += 1.0 + ((matches.saturating_sub(1)) as f64 * 0.25);
        }
    }

    score
}

/// Truncate a string by Unicode scalar count.
fn trim_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }

    let mut out: String = input.chars().take(max_chars).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;

    /// Create a unique temporary workspace for memory tests.
    fn unique_workspace() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sa-memory-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create workspace");
        dir
    }

    #[test]
    fn memory_reference_detection_matches_expected_paths() {
        assert!(is_memory_reference("MEMORY.md"));
        assert!(is_memory_reference("memory.md"));
        assert!(is_memory_reference("memory/2026-03-06.md"));
        assert!(!is_memory_reference("notes/memory.md"));
        assert!(!is_memory_reference("memory/2026-03-06.txt"));
    }

    #[tokio::test]
    async fn prompt_block_loads_root_memory_files_only() {
        let workspace = unique_workspace();
        fs::write(workspace.join("MEMORY.md"), "root memory").expect("write MEMORY.md");
        fs::create_dir_all(workspace.join("memory")).expect("create memory dir");
        fs::write(
            workspace.join("memory").join("2026-03-06.md"),
            "daily memory",
        )
        .expect("write daily memory");

        let block = build_prompt_block(&workspace).await.expect("prompt block");
        assert!(block.contains("MEMORY.md"));
        assert!(block.contains("root memory"));
        assert!(!block.contains("daily memory"));
    }

    #[tokio::test]
    async fn memory_search_reads_memory_md_and_daily_files() {
        let workspace = unique_workspace();
        fs::write(
            workspace.join("MEMORY.md"),
            "Preference: likes Rust and careful reviews.",
        )
        .expect("write MEMORY.md");
        fs::create_dir_all(workspace.join("memory")).expect("create memory dir");
        fs::write(
            workspace.join("memory").join("2026-03-06.md"),
            "2026-03-06\nDecision: use skill sandbox instead of exposing host paths.",
        )
        .expect("write daily memory");

        let hits = search_markdown_memory(&workspace, "skill sandbox host paths", Some(5), None)
            .await
            .expect("memory search");
        assert!(!hits.is_empty());
        assert!(hits.iter().any(|hit| hit.path == "memory/2026-03-06.md"));
    }

    #[tokio::test]
    async fn memory_get_rejects_outside_paths() {
        let workspace = unique_workspace();
        let err = read_markdown_memory(&workspace, "../secret.md", None, None)
            .await
            .expect_err("outside path must fail");
        assert!(err.to_string().contains("only allows"));
    }
}
