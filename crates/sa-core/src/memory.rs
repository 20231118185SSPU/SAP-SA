//! Markdown-based memory loading and retrieval for StudyAdministrator (SA).
//!
//! The user asked that memory should work more like OpenClaw:
//! - Stable memory files live in the workspace (`MEMORY.md` / `memory.md`).
//! - Daily memory lives under `memory/*.md`.
//! - Topic memory lives under `memory/topics/**/*.md`.
//! - Dream audit files may exist under `memory/dreams/**/*.md`, but they should
//!   not pollute ordinary memory search results.
//! - The model should not receive a synthetic JSONL summary blob each turn.
//! - Instead, it should use dedicated tools to search and read memory on demand.
//!
//! This module therefore provides three pieces:
//! 1. A small prompt block that injects only root memory files (`MEMORY.md`
//!    and/or `memory.md`) when they exist.
//! 2. `MemorySearch`: lexical search over `MEMORY.md`, `memory.md`,
//!    `memory/*.md`, and `memory/topics/**/*.md`.
//! 3. `MemoryGet`: bounded file/line reads restricted to those same memory
//!    files, plus optional audit reads under `memory/dreams/**/*.md`.
//!
//! Important design notes:
//! - We intentionally keep the search implementation simple and inspectable.
//!   It is not embedding-based semantic search.
//! - Security matters more than convenience:
//!   - only files inside the workspace are allowed
//!   - only `MEMORY.md`, `memory.md`, `memory/*.md`, `memory/topics/**/*.md`,
//!     and `memory/dreams/**/*.md` are readable
//!   - path traversal and symlink escapes are rejected

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use chrono;
use std::cmp::Ordering;
use std::io::Write;
use std::path::{Path, PathBuf};
use crate::cache::MemoryCache;
use crate::memory_scope::MemoryScope;
use crate::memory_filter::WriteFilterResult;

/// Primary long-term memory filename used by OpenClaw-style workspaces.
pub const PRIMARY_MEMORY_FILE: &str = "MEMORY.md";

/// Alternate lowercase memory filename accepted for compatibility.
pub const ALT_MEMORY_FILE: &str = "memory.md";

/// Directory that stores daily / rolling Markdown memory notes.
pub const MEMORY_DIR: &str = "memory";

/// Directory prefix that stores curated topic memories.
pub const MEMORY_TOPICS_DIR: &str = "memory/topics";

/// Directory prefix that stores dream audit records.
pub const MEMORY_DREAMS_DIR: &str = "memory/dreams";

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
    /// Recall paths that contributed to this result.
    #[serde(default)]
    pub contributing_paths: Vec<String>,
    /// Optional embedding vector for MMR reranking.
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
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

/// Cached memory searcher that wraps `search_markdown_memory` with a moka cache.
pub struct MemorySearcher {
    cache: MemoryCache,
}

impl MemorySearcher {
    pub fn new() -> Self {
        Self {
            cache: MemoryCache::new(),
        }
    }

    pub async fn search_with_cache(
        &self,
        workspace_root: &Path,
        query: &str,
        max_results: Option<usize>,
        min_score: Option<f64>,
        weights: Option<[f64; 3]>,
        filter_tags: Option<&[String]>,
    ) -> anyhow::Result<Vec<MemorySearchResult>> {
        if let Some(cached) = self.cache.get_search_result(query) {
            return Ok(cached);
        }

        let results = search_markdown_memory(
            workspace_root,
            query,
            max_results,
            min_score,
            weights,
            filter_tags,
        )
        .await?;

        self.cache
            .cache_search_result(query.to_string(), results.clone());

        Ok(results)
    }
}

impl Default for MemorySearcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the prompt block that injects root memory files, mirroring the
/// OpenClaw idea that stable curated memory may be included in prompt context.
///
/// Daily memory files under `memory/` are intentionally excluded here so they
/// remain on-demand via tools.
///
/// If MEMORY.md is pointer-based (contains markdown links), the function will:
/// 1. Include the pointer summary (directory structure)
/// 2. Resolve and include content from pointer targets
pub async fn build_prompt_block(workspace_root: &Path) -> anyhow::Result<String> {
    let workspace_root = canonical_workspace_root(workspace_root)?;
    let mut remaining = MAX_PROMPT_MEMORY_TOTAL_CHARS;
    let mut loaded = 0usize;
    let mut out = String::from("## Memory Context\n\n");

    // Check if MEMORY.md is pointer-based
    let memory_path = workspace_root.join("MEMORY.md");
    let alt_path = workspace_root.join("memory.md");
    let main_path = if memory_path.exists() {
        memory_path
    } else if alt_path.exists() {
        alt_path
    } else {
        // No memory file found, return early
        out.push_str("- (no `MEMORY.md` / `memory.md` was loaded)\n");
        return Ok(out);
    };

    let raw = tokio::fs::read_to_string(&main_path)
        .await
        .with_context(|| {
            format!("Failed to read memory file: {}", main_path.display())
        })?;

    // Detect if this is pointer-based memory
    let is_pointer_based = raw.contains("- [") && raw.contains("](");

    if is_pointer_based {
        // Pointer-based memory: include summary and resolve pointers
        use crate::memory_pointer::MemoryPointerCollection;

        let collection = MemoryPointerCollection::parse_from_content(
            &raw,
            main_path.clone(),
        );

        // Include pointer summary
        let summary = collection.summary();
        if !summary.is_empty() {
            let capped = trim_chars(&summary, MAX_PROMPT_MEMORY_CHARS_PER_FILE.min(remaining));
            if !capped.is_empty() {
                loaded += 1;
                remaining = remaining.saturating_sub(capped.chars().count());
                out.push_str("### 记忆目录\n\n```text\n");
                out.push_str(&capped);
                out.push_str("\n```\n\n");
            }
        }

        // Resolve and include pointer targets
        for pointer in collection.active_pointers() {
            if remaining == 0 {
                break;
            }

            let target_path = collection.resolve_path(pointer, &workspace_root);
            if !target_path.exists() {
                continue;
            }

            let target_content = tokio::fs::read_to_string(&target_path)
                .await
                .with_context(|| {
                    format!("Failed to read memory pointer target: {}", target_path.display())
                })?;

            let trimmed = target_content.trim();
            if trimmed.is_empty() {
                continue;
            }

            let capped = trim_chars(trimmed, MAX_PROMPT_MEMORY_CHARS_PER_FILE.min(remaining));
            if capped.is_empty() {
                break;
            }

            loaded += 1;
            remaining = remaining.saturating_sub(capped.chars().count());
            let display = workspace_relative_display(&workspace_root, &target_path);
            out.push_str(&format!("### `{}` ({})\n\n```text\n{capped}\n```\n\n", pointer.label, display));
        }
    } else {
        // Traditional memory: include content directly
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let capped = trim_chars(trimmed, MAX_PROMPT_MEMORY_CHARS_PER_FILE.min(remaining));
            if !capped.is_empty() {
                loaded += 1;
                remaining = remaining.saturating_sub(capped.chars().count());
                let display = workspace_relative_display(&workspace_root, &main_path);
                out.push_str(&format!("### `{display}`\n\n```text\n{capped}\n```\n\n"));
            }
        }
    }

    // Also load other candidate memory files (excluding the main one)
    for candidate in candidate_prompt_memory_paths(&workspace_root) {
        if remaining == 0 {
            break;
        }

        // Skip the main memory file (already processed)
        if candidate == main_path {
            continue;
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

    lower == "memory.md"
        || is_daily_memory_path(&lower)
        || is_topics_memory_path(&lower)
        || is_dream_audit_path(&lower)
}

/// Build a set of line ranges (0-indexed) in `content` that correspond to entries
/// matching the given `filter_tags`. If `filter_tags` is None, returns None (no filter).
fn tagged_line_ranges(content: &str, filter_tags: Option<&[String]>) -> Option<Vec<(usize, usize)>> {
    let tags = filter_tags?;
    if tags.is_empty() {
        return None;
    }
    let tag_set: std::collections::HashSet<&str> = tags.iter().map(|s| s.as_str()).collect();
    let lines: Vec<&str> = content.lines().collect();
    let mut dash_pos: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() == "---" {
            dash_pos.push(i);
        }
    }
    let mut allowed = Vec::new();
    for window in dash_pos.windows(2) {
        let entry_start = window[0];
        let entry_end = window[1];
        let fm_text = lines[entry_start..entry_end].join("\n");
        let (meta, _, _) = parse_yaml_front_matter(&fm_text);
        let entry_tags = match &meta.tags {
            Some(t) => t,
            None => continue,
        };
        if entry_tags.iter().any(|t| tag_set.contains(t.as_str())) {
            allowed.push((entry_start, entry_end));
        }
    }
    if let Some(&last) = dash_pos.last() {
        let fm_text = lines[last..].join("\n");
        let (meta, _, _) = parse_yaml_front_matter(&fm_text);
        if let Some(ref entry_tags) = meta.tags {
            if entry_tags.iter().any(|t| tag_set.contains(t.as_str())) {
                allowed.push((last, lines.len()));
            }
        }
    }
    Some(allowed)
}

/// Search workspace memory files and return the most relevant snippets.
pub async fn search_markdown_memory(
    workspace_root: &Path,
    query: &str,
    max_results: Option<usize>,
    min_score: Option<f64>,
    _weights: Option<[f64; 3]>,
    filter_tags: Option<&[String]>,
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
                contributing_paths: vec!["Text".to_string()],
                embedding: None,
            });
        }

        // Post-filter by tags if requested
        if let Some(allowed_ranges) = tagged_line_ranges(&content, filter_tags) {
            hits.retain(|h| {
                let s = h.start_line.saturating_sub(1); // convert to 0-indexed
                let e = h.end_line;
                allowed_ranges.iter().any(|&(rs, re)| s < re && e > rs)
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

/// Hybrid search combining BM25 lexical search with vector similarity via Reciprocal Rank Fusion.
pub async fn search_markdown_memory_hybrid(
    workspace_root: &Path,
    query: &str,
    max_results: Option<usize>,
    min_score: Option<f64>,
    vector_hits: Vec<crate::vector_store::VectorHit>,
    weights: Option<[f64; 3]>,
    filter_tags: Option<&[String]>,
) -> anyhow::Result<Vec<MemorySearchResult>> {
    // Run lexical search first
    let lexical_hits = search_markdown_memory(workspace_root, query, max_results, min_score, weights, filter_tags).await?;

    // Convert vector hits to MemorySearchResult
    let vector_results: Vec<MemorySearchResult> = vector_hits
        .into_iter()
        .map(|vh| MemorySearchResult {
            path: vh.path.clone(),
            start_line: vh.start_line,
            end_line: vh.end_line,
            score: vh.score,
            snippet: trim_chars(&vh.snippet, 700),
            contributing_paths: vec!["Vector".to_string()],
            embedding: Some(vh.embedding),
        })
        .collect();

    // Reciprocal Rank Fusion: merge lexical and vector results
    let rrf_k = 60.0_f64;
    let mut merged: std::collections::HashMap<String, MemorySearchResult> = std::collections::HashMap::new();
    let mut merged_scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    // Add lexical results with RRF scoring
    for (rank, hit) in lexical_hits.iter().enumerate() {
        let key = format!("{}:{}", hit.path, hit.start_line);
        let rrf_score = 1.0 / (rrf_k + rank as f64 + 1.0);
        *merged_scores.entry(key.clone()).or_insert(0.0) += rrf_score * 0.5;
        merged.entry(key).or_insert_with(|| hit.clone());
    }

    // Add vector results with RRF scoring (skip when tag filter is active)
    if filter_tags.is_none() {
        for (rank, hit) in vector_results.iter().enumerate() {
            let key = format!("{}:{}", hit.path, hit.start_line);
            let rrf_score = 1.0 / (rrf_k + rank as f64 + 1.0);
            *merged_scores.entry(key.clone()).or_insert(0.0) += rrf_score * 0.5;
            let entry = merged.entry(key).or_insert_with(|| {
                let mut h = hit.clone();
                h.contributing_paths = vec!["Vector".to_string()];
                h
            });
            if !entry.contributing_paths.contains(&"Vector".to_string()) {
                entry.contributing_paths.push("Vector".to_string());
            }
        }
    }

    // Update scores with merged RRF scores
    let mut results: Vec<MemorySearchResult> = merged
        .into_iter()
        .map(|(key, mut result)| {
            result.score = *merged_scores.get(&key).unwrap_or(&0.0);
            result
        })
        .collect();

    // D9: Emotional and entity recall path boosting.
    // Detect query emotion and entities, boost matching results.
    let (_q_valence, q_arousal) = detect_emotion(query);
    let q_entities = extract_entities_simple(query);
    if q_arousal > 0.0 || !q_entities.is_empty() {
        let emotion_boost = 0.05;
        let entity_boost = 0.03;
        for result in &mut results {
            let snippet_lower = result.snippet.to_ascii_lowercase();
            // Emotional boost: if query has high arousal and snippet also has emotion keywords.
            if q_arousal > 0.0 {
                let (_, s_arousal) = detect_emotion(&result.snippet);
                if s_arousal > 0.0 {
                    result.score += emotion_boost * q_arousal.min(s_arousal);
                    if !result.contributing_paths.contains(&"Emotion".to_string()) {
                        result.contributing_paths.push("Emotion".to_string());
                    }
                }
            }
            // Entity boost: if snippet contains query entities.
            for entity in &q_entities {
                if snippet_lower.contains(&entity.to_ascii_lowercase()) {
                    result.score += entity_boost;
                    if !result.contributing_paths.contains(&"Entity".to_string()) {
                        result.contributing_paths.push("Entity".to_string());
                    }
                    break; // one boost per result
                }
            }
        }
    }

    let max_results = max_results
        .unwrap_or(DEFAULT_MEMORY_SEARCH_RESULTS)
        .clamp(1, MAX_MEMORY_SEARCH_RESULTS);

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
    });
    results.truncate(max_results);

    Ok(results)
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
pub(crate) fn indexed_memory_files(workspace_root: &Path) -> anyhow::Result<Vec<PathBuf>> {
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

            let display = workspace_relative_display(workspace_root, &canonical);
            let display_lower = display.to_ascii_lowercase();
            if !(is_daily_memory_path(&display_lower) || is_topics_memory_path(&display_lower)) {
                continue;
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
            "MemoryGet only allows `MEMORY.md`, `memory.md`, `memory/*.md`, `memory/topics/**/*.md`, or `memory/dreams/**/*.md`: {}",
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

/// Return `true` when the path points at a flat daily note under `memory/`.
fn is_daily_memory_path(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("memory/") else {
        return false;
    };
    rest.ends_with(".md") && !rest.contains('/')
}

/// Return `true` when the path points at a curated topic memory.
fn is_topics_memory_path(path: &str) -> bool {
    path.starts_with("memory/topics/") && path.ends_with(".md")
}

/// Return `true` when the path points at a dream audit file.
fn is_dream_audit_path(path: &str) -> bool {
    path.starts_with("memory/dreams/") && path.ends_with(".md")
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

// =============================================================================
// Memory Metadata & Data Structures
// =============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub title: Option<String>,
    pub tags: Option<Vec<String>>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub importance: Option<f64>,
    pub summary: Option<String>,
    pub emotion: Option<Emotion>,
    pub emotions: Option<Emotion>,
    pub related_notes: Option<Vec<String>>,
    pub entities: Option<Vec<String>>,
    pub access_count: Option<u32>,
    pub last_strengthen: Option<String>,
    pub scope: Option<MemoryScope>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Emotion {
    pub valence: Option<f64>,
    pub arousal: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactType {
    Static,
    Dynamic,
    Transient,
}

impl Default for FactType {
    fn default() -> Self { FactType::Dynamic }
}

impl std::fmt::Display for FactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactType::Static => write!(f, "static"),
            FactType::Dynamic => write!(f, "dynamic"),
            FactType::Transient => write!(f, "transient"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub content: String,
    pub category: String,
    pub fact_type: FactType,
    pub confidence: f64,
    pub source_date: String,
    pub valid_from: String,
    pub valid_to: Option<String>,
}

impl Fact {
    /// Map fact_type to a predicate string.
    pub fn predicate(&self) -> &str {
        match self.fact_type {
            FactType::Static => "is",
            FactType::Dynamic => "changed",
            FactType::Transient => "noted",
        }
    }
}

// =============================================================================
// D3 — Importance Scoring & Emotion Detection
// =============================================================================

const PREFERENCE_KEYWORDS: &[&str] = &[
    "喜欢", "不喜欢", "偏好", "习惯", "爱好", "爱",
];
const EXPLICIT_MARKERS: &[&str] = &[
    "remember", "记住", "切记", "不要忘", "牢记", "重要",
];
const EMOTION_KEYWORDS: &[(&str, f64)] = &[
    ("开心", 0.8), ("高兴", 0.7), ("难过", 0.6), ("伤心", 0.7),
    ("生气", 0.7), ("焦虑", 0.6), ("紧张", 0.5), ("兴奋", 0.9),
    ("失望", 0.5), ("害怕", 0.6), ("快乐", 0.8), ("愤怒", 0.8),
    ("沮丧", 0.5), ("感动", 0.7), ("温暖", 0.5),
];
const ENTITY_SIGNALS: &[&str] = &[
    "昨天", "前天", "今天", "最近", "上次",
];

pub fn score_content_importance(content: &str) -> f64 {
    let mut score = 0.0_f64;

    // entity_signal: dates, numbers, paths
    if ENTITY_SIGNALS.iter().any(|s| content.contains(*s)) {
        score += 0.3;
    }
    // preference_keyword
    if PREFERENCE_KEYWORDS.iter().any(|s| content.contains(*s)) {
        score += 0.3;
    }
    // emotion_signal
    let (_, arousal) = detect_emotion(content);
    score += arousal * 0.2;
    // explicit_marker
    if EXPLICIT_MARKERS.iter().any(|s| content.contains(*s)) {
        score += 0.2;
    }

    score.min(1.0)
}

pub fn detect_emotion(content: &str) -> (f64, f64) {
    let mut max_arousal = 0.0_f64;
    let mut valence_sum = 0.0_f64;
    let mut count = 0u32;

    for (keyword, arousal) in EMOTION_KEYWORDS {
        if content.contains(keyword) {
            max_arousal = max_arousal.max(*arousal);
            valence_sum += if *arousal > 0.6 { 0.3 } else { -0.3 };
            count += 1;
        }
    }

    let valence = if count > 0 { valence_sum / count as f64 } else { 0.0 };
    (valence.clamp(-1.0, 1.0), max_arousal)
}

pub fn extract_entities_simple(content: &str) -> Vec<String> {
    let skip: &[&str] = &["我", "你", "他", "她", "它", "我们", "是", "的", "了", "在", "和", "与"];
    content
        .split(|c: char| !c.is_alphanumeric() && c != '_' && !c.is_ascii())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .filter(|w| !skip.iter().any(|s| *s == *w))
        .take(10)
        .map(|s| s.to_string())
        .collect()
}

pub fn enrich_metadata(metadata: &MemoryMetadata, content: &str) -> MemoryMetadata {
    let mut meta = metadata.clone();
    if meta.importance.is_none() {
        meta.importance = Some(score_content_importance(content));
    }
    if meta.emotion.is_none() {
        let (valence, arousal) = detect_emotion(content);
        if arousal > 0.0 {
            meta.emotion = Some(Emotion { valence: Some(valence), arousal: Some(arousal) });
        }
    }
    if meta.entities.is_none() {
        let entities = extract_entities_simple(content);
        if !entities.is_empty() {
            meta.entities = Some(entities);
        }
    }
    meta
}

// =============================================================================
// D5 — Fact Extraction
// =============================================================================

pub fn extract_facts(content: &str, noted_date: &str) -> Vec<Fact> {
    let mut facts = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        // 跳过空行、纯标题行、过短行
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // 去掉 markdown 列表前缀 "- "
        let body = line.strip_prefix("- ").unwrap_or(line).trim();
        if body.len() < 6 {
            continue;
        }

        let category = if line.contains("决定") || line.contains("选择")
            || line.contains("改为") || line.contains("切换到")
            || line.contains("启用") || line.contains("禁用")
            || line.contains("统一使用") || line.contains("结论")
        {
            "decision"
        } else if PREFERENCE_KEYWORDS.iter().any(|k| line.contains(k)) {
            "preference"
        } else if line.contains("生日是") || line.contains("出生于")
            || line.contains("工作单位") || line.contains("公司是")
            || line.contains("简历") || line.contains("姓名")
        {
            "person"
        } else if line.contains("项目") || line.contains("计划")
            || line.contains("待办") || line.contains("TODO")
            || line.contains("任务") || line.contains("产出")
        {
            "project"
        } else if line.contains("配置") || line.contains("sa.toml")
            || line.contains("模型") || line.contains("参数")
            || line.contains("tokens") || line.contains("预算")
            || line.contains("窗口") || line.contains("设置")
        {
            "config"
        } else if line.contains("根因") || line.contains("Bug")
            || line.contains("报错") || line.contains("错误")
            || line.contains("修复") || line.contains("排查")
            || line.contains("失败") || line.contains("重复")
        {
            "bug"
        } else if line.contains("教训") || line.contains("经验")
            || line.contains("总结") || line.contains("注意")
            || line.contains("避免") || line.contains("小心")
            || line.contains("小结")
        {
            "lesson"
        } else if line.contains("上下文") || line.contains("压缩")
            || line.contains("session") || line.contains("摘要")
            || line.contains("分段") || line.contains("缓存")
        {
            "context"
        } else if line.contains("启动") || line.contains("恢复")
            || line.contains("重载") || line.contains("Reload")
            || line.contains("skills") || line.contains("MCP")
            || line.contains("模块") || line.contains("技能")
            || line.contains("注册") || line.contains("创建")
            || line.contains("更新") || line.contains("已更新")
        {
            "system"
        } else if line.contains("完成") || line.contains("成功")
            || line.contains("读取") || line.contains("分析")
            || line.contains("提供") || line.contains("确认")
            || line.contains("检查") || line.contains("清理")
            || line.contains("补充") || line.contains("新增")
        {
            "task"
        } else {
            continue;
        };

        let fact_type = if line.contains("永远") || line.contains("一直")
            || line.contains("统一") || line.contains("始终")
        {
            FactType::Static
        } else {
            FactType::Dynamic
        };

        // 根据类别调整置信度
        let confidence = match category {
            "decision" | "lesson" => 0.8,
            "config" | "bug" => 0.75,
            "person" | "preference" => 0.7,
            _ => 0.65,
        };

        facts.push(Fact {
            content: body.to_string(),
            category: category.to_string(),
            fact_type,
            confidence,
            source_date: noted_date.to_string(),
            valid_from: noted_date.to_string(),
            valid_to: None,
        });
    }

    facts
}

// =============================================================================
// D4 — Write Gate: Similarity-Based Deduplication
// =============================================================================

const SIMILARITY_NGRAM_WINDOW: usize = 3;
const SIMILARITY_DUPLICATE_THRESHOLD: f64 = 0.92;
const SIMILARITY_MERGE_THRESHOLD: f64 = 0.70;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteGateAction {
    Skip,
    Merge,
    Write,
}

fn ngrams(text: &str, n: usize) -> std::collections::HashSet<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        let mut set = std::collections::HashSet::new();
        set.insert(text.to_string());
        return set;
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

pub fn compute_similarity(a: &str, b: &str) -> f64 {
    let a_lower = a.to_ascii_lowercase();
    let b_lower = b.to_ascii_lowercase();
    if a_lower == b_lower { return 1.0; }

    let ga = ngrams(&a_lower, SIMILARITY_NGRAM_WINDOW);
    let gb = ngrams(&b_lower, SIMILARITY_NGRAM_WINDOW);
    let intersection = ga.intersection(&gb).count();
    let union = ga.union(&gb).count();
    if union == 0 { return 0.0; }
    intersection as f64 / union as f64
}

pub fn check_write_gate(entries: &[(String, MemoryMetadata)], new_content: &str) -> (WriteGateAction, Option<usize>) {
    let mut best_sim = 0.0_f64;
    let mut best_idx = None;

    for (idx, (old_content, _)) in entries.iter().enumerate() {
        let sim = compute_similarity(new_content, old_content);
        if sim > best_sim {
            best_sim = sim;
            best_idx = Some(idx);
        }
    }

    if best_sim > SIMILARITY_DUPLICATE_THRESHOLD {
        (WriteGateAction::Skip, best_idx)
    } else if best_sim > SIMILARITY_MERGE_THRESHOLD {
        (WriteGateAction::Merge, best_idx)
    } else {
        (WriteGateAction::Write, None)
    }
}

pub(crate) fn extract_entries_from_daily(content: &str) -> Vec<(String, MemoryMetadata)> {
    let mut results = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return results;
    }

    let mut entry_boundaries: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() == "---" {
            entry_boundaries.push(i);
        }
    }

    for window in entry_boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        let entry_text = lines[start..end].join("\n");
        let (meta, body, _) = parse_yaml_front_matter(&entry_text);
        results.push((body.trim().to_string(), meta));
    }

    if let Some(&last) = entry_boundaries.last() {
        if last + 1 < lines.len() {
            let entry_text = lines[last..].join("\n");
            let (meta, body, _) = parse_yaml_front_matter(&entry_text);
            if !body.trim().is_empty() {
                results.push((body.trim().to_string(), meta));
            }
        }
    }

    results
}

// =============================================================================
// YAML Front Matter
// =============================================================================

pub(crate) fn generate_yaml_front_matter(metadata: &MemoryMetadata, created_at: &str) -> String {
    let mut yaml = String::from("---\n");
    yaml.push_str(&format!("created_at: {}\n", created_at));

    if let Some(importance) = metadata.importance {
        yaml.push_str(&format!("importance: {}\n", importance));
    }
    if let Some(emotion) = &metadata.emotion {
        yaml.push_str("emotion:\n");
        if let Some(valence) = emotion.valence {
            yaml.push_str(&format!("  valence: {}\n", valence));
        }
        if let Some(arousal) = emotion.arousal {
            yaml.push_str(&format!("  arousal: {}\n", arousal));
        }
    }
    if let Some(entities) = &metadata.entities {
        yaml.push_str("entities:\n");
        for entity in entities {
            yaml.push_str(&format!("  - {}\n", entity));
        }
    }
    if let Some(access_count) = metadata.access_count {
        yaml.push_str(&format!("access_count: {}\n", access_count));
    }
    if let Some(last_strengthen) = &metadata.last_strengthen {
        yaml.push_str(&format!("last_strengthen: {}\n", last_strengthen));
    }
    if let Some(scope) = &metadata.scope {
        yaml.push_str(&format!("scope: {}\n", scope));
    }
    if let Some(title) = &metadata.title {
        yaml.push_str(&format!("title: {}\n", title));
    }
    if let Some(tags) = &metadata.tags {
        yaml.push_str("tags:\n");
        for tag in tags {
            yaml.push_str(&format!("  - {}\n", tag));
        }
    }
    if let Some(summary) = &metadata.summary {
        yaml.push_str(&format!("summary: {}\n", summary));
    }
    yaml.push_str("---\n");
    yaml
}

pub(crate) fn parse_yaml_front_matter(content: &str) -> (MemoryMetadata, String, usize) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 2 || lines[0].trim() != "---" {
        return (MemoryMetadata::default(), content.to_string(), 0);
    }

    let mut front_matter_lines = Vec::new();
    let mut end_idx = 0usize;
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim() == "---" {
            end_idx = i;
            break;
        }
        front_matter_lines.push(*line);
    }

    let front_matter_str = front_matter_lines.join("\n");
    let metadata: MemoryMetadata = serde_yaml::from_str(&front_matter_str).unwrap_or_default();

    let content_start = end_idx + 1;
    let content_without = lines[content_start..].join("\n");

    (metadata, content_without, end_idx + 1)
}

fn rebuild_daily_with_updated_entry(
    file_path: &Path,
    entries: &[(String, MemoryMetadata)],
    update_idx: usize,
    updated_meta: &MemoryMetadata,
) -> anyhow::Result<()> {
    let mut content = String::new();
    for (idx, (body, meta)) in entries.iter().enumerate() {
        let meta_to_use = if idx == update_idx { updated_meta } else { meta };
        let created_at = meta_to_use.created_at.clone().unwrap_or_else(|| {
            chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+08:00").to_string()
        });
        let fm = generate_yaml_front_matter(meta_to_use, &created_at);
        content.push_str(&fm);
        content.push_str(body);
        content.push_str("\n");
    }
    std::fs::write(file_path, content)?;
    Ok(())
}

/// Rebuild a daily memory file, replacing both metadata and optionally content
/// for the entry at `edit_idx`. Other entries remain unchanged.
pub(crate) fn rebuild_daily_with_edited_entry(
    file_path: &Path,
    entries: &[(String, MemoryMetadata)],
    edit_idx: usize,
    updated_meta: &MemoryMetadata,
    new_content: Option<&str>,
) -> anyhow::Result<()> {
    let mut output = String::new();
    for (idx, (body, meta)) in entries.iter().enumerate() {
        let meta_to_use = if idx == edit_idx { updated_meta } else { meta };
        let created_at = meta_to_use.created_at.clone().unwrap_or_else(|| {
            chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+08:00").to_string()
        });
        let body_to_use = if idx == edit_idx {
            new_content.unwrap_or(body)
        } else {
            body
        };
        let fm = generate_yaml_front_matter(meta_to_use, &created_at);
        output.push_str(&fm);
        output.push_str(body_to_use);
        output.push_str("\n");
    }
    std::fs::write(file_path, output)?;
    Ok(())
}

// =============================================================================
// Write Daily Memory
// =============================================================================

fn write_topic_memory_blocking(
    workspace_root: &Path,
    topic: &str,
    facts: &[Fact],
) -> anyhow::Result<()> {
    let topics_dir = workspace_root.join(MEMORY_TOPICS_DIR);
    if !topics_dir.exists() {
        std::fs::create_dir_all(&topics_dir)?;
    }
    let file_path = topics_dir.join(format!("{}.md", topic));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)?;

    for fact in facts {
        writeln!(file, "- [{}] {} (confidence: {}, from: {})",
            fact.fact_type, fact.content, fact.confidence, fact.source_date)?;
    }
    Ok(())
}

pub fn write_daily_memory(
    workspace_root: &Path,
    date: &str,
    content: &str,
    metadata: &MemoryMetadata,
    filter: Option<&WriteFilterResult>,
) -> anyhow::Result<()> {
    // Apply write filter if provided (PII sanitization + scope enforcement)
    let (content, metadata) = if let Some(f) = filter {
        let mut meta = metadata.clone();
        meta.scope = Some(f.scope);
        (&f.sanitized_content[..], meta)
    } else {
        (content, metadata.clone())
    };

    let metadata = enrich_metadata(&metadata, content);

    // D5: Extract facts and write to topic memory
    let facts = extract_facts(content, date);
    if !facts.is_empty() {
        let mut topics: std::collections::HashMap<String, Vec<Fact>> = std::collections::HashMap::new();
        for fact in facts {
            let topic = match fact.category.as_str() {
                "decision" => "decisions",
                "preference" => "preferences",
                "person" => "people",
                "project" => "projects",
                _ => "general",
            };
            topics.entry(topic.to_string()).or_default().push(fact);
        }
        for (topic, topic_facts) in topics {
            if let Err(e) = write_topic_memory_blocking(workspace_root, &topic, &topic_facts) {
                tracing::warn!("Failed to write topic memory '{}': {}", topic, e);
            }
        }
    }

    let memory_dir = workspace_root.join(MEMORY_DIR);
    if !memory_dir.exists() {
        std::fs::create_dir_all(&memory_dir)
            .with_context(|| format!("Failed to create memory dir: {}", memory_dir.display()))?;
    }
    let file_path = memory_dir.join(format!("{}.md", date));
    let created_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+08:00").to_string();

    // D4 Write Gate: check similarity against existing entries
    if file_path.exists() {
        let existing_content = std::fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read daily memory: {}", file_path.display()))?;
        let entries = extract_entries_from_daily(&existing_content);
        let (gate_action, best_idx) = check_write_gate(&entries, content);

        if gate_action == WriteGateAction::Skip {
            if let Some(idx) = best_idx {
                let new_access_count = entries[idx].1.access_count.unwrap_or(0).saturating_add(1);
                let mut updated_metadata = entries[idx].1.clone();
                updated_metadata.access_count = Some(new_access_count);
                rebuild_daily_with_updated_entry(&file_path, &entries, idx, &updated_metadata)?;
            }
            return Ok(());
        }
    }

    // Write new entry
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file_path)?;
    let front_matter = generate_yaml_front_matter(&metadata, &created_at);
    file.write_all(front_matter.as_bytes())?;
    file.write_all(content.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;

    Ok(())
}




// ── D10: Similar Memory Clustering ─────────────────────────────────────────

/// Cluster similar entries across recent daily memory files.
///
/// Scans daily files within `lookback_days`, splits each file into
/// YAML-fronted blocks, and groups blocks whose similarity exceeds
/// `threshold`. Returns a list of groups, each containing
/// `(relative_path, block_text)` tuples.
pub fn find_similar_clusters(
    workspace_root: &Path,
    lookback_days: usize,
    threshold: f64,
) -> Vec<Vec<(String, String)>> {
    use chrono::{Local, Duration};

    let memory_dir = workspace_root.join(MEMORY_DIR);
    if !memory_dir.is_dir() {
        return vec![];
    }

    // Collect all blocks from recent files.
    let mut all_blocks: Vec<(String, String)> = Vec::new(); // (rel_path, block_text)
    for days_ago in 0..lookback_days {
        let date = (Local::now() - Duration::days(days_ago as i64))
            .format("%Y-%m-%d")
            .to_string();
        let file_path = memory_dir.join(format!("{}.md", &date));
        if !file_path.is_file() {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file_path) else { continue };
        let rel = format!("memory/{}.md", &date);

        // Split by YAML front matter delimiters.
        let parts: Vec<&str> = text.split("
---
").collect();
        for chunk in parts {
            let trimmed = chunk.trim();
            if trimmed.len() < 20 {
                continue;
            }
            all_blocks.push((rel.clone(), trimmed.to_string()));
        }
    }

    if all_blocks.len() < 2 {
        return vec![];
    }

    // Greedy clustering: group blocks with pairwise similarity > threshold.
    let mut used = vec![false; all_blocks.len()];
    let mut clusters: Vec<Vec<(String, String)>> = Vec::new();

    for i in 0..all_blocks.len() {
        if used[i] {
            continue;
        }
        let mut cluster = vec![all_blocks[i].clone()];
        used[i] = true;

        for j in (i + 1)..all_blocks.len() {
            if used[j] {
                continue;
            }
            // Compare against any member of the cluster.
            let sim = compute_similarity(&all_blocks[i].1, &all_blocks[j].1);
            if sim >= threshold {
                cluster.push(all_blocks[j].clone());
                used[j] = true;
            }
        }

        if cluster.len() >= 2 {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Search semantic memory facts from `memory/semantic.db`.
/// Returns a vector of (subject, predicate, object, confidence, source) tuples.
pub fn search_semantic_facts(
    workspace_root: &Path,
    query: &str,
    max_results: usize,
) -> Vec<(String, String, String, f64, Option<String>)> {
    use rusqlite::{params, Connection, OpenFlags};
    let db_path = workspace_root.join("memory/semantic.db");
    if !db_path.exists() {
        return Vec::new();
    }
    let conn = match Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let pattern = format!("%{}%", query);
    let mut stmt = match conn.prepare(
        "SELECT subject, predicate, object, confidence, source \
         FROM facts \
         WHERE subject LIKE ?1 OR predicate LIKE ?1 OR object LIKE ?1 \
         ORDER BY confidence DESC \
         LIMIT ?2",
    ) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let rows = stmt.query_map(params![pattern, max_results], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, f64>(3)?,
            row.get::<_, Option<String>>(4)?,
        ))
    });
    match rows {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(_) => Vec::new(),
    }
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
        assert!(is_memory_reference("memory/topics/preferences.md"));
        assert!(is_memory_reference("memory/topics/rust/testing.md"));
        assert!(is_memory_reference("memory/dreams/2026-04-13.md"));
        assert!(!is_memory_reference("notes/memory.md"));
        assert!(!is_memory_reference("memory/misc/note.md"));
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
    async fn memory_search_reads_memory_md_daily_files_and_topics_but_skips_dream_audits() {
        let workspace = unique_workspace();
        fs::write(
            workspace.join("MEMORY.md"),
            "Preference: likes Rust and careful reviews.",
        )
        .expect("write MEMORY.md");
        fs::create_dir_all(workspace.join("memory")).expect("create memory dir");
        fs::create_dir_all(workspace.join("memory").join("topics")).expect("create topics dir");
        fs::create_dir_all(workspace.join("memory").join("dreams")).expect("create dreams dir");
        fs::write(
            workspace.join("memory").join("2026-03-06.md"),
            "2026-03-06\nDecision: use skill sandbox instead of exposing host paths.",
        )
        .expect("write daily memory");
        fs::write(
            workspace
                .join("memory")
                .join("topics")
                .join("preferences.md"),
            "User prefers concise progress updates and clear boundaries.",
        )
        .expect("write topic memory");
        fs::write(
            workspace
                .join("memory")
                .join("dreams")
                .join("2026-03-06.md"),
            "Dream audit: remove stale sandbox note after policy change.",
        )
        .expect("write dream audit");

        let hits = search_markdown_memory(&workspace, "skill sandbox host paths", Some(5), None, None, None)
            .await
            .expect("memory search");
        assert!(!hits.is_empty());
        assert!(hits.iter().any(|hit| hit.path == "memory/2026-03-06.md"));
        assert!(
            !hits
                .iter()
                .any(|hit| hit.path == "memory/dreams/2026-03-06.md")
        );

        let topic_hits =
            search_markdown_memory(&workspace, "concise progress updates", Some(5), None, None, None)
                .await
                .expect("topic memory search");
        assert!(
            topic_hits
                .iter()
                .any(|hit| hit.path == "memory/topics/preferences.md")
        );
    }

    #[tokio::test]
    async fn memory_get_rejects_outside_paths() {
        let workspace = unique_workspace();
        let err = read_markdown_memory(&workspace, "../secret.md", None, None)
            .await
            .expect_err("outside path must fail");
        assert!(err.to_string().contains("only allows"));
    }

    #[tokio::test]
    async fn memory_get_allows_dream_audit_reads_for_explicit_auditing() {
        let workspace = unique_workspace();
        fs::create_dir_all(workspace.join("memory").join("dreams")).expect("create dreams dir");
        fs::write(
            workspace
                .join("memory")
                .join("dreams")
                .join("2026-04-13.md"),
            "Dream audit line 1\nDream audit line 2",
        )
        .expect("write dream audit");

        let result =
            read_markdown_memory(&workspace, "memory/dreams/2026-04-13.md", Some(1), Some(2))
                .await
                .expect("dream audit read should succeed");
        assert_eq!(result.path, "memory/dreams/2026-04-13.md");
        assert!(result.text.contains("Dream audit line 1"));
    }
}
