//! Batch file indexing module for the SA agent.
//!
//! Provides `BuildIndex` and `SearchIndex` built-in tools:
//! - **BuildIndex**: scans the workspace, extracts metadata + preview from each
//!   file, and persists the index to `workspace/runtime/file_index.json`.
//!   Supports incremental updates by comparing file `mtime`.
//! - **SearchIndex**: performs keyword search over the persisted index.

use crate::cancel::CancelToken;
use crate::file_search::PathResolver;
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::tool_cache::FileIndexCache;
use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Default index file path relative to workspace root.
const INDEX_RELATIVE_PATH: &str = "runtime/file_index.json";

/// Maximum number of characters to preview per file.
const PREVIEW_CHARS: usize = 500;

/// Maximum directory depth for indexing (0 = unlimited).
const MAX_INDEX_DEPTH: usize = 20;

/// Directories to skip during indexing.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".next",
    "dist",
    "build",
    ".venv",
    "venv",
    ".mcp",
];

/// File extensions to skip during indexing (binary / generated).
const SKIP_EXTENSIONS: &[&str] = &[
    "exe", "dll", "so", "dylib", "o", "a", "lib", "pdb", "wasm", "png", "jpg", "jpeg", "gif",
    "webp", "bmp", "ico", "svg", "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "mp3", "mp4", "avi",
    "mkv", "mov", "wav", "flac",
    "pdf", // PDFs are handled via MCP OCR, skip raw content indexing
    "pyc",
];

/// Tool definition for `BuildIndex`.
pub fn build_index_definition() -> ToolDefinition {
    ToolDefinition {
        kind: "function".to_string(),
        function: ToolFunctionDefinition {
            name: "BuildIndex".to_string(),
            description: "Build or incrementally update a keyword index of workspace files. The index stores file metadata and content previews for fast searching. Call this before using SearchIndex."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "If true, rebuild the index from scratch. If false (default), perform incremental update based on file modification times."
                    }
                }
            }),
        },
    }
}

/// Tool definition for `SearchIndex`.
pub fn search_index_definition() -> ToolDefinition {
    ToolDefinition {
        kind: "function".to_string(),
        function: ToolFunctionDefinition {
            name: "SearchIndex".to_string(),
            description: "Search the workspace file index by keyword. Returns matching files ranked by relevance with content previews. BuildIndex must have been called first."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "One or more keywords to search for. Multiple keywords are ANDed."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results. Defaults to 10 and is capped at 50."
                    }
                },
                "required": ["query"]
            }),
        },
    }
}

/// One entry in the file index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Path relative to workspace root.
    pub relative_path: String,
    /// File extension (lowercase).
    pub extension: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last modification time (Unix timestamp seconds).
    pub modified_ts: i64,
    /// Category classification.
    pub category: String,
    /// First N characters of content (for text files).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}

/// The full persisted index.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileIndex {
    /// Workspace root at index build time (for validation).
    pub workspace_root: String,
    /// When the index was last built/updated.
    pub built_at: String,
    /// Total number of indexed files.
    pub total_files: usize,
    /// Indexed entries keyed by relative path.
    pub entries: BTreeMap<String, IndexEntry>,
}

impl FileIndex {
    /// Load index from disk, or return an empty index if not found.
    fn load(index_path: &Path) -> anyhow::Result<Self> {
        if !index_path.exists() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(index_path)
            .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;
        serde_json::from_str(&data)
            .with_context(|| format!("Failed to parse index file: {}", index_path.display()))
    }

    /// Persist index to disk.
    fn save(&self, index_path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self).context("Failed to serialize file index")?;
        std::fs::write(index_path, data)?;
        Ok(())
    }
}

/// Execute `BuildIndex`.
pub async fn execute_build_index(
    workspace_root: &Path,
    _path_resolver: &dyn PathResolver,
    args: serde_json::Value,
    _cancel: &CancelToken,
    index_cache: &FileIndexCache,
) -> anyhow::Result<String> {
    #[derive(Debug, Deserialize)]
    struct Args {
        #[serde(default)]
        force: bool,
    }
    let args: Args = serde_json::from_value(args).unwrap_or(Args { force: false });

    let index_path = workspace_root.join(INDEX_RELATIVE_PATH);

    // Clear memory cache on force rebuild.
    if args.force {
        index_cache.invalidate().await;
    }

    let mut index = if args.force {
        FileIndex::default()
    } else {
        // Try memory cache first, then fall back to disk.
        match index_cache.get().await {
            Some(cached) => cached,
            None => FileIndex::load(&index_path)?,
        }
    };

    let mut new_count = 0usize;
    let mut updated_count = 0usize;
    let mut skipped_count = 0usize;

    // Walk the workspace directory.
    walk_dir_index(
        workspace_root,
        workspace_root,
        0,
        &mut index,
        &mut new_count,
        &mut updated_count,
        &mut skipped_count,
    )?;

    index.workspace_root = workspace_root.display().to_string();
    index.built_at = chrono::Utc::now().to_rfc3339();
    index.total_files = index.entries.len();
    index.save(&index_path)?;

    // Persist to memory cache for fast SearchIndex access.
    index_cache.put(index.clone()).await;

    Ok(serde_json::json!({
        "status": "ok",
        "index_path": index_path.display().to_string(),
        "total_files": index.total_files,
        "new_files": new_count,
        "updated_files": updated_count,
        "skipped_files": skipped_count,
    })
    .to_string())
}

/// Execute `SearchIndex`.
pub async fn execute_search_index(
    workspace_root: &Path,
    _path_resolver: &dyn PathResolver,
    args: serde_json::Value,
    _cancel: &CancelToken,
    index_cache: &FileIndexCache,
) -> anyhow::Result<String> {
    #[derive(Debug, Deserialize)]
    struct Args {
        query: String,
        max_results: Option<usize>,
    }

    let args: Args = serde_json::from_value(args).context("Invalid arguments for SearchIndex")?;
    let max_results = args.max_results.unwrap_or(10).min(50);

    // Try memory cache first, then fall back to disk.
    let index = match index_cache.get().await {
        Some(cached) if !cached.entries.is_empty() => cached,
        _ => {
            let index_path = workspace_root.join(INDEX_RELATIVE_PATH);
            let from_disk = FileIndex::load(&index_path)?;
            if !from_disk.entries.is_empty() {
                // Warm the memory cache from disk.
                index_cache.put(from_disk.clone()).await;
                from_disk
            } else {
                from_disk
            }
        }
    };

    if index.entries.is_empty() {
        return Ok("File index is empty. Call BuildIndex first to create the index.".to_string());
    }

    // Parse query into keywords (split on whitespace).
    let keywords: Vec<String> = args
        .query
        .split_whitespace()
        .map(|k| k.to_lowercase())
        .collect();

    if keywords.is_empty() {
        anyhow::bail!("SearchIndex requires at least one keyword");
    }

    // Score each entry by keyword matches.
    let mut scored: Vec<(i32, &IndexEntry)> = index
        .entries
        .values()
        .map(|entry| {
            let mut score = 0i32;
            let path_lower = entry.relative_path.to_lowercase();
            let preview_lower = entry.preview.as_deref().unwrap_or("").to_lowercase();

            for kw in &keywords {
                // Filename match (higher weight).
                let file_name = Path::new(&entry.relative_path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if file_name.contains(kw) {
                    score += 10;
                }
                // Path match.
                if path_lower.contains(kw) {
                    score += 5;
                }
                // Content preview match.
                let count = preview_lower.matches(kw.as_str()).count() as i32;
                if count > 0 {
                    score += count;
                }
            }
            (score, entry)
        })
        .filter(|(score, _)| *score > 0)
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.truncate(max_results);

    let results: Vec<serde_json::Value> = scored
        .iter()
        .map(|(score, entry)| {
            serde_json::json!({
                "path": entry.relative_path,
                "type": entry.category,
                "size": entry.size_bytes,
                "score": score,
                "preview": entry.preview.as_deref().unwrap_or(""),
            })
        })
        .collect();

    Ok(serde_json::json!({
        "query": args.query,
        "results": results,
        "total_matches": results.len(),
    })
    .to_string())
}

/// Recursively walk and index files.
fn walk_dir_index(
    base: &Path,
    current: &Path,
    depth: usize,
    index: &mut FileIndex,
    new_count: &mut usize,
    updated_count: &mut usize,
    skipped_count: &mut usize,
) -> anyhow::Result<()> {
    if depth > MAX_INDEX_DEPTH {
        return Ok(());
    }

    let entries = std::fs::read_dir(current)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        // Skip hidden files/dirs and well-known skip dirs.
        if file_name_str.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if SKIP_DIRS.contains(&file_name_str.as_ref()) {
                continue;
            }
            walk_dir_index(
                base,
                &path,
                depth + 1,
                index,
                new_count,
                updated_count,
                skipped_count,
            )?;
            continue;
        }

        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        if SKIP_EXTENSIONS.contains(&ext.as_str()) {
            continue;
        }

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Skip very large files.
        if meta.len() > 1_000_000 {
            continue;
        }

        let relative_path = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        let modified_ts = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        // Check if file needs updating.
        let needs_update = match index.entries.get(&relative_path) {
            Some(existing) => existing.modified_ts != modified_ts,
            None => true,
        };

        if !needs_update {
            *skipped_count += 1;
            continue;
        }

        // Read content preview for text files.
        let (category, preview) = classify_and_preview(&path, &ext);

        let is_new = !index.entries.contains_key(&relative_path);
        if is_new {
            *new_count += 1;
        } else {
            *updated_count += 1;
        }

        index.entries.insert(
            relative_path,
            IndexEntry {
                relative_path: path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string(),
                extension: ext,
                size_bytes: meta.len(),
                modified_ts,
                category,
                preview,
            },
        );
    }

    Ok(())
}

/// Classify a file and extract a text preview.
fn classify_and_preview(path: &Path, ext: &str) -> (String, Option<String>) {
    let text_extensions = [
        "txt",
        "md",
        "rst",
        "log",
        "rs",
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "go",
        "java",
        "c",
        "cpp",
        "h",
        "hpp",
        "css",
        "scss",
        "less",
        "html",
        "htm",
        "xml",
        "yaml",
        "yml",
        "toml",
        "json",
        "jsonl",
        "csv",
        "tsv",
        "sql",
        "sh",
        "bash",
        "zsh",
        "fish",
        "ps1",
        "bat",
        "cmd",
        "makefile",
        "dockerfile",
        "gitignore",
        "env",
        "rb",
        "php",
        "swift",
        "kt",
        "scala",
        "lua",
        "vim",
        "el",
    ];

    if text_extensions.contains(&ext)
        || ext.is_empty()
        || path.file_name().is_some_and(|n| {
            let name = n.to_string_lossy().to_lowercase();
            name == "makefile" || name == "dockerfile" || name == "license" || name == "readme"
        })
    {
        let preview = std::fs::read_to_string(path).ok().map(|content| {
            let chars: Vec<char> = content.chars().take(PREVIEW_CHARS).collect();
            let mut result: String = chars.into_iter().collect();
            if content.chars().count() > PREVIEW_CHARS {
                result.push_str("...");
            }
            result
        });
        ("text".to_string(), preview)
    } else {
        ("binary".to_string(), None)
    }
}
