//! File search tools: ListDir, Glob, Grep.
//!
//! This module provides three built-in tools that let the agent explore the
//! workspace filesystem natively without falling back to `Bash ls/find/grep`.
//!
//! Design goals:
//! - Keep the implementation independent from `tools.rs` (collaboration rule §6).
//! - Reuse `path_guard` and `ToolContext::resolve_under_workspace` for safety.
//! - Respect `walkdir` (already in the dependency tree) for recursive traversal.

use crate::cancel::CancelToken;
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::path_guard::{PathOperation, validate_resolved_tool_path, validate_tool_path_input};
use anyhow::Context as _;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Trait for resolving a user-provided path string to an absolute path.
///
/// This decouples `file_search` from `tools::ToolContext` to avoid a circular
/// dependency (tools.rs → file_search.rs → tools.rs).
pub trait PathResolver: Send + Sync {
    fn resolve_under_workspace(&self, raw: &str) -> anyhow::Result<PathBuf>;
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Directories that should be silently skipped during recursive traversal.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".cache",
];

/// Maximum recursion depth for ListDir.
const MAX_LISTDIR_DEPTH: usize = 10;

/// Maximum entries returned by ListDir.
const MAX_LISTDIR_ENTRIES: usize = 1_000;

/// Maximum matches returned by Glob.
const MAX_GLOB_MATCHES: usize = 200;

/// Maximum matches returned by Grep.
const MAX_GREP_MATCHES: usize = 200;

/// Number of bytes to sample when detecting binary files.
const BINARY_DETECT_BYTES: usize = 8_192;

// ---------------------------------------------------------------------------
// ToolDefinition builders
// ---------------------------------------------------------------------------

/// `ListDir` — list directory contents with optional recursion and glob filter.
pub fn list_dir_definition() -> ToolDefinition {
    ToolDefinition {
        kind: "function".to_string(),
        function: ToolFunctionDefinition {
            name: "ListDir".to_string(),
            description: "List files and directories at a given path. \
                Supports optional recursion and glob pattern filtering. \
                Returns names, types (file/dir/link), and sizes."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path, relative to the workspace root or absolute under it."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter results (e.g. '*.rs', 'src/**/*.ts'). Only applies when recursive is true."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list subdirectories recursively. Defaults to false."
                    }
                },
                "required": ["path"]
            }),
        },
    }
}

/// `Glob` — search for files matching a glob pattern.
pub fn glob_definition() -> ToolDefinition {
    ToolDefinition {
        kind: "function".to_string(),
        function: ToolFunctionDefinition {
            name: "Glob".to_string(),
            description: "Search for files matching a glob pattern (e.g. '*.rs', 'src/**/*.ts'). \
                Returns matching file paths relative to the workspace root."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files against."
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in. Defaults to workspace root."
                    }
                },
                "required": ["pattern"]
            }),
        },
    }
}

/// `Grep` — search file contents for lines matching a regex pattern.
pub fn grep_definition() -> ToolDefinition {
    ToolDefinition {
        kind: "function".to_string(),
        function: ToolFunctionDefinition {
            name: "Grep".to_string(),
            description: "Search file contents for lines matching a regex pattern. \
                Returns file paths, line numbers, and matching lines with context."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Defaults to workspace root."
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional file glob to filter which files to search (e.g. '*.rs')."
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum total matches to return. Defaults to 50, capped at 200."
                    }
                },
                "required": ["pattern"]
            }),
        },
    }
}

// ---------------------------------------------------------------------------
// Execution functions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ListDirArgs {
    path: String,
    pattern: Option<String>,
    recursive: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct GlobArgs {
    pattern: String,
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GrepArgs {
    pattern: String,
    path: Option<String>,
    include: Option<String>,
    max_matches: Option<usize>,
}

/// Execute the `ListDir` tool.
pub async fn execute_list_dir(
    workspace_root: &Path,
    resolver: &dyn PathResolver,
    args: serde_json::Value,
    cancel: &CancelToken,
) -> anyhow::Result<String> {
    if cancel.is_cancelled() {
        anyhow::bail!("ListDir cancelled");
    }

    let args: ListDirArgs =
        serde_json::from_value(args).context("Invalid arguments for ListDir")?;

    validate_tool_path_input(&args.path, PathOperation::Read)?;
    let target_dir = resolver.resolve_under_workspace(&args.path)?;
    validate_resolved_tool_path(workspace_root, &target_dir, PathOperation::Read)?;

    let meta = tokio::fs::metadata(&target_dir)
        .await
        .with_context(|| format!("Failed to stat path: {}", target_dir.display()))?;
    if !meta.is_dir() {
        anyhow::bail!(
            "ListDir requires a directory, not a file: {}",
            target_dir.display()
        );
    }

    let recursive = args.recursive.unwrap_or(false);

    if recursive {
        list_dir_recursive(workspace_root, &target_dir, args.pattern.as_deref(), cancel).await
    } else {
        list_dir_flat(&target_dir).await
    }
}

/// Execute the `Glob` tool.
pub async fn execute_glob_search(
    workspace_root: &Path,
    resolver: &dyn PathResolver,
    args: serde_json::Value,
    cancel: &CancelToken,
) -> anyhow::Result<String> {
    if cancel.is_cancelled() {
        anyhow::bail!("Glob cancelled");
    }

    let args: GlobArgs =
        serde_json::from_value(args).context("Invalid arguments for Glob")?;

    if args.pattern.contains("..") {
        anyhow::bail!("Glob pattern must not contain '..' path traversal: {}", args.pattern);
    }

    let base_dir = match &args.path {
        Some(p) => {
            validate_tool_path_input(p, PathOperation::Read)?;
            let resolved = resolver.resolve_under_workspace(p)?;
            validate_resolved_tool_path(workspace_root, &resolved, PathOperation::Read)?;
            resolved
        }
        None => workspace_root.to_path_buf(),
    };

    // Build the full glob pattern: base_dir/pattern
    let full_pattern = base_dir.join(&args.pattern);
    let pattern_str = full_pattern
        .to_str()
        .with_context(|| format!("Invalid glob path: {}", full_pattern.display()))?;

    let mut entries: Vec<String> = Vec::new();

    for entry in glob::glob(pattern_str)
        .with_context(|| format!("Invalid glob pattern: {}", pattern_str))?
    {
        if cancel.is_cancelled() {
            anyhow::bail!("Glob cancelled");
        }

        match entry {
            Ok(path) => {
                // Skip directories, hidden files, and files outside workspace
                if path.is_dir() {
                    continue;
                }
                if is_hidden_entry(&path) {
                    continue;
                }
                if !path.starts_with(workspace_root) {
                    continue;
                }

                // Convert to relative path
                if let Ok(rel) = path.strip_prefix(workspace_root) {
                    if let Some(rel_str) = rel.to_str() {
                        entries.push(rel_str.replace('\\', "/"));
                    }
                }

                if entries.len() >= MAX_GLOB_MATCHES {
                    break;
                }
            }
            Err(e) => {
                // Log but don't fail on permission errors etc.
                tracing::debug!("Glob entry error: {e}");
            }
        }
    }

    if entries.is_empty() {
        return Ok("No files matched the given pattern.".to_string());
    }

    // Deduplicate and sort
    entries.sort();
    entries.dedup();

    let total = entries.len();
    let truncated = total >= MAX_GLOB_MATCHES;
    let display: Vec<&str> = entries.iter().map(|s| s.as_str()).collect();
    let mut result = display.join("\n");
    if truncated {
        result.push_str(&format!(
            "\n\n(showing {MAX_GLOB_MATCHES} of {total}+ matches)"
        ));
    }
    Ok(result)
}

/// Execute the `Grep` tool.
pub async fn execute_grep_search(
    workspace_root: &Path,
    resolver: &dyn PathResolver,
    args: serde_json::Value,
    cancel: &CancelToken,
) -> anyhow::Result<String> {
    if cancel.is_cancelled() {
        anyhow::bail!("Grep cancelled");
    }

    let args: GrepArgs =
        serde_json::from_value(args).context("Invalid arguments for Grep")?;

    let re = regex::Regex::new(&args.pattern)
        .with_context(|| format!("Invalid regex pattern: {}", args.pattern))?;

    let target_path = match &args.path {
        Some(p) => {
            validate_tool_path_input(p, PathOperation::Read)?;
            let resolved = resolver.resolve_under_workspace(p)?;
            validate_resolved_tool_path(workspace_root, &resolved, PathOperation::Read)?;
            resolved
        }
        None => workspace_root.to_path_buf(),
    };

    let max_matches = args.max_matches.unwrap_or(50).min(MAX_GREP_MATCHES);

    // Build optional include glob matcher
    let include_matcher = match &args.include {
        Some(glob_str) => {
            let pat = if glob_str.starts_with('*') {
                // Simple extension pattern like "*.rs"
                glob_str.to_string()
            } else {
                format!("*{glob_str}")
            };
            Some(glob::Pattern::new(&pat).with_context(|| {
                format!("Invalid include glob pattern: {}", args.include.as_deref().unwrap())
            })?)
        }
        None => None,
    };

    let mut matches: Vec<GrepMatch> = Vec::new();

    // Check if target is a single file or a directory
    let target_meta = tokio::fs::metadata(&target_path).await.with_context(|| {
        format!("Failed to stat path: {}", target_path.display())
    })?;

    if target_meta.is_file() {
        grep_file(
            &target_path,
            workspace_root,
            &re,
            &include_matcher,
            &mut matches,
            max_matches,
            cancel,
        )
        .await?;
    } else if target_meta.is_dir() {
        grep_directory(
            &target_path,
            workspace_root,
            &re,
            &include_matcher,
            &mut matches,
            max_matches,
            cancel,
        )
        .await?;
    } else {
        anyhow::bail!(
            "Grep target is neither a file nor a directory: {}",
            target_path.display()
        );
    }

    if matches.is_empty() {
        return Ok(format!(
            "No matches found for pattern: {}",
            args.pattern
        ));
    }

    let truncated = matches.len() >= max_matches;
    let result = format_grep_output(&matches);
    let mut output = result;
    if truncated {
        output.push_str(&format!(
            "\n\n(showing {max_matches} of potentially more matches)"
        ));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Grep output types
// ---------------------------------------------------------------------------

struct GrepMatch {
    rel_path: String,
    line_num: usize,
    line_text: String,
}

fn format_grep_output(matches: &[GrepMatch]) -> String {
    // Group by file
    let mut by_file: Vec<(&str, Vec<&GrepMatch>)> = Vec::new();
    let mut seen_files: Vec<&str> = Vec::new();

    for m in matches {
        if !seen_files.contains(&m.rel_path.as_str()) {
            seen_files.push(&m.rel_path);
        }
    }

    for file in &seen_files {
        let file_matches: Vec<&GrepMatch> = matches
            .iter()
            .filter(|m| m.rel_path.as_str() == *file)
            .collect();
        by_file.push((*file, file_matches));
    }

    let mut output = String::new();
    for (i, (file, file_matches)) in by_file.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(file);
        output.push_str(":\n");
        for m in file_matches {
            output.push_str(&format!("  {:>5}: {}\n", m.line_num, m.line_text));
        }
    }
    output
}

// ---------------------------------------------------------------------------
// ListDir implementation
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct DirEntry {
    name: String,
    #[serde(rename = "type")]
    entry_type: String, // "file", "dir", "symlink"
    size: u64,
}

async fn list_dir_flat(dir: &Path) -> anyhow::Result<String> {
    let mut entries: Vec<DirEntry> = Vec::new();

    let mut rd = tokio::fs::read_dir(dir)
        .await
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    while let Some(entry) = rd.next_entry().await? {
        let path = entry.path();
        if is_hidden_entry(&path) {
            continue;
        }

        let name = entry
            .file_name()
            .to_str()
            .unwrap_or("<invalid>")
            .to_string();
        let meta = entry.metadata().await?;
        let entry_type = if meta.is_symlink() {
            "symlink".to_string()
        } else if meta.is_dir() {
            "dir".to_string()
        } else {
            "file".to_string()
        };
        let size = meta.len();

        entries.push(DirEntry {
            name,
            entry_type,
            size,
        });

        if entries.len() >= MAX_LISTDIR_ENTRIES {
            break;
        }
    }

    if entries.is_empty() {
        return Ok("(empty directory)".to_string());
    }

    // Sort: directories first, then files, alphabetically within each group
    entries.sort_by(|a, b| {
        match (a.entry_type.as_str(), b.entry_type.as_str()) {
            ("dir", "file") | ("dir", "symlink") => std::cmp::Ordering::Less,
            ("file", "dir") | ("symlink", "dir") => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        }
    });

    let output = entries
        .iter()
        .map(|e| {
            let size_str = if e.entry_type == "dir" {
                "".to_string()
            } else if e.size < 1024 {
                format!("  ({} B)", e.size)
            } else if e.size < 1024 * 1024 {
                format!("  ({:.1} KB)", e.size as f64 / 1024.0)
            } else {
                format!("  ({:.1} MB)", e.size as f64 / (1024.0 * 1024.0))
            };
            format!(
                "{}  [{}]{}",
                e.name,
                match e.entry_type.as_str() {
                    "dir" => "D",
                    "symlink" => "L",
                    _ => "F",
                },
                size_str
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(output)
}

async fn list_dir_recursive(
    _workspace_root: &Path,
    root_dir: &Path,
    pattern: Option<&str>,
    cancel: &CancelToken,
) -> anyhow::Result<String> {
    let pattern_matcher = match pattern {
        Some(p) => Some(glob::Pattern::new(p).with_context(|| format!("Invalid glob pattern: {p}"))?),
        None => None,
    };

    let mut entries: Vec<String> = Vec::new();

    // Collect file paths first using walkdir's filter_entry, then process.
    // walkdir is a sync iterator so we cannot call .await inside the loop.
    let walk_iter = walkdir::WalkDir::new(root_dir)
        .max_depth(MAX_LISTDIR_DEPTH)
        .into_iter()
        .filter_entry(|e| {
            let path = e.path();
            // Skip hidden entries
            if is_hidden_entry(path) {
                return false;
            }
            // Skip well-known heavy directories
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if SKIP_DIRS.contains(&name) && e.file_type().is_dir() {
                    return false;
                }
            }
            true
        });

    for entry in walk_iter {
        if cancel.is_cancelled() {
            anyhow::bail!("ListDir cancelled");
        }

        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("WalkDir error: {e}");
                continue;
            }
        };

        let path = entry.path();

        let rel = match path.strip_prefix(root_dir) {
            Ok(r) => r.to_str().with_context(|| format!("Invalid path: {}", path.display()))?,
            Err(_) => continue,
        };

        // Apply glob filter if provided
        if let Some(matcher) = pattern_matcher.as_ref() {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if !matcher.matches(file_name) && !matcher.matches(rel.replace('\\', "/").as_str()) {
                continue;
            }
        }

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let rel_display = rel.replace('\\', "/");

        if meta.is_dir() {
            entries.push(format!("{rel_display}/"));
        } else {
            let size_str = if meta.len() < 1024 {
                format!("  ({} B)", meta.len())
            } else if meta.len() < 1024 * 1024 {
                format!("  ({:.1} KB)", meta.len() as f64 / 1024.0)
            } else {
                format!("  ({:.1} MB)", meta.len() as f64 / (1024.0 * 1024.0))
            };
            entries.push(format!("{rel_display}{size_str}"));
        }

        if entries.len() >= MAX_LISTDIR_ENTRIES {
            break;
        }
    }

    if entries.is_empty() {
        return Ok("(no matching entries found)".to_string());
    }

    entries.sort();
    let output = entries.join("\n");

    let truncated = entries.len() >= MAX_LISTDIR_ENTRIES;
    let mut result = output;
    if truncated {
        result.push_str(&format!(
            "\n\n(showing {MAX_LISTDIR_ENTRIES} of potentially more entries)"
        ));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Grep implementation helpers
// ---------------------------------------------------------------------------

async fn grep_file(
    file_path: &Path,
    workspace_root: &Path,
    re: &regex::Regex,
    include_matcher: &Option<glob::Pattern>,
    matches: &mut Vec<GrepMatch>,
    max_matches: usize,
    cancel: &CancelToken,
) -> anyhow::Result<()> {
    // Check include filter
    if let Some(matcher) = include_matcher.as_ref() {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if !matcher.matches(file_name) {
            return Ok(());
        }
    }

    // Read file content (async)
    let content = tokio::fs::read(file_path).await.with_context(|| {
        format!("Failed to read file: {}", file_path.display())
    })?;

    // Binary detection
    if is_binary_content(&content) {
        return Ok(());
    }

    let text = String::from_utf8_lossy(&content);

    for (i, line) in text.lines().enumerate() {
        if cancel.is_cancelled() {
            anyhow::bail!("Grep cancelled");
        }
        if re.is_match(line) {
            let rel_path = file_path
                .strip_prefix(workspace_root)
                .unwrap_or(file_path)
                .to_str()
                .unwrap_or("<invalid>")
                .replace('\\', "/");
            matches.push(GrepMatch {
                rel_path,
                line_num: i + 1,
                line_text: line.to_string(),
            });
            if matches.len() >= max_matches {
                return Ok(());
            }
        }
    }

    Ok(())
}

async fn grep_directory(
    dir: &Path,
    workspace_root: &Path,
    re: &regex::Regex,
    include_matcher: &Option<glob::Pattern>,
    matches: &mut Vec<GrepMatch>,
    max_matches: usize,
    cancel: &CancelToken,
) -> anyhow::Result<()> {
    // Phase 1: collect candidate file paths synchronously via walkdir
    let mut candidate_files: Vec<PathBuf> = Vec::new();

    let walk_iter = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_entry(|e| {
            let path = e.path();
            if is_hidden_entry(path) {
                return false;
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if SKIP_DIRS.contains(&name) && e.file_type().is_dir() {
                    return false;
                }
            }
            true
        });

    for entry in walk_iter {
        if cancel.is_cancelled() {
            anyhow::bail!("Grep cancelled");
        }
        if candidate_files.len() >= max_matches * 2 {
            break; // collected enough candidates
        }

        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path().to_path_buf();

        // Only process files
        if !entry.file_type().is_file() {
            continue;
        }

        // Check include filter
        if let Some(matcher) = include_matcher.as_ref() {
            let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !matcher.matches(fname) {
                continue;
            }
        }

        candidate_files.push(path);
    }

    // Phase 2: read and search files asynchronously
    for file_path in candidate_files {
        if cancel.is_cancelled() {
            anyhow::bail!("Grep cancelled");
        }
        if matches.len() >= max_matches {
            break;
        }

        let content = match tokio::fs::read(&file_path).await {
            Ok(c) => c,
            Err(_) => continue,
        };

        if is_binary_content(&content) {
            continue;
        }

        let text = String::from_utf8_lossy(&content);

        for (i, line) in text.lines().enumerate() {
            if cancel.is_cancelled() {
                anyhow::bail!("Grep cancelled");
            }
            if re.is_match(line) {
                let rel_path = file_path
                    .strip_prefix(workspace_root)
                    .unwrap_or(&file_path)
                    .to_str()
                    .unwrap_or("<invalid>")
                    .replace('\\', "/");
                matches.push(GrepMatch {
                    rel_path,
                    line_num: i + 1,
                    line_text: line.to_string(),
                });
                if matches.len() >= max_matches {
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared utility helpers
// ---------------------------------------------------------------------------

/// Check if a path entry should be hidden (starts with `.`).
fn is_hidden_entry(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

/// Detect whether a byte slice looks like binary content.
///
/// Uses a simple heuristic: if the first `BINARY_DETECT_BYTES` contain a NUL
/// byte, it is almost certainly a binary file (text editors and language
/// parsers agree on this convention).
fn is_binary_content(bytes: &[u8]) -> bool {
    let sample = &bytes[..bytes.len().min(BINARY_DETECT_BYTES)];
    sample.contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hidden_entry() {
        assert!(is_hidden_entry(Path::new(".git")));
        assert!(is_hidden_entry(Path::new(".env")));
        assert!(!is_hidden_entry(Path::new("src")));
        assert!(!is_hidden_entry(Path::new("file.rs")));
    }

    #[test]
    fn test_is_binary_content() {
        assert!(!is_binary_content(b"hello world\n"));
        assert!(is_binary_content(b"\x00hello"));
        assert!(is_binary_content(&[0xFF, 0xFE, 0x00, 0x01]));
    }

    #[test]
    fn test_format_grep_output() {
        let matches = vec![
            GrepMatch {
                rel_path: "src/main.rs".to_string(),
                line_num: 10,
                line_text: "fn main() {".to_string(),
            },
            GrepMatch {
                rel_path: "src/main.rs".to_string(),
                line_num: 15,
                line_text: "    println!(\"hello\");".to_string(),
            },
            GrepMatch {
                rel_path: "src/lib.rs".to_string(),
                line_num: 3,
                line_text: "pub fn greet() {".to_string(),
            },
        ];
        let output = format_grep_output(&matches);
        assert!(output.contains("src/main.rs:"));
        assert!(output.contains("src/lib.rs:"));
        assert!(output.contains("10: fn main() {"));
        assert!(output.contains("15:     println!(\"hello\");"));
    }
}
