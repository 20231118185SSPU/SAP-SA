//! Shared filesystem safety checks for `Read` / `Write` / `Edit`.
//!
//! SA already keeps paths inside the workspace root via
//! `ToolContext::resolve_under_workspace()`. That is necessary but not
//! sufficient. Claude Code also guards a second layer:
//! - suspicious raw path syntax that may hide shell expansion intent;
//! - dangerous control/configuration files that should not be edited by a
//!   general autonomous agent;
//! - internal runtime directories that should not be mutated through generic
//!   file tools.

use std::path::Path;

/// File-tool operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathOperation {
    /// Plain file read.
    Read,
    /// Create a new file.
    Write,
    /// Modify an existing file.
    Edit,
}

/// Dangerous directories that generic file tools must not mutate.
const DANGEROUS_DIRECTORIES: &[&str] = &[".git", ".vscode", ".idea"];

/// Dangerous file names that generic file tools must not mutate.
const DANGEROUS_FILE_NAMES: &[&str] = &[
    ".gitconfig",
    ".bashrc",
    ".bash_profile",
    ".profile",
    ".zshrc",
    ".zprofile",
    ".mcp.json",
    ".env",
    ".env.local",
    ".env.production",
    "sa.toml",
];

/// Root-level control files that define SA's own runtime behavior.
const ROOT_CONTROL_FILES: &[&str] = &["AGENTS.md", "SubAgents.md", "prompt.md", "compact.md"];

/// Validate the user-provided raw path string before any workspace resolution.
pub fn validate_tool_path_input(raw: &str, operation: PathOperation) -> anyhow::Result<()> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("Path must not be empty");
    }

    if trimmed.contains('\0') {
        anyhow::bail!("Path contains a NUL byte, which SA refuses to process");
    }

    if trimmed.starts_with("\\\\") || trimmed.starts_with("//") {
        anyhow::bail!("UNC/network paths are not allowed in SA file tools: {trimmed}");
    }

    if trimmed.starts_with('~') {
        anyhow::bail!("Shell-style `~` expansion is not allowed in SA file tools: {trimmed}");
    }

    if trimmed.starts_with('=') {
        anyhow::bail!(
            "Shell-style `=cmd` path expansion is not allowed in SA file tools: {trimmed}"
        );
    }

    if trimmed.contains('$') || contains_windows_env_expansion(trimmed) {
        anyhow::bail!(
            "Shell or environment-variable expansion syntax is not allowed in SA file tools: {trimmed}"
        );
    }

    if has_suspicious_windows_path_pattern(trimmed) {
        anyhow::bail!(
            "Path contains suspicious Windows-specific syntax that SA refuses to trust: {trimmed}"
        );
    }

    if matches!(operation, PathOperation::Write | PathOperation::Edit)
        && contains_glob_character(trimmed)
    {
        anyhow::bail!("Write/Edit paths must be literal paths, not globs: {trimmed}");
    }

    Ok(())
}

/// Validate the resolved canonical workspace path for the requested operation.
pub fn validate_resolved_tool_path(
    workspace_root: &Path,
    resolved_path: &Path,
    operation: PathOperation,
) -> anyhow::Result<()> {
    if !resolved_path.starts_with(workspace_root) {
        anyhow::bail!(
            "Resolved path escapes workspace root (root: {}, path: {})",
            workspace_root.display(),
            resolved_path.display()
        );
    }

    if matches!(operation, PathOperation::Read) {
        if path_contains_dangerous_directory(resolved_path) {
            anyhow::bail!(
                "SA blocks reading dangerous internal/config directories through generic file tools: {}",
                resolved_path.display()
            );
        }

        if resolved_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(is_dangerous_file_name)
            .unwrap_or(false)
        {
            anyhow::bail!(
                "SA blocks reading sensitive configuration files through generic file tools: {}",
                resolved_path.display()
            );
        }

        if is_internal_runtime_path(workspace_root, resolved_path) {
            anyhow::bail!(
                "SA blocks reading internal runtime/session data through generic file tools: {}",
                resolved_path.display()
            );
        }

        return Ok(());
    }

    if path_contains_dangerous_directory(resolved_path) {
        anyhow::bail!(
            "SA blocks generic file mutation inside dangerous directories: {}",
            resolved_path.display()
        );
    }

    if resolved_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(is_dangerous_file_name)
        .unwrap_or(false)
    {
        anyhow::bail!(
            "SA blocks generic file mutation for dangerous configuration files: {}",
            resolved_path.display()
        );
    }

    if is_root_control_file(workspace_root, resolved_path) {
        anyhow::bail!(
            "SA blocks generic file mutation for its own control files: {}",
            resolved_path.display()
        );
    }

    if is_internal_runtime_path(workspace_root, resolved_path) {
        anyhow::bail!(
            "SA blocks generic file mutation inside internal runtime directories: {}",
            resolved_path.display()
        );
    }

    Ok(())
}

/// Detect `%VAR%`-style expansion syntax used by `cmd.exe`.
fn contains_windows_env_expansion(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'%' {
            continue;
        }
        for end in (start + 1)..bytes.len() {
            if bytes[end] != b'%' {
                continue;
            }
            if end > start + 1 {
                return true;
            }
        }
    }
    false
}

/// Detect simple glob characters that generic file mutations should not accept.
fn contains_glob_character(raw: &str) -> bool {
    raw.chars().any(|ch| matches!(ch, '*' | '?' | '[' | ']'))
}

/// Detect suspicious Windows path syntax often used to bypass path filters.
fn has_suspicious_windows_path_pattern(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();

    if lower.starts_with(r"\\?\")
        || lower.starts_with(r"\\.\")
        || lower.starts_with("//?/")
        || lower.starts_with("//./")
    {
        return true;
    }

    let colon_index = raw
        .chars()
        .enumerate()
        .find_map(|(index, ch)| if ch == ':' { Some(index) } else { None });
    if let Some(index) = colon_index
        && index > 1
    {
        return true;
    }

    let bytes = raw.as_bytes();
    for index in 0..bytes.len().saturating_sub(1) {
        if bytes[index] == b'~' && bytes[index + 1].is_ascii_digit() {
            return true;
        }
    }

    raw.split(['/', '\\']).any(|segment| {
        let trimmed = segment.trim_end_matches(['.', ' ']);
        trimmed.len() != segment.len()
    })
}

/// Detect whether the canonical path traverses a protected directory name.
fn path_contains_dangerous_directory(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .map(|segment| {
                DANGEROUS_DIRECTORIES
                    .iter()
                    .any(|dangerous| segment.eq_ignore_ascii_case(dangerous))
            })
            .unwrap_or(false)
    })
}

/// Detect whether the file name is a protected configuration file.
fn is_dangerous_file_name(name: &str) -> bool {
    DANGEROUS_FILE_NAMES
        .iter()
        .any(|dangerous| name.eq_ignore_ascii_case(dangerous))
}

/// Detect whether the path points at one of SA's own root control files.
fn is_root_control_file(workspace_root: &Path, resolved_path: &Path) -> bool {
    let Some(parent) = resolved_path.parent() else {
        return false;
    };
    if parent != workspace_root {
        return false;
    }

    resolved_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            ROOT_CONTROL_FILES
                .iter()
                .any(|file| name.eq_ignore_ascii_case(file))
        })
        .unwrap_or(false)
}

/// Detect whether the path points into SA's internal runtime directories.
fn is_internal_runtime_path(workspace_root: &Path, resolved_path: &Path) -> bool {
    [
        workspace_root.join("runtime"),
        workspace_root.join("sessions"),
        workspace_root.join("interactions"),
    ]
    .iter()
    .any(|prefix| resolved_path == prefix || resolved_path.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::{PathOperation, validate_resolved_tool_path, validate_tool_path_input};
    use tempfile::TempDir;

    /// Build one temporary workspace root for path-guard tests.
    fn workspace() -> TempDir {
        tempfile::tempdir().expect("temp workspace")
    }

    #[test]
    fn raw_path_rejects_unc() {
        let err = validate_tool_path_input(r"\\server\share\file.txt", PathOperation::Read)
            .expect_err("UNC path must fail");
        assert!(err.to_string().contains("UNC"));
    }

    #[test]
    fn raw_path_rejects_env_expansion() {
        let err = validate_tool_path_input("$HOME/file.txt", PathOperation::Read)
            .expect_err("shell expansion must fail");
        assert!(err.to_string().contains("expansion"));
    }

    #[test]
    fn raw_path_rejects_glob_for_write() {
        let err = validate_tool_path_input("notes/*.md", PathOperation::Write)
            .expect_err("glob write path must fail");
        assert!(err.to_string().contains("literal"));
    }

    #[test]
    fn resolved_path_rejects_git_mutation() {
        let workspace = workspace();
        let path = workspace.path().join(".git").join("config");
        let err = validate_resolved_tool_path(workspace.path(), &path, PathOperation::Edit)
            .expect_err(".git mutation must fail");
        assert!(err.to_string().contains("dangerous directories"));
    }

    #[test]
    fn resolved_path_rejects_sensitive_read() {
        let workspace = workspace();
        let path = workspace.path().join("sa.toml");
        let err = validate_resolved_tool_path(workspace.path(), &path, PathOperation::Read)
            .expect_err("sensitive read must fail");
        assert!(err.to_string().contains("sensitive configuration files"));
    }

    #[test]
    fn resolved_path_rejects_runtime_read() {
        let workspace = workspace();
        let path = workspace.path().join("sessions").join("current.jsonl");
        let err = validate_resolved_tool_path(workspace.path(), &path, PathOperation::Read)
            .expect_err("runtime read must fail");
        assert!(err.to_string().contains("runtime/session data"));
    }

    #[test]
    fn resolved_path_rejects_root_control_file_mutation() {
        let workspace = workspace();
        let path = workspace.path().join("prompt.md");
        let err = validate_resolved_tool_path(workspace.path(), &path, PathOperation::Edit)
            .expect_err("prompt.md mutation must fail");
        assert!(err.to_string().contains("control files"));
    }

    #[test]
    fn resolved_path_rejects_runtime_mutation() {
        let workspace = workspace();
        let path = workspace
            .path()
            .join("runtime")
            .join("tasks")
            .join("task.log");
        let err = validate_resolved_tool_path(workspace.path(), &path, PathOperation::Write)
            .expect_err("runtime mutation must fail");
        assert!(err.to_string().contains("runtime directories"));
    }
}
