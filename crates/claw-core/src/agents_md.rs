//! Loading `Agents.md`.
//!
//! The request explicitly asked that the minimal agent should read `Agents.md`
//! and use it as part of the "system" (or "developer") prompt.
//!
//! This module provides a small helper that:
//! - Reads `Agents.md` as UTF-8 text.
//! - Treats a missing file as "empty instructions" (but returns `found=false`
//!   so the caller can log a warning).

use std::path::{Path, PathBuf};

/// Result of reading `Agents.md`.
#[derive(Debug, Clone)]
pub struct AgentsMd {
    /// The path we attempted to read.
    pub path: PathBuf,
    /// The file contents (empty if missing).
    pub content: String,
    /// Whether the file existed and was read successfully.
    pub found: bool,
}

/// Load `Agents.md` from disk.
pub async fn load_agents_md(path: PathBuf) -> anyhow::Result<AgentsMd> {
    // If the file exists, read it.
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Ok(AgentsMd {
            path,
            content,
            found: true,
        }),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(AgentsMd {
            path,
            content: String::new(),
            found: false,
        }),
        Err(err) => Err(err.into()),
    }
}

/// Build the "instructions" block we inject into the prompt.
///
/// We keep this formatting stable so it is easy to debug and diff.
pub fn format_agents_md_block(agents_md: &AgentsMd) -> String {
    // If the file is missing, we still include a placeholder so the model
    // understands what happened.
    if !agents_md.found {
        return format!(
            "## Agents.md\n\n(Agents.md not found at `{}`)\n",
            agents_md.path.display()
        );
    }

    // Normal case: include the file content verbatim.
    format!(
        "## Agents.md\n\n(loaded from `{}`)\n\n{}\n",
        agents_md.path.display(),
        agents_md.content
    )
}

/// Helper: determine whether a path looks like `Agents.md`.
///
/// This is not used by the agent loop, but is handy for tests and future
/// enhancements (like auto-discovery).
pub fn is_agents_md_path(path: &Path) -> bool {
    // We check case-insensitively because some Windows setups are case-insensitive.
    let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    file_name.eq_ignore_ascii_case("Agents.md")
}

