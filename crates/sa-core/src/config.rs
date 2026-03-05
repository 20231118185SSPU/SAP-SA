//! Configuration loading for StudyAdministrator (SA).
//!
//! The user request explicitly asked for:
//! - A TOML configuration file.
//! - Configurable `base_url`, `api_key`, `model`, etc.
//! - A special option that controls the **role name** used for the "system"
//!   instructions message:
//!   - If not set, default to `"system"`.
//!   - Some OpenAI-compatible providers reject `role: "system"` and only accept
//!     `role: "developer"`; the config makes this adjustable.

use anyhow::Context as _;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Root configuration object (maps to the full `sa.toml`).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// LLM/provider configuration (`[llm]`).
    pub llm: LlmConfig,

    /// WebSocket daemon configuration (`[server]`).
    pub server: ServerConfig,

    /// Workspace configuration (`[workspace]`).
    pub workspace: WorkspaceConfig,

    /// Skills discovery configuration (`[skills]`).
    #[serde(default)]
    pub skills: SkillsConfig,
}

/// LLM/provider configuration (`[llm]` section).
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// OpenAI-compatible base URL (example: `https://example.com/v1`).
    pub base_url: String,

    /// OpenAI-compatible API key (sent as `Authorization: Bearer ...`).
    pub api_key: String,

    /// Model name (example: `gpt-5.2`).
    pub model: String,

    /// Role name used for the "system instructions" message.
    ///
    /// - Missing / empty => `"system"`.
    /// - Common override => `"developer"`.
    #[serde(default)]
    pub system_role_name: Option<String>,

    /// Maximum tool-calling steps per task (safety cap).
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,
}

/// Default maximum steps.
///
/// We keep it reasonably small so the agent cannot "run forever" by default.
fn default_max_steps() -> u32 {
    32
}

impl LlmConfig {
    /// Return the effective role name for "system instructions".
    pub fn effective_system_role_name(&self) -> &str {
        // If missing => "system".
        let Some(raw) = self.system_role_name.as_deref() else {
            return "system";
        };

        // If present but blank => treat as unset.
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return "system";
        }

        // Otherwise use the user-provided role name (e.g. "developer").
        trimmed
    }
}

/// WebSocket daemon configuration (`[server]` section).
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Listen address (example: `127.0.0.1:8765`).
    pub bind: String,

    /// WebSocket path (example: `/ws`).
    pub ws_path: String,
}

/// Workspace configuration (`[workspace]` section).
#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceConfig {
    /// The directory the agent is allowed to operate in.
    pub root_dir: String,

    /// The instruction file to read and inject into the prompt.
    pub agents_md: String,
}

impl WorkspaceConfig {
    /// Resolve the configured root directory into an absolute path.
    pub fn root_dir_path(&self, config_file_dir: &Path) -> PathBuf {
        // If the user configured an absolute path, keep it.
        let raw = PathBuf::from(&self.root_dir);
        if raw.is_absolute() {
            return raw;
        }

        // Otherwise interpret it as relative to the config file's directory.
        config_file_dir.join(raw)
    }

    /// Resolve the `Agents.md` path (relative to `root_dir` if needed).
    pub fn agents_md_path(&self, workspace_root: &Path) -> PathBuf {
        // `agents_md` is typically `Agents.md` at the workspace root.
        let raw = PathBuf::from(&self.agents_md);
        if raw.is_absolute() {
            return raw;
        }
        workspace_root.join(raw)
    }
}

/// Skills discovery configuration (`[skills]` section).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SkillsConfig {
    /// Directories to scan for installed skills.
    ///
    /// Each skill is expected to be a directory containing a `SKILL.md`.
    #[serde(default)]
    pub dirs: Vec<String>,
}

impl SkillsConfig {
    /// Resolve the configured skill directories into paths.
    ///
    /// This expands `~` to the user home directory (best-effort).
    pub fn dirs_as_paths(&self) -> Vec<PathBuf> {
        self.dirs.iter().map(|raw| expand_tilde(raw)).collect()
    }
}

/// Load and parse `sa.toml`.
///
/// We deliberately load from an explicit path so:
/// - The daemon and CLI can share the same config loading logic.
/// - The caller can choose a project-local `sa.toml`.
pub fn load_config_from_file(path: &Path) -> anyhow::Result<Config> {
    // Read the file first so parse errors have a stable "source of truth".
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    // Parse TOML into typed config.
    toml::from_str::<Config>(&raw)
        .with_context(|| format!("Failed to parse TOML config file: {}", path.display()))
}

/// Best-effort expansion of `~` (tilde) to the current user's home directory.
///
/// This is useful for configs like `~/.codex/skills`.
pub fn expand_tilde(path: &str) -> PathBuf {
    // We only treat a leading `~` as special. Anything else is returned as-is.
    let Some(rest) = path.strip_prefix('~') else {
        return PathBuf::from(path);
    };

    // On Unix, `HOME` is standard. On Windows, `USERPROFILE` is common.
    // If neither is set, we fall back to the raw path so callers get a useful
    // error later.
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    let Ok(home) = home else {
        return PathBuf::from(path);
    };

    // `~` alone means "home", `~/...` means "home/...".
    let home = PathBuf::from(home);
    if rest.is_empty() {
        return home;
    }

    // Strip leading path separators from the remainder to avoid `join`
    // treating it as an absolute path.
    let rest = rest.trim_start_matches(['/', '\\']);
    home.join(rest)
}
