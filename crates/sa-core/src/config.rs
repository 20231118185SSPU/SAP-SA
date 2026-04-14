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

use crate::compact::CompactionConfig;
use crate::dream::DreamConfig;
use crate::openai::{AuthStyle, WireApi};
use anyhow::Context as _;
use serde::de::Error as SerdeError;
use serde::{Deserialize, Deserializer};
use std::collections::{BTreeMap, HashMap};
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

    /// Long-session history compaction configuration (`[compaction]`).
    #[serde(default)]
    pub compaction: CompactionConfig,

    /// Nightly dream-memory distillation configuration (`[dream]`).
    #[serde(default)]
    pub dream: DreamConfig,

    /// External MCP server configuration (`[mcp]`).
    #[serde(default)]
    pub mcp: McpConfig,

    /// Non-interactive permission policy (`[permissions]`).
    #[serde(default)]
    pub permissions: PermissionsConfig,

    /// Durable multi-agent runtime configuration (`[team]`).
    #[serde(default)]
    pub team: TeamConfig,
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

    /// Wire protocol used when talking to the OpenAI-compatible provider.
    ///
    /// Supported values:
    /// - `chat_completions`
    /// - `responses`
    /// - `anthropic_messages`
    ///
    /// Alias spellings such as `chat-completions`, `chat`, `anthropic`, and
    /// `claude` are also accepted.
    ///
    /// If this field is absent, SA keeps the historical default:
    /// `chat_completions`.
    #[serde(default)]
    pub wire_api: Option<WireApi>,

    /// Authentication style used for outbound provider calls.
    ///
    /// Common values:
    /// - `bearer`
    /// - `x_api_key`
    /// - `anthropic_auto`
    ///
    /// If omitted:
    /// - OpenAI-compatible protocols default to `bearer`
    /// - `anthropic_messages` defaults to `anthropic_auto`
    #[serde(default)]
    pub auth_style: Option<AuthStyle>,

    /// Role name used for the "system instructions" message.
    ///
    /// - Missing / empty => `"system"`.
    /// - Common override => `"developer"`.
    #[serde(default)]
    pub system_role_name: Option<String>,

    /// Optional reasoning depth / effort passed through to compatible GPT
    /// models.
    ///
    /// Common values seen on GPT-family reasoning models include:
    /// - `none`
    /// - `minimal`
    /// - `low`
    /// - `medium`
    /// - `high`
    /// - `xhigh`
    ///
    /// We intentionally keep this as a free-form string because:
    /// - different OpenAI-compatible providers may expose different subsets
    /// - future providers may add new values
    ///
    /// If this field is absent or blank, we omit it from the request body.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Durable multi-agent runtime configuration (`[team]` section).
#[derive(Debug, Clone, Deserialize)]
pub struct TeamConfig {
    /// Whether unfinished work should be resumed automatically on boot.
    #[serde(default = "default_team_auto_resume")]
    pub auto_resume: bool,

    /// Soft ceiling for concurrently tracked live agents.
    #[serde(default = "default_team_max_active_agents")]
    pub max_active_agents: usize,

    /// Maximum concurrent model calls shared by all agents.
    #[serde(default = "default_team_max_concurrent_model_calls")]
    pub max_concurrent_model_calls: usize,
}

/// High-level permission mode.
///
/// SA currently keeps the model in a fully non-interactive posture, so the
/// practical behavior is:
/// - `bypass`: no approval prompts; only explicit deny rules and command-scope
///   allowlists apply
///
/// The enum still exists so the config format remains explicit and can grow
/// later without another breaking TOML change.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// No approval UI. Global deny rules and command-scoped restrictions still
    /// apply.
    #[default]
    Bypass,
}

/// Non-interactive permission controls (`[permissions]` section).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PermissionsConfig {
    /// High-level mode. Defaults to `bypass`.
    #[serde(default)]
    pub mode: PermissionMode,

    /// Glob-style tool deny patterns applied before every execution.
    ///
    /// Examples:
    /// - `Bash`
    /// - `mcp__playwright__*`
    #[serde(default)]
    pub deny_tools: Vec<String>,

    /// Glob-style command/skill deny patterns.
    ///
    /// These hide matching commands from the prompt and also block
    /// `Skill(action="read"|"invoke")` for those names.
    #[serde(default)]
    pub deny_commands: Vec<String>,
}

impl Default for TeamConfig {
    fn default() -> Self {
        Self {
            auto_resume: default_team_auto_resume(),
            max_active_agents: default_team_max_active_agents(),
            max_concurrent_model_calls: default_team_max_concurrent_model_calls(),
        }
    }
}

/// Default auto-resume policy for the durable runtime.
const fn default_team_auto_resume() -> bool {
    true
}

/// Default maximum number of live agents.
const fn default_team_max_active_agents() -> usize {
    4096
}

/// Default maximum number of concurrent model calls.
const fn default_team_max_concurrent_model_calls() -> usize {
    8
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

    /// Return the effective reasoning effort, if configured.
    pub fn effective_reasoning_effort(&self) -> Option<&str> {
        let raw = self.reasoning_effort.as_deref()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(trimmed)
    }

    /// Return the effective OpenAI-compatible wire protocol.
    pub fn effective_wire_api(&self) -> WireApi {
        self.wire_api.unwrap_or_default()
    }

    /// Return the effective authentication style for the selected wire
    /// protocol.
    pub fn effective_auth_style(&self, wire_api: WireApi) -> AuthStyle {
        self.auth_style.unwrap_or(match wire_api {
            WireApi::AnthropicMessages => AuthStyle::AnthropicAuto,
            WireApi::ChatCompletions | WireApi::Responses => AuthStyle::Bearer,
        })
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

/// Transport type for MCP server connections.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    /// Spawn a local process and communicate over stdin/stdout.
    #[default]
    Stdio,
    /// Connect via HTTP POST.
    Http,
    /// Connect via HTTP + Server-Sent Events.
    Sse,
}

/// Configuration for one external MCP server.
#[derive(Debug, Clone, Default)]
pub struct McpServerConfig {
    /// Display name used as the tool prefix (`<server>__<tool>`).
    pub name: String,
    /// Transport type.
    pub transport: McpTransport,
    /// URL for HTTP/SSE transports.
    pub url: Option<String>,
    /// Executable for stdio transport.
    pub command: String,
    /// Arguments for stdio transport.
    pub args: Vec<String>,
    /// Optional stdio child working directory.
    ///
    /// Relative values from `sa.toml` are resolved against the config file
    /// directory during config loading, so runtime code receives the final
    /// absolute path.
    pub cwd: Option<PathBuf>,
    /// Extra environment variables for stdio transport.
    pub env: std::collections::HashMap<String, String>,
    /// Extra HTTP headers for HTTP/SSE transports.
    pub headers: std::collections::HashMap<String, String>,
    /// Optional per-call timeout in seconds.
    pub tool_timeout_secs: Option<u64>,
}

/// External MCP client configuration (`[mcp]`).
///
/// Supported format:
/// - `[mcp]`
/// - `[mcp.<server_name>]`
#[derive(Debug, Clone, Default)]
pub struct McpConfig {
    /// Whether MCP support is enabled.
    pub enabled: bool,
    /// Configured MCP servers.
    pub servers: Vec<McpServerConfig>,
}

impl<'de> Deserialize<'de> for McpServerConfig {
    /// Parse one MCP server entry.
    ///
    /// Supported style:
    /// - named-table item under `[mcp.<name>]`
    ///
    /// Transport behavior:
    /// - if `transport` is present, we obey it
    /// - otherwise:
    ///   - `command` => `stdio`
    ///   - `url` => `http`
    /// - `sse` remains available via explicit `transport = "sse"`
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawMcpServerConfig::deserialize(deserializer)?;

        let name = raw.name.unwrap_or_default().trim().to_string();
        let command = raw.command.unwrap_or_default();
        let url = raw.url.unwrap_or_default();
        let cwd = raw
            .cwd
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let has_command = !command.trim().is_empty();
        let has_url = !url.trim().is_empty();

        if has_command && has_url {
            return Err(SerdeError::custom(
                "MCP server config must not set both `command` and `url`",
            ));
        }

        let transport = match raw.transport {
            Some(transport) => transport,
            None if has_command => McpTransport::Stdio,
            None if has_url => McpTransport::Http,
            None => {
                return Err(SerdeError::custom(
                    "MCP server config must set either `command` or `url`",
                ));
            }
        };

        match transport {
            McpTransport::Stdio if !has_command => {
                return Err(SerdeError::custom(
                    "MCP server with transport=stdio requires non-empty `command`",
                ));
            }
            McpTransport::Http | McpTransport::Sse if !has_url => {
                return Err(SerdeError::custom(format!(
                    "MCP server with transport={} requires non-empty `url`",
                    match transport {
                        McpTransport::Http => "http",
                        McpTransport::Sse => "sse",
                        McpTransport::Stdio => "stdio",
                    }
                )));
            }
            _ => {}
        }

        Ok(Self {
            name,
            transport,
            url: has_url.then_some(url),
            command,
            args: raw.args,
            cwd,
            env: raw.env,
            headers: raw.headers,
            tool_timeout_secs: raw.tool_timeout_secs,
        })
    }
}

impl<'de> Deserialize<'de> for McpConfig {
    /// Parse `[mcp]`.
    ///
    /// Accepted input form:
    /// - `[mcp]`
    /// - `[mcp.filesystem]`
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawMcpConfig::deserialize(deserializer)?;
        let mut servers = Vec::<McpServerConfig>::new();

        for (table_name, mut server) in raw.named_servers {
            let table_name = table_name.trim().to_string();
            if table_name.is_empty() {
                return Err(SerdeError::custom("MCP named table key must not be empty"));
            }

            if server.name.trim().is_empty() {
                server.name = table_name;
            } else if server.name.trim() != table_name {
                return Err(SerdeError::custom(format!(
                    "MCP named table `[mcp.{table_name}]` conflicts with inline name `{}`",
                    server.name.trim()
                )));
            }

            servers.push(server);
        }

        Ok(Self {
            enabled: raw.enabled,
            servers,
        })
    }
}

/// Raw single-server shape accepted in `sa.toml`.
///
/// We keep this separate from [`McpServerConfig`] because:
/// - `[mcp.<name>]` tables do not need an inline `name`
/// - `transport` is now optional and may be inferred
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct RawMcpServerConfig {
    /// Optional explicit display name.
    name: Option<String>,
    /// Optional explicit transport selector.
    transport: Option<McpTransport>,
    /// URL for HTTP/SSE transports.
    url: Option<String>,
    /// Executable for stdio transport.
    command: Option<String>,
    /// Arguments for stdio transport.
    args: Vec<String>,
    /// Optional stdio child working directory.
    cwd: Option<String>,
    /// Extra environment variables for stdio transport.
    env: HashMap<String, String>,
    /// Extra HTTP headers for HTTP/SSE transports.
    headers: HashMap<String, String>,
    /// Optional per-call timeout in seconds.
    tool_timeout_secs: Option<u64>,
}

/// Raw `[mcp]` table shape accepted in `sa.toml`.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct RawMcpConfig {
    /// Whether MCP support is enabled.
    enabled: bool,
    /// Named-table format: `[mcp.<name>]`.
    #[serde(flatten)]
    named_servers: BTreeMap<String, McpServerConfig>,
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
    let mut config = toml::from_str::<Config>(&raw)
        .with_context(|| format!("Failed to parse TOML config file: {}", path.display()))?;
    let config_file_dir = resolve_config_file_dir(path)?;
    normalize_skill_config_paths(&mut config.skills, &config_file_dir)?;
    normalize_mcp_config_paths(&mut config.mcp, &config_file_dir)?;
    validate_mcp_config(&config.mcp)?;
    Ok(config)
}

/// Resolve the directory containing one config file path.
///
/// Important edge case:
/// - `Path::new("sa.toml").parent()` may behave like an empty path on some
///   platforms
/// - `std::path::absolute("")` then fails with
///   "cannot make an empty path absolute"
///
/// So a bare relative filename must be treated as `./sa.toml`, whose parent is
/// the current working directory.
pub fn resolve_config_file_dir(path: &Path) -> anyhow::Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    std::path::absolute(parent).with_context(|| {
        format!(
            "Failed to resolve config file directory for {}",
            path.display()
        )
    })
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

/// Hard safety ceiling for MCP per-tool call timeouts.
const MCP_MAX_TOOL_TIMEOUT_SECS: u64 = 600;

/// Resolve all skill scan directories into their final runtime form.
///
/// We intentionally allow nonexistent paths because users may preconfigure a
/// directory before installing any skills there; the scanner already treats
/// missing directories as empty.
fn normalize_skill_config_paths(
    config: &mut SkillsConfig,
    config_file_dir: &Path,
) -> anyhow::Result<()> {
    for dir in &mut config.dirs {
        let expanded = expand_tilde(dir);
        let resolved = resolve_config_relative_path(&expanded, config_file_dir)
            .with_context(|| format!("failed to resolve skills.dir `{dir}`"))?;
        *dir = resolved.display().to_string();
    }

    Ok(())
}

/// Resolve all MCP filesystem paths into their final runtime form.
///
/// We do this during config loading so startup fails early with a precise
/// error instead of deferring path issues to MCP connection time.
fn normalize_mcp_config_paths(
    config: &mut McpConfig,
    config_file_dir: &Path,
) -> anyhow::Result<()> {
    for server in &mut config.servers {
        let Some(raw_cwd) = server.cwd.as_ref() else {
            continue;
        };

        let resolved_cwd =
            resolve_config_relative_path(raw_cwd, config_file_dir).with_context(|| {
                format!(
                    "failed to resolve working directory for MCP server `{}`",
                    server.name
                )
            })?;

        if !resolved_cwd.exists() {
            anyhow::bail!(
                "mcp.{}.cwd does not exist: {}",
                server.name,
                resolved_cwd.display()
            );
        }
        if !resolved_cwd.is_dir() {
            anyhow::bail!(
                "mcp.{}.cwd must point to a directory: {}",
                server.name,
                resolved_cwd.display()
            );
        }

        server.cwd = Some(resolved_cwd);
    }

    Ok(())
}

/// Resolve one `sa.toml` path against the directory containing that config
/// file.
fn resolve_config_relative_path(path: &Path, config_file_dir: &Path) -> anyhow::Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        config_file_dir.join(path)
    };

    std::path::absolute(&candidate).with_context(|| {
        format!(
            "failed to resolve config-relative path `{}`",
            candidate.display()
        )
    })
}

/// Validate MCP configuration early so startup errors are explicit.
fn validate_mcp_config(config: &McpConfig) -> anyhow::Result<()> {
    let mut seen_names = std::collections::HashSet::<String>::new();

    for server in &config.servers {
        let name = server.name.trim();
        if name.is_empty() {
            anyhow::bail!("MCP server name must not be empty; use `[mcp.<name>]`");
        }
        if !seen_names.insert(name.to_ascii_lowercase()) {
            anyhow::bail!("mcp contains duplicate server name: {name}");
        }

        let location = format!("mcp.{name}");

        if let Some(timeout) = server.tool_timeout_secs {
            if timeout == 0 {
                anyhow::bail!("{location}.tool_timeout_secs must be greater than 0");
            }
            if timeout > MCP_MAX_TOOL_TIMEOUT_SECS {
                anyhow::bail!(
                    "{location}.tool_timeout_secs exceeds max {MCP_MAX_TOOL_TIMEOUT_SECS}"
                );
            }
        }

        match server.transport {
            McpTransport::Stdio => {
                if server.command.trim().is_empty() {
                    anyhow::bail!("{location} with transport=stdio requires non-empty command");
                }
                if let Some(cwd) = &server.cwd {
                    if !cwd.is_absolute() {
                        anyhow::bail!("{location}.cwd must be an absolute path");
                    }
                    if !cwd.exists() {
                        anyhow::bail!("{location}.cwd does not exist: {}", cwd.display());
                    }
                    if !cwd.is_dir() {
                        anyhow::bail!(
                            "{location}.cwd must point to a directory: {}",
                            cwd.display()
                        );
                    }
                }
            }
            McpTransport::Http | McpTransport::Sse => {
                if server.cwd.is_some() {
                    anyhow::bail!("{location}.cwd is only supported for transport=stdio");
                }
                let url = server
                    .url
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{location} with transport={} requires url",
                            match server.transport {
                                McpTransport::Http => "http",
                                McpTransport::Sse => "sse",
                                McpTransport::Stdio => "stdio",
                            }
                        )
                    })?;
                let parsed = reqwest::Url::parse(url)
                    .with_context(|| format!("{location}.url is not a valid URL"))?;
                if !matches!(parsed.scheme(), "http" | "https") {
                    anyhow::bail!("{location}.url must use http/https");
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Config, LlmConfig, McpConfig, McpServerConfig, McpTransport, PermissionMode, TeamConfig,
        load_config_from_file, resolve_config_file_dir, validate_mcp_config,
    };
    use crate::openai::{AuthStyle, WireApi};
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn sample_llm() -> LlmConfig {
        LlmConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: "sk-test".to_string(),
            model: "gpt-5.2".to_string(),
            wire_api: None,
            auth_style: None,
            system_role_name: None,
            reasoning_effort: None,
        }
    }

    fn sample_team() -> TeamConfig {
        TeamConfig::default()
    }

    #[test]
    fn effective_wire_api_defaults_to_chat_completions() {
        let cfg = sample_llm();
        assert_eq!(cfg.effective_wire_api(), WireApi::ChatCompletions);
    }

    #[test]
    fn llm_wire_api_aliases_deserialize() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"
wire_api = "chat"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"
"#;

        let cfg = toml::from_str::<Config>(raw).expect("wire_api alias should parse");
        assert_eq!(cfg.llm.effective_wire_api(), WireApi::ChatCompletions);

        let raw = raw.replace("wire_api = \"chat\"", "wire_api = \"responses\"");
        let cfg = toml::from_str::<Config>(&raw).expect("responses wire_api should parse");
        assert_eq!(cfg.llm.effective_wire_api(), WireApi::Responses);

        let raw = raw.replace("wire_api = \"responses\"", "wire_api = \"claude\"");
        let cfg = toml::from_str::<Config>(&raw).expect("claude wire_api alias should parse");
        assert_eq!(cfg.llm.effective_wire_api(), WireApi::AnthropicMessages);
    }

    #[test]
    fn effective_auth_style_defaults_follow_wire_api() {
        let cfg = sample_llm();
        assert_eq!(
            cfg.effective_auth_style(WireApi::ChatCompletions),
            AuthStyle::Bearer
        );
        assert_eq!(
            cfg.effective_auth_style(WireApi::Responses),
            AuthStyle::Bearer
        );
        assert_eq!(
            cfg.effective_auth_style(WireApi::AnthropicMessages),
            AuthStyle::AnthropicAuto
        );
    }

    #[test]
    fn auth_style_aliases_deserialize() {
        let raw = r#"
[llm]
base_url = "https://example.com"
api_key = "sk-test"
model = "claude-sonnet-4-5"
wire_api = "anthropic_messages"
auth_style = "x-api-key"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"
"#;

        let cfg = toml::from_str::<Config>(raw).expect("auth_style alias should parse");
        assert_eq!(cfg.llm.auth_style, Some(AuthStyle::XApiKey));
    }

    #[test]
    fn effective_reasoning_effort_is_none_when_unset_or_blank() {
        let cfg = sample_llm();
        assert_eq!(cfg.effective_reasoning_effort(), None);

        let mut cfg = sample_llm();
        cfg.reasoning_effort = Some("   ".to_string());
        assert_eq!(cfg.effective_reasoning_effort(), None);
    }

    #[test]
    fn effective_reasoning_effort_trims_value() {
        let mut cfg = sample_llm();
        cfg.reasoning_effort = Some("  xhigh  ".to_string());
        assert_eq!(cfg.effective_reasoning_effort(), Some("xhigh"));
    }

    #[test]
    fn team_defaults_match_expected_runtime_policy() {
        let cfg = sample_team();
        assert!(cfg.auto_resume);
        assert_eq!(cfg.max_active_agents, 4096);
        assert_eq!(cfg.max_concurrent_model_calls, 8);
    }

    #[test]
    fn team_defaults_load_when_section_is_missing() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"
"#;

        let cfg = toml::from_str::<Config>(raw).expect("config without team section should parse");
        assert!(cfg.team.auto_resume);
        assert_eq!(cfg.team.max_active_agents, 4096);
        assert_eq!(cfg.team.max_concurrent_model_calls, 8);
    }

    #[test]
    fn permissions_defaults_load_when_section_is_missing() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"
"#;

        let cfg =
            toml::from_str::<Config>(raw).expect("config without permissions section should parse");
        assert_eq!(cfg.permissions.mode, PermissionMode::Bypass);
        assert!(cfg.permissions.deny_tools.is_empty());
        assert!(cfg.permissions.deny_commands.is_empty());
    }

    #[test]
    fn permissions_section_parses_explicit_deny_lists() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[permissions]
mode = "bypass"
deny_tools = ["Show", "mcp__playwright__*"]
deny_commands = ["dangerous-*", "mcp__internal__*"]
"#;

        let cfg = toml::from_str::<Config>(raw).expect("permissions section should parse");
        assert_eq!(cfg.permissions.mode, PermissionMode::Bypass);
        assert_eq!(
            cfg.permissions.deny_tools,
            vec!["Show".to_string(), "mcp__playwright__*".to_string()]
        );
        assert_eq!(
            cfg.permissions.deny_commands,
            vec!["dangerous-*".to_string(), "mcp__internal__*".to_string()]
        );
    }

    #[test]
    fn team_partial_override_preserves_other_defaults() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[team]
max_concurrent_model_calls = 3
"#;

        let cfg =
            toml::from_str::<Config>(raw).expect("config with partial team override should parse");
        assert!(cfg.team.auto_resume);
        assert_eq!(cfg.team.max_active_agents, 4096);
        assert_eq!(cfg.team.max_concurrent_model_calls, 3);
    }

    #[test]
    fn compaction_defaults_load_when_section_is_missing() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"
"#;

        let cfg =
            toml::from_str::<Config>(raw).expect("config without compaction section should parse");
        assert!(cfg.compaction.enabled);
        assert_eq!(cfg.compaction.trigger_tokens, 180_000);
        assert_eq!(cfg.compaction.keep_recent_tokens, 8_000);
        assert_eq!(cfg.compaction.reserve_summary_tokens, 4_096);
        assert_eq!(cfg.compaction.min_messages_to_compact, 8);
        assert!(cfg.dream.enabled);
        assert_eq!(cfg.dream.daily_note_lookback_days, 3);
        assert_eq!(cfg.dream.recent_session_segments, 6);
        assert_eq!(cfg.dream.recent_topic_files, 24);
    }

    #[test]
    fn compaction_partial_override_preserves_other_defaults() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[compaction]
trigger_tokens = 12345
keep_recent_tokens = 6789
"#;

        let cfg = toml::from_str::<Config>(raw)
            .expect("config with partial compaction override should parse");
        assert!(cfg.compaction.enabled);
        assert_eq!(cfg.compaction.trigger_tokens, 12_345);
        assert_eq!(cfg.compaction.keep_recent_tokens, 6_789);
        assert_eq!(cfg.compaction.reserve_summary_tokens, 4_096);
        assert_eq!(cfg.compaction.min_messages_to_compact, 8);
    }

    #[test]
    fn dream_partial_override_preserves_other_defaults() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[dream]
enabled = true
recent_session_segments = 9
"#;

        let cfg =
            toml::from_str::<Config>(raw).expect("config with partial dream override should parse");
        assert!(cfg.dream.enabled);
        assert_eq!(cfg.dream.daily_note_lookback_days, 3);
        assert_eq!(cfg.dream.recent_session_segments, 9);
        assert_eq!(cfg.dream.recent_topic_files, 24);
    }

    #[test]
    fn mcp_validation_rejects_duplicate_server_names() {
        let config = McpConfig {
            enabled: true,
            servers: vec![
                McpServerConfig {
                    name: "fs".to_string(),
                    transport: McpTransport::Stdio,
                    url: None,
                    command: "server-a".to_string(),
                    args: Vec::new(),
                    cwd: None,
                    env: HashMap::new(),
                    headers: HashMap::new(),
                    tool_timeout_secs: None,
                },
                McpServerConfig {
                    name: "FS".to_string(),
                    transport: McpTransport::Stdio,
                    url: None,
                    command: "server-b".to_string(),
                    args: Vec::new(),
                    cwd: None,
                    env: HashMap::new(),
                    headers: HashMap::new(),
                    tool_timeout_secs: None,
                },
            ],
        };

        let err = validate_mcp_config(&config).expect_err("duplicate names must fail");
        assert!(err.to_string().contains("duplicate server name"));
    }

    #[test]
    fn mcp_named_table_format_parses_and_infers_stdio_transport() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp]
enabled = true

[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
cwd = "./tooling"
tool_timeout_secs = 180
"#;

        let cfg = toml::from_str::<Config>(raw).expect("named mcp table should parse");
        assert!(cfg.mcp.enabled);
        assert_eq!(cfg.mcp.servers.len(), 1);
        let server = &cfg.mcp.servers[0];
        assert_eq!(server.name, "filesystem");
        assert_eq!(server.transport, McpTransport::Stdio);
        assert_eq!(server.command, "npx");
        assert_eq!(
            server.cwd.as_deref(),
            Some(std::path::Path::new("./tooling"))
        );
        assert_eq!(server.tool_timeout_secs, Some(180));
    }

    #[test]
    fn mcp_named_table_format_parses_and_infers_http_transport() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp.remote]
url = "https://example.com/mcp"
"#;

        let cfg = toml::from_str::<Config>(raw).expect("named HTTP mcp table should parse");
        assert_eq!(cfg.mcp.servers.len(), 1);
        let server = &cfg.mcp.servers[0];
        assert_eq!(server.name, "remote");
        assert_eq!(server.transport, McpTransport::Http);
        assert_eq!(server.url.as_deref(), Some("https://example.com/mcp"));
    }

    #[test]
    fn mcp_legacy_array_format_is_rejected() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp]
enabled = true

[[mcp.servers]]
name = "legacy"
transport = "stdio"
command = "legacy-mcp"
"#;

        let err = toml::from_str::<Config>(raw).expect_err("legacy mcp array should fail");
        assert!(
            err.to_string().contains("servers")
                || err.to_string().contains("expected")
                || err.to_string().contains("unknown field")
        );
    }

    #[test]
    fn mcp_named_table_rejects_inline_name_mismatch() {
        let raw = r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp.docs]
name = "other"
command = "mcp-docs"
"#;

        let err = toml::from_str::<Config>(raw).expect_err("mismatched inline name must fail");
        assert!(err.to_string().contains("conflicts with inline name"));
    }

    #[test]
    fn mcp_validation_rejects_missing_stdio_command() {
        let config = McpConfig {
            enabled: true,
            servers: vec![McpServerConfig {
                name: "fs".to_string(),
                transport: McpTransport::Stdio,
                url: None,
                command: "   ".to_string(),
                args: Vec::new(),
                cwd: None,
                env: HashMap::new(),
                headers: HashMap::new(),
                tool_timeout_secs: None,
            }],
        };

        let err = validate_mcp_config(&config).expect_err("stdio command must be required");
        assert!(err.to_string().contains("requires non-empty command"));
    }

    #[test]
    fn load_config_resolves_mcp_cwd_relative_to_config_file_dir() {
        let temp = TempDir::new().expect("temp dir should be created");
        let config_dir = temp.path().join("config");
        let mcp_cwd = config_dir.join("mcp-workdir");
        fs::create_dir_all(&mcp_cwd).expect("mcp cwd should be created");

        let config_path = config_dir.join("sa.toml");
        fs::create_dir_all(&config_dir).expect("config dir should be created");
        fs::write(
            &config_path,
            r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp.playwright]
command = "npx"
args = ["@playwright/mcp@latest"]
cwd = "mcp-workdir"
"#,
        )
        .expect("config file should be written");

        let cfg =
            load_config_from_file(&config_path).expect("config with relative cwd should load");
        assert_eq!(cfg.mcp.servers.len(), 1);
        assert_eq!(cfg.mcp.servers[0].cwd.as_deref(), Some(mcp_cwd.as_path()));
    }

    #[test]
    fn load_config_resolves_skill_dirs_relative_to_config_file_dir() {
        let temp = TempDir::new().expect("temp dir should be created");
        let config_dir = temp.path().join("config");
        let skills_dir = config_dir.join("skills");
        fs::create_dir_all(&skills_dir).expect("skills dir should be created");

        let config_path = config_dir.join("sa.toml");
        fs::create_dir_all(&config_dir).expect("config dir should be created");
        fs::write(
            &config_path,
            r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[skills]
dirs = ["./skills"]
"#,
        )
        .expect("config file should be written");

        let cfg =
            load_config_from_file(&config_path).expect("config with relative skills should load");
        assert_eq!(cfg.skills.dirs, vec![skills_dir.display().to_string()]);
    }

    #[test]
    fn resolve_config_file_dir_handles_relative_leaf_filename() {
        let resolved =
            resolve_config_file_dir(Path::new("sa.toml")).expect("relative sa.toml should resolve");
        let current = std::path::absolute(Path::new(".")).expect("cwd should resolve");
        assert_eq!(resolved, current);
    }

    #[test]
    fn load_config_rejects_missing_mcp_cwd_directory() {
        let temp = TempDir::new().expect("temp dir should be created");
        let config_dir = temp.path().join("config");
        fs::create_dir_all(&config_dir).expect("config dir should be created");

        let config_path = config_dir.join("sa.toml");
        fs::write(
            &config_path,
            r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp.playwright]
command = "npx"
args = ["@playwright/mcp@latest"]
cwd = "missing-dir"
"#,
        )
        .expect("config file should be written");

        let err =
            load_config_from_file(&config_path).expect_err("missing mcp cwd should be rejected");
        assert!(
            err.to_string()
                .contains("mcp.playwright.cwd does not exist")
        );
    }

    #[test]
    fn load_config_rejects_cwd_on_http_transport() {
        let temp = TempDir::new().expect("temp dir should be created");
        let config_dir = temp.path().join("config");
        let http_cwd = config_dir.join("http-cwd");
        fs::create_dir_all(&http_cwd).expect("dummy cwd should be created");

        let config_path = config_dir.join("sa.toml");
        fs::write(
            &config_path,
            r#"
[llm]
base_url = "https://example.com/v1"
api_key = "sk-test"
model = "gpt-5.2"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = "."
agents_md = "Agents.md"

[mcp.remote]
url = "https://example.com/mcp"
cwd = "http-cwd"
"#,
        )
        .expect("config file should be written");

        let err = load_config_from_file(&config_path).expect_err("HTTP mcp cwd should be rejected");
        assert!(
            err.to_string()
                .contains("mcp.remote.cwd is only supported for transport=stdio")
        );
    }
}
