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
use crate::openai::{AuthStyle, WireApi};
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

    /// Long-session history compaction configuration (`[compaction]`).
    #[serde(default)]
    pub compaction: CompactionConfig,

    /// External MCP server configuration (`[mcp]`).
    #[serde(default)]
    pub mcp: McpConfig,
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
#[derive(Debug, Clone, Deserialize, Default)]
pub struct McpServerConfig {
    /// Display name used as the tool prefix (`<server>__<tool>`).
    pub name: String,
    /// Transport type.
    #[serde(default)]
    pub transport: McpTransport,
    /// URL for HTTP/SSE transports.
    #[serde(default)]
    pub url: Option<String>,
    /// Executable for stdio transport.
    #[serde(default)]
    pub command: String,
    /// Arguments for stdio transport.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for stdio transport.
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    /// Extra HTTP headers for HTTP/SSE transports.
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
    /// Optional per-call timeout in seconds.
    #[serde(default)]
    pub tool_timeout_secs: Option<u64>,
}

/// External MCP client configuration (`[mcp]`).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct McpConfig {
    /// Whether MCP support is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Configured MCP servers.
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
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
    let config = toml::from_str::<Config>(&raw)
        .with_context(|| format!("Failed to parse TOML config file: {}", path.display()))?;
    validate_mcp_config(&config.mcp)?;
    Ok(config)
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

/// Validate MCP configuration early so startup errors are explicit.
fn validate_mcp_config(config: &McpConfig) -> anyhow::Result<()> {
    let mut seen_names = std::collections::HashSet::<String>::new();

    for (index, server) in config.servers.iter().enumerate() {
        let name = server.name.trim();
        if name.is_empty() {
            anyhow::bail!("mcp.servers[{index}].name must not be empty");
        }
        if !seen_names.insert(name.to_ascii_lowercase()) {
            anyhow::bail!("mcp.servers contains duplicate name: {name}");
        }

        if let Some(timeout) = server.tool_timeout_secs {
            if timeout == 0 {
                anyhow::bail!("mcp.servers[{index}].tool_timeout_secs must be greater than 0");
            }
            if timeout > MCP_MAX_TOOL_TIMEOUT_SECS {
                anyhow::bail!(
                    "mcp.servers[{index}].tool_timeout_secs exceeds max {MCP_MAX_TOOL_TIMEOUT_SECS}"
                );
            }
        }

        match server.transport {
            McpTransport::Stdio => {
                if server.command.trim().is_empty() {
                    anyhow::bail!(
                        "mcp.servers[{index}] with transport=stdio requires non-empty command"
                    );
                }
            }
            McpTransport::Http | McpTransport::Sse => {
                let url = server
                    .url
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "mcp.servers[{index}] with transport={} requires url",
                            match server.transport {
                                McpTransport::Http => "http",
                                McpTransport::Sse => "sse",
                                McpTransport::Stdio => "stdio",
                            }
                        )
                    })?;
                let parsed = reqwest::Url::parse(url)
                    .with_context(|| format!("mcp.servers[{index}].url is not a valid URL"))?;
                if !matches!(parsed.scheme(), "http" | "https") {
                    anyhow::bail!("mcp.servers[{index}].url must use http/https");
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Config, LlmConfig, McpConfig, McpServerConfig, McpTransport, validate_mcp_config};
    use crate::openai::{AuthStyle, WireApi};
    use std::collections::HashMap;

    fn sample_llm() -> LlmConfig {
        LlmConfig {
            base_url: "https://example.com/v1".to_string(),
            api_key: "sk-test".to_string(),
            model: "gpt-5.2".to_string(),
            wire_api: None,
            auth_style: None,
            system_role_name: None,
            reasoning_effort: None,
            max_steps: 32,
        }
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
                    env: HashMap::new(),
                    headers: HashMap::new(),
                    tool_timeout_secs: None,
                },
            ],
        };

        let err = validate_mcp_config(&config).expect_err("duplicate names must fail");
        assert!(err.to_string().contains("duplicate name"));
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
                env: HashMap::new(),
                headers: HashMap::new(),
                tool_timeout_secs: None,
            }],
        };

        let err = validate_mcp_config(&config).expect_err("stdio command must be required");
        assert!(err.to_string().contains("requires non-empty command"));
    }
}
