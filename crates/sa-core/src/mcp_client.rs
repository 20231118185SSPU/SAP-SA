//! MCP (Model Context Protocol) client for external tool servers.
//!
//! This module is adapted from `zeroclaw`'s mature implementation, then shaped
//! for SA's smaller tool system:
//! - connect to multiple servers
//! - fetch their tools/resources/prompts
//! - flatten tools into `mcp__server__tool` names
//! - expose MCP prompts as command-like entries
//! - dispatch tool/resource/prompt calls back to the right server

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use shlex::split as shlex_split;
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};

use crate::commands::McpPromptCommand;
use crate::config::McpServerConfig;
use crate::mcp_protocol::{
    JsonRpcRequest, MCP_PROTOCOL_VERSION, McpPromptDef, McpPromptGetResult, McpPromptsListResult,
    McpResourceDef, McpResourceReadResult, McpResourcesListResult, McpToolDef, McpToolsListResult,
};
use crate::mcp_transport::{McpTransportConn, create_transport};
use crate::openai::{ToolDefinition, ToolFunctionDefinition};

/// Timeout for initialize/list calls.
const RECV_TIMEOUT_SECS: u64 = 30;

/// Default tool-call timeout.
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 180;

/// Hard maximum timeout.
const MAX_TOOL_TIMEOUT_SECS: u64 = 600;

/// One live MCP server connection.
struct McpServerInner {
    /// Original configuration.
    config: McpServerConfig,
    /// Active transport.
    transport: Box<dyn McpTransportConn>,
    /// Request id sequence.
    next_id: AtomicU64,
    /// Tools advertised by this server.
    tools: Vec<McpToolDef>,
    /// Resources advertised by this server.
    resources: Vec<McpResourceDef>,
    /// Prompts advertised by this server.
    prompts: Vec<McpPromptDef>,
}

/// Public handle for one MCP server.
#[derive(Clone)]
pub struct McpServer {
    /// Shared mutable server state.
    inner: Arc<Mutex<McpServerInner>>,
}

impl McpServer {
    /// Connect, initialize, and list tools/resources/prompts.
    pub async fn connect(config: McpServerConfig) -> Result<Self> {
        let mut transport = create_transport(&config).with_context(|| {
            format!(
                "failed to create transport for MCP server `{}`",
                config.name
            )
        })?;

        let initialize = JsonRpcRequest::new(
            1,
            "initialize",
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "sa",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        );
        let initialize_response = timeout(
            Duration::from_secs(RECV_TIMEOUT_SECS),
            transport.send_and_recv(&initialize),
        )
        .await
        .with_context(|| {
            format!(
                "MCP server `{}` timed out after {}s waiting for initialize response",
                config.name, RECV_TIMEOUT_SECS
            )
        })??;
        if initialize_response.error.is_some() {
            bail!(
                "MCP server `{}` rejected initialize: {:?}",
                config.name,
                initialize_response.error
            );
        }

        let initialized =
            JsonRpcRequest::notification("notifications/initialized", serde_json::json!({}));
        let _ = transport.send_and_recv(&initialized).await;

        let list_request = JsonRpcRequest::new(2, "tools/list", serde_json::json!({}));
        let list_response = timeout(
            Duration::from_secs(RECV_TIMEOUT_SECS),
            transport.send_and_recv(&list_request),
        )
        .await
        .with_context(|| {
            format!(
                "MCP server `{}` timed out after {}s waiting for tools/list response",
                config.name, RECV_TIMEOUT_SECS
            )
        })??;

        let result = list_response
            .result
            .ok_or_else(|| anyhow!("tools/list returned no result from `{}`", config.name))?;
        let tool_list: McpToolsListResult = serde_json::from_value(result)
            .with_context(|| format!("failed to parse tools/list from `{}`", config.name))?;

        tracing::info!(
            "MCP server `{}` connected — {} tool(s) available",
            config.name,
            tool_list.tools.len()
        );

        let resources = match timeout(
            Duration::from_secs(RECV_TIMEOUT_SECS),
            transport.send_and_recv(&JsonRpcRequest::new(
                3,
                "resources/list",
                serde_json::json!({}),
            )),
        )
        .await
        {
            Ok(Ok(response)) => match response.result {
                Some(result) => serde_json::from_value::<McpResourcesListResult>(result)
                    .map(|result| result.resources)
                    .unwrap_or_default(),
                None => Vec::new(),
            },
            Ok(Err(_)) | Err(_) => Vec::new(),
        };

        let prompts = match timeout(
            Duration::from_secs(RECV_TIMEOUT_SECS),
            transport.send_and_recv(&JsonRpcRequest::new(
                4,
                "prompts/list",
                serde_json::json!({}),
            )),
        )
        .await
        {
            Ok(Ok(response)) => match response.result {
                Some(result) => serde_json::from_value::<McpPromptsListResult>(result)
                    .map(|result| result.prompts)
                    .unwrap_or_default(),
                None => Vec::new(),
            },
            Ok(Err(_)) | Err(_) => Vec::new(),
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(McpServerInner {
                config,
                transport,
                next_id: AtomicU64::new(5),
                tools: tool_list.tools,
                resources,
                prompts,
            })),
        })
    }

    /// Clone the current tool list.
    async fn tools(&self) -> Vec<McpToolDef> {
        self.inner.lock().await.tools.clone()
    }

    /// Clone the current resource list.
    async fn resources(&self) -> Vec<McpResourceDef> {
        self.inner.lock().await.resources.clone()
    }

    /// Clone the current prompt list.
    async fn prompts(&self) -> Vec<McpPromptDef> {
        self.inner.lock().await.prompts.clone()
    }

    /// Call one tool on this server.
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut inner = self.inner.lock().await;
        let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(
            id,
            "tools/call",
            json!({ "name": tool_name, "arguments": arguments }),
        );

        let tool_timeout = inner
            .config
            .tool_timeout_secs
            .unwrap_or(DEFAULT_TOOL_TIMEOUT_SECS)
            .min(MAX_TOOL_TIMEOUT_SECS);

        let response = timeout(
            Duration::from_secs(tool_timeout),
            inner.transport.send_and_recv(&request),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "MCP server `{}` timed out after {}s during tool call `{tool_name}`",
                inner.config.name,
                tool_timeout
            )
        })?
        .with_context(|| {
            format!(
                "MCP server `{}` error during tool call `{tool_name}`",
                inner.config.name
            )
        })?;

        if let Some(error) = response.error {
            bail!(
                "MCP tool `{tool_name}` error {}: {}",
                error.code,
                error.message
            );
        }

        Ok(response.result.unwrap_or(serde_json::Value::Null))
    }

    /// Read one resource from this server.
    async fn read_resource(&self, uri: &str) -> Result<McpResourceReadResult> {
        let mut inner = self.inner.lock().await;
        let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, "resources/read", json!({ "uri": uri }));

        let response = timeout(
            Duration::from_secs(RECV_TIMEOUT_SECS),
            inner.transport.send_and_recv(&request),
        )
        .await
        .with_context(|| {
            format!(
                "MCP server `{}` timed out during resources/read",
                inner.config.name
            )
        })??;

        if let Some(error) = response.error {
            bail!("MCP resources/read error {}: {}", error.code, error.message);
        }

        let result = response.result.ok_or_else(|| {
            anyhow!(
                "resources/read returned no result from `{}`",
                inner.config.name
            )
        })?;
        serde_json::from_value(result).with_context(|| {
            format!(
                "failed to parse resources/read from `{}`",
                inner.config.name
            )
        })
    }

    /// Expand one prompt from this server.
    async fn get_prompt(
        &self,
        prompt_name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<McpPromptGetResult> {
        let mut inner = self.inner.lock().await;
        let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(
            id,
            "prompts/get",
            json!({
                "name": prompt_name,
                "arguments": arguments,
            }),
        );

        let response = timeout(
            Duration::from_secs(RECV_TIMEOUT_SECS),
            inner.transport.send_and_recv(&request),
        )
        .await
        .with_context(|| {
            format!(
                "MCP server `{}` timed out during prompts/get `{prompt_name}`",
                inner.config.name
            )
        })??;

        if let Some(error) = response.error {
            bail!("MCP prompts/get error {}: {}", error.code, error.message);
        }

        let result = response.result.ok_or_else(|| {
            anyhow!(
                "prompts/get returned no result from `{}`",
                inner.config.name
            )
        })?;
        serde_json::from_value(result)
            .with_context(|| format!("failed to parse prompts/get from `{}`", inner.config.name))
    }
}

/// Flattened registry of all connected MCP servers and tools.
#[derive(Clone)]
pub struct McpRegistry {
    /// Connected servers.
    servers: Vec<McpServer>,
    /// Server names aligned with `servers`.
    server_names: Vec<String>,
    /// Prefixed tool name -> (server index, original tool name).
    routes: HashMap<String, (usize, String)>,
    /// Prefixed tool name -> MCP definition.
    defs: HashMap<String, McpToolDef>,
    /// Original server name -> server index.
    server_indexes: HashMap<String, usize>,
    /// Original server name -> advertised resources.
    resources: HashMap<String, Vec<McpResourceDef>>,
    /// Prefixed prompt command name -> (server index, prompt definition).
    prompt_defs: HashMap<String, (usize, McpPromptDef)>,
}

impl McpRegistry {
    /// Connect to all configured servers. Individual failures are non-fatal.
    pub async fn connect_all(configs: &[McpServerConfig]) -> Result<Self> {
        let mut servers = Vec::new();
        let mut server_names = Vec::new();
        let mut routes = HashMap::new();
        let mut defs = HashMap::new();
        let mut server_indexes = HashMap::new();
        let mut resources = HashMap::new();
        let mut prompt_defs = HashMap::new();
        let mut normalized_server_names = HashMap::<String, String>::new();

        for config in configs {
            let normalized_server = normalize_name_for_mcp(&config.name);
            if let Some(previous) =
                normalized_server_names.insert(normalized_server.clone(), config.name.clone())
            {
                anyhow::bail!(
                    "MCP server name collision after normalization: `{}` and `{}` both map to `mcp__{}`",
                    previous,
                    config.name,
                    normalized_server
                );
            }

            match McpServer::connect(config.clone()).await {
                Ok(server) => {
                    let server_index = servers.len();
                    let tools = server.tools().await;
                    let server_resources = server.resources().await;
                    let prompts = server.prompts().await;
                    for tool in tools {
                        let prefixed_name = format!("mcp__{}__{}", normalized_server, tool.name);
                        routes.insert(prefixed_name.clone(), (server_index, tool.name.clone()));
                        defs.insert(prefixed_name, tool);
                    }
                    for prompt in prompts {
                        let prefixed_name = format!("mcp__{}__{}", normalized_server, prompt.name);
                        prompt_defs.insert(prefixed_name, (server_index, prompt));
                    }
                    server_indexes.insert(config.name.clone(), server_index);
                    resources.insert(config.name.clone(), server_resources);
                    server_names.push(config.name.clone());
                    servers.push(server);
                }
                Err(error) => {
                    tracing::error!(
                        "Failed to connect to MCP server `{}`: {error:#}",
                        config.name
                    );
                }
            }
        }

        Ok(Self {
            servers,
            server_names,
            routes,
            defs,
            server_indexes,
            resources,
            prompt_defs,
        })
    }

    /// Build OpenAI-compatible tool definitions for all connected MCP tools.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut entries: Vec<_> = self.defs.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        entries
            .into_iter()
            .map(|(prefixed_name, def)| ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: prefixed_name.clone(),
                    description: def
                        .description
                        .clone()
                        .unwrap_or_else(|| "MCP tool".to_string()),
                    parameters: def.input_schema.clone(),
                },
            })
            .collect()
    }

    /// Build command descriptors for MCP prompts.
    pub fn prompt_commands(&self) -> Vec<McpPromptCommand> {
        let mut entries: Vec<_> = self.prompt_defs.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        entries
            .into_iter()
            .map(|(prefixed_name, (server_index, prompt))| McpPromptCommand {
                name: prefixed_name.clone(),
                server_name: self.server_names[*server_index].clone(),
                prompt_name: prompt.name.clone(),
                description: prompt.description.clone().unwrap_or_default(),
                arguments: prompt
                    .arguments
                    .iter()
                    .map(|argument| argument.name.clone())
                    .collect(),
            })
            .collect()
    }

    /// Return whether the registry exposes a given prefixed tool.
    pub fn has_tool(&self, prefixed_name: &str) -> bool {
        self.routes.contains_key(prefixed_name)
    }

    /// Execute a prefixed MCP tool name.
    pub async fn call_tool(
        &self,
        prefixed_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String> {
        let (server_index, original_name) = self
            .routes
            .get(prefixed_name)
            .ok_or_else(|| anyhow!("unknown MCP tool `{prefixed_name}`"))?;

        let result = self.servers[*server_index]
            .call_tool(original_name, arguments)
            .await?;
        serde_json::to_string_pretty(&result)
            .with_context(|| format!("failed to serialize result of MCP tool `{prefixed_name}`"))
    }

    /// Number of connected servers.
    pub fn server_count(&self) -> usize {
        self.servers.len()
    }

    /// Number of flattened tools.
    pub fn tool_count(&self) -> usize {
        self.routes.len()
    }

    /// Whether there are no connected servers.
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Expand one MCP prompt into plain text content.
    pub async fn expand_prompt(
        &self,
        server_name: &str,
        prompt_name: &str,
        argument_names: &[String],
        raw_args: Option<&str>,
    ) -> Result<String> {
        let Some(server_index) = self.server_indexes.get(server_name) else {
            bail!("unknown MCP server `{server_name}`");
        };

        let arguments = build_prompt_arguments(argument_names, raw_args.unwrap_or_default());
        let result = self.servers[*server_index]
            .get_prompt(prompt_name, arguments)
            .await?;
        Ok(format_prompt_result(server_name, prompt_name, &result))
    }

    /// List resources from one server or from all servers.
    pub async fn list_resources(&self, server_name: Option<&str>) -> Result<String> {
        let payload = if let Some(server_name) = server_name {
            let Some(resources) = self.resources.get(server_name) else {
                bail!("unknown MCP server `{server_name}`");
            };
            serde_json::json!({
                "server": server_name,
                "resources": resources,
            })
        } else {
            let mut entries: Vec<_> = self.resources.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let resources = entries
                .into_iter()
                .flat_map(|(server, items)| {
                    items.iter().map(move |item| {
                        serde_json::json!({
                            "server": server,
                            "uri": item.uri,
                            "name": item.name,
                            "description": item.description,
                            "mimeType": item.mime_type,
                        })
                    })
                })
                .collect::<Vec<_>>();
            serde_json::json!({ "resources": resources })
        };

        serde_json::to_string_pretty(&payload).context("failed to serialize MCP resources list")
    }

    /// Read one resource from a specific server.
    pub async fn read_resource(&self, server_name: &str, uri: &str) -> Result<String> {
        let Some(server_index) = self.server_indexes.get(server_name) else {
            bail!("unknown MCP server `{server_name}`");
        };
        let result = self.servers[*server_index].read_resource(uri).await?;
        serde_json::to_string_pretty(&serde_json::json!({
            "server": server_name,
            "uri": uri,
            "contents": result.contents,
        }))
        .context("failed to serialize MCP resource read result")
    }
}

/// Normalize one MCP server name so it is safe inside `mcp__...` tool names.
fn normalize_name_for_mcp(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// Build named prompt arguments from one raw whitespace-delimited argument
/// string.
fn build_prompt_arguments(argument_names: &[String], raw_args: &str) -> HashMap<String, String> {
    let parsed = shlex_split(raw_args).unwrap_or_else(|| {
        let trimmed = raw_args.trim();
        if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed.to_string()]
        }
    });

    argument_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| parsed.get(index).map(|value| (name.clone(), value.clone())))
        .collect()
}

/// Flatten one MCP prompt result into plain text that can be injected into the
/// main agent conversation as a tool result.
fn format_prompt_result(
    server_name: &str,
    prompt_name: &str,
    result: &McpPromptGetResult,
) -> String {
    let mut lines = vec![format!(
        "Loaded MCP prompt `mcp__{}__{}` from server `{}`.",
        normalize_name_for_mcp(server_name),
        prompt_name,
        server_name
    )];

    if let Some(description) = result.description.as_deref() {
        let description = description.trim();
        if !description.is_empty() {
            lines.push(format!("Description: {description}"));
        }
    }

    for message in &result.messages {
        let role = message.role.as_deref().unwrap_or("message");
        let content = flatten_prompt_content(&message.content);
        if content.trim().is_empty() {
            continue;
        }
        lines.push(format!("[{role}] {content}"));
    }

    lines.join("\n\n")
}

/// Flatten one MCP prompt content payload into human-readable text.
fn flatten_prompt_content(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(flatten_prompt_content)
            .filter(|item| !item.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Object(map) => match map.get("type").and_then(|value| value.as_str()) {
            Some("text") => map
                .get("text")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            Some("resource") => map
                .get("resource")
                .map(flatten_prompt_content)
                .unwrap_or_default(),
            _ => {
                if let Some(text) = map.get("text").and_then(|value| value.as_str()) {
                    return text.to_string();
                }
                if let Some(blob) = map.get("blob").and_then(|value| value.as_str()) {
                    return format!("[binary blob: {} base64 chars]", blob.len());
                }
                serde_json::to_string_pretty(content).unwrap_or_else(|_| content.to_string())
            }
        },
        _ => serde_json::to_string_pretty(content).unwrap_or_else(|_| content.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::McpTransport;
    use std::collections::HashMap;

    #[test]
    fn prefixed_tool_name_format_is_stable() {
        let prefixed = format!("mcp__{}__{}", "filesystem", "read_file");
        assert_eq!(prefixed, "mcp__filesystem__read_file");
    }

    #[tokio::test]
    async fn connect_all_is_non_fatal_on_failure() {
        let configs = vec![McpServerConfig {
            name: "bad".to_string(),
            transport: McpTransport::Stdio,
            url: None,
            command: "/definitely/missing/sa-mcp-test".to_string(),
            args: Vec::new(),
            cwd: None,
            env: HashMap::new(),
            headers: HashMap::new(),
            tool_timeout_secs: None,
        }];

        let registry = McpRegistry::connect_all(&configs)
            .await
            .expect("connect_all should not fail the whole registry");
        assert!(registry.is_empty());
        assert_eq!(registry.tool_count(), 0);
    }

    #[tokio::test]
    async fn connect_all_rejects_normalized_server_name_collision() {
        let configs = vec![
            McpServerConfig {
                name: "play.wright".to_string(),
                transport: McpTransport::Stdio,
                url: None,
                command: "/definitely/missing/sa-mcp-test".to_string(),
                args: Vec::new(),
                cwd: None,
                env: HashMap::new(),
                headers: HashMap::new(),
                tool_timeout_secs: None,
            },
            McpServerConfig {
                name: "play/wright".to_string(),
                transport: McpTransport::Stdio,
                url: None,
                command: "/definitely/missing/sa-mcp-test".to_string(),
                args: Vec::new(),
                cwd: None,
                env: HashMap::new(),
                headers: HashMap::new(),
                tool_timeout_secs: None,
            },
        ];

        let err = match McpRegistry::connect_all(&configs).await {
            Ok(_) => panic!("normalized collision must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("MCP server name collision"));
    }
}
