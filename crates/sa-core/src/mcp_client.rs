//! MCP (Model Context Protocol) client for external tool servers.
//!
//! This module is adapted from `zeroclaw`'s mature implementation, then shaped
//! for SA's smaller tool system:
//! - connect to multiple servers
//! - fetch their tool list
//! - flatten tools into `server__tool` names
//! - dispatch tool calls back to the right server

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};

use crate::config::McpServerConfig;
use crate::mcp_protocol::{JsonRpcRequest, MCP_PROTOCOL_VERSION, McpToolDef, McpToolsListResult};
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
}

/// Public handle for one MCP server.
#[derive(Clone)]
pub struct McpServer {
    /// Shared mutable server state.
    inner: Arc<Mutex<McpServerInner>>,
}

impl McpServer {
    /// Connect, initialize, and list tools.
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

        Ok(Self {
            inner: Arc::new(Mutex::new(McpServerInner {
                config,
                transport,
                next_id: AtomicU64::new(3),
                tools: tool_list.tools,
            })),
        })
    }

    /// Clone the current tool list.
    async fn tools(&self) -> Vec<McpToolDef> {
        self.inner.lock().await.tools.clone()
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
}

/// Flattened registry of all connected MCP servers and tools.
#[derive(Clone)]
pub struct McpRegistry {
    /// Connected servers.
    servers: Vec<McpServer>,
    /// Prefixed tool name -> (server index, original tool name).
    routes: HashMap<String, (usize, String)>,
    /// Prefixed tool name -> MCP definition.
    defs: HashMap<String, McpToolDef>,
}

impl McpRegistry {
    /// Connect to all configured servers. Individual failures are non-fatal.
    pub async fn connect_all(configs: &[McpServerConfig]) -> Result<Self> {
        let mut servers = Vec::new();
        let mut routes = HashMap::new();
        let mut defs = HashMap::new();

        for config in configs {
            match McpServer::connect(config.clone()).await {
                Ok(server) => {
                    let server_index = servers.len();
                    let tools = server.tools().await;
                    for tool in tools {
                        let prefixed_name = format!("{}__{}", config.name, tool.name);
                        routes.insert(prefixed_name.clone(), (server_index, tool.name.clone()));
                        defs.insert(prefixed_name, tool);
                    }
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
            routes,
            defs,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::McpTransport;
    use std::collections::HashMap;

    #[test]
    fn prefixed_tool_name_format_is_stable() {
        let prefixed = format!("{}__{}", "filesystem", "read_file");
        assert_eq!(prefixed, "filesystem__read_file");
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
}
