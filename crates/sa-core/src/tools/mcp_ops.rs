//! MCP resource tool implementations for ToolExecutor.

use crate::cancel::CancelToken;
use super::ToolExecutor;
use anyhow::Context as _;
use serde::Deserialize;
use serde_json;

impl ToolExecutor {
    /// `ListMcpResources`: enumerate connected MCP resources.
    pub(crate) async fn list_mcp_resources(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("ListMcpResources cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            server: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ListMcpResources")?;
        let Some(registry) = &self.mcp_registry else {
            anyhow::bail!("No MCP registry is configured");
        };

        registry.list_resources(args.server.as_deref()).await
    }

    /// `ReadMcpResource`: read one resource payload from a connected MCP
    /// server.

    /// `ReadMcpResource`: read one resource payload from a connected MCP
    /// server.
    pub(crate) async fn read_mcp_resource(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("ReadMcpResource cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            server: String,
            uri: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ReadMcpResource")?;
        let Some(registry) = &self.mcp_registry else {
            anyhow::bail!("No MCP registry is configured");
        };

        registry.read_resource(&args.server, &args.uri).await
    }
}
