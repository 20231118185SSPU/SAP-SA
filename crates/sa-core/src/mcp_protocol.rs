//! MCP (Model Context Protocol) JSON-RPC 2.0 protocol types.
//!
//! This module is ported from the mature MCP implementation already used in
//! `zeroclaw`, then reduced to the pieces SA needs for external tool servers.
//! Keeping the protocol layer small and explicit lowers the chance of subtle
//! serialization bugs.

use serde::{Deserialize, Serialize};

/// JSON-RPC protocol version used by MCP.
pub const JSONRPC_VERSION: &str = "2.0";

/// MCP protocol version advertised during `initialize`.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Standard JSON-RPC internal error code.
pub const INTERNAL_ERROR: i32 = -32603;

/// Outbound JSON-RPC request (client -> MCP server).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version marker.
    pub jsonrpc: String,
    /// Request id for normal method calls; omitted for notifications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    /// Method name, for example `initialize` or `tools/call`.
    pub method: String,
    /// Optional method parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    /// Build a method call request with a numeric id.
    pub fn new(id: u64, method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(serde_json::json!(id)),
            method: method.into(),
            params: Some(params),
        }
    }

    /// Build a notification that expects no response.
    pub fn notification(method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: None,
            method: method.into(),
            params: Some(params),
        }
    }
}

/// Inbound JSON-RPC response (MCP server -> client).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version marker.
    pub jsonrpc: String,
    /// Response id, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    /// Successful result payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error payload for failed calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Numeric error code.
    pub code: i32,
    /// Human-readable message.
    pub message: String,
    /// Optional extra data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// One MCP tool definition returned by `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    /// Tool name as advertised by the server.
    pub name: String,
    /// Optional human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema for input arguments.
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// `tools/list` result payload.
#[derive(Debug, Deserialize)]
pub struct McpToolsListResult {
    /// Advertised tools.
    pub tools: Vec<McpToolDef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_with_id() {
        let req = JsonRpcRequest::new(1, "tools/list", serde_json::json!({}));
        let serialized = serde_json::to_string(&req).expect("serialize request");
        assert!(serialized.contains("\"id\":1"));
        assert!(serialized.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn notification_omits_id() {
        let req = JsonRpcRequest::notification("notifications/initialized", serde_json::json!({}));
        let serialized = serde_json::to_string(&req).expect("serialize request");
        assert!(!serialized.contains("\"id\""));
    }
}
