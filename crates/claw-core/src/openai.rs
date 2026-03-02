//! OpenAI-compatible Chat Completions client.
//!
//! The request requires: "how to call an OpenAI-compatible API".
//! We implement the minimal subset of `POST /v1/chat/completions` that we need:
//! - Send `model` and `messages`.
//! - Send `tools` definitions to enable tool calling.
//! - Parse assistant messages, including `tool_calls`.
//!
//! References in `../zeroclaw`:
//! - `zeroclaw/src/providers/compatible.rs` implements a large, production-grade
//!   OpenAI-compatible provider adapter.
//! - Here we extract the smallest useful slice for a minimal agent.

use anyhow::Context as _;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

/// Minimal OpenAI-compatible client.
#[derive(Debug, Clone)]
pub struct OpenAiClient {
    /// Preconfigured HTTP client (connection pooling, timeouts, TLS).
    http: reqwest::Client,
    /// Base URL (example: `https://example.com/v1`).
    base_url: String,
}

impl OpenAiClient {
    /// Create a new client.
    pub fn new(base_url: String, api_key: String) -> anyhow::Result<Self> {
        // Normalize base URL by trimming trailing slashes. This avoids double
        // slashes when we append `/chat/completions`.
        let base_url = base_url.trim_end_matches('/').to_string();

        // Build default headers.
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Note: we do not log the API key. We also avoid embedding it into errors.
        let bearer = format!("Bearer {api_key}");
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&bearer).context("Invalid API key for Authorization header")?,
        );

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self { http, base_url })
    }

    /// Compute the `chat/completions` URL.
    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Send a `POST /chat/completions` request.
    pub async fn chat_completions(
        &self,
        req: &ChatCompletionsRequest,
    ) -> anyhow::Result<ChatCompletionsResponse> {
        let url = self.chat_completions_url();

        // Send request.
        let resp = self
            .http
            .post(url)
            .json(req)
            .send()
            .await
            .context("Chat completions request failed")?;

        // Handle non-2xx responses with a readable error.
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());

            // IMPORTANT: do not include the API key in the error.
            // (We keep `self.api_key` only for the header; we never print it.)
            anyhow::bail!("Chat completions error ({status}): {body}");
        }

        // Parse JSON.
        resp.json::<ChatCompletionsResponse>()
            .await
            .context("Failed to parse chat completions JSON response")
    }
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionsRequest {
    /// Model name.
    pub model: String,
    /// Chat messages.
    pub messages: Vec<ChatMessage>,

    /// Tool definitions (OpenAI function-calling style).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Tool choice. We set `"auto"` to let the model choose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    /// Whether to use streaming. (We keep it `false` in this minimal agent.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A single message in the OpenAI Chat Completions format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role name (e.g. `system`, `developer`, `user`, `assistant`, `tool`).
    pub role: String,

    /// Text content.
    ///
    /// For tool-calls, providers often return `null` content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls emitted by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// For tool result messages: which tool-call this result corresponds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Construct a normal text message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Construct a tool result message (`role: "tool"`).
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Tool definition used in `tools`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    /// Always `"function"` in this minimal implementation.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function definition.
    pub function: ToolFunctionDefinition,
}

/// A single function tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct ToolFunctionDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for parameters.
    pub parameters: serde_json::Value,
}

/// Tool call emitted by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-generated identifier.
    pub id: String,
    /// Type discriminator, usually `"function"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function call payload.
    pub function: ToolFunctionCall,
}

/// Function call payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionCall {
    /// Name of the tool to call.
    pub name: String,
    /// Arguments in JSON string form.
    pub arguments: String,
}

/// Response body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionsResponse {
    /// Model outputs.
    pub choices: Vec<ChatChoice>,
}

/// A single model output choice.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    /// The assistant message.
    pub message: ChatMessage,
    /// Finish reason (e.g. `"stop"`, `"tool_calls"`).
    pub finish_reason: Option<String>,
}

impl ChatCompletionsResponse {
    /// Convenience: return the first choice, or an error if missing.
    pub fn first_choice(&self) -> anyhow::Result<&ChatChoice> {
        self.choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("OpenAI response contained no choices"))
    }
}
