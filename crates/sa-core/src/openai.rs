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
use std::fmt;

/// Error returned by [`OpenAiClient::chat_completions`].
///
/// We keep this error structured so the agent loop can decide whether an error
/// is retryable (temporary) or non-retryable (permanent, such as invalid auth).
#[derive(Debug)]
pub enum ChatCompletionsError {
    /// Transport-level errors (DNS failure, connection refused, TLS, timeout, etc.).
    Transport(reqwest::Error),
    /// Non-2xx HTTP status with the raw response body.
    Http {
        /// HTTP status code returned by the server.
        status: reqwest::StatusCode,
        /// Raw response body (best-effort, may be truncated by the server).
        body: String,
    },
}

impl ChatCompletionsError {
    /// Return `true` if we should retry this error.
    ///
    /// We follow a conservative policy:
    /// - Transport errors are usually retryable (except builder errors).
    /// - HTTP 408/429/5xx are treated as retryable.
    /// - Authentication/validation errors (4xx) are treated as non-retryable.
    pub fn is_retriable(&self) -> bool {
        match self {
            ChatCompletionsError::Transport(err) => !err.is_builder(),
            ChatCompletionsError::Http { status, .. } => {
                *status == reqwest::StatusCode::REQUEST_TIMEOUT
                    || *status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error()
            }
        }
    }

    /// Return the HTTP status code, if this is an HTTP error.
    pub fn status(&self) -> Option<reqwest::StatusCode> {
        match self {
            ChatCompletionsError::Transport(_) => None,
            ChatCompletionsError::Http { status, .. } => Some(*status),
        }
    }
}

impl fmt::Display for ChatCompletionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatCompletionsError::Transport(err) => write!(f, "transport error: {err}"),
            ChatCompletionsError::Http { status, body } => {
                write!(f, "http error ({status}): {body}")
            }
        }
    }
}

impl std::error::Error for ChatCompletionsError {}

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
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.chat_completions_url();

        // Send request.
        let resp = self
            .http
            .post(url)
            .json(req)
            .send()
            .await
            .map_err(ChatCompletionsError::Transport)?;

        // Handle non-2xx responses with a readable error.
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            let body = truncate_error_body(body);

            // IMPORTANT:
            // - We do not include the API key in the error.
            // - We keep the raw body because it is often the only traceable clue.
            return Err(ChatCompletionsError::Http { status, body });
        }

        // Parse JSON.
        resp.json::<ChatCompletionsResponse>()
            .await
            .map_err(ChatCompletionsError::Transport)
    }
}

/// Truncate a response body that is going to be included in an error.
///
/// Rationale:
/// - Many API gateways return large HTML error pages on 502/503.
/// - Emitting megabytes into the agent event stream is not useful and can
///   degrade the CLI UX.
fn truncate_error_body(body: String) -> String {
    const MAX_CHARS: usize = 4_000;

    // If already small, keep as-is.
    if body.chars().count() <= MAX_CHARS {
        return body;
    }

    // Truncate by *characters* so we do not split UTF-8 sequences.
    let truncated: String = body.chars().take(MAX_CHARS).collect();
    format!("{truncated}…(truncated)")
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionsRequest {
    /// Model name.
    pub model: String,
    /// Chat messages.
    pub messages: Vec<ChatMessage>,

    /// Optional output token cap.
    ///
    /// We use the classic OpenAI-compatible `max_tokens` field because many
    /// compatible gateways and self-hosted proxies still accept this shape on
    /// `/v1/chat/completions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Optional reasoning depth / effort for GPT-family reasoning models.
    ///
    /// This is forwarded as the OpenAI-compatible top-level
    /// `reasoning_effort` field when configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

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

    /// Internal request-usage snapshot copied from the top-level response that
    /// produced this assistant turn.
    ///
    /// This field is intentionally not sent back to the provider. SA only keeps
    /// it locally so compaction can reuse the latest authoritative usage as a
    /// token-estimation anchor.
    ///
    /// Important:
    /// - this is **request-scoped** usage anchored to the assistant turn
    /// - it is not a claim about the cost of this one message in isolation
    #[serde(skip_serializing, default)]
    pub request_usage: Option<ChatUsage>,
}

impl ChatMessage {
    /// Construct a normal text message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            request_usage: None,
        }
    }

    /// Construct a tool result message (`role: "tool"`).
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            request_usage: None,
        }
    }
}

/// Provider-reported usage for one chat completion response.
///
/// The field aliases keep SA compatible with a wider range of
/// OpenAI-compatible gateways.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ChatUsage {
    /// Prompt-side tokens.
    #[serde(default, alias = "prompt_tokens")]
    pub input_tokens: Option<u64>,

    /// Completion-side tokens.
    #[serde(default, alias = "completion_tokens")]
    pub output_tokens: Option<u64>,

    /// Provider-reported total tokens, if available.
    #[serde(default, alias = "total")]
    pub total_tokens: Option<u64>,

    /// Nested prompt token detail payload.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokenDetails>,

    /// Responses-style input token detail payload.
    #[serde(default)]
    pub input_tokens_details: Option<InputTokenDetails>,

    /// Cache-read tokens exposed by some OpenAI-compatible bridges.
    #[serde(default, alias = "cache_read_tokens")]
    pub cache_read_input_tokens: Option<u64>,

    /// Cache-write tokens exposed by some OpenAI-compatible bridges.
    #[serde(default, alias = "cache_write_tokens")]
    pub cache_creation_input_tokens: Option<u64>,

    /// Chat Completions-style completion token detail payload.
    #[serde(default)]
    pub completion_tokens_details: Option<OutputTokenDetails>,

    /// Responses-style output token detail payload.
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokenDetails>,
}

impl ChatUsage {
    /// Return cache-read tokens explicitly reported as an additional usage
    /// component, as seen on Anthropic/Bedrock-style payloads.
    pub fn explicit_cache_read_tokens(&self) -> u64 {
        self.cache_read_input_tokens.unwrap_or(0)
    }

    /// Return cache-read tokens reported inside OpenAI-style nested detail
    /// objects.
    ///
    /// These nested values are usually informational details about prompt/input
    /// tokens, not always standalone totals. Callers that compute a total usage
    /// estimate must decide whether adding them would double-count.
    pub fn cached_input_tokens_detail(&self) -> u64 {
        self.prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .or(self
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens))
            .unwrap_or(0)
    }

    /// Return the best available cache-read token count for display/debug use.
    pub fn cache_read_tokens(&self) -> u64 {
        self.cache_read_input_tokens
            .or(self
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens))
            .or(self
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens))
            .unwrap_or(0)
    }

    /// Return cache-write tokens, if the gateway reports them.
    pub fn cache_write_tokens(&self) -> u64 {
        self.cache_creation_input_tokens.unwrap_or(0)
    }

    /// Return reasoning tokens reported by OpenAI-style detailed output usage.
    pub fn reasoning_tokens(&self) -> u64 {
        self.completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .or(self
                .output_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens))
            .unwrap_or(0)
    }
}

/// Nested prompt token detail payload used by some OpenAI-compatible APIs.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct PromptTokenDetails {
    /// Cached prompt tokens, if present.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// Responses-style input token detail payload.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct InputTokenDetails {
    /// Cached input tokens, if present.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// Output token detail payload used by newer OpenAI-compatible responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct OutputTokenDetails {
    /// Reasoning tokens, if present.
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
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
    /// Provider-reported usage for the whole request, if available.
    #[serde(default)]
    pub usage: Option<ChatUsage>,
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

#[cfg(test)]
mod tests {
    use super::{
        ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, ChatUsage, InputTokenDetails,
        OutputTokenDetails,
    };

    #[test]
    fn request_serializes_reasoning_effort_when_present() {
        let req = ChatCompletionsRequest {
            model: "gpt-5.2".to_string(),
            messages: vec![ChatMessage::text("user", "hello")],
            max_tokens: None,
            reasoning_effort: Some("xhigh".to_string()),
            tools: None,
            tool_choice: None,
            stream: Some(false),
        };

        let value = serde_json::to_value(req).expect("serialize request");
        assert_eq!(value["reasoning_effort"], "xhigh");
    }

    #[test]
    fn request_omits_reasoning_effort_when_absent() {
        let req = ChatCompletionsRequest {
            model: "gpt-5.2".to_string(),
            messages: vec![ChatMessage::text("user", "hello")],
            max_tokens: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            stream: Some(false),
        };

        let value = serde_json::to_value(req).expect("serialize request");
        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn chat_message_serialization_omits_internal_usage_snapshot() {
        let mut message = ChatMessage::text("assistant", "done");
        message.request_usage = Some(ChatUsage {
            input_tokens: Some(100),
            output_tokens: Some(20),
            total_tokens: Some(120),
            prompt_tokens_details: None,
            input_tokens_details: None,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            completion_tokens_details: None,
            output_tokens_details: None,
        });

        let value = serde_json::to_value(message).expect("serialize message");
        assert!(value.get("request_usage").is_none());
    }

    #[test]
    fn response_deserializes_usage_with_openai_aliases() {
        let raw = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 34,
                "total_tokens": 1234,
                "prompt_tokens_details": {
                    "cached_tokens": 1000
                }
            }
        });

        let response =
            serde_json::from_value::<ChatCompletionsResponse>(raw).expect("deserialize response");
        let usage = response.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, Some(1200));
        assert_eq!(usage.output_tokens, Some(34));
        assert_eq!(usage.total_tokens, Some(1234));
        assert_eq!(usage.cache_read_tokens(), 1000);
        assert_eq!(usage.cache_write_tokens(), 0);
    }

    #[test]
    fn response_deserializes_usage_with_responses_detail_aliases() {
        let raw = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 800,
                "output_tokens": 120,
                "input_tokens_details": {
                    "cached_tokens": 300
                },
                "output_tokens_details": {
                    "reasoning_tokens": 40
                },
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 25
            }
        });

        let response =
            serde_json::from_value::<ChatCompletionsResponse>(raw).expect("deserialize response");
        let usage = response.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, Some(800));
        assert_eq!(usage.output_tokens, Some(120));
        assert_eq!(usage.cached_input_tokens_detail(), 300);
        assert_eq!(usage.explicit_cache_read_tokens(), 50);
        assert_eq!(usage.cache_read_tokens(), 50);
        assert_eq!(usage.cache_write_tokens(), 25);
        assert_eq!(usage.reasoning_tokens(), 40);
    }

    #[test]
    fn chat_usage_display_helpers_fall_back_to_nested_detail_fields() {
        let usage = ChatUsage {
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
            prompt_tokens_details: None,
            input_tokens_details: Some(InputTokenDetails {
                cached_tokens: Some(222),
            }),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            completion_tokens_details: Some(OutputTokenDetails {
                reasoning_tokens: Some(17),
            }),
            output_tokens_details: None,
        };

        assert_eq!(usage.cached_input_tokens_detail(), 222);
        assert_eq!(usage.cache_read_tokens(), 222);
        assert_eq!(usage.reasoning_tokens(), 17);
    }
}
