//! OpenAI-compatible client used by SA.
//!
//! SA keeps one canonical internal request/response model based on chat-style
//! messages and tool calls, then maps that model to the configured wire API:
//! - `chat_completions` => `POST /v1/chat/completions`
//! - `responses` => `POST /v1/responses`
//!
//! This keeps the agent loop and compaction logic stable while still allowing
//! compatibility with providers that expose only the newer Responses endpoint.

use anyhow::Context as _;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use uuid::Uuid;

/// Wire protocol used for one OpenAI-compatible endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WireApi {
    /// Classic `POST /v1/chat/completions`.
    #[default]
    #[serde(
        rename = "chat_completions",
        alias = "chat-completions",
        alias = "chat",
        alias = "chatcompletions"
    )]
    ChatCompletions,

    /// Newer `POST /v1/responses`.
    #[serde(rename = "responses", alias = "response")]
    Responses,
}

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
    /// Selected wire protocol.
    wire_api: WireApi,
}

impl OpenAiClient {
    /// Create a new client using the historical default wire protocol:
    /// Chat Completions.
    pub fn new(base_url: String, api_key: String) -> anyhow::Result<Self> {
        Self::with_wire_api(base_url, api_key, WireApi::ChatCompletions)
    }

    /// Create a new client with an explicit wire protocol.
    pub fn with_wire_api(
        base_url: String,
        api_key: String,
        wire_api: WireApi,
    ) -> anyhow::Result<Self> {
        // Normalize base URL by trimming trailing slashes. This avoids double
        // slashes when we append endpoint suffixes.
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

        Ok(Self {
            http,
            base_url,
            wire_api,
        })
    }

    /// Return the configured wire protocol.
    pub fn wire_api(&self) -> WireApi {
        self.wire_api
    }

    /// Compute the `chat/completions` URL.
    fn chat_completions_url(&self) -> String {
        if self.path_ends_with("/chat/completions") {
            return self.base_url.clone();
        }

        format!("{}/chat/completions", self.base_url)
    }

    /// Compute the `responses` URL.
    fn responses_url(&self) -> String {
        if self.path_ends_with("/responses") {
            return self.base_url.clone();
        }

        let normalized_base = self.base_url.trim_end_matches('/');

        if let Some(prefix) = normalized_base.strip_suffix("/chat/completions") {
            return format!("{prefix}/responses");
        }

        if self.has_explicit_api_path() {
            format!("{normalized_base}/responses")
        } else {
            format!("{normalized_base}/v1/responses")
        }
    }

    /// Return whether the configured base URL already ends with one exact suffix.
    fn path_ends_with(&self, suffix: &str) -> bool {
        if let Ok(url) = reqwest::Url::parse(&self.base_url) {
            return url.path().trim_end_matches('/').ends_with(suffix);
        }

        self.base_url.trim_end_matches('/').ends_with(suffix)
    }

    /// Return whether the configured base URL already carries a concrete path.
    fn has_explicit_api_path(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.base_url) else {
            return false;
        };

        let path = url.path().trim_end_matches('/');
        !path.is_empty() && path != "/"
    }

    /// Send one canonical SA request through the configured provider wire API.
    pub async fn chat_completions(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        match self.wire_api {
            WireApi::ChatCompletions => self.send_chat_completions_request(req).await,
            WireApi::Responses => self.send_responses_request(req).await,
        }
    }

    /// Send a classic `POST /chat/completions` request.
    async fn send_chat_completions_request(
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

    /// Send one `POST /responses` request and normalize it back into SA's
    /// canonical chat-style response model.
    async fn send_responses_request(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.responses_url();
        let wire_request = ResponsesRequest::from_chat_request(req);

        let resp = self
            .http
            .post(url)
            .json(&wire_request)
            .send()
            .await
            .map_err(ChatCompletionsError::Transport)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            let body = truncate_error_body(body);
            return Err(ChatCompletionsError::Http { status, body });
        }

        let wire_response = resp
            .json::<ResponsesResponse>()
            .await
            .map_err(ChatCompletionsError::Transport)?;

        Ok(normalize_responses_response(wire_response))
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

/// Canonical SA request body.
///
/// Important:
/// - When `wire_api = "chat_completions"`, this struct is serialized directly.
/// - When `wire_api = "responses"`, this struct is converted into the Responses
///   API request shape first.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionsRequest {
    /// Model name.
    pub model: String,
    /// Chat messages.
    pub messages: Vec<ChatMessage>,

    /// Optional output token cap.
    ///
    /// We keep the internal field name as `max_tokens`.
    ///
    /// Mapping by wire protocol:
    /// - Chat Completions => `max_tokens`
    /// - Responses => `max_output_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Optional reasoning depth / effort for GPT-family reasoning models.
    ///
    /// Mapping by wire protocol:
    /// - Chat Completions => top-level `reasoning_effort`
    /// - Responses => top-level `reasoning: { effort }`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Tool definitions (OpenAI function-calling style).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Tool choice. We set `"auto"` to let the model choose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    /// Whether to use streaming. (We keep it `false` in this minimal agent.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A single message in SA's canonical history format.
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

    /// Normalized Responses API items that originally produced this assistant
    /// turn.
    ///
    /// Why keep this around?
    /// - reasoning-capable Responses models may emit `reasoning` items
    /// - subsequent tool turns should resend those items instead of trying to
    ///   reconstruct them from plain text
    /// - this preserves fidelity across multi-step tool loops when
    ///   `wire_api = "responses"`
    #[serde(skip_serializing, default)]
    pub responses_input_items: Option<Vec<ResponsesInputItem>>,
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
            responses_input_items: None,
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
            responses_input_items: None,
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
    pub parameters: Value,
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

/// Canonical SA response body.
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

/// OpenAI Responses wire request.
#[derive(Debug, Clone, Serialize)]
struct ResponsesRequest {
    /// Model name.
    model: String,
    /// Conversation items.
    input: Vec<ResponsesInputItem>,
    /// Flattened system/developer instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    /// Output cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    /// Optional reasoning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    /// Tool choice policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    /// Streaming toggle.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

impl ResponsesRequest {
    /// Convert SA's canonical chat-style request into Responses API input items.
    fn from_chat_request(req: &ChatCompletionsRequest) -> Self {
        let mut instructions = Vec::<String>::new();
        let mut input = Vec::<ResponsesInputItem>::new();

        for message in &req.messages {
            match message.role.as_str() {
                "assistant" => {
                    if let Some(items) = message.responses_input_items.as_ref() {
                        input.extend(items.iter().cloned());
                    } else {
                        input.extend(build_assistant_responses_items(message));
                    }
                }
                "tool" => {
                    if let Some(item) = build_tool_output_item(message) {
                        input.push(item);
                    } else if let Some(text) = non_empty_text(message.content.as_deref()) {
                        input.push(ResponsesInputItem::user_text(format!(
                            "[Tool result]\n{text}"
                        )));
                    }
                }
                "user" => {
                    if let Some(text) = message.content.as_deref() {
                        input.push(ResponsesInputItem::user_text(text.to_string()));
                    }
                }
                _ => {
                    if let Some(text) = non_empty_text(message.content.as_deref()) {
                        instructions.push(text.to_string());
                    }
                }
            }
        }

        Self {
            model: req.model.clone(),
            input,
            instructions: (!instructions.is_empty()).then(|| instructions.join("\n\n")),
            max_output_tokens: req.max_tokens,
            reasoning: req
                .reasoning_effort
                .as_deref()
                .and_then(|effort| non_empty_text(Some(effort)))
                .map(|effort| ResponsesReasoning {
                    effort: effort.to_string(),
                }),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
            stream: req.stream,
        }
    }
}

/// OpenAI Responses message/tool/reasoning input item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputItem {
    /// Text message history item.
    Message {
        /// Sender role.
        role: String,
        /// Message content parts.
        content: Vec<ResponsesContentPart>,
    },

    /// Assistant function-call item.
    FunctionCall {
        /// Optional provider-side item id.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool-call correlation id used by later `function_call_output`.
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        /// Tool name.
        name: String,
        /// JSON-encoded arguments string.
        arguments: String,
    },

    /// Tool result item.
    FunctionCallOutput {
        /// The prior function-call correlation id.
        call_id: String,
        /// Tool output text.
        output: String,
    },

    /// Reasoning breadcrumb item preserved across turns.
    Reasoning {
        /// Optional plaintext reasoning fragment.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// Optional encrypted reasoning blob.
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        /// Optional reasoning summary.
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
    },
}

impl ResponsesInputItem {
    /// Build one text message item.
    fn message_text(role: impl Into<String>, kind: impl Into<String>, text: String) -> Self {
        Self::Message {
            role: role.into(),
            content: vec![ResponsesContentPart::text(kind, text)],
        }
    }

    /// Build one user text item.
    fn user_text(text: String) -> Self {
        Self::message_text("user", "input_text", text)
    }

    /// Return the first textual payload carried by this item, if any.
    fn text(&self) -> Option<String> {
        match self {
            ResponsesInputItem::Message { content, .. } => {
                let parts = content
                    .iter()
                    .filter_map(|part| non_empty_text(part.text.as_deref()).map(str::to_string))
                    .collect::<Vec<_>>();
                (!parts.is_empty()).then(|| parts.join("\n"))
            }
            ResponsesInputItem::FunctionCall { .. }
            | ResponsesInputItem::FunctionCallOutput { .. }
            | ResponsesInputItem::Reasoning { .. } => None,
        }
    }
}

/// Text content part inside one Responses message item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponsesContentPart {
    /// Part discriminator such as `input_text` or `output_text`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Text payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl ResponsesContentPart {
    /// Build one text content part.
    fn text(kind: impl Into<String>, text: String) -> Self {
        Self {
            kind: kind.into(),
            text: Some(text),
        }
    }
}

/// Minimal reasoning options for the Responses request.
#[derive(Debug, Clone, Serialize)]
struct ResponsesReasoning {
    /// Requested reasoning effort.
    effort: String,
}

/// Responses wire response.
#[derive(Debug, Clone, Deserialize)]
struct ResponsesResponse {
    /// Output items emitted by the provider.
    #[serde(default)]
    output: Vec<ResponsesOutputItem>,
    /// Some providers also mirror final text here.
    #[serde(default)]
    output_text: Option<String>,
    /// Provider-reported usage.
    #[serde(default)]
    usage: Option<ChatUsage>,
}

/// One raw output item from the Responses API.
#[derive(Debug, Clone, Deserialize)]
struct ResponsesOutputItem {
    /// Item discriminator.
    #[serde(rename = "type", default)]
    kind: Option<String>,
    /// Optional provider item id.
    #[serde(default)]
    id: Option<String>,
    /// Optional tool call id.
    #[serde(default)]
    call_id: Option<String>,
    /// Function name for `function_call`.
    #[serde(default)]
    name: Option<String>,
    /// Function arguments for `function_call`.
    #[serde(default)]
    arguments: Option<String>,
    /// Role for `message`.
    #[serde(default)]
    role: Option<String>,
    /// Raw content payload.
    ///
    /// Different item kinds reuse the same field name with different shapes:
    /// - `message` => array of content parts
    /// - some `reasoning` payloads => plain string
    #[serde(default)]
    content: Option<Value>,
    /// Plain text for direct `output_text` items or loose payloads.
    #[serde(default)]
    text: Option<String>,
    /// Optional reasoning encrypted blob.
    #[serde(default)]
    encrypted_content: Option<String>,
    /// Optional reasoning summary.
    #[serde(default)]
    summary: Option<String>,
}

/// Normalize a Responses reply back into SA's canonical chat-style response.
fn normalize_responses_response(response: ResponsesResponse) -> ChatCompletionsResponse {
    let (content, tool_calls, history_items) =
        normalize_responses_output_items(&response.output, response.output_text.as_deref());

    let finish_reason = if tool_calls.is_empty() {
        Some("stop".to_string())
    } else {
        Some("tool_calls".to_string())
    };

    let message = ChatMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
        request_usage: None,
        responses_input_items: (!history_items.is_empty()).then_some(history_items),
    };

    ChatCompletionsResponse {
        choices: vec![ChatChoice {
            message,
            finish_reason,
        }],
        usage: response.usage,
    }
}

/// Convert raw Responses output items into:
/// - assistant visible text
/// - assistant tool calls
/// - normalized history items that can be replayed on the next `/responses` turn
fn normalize_responses_output_items(
    output: &[ResponsesOutputItem],
    top_level_output_text: Option<&str>,
) -> (Option<String>, Vec<ToolCall>, Vec<ResponsesInputItem>) {
    let mut assistant_text: Option<String> = None;
    let mut tool_calls = Vec::<ToolCall>::new();
    let mut history_items = Vec::<ResponsesInputItem>::new();

    for item in output {
        match item.kind.as_deref() {
            Some("message") => {
                let role =
                    normalize_openresponses_role(item.role.as_deref().unwrap_or("assistant"));
                let parts = normalize_responses_message_parts(role, item.content.as_ref());
                if parts.is_empty() {
                    continue;
                }

                let normalized_item = ResponsesInputItem::Message {
                    role: role.to_string(),
                    content: parts,
                };

                if role == "assistant" && assistant_text.is_none() {
                    assistant_text = normalized_item.text();
                }

                history_items.push(normalized_item);
            }
            Some("function_call") => {
                let Some(name) = non_empty_text(item.name.as_deref()) else {
                    continue;
                };
                let arguments = item.arguments.clone().unwrap_or_else(|| "{}".to_string());
                let call_id = sanitize_id(item.call_id.as_deref());
                let item_id = sanitize_id(item.id.as_deref());
                let normalized_call_id = call_id
                    .clone()
                    .or_else(|| item_id.clone())
                    .unwrap_or_else(|| Uuid::new_v4().to_string());

                tool_calls.push(ToolCall {
                    id: normalized_call_id.clone(),
                    kind: "function".to_string(),
                    function: ToolFunctionCall {
                        name: name.to_string(),
                        arguments: arguments.clone(),
                    },
                });

                history_items.push(ResponsesInputItem::FunctionCall {
                    id: item_id,
                    call_id: Some(normalized_call_id),
                    name: name.to_string(),
                    arguments,
                });
            }
            Some("reasoning") => {
                let content = non_empty_text(
                    extract_responses_reasoning_text(item.content.as_ref())
                        .as_deref()
                        .or(item.text.as_deref())
                        .or(item.summary.as_deref()),
                )
                .map(str::to_string);
                let encrypted_content =
                    non_empty_text(item.encrypted_content.as_deref()).map(str::to_string);
                let summary = non_empty_text(item.summary.as_deref()).map(str::to_string);

                if content.is_none() && encrypted_content.is_none() && summary.is_none() {
                    continue;
                }

                history_items.push(ResponsesInputItem::Reasoning {
                    content,
                    encrypted_content,
                    summary,
                });
            }
            Some("output_text") => {
                let Some(text) = non_empty_text(item.text.as_deref()).map(str::to_string) else {
                    continue;
                };

                if assistant_text.is_none() {
                    assistant_text = Some(text.clone());
                }

                history_items.push(ResponsesInputItem::message_text(
                    "assistant",
                    "output_text",
                    text,
                ));
            }
            _ => {}
        }
    }

    if assistant_text.is_none() {
        if let Some(text) = non_empty_text(top_level_output_text) {
            let text = text.to_string();
            assistant_text = Some(text.clone());
            history_items.push(ResponsesInputItem::message_text(
                "assistant",
                "output_text",
                text,
            ));
        }
    }

    (assistant_text, tool_calls, history_items)
}

/// Normalize raw message content parts so they can be replayed back into
/// `input[]` on a later Responses turn.
fn normalize_responses_message_parts(
    role: &str,
    raw_content: Option<&Value>,
) -> Vec<ResponsesContentPart> {
    let default_kind = match role {
        "assistant" => "output_text",
        _ => "input_text",
    };

    let Some(parts) = raw_content.and_then(Value::as_array) else {
        return Vec::new();
    };

    parts
        .iter()
        .filter_map(|part| {
            let text = non_empty_text(part.get("text").and_then(Value::as_str))?.to_string();
            let kind = part
                .get("type")
                .and_then(Value::as_str)
                .filter(|kind| matches!(*kind, "input_text" | "output_text"))
                .unwrap_or(default_kind);
            Some(ResponsesContentPart::text(kind, text))
        })
        .collect()
}

/// Extract plaintext reasoning content from a polymorphic `content` field.
fn extract_responses_reasoning_text(raw_content: Option<&Value>) -> Option<String> {
    match raw_content {
        Some(Value::String(text)) => non_empty_text(Some(text)).map(str::to_string),
        _ => None,
    }
}

/// Build assistant-side Responses items from SA's canonical assistant message.
fn build_assistant_responses_items(message: &ChatMessage) -> Vec<ResponsesInputItem> {
    let mut items = Vec::<ResponsesInputItem>::new();

    if let Some(text) = message.content.as_deref() {
        items.push(ResponsesInputItem::message_text(
            "assistant",
            "output_text",
            text.to_string(),
        ));
    }

    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for tool_call in tool_calls {
            let call_id =
                sanitize_id(Some(&tool_call.id)).unwrap_or_else(|| Uuid::new_v4().to_string());
            items.push(ResponsesInputItem::FunctionCall {
                id: None,
                call_id: Some(call_id),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
            });
        }
    }

    items
}

/// Build a `function_call_output` item from one canonical SA tool result message.
fn build_tool_output_item(message: &ChatMessage) -> Option<ResponsesInputItem> {
    let call_id = sanitize_id(message.tool_call_id.as_deref())?;
    let output = message.content.clone().unwrap_or_default();
    Some(ResponsesInputItem::FunctionCallOutput { call_id, output })
}

/// Map arbitrary chat-style roles into the roles accepted by the OpenResponses schema.
fn normalize_openresponses_role(role: &str) -> &'static str {
    match role {
        "system" => "system",
        "developer" => "developer",
        "assistant" => "assistant",
        _ => "user",
    }
}

/// Strip and reject empty text values.
fn non_empty_text(text: Option<&str>) -> Option<&str> {
    let text = text?.trim();
    if text.is_empty() {
        return None;
    }
    Some(text)
}

/// Normalize an id-like field by trimming whitespace and rejecting empty values.
fn sanitize_id(value: Option<&str>) -> Option<String> {
    non_empty_text(value).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::{
        ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, ChatUsage, InputTokenDetails,
        OutputTokenDetails, ResponsesInputItem, ResponsesRequest, ToolCall, ToolDefinition,
        ToolFunctionCall, ToolFunctionDefinition, WireApi, normalize_responses_response,
    };

    #[test]
    fn wire_api_aliases_deserialize() {
        let parsed = serde_json::from_str::<WireApi>(r#""chat""#).expect("chat alias should parse");
        assert_eq!(parsed, WireApi::ChatCompletions);

        let parsed =
            serde_json::from_str::<WireApi>(r#""responses""#).expect("responses should parse");
        assert_eq!(parsed, WireApi::Responses);
    }

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
    fn chat_message_serialization_omits_internal_fields() {
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
        message.responses_input_items = Some(vec![ResponsesInputItem::user_text(
            "shadow state".to_string(),
        )]);

        let value = serde_json::to_value(message).expect("serialize message");
        assert!(value.get("request_usage").is_none());
        assert!(value.get("responses_input_items").is_none());
    }

    #[test]
    fn responses_request_converts_tool_history_and_reasoning_effort() {
        let req = ChatCompletionsRequest {
            model: "gpt-5.2".to_string(),
            messages: vec![
                ChatMessage::text("developer", "policy"),
                ChatMessage::text("user", "帮我看一下"),
                ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_123".to_string(),
                        kind: "function".to_string(),
                        function: ToolFunctionCall {
                            name: "Read".to_string(),
                            arguments: "{\"path\":\"a.txt\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    request_usage: None,
                    responses_input_items: None,
                },
                ChatMessage::tool_result("call_123", "文件内容"),
            ],
            max_tokens: Some(512),
            reasoning_effort: Some("high".to_string()),
            tools: Some(vec![ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Read".to_string(),
                    description: "读取文件".to_string(),
                    parameters: serde_json::json!({"type":"object"}),
                },
            }]),
            tool_choice: Some(serde_json::json!("auto")),
            stream: Some(false),
        };

        let wire = ResponsesRequest::from_chat_request(&req);
        let value = serde_json::to_value(wire).expect("serialize responses request");

        assert_eq!(value["instructions"], "policy");
        assert_eq!(value["max_output_tokens"], 512);
        assert_eq!(value["reasoning"]["effort"], "high");
        assert_eq!(value["input"][0]["type"], "message");
        assert_eq!(value["input"][0]["role"], "user");
        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][1]["type"], "function_call");
        assert_eq!(value["input"][1]["call_id"], "call_123");
        assert_eq!(value["input"][2]["type"], "function_call_output");
        assert_eq!(value["input"][2]["call_id"], "call_123");
    }

    #[test]
    fn normalize_responses_response_preserves_reasoning_and_function_call_history() {
        let raw = serde_json::json!({
            "output": [
                {
                    "type": "reasoning",
                    "encrypted_content": "ciphertext"
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "Read",
                    "arguments": "{\"path\":\"a.txt\"}"
                }
            ],
            "usage": {
                "input_tokens": 800,
                "output_tokens": 120
            }
        });

        let wire = serde_json::from_value(raw).expect("responses payload should deserialize");
        let normalized = normalize_responses_response(wire);
        let choice = normalized
            .first_choice()
            .expect("normalized choice should exist");
        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls should exist");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].function.name, "Read");
        assert!(choice.message.content.is_none());
        let history = choice
            .message
            .responses_input_items
            .as_ref()
            .expect("responses history items should exist");
        assert_eq!(history.len(), 2);
        assert_eq!(
            normalized.usage.expect("usage should exist").input_tokens,
            Some(800)
        );
    }

    #[test]
    fn normalize_responses_response_extracts_message_text() {
        let raw = serde_json::json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        { "type": "output_text", "text": "第一行" },
                        { "type": "output_text", "text": "第二行" }
                    ]
                }
            ]
        });

        let response = serde_json::from_value(raw).expect("responses payload should deserialize");
        let normalized = normalize_responses_response(response);
        let choice = normalized
            .first_choice()
            .expect("normalized choice should exist");
        assert_eq!(choice.message.content.as_deref(), Some("第一行\n第二行"));
    }

    #[test]
    fn normalize_responses_response_falls_back_to_top_level_output_text() {
        let raw = serde_json::json!({
            "output": [],
            "output_text": "hello from top-level"
        });

        let response = serde_json::from_value(raw).expect("responses payload should deserialize");
        let normalized = normalize_responses_response(response);
        let choice = normalized
            .first_choice()
            .expect("normalized choice should exist");
        assert_eq!(
            choice.message.content.as_deref(),
            Some("hello from top-level")
        );
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
