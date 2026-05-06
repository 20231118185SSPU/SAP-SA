//! OpenAI/Anthropic-compatible client used by SA.
//!
//! SA keeps one canonical internal request/response model based on chat-style
//! messages and tool calls, then maps that model to the configured wire API:
//! - `chat_completions` => `POST /v1/chat/completions`
//! - `responses` => `POST /v1/responses`
//! - `anthropic_messages` => `POST /v1/messages`
//!
//! This keeps the agent loop and compaction logic stable while still allowing
//! compatibility with providers that expose different upstream wire formats.

use anyhow::Context as _;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use ts_rs::TS;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use tokio_stream::StreamExt;
use uuid::Uuid;

/// Wire protocol used for one OpenAI-compatible endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, TS)]
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

    /// Anthropic-style `POST /v1/messages`.
    ///
    /// Alias values such as `anthropic` and `claude` are accepted because the
    /// user-facing need is usually "talk to a Claude-compatible endpoint"
    /// rather than "I specifically know the endpoint is `/v1/messages`".
    #[serde(
        rename = "anthropic_messages",
        alias = "anthropic-messages",
        alias = "anthropic",
        alias = "claude"
    )]
    AnthropicMessages,
}

/// Authentication style used when sending requests to the configured provider.
///
/// Why make this explicit?
/// - OpenAI-style gateways normally expect `Authorization: Bearer ...`
/// - Anthropic-style gateways normally expect `x-api-key: ...`
/// - Anthropic setup-token / OAuth flows instead expect `Authorization`
/// - some vendors proxy Anthropic but keep Anthropic auth semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, TS)]
pub enum AuthStyle {
    /// `Authorization: Bearer <token>`
    #[default]
    #[serde(rename = "bearer", alias = "authorization")]
    Bearer,

    /// `x-api-key: <token>`
    #[serde(rename = "x_api_key", alias = "x-api-key", alias = "api-key")]
    XApiKey,

    /// Anthropic-compatible auto detection:
    /// - setup/OAuth tokens => `Authorization: Bearer ...` + `anthropic-beta`
    /// - normal API keys => `x-api-key: ...`
    #[serde(
        rename = "anthropic_auto",
        alias = "anthropic-auto",
        alias = "anthropic"
    )]
    AnthropicAuto,
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
    /// The provider returned a payload that SA could not parse as the selected
    /// wire protocol.
    InvalidResponse(String),
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
            ChatCompletionsError::InvalidResponse(_) => false,
        }
    }

    /// Return the HTTP status code, if this is an HTTP error.
    pub fn status(&self) -> Option<reqwest::StatusCode> {
        match self {
            ChatCompletionsError::Transport(_) => None,
            ChatCompletionsError::Http { status, .. } => Some(*status),
            ChatCompletionsError::InvalidResponse(_) => None,
        }
    }

    /// Return `true` if this 429 error body indicates the quota is fully
    /// exhausted (as opposed to a temporary rate limit that will reset).
    ///
    /// Checks for these substrings (case-insensitive) in the error body:
    /// - `quota exhausted`
    /// - `quota exceeded`
    /// - `insufficient_quota`
    ///
    /// Returns `false` for non-429 errors and transient rate-limit 429s.
    pub fn is_quota_exhausted(&self) -> bool {
        let body = match self {
            ChatCompletionsError::Http { status, body }
                if *status == reqwest::StatusCode::TOO_MANY_REQUESTS =>
            {
                body
            }
            _ => return false,
        };
        let lower = body.to_lowercase();
        lower.contains("quota exhausted")
            || lower.contains("quota exceeded")
            || lower.contains("insufficient_quota")
    }
}

impl fmt::Display for ChatCompletionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatCompletionsError::Transport(err) => write!(f, "transport error: {err}"),
            ChatCompletionsError::Http { status, body } => {
                write!(f, "http error ({status}): {body}")
            }
            ChatCompletionsError::InvalidResponse(message) => {
                write!(f, "invalid response: {message}")
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
    /// Authentication headers cloned onto each request.
    auth_headers: HeaderMap,
}

impl OpenAiClient {
    /// Create a new client using the historical default wire protocol:
    /// Chat Completions.
    pub fn new(base_url: String, api_key: String) -> anyhow::Result<Self> {
        Self::with_wire_api_and_auth_style(
            base_url,
            api_key,
            WireApi::ChatCompletions,
            AuthStyle::Bearer,
        )
    }

    /// Create a new client with an explicit wire protocol.
    pub fn with_wire_api(
        base_url: String,
        api_key: String,
        wire_api: WireApi,
    ) -> anyhow::Result<Self> {
        let auth_style = match wire_api {
            WireApi::AnthropicMessages => AuthStyle::AnthropicAuto,
            WireApi::ChatCompletions | WireApi::Responses => AuthStyle::Bearer,
        };

        Self::with_wire_api_and_auth_style(base_url, api_key, wire_api, auth_style)
    }

    /// Create a new client with both explicit wire protocol and explicit
    /// authentication style.
    pub fn with_wire_api_and_auth_style(
        base_url: String,
        api_key: String,
        wire_api: WireApi,
        auth_style: AuthStyle,
    ) -> anyhow::Result<Self> {
        // Normalize base URL by trimming trailing slashes. This avoids double
        // slashes when we append endpoint suffixes.
        let base_url = base_url.trim_end_matches('/').to_string();

        // Build default content headers. Authorization-style headers are kept
        // separately because different protocols need different auth schemes.
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("Failed to build HTTP client")?;
        let auth_headers = build_auth_headers(&api_key, auth_style)?;

        Ok(Self {
            http,
            base_url,
            wire_api,
            auth_headers,
        })
    }

    /// Return the configured wire protocol.
    pub fn wire_api(&self) -> WireApi {
        self.wire_api
    }

    /// Resolve the model name based on category and routing table.
    pub fn resolve_model(&self, category: Option<&str>, routing: Option<&HashMap<String, String>>, default_model: &str) -> String {
        if let Some(cat) = category {
            if let Some(routing_table) = routing {
                if let Some(model) = routing_table.get(cat) {
                    return model.clone();
                }
            }
        }
        default_model.to_string()
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

    /// Compute the Anthropic-style `messages` URL.
    fn anthropic_messages_url(&self) -> String {
        if self.path_ends_with("/messages") {
            return self.base_url.clone();
        }

        let normalized_base = self.base_url.trim_end_matches('/');

        if let Some(prefix) = normalized_base.strip_suffix("/chat/completions") {
            return format!("{prefix}/messages");
        }

        if let Some(prefix) = normalized_base.strip_suffix("/responses") {
            return format!("{prefix}/messages");
        }

        if self.has_explicit_api_path() {
            format!("{normalized_base}/messages")
        } else {
            format!("{normalized_base}/v1/messages")
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
        // Filter out empty assistant/tool messages that some providers reject.
        let sanitized = req.sanitized();
        match self.wire_api {
            WireApi::ChatCompletions => self.send_chat_completions_request(&sanitized).await,
            WireApi::Responses => self.send_responses_request(&sanitized).await,
            WireApi::AnthropicMessages => self.send_anthropic_messages_request(&sanitized).await,
        }
    }

    /// Apply the configured authentication headers to one outgoing request.
    fn apply_auth_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request.headers(self.auth_headers.clone())
    }

    /// Send a classic `POST /chat/completions` request.
    async fn send_chat_completions_request(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        if req.stream.unwrap_or(false) {
            match self.send_chat_completions_streaming_request(req).await {
                Ok(response) => return Ok(response),
                Err(error) if should_fallback_from_streaming(&error) => {
                    let mut fallback_req = req.clone();
                    fallback_req.stream = Some(false);
                    return self
                        .send_chat_completions_request_non_streaming(&fallback_req)
                        .await;
                }
                Err(error) => return Err(error),
            }
        }

        self.send_chat_completions_request_non_streaming(req).await
    }

    /// Send a classic non-streaming `POST /chat/completions` request.
    async fn send_chat_completions_request_non_streaming(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.chat_completions_url();

        // Send request.
        let resp = self
            .apply_auth_headers(self.http.post(url))
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

    /// Send a streaming `POST /chat/completions` request and aggregate it back
    /// into the canonical non-streaming SA response shape.
    async fn send_chat_completions_streaming_request(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.chat_completions_url();
        let resp = self
            .apply_auth_headers(self.http.post(url))
            .header("Accept", "text/event-stream")
            .json(req)
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

        let mut accumulator = ChatCompletionsStreamAccumulator::default();
        self.read_sse_response(resp, |event_name, data| {
            if data == "[DONE]" {
                return Ok(Some(()));
            }
            if event_name.is_some_and(|event| !event.eq_ignore_ascii_case("message")) {
                return Ok(None);
            }

            let chunk =
                serde_json::from_str::<ChatCompletionsStreamChunk>(data).map_err(|err| {
                    invalid_response(format!(
                        "chat-completions stream chunk is not valid JSON: {err}; body={}",
                        summarize_stream_payload(data)
                    ))
                })?;
            accumulator.apply_chunk(chunk);
            Ok(None)
        })
        .await?;

        accumulator.into_response()
    }

    /// Send one `POST /responses` request and normalize it back into SA's
    /// canonical chat-style response model.
    async fn send_responses_request(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        if req.stream.unwrap_or(false) {
            match self.send_responses_streaming_request(req).await {
                Ok(response) => return Ok(response),
                Err(error) if should_fallback_from_streaming(&error) => {
                    let mut fallback_req = req.clone();
                    fallback_req.stream = Some(false);
                    return self
                        .send_responses_request_non_streaming(&fallback_req)
                        .await;
                }
                Err(error) => return Err(error),
            }
        }

        self.send_responses_request_non_streaming(req).await
    }

    /// Send one non-streaming `POST /responses` request and normalize it back into SA's
    /// canonical chat-style response model.
    async fn send_responses_request_non_streaming(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.responses_url();
        let wire_request = ResponsesRequest::from_chat_request(req);

        let resp = self
            .apply_auth_headers(self.http.post(url))
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

    /// Send one streaming `POST /responses` request and aggregate the SSE event
    /// stream into SA's canonical response model.
    async fn send_responses_streaming_request(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.responses_url();
        let wire_request = ResponsesRequest::from_chat_request(req);

        let resp = self
            .apply_auth_headers(self.http.post(url))
            .header("Accept", "text/event-stream")
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

        let mut accumulator = ResponsesStreamAccumulator::default();
        self.read_sse_response(resp, |_, data| {
            let event = serde_json::from_str::<Value>(data).map_err(|err| {
                invalid_response(format!(
                    "responses stream event is not valid JSON: {err}; body={}",
                    summarize_stream_payload(data)
                ))
            })?;
            accumulator.apply_event(&event)
        })
        .await?;

        let response = accumulator
            .completed_response
            .clone()
            .unwrap_or_else(|| accumulator.synthetic_response());
        Ok(normalize_responses_response(response))
    }

    /// Send one Anthropic-style `POST /messages` request and normalize it back
    /// into SA's canonical chat-style response model.
    ///
    /// Current design choice:
    /// - SA always uses the unary `/messages` response here, even when the
    ///   canonical request asked for streaming.
    /// - This keeps Claude-compatible support small and robust first.
    /// - If later needed, Anthropic SSE can be added without changing the
    ///   higher-level agent loop because the normalization boundary stays here.
    async fn send_anthropic_messages_request(
        &self,
        req: &ChatCompletionsRequest,
    ) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let url = self.anthropic_messages_url();
        let wire_request = AnthropicMessagesRequest::from_chat_request(req);

        let resp = self
            .apply_auth_headers(self.http.post(url))
            .header("anthropic-version", "2023-06-01")
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
            .json::<AnthropicMessagesResponse>()
            .await
            .map_err(ChatCompletionsError::Transport)?;

        Ok(normalize_anthropic_messages_response(wire_response))
    }

    /// Read one SSE response until the supplied handler returns a value.
    async fn read_sse_response<T, F>(
        &self,
        response: reqwest::Response,
        mut on_event: F,
    ) -> Result<T, ChatCompletionsError>
    where
        F: FnMut(Option<&str>, &str) -> Result<Option<T>, ChatCompletionsError>,
    {
        let mut buffer = String::new();
        let mut current_event: Option<String> = None;
        let mut current_data: Vec<String> = Vec::new();
        let mut stream = response.bytes_stream();

        while let Some(item) = stream.next().await {
            let bytes = item.map_err(ChatCompletionsError::Transport)?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|err| invalid_response(format!("stream payload is not UTF-8: {err}")))?;
            buffer.push_str(text);

            while let Some(pos) = buffer.find('\n') {
                let mut line = buffer.drain(..=pos).collect::<String>();
                if line.ends_with('\n') {
                    line.pop();
                }
                if line.ends_with('\r') {
                    line.pop();
                }

                if let Some(result) =
                    process_sse_line(&line, &mut current_event, &mut current_data, &mut on_event)?
                {
                    return Ok(result);
                }
            }
        }

        if !buffer.is_empty() {
            let line = std::mem::take(&mut buffer);
            if let Some(result) =
                process_sse_line(&line, &mut current_event, &mut current_data, &mut on_event)?
            {
                return Ok(result);
            }
        }

        if let Some(result) = flush_sse_event(&mut current_event, &mut current_data, &mut on_event)?
        {
            return Ok(result);
        }

        Err(invalid_response(
            "stream ended before a terminal completion event arrived".to_string(),
        ))
    }
}

/// Input for embedding requests — single text or batch.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

/// Response wrapper for the embeddings endpoint.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAiClient {
    /// Call `POST /v1/embeddings` and return embedding vectors.
    pub async fn embeddings(
        &self,
        model: &str,
        input: EmbeddingInput,
        stream: bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let _ = stream; // embeddings API does not stream
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": model,
            "input": match &input {
                EmbeddingInput::Single(s) => serde_json::Value::String(s.clone()),
                EmbeddingInput::Batch(v) => serde_json::Value::Array(
                    v.iter().map(|s| serde_json::Value::String(s.clone())).collect()
                ),
            },
        });
        let resp = self
            .http
            .post(&url)
            .headers(self.auth_headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("embeddings request failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("embeddings API returned {status}: {text}");
        }
        let parsed: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("embeddings response parse failed: {e}"))?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// Build the authentication headers that should be attached to every request
/// made by this client.
fn build_auth_headers(api_key: &str, auth_style: AuthStyle) -> anyhow::Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let api_key = api_key.trim();

    match auth_style {
        AuthStyle::Bearer => {
            let bearer = format!("Bearer {api_key}");
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&bearer)
                    .context("Invalid API key for Authorization header")?,
            );
        }
        AuthStyle::XApiKey => {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key).context("Invalid API key for x-api-key header")?,
            );
        }
        AuthStyle::AnthropicAuto => match detect_anthropic_auth_kind(api_key) {
            AnthropicResolvedAuthKind::ApiKey => {
                headers.insert(
                    "x-api-key",
                    HeaderValue::from_str(api_key)
                        .context("Invalid API key for x-api-key header")?,
                );
            }
            AnthropicResolvedAuthKind::Authorization => {
                let bearer = format!("Bearer {api_key}");
                headers.insert(
                    AUTHORIZATION,
                    HeaderValue::from_str(&bearer)
                        .context("Invalid API key for Authorization header")?,
                );
                headers.insert(
                    "anthropic-beta",
                    HeaderValue::from_static("oauth-2025-04-20"),
                );
            }
        },
    }

    Ok(headers)
}

/// Resolved authentication style used for Anthropic-compatible endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnthropicResolvedAuthKind {
    /// `x-api-key: ...`
    ApiKey,
    /// `Authorization: Bearer ...`
    Authorization,
}

/// Best-effort detection for Anthropic authentication style.
///
/// This intentionally mirrors the mature logic already used in `zeroclaw`:
/// - setup/OAuth tokens often look JWT-like or use Anthropic setup prefixes
/// - regular Anthropic platform keys should go via `x-api-key`
fn detect_anthropic_auth_kind(token: &str) -> AnthropicResolvedAuthKind {
    let trimmed = token.trim();

    if trimmed.starts_with("sk-ant-oat01-") || trimmed.matches('.').count() >= 2 {
        return AnthropicResolvedAuthKind::Authorization;
    }

    AnthropicResolvedAuthKind::ApiKey
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

/// Return whether streaming should fall back to a non-streaming retry.
///
/// We only fall back on errors that strongly suggest "streaming is unsupported
/// or malformed here", not on ordinary transient provider failures like 503.
fn should_fallback_from_streaming(error: &ChatCompletionsError) -> bool {
    match error {
        ChatCompletionsError::Http { status, .. } => matches!(
            *status,
            reqwest::StatusCode::BAD_REQUEST
                | reqwest::StatusCode::NOT_FOUND
                | reqwest::StatusCode::METHOD_NOT_ALLOWED
                | reqwest::StatusCode::NOT_ACCEPTABLE
                | reqwest::StatusCode::UNPROCESSABLE_ENTITY
                | reqwest::StatusCode::NOT_IMPLEMENTED
        ),
        ChatCompletionsError::InvalidResponse(_) => true,
        ChatCompletionsError::Transport(_) => false,
    }
}

/// Build one protocol/shape error for provider payloads that are syntactically
/// reachable but semantically unusable.
fn invalid_response(message: String) -> ChatCompletionsError {
    ChatCompletionsError::InvalidResponse(message)
}

/// Summarize one raw stream payload so parser errors stay readable.
fn summarize_stream_payload(payload: &str) -> String {
    const MAX_CHARS: usize = 500;
    let compact = payload.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= MAX_CHARS {
        return compact;
    }
    let truncated: String = compact.chars().take(MAX_CHARS).collect();
    format!("{truncated}…(truncated)")
}

/// Process one SSE line and flush completed events when a blank separator is seen.
fn process_sse_line<T, F>(
    line: &str,
    current_event: &mut Option<String>,
    current_data: &mut Vec<String>,
    on_event: &mut F,
) -> Result<Option<T>, ChatCompletionsError>
where
    F: FnMut(Option<&str>, &str) -> Result<Option<T>, ChatCompletionsError>,
{
    if line.is_empty() {
        return flush_sse_event(current_event, current_data, on_event);
    }

    if line.starts_with(':') {
        return Ok(None);
    }

    if let Some(rest) = line.strip_prefix("event:") {
        *current_event = Some(rest.trim().to_string());
        return Ok(None);
    }

    if let Some(rest) = line.strip_prefix("data:") {
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        current_data.push(rest.to_string());
    }

    Ok(None)
}

/// Flush one accumulated SSE event frame into the supplied event handler.
fn flush_sse_event<T, F>(
    current_event: &mut Option<String>,
    current_data: &mut Vec<String>,
    on_event: &mut F,
) -> Result<Option<T>, ChatCompletionsError>
where
    F: FnMut(Option<&str>, &str) -> Result<Option<T>, ChatCompletionsError>,
{
    if current_event.is_none() && current_data.is_empty() {
        return Ok(None);
    }

    let event = current_event.take();
    let data = current_data.join("\n");
    current_data.clear();
    let data = data.trim();

    if data.is_empty() {
        return Ok(None);
    }

    on_event(event.as_deref(), data)
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
    /// - Anthropic Messages => `max_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Optional reasoning depth / effort for GPT-family reasoning models.
    ///
    /// Mapping by wire protocol:
    /// - Chat Completions => top-level `reasoning_effort`
    /// - Responses => top-level `reasoning: { effort }`
    /// - Anthropic Messages => currently ignored
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

    /// Sampling temperature (0.0 – 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus sampling parameter (0.0 – 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl ChatCompletionsRequest {
    /// Return a copy of this request with empty messages filtered out.
    ///
    /// Some providers reject assistant messages that carry neither `content`
    /// nor `tool_calls`.  A tool-execution loop can occasionally produce such
    /// messages (e.g. when a tool result arrives but the prior assistant turn
    /// had no visible text or tool calls).  Stripping them before dispatch
    /// prevents 400 Bad Request errors.
    pub fn sanitized(&self) -> Self {
        let messages = self
            .messages
            .iter()
            .filter(|msg| match msg.role.as_str() {
                "assistant" => msg.content.is_some() || msg.tool_calls.is_some(),
                "tool" => msg.content.is_some() && msg.tool_call_id.is_some(),
                _ => true,
            })
            .cloned()
            .collect();
        Self {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            reasoning_effort: self.reasoning_effort.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            stream: self.stream,
            temperature: self.temperature,
            top_p: self.top_p,
        }
    }
}

/// Content of a chat message: either plain text or a multimodal list.
///
/// Uses `#[serde(untagged)]` so that existing on-disk sessions containing a
/// plain JSON string deserialise into the `Text` variant transparently.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text string (backward-compatible with existing serialized data).
    Text(String),
    /// Multimodal content parts (text + images).
    Parts(Vec<ContentPart>),
}

/// One part inside a multimodal message content array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text block.
    Text { text: String },
    /// An image block with a URL (data: or https:).
    Image {
        /// Image URL details.
        image_url: ImageUrl,
    },
}

/// Image URL wrapper matching OpenAI's content-part format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUrl {
    /// Either a `data:image/...;base64,...` URL or an `https://` URL.
    pub url: String,
    /// Detail level: `"auto"`, `"low"`, or `"high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A single message in SA's canonical history format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role name (e.g. `system`, `developer`, `user`, `assistant`, `tool`).
    pub role: String,

    /// Text or multimodal content.
    ///
    /// For tool-calls, providers often return `null` content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,

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
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
            request_usage: None,
            responses_input_items: None,
        }
    }

    /// Construct a multimodal message with text + image content parts.
    pub fn multimodal(role: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        Self {
            role: role.into(),
            content: Some(MessageContent::Parts(parts)),
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
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            request_usage: None,
            responses_input_items: None,
        }
    }

    /// Extract plain text from content, regardless of `Text` or `Parts` variant.
    ///
    /// For `Parts`, only `ContentPart::Text` items are included; images are
    /// represented as a placeholder string like `[Image: data:image/png;base64,...]`.
    pub fn text_content(&self) -> Option<String> {
        self.content.as_ref().and_then(|c| match c {
            MessageContent::Text(s) => {
                let trimmed = s.trim().to_string();
                (!trimmed.is_empty()).then_some(trimmed)
            }
            MessageContent::Parts(parts) => {
                let mut pieces = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            let t = text.trim();
                            if !t.is_empty() {
                                pieces.push(t.to_string());
                            }
                        }
                        ContentPart::Image { image_url } => {
                            // Use a compact placeholder so the agent knows an image is present.
                            let preview = if image_url.url.len() > 80 {
                                format!("[Image: {}...]", &image_url.url[..60])
                            } else {
                                format!("[Image: {}]", image_url.url)
                            };
                            pieces.push(preview);
                        }
                    }
                }
                (!pieces.is_empty()).then_some(pieces.join("\n"))
            }
        })
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

/// One streamed chat-completions chunk.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsStreamChunk {
    /// Stream choices.
    #[serde(default)]
    choices: Vec<ChatCompletionsStreamChoice>,
    /// Optional streamed usage object.
    #[serde(default)]
    usage: Option<ChatUsage>,
}

/// One streamed choice delta.
#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsStreamChoice {
    /// Delta payload.
    #[serde(default)]
    delta: ChatCompletionsStreamDelta,
    /// Finish reason, if this chunk closes the turn.
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Streamed assistant delta payload.
#[derive(Debug, Clone, Deserialize, Default)]
struct ChatCompletionsStreamDelta {
    /// Optional role.
    #[serde(default)]
    role: Option<String>,
    /// Streamed text content delta.
    #[serde(default)]
    content: Option<String>,
    /// Streamed tool-call fragments.
    #[serde(default)]
    tool_calls: Option<Vec<ChatCompletionsStreamToolCallDelta>>,
}

/// Streamed tool-call fragment.
#[derive(Debug, Clone, Deserialize, Default)]
struct ChatCompletionsStreamToolCallDelta {
    /// Stable tool-call slot index.
    #[serde(default)]
    index: usize,
    /// Optional tool-call id fragment.
    #[serde(default)]
    id: Option<String>,
    /// Optional type discriminator.
    #[serde(rename = "type", default)]
    kind: Option<String>,
    /// Function payload delta.
    #[serde(default)]
    function: Option<ChatCompletionsStreamFunctionDelta>,
}

/// Streamed function-call fragment.
#[derive(Debug, Clone, Deserialize, Default)]
struct ChatCompletionsStreamFunctionDelta {
    /// Optional function name fragment.
    #[serde(default)]
    name: Option<String>,
    /// Optional function arguments fragment.
    #[serde(default)]
    arguments: Option<String>,
}

/// Mutable accumulator for one streamed chat-completions response.
#[derive(Debug, Clone, Default)]
struct ChatCompletionsStreamAccumulator {
    /// Assistant role if streamed explicitly.
    role: Option<String>,
    /// Accumulated assistant text.
    content: String,
    /// Incrementally reconstructed tool calls.
    tool_calls: Vec<ChatCompletionsStreamToolCallAccumulator>,
    /// Final finish reason, if received.
    finish_reason: Option<String>,
    /// Optional provider usage.
    usage: Option<ChatUsage>,
}

impl ChatCompletionsStreamAccumulator {
    /// Merge one streamed chunk.
    fn apply_chunk(&mut self, chunk: ChatCompletionsStreamChunk) {
        if chunk.usage.is_some() {
            self.usage = chunk.usage;
        }

        for choice in chunk.choices {
            if let Some(role) = choice.delta.role {
                self.role = Some(role);
            }

            if let Some(text) = choice.delta.content {
                self.content.push_str(&text);
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    self.apply_tool_call_delta(tool_call);
                }
            }

            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }
        }
    }

    /// Merge one streamed tool-call delta by index.
    fn apply_tool_call_delta(&mut self, delta: ChatCompletionsStreamToolCallDelta) {
        while self.tool_calls.len() <= delta.index {
            self.tool_calls
                .push(ChatCompletionsStreamToolCallAccumulator::default());
        }

        let slot = &mut self.tool_calls[delta.index];
        if let Some(id) = delta.id {
            slot.id.get_or_insert(id);
        }
        if let Some(kind) = delta.kind {
            slot.kind.get_or_insert(kind);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                slot.name.get_or_insert(name);
            }
            if let Some(arguments) = function.arguments {
                slot.arguments.push_str(&arguments);
            }
        }
    }

    /// Convert the accumulated stream state into SA's canonical response.
    fn into_response(self) -> Result<ChatCompletionsResponse, ChatCompletionsError> {
        let content = (!self.content.trim().is_empty()).then_some(self.content);
        let tool_calls = self
            .tool_calls
            .into_iter()
            .filter_map(ChatCompletionsStreamToolCallAccumulator::into_tool_call)
            .collect::<Vec<_>>();

        let finish_reason = self.finish_reason.or_else(|| {
            if tool_calls.is_empty() {
                Some("stop".to_string())
            } else {
                Some("tool_calls".to_string())
            }
        });

        Ok(ChatCompletionsResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: self.role.unwrap_or_else(|| "assistant".to_string()),
                    content: content.map(MessageContent::Text),
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    tool_call_id: None,
                    request_usage: None,
                    responses_input_items: None,
                },
                finish_reason,
            }],
            usage: self.usage,
        })
    }
}

/// One incrementally reconstructed streamed tool call.
#[derive(Debug, Clone, Default)]
struct ChatCompletionsStreamToolCallAccumulator {
    /// Tool-call id.
    id: Option<String>,
    /// Type discriminator.
    kind: Option<String>,
    /// Function name.
    name: Option<String>,
    /// Concatenated JSON argument string.
    arguments: String,
}

impl ChatCompletionsStreamToolCallAccumulator {
    /// Convert a reconstructed tool call into the canonical SA form.
    fn into_tool_call(self) -> Option<ToolCall> {
        let name = sanitize_id(self.name.as_deref())?;
        Some(ToolCall {
            id: self.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            kind: self.kind.unwrap_or_else(|| "function".to_string()),
            function: ToolFunctionCall {
                name,
                arguments: if self.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    self.arguments
                },
            },
        })
    }
}

/// Conservative default `max_tokens` for Anthropic `/messages` when SA did not
/// set an explicit completion budget.
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 4_096;

/// Anthropic `/messages` request body.
#[derive(Debug, Clone, Serialize)]
struct AnthropicMessagesRequest {
    /// Model identifier.
    model: String,
    /// Completion budget.
    max_tokens: u32,
    /// Top-level system/developer instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    /// Conversation history.
    messages: Vec<AnthropicMessage>,
    /// Optional native tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicToolDefinition>>,
}

impl AnthropicMessagesRequest {
    /// Convert SA's canonical request into Anthropic `/messages`.
    fn from_chat_request(req: &ChatCompletionsRequest) -> Self {
        let (system, messages) = convert_messages_to_anthropic(&req.messages);
        let tools = req.tools.as_ref().and_then(|tools| {
            let native = tools
                .iter()
                .map(AnthropicToolDefinition::from_tool_definition)
                .collect::<Vec<_>>();
            (!native.is_empty()).then_some(native)
        });

        // Inject prompt caching breakpoints:
        // - last tool_result message (highest reuse value)
        // - last user message (catches the latest context)
        let mut messages = messages;
        inject_anthropic_cache_breakpoints(&mut messages);

        Self {
            model: req.model.clone(),
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS),
            system,
            messages,
            tools,
        }
    }
}

/// Anthropic conversation message.
#[derive(Debug, Clone, Serialize)]
struct AnthropicMessage {
    /// Anthropic role (`user` or `assistant`).
    role: String,
    /// Structured content blocks.
    content: Vec<AnthropicContentOut>,
}

impl AnthropicMessage {
    /// Build a simple text-only user message.
    fn user_text(text: String) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![AnthropicContentOut::Text { text, cache_control: None }],
        }
    }
}

/// Anthropic cache control directive.
#[derive(Debug, Clone, Serialize)]
struct CacheControl {
    /// Cache type — always `"ephemeral"` for prompt caching.
    #[serde(rename = "type")]
    type_: String,
}

impl CacheControl {
    /// Create an ephemeral cache control directive.
    fn ephemeral() -> Self {
        Self { type_: "ephemeral".to_string() }
    }
}

/// Outbound Anthropic content block.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentOut {
    /// Plain text block.
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Assistant-native tool-use block.
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Provider-stable tool-use id.
        id: String,
        /// Tool name.
        name: String,
        /// JSON arguments object.
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// User-side tool-result block answering one previous `tool_use`.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// Previously emitted `tool_use.id`.
        tool_use_id: String,
        /// Textual tool result body.
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image block for multimodal messages.
    #[serde(rename = "image")]
    Image {
        /// Image source details.
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Image source for Anthropic's `image` content block.
#[derive(Debug, Clone, Serialize)]
struct AnthropicImageSource {
    /// Source type — always `"base64"` for inline images.
    #[serde(rename = "type")]
    type_: String,
    /// MIME type, e.g. `"image/png"`.
    media_type: String,
    /// Base64-encoded image bytes.
    data: String,
}

/// Native Anthropic tool definition.
#[derive(Debug, Clone, Serialize)]
struct AnthropicToolDefinition {
    /// Tool name.
    name: String,
    /// Human-readable description.
    description: String,
    /// JSON schema for tool input.
    input_schema: Value,
}

impl AnthropicToolDefinition {
    /// Convert SA's canonical function tool schema into Anthropic tool format.
    fn from_tool_definition(tool: &ToolDefinition) -> Self {
        Self {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            input_schema: tool.function.parameters.clone(),
        }
    }
}

/// Unary Anthropic `/messages` response.
#[derive(Debug, Clone, Deserialize)]
struct AnthropicMessagesResponse {
    /// Structured content blocks.
    #[serde(default)]
    content: Vec<AnthropicContentIn>,
    /// Provider-native stop reason.
    #[serde(default)]
    stop_reason: Option<String>,
    /// Provider-reported usage.
    #[serde(default)]
    usage: Option<ChatUsage>,
}

/// Inbound Anthropic content block.
#[derive(Debug, Clone, Deserialize)]
struct AnthropicContentIn {
    /// Block discriminator.
    #[serde(rename = "type")]
    kind: String,
    /// Optional text content.
    #[serde(default)]
    text: Option<String>,
    /// Optional tool-use id.
    #[serde(default)]
    id: Option<String>,
    /// Optional tool name.
    #[serde(default)]
    name: Option<String>,
    /// Optional tool input object.
    #[serde(default)]
    input: Option<Value>,
}

/// Convert canonical SA history into Anthropic system text plus `/messages`
/// message blocks.
fn convert_messages_to_anthropic(
    messages: &[ChatMessage],
) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system_parts = Vec::<String>::new();
    let mut native_messages = Vec::<AnthropicMessage>::new();

    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => {
                if let Some(text) = message.text_content() {
                    system_parts.push(text);
                }
            }
            "assistant" => {
                let native = build_anthropic_assistant_message(message);
                if let Some(native) = native {
                    native_messages.push(native);
                }
            }
            "tool" => {
                if let Some(native) = build_anthropic_tool_result_message(message) {
                    native_messages.push(native);
                } else if let Some(text) = message.text_content() {
                    native_messages.push(AnthropicMessage::user_text(text));
                }
            }
            _ => {
                // user (and any other role) → convert multimodal content
                let native = build_anthropic_user_message(message);
                if let Some(native) = native {
                    native_messages.push(native);
                }
            }
        }
    }

    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    (system, native_messages)
}

/// Inject Anthropic prompt caching breakpoints on the last tool_result
/// message and the last user message for maximum cache reuse.
///
/// This marks content blocks with `cache_control: {"type": "ephemeral"}`
/// so Anthropic can cache those breakpoints and avoid re-processing
/// the prefix on subsequent requests.
fn inject_anthropic_cache_breakpoints(messages: &mut [AnthropicMessage]) {
    // Mark the last tool_result message's content blocks.
    if let Some(last_tool_msg) = messages.iter_mut().rev().find(|m| {
        m.role == "user"
            && m.content
                .iter()
                .any(|c| matches!(c, AnthropicContentOut::ToolResult { .. }))
    }) {
        for block in &mut last_tool_msg.content {
            if let AnthropicContentOut::ToolResult { cache_control, .. } = block {
                *cache_control = Some(CacheControl::ephemeral());
            }
        }
    }

    // Mark the last user message's content blocks (skip if already marked above).
    if let Some(last_user_msg) = messages.iter_mut().rev().find(|m| {
        m.role == "user"
            && !m.content.iter().any(|c| matches!(c, AnthropicContentOut::ToolResult { cache_control: Some(_), .. }))
    }) {
        if let Some(last_block) = last_user_msg.content.last_mut() {
            match last_block {
                AnthropicContentOut::Text { cache_control, .. }
                | AnthropicContentOut::Image { cache_control, .. } => {
                    *cache_control = Some(CacheControl::ephemeral());
                }
                _ => {}
            }
        }
    }
}

/// Build an Anthropic user message from SA's canonical message, handling both
/// plain text and multimodal (text + images) content.
fn build_anthropic_user_message(message: &ChatMessage) -> Option<AnthropicMessage> {
    let content = message.content.as_ref()?;

    match content {
        MessageContent::Text(text) => {
            let text = text.trim();
            if text.is_empty() {
                return None;
            }
            Some(AnthropicMessage::user_text(text.to_string()))
        }
        MessageContent::Parts(parts) => {
            let mut blocks = Vec::<AnthropicContentOut>::new();
            let mut has_content = false;

            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        let t = text.trim();
                        if !t.is_empty() {
                            blocks.push(AnthropicContentOut::Text {
                                text: t.to_string(),
                                cache_control: None,
                            });
                            has_content = true;
                        }
                    }
                    ContentPart::Image { image_url } => {
                        if let Some(source) = parse_data_url_to_anthropic_source(&image_url.url) {
                            blocks.push(AnthropicContentOut::Image { source, cache_control: None });
                            has_content = true;
                        }
                    }
                }
            }

            has_content.then(|| AnthropicMessage {
                role: "user".to_string(),
                content: blocks,
            })
        }
    }
}

/// Parse a `data:<media_type>;base64,<data>` URL into an Anthropic image source.
fn parse_data_url_to_anthropic_source(data_url: &str) -> Option<AnthropicImageSource> {
    let url = data_url.trim();
    if !url.starts_with("data:") {
        return None;
    }
    // Expected format: data:image/png;base64,iVBOR...
    let rest = &url[5..];
    let semi = rest.find(';')?;
    let comma = rest.find(',')?;
    if semi > comma {
        return None;
    }
    let media_type = &rest[..semi];
    let encoding = &rest[semi + 1..comma];
    if !encoding.eq_ignore_ascii_case("base64") {
        return None;
    }
    let data = &rest[comma + 1..];
    if data.is_empty() {
        return None;
    }

    Some(AnthropicImageSource {
        type_: "base64".to_string(),
        media_type: media_type.to_string(),
        data: data.to_string(),
    })
}

/// Build one Anthropic assistant message from SA's canonical assistant turn.
fn build_anthropic_assistant_message(message: &ChatMessage) -> Option<AnthropicMessage> {
    let mut content = Vec::<AnthropicContentOut>::new();

    if let Some(text) = message.text_content() {
        content.push(AnthropicContentOut::Text { text, cache_control: None });
    }

    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for tool_call in tool_calls {
            let id = sanitize_id(Some(&tool_call.id)).unwrap_or_else(|| Uuid::new_v4().to_string());
            let input = serde_json::from_str::<Value>(&tool_call.function.arguments)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
            content.push(AnthropicContentOut::ToolUse {
                id,
                name: tool_call.function.name.clone(),
                input,
                cache_control: None,
            });
        }
    }

    (!content.is_empty()).then(|| AnthropicMessage {
        role: "assistant".to_string(),
        content,
    })
}

/// Build one Anthropic user-side tool-result message from SA's canonical tool
/// result history item.
fn build_anthropic_tool_result_message(message: &ChatMessage) -> Option<AnthropicMessage> {
    let tool_use_id = sanitize_id(message.tool_call_id.as_deref())?;
    let content = message.text_content().unwrap_or_default();
    Some(AnthropicMessage {
        role: "user".to_string(),
        content: vec![AnthropicContentOut::ToolResult {
            tool_use_id,
            content,
            cache_control: None,
        }],
    })
}

/// Normalize an Anthropic `/messages` reply back into SA's canonical
/// chat-style response.
fn normalize_anthropic_messages_response(
    response: AnthropicMessagesResponse,
) -> ChatCompletionsResponse {
    let mut text_parts = Vec::<String>::new();
    let mut tool_calls = Vec::<ToolCall>::new();

    for block in response.content {
        match block.kind.as_str() {
            "text" => {
                if let Some(text) = non_empty_text(block.text.as_deref()) {
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                let Some(name) = non_empty_text(block.name.as_deref()) else {
                    continue;
                };
                let arguments = block
                    .input
                    .unwrap_or_else(|| Value::Object(serde_json::Map::new()))
                    .to_string();
                tool_calls.push(ToolCall {
                    id: sanitize_id(block.id.as_deref())
                        .unwrap_or_else(|| Uuid::new_v4().to_string()),
                    kind: "function".to_string(),
                    function: ToolFunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
            _ => {}
        }
    }

    let finish_reason = if tool_calls.is_empty() {
        normalize_anthropic_stop_reason(response.stop_reason.as_deref())
    } else {
        Some("tool_calls".to_string())
    };
    let content = (!text_parts.is_empty()).then(|| text_parts.join("\n"));

    ChatCompletionsResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".to_string(),
                content: content.map(MessageContent::Text),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
                request_usage: None,
                responses_input_items: None,
            },
            finish_reason,
        }],
        usage: response.usage,
    }
}

/// Convert Anthropic-native stop reasons into the canonical finish reasons SA
/// already uses elsewhere.
fn normalize_anthropic_stop_reason(stop_reason: Option<&str>) -> Option<String> {
    match stop_reason {
        Some("max_tokens") => Some("length".to_string()),
        Some("tool_use") => Some("tool_calls".to_string()),
        Some("end_turn") | Some("stop_sequence") => Some("stop".to_string()),
        Some(other) if !other.trim().is_empty() => Some(other.to_string()),
        _ => Some("stop".to_string()),
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
    ///
    /// `codex` always sends a concrete string here, even when it is empty. We
    /// follow the same shape so providers that validate strictly against the
    /// canonical Responses schema see the expected field type.
    instructions: String,
    /// Output cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    /// Optional reasoning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    /// Tool definitions.
    ///
    /// The official Responses request accepts arbitrary JSON tool
    /// specifications, not just function tools, so we serialize into raw JSON
    /// values here.
    tools: Vec<Value>,
    /// Tool choice policy.
    tool_choice: String,
    /// Whether the model may emit more than one tool call in one turn.
    parallel_tool_calls: bool,
    /// Whether the provider should persist the response server-side.
    ///
    /// SA does not currently rely on provider-side storage, so we keep this
    /// disabled by default while still emitting the canonical field.
    store: bool,
    /// Streaming toggle.
    stream: bool,
    /// Extra response fields requested from the provider.
    include: Vec<String>,
    /// Optional text controls.
    ///
    /// We keep the field available for official-shape compatibility even
    /// though SA does not yet expose verbosity/schema controls in its own
    /// configuration surface.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponsesTextControls>,
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
                    } else if let Some(text) = message.text_content() {
                        input.push(ResponsesInputItem::user_text(format!(
                            "[Tool result]\n{text}"
                        )));
                    }
                }
                "user" => {
                    if let Some(content) = message.content.as_ref() {
                        match content {
                            MessageContent::Text(text) => {
                                input.push(ResponsesInputItem::user_text(text.clone()));
                            }
                            MessageContent::Parts(parts) => {
                                let items: Vec<ResponsesContentItem> = parts
                                    .iter()
                                    .map(|part| match part {
                                        ContentPart::Text { text } => {
                                            ResponsesContentItem::InputText {
                                                text: text.clone(),
                                            }
                                        }
                                        ContentPart::Image { image_url } => {
                                            ResponsesContentItem::InputImage {
                                                image_url: image_url.url.clone(),
                                            }
                                        }
                                    })
                                    .collect();
                                input.push(ResponsesInputItem::Message {
                                    role: "user".to_string(),
                                    content: items,
                                });
                            }
                        }
                    }
                }
                _ => {
                    if let Some(text) = message.text_content() {
                        instructions.push(text);
                    }
                }
            }
        }

        let reasoning = req
            .reasoning_effort
            .as_deref()
            .and_then(|effort| non_empty_text(Some(effort)))
            .map(|effort| ResponsesReasoning {
                effort: effort.to_string(),
            });
        let include = if reasoning.is_some() {
            vec!["reasoning.encrypted_content".to_string()]
        } else {
            Vec::new()
        };
        let tools = req
            .tools
            .as_ref()
            .map(|definitions| {
                definitions
                    .iter()
                    .map(tool_definition_to_responses_value)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let has_tools = !tools.is_empty();

        Self {
            model: req.model.clone(),
            input,
            instructions: instructions.join("\n\n"),
            max_output_tokens: req.max_tokens,
            reasoning,
            tools,
            tool_choice: normalize_responses_tool_choice(req.tool_choice.as_ref(), has_tools),
            parallel_tool_calls: has_tools,
            store: false,
            stream: req.stream.unwrap_or(false),
            include,
            text: None,
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
        content: Vec<ResponsesContentItem>,
    },

    /// Assistant function-call item.
    FunctionCall {
        /// Optional provider-side item id.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool-call correlation id used by later `function_call_output`.
        call_id: String,
        /// Tool name.
        name: String,
        /// JSON-encoded arguments string.
        arguments: String,
    },

    /// Tool result item.
    FunctionCallOutput {
        /// The prior function-call correlation id.
        call_id: String,
        /// Tool output payload encoded exactly like the official Responses API:
        /// either a plain string or an array of structured content items.
        output: ResponsesFunctionCallOutputPayload,
    },

    /// Reasoning breadcrumb item preserved across turns.
    Reasoning {
        /// Optional provider-side item id.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Optional reasoning summary items.
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummaryItem>,
        /// Optional plaintext reasoning fragment.
        #[serde(default, skip_serializing_if = "responses_reasoning_content_is_empty")]
        content: Option<Vec<ResponsesReasoningContentItem>>,
        /// Optional encrypted reasoning blob.
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

impl ResponsesInputItem {
    /// Build one user/developer/system text message item.
    fn message_input_text(role: impl Into<String>, text: String) -> Self {
        Self::Message {
            role: role.into(),
            content: vec![ResponsesContentItem::InputText { text }],
        }
    }

    /// Build one assistant text message item.
    fn message_output_text(role: impl Into<String>, text: String) -> Self {
        Self::Message {
            role: role.into(),
            content: vec![ResponsesContentItem::OutputText { text }],
        }
    }

    /// Build one user text item.
    fn user_text(text: String) -> Self {
        Self::message_input_text("user", text)
    }

    /// Return the first textual payload carried by this item, if any.
    fn text(&self) -> Option<String> {
        match self {
            ResponsesInputItem::Message { content, .. } => {
                let parts = content
                    .iter()
                    .filter_map(ResponsesContentItem::text)
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                (!parts.is_empty()).then(|| parts.join("\n"))
            }
            ResponsesInputItem::FunctionCall { .. }
            | ResponsesInputItem::FunctionCallOutput { .. }
            | ResponsesInputItem::Reasoning { .. } => None,
        }
    }
}

/// Content part inside one Responses message item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentItem {
    /// User/developer/system text.
    InputText {
        /// Plain text payload.
        text: String,
    },
    /// Assistant text.
    OutputText {
        /// Plain text payload.
        text: String,
    },
    /// User image reference.
    InputImage {
        /// Image URL or data URL.
        image_url: String,
    },
}

impl ResponsesContentItem {
    /// Return the visible text carried by this content item, if any.
    fn text(&self) -> Option<&str> {
        match self {
            ResponsesContentItem::InputText { text }
            | ResponsesContentItem::OutputText { text } => non_empty_text(Some(text)),
            ResponsesContentItem::InputImage { .. } => None,
        }
    }
}

/// One reasoning-summary part preserved across Responses turns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningSummaryItem {
    /// Human-readable summary text.
    SummaryText {
        /// Summary payload.
        text: String,
    },
}

/// One reasoning-content part preserved across Responses turns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningContentItem {
    /// Explicit reasoning-text payload.
    ReasoningText {
        /// Reasoning payload.
        text: String,
    },
    /// Plain text payload seen on some providers.
    Text {
        /// Text payload.
        text: String,
    },
}

/// Tool-call output content items compatible with the official Responses API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesFunctionCallOutputContentItem {
    /// Text output returned by the tool.
    InputText {
        /// Text payload.
        text: String,
    },
    /// Image output returned by the tool.
    InputImage {
        /// Image URL or data URL.
        image_url: String,
    },
}

/// Wire body for `function_call_output.output`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ResponsesFunctionCallOutputBody {
    /// Plain-text tool output.
    Text(String),
    /// Structured multimodal tool output.
    ContentItems(Vec<ResponsesFunctionCallOutputContentItem>),
}

/// Wrapper that keeps SA's internal type explicit while serializing exactly as
/// the official Responses API expects for `function_call_output.output`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponsesFunctionCallOutputPayload {
    /// Actual wire body.
    body: ResponsesFunctionCallOutputBody,
}

impl ResponsesFunctionCallOutputPayload {
    /// Build a plain-text tool output payload.
    fn from_text(text: String) -> Self {
        Self {
            body: ResponsesFunctionCallOutputBody::Text(text),
        }
    }
}

impl Serialize for ResponsesFunctionCallOutputPayload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.body.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ResponsesFunctionCallOutputPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self {
            body: ResponsesFunctionCallOutputBody::deserialize(deserializer)?,
        })
    }
}

/// Minimal reasoning options for the Responses request.
#[derive(Debug, Clone, Serialize)]
struct ResponsesReasoning {
    /// Requested reasoning effort.
    effort: String,
}

/// Minimal text controls for the Responses request.
#[derive(Debug, Clone, Serialize)]
struct ResponsesTextControls {
    /// Optional verbosity control.
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<ResponsesVerbosity>,
}

/// Verbosity variants accepted by the Responses API text controls.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
#[allow(dead_code)]
enum ResponsesVerbosity {
    /// Minimal output verbosity.
    Low,
    /// Balanced output verbosity.
    Medium,
    /// High output verbosity.
    High,
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

/// Mutable accumulator for one streamed `/responses` SSE session.
#[derive(Debug, Clone, Default)]
struct ResponsesStreamAccumulator {
    /// Incremental text deltas, if the provider emits them.
    output_text: String,
    /// Whether at least one delta chunk has been observed.
    saw_output_text_delta: bool,
    /// Completed output items, when emitted individually.
    output_items: Vec<ResponsesOutputItem>,
    /// Incremental reasoning content deltas keyed by `content_index`.
    reasoning_content_deltas: BTreeMap<i64, String>,
    /// Incremental reasoning-summary deltas keyed by `summary_index`.
    reasoning_summary_deltas: BTreeMap<i64, String>,
    /// Terminal full response object, if the provider emitted one.
    completed_response: Option<ResponsesResponse>,
}

impl ResponsesStreamAccumulator {
    /// Apply one streamed event. Returning `Some` means the response is
    /// complete and can be converted immediately.
    fn apply_event(&mut self, event: &Value) -> Result<Option<()>, ChatCompletionsError> {
        let event_type = event.get("type").and_then(Value::as_str);

        if event_type == Some("error") {
            let message = event
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| {
                    event
                        .get("error")
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                })
                .or_else(|| event.get("code").and_then(Value::as_str))
                .unwrap_or("responses stream returned an error event");
            return Err(invalid_response(message.to_string()));
        }

        if event_type == Some("response.failed") {
            let message = event
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("responses stream reported failure");
            return Err(invalid_response(message.to_string()));
        }

        match event_type {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.saw_output_text_delta = true;
                    self.output_text.push_str(delta);
                }
                Ok(None)
            }
            Some("response.output_text.done") if !self.saw_output_text_delta => {
                if let Some(text) = event.get("text").and_then(Value::as_str) {
                    self.output_text = text.to_string();
                }
                Ok(None)
            }
            Some("response.output_item.added") | Some("response.output_item.done") => {
                if let Some(item) = event.get("item").cloned() {
                    if let Ok(parsed) = serde_json::from_value::<ResponsesOutputItem>(item) {
                        upsert_responses_output_item(&mut self.output_items, parsed);
                    }
                }
                Ok(None)
            }
            Some("response.reasoning_text.delta") => {
                let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                    return Ok(None);
                };
                let index = event
                    .get("content_index")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                self.reasoning_content_deltas
                    .entry(index)
                    .or_default()
                    .push_str(delta);
                Ok(None)
            }
            Some("response.reasoning_summary_part.added") => {
                let index = event
                    .get("summary_index")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                self.reasoning_summary_deltas.entry(index).or_default();
                Ok(None)
            }
            Some("response.reasoning_summary_text.delta") => {
                let Some(delta) = event.get("delta").and_then(Value::as_str) else {
                    return Ok(None);
                };
                let index = event
                    .get("summary_index")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                self.reasoning_summary_deltas
                    .entry(index)
                    .or_default()
                    .push_str(delta);
                Ok(None)
            }
            Some("response.completed") | Some("response.done") => {
                let Some(response) = event.get("response").cloned() else {
                    self.completed_response = Some(self.synthetic_response());
                    return Ok(Some(()));
                };

                let mut parsed =
                    serde_json::from_value::<ResponsesResponse>(response).map_err(|err| {
                        invalid_response(format!(
                            "responses completion event carried an invalid response object: {err}"
                        ))
                    })?;

                if parsed.output_text.is_none() && !self.output_text.is_empty() {
                    parsed.output_text = Some(self.output_text.clone());
                }
                merge_responses_output_items(&mut parsed.output, &self.synthetic_output_items());

                self.completed_response = Some(parsed);
                Ok(Some(()))
            }
            _ => Ok(None),
        }
    }

    /// Build a best-effort synthetic response when the provider never sent an
    /// explicit `response.completed` object.
    fn synthetic_response(&self) -> ResponsesResponse {
        ResponsesResponse {
            output: self.synthetic_output_items(),
            output_text: (!self.output_text.is_empty()).then(|| self.output_text.clone()),
            usage: None,
        }
    }

    /// Build the best-effort output item list we can reconstruct from stream
    /// fragments alone.
    fn synthetic_output_items(&self) -> Vec<ResponsesOutputItem> {
        let mut items = self.output_items.clone();
        if !items.iter().any(ResponsesOutputItem::is_reasoning_item) {
            let synthesized_reasoning = self.synthesized_reasoning_item();
            if let Some(item) = synthesized_reasoning {
                items.push(item);
            }
        }
        items
    }

    /// Reconstruct one reasoning output item from the accumulated delta-only
    /// stream events.
    fn synthesized_reasoning_item(&self) -> Option<ResponsesOutputItem> {
        let content_items = self
            .reasoning_content_deltas
            .values()
            .filter_map(|text| {
                non_empty_text(Some(text)).map(|trimmed| {
                    serde_json::json!({
                        "type": "reasoning_text",
                        "text": trimmed,
                    })
                })
            })
            .collect::<Vec<_>>();
        let summary_items = self
            .reasoning_summary_deltas
            .values()
            .filter_map(|text| {
                non_empty_text(Some(text)).map(|trimmed| {
                    serde_json::json!({
                        "type": "summary_text",
                        "text": trimmed,
                    })
                })
            })
            .collect::<Vec<_>>();

        if content_items.is_empty() && summary_items.is_empty() {
            return None;
        }

        Some(ResponsesOutputItem {
            kind: Some("reasoning".to_string()),
            id: None,
            call_id: None,
            status: None,
            name: None,
            arguments: None,
            role: None,
            content: (!content_items.is_empty()).then(|| Value::Array(content_items)),
            text: None,
            encrypted_content: None,
            summary: (!summary_items.is_empty()).then(|| Value::Array(summary_items)),
        })
    }
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
    /// Optional item status.
    #[serde(default)]
    #[allow(dead_code)]
    status: Option<String>,
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
    summary: Option<Value>,
}

impl ResponsesOutputItem {
    /// Return `true` if this output item is a reasoning item.
    fn is_reasoning_item(&self) -> bool {
        self.kind.as_deref() == Some("reasoning")
    }

    /// Determine whether two output items refer to the same logical provider
    /// item.
    ///
    /// We prefer stable provider IDs. If they do not exist, `call_id` still
    /// lets us coalesce `function_call` items emitted via both
    /// `response.output_item.added` and `response.output_item.done`.
    fn matches_identity(&self, other: &Self) -> bool {
        if let (Some(left), Some(right)) = (self.id.as_deref(), other.id.as_deref()) {
            return left == right;
        }

        if let (Some(left), Some(right)) = (self.call_id.as_deref(), other.call_id.as_deref()) {
            return left == right;
        }

        false
    }
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
        content: content.map(MessageContent::Text),
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
                    call_id: normalized_call_id,
                    name: name.to_string(),
                    arguments,
                });
            }
            Some("reasoning") => {
                let content =
                    normalize_responses_reasoning_content(item.content.as_ref()).or_else(|| {
                        non_empty_text(item.text.as_deref()).map(|text| {
                            vec![ResponsesReasoningContentItem::ReasoningText {
                                text: text.to_string(),
                            }]
                        })
                    });
                let encrypted_content =
                    non_empty_text(item.encrypted_content.as_deref()).map(str::to_string);
                let summary = normalize_responses_reasoning_summary(item.summary.as_ref());

                if content.is_none() && encrypted_content.is_none() && summary.is_empty() {
                    continue;
                }

                history_items.push(ResponsesInputItem::Reasoning {
                    id: sanitize_id(item.id.as_deref()),
                    summary,
                    content,
                    encrypted_content,
                });
            }
            Some("output_text") => {
                let Some(text) = non_empty_text(item.text.as_deref()).map(str::to_string) else {
                    continue;
                };

                if assistant_text.is_none() {
                    assistant_text = Some(text.clone());
                }

                history_items.push(ResponsesInputItem::message_output_text("assistant", text));
            }
            _ => {}
        }
    }

    if assistant_text.is_none() {
        if let Some(text) = non_empty_text(top_level_output_text) {
            let text = text.to_string();
            assistant_text = Some(text.clone());
            history_items.push(ResponsesInputItem::message_output_text("assistant", text));
        }
    }

    (assistant_text, tool_calls, history_items)
}

/// Normalize raw message content parts so they can be replayed back into
/// `input[]` on a later Responses turn.
fn normalize_responses_message_parts(
    role: &str,
    raw_content: Option<&Value>,
) -> Vec<ResponsesContentItem> {
    let default_is_output = matches!(role, "assistant");

    if let Some(Value::String(text)) = raw_content
        && let Some(text) = non_empty_text(Some(text))
    {
        return vec![if default_is_output {
            ResponsesContentItem::OutputText {
                text: text.to_string(),
            }
        } else {
            ResponsesContentItem::InputText {
                text: text.to_string(),
            }
        }];
    }

    let Some(parts) = raw_content.and_then(Value::as_array) else {
        return Vec::new();
    };

    parts
        .iter()
        .filter_map(|part| match part.get("type").and_then(Value::as_str) {
            Some("input_image") => {
                let image_url = non_empty_text(part.get("image_url").and_then(Value::as_str))?;
                Some(ResponsesContentItem::InputImage {
                    image_url: image_url.to_string(),
                })
            }
            Some("output_text") => {
                let text = non_empty_text(part.get("text").and_then(Value::as_str))?;
                Some(ResponsesContentItem::OutputText {
                    text: text.to_string(),
                })
            }
            Some("input_text") => {
                let text = non_empty_text(part.get("text").and_then(Value::as_str))?;
                Some(ResponsesContentItem::InputText {
                    text: text.to_string(),
                })
            }
            _ => {
                let text = non_empty_text(part.get("text").and_then(Value::as_str))?;
                Some(if default_is_output {
                    ResponsesContentItem::OutputText {
                        text: text.to_string(),
                    }
                } else {
                    ResponsesContentItem::InputText {
                        text: text.to_string(),
                    }
                })
            }
        })
        .collect()
}

/// Normalize a raw reasoning-summary payload into official Responses summary
/// items.
fn normalize_responses_reasoning_summary(
    raw_summary: Option<&Value>,
) -> Vec<ResponsesReasoningSummaryItem> {
    match raw_summary {
        Some(Value::String(text)) => non_empty_text(Some(text))
            .map(|text| {
                vec![ResponsesReasoningSummaryItem::SummaryText {
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some(Value::Object(_)) => normalize_responses_reasoning_summary(Some(&Value::Array(vec![
            raw_summary.cloned().unwrap_or(Value::Null),
        ]))),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| {
                let text = non_empty_text(item.get("text").and_then(Value::as_str))?;
                Some(ResponsesReasoningSummaryItem::SummaryText {
                    text: text.to_string(),
                })
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Normalize a raw reasoning-content payload into official Responses content
/// items.
fn normalize_responses_reasoning_content(
    raw_content: Option<&Value>,
) -> Option<Vec<ResponsesReasoningContentItem>> {
    let normalized = match raw_content {
        Some(Value::String(text)) => non_empty_text(Some(text))
            .map(|text| {
                vec![ResponsesReasoningContentItem::ReasoningText {
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some(Value::Object(_)) => {
            let raw_item = raw_content.cloned().unwrap_or(Value::Null);
            normalize_responses_reasoning_content(Some(&Value::Array(vec![raw_item])))
                .unwrap_or_default()
        }
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| {
                let text = non_empty_text(item.get("text").and_then(Value::as_str))?;
                let kind = item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("reasoning_text");
                Some(match kind {
                    "text" => ResponsesReasoningContentItem::Text {
                        text: text.to_string(),
                    },
                    _ => ResponsesReasoningContentItem::ReasoningText {
                        text: text.to_string(),
                    },
                })
            })
            .collect(),
        _ => Vec::new(),
    };

    (!normalized.is_empty()).then_some(normalized)
}

/// Build assistant-side Responses items from SA's canonical assistant message.
fn build_assistant_responses_items(message: &ChatMessage) -> Vec<ResponsesInputItem> {
    let mut items = Vec::<ResponsesInputItem>::new();

    if let Some(text) = message.text_content() {
        items.push(ResponsesInputItem::message_output_text(
            "assistant",
            text,
        ));
    }

    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for tool_call in tool_calls {
            let call_id =
                sanitize_id(Some(&tool_call.id)).unwrap_or_else(|| Uuid::new_v4().to_string());
            items.push(ResponsesInputItem::FunctionCall {
                id: None,
                call_id,
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
    let output = message.text_content().unwrap_or_default();
    Some(ResponsesInputItem::FunctionCallOutput {
        call_id,
        output: ResponsesFunctionCallOutputPayload::from_text(output),
    })
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

/// `Reasoning.content` is optional; when present but empty it should still be
/// omitted so the replayed item stays compact.
fn responses_reasoning_content_is_empty(
    content: &Option<Vec<ResponsesReasoningContentItem>>,
) -> bool {
    match content {
        Some(items) => items.is_empty(),
        None => true,
    }
}

/// Convert one tool definition into the raw JSON shape used by the official
/// Responses request.
fn tool_definition_to_responses_value(definition: &ToolDefinition) -> Value {
    serde_json::to_value(definition)
        .expect("ToolDefinition contains only serializable JSON-compatible fields")
}

/// Normalize SA's internal `tool_choice` into the canonical Responses string
/// form used by `codex`.
fn normalize_responses_tool_choice(tool_choice: Option<&Value>, has_tools: bool) -> String {
    if let Some(Value::String(choice)) = tool_choice
        && let Some(choice) = non_empty_text(Some(choice))
    {
        return choice.to_string();
    }

    if has_tools {
        "auto".to_string()
    } else {
        "none".to_string()
    }
}

/// Insert or replace one Responses output item in-place.
fn upsert_responses_output_item(items: &mut Vec<ResponsesOutputItem>, item: ResponsesOutputItem) {
    if let Some(index) = items
        .iter()
        .position(|existing| existing.matches_identity(&item))
    {
        items[index] = item;
    } else {
        items.push(item);
    }
}

/// Merge one list of output items into another while coalescing identical
/// provider items.
fn merge_responses_output_items(
    target: &mut Vec<ResponsesOutputItem>,
    incoming: &[ResponsesOutputItem],
) {
    for item in incoming {
        upsert_responses_output_item(target, item.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AnthropicMessagesRequest, AuthStyle, ChatCompletionsRequest, ChatCompletionsResponse,
        ChatCompletionsStreamAccumulator, ChatCompletionsStreamChunk, ChatMessage, ChatUsage,
        InputTokenDetails, OutputTokenDetails, ResponsesFunctionCallOutputContentItem,
        ResponsesInputItem, ResponsesReasoningContentItem, ResponsesReasoningSummaryItem,
        ResponsesRequest, ResponsesStreamAccumulator, ToolCall, ToolDefinition, ToolFunctionCall,
        ToolFunctionDefinition, WireApi, build_auth_headers, normalize_anthropic_messages_response,
        normalize_responses_response, process_sse_line,
    };

    #[test]
    fn wire_api_aliases_deserialize() {
        let parsed = serde_json::from_str::<WireApi>(r#""chat""#).expect("chat alias should parse");
        assert_eq!(parsed, WireApi::ChatCompletions);

        let parsed =
            serde_json::from_str::<WireApi>(r#""responses""#).expect("responses should parse");
        assert_eq!(parsed, WireApi::Responses);

        let parsed =
            serde_json::from_str::<WireApi>(r#""claude""#).expect("claude alias should parse");
        assert_eq!(parsed, WireApi::AnthropicMessages);
    }

    #[test]
    fn anthropic_auto_auth_uses_x_api_key_for_regular_keys() {
        let headers = build_auth_headers("sk-ant-api03-demo", AuthStyle::AnthropicAuto)
            .expect("headers should build");
        assert_eq!(
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok()),
            Some("sk-ant-api03-demo")
        );
        assert!(headers.get("authorization").is_none());
    }

    #[test]
    fn anthropic_auto_auth_uses_bearer_for_setup_tokens() {
        let headers = build_auth_headers("sk-ant-oat01-demo", AuthStyle::AnthropicAuto)
            .expect("headers should build");
        assert_eq!(
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer sk-ant-oat01-demo")
        );
        assert_eq!(
            headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok()),
            Some("oauth-2025-04-20")
        );
    }

    #[test]
    fn process_sse_line_reassembles_multiline_event_data() {
        let mut current_event = None;
        let mut current_data = Vec::new();
        let mut events = Vec::<(Option<String>, String)>::new();

        for line in [
            "event: message",
            "data: {\"hello\":",
            "data: \"world\"}",
            "",
        ] {
            let maybe = process_sse_line(
                line,
                &mut current_event,
                &mut current_data,
                &mut |event, data| {
                    events.push((event.map(str::to_string), data.to_string()));
                    Ok(None::<()>)
                },
            )
            .expect("SSE line should parse");
            assert!(maybe.is_none());
        }

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0.as_deref(), Some("message"));
        assert_eq!(events[0].1, "{\"hello\":\n\"world\"}");
    }

    #[test]
    fn chat_completions_stream_accumulator_reconstructs_tool_call_arguments() {
        let mut acc = ChatCompletionsStreamAccumulator::default();

        let first = serde_json::json!({
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": "{\"path\":"
                        }
                    }]
                }
            }]
        });
        let second = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "\"a.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        acc.apply_chunk(
            serde_json::from_value::<ChatCompletionsStreamChunk>(first)
                .expect("first stream chunk should deserialize"),
        );
        acc.apply_chunk(
            serde_json::from_value::<ChatCompletionsStreamChunk>(second)
                .expect("second stream chunk should deserialize"),
        );

        let response = acc.into_response().expect("stream should normalize");
        let choice = response.first_choice().expect("choice should exist");
        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls should exist");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "Read");
        assert_eq!(tool_calls[0].function.arguments, "{\"path\":\"a.txt\"}");
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn responses_stream_accumulator_prefers_completed_response_object() {
        let mut acc = ResponsesStreamAccumulator::default();
        let event = serde_json::json!({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "Read",
                    "arguments": "{\"path\":\"a.txt\"}"
                }],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3
                }
            }
        });

        acc.apply_event(&event)
            .expect("stream event should parse")
            .expect("completed response should terminate");
        let normalized = normalize_responses_response(
            acc.completed_response
                .clone()
                .expect("completed response should be stored"),
        );
        let choice = normalized.first_choice().expect("choice should exist");
        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls should exist");
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(
            normalized.usage.expect("usage should exist").input_tokens,
            Some(12)
        );
    }

    #[test]
    fn responses_stream_accumulator_builds_synthetic_response_from_deltas() {
        let mut acc = ResponsesStreamAccumulator::default();

        acc.apply_event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "hello "
        }))
        .expect("delta should parse");
        acc.apply_event(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "world"
        }))
        .expect("delta should parse");
        acc.apply_event(&serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [
                    { "type": "output_text", "text": "hello world" }
                ]
            }
        }))
        .expect("output item should parse");

        let response = acc.synthetic_response();
        let normalized = normalize_responses_response(response);
        let choice = normalized.first_choice().expect("choice should exist");
        assert_eq!(choice.message.text_content().as_deref(), Some("hello world"));
    }

    #[test]
    fn anthropic_request_extracts_system_tools_and_tool_results() {
        let req = ChatCompletionsRequest {
            model: "claude-sonnet-4-5".to_string(),
            messages: vec![
                ChatMessage::text("system", "系统规则"),
                ChatMessage::text("developer", "开发规则"),
                ChatMessage::text("user", "看一下 a.txt"),
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(MessageContent::Text("先读文件".to_string())),
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
            stream: Some(true),
        };

        let wire = AnthropicMessagesRequest::from_chat_request(&req);
        let value = serde_json::to_value(wire).expect("serialize anthropic request");

        assert_eq!(value["system"], "系统规则\n\n开发规则");
        assert_eq!(value["max_tokens"], 512);
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"][0]["type"], "text");
        assert_eq!(value["messages"][1]["role"], "assistant");
        assert_eq!(value["messages"][1]["content"][0]["type"], "text");
        assert_eq!(value["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(value["messages"][1]["content"][1]["id"], "call_123");
        assert_eq!(value["messages"][2]["role"], "user");
        assert_eq!(value["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(
            value["messages"][2]["content"][0]["tool_use_id"],
            "call_123"
        );
        assert_eq!(value["tools"][0]["name"], "Read");
    }

    #[test]
    fn normalize_anthropic_response_extracts_text_and_tool_calls() {
        let raw = serde_json::json!({
            "content": [
                {
                    "type": "text",
                    "text": "先读取配置"
                },
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "Read",
                    "input": {
                        "path": "sa.toml"
                    }
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_creation_input_tokens": 10
            }
        });

        let response = serde_json::from_value(raw).expect("anthropic response should deserialize");
        let normalized = normalize_anthropic_messages_response(response);
        let usage = normalized.usage.clone().expect("usage should exist");
        let choice = normalized.first_choice().expect("choice should exist");
        assert_eq!(choice.message.text_content().as_deref(), Some("先读取配置"));
        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls should exist");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tool_1");
        assert_eq!(tool_calls[0].function.name, "Read");
        assert_eq!(usage.cache_write_tokens(), 10);
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn responses_stream_accumulator_reconstructs_reasoning_from_deltas() {
        let mut acc = ResponsesStreamAccumulator::default();

        acc.apply_event(&serde_json::json!({
            "type": "response.reasoning_summary_part.added",
            "summary_index": 0
        }))
        .expect("summary part should parse");
        acc.apply_event(&serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "summary_index": 0,
            "delta": "先读配置"
        }))
        .expect("summary delta should parse");
        acc.apply_event(&serde_json::json!({
            "type": "response.reasoning_text.delta",
            "content_index": 0,
            "delta": "我需要先确认字段。"
        }))
        .expect("reasoning delta should parse");

        let response = acc.synthetic_response();
        let normalized = normalize_responses_response(response);
        let choice = normalized.first_choice().expect("choice should exist");
        let history = choice
            .message
            .responses_input_items
            .as_ref()
            .expect("responses history should exist");

        assert!(history.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Reasoning {
                summary,
                content: Some(content),
                ..
            } if summary == &vec![ResponsesReasoningSummaryItem::SummaryText {
                text: "先读配置".to_string(),
            }] && content == &vec![ResponsesReasoningContentItem::ReasoningText {
                text: "我需要先确认字段。".to_string(),
            }]
        )));
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
        assert_eq!(value["tool_choice"], "auto");
        assert_eq!(value["parallel_tool_calls"], true);
        assert_eq!(value["store"], false);
        assert_eq!(value["stream"], false);
        assert_eq!(
            value["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
        assert_eq!(value["input"][0]["type"], "message");
        assert_eq!(value["input"][0]["role"], "user");
        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][1]["type"], "function_call");
        assert_eq!(value["input"][1]["call_id"], "call_123");
        assert_eq!(value["input"][2]["type"], "function_call_output");
        assert_eq!(value["input"][2]["call_id"], "call_123");
        assert_eq!(value["input"][2]["output"], "文件内容");
        assert_eq!(value["tools"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn normalize_responses_response_preserves_reasoning_and_function_call_history() {
        let raw = serde_json::json!({
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "先看文件"
                        }
                    ],
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "先确认工具输出。"
                        }
                    ],
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
        assert!(matches!(
            &history[0],
            ResponsesInputItem::Reasoning {
                summary,
                content: Some(content),
                encrypted_content: Some(encrypted_content),
                ..
            } if summary == &vec![ResponsesReasoningSummaryItem::SummaryText {
                text: "先看文件".to_string(),
            }] && content == &vec![ResponsesReasoningContentItem::ReasoningText {
                text: "先确认工具输出。".to_string(),
            }] && encrypted_content == "ciphertext"
        ));
        assert_eq!(
            normalized.usage.expect("usage should exist").input_tokens,
            Some(800)
        );
    }

    #[test]
    fn responses_function_call_output_payload_serializes_like_codex_wire_format() {
        let item = ResponsesInputItem::FunctionCallOutput {
            call_id: "call_1".to_string(),
            output: super::ResponsesFunctionCallOutputPayload {
                body: super::ResponsesFunctionCallOutputBody::ContentItems(vec![
                    ResponsesFunctionCallOutputContentItem::InputText {
                        text: "line 1".to_string(),
                    },
                    ResponsesFunctionCallOutputContentItem::InputImage {
                        image_url: "data:image/png;base64,AAA".to_string(),
                    },
                ]),
            },
        };

        let value = serde_json::to_value(&item).expect("serialize tool output");
        assert_eq!(value["type"], "function_call_output");
        assert_eq!(value["call_id"], "call_1");
        assert_eq!(value["output"][0]["type"], "input_text");
        assert_eq!(value["output"][1]["type"], "input_image");

        let parsed =
            serde_json::from_value::<ResponsesInputItem>(value).expect("deserialize tool output");
        assert_eq!(parsed, item);
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
        assert_eq!(choice.message.text_content().as_deref(), Some("第一行\n第二行"));
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
            choice.message.text_content().as_deref(),
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
