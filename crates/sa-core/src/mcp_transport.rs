//! MCP transport abstraction — supports stdio, HTTP, and SSE transports.
//!
//! This module is adapted from `zeroclaw`'s mature MCP transport layer.
//! Keeping the transport code close to that implementation reduces protocol
//! edge cases, especially for SSE framing and mixed HTTP/SSE servers.

use std::borrow::Cow;
use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Notify, oneshot};
use tokio::time::{Duration, timeout};
use tokio_stream::StreamExt;

use crate::config::{McpServerConfig, McpTransport};
use crate::mcp_protocol::{INTERNAL_ERROR, JsonRpcError, JsonRpcRequest, JsonRpcResponse};

/// Maximum bytes accepted for a single JSON-RPC line over stdio.
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

/// Timeout used while waiting for initialization/list responses.
const RECV_TIMEOUT_SECS: u64 = 30;

/// Accept header used by streamable MCP HTTP servers.
const MCP_STREAMABLE_ACCEPT: &str = "application/json, text/event-stream";

/// Default content type for JSON-RPC request bodies.
const MCP_JSON_CONTENT_TYPE: &str = "application/json";

/// Abstract MCP transport connection.
#[async_trait]
pub trait McpTransportConn: Send + Sync {
    /// Send one JSON-RPC request and receive the matching response.
    async fn send_and_recv(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse>;

    /// Best-effort close hook.
    async fn close(&mut self) -> Result<()>;
}

/// Stdio transport used for local MCP servers.
pub struct StdioTransport {
    /// Child process kept alive for the lifetime of the connection.
    _child: Child,
    /// Child stdin.
    stdin: tokio::process::ChildStdin,
    /// Line-based stdout reader.
    stdout_lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
}

impl StdioTransport {
    /// Spawn the configured stdio MCP server.
    pub fn new(config: &McpServerConfig) -> Result<Self> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .envs(&config.env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to spawn MCP server `{}`", config.name))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("no stdin on MCP server `{}`", config.name))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no stdout on MCP server `{}`", config.name))?;

        Ok(Self {
            _child: child,
            stdin,
            stdout_lines: BufReader::new(stdout).lines(),
        })
    }

    /// Send one raw JSON line.
    async fn send_raw(&mut self, line: &str) -> Result<()> {
        self.stdin
            .write_all(line.as_bytes())
            .await
            .context("failed to write to MCP server stdin")?;
        self.stdin
            .write_all(b"\n")
            .await
            .context("failed to write newline to MCP server stdin")?;
        self.stdin.flush().await.context("failed to flush stdin")?;
        Ok(())
    }

    /// Read one raw JSON line.
    async fn recv_raw(&mut self) -> Result<String> {
        let line = self
            .stdout_lines
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("MCP server closed stdout"))?;
        if line.len() > MAX_LINE_BYTES {
            bail!("MCP response too large: {} bytes", line.len());
        }
        Ok(line)
    }
}

#[async_trait]
impl McpTransportConn for StdioTransport {
    async fn send_and_recv(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse> {
        let line = serde_json::to_string(request)?;
        self.send_raw(&line).await?;

        if request.id.is_none() {
            return Ok(JsonRpcResponse {
                jsonrpc: crate::mcp_protocol::JSONRPC_VERSION.to_string(),
                id: None,
                result: None,
                error: None,
            });
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(RECV_TIMEOUT_SECS);
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                bail!("timeout waiting for MCP response");
            }

            let raw = timeout(remaining, self.recv_raw())
                .await
                .context("timeout waiting for MCP response")??;
            let response: JsonRpcResponse = serde_json::from_str(&raw)
                .with_context(|| format!("invalid JSON-RPC response: {raw}"))?;

            // Notifications can legally arrive while we wait for our actual response.
            if response.id.is_none() {
                tracing::debug!("MCP stdio: skipping notification while waiting for response");
                continue;
            }

            return Ok(response);
        }
    }

    async fn close(&mut self) -> Result<()> {
        let _ = self.stdin.shutdown().await;
        Ok(())
    }
}

/// HTTP transport used by MCP servers that accept POST requests directly.
pub struct HttpTransport {
    /// Base URL.
    url: String,
    /// Shared HTTP client.
    client: reqwest::Client,
    /// Extra configured headers.
    headers: HashMap<String, String>,
}

impl HttpTransport {
    /// Construct one HTTP transport.
    pub fn new(config: &McpServerConfig) -> Result<Self> {
        let url = config
            .url
            .as_ref()
            .ok_or_else(|| anyhow!("URL required for HTTP transport"))?
            .clone();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            url,
            client,
            headers: config.headers.clone(),
        })
    }
}

#[async_trait]
impl McpTransportConn for HttpTransport {
    async fn send_and_recv(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse> {
        let body = serde_json::to_string(request)?;
        let has_accept = self
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("Accept"));
        let has_content_type = self
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("Content-Type"));

        let mut req = self.client.post(&self.url).body(body);
        if !has_content_type {
            req = req.header("Content-Type", MCP_JSON_CONTENT_TYPE);
        }
        for (key, value) in &self.headers {
            req = req.header(key, value);
        }
        if !has_accept {
            req = req.header("Accept", MCP_STREAMABLE_ACCEPT);
        }

        let response = req
            .send()
            .await
            .context("HTTP request to MCP server failed")?;
        if !response.status().is_success() {
            bail!("MCP server returned HTTP {}", response.status());
        }

        if request.id.is_none() {
            return Ok(JsonRpcResponse {
                jsonrpc: crate::mcp_protocol::JSONRPC_VERSION.to_string(),
                id: None,
                result: None,
                error: None,
            });
        }

        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"));
        if is_sse {
            let maybe_response = timeout(
                Duration::from_secs(RECV_TIMEOUT_SECS),
                read_first_jsonrpc_from_sse_response(response),
            )
            .await
            .context("timeout waiting for MCP response from streamable HTTP SSE stream")??;
            return maybe_response
                .ok_or_else(|| anyhow!("MCP server returned no response in SSE stream"));
        }

        let text = response
            .text()
            .await
            .context("failed to read HTTP response")?;
        parse_jsonrpc_response_text(&text)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Connection state for the SSE reader.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SseStreamState {
    /// We have not tried connecting yet.
    Unknown,
    /// The SSE reader task is alive.
    Connected,
    /// The server did not expose SSE semantics; fall back to direct POST reads.
    Unsupported,
}

/// SSE transport used by MCP servers that split POST requests and event streams.
pub struct SseTransport {
    /// Base SSE URL.
    sse_url: String,
    /// Server display name for diagnostics.
    server_name: String,
    /// Shared HTTP client.
    client: reqwest::Client,
    /// Extra configured headers.
    headers: HashMap<String, String>,
    /// Reader state.
    stream_state: SseStreamState,
    /// Shared pending-response map.
    shared: std::sync::Arc<Mutex<SseSharedState>>,
    /// Notifies when the server advertises a different POST endpoint.
    notify: std::sync::Arc<Notify>,
    /// Shutdown signal for the reader task.
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Background reader task.
    reader_task: Option<tokio::task::JoinHandle<()>>,
}

impl SseTransport {
    /// Construct one SSE transport.
    pub fn new(config: &McpServerConfig) -> Result<Self> {
        let sse_url = config
            .url
            .as_ref()
            .ok_or_else(|| anyhow!("URL required for SSE transport"))?
            .clone();
        let client = reqwest::Client::builder()
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self {
            sse_url,
            server_name: config.name.clone(),
            client,
            headers: config.headers.clone(),
            stream_state: SseStreamState::Unknown,
            shared: std::sync::Arc::new(Mutex::new(SseSharedState::default())),
            notify: std::sync::Arc::new(Notify::new()),
            shutdown_tx: None,
            reader_task: None,
        })
    }

    /// Ensure the background SSE reader is running when the server supports it.
    async fn ensure_connected(&mut self) -> Result<()> {
        if self.stream_state == SseStreamState::Unsupported {
            return Ok(());
        }
        if let Some(task) = &self.reader_task {
            if !task.is_finished() {
                self.stream_state = SseStreamState::Connected;
                return Ok(());
            }
        }

        let has_accept = self
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("Accept"));

        let mut req = self
            .client
            .get(&self.sse_url)
            .header("Cache-Control", "no-cache");
        for (key, value) in &self.headers {
            req = req.header(key, value);
        }
        if !has_accept {
            req = req.header("Accept", MCP_STREAMABLE_ACCEPT);
        }

        let response = req.send().await.context("SSE GET to MCP server failed")?;
        if response.status() == reqwest::StatusCode::NOT_FOUND
            || response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED
        {
            self.stream_state = SseStreamState::Unsupported;
            return Ok(());
        }
        if !response.status().is_success() {
            return Err(anyhow!("MCP server returned HTTP {}", response.status()));
        }

        let is_event_stream = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"));
        if !is_event_stream {
            self.stream_state = SseStreamState::Unsupported;
            return Ok(());
        }

        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
        self.shutdown_tx = Some(shutdown_tx);

        let shared = self.shared.clone();
        let notify = self.notify.clone();
        let sse_url = self.sse_url.clone();
        let server_name = self.server_name.clone();

        self.reader_task = Some(tokio::spawn(async move {
            let stream = response
                .bytes_stream()
                .map(|item| item.map_err(std::io::Error::other));
            let reader = tokio_util::io::StreamReader::new(stream);
            let mut lines = BufReader::new(reader).lines();

            let mut current_event: Option<String> = None;
            let mut current_id: Option<String> = None;
            let mut current_data: Vec<String> = Vec::new();

            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    line = lines.next_line() => {
                        let Ok(line_opt) = line else { break; };
                        let Some(mut line) = line_opt else { break; };
                        if line.ends_with('\r') {
                            line.pop();
                        }

                        if line.is_empty() {
                            if current_event.is_none() && current_id.is_none() && current_data.is_empty() {
                                continue;
                            }

                            let event = current_event.take();
                            let data = current_data.join("\n");
                            current_data.clear();
                            let id = current_id.take();
                            handle_sse_event(
                                &server_name,
                                &sse_url,
                                &shared,
                                &notify,
                                event.as_deref(),
                                id.as_deref(),
                                data,
                            )
                            .await;
                            continue;
                        }

                        if line.starts_with(':') {
                            continue;
                        }
                        if let Some(rest) = line.strip_prefix("event:") {
                            current_event = Some(rest.trim().to_string());
                        }
                        if let Some(rest) = line.strip_prefix("data:") {
                            let rest = rest.strip_prefix(' ').unwrap_or(rest);
                            current_data.push(rest.to_string());
                        }
                        if let Some(rest) = line.strip_prefix("id:") {
                            current_id = Some(rest.trim().to_string());
                        }
                    }
                }
            }

            let pending = {
                let mut guard = shared.lock().await;
                std::mem::take(&mut guard.pending)
            };
            for (_, tx) in pending {
                let _ = tx.send(JsonRpcResponse {
                    jsonrpc: crate::mcp_protocol::JSONRPC_VERSION.to_string(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: INTERNAL_ERROR,
                        message: "SSE connection closed".to_string(),
                        data: None,
                    }),
                });
            }
        }));

        self.stream_state = SseStreamState::Connected;
        Ok(())
    }

    /// Determine the POST endpoint used for `tools/call` requests.
    async fn get_message_url(&self) -> Result<(String, bool)> {
        let guard = self.shared.lock().await;
        if let Some(url) = &guard.message_url {
            return Ok((url.clone(), guard.message_url_from_endpoint));
        }
        drop(guard);

        let derived = derive_message_url(&self.sse_url, "messages")
            .or_else(|| derive_message_url(&self.sse_url, "message"))
            .ok_or_else(|| anyhow!("invalid SSE URL"))?;
        let mut guard = self.shared.lock().await;
        if guard.message_url.is_none() {
            guard.message_url = Some(derived.clone());
            guard.message_url_from_endpoint = false;
        }
        Ok((derived, false))
    }
}

/// Shared mutable state for the SSE reader and request sender.
#[derive(Default)]
struct SseSharedState {
    /// Message POST endpoint.
    message_url: Option<String>,
    /// Whether the endpoint came from an explicit server event.
    message_url_from_endpoint: bool,
    /// Pending response channels by JSON-RPC request id.
    pending: HashMap<u64, oneshot::Sender<JsonRpcResponse>>,
}

/// Derive a best-effort POST endpoint from the SSE URL.
fn derive_message_url(sse_url: &str, message_path: &str) -> Option<String> {
    let url = reqwest::Url::parse(sse_url).ok()?;
    let mut segments: Vec<&str> = url.path_segments()?.collect();
    if segments.is_empty() {
        return None;
    }
    if segments.last().copied() == Some("sse") {
        segments.pop();
        segments.push(message_path);
        let mut new_url = url.clone();
        new_url.set_path(&format!("/{}", segments.join("/")));
        return Some(new_url.to_string());
    }

    let mut new_url = url.clone();
    let mut path = url.path().trim_end_matches('/').to_string();
    path.push('/');
    path.push_str(message_path);
    new_url.set_path(&path);
    Some(new_url.to_string())
}

/// Handle one completed SSE event frame.
async fn handle_sse_event(
    server_name: &str,
    sse_url: &str,
    shared: &std::sync::Arc<Mutex<SseSharedState>>,
    notify: &std::sync::Arc<Notify>,
    event: Option<&str>,
    _id: Option<&str>,
    data: String,
) {
    let event = event.unwrap_or("message");
    let trimmed = data.trim();
    if trimmed.is_empty() {
        return;
    }

    if event.eq_ignore_ascii_case("endpoint") || event.eq_ignore_ascii_case("mcp-endpoint") {
        if let Some(url) = parse_endpoint_from_data(sse_url, trimmed) {
            let mut guard = shared.lock().await;
            guard.message_url = Some(url);
            guard.message_url_from_endpoint = true;
            drop(guard);
            notify.notify_waiters();
        }
        return;
    }

    if !event.eq_ignore_ascii_case("message") {
        return;
    }

    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return;
    };
    let Ok(response) = serde_json::from_value::<JsonRpcResponse>(value.clone()) else {
        let _ = serde_json::from_value::<JsonRpcRequest>(value);
        return;
    };

    let Some(id_value) = response.id.clone() else {
        return;
    };
    let Some(id) = id_value.as_u64() else {
        return;
    };

    let tx = {
        let mut guard = shared.lock().await;
        guard.pending.remove(&id)
    };

    if let Some(tx) = tx {
        let _ = tx.send(response);
    } else {
        tracing::debug!(
            "MCP SSE `{}` received response for unknown id {}",
            server_name,
            id
        );
    }
}

/// Parse an alternate endpoint announcement from SSE event data.
fn parse_endpoint_from_data(sse_url: &str, data: &str) -> Option<String> {
    if data.starts_with('{') {
        let value: serde_json::Value = serde_json::from_str(data).ok()?;
        let endpoint = value.get("endpoint")?.as_str()?;
        return parse_endpoint_from_data(sse_url, endpoint);
    }
    if data.starts_with("http://") || data.starts_with("https://") {
        return Some(data.to_string());
    }
    let base = reqwest::Url::parse(sse_url).ok()?;
    base.join(data).ok().map(|url| url.to_string())
}

/// Extract the JSON payload from an SSE frame.
fn extract_json_from_sse_text(response_text: &str) -> Cow<'_, str> {
    let text = response_text.trim_start_matches('\u{feff}');
    let mut current_data_lines: Vec<&str> = Vec::new();
    let mut last_event_data_lines: Vec<&str> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim_end_matches('\r').trim_start();
        if line.is_empty() {
            if !current_data_lines.is_empty() {
                last_event_data_lines = std::mem::take(&mut current_data_lines);
            }
            continue;
        }
        if line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            current_data_lines.push(rest);
        }
    }

    if !current_data_lines.is_empty() {
        last_event_data_lines = current_data_lines;
    }

    if last_event_data_lines.is_empty() {
        return Cow::Borrowed(text.trim());
    }
    if last_event_data_lines.len() == 1 {
        return Cow::Borrowed(last_event_data_lines[0].trim());
    }

    Cow::Owned(last_event_data_lines.join("\n").trim().to_string())
}

/// Parse either plain JSON or SSE-framed JSON-RPC text.
fn parse_jsonrpc_response_text(response_text: &str) -> Result<JsonRpcResponse> {
    let trimmed = response_text.trim();
    if trimmed.is_empty() {
        bail!("MCP server returned no response");
    }

    let json_text = if looks_like_sse_text(trimmed) {
        extract_json_from_sse_text(trimmed)
    } else {
        Cow::Borrowed(trimmed)
    };

    serde_json::from_str(json_text.as_ref())
        .with_context(|| format!("invalid JSON-RPC response: {response_text}"))
}

/// Detect likely SSE framing.
fn looks_like_sse_text(text: &str) -> bool {
    text.starts_with("data:")
        || text.starts_with("event:")
        || text.contains("\ndata:")
        || text.contains("\nevent:")
}

/// Read the first JSON-RPC response from an event stream response body.
async fn read_first_jsonrpc_from_sse_response(
    response: reqwest::Response,
) -> Result<Option<JsonRpcResponse>> {
    let stream = response
        .bytes_stream()
        .map(|item| item.map_err(std::io::Error::other));
    let reader = tokio_util::io::StreamReader::new(stream);
    let mut lines = BufReader::new(reader).lines();

    let mut current_event: Option<String> = None;
    let mut current_data: Vec<String> = Vec::new();

    while let Ok(line_opt) = lines.next_line().await {
        let Some(mut line) = line_opt else { break };
        if line.ends_with('\r') {
            line.pop();
        }

        if line.is_empty() {
            if current_event.is_none() && current_data.is_empty() {
                continue;
            }
            let event = current_event.take();
            let data = current_data.join("\n");
            current_data.clear();

            let event = event.unwrap_or_else(|| "message".to_string());
            if event.eq_ignore_ascii_case("endpoint") || event.eq_ignore_ascii_case("mcp-endpoint")
            {
                continue;
            }
            if !event.eq_ignore_ascii_case("message") {
                continue;
            }

            let trimmed = data.trim();
            if trimmed.is_empty() {
                continue;
            }

            let json = extract_json_from_sse_text(trimmed);
            if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(json.as_ref()) {
                return Ok(Some(response));
            }
            continue;
        }

        if line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            current_event = Some(rest.trim().to_string());
        }
        if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            current_data.push(rest.to_string());
        }
    }

    Ok(None)
}

#[async_trait]
impl McpTransportConn for SseTransport {
    async fn send_and_recv(&mut self, request: &JsonRpcRequest) -> Result<JsonRpcResponse> {
        self.ensure_connected().await?;

        let request_id = request.id.as_ref().and_then(|value| value.as_u64());
        let body = serde_json::to_string(request)?;
        let (message_url, _from_endpoint) = self.get_message_url().await?;

        let mut response_rx = None;
        if let Some(id) = request_id {
            if self.stream_state == SseStreamState::Connected {
                let (tx, rx) = oneshot::channel();
                {
                    let mut guard = self.shared.lock().await;
                    guard.pending.insert(id, tx);
                }
                response_rx = Some((id, rx));
            }
        }

        let has_accept = self
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("Accept"));
        let has_content_type = self
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("Content-Type"));

        let mut req = self
            .client
            .post(&message_url)
            .timeout(Duration::from_secs(120))
            .body(body);
        if !has_content_type {
            req = req.header("Content-Type", MCP_JSON_CONTENT_TYPE);
        }
        for (key, value) in &self.headers {
            req = req.header(key, value);
        }
        if !has_accept {
            req = req.header("Accept", MCP_STREAMABLE_ACCEPT);
        }

        let response = req.send().await.context("SSE POST to MCP server failed")?;
        if !response.status().is_success() {
            if let Some((id, _)) = response_rx.as_ref() {
                let mut guard = self.shared.lock().await;
                guard.pending.remove(id);
            }
            bail!("MCP server returned HTTP {}", response.status());
        }

        if request.id.is_none() {
            return Ok(JsonRpcResponse {
                jsonrpc: crate::mcp_protocol::JSONRPC_VERSION.to_string(),
                id: None,
                result: None,
                error: None,
            });
        }

        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"));
        if is_sse {
            if let Some(response) = read_first_jsonrpc_from_sse_response(response).await? {
                if let Some((id, _)) = response_rx.as_ref() {
                    let mut guard = self.shared.lock().await;
                    guard.pending.remove(id);
                }
                return Ok(response);
            }
        } else {
            let text = response.text().await.unwrap_or_default();
            if !text.trim().is_empty() {
                if let Some((id, _)) = response_rx.as_ref() {
                    let mut guard = self.shared.lock().await;
                    guard.pending.remove(id);
                }
                return parse_jsonrpc_response_text(&text);
            }
        }

        let Some((_id, rx)) = response_rx else {
            bail!("MCP server returned no response");
        };
        rx.await.map_err(|_| anyhow!("SSE response channel closed"))
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
        Ok(())
    }
}

/// Create a transport from configuration.
pub fn create_transport(config: &McpServerConfig) -> Result<Box<dyn McpTransportConn>> {
    match config.transport {
        McpTransport::Stdio => Ok(Box::new(StdioTransport::new(config)?)),
        McpTransport::Http => Ok(Box::new(HttpTransport::new(config)?)),
        McpTransport::Sse => Ok(Box::new(SseTransport::new(config)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_json_response() {
        let parsed = parse_jsonrpc_response_text("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}")
            .expect("parse plain JSON");
        assert_eq!(parsed.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn parse_sse_framed_json_response() {
        let parsed = parse_jsonrpc_response_text(
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}\n\n",
        )
        .expect("parse SSE-framed JSON");
        assert_eq!(parsed.id, Some(serde_json::json!(2)));
    }
}
