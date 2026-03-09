//! WebSocket JSON protocol shared by the daemon and CLI.
//!
//! Handshake overview:
//! - The client must prove it is an expected SA frontend first.
//! - The server verifies that proof.
//! - Only after verification does the server return its own proof.
//! - Once both sides verify each other, the normal WS message stream begins.
//!
//! This is intentionally explicit JSON with tagged enums so reconnect logic,
//! manual debugging, and protocol evolution all stay traceable.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Client -> server messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    /// Mandatory first packet for mutual protocol identification.
    #[serde(rename = "client_hello")]
    ClientHello { hello: ClientHello },

    /// Submit a new task to the agent.
    #[serde(rename = "submit")]
    Submit {
        /// Client-generated task id for idempotent retries.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<Uuid>,
        /// The task text for the agent.
        task: String,
    },

    /// Request event history starting after `from_event_id`.
    #[serde(rename = "get_history")]
    GetHistory { from_event_id: u64 },

    /// Interrupt a running task.
    #[serde(rename = "interrupt")]
    Interrupt { task_id: Uuid },

    /// Answer a pending structured question.
    #[serde(rename = "answer_question")]
    AnswerQuestion { answer: UserQuestionAnswer },
}

/// Server -> client messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// Server proof returned only after the client's `client_hello` passed.
    #[serde(rename = "server_hello")]
    ServerHello { hello: ServerHello },

    /// Handshake rejection. The client should treat the connection as invalid.
    #[serde(rename = "hello_reject")]
    HelloReject { reject: HelloReject },

    /// Task accepted and queued.
    #[serde(rename = "accepted")]
    Accepted { task_id: Uuid },

    /// Buffered event history.
    #[serde(rename = "history")]
    History { events: Vec<Event> },

    /// One live event.
    #[serde(rename = "event")]
    Event { event: Event },

    /// A newly issued structured question.
    #[serde(rename = "question")]
    Question { question: UserQuestion },

    /// Snapshot of all pending questions.
    #[serde(rename = "pending_questions")]
    PendingQuestions { questions: Vec<UserQuestion> },

    /// A question has been resolved or cancelled.
    #[serde(rename = "question_resolved")]
    QuestionResolved { question_id: Uuid },

    /// A user-visible file payload emitted by the backend `Show` tool.
    #[serde(rename = "show")]
    Show { file: UserVisibleFile },

    /// Snapshot of the most recent shown files for reconnecting clients.
    #[serde(rename = "recent_shows")]
    RecentShows { files: Vec<UserVisibleFile> },

    /// Protocol or request error.
    #[serde(rename = "error")]
    Error { message: String },
}

/// The client's proof-of-identity packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientHello {
    /// Stable protocol identifier (example: `sa-ws/v1`).
    pub protocol: String,
    /// Hash algorithm identifier used by the proofs.
    pub hash_algo: String,
    /// Time bucket width in seconds.
    pub time_step_secs: u64,
    /// Allowed clock skew in time buckets.
    pub allowed_skew_buckets: u64,
    /// Expected frontend name (currently `sa-cli`).
    pub client_name: String,
    /// Frontend binary version.
    pub client_version: String,
    /// UTC time bucket used to derive the proof.
    pub time_bucket: i64,
    /// Short hint derived from the local machine fingerprint.
    pub machine_hint: String,
    /// Client-generated nonce to bind the server proof to this exact handshake.
    pub client_nonce: String,
    /// Hex-encoded SHA-256 proof.
    pub proof: String,
}

/// The server's proof-of-identity packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHello {
    /// Stable protocol identifier (example: `sa-ws/v1`).
    pub protocol: String,
    /// Hash algorithm identifier used by the proofs.
    pub hash_algo: String,
    /// Time bucket width in seconds.
    pub time_step_secs: u64,
    /// Allowed clock skew in time buckets.
    pub allowed_skew_buckets: u64,
    /// Expected backend name (currently `sa`).
    pub server_name: String,
    /// Backend binary version.
    pub server_version: String,
    /// UTC time bucket used to derive the proof.
    pub time_bucket: i64,
    /// Short hint derived from the local machine fingerprint.
    pub machine_hint: String,
    /// Echo of the client's nonce.
    pub client_nonce: String,
    /// Server-generated nonce.
    pub server_nonce: String,
    /// Hex-encoded SHA-256 proof.
    pub proof: String,
}

/// Rejection payload returned when the mandatory handshake fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloReject {
    /// Stable protocol identifier.
    pub protocol: String,
    /// Server name emitting the rejection.
    pub server_name: String,
    /// Server binary version.
    pub server_version: String,
    /// Human-readable rejection reason.
    pub reason: String,
}

/// One event emitted by the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Monotonic event id assigned by the server.
    pub event_id: u64,
    /// UTC timestamp of the event.
    pub ts: DateTime<Utc>,
    /// Task that produced the event.
    pub task_id: Uuid,
    /// Event classification used by the UI.
    pub kind: EventKind,
    /// Human-readable message.
    pub message: String,
}

/// Structured question emitted by the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestion {
    /// Unique question id.
    pub question_id: Uuid,
    /// Top-level task id this question belongs to.
    pub task_id: Uuid,
    /// User-facing prompt.
    pub prompt: String,
    /// Expected answer shape.
    pub mode: QuestionMode,
    /// Choice options for non-text questions.
    #[serde(default)]
    pub options: Vec<QuestionOption>,
    /// Whether extra free text is allowed.
    #[serde(default)]
    pub allow_free_text: bool,
}

/// One selectable option inside a structured question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    /// Stable option id used in answers.
    pub id: String,
    /// User-facing label.
    pub label: String,
    /// Optional descriptive help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// How the user should answer the question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionMode {
    /// Exactly one choice.
    SingleChoice,
    /// Zero or more choices.
    MultiChoice,
    /// Pure free text.
    Text,
}

/// User answer sent back to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestionAnswer {
    /// Which question this answers.
    pub question_id: Uuid,
    /// Selected option ids, if any.
    #[serde(default)]
    pub selected_option_ids: Vec<String>,
    /// Optional free-text payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_text: Option<String>,
}

/// A file payload explicitly shown to the user by the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserVisibleFile {
    /// Unique show id.
    pub show_id: Uuid,
    /// Top-level task that produced this file.
    pub task_id: Uuid,
    /// Original backend path string.
    pub path: String,
    /// Optional user-facing title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Best-effort media type hint.
    pub media_type: String,
    /// How `content` should be interpreted.
    pub encoding: UserVisibleFileEncoding,
    /// Text body or base64-encoded bytes.
    pub content: String,
    /// Original byte size before transport encoding.
    pub bytes: usize,
}

/// Encoding used for `UserVisibleFile.content`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserVisibleFileEncoding {
    /// UTF-8 text sent directly.
    Utf8,
    /// Binary bytes encoded as base64 text.
    Base64,
}

/// Event kind used by the CLI for formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Normal logging/progress output.
    Log,
    /// Tool execution details.
    Tool,
    /// A direct user-facing message.
    Message,
    /// Final answer / completion.
    Final,
    /// Recoverable error output.
    Error,
}
