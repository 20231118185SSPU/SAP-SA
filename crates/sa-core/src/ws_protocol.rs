//! WebSocket JSON protocol shared by the daemon and CLI.
//!
//! The request asked for:
//! - A WS connection between a CLI frontend and the agent daemon.
//! - Communication interruptions must not affect the agent's execution.
//! - The agent can proactively send messages and structured questions to users.
//!
//! To make that work reliably, we use:
//! - A monotonically increasing `event_id` so clients can reconnect and
//!   request history since the last seen event.
//! - A stable, explicit JSON schema (tagged enums).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Client → server messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    /// Submit a new task to the agent.
    #[serde(rename = "submit")]
    Submit {
        /// Client-generated task id for idempotent retries.
        ///
        /// If omitted, the server generates one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<Uuid>,

        /// The task text for the agent.
        task: String,
    },

    /// Request event history starting *after* `from_event_id`.
    #[serde(rename = "get_history")]
    GetHistory { from_event_id: u64 },

    /// Interrupt (cancel) a running task.
    ///
    /// This is the key feature that enables the CLI to "send a message at any
    /// time" and stop the current autonomous loop without killing the daemon.
    #[serde(rename = "interrupt")]
    Interrupt { task_id: Uuid },

    /// Answer a structured question previously asked by the agent.
    #[serde(rename = "answer_question")]
    AnswerQuestion { answer: UserQuestionAnswer },
}

/// Server → client messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// Task accepted and queued.
    #[serde(rename = "accepted")]
    Accepted { task_id: Uuid },

    /// A chunk of event history.
    #[serde(rename = "history")]
    History { events: Vec<Event> },

    /// A single real-time event.
    #[serde(rename = "event")]
    Event { event: Event },

    /// A newly issued structured question from the agent.
    #[serde(rename = "question")]
    Question { question: UserQuestion },

    /// All questions still waiting for a user answer.
    #[serde(rename = "pending_questions")]
    PendingQuestions { questions: Vec<UserQuestion> },

    /// A previously pending question has been resolved or cancelled.
    #[serde(rename = "question_resolved")]
    QuestionResolved { question_id: Uuid },

    /// A user-visible file payload emitted by the `Show` tool.
    #[serde(rename = "show")]
    Show { file: UserVisibleFile },

    /// Snapshot of the most recent `Show` payloads for reconnecting clients.
    #[serde(rename = "recent_shows")]
    RecentShows { files: Vec<UserVisibleFile> },

    /// A protocol-level error (bad request, parse error, etc.).
    #[serde(rename = "error")]
    Error { message: String },
}

/// An event emitted by the agent daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Monotonically increasing identifier assigned by the server.
    pub event_id: u64,

    /// UTC timestamp of the event.
    pub ts: DateTime<Utc>,

    /// Task that produced the event.
    pub task_id: Uuid,

    /// Classification used by UIs.
    pub kind: EventKind,

    /// Human-readable message.
    pub message: String,
}

/// A structured question that the agent asks the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestion {
    /// Unique question id used to correlate the eventual answer.
    pub question_id: Uuid,

    /// Top-level task this question belongs to.
    pub task_id: Uuid,

    /// Human-readable prompt shown to the user.
    pub prompt: String,

    /// Question mode (single choice, multi choice, or free text).
    pub mode: QuestionMode,

    /// Available options for choice-based questions.
    #[serde(default)]
    pub options: Vec<QuestionOption>,

    /// Whether the user may additionally provide free text.
    #[serde(default)]
    pub allow_free_text: bool,
}

/// One selectable option in a structured question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    /// Stable option id used in answers.
    pub id: String,

    /// User-facing label.
    pub label: String,

    /// Optional explanatory text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// How the user is expected to answer a structured question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionMode {
    /// Exactly one option may be chosen.
    SingleChoice,

    /// Zero or more options may be chosen.
    MultiChoice,

    /// A free-text answer is expected.
    Text,
}

/// A user answer to a previously asked structured question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestionAnswer {
    /// Which question this answer belongs to.
    pub question_id: Uuid,

    /// Selected option ids (if any).
    #[serde(default)]
    pub selected_option_ids: Vec<String>,

    /// Optional free-text input from the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_text: Option<String>,
}

/// A file payload explicitly shown to the user by the `Show` tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserVisibleFile {
    /// Unique show id.
    pub show_id: Uuid,

    /// Top-level task that produced this file.
    pub task_id: Uuid,

    /// Original workspace-relative or absolute path resolved by the backend.
    pub path: String,

    /// Optional user-facing title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Best-effort media type hint.
    pub media_type: String,

    /// How to interpret `content`.
    pub encoding: UserVisibleFileEncoding,

    /// Text body or base64-encoded bytes, depending on `encoding`.
    pub content: String,

    /// Original byte size before transport encoding.
    pub bytes: usize,
}

/// How a shown file's `content` field is encoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserVisibleFileEncoding {
    /// UTF-8 text sent directly.
    Utf8,

    /// Binary bytes encoded as base64 text.
    Base64,
}

/// Event kind (used for CLI coloring/filtering later).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Normal logging/progress output.
    Log,
    /// Tool execution (requests or results).
    Tool,
    /// A direct user-facing message sent by the agent.
    Message,
    /// Final answer / completion.
    Final,
    /// Errors that did not necessarily crash the daemon.
    Error,
}
