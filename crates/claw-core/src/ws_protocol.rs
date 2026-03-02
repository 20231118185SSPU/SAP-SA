//! WebSocket JSON protocol shared by the daemon and CLI.
//!
//! The request asked for:
//! - A WS connection between a CLI frontend and the agent daemon.
//! - Communication interruptions must not affect the agent's execution.
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

/// Event kind (used for CLI coloring/filtering later).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Normal logging/progress output.
    Log,
    /// Tool execution (requests or results).
    Tool,
    /// Final answer / completion.
    Final,
    /// Errors that did not necessarily crash the daemon.
    Error,
}
