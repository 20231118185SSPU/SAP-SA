//! Durable JSON-serializable runtime state shared by the daemon and tests.
//!
//! The new SA runtime is actor-based:
//! - every agent is long-lived
//! - every agent may have one active work item at a time
//! - background terminal jobs also become first-class runtime tasks
//! - restart recovery is driven from these persisted records

use crate::openai::{ChatMessage, ToolCall};
use crate::skills::ActiveCommandInvocation;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Wire/schema version for the durable runtime state.
pub const TEAM_RUNTIME_PROTOCOL_VERSION: u32 = 1;

/// Stable class of one durable agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    /// Root conversation agent owned directly by the current workspace.
    Root,
    /// General-purpose child agent created by another agent.
    Worker,
    /// Compact-triggered memory refresh agent.
    MemoryRefresh,
    /// Nightly dream-memory distillation agent.
    Dream,
}

/// Runtime lifecycle state of one agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// No active work; ready to be woken by mailbox traffic.
    Idle,
    /// Currently executing a model/tool turn.
    Running,
    /// Blocked on a user answer.
    WaitingUser,
    /// Blocked on a dependency such as another agent or background task.
    WaitingDependency,
    /// Restart recovery is reconstructing this agent.
    Recovering,
    /// Root-only administrative deletion is in progress.
    Deleting,
    /// Agent has been deleted from the live runtime.
    Deleted,
    /// Agent crashed in a way that prevented automatic recovery.
    Failed,
}

/// Durable record of the current root-level team state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TeamState {
    /// Stable root agent id for this workspace.
    pub root_agent_id: Uuid,
    /// Agent currently holding free-form user input ownership.
    pub input_owner_agent_id: Option<Uuid>,
    /// Schema version used by this state file.
    pub protocol_version: u32,
    /// Last mutation timestamp for audit/debugging.
    pub updated_at: DateTime<Utc>,
}

impl TeamState {
    /// Build a new root-owned team state.
    pub fn new(root_agent_id: Uuid) -> Self {
        Self {
            root_agent_id,
            input_owner_agent_id: Some(root_agent_id),
            protocol_version: TEAM_RUNTIME_PROTOCOL_VERSION,
            updated_at: Utc::now(),
        }
    }
}

/// Pending confirmation state for a `Finish` that still needs explicit
/// confirmation or follow-up output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingFinishConfirmation {
    /// Provisional finish reason captured from the tool call.
    pub reason: String,
    /// Provisional result summary captured from the tool call.
    pub result: String,
    /// Why an extra confirmation step is required.
    pub mode: PendingFinishMode,
}

/// Reason why the runtime is holding a provisional finish.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PendingFinishMode {
    /// Foreground root finished without `Send` or `Show`.
    RootNeedsOutputOrConfirmation,
    /// Child finished without explicitly messaging its parent.
    ChildNeedsParentAckOrConfirmation,
}

/// Assistant message data that must survive a crash before the runtime fully
/// applies one control decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingAssistantMessage {
    /// Assistant role name.
    pub role: String,
    /// Optional text content.
    pub content: Option<String>,
    /// Optional tool calls emitted by the assistant.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional tool-call id for tool-result messages.
    pub tool_call_id: Option<String>,
}

impl From<&ChatMessage> for PendingAssistantMessage {
    fn from(value: &ChatMessage) -> Self {
        Self {
            role: value.role.clone(),
            content: value.content.clone(),
            tool_calls: value.tool_calls.clone(),
            tool_call_id: value.tool_call_id.clone(),
        }
    }
}

impl From<&PendingAssistantMessage> for ChatMessage {
    fn from(value: &PendingAssistantMessage) -> Self {
        ChatMessage {
            role: value.role.clone(),
            content: value.content.clone(),
            tool_calls: value.tool_calls.clone(),
            tool_call_id: value.tool_call_id.clone(),
            request_usage: None,
            responses_input_items: None,
        }
    }
}

/// One durable control checkpoint captured between `run_quantum()` returning a
/// control decision and the daemon fully applying the corresponding runtime
/// transition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PendingControlAction {
    /// An `Ask` tool-call is waiting to be durably materialized into
    /// `pending_question.json`.
    Ask {
        assistant_message: PendingAssistantMessage,
        tool_call_id: String,
        prompt: String,
        mode: String,
        options_json: String,
        allow_free_text: bool,
    },
    /// A `Wait(...)` tool-call must be durably turned into `waiting_on`.
    Wait {
        assistant_message: PendingAssistantMessage,
        target_kind: WaitKind,
        target_id: Uuid,
        until: Option<WaitUntil>,
        timeout_seconds: Option<u64>,
    },
    /// A `Finish(...)` or `FinishWithoutOutput()` call must be durably applied.
    Finish {
        assistant_message: PendingAssistantMessage,
        reason: String,
        result: String,
        without_output_confirmation: bool,
    },
}

/// Durable state for one agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentState {
    /// Stable agent id.
    pub agent_id: Uuid,
    /// Parent agent id, if this is not the root.
    pub parent_agent_id: Option<Uuid>,
    /// Stable root agent for this whole tree.
    pub root_agent_id: Uuid,
    /// High-level kind of agent.
    pub kind: AgentKind,
    /// Human-friendly label used in logs and UI metadata.
    pub label: String,
    /// Current runtime lifecycle state.
    pub status: AgentStatus,
    /// Whether this agent may directly `Send`.
    pub allow_user_send: bool,
    /// Whether this agent may directly `Show`.
    pub allow_user_show: bool,
    /// Whether this agent may directly `Ask`.
    pub allow_user_ask: bool,
    /// Whether root may transfer input ownership to this agent.
    pub allow_input_transfer_target: bool,
    /// Active session file path for this agent.
    pub current_session_path: String,
    /// Previous compacted-away session file path for this agent.
    pub previous_session_path: Option<String>,
    /// Files currently eligible for `Edit` because they were freshly `Read`.
    #[serde(default)]
    pub tool_session_read_set: Vec<String>,
    /// Commands/skills currently active in this agent's long-lived context.
    ///
    /// These reminders survive restarts so command-scoped tool permissions and
    /// prompt guidance do not silently disappear after compaction or recovery.
    #[serde(default)]
    pub active_command_invocations: Vec<ActiveCommandInvocation>,
    /// Conditional commands that have already been activated by touched
    /// workspace paths.
    #[serde(default)]
    pub activated_conditional_commands: Vec<String>,
    /// Last consumed mailbox offset.
    pub last_mailbox_offset: u64,
    /// Currently active work id, if any.
    pub active_work_id: Option<Uuid>,
    /// Human-readable summary of the active work.
    pub active_work_summary: Option<String>,
    /// When the current work started.
    pub active_started_at: Option<DateTime<Utc>>,
    /// Whether this work has already emitted a direct `Send` or `Show`.
    pub work_has_user_output: bool,
    /// Whether this work has explicitly messaged the parent agent.
    pub work_has_parent_message: bool,
    /// Parent work that created the current active work, when this is a child
    /// agent. This keeps parent-child routing stable even if the parent later
    /// switches to another active work.
    pub parent_work_id: Option<Uuid>,
    /// Whether the next quantum should inject the temporary finish reminder.
    pub needs_finish_reminder: bool,
    /// Persisted dependency wait, if this work is currently suspended.
    pub waiting_on: Option<WaitingDependency>,
    /// Pending finish confirmation, if the runtime requested an extra check.
    pub pending_finish_confirmation: Option<PendingFinishConfirmation>,
    /// Pending control decision captured before the runtime fully applied it.
    pub pending_control: Option<PendingControlAction>,
    /// Last successful finish reason.
    pub last_finish_reason: Option<String>,
    /// Last successful finish result summary.
    pub last_finish_result: Option<String>,
    /// Most recent finished work id.
    pub last_finished_work_id: Option<Uuid>,
    /// When the most recent finished work completed.
    pub last_finished_at: Option<DateTime<Utc>>,
}

impl AgentState {
    /// Create a new root agent state.
    pub fn new_root(agent_id: Uuid, current_session_path: String) -> Self {
        Self {
            agent_id,
            parent_agent_id: None,
            root_agent_id: agent_id,
            kind: AgentKind::Root,
            label: "root".to_string(),
            status: AgentStatus::Idle,
            allow_user_send: true,
            allow_user_show: true,
            allow_user_ask: true,
            allow_input_transfer_target: false,
            current_session_path,
            previous_session_path: None,
            tool_session_read_set: Vec::new(),
            active_command_invocations: Vec::new(),
            activated_conditional_commands: Vec::new(),
            last_mailbox_offset: 0,
            active_work_id: None,
            active_work_summary: None,
            active_started_at: None,
            work_has_user_output: false,
            work_has_parent_message: false,
            parent_work_id: None,
            needs_finish_reminder: false,
            waiting_on: None,
            pending_finish_confirmation: None,
            pending_control: None,
            last_finish_reason: None,
            last_finish_result: None,
            last_finished_work_id: None,
            last_finished_at: None,
        }
    }

    /// Create a new child agent state inheriting the current root tree.
    #[allow(clippy::too_many_arguments)]
    pub fn new_child(
        agent_id: Uuid,
        parent_agent_id: Uuid,
        root_agent_id: Uuid,
        label: String,
        allow_user_send: bool,
        allow_user_show: bool,
        allow_user_ask: bool,
        allow_input_transfer_target: bool,
        current_session_path: String,
    ) -> Self {
        Self {
            agent_id,
            parent_agent_id: Some(parent_agent_id),
            root_agent_id,
            kind: AgentKind::Worker,
            label,
            status: AgentStatus::Idle,
            allow_user_send,
            allow_user_show,
            allow_user_ask,
            allow_input_transfer_target,
            current_session_path,
            previous_session_path: None,
            tool_session_read_set: Vec::new(),
            active_command_invocations: Vec::new(),
            activated_conditional_commands: Vec::new(),
            last_mailbox_offset: 0,
            active_work_id: None,
            active_work_summary: None,
            active_started_at: None,
            work_has_user_output: false,
            work_has_parent_message: false,
            parent_work_id: None,
            needs_finish_reminder: false,
            waiting_on: None,
            pending_finish_confirmation: None,
            pending_control: None,
            last_finish_reason: None,
            last_finish_result: None,
            last_finished_work_id: None,
            last_finished_at: None,
        }
    }
}

/// Kind of waitable runtime target.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WaitKind {
    /// Wait on an agent lifecycle state.
    Agent,
    /// Wait on one specific work item.
    Work,
    /// Wait on one background runtime task.
    Task,
}

/// Condition a `Wait` is watching for.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WaitUntil {
    /// Agent has become idle.
    Idle,
    /// Work item has finished via explicit `Finish`.
    Finished,
    /// Background task has exited.
    Exited,
}

/// Persisted dependency wait used to restore `Wait(...)` after a restart.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WaitingDependency {
    /// Target kind this agent is waiting on.
    pub kind: WaitKind,
    /// Target id.
    pub id: Uuid,
    /// Desired target state.
    pub until: WaitUntil,
    /// Absolute timeout deadline, if the wait is bounded.
    pub timeout_at: Option<DateTime<Utc>>,
}

/// Durable lifecycle state of one logical work item executed by an agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeWorkStatus {
    /// Work is currently active.
    Running,
    /// Work finished normally via explicit `Finish`.
    Finished,
    /// Work was cancelled intentionally, for example by interrupt.
    Cancelled,
}

/// Durable record for one logical work item so `Wait(kind=work)` can remain
/// valid even after the owning agent starts newer work items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeWorkState {
    /// Stable work id.
    pub work_id: Uuid,
    /// Current owner agent.
    pub owner_agent_id: Uuid,
    /// Root tree containing this work.
    pub root_agent_id: Uuid,
    /// Human-readable summary captured when the work started.
    pub summary: String,
    /// Current lifecycle state.
    pub status: RuntimeWorkStatus,
    /// When the work started.
    pub started_at: DateTime<Utc>,
    /// When the work reached a terminal state, if applicable.
    pub finished_at: Option<DateTime<Utc>>,
    /// Optional human-readable outcome.
    pub result_summary: Option<String>,
}

/// Kind of durable runtime task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTaskKind {
    /// Background terminal/Bash process.
    Terminal,
}

/// Lifecycle state for one runtime task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTaskStatus {
    /// Task is still running.
    Running,
    /// Task completed normally.
    Exited,
    /// Task failed before it could produce a normal exit.
    Failed,
    /// Task was cancelled intentionally.
    Cancelled,
}

/// Durable state for one runtime task such as a background Bash command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeTaskState {
    /// Stable task id.
    pub task_id: Uuid,
    /// Owning agent that spawned the task.
    pub owner_agent_id: Uuid,
    /// Runtime task kind.
    pub kind: RuntimeTaskKind,
    /// Current task lifecycle state.
    pub status: RuntimeTaskStatus,
    /// Command summary for audit/debugging.
    pub command: String,
    /// Working directory used by the task.
    pub workdir: String,
    /// When the task started.
    pub started_at: DateTime<Utc>,
    /// When the task finished, if applicable.
    pub finished_at: Option<DateTime<Utc>>,
    /// Exit code, when available.
    pub exit_code: Option<i32>,
    /// Output file path storing stdout/stderr or other task artifacts.
    pub output_path: Option<String>,
    /// Best-effort summary of the final outcome.
    pub summary: Option<String>,
    /// Free-form metadata for forward-compatible extensions.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

/// Durable mailbox entry kind.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MailboxEntryKind {
    /// Free-form user input routed to an agent.
    UserInput,
    /// Direct point-to-point agent message.
    AgentMessage,
    /// Broadcast fan-out message.
    BroadcastMessage,
    /// Child work finished and reported back to its parent.
    ChildFinished,
    /// Background task finished and reported back to its owner.
    TaskFinished,
    /// System-generated runtime note.
    SystemNotice,
}

/// One append-only mailbox entry delivered to an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MailboxEntry {
    /// Monotonic mailbox offset for one agent.
    pub offset: u64,
    /// Stable message id for idempotency.
    pub entry_id: Uuid,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Entry class.
    pub kind: MailboxEntryKind,
    /// Sender agent id when applicable.
    pub from_agent_id: Option<Uuid>,
    /// Human-friendly sender label when available.
    pub from_label: Option<String>,
    /// Text payload.
    pub message: String,
    /// Associated work id when available.
    pub work_id: Option<Uuid>,
    /// Related entity id such as another agent/work/task.
    pub related_id: Option<Uuid>,
}

/// Durable pending-question state written before an agent blocks on `Ask`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingQuestionState {
    /// Agent waiting for the answer.
    pub agent_id: Uuid,
    /// Work blocked by this question.
    pub work_id: Uuid,
    /// Stable question id returned to the frontend.
    pub question_id: Uuid,
    /// Assistant tool-call id that should receive the eventual tool-result
    /// message once the question is answered.
    pub tool_call_id: String,
    /// Prompt text.
    pub prompt: String,
    /// Raw question mode string.
    pub mode: String,
    /// Serialized options payload.
    pub options_json: String,
    /// Whether free text is allowed.
    pub allow_free_text: bool,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn team_state_defaults_to_root_holding_input() {
        let root = Uuid::new_v4();
        let state = TeamState::new(root);
        assert_eq!(state.root_agent_id, root);
        assert_eq!(state.input_owner_agent_id, Some(root));
        assert_eq!(state.protocol_version, TEAM_RUNTIME_PROTOCOL_VERSION);
    }

    #[test]
    fn root_agent_defaults_allow_direct_user_io() {
        let state = AgentState::new_root(Uuid::new_v4(), "sessions/root.jsonl".to_string());
        assert_eq!(state.kind, AgentKind::Root);
        assert_eq!(state.status, AgentStatus::Idle);
        assert!(state.allow_user_send);
        assert!(state.allow_user_show);
        assert!(state.allow_user_ask);
        assert!(state.active_work_id.is_none());
    }

    #[test]
    fn runtime_task_state_round_trips_through_json() {
        let task = RuntimeTaskState {
            task_id: Uuid::new_v4(),
            owner_agent_id: Uuid::new_v4(),
            kind: RuntimeTaskKind::Terminal,
            status: RuntimeTaskStatus::Exited,
            command: "cargo test".to_string(),
            workdir: "G:/repo".to_string(),
            started_at: Utc::now(),
            finished_at: Some(Utc::now()),
            exit_code: Some(0),
            output_path: Some("runtime/tasks/task-1.log".to_string()),
            summary: Some("completed".to_string()),
            metadata: BTreeMap::from([(String::from("key"), String::from("value"))]),
        };

        let json = serde_json::to_string(&task).expect("task should serialize");
        let parsed =
            serde_json::from_str::<RuntimeTaskState>(&json).expect("task should deserialize");
        assert_eq!(parsed.kind, RuntimeTaskKind::Terminal);
        assert_eq!(parsed.status, RuntimeTaskStatus::Exited);
        assert_eq!(parsed.exit_code, Some(0));
        assert_eq!(
            parsed.metadata.get("key").map(String::as_str),
            Some("value")
        );
    }
}
