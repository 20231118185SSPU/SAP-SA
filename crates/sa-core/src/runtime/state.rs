//! Durable JSON-serializable runtime state shared by the daemon and tests.
//!
//! The new SA runtime is actor-based:
//! - every agent is long-lived
//! - every agent may have one active work item at a time
//! - background terminal jobs also become first-class runtime tasks
//! - restart recovery is driven from these persisted records

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
    pub tool_session_read_set: Vec<String>,
    /// Last consumed mailbox offset.
    pub last_mailbox_offset: u64,
    /// Currently active work id, if any.
    pub active_work_id: Option<Uuid>,
    /// Human-readable summary of the active work.
    pub active_work_summary: Option<String>,
    /// When the current work started.
    pub active_started_at: Option<DateTime<Utc>>,
    /// Pending finish confirmation, if the runtime requested an extra check.
    pub pending_finish_confirmation: Option<PendingFinishConfirmation>,
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
            last_mailbox_offset: 0,
            active_work_id: None,
            active_work_summary: None,
            active_started_at: None,
            pending_finish_confirmation: None,
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
        assert_eq!(parsed.metadata.get("key").map(String::as_str), Some("value"));
    }
}

