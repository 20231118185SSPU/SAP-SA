//! Built-in tools for the StudyAdministrator (SA) agent.
//!
//! The user requested a deliberate redesign of the built-in toolset. The
//! previous minimal set (`shell_command`, `read_file`, `write_file`, etc.) is
//! replaced by these exact tools:
//! - `Read`: read a UTF-8 text file.
//! - `Write`: create a new UTF-8 text file; refuse if the file already exists.
//! - `Edit`: edit an existing UTF-8 text file; the file must have been `Read`
//!   earlier in the same agent session.
//! - `Bash`: execute a shell command via Git Bash (`bash -lc`).
//! - `Fetch`: make a direct HTTP request to a known URL.
//! - `Search`: perform a web search to discover relevant URLs.
//! - `MemorySearch`: search Markdown memory files on demand.
//! - `MemoryGet`: read one bounded slice from a Markdown memory file.
//! - `Send`: send a user-facing message without blocking for a reply.
//! - `Ask`: ask the user a structured question and wait for the answer.
//! - `Show`: transmit a file's contents to the user over WebSocket.
//! - `Skill`: read `SKILL.md` or another skill-relative file from a named skill.
//! - `SubAgent`: launch a nested sub-agent and return its final answer.
//!
//! Design goals:
//! - Keep the file/workspace boundary explicit and verifiable.
//! - Make the `Edit` safety rule enforceable in Rust rather than hoping the
//!   model follows it.
//! - Allow the daemon layer to inject user-interaction and sub-agent behavior
//!   without coupling this crate to any particular transport.

use crate::bash_safety::{BashSafetyDecision, validate_bash_command_safety};
use crate::cancel::CancelToken;
use crate::fetch_safety::{
    FETCH_TIMEOUT_SECS, MAX_FETCH_REDIRECTS, MAX_FETCH_TRANSFER_BYTES, is_permitted_redirect,
    validate_fetch_request,
};
use crate::interaction_history::{
    InteractionDisclosureMode, InteractionReadOptions, InteractionStore,
};
use crate::mcp_client::McpRegistry;
use crate::memory::{read_markdown_memory, search_markdown_memory};
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::path_guard::{PathOperation, validate_resolved_tool_path, validate_tool_path_input};
use crate::runtime::state::{AgentStatus, RuntimeTaskStatus, WaitKind, WaitUntil};
use crate::skills::{ActiveCommandInvocation, SkillRegistry, matches_command_patterns};
use crate::ws_protocol::{
    QuestionMode, QuestionOption, UserQuestionAnswer, UserVisibleFile, UserVisibleFileEncoding,
};
use anyhow::Context as _;
use base64::Engine as _;
use serde::de::Error as _;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::OsString;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Maximum file size accepted by `Read`.
///
/// This is a context-window guardrail. Large files should be explored through
/// more targeted reads or shell commands rather than dumping megabytes into the
/// model context.
const MAX_READ_FILE_BYTES: u64 = 200_000;

/// Default timeout for `Bash` if the caller does not specify one.
const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_secs(300);

/// Maximum timeout accepted by `Bash`.
///
/// The tool should still be practical for builds/tests, but we keep an upper
/// bound so one tool call cannot hang indefinitely.
const MAX_BASH_TIMEOUT: Duration = Duration::from_secs(1_800);

/// Maximum response bytes returned by `Fetch`.
const MAX_FETCH_RESPONSE_BYTES: usize = 200_000;

/// Maximum file size transported by `Show`.
const MAX_SHOW_FILE_BYTES: usize = 512 * 1024;

/// Safety limit for nested sub-agents.
///
/// The user explicitly asked for recursive sub-agents. We still need a hard
/// ceiling so a prompt bug cannot recurse forever.
pub const MAX_SUBAGENT_DEPTH: u32 = 6;

/// Boxed async return type used by runtime callbacks.
type ToolFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Deserialize one optional string field, treating empty/blank strings as
/// `None`.
fn deserialize_optional_nonempty_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.and_then(|raw| {
        if raw.trim().is_empty() {
            None
        } else {
            Some(raw)
        }
    }))
}

/// Deserialize one optional UUID field, treating empty/blank strings as
/// `None`.
fn deserialize_optional_uuid_or_empty<'de, D>(deserializer: D) -> Result<Option<Uuid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(raw) if raw.trim().is_empty() => Ok(None),
        Some(raw) => Uuid::parse_str(raw.trim())
            .map(Some)
            .map_err(D::Error::custom),
    }
}

/// Callback used by `Send`.
pub type SendMessageFn =
    Arc<dyn Fn(String) -> ToolFuture<anyhow::Result<()>> + Send + Sync + 'static>;

/// Callback used by `Ask`.
pub type AskQuestionFn = Arc<
    dyn Fn(AskRequest, CancelToken) -> ToolFuture<anyhow::Result<UserQuestionAnswer>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by `Show`.
pub type ShowFileFn =
    Arc<dyn Fn(UserVisibleFile) -> ToolFuture<anyhow::Result<()>> + Send + Sync + 'static>;

/// Callback used by `SubAgent`.
pub type RunSubAgentFn = Arc<
    dyn Fn(SubAgentRequest, CancelToken) -> ToolFuture<anyhow::Result<SubAgentHandle>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by `NotifyParent`.
pub type NotifyParentFn =
    Arc<dyn Fn(String) -> ToolFuture<anyhow::Result<AgentMessageReceipt>> + Send + Sync + 'static>;

/// Callback used by `MessageAgent`.
pub type MessageAgentFn = Arc<
    dyn Fn(AgentMessageRequest) -> ToolFuture<anyhow::Result<AgentMessageReceipt>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by `BroadcastAgents`.
pub type BroadcastAgentsFn = Arc<
    dyn Fn(BroadcastAgentsRequest) -> ToolFuture<anyhow::Result<BroadcastReceipt>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by `ListAgents`.
pub type ListAgentsFn = Arc<
    dyn Fn(ListAgentsRequest) -> ToolFuture<anyhow::Result<Vec<AgentInfo>>> + Send + Sync + 'static,
>;

/// Callback used by `GetAgent`.
pub type GetAgentFn =
    Arc<dyn Fn(Uuid) -> ToolFuture<anyhow::Result<AgentInfo>> + Send + Sync + 'static>;

/// Callback used by `TransferInput`.
pub type TransferInputFn = Arc<
    dyn Fn(TransferInputRequest) -> ToolFuture<anyhow::Result<TransferInputReceipt>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by background Bash execution.
pub type StartTerminalTaskFn = Arc<
    dyn Fn(StartTerminalTaskRequest, CancelToken) -> ToolFuture<anyhow::Result<TerminalTaskHandle>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by `GetTask`.
pub type GetTaskFn =
    Arc<dyn Fn(Uuid) -> ToolFuture<anyhow::Result<TerminalTaskInfo>> + Send + Sync + 'static>;

/// Callback used by `Reload`.
pub type ReloadRuntimeFn =
    Arc<dyn Fn() -> ToolFuture<anyhow::Result<ReloadRuntimeReceipt>> + Send + Sync + 'static>;

/// Structured result returned by the runtime hot-reload callback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReloadRuntimeReceipt {
    /// Human-readable summary of what was reloaded.
    pub summary: String,
}

/// Structured request emitted by the `Ask` tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskRequest {
    /// Human-readable question prompt.
    pub prompt: String,
    /// How the answer should be collected.
    pub mode: QuestionMode,
    /// Selectable options for choice-based prompts.
    pub options: Vec<QuestionOption>,
    /// Whether optional free text is allowed in addition to selections.
    pub allow_free_text: bool,
}

impl AskRequest {
    /// Validate the request before it leaves the tool layer.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.prompt.trim().is_empty() {
            anyhow::bail!("Ask prompt must not be empty");
        }

        match self.mode {
            QuestionMode::SingleChoice | QuestionMode::MultiChoice => {
                if self.options.is_empty() {
                    anyhow::bail!("Choice-based Ask requests must provide at least one option");
                }
            }
            QuestionMode::Text => {
                if !self.options.is_empty() {
                    anyhow::bail!("Text Ask requests must not provide options");
                }
            }
        }

        let mut seen = HashSet::<String>::new();
        for option in &self.options {
            if option.id.trim().is_empty() {
                anyhow::bail!("Ask option ids must not be empty");
            }
            if option.label.trim().is_empty() {
                anyhow::bail!("Ask option labels must not be empty");
            }
            if !seen.insert(option.id.clone()) {
                anyhow::bail!("Ask option ids must be unique: {}", option.id);
            }
        }

        Ok(())
    }
}

/// Control signal returned by a tool instead of a plain observation payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolControl {
    /// Persist a structured user question and suspend until the runtime injects
    /// the eventual answer as a tool-result message.
    Ask(AskRequest),
    /// Explicit request to finish the current work.
    Finish(FinishRequest),
    /// Explicit confirmation that finishing without extra outward output is
    /// acceptable for this work.
    FinishWithoutOutput,
    /// Request to suspend execution until a dependency reaches a target state.
    Wait(WaitRequest),
}

/// Result returned from one tool execution.
#[derive(Debug, Clone)]
pub enum ToolExecutionResult {
    /// Normal tool output that should be appended as a tool-result message.
    Observation(String),
    /// Control signal that changes the agent runtime state.
    Control(ToolControl),
    /// Image payload that should be injected as a multimodal user message
    /// instead of a plain-text tool result, so the LLM can "see" the image.
    ImagePayload {
        /// A `data:image/...;base64,...` URL or a file path that the agent
        /// runtime will embed into the next user turn.
        data_url: String,
        /// MIME type of the image (e.g. `image/png`).
        media_type: String,
        /// Optional text the model supplied alongside the image (e.g. a prompt
        /// describing what to analyse).  If absent a generic prompt is used.
        prompt: Option<String>,
    },
}

/// Structured finish request emitted by the `Finish` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinishRequest {
    /// Human-readable reason explaining why the current work is complete.
    pub reason: String,
    /// Result summary for later audit/recovery and parent-agent coordination.
    pub result: String,
}

impl FinishRequest {
    /// Validate `Finish` arguments before they reach the runtime.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.reason.trim().is_empty() {
            anyhow::bail!("Finish reason must not be empty");
        }
        if self.result.trim().is_empty() {
            anyhow::bail!("Finish result must not be empty");
        }
        Ok(())
    }
}

/// Supported scopes for agent discovery and broadcasts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentScope {
    /// Direct children only.
    Children,
    /// All descendants below the current agent.
    Descendants,
    /// Agents that share the same parent.
    Siblings,
    /// Every agent under the current root tree.
    AllUnderRoot,
}

/// One spawned or reused child-agent handle returned by `SubAgent`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubAgentHandle {
    /// Stable child agent id.
    pub agent_id: Uuid,
    /// Stable work id accepted by that child.
    pub work_id: Uuid,
    /// Human-friendly label for logs/UI metadata.
    pub label: String,
    /// Best-effort runtime status snapshot.
    pub status: AgentStatus,
}

/// One direct agent-to-agent message request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentMessageRequest {
    /// Target agent receiving the message.
    pub target_agent_id: Uuid,
    /// Plain-text message content.
    pub message: String,
}

impl AgentMessageRequest {
    /// Validate that the message is non-empty before dispatch.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.message.trim().is_empty() {
            anyhow::bail!("Agent message must not be empty");
        }
        Ok(())
    }
}

/// Delivery receipt returned after one agent-to-agent message is queued.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentMessageReceipt {
    /// Target agent that accepted the message.
    pub target_agent_id: Uuid,
    /// Whether the runtime accepted the message for delivery.
    pub delivered: bool,
}

/// Request emitted by `BroadcastAgents`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BroadcastAgentsRequest {
    /// Broadcast scope under the current root tree.
    pub scope: AgentScope,
    /// Plain-text message content.
    pub message: String,
}

impl BroadcastAgentsRequest {
    /// Validate the broadcast payload.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.message.trim().is_empty() {
            anyhow::bail!("Broadcast message must not be empty");
        }
        Ok(())
    }
}

/// Receipt returned after a broadcast fan-out is queued.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BroadcastReceipt {
    /// Scope used by the broadcast.
    pub scope: AgentScope,
    /// Number of recipients reached.
    pub recipients: usize,
}

/// Query emitted by `ListAgents`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ListAgentsRequest {
    /// Which subset of the current root tree to enumerate.
    pub scope: AgentScope,
}

/// Runtime metadata returned by `ListAgents` and `GetAgent`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentInfo {
    /// Stable agent id.
    pub agent_id: Uuid,
    /// Optional parent agent id.
    pub parent_agent_id: Option<Uuid>,
    /// Human-friendly label.
    pub label: String,
    /// Current status.
    pub status: AgentStatus,
    /// Current active work id, if any.
    pub active_work_id: Option<Uuid>,
    /// Whether the agent may directly `Send`.
    pub allow_user_send: bool,
    /// Whether the agent may directly `Show`.
    pub allow_user_show: bool,
    /// Whether the agent may directly `Ask`.
    pub allow_user_ask: bool,
}

/// Input-ownership transfer request emitted by `TransferInput`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransferInputRequest {
    /// New input owner, or `None` to return ownership to the root.
    pub target_agent_id: Option<Uuid>,
}

/// Receipt returned after input ownership changes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransferInputReceipt {
    /// Effective input owner after the transfer.
    pub input_owner_agent_id: Uuid,
}

/// Background terminal-task launch request emitted by `Bash`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StartTerminalTaskRequest {
    /// Shell command passed to `bash -lc`.
    pub command: String,
    /// Working directory under the workspace root.
    pub workdir: String,
    /// Optional timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Safety warning computed when the foreground tool call was accepted.
    pub safety_warning: Option<String>,
}

/// Handle returned when a background terminal task is accepted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminalTaskHandle {
    /// Stable runtime task id.
    pub task_id: Uuid,
    /// Initial task status.
    pub status: RuntimeTaskStatus,
    /// Path storing command output, if already allocated.
    pub output_path: Option<String>,
}

/// Detailed runtime task metadata returned by `GetTask`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminalTaskInfo {
    /// Stable task id.
    pub task_id: Uuid,
    /// Current lifecycle state.
    pub status: RuntimeTaskStatus,
    /// Command summary.
    pub command: String,
    /// Working directory.
    pub workdir: String,
    /// Exit code when available.
    pub exit_code: Option<i32>,
    /// Output file path when available.
    pub output_path: Option<String>,
    /// Best-effort summary of the final outcome.
    pub summary: Option<String>,
}

/// Wait request emitted by the `Wait` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WaitRequest {
    /// Kind of wait target.
    pub kind: WaitKind,
    /// Agent/work/task id depending on `kind`.
    pub id: Uuid,
    /// Desired completion condition.
    pub until: Option<WaitUntil>,
    /// Optional timeout in seconds.
    pub timeout_seconds: Option<u64>,
}

impl WaitRequest {
    /// Validate that the wait request has a sensible target condition.
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(timeout) = self.timeout_seconds
            && timeout == 0
        {
            anyhow::bail!("Wait timeout_seconds must be greater than 0 when provided");
        }

        match (self.kind, self.until) {
            (WaitKind::Agent, Some(WaitUntil::Finished | WaitUntil::Exited)) => {
                anyhow::bail!("Wait(kind=agent) only supports until=idle");
            }
            (WaitKind::Work, Some(WaitUntil::Idle | WaitUntil::Exited)) => {
                anyhow::bail!("Wait(kind=work) only supports until=finished");
            }
            (WaitKind::Task, Some(WaitUntil::Idle | WaitUntil::Finished)) => {
                anyhow::bail!("Wait(kind=task) only supports until=exited");
            }
            _ => Ok(()),
        }
    }
}

/// Request emitted by the `SubAgent` tool.
#[derive(Debug, Clone)]
pub struct SubAgentRequest {
    /// Optional traceable label shown in logs.
    pub label: Option<String>,
    /// Concrete task the child agent should complete.
    pub task: String,
    /// Parent-provided context that will be injected into the child prompt.
    pub context: String,
    /// Optional direct prompt block provided by the parent agent.
    pub prompt: Option<String>,
    /// Optional workspace prompt file such as `DESIGNER.md`.
    pub prompt_file: Option<String>,
    /// Optional skill name used as a persona prompt; defaults to that skill's
    /// `SKILL.md`.
    pub prompt_skill: Option<String>,
    /// Whether the child may directly `Send`.
    pub allow_user_send: bool,
    /// Whether the child may directly `Show`.
    pub allow_user_show: bool,
    /// Whether the child may directly `Ask`.
    pub allow_user_ask: bool,
    /// Whether root may transfer input ownership to this child.
    pub allow_input_transfer_target: bool,
    /// Reuse an existing child agent instead of creating a new one.
    pub existing_agent_id: Option<Uuid>,
    /// Model routing category (e.g., "quick" / "reasoning" / "coding").
    pub category: Option<String>,
}

impl SubAgentRequest {
    /// Validate that the request is well-formed before the daemon executes it.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.task.trim().is_empty() {
            anyhow::bail!("SubAgent task must not be empty");
        }

        if self.context.trim().is_empty() {
            anyhow::bail!("SubAgent context must not be empty");
        }

        if let Some(prompt) = self.prompt.as_deref()
            && prompt.trim().is_empty()
        {
            anyhow::bail!("SubAgent prompt must not be empty when provided");
        }

        if let Some(prompt_file) = self.prompt_file.as_deref()
            && prompt_file.trim().is_empty()
        {
            anyhow::bail!("SubAgent prompt_file must not be empty when provided");
        }

        if let Some(prompt_skill) = self.prompt_skill.as_deref()
            && prompt_skill.trim().is_empty()
        {
            anyhow::bail!("SubAgent prompt_skill must not be empty when provided");
        }

        Ok(())
    }
}

/// Base prompt template selected for one agent runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptProfile {
    /// Main root-agent prompt.
    Root,
    /// Ordinary durable worker / child-agent prompt.
    SubAgent,
    /// Internal background runtime such as memory-refresh.
    Background,
}

/// Shared callbacks injected by the daemon layer.
///
/// The core tool implementation does not know how to talk to the user or how
/// to create child agents. Those behaviors are supplied here by the backend.
#[derive(Clone)]
pub struct ToolRuntime {
    /// Stable current agent id.
    pub agent_id: Uuid,
    /// Parent agent id, if any.
    pub parent_agent_id: Option<Uuid>,
    /// Stable root agent id.
    pub root_agent_id: Uuid,
    /// Human-friendly label for this runtime.
    pub agent_label: String,
    /// Base prompt profile used when composing the system prompt.
    pub prompt_profile: PromptProfile,
    /// Whether this agent is currently the root agent.
    pub is_root: bool,
    /// Whether this agent currently holds user input ownership.
    pub holds_input_ownership: bool,
    /// Whether `FinishWithoutOutput()` is temporarily available for this turn.
    pub allow_finish_without_output: bool,
    /// Whether this agent may directly `Send`.
    pub allow_user_send: bool,
    /// Whether this agent may directly `Show`.
    pub allow_user_show: bool,
    /// Whether this agent may directly `Ask`.
    pub allow_user_ask: bool,
    /// Whether root may transfer input ownership to this agent.
    pub allow_input_transfer_target: bool,
    /// Whether this runtime may trigger backend hot reload.
    pub allow_runtime_reload: bool,
    /// Glob-style tool deny patterns that are always enforced.
    pub denied_tools: Vec<String>,
    /// Glob-style command/skill deny patterns.
    pub denied_commands: Vec<String>,
    /// Command-scoped allowlist patterns currently active for this agent.
    pub allowed_tool_patterns: Vec<String>,
    /// Commands currently active in this agent's durable context.
    pub active_command_invocations: Vec<ActiveCommandInvocation>,
    /// Conditional commands that have already been activated by touched paths.
    pub activated_conditional_commands: Vec<String>,
    /// Effective model override produced by active commands, if any.
    pub model_override: Option<String>,
    /// Effective reasoning-effort override produced by active commands, if any.
    pub effort_override: Option<String>,
    /// Non-blocking "tell the user" channel.
    send_message: SendMessageFn,
    /// Blocking "ask the user" channel.
    ask_question: AskQuestionFn,
    /// Structured file display channel.
    show_file: ShowFileFn,
    /// Nested-agent launcher.
    run_subagent: RunSubAgentFn,
    /// Notify the current parent agent.
    notify_parent: NotifyParentFn,
    /// Send a direct message to one agent.
    message_agent: MessageAgentFn,
    /// Broadcast a message to a scoped set of agents.
    broadcast_agents: BroadcastAgentsFn,
    /// List visible agents under the current root tree.
    list_agents: ListAgentsFn,
    /// Inspect one visible agent.
    get_agent: GetAgentFn,
    /// Transfer input ownership (root only).
    transfer_input: TransferInputFn,
    /// Launch one background terminal task.
    start_terminal_task: StartTerminalTaskFn,
    /// Inspect one runtime task.
    get_task: GetTaskFn,
    /// Trigger one in-process backend hot reload.
    reload_runtime: ReloadRuntimeFn,
}

impl ToolRuntime {
    /// Construct a runtime from concrete callbacks.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent_id: Uuid,
        parent_agent_id: Option<Uuid>,
        root_agent_id: Uuid,
        agent_label: String,
        prompt_profile: PromptProfile,
        is_root: bool,
        holds_input_ownership: bool,
        allow_finish_without_output: bool,
        allow_user_send: bool,
        allow_user_show: bool,
        allow_user_ask: bool,
        allow_input_transfer_target: bool,
        allow_runtime_reload: bool,
        denied_tools: Vec<String>,
        denied_commands: Vec<String>,
        allowed_tool_patterns: Vec<String>,
        active_command_invocations: Vec<ActiveCommandInvocation>,
        activated_conditional_commands: Vec<String>,
        model_override: Option<String>,
        effort_override: Option<String>,
        send_message: SendMessageFn,
        ask_question: AskQuestionFn,
        show_file: ShowFileFn,
        run_subagent: RunSubAgentFn,
        notify_parent: NotifyParentFn,
        message_agent: MessageAgentFn,
        broadcast_agents: BroadcastAgentsFn,
        list_agents: ListAgentsFn,
        get_agent: GetAgentFn,
        transfer_input: TransferInputFn,
        start_terminal_task: StartTerminalTaskFn,
        get_task: GetTaskFn,
        reload_runtime: ReloadRuntimeFn,
    ) -> Self {
        Self {
            agent_id,
            parent_agent_id,
            root_agent_id,
            agent_label,
            prompt_profile,
            is_root,
            holds_input_ownership,
            allow_finish_without_output,
            allow_user_send,
            allow_user_show,
            allow_user_ask,
            allow_input_transfer_target,
            allow_runtime_reload,
            denied_tools,
            denied_commands,
            allowed_tool_patterns,
            active_command_invocations,
            activated_conditional_commands,
            model_override,
            effort_override,
            send_message,
            ask_question,
            show_file,
            run_subagent,
            notify_parent,
            message_agent,
            broadcast_agents,
            list_agents,
            get_agent,
            transfer_input,
            start_terminal_task,
            get_task,
            reload_runtime,
        }
    }

    /// Detached runtime used in unit tests.
    ///
    /// Any attempt to use a runtime-only tool fails with a clear error.
    pub fn detached() -> Self {
        let nil = Uuid::nil();
        let send_message: SendMessageFn = Arc::new(|_message| {
            Box::pin(async { anyhow::bail!("Send runtime is not configured") })
        });
        let ask_question: AskQuestionFn = Arc::new(|_request, _cancel| {
            Box::pin(async { anyhow::bail!("Ask runtime is not configured") })
        });
        let show_file: ShowFileFn =
            Arc::new(|_file| Box::pin(async { anyhow::bail!("Show runtime is not configured") }));
        let run_subagent: RunSubAgentFn = Arc::new(|_request, _cancel| {
            Box::pin(async { anyhow::bail!("SubAgent runtime is not configured") })
        });
        let notify_parent: NotifyParentFn = Arc::new(|_message| {
            Box::pin(async { anyhow::bail!("NotifyParent runtime is not configured") })
        });
        let message_agent: MessageAgentFn = Arc::new(|_request| {
            Box::pin(async { anyhow::bail!("MessageAgent runtime is not configured") })
        });
        let broadcast_agents: BroadcastAgentsFn = Arc::new(|_request| {
            Box::pin(async { anyhow::bail!("BroadcastAgents runtime is not configured") })
        });
        let list_agents: ListAgentsFn = Arc::new(|_request| {
            Box::pin(async { anyhow::bail!("ListAgents runtime is not configured") })
        });
        let get_agent: GetAgentFn = Arc::new(|_agent_id| {
            Box::pin(async { anyhow::bail!("GetAgent runtime is not configured") })
        });
        let transfer_input: TransferInputFn = Arc::new(|_request| {
            Box::pin(async { anyhow::bail!("TransferInput runtime is not configured") })
        });
        let start_terminal_task: StartTerminalTaskFn = Arc::new(|_request, _cancel| {
            Box::pin(async { anyhow::bail!("Background Bash runtime is not configured") })
        });
        let get_task: GetTaskFn = Arc::new(|_task_id| {
            Box::pin(async { anyhow::bail!("GetTask runtime is not configured") })
        });
        let reload_runtime: ReloadRuntimeFn =
            Arc::new(|| Box::pin(async { anyhow::bail!("Reload runtime is not configured") }));

        Self::new(
            nil,
            None,
            nil,
            "detached".to_string(),
            PromptProfile::Root,
            true,
            true,
            false,
            true,
            true,
            true,
            false,
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            send_message,
            ask_question,
            show_file,
            run_subagent,
            notify_parent,
            message_agent,
            broadcast_agents,
            list_agents,
            get_agent,
            transfer_input,
            start_terminal_task,
            get_task,
            reload_runtime,
        )
    }

    /// Invoke the `Send` callback.
    pub async fn send_message(&self, message: String) -> anyhow::Result<()> {
        (self.send_message)(message).await
    }

    /// Invoke the `Ask` callback.
    pub async fn ask_question(
        &self,
        request: AskRequest,
        cancel: CancelToken,
    ) -> anyhow::Result<UserQuestionAnswer> {
        (self.ask_question)(request, cancel).await
    }

    /// Invoke the `Show` callback.
    pub async fn show_file(&self, file: UserVisibleFile) -> anyhow::Result<()> {
        (self.show_file)(file).await
    }

    /// Invoke the `SubAgent` callback.
    pub async fn run_subagent(
        &self,
        request: SubAgentRequest,
        cancel: CancelToken,
    ) -> anyhow::Result<SubAgentHandle> {
        (self.run_subagent)(request, cancel).await
    }

    /// Invoke the `NotifyParent` callback.
    pub async fn notify_parent(&self, message: String) -> anyhow::Result<AgentMessageReceipt> {
        (self.notify_parent)(message).await
    }

    /// Invoke the `MessageAgent` callback.
    pub async fn message_agent(
        &self,
        request: AgentMessageRequest,
    ) -> anyhow::Result<AgentMessageReceipt> {
        (self.message_agent)(request).await
    }

    /// Invoke the `BroadcastAgents` callback.
    pub async fn broadcast_agents(
        &self,
        request: BroadcastAgentsRequest,
    ) -> anyhow::Result<BroadcastReceipt> {
        (self.broadcast_agents)(request).await
    }

    /// Invoke the `ListAgents` callback.
    pub async fn list_agents(&self, request: ListAgentsRequest) -> anyhow::Result<Vec<AgentInfo>> {
        (self.list_agents)(request).await
    }

    /// Invoke the `GetAgent` callback.
    pub async fn get_agent(&self, agent_id: Uuid) -> anyhow::Result<AgentInfo> {
        (self.get_agent)(agent_id).await
    }

    /// Invoke the `TransferInput` callback.
    pub async fn transfer_input(
        &self,
        request: TransferInputRequest,
    ) -> anyhow::Result<TransferInputReceipt> {
        (self.transfer_input)(request).await
    }

    /// Start one background terminal task.
    pub async fn start_terminal_task(
        &self,
        request: StartTerminalTaskRequest,
        cancel: CancelToken,
    ) -> anyhow::Result<TerminalTaskHandle> {
        (self.start_terminal_task)(request, cancel).await
    }

    /// Inspect one runtime task.
    pub async fn get_task(&self, task_id: Uuid) -> anyhow::Result<TerminalTaskInfo> {
        (self.get_task)(task_id).await
    }

    /// Trigger one in-process backend hot reload.
    pub async fn reload_runtime(&self) -> anyhow::Result<ReloadRuntimeReceipt> {
        (self.reload_runtime)().await
    }
}

/// Shared context used by all tool invocations.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Workspace root directory; all file activity is constrained to this tree.
    pub workspace_root: PathBuf,
    /// Registry of discovered skills.
    pub skills: Arc<SkillRegistry>,
    /// Multi-backend search configuration.
    pub search_config: crate::config::SearchConfig,
}

impl ToolContext {
    /// Create a new tool context.
    pub fn new(workspace_root: PathBuf, skills: Arc<SkillRegistry>) -> anyhow::Result<Self> {
        let workspace_root = std::fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root: {}",
                workspace_root.display()
            )
        })?;

        Ok(Self {
            workspace_root,
            skills,
            search_config: crate::config::SearchConfig::default(),
        })
    }

    /// Builder-style setter for search configuration.
    pub fn with_search_config(mut self, config: crate::config::SearchConfig) -> Self {
        self.search_config = config;
        self
    }


    /// Resolve a user-provided path into an absolute path under `workspace_root`.
    ///
    /// This must work for both existing and not-yet-existing paths.
    pub fn resolve_under_workspace(&self, raw: &str) -> anyhow::Result<PathBuf> {
        let raw_path = PathBuf::from(raw);
        let candidate = if raw_path.is_absolute() {
            raw_path
        } else {
            self.workspace_root.join(raw_path)
        };

        let mut ancestor = candidate.as_path();
        let mut suffix: Vec<OsString> = Vec::new();
        while !ancestor.exists() {
            let Some(name) = ancestor.file_name() else {
                anyhow::bail!("Invalid path: {raw}");
            };
            suffix.push(name.to_os_string());
            let Some(parent) = ancestor.parent() else {
                anyhow::bail!("Invalid path: {raw}");
            };
            ancestor = parent;
        }

        let mut canonical = std::fs::canonicalize(ancestor).with_context(|| {
            format!(
                "Failed to canonicalize existing ancestor: {}",
                ancestor.display()
            )
        })?;
        for component in suffix.into_iter().rev() {
            canonical.push(component);
        }

        if !canonical.starts_with(&self.workspace_root) {
            anyhow::bail!(
                "Path escapes workspace root (root: {}, requested: {})",
                self.workspace_root.display(),
                canonical.display()
            );
        }

        Ok(canonical)
    }
}

/// Per-agent-session tool state.
///
/// The important rule here is the `Edit` precondition:
/// - a file must be `Read` before it is `Edit`ed
/// - after a successful `Edit`, the "fresh read" marker is cleared again
///   because the file content has changed
#[derive(Debug, Clone, Default)]
pub struct ToolSession {
    /// Canonical file paths that are currently eligible for `Edit`.
    readable_for_edit: HashMap<PathBuf, Option<FileReadVersion>>,
    /// Commands that remain active for future turns.
    active_command_invocations: Vec<ActiveCommandInvocation>,
    /// Conditional commands already unlocked by touched paths.
    activated_conditional_commands: BTreeSet<String>,
    /// Workspace paths touched during the current quantum.
    touched_paths: BTreeSet<PathBuf>,
}

/// Lightweight version fingerprint captured when a file is read for later
/// `Edit` safety checks.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileReadVersion {
    /// Byte length of the file at read time.
    bytes: usize,
    /// Fast content hash used to detect silent external modifications.
    hash: [u8; 32],
}

impl FileReadVersion {
    /// Build one deterministic fingerprint from UTF-8 content.
    fn from_content(content: &str) -> Self {
        use sha2::Digest as _;

        let mut hasher = sha2::Sha256::new();
        hasher.update(content.as_bytes());
        let hash: [u8; 32] = hasher.finalize().into();
        Self {
            bytes: content.len(),
            hash,
        }
    }
}

impl ToolSession {
    /// Restore one tool session from persisted durable state.
    pub fn from_state(
        paths: impl IntoIterator<Item = PathBuf>,
        active_command_invocations: impl IntoIterator<Item = ActiveCommandInvocation>,
        activated_conditional_commands: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            readable_for_edit: paths.into_iter().map(|path| (path, None)).collect(),
            active_command_invocations: active_command_invocations.into_iter().collect(),
            activated_conditional_commands: activated_conditional_commands.into_iter().collect(),
            touched_paths: BTreeSet::new(),
        }
    }

    /// Restore one tool session from a persisted list of canonical paths.
    ///
    /// This helper keeps older unit tests concise.
    pub fn from_readable_paths(paths: impl IntoIterator<Item = PathBuf>) -> Self {
        Self::from_state(
            paths,
            Vec::<ActiveCommandInvocation>::new(),
            Vec::<String>::new(),
        )
    }

    /// Export the current "freshly read" set for persistence.
    pub fn readable_paths(&self) -> Vec<PathBuf> {
        self.readable_for_edit.keys().cloned().collect()
    }

    /// Export the active command reminder set for persistence.
    pub fn active_command_invocations(&self) -> Vec<ActiveCommandInvocation> {
        self.active_command_invocations.clone()
    }

    /// Export the activated conditional command names for persistence.
    pub fn activated_conditional_commands(&self) -> Vec<String> {
        self.activated_conditional_commands
            .iter()
            .cloned()
            .collect()
    }

    /// Export the paths touched during the just-finished quantum.
    pub fn touched_paths(&self) -> Vec<PathBuf> {
        self.touched_paths.iter().cloned().collect()
    }

    /// Mark that a file has just been read in this session.
    fn note_read(&mut self, path: PathBuf, version: FileReadVersion) {
        self.readable_for_edit.insert(path.clone(), Some(version));
        self.note_touched_path(path);
    }

    /// Enforce the "must read before edit" rule.
    fn require_fresh_read(&self, path: &Path, current_content: &str) -> anyhow::Result<()> {
        let Some(version) = self.readable_for_edit.get(path) else {
            anyhow::bail!(
                "Edit is not allowed until the file has been Read in this session: {}",
                path.display()
            );
        };

        let Some(version) = version else {
            anyhow::bail!(
                "Edit requires a fresh re-Read after restart because the previous file version metadata is unavailable: {}",
                path.display()
            );
        };

        let current = FileReadVersion::from_content(current_content);
        if version != &current {
            anyhow::bail!(
                "Edit requires a fresh re-Read because the file changed since it was last Read in this session: {}",
                path.display()
            );
        }

        Ok(())
    }

    /// After an edit, the caller must re-read the file before the next edit.
    fn invalidate_after_edit(&mut self, path: &Path) {
        self.readable_for_edit.remove(path);
    }

    /// Record that one workspace path participated in the current quantum.
    fn note_touched_path(&mut self, path: PathBuf) {
        self.touched_paths.insert(path);
    }

    /// Activate or refresh one command reminder.
    fn note_command_invocation(&mut self, invocation: ActiveCommandInvocation) {
        self.active_command_invocations
            .retain(|existing| existing.name != invocation.name);
        self.active_command_invocations.push(invocation);
    }

    /// Mark a conditional command as activated for future prompt visibility.
    pub fn activate_conditional_command(&mut self, name: impl Into<String>) {
        self.activated_conditional_commands.insert(name.into());
    }
}

/// Concrete tool executor.
#[derive(Clone)]
pub struct ToolExecutor {
    /// Shared tool context.
    pub ctx: ToolContext,
    /// Optional external MCP registry.
    mcp_registry: Option<Arc<McpRegistry>>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("ctx", &self.ctx)
            .field("has_mcp_registry", &self.mcp_registry.is_some())
            .finish()
    }
}

/// Internal result type for `ImageAnalyze` – decoupled from `ToolExecutionResult`.
struct ImageAnalyzeOutput {
    data_url: String,
    media_type: String,
    prompt: Option<String>,
}

impl ToolExecutor {
    /// Create a new executor.
    pub fn new(ctx: ToolContext, mcp_registry: Option<Arc<McpRegistry>>) -> Self {
        Self { ctx, mcp_registry }
    }

    /// Tool definitions advertised to the model.
    pub fn tool_definitions(&self, runtime: &ToolRuntime) -> Vec<ToolDefinition> {
        let mut definitions = vec![
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Read".to_string(),
                    description: "Read a text file inside the workspace. Supports partial reads via offset/limit and non-UTF-8 encodings."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path, relative to the workspace root or absolute under it."
                            },
                            "offset": {
                                "type": "integer",
                                "description": "1-based starting line number. Defaults to 1 (first line)."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of lines to return. Defaults to reading until end of file."
                            },
                            "encoding": {
                                "type": "string",
                                "description": "Character encoding, e.g. \"utf-8\", \"gbk\", \"gb2312\", \"shift_jis\", \"euc-kr\". Defaults to \"utf-8\"."
                            }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Write".to_string(),
                    description: "Create a new UTF-8 text file. Refuses to overwrite existing files."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "New file path, relative to the workspace root or absolute under it."
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete file content to create."
                            }
                        },
                        "required": ["path", "content"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Edit".to_string(),
                    description: "Edit an existing UTF-8 text file that has already been Read in this session."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Existing file path, relative to the workspace root or absolute under it."
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to replace. Must be non-empty."
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Replacement text."
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "If true, replace all matches. If false or omitted, exactly one match must exist."
                            }
                        },
                        "required": ["path", "old_text", "new_text"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Bash".to_string(),
                    description: "Run a command via Git Bash (`bash -lc`) inside the workspace."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command string passed to `bash -lc`."
                            },
                            "workdir": {
                                "type": "string",
                                "description": "Optional working directory, relative to the workspace root or absolute under it."
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Optional timeout in seconds. Defaults to 300 and is capped at 1800."
                            },
                            "run_in_background": {
                                "type": "boolean",
                                "description": "If true, start the command as a background runtime task and return its task_id instead of blocking for completion."
                            }
                        },
                        "required": ["command"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Fetch".to_string(),
                    description: "Fetch a public HTTP(S) URL in SA's read-only safety model and return the response body."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "HTTP or HTTPS URL to request."
                            },
                            "method": {
                                "type": "string",
                                "description": "HTTP method. Only `GET` and `HEAD` are supported. Defaults to `GET`."
                            },
                            "headers": {
                                "type": "object",
                                "description": "Optional request headers as string key/value pairs. Sensitive headers such as `Authorization` and `Cookie` are blocked."
                            },
                            "body": {
                                "type": "string",
                                "description": "Not currently supported in SA's read-only Fetch safety model."
                            },
                            "max_bytes": {
                                "type": "integer",
                                "description": "Optional response size cap in bytes. Defaults to 200000 and is capped at 200000."
                            }
                        },
                        "required": ["url"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Search".to_string(),
                    description: "Search the web for relevant pages, returning titles, snippets, and URLs."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return. Defaults to 5 and is capped at 10."
                            }
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Send".to_string(),
                    description: "Send a user-facing message without waiting for a reply."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message content shown to the user."
                            }
                        },
                        "required": ["message"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "MemorySearch".to_string(),
                    description:
                        "Search `MEMORY.md`, `memory.md`, `memory/*.md`, and `memory/topics/**/*.md` for prior decisions, dates, preferences, or long-term context. Dream audit files are excluded from normal search."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for memory recall."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Optional maximum number of results. Defaults to 5 and is capped at 10."
                            },
                            "min_score": {
                                "type": "number",
                                "description": "Optional minimum lexical score threshold."
                            }
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "MemoryGet".to_string(),
                    description:
                        "Read one allowed memory Markdown file (`MEMORY.md`, `memory.md`, `memory/*.md`, `memory/topics/**/*.md`, or explicit dream audit files under `memory/dreams/**/*.md`) with an optional line range."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Memory file path. Must be `MEMORY.md`, `memory.md`, a daily note under `memory/`, a topic file under `memory/topics/`, or an audit file under `memory/dreams/`."
                            },
                            "from": {
                                "type": "integer",
                                "description": "Optional 1-based starting line."
                            },
                            "lines": {
                                "type": "integer",
                                "description": "Optional number of lines to read."
                            }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Show".to_string(),
                    description: "Display an existing workspace file to the user through the frontend."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Existing file path, relative to the workspace root or absolute under it."
                            },
                            "title": {
                                "type": "string",
                                "description": "Optional user-facing title shown above the file content."
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Required concise internal description of what the shown file contains or why it is being shown. This is stored in interaction history and may appear in raw event streams."
                            }
                        },
                        "required": ["path", "prompt"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "GetInteractionEntry".to_string(),
                    description:
                        "Read one entry from the durable interaction log by id with progressive disclosure. Raw log path: `interactions/history.jsonl`; grep it first, then use this tool for bounded retrieval."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Interaction entry UUID."
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["summary", "full", "slice"],
                                "description": "Disclosure mode. Defaults to `summary`."
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Optional character offset used by `slice` mode."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Optional character count used by `slice` mode."
                            }
                        },
                        "required": ["id"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Ask".to_string(),
                    description: "Ask the user a structured question and wait for the answer."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Human-readable question prompt."
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["single_choice", "multi_choice", "text"],
                                "description": "How the answer should be collected."
                            },
                            "options": {
                                "type": "array",
                                "description": "Selectable options for choice-based questions.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": { "type": "string" },
                                        "label": { "type": "string" },
                                        "description": { "type": "string" }
                                    },
                                    "required": ["id", "label"]
                                }
                            },
                            "allow_free_text": {
                                "type": "boolean",
                                "description": "Whether the user may additionally provide free text."
                            }
                        },
                        "required": ["prompt", "mode"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Finish".to_string(),
                    description: "Explicitly finish the current work and return this agent to idle."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why the current work is complete."
                            },
                            "result": {
                                "type": "string",
                                "description": "Result summary for audit/recovery and parent-agent coordination."
                            }
                        },
                        "required": ["reason", "result"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Wait".to_string(),
                    description: "Suspend the current work until an agent, work item, or background task reaches the desired state."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["agent", "work", "task"]
                            },
                            "id": {
                                "type": "string",
                                "description": "UUID of the target agent/work/task."
                            },
                            "until": {
                                "type": "string",
                                "enum": ["idle", "finished", "exited"],
                                "description": "Optional target condition. Defaults depend on `kind`."
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Optional timeout in seconds."
                            }
                        },
                        "required": ["kind", "id"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "UpdateWorkCheckpoint".to_string(),
                    description: "保存当前工作的关键信息到持久化检查点，用于超长任务中断后的恢复。传入关键摘要信息。"
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "key_info": {
                                "type": "string",
                                "description": "要保存的关键信息摘要（当前进度、已完成步骤、待办事项等）"
                            }
                        },
                        "required": ["key_info"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "SearchSkill".to_string(),
                    description: "搜索已注册的技能。传入关键词或自然语言描述，返回匹配的技能名称和描述。用于帮助发现和选择正确的技能。例如 query: PDF处理, 生成表格等。"
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词或描述（如 PDF处理、生成表格 等）"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "最多返回的结果数，默认 10"
                            }
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Skill".to_string(),
                    description:
                        "Invoke one registered command/skill, or read one local skill-relative file without exposing the real host path."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["invoke", "read"],
                                "description": "Whether to invoke the command or read one local skill file. Defaults to `invoke`."
                            },
                            "name": {
                                "type": "string",
                                "description": "Command/skill name from the prompt's command metadata list."
                            },
                            "args": {
                                "type": "string",
                                "description": "Optional raw argument string used when `action = invoke`."
                            },
                            "path": {
                                "type": "string",
                                "description": "Optional skill-relative path used when `action = read`. Defaults to `SKILL.md`."
                            }
                        },
                        "required": ["name"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "ListMcpResources".to_string(),
                    description: "List resources exposed by connected MCP servers.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "server": {
                                "type": "string",
                                "description": "Optional server name filter."
                            }
                        }
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "ReadMcpResource".to_string(),
                    description: "Read one resource from a connected MCP server by URI.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "server": {
                                "type": "string",
                                "description": "MCP server name."
                            },
                            "uri": {
                                "type": "string",
                                "description": "Resource URI returned by `ListMcpResources`."
                            }
                        },
                        "required": ["server", "uri"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "SubAgent".to_string(),
                    description: "Launch a nested sub-agent with explicit parent-provided context."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Optional short label used for tracing in logs."
                            },
                            "task": {
                                "type": "string",
                                "description": "Concrete task the child agent should complete."
                            },
                            "context": {
                                "type": "string",
                                "description": "Parent-provided context, constraints, and findings for the child agent."
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Optional direct prompt block injected into the child agent's prompt."
                            },
                            "prompt_file": {
                                "type": "string",
                                "description": "Optional workspace prompt file such as `DESIGNER.md`."
                            },
                            "prompt_skill": {
                                "type": "string",
                                "description": "Optional skill name used as the child agent's persona prompt. Defaults to that skill's `SKILL.md`."
                            },
                            "allow_user_send": {
                                "type": "boolean",
                                "description": "Whether the child may directly Send."
                            },
                            "allow_user_show": {
                                "type": "boolean",
                                "description": "Whether the child may directly Show."
                            },
                            "allow_user_ask": {
                                "type": "boolean",
                                "description": "Whether the child may directly Ask."
                            },
                            "allow_input_transfer_target": {
                                "type": "boolean",
                                "description": "Whether root may transfer input ownership to this child."
                            },
                            "existing_agent_id": {
                                "type": "string",
                                "description": "Optional existing child agent id to reuse."
                            }
                        },
                        "required": ["task", "context"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "NotifyParent".to_string(),
                    description: "Send a direct single message to the current parent agent."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message content delivered to the parent agent."
                            }
                        },
                        "required": ["message"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "MessageAgent".to_string(),
                    description: "Send a direct one-way message to another agent in the same root tree."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "target_agent_id": {
                                "type": "string",
                                "description": "Target agent UUID."
                            },
                            "message": {
                                "type": "string",
                                "description": "Message content delivered to the target agent."
                            }
                        },
                        "required": ["target_agent_id", "message"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "BroadcastAgents".to_string(),
                    description: "Broadcast a one-way message to a scoped set of agents under the same root."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "enum": ["children", "descendants", "siblings", "all_under_root"]
                            },
                            "message": {
                                "type": "string",
                                "description": "Broadcast message content."
                            }
                        },
                        "required": ["message"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "ListAgents".to_string(),
                    description: "List visible agents under the current root tree."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "enum": ["children", "descendants", "siblings", "all_under_root"]
                            }
                        }
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "GetAgent".to_string(),
                    description: "Inspect one visible agent by UUID.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent UUID."
                            }
                        },
                        "required": ["agent_id"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "GetTask".to_string(),
                    description: "Inspect one background runtime task such as a Bash task."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task UUID."
                            }
                        },
                        "required": ["task_id"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "TransferInput".to_string(),
                    description: "Transfer free-form user input ownership to another agent, or return it to the root. Ownership stays with the target until the root explicitly transfers it again or returns it to the root. If you want a child agent to directly reply to the classmate, first transfer input to that child."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "target_agent_id": {
                                "type": ["string", "null"],
                                "description": "Target agent UUID, or null to return input ownership to the root."
                            }
                        }
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Reload".to_string(),
                    description: "Hot-reload SA runtime state from disk without restarting the process or dropping WS connections. This reloads config-backed runtime state such as model settings, skills, permissions, MCP, and Agents.md path when the change is safe to apply in place."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "ImageAnalyze".to_string(),
                    description: "Read an image file and return its base64-encoded data URL so the agent can include it in a multimodal message for analysis."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Image file path, relative to the workspace root or absolute under it."
                            }
                        },
                        "required": ["path"]
                    }),
                },
            },
        ];

        if runtime.allow_finish_without_output {
            definitions.push(ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "FinishWithoutOutput".to_string(),
                    description: "Confirm that the current work may finish without extra outward output or extra explicit parent messaging."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
            });
        }

        if let Some(registry) = &self.mcp_registry {
            definitions.extend(registry.tool_definitions());
        }

        definitions
            .into_iter()
            .filter(|definition| self.tool_is_model_visible(runtime, &definition.function.name))
            .collect()
    }

    /// Execute one tool call.
    pub async fn execute(
        &self,
        session: &mut ToolSession,
        runtime: &ToolRuntime,
        name: &str,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        self.ensure_tool_allowed(runtime, name, &args)?;

        match name {
            "Read" => self
                .read(session, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Write" => self
                .write(session, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Edit" => self
                .edit(session, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Bash" => self
                .bash(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Fetch" => self
                .fetch(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Search" => self
                .search(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "MemorySearch" => self
                .memory_search(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "MemoryGet" => self
                .memory_get(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Send" => self
                .send(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Show" => self
                .show(session, runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "GetInteractionEntry" => self
                .get_interaction_entry(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Ask" => self.ask(runtime, args, cancel).await,
            "UpdateWorkCheckpoint" => self
                .update_work_checkpoint(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "SearchSkill" => self
                .search_skill(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Skill" => self
                .skill(session, runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "ListMcpResources" => self
                .list_mcp_resources(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "ReadMcpResource" => self
                .read_mcp_resource(args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "SubAgent" => self
                .subagent(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Finish" => self.finish(args, cancel).await,
            "FinishWithoutOutput" => self.finish_without_output(cancel).await,
            "NotifyParent" => self
                .notify_parent(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "MessageAgent" => self
                .message_agent(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "BroadcastAgents" => self
                .broadcast_agents(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "ListAgents" => self
                .list_agents(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "GetAgent" => self
                .get_agent(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "TransferInput" => self
                .transfer_input(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "Reload" => self
                .reload(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            "ImageAnalyze" => {
                match self.image_analyze(session, args, cancel).await {
                    Ok(out) => Ok(ToolExecutionResult::ImagePayload {
                        data_url: out.data_url,
                        media_type: out.media_type,
                        prompt: out.prompt,
                    }),
                    Err(e) => Err(e),
                }
            }
            "Wait" => self.wait(args, cancel).await,
            "GetTask" => self
                .get_task(runtime, args, cancel)
                .await
                .map(ToolExecutionResult::Observation),
            _ => {
                if let Some(registry) = &self.mcp_registry {
                    if registry.has_tool(name) {
                        return registry
                            .call_tool(name, args)
                            .await
                            .map(ToolExecutionResult::Observation);
                    }
                }
                anyhow::bail!("Unknown tool: {name}")
            }
        }
    }

    /// Whether a tool should be visible in the advertised model tool list.
    fn tool_is_model_visible(&self, runtime: &ToolRuntime, tool_name: &str) -> bool {
        match tool_name {
            "Send" if !runtime.allow_user_send => return false,
            "Show" if !runtime.allow_user_show => return false,
            "Ask" if !runtime.allow_user_ask => return false,
            "NotifyParent" if runtime.parent_agent_id.is_none() => return false,
            "TransferInput" if !runtime.is_root => return false,
            "Reload" if !runtime.allow_runtime_reload => return false,
            _ => {}
        }

        if matches_tool_patterns(tool_name, None, &runtime.denied_tools) {
            return false;
        }

        if runtime.allowed_tool_patterns.is_empty() {
            return true;
        }

        runtime
            .allowed_tool_patterns
            .iter()
            .map(|pattern| pattern.trim())
            .filter(|pattern| !pattern.is_empty())
            .any(|pattern| pattern_allows_tool_visibility(pattern, tool_name))
    }

    /// Enforce the current runtime permission policy for one concrete tool call.
    fn ensure_tool_allowed(
        &self,
        runtime: &ToolRuntime,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let bash_command = extract_bash_command(tool_name, args);
        if matches_tool_patterns(tool_name, bash_command.as_deref(), &runtime.denied_tools) {
            anyhow::bail!("Tool `{tool_name}` is denied by the current permission policy");
        }

        if runtime.allowed_tool_patterns.is_empty() {
            return Ok(());
        }

        if matches_tool_patterns(
            tool_name,
            bash_command.as_deref(),
            &runtime.allowed_tool_patterns,
        ) {
            return Ok(());
        }

        anyhow::bail!(
            "Tool `{tool_name}` is not allowed by the current command scope; allowed patterns: {}",
            runtime.allowed_tool_patterns.join(", ")
        );
    }

    /// `Read`: read a UTF-8 text file and mark it as eligible for `Edit`.
    async fn read(
        &self,
        session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Read cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            offset: Option<u64>,
            limit: Option<u64>,
            encoding: Option<String>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Read")?;
        validate_tool_path_input(&args.path, PathOperation::Read)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Read)?;

        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "Read requires a file path, not a directory: {}",
                path.display()
            );
        }
        if meta.len() > MAX_READ_FILE_BYTES {
            anyhow::bail!(
                "File is too large to Read via tool ({} bytes): {}",
                meta.len(),
                path.display()
            );
        }

        // Determine encoding; default to UTF-8.
        let encoding_label = args.encoding.as_deref().unwrap_or("utf-8");
        let content = if encoding_label.eq_ignore_ascii_case("utf-8") {
            tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("Failed to read file: {}", path.display()))?
        } else {
            let raw_bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("Failed to read file: {}", path.display()))?;
            let encoding = encoding_rs::Encoding::for_label(encoding_label.as_bytes())
                .ok_or_else(|| anyhow::anyhow!("Unknown encoding: \"{encoding_label}\""))?;
            let (cow, _encoding_used, had_errors) = encoding.decode(&raw_bytes);
            if had_errors {
                tracing::warn!(
                    "Read {}: some bytes could not be decoded with encoding \"{}\"",
                    path.display(),
                    encoding_label
                );
            }
            cow.into_owned()
        };

        // Apply offset / limit (1-based line numbers).
        let is_partial = args.offset.is_some() || args.limit.is_some();
        let (content, line_range) = if is_partial {
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len() as u64;

            let offset = args.offset.unwrap_or(1).max(1);
            if offset > total_lines {
                anyhow::bail!(
                    "Read offset {} exceeds total line count {} for: {}",
                    offset,
                    total_lines,
                    path.display()
                );
            }

            let start = (offset - 1) as usize;
            let remaining = total_lines - offset + 1;
            let limit = args.limit.unwrap_or(remaining).min(remaining) as usize;
            let end = (start + limit).min(lines.len());

            let sliced: String = lines[start..end].join("\n");
            (sliced, (offset, end as u64))
        } else {
            // Full read: compute line range for the response.
            let total = content.lines().count() as u64;
            (content, (1, total))
        };

        // Only record a read version for full (non-partial) reads so that Edit
        // safety checks remain correct — partial content would produce a
        // mismatching SHA-256 hash.
        if !is_partial {
            session.note_read(path.clone(), FileReadVersion::from_content(&content));
        }

        Ok(serde_json::json!({
            "path": path.display().to_string(),
            "bytes": content.len(),
            "lines": [line_range.0, line_range.1],
            "content": content,
        })
        .to_string())
    }

    /// `ImageAnalyze`: read an image file and return its base64-encoded data URL
    /// along with enough metadata for the agent runtime to inject it as a
    /// multimodal user message.
    async fn image_analyze(
        &self,
        _session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ImageAnalyzeOutput> {
        if cancel.is_cancelled() {
            anyhow::bail!("ImageAnalyze cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            prompt: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ImageAnalyze")?;
        validate_tool_path_input(&args.path, PathOperation::Read)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Read)?;

        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "ImageAnalyze requires a file path, not a directory: {}",
                path.display()
            );
        }

        let raw_bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        // Infer MIME type from extension.
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        let media_type: String = match ext.as_deref() {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        }
        .to_string();

        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw_bytes);
        let data_url = format!("data:{media_type};base64,{b64}");

        Ok(ImageAnalyzeOutput {
            data_url,
            media_type,
            prompt: args.prompt,
        })
    }

    /// `Write`: create a new file and refuse to overwrite an existing one.
    async fn write(
        &self,
        session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Write cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            content: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Write")?;
        validate_tool_path_input(&args.path, PathOperation::Write)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Write)?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create parent directories: {}", parent.display())
            })?;
        }

        {
            use tokio::io::AsyncWriteExt as _;

            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .await
                .with_context(|| {
                    format!(
                        "Write refuses to overwrite an existing path: {}",
                        path.display()
                    )
                })?;
            file.write_all(args.content.as_bytes())
                .await
                .with_context(|| format!("Failed to write file: {}", path.display()))?;
            file.flush()
                .await
                .with_context(|| format!("Failed to flush file: {}", path.display()))?;
        }
        session.note_touched_path(path.clone());

        Ok(serde_json::json!({
            "created": true,
            "path": path.display().to_string(),
            "bytes": args.content.len(),
        })
        .to_string())
    }

    /// `UpdateWorkCheckpoint`: persist a checkpoint for long-running task recovery.
    async fn update_work_checkpoint(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("UpdateWorkCheckpoint cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            key_info: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for UpdateWorkCheckpoint")?;

        let runtime_dir = self.ctx.workspace_root.join("runtime");
        tokio::fs::create_dir_all(&runtime_dir)
            .await
            .context("Failed to create runtime directory for checkpoint")?;

        let checkpoint_path = runtime_dir.join("checkpoint.md");
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let content = format!(
            "# Checkpoint ({timestamp})\n\n{key_info}\n",
            key_info = args.key_info.trim()
        );

        tokio::fs::write(&checkpoint_path, content.as_bytes())
            .await
            .with_context(|| {
                format!(
                    "Failed to write checkpoint: {}",
                    checkpoint_path.display()
                )
            })?;

        Ok(serde_json::json!({
            "checkpoint_written": true,
            "path": checkpoint_path.display().to_string(),
            "bytes": content.len(),
        })
        .to_string())
    }

    /// `SearchSkill`: search registered skills by keyword query.
    async fn search_skill(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("SearchSkill cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            #[serde(default = "default_limit")]
            limit: usize,
        }

        fn default_limit() -> usize {
            10
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for SearchSkill")?;
        let limit = args.limit.min(20);

        let index = crate::skill_search::SkillIndex::from_arc(&self.ctx.skills);
        let results = index.search(&args.query, limit);

        Ok(serde_json::json!({
            "query": args.query,
            "total_matches": results.len(),
            "results": results,
        })
        .to_string())
    }

    /// `Edit`: replace text in an existing file that was previously `Read`.
    async fn edit(
        &self,
        session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Edit cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            old_text: String,
            new_text: String,
            replace_all: Option<bool>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Edit")?;
        let replace_all = args.replace_all.unwrap_or(false);
        validate_tool_path_input(&args.path, PathOperation::Edit)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Edit)?;

        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file for edit: {}", path.display()))?;
        session.require_fresh_read(&path, &content)?;

        if args.old_text.is_empty() {
            anyhow::bail!("Edit old_text must be non-empty");
        }

        let match_count = content.matches(&args.old_text).count();
        if match_count == 0 {
            anyhow::bail!("Edit could not find the target text in {}", path.display());
        }

        let new_content = if replace_all {
            content.replace(&args.old_text, &args.new_text)
        } else {
            if match_count != 1 {
                anyhow::bail!(
                    "Edit expected exactly one match in {}, but found {}. Use replace_all=true or provide a more specific old_text.",
                    path.display(),
                    match_count
                );
            }
            content.replacen(&args.old_text, &args.new_text, 1)
        };

        tokio::fs::write(&path, new_content.as_bytes())
            .await
            .with_context(|| format!("Failed to write edited file: {}", path.display()))?;

        // 5.7 Atomic replace verification: confirm new_text is actually in the file.
        let verify = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read back file for verification: {}", path.display()))?;
        if !verify.contains(&args.new_text) {
            anyhow::bail!(
                "Edit 写入验证失败：替换后文件中未找到 new_text，文件可能未正确更新: {}",
                path.display()
            );
        }

        session.invalidate_after_edit(&path);
        session.note_touched_path(path.clone());

        Ok(serde_json::json!({
            "edited": true,
            "path": path.display().to_string(),
            "replace_all": replace_all,
            "matches_replaced": if replace_all { match_count } else { 1 },
            "new_bytes": new_content.len(),
        })
        .to_string())
    }

    /// `Bash`: run a command through Git Bash.
    async fn bash(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            command: String,
            workdir: Option<String>,
            timeout_seconds: Option<u64>,
            #[serde(default)]
            run_in_background: bool,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Bash")?;
        let workdir = match args.workdir.as_deref() {
            Some(raw) => self.ctx.resolve_under_workspace(raw)?,
            None => self.ctx.workspace_root.clone(),
        };

        let timeout = args
            .timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_BASH_TIMEOUT)
            .min(MAX_BASH_TIMEOUT);

        let safety_warning =
            match validate_bash_command_safety(&args.command, &self.ctx.workspace_root, &workdir) {
                BashSafetyDecision::Allow { warning } => warning,
                BashSafetyDecision::Block { reason } => {
                    anyhow::bail!("Bash blocked by SA safety checks: {reason}");
                }
            };

        if args.run_in_background {
            let handle = runtime
                .start_terminal_task(
                    StartTerminalTaskRequest {
                        command: args.command.clone(),
                        workdir: workdir.display().to_string(),
                        timeout_seconds: args.timeout_seconds,
                        safety_warning: safety_warning.clone(),
                    },
                    cancel.clone(),
                )
                .await?;

            return Ok(serde_json::json!({
                "started": true,
                "task_id": handle.task_id,
                "status": handle.status,
                "output_path": handle.output_path,
                "workdir": workdir.display().to_string(),
                "safety_warning": safety_warning,
            })
            .to_string());
        }

        let programs = candidate_bash_programs();
        let mut last_not_found: Option<anyhow::Error> = None;

        for program in programs {
            if cancel.is_cancelled() {
                anyhow::bail!("Bash cancelled");
            }

            let mut cmd = tokio::process::Command::new(&program);
            cmd.arg("-lc");
            cmd.arg(&args.command);
            cmd.current_dir(&workdir);
            cmd.kill_on_drop(true);

            let result = tokio::select! {
                _ = cancel.cancelled() => {
                    anyhow::bail!("Bash cancelled");
                }
                output = tokio::time::timeout(timeout, cmd.output()) => {
                    output.context("Bash command timed out")?
                }
            };

            match result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let exit_code = output.status.code().unwrap_or(-1);

                    return Ok(serde_json::json!({
                        "program": program.display().to_string(),
                        "workdir": workdir.display().to_string(),
                        "exit_code": exit_code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "safety_warning": safety_warning,
                    })
                    .to_string());
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    last_not_found = Some(anyhow::Error::new(err).context(format!(
                        "Bash executable not found at {}",
                        program.display()
                    )));
                    continue;
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!("Failed to execute Bash via {}", program.display())
                    });
                }
            }
        }

        Err(last_not_found.unwrap_or_else(|| {
            anyhow::anyhow!(
                "No usable bash executable was found. Install Git Bash or ensure `bash` is on PATH."
            )
        }))
    }

    /// `Fetch`: make a direct HTTP request to a known URL.
    async fn fetch(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            url: String,
            method: Option<String>,
            #[serde(default)]
            headers: serde_json::Map<String, serde_json::Value>,
            body: Option<String>,
            max_bytes: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Fetch")?;
        if cancel.is_cancelled() {
            anyhow::bail!("Fetch cancelled");
        }

        let method = args
            .method
            .as_deref()
            .unwrap_or("GET")
            .parse::<reqwest::Method>()
            .context("Fetch method must be a valid HTTP method")?;
        let validated =
            validate_fetch_request(&args.url, &method, &args.headers, args.body.as_deref()).await?;
        let max_bytes = args
            .max_bytes
            .unwrap_or(MAX_FETCH_RESPONSE_BYTES)
            .min(MAX_FETCH_RESPONSE_BYTES);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("StudyAdministrator/0.6 Fetch")
            .build()
            .context("Failed to build Fetch HTTP client")?;
        let mut current_url = validated.url.clone();
        let mut redirect_hops = 0usize;

        loop {
            let mut request = client.request(method.clone(), current_url.clone());
            for (name, value) in &args.headers {
                let Some(value) = value.as_str() else {
                    anyhow::bail!("Fetch headers must be string key/value pairs");
                };
                request = request.header(name, value);
            }
            let mut response = tokio::select! {
                _ = cancel.cancelled() => {
                    anyhow::bail!("Fetch cancelled");
                }
                response = request.send() => {
                    response.context("Fetch request failed")?
                }
            };

            let status = response.status();
            if matches!(status.as_u16(), 301 | 302 | 303 | 307 | 308) {
                let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
                    anyhow::bail!("Fetch redirect response is missing a Location header");
                };
                let location = location
                    .to_str()
                    .context("Fetch redirect Location header is not valid UTF-8")?;
                let redirect_url = current_url.join(location).with_context(|| {
                    format!("Failed to resolve Fetch redirect target `{location}`")
                })?;

                if redirect_hops >= MAX_FETCH_REDIRECTS {
                    anyhow::bail!(
                        "Fetch exceeded SA's redirect safety limit of {} hops",
                        MAX_FETCH_REDIRECTS
                    );
                }

                let allows_auto_follow =
                    matches!(method, reqwest::Method::GET | reqwest::Method::HEAD)
                        && is_permitted_redirect(&current_url, &redirect_url);
                if !allows_auto_follow {
                    return Ok(serde_json::json!({
                        "redirect": true,
                        "redirect_blocked": true,
                        "original_url": args.url,
                        "current_url": current_url.as_str(),
                        "redirect_url": redirect_url.as_str(),
                        "method": method.as_str(),
                        "status": status.as_u16(),
                        "message": "Fetch detected a redirect that SA will not follow automatically. Re-run Fetch explicitly with the redirected URL if you intend to trust it.",
                        "safety_warning": validated.safety_warning,
                    })
                    .to_string());
                }

                validate_fetch_request(redirect_url.as_str(), &method, &args.headers, None).await?;
                redirect_hops += 1;
                current_url = redirect_url;
                continue;
            }

            if let Some(content_length) = response.content_length()
                && content_length > MAX_FETCH_TRANSFER_BYTES as u64
            {
                anyhow::bail!(
                    "Fetch response exceeds SA's {}-byte transport safety limit",
                    MAX_FETCH_TRANSFER_BYTES
                );
            }

            let headers = response.headers().clone();
            let content_type = headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            let (body_bytes, total_bytes, truncated) =
                read_fetch_body_limited(&mut response, max_bytes, cancel).await?;
            let body_text = String::from_utf8_lossy(&body_bytes).to_string();
            let response_headers = headers
                .iter()
                .filter_map(|(name, value)| {
                    value.to_str().ok().map(|value| {
                        (
                            name.as_str().to_string(),
                            serde_json::Value::String(value.to_string()),
                        )
                    })
                })
                .collect::<serde_json::Map<_, _>>();

            return Ok(serde_json::json!({
                "url": current_url.as_str(),
                "original_url": args.url,
                "method": method.as_str(),
                "status": status.as_u16(),
                "content_type": content_type,
                "headers": response_headers,
                "bytes": total_bytes,
                "truncated": truncated,
                "body": body_text,
                "safety_warning": validated.safety_warning,
            })
            .to_string());
        }
    }

    /// `Search`: discover relevant URLs before a more targeted `Fetch`.
    async fn search(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            max_results: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Search")?;
        if cancel.is_cancelled() {
            anyhow::bail!("Search cancelled");
        }

        let max_results = args.max_results.unwrap_or(5).clamp(1, 10);

        let results = crate::search_backends::multi_backend_search(
            args.query.trim(),
            max_results,
            &self.ctx.search_config,
            cancel,
        )
        .await?;

        Ok(serde_json::json!({
            "query": args.query,
            "results": results,
        })
        .to_string())
    }

    /// `MemorySearch`: search Markdown memory files on demand.
    async fn memory_search(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            max_results: Option<usize>,
            min_score: Option<f64>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MemorySearch")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemorySearch cancelled");
        }

        let results = search_markdown_memory(
            &self.ctx.workspace_root,
            &args.query,
            args.max_results,
            args.min_score,
            None,
            None,
        )
        .await?;

        Ok(serde_json::json!({
            "query": args.query,
            "results": results,
            "engine": "markdown_lexical_v1",
        })
        .to_string())
    }

    /// `MemoryGet`: read a bounded slice from one allowed memory file.
    async fn memory_get(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            from: Option<usize>,
            lines: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for MemoryGet")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemoryGet cancelled");
        }

        let result =
            read_markdown_memory(&self.ctx.workspace_root, &args.path, args.from, args.lines)
                .await?;

        Ok(serde_json::to_string(&result)?)
    }

    /// `Send`: forward a message to the user through the daemon/runtime layer.
    async fn send(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if !runtime.allow_user_send {
            anyhow::bail!("Send is not allowed for this agent");
        }
        if cancel.is_cancelled() {
            anyhow::bail!("Send cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            message: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Send")?;
        if args.message.trim().is_empty() {
            anyhow::bail!("Send message must not be empty");
        }

        runtime.send_message(args.message.clone()).await?;

        Ok(serde_json::json!({
            "sent": true,
            "message": args.message,
        })
        .to_string())
    }

    /// `Show`: transport a file payload to the user-facing frontend.
    async fn show(
        &self,
        session: &mut ToolSession,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if !runtime.allow_user_show {
            anyhow::bail!("Show is not allowed for this agent");
        }
        if cancel.is_cancelled() {
            anyhow::bail!("Show cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            title: Option<String>,
            prompt: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Show")?;
        if args.prompt.trim().is_empty() {
            anyhow::bail!("Show prompt must not be empty");
        }
        validate_tool_path_input(&args.path, PathOperation::Read)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Read)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file for Show: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "Show requires a file path, not a directory: {}",
                path.display()
            );
        }
        if meta.len() as usize > MAX_SHOW_FILE_BYTES {
            anyhow::bail!(
                "Show refuses files larger than {} bytes: {}",
                MAX_SHOW_FILE_BYTES,
                path.display()
            );
        }

        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("Failed to read file for Show: {}", path.display()))?;

        let (encoding, content) = match std::str::from_utf8(&bytes) {
            Ok(text) => (UserVisibleFileEncoding::Utf8, text.to_string()),
            Err(_) => (
                UserVisibleFileEncoding::Base64,
                base64::engine::general_purpose::STANDARD.encode(&bytes),
            ),
        };

        let file = UserVisibleFile {
            show_id: uuid::Uuid::new_v4(),
            task_id: uuid::Uuid::nil(),
            agent: None,
            path: path.display().to_string(),
            title: args.title.filter(|title| !title.trim().is_empty()),
            prompt: args.prompt,
            media_type: guess_media_type(&path, &encoding),
            encoding,
            content,
            bytes: bytes.len(),
        };

        runtime.show_file(file.clone()).await?;
        session.note_touched_path(path.clone());

        Ok(serde_json::json!({
            "shown": true,
            "path": file.path,
            "title": file.title,
            "bytes": file.bytes,
            "media_type": file.media_type,
            "encoding": file.encoding,
            "prompt": file.prompt,
        })
        .to_string())
    }

    /// `GetInteractionEntry`: retrieve one durable interaction-log entry by id
    /// with progressive disclosure.
    async fn get_interaction_entry(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("GetInteractionEntry cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            id: Uuid,
            mode: Option<String>,
            offset: Option<usize>,
            limit: Option<usize>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for GetInteractionEntry")?;
        let mode = match args.mode.as_deref().unwrap_or("summary") {
            "summary" => InteractionDisclosureMode::Summary,
            "full" => InteractionDisclosureMode::Full,
            "slice" => InteractionDisclosureMode::Slice,
            other => anyhow::bail!(
                "GetInteractionEntry mode must be one of `summary`, `full`, `slice`, got `{other}`"
            ),
        };
        let store = InteractionStore::new(self.ctx.workspace_root.clone())?;
        let Some(serialized) = store.serialize_entry_by_id(
            args.id,
            InteractionReadOptions {
                mode,
                offset: args.offset.unwrap_or(0),
                limit: args.limit.unwrap_or_default(),
            },
        )?
        else {
            anyhow::bail!("Interaction entry not found: {}", args.id);
        };

        Ok(serialized)
    }

    /// `Ask`: block until the user answers a structured question.
    async fn ask(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if !runtime.allow_user_ask {
            anyhow::bail!("Ask is not allowed for this agent");
        }
        if cancel.is_cancelled() {
            anyhow::bail!("Ask cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            prompt: String,
            mode: QuestionMode,
            #[serde(default)]
            options: Vec<QuestionOption>,
            #[serde(default)]
            allow_free_text: bool,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Ask")?;
        let request = AskRequest {
            prompt: args.prompt,
            mode: args.mode,
            options: args.options,
            allow_free_text: args.allow_free_text,
        };
        request.validate()?;
        Ok(ToolExecutionResult::Control(ToolControl::Ask(request)))
    }

    /// `Skill`: invoke one registered command or read one local skill file.
    async fn skill(
        &self,
        session: &mut ToolSession,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Skill cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            #[serde(default = "default_skill_action")]
            action: String,
            name: String,
            args: Option<String>,
            path: Option<String>,
            session_id: Option<String>,
        }

        fn default_skill_action() -> String {
            "invoke".to_string()
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Skill")?;
        if matches_command_patterns(&args.name, &runtime.denied_commands) {
            anyhow::bail!(
                "Command `{}` is denied by the current permission policy",
                args.name
            );
        }
        let Some(skill) = self.ctx.skills.get(&args.name) else {
            anyhow::bail!("Skill not found: {}", args.name);
        };

        match args.action.as_str() {
            "read" => {
                let (path, content) = self
                    .ctx
                    .skills
                    .load_skill_file(&args.name, args.path.as_deref())
                    .await?;

                Ok(serde_json::json!({
                    "name": skill.name,
                    "description": skill.description,
                    "path": path,
                    "content": content,
                })
                .to_string())
            }
            "invoke" => {
                if skill.disable_model_invocation {
                    anyhow::bail!(
                        "Command `{}` is not model-invocable and must not be invoked automatically",
                        skill.name
                    );
                }

                let instructions = if let Some(prompt) = skill.mcp_prompt() {
                    let Some(registry) = &self.mcp_registry else {
                        anyhow::bail!("No MCP registry is configured for MCP prompt invocation");
                    };
                    registry
                        .expand_prompt(
                            &prompt.server_name,
                            &prompt.prompt_name,
                            &prompt.arguments,
                            args.args.as_deref(),
                        )
                        .await?
                } else {
                    let session_id = args
                        .session_id
                        .clone()
                        .unwrap_or_else(|| runtime.agent_id.to_string());
                    skill
                        .expand_invocation(args.args.as_deref(), &session_id)?
                        .instructions
                };

                session.note_command_invocation(skill.reminder());

                Ok(instructions)
            }
            other => anyhow::bail!("Skill action must be `invoke` or `read`, got `{other}`"),
        }
    }

    /// `ListMcpResources`: enumerate connected MCP resources.
    async fn list_mcp_resources(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("ListMcpResources cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            server: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ListMcpResources")?;
        let Some(registry) = &self.mcp_registry else {
            anyhow::bail!("No MCP registry is configured");
        };

        registry.list_resources(args.server.as_deref()).await
    }

    /// `ReadMcpResource`: read one resource payload from a connected MCP
    /// server.
    async fn read_mcp_resource(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("ReadMcpResource cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            server: String,
            uri: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ReadMcpResource")?;
        let Some(registry) = &self.mcp_registry else {
            anyhow::bail!("No MCP registry is configured");
        };

        registry.read_resource(&args.server, &args.uri).await
    }

    /// `SubAgent`: delegate a focused sub-task to a nested agent.
    async fn subagent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("SubAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            #[serde(default, deserialize_with = "deserialize_optional_nonempty_string")]
            label: Option<String>,
            task: String,
            context: String,
            #[serde(default, deserialize_with = "deserialize_optional_nonempty_string")]
            prompt: Option<String>,
            #[serde(default, deserialize_with = "deserialize_optional_nonempty_string")]
            prompt_file: Option<String>,
            #[serde(default, deserialize_with = "deserialize_optional_nonempty_string")]
            prompt_skill: Option<String>,
            #[serde(default)]
            allow_user_send: bool,
            #[serde(default)]
            allow_user_show: bool,
            #[serde(default)]
            allow_user_ask: bool,
            #[serde(default)]
            allow_input_transfer_target: bool,
            #[serde(default, deserialize_with = "deserialize_optional_uuid_or_empty")]
            existing_agent_id: Option<Uuid>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for SubAgent")?;
        let request = SubAgentRequest {
            label: args.label,
            task: args.task,
            context: args.context,
            prompt: args.prompt,
            prompt_file: args.prompt_file,
            prompt_skill: args.prompt_skill,
            allow_user_send: args.allow_user_send,
            allow_user_show: args.allow_user_show,
            allow_user_ask: args.allow_user_ask,
            allow_input_transfer_target: args.allow_input_transfer_target,
            existing_agent_id: args.existing_agent_id,
            category: None,
        };
        request.validate()?;

        let handle = runtime.run_subagent(request, cancel.clone()).await?;

        Ok(serde_json::json!({
            "agent_id": handle.agent_id,
            "work_id": handle.work_id,
            "label": handle.label,
            "status": handle.status,
        })
        .to_string())
    }

    /// `Finish`: explicitly mark the current work as complete.
    async fn finish(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if cancel.is_cancelled() {
            anyhow::bail!("Finish cancelled");
        }

        let args: FinishRequest =
            serde_json::from_value(args).context("Invalid arguments for Finish")?;
        args.validate()?;

        Ok(ToolExecutionResult::Control(ToolControl::Finish(args)))
    }

    /// `FinishWithoutOutput`: explicit confirmation for a temporary
    /// no-extra-output finish path.
    async fn finish_without_output(
        &self,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if cancel.is_cancelled() {
            anyhow::bail!("FinishWithoutOutput cancelled");
        }

        Ok(ToolExecutionResult::Control(
            ToolControl::FinishWithoutOutput,
        ))
    }

    /// `NotifyParent`: convenience one-way message to the current parent.
    async fn notify_parent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("NotifyParent cancelled");
        }
        if runtime.parent_agent_id.is_none() {
            anyhow::bail!("NotifyParent is only available for child agents");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            message: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for NotifyParent")?;
        if args.message.trim().is_empty() {
            anyhow::bail!("NotifyParent message must not be empty");
        }

        let receipt = runtime.notify_parent(args.message.clone()).await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `MessageAgent`: direct one-way message to another agent.
    async fn message_agent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("MessageAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            target_agent_id: Uuid,
            message: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MessageAgent")?;
        let request = AgentMessageRequest {
            target_agent_id: args.target_agent_id,
            message: args.message,
        };
        request.validate()?;
        let receipt = runtime.message_agent(request).await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `BroadcastAgents`: broadcast a one-way message to a scoped set of agents.
    async fn broadcast_agents(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("BroadcastAgents cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            scope: Option<AgentScope>,
            message: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for BroadcastAgents")?;
        let request = BroadcastAgentsRequest {
            scope: args.scope.unwrap_or(AgentScope::Descendants),
            message: args.message,
        };
        request.validate()?;
        let receipt = runtime.broadcast_agents(request).await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `ListAgents`: inspect visible agents.
    async fn list_agents(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("ListAgents cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            scope: Option<AgentScope>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ListAgents")?;
        let items = runtime
            .list_agents(ListAgentsRequest {
                scope: args.scope.unwrap_or(AgentScope::Descendants),
            })
            .await?;
        Ok(serde_json::to_string(&items)?)
    }

    /// `GetAgent`: inspect one visible agent.
    async fn get_agent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("GetAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            agent_id: Uuid,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for GetAgent")?;
        let item = runtime.get_agent(args.agent_id).await?;
        Ok(serde_json::to_string(&item)?)
    }

    /// `TransferInput`: move free-form user input ownership.
    async fn transfer_input(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("TransferInput cancelled");
        }
        if !runtime.is_root {
            anyhow::bail!("TransferInput is only available to the root agent");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            target_agent_id: Option<Uuid>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for TransferInput")?;
        let receipt = runtime
            .transfer_input(TransferInputRequest {
                target_agent_id: args.target_agent_id,
            })
            .await?;
        Ok(serde_json::to_string(&receipt)?)
    }

    /// `Wait`: suspend the current work until a dependency reaches the desired
    /// state.
    async fn wait(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if cancel.is_cancelled() {
            anyhow::bail!("Wait cancelled");
        }

        let args: WaitRequest =
            serde_json::from_value(args).context("Invalid arguments for Wait")?;
        args.validate()?;
        Ok(ToolExecutionResult::Control(ToolControl::Wait(args)))
    }

    /// `GetTask`: inspect one runtime task.
    async fn get_task(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("GetTask cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            task_id: Uuid,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for GetTask")?;
        let task = runtime.get_task(args.task_id).await?;
        Ok(serde_json::to_string(&task)?)
    }

    /// `Reload`: ask the backend to hot-reload runtime state from disk.
    async fn reload(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Reload cancelled");
        }
        if !runtime.allow_runtime_reload {
            anyhow::bail!("Reload is not available in the current runtime");
        }

        #[derive(Debug, Deserialize, Default)]
        struct Args {}

        let _: Args = serde_json::from_value(args).context("Invalid arguments for Reload")?;
        let receipt = runtime.reload_runtime().await?;
        Ok(serde_json::json!({
            "reloaded": true,
            "summary": receipt.summary,
        })
        .to_string())
    }
}

/// Candidate `bash` programs to try, in order.
///
/// Strategy:
/// - On Windows, prefer Git Bash specifically.
/// - Do not prefer the generic `System32\\bash.exe` / WSL shim because it can
///   launch a different environment where `git` is unavailable, which is the
///   exact failure mode SA must avoid.
/// - On non-Windows platforms, plain `bash` remains fine.
fn candidate_bash_programs() -> Vec<PathBuf> {
    #[cfg(not(windows))]
    {
        return vec![PathBuf::from("bash")];
    }

    #[cfg(windows)]
    {
        let mut out = Vec::<PathBuf>::new();

        if let Ok(git_paths) = which::which_all("git") {
            for git_path in git_paths {
                out.extend(infer_git_bash_candidates_from_git_path(&git_path));
            }
        }

        if let Ok(bash_paths) = which::which_all("bash") {
            for bash_path in bash_paths {
                if is_preferred_windows_bash_candidate(&bash_path) {
                    out.push(bash_path);
                }
            }
        }

        for env_name in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Ok(root) = std::env::var(env_name) {
                let root = PathBuf::from(root);
                out.push(root.join("Git").join("bin").join("bash.exe"));
                out.push(root.join("Git").join("usr").join("bin").join("bash.exe"));
            }
        }

        dedup_paths_preserve_order(out)
    }
}

/// Infer likely Git Bash locations from one discovered `git.exe` path.
#[cfg(windows)]
fn infer_git_bash_candidates_from_git_path(git_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(parent) = git_path.parent() else {
        return out;
    };
    let Some(root) = parent.parent() else {
        return out;
    };

    out.push(root.join("bin").join("bash.exe"));
    out.push(root.join("usr").join("bin").join("bash.exe"));
    out
}

/// Return whether a Windows `bash.exe` path looks like Git Bash rather than a
/// WSL or Windows shim.
#[cfg(windows)]
fn is_preferred_windows_bash_candidate(path: &Path) -> bool {
    let normalized = path
        .to_string_lossy()
        .replace('/', "\\")
        .to_ascii_lowercase();
    normalized.ends_with("\\git\\bin\\bash.exe")
        || normalized.ends_with("\\git\\usr\\bin\\bash.exe")
}

/// Deduplicate paths while preserving their first-seen order.
fn dedup_paths_preserve_order(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    use std::collections::HashSet;

    let mut seen = HashSet::<PathBuf>::new();
    let mut out = Vec::new();
    for path in paths {
        if seen.insert(path.clone()) {
            out.push(path);
        }
    }
    out
}

/// Validate that a network URL is HTTP(S) and therefore appropriate for
/// `Fetch`.
pub(crate) fn validate_network_url(raw: &str) -> anyhow::Result<reqwest::Url> {
    let url = reqwest::Url::parse(raw).context("Fetch URL must be a valid absolute URL")?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => anyhow::bail!("Fetch only allows http/https URLs, got scheme `{other}`"),
    }
}

/// Read a Fetch response body while enforcing SA's transport safety ceiling.
async fn read_fetch_body_limited(
    response: &mut reqwest::Response,
    return_limit: usize,
    cancel: &CancelToken,
) -> anyhow::Result<(Vec<u8>, usize, bool)> {
    let mut total_bytes = 0usize;
    let mut clipped = Vec::<u8>::new();
    let mut truncated = false;

    loop {
        let next_chunk = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("Fetch cancelled");
            }
            chunk = response.chunk() => {
                chunk.context("Failed to read Fetch response body chunk")?
            }
        };

        let Some(chunk) = next_chunk else {
            break;
        };

        total_bytes = total_bytes.saturating_add(chunk.len());
        if total_bytes > MAX_FETCH_TRANSFER_BYTES {
            anyhow::bail!(
                "Fetch response exceeded SA's {}-byte transport safety limit while streaming",
                MAX_FETCH_TRANSFER_BYTES
            );
        }

        let remaining = return_limit.saturating_sub(clipped.len());
        if remaining > 0 {
            let take = remaining.min(chunk.len());
            clipped.extend_from_slice(&chunk[..take]);
        }
        if chunk.len() > remaining {
            truncated = true;
        }
    }

    if total_bytes > return_limit {
        truncated = true;
    }

    Ok((clipped, total_bytes, truncated))
}

/// Infer a reasonable media type for `Show`.
fn guess_media_type(path: &Path, encoding: &UserVisibleFileEncoding) -> String {
    match encoding {
        UserVisibleFileEncoding::Base64 => {
            let ext = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase());
            match ext.as_deref() {
                Some("png") => "image/png".to_string(),
                Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
                Some("gif") => "image/gif".to_string(),
                Some("webp") => "image/webp".to_string(),
                Some("pdf") => "application/pdf".to_string(),
                Some("zip") => "application/zip".to_string(),
                _ => "application/octet-stream".to_string(),
            }
        }
        UserVisibleFileEncoding::Utf8 => {
            let ext = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase());
            match ext.as_deref() {
                Some("md") => "text/markdown".to_string(),
                Some("json") => "application/json".to_string(),
                Some("toml") => "application/toml".to_string(),
                Some("yaml") | Some("yml") => "application/yaml".to_string(),
                Some("rs") => "text/x-rust".to_string(),
                Some("sh") => "text/x-shellscript".to_string(),
                Some("txt") | Some("log") => "text/plain".to_string(),
                _ => "text/plain; charset=utf-8".to_string(),
            }
        }
    }
}

/// Extract the raw Bash command string when the current call targets `Bash`.
fn extract_bash_command<'a>(tool_name: &str, args: &'a serde_json::Value) -> Option<&'a str> {
    if tool_name != "Bash" {
        return None;
    }

    args.get("command")
        .and_then(|value| value.as_str())
        .map(str::trim)
}

/// Decide whether one allow/deny pattern should keep a tool visible in the
/// model-facing tool list.
///
/// Visibility is intentionally approximate for argument-scoped rules such as
/// `Bash(git:*)`: the model still needs to see `Bash`, while concrete command
/// enforcement happens later in [`matches_tool_patterns`].
fn pattern_allows_tool_visibility(pattern: &str, tool_name: &str) -> bool {
    if let Some((pattern_tool_name, _)) = split_tool_scope_pattern(pattern) {
        return pattern_tool_name == tool_name;
    }

    matches_command_patterns(tool_name, &[pattern.to_string()])
}

/// Match concrete tool calls against plain tool patterns and `Bash(...)`
/// command-prefix patterns.
fn matches_tool_patterns(tool_name: &str, bash_command: Option<&str>, patterns: &[String]) -> bool {
    patterns
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.is_empty())
        .any(|pattern| {
            if let Some((pattern_tool_name, scoped_pattern)) = split_tool_scope_pattern(pattern) {
                if pattern_tool_name != tool_name {
                    return false;
                }
                return match (pattern_tool_name, bash_command) {
                    ("Bash", Some(command)) => matches_bash_scoped_pattern(command, scoped_pattern),
                    _ => false,
                };
            }

            matches_command_patterns(tool_name, &[pattern.to_string()])
        })
}

/// Parse `ToolName(scope-pattern)` forms used by command-scoped allowlists.
fn split_tool_scope_pattern(pattern: &str) -> Option<(&str, &str)> {
    let open = pattern.find('(')?;
    let close = pattern.rfind(')')?;
    if close <= open {
        return None;
    }

    let tool_name = pattern[..open].trim();
    let scoped_pattern = pattern[open + 1..close].trim();
    if tool_name.is_empty() || scoped_pattern.is_empty() {
        return None;
    }

    Some((tool_name, scoped_pattern))
}

/// Match one Bash command against the scoped pattern syntax used by
/// `allowed-tools`, for example `git:*`.
///
/// The `:<star>` suffix is treated as a shell-command prefix rule rather than
/// a literal colon match, so `git:*` means "the command starts with `git`".
fn matches_bash_scoped_pattern(command: &str, pattern: &str) -> bool {
    let command = command.trim();
    let pattern = pattern.trim();

    if let Some(prefix) = pattern.strip_suffix(":*").map(str::trim) {
        return command == prefix
            || command.starts_with(&format!("{prefix} "))
            || command.starts_with(&format!("{prefix}\t"));
    }

    matches_command_patterns(command, &[pattern.to_string()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_history::{InteractionPayload, InteractionStore};
    use crate::skills::SkillRegistry;
    use std::fs;
    use uuid::Uuid;

    /// Create a unique temp directory for tests without adding extra deps.
    fn unique_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sa-tools-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Create a minimal tool context rooted at a fresh temp directory.
    fn test_context() -> ToolContext {
        ToolContext::new(unique_temp_dir(), Arc::new(SkillRegistry::default())).expect("context")
    }

    /// Create a tool context with one local skill directory already scanned.
    fn test_context_with_skill(skill_name: &str, skill_markdown: &str) -> ToolContext {
        let root = unique_temp_dir();
        let skill_dir = root.join(skill_name);
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::write(skill_dir.join("SKILL.md"), skill_markdown).expect("write SKILL.md");
        let registry = Arc::new(SkillRegistry::scan(&[root]).expect("scan skills"));
        ToolContext::new(unique_temp_dir(), registry).expect("context")
    }

    #[test]
    fn resolve_under_workspace_rejects_escape() {
        let ctx = test_context();
        let err = ctx
            .resolve_under_workspace("..\\outside.txt")
            .expect_err("path traversal must fail");
        assert!(err.to_string().contains("escapes workspace root"));
    }

    #[test]
    fn validate_network_url_rejects_non_http() {
        let err = validate_network_url("file:///tmp/secret.txt")
            .expect_err("non-http URL must be rejected");
        assert!(err.to_string().contains("http/https"));
    }

    #[tokio::test]
    async fn fetch_rejects_localhost_before_network() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let cancel = crate::cancel::cancel_pair().1;

        let err = executor
            .fetch(
                serde_json::json!({
                    "url": "http://localhost:8080/private"
                }),
                &cancel,
            )
            .await
            .expect_err("localhost fetch must fail");

        assert!(err.to_string().contains("localhost"));
    }

    #[tokio::test]
    async fn fetch_rejects_sensitive_auth_headers_before_network() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let cancel = crate::cancel::cancel_pair().1;

        let err = executor
            .fetch(
                serde_json::json!({
                    "url": "https://example.com/docs",
                    "headers": {
                        "Authorization": "Bearer secret"
                    }
                }),
                &cancel,
            )
            .await
            .expect_err("auth header fetch must fail");

        assert!(err.to_string().contains("sensitive request header"));
    }

    #[tokio::test]
    async fn write_refuses_existing_file() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let path = ctx.workspace_root.join("already.txt");
        fs::write(&path, "hello").expect("seed file");

        let err = executor
            .write(
                &mut session,
                serde_json::json!({
                    "path": "already.txt",
                    "content": "new",
                }),
                &crate::cancel::cancel_pair().1,
            )
            .await
            .expect_err("overwrite must fail");

        assert!(err.to_string().contains("refuses to overwrite"));
    }

    #[tokio::test]
    async fn write_rejects_sensitive_config_path() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();

        let err = executor
            .write(
                &mut session,
                serde_json::json!({
                    "path": ".env",
                    "content": "API_KEY=secret",
                }),
                &crate::cancel::cancel_pair().1,
            )
            .await
            .expect_err("sensitive config path must fail");

        assert!(err.to_string().contains("dangerous configuration files"));
    }

    #[tokio::test]
    async fn edit_requires_read_first() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("note.txt"), "alpha beta").expect("seed file");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect_err("edit without read must fail");

        assert!(err.to_string().contains("has been Read"));
    }

    #[tokio::test]
    async fn edit_rejects_root_control_file_even_after_read() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("prompt.md"), "alpha beta").expect("seed prompt.md");
        executor
            .read(
                &mut session,
                serde_json::json!({
                    "path": "prompt.md"
                }),
                &cancel,
            )
            .await
            .expect("read prompt.md");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "prompt.md",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect_err("control file edit must fail");

        assert!(err.to_string().contains("control files"));
    }

    #[tokio::test]
    async fn read_rejects_sensitive_config_path() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("sa.toml"), "api_key = 'secret'").expect("seed sa.toml");

        let err = executor
            .read(
                &mut session,
                serde_json::json!({
                    "path": "sa.toml"
                }),
                &cancel,
            )
            .await
            .expect_err("sensitive read must fail");

        assert!(err.to_string().contains("sensitive configuration files"));
    }

    #[tokio::test]
    async fn edit_invalidates_read_marker_after_success() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("note.txt"), "alpha beta").expect("seed file");

        executor
            .read(
                &mut session,
                serde_json::json!({ "path": "note.txt" }),
                &cancel,
            )
            .await
            .expect("read");

        executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect("edit");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "gamma",
                    "new_text": "delta",
                }),
                &cancel,
            )
            .await
            .expect_err("second edit without reread must fail");

        assert!(err.to_string().contains("has been Read"));
    }

    #[tokio::test]
    async fn edit_rejects_file_changed_since_read() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        let path = ctx.workspace_root.join("note.txt");
        fs::write(&path, "alpha beta").expect("seed file");

        executor
            .read(
                &mut session,
                serde_json::json!({
                    "path": "note.txt"
                }),
                &cancel,
            )
            .await
            .expect("read note");

        fs::write(&path, "alpha beta changed").expect("mutate file externally");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect_err("edit after external change must fail");

        assert!(
            err.to_string()
                .contains("file changed since it was last Read")
        );
    }

    #[tokio::test]
    async fn edit_rejects_restored_read_set_without_version_metadata() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::from_readable_paths([ctx.workspace_root.join("note.txt")]);
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("note.txt"), "alpha beta").expect("seed file");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect_err("restored read-set without version metadata must fail");

        assert!(err.to_string().contains("fresh re-Read after restart"));
    }

    #[tokio::test]
    async fn show_requires_non_empty_prompt() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("note.txt"), "hello").expect("seed file");

        let err = executor
            .show(
                &mut session,
                &ToolRuntime::detached(),
                serde_json::json!({
                    "path": "note.txt",
                    "prompt": ""
                }),
                &cancel,
            )
            .await
            .expect_err("empty prompt must fail");

        assert!(err.to_string().contains("Show prompt must not be empty"));
    }

    #[tokio::test]
    async fn show_rejects_sensitive_runtime_file() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;
        let sessions_dir = ctx.workspace_root.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("create sessions dir");
        fs::write(sessions_dir.join("current.jsonl"), "{}\n").expect("seed session file");

        let err = executor
            .show(
                &mut session,
                &ToolRuntime::detached(),
                serde_json::json!({
                    "path": "sessions/current.jsonl",
                    "prompt": "show runtime data"
                }),
                &cancel,
            )
            .await
            .expect_err("runtime Show must fail");

        assert!(err.to_string().contains("runtime/session data"));
    }

    #[tokio::test]
    async fn get_interaction_entry_reads_summary_from_log() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone(), None);
        let cancel = crate::cancel::cancel_pair().1;
        let store = InteractionStore::new(ctx.workspace_root.clone()).expect("store");
        let entry = store
            .append_new(
                Some(Uuid::new_v4()),
                Some(Uuid::new_v4()),
                InteractionPayload::Send {
                    message: "history message".to_string(),
                },
            )
            .expect("entry should append");

        let raw = executor
            .get_interaction_entry(
                serde_json::json!({
                    "id": entry.id,
                    "mode": "summary"
                }),
                &cancel,
            )
            .await
            .expect("tool should read interaction entry");
        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("tool output should be valid JSON");
        assert_eq!(value["kind"], "send");
        assert_eq!(value["id"], entry.id.to_string());
        assert_eq!(value["message_preview"], "history message");
    }

    #[tokio::test]
    async fn skill_invoke_expands_inline_command_body() {
        let ctx = test_context_with_skill(
            "writer",
            r#"---
name: writer
description: Writes polished reports
arguments:
  - topic
---

Write about $topic in session ${SA_SESSION_ID}.
"#,
        );
        let executor = ToolExecutor::new(ctx, None);
        let cancel = crate::cancel::cancel_pair().1;
        let runtime = ToolRuntime::detached();
        let mut session = ToolSession::default();

        let raw = executor
            .skill(
                &mut session,
                &runtime,
                serde_json::json!({
                    "action": "invoke",
                    "name": "writer",
                    "args": "study-notes",
                    "session_id": "session-test"
                }),
                &cancel,
            )
            .await
            .expect("skill invoke should succeed");

        assert!(raw.contains("study-notes"));
        assert!(raw.contains("session-test"));
        assert!(!raw.contains("\"path\""));
    }

    #[test]
    fn tool_definitions_include_mcp_resource_tools_and_skill_action_schema() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let runtime = ToolRuntime::detached();
        let definitions = executor.tool_definitions(&runtime);

        let skill = definitions
            .iter()
            .find(|definition| definition.function.name == "Skill")
            .expect("Skill definition should exist");
        assert!(
            skill.function.parameters["properties"]
                .get("action")
                .is_some()
        );

        assert!(
            definitions
                .iter()
                .any(|definition| definition.function.name == "ListMcpResources")
        );
        assert!(
            definitions
                .iter()
                .any(|definition| definition.function.name == "ReadMcpResource")
        );
    }

    /// Build one detached runtime and override the permission fields relevant
    /// to these unit tests.
    fn detached_runtime_with_permissions(
        deny_tools: &[&str],
        deny_commands: &[&str],
        allowed_tool_patterns: &[&str],
    ) -> ToolRuntime {
        let mut runtime = ToolRuntime::detached();
        runtime.denied_tools = deny_tools
            .iter()
            .map(|value| (*value).to_string())
            .collect();
        runtime.denied_commands = deny_commands
            .iter()
            .map(|value| (*value).to_string())
            .collect();
        runtime.allowed_tool_patterns = allowed_tool_patterns
            .iter()
            .map(|value| (*value).to_string())
            .collect();
        runtime
    }

    #[test]
    fn matches_tool_patterns_supports_bash_scope_rules() {
        assert!(matches_tool_patterns(
            "Bash",
            Some("git status"),
            &[String::from("Bash(git:*)")]
        ));
        assert!(!matches_tool_patterns(
            "Bash",
            Some("cargo test"),
            &[String::from("Bash(git:*)")]
        ));
        assert!(matches_tool_patterns(
            "mcp__playwright__navigate",
            None,
            &[String::from("mcp__playwright__*")]
        ));
    }

    #[test]
    fn tool_definitions_hide_denied_and_out_of_scope_tools() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let runtime = detached_runtime_with_permissions(&["Show"], &[], &["Read", "Bash(git:*)"]);
        let definitions = executor.tool_definitions(&runtime);
        let names = definitions
            .iter()
            .map(|definition| definition.function.name.as_str())
            .collect::<Vec<_>>();

        assert!(names.contains(&"Read"));
        assert!(names.contains(&"Bash"));
        assert!(!names.contains(&"Show"));
        assert!(!names.contains(&"Send"));
    }

    #[test]
    fn tool_definitions_hide_interaction_tools_when_runtime_lacks_permissions() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let mut runtime = ToolRuntime::detached();
        runtime.allow_user_send = false;
        runtime.allow_user_show = false;
        runtime.allow_user_ask = false;

        let names = executor
            .tool_definitions(&runtime)
            .into_iter()
            .map(|definition| definition.function.name)
            .collect::<Vec<_>>();

        assert!(!names.iter().any(|name| name == "Send"));
        assert!(!names.iter().any(|name| name == "Show"));
        assert!(!names.iter().any(|name| name == "Ask"));
    }

    #[tokio::test]
    async fn reload_tool_invokes_runtime_callback_and_returns_summary() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let cancel = crate::cancel::cancel_pair().1;
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = ToolRuntime::detached();
        runtime.allow_runtime_reload = true;
        runtime.reload_runtime = {
            let calls = Arc::clone(&calls);
            Arc::new(move || {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(ReloadRuntimeReceipt {
                        summary: "reloaded model=test-model skills=3 mcp_servers=1".to_string(),
                    })
                })
            })
        };
        let mut session = ToolSession::default();

        let raw = executor
            .execute(
                &mut session,
                &runtime,
                "Reload",
                serde_json::json!({}),
                &cancel,
            )
            .await
            .expect("Reload should succeed");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let ToolExecutionResult::Observation(raw) = raw else {
            panic!("Reload should return an observation payload");
        };
        assert!(raw.contains("reloaded model=test-model"));
    }

    #[tokio::test]
    async fn subagent_accepts_empty_optional_fields_as_absent() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let cancel = crate::cancel::cancel_pair().1;
        let mut runtime = ToolRuntime::detached();
        runtime.run_subagent = Arc::new(|request, _cancel| {
            Box::pin(async move {
                assert_eq!(request.label.as_deref(), Some("chat-worker"));
                assert_eq!(request.prompt, None);
                assert_eq!(request.prompt_file, None);
                assert_eq!(request.prompt_skill, None);
                assert_eq!(request.existing_agent_id, None);
                Ok(SubAgentHandle {
                    agent_id: Uuid::new_v4(),
                    work_id: Uuid::new_v4(),
                    label: request.label.unwrap_or_else(|| "child".to_string()),
                    status: AgentStatus::Idle,
                })
            })
        });

        let raw = executor
            .subagent(
                &runtime,
                serde_json::json!({
                    "label": "chat-worker",
                    "task": "talk to the user",
                    "context": "take over the conversation after verification",
                    "prompt": "",
                    "prompt_file": "",
                    "prompt_skill": "",
                    "existing_agent_id": "",
                    "allow_user_send": true,
                    "allow_user_ask": true,
                    "allow_user_show": false,
                    "allow_input_transfer_target": true
                }),
                &cancel,
            )
            .await
            .expect("SubAgent should treat empty optional fields as absent");

        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("tool output should be valid JSON");
        assert_eq!(value["label"], "chat-worker");
    }

    #[cfg(windows)]
    #[test]
    fn infer_git_bash_candidates_from_git_path_prefers_git_install_root() {
        let git_path = PathBuf::from(r"D:\Git\cmd\git.exe");
        let candidates = infer_git_bash_candidates_from_git_path(&git_path);

        assert_eq!(candidates[0], PathBuf::from(r"D:\Git\bin\bash.exe"));
        assert_eq!(candidates[1], PathBuf::from(r"D:\Git\usr\bin\bash.exe"));
    }

    #[cfg(windows)]
    #[test]
    fn preferred_windows_bash_candidate_rejects_wsl_shims() {
        assert!(is_preferred_windows_bash_candidate(Path::new(
            r"D:\Git\bin\bash.exe"
        )));
        assert!(!is_preferred_windows_bash_candidate(Path::new(
            r"C:\Windows\System32\bash.exe"
        )));
        assert!(!is_preferred_windows_bash_candidate(Path::new(
            r"C:\Users\name\AppData\Local\Microsoft\WindowsApps\bash.exe"
        )));
    }

    #[tokio::test]
    async fn bash_tool_blocks_unsafe_command_before_execution() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx, None);
        let cancel = crate::cancel::cancel_pair().1;

        let err = executor
            .bash(
                &ToolRuntime::detached(),
                serde_json::json!({
                    "command": "echo $(whoami)"
                }),
                &cancel,
            )
            .await
            .expect_err("unsafe Bash command must be blocked");

        assert!(err.to_string().contains("Bash blocked by SA safety checks"));
    }

    #[tokio::test]
    async fn execute_rejects_denied_tool_before_running_it() {
        let ctx = test_context();
        fs::write(ctx.workspace_root.join("note.txt"), "hello").expect("seed file");
        let executor = ToolExecutor::new(ctx, None);
        let runtime = detached_runtime_with_permissions(&["Read"], &[], &[]);
        let cancel = crate::cancel::cancel_pair().1;
        let mut session = ToolSession::default();

        let err = executor
            .execute(
                &mut session,
                &runtime,
                "Read",
                serde_json::json!({ "path": "note.txt" }),
                &cancel,
            )
            .await
            .expect_err("denied tool must fail");

        assert!(
            err.to_string()
                .contains("denied by the current permission policy")
        );
    }

    #[tokio::test]
    async fn skill_respects_denied_commands() {
        let ctx = test_context_with_skill(
            "writer",
            r#"---
name: writer
description: Writes polished reports
---

Write the report.
"#,
        );
        let executor = ToolExecutor::new(ctx, None);
        let runtime = detached_runtime_with_permissions(&[], &["writer"], &[]);
        let cancel = crate::cancel::cancel_pair().1;
        let mut session = ToolSession::default();

        let err = executor
            .skill(
                &mut session,
                &runtime,
                serde_json::json!({
                    "action": "invoke",
                    "name": "writer"
                }),
                &cancel,
            )
            .await
            .expect_err("denied command must fail");

        assert!(
            err.to_string()
                .contains("denied by the current permission policy")
        );
    }
}
