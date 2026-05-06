//! `sa` 闁?the StudyAdministrator (SA) backend agent daemon.
//!
//! Responsibilities:
//! - Load `sa.toml` (TOML config).
//! - Read `Agents.md` and discover skills (`SKILL.md`).
//! - Run an autonomous agent loop in the background (OpenAI tool calling).
//! - Expose a **WebSocket** interface for a CLI frontend.
//! - Ensure that **client disconnects do not stop the agent**:
//!   - tasks are queued and processed independently of WS connections
//!   - events are buffered and can be replayed on reconnect
//!
//! Frontend note:
//! - The CLI frontend is a separate project located at `../sa-cli`.

use anyhow::Context as _;
use axum::Router;
use axum::extract::State;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code};
use axum::response::IntoResponse;
use axum::routing::get;
use clap::Parser;
use futures_util::{SinkExt as _, StreamExt as _};
use sa_core::agent::{
    AgentQuantumOutcome, AgentRunner, AgentRunnerConfig, DrainQueuedUserMessagesFn,
    EmitContextInfoFn, EmitEventFn,
};
use sa_core::agents_md::{extract_markdown_file_references, load_agents_md};
use sa_core::cancel::{CancelHandle, cancel_pair};
use sa_core::config::{
    Config, PermissionsConfig, TeamConfig, load_config_from_file, resolve_config_file_dir,
};
use sa_core::dream::DreamManager;
use sa_core::interaction_history::{
    InteractionPayload, InteractionStore, show_payload_from_visible_file,
};
use sa_core::mcp_client::McpRegistry;
use sa_core::memory::{
    build_prompt_block as build_memory_prompt_block, extract_facts, find_similar_clusters,
    is_memory_reference,
};
use sa_core::openai::{AuthStyle, ChatMessage, OpenAiClient, WireApi};
use sa_core::runtime::state::AgentStatus;
use sa_core::runtime::state::{
    AgentKind, AgentState, MailboxEntry, MailboxEntryKind, PendingAssistantMessage,
    PendingControlAction, PendingFinishConfirmation, PendingFinishMode, PendingQuestionState,
    RuntimeTaskKind, RuntimeTaskState, RuntimeTaskStatus, RuntimeWorkState, RuntimeWorkStatus,
    TeamState, WaitKind, WaitUntil, WaitingDependency,
};
use sa_core::runtime::store::{RootMarker, RuntimeStore};
use sa_core::session::SessionStore;
use sa_core::skills::SkillRegistry;
use sa_core::task_audit::{TaskAuditPhase, TaskAuditStore};
use sa_core::tools::{
    AgentInfo, AgentMessageReceipt, AgentMessageRequest, AgentScope, AskQuestionFn, AskRequest,
    BroadcastAgentsFn, BroadcastAgentsRequest, BroadcastReceipt, GetAgentFn, GetTaskFn,
    ListAgentsFn, ListAgentsRequest, MAX_SUBAGENT_DEPTH, MessageAgentFn, NotifyParentFn,
    PromptProfile, ReloadRuntimeFn, ReloadRuntimeReceipt, RunSubAgentFn, SendMessageFn, ShowFileFn,
    StartTerminalTaskFn, StartTerminalTaskRequest, SubAgentHandle, SubAgentRequest,
    TerminalTaskHandle, TerminalTaskInfo, ToolContext, ToolExecutor, ToolRuntime, TransferInputFn,
    TransferInputReceipt, TransferInputRequest,
};
use sa_core::ws_identity::{
    LocalIdentity, WS_HANDSHAKE_TIMEOUT_SECS, build_hello_reject, build_server_hello,
    load_local_identity, verify_client_hello,
};
use sa_core::ws_protocol::{
    AgentIdentity, ChatMode, ClientMessage, Event, EventKind, InitCompleted, InitFailed, InitMethod,
    InitMethodOption, InitRequired, InitializeConfigRequest, PlanAction, PlanStepStatus,
    QuestionMode, ServerMessage, UserQuestion, UserQuestionAnswer, UserVisibleFile,
    UserVisibleFileEncoding,
};
use sa_core::workflow::NodeStatus;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex as AsyncMutex, Semaphore, broadcast, mpsc, oneshot};
use tracing::Level;
use uuid::Uuid;

/// Max number of events buffered in memory.
///
/// The buffer is used so a CLI can reconnect and request event history.
const MAX_BUFFERED_EVENTS: usize = 10_000;

/// Max number of recent `Show` payloads kept for reconnecting clients.
const MAX_BUFFERED_SHOWS: usize = 32;

/// Close-frame reason used after a handshake rejection.
const WS_HANDSHAKE_REJECT_CLOSE_REASON: &str = "sa handshake rejected";

/// Default listen address used when `sa.toml` does not exist yet.
const DEFAULT_BOOTSTRAP_BIND: &str = "127.0.0.1:8765";

/// Default WS path used when `sa.toml` does not exist yet.
const DEFAULT_BOOTSTRAP_WS_PATH: &str = "/ws";

/// CLI arguments.
#[derive(clap::Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to `sa.toml`.
    #[arg(long, default_value = r"E:\SA\SAP-6.2\sa\sa.toml")]
    config: PathBuf,
}

/// Bootstrap metadata kept while the backend waits for the first config file.
#[derive(Debug, Clone)]
struct BootstrapState {
    /// Final path where the generated `sa.toml` should be written.
    config_path: PathBuf,
    /// Directory containing the future config file.
    config_dir: PathBuf,
    /// Workspace root the generated config should point at.
    workspace_root: PathBuf,
    /// `Agents.md` path the generated config should point at.
    agents_md_path: PathBuf,
    /// Bind address used by the already running bootstrap listener.
    bind: String,
    /// WS path used by the already running bootstrap listener.
    ws_path: String,
}

impl BootstrapState {
    /// Build the `init_required` payload shown by the frontend onboarding page.
    fn init_required_message(&self) -> ServerMessage {
        ServerMessage::InitRequired {
            request: InitRequired {
                config_path: self.config_path.display().to_string(),
                workspace_root: self.workspace_root.display().to_string(),
                agents_md_path: self.agents_md_path.display().to_string(),
                methods: vec![
                    InitMethodOption {
                        id: InitMethod::OpenAiCompatible,
                        label: "OpenAI / 兼容接口".to_string(),
                        description:
                            "适用于 OpenAI 兼容网关、常见 Bearer 鉴权接口，以及默认 chat_completions 路径。"
                                .to_string(),
                        default_wire_api: WireApi::ChatCompletions,
                        default_auth_style: AuthStyle::Bearer,
                        default_system_role_name: "system".to_string(),
                        recommended: true,
                    },
                    InitMethodOption {
                        id: InitMethod::AnthropicCompatible,
                        label: "Claude / Anthropic".to_string(),
                        description:
                            "适用于 `/v1/messages` 风格接口，默认使用 anthropic_auto 鉴权策略。"
                                .to_string(),
                        default_wire_api: WireApi::AnthropicMessages,
                        default_auth_style: AuthStyle::AnthropicAuto,
                        default_system_role_name: "system".to_string(),
                        recommended: false,
                    },
                    InitMethodOption {
                        id: InitMethod::Custom,
                        label: "高级自定义".to_string(),
                        description:
                            "手动指定 wire_api、auth_style 与 system_role_name，适合特殊供应商或网关。"
                                .to_string(),
                        default_wire_api: WireApi::Responses,
                        default_auth_style: AuthStyle::Bearer,
                        default_system_role_name: "system".to_string(),
                        recommended: false,
                    },
                ],
                recommended_method: InitMethod::OpenAiCompatible,
            },
        }
    }
}

/// Runtime state shared by every WS connection.
#[derive(Debug, Clone)]
enum RuntimeState {
    /// First-run bootstrap mode without `sa.toml`.
    Bootstrap(BootstrapState),
    /// Fully initialized backend runtime.
    Ready(Arc<Hub>),
}

/// Top-level server state shared across all connections.
#[derive(Debug)]
struct ServerState {
    /// Stable machine identity used for WS proofs.
    ws_identity: LocalIdentity,
    /// Current runtime mode.
    runtime: AsyncMutex<RuntimeState>,
}

/// A single queued task.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TaskRequest {
    /// Task id (UUID).
    task_id: Uuid,
    /// Task text.
    task: String,
}

/// Information about the task currently being executed by the worker loop.
///
/// This enables cooperative cancellation: a WebSocket client can request an
/// interrupt, and the worker loop will stop at the next cancellation point.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CurrentTask {
    /// Currently running task id.
    task_id: Uuid,
    /// Cancellation handle for the task.
    cancel: CancelHandle,
    /// Follow-up user messages that should be injected into the same session
    /// once the current model/tool step reaches a safe boundary.
    follow_up_messages: Arc<Mutex<VecDeque<String>>>,
}

/// One question currently waiting for a user answer.
#[derive(Debug)]
struct PendingQuestionEntry {
    /// Agent currently blocked by this question.
    agent_id: Uuid,
    /// Work currently blocked by this question.
    work_id: Uuid,
    /// Assistant tool-call id that should receive the eventual tool-result
    /// message.
    tool_call_id: String,
    /// Structured question visible to clients.
    question: UserQuestion,
    /// Legacy one-shot waiter still used by the old single-task runner.
    answer_tx: Option<oneshot::Sender<UserQuestionAnswer>>,
}

/// Per-agent in-memory wake bookkeeping.
#[derive(Debug, Default, Clone, Copy)]
struct AgentWakeState {
    /// Whether one quantum for this agent is currently running or already
    /// queued into the supervisor.
    running_or_queued: bool,
    /// Whether another wake was requested while the current quantum was still
    /// running.
    rerun_requested: bool,
}

/// One actively executing agent quantum.
#[derive(Debug, Clone)]
struct RunningAgentExecution {
    /// Work currently being executed by the agent.
    work_id: Uuid,
    /// Cancellation handle for the in-flight quantum.
    cancel: CancelHandle,
}

/// Runtime components that can be replaced in place by hot reload.
#[derive(Debug, Clone)]
struct RuntimeReloadState {
    /// Current agent runner (model client + tools + prompt config).
    runner: AgentRunner,
    /// Current workspace tool context used for skills and path resolution.
    preload_ctx: ToolContext,
    /// Current `Agents.md` path injected into prompts.
    agents_md_path: PathBuf,
    /// Current team-level runtime configuration.
    team_cfg: TeamConfig,
    /// Current non-interactive permission policy.
    permissions: PermissionsConfig,
    /// Current global concurrent model-call ceiling.
    max_concurrent_model_calls: usize,
}

/// Fully prepared runtime returned from parsed config before the hub is
/// started or hot-reloaded.
#[derive(Debug, Clone)]
struct PreparedRuntime {
    /// Bind address selected by the config.
    bind: String,
    /// WS path selected by the config.
    ws_path: String,
    /// Canonical workspace root.
    workspace_root: PathBuf,
    /// Components that may be hot-swapped into an existing hub.
    reloadable: RuntimeReloadState,
    /// Dream scheduler manager built from the same config.
    dream_manager: DreamManager,
    /// Human-readable summary returned by `Reload`.
    reload_summary: String,
}

/// One live dream scheduler task.
#[derive(Debug)]
struct DreamSchedulerHandle {
    /// Cancellation handle used to stop the background loop on reload.
    cancel: CancelHandle,
    /// Spawned background task.
    task: tokio::task::JoinHandle<()>,
}

/// Shared daemon state.
///
/// This object is designed so:
/// - The agent worker loop can run without any WS connection.
/// - WS connections can subscribe to events and submit tasks.
#[derive(Debug)]
struct Hub {
    /// Task queue sender (worker loop receives from this).
    #[allow(dead_code)]
    task_tx: mpsc::Sender<TaskRequest>,

    /// Broadcast channel for live server messages (events + questions).
    events_tx: broadcast::Sender<ServerMessage>,

    /// In-memory event buffer for reconnect/history.
    events_buf: Mutex<VecDeque<Event>>,

    /// Monotonic event id generator.
    next_event_id: AtomicU64,

    /// Mapping from submit ids to effective durable work ids (idempotency for retries).
    ///
    /// This now covers both:
    /// - top-level tasks
    /// - queued follow-up messages sent while a task is still running
    seen_tasks: Mutex<HashMap<Uuid, Uuid>>,

    /// Currently running task (if any), so we can cancel it.
    #[allow(dead_code)]
    current_task: Mutex<Option<CurrentTask>>,

    /// Durable agent wake queue consumed by the supervisor loop.
    wake_tx: mpsc::UnboundedSender<Uuid>,

    /// In-memory deduplication state for queued/running agents.
    wake_state: AsyncMutex<HashMap<Uuid, AgentWakeState>>,

    /// Currently executing agent quanta that may be interrupted.
    running_agents: AsyncMutex<HashMap<Uuid, RunningAgentExecution>>,

    /// Per-agent session stores keyed by stable agent id.
    agent_session_stores: AsyncMutex<HashMap<Uuid, Arc<SessionStore>>>,

    /// Limits the number of concurrent model calls across the whole runtime.
    model_call_semaphore: RwLock<Arc<Semaphore>>,

    /// Questions waiting for a user answer.
    pending_questions: Mutex<Vec<PendingQuestionEntry>>,

    /// Recent files explicitly shown to the user.
    recent_shows: Mutex<VecDeque<UserVisibleFile>>,

    /// Durable append-only interaction history.
    interaction_store: Arc<InteractionStore>,
    /// Durable append-only background-task audit history.
    task_audit_store: Arc<TaskAuditStore>,

    /// Stable config file path used by hot reload.
    config_path: PathBuf,
    /// Stable bind address; changing it still requires a restart.
    bind: String,
    /// Stable WS path; changing it still requires a restart.
    ws_path: String,
    /// Stable canonical workspace root. Hot reload may not change this.
    workspace_root: PathBuf,

    /// Context used for safe path resolution when preloading files.
    preload_ctx: RwLock<ToolContext>,

    /// The agent runner (OpenAI + tools loop).
    runner: RwLock<AgentRunner>,

    /// Path to Agents.md (reloaded for every task).
    ///
    /// Why reload each task?
    /// - Users often edit `Agents.md` while the daemon is running.
    /// - Users may create `Agents.md` after the daemon starts.
    /// Reloading makes this behavior verifiable and fixes "Agents.md not loaded" confusion.
    agents_md_path: RwLock<PathBuf>,

    /// Durable top-level conversation store rooted at `workspace/sessions/`.
    #[allow(dead_code)]
    session_store: Arc<SessionStore>,
    /// Durable runtime metadata store rooted at `workspace/runtime/`.
    runtime_store: RuntimeStore,
    /// Current persisted team state snapshot.
    team_state: AsyncMutex<TeamState>,
    /// Team-level runtime tuning flags loaded from config.
    team_cfg: RwLock<TeamConfig>,
    /// Non-interactive permission policy loaded from config.
    permissions: RwLock<PermissionsConfig>,
    /// Current dream scheduler task, if enabled.
    dream_scheduler: Mutex<Option<DreamSchedulerHandle>>,

    /// Cache performance monitor for tracking token usage and cache hits.
    cache_monitor: Mutex<sa_core::cache_monitor::CacheMonitor>,
}

impl Hub {
    /// Create a new hub and spawn the background worker.
    fn new(
        config_path: PathBuf,
        bind: String,
        ws_path: String,
        workspace_root: PathBuf,
        reloadable: RuntimeReloadState,
        session_store: Arc<SessionStore>,
        interaction_store: Arc<InteractionStore>,
        task_audit_store: Arc<TaskAuditStore>,
        runtime_store: RuntimeStore,
        team_state: TeamState,
    ) -> Arc<Self> {
        // Task queue capacity (small but adequate for minimal agent).
        let (task_tx, task_rx) = mpsc::channel::<TaskRequest>(128);
        let (wake_tx, wake_rx) = mpsc::unbounded_channel::<Uuid>();

        // Broadcast event channel. Capacity controls how many events a slow
        // subscriber can lag behind before it gets a `Lagged` error.
        let (events_tx, _) = broadcast::channel::<ServerMessage>(1024);

        let root_session_store = Arc::clone(&session_store);
        let root_agent_id = team_state.root_agent_id;
        let mut session_stores = HashMap::new();
        session_stores.insert(root_agent_id, root_session_store);
        let max_concurrent_model_calls = reloadable.max_concurrent_model_calls.max(1);
        let model_name_for_cache = reloadable.runner.model_name().to_string();

        // Build hub.
        let hub = Arc::new(Self {
            task_tx,
            events_tx,
            events_buf: Mutex::new(VecDeque::new()),
            next_event_id: AtomicU64::new(0),
            seen_tasks: Mutex::new(HashMap::new()),
            current_task: Mutex::new(None),
            wake_tx,
            wake_state: AsyncMutex::new(HashMap::new()),
            running_agents: AsyncMutex::new(HashMap::new()),
            agent_session_stores: AsyncMutex::new(session_stores),
            model_call_semaphore: RwLock::new(Arc::new(Semaphore::new(max_concurrent_model_calls))),
            pending_questions: Mutex::new(Vec::new()),
            recent_shows: Mutex::new(VecDeque::new()),
            interaction_store,
            task_audit_store,
            config_path,
            bind,
            ws_path,
            workspace_root,
            preload_ctx: RwLock::new(reloadable.preload_ctx),
            runner: RwLock::new(reloadable.runner),
            agents_md_path: RwLock::new(reloadable.agents_md_path),
            session_store,
            runtime_store,
            team_state: AsyncMutex::new(team_state),
            team_cfg: RwLock::new(reloadable.team_cfg),
            permissions: RwLock::new(reloadable.permissions),
            dream_scheduler: Mutex::new(None),
            cache_monitor: Mutex::new(sa_core::cache_monitor::CacheMonitor::for_model(&model_name_for_cache)),
        });

        // Wire up the context info callback so the agent can push real-time
        // context window usage data to the frontend after each LLM response.
        {
            let hub_for_ctx = Arc::clone(&hub);
            let context_info_emit: EmitContextInfoFn =
                Arc::new(move |used_tokens, max_tokens, message_count| {
                    hub_for_ctx.broadcast_server_message(
                        ServerMessage::ContextInfo {
                            used_tokens,
                            max_tokens,
                            message_count,
                        },
                    );
                });
            hub.runner
                .write()
                .expect("runner rwlock poisoned")
                .set_context_info_emit(context_info_emit);
        }

        // Spawn the durable supervisor loop.
        let hub_clone = Arc::clone(&hub);
        tokio::spawn(async move {
            drop(task_rx);
            hub_clone.supervisor_loop(wake_rx).await;
        });

        hub
    }

    /// Return the current hot-swappable tool context snapshot.
    fn current_preload_ctx(&self) -> ToolContext {
        self.preload_ctx
            .read()
            .expect("preload_ctx rwlock poisoned")
            .clone()
    }

    /// Return the current hot-swappable runner snapshot.
    fn current_runner(&self) -> AgentRunner {
        self.runner.read().expect("runner rwlock poisoned").clone()
    }

    /// Return the current configured `Agents.md` path.
    fn current_agents_md_path(&self) -> PathBuf {
        self.agents_md_path
            .read()
            .expect("agents_md_path rwlock poisoned")
            .clone()
    }

    /// Return the current team configuration snapshot.
    fn current_team_cfg(&self) -> TeamConfig {
        self.team_cfg
            .read()
            .expect("team_cfg rwlock poisoned")
            .clone()
    }

    /// Return the current permission policy snapshot.
    fn current_permissions(&self) -> PermissionsConfig {
        self.permissions
            .read()
            .expect("permissions rwlock poisoned")
            .clone()
    }

    /// Return the current model-call semaphore snapshot.
    fn current_model_call_semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(
            &self
                .model_call_semaphore
                .read()
                .expect("model_call_semaphore rwlock poisoned"),
        )
    }

    /// Return the current runner config snapshot.
    fn current_runner_config(&self) -> AgentRunnerConfig {
        self.current_runner().config().clone()
    }

    /// Replace the background dream scheduler with one built from the latest
    /// config snapshot.
    fn replace_dream_scheduler(self: &Arc<Self>, dream_manager: DreamManager) {
        let mut guard = self
            .dream_scheduler
            .lock()
            .expect("dream_scheduler mutex poisoned");

        if let Some(existing) = guard.take() {
            existing.cancel.cancel();
            drop(existing.task);
        }

        if !dream_manager.config().enabled {
            tracing::info!("Dream scheduler disabled by config.");
            return;
        }

        let (cancel_handle, cancel_token) = cancel_pair();
        let hub = Arc::clone(self);
        let task = tokio::spawn(async move {
            run_dream_scheduler(hub, dream_manager, cancel_token).await;
        });
        *guard = Some(DreamSchedulerHandle {
            cancel: cancel_handle,
            task,
        });
    }

    /// Hot-reload the current runtime from `sa.toml` without restarting the
    /// process or dropping existing WS connections.
    async fn reload_runtime_from_disk(self: &Arc<Self>) -> anyhow::Result<String> {
        let cfg = load_config_from_file(&self.config_path)?;
        let prepared = prepare_runtime_from_config(cfg, &self.config_path).await?;

        if prepared.bind != self.bind {
            anyhow::bail!(
                "hot reload cannot change bind address from {} to {}; restart SA instead",
                self.bind,
                prepared.bind
            );
        }
        if prepared.ws_path != self.ws_path {
            anyhow::bail!(
                "hot reload cannot change ws_path from {} to {}; restart SA instead",
                self.ws_path,
                prepared.ws_path
            );
        }
        if prepared.workspace_root != self.workspace_root {
            anyhow::bail!(
                "hot reload cannot change workspace root from {} to {}; restart SA instead",
                self.workspace_root.display(),
                prepared.workspace_root.display()
            );
        }

        {
            *self.runner.write().expect("runner rwlock poisoned") = prepared.reloadable.runner;
            *self
                .preload_ctx
                .write()
                .expect("preload_ctx rwlock poisoned") = prepared.reloadable.preload_ctx;
            *self
                .agents_md_path
                .write()
                .expect("agents_md_path rwlock poisoned") = prepared.reloadable.agents_md_path;
            *self.team_cfg.write().expect("team_cfg rwlock poisoned") =
                prepared.reloadable.team_cfg;
            *self
                .permissions
                .write()
                .expect("permissions rwlock poisoned") = prepared.reloadable.permissions;
            *self
                .model_call_semaphore
                .write()
                .expect("model_call_semaphore rwlock poisoned") = Arc::new(Semaphore::new(
                prepared.reloadable.max_concurrent_model_calls.max(1),
            ));
        }

        // Re-set the context_info_emit callback on the newly replaced runner,
        // so context window usage continues to be broadcast after hot-reload.
        {
            let hub_for_ctx = Arc::clone(self);
            let context_info_emit: EmitContextInfoFn =
                Arc::new(move |used_tokens, max_tokens, message_count| {
                    hub_for_ctx.broadcast_server_message(
                        ServerMessage::ContextInfo {
                            used_tokens,
                            max_tokens,
                            message_count,
                        },
                    );
                });
            self.runner
                .write()
                .expect("runner rwlock poisoned")
                .set_context_info_emit(context_info_emit);
        }

        // Update cache monitor pricing to match the new model.
        {
            let new_model = self.runner.read().expect("runner rwlock poisoned").model_name().to_string();
            self.cache_monitor.lock().unwrap_or_else(|e| e.into_inner()).set_pricing(
                sa_core::cost_budget::ModelPricing::for_model(&new_model),
            );
        }

        self.replace_dream_scheduler(prepared.dream_manager);
        let root_agent_id = self.team_state.lock().await.root_agent_id;
        self.publish(
            EventKind::Log,
            root_agent_id,
            format!("Runtime hot reload applied: {}", prepared.reload_summary),
        );
        Ok(prepared.reload_summary)
    }

    /// Publish an event:
    /// - assign `event_id`
    /// - store in buffer
    /// - broadcast to subscribers
    fn publish(&self, kind: EventKind, task_id: Uuid, message: String) {
        self.publish_with_agent(kind, task_id, message, None);
    }

    /// Publish one event with optional structured sender identity metadata.
    fn publish_with_agent(
        &self,
        kind: EventKind,
        task_id: Uuid,
        message: String,
        agent: Option<AgentIdentity>,
    ) {
        // Monotonic id: start at 1.
        let event_id = self.next_event_id.fetch_add(1, Ordering::Relaxed) + 1;

        // Timestamp.
        let ts = chrono::Utc::now();

        // Build event.
        let event = Event {
            event_id,
            ts,
            task_id,
            kind,
            message,
            agent,
        };

        mirror_server_message(&ServerMessage::Event {
            event: event.clone(),
        });

        // Store in buffer.
        //
        // NOTE: We use a standard mutex here because `publish` must be callable
        // from a non-async callback (`EmitEventFn`). We keep the critical
        // section small to minimize contention.
        {
            let mut buf = self.events_buf.lock().expect("events_buf mutex poisoned");
            buf.push_back(event.clone());
            while buf.len() > MAX_BUFFERED_EVENTS {
                buf.pop_front();
            }
        }

        // Broadcast (ignore errors if no receivers).
        let _ = self.events_tx.send(ServerMessage::Event { event });
    }

    /// Broadcast a non-history server message to all connected clients.
    fn broadcast_server_message(&self, msg: ServerMessage) {
        mirror_server_message(&msg);
        match &msg {
            ServerMessage::MemoryUpdated { source, report } => {
                tracing::info!("[broadcast] MemoryUpdated source={} report={:?}", source, report);
            }
            _ => {}
        }
        let _ = self.events_tx.send(msg);
    }

    /// Broadcast current cache performance statistics to all clients.
    fn broadcast_cache_stats(&self) {
        let monitor = self
            .cache_monitor
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let report = monitor.generate_report();
        let model_breakdown = monitor
            .get_model_breakdown()
            .into_iter()
            .map(|(model, stats)| sa_core::ws_protocol::CacheModelStats {
                model,
                requests: stats.requests,
                cache_hits: stats.cache_hits,
                cache_misses: stats.cache_misses,
                output_tokens: stats.output_tokens,
                hit_rate: stats.hit_rate,
            })
            .collect();

        self.broadcast_server_message(ServerMessage::CacheStats {
            total_requests: report.total_requests,
            total_cache_hits: report.total_cache_hit_tokens,
            total_cache_misses: report.total_cache_miss_tokens,
            total_output_tokens: report.total_output_tokens,
            hit_rate: report.average_hit_rate,
            estimated_savings: report.total_estimated_savings_usd,
            model_breakdown,
        });
    }

    /// Request interruption (cancellation) of a running task.
    ///
    /// This is called from the WS handler when a client sends:
    /// `{"type":"interrupt","task_id":"..."}`
    async fn request_interrupt(self: &Arc<Self>, task_id: Uuid) -> anyhow::Result<()> {
        let team_state = self.team_state.lock().await.clone();
        let mut target_agent_ids = self
            .runtime_store
            .list_agent_states()?
            .into_iter()
            .filter(|state| state.active_work_id == Some(task_id))
            .map(|state| state.agent_id)
            .collect::<HashSet<_>>();

        if let Some(input_owner_agent_id) = team_state.input_owner_agent_id
            && input_owner_agent_id != team_state.root_agent_id
        {
            let root_state = self
                .runtime_store
                .load_agent_state(team_state.root_agent_id)?;
            let input_state = self.runtime_store.load_agent_state(input_owner_agent_id)?;
            let child_holds_task =
                input_state.as_ref().and_then(|state| state.active_work_id) == Some(task_id);
            let root_holds_task =
                root_state.as_ref().and_then(|state| state.active_work_id) == Some(task_id);
            if root_holds_task {
                target_agent_ids.insert(team_state.root_agent_id);
            }
            if child_holds_task {
                target_agent_ids.insert(input_owner_agent_id);
                self.set_input_owner(team_state.root_agent_id, Some(task_id), "interrupt")
                    .await?;
            }
        }

        if target_agent_ids.is_empty() {
            self.publish(
                EventKind::Error,
                task_id,
                "Interrupt requested, but no matching active work was found.".to_string(),
            );
            return Ok(());
        }

        let running_agents = self.running_agents.lock().await.clone();
        for agent_id in &target_agent_ids {
            if let Some(execution) = running_agents.get(agent_id) {
                execution.cancel.cancel();
                self.publish(
                    EventKind::Log,
                    task_id,
                    format!(
                        "Interrupt requested; cancelling agent {} work {}.",
                        agent_id, execution.work_id
                    ),
                );
            }
        }
        drop(running_agents);

        for agent_id in target_agent_ids {
            let Some(mut state) = self.runtime_store.load_agent_state(agent_id)? else {
                continue;
            };

            if matches!(state.status, AgentStatus::WaitingUser) {
                self.runtime_store.clear_pending_question(agent_id)?;
                let mut resolved_question_ids = Vec::<Uuid>::new();
                {
                    let mut pending = self
                        .pending_questions
                        .lock()
                        .expect("pending_questions mutex poisoned");
                    pending.retain(|entry| {
                        if entry.agent_id == agent_id {
                            resolved_question_ids.push(entry.question.question_id);
                            false
                        } else {
                            true
                        }
                    });
                }
                for question_id in resolved_question_ids {
                    self.broadcast_server_message(ServerMessage::QuestionResolved { question_id });
                }
            }

            let _ = self
                .cancel_active_work(&mut state, "user interrupt")
                .await?;
        }

        Ok(())
    }

    /// Return all buffered events with `event_id > from_event_id`.
    async fn history_since(&self, from_event_id: u64) -> Vec<Event> {
        let buf = self.events_buf.lock().expect("events_buf mutex poisoned");
        buf.iter()
            .filter(|e| e.event_id > from_event_id)
            .cloned()
            .collect()
    }

    /// Snapshot the currently pending questions for reconnecting clients.
    fn pending_questions_snapshot(&self) -> Vec<UserQuestion> {
        let pending = self
            .pending_questions
            .lock()
            .expect("pending_questions mutex poisoned");
        pending.iter().map(|entry| entry.question.clone()).collect()
    }

    /// Snapshot the most recent `Show` payloads for reconnecting clients.
    fn recent_shows_snapshot(&self) -> Vec<UserVisibleFile> {
        let shows = self
            .recent_shows
            .lock()
            .expect("recent_shows mutex poisoned");
        shows.iter().cloned().collect()
    }

    /// Store and broadcast a user-visible file payload.
    fn publish_show(&self, file: UserVisibleFile) {
        {
            let mut shows = self
                .recent_shows
                .lock()
                .expect("recent_shows mutex poisoned");
            shows.push_back(file.clone());
            while shows.len() > MAX_BUFFERED_SHOWS {
                shows.pop_front();
            }
        }

        self.broadcast_server_message(ServerMessage::Show { file });
    }

    /// Append one entry to the durable interaction history.
    fn record_interaction(
        &self,
        work_id: Option<Uuid>,
        agent_id: Option<Uuid>,
        payload: InteractionPayload,
    ) -> anyhow::Result<Uuid> {
        Ok(self
            .interaction_store
            .append_new(work_id, agent_id, payload)?
            .id)
    }

    /// Return the best current task id for user-visible events emitted by one
    /// agent.
    #[allow(dead_code)]
    fn active_task_id_for_agent(state: &AgentState) -> Uuid {
        state.active_work_id.unwrap_or(state.agent_id)
    }

    /// Return the current root agent id.
    #[allow(dead_code)]
    async fn root_agent_id(&self) -> Uuid {
        self.team_state.lock().await.root_agent_id
    }

    /// Load one persisted agent state and fail if it is missing.
    fn load_agent_state_required(&self, agent_id: Uuid) -> anyhow::Result<AgentState> {
        self.runtime_store
            .load_agent_state(agent_id)?
            .with_context(|| format!("Agent state not found: {agent_id}"))
    }

    /// Get or lazily create the per-agent durable session store.
    async fn session_store_for_agent(
        self: &Arc<Self>,
        agent_id: Uuid,
    ) -> anyhow::Result<Arc<SessionStore>> {
        if let Some(existing) = self
            .agent_session_stores
            .lock()
            .await
            .get(&agent_id)
            .cloned()
        {
            return Ok(existing);
        }

        let relative_dir = PathBuf::from("sessions")
            .join("agents")
            .join(agent_id.to_string());
        let store = Arc::new(SessionStore::new_in_relative_dir(
            self.workspace_root.clone(),
            &relative_dir,
        )?);

        let mut guard = self.agent_session_stores.lock().await;
        Ok(guard
            .entry(agent_id)
            .or_insert_with(|| Arc::clone(&store))
            .clone())
    }

    /// Synchronize the persisted session-path metadata recorded inside one
    /// agent state.
    fn sync_agent_session_paths(
        &self,
        state: &mut AgentState,
        session_store: &SessionStore,
    ) -> anyhow::Result<()> {
        let snapshot = session_store.load_snapshot()?;
        state.current_session_path = snapshot.descriptor.current_session_path;
        state.previous_session_path = snapshot.descriptor.previous_session_path;
        Ok(())
    }

    /// Queue one agent for supervisor execution.
    async fn wake_agent(self: &Arc<Self>, agent_id: Uuid) -> anyhow::Result<()> {
        let mut wake_state = self.wake_state.lock().await;
        let entry = wake_state.entry(agent_id).or_default();
        if entry.running_or_queued {
            entry.rerun_requested = true;
            return Ok(());
        }

        entry.running_or_queued = true;
        entry.rerun_requested = false;
        drop(wake_state);

        self.wake_tx
            .send(agent_id)
            .map_err(|_| anyhow::anyhow!("Agent supervisor is no longer running"))
    }

    /// Complete one wake cycle and requeue the agent when needed.
    async fn finish_agent_wake_cycle(
        self: &Arc<Self>,
        agent_id: Uuid,
        force_rerun: bool,
    ) -> anyhow::Result<()> {
        let should_rerun = {
            let mut wake_state = self.wake_state.lock().await;
            let entry = wake_state.entry(agent_id).or_default();
            let should_rerun = force_rerun || entry.rerun_requested;
            if should_rerun {
                entry.rerun_requested = false;
            } else {
                entry.running_or_queued = false;
            }
            should_rerun
        };

        if should_rerun {
            self.wake_tx
                .send(agent_id)
                .map_err(|_| anyhow::anyhow!("Agent supervisor is no longer running"))?;
        }

        Ok(())
    }

    /// Persist a new input owner and emit a trace event for auditability.
    async fn set_input_owner(
        self: &Arc<Self>,
        input_owner_agent_id: Uuid,
        event_task_id: Option<Uuid>,
        reason: &str,
    ) -> anyhow::Result<()> {
        let mut team_state = self.team_state.lock().await;
        let previous_input_owner_agent_id = team_state.input_owner_agent_id;
        if team_state.input_owner_agent_id == Some(input_owner_agent_id) {
            return Ok(());
        }

        team_state.input_owner_agent_id = Some(input_owner_agent_id);
        team_state.updated_at = chrono::Utc::now();
        self.runtime_store.save_team_state(&team_state)?;
        let task_id = event_task_id
            .or_else(|| {
                self.runtime_store
                    .load_agent_state(team_state.root_agent_id)
                    .ok()
                    .flatten()
                    .and_then(|state| state.active_work_id)
            })
            .unwrap_or(team_state.root_agent_id);
        drop(team_state);

        let previous_input_owner = previous_input_owner_agent_id
            .and_then(|agent_id| self.runtime_store.load_agent_state(agent_id).ok().flatten());
        let new_input_owner = self.runtime_store.load_agent_state(input_owner_agent_id)?;

        if let Some(previous_input_owner) = previous_input_owner.as_ref()
            && previous_input_owner.agent_id != input_owner_agent_id
        {
            self.append_mailbox_message(
                previous_input_owner.agent_id,
                MailboxEntryKind::SystemNotice,
                None,
                Some("runtime".to_string()),
                format!(
                    "同学自由输入权限已从你这里移走，当前交给 `{}`（agent_id: {}）。后续同学输入不会再路由到你，除非主代理再次转交。",
                    if input_owner_agent_id == previous_input_owner.root_agent_id {
                        "学习委员".to_string()
                    } else {
                        new_input_owner
                            .as_ref()
                            .map(|state| state.label.clone())
                            .unwrap_or_else(|| input_owner_agent_id.to_string())
                    },
                    input_owner_agent_id
                ),
                previous_input_owner.active_work_id.or(event_task_id),
                Some(input_owner_agent_id),
            )
            .await?;
        }

        if let Some(new_input_owner) = new_input_owner.as_ref() {
            self.append_mailbox_message(
                new_input_owner.agent_id,
                MailboxEntryKind::SystemNotice,
                None,
                Some("runtime".to_string()),
                format!(
                    "同学自由输入权限已交给你。后续同学输入会继续路由到你，直到主代理显式收回或再次转交。来源：{}。",
                    reason
                ),
                new_input_owner.active_work_id.or(event_task_id),
                previous_input_owner_agent_id.or(event_task_id),
            )
            .await?;
        }

        self.publish(
            EventKind::Log,
            task_id,
            format!("Input ownership moved to agent {input_owner_agent_id} ({reason})."),
        );
        Ok(())
    }

    /// Return whether an agent is currently a valid free-form input owner.
    fn agent_can_hold_input(&self, state: &AgentState) -> bool {
        state.root_agent_id
            == self
                .runtime_store
                .load_root_marker()
                .ok()
                .flatten()
                .map(|marker| marker.root_agent_id)
                .unwrap_or(state.root_agent_id)
            && !matches!(
                state.status,
                AgentStatus::Deleted | AgentStatus::Deleting | AgentStatus::Failed
            )
            && (state.parent_agent_id.is_none() || state.allow_input_transfer_target)
    }

    /// Resolve the effective current input owner, falling back to root if the
    /// persisted owner is no longer valid.
    async fn effective_input_owner(self: &Arc<Self>) -> anyhow::Result<Uuid> {
        let mut team_state = self.team_state.lock().await;
        let root_agent_id = team_state.root_agent_id;
        let candidate = team_state.input_owner_agent_id.unwrap_or(root_agent_id);
        let Some(candidate_state) = self.runtime_store.load_agent_state(candidate)? else {
            team_state.input_owner_agent_id = Some(root_agent_id);
            team_state.updated_at = chrono::Utc::now();
            self.runtime_store.save_team_state(&team_state)?;
            return Ok(root_agent_id);
        };
        if !self.agent_can_hold_input(&candidate_state) {
            team_state.input_owner_agent_id = Some(root_agent_id);
            team_state.updated_at = chrono::Utc::now();
            self.runtime_store.save_team_state(&team_state)?;
            return Ok(root_agent_id);
        }
        Ok(candidate)
    }

    /// Compute the durable depth of one agent within its current root tree.
    fn agent_depth(&self, agent_id: Uuid) -> anyhow::Result<u32> {
        let mut depth = 0u32;
        let mut cursor = self
            .runtime_store
            .load_agent_state(agent_id)?
            .and_then(|state| state.parent_agent_id);
        while let Some(parent_id) = cursor {
            depth = depth.saturating_add(1);
            cursor = self
                .runtime_store
                .load_agent_state(parent_id)?
                .and_then(|state| state.parent_agent_id);
        }
        Ok(depth)
    }

    /// Mark that one work emitted direct user-visible output.
    async fn mark_work_user_output(&self, agent_id: Uuid, work_id: Uuid) -> anyhow::Result<()> {
        let Some(mut state) = self.runtime_store.load_agent_state(agent_id)? else {
            return Ok(());
        };
        if state.active_work_id != Some(work_id) {
            return Ok(());
        }
        if state.work_has_user_output {
            return Ok(());
        }
        state.work_has_user_output = true;
        self.runtime_store.save_agent_state(&state)
    }

    /// Mark that one child work explicitly talked back to its parent.
    async fn mark_work_parent_message(&self, agent_id: Uuid, work_id: Uuid) -> anyhow::Result<()> {
        let Some(mut state) = self.runtime_store.load_agent_state(agent_id)? else {
            return Ok(());
        };
        if state.active_work_id != Some(work_id) {
            return Ok(());
        }
        if state.work_has_parent_message {
            return Ok(());
        }
        state.work_has_parent_message = true;
        self.runtime_store.save_agent_state(&state)
    }

    /// Append one durable mailbox entry and optionally wake the target agent.
    async fn append_mailbox_message(
        self: &Arc<Self>,
        target_agent_id: Uuid,
        kind: MailboxEntryKind,
        from_agent_id: Option<Uuid>,
        from_label: Option<String>,
        message: String,
        work_id: Option<Uuid>,
        related_id: Option<Uuid>,
    ) -> anyhow::Result<MailboxEntry> {
        let entry = self.runtime_store.append_mailbox_entry(
            target_agent_id,
            MailboxEntry {
                offset: 0,
                entry_id: Uuid::new_v4(),
                created_at: chrono::Utc::now(),
                kind,
                from_agent_id,
                from_label,
                message,
                work_id,
                related_id,
            },
        )?;

        if let Some(state) = self.runtime_store.load_agent_state(target_agent_id)? {
            if !matches!(
                state.status,
                AgentStatus::WaitingUser | AgentStatus::WaitingDependency | AgentStatus::Deleted
            ) {
                self.wake_agent(target_agent_id).await?;
            }
        }

        Ok(entry)
    }

    /// Convert one agent state into the public tool-level summary shape.
    fn agent_info_from_state(state: &AgentState) -> AgentInfo {
        AgentInfo {
            agent_id: state.agent_id,
            parent_agent_id: state.parent_agent_id,
            label: state.label.clone(),
            status: state.status,
            active_work_id: state.active_work_id,
            allow_user_send: state.allow_user_send,
            allow_user_show: state.allow_user_show,
            allow_user_ask: state.allow_user_ask,
        }
    }

    /// Build stable frontend-facing identity metadata for one agent.
    fn frontend_identity_from_state(state: &AgentState) -> AgentIdentity {
        Self::frontend_identity(
            state.agent_id,
            state.label.clone(),
            state.parent_agent_id.is_none(),
        )
    }

    /// Build stable frontend-facing identity metadata from runtime fields.
    fn frontend_identity(agent_id: Uuid, agent_label: String, is_root: bool) -> AgentIdentity {
        let display_name = if is_root {
            "学习委员".to_string()
        } else {
            agent_label.clone()
        };
        AgentIdentity {
            agent_id,
            agent_label,
            display_name,
            is_root,
        }
    }

    /// Return whether `candidate_agent_id` is a descendant of `ancestor_id`.
    fn is_descendant_of(
        states: &HashMap<Uuid, AgentState>,
        candidate_agent_id: Uuid,
        ancestor_id: Uuid,
    ) -> bool {
        let mut cursor = states
            .get(&candidate_agent_id)
            .and_then(|state| state.parent_agent_id);
        while let Some(parent_id) = cursor {
            if parent_id == ancestor_id {
                return true;
            }
            cursor = states
                .get(&parent_id)
                .and_then(|state| state.parent_agent_id);
        }
        false
    }

    /// Resolve one scope query under the current root tree.
    fn resolve_agent_scope(
        &self,
        requester: &AgentState,
        scope: AgentScope,
        include_requester: bool,
    ) -> anyhow::Result<Vec<AgentState>> {
        let states = self.runtime_store.list_agent_states()?;
        let state_map = states
            .iter()
            .cloned()
            .map(|state| (state.agent_id, state))
            .collect::<HashMap<_, _>>();

        let mut out = Vec::new();
        for state in states {
            if state.root_agent_id != requester.root_agent_id {
                continue;
            }
            if !include_requester && state.agent_id == requester.agent_id {
                continue;
            }
            if matches!(state.status, AgentStatus::Deleted) {
                continue;
            }

            let visible = match scope {
                AgentScope::Children => state.parent_agent_id == Some(requester.agent_id),
                AgentScope::Descendants => {
                    Self::is_descendant_of(&state_map, state.agent_id, requester.agent_id)
                }
                AgentScope::Siblings => {
                    requester.parent_agent_id.is_some()
                        && state.parent_agent_id == requester.parent_agent_id
                }
                AgentScope::AllUnderRoot => true,
            };

            if visible {
                out.push(state);
            }
        }

        out.sort_by_key(|state| state.agent_id);
        Ok(out)
    }

    /// Return a human-readable notice when a wait dependency is already
    /// satisfied.
    fn wait_satisfied_notice(
        &self,
        waiting_on: &WaitingDependency,
    ) -> anyhow::Result<Option<String>> {
        let now = chrono::Utc::now();
        if let Some(timeout_at) = waiting_on.timeout_at
            && timeout_at <= now
        {
            return Ok(Some(format!(
                "Wait for {:?} {} timed out at {}.",
                waiting_on.kind, waiting_on.id, timeout_at
            )));
        }

        match waiting_on.kind {
            WaitKind::Agent => {
                let Some(state) = self.runtime_store.load_agent_state(waiting_on.id)? else {
                    return Ok(Some(format!(
                        "Wait target agent {} no longer exists.",
                        waiting_on.id
                    )));
                };
                if matches!(state.status, AgentStatus::Idle | AgentStatus::Deleted) {
                    return Ok(Some(format!(
                        "Agent {} reached status {:?}.",
                        state.agent_id, state.status
                    )));
                }
            }
            WaitKind::Work => {
                if let Some(work) = self.runtime_store.load_work_state(waiting_on.id)? {
                    if matches!(
                        work.status,
                        RuntimeWorkStatus::Finished | RuntimeWorkStatus::Cancelled
                    ) {
                        return Ok(Some(format!(
                            "Work {} completed with status {:?}.",
                            waiting_on.id, work.status
                        )));
                    }
                }
            }
            WaitKind::Task => {
                let Some(task) = self.runtime_store.load_task_state(waiting_on.id)? else {
                    return Ok(Some(format!(
                        "Runtime task {} no longer exists.",
                        waiting_on.id
                    )));
                };
                if !matches!(task.status, RuntimeTaskStatus::Running) {
                    return Ok(Some(format!(
                        "Runtime task {} completed with status {:?}.",
                        task.task_id, task.status
                    )));
                }
            }
        }

        Ok(None)
    }

    /// Schedule a timeout wake-up for one waiting agent.
    fn schedule_wait_timeout(self: &Arc<Self>, agent_id: Uuid, waiting_on: WaitingDependency) {
        let Some(timeout_at) = waiting_on.timeout_at else {
            return;
        };
        let hub = Arc::clone(self);
        tokio::spawn(async move {
            let now = chrono::Utc::now();
            let sleep_for = if timeout_at <= now {
                std::time::Duration::from_secs(0)
            } else {
                (timeout_at - now)
                    .to_std()
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            };
            tokio::time::sleep(sleep_for).await;
            if let Err(err) = hub.handle_wait_timeout(agent_id, waiting_on).await {
                tracing::warn!("Failed to process wait timeout: {err:#}");
            }
        });
    }

    /// Wake one agent whose wait timed out and inject a notice into its
    /// mailbox.
    async fn handle_wait_timeout(
        self: &Arc<Self>,
        agent_id: Uuid,
        waiting_on: WaitingDependency,
    ) -> anyhow::Result<()> {
        let Some(mut state) = self.runtime_store.load_agent_state(agent_id)? else {
            return Ok(());
        };
        if state.waiting_on.as_ref() != Some(&waiting_on) {
            return Ok(());
        }

        state.waiting_on = None;
        state.status = AgentStatus::Idle;
        self.runtime_store.save_agent_state(&state)?;
        self.append_mailbox_message(
            agent_id,
            MailboxEntryKind::SystemNotice,
            None,
            Some("runtime".to_string()),
            format!(
                "Wait for {:?} {} timed out{}.",
                waiting_on.kind,
                waiting_on.id,
                waiting_on
                    .timeout_at
                    .map(|at| format!(" at {at}"))
                    .unwrap_or_default()
            ),
            state.active_work_id,
            Some(waiting_on.id),
        )
        .await?;
        Ok(())
    }

    /// Wake all agents currently waiting on one dependency that has just
    /// completed.
    async fn wake_waiters_for_dependency(
        self: &Arc<Self>,
        kind: WaitKind,
        id: Uuid,
        message: String,
    ) -> anyhow::Result<()> {
        for mut state in self.runtime_store.list_agent_states()? {
            let Some(waiting_on) = state.waiting_on.clone() else {
                continue;
            };
            if waiting_on.kind != kind || waiting_on.id != id {
                continue;
            }

            state.waiting_on = None;
            state.status = AgentStatus::Idle;
            self.runtime_store.save_agent_state(&state)?;
            self.append_mailbox_message(
                state.agent_id,
                MailboxEntryKind::SystemNotice,
                None,
                Some("runtime".to_string()),
                message.clone(),
                state.active_work_id,
                Some(id),
            )
            .await?;
        }

        Ok(())
    }

    /// Finalize one active work as cancelled and wake any dependent waiters.
    async fn cancel_active_work(
        self: &Arc<Self>,
        state: &mut AgentState,
        reason: &str,
    ) -> anyhow::Result<Option<Uuid>> {
        let Some(work_id) = state.active_work_id else {
            state.status = AgentStatus::Idle;
            state.waiting_on = None;
            state.needs_finish_reminder = false;
            self.runtime_store.save_agent_state(state)?;
            return Ok(None);
        };
        let work_summary = state
            .active_work_summary
            .clone()
            .unwrap_or_else(|| "(cancelled work)".to_string());
        let started_at = state.active_started_at.unwrap_or_else(chrono::Utc::now);
        let finished_at = chrono::Utc::now();

        state.status = AgentStatus::Idle;
        state.active_work_id = None;
        state.active_work_summary = None;
        state.active_started_at = None;
        state.work_has_user_output = false;
        state.work_has_parent_message = false;
        state.parent_work_id = None;
        state.needs_finish_reminder = false;
        state.waiting_on = None;
        state.pending_finish_confirmation = None;
        state.pending_control = None;
        self.runtime_store.clear_pending_question(state.agent_id)?;
        self.runtime_store.save_agent_state(state)?;
        self.persist_terminal_work(
            state.agent_id,
            state.root_agent_id,
            work_id,
            work_summary,
            started_at,
            RuntimeWorkStatus::Cancelled,
            finished_at,
            format!("Cancelled: {}", reason.trim()),
        )?;

        self.publish(
            EventKind::Final,
            work_id,
            format!("WORK_CANCELLED\nreason: {}", reason.trim()),
        );
        self.wake_waiters_for_dependency(
            WaitKind::Agent,
            state.agent_id,
            format!("Agent {} became idle after cancellation.", state.agent_id),
        )
        .await?;
        self.wake_waiters_for_dependency(
            WaitKind::Work,
            work_id,
            format!("Work {work_id} was cancelled: {}.", reason.trim()),
        )
        .await?;

        Ok(Some(work_id))
    }

    /// Build a short human-readable work summary from free-form text.
    fn summarize_work_text(text: &str) -> String {
        let first_line = text
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("(empty work)");
        let mut summary = first_line.trim().to_string();
        if summary.len() > 120 {
            let mut safe_boundary = 120usize;
            while safe_boundary > 0 && !summary.is_char_boundary(safe_boundary) {
                safe_boundary -= 1;
            }
            summary.truncate(safe_boundary);
            summary.push_str("...");
        }
        summary
    }

    /// Persist one newly started work into the durable work index.
    fn persist_started_work(
        &self,
        owner_agent_id: Uuid,
        root_agent_id: Uuid,
        work_id: Uuid,
        summary: String,
        started_at: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<()> {
        self.runtime_store.save_work_state(&RuntimeWorkState {
            work_id,
            owner_agent_id,
            root_agent_id,
            summary,
            status: RuntimeWorkStatus::Running,
            started_at,
            finished_at: None,
            result_summary: None,
        })
    }

    /// Persist a terminal work state.
    fn persist_terminal_work(
        &self,
        owner_agent_id: Uuid,
        root_agent_id: Uuid,
        work_id: Uuid,
        summary: String,
        started_at: chrono::DateTime<chrono::Utc>,
        status: RuntimeWorkStatus,
        finished_at: chrono::DateTime<chrono::Utc>,
        result_summary: String,
    ) -> anyhow::Result<()> {
        self.runtime_store.save_work_state(&RuntimeWorkState {
            work_id,
            owner_agent_id,
            root_agent_id,
            summary,
            status,
            started_at,
            finished_at: Some(finished_at),
            result_summary: Some(result_summary),
        })
    }

    /// Convert one durable mailbox entry into a chat-style user turn.
    ///
    /// If the message body contains the sentinel `__IMAGE__:data:...`, a
    /// multimodal user message with a `ContentPart::Image` is returned instead
    /// of a plain-text message.
    fn mailbox_entry_to_message(entry: &MailboxEntry) -> ChatMessage {
        // Check for image sentinel injected by `inject_user_image`.
        if let Some(image_start) = entry.message.find("__IMAGE__:") {
            // Extract the text before the sentinel as caption/prompt.
            let caption = entry.message[..image_start].trim();
            let rest = &entry.message[image_start..];

            // Parse `__IMAGE__:data:image/...;base64,...\n__IMAGE_PATH__:/path/to/file`
            let data_url = if let Some(line_end) = rest.find('\n') {
                &rest["__IMAGE__:".len()..line_end]
            } else {
                &rest["__IMAGE__:".len()..]
            };

            let prompt = if caption.is_empty() {
                "同学上传了一张图片，请分析。".to_string()
            } else {
                format!("同学上传了一张图片：{caption}")
            };

            let parts = vec![
                sa_core::openai::ContentPart::Text { text: prompt },
                sa_core::openai::ContentPart::Image {
                    image_url: sa_core::openai::ImageUrl {
                        url: data_url.to_string(),
                        detail: Some("auto".to_string()),
                    },
                },
            ];
            return ChatMessage::multimodal("user", parts);
        }

        let content = match entry.kind {
            MailboxEntryKind::UserInput => entry.message.clone(),
            MailboxEntryKind::AgentMessage => match entry.from_label.as_deref() {
                Some(label) => format!("来自代理 `{label}` 的消息：\n{}", entry.message),
                None => format!("来自其他代理的消息：\n{}", entry.message),
            },
            MailboxEntryKind::BroadcastMessage => match entry.from_label.as_deref() {
                Some(label) => format!("来自代理 `{label}` 的广播：\n{}", entry.message),
                None => format!("来自其他代理的广播：\n{}", entry.message),
            },
            MailboxEntryKind::ChildFinished
            | MailboxEntryKind::TaskFinished
            | MailboxEntryKind::SystemNotice => entry.message.clone(),
        };
        ChatMessage::text("user", content)
    }

    /// Build one temporary runtime-only reminder block injected into the next
    /// request without persisting it into session history.
    fn temporary_runtime_reminder_block(state: &AgentState) -> Option<String> {
        if let Some(pending_finish) = state.pending_finish_confirmation.as_ref() {
            let header = match pending_finish.mode {
                PendingFinishMode::RootNeedsOutputOrConfirmation => {
                    "你刚刚调用了 `Finish(reason, result)`，但当前工作尚未通过 `Send` 或 `Show` 向同学输出任何内容。"
                }
                PendingFinishMode::ChildNeedsParentAckOrConfirmation => {
                    "你刚刚调用了 `Finish(reason, result)`，但当前工作尚未显式与父代理交互。"
                }
            };
            return Some(format!(
                "## 临时运行时提醒\n\n- {header}\n- 如果任务确实已经完成且不需要额外输出，请立即调用 `FinishWithoutOutput()`。\n- `FinishWithoutOutput()` 没有参数，只用于这一次确认；不要把它当作常规工具。\n- 如果任务还没有真正结束，请继续工作，并在准备好后重新调用 `Finish(reason, result)`。\n- 上一次暂存的 Finish reason：`{}`\n- 上一次暂存的 Finish result：`{}`",
                pending_finish.reason.trim(),
                pending_finish.result.trim(),
            ));
        }

        if state.needs_finish_reminder {
            return Some(
                "## 临时运行时提醒\n\n- 你上一轮没有调用任何工具，也没有调用 `Finish`。\n- 如果任务已经结束，请调用 `Finish(reason, result)`。\n- 如果任务尚未结束，请继续使用工具推进，不要空转。".to_string(),
            );
        }

        None
    }

    /// Build the extra prompt block for one durable agent quantum.
    async fn build_agent_extra_prompt(
        &self,
        state: &AgentState,
        task_id: Uuid,
        agents_md: &sa_core::agents_md::AgentsMd,
    ) -> String {
        let mut blocks = Vec::<String>::new();

        if matches!(state.kind, AgentKind::Root) {
            blocks.push(self.memory_prompt_block(task_id).await);
        }
        if matches!(state.kind, AgentKind::Worker) {
            blocks.extend(self.load_subagent_prompt_blocks(state, task_id).await);
        }
        blocks.push(self.preload_agents_md_references(agents_md).await);
        if !state.active_command_invocations.is_empty() {
            let mut block = String::from("## 已激活命令上下文\n\n");
            block.push_str(
                "以下命令已经在当前代理上下文中激活。相关要求、作用域工具限制、模型覆盖会继续生效，直到后续工作自然脱离这些上下文。\n\n",
            );
            for command in &state.active_command_invocations {
                block.push_str(&format!(
                    "- `{}`：{}\n",
                    command.name,
                    command.description.trim()
                ));
                if let Some(when_to_use) = command.when_to_use.as_deref() {
                    block.push_str(&format!("  适用时机：{}\n", when_to_use.trim()));
                }
                if !command.allowed_tools.is_empty() {
                    block.push_str(&format!(
                        "  当前允许工具：{}\n",
                        command.allowed_tools.join(", ")
                    ));
                }
                if let Some(model) = command.model_override.as_deref() {
                    block.push_str(&format!("  当前模型覆盖：`{model}`\n"));
                }
                if let Some(effort) = command.effort_override.as_deref() {
                    block.push_str(&format!("  当前思维深度覆盖：`{effort}`\n"));
                }
            }
            block.push_str("\n如果这些命令带有明确步骤、限制或输出要求，你必须继续遵守。");
            blocks.push(block);
        }

        if let Some(reminder) = Self::temporary_runtime_reminder_block(state) {
            blocks.push(reminder);
        }

        blocks.join("\n\n")
    }

    /// Load worker-only prompt blocks such as persona skill, persona file, and
    /// parent-provided prompt text.
    async fn load_subagent_prompt_blocks(&self, state: &AgentState, task_id: Uuid) -> Vec<String> {
        let mut blocks = Vec::<String>::new();
        let preload_ctx = self.current_preload_ctx();

        if let Some(skill_name) = state.subagent_prompt_skill.as_deref() {
            match preload_ctx.skills.load_skill_file(skill_name, None).await {
                Ok((path, content)) => blocks.push(format!(
                    "## 子代理主人格 Skill\n\n- 名称：`{skill_name}`\n- 文件：`{path}`\n\n{content}",
                )),
                Err(err) => {
                    self.publish(
                        EventKind::Error,
                        task_id,
                        format!("Failed to load subagent prompt skill `{skill_name}`: {err}"),
                    );
                }
            }
        }

        if let Some(raw_path) = state.subagent_prompt_file.as_deref() {
            match self.load_workspace_prompt_file(raw_path).await {
                Ok(content) => blocks.push(format!(
                    "## 子代理人格提示词文件\n\n- 文件：`{}`\n\n{}",
                    raw_path.trim(),
                    content
                )),
                Err(err) => {
                    self.publish(
                        EventKind::Error,
                        task_id,
                        format!("Failed to load subagent prompt file `{raw_path}`: {err}"),
                    );
                }
            }
        }

        if let Some(prompt) = state.subagent_prompt.as_deref() {
            let trimmed = prompt.trim();
            if !trimmed.is_empty() {
                blocks.push(format!("## 父代理附加提示词\n\n{trimmed}"));
            }
        }

        blocks
    }

    /// Load one workspace text prompt file with the same workspace safety
    /// boundary used by other file tools.
    async fn load_workspace_prompt_file(&self, raw_path: &str) -> anyhow::Result<String> {
        let preload_ctx = self.current_preload_ctx();
        let path = preload_ctx.resolve_under_workspace(raw_path)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat prompt file: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "Prompt file must be a regular file, not a directory: {}",
                path.display()
            );
        }
        if meta.len() > 200_000 {
            anyhow::bail!(
                "Prompt file is too large to inject ({} bytes): {}",
                meta.len(),
                path.display()
            );
        }

        tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read prompt file: {}", path.display()))
    }

    /// Append one assistant control message into the durable session only when
    /// it is not already present.
    fn append_control_message_if_missing(
        &self,
        session_store: &SessionStore,
        message: &PendingAssistantMessage,
    ) -> anyhow::Result<()> {
        let snapshot = session_store.load_snapshot()?;
        let already_present = snapshot.messages.last().is_some_and(|last| {
            last.role == message.role
                && last.content == message.content
                && last.tool_calls == message.tool_calls
                && last.tool_call_id == message.tool_call_id
        });
        if !already_present {
            let chat_message = ChatMessage::from(message);
            session_store.append_message(&chat_message)?;
        }
        Ok(())
    }

    /// Apply one durable pending control checkpoint, including recovery after a
    /// crash that happened between model decision and runtime transition.
    async fn apply_pending_control(
        self: &Arc<Self>,
        state: &mut AgentState,
        session_store: &SessionStore,
    ) -> anyhow::Result<Option<bool>> {
        let Some(pending_control) = state.pending_control.clone() else {
            return Ok(None);
        };

        match pending_control {
            PendingControlAction::Ask {
                assistant_message,
                tool_call_id,
                prompt,
                mode,
                options_json,
                allow_free_text,
            } => {
                self.append_control_message_if_missing(session_store, &assistant_message)?;

                if self
                    .runtime_store
                    .load_pending_question(state.agent_id)?
                    .is_some()
                {
                    state.pending_control = None;
                    self.runtime_store.save_agent_state(state)?;
                    return Ok(Some(false));
                }

                let request = AskRequest {
                    prompt,
                    mode: serde_json::from_str(&mode)?,
                    options: serde_json::from_str(&options_json)?,
                    allow_free_text,
                };
                self.persist_pending_question(
                    state,
                    state.active_work_id.unwrap_or(state.agent_id),
                    tool_call_id,
                    request,
                )
                .await?;
                let mut refreshed = self.load_agent_state_required(state.agent_id)?;
                refreshed.pending_control = None;
                self.runtime_store.save_agent_state(&refreshed)?;
                *state = refreshed;
                Ok(Some(false))
            }
            PendingControlAction::Wait {
                assistant_message,
                target_kind,
                target_id,
                until,
                timeout_seconds,
            } => {
                self.append_control_message_if_missing(session_store, &assistant_message)?;
                let waiting_on = WaitingDependency {
                    kind: target_kind,
                    id: target_id,
                    until: until.unwrap_or(match target_kind {
                        WaitKind::Agent => WaitUntil::Idle,
                        WaitKind::Work => WaitUntil::Finished,
                        WaitKind::Task => WaitUntil::Exited,
                    }),
                    timeout_at: timeout_seconds.map(|seconds| {
                        chrono::Utc::now() + chrono::Duration::seconds(seconds as i64)
                    }),
                };

                if let Some(notice) = self.wait_satisfied_notice(&waiting_on)? {
                    state.status = AgentStatus::Idle;
                    state.waiting_on = None;
                    state.pending_control = None;
                    self.runtime_store.save_agent_state(state)?;
                    self.append_mailbox_message(
                        state.agent_id,
                        MailboxEntryKind::SystemNotice,
                        None,
                        Some("runtime".to_string()),
                        notice,
                        state.active_work_id,
                        Some(waiting_on.id),
                    )
                    .await?;
                    return Ok(Some(true));
                }

                state.status = AgentStatus::WaitingDependency;
                state.waiting_on = Some(waiting_on.clone());
                state.pending_control = None;
                state.needs_finish_reminder = false;
                self.runtime_store.save_agent_state(state)?;
                self.schedule_wait_timeout(state.agent_id, waiting_on);
                Ok(Some(false))
            }
            PendingControlAction::Finish {
                assistant_message,
                reason,
                result,
                without_output_confirmation,
            } => {
                self.append_control_message_if_missing(session_store, &assistant_message)?;
                if state.active_work_id.is_none() {
                    state.pending_control = None;
                    self.runtime_store.save_agent_state(state)?;
                    return Ok(Some(false));
                }
                let had_pending_finish_confirmation = state.pending_finish_confirmation.is_some();
                let rerun = self
                    .handle_finish_outcome(state, reason, result, without_output_confirmation)
                    .await?;
                let mut refreshed = self.load_agent_state_required(state.agent_id)?;
                refreshed.pending_control = None;
                self.runtime_store.save_agent_state(&refreshed)?;
                *state = refreshed;
                Ok(Some(rerun && !had_pending_finish_confirmation))
            }
        }
    }

    /// Persist one runtime-level structured question and surface it to all
    /// connected frontends.
    async fn persist_pending_question(
        self: &Arc<Self>,
        state: &mut AgentState,
        work_id: Uuid,
        tool_call_id: String,
        request: AskRequest,
    ) -> anyhow::Result<()> {
        let question_id = Uuid::new_v4();
        let created_at = chrono::Utc::now();
        let question = UserQuestion {
            question_id,
            task_id: work_id,
            created_at,
            agent: Some(Self::frontend_identity_from_state(state)),
            prompt: request.prompt.clone(),
            mode: request.mode.clone(),
            options: request.options.clone(),
            allow_free_text: request.allow_free_text || request.mode == QuestionMode::Text,
        };
        let question_state = PendingQuestionState {
            agent_id: state.agent_id,
            work_id,
            question_id,
            tool_call_id: tool_call_id.clone(),
            prompt: request.prompt,
            mode: serde_json::to_string(&question.mode)?,
            options_json: serde_json::to_string(&question.options)?,
            allow_free_text: question.allow_free_text,
            created_at,
        };

        self.runtime_store.save_pending_question(&question_state)?;
        state.status = AgentStatus::WaitingUser;
        state.needs_finish_reminder = false;
        self.runtime_store.save_agent_state(state)?;

        {
            let mut pending = self
                .pending_questions
                .lock()
                .expect("pending_questions mutex poisoned");
            pending.push(PendingQuestionEntry {
                agent_id: state.agent_id,
                work_id,
                tool_call_id,
                question: question.clone(),
                answer_tx: None,
            });
        }

        self.record_interaction(
            Some(work_id),
            Some(state.agent_id),
            InteractionPayload::Ask {
                question_id,
                prompt: question.prompt.clone(),
                mode: question.mode.clone(),
                options: question.options.clone(),
                allow_free_text: question.allow_free_text,
            },
        )?;
        self.broadcast_server_message(ServerMessage::Question { question });
        Ok(())
    }

    /// Apply the `Finish` / `FinishWithoutOutput` runtime semantics for one
    /// agent.
    async fn handle_finish_outcome(
        self: &Arc<Self>,
        state: &mut AgentState,
        reason: String,
        result: String,
        without_output_confirmation: bool,
    ) -> anyhow::Result<bool> {
        let work_id = state
            .active_work_id
            .ok_or_else(|| anyhow::anyhow!("Finish reached an agent without active work"))?;

        let (effective_reason, effective_result) = if without_output_confirmation {
            let pending = state.pending_finish_confirmation.clone().ok_or_else(|| {
                anyhow::anyhow!("FinishWithoutOutput was used without a pending confirmation")
            })?;
            (pending.reason, pending.result)
        } else {
            (reason, result)
        };

        let root_owner = self.team_state.lock().await.input_owner_agent_id;
        let root_needs_confirmation = state.parent_agent_id.is_none()
            && root_owner == Some(state.agent_id)
            && !state.work_has_user_output
            && !without_output_confirmation;
        let child_needs_confirmation = state.parent_agent_id.is_some()
            && !state.work_has_parent_message
            && !without_output_confirmation;

        if root_needs_confirmation || child_needs_confirmation {
            state.status = AgentStatus::Idle;
            state.pending_finish_confirmation = Some(PendingFinishConfirmation {
                reason: effective_reason,
                result: effective_result,
                mode: if root_needs_confirmation {
                    PendingFinishMode::RootNeedsOutputOrConfirmation
                } else {
                    PendingFinishMode::ChildNeedsParentAckOrConfirmation
                },
            });
            state.needs_finish_reminder = false;
            self.runtime_store.save_agent_state(state)?;
            return Ok(true);
        }

        let parent_agent_id = state.parent_agent_id;
        let parent_work_id = state.parent_work_id;
        let agent_id = state.agent_id;
        let agent_label = state.label.clone();
        let root_agent_id = state.root_agent_id;
        let finished_at = chrono::Utc::now();
        let work_summary = state
            .active_work_summary
            .clone()
            .unwrap_or_else(|| "(finished work)".to_string());
        let started_at = state.active_started_at.unwrap_or_else(chrono::Utc::now);

        state.status = AgentStatus::Idle;
        state.active_work_id = None;
        state.active_work_summary = None;
        state.active_started_at = None;
        state.work_has_user_output = false;
        state.work_has_parent_message = false;
        state.parent_work_id = None;
        state.needs_finish_reminder = false;
        state.waiting_on = None;
        state.pending_finish_confirmation = None;
        state.pending_control = None;
        state.last_finish_reason = Some(effective_reason.clone());
        state.last_finish_result = Some(effective_result.clone());
        state.last_finished_work_id = Some(work_id);
        state.last_finished_at = Some(finished_at);
        self.runtime_store.clear_pending_question(agent_id)?;
        self.runtime_store.save_agent_state(state)?;
        self.persist_terminal_work(
            agent_id,
            root_agent_id,
            work_id,
            work_summary,
            started_at,
            RuntimeWorkStatus::Finished,
            finished_at,
            effective_result.clone(),
        )?;

        self.publish(
            EventKind::Final,
            work_id,
            format!(
                "WORK_FINISH\nreason: {}\nresult: {}",
                effective_reason.trim(),
                effective_result.trim()
            ),
        );

        if let Some(parent_agent_id) = parent_agent_id {
            self.append_mailbox_message(
                parent_agent_id,
                MailboxEntryKind::ChildFinished,
                Some(agent_id),
                Some(agent_label.clone()),
                format!(
                    "子代理 `{agent_label}` 已完成。\n- agent_id: {agent_id}\n- work_id: {work_id}\n- reason: {}\n- result: {}",
                    effective_reason.trim(),
                    effective_result.trim()
                ),
                parent_work_id,
                Some(work_id),
            )
            .await?;
        }

        self.wake_waiters_for_dependency(
            WaitKind::Agent,
            agent_id,
            format!("Agent {agent_id} became idle."),
        )
        .await?;
        self.wake_waiters_for_dependency(
            WaitKind::Work,
            work_id,
            format!("Work {work_id} finished via agent {agent_id}."),
        )
        .await?;

        Ok(false)
    }

    /// Background supervisor that serializes agent wake-ups into one-quantum
    /// executions.
    async fn supervisor_loop(self: Arc<Self>, mut wake_rx: mpsc::UnboundedReceiver<Uuid>) {
        while let Some(agent_id) = wake_rx.recv().await {
            let semaphore = self.current_model_call_semaphore();
            let permit = match semaphore.acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => break,
            };
            let hub = Arc::clone(&self);
            tokio::spawn(async move {
                let _permit = permit;
                let rerun = match hub.execute_agent_quantum(agent_id).await {
                    Ok(rerun) => rerun,
                    Err(err) => {
                        tracing::error!("Agent quantum failed for {agent_id}: {err:#}");
                        false
                    }
                };
                if let Err(err) = hub.finish_agent_wake_cycle(agent_id, rerun).await {
                    tracing::error!("Failed to finalize wake cycle for {agent_id}: {err:#}");
                }
            });
        }
    }

    /// Rebuild in-memory runtime indexes from the durable store and wake any
    /// agent that should resume automatically.
    async fn recover_runtime(self: &Arc<Self>) -> anyhow::Result<()> {
        let team_state = self.team_state.lock().await.clone();
        if self
            .runtime_store
            .load_agent_state(team_state.root_agent_id)?
            .is_none()
        {
            anyhow::bail!("Root agent state is missing during runtime recovery");
        }

        let mut restored_questions = Vec::<PendingQuestionEntry>::new();
        let mut resumed_agents = 0usize;
        let mut waiting_agents = 0usize;

        for mut task in self.runtime_store.list_task_states()? {
            if !matches!(task.status, RuntimeTaskStatus::Running) {
                continue;
            }
            task.status = RuntimeTaskStatus::Failed;
            task.finished_at = Some(chrono::Utc::now());
            task.summary = Some(
                "Task was still marked running when SA restarted; it has been reconciled as failed."
                    .to_string(),
            );
            self.runtime_store.save_task_state(&task)?;
            if let Err(err) = self.task_audit_store.append_new(
                task.task_id,
                task.owner_agent_id,
                self.runtime_store
                    .load_agent_state(task.owner_agent_id)?
                    .and_then(|owner| owner.active_work_id),
                TaskAuditPhase::Finished,
                "terminal",
                task.command.clone(),
                task.workdir.clone(),
                task.output_path.clone(),
                Some(task.status),
                task.exit_code,
                task.metadata.get("safety_warning").cloned(),
                task.summary.clone(),
            ) {
                tracing::warn!("Failed to append recovered task audit entry: {err:#}");
            }
            let owner_work_id = self
                .runtime_store
                .load_agent_state(task.owner_agent_id)?
                .and_then(|owner| owner.active_work_id);
            self.append_mailbox_message(
                task.owner_agent_id,
                MailboxEntryKind::TaskFinished,
                None,
                Some("runtime".to_string()),
                format!(
                    "后台任务 `{}` 在重启恢复时被标记为失败，因为进程重启前它仍处于 Running 状态。",
                    task.task_id
                ),
                owner_work_id,
                Some(task.task_id),
            )
            .await?;
            self.wake_waiters_for_dependency(
                WaitKind::Task,
                task.task_id,
                format!(
                    "Runtime task {} was reconciled as failed during recovery.",
                    task.task_id
                ),
            )
            .await?;
        }

        for mut state in self.runtime_store.list_agent_states()? {
            let _ = self.session_store_for_agent(state.agent_id).await?;

            if let Some(work_id) = state.active_work_id
                && self.runtime_store.load_work_state(work_id)?.is_none()
            {
                self.persist_started_work(
                    state.agent_id,
                    state.root_agent_id,
                    work_id,
                    state
                        .active_work_summary
                        .clone()
                        .unwrap_or_else(|| "(recovered work)".to_string()),
                    state.active_started_at.unwrap_or_else(chrono::Utc::now),
                )?;
            }

            if let Some(question_state) =
                self.runtime_store.load_pending_question(state.agent_id)?
            {
                let mode = serde_json::from_str::<QuestionMode>(&question_state.mode)?;
                let options = serde_json::from_str::<Vec<sa_core::ws_protocol::QuestionOption>>(
                    &question_state.options_json,
                )?;
                state.status = AgentStatus::WaitingUser;
                self.runtime_store.save_agent_state(&state)?;
                restored_questions.push(PendingQuestionEntry {
                    agent_id: question_state.agent_id,
                    work_id: question_state.work_id,
                    tool_call_id: question_state.tool_call_id,
                    question: UserQuestion {
                        question_id: question_state.question_id,
                        task_id: question_state.work_id,
                        created_at: question_state.created_at,
                        agent: Some(Self::frontend_identity_from_state(&state)),
                        prompt: question_state.prompt,
                        mode,
                        options,
                        allow_free_text: question_state.allow_free_text,
                    },
                    answer_tx: None,
                });
                waiting_agents = waiting_agents.saturating_add(1);
                continue;
            }

            if let Some(waiting_on) = state.waiting_on.clone() {
                if let Some(notice) = self.wait_satisfied_notice(&waiting_on)? {
                    state.waiting_on = None;
                    state.status = AgentStatus::Idle;
                    self.runtime_store.save_agent_state(&state)?;
                    self.append_mailbox_message(
                        state.agent_id,
                        MailboxEntryKind::SystemNotice,
                        None,
                        Some("runtime".to_string()),
                        notice,
                        state.active_work_id,
                        Some(waiting_on.id),
                    )
                    .await?;
                } else {
                    self.schedule_wait_timeout(state.agent_id, waiting_on);
                    waiting_agents = waiting_agents.saturating_add(1);
                }
                continue;
            }

            let unread_mailbox = self
                .runtime_store
                .read_mailbox_after(state.agent_id, state.last_mailbox_offset)?;
            let should_resume = self.current_team_cfg().auto_resume
                && !matches!(
                    state.status,
                    AgentStatus::Deleted | AgentStatus::Deleting | AgentStatus::Failed
                )
                && (state.active_work_id.is_some()
                    || !unread_mailbox.is_empty()
                    || state.pending_control.is_some()
                    || state.pending_finish_confirmation.is_some()
                    || state.needs_finish_reminder);

            if should_resume {
                state.status = AgentStatus::Idle;
                self.runtime_store.save_agent_state(&state)?;
                self.wake_agent(state.agent_id).await?;
                resumed_agents = resumed_agents.saturating_add(1);
            }
        }

        {
            let mut pending = self
                .pending_questions
                .lock()
                .expect("pending_questions mutex poisoned");
            *pending = restored_questions;
        }

        self.publish(
            EventKind::Log,
            team_state.root_agent_id,
            format!(
                "Recovered durable runtime: resumed_agents={}, waiting_agents={}, restored_questions={}.",
                resumed_agents,
                waiting_agents,
                self.pending_questions
                    .lock()
                    .expect("pending_questions mutex poisoned")
                    .len()
            ),
        );

        Ok(())
    }

    /// Execute exactly one durable quantum for one persisted agent.
    async fn execute_agent_quantum(self: &Arc<Self>, agent_id: Uuid) -> anyhow::Result<bool> {
        let mut state = self.load_agent_state_required(agent_id)?;
        if matches!(
            state.status,
            AgentStatus::Deleted | AgentStatus::Deleting | AgentStatus::Failed
        ) {
            return Ok(false);
        }

        let session_store = self.session_store_for_agent(agent_id).await?;
        self.sync_agent_session_paths(&mut state, &session_store)?;
        self.runtime_store.save_agent_state(&state)?;

        if let Some(rerun) = self
            .apply_pending_control(&mut state, &session_store)
            .await?
        {
            return Ok(rerun);
        }

        if let Some(waiting_on) = state.waiting_on.clone() {
            if let Some(notice) = self.wait_satisfied_notice(&waiting_on)? {
                state.waiting_on = None;
                state.status = AgentStatus::Idle;
                self.runtime_store.save_agent_state(&state)?;
                self.append_mailbox_message(
                    agent_id,
                    MailboxEntryKind::SystemNotice,
                    None,
                    Some("runtime".to_string()),
                    notice,
                    state.active_work_id,
                    Some(waiting_on.id),
                )
                .await?;
            } else {
                self.schedule_wait_timeout(agent_id, waiting_on);
                return Ok(false);
            }
        }

        if matches!(state.status, AgentStatus::WaitingUser) {
            if self
                .runtime_store
                .load_pending_question(agent_id)?
                .is_some()
            {
                return Ok(false);
            }
            state.status = AgentStatus::Idle;
            self.runtime_store.save_agent_state(&state)?;
        }

        let mailbox_entries = self
            .runtime_store
            .read_mailbox_after(agent_id, state.last_mailbox_offset)?;
        if state.active_work_id.is_none() && mailbox_entries.is_empty() {
            return Ok(false);
        }

        if state.active_work_id.is_none() {
            let work_id = mailbox_entries
                .iter()
                .find_map(|entry| entry.work_id)
                .unwrap_or_else(Uuid::new_v4);
            let summary_source = mailbox_entries
                .first()
                .map(|entry| entry.message.as_str())
                .unwrap_or("(runtime mailbox)");
            let started_at = chrono::Utc::now();
            let summary = Self::summarize_work_text(summary_source);
            state.active_work_id = Some(work_id);
            state.active_work_summary = Some(summary.clone());
            state.active_started_at = Some(started_at);
            state.work_has_user_output = false;
            state.work_has_parent_message = false;
            state.parent_work_id = None;
            state.pending_finish_confirmation = None;
            state.needs_finish_reminder = false;
            self.persist_started_work(
                state.agent_id,
                state.root_agent_id,
                work_id,
                summary,
                started_at,
            )?;
        }

        let work_id = state
            .active_work_id
            .ok_or_else(|| anyhow::anyhow!("Agent wake reached a state without active work"))?;
        let max_offset = mailbox_entries.last().map(|entry| entry.offset);
        let incoming_messages = mailbox_entries
            .iter()
            .map(Self::mailbox_entry_to_message)
            .collect::<Vec<_>>();
        let had_finish_reminder = state.needs_finish_reminder;
        let had_pending_finish_confirmation = state.pending_finish_confirmation.is_some();

        state.status = AgentStatus::Running;
        self.runtime_store.save_agent_state(&state)?;

        let agents_md_path = self.current_agents_md_path();
        let agents_md = match load_agents_md(agents_md_path.clone()).await {
            Ok(agents_md) => agents_md,
            Err(err) => {
                self.publish(
                    EventKind::Error,
                    work_id,
                    format!("Failed to read Agents.md: {err}"),
                );
                sa_core::agents_md::AgentsMd {
                    path: agents_md_path,
                    content: String::new(),
                    found: false,
                }
            }
        };
        let extra_prompt = self
            .build_agent_extra_prompt(&state, work_id, &agents_md)
            .await;

        let hub_for_emit = Arc::clone(self);
        let emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
            hub_for_emit.publish(kind, task_id, message);
        });
        let holds_input_ownership =
            self.team_state.lock().await.input_owner_agent_id == Some(agent_id);
        let runtime = self.build_agent_tool_runtime(&state, work_id, holds_input_ownership);
        let tool_session = sa_core::tools::ToolSession::from_state(
            state.tool_session_read_set.iter().map(PathBuf::from),
            state.active_command_invocations.clone(),
            state.activated_conditional_commands.clone(),
        );
        let (cancel_handle, cancel_token) = cancel_pair();
        {
            let mut running_agents = self.running_agents.lock().await;
            running_agents.insert(
                agent_id,
                RunningAgentExecution {
                    work_id,
                    cancel: cancel_handle,
                },
            );
        }

        let runner = self.current_runner();
        let quantum = runner
            .run_quantum(
                work_id,
                incoming_messages,
                &agents_md,
                Some(extra_prompt.as_str()),
                Some(Arc::clone(&session_store)),
                runtime,
                &cancel_token,
                emit,
                tool_session,
            )
            .await;

        self.running_agents.lock().await.remove(&agent_id);
        let mut state = self.load_agent_state_required(agent_id)?;
        self.sync_agent_session_paths(&mut state, &session_store)?;

        match quantum {
            Ok(quantum) => {
                if let Some(offset) = max_offset {
                    state.last_mailbox_offset = offset;
                }
                state.tool_session_read_set = quantum
                    .tool_session
                    .readable_paths()
                    .into_iter()
                    .map(|path| path.display().to_string())
                    .collect();
                state.active_command_invocations =
                    quantum.tool_session.active_command_invocations();

                let mut activated_conditional = quantum
                    .tool_session
                    .activated_conditional_commands()
                    .into_iter()
                    .collect::<BTreeSet<_>>();
                let preload_ctx = self.current_preload_ctx();
                let newly_activated = preload_ctx.skills.conditional_matches_for_paths(
                    &preload_ctx.workspace_root,
                    &quantum.tool_session.touched_paths(),
                    &activated_conditional,
                );
                activated_conditional.extend(newly_activated);
                state.activated_conditional_commands =
                    activated_conditional.into_iter().collect::<Vec<_>>();

                // Record cache/token usage from this quantum.
                let assistant_usage = match &quantum.outcome {
                    AgentQuantumOutcome::Ask { assistant_message, .. }
                    | AgentQuantumOutcome::Wait { assistant_message, .. }
                    | AgentQuantumOutcome::Finish { assistant_message, .. } => {
                        assistant_message.request_usage.clone()
                    }
                    AgentQuantumOutcome::Continue { .. } => None,
                };
                if let Some(usage) = &assistant_usage {
                    let model = self.current_runner().model_name().to_string();
                    let request_id = work_id.to_string();
                    self.cache_monitor
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .record_from_usage(request_id, model, usage);
                    self.broadcast_cache_stats();
                }

                match quantum.outcome {
                    AgentQuantumOutcome::Ask {
                        assistant_message,
                        tool_call_id,
                        request,
                    } => {
                        state.pending_control = Some(PendingControlAction::Ask {
                            assistant_message: PendingAssistantMessage::from(&assistant_message),
                            tool_call_id,
                            prompt: request.prompt.clone(),
                            mode: serde_json::to_string(&request.mode)?,
                            options_json: serde_json::to_string(&request.options)?,
                            allow_free_text: request.allow_free_text,
                        });
                        self.runtime_store.save_agent_state(&state)?;
                        self.apply_pending_control(&mut state, &session_store)
                            .await?
                            .ok_or_else(|| anyhow::anyhow!("Pending Ask control was not applied"))
                    }
                    AgentQuantumOutcome::Continue {
                        needs_finish_reminder,
                        assistant_text: _,
                    } => {
                        state.status = AgentStatus::Idle;
                        state.needs_finish_reminder = needs_finish_reminder;
                        state.pending_finish_confirmation = None;
                        self.runtime_store.save_agent_state(&state)?;
                        Ok(!needs_finish_reminder || !had_finish_reminder)
                    }
                    AgentQuantumOutcome::Wait {
                        assistant_message,
                        request,
                    } => {
                        state.pending_control = Some(PendingControlAction::Wait {
                            assistant_message: PendingAssistantMessage::from(&assistant_message),
                            target_kind: request.kind,
                            target_id: request.id,
                            until: request.until,
                            timeout_seconds: request.timeout_seconds,
                        });
                        self.runtime_store.save_agent_state(&state)?;
                        self.apply_pending_control(&mut state, &session_store)
                            .await?
                            .ok_or_else(|| anyhow::anyhow!("Pending Wait control was not applied"))
                    }
                    AgentQuantumOutcome::Finish {
                        assistant_message,
                        reason,
                        result,
                        without_output_confirmation,
                    } => {
                        state.pending_control = Some(PendingControlAction::Finish {
                            assistant_message: PendingAssistantMessage::from(&assistant_message),
                            reason,
                            result,
                            without_output_confirmation,
                        });
                        self.runtime_store.save_agent_state(&state)?;
                        let rerun = self
                            .apply_pending_control(&mut state, &session_store)
                            .await?
                            .ok_or_else(|| {
                                anyhow::anyhow!("Pending Finish control was not applied")
                            })?;
                        Ok(rerun && !had_pending_finish_confirmation)
                    }
                }
            }
            Err(err) => {
                if cancel_token.is_cancelled() {
                    if let Some(offset) = max_offset {
                        state.last_mailbox_offset = offset;
                    }
                    state.status = AgentStatus::Idle;
                    state.needs_finish_reminder = false;
                    self.runtime_store.save_agent_state(&state)?;
                    self.publish(
                        EventKind::Log,
                        work_id,
                        "Task cancelled by user interrupt.".to_string(),
                    );
                    self.wake_waiters_for_dependency(
                        WaitKind::Agent,
                        agent_id,
                        format!("Agent {agent_id} became idle after interrupt."),
                    )
                    .await?;
                    return Ok(false);
                }

                state.status = AgentStatus::Idle;
                self.runtime_store.save_agent_state(&state)?;
                self.publish(
                    EventKind::Error,
                    work_id,
                    format!(
                        "Task crashed and mailbox state was preserved for retry/recovery: {err:#}"
                    ),
                );
                Ok(false)
            }
        }
    }

    /// Start one background Git-Bash task and persist its runtime task state.
    async fn start_terminal_task(
        self: &Arc<Self>,
        owner_agent_id: Uuid,
        request: StartTerminalTaskRequest,
    ) -> anyhow::Result<TerminalTaskHandle> {
        let task_id = Uuid::new_v4();
        let owner_work_id = self
            .runtime_store
            .load_agent_state(owner_agent_id)?
            .and_then(|owner| owner.active_work_id);
        let output_path = self
            .runtime_store
            .runtime_dir()
            .join("tasks")
            .join(format!("{task_id}.log"));
        let output_path_string = output_path.display().to_string();
        let mut metadata = std::collections::BTreeMap::new();
        if let Some(warning) = request.safety_warning.clone() {
            metadata.insert("safety_warning".to_string(), warning);
        }
        let initial = RuntimeTaskState {
            task_id,
            owner_agent_id,
            kind: RuntimeTaskKind::Terminal,
            status: RuntimeTaskStatus::Running,
            command: request.command.clone(),
            workdir: request.workdir.clone(),
            started_at: chrono::Utc::now(),
            finished_at: None,
            exit_code: None,
            output_path: Some(output_path_string.clone()),
            summary: None,
            metadata,
        };
        self.runtime_store.save_task_state(&initial)?;
        self.task_audit_store.append_new(
            task_id,
            owner_agent_id,
            owner_work_id,
            TaskAuditPhase::Started,
            "terminal",
            request.command.clone(),
            request.workdir.clone(),
            Some(output_path_string.clone()),
            Some(RuntimeTaskStatus::Running),
            None,
            request.safety_warning.clone(),
            None,
        )?;

        let store = self.runtime_store.clone();
        let task_audit_store = Arc::clone(&self.task_audit_store);
        let hub = Arc::clone(self);
        tokio::spawn(async move {
            let result = run_background_bash_command(&request, &output_path).await;
            let mut state = initial.clone();
            state.finished_at = Some(chrono::Utc::now());

            match result {
                Ok(exit_code) => {
                    state.status = RuntimeTaskStatus::Exited;
                    state.exit_code = Some(exit_code);
                    state.summary = Some(format!("Task exited with code {exit_code}."));
                }
                Err(err) => {
                    state.status = RuntimeTaskStatus::Failed;
                    state.summary = Some(format!("{err:#}"));
                }
            }

            if let Err(err) = store.save_task_state(&state) {
                tracing::error!("Failed to persist finished runtime task state: {err:#}");
            }

            let owner_work_id = match store.load_agent_state(owner_agent_id) {
                Ok(Some(owner)) => owner.active_work_id,
                Ok(None) => None,
                Err(err) => {
                    tracing::warn!("Failed to load task owner state: {err:#}");
                    None
                }
            };
            if let Err(err) = task_audit_store.append_new(
                task_id,
                owner_agent_id,
                owner_work_id,
                TaskAuditPhase::Finished,
                "terminal",
                state.command.clone(),
                state.workdir.clone(),
                state.output_path.clone(),
                Some(state.status),
                state.exit_code,
                state.metadata.get("safety_warning").cloned(),
                state.summary.clone(),
            ) {
                tracing::warn!("Failed to append task audit entry: {err:#}");
            }
            if let Err(err) = hub
                .append_mailbox_message(
                    owner_agent_id,
                    MailboxEntryKind::TaskFinished,
                    None,
                    Some("runtime".to_string()),
                    format!(
                        "后台 Bash 任务 `{task_id}` 已结束。\n- status: {:?}\n- exit_code: {:?}\n- output_path: {}\n- summary: {}",
                        state.status,
                        state.exit_code,
                        state.output_path.clone().unwrap_or_default(),
                        state.summary.clone().unwrap_or_default()
                    ),
                    owner_work_id,
                    Some(task_id),
                )
                .await
            {
                tracing::warn!("Failed to deliver task completion mailbox entry: {err:#}");
            }
            if let Err(err) = hub
                .wake_waiters_for_dependency(
                    WaitKind::Task,
                    task_id,
                    format!(
                        "Runtime task {task_id} completed with status {:?}.",
                        state.status
                    ),
                )
                .await
            {
                tracing::warn!("Failed to wake task waiters: {err:#}");
            }

            hub.publish(
                EventKind::Log,
                owner_work_id.unwrap_or(owner_agent_id),
                format!(
                    "Background Bash task {task_id} completed with status {:?}.",
                    state.status
                ),
            );
        });

        Ok(TerminalTaskHandle {
            task_id,
            status: RuntimeTaskStatus::Running,
            output_path: Some(output_path_string),
        })
    }

    /// Load one background runtime task snapshot.
    fn get_task_info(&self, task_id: Uuid) -> anyhow::Result<TerminalTaskInfo> {
        let Some(task) = self.runtime_store.load_task_state(task_id)? else {
            anyhow::bail!("Runtime task not found: {task_id}");
        };

        Ok(TerminalTaskInfo {
            task_id: task.task_id,
            status: task.status,
            command: task.command,
            workdir: task.workdir,
            exit_code: task.exit_code,
            output_path: task.output_path,
            summary: task.summary,
        })
    }

    /// Build the durable tool runtime for one persisted agent.
    fn build_agent_tool_runtime(
        self: &Arc<Self>,
        state: &AgentState,
        work_id: Uuid,
        holds_input_ownership: bool,
    ) -> ToolRuntime {
        let agent_id = state.agent_id;
        let parent_agent_id = state.parent_agent_id;
        let root_agent_id = state.root_agent_id;
        let is_root = parent_agent_id.is_none();
        let allow_finish_without_output = state.pending_finish_confirmation.is_some();
        let allow_user_send = state.allow_user_send;
        let allow_user_show = state.allow_user_show;
        let allow_user_ask = state.allow_user_ask;
        let allow_input_transfer_target = state.allow_input_transfer_target;
        let permissions = self.current_permissions();
        let denied_tools = permissions.deny_tools;
        let denied_commands = permissions.deny_commands;
        let frontend_identity = Self::frontend_identity(agent_id, state.label.clone(), is_root);
        let allowed_tool_patterns = state
            .active_command_invocations
            .iter()
            .flat_map(|invocation| invocation.allowed_tools.iter().cloned())
            .collect::<Vec<_>>();
        let active_command_invocations = state.active_command_invocations.clone();
        let activated_conditional_commands = state.activated_conditional_commands.clone();
        let model_override = state
            .active_command_invocations
            .iter()
            .rev()
            .find_map(|invocation| invocation.model_override.clone());
        let effort_override = state
            .active_command_invocations
            .iter()
            .rev()
            .find_map(|invocation| invocation.effort_override.clone());
        let runtime_label = state.label.clone();
        let notify_parent_label = state.label.clone();
        let message_agent_label = state.label.clone();
        let broadcast_agent_label = state.label.clone();
        let parent_work_id = state.parent_work_id;
        let send_frontend_identity = frontend_identity.clone();
        let show_frontend_identity = frontend_identity.clone();

        let hub_for_send = Arc::clone(self);
        let send_message: SendMessageFn = Arc::new(move |message: String| {
            let hub = Arc::clone(&hub_for_send);
            let frontend_identity = send_frontend_identity.clone();
            Box::pin(async move {
                hub.mark_work_user_output(agent_id, work_id).await?;
                hub.record_interaction(
                    Some(work_id),
                    Some(agent_id),
                    InteractionPayload::Send {
                        message: message.clone(),
                    },
                )?;
                hub.publish_with_agent(
                    EventKind::Message,
                    work_id,
                    message,
                    Some(frontend_identity),
                );
                Ok(())
            })
        });

        let ask_question: AskQuestionFn = Arc::new(move |_request: AskRequest, _cancel| {
            Box::pin(async move {
                anyhow::bail!(
                    "Ask is handled as a durable control tool and must not call the runtime callback directly"
                )
            })
        });

        let hub_for_show = Arc::clone(self);
        let show_file: ShowFileFn = Arc::new(move |mut file: UserVisibleFile| {
            let hub = Arc::clone(&hub_for_show);
            let frontend_identity = show_frontend_identity.clone();
            Box::pin(async move {
                hub.mark_work_user_output(agent_id, work_id).await?;
                file.task_id = work_id;
                file.agent = Some(frontend_identity);
                hub.record_interaction(
                    Some(work_id),
                    Some(agent_id),
                    show_payload_from_visible_file(&file),
                )?;
                hub.publish_show(file);
                Ok(())
            })
        });

        let hub_for_subagent = Arc::clone(self);
        let run_subagent: RunSubAgentFn = Arc::new(move |request: SubAgentRequest, cancel| {
            let hub = Arc::clone(&hub_for_subagent);
            Box::pin(async move {
                hub.run_durable_subagent(agent_id, work_id, request, cancel)
                    .await
            })
        });

        let hub_for_notify_parent = Arc::clone(self);
        let notify_parent: NotifyParentFn = Arc::new(move |message: String| {
            let hub = Arc::clone(&hub_for_notify_parent);
            let agent_label = notify_parent_label.clone();
            Box::pin(async move {
                let parent_agent_id = parent_agent_id.ok_or_else(|| {
                    anyhow::anyhow!("NotifyParent is only available for child agents")
                })?;
                hub.append_mailbox_message(
                    parent_agent_id,
                    MailboxEntryKind::AgentMessage,
                    Some(agent_id),
                    Some(agent_label),
                    message,
                    parent_work_id,
                    Some(work_id),
                )
                .await?;
                hub.mark_work_parent_message(agent_id, work_id).await?;
                Ok(AgentMessageReceipt {
                    target_agent_id: parent_agent_id,
                    delivered: true,
                })
            })
        });

        let hub_for_message_agent = Arc::clone(self);
        let message_agent: MessageAgentFn = Arc::new(move |request: AgentMessageRequest| {
            let hub = Arc::clone(&hub_for_message_agent);
            let agent_label = message_agent_label.clone();
            Box::pin(async move {
                let requester = hub.load_agent_state_required(agent_id)?;
                let target = hub.load_agent_state_required(request.target_agent_id)?;
                if requester.root_agent_id != target.root_agent_id {
                    anyhow::bail!("MessageAgent target is outside the current root tree");
                }
                hub.append_mailbox_message(
                    target.agent_id,
                    MailboxEntryKind::AgentMessage,
                    Some(agent_id),
                    Some(agent_label),
                    request.message,
                    if requester.parent_agent_id == Some(target.agent_id) {
                        requester.parent_work_id
                    } else {
                        target.active_work_id
                    },
                    Some(work_id),
                )
                .await?;
                if requester.parent_agent_id == Some(target.agent_id) {
                    hub.mark_work_parent_message(agent_id, work_id).await?;
                }
                Ok(AgentMessageReceipt {
                    target_agent_id: target.agent_id,
                    delivered: true,
                })
            })
        });

        let hub_for_broadcast = Arc::clone(self);
        let broadcast_agents: BroadcastAgentsFn =
            Arc::new(move |request: BroadcastAgentsRequest| {
                let hub = Arc::clone(&hub_for_broadcast);
                let agent_label = broadcast_agent_label.clone();
                Box::pin(async move {
                    let requester = hub.load_agent_state_required(agent_id)?;
                    let recipients = hub.resolve_agent_scope(&requester, request.scope, false)?;
                    let mut parent_notified = false;
                    for recipient in &recipients {
                        hub.append_mailbox_message(
                            recipient.agent_id,
                            MailboxEntryKind::BroadcastMessage,
                            Some(agent_id),
                            Some(agent_label.clone()),
                            request.message.clone(),
                            recipient.active_work_id,
                            Some(work_id),
                        )
                        .await?;
                        if requester.parent_agent_id == Some(recipient.agent_id) {
                            parent_notified = true;
                        }
                    }
                    if parent_notified {
                        hub.mark_work_parent_message(agent_id, work_id).await?;
                    }
                    Ok(BroadcastReceipt {
                        scope: request.scope,
                        recipients: recipients.len(),
                    })
                })
            });

        let hub_for_list_agents = Arc::clone(self);
        let list_agents: ListAgentsFn = Arc::new(move |request: ListAgentsRequest| {
            let hub = Arc::clone(&hub_for_list_agents);
            Box::pin(async move {
                let requester = hub.load_agent_state_required(agent_id)?;
                Ok(hub
                    .resolve_agent_scope(&requester, request.scope, true)?
                    .into_iter()
                    .map(|state| Self::agent_info_from_state(&state))
                    .collect())
            })
        });

        let hub_for_get_agent = Arc::clone(self);
        let get_agent: GetAgentFn = Arc::new(move |target_agent_id| {
            let hub = Arc::clone(&hub_for_get_agent);
            Box::pin(async move {
                let requester = hub.load_agent_state_required(agent_id)?;
                let target = hub.load_agent_state_required(target_agent_id)?;
                if requester.root_agent_id != target.root_agent_id {
                    anyhow::bail!("GetAgent target is outside the current root tree");
                }
                Ok(Self::agent_info_from_state(&target))
            })
        });

        let hub_for_transfer_input = Arc::clone(self);
        let transfer_input: TransferInputFn = Arc::new(move |request: TransferInputRequest| {
            let hub = Arc::clone(&hub_for_transfer_input);
            Box::pin(async move {
                let requester = hub.load_agent_state_required(agent_id)?;
                if requester.parent_agent_id.is_some() {
                    anyhow::bail!("TransferInput is only available to the root agent");
                }
                let target_agent_id = match request.target_agent_id {
                    Some(target_agent_id) => {
                        let target = hub.load_agent_state_required(target_agent_id)?;
                        if target.root_agent_id != requester.root_agent_id {
                            anyhow::bail!("TransferInput target is outside the current root tree");
                        }
                        if !hub.agent_can_hold_input(&target) {
                            anyhow::bail!(
                                "Target agent is not currently eligible to hold input: {}",
                                target.agent_id
                            );
                        }
                        if !target.allow_input_transfer_target {
                            anyhow::bail!(
                                "Target agent is not allowed to hold free-form input: {}",
                                target.agent_id
                            );
                        }
                        target.agent_id
                    }
                    None => requester.agent_id,
                };
                hub.set_input_owner(target_agent_id, Some(work_id), "TransferInput tool")
                    .await?;
                Ok(TransferInputReceipt {
                    input_owner_agent_id: target_agent_id,
                })
            })
        });

        let hub_for_terminal_task = Arc::clone(self);
        let start_terminal_task: StartTerminalTaskFn = Arc::new(move |request, _cancel| {
            let hub = Arc::clone(&hub_for_terminal_task);
            Box::pin(async move { hub.start_terminal_task(agent_id, request).await })
        });

        let hub_for_get_task = Arc::clone(self);
        let get_task: GetTaskFn = Arc::new(move |task_id| {
            let hub = Arc::clone(&hub_for_get_task);
            Box::pin(async move { hub.get_task_info(task_id) })
        });
        let hub_for_reload = Arc::clone(self);
        let reload_runtime: ReloadRuntimeFn = Arc::new(move || {
            let hub = Arc::clone(&hub_for_reload);
            Box::pin(async move {
                hub.reload_runtime_from_disk()
                    .await
                    .map(|summary| ReloadRuntimeReceipt { summary })
            })
        });

        ToolRuntime::new(
            agent_id,
            parent_agent_id,
            root_agent_id,
            runtime_label,
            if is_root {
                PromptProfile::Root
            } else {
                PromptProfile::SubAgent
            },
            is_root,
            holds_input_ownership,
            allow_finish_without_output,
            allow_user_send,
            allow_user_show,
            allow_user_ask,
            allow_input_transfer_target,
            is_root,
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
        )
    }

    /// Create or wake one durable child agent and return its stable handle.
    async fn run_durable_subagent(
        self: &Arc<Self>,
        parent_agent_id: Uuid,
        parent_work_id: Uuid,
        request: SubAgentRequest,
        _cancel: sa_core::cancel::CancelToken,
    ) -> anyhow::Result<SubAgentHandle> {
        request.validate()?;

        let parent_state = self.load_agent_state_required(parent_agent_id)?;
        let normalized_prompt = request
            .prompt
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string);
        let normalized_prompt_file = match request.prompt_file.as_deref() {
            Some(raw_path) => {
                let trimmed = raw_path.trim();
                let _ = self.load_workspace_prompt_file(trimmed).await?;
                Some(trimmed.to_string())
            }
            None => None,
        };
        let prompt_skill_reminder = match request.prompt_skill.as_deref() {
            Some(raw_name) => {
                let trimmed = raw_name.trim();
                let preload_ctx = self.current_preload_ctx();
                let Some(skill) = preload_ctx.skills.get(trimmed) else {
                    anyhow::bail!("SubAgent prompt_skill not found: {trimmed}");
                };
                Some((trimmed.to_string(), skill.reminder()))
            }
            None => None,
        };
        let normalized_prompt_skill = prompt_skill_reminder
            .as_ref()
            .map(|(skill_name, _)| skill_name.clone());
        let child_depth = self.agent_depth(parent_agent_id)?.saturating_add(1);
        if child_depth > MAX_SUBAGENT_DEPTH {
            anyhow::bail!(
                "SubAgent depth limit exceeded (requested depth={}, max={})",
                child_depth,
                MAX_SUBAGENT_DEPTH
            );
        }
        let child_agent_id = request.existing_agent_id.unwrap_or_else(Uuid::new_v4);

        if request.existing_agent_id.is_none()
            && self.runtime_store.list_agent_states()?.len()
                >= self.current_team_cfg().max_active_agents
        {
            anyhow::bail!(
                "SubAgent refused because the runtime already reached max_active_agents={}",
                self.current_team_cfg().max_active_agents
            );
        }

        let child_session_store = self.session_store_for_agent(child_agent_id).await?;
        let mut child_state = match self.runtime_store.load_agent_state(child_agent_id)? {
            Some(existing) => {
                if existing.root_agent_id != parent_state.root_agent_id {
                    anyhow::bail!("SubAgent target belongs to a different root tree");
                }
                if existing.parent_agent_id != Some(parent_agent_id) {
                    anyhow::bail!(
                        "SubAgent may only reuse a direct child agent owned by the caller"
                    );
                }
                if !matches!(existing.kind, AgentKind::Worker) {
                    anyhow::bail!("SubAgent may only reuse normal worker agents");
                }
                if !matches!(existing.status, AgentStatus::Idle) {
                    anyhow::bail!("SubAgent may only reuse idle agents");
                }
                if existing.active_work_id.is_some()
                    || existing.waiting_on.is_some()
                    || existing.pending_finish_confirmation.is_some()
                    || self
                        .runtime_store
                        .load_pending_question(existing.agent_id)?
                        .is_some()
                {
                    anyhow::bail!("SubAgent may not reuse a busy or suspended agent");
                }
                if existing.allow_user_send != request.allow_user_send
                    || existing.allow_user_show != request.allow_user_show
                    || existing.allow_user_ask != request.allow_user_ask
                    || existing.allow_input_transfer_target != request.allow_input_transfer_target
                    || existing.subagent_prompt != normalized_prompt
                    || existing.subagent_prompt_file != normalized_prompt_file
                    || existing.subagent_prompt_skill != normalized_prompt_skill
                {
                    anyhow::bail!(
                        "SubAgent may not reuse an agent with different capability settings"
                    );
                }
                if let Some(label) = request.label.as_deref()
                    && existing.label != label
                {
                    anyhow::bail!("SubAgent may not reuse an agent with a different label");
                }
                existing
            }
            None => AgentState::new_child(
                child_agent_id,
                parent_agent_id,
                parent_state.root_agent_id,
                request
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("agent-{}", &child_agent_id.to_string()[..8])),
                request.allow_user_send,
                request.allow_user_show,
                request.allow_user_ask,
                request.allow_input_transfer_target,
                normalized_prompt.clone(),
                normalized_prompt_file.clone(),
                normalized_prompt_skill.clone(),
                child_session_store.current_session_path(),
            ),
        };

        self.sync_agent_session_paths(&mut child_state, &child_session_store)?;
        let child_work_id = Uuid::new_v4();
        let started_at = chrono::Utc::now();
        let summary = Self::summarize_work_text(&request.task);
        child_state.active_work_id = Some(child_work_id);
        child_state.active_work_summary = Some(summary.clone());
        child_state.active_started_at = Some(started_at);
        child_state.work_has_user_output = false;
        child_state.work_has_parent_message = false;
        child_state.parent_work_id = Some(parent_work_id);
        child_state.needs_finish_reminder = false;
        child_state.pending_finish_confirmation = None;
        child_state.pending_control = None;
        child_state.waiting_on = None;
        child_state.status = AgentStatus::Idle;
        child_state.subagent_prompt = normalized_prompt;
        child_state.subagent_prompt_file = normalized_prompt_file;
        child_state.subagent_prompt_skill = normalized_prompt_skill;
        if let Some((_skill_name, reminder)) = prompt_skill_reminder {
            child_state
                .active_command_invocations
                .retain(|invocation| invocation.name != reminder.name);
            child_state.active_command_invocations.push(reminder);
        }
        self.persist_started_work(
            child_state.agent_id,
            child_state.root_agent_id,
            child_work_id,
            summary,
            started_at,
        )?;

        self.runtime_store.save_agent_state(&child_state)?;

        self.append_mailbox_message(
            child_agent_id,
            MailboxEntryKind::UserInput,
            Some(parent_agent_id),
            Some(parent_state.label.clone()),
            format!(
                "父代理 `{}` 给你分配了新的任务。\n\n## 任务\n{}\n\n## 上下文\n{}",
                parent_state.label,
                request.task.trim(),
                request.context.trim()
            ),
            Some(child_work_id),
            Some(parent_work_id),
        )
        .await?;

        Ok(SubAgentHandle {
            agent_id: child_agent_id,
            work_id: child_work_id,
            label: child_state.label,
            status: child_state.status,
        })
    }

    /// Build the OpenClaw-style root memory prompt block.
    ///
    /// We only inject stable root memory files (`MEMORY.md` / `memory.md`) for
    /// top-level tasks. Daily memory remains on-demand via `MemorySearch` /
    /// `MemoryGet`.
    async fn memory_prompt_block(&self, task_id: Uuid) -> String {
        let preload_ctx = self.current_preload_ctx();
        match build_memory_prompt_block(&preload_ctx.workspace_root).await {
            Ok(block) => block,
            Err(err) => {
                self.publish(
                    EventKind::Error,
                    task_id,
                    format!("Failed to build memory prompt block: {err}"),
                );
                "## Memory Context\n\n- (failed to load memory bootstrap files)\n".to_string()
            }
        }
    }

    /// Validate and deliver an answer coming from a client.
    async fn answer_question(self: &Arc<Self>, answer: UserQuestionAnswer) -> anyhow::Result<()> {
        let durable_entry = {
            let mut pending = self
                .pending_questions
                .lock()
                .expect("pending_questions mutex poisoned");

            let Some(index) = pending
                .iter()
                .position(|entry| entry.question.question_id == answer.question_id)
            else {
                anyhow::bail!(
                    "Question not found or already resolved: {}",
                    answer.question_id
                );
            };

            validate_user_answer(&pending[index].question, &answer)?;

            if pending[index].answer_tx.is_some() {
                let entry = pending.remove(index);
                drop(pending);
                let Some(answer_tx) = entry.answer_tx else {
                    anyhow::bail!("Legacy Ask waiter disappeared unexpectedly");
                };
                if answer_tx.send(answer.clone()).is_err() {
                    anyhow::bail!("Question waiter dropped before receiving the answer");
                }
                self.record_interaction(
                    Some(entry.work_id),
                    Some(entry.agent_id),
                    InteractionPayload::AskAnswer {
                        question_id: answer.question_id,
                        selected_option_ids: answer.selected_option_ids.clone(),
                        selected_labels: Vec::new(),
                        free_text: answer.free_text.clone(),
                    },
                )?;
                self.broadcast_server_message(ServerMessage::QuestionResolved {
                    question_id: answer.question_id,
                });
                return Ok(());
            }

            (
                pending[index].agent_id,
                pending[index].work_id,
                pending[index].tool_call_id.clone(),
                pending[index].question.clone(),
            )
        };

        let (agent_id, work_id, tool_call_id, question) = durable_entry;

        let selected_labels: Vec<String> = answer
            .selected_option_ids
            .iter()
            .filter_map(|id| {
                question
                    .options
                    .iter()
                    .find(|option| option.id == *id)
                    .map(|option| option.label.clone())
            })
            .collect();
        let payload = serde_json::json!({
            "question_prompt": question.prompt,
            "mode": question.mode,
            "selected_option_ids": answer.selected_option_ids,
            "selected_labels": selected_labels,
            "free_text": answer.free_text,
        })
        .to_string();

        let session_store = self.session_store_for_agent(agent_id).await?;
        let already_applied = session_store.has_raw_tool_result(&tool_call_id)?;
        if !already_applied {
            session_store.append_message(&ChatMessage::tool_result(tool_call_id, payload))?;
        }
        self.runtime_store.clear_pending_question(agent_id)?;

        if let Some(mut state) = self.runtime_store.load_agent_state(agent_id)? {
            if state.active_work_id == Some(work_id) {
                state.status = AgentStatus::Idle;
                state.needs_finish_reminder = false;
                self.runtime_store.save_agent_state(&state)?;
                self.wake_agent(agent_id).await?;
            }
        }

        self.record_interaction(
            Some(work_id),
            Some(agent_id),
            InteractionPayload::AskAnswer {
                question_id: answer.question_id,
                selected_option_ids: answer.selected_option_ids.clone(),
                selected_labels,
                free_text: answer.free_text.clone(),
            },
        )?;

        let mut pending = self
            .pending_questions
            .lock()
            .expect("pending_questions mutex poisoned");
        if let Some(index) = pending
            .iter()
            .position(|entry| entry.question.question_id == answer.question_id)
        {
            pending.remove(index);
        }
        drop(pending);

        self.broadcast_server_message(ServerMessage::QuestionResolved {
            question_id: answer.question_id,
        });

        Ok(())
    }

    /// Accept one user message (idempotent).
    ///
    /// Behavior:
    /// - if no task is currently running, this becomes a new top-level task
    /// - if a task is currently running, this is queued as a follow-up user
    ///   turn for that same session
    /// - if the current task is already being cancelled, the message is queued
    ///   as the next top-level task instead of being attached to the doomed run
    async fn submit_task(self: &Arc<Self>, task_id: Uuid, task: String) -> anyhow::Result<Uuid> {
        // Ensure idempotency: if we have already seen this submit id, do not process it again.
        {
            let seen = self.seen_tasks.lock().expect("seen_tasks mutex poisoned");
            if let Some(existing_work_id) = seen.get(&task_id).copied() {
                return Ok(existing_work_id);
            }
        }

        let owner_agent_id = self.effective_input_owner().await?;
        let mut owner_state = self.load_agent_state_required(owner_agent_id)?;
        let is_new_work = owner_state.active_work_id.is_none();
        let effective_work_id = owner_state.active_work_id.unwrap_or(task_id);

        if is_new_work {
            let started_at = chrono::Utc::now();
            let summary = Self::summarize_work_text(&task);
            owner_state.active_work_id = Some(effective_work_id);
            owner_state.active_work_summary = Some(summary.clone());
            owner_state.active_started_at = Some(started_at);
            owner_state.work_has_user_output = false;
            owner_state.work_has_parent_message = false;
            owner_state.parent_work_id = None;
            owner_state.needs_finish_reminder = false;
            owner_state.pending_finish_confirmation = None;
            owner_state.pending_control = None;
            owner_state.waiting_on = None;
            owner_state.status = AgentStatus::Idle;
            self.persist_started_work(
                owner_state.agent_id,
                owner_state.root_agent_id,
                effective_work_id,
                summary,
                started_at,
            )?;
        }
        self.runtime_store.save_agent_state(&owner_state)?;
        {
            let mut seen = self.seen_tasks.lock().expect("seen_tasks mutex poisoned");
            seen.insert(task_id, effective_work_id);
        }

        self.append_mailbox_message(
            owner_agent_id,
            MailboxEntryKind::UserInput,
            None,
            Some("classmate".to_string()),
            task.clone(),
            Some(effective_work_id),
            None,
        )
        .await?;
        self.record_interaction(
            Some(effective_work_id),
            Some(owner_agent_id),
            InteractionPayload::UserMessage {
                submit_id: task_id,
                input_owner_agent_id: owner_agent_id,
                message: task.clone(),
            },
        )?;

        self.publish(
            EventKind::Log,
            effective_work_id,
            if is_new_work {
                format!("Accepted new work for agent {owner_agent_id}.")
            } else {
                format!(
                    "Queued a follow-up message for agent {owner_agent_id} work {effective_work_id}."
                )
            },
        );

        Ok(effective_work_id)
    }

    /// Ask the user a structured question and wait until an answer arrives.
    #[allow(dead_code)]
    async fn ask_user(
        &self,
        task_id: Uuid,
        request: AskRequest,
        cancel: &sa_core::cancel::CancelToken,
    ) -> anyhow::Result<UserQuestionAnswer> {
        request.validate()?;

        let question = UserQuestion {
            question_id: Uuid::new_v4(),
            task_id,
            created_at: chrono::Utc::now(),
            agent: None,
            prompt: request.prompt,
            mode: request.mode,
            options: request.options,
            allow_free_text: request.allow_free_text,
        };

        let (answer_tx, answer_rx) = oneshot::channel::<UserQuestionAnswer>();
        {
            let mut pending = self
                .pending_questions
                .lock()
                .expect("pending_questions mutex poisoned");
            pending.push(PendingQuestionEntry {
                agent_id: task_id,
                work_id: task_id,
                tool_call_id: String::new(),
                question: question.clone(),
                answer_tx: Some(answer_tx),
            });
        }

        self.broadcast_server_message(ServerMessage::Question {
            question: question.clone(),
        });

        let answer = tokio::select! {
            _ = cancel.cancelled() => {
                let mut pending = self
                    .pending_questions
                    .lock()
                    .expect("pending_questions mutex poisoned");
                if let Some(index) = pending
                    .iter()
                    .position(|entry| entry.question.question_id == question.question_id)
                {
                    pending.remove(index);
                    self.broadcast_server_message(ServerMessage::QuestionResolved {
                        question_id: question.question_id,
                    });
                }
                anyhow::bail!("Ask cancelled");
            }
            answer = answer_rx => {
                answer.context("Ask waiter dropped before an answer arrived")?
            }
        };

        Ok(answer)
    }

    /// Build the per-task runtime callbacks used by the tool layer.
    #[allow(dead_code)]
    fn build_tool_runtime(self: &Arc<Self>, task_id: Uuid, depth: u32) -> ToolRuntime {
        let agent_id = task_id;
        let parent_agent_id = (depth > 0).then_some(task_id);
        let hub_for_send = Arc::clone(self);
        let send_message: SendMessageFn = Arc::new(move |message: String| {
            let hub = Arc::clone(&hub_for_send);
            Box::pin(async move {
                hub.publish(
                    EventKind::Message,
                    task_id,
                    decorate_nested_text(depth, &message),
                );
                Ok(())
            })
        });

        let hub_for_ask = Arc::clone(self);
        let ask_question: AskQuestionFn = Arc::new(move |request: AskRequest, cancel| {
            let hub = Arc::clone(&hub_for_ask);
            Box::pin(async move {
                let request = AskRequest {
                    prompt: decorate_nested_prompt(depth, &request.prompt),
                    ..request
                };
                hub.ask_user(task_id, request, &cancel).await
            })
        });

        let hub_for_show = Arc::clone(self);
        let show_file: ShowFileFn = Arc::new(move |mut file: UserVisibleFile| {
            let hub = Arc::clone(&hub_for_show);
            Box::pin(async move {
                file.task_id = task_id;
                if depth > 0 {
                    file.title = Some(match file.title {
                        Some(title) => format!("[subagent depth={depth}] {title}"),
                        None => format!("[subagent depth={depth}] {}", file.path),
                    });
                }
                hub.publish_show(file);
                Ok(())
            })
        });

        let hub_for_subagent = Arc::clone(self);
        let run_subagent: RunSubAgentFn = Arc::new(move |request: SubAgentRequest, cancel| {
            let hub = Arc::clone(&hub_for_subagent);
            Box::pin(async move { hub.run_subagent(task_id, depth, request, cancel).await })
        });
        let notify_parent: NotifyParentFn = Arc::new(move |_message: String| {
            Box::pin(async move { anyhow::bail!("NotifyParent is not wired in the legacy hub") })
        });
        let message_agent: MessageAgentFn = Arc::new(move |_request| {
            Box::pin(async move { anyhow::bail!("MessageAgent is not wired in the legacy hub") })
        });
        let broadcast_agents: BroadcastAgentsFn = Arc::new(move |_request| {
            Box::pin(async move { anyhow::bail!("BroadcastAgents is not wired in the legacy hub") })
        });
        let list_agents: ListAgentsFn =
            Arc::new(move |_request| Box::pin(async move { Ok(Vec::<AgentInfo>::new()) }));
        let get_agent: GetAgentFn = Arc::new(move |_agent_id| {
            Box::pin(async move { anyhow::bail!("GetAgent is not wired in the legacy hub") })
        });
        let transfer_input: TransferInputFn = Arc::new(move |_request| {
            Box::pin(async move { anyhow::bail!("TransferInput is not wired in the legacy hub") })
        });
        let hub_for_terminal_task = Arc::clone(self);
        let start_terminal_task: StartTerminalTaskFn = Arc::new(move |request, _cancel| {
            let hub = Arc::clone(&hub_for_terminal_task);
            Box::pin(async move { hub.start_terminal_task(agent_id, request).await })
        });
        let hub_for_get_task = Arc::clone(self);
        let get_task: GetTaskFn = Arc::new(move |task_id| {
            let hub = Arc::clone(&hub_for_get_task);
            Box::pin(async move { hub.get_task_info(task_id) })
        });
        let hub_for_reload = Arc::clone(self);
        let reload_runtime: ReloadRuntimeFn = Arc::new(move || {
            let hub = Arc::clone(&hub_for_reload);
            Box::pin(async move {
                hub.reload_runtime_from_disk()
                    .await
                    .map(|summary| ReloadRuntimeReceipt { summary })
            })
        });

        ToolRuntime::new(
            agent_id,
            parent_agent_id,
            task_id,
            if depth == 0 {
                "root".to_string()
            } else {
                format!("subagent-depth-{depth}")
            },
            if depth == 0 {
                PromptProfile::Root
            } else {
                PromptProfile::SubAgent
            },
            depth == 0,
            depth == 0,
            false,
            true,
            true,
            true,
            false,
            depth == 0,
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

    /// Build a restricted runtime used by internal background tasks such as
    /// nightly dream consolidation.
    ///
    /// Background tasks must not talk to the user directly or block on `Ask`.
    /// Instead:
    /// - `Send` becomes a trace log entry
    /// - `Show` becomes a trace log entry
    /// - `Ask` is rejected
    /// - `SubAgent` is rejected to keep the background control surface small
    fn build_background_tool_runtime(
        self: &Arc<Self>,
        task_id: Uuid,
        label: &'static str,
    ) -> ToolRuntime {
        let notify_parent: NotifyParentFn = Arc::new(move |_message: String| {
            Box::pin(
                async move { anyhow::bail!("background {label} task must not use NotifyParent") },
            )
        });
        let hub_for_send = Arc::clone(self);
        let send_message: SendMessageFn = Arc::new(move |message: String| {
            let hub = Arc::clone(&hub_for_send);
            Box::pin(async move {
                hub.publish(
                    EventKind::Log,
                    task_id,
                    format!("[{label}] background Send suppressed: {message}"),
                );
                Ok(())
            })
        });

        let ask_question: AskQuestionFn = Arc::new(move |_request: AskRequest, _cancel| {
            Box::pin(async move { anyhow::bail!("background {label} task must not use Ask") })
        });

        let hub_for_show = Arc::clone(self);
        let show_file: ShowFileFn = Arc::new(move |file: UserVisibleFile| {
            let hub = Arc::clone(&hub_for_show);
            Box::pin(async move {
                hub.publish(
                    EventKind::Log,
                    task_id,
                    format!("[{label}] background Show suppressed: {}", file.path),
                );
                Ok(())
            })
        });

        let run_subagent: RunSubAgentFn = Arc::new(move |_request: SubAgentRequest, _cancel| {
            Box::pin(async move { anyhow::bail!("background {label} task must not use SubAgent") })
        });
        let message_agent: MessageAgentFn = Arc::new(move |_request| {
            Box::pin(
                async move { anyhow::bail!("background {label} task must not use MessageAgent") },
            )
        });
        let broadcast_agents: BroadcastAgentsFn = Arc::new(move |_request| {
            Box::pin(async move {
                anyhow::bail!("background {label} task must not use BroadcastAgents")
            })
        });
        let list_agents: ListAgentsFn =
            Arc::new(move |_request| Box::pin(async move { Ok(Vec::<AgentInfo>::new()) }));
        let get_agent: GetAgentFn = Arc::new(move |_agent_id| {
            Box::pin(async move { anyhow::bail!("background {label} task must not use GetAgent") })
        });
        let transfer_input: TransferInputFn = Arc::new(move |_request| {
            Box::pin(
                async move { anyhow::bail!("background {label} task must not use TransferInput") },
            )
        });
        let hub_for_terminal_task = Arc::clone(self);
        let start_terminal_task: StartTerminalTaskFn = Arc::new(move |request, _cancel| {
            let hub = Arc::clone(&hub_for_terminal_task);
            Box::pin(async move { hub.start_terminal_task(task_id, request).await })
        });
        let hub_for_get_task = Arc::clone(self);
        let get_task: GetTaskFn = Arc::new(move |task_id| {
            let hub = Arc::clone(&hub_for_get_task);
            Box::pin(async move { hub.get_task_info(task_id) })
        });
        let reload_runtime: ReloadRuntimeFn = Arc::new(move || {
            Box::pin(async move { anyhow::bail!("background {label} task must not use Reload") })
        });

        ToolRuntime::new(
            task_id,
            None,
            task_id,
            label.to_string(),
            PromptProfile::Background,
            true,
            false,
            false,
            false,
            false,
            false,
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

    /// Execute one isolated background dream run.
    async fn run_background_dream(self: &Arc<Self>, dream: &DreamManager) -> anyhow::Result<()> {
        let started_at = chrono::Local::now();
        if !dream.should_run_now(started_at)? {
            return Ok(());
        }

        let Some(_lock) = dream.try_acquire_lock()? else {
            self.publish(
                EventKind::Log,
                Uuid::nil(),
                "[dream] skipped because another dream run already holds the lock.".to_string(),
            );
            return Ok(());
        };

        let prepared = dream.prepare_run(started_at)?;
        dream.mark_started(started_at)?;

        let task_id = Uuid::new_v4();
        self.publish(
            EventKind::Log,
            task_id,
            format!(
                "[dream] starting nightly memory distillation; report={}",
                prepared.report_relative_path
            ),
        );

        let agents_md_path = self.current_agents_md_path();
        let agents_md = load_agents_md(agents_md_path.clone())
            .await
            .context("Failed to load Agents.md for dream run")?;

        let hub_for_emit = Arc::clone(self);
        let emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
            // Dream 子代理 Final 冗余，各步骤已作为 Log emit
            if matches!(kind, EventKind::Final) {
                return;
            }
            hub_for_emit.publish(kind, task_id, format!("[dream] {message}"));
        });

        let runtime = self.build_background_tool_runtime(task_id, "dream");
        let (_cancel_handle, cancel_token) = cancel_pair();

        let runner = self.current_runner();
        match runner
            .run_task(
                task_id,
                prepared.task,
                &agents_md,
                Some(prepared.extra_system_prompt.as_str()),
                None,
                runtime,
                &cancel_token,
                None,
                emit,
            )
            .await
        {
            Ok(_final_text) => {
                dream.mark_completed(
                    started_at.date_naive(),
                    chrono::Local::now(),
                    &prepared.report_relative_path,
                )?;
                self.broadcast_server_message(ServerMessage::MemoryUpdated {
                    source: "dream".to_string(),
                    report: Some(prepared.report_relative_path.clone()),
                });
                self.publish(
                    EventKind::Log,
                    task_id,
                    format!(
                        "[dream] completed successfully; report={}",
                        prepared.report_relative_path
                    ),
                );
                Ok(())
            }
            Err(err) => {
                self.publish(
                    EventKind::Error,
                    task_id,
                    format!("[dream] failed: {err:#}"),
                );
                Err(err)
            }
        }
    }

    /// Run a nested sub-agent while keeping all output attached to the
    /// top-level task event stream.
    #[allow(dead_code)]
    async fn run_subagent(
        self: &Arc<Self>,
        task_id: Uuid,
        parent_depth: u32,
        request: SubAgentRequest,
        cancel: sa_core::cancel::CancelToken,
    ) -> anyhow::Result<SubAgentHandle> {
        let depth = parent_depth.saturating_add(1);
        if depth > MAX_SUBAGENT_DEPTH {
            anyhow::bail!(
                "SubAgent depth limit exceeded (requested depth={}, max={})",
                depth,
                MAX_SUBAGENT_DEPTH
            );
        }
        let child_agent_id = request.existing_agent_id.unwrap_or_else(Uuid::new_v4);
        let child_work_id = Uuid::new_v4();

        let label = request
            .label
            .clone()
            .unwrap_or_else(|| format!("subagent-depth-{depth}"));

        self.publish(
            EventKind::Log,
            task_id,
            format!(
                "[subagent depth={depth} label={label}] started: {}",
                request.task
            ),
        );

        let agents_md_path = self.current_agents_md_path();
        let agents_md = match load_agents_md(agents_md_path.clone()).await {
            Ok(a) => a,
            Err(err) => {
                self.publish(
                    EventKind::Error,
                    task_id,
                    format!("Failed to read Agents.md for subagent: {err}"),
                );
                sa_core::agents_md::AgentsMd {
                    path: agents_md_path,
                    content: String::new(),
                    found: false,
                }
            }
        };

        let preload_block = self.preload_agents_md_references(&agents_md).await;
        let extra_prompt = format!(
            "{preload_block}\n\n## Parent-provided SubAgent Context\n\n- depth: {depth}\n- label: {label}\n\n```text\n{}\n```",
            request.context.trim()
        );

        let hub_for_emit = Arc::clone(self);
        let emit_label = label.clone();
        let emit: EmitEventFn = Arc::new(move |kind, _ignored_task_id, message| {
            // 子代理 Final 冗余，各步骤已作为 Log emit
            if matches!(kind, EventKind::Final) {
                return;
            }
            hub_for_emit.publish(
                kind,
                task_id,
                format!(
                    "[subagent depth={} label={}] {}",
                    depth, emit_label, message
                ),
            );
        });

        let runtime = self.build_tool_runtime(task_id, depth);
        let runner = self.current_runner();
        let result = runner
            .run_task(
                child_work_id,
                request.task.clone(),
                &agents_md,
                Some(extra_prompt.as_str()),
                None,
                runtime,
                &cancel,
                None,
                emit,
            )
            .await;

        match &result {
            Ok(final_answer) => {
                self.publish(
                    EventKind::Log,
                    task_id,
                    format!(
                        "[subagent depth={depth} label={label}] completed: {}",
                        final_answer.trim()
                    ),
                );
            }
            Err(err) => {
                self.publish(
                    EventKind::Error,
                    task_id,
                    format!("[subagent depth={depth} label={label}] crashed: {err}"),
                );
            }
        }

        result?;
        Ok(SubAgentHandle {
            agent_id: child_agent_id,
            work_id: child_work_id,
            label,
            status: AgentStatus::Idle,
        })
    }

    /// Background worker loop that executes tasks sequentially.
    #[allow(dead_code)]
    async fn worker_loop(self: Arc<Self>, mut task_rx: mpsc::Receiver<TaskRequest>) {
        while let Some(req) = task_rx.recv().await {
            // Create a per-task cancellation token and register it as the "current task".
            //
            // This is the backend part of: "the CLI can send messages anytime to interrupt".
            let (cancel_handle, cancel_token) = cancel_pair();
            let follow_up_messages = Arc::new(Mutex::new(VecDeque::<String>::new()));
            {
                let mut current = self
                    .current_task
                    .lock()
                    .expect("current_task mutex poisoned");
                *current = Some(CurrentTask {
                    task_id: req.task_id,
                    cancel: cancel_handle,
                    follow_up_messages: Arc::clone(&follow_up_messages),
                });
            }

            // Emit task start.
            self.publish(
                EventKind::Log,
                req.task_id,
                format!("Task started: {}", req.task),
            );

            // Reload Agents.md for every task (see comment on `agents_md_path`).
            let agents_md_path = self.current_agents_md_path();
            let agents_md = match load_agents_md(agents_md_path.clone()).await {
                Ok(a) => a,
                Err(err) => {
                    self.publish(
                        EventKind::Error,
                        req.task_id,
                        format!("Failed to read Agents.md: {err}"),
                    );
                    sa_core::agents_md::AgentsMd {
                        path: agents_md_path,
                        content: String::new(),
                        found: false,
                    }
                }
            };

            // Extra context injected into the system prompt:
            // - root memory files (`MEMORY.md` / `memory.md`)
            // - proactive preloading of non-memory files referenced in Agents.md
            let memory_block = self.memory_prompt_block(req.task_id).await;
            let preload_block = self.preload_agents_md_references(&agents_md).await;
            let mut extra_prompt = format!("{memory_block}\n\n{preload_block}");

            // Check for skill trigger word matching.
            let preload_ctx = self.current_preload_ctx();
            if let Some(skill) = preload_ctx.skills.match_triggers(&req.task) {
                // Send skill_activated notification to frontend.
                self.broadcast_server_message(
                    sa_core::ws_protocol::ServerMessage::SkillActivated {
                        skill_name: skill.name.clone(),
                        description: skill.description.clone(),
                        trigger: skill.triggers.iter().find(|t| req.task.to_lowercase().contains(&t.to_lowercase())).cloned().unwrap_or_default(),
                        orchestration_mode: skill.orchestration.as_ref().and_then(|o| o.mode.clone()),
                    },
                );
                // Inject skill instructions into extra_system_prompt.
                if let Ok(expanded) = skill.expand_invocation(Some(&req.task), &req.task_id.to_string()) {
                    extra_prompt = format!("{extra_prompt}\n\n## Skill Activation: {}\n\n{}", skill.name, expanded.instructions);
                }
            }

            // Build the emit callback for the agent runner.
            let hub_for_emit = Arc::clone(&self);
            let emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
                hub_for_emit.publish(kind, task_id, message);
            });
            let drain_queued_user_messages: DrainQueuedUserMessagesFn = Arc::new(move || {
                let mut queued = follow_up_messages
                    .lock()
                    .expect("follow_up_messages mutex poisoned");
                queued.drain(..).collect()
            });
            let runtime = self.build_tool_runtime(req.task_id, 0);

            // Run the agent loop.
            let runner = self.current_runner();
            match runner
                .run_task(
                    req.task_id,
                    req.task.clone(),
                    &agents_md,
                    Some(extra_prompt.as_str()),
                    Some(Arc::clone(&self.session_store)),
                    runtime,
                    &cancel_token,
                    Some(drain_queued_user_messages),
                    emit.clone(),
                )
                .await
            {
                Ok(final_text) => {
                    // The agent runner already emits `Final`, but we keep an explicit
                    // completion log for clarity.
                    self.publish(EventKind::Log, req.task_id, "Task finished.".to_string());

                    // Trigger lightweight post-task memory collection (async, non-blocking).
                    let hub_for_memory = Arc::clone(&self);
                    let mem_task_id = req.task_id;
                    let mem_agents_md = agents_md.clone();
                    let mem_session_store = Some(Arc::clone(&self.session_store));
                    let mem_cancel = cancel_token.clone();
                    let mem_emit = emit.clone();
                    let mem_summary = format!("Task completed: {}", req.task);

                    tokio::spawn(async move {
                        let runner = hub_for_memory.current_runner();
                        if let Err(e) = sa_core::agent::post_task_memory_collect(
                            &runner,
                            mem_task_id,
                            &mem_agents_md,
                            None,
                            mem_session_store.as_ref().unwrap(),
                            &mem_cancel,
                            &mem_emit,
                            &mem_summary,
                        )
                        .await
                        {
                            tracing::warn!("Post-task memory collection failed: {e:#}");
                        }
                        hub_for_memory.broadcast_server_message(
                            sa_core::ws_protocol::ServerMessage::MemoryUpdated {
                                source: "post_task".to_string(),
                                report: None,
                            },
                        );

                        // Skillify: Auto-extract reusable patterns from task execution.
                        // This is a simplified placeholder - in a full implementation,
                        // we would analyze tool call patterns and generate SKILL.md files.
                        tracing::info!("Skillify: Task pattern extraction completed (placeholder)");
                    });

                    let _ = final_text;
                }
                Err(err) => {
                    let msg = format!("Task crashed: {err}");
                    self.publish(EventKind::Error, req.task_id, msg.clone());
                    // Emit `Final` so clients can stop waiting even on crash.
                    self.publish(EventKind::Final, req.task_id, msg);
                }
            }

            // Clear the "current task" slot if it still points to this task id.
            //
            // This prevents interrupts from accidentally cancelling the next task.
            {
                let mut current = self
                    .current_task
                    .lock()
                    .expect("current_task mutex poisoned");
                if current.as_ref().is_some_and(|c| c.task_id == req.task_id) {
                    *current = None;
                }
            }
        }
    }

    /// Preload workspace files referenced by `Agents.md` and return a prompt block.
    async fn preload_agents_md_references(
        &self,
        agents_md: &sa_core::agents_md::AgentsMd,
    ) -> String {
        // If Agents.md is missing or empty, we return a short block for traceability.
        if !agents_md.found || agents_md.content.trim().is_empty() {
            return "## Preloaded files (from Agents.md)\n\n- (Agents.md missing or empty)\n"
                .to_string();
        }

        let refs = extract_markdown_file_references(&agents_md.content);
        let refs = expand_date_placeholders(refs);

        // Hard limits to keep prompts bounded.
        let max_files: usize = 10;
        let max_bytes_per_file: u64 = 80_000;

        let mut out = String::new();
        out.push_str("## Preloaded files (from Agents.md)\n\n");
        let preload_ctx = self.current_preload_ctx();

        let mut loaded = 0usize;
        for r in refs {
            if is_memory_reference(&r) {
                out.push_str(&format!(
                    "- `{r}`: skipped; use `MemorySearch` / `MemoryGet` for memory files\n"
                ));
                continue;
            }

            if loaded >= max_files {
                out.push_str("- (truncated: too many referenced files)\n");
                break;
            }

            // Resolve safely under the workspace root.
            let abs = match preload_ctx.resolve_under_workspace(&r) {
                Ok(p) => p,
                Err(err) => {
                    out.push_str(&format!("- `{r}`: resolve error: {err}\n"));
                    continue;
                }
            };

            // Only preload existing files.
            let meta = match tokio::fs::metadata(&abs).await {
                Ok(m) => m,
                Err(_) => {
                    out.push_str(&format!("- `{r}`: (missing)\n"));
                    continue;
                }
            };

            if meta.len() > max_bytes_per_file {
                out.push_str(&format!(
                    "- `{r}`: (skipped; too large: {} bytes)\n",
                    meta.len()
                ));
                continue;
            }

            let content = match tokio::fs::read_to_string(&abs).await {
                Ok(c) => c,
                Err(err) => {
                    out.push_str(&format!("- `{r}`: read error: {err}\n"));
                    continue;
                }
            };

            loaded += 1;
            out.push_str(&format!("\n### `{r}`\n\n"));
            out.push_str("```text\n");
            out.push_str(&content);
            if !content.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("```\n");
        }

        if loaded == 0 {
            out.push_str("- (no referenced files were loaded)\n");
        }

        out
    }
}

/// Fully built runtime returned from one successfully loaded configuration.
struct LoadedRuntime {
    /// Live background hub.
    hub: Arc<Hub>,
    /// Bind address selected by the config.
    bind: String,
    /// WS path selected by the config.
    ws_path: String,
    /// Resolved workspace root.
    workspace_root: PathBuf,
}

impl ServerState {
    /// Return a snapshot of the current runtime mode.
    async fn runtime_snapshot(&self) -> RuntimeState {
        self.runtime.lock().await.clone()
    }

    /// Create `sa.toml`, validate it, build the runtime, and atomically switch
    /// the daemon out of bootstrap mode.
    async fn initialize_from_request(
        &self,
        request: InitializeConfigRequest,
    ) -> anyhow::Result<InitCompleted> {
        let bootstrap = {
            let runtime = self.runtime.lock().await;
            match &*runtime {
                RuntimeState::Bootstrap(bootstrap) => bootstrap.clone(),
                RuntimeState::Ready(_) => {
                    anyhow::bail!("SA 已经完成初始化，无需再次创建配置文件")
                }
            }
        };

        if bootstrap.config_path.exists() {
            anyhow::bail!("目标配置文件已经存在：{}", bootstrap.config_path.display());
        }

        std::fs::create_dir_all(&bootstrap.config_dir).with_context(|| {
            format!(
                "Failed to create config directory: {}",
                bootstrap.config_dir.display()
            )
        })?;

        let rendered = render_initial_config_toml(&bootstrap, &request)?;
        let temp_config_path = bootstrap
            .config_dir
            .join(format!(".sa.init.{}.toml", Uuid::new_v4()));

        std::fs::write(&temp_config_path, rendered).with_context(|| {
            format!(
                "Failed to write temporary config file: {}",
                temp_config_path.display()
            )
        })?;

        let load_result = load_config_from_file(&temp_config_path);
        let runtime_result = match load_result {
            Ok(cfg) => build_runtime_from_config(cfg, &temp_config_path).await,
            Err(err) => Err(err),
        };

        let runtime = match runtime_result {
            Ok(runtime) => runtime,
            Err(err) => {
                let _ = std::fs::remove_file(&temp_config_path);
                return Err(err);
            }
        };

        if runtime.bind != bootstrap.bind || runtime.ws_path != bootstrap.ws_path {
            let _ = std::fs::remove_file(&temp_config_path);
            anyhow::bail!(
                "初始化生成的监听配置与当前 bootstrap 监听器不一致：expected {}{} but got {}{}",
                bootstrap.bind,
                bootstrap.ws_path,
                runtime.bind,
                runtime.ws_path
            );
        }

        std::fs::rename(&temp_config_path, &bootstrap.config_path).with_context(|| {
            format!(
                "Failed to move generated config into place: {}",
                bootstrap.config_path.display()
            )
        })?;

        {
            let mut runtime_state = self.runtime.lock().await;
            match &*runtime_state {
                RuntimeState::Bootstrap(_) => {
                    *runtime_state = RuntimeState::Ready(Arc::clone(&runtime.hub));
                }
                RuntimeState::Ready(_) => {
                    tracing::warn!(
                        "Bootstrap initialization raced with another ready runtime; keeping the existing runtime state"
                    );
                }
            }
        }

        Ok(InitCompleted {
            config_path: bootstrap.config_path.display().to_string(),
            workspace_root: runtime.workspace_root.display().to_string(),
            message: "初始化完成，新的 `sa.toml` 已写入并立即生效。".to_string(),
        })
    }
}

/// Build the bootstrap state used when `sa.toml` does not exist yet.
fn build_bootstrap_state(config_path: &Path) -> anyhow::Result<BootstrapState> {
    let config_path = std::path::absolute(config_path).with_context(|| {
        format!(
            "Failed to resolve bootstrap config path: {}",
            config_path.display()
        )
    })?;
    let config_dir = config_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let workspace_root = config_dir.clone();
    let agents_md_path = workspace_root.join("Agents.md");

    Ok(BootstrapState {
        config_path,
        config_dir,
        workspace_root,
        agents_md_path,
        bind: DEFAULT_BOOTSTRAP_BIND.to_string(),
        ws_path: DEFAULT_BOOTSTRAP_WS_PATH.to_string(),
    })
}

/// Build one fully initialized runtime from an already parsed config object.
async fn prepare_runtime_from_config(
    cfg: Config,
    config_path: &Path,
) -> anyhow::Result<PreparedRuntime> {
    let config_dir = resolve_config_file_dir(config_path)?;
    let bind = cfg.server.bind.clone();
    let ws_path = cfg.server.ws_path.clone();

    // Resolve workspace paths.
    let workspace_root = cfg.workspace.root_dir_path(&config_dir);
    let agents_md_path = cfg.workspace.agents_md_path(&workspace_root);

    tracing::info!("Workspace root: {}", workspace_root.display());
    tracing::info!("Agents.md path: {}", agents_md_path.display());

    // Best-effort load `Agents.md` once for startup logging.
    match load_agents_md(agents_md_path.clone()).await {
        Ok(a) if a.found => {
            tracing::info!("Agents.md detected ({} bytes).", a.content.len());
        }
        Ok(_) => {
            tracing::info!("Agents.md not found at startup (this is OK).");
        }
        Err(err) => {
            tracing::warn!("Failed to read Agents.md at startup: {err}");
        }
    }

    // Discover skills.
    let skill_dirs = cfg.skills.dirs_as_paths();
    let mut skills_registry = SkillRegistry::scan(&skill_dirs)?;
    let local_skill_count = skills_registry.list().len();
    tracing::info!("Discovered {} local skill(s).", local_skill_count);

    // Connect external MCP servers before freezing the tool registry.
    let mut connected_mcp_servers = 0usize;
    let mcp_registry = if cfg.mcp.enabled && !cfg.mcp.servers.is_empty() {
        tracing::info!(
            "Initializing MCP client — {} server(s) configured",
            cfg.mcp.servers.len()
        );
        match McpRegistry::connect_all(&cfg.mcp.servers).await {
            Ok(registry) if !registry.is_empty() => {
                connected_mcp_servers = registry.server_count();
                skills_registry.extend_mcp_prompts(registry.prompt_commands());
                tracing::info!(
                    "MCP: {} tool(s) and {} prompt command(s) registered from {} server(s)",
                    registry.tool_count(),
                    registry.prompt_commands().len(),
                    registry.server_count()
                );
                Some(Arc::new(registry))
            }
            Ok(_) => {
                tracing::warn!("MCP enabled, but no MCP servers connected successfully.");
                None
            }
            Err(error) => {
                tracing::error!("MCP registry failed to initialize: {error:#}");
                None
            }
        }
    } else {
        None
    };
    let skills = Arc::new(skills_registry);

    // Build tools.
    let tool_ctx = ToolContext::new(workspace_root.clone(), Arc::clone(&skills))?
        .with_search_config(cfg.search.clone());
    let preload_ctx = tool_ctx.clone();
    let tools = ToolExecutor::new(tool_ctx, mcp_registry);

    // Build LLM client.
    let system_role_name = cfg.llm.effective_system_role_name().to_string();
    let reasoning_effort = cfg.llm.effective_reasoning_effort().map(str::to_string);
    let reasoning_effort_summary = reasoning_effort
        .clone()
        .unwrap_or_else(|| "none".to_string());
    let wire_api = cfg.llm.effective_wire_api();
    let auth_style = cfg.llm.effective_auth_style(wire_api);
    let llm = OpenAiClient::with_wire_api_and_auth_style(
        cfg.llm.base_url,
        cfg.llm.api_key,
        wire_api,
        auth_style,
    )?;

    // Build agent runner.
    let runner_cfg = AgentRunnerConfig {
        model: cfg.llm.model.clone(),
        system_role_name: system_role_name.clone(),
        reasoning_effort,
        compaction: cfg.compaction,
        temperature: cfg.llm.temperature,
        top_p: cfg.llm.top_p,
        max_output_tokens: cfg.llm.max_output_tokens,
        fallback_model: cfg.llm.fallback_model.clone(),
        max_consecutive_failures: cfg.llm.max_consecutive_failures,
        max_retries: cfg.llm.max_retries,
        model_routing: if cfg.model_routing.enabled {
            Some(cfg.model_routing.categories.clone())
        } else {
            None
        },
    };
    let runner = AgentRunner::new(llm, tools, Arc::clone(&skills), runner_cfg);
    let dream_manager = DreamManager::new(workspace_root.clone(), cfg.dream.clone())?;
    let max_concurrent_model_calls = cfg.team.max_concurrent_model_calls;
    let reload_summary = format!(
        "reloaded model={} system_role={} reasoning_effort={} local_skills={} mcp_servers={} agents_md={}",
        cfg.llm.model,
        system_role_name,
        reasoning_effort_summary,
        local_skill_count,
        connected_mcp_servers,
        agents_md_path.display()
    );

    Ok(PreparedRuntime {
        bind,
        ws_path,
        workspace_root: preload_ctx.workspace_root.clone(),
        reloadable: RuntimeReloadState {
            runner,
            preload_ctx,
            agents_md_path,
            team_cfg: cfg.team,
            permissions: cfg.permissions,
            max_concurrent_model_calls,
        },
        dream_manager,
        reload_summary,
    })
}

/// Build one fully initialized runtime from an already parsed config object.
async fn build_runtime_from_config(
    cfg: Config,
    config_path: &Path,
) -> anyhow::Result<LoadedRuntime> {
    let prepared = prepare_runtime_from_config(cfg, config_path).await?;
    let runtime_store = RuntimeStore::new(prepared.workspace_root.clone())?;

    let root_agent_id = match runtime_store.load_root_marker()? {
        Some(marker) => marker.root_agent_id,
        None => {
            let root_id = Uuid::new_v4();
            runtime_store.save_root_marker(&RootMarker {
                root_agent_id: root_id,
            })?;
            root_id
        }
    };
    let team_state = match runtime_store.load_team_state()? {
        Some(state) => state,
        None => {
            let state = TeamState::new(root_agent_id);
            runtime_store.save_team_state(&state)?;
            state
        }
    };

    if runtime_store.load_agent_state(root_agent_id)?.is_none() {
        let root_session_store = SessionStore::new_in_relative_dir(
            prepared.workspace_root.clone(),
            Path::new(&format!("sessions/agents/{root_agent_id}")),
        )?;
        runtime_store.save_agent_state(&AgentState::new_root(
            root_agent_id,
            root_session_store.current_session_path(),
        ))?;
    }

    let session_store = Arc::new(SessionStore::new(prepared.workspace_root.clone())?);
    let interaction_store = Arc::new(InteractionStore::new(prepared.workspace_root.clone())?);
    let task_audit_store = Arc::new(TaskAuditStore::new(prepared.workspace_root.clone())?);
    let config_path = std::path::absolute(config_path)
        .with_context(|| format!("Failed to resolve config path: {}", config_path.display()))?;

    let hub = Hub::new(
        config_path,
        prepared.bind.clone(),
        prepared.ws_path.clone(),
        prepared.workspace_root.clone(),
        prepared.reloadable,
        session_store,
        interaction_store,
        task_audit_store,
        runtime_store,
        team_state,
    );
    hub.recover_runtime().await?;
    hub.replace_dream_scheduler(prepared.dream_manager);

    Ok(LoadedRuntime {
        hub,
        bind: prepared.bind,
        ws_path: prepared.ws_path,
        workspace_root: prepared.workspace_root,
    })
}

/// Run the background dream scheduler for one ready runtime.
///
/// Policy:
/// - immediately attempt a catch-up run when today has not been processed yet
/// - afterwards sleep until the next local midnight and re-check
async fn run_dream_scheduler(
    hub: Arc<Hub>,
    dream: DreamManager,
    cancel: sa_core::cancel::CancelToken,
) {
    tracing::info!("Dream scheduler started.");

    loop {
        if cancel.is_cancelled() {
            tracing::info!("Dream scheduler stopping after reload request.");
            return;
        }

        if let Err(err) = hub.run_background_dream(&dream).await {
            tracing::warn!("Dream run failed: {err:#}");
        }

        let now = chrono::Local::now();
        let next = dream.next_run_after(now);
        let sleep_for = (next - now)
            .to_std()
            .unwrap_or_else(|_| std::time::Duration::from_secs(1));
        tracing::info!(
            "Next dream run scheduled for {} (sleep {:?}).",
            next.to_rfc3339(),
            sleep_for
        );
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::info!("Dream scheduler cancelled before next wake.");
                return;
            }
            _ = tokio::time::sleep(sleep_for) => {}
        }
    }
}

/// Normalize optional free-form config values.
fn normalize_optional_string(value: Option<String>) -> Option<String> {
    let value = value?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

/// Decide the effective wire protocol for a first-run initialization request.
fn effective_init_wire_api(request: &InitializeConfigRequest) -> WireApi {
    request.wire_api.unwrap_or(match request.method {
        InitMethod::OpenAiCompatible => WireApi::ChatCompletions,
        InitMethod::AnthropicCompatible => WireApi::AnthropicMessages,
        InitMethod::Custom => WireApi::Responses,
    })
}

/// Decide the effective authentication style for a first-run initialization request.
fn effective_init_auth_style(request: &InitializeConfigRequest, wire_api: WireApi) -> AuthStyle {
    request.auth_style.unwrap_or(match request.method {
        InitMethod::OpenAiCompatible => AuthStyle::Bearer,
        InitMethod::AnthropicCompatible => AuthStyle::AnthropicAuto,
        InitMethod::Custom => match wire_api {
            WireApi::AnthropicMessages => AuthStyle::AnthropicAuto,
            WireApi::ChatCompletions | WireApi::Responses => AuthStyle::Bearer,
        },
    })
}

/// Render the initial `sa.toml` contents from one frontend onboarding request.
fn render_initial_config_toml(
    bootstrap: &BootstrapState,
    request: &InitializeConfigRequest,
) -> anyhow::Result<String> {
    let base_url = request.base_url.trim();
    let api_key = request.api_key.trim();
    let model = request.model.trim();
    if base_url.is_empty() {
        anyhow::bail!("`base_url` 不能为空");
    }
    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        anyhow::bail!("`base_url` 必须以 http:// 或 https:// 开头");
    }
    if api_key.is_empty() {
        anyhow::bail!("`api_key` 不能为空");
    }
    if model.is_empty() {
        anyhow::bail!("`model` 不能为空");
    }

    let wire_api = effective_init_wire_api(request);
    let auth_style = effective_init_auth_style(request, wire_api);
    let system_role_name = normalize_optional_string(request.system_role_name.clone());
    let reasoning_effort = normalize_optional_string(request.reasoning_effort.clone());

    let mut out = String::new();
    out.push_str("# Generated by SA first-run initialization.\n");
    out.push_str("# You can edit this file later if you need more advanced settings.\n\n");
    out.push_str("[llm]\n");
    out.push_str(&format!("base_url = {}\n", toml_string(base_url)));
    out.push_str(&format!("api_key = {}\n", toml_string(api_key)));
    out.push_str(&format!("model = {}\n", toml_string(model)));
    out.push_str(&format!(
        "wire_api = {}\n",
        toml_string(match wire_api {
            WireApi::ChatCompletions => "chat_completions",
            WireApi::Responses => "responses",
            WireApi::AnthropicMessages => "anthropic_messages",
        })
    ));
    out.push_str(&format!(
        "auth_style = {}\n",
        toml_string(match auth_style {
            AuthStyle::Bearer => "bearer",
            AuthStyle::XApiKey => "x_api_key",
            AuthStyle::AnthropicAuto => "anthropic_auto",
        })
    ));
    if let Some(system_role_name) = system_role_name {
        out.push_str(&format!(
            "system_role_name = {}\n",
            toml_string(&system_role_name)
        ));
    }
    if let Some(reasoning_effort) = reasoning_effort {
        out.push_str(&format!(
            "reasoning_effort = {}\n",
            toml_string(&reasoning_effort)
        ));
    }
    out.push('\n');
    out.push_str("[server]\n");
    out.push_str(&format!("bind = {}\n", toml_string(&bootstrap.bind)));
    out.push_str(&format!(
        "ws_path = {}\n\n",
        toml_string(&bootstrap.ws_path)
    ));
    out.push_str("[workspace]\n");
    out.push_str(&format!(
        "root_dir = {}\n",
        toml_string(
            bootstrap
                .workspace_root
                .strip_prefix(&bootstrap.config_dir)
                .ok()
                .and_then(|path| if path.as_os_str().is_empty() {
                    Some(".")
                } else {
                    path.to_str()
                })
                .unwrap_or(".")
        )
    ));
    out.push_str(&format!(
        "agents_md = {}\n",
        toml_string(
            bootstrap
                .agents_md_path
                .strip_prefix(&bootstrap.workspace_root)
                .ok()
                .and_then(Path::to_str)
                .unwrap_or("Agents.md")
        )
    ));
    let dream = sa_core::dream::DreamConfig::default();
    out.push_str("\n[dream]\n");
    out.push_str(&format!("enabled = {}\n", dream.enabled));
    out.push_str(&format!(
        "daily_note_lookback_days = {}\n",
        dream.daily_note_lookback_days
    ));
    out.push_str(&format!(
        "recent_session_segments = {}\n",
        dream.recent_session_segments
    ));
    out.push_str(&format!(
        "recent_topic_files = {}\n",
        dream.recent_topic_files
    ));
    let team = sa_core::config::TeamConfig::default();
    out.push_str("\n[team]\n");
    out.push_str(&format!("auto_resume = {}\n", team.auto_resume));
    out.push_str(&format!("max_active_agents = {}\n", team.max_active_agents));
    out.push_str(&format!(
        "max_concurrent_model_calls = {}\n",
        team.max_concurrent_model_calls
    ));

    Ok(out)
}

/// Serialize one plain string as a TOML double-quoted string literal.
fn toml_string(value: &str) -> String {
    format!("{value:?}")
}

/// Send the current ready-runtime snapshots to one freshly connected client.
fn send_ready_runtime_snapshot(hub: &Arc<Hub>, out_tx: &mpsc::UnboundedSender<ServerMessage>) {
    send_direct_server_message(
        out_tx,
        ServerMessage::PendingQuestions {
            questions: hub.pending_questions_snapshot(),
        },
    );
    send_direct_server_message(
        out_tx,
        ServerMessage::RecentShows {
            files: hub.recent_shows_snapshot(),
        },
    );
}

/// Start forwarding broadcast events from one ready runtime into one WS
/// connection.
fn spawn_event_forwarder_for_connection(
    hub: Arc<Hub>,
    out_tx: mpsc::UnboundedSender<ServerMessage>,
) -> tokio::task::JoinHandle<()> {
    let mut events_rx = hub.events_tx.subscribe();
    tokio::spawn(async move {
        loop {
            match events_rx.recv().await {
                Ok(msg) => {
                    let _ = out_tx.send(msg);
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    send_direct_server_message(
                        &out_tx,
                        ServerMessage::Error {
                            message: format!("Lagged in event stream; skipped {skipped} events"),
                        },
                    );
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Expand special placeholders in referenced paths.
///
/// Currently supported:
/// - `YYYY-MM-DD` 闁?replaced with today's date, and also yesterday's date
///   (to match common "load today + yesterday" instructions).
fn expand_date_placeholders(mut refs: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();

    let today = chrono::Local::now().date_naive();
    let yesterday = today - chrono::Duration::days(1);

    for r in refs.drain(..) {
        if r.contains("YYYY-MM-DD") {
            out.push(r.replace("YYYY-MM-DD", &today.to_string()));
            out.push(r.replace("YYYY-MM-DD", &yesterday.to_string()));
        } else {
            out.push(r);
        }
    }

    // De-duplicate while preserving order.
    let mut seen = HashSet::<String>::new();
    out.into_iter().filter(|r| seen.insert(r.clone())).collect()
}

/// Candidate `bash` programs to try when starting a background task.
fn candidate_bash_programs() -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("bash")];

    for env_name in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(root) = std::env::var(env_name) {
            let root = PathBuf::from(root);
            out.push(root.join("Git").join("bin").join("bash.exe"));
            out.push(root.join("Git").join("usr").join("bin").join("bash.exe"));
        }
    }

    out
}

/// Pump one child-process pipe into the central log writer channel.
///
/// The background task runtime keeps a single on-disk log file. Stdout and
/// stderr are read concurrently in small chunks and forwarded to the main task,
/// which serializes them into that file. This keeps output visible while the
/// process is still running instead of buffering everything until exit.
async fn stream_background_pipe_to_channel<R>(
    mut reader: R,
    tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt as _;

    let mut buffer = vec![0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        if tx.send(buffer[..read].to_vec()).is_err() {
            return Ok(());
        }
    }
}

/// Run one background Bash command and tee stdout/stderr into one log file.
async fn run_background_bash_command(
    request: &StartTerminalTaskRequest,
    output_path: &Path,
) -> anyhow::Result<i32> {
    use tokio::io::AsyncWriteExt as _;
    use tokio::process::Command;
    use tokio::sync::mpsc;

    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!(
                "Failed to create runtime task output directory: {}",
                parent.display()
            )
        })?;
    }

    let timeout = request
        .timeout_seconds
        .map(std::time::Duration::from_secs)
        .unwrap_or(std::time::Duration::from_secs(300))
        .min(std::time::Duration::from_secs(1800));

    let mut last_not_found: Option<anyhow::Error> = None;
    for program in candidate_bash_programs() {
        let mut cmd = Command::new(&program);
        cmd.arg("-lc");
        cmd.arg(&request.command);
        cmd.current_dir(&request.workdir);
        cmd.kill_on_drop(true);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                last_not_found = Some(anyhow::Error::new(err).context(format!(
                    "Bash executable not found at {}",
                    program.display()
                )));
                continue;
            }
            Err(err) => {
                return Err(anyhow::Error::new(err))
                    .with_context(|| format!("Failed to execute Bash via {}", program.display()));
            }
        };

        let stdout = child.stdout.take().with_context(|| {
            format!(
                "Failed to capture stdout for background Bash via {}",
                program.display()
            )
        })?;
        let stderr = child.stderr.take().with_context(|| {
            format!(
                "Failed to capture stderr for background Bash via {}",
                program.display()
            )
        })?;

        let mut file = match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)
            .await
        {
            Ok(file) => file,
            Err(err) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err(anyhow::Error::new(err).context(format!(
                    "Failed to create background task log securely: {}",
                    output_path.display()
                )));
            }
        };

        let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let stdout_task = {
            let tx = chunk_tx.clone();
            tokio::spawn(async move { stream_background_pipe_to_channel(stdout, tx).await })
        };
        let stderr_task = {
            let tx = chunk_tx.clone();
            tokio::spawn(async move { stream_background_pipe_to_channel(stderr, tx).await })
        };
        drop(chunk_tx);

        let mut timed_out = false;
        let mut child_status: Option<std::process::ExitStatus> = None;
        let mut streams_closed = false;
        let timeout_sleep = tokio::time::sleep(timeout);
        tokio::pin!(timeout_sleep);
        let mut poll_interval = tokio::time::interval(std::time::Duration::from_millis(25));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            if child_status.is_some() && streams_closed {
                break;
            }

            tokio::select! {
                maybe_chunk = chunk_rx.recv(), if !streams_closed => {
                    match maybe_chunk {
                        Some(chunk) => {
                            file.write_all(&chunk).await.with_context(|| {
                                format!("Failed to append streamed log: {}", output_path.display())
                            })?;
                            file.flush().await.with_context(|| {
                                format!("Failed to flush background task log: {}", output_path.display())
                            })?;
                        }
                        None => {
                            streams_closed = true;
                        }
                    }
                }
                _ = poll_interval.tick(), if child_status.is_none() => {
                    child_status = child.try_wait().with_context(|| {
                        format!("Failed to poll background Bash via {}", program.display())
                    })?;
                }
                _ = &mut timeout_sleep, if !timed_out && child_status.is_none() => {
                    timed_out = true;
                    match child.kill().await {
                        Ok(()) => {}
                        Err(err) if err.kind() == std::io::ErrorKind::InvalidInput => {}
                        Err(err) => {
                            return Err(anyhow::Error::new(err)).with_context(|| {
                                format!(
                                    "Failed to terminate timed-out background Bash via {}",
                                    program.display()
                                )
                            });
                        }
                    }
                    child_status = Some(child.wait().await.with_context(|| {
                        format!(
                            "Failed to wait for timed-out background Bash via {}",
                            program.display()
                        )
                    })?);
                }
            }

            if child_status.is_none() {
                child_status = child.try_wait().with_context(|| {
                    format!("Failed to poll background Bash via {}", program.display())
                })?;
            }
        }

        for (label, task) in [("stdout", stdout_task), ("stderr", stderr_task)] {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!(
                            "Failed while streaming background {label} for Bash via {}",
                            program.display()
                        )
                    });
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!(
                            "Background {label} stream task crashed for Bash via {}",
                            program.display()
                        )
                    });
                }
            }
        }

        file.flush().await.with_context(|| {
            format!(
                "Failed to flush background task log: {}",
                output_path.display()
            )
        })?;

        if timed_out {
            return Err(anyhow::anyhow!(
                "Background Bash command timed out after {:?}",
                timeout
            ));
        }

        let status = child_status.expect("background Bash loop must observe child exit");
        return Ok(status.code().unwrap_or(-1));
    }

    Err(last_not_found.unwrap_or_else(|| {
        anyhow::anyhow!("No usable bash executable was found for background task execution.")
    }))
}

/// Decorate nested agent text so user-visible messages stay traceable.
#[allow(dead_code)]
fn decorate_nested_text(depth: u32, text: &str) -> String {
    if depth == 0 {
        return text.to_string();
    }

    format!("[subagent depth={depth}] {text}")
}

/// Decorate nested agent prompts for `Ask`.
#[allow(dead_code)]
fn decorate_nested_prompt(depth: u32, prompt: &str) -> String {
    if depth == 0 {
        return prompt.to_string();
    }

    format!("[subagent depth={depth}] {prompt}")
}

/// Validate a user answer against the question schema that produced it.
fn validate_user_answer(
    question: &UserQuestion,
    answer: &UserQuestionAnswer,
) -> anyhow::Result<()> {
    let free_text = answer.free_text.as_deref().map(str::trim).unwrap_or("");

    let allows_free_text = question.allow_free_text || question.mode == QuestionMode::Text;
    if !allows_free_text && !free_text.is_empty() {
        anyhow::bail!("This question does not accept free-text input");
    }

    let mut seen_ids = HashSet::<String>::new();
    for id in &answer.selected_option_ids {
        if !seen_ids.insert(id.clone()) {
            anyhow::bail!("Duplicate option id in answer: {id}");
        }
        if !question.options.iter().any(|option| option.id == *id) {
            anyhow::bail!("Unknown option id in answer: {id}");
        }
    }

    match question.mode {
        QuestionMode::Text => {
            if !answer.selected_option_ids.is_empty() {
                anyhow::bail!("Text questions do not accept selected options");
            }
            if free_text.is_empty() {
                anyhow::bail!("Text questions require a free-text answer");
            }
        }
        QuestionMode::SingleChoice => {
            if answer.selected_option_ids.len() > 1 {
                anyhow::bail!("Single-choice questions accept at most one selected option");
            }
            if answer.selected_option_ids.is_empty() && free_text.is_empty() {
                anyhow::bail!("Single-choice questions require one selection or free text");
            }
        }
        QuestionMode::MultiChoice => {
            if answer.selected_option_ids.is_empty() && free_text.is_empty() {
                anyhow::bail!("Multi-choice questions require at least one selection or free text");
            }
        }
    }

    Ok(())
}

// ── Memory panel helpers ────────────────────────────────────────────────────────

use sa_core::ws_protocol::MemoryFact;

/// Scan memory markdown files and compute aggregate statistics.
async fn compute_memory_stats(
    memory_dir: &std::path::Path,
) -> anyhow::Result<(usize, usize, usize, f64, Option<String>, Option<String>)> {
    if !memory_dir.exists() {
        return Ok((0, 0, 0, 0.0, None, None));
    }

    let all_facts = collect_all_facts(memory_dir).await?;
    let mut subjects: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut predicates: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_confidence = 0.0f64;
    let mut oldest: Option<String> = None;
    let mut newest: Option<String> = None;

    for fact in &all_facts {
        subjects.insert(fact.subject.clone());
        predicates.insert(fact.predicate.clone());
        total_confidence += fact.confidence;
        if oldest.is_none() || fact.created_at < *oldest.as_ref().unwrap() {
            oldest = Some(fact.created_at.clone());
        }
        if newest.is_none() || fact.created_at > *newest.as_ref().unwrap() {
            newest = Some(fact.created_at.clone());
        }
    }

    let count = all_facts.len();
    Ok((
        count,
        subjects.len(),
        predicates.len(),
        if count > 0 { total_confidence / count as f64 } else { 0.0 },
        oldest,
        newest,
    ))
}

/// Query memory facts with optional search and pagination.
async fn query_memory_facts(
    memory_dir: &std::path::Path,
    query: Option<&str>,
    limit: usize,
    offset: usize,
) -> anyhow::Result<(Vec<MemoryFact>, usize)> {
    let all_facts = collect_all_facts(memory_dir).await?;

    let filtered: Vec<MemoryFact> = if let Some(q) = query {
        let q_lower = q.to_lowercase();
        all_facts
            .into_iter()
            .filter(|f| {
                f.subject.to_lowercase().contains(&q_lower)
                    || f.predicate.to_lowercase().contains(&q_lower)
                    || f.object.to_lowercase().contains(&q_lower)
            })
            .collect()
    } else {
        all_facts
    };

    let total = filtered.len();
    let page: Vec<MemoryFact> = filtered.into_iter().skip(offset).take(limit).collect();
    Ok((page, total))
}

/// Traverse memory graph from a starting subject.
async fn traverse_memory(
    memory_dir: &std::path::Path,
    subject: &str,
    _max_hops: usize,
    max_results: usize,
) -> anyhow::Result<Vec<MemoryFact>> {
    let all_facts = collect_all_facts(memory_dir).await?;
    let subject_lower = subject.to_lowercase();

    let direct: Vec<MemoryFact> = all_facts
        .iter()
        .filter(|f| {
            f.subject.to_lowercase().contains(&subject_lower)
                || f.object.to_lowercase().contains(&subject_lower)
        })
        .cloned()
        .take(max_results)
        .collect();

    Ok(direct)
}

/// Apply confidence decay to all facts (placeholder — marks files with decay timestamp).
async fn decay_memory(memory_dir: &std::path::Path) -> anyhow::Result<usize> {
    if !memory_dir.exists() {
        return Ok(0);
    }
    let mut entries = tokio::fs::read_dir(memory_dir).await?;
    let mut affected = 0usize;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let content = tokio::fs::read_to_string(&path).await?;
        let noted_date = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let facts = extract_facts(&content, noted_date);
        affected += facts.len();
    }
    Ok(affected)
}

/// Prune facts below a confidence threshold (placeholder).
async fn prune_memory(memory_dir: &std::path::Path, min_confidence: f64) -> anyhow::Result<usize> {
    if !memory_dir.exists() {
        return Ok(0);
    }
    let all_facts = collect_all_facts(memory_dir).await?;
    let pruned = all_facts.iter().filter(|f| f.confidence < min_confidence).count();
    Ok(pruned)
}

/// Helper: scan all .md files in memory dir, extract facts as MemoryFact.
async fn collect_all_facts(memory_dir: &std::path::Path) -> anyhow::Result<Vec<MemoryFact>> {
    if !memory_dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = tokio::fs::read_dir(memory_dir).await?;
    let mut all_facts = Vec::new();

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let content = tokio::fs::read_to_string(&path).await?;
        let noted_date = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let facts = extract_facts(&content, noted_date);
        for (i, fact) in facts.iter().enumerate() {
            all_facts.push(MemoryFact {
                id: format!("{}-{}", noted_date, i),
                subject: fact.category.clone(),
                predicate: fact.predicate().to_string(),
                object: fact.content.clone(),
                confidence: fact.confidence,
                source: Some(noted_date.to_string()),
                created_at: fact.source_date.clone(),
                updated_at: fact.valid_from.clone(),
            });
        }
    }

    all_facts.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(all_facts)
}

/// Mirror one event in a stable multi-line format.
fn mirror_event_line(prefix: &str, event: &Event) {
    let agent_meta = event
        .agent
        .as_ref()
        .map(|agent| {
            format!(
                "[agent={}][agent_id={}][agent_label={}]",
                agent.display_name, agent.agent_id, agent.agent_label
            )
        })
        .unwrap_or_default();
    let header = format!(
        "{prefix}[kind={:?}][task={}][event_id={}][ts={}]{} ",
        event.kind, event.task_id, event.event_id, event.ts, agent_meta
    );

    let mut lines = event.message.lines();
    match lines.next() {
        Some(first) => {
            eprintln!("{header}{first}");
            for line in lines {
                eprintln!("{}{}", " ".repeat(header.len()), line);
            }
        }
        None => eprintln!("{header}"),
    }
}

/// Mirror one `Show` payload in a stable format.
fn mirror_shown_file(prefix: &str, file: &UserVisibleFile) {
    let title = file.title.as_deref().unwrap_or(&file.path);
    let agent_meta = file
        .agent
        .as_ref()
        .map(|agent| {
            format!(
                "[agent={}][agent_id={}][agent_label={}]",
                agent.display_name, agent.agent_id, agent.agent_label
            )
        })
        .unwrap_or_default();
    eprintln!(
        "{prefix}[task={}]{agent_meta}[title={}][prompt={}] path={} media_type={} bytes={}",
        file.task_id, title, file.prompt, file.path, file.media_type, file.bytes
    );
    match file.encoding {
        UserVisibleFileEncoding::Utf8 => {
            eprintln!("{}", file.content);
        }
        UserVisibleFileEncoding::Base64 => {
            eprintln!(
                "<binary payload omitted; {} bytes encoded as base64>",
                file.bytes
            );
        }
    }
}

/// Mirror one outbound frontend message to the backend terminal.
///
/// This stays global instead of hanging off `Hub` so both the fully initialized
/// runtime and the first-run bootstrap mode can emit the same diagnostics.
fn mirror_server_message(msg: &ServerMessage) {
    match msg {
        ServerMessage::ServerHello { hello } => {
            eprintln!(
                "[frontend][server_hello][protocol={}][server={}][version={}][bucket={}][machine_hint={}]",
                hello.protocol,
                hello.server_name,
                hello.server_version,
                hello.time_bucket,
                hello.machine_hint
            );
        }
        ServerMessage::HelloReject { reject } => {
            eprintln!(
                "[frontend][hello_reject][protocol={}][server={}][version={}] {}",
                reject.protocol, reject.server_name, reject.server_version, reject.reason
            );
        }
        ServerMessage::Accepted { task_id } => {
            eprintln!("[frontend][accepted][task={task_id}]");
        }
        ServerMessage::History { events } => {
            eprintln!("[frontend][history][count={}]", events.len());
            for event in events {
                mirror_event_line("[frontend][history-event]", event);
            }
        }
        ServerMessage::Event { event } => {
            mirror_event_line("[frontend][event]", event);
        }
        ServerMessage::Question { question } => {
            let agent_meta = question
                .agent
                .as_ref()
                .map(|agent| {
                    format!(
                        "[agent={}][agent_id={}][agent_label={}]",
                        agent.display_name, agent.agent_id, agent.agent_label
                    )
                })
                .unwrap_or_default();
            eprintln!(
                "[frontend][ask][task={}][id={}]{} {}",
                question.task_id, question.question_id, agent_meta, question.prompt
            );
            for (index, option) in question.options.iter().enumerate() {
                match option.description.as_deref() {
                    Some(description) => eprintln!(
                        "  {}. {} [{}] - {}",
                        index + 1,
                        option.label,
                        option.id,
                        description
                    ),
                    None => eprintln!("  {}. {} [{}]", index + 1, option.label, option.id),
                }
            }
            if question.allow_free_text {
                eprintln!("  free text: allowed");
            }
        }
        ServerMessage::PendingQuestions { questions } => {
            eprintln!("[frontend][pending_questions][count={}]", questions.len());
            for question in questions {
                eprintln!(
                    "  [task={}][id={}] {}",
                    question.task_id, question.question_id, question.prompt
                );
            }
        }
        ServerMessage::QuestionResolved { question_id } => {
            eprintln!("[frontend][question_resolved][id={question_id}]");
        }
        ServerMessage::Show { file } => {
            mirror_shown_file("[frontend][show]", file);
        }
        ServerMessage::RecentShows { files } => {
            eprintln!("[frontend][recent_shows][count={}]", files.len());
            for file in files {
                mirror_shown_file("  [recent_show]", file);
            }
        }
        ServerMessage::InitRequired { request } => {
            eprintln!(
                "[frontend][init_required] config_path={} workspace_root={} agents_md={}",
                request.config_path, request.workspace_root, request.agents_md_path
            );
            for method in &request.methods {
                eprintln!(
                    "  [method={:?}] {} - {}",
                    method.id, method.label, method.description
                );
            }
        }
        ServerMessage::InitCompleted { info } => {
            eprintln!(
                "[frontend][init_completed] {} config_path={} workspace_root={}",
                info.message, info.config_path, info.workspace_root
            );
        }
        ServerMessage::InitFailed { error } => {
            eprintln!("[frontend][init_failed] {}", error.message);
            if let Some(detail) = &error.detail {
                eprintln!("{detail}");
            }
        }
        ServerMessage::Error { message } => {
            eprintln!("[frontend][error] {message}");
        }
        ServerMessage::ImageUploaded { upload_id, saved_path } => {
            eprintln!("[frontend][image_uploaded] upload_id={upload_id} saved={saved_path}");
        }
        ServerMessage::ConfigSnapshot { config_path, .. } => {
            eprintln!("[frontend][config_snapshot] path={config_path}");
        }
        ServerMessage::ConfigUpdated { summary } => {
            eprintln!("[frontend][config_updated] {summary}");
        }
        ServerMessage::ConfigUpdateFailed { message, .. } => {
            eprintln!("[frontend][config_update_failed] {message}");
        }
        ServerMessage::MemoryFacts { facts, total, offset } => {
            eprintln!("[frontend][memory_facts] count={} total={} offset={}", facts.len(), total, offset);
        }
        ServerMessage::MemoryGraph { subject, facts, hops } => {
            eprintln!("[frontend][memory_graph] subject={subject} facts={} hops={hops}", facts.len());
        }
        ServerMessage::MemoryStats { total_facts, .. } => {
            eprintln!("[frontend][memory_stats] total_facts={total_facts}");
        }
        ServerMessage::MemoryOperationResult { operation, affected, summary } => {
            eprintln!("[frontend][memory_operation_result] op={operation} affected={affected} {summary}");
        }
        ServerMessage::CacheStats { total_requests, total_cache_hits, total_cache_misses, hit_rate, estimated_savings, .. } => {
            eprintln!(
                "[frontend][cache_stats] requests={} hits={} misses={} hit_rate={:.1}% savings=${:.4}",
                total_requests, total_cache_hits, total_cache_misses, hit_rate * 100.0, estimated_savings
            );
        }
        ServerMessage::ContextInfo { used_tokens, max_tokens, message_count } => {
            eprintln!(
                "[frontend][context_info] used={}K max={}K messages={}",
                used_tokens / 1000, max_tokens / 1000, message_count
            );
        }
        ServerMessage::WorkflowUpdate { workflow_id, nodes, edges } => {
            eprintln!(
                "[frontend][workflow_update] id={} nodes={} edges={}",
                workflow_id, nodes.len(), edges.len()
            );
        }
        ServerMessage::WorkflowStarted { run_id, workflow_name, .. } => {
            eprintln!("[frontend][workflow_started] run={run_id} name={workflow_name}");
        }
        ServerMessage::WorkflowNodeUpdate { run_id, node_id, status, .. } => {
            eprintln!("[frontend][workflow_node_update] run={run_id} node={node_id} status={status:?}");
        }
        ServerMessage::WorkflowCompleted { run_id, summary } => {
            eprintln!("[frontend][workflow_completed] run={run_id} {summary}");
        }
        ServerMessage::WorkflowFailed { run_id, node_id, error } => {
            eprintln!("[frontend][workflow_failed] run={run_id} node={node_id:?} {error}");
        }
        ServerMessage::ApprovalRequested { run_id, node_id, prompt } => {
            eprintln!("[frontend][approval_requested] run={run_id} node={node_id} prompt={prompt}");
        }
        ServerMessage::PlanCreated { plan_id, steps } => {
            eprintln!("[frontend][plan_created] plan={plan_id} steps={}", steps.len());
        }
        ServerMessage::PlanStepUpdate { plan_id, step_id, status } => {
            eprintln!("[frontend][plan_step_update] plan={plan_id} step={step_id} status={status:?}");
        }
        ServerMessage::ChatModeChanged { mode } => {
            eprintln!("[frontend][chat_mode_changed] mode={mode:?}");
        }
        ServerMessage::MemoryUpdated { source, report } => {
            eprintln!("[frontend][memory_updated] source={source} report={:?}", report);
        }        ServerMessage::MemoryReport { report_path, content } => {
            eprintln!("[frontend][memory_report] path={report_path} bytes={}", content.len());
        }
        ServerMessage::SkillActivated { skill_name, description, trigger, orchestration_mode } => {
            eprintln!("[frontend][skill_activated] skill={skill_name} trigger={trigger} mode={:?}", orchestration_mode);
            let _ = description;
        }        ServerMessage::SkillsList { skills } => {
            eprintln!("[frontend][skills_list] count={}", skills.len());
        }
        ServerMessage::VerifyPass { step, message } => {
            eprintln!("[frontend][verify_pass] step={step} message={message}");
        }
        ServerMessage::VerifyFail { step, error, retry_count } => {
            eprintln!("[frontend][verify_fail] step={step} error={error} retries={:?}", retry_count);
        }
        ServerMessage::SkillDoc { name, description, content } => {
            eprintln!("[frontend][skill_doc] name={name} description_len={} content_len={}", description.len(), content.len());
        }
        ServerMessage::ApiKeyTestResult { ok, model, provider, latency_ms, error } => {
            eprintln!("[frontend][api_key_test_result] ok={ok} model={model} provider={provider} latency={latency_ms}ms error={:?}", error);
        }
    }
}


/// Send one direct per-connection message and mirror it to the backend
/// terminal.
fn send_direct_server_message(out_tx: &mpsc::UnboundedSender<ServerMessage>, msg: ServerMessage) {
    mirror_server_message(&msg);
    let _ = out_tx.send(msg);
}

/// Send one handshake message before the per-connection writer task exists.
async fn send_handshake_server_message(
    ws_tx: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    msg: ServerMessage,
) -> anyhow::Result<()> {
    mirror_server_message(&msg);

    let text = serde_json::to_string(&msg).context("Failed to serialize WS handshake message")?;
    ws_tx
        .send(Message::Text(text.into()))
        .await
        .context("Failed to send WS handshake message")?;

    Ok(())
}

/// Close one connection after the handshake has already been rejected.
async fn close_rejected_handshake(
    connection_id: Uuid,
    stage: &'static str,
    ws_tx: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) {
    let close_frame = CloseFrame {
        code: close_code::POLICY,
        reason: WS_HANDSHAKE_REJECT_CLOSE_REASON.into(),
    };

    match ws_tx.send(Message::Close(Some(close_frame.clone()))).await {
        Ok(()) => {
            tracing::warn!(
                %connection_id,
                stage,
                close_code = close_frame.code,
                close_reason = %close_frame.reason,
                "Closed WS connection after handshake rejection"
            );
        }
        Err(err) => {
            tracing::warn!(
                %connection_id,
                stage,
                %err,
                "Failed to close WS connection after handshake rejection"
            );
        }
    }
}

/// Reject the handshake, send `hello_reject`, and then explicitly close the socket.
async fn reject_handshake(
    connection_id: Uuid,
    stage: &'static str,
    server_version: &str,
    ws_tx: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    reason: impl Into<String>,
) {
    let reason = reason.into();
    tracing::warn!(
        %connection_id,
        stage,
        reason = %reason,
        "Rejecting WS handshake"
    );

    if let Err(err) = send_handshake_server_message(
        ws_tx,
        ServerMessage::HelloReject {
            reject: build_hello_reject(reason, server_version),
        },
    )
    .await
    {
        tracing::warn!(
            %connection_id,
            stage,
            %err,
            "Failed to send hello_reject for rejected WS handshake"
        );
    }

    close_rejected_handshake(connection_id, stage, ws_tx).await;
}

/// WS upgrade handler.
async fn ws_route(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_session(socket, state))
}

/// Handle a single WS connection with the production handshake timeout.
async fn ws_session(socket: WebSocket, state: Arc<ServerState>) {
    ws_session_with_timeout(
        socket,
        state,
        std::time::Duration::from_secs(WS_HANDSHAKE_TIMEOUT_SECS),
    )
    .await;
}

/// Handle a single WS connection with an explicit handshake timeout.
///
/// Tests use this variant directly so timeout behavior can be verified without
/// sleeping for the full production timeout.

/// Inject a user-uploaded image into the current input-owner agent's
/// conversation by sending a specially-formatted mailbox message.
///
/// The message uses the prefix `__IMAGE__:data:...` so that
/// `mailbox_entry_to_message` can recognise it and build a multimodal
/// `ChatMessage` with `ContentPart::Image`.
async fn inject_user_image(
    hub: &Arc<Hub>,
    data_url: &str,
    saved_path: &str,
) -> anyhow::Result<()> {
    let target_agent_id = hub.effective_input_owner().await?;

    let caption = "同学上传了一张图片，请查看并分析。";
    let mailbox_text = format!(
        "__IMAGE__:{data_url}\n__IMAGE_PATH__:{saved_path}"
    );
    let full_message = format!("{caption}\n{mailbox_text}");

    hub.append_mailbox_message(
        target_agent_id,
        sa_core::runtime::state::MailboxEntryKind::UserInput,
        None,
        Some("同学".to_string()),
        full_message,
        None,
        None,
    )
    .await?;

    Ok(())
}

async fn ws_session_with_timeout(
    socket: WebSocket,
    state: Arc<ServerState>,
    handshake_timeout: std::time::Duration,
) {
    let connection_id = Uuid::new_v4();
    let handshake_timeout_ms = handshake_timeout.as_millis() as u64;
    let server_version = env!("CARGO_PKG_VERSION");

    tracing::info!(
        %connection_id,
        handshake_timeout_ms,
        "WS connection opened"
    );

    // We split the socket so we can read and write concurrently.
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Mandatory client-first handshake.
    let first_frame = match tokio::time::timeout(handshake_timeout, ws_rx.next()).await {
        Ok(Some(Ok(frame))) => frame,
        Ok(Some(Err(err))) => {
            reject_handshake(
                connection_id,
                "read_initial_frame",
                server_version,
                &mut ws_tx,
                format!("Failed to read initial handshake frame: {err}"),
            )
            .await;
            return;
        }
        Ok(None) => {
            tracing::info!(
                %connection_id,
                "WS peer closed the connection before completing the handshake"
            );
            return;
        }
        Err(_) => {
            reject_handshake(
                connection_id,
                "handshake_timeout",
                server_version,
                &mut ws_tx,
                format!("Handshake timed out after {} ms", handshake_timeout_ms),
            )
            .await;
            return;
        }
    };

    let Message::Text(first_text) = first_frame else {
        reject_handshake(
            connection_id,
            "first_frame_not_text",
            server_version,
            &mut ws_tx,
            "The first client frame must be a UTF-8 JSON text frame",
        )
        .await;
        return;
    };

    let first_msg = match serde_json::from_str::<ClientMessage>(&first_text) {
        Ok(msg) => msg,
        Err(err) => {
            reject_handshake(
                connection_id,
                "invalid_handshake_json",
                server_version,
                &mut ws_tx,
                format!("Invalid handshake JSON: {err}"),
            )
            .await;
            return;
        }
    };

    let client_hello = match first_msg {
        ClientMessage::ClientHello { hello } => hello,
        other => {
            reject_handshake(
                connection_id,
                "first_message_wrong_type",
                server_version,
                &mut ws_tx,
                format!(
                    "The first client message must be `client_hello`, got `{:?}`",
                    other
                ),
            )
            .await;
            return;
        }
    };

    if let Err(err) = verify_client_hello(&client_hello, std::time::SystemTime::now()) {
        reject_handshake(
            connection_id,
            "client_hello_verification_failed",
            server_version,
            &mut ws_tx,
            format!("Client hello verification failed: {err}"),
        )
        .await;
        return;
    }

    let server_hello = match build_server_hello(
        &state.ws_identity,
        server_version,
        &client_hello.client_nonce,
    ) {
        Ok(hello) => hello,
        Err(err) => {
            reject_handshake(
                connection_id,
                "build_server_hello_failed",
                server_version,
                &mut ws_tx,
                format!("Failed to build server hello: {err}"),
            )
            .await;
            return;
        }
    };

    if let Err(err) = send_handshake_server_message(
        &mut ws_tx,
        ServerMessage::ServerHello {
            hello: server_hello,
        },
    )
    .await
    {
        tracing::warn!(
            %connection_id,
            %err,
            "Failed to send server_hello during WS handshake"
        );
        return;
    }

    tracing::info!(
        %connection_id,
        client_name = %client_hello.client_name,
        client_version = %client_hello.client_version,
        time_bucket = client_hello.time_bucket,
        "Accepted WS handshake"
    );

    // Outbound message channel (single writer task owns `ws_tx`).
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<ServerMessage>();

    // Writer task: serialize ServerMessage -> WS text frame.
    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            let Ok(text) = serde_json::to_string(&msg) else {
                continue;
            };
            if ws_tx.send(Message::Text(text.into())).await.is_err() {
                break;
            }
        }
    });

    // Shared approval channels for workflow nodes.
    // Maps (run_id, node_id) -> sender.
    use sa_core::workflow_engine::ApprovalResponse;
    let approval_txs: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<(Uuid, String), tokio::sync::mpsc::UnboundedSender<ApprovalResponse>>>> =
        std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new()));

    // Chat mode tracking (per-connection).
    let mut chat_mode: ChatMode = ChatMode::default();

    let mut forwarder: Option<tokio::task::JoinHandle<()>> = None;
    match state.runtime_snapshot().await {
        RuntimeState::Ready(hub) => {
            forwarder = Some(spawn_event_forwarder_for_connection(
                Arc::clone(&hub),
                out_tx.clone(),
            ));
            send_ready_runtime_snapshot(&hub, &out_tx);
        }
        RuntimeState::Bootstrap(bootstrap) => {
            send_direct_server_message(&out_tx, bootstrap.init_required_message());
        }
    }

    // Reader loop: handle client requests.
    while let Some(Ok(frame)) = ws_rx.next().await {
        match frame {
            Message::Text(text) => {
                let msg = match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(msg) => msg,
                    Err(err) => {
                        // Extract the message type from raw JSON for better diagnostics.
                        let msg_type = serde_json::from_str::<serde_json::Value>(&text)
                            .ok()
                            .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(String::from))
                            .unwrap_or_else(|| "<unknown>".into());

                        let error_message = if err.to_string().contains("unknown variant") {
                            format!(
                                "后端暂不支持 '{msg_type}' 消息类型。请更新 sa.exe 或运行 `cargo build -p sa --release` 重新构建。原始错误: {err}"
                            )
                        } else {
                            format!("消息解析失败（type={msg_type}）: {err}")
                        };

                        send_direct_server_message(
                            &out_tx,
                            ServerMessage::Error {
                                message: error_message,
                            },
                        );
                        continue;
                    }
                };
                // Silently ignore known-but-disabled message types.
                // Future protocol additions go here before the match below.

                match msg {
                    ClientMessage::ClientHello { .. } => {
                        send_direct_server_message(
                            &out_tx,
                            ServerMessage::Error {
                                message: "`client_hello` is only allowed as the first message on a connection".to_string(),
                            },
                        );
                    }
                    ClientMessage::InitializeConfig { request } => {
                        match state.initialize_from_request(request).await {
                            Ok(info) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::InitCompleted { info },
                                );

                                if forwarder.is_none() {
                                    if let RuntimeState::Ready(hub) = state.runtime_snapshot().await
                                    {
                                        send_ready_runtime_snapshot(&hub, &out_tx);
                                        forwarder = Some(spawn_event_forwarder_for_connection(
                                            hub,
                                            out_tx.clone(),
                                        ));
                                    }
                                }
                            }
                            Err(err) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::InitFailed {
                                        error: InitFailed {
                                            message: "初始化失败，配置文件尚未生效。".to_string(),
                                            detail: Some(format!("{err:#}")),
                                        },
                                    },
                                );
                            }
                        }
                    }
                    ClientMessage::Submit { task_id, task } => {
                        if chat_mode == ChatMode::Plan {
                            // Plan mode: generate a plan from the user request.
                            let task_id = task_id.unwrap_or_else(Uuid::new_v4);
                            let out_tx_plan = out_tx.clone();
                            let config_path = match state.runtime_snapshot().await {
                                RuntimeState::Ready(ref hub) => hub.config_path.clone(),
                                _ => {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: "Runtime not ready for plan generation".to_string(),
                                        },
                                    );
                                    continue;
                                }
                            };

                            // Get LLM config
                            let llm_config = match load_config_from_file(&config_path) {
                                Ok(cfg) => sa_core::plan_engine::PlanLlmConfig {
                                    base_url: cfg.llm.base_url.clone(),
                                    api_key: cfg.llm.api_key.clone(),
                                    model: cfg.llm.model.clone(),
                                },
                                Err(err) => {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: format!("Failed to load config: {err}"),
                                        },
                                    );
                                    continue;
                                }
                            };

                            let context = "You are an AI planning assistant. Generate clear, actionable steps that can be executed sequentially.";

                            tokio::spawn(async move {
                                match sa_core::plan_engine::generate_plan(&task, context, &llm_config).await {
                                    Ok(plan) => {
                                        let _ = sa_core::plan_engine::save_plan(
                                            &std::path::PathBuf::from("."),
                                            &plan,
                                        ).await;
                                        let _ = out_tx_plan.send(ServerMessage::PlanCreated {
                                            plan_id: plan.plan_id,
                                            steps: plan.steps,
                                        });
                                    }
                                    Err(err) => {
                                        let _ = out_tx_plan.send(ServerMessage::Error {
                                            message: format!("Plan generation failed: {err}"),
                                        });
                                    }
                                }
                            });

                            send_direct_server_message(
                                &out_tx,
                                ServerMessage::Accepted { task_id },
                            );
                        } else if chat_mode == ChatMode::Workflow {
                            // Workflow mode: LLM decomposes task into DAG, then execute.
                            let task_id = task_id.unwrap_or_else(Uuid::new_v4);
                            let out_tx_wf = out_tx.clone();
                            let config_path = match state.runtime_snapshot().await {
                                RuntimeState::Ready(ref hub) => hub.config_path.clone(),
                                _ => {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: "Runtime not ready for workflow generation".to_string(),
                                        },
                                    );
                                    continue;
                                }
                            };

                            // Get LLM config
                            let llm_config = match load_config_from_file(&config_path) {
                                Ok(cfg) => sa_core::plan_engine::PlanLlmConfig {
                                    base_url: cfg.llm.base_url.clone(),
                                    api_key: cfg.llm.api_key.clone(),
                                    model: cfg.llm.model.clone(),
                                },
                                Err(err) => {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: format!("Failed to load config: {err}"),
                                        },
                                    );
                                    continue;
                                }
                            };

                            // Get workspace root and approval_txs
                            let (ws_root, approval_txs_clone) = match state.runtime_snapshot().await {
                                RuntimeState::Ready(hub) => (hub.workspace_root.clone(), approval_txs.clone()),
                                _ => {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: "Runtime not ready".to_string(),
                                        },
                                    );
                                    continue;
                                }
                            };

                            let context = "You are a workflow decomposition assistant. Break tasks into a DAG of executable nodes.";

                            // Send Accepted immediately
                            send_direct_server_message(
                                &out_tx,
                                ServerMessage::Accepted { task_id },
                            );

                            tokio::spawn(async move {
                                // 1. LLM generates the workflow DAG
                                let def = match sa_core::plan_engine::generate_workflow(&task, context, &llm_config).await {
                                    Ok(def) => {
                                        tracing::info!("Generated workflow '{}' with {} nodes:", def.name, def.nodes.len());
                                        for n in &def.nodes {
                                            tracing::info!("  Node '{}' type={:?} deps={:?} prompt={:?} cmd={:?}",
                                                n.id, n.node_type, n.depends_on,
                                                n.prompt.as_deref().map(|s| &s[..s.len().min(80)]),
                                                n.command.as_deref().map(|s| &s[..s.len().min(80)]));
                                        }
                                        def
                                    }
                                    Err(err) => {
                                        let _ = out_tx_wf.send(ServerMessage::Error {
                                            message: format!("Workflow generation failed: {err}"),
                                        });
                                        return;
                                    }
                                };

                                // 2. Create channels for progress and approval
                                let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
                                let (approval_tx, approval_rx) = tokio::sync::mpsc::unbounded_channel::<sa_core::workflow_engine::ApprovalResponse>();

                                // 3. Spawn forwarder: WorkflowEvent → ServerMessage
                                let forward_tx = out_tx_wf.clone();
                                let approval_txs_forward = approval_txs_clone.clone();
                                let forwarder = tokio::spawn(async move {
                                    while let Some(event) = progress_rx.recv().await {
                                        use sa_core::workflow_engine::WorkflowEvent;
                                        let msg = match event {
                                            WorkflowEvent::Started { run_id, workflow_name } => {
                                                sa_core::ws_protocol::ServerMessage::WorkflowStarted {
                                                    run_id,
                                                    workflow_name,
                                                    nodes: Vec::new(),
                                                }
                                            }
                                            WorkflowEvent::NodeUpdate { run_id, node_id, status, output } => {
                                                sa_core::ws_protocol::ServerMessage::WorkflowNodeUpdate {
                                                    run_id,
                                                    node_id,
                                                    status,
                                                    output,
                                                }
                                            }
                                            WorkflowEvent::ApprovalRequested { run_id, node_id, prompt } => {
                                                {
                                                    let mut map = approval_txs_forward.lock().await;
                                                    map.insert((run_id, node_id.clone()), approval_tx.clone());
                                                }
                                                sa_core::ws_protocol::ServerMessage::ApprovalRequested {
                                                    run_id,
                                                    node_id,
                                                    prompt,
                                                }
                                            }
                                            WorkflowEvent::Completed { run_id, summary } => {
                                                {
                                                    let mut map = approval_txs_forward.lock().await;
                                                    map.retain(|(rid, _), _| *rid != run_id);
                                                }
                                                sa_core::ws_protocol::ServerMessage::WorkflowCompleted {
                                                    run_id,
                                                    summary,
                                                }
                                            }
                                            WorkflowEvent::Failed { run_id, node_id, error } => {
                                                {
                                                    let mut map = approval_txs_forward.lock().await;
                                                    map.retain(|(rid, _), _| *rid != run_id);
                                                }
                                                sa_core::ws_protocol::ServerMessage::WorkflowFailed {
                                                    run_id,
                                                    node_id,
                                                    error,
                                                }
                                            }
                                        };
                                        let _ = forward_tx.send(msg);
                                    }
                                });

                                // 4. Build workflow LLM config
                                let wf_llm_config = sa_core::workflow_engine::WorkflowLlmConfig {
                                    base_url: llm_config.base_url.clone(),
                                    api_key: llm_config.api_key.clone(),
                                    default_model: llm_config.model.clone(),
                                };

                                // 5. Create tool resources for workflow nodes
                                let wf_tool_ctx = match sa_core::tools::ToolContext::new(ws_root.clone(), std::sync::Arc::new(sa_core::skills::SkillRegistry::default())) {
                                    Ok(ctx) => ctx,
                                    Err(e) => {
                                        tracing::warn!("Failed to create tool context for workflow: {e}");
                                        sa_core::tools::ToolContext::new(
                                            std::path::PathBuf::from("."),
                                            std::sync::Arc::new(sa_core::skills::SkillRegistry::default()),
                                        ).unwrap()
                                    }
                                };
                                let wf_tools = sa_core::tools::ToolExecutor::new(wf_tool_ctx, None);
                                let wf_runtime = sa_core::tools::ToolRuntime::detached();
                                let wf_cancel = sa_core::cancel::cancel_pair().1;

                                // 6. Run the workflow
                                let mut inputs = std::collections::HashMap::new();
                                inputs.insert("user_request".to_string(), task);
                                let _run = sa_core::workflow_engine::run_workflow(
                                    def,
                                    inputs,
                                    ws_root,
                                    progress_tx,
                                    Some(wf_llm_config),
                                    approval_rx,
                                    Some((wf_tools, wf_runtime, wf_cancel)),
                                ).await;
                                let _ = forwarder.await;
                            });
                        } else {
                            // Normal Chat mode: submit to agent runner.
                            match state.runtime_snapshot().await {
                                RuntimeState::Ready(hub) => {
                                    let task_id = task_id.unwrap_or_else(Uuid::new_v4);

                                    match hub.submit_task(task_id, task).await {
                                        Ok(work_id) => {
                                            send_direct_server_message(
                                                &out_tx,
                                                ServerMessage::Accepted { task_id: work_id },
                                            );
                                        }
                                        Err(err) => {
                                            send_direct_server_message(
                                                &out_tx,
                                                ServerMessage::Error {
                                                    message: format!("Failed to submit task: {err}"),
                                                },
                                            );
                                        }
                                    }
                                }
                                RuntimeState::Bootstrap(bootstrap) => {
                                    send_direct_server_message(
                                        &out_tx,
                                        bootstrap.init_required_message(),
                                    );
                                }
                            }
                        }
                    }
                    ClientMessage::GetHistory { from_event_id } => {
                        match state.runtime_snapshot().await {
                            RuntimeState::Ready(hub) => {
                                let events = hub.history_since(from_event_id).await;
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::History { events },
                                );
                            }
                            RuntimeState::Bootstrap(_) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::History { events: Vec::new() },
                                );
                            }
                        }
                    }
                    ClientMessage::Interrupt { task_id } => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            if let Err(err) = hub.request_interrupt(task_id).await {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::Error {
                                        message: format!("Failed to interrupt task: {err}"),
                                    },
                                );
                            }
                        }
                    }
                    ClientMessage::AnswerQuestion { answer } => {
                        match state.runtime_snapshot().await {
                            RuntimeState::Ready(hub) => {
                                if let Err(err) = hub.answer_question(answer).await {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: format!("Failed to answer question: {err}"),
                                        },
                                    );
                                }
                            }
                            RuntimeState::Bootstrap(bootstrap) => {
                                send_direct_server_message(
                                    &out_tx,
                                    bootstrap.init_required_message(),
                                );
                            }
                        }
                    }
                    ClientMessage::ImageUpload {
                        upload_id,
                        filename,
                        mime_type,
                        data,
                        task_id: _,
                    } => {
                        // Decode base64 payload and persist to workspace uploads/.
                        use base64::Engine as _;
                        match base64::engine::general_purpose::STANDARD.decode(&data) {
                            Ok(bytes) => {
                                // Derive a safe extension from the original filename or
                                // fall back to the mime type.
                                let ext = std::path::Path::new(&filename)
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .unwrap_or_else(|| {
                                        if mime_type.contains("jpeg") || mime_type.contains("jpg") {
                                            "jpg"
                                        } else if mime_type.contains("png") {
                                            "png"
                                        } else if mime_type.contains("webp") {
                                            "webp"
                                        } else if mime_type.contains("gif") {
                                            "gif"
                                        } else {
                                            "bin"
                                        }
                                    });

                                let uploads_dir = {
                                    let runtime = state.runtime.lock().await;
                                    match &*runtime {
                                        RuntimeState::Ready(hub) => hub.workspace_root.join("uploads"),
                                        RuntimeState::Bootstrap(_) => {
                                            send_direct_server_message(
                                                &out_tx,
                                                ServerMessage::Error {
                                                    message: "image_upload: SA is not yet initialized (no workspace)".to_string(),
                                                },
                                            );
                                            return;
                                        }
                                    }
                                };
                                let save_result = (|| -> std::io::Result<std::path::PathBuf> {
                                    std::fs::create_dir_all(&uploads_dir)?;
                                    let fname = format!("{}.{}", upload_id, ext);
                                    let dest = uploads_dir.join(&fname);
                                    std::fs::write(&dest, &bytes)?;
                                    Ok(dest)
                                })();

                                match save_result {
                                    Ok(saved_path) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ImageUploaded {
                                                upload_id,
                                                saved_path: saved_path.to_string_lossy().to_string(),
                                            },
                                        );

                                        // Inject the uploaded image into the current agent's
                                        // conversation so the LLM can analyse it.
                                        let data_url = format!(
                                            "data:{mime_type};base64,{data}"
                                        );
                                        // Grab the Hub from the ready runtime state.
                                        let hub = {
                                            let runtime = state.runtime.lock().await;
                                            match &*runtime {
                                                RuntimeState::Ready(hub) => Arc::clone(hub),
                                                RuntimeState::Bootstrap(_) => {
                                                    // No agent to inject into.
                                                    continue;
                                                }
                                            }
                                        };
                                        if let Err(inject_err) = inject_user_image(
                                            &hub,
                                            &data_url,
                                            &saved_path.to_string_lossy(),
                                        )
                                        .await
                                        {
                                            // Non-fatal: log but don't fail the upload.
                                            eprintln!(
                                                "[image_upload] failed to inject image into agent conversation: {inject_err}"
                                            );
                                        }
                                    }
                                    Err(io_err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!(
                                                    "Failed to save uploaded image: {io_err}"
                                                ),
                                            },
                                        );
                                    }
                                }
                            }
                            Err(decode_err) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::Error {
                                        message: format!(
                                            "image_upload: invalid base64 payload: {decode_err}"
                                        ),
                                    },
                                );
                            }
                        }
                    }
                    ClientMessage::GetConfig => {
                        match state.runtime_snapshot().await {
                            RuntimeState::Ready(hub) => {
                                let cfg = match load_config_from_file(&hub.config_path) {
                                    Ok(c) => c,
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdateFailed {
                                                message: "读取配置文件失败。".to_string(),
                                                detail: Some(format!("{err:#}")),
                                            },
                                        );
                                        continue;
                                    }
                                };
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::ConfigSnapshot {
                                        config: cfg.to_toml_value(),
                                        config_path: hub.config_path.display().to_string(),
                                    },
                                );
                            }
                            RuntimeState::Bootstrap(_) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::ConfigUpdateFailed {
                                        message: "SA 尚未初始化，无法读取配置。".to_string(),
                                        detail: None,
                                    },
                                );
                            }
                        }
                    }
                    ClientMessage::UpdateConfig { config } => {
                        match state.runtime_snapshot().await {
                            RuntimeState::Ready(hub) => {
                                // Load the current (real) config to merge secrets.
                                let current_cfg = match load_config_from_file(&hub.config_path) {
                                    Ok(c) => c,
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdateFailed {
                                                message: "读取当前配置失败。".to_string(),
                                                detail: Some(format!("{err:#}")),
                                            },
                                        );
                                        continue;
                                    }
                                };

                                // Merge frontend edits into a full Config.
                                let merged = match Config::merge_from_frontend_value(&config, &current_cfg) {
                                    Ok(c) => c,
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdateFailed {
                                                message: "配置校验失败。".to_string(),
                                                detail: Some(format!("{err:#}")),
                                            },
                                        );
                                        continue;
                                    }
                                };

                                // Serialize to TOML and write to disk.
                                let toml_str = match toml::to_string_pretty(&merged) {
                                    Ok(s) => s,
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdateFailed {
                                                message: "序列化配置失败。".to_string(),
                                                detail: Some(format!("{err:#}")),
                                            },
                                        );
                                        continue;
                                    }
                                };

                                if let Err(err) = std::fs::write(&hub.config_path, &toml_str) {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::ConfigUpdateFailed {
                                            message: "写入配置文件失败。".to_string(),
                                            detail: Some(format!("{err:#}")),
                                        },
                                    );
                                    continue;
                                }

                                // Hot-reload the runtime from disk.
                                match hub.reload_runtime_from_disk().await {
                                    Ok(summary) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdated { summary },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdateFailed {
                                                message: "配置已写入但热重载失败，可能需要重启 SA。".to_string(),
                                                detail: Some(format!("{err:#}")),
                                            },
                                        );
                                    }
                                }
                            }
                            RuntimeState::Bootstrap(_) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::ConfigUpdateFailed {
                                        message: "SA 尚未初始化，无法更新配置。".to_string(),
                                        detail: None,
                                    },
                                );
                            }
                        }
                    }


                    ClientMessage::GetSkillDoc { name } => {
                        match state.runtime_snapshot().await {
                            RuntimeState::Ready(hub) => {
                                // 克隆需要的数据，然后释放锁
                                let skills_clone = {
                                    let preload_ctx = hub.preload_ctx.read().unwrap();
                                    preload_ctx.skills.clone()
                                };
                                
                                match skills_clone.load_skill_file(&name, None).await {
                                    Ok((path, content)) => {
                                        // 获取技能描述
                                        let description = skills_clone.get(&name)
                                            .map(|skill| skill.description.clone())
                                            .unwrap_or_default();
                                        
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::SkillDoc {
                                                name: name.clone(),
                                                description,
                                                content,
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ConfigUpdateFailed {
                                                message: format!("读取技能文档失败: {name}"),
                                                detail: Some(format!("{err:#}")),
                                            },
                                        );
                                    }
                                }
                            }
                            RuntimeState::Bootstrap(_) => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::ConfigUpdateFailed {
                                        message: "SA 尚未初始化，无法读取技能文档。".to_string(),
                                        detail: None,
                                    },
                                );
                            }
                        }
                    }





                    // ── Memory panel handlers ────────────────────────────

                    ClientMessage::GetMemoryStats => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let memory_dir = hub.workspace_root.join("memory");
                            let out_tx = out_tx.clone();
                            tokio::spawn(async move {
                                let result = compute_memory_stats(&memory_dir).await;
                                match result {
                                    Ok((total_facts, unique_subjects, unique_predicates, avg_confidence, oldest_fact, newest_fact)) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::MemoryStats {
                                                total_facts,
                                                unique_subjects,
                                                unique_predicates,
                                                avg_confidence,
                                                oldest_fact,
                                                newest_fact,
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to get memory stats: {err}"),
                                            },
                                        );
                                    }
                                }
                            });
                        }
                    }

                    ClientMessage::GetMemoryFacts { query, limit, offset } => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let memory_dir = hub.workspace_root.join("memory");
                            let out_tx = out_tx.clone();
                            tokio::spawn(async move {
                                let result = query_memory_facts(&memory_dir, query.as_deref(), limit, offset).await;
                                match result {
                                    Ok((facts, total)) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::MemoryFacts { facts, total, offset },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to query memory facts: {err}"),
                                            },
                                        );
                                    }
                                }
                            });
                        }
                    }

                    ClientMessage::TraverseMemoryGraph { subject, max_hops, max_results } => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let memory_dir = hub.workspace_root.join("memory");
                            let out_tx = out_tx.clone();
                            tokio::spawn(async move {
                                let result = traverse_memory(&memory_dir, &subject, max_hops, max_results).await;
                                match result {
                                    Ok(facts) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::MemoryGraph {
                                                subject,
                                                facts,
                                                hops: max_hops,
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to traverse memory graph: {err}"),
                                            },
                                        );
                                    }
                                }
                            });
                        }
                    }

                    ClientMessage::DecayMemoryFacts => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let memory_dir = hub.workspace_root.join("memory");
                            let out_tx = out_tx.clone();
                            tokio::spawn(async move {
                                let result = decay_memory(&memory_dir).await;
                                match result {
                                    Ok(affected) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::MemoryOperationResult {
                                                operation: "decay".to_string(),
                                                affected,
                                                summary: format!("已对 {affected} 条记忆执行衰减"),
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to decay memory: {err}"),
                                            },
                                        );
                                    }
                                }
                            });
                        }
                    }

                    ClientMessage::PruneMemoryFacts { min_confidence } => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let memory_dir = hub.workspace_root.join("memory");
                            let out_tx = out_tx.clone();
                            tokio::spawn(async move {
                                let result = prune_memory(&memory_dir, min_confidence).await;
                                match result {
                                    Ok(affected) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::MemoryOperationResult {
                                                operation: "prune".to_string(),
                                                affected,
                                                summary: format!("已剪除 {affected} 条低置信度记忆"),
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to prune memory: {err}"),
                                            },
                                        );
                                    }
                                }
                            });
                        }
                    }
                    ClientMessage::GetMemoryReport { report_path } => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let memory_dir = hub.workspace_root.join("memory");
                            let out_tx = out_tx.clone();
                            tokio::spawn(async move {
                                // Sanitize: normalize separators and reject traversal
                                let normalized = report_path.replace('\\', "/");
                                if normalized.contains("..") {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::Error {
                                            message: "Invalid report path: directory traversal not allowed".to_string(),
                                        },
                                    );
                                    return;
                                }
                                let full_path = memory_dir.join(&normalized);
                                match tokio::fs::read_to_string(&full_path).await {
                                    Ok(content) => {
                                        tracing::info!("[GetMemoryReport] sending report '{}' ({} bytes)", normalized, content.len());
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::MemoryReport {
                                                report_path: normalized,
                                                content,
                                            },
                                        );
                                    }
                                    Err(err) => {
                                        if err.kind() == std::io::ErrorKind::NotFound {
                                            let msg = format!("📭 报告尚未生成: {}\n\n该报告将在下一次 Dream 运行后自动生成。", normalized);
                                            send_direct_server_message(
                                                &out_tx,
                                                ServerMessage::MemoryReport {
                                                    report_path: normalized,
                                                    content: msg,
                                                },
                                            );
                                        } else {
                                            send_direct_server_message(
                                                &out_tx,
                                                ServerMessage::Error {
                                                    message: format!("Failed to read report '{normalized}': {err}"),
                                                },
                                            );
                                        }
                                    }
                                }
                            });
                        }
                    }


                    // ── Workflow engine handlers ─────────────────────────────

                    ClientMessage::StartWorkflow { workflow_name, inputs } => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let workflows_dir = hub.workspace_root.parent()
                                .unwrap_or(&hub.workspace_root)
                                .join("sa")
                                .join("workflows");
                            let workflow_path = workflows_dir.join(format!("{workflow_name}.yaml"));
                            let out_tx = out_tx.clone();

                            // Build LLM config from sa.toml for workflow engine
                            let llm_config = match load_config_from_file(&hub.config_path) {
                                Ok(cfg) => Some(sa_core::workflow_engine::WorkflowLlmConfig {
                                    base_url: cfg.llm.base_url.clone(),
                                    api_key: cfg.llm.api_key.clone(),
                                    default_model: cfg.llm.model.clone(),
                                }),
                                Err(_) => None,
                            };

                            let approval_txs_clone = approval_txs.clone();

                            tokio::spawn(async move {
                                match sa_core::workflow_engine::load_workflow_def(&workflow_path).await {
                                    Ok(def) => {
                                        let ws_root = hub.workspace_root.clone();
                                        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

                                        // Create approval channel
                                        let (approval_tx, approval_rx) = tokio::sync::mpsc::unbounded_channel::<ApprovalResponse>();

                                        // Spawn a task to forward progress events to the WS client
                                        let forward_tx = out_tx.clone();
                                        let approval_txs_forward = approval_txs_clone.clone();
                                        let forwarder = tokio::spawn(async move {
                                            while let Some(event) = rx.recv().await {
                                                use sa_core::workflow_engine::WorkflowEvent;
                                                let msg = match event {
                                                    WorkflowEvent::Started { run_id, workflow_name } => {
                                                        sa_core::ws_protocol::ServerMessage::WorkflowStarted {
                                                            run_id,
                                                            workflow_name,
                                                            nodes: Vec::new(),
                                                        }
                                                    }
                                                    WorkflowEvent::NodeUpdate { run_id, node_id, status, output } => {
                                                        sa_core::ws_protocol::ServerMessage::WorkflowNodeUpdate {
                                                            run_id,
                                                            node_id,
                                                            status,
                                                            output,
                                                        }
                                                    }
                                                    WorkflowEvent::ApprovalRequested { run_id, node_id, prompt } => {
                                                        // Register the approval sender so RespondApproval can find it.
                                                        {
                                                            let mut map = approval_txs_forward.lock().await;
                                                            map.insert((run_id, node_id.clone()), approval_tx.clone());
                                                        }
                                                        sa_core::ws_protocol::ServerMessage::ApprovalRequested {
                                                            run_id,
                                                            node_id,
                                                            prompt,
                                                        }
                                                    }
                                                    WorkflowEvent::Completed { run_id, summary } => {
                                                        // Clean up any remaining approval senders for this run.
                                                        {
                                                            let mut map = approval_txs_forward.lock().await;
                                                            map.retain(|(rid, _), _| *rid != run_id);
                                                        }
                                                        sa_core::ws_protocol::ServerMessage::WorkflowCompleted {
                                                            run_id,
                                                            summary,
                                                        }
                                                    }
                                                    WorkflowEvent::Failed { run_id, node_id, error } => {
                                                        // Clean up any remaining approval senders for this run.
                                                        {
                                                            let mut map = approval_txs_forward.lock().await;
                                                            map.retain(|(rid, _), _| *rid != run_id);
                                                        }
                                                        sa_core::ws_protocol::ServerMessage::WorkflowFailed {
                                                            run_id,
                                                            node_id,
                                                            error,
                                                        }
                                                    }
                                                };
                                                let _ = forward_tx.send(msg);
                                            }
                                        });

                                        let _run = sa_core::workflow_engine::run_workflow(
                                            def,
                                            inputs,
                                            ws_root,
                                            tx,
                                            llm_config,
                                            approval_rx,
                                            None, // YAML workflows don't need LLM tool-calling
                                        ).await;
                                        let _ = forwarder.await;
                                    }
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to load workflow '{workflow_name}': {err}"),
                                            },
                                        );
                                    }
                                }
                            });
                        } else {
                            send_direct_server_message(
                                &out_tx,
                                ServerMessage::Error {
                                    message: "Runtime not ready".to_string(),
                                },
                            );
                        }
                    }

                    ClientMessage::CancelWorkflow { run_id: _ } => {
                        // Cancel is handled by tracking active workflow handles.
                        // For now, acknowledge the request.
                        send_direct_server_message(
                            &out_tx,
                            ServerMessage::Error {
                                message: "Workflow cancellation not yet implemented".to_string(),
                            },
                        );
                    }

                    ClientMessage::InjectWorkflowContext { run_id, node_id, context } => {
                        // Forward the injected context to the workflow via approval channel.
                        let mut map = approval_txs.lock().await;
                        if let Some(sender) = map.remove(&(run_id, node_id.clone())) {
                            let _ = sender.send(ApprovalResponse {
                                run_id,
                                node_id: node_id.clone(),
                                approved: true,
                                feedback: Some(format!("[INJECTED CONTEXT] {context}")),
                            });
                            tracing::info!("Context injected for run={run_id} node={node_id}");
                        } else {
                            send_direct_server_message(
                                &out_tx,
                                ServerMessage::Error {
                                    message: format!("No pending node for injection: run={run_id} node={node_id}"),
                                },
                            );
                        }
                    }

                    ClientMessage::PauseWorkflow { run_id } => {
                        // Workflow pause — mark all pending approvals in this run as rejected.
                        tracing::info!("Pausing workflow run={run_id}");
                        send_direct_server_message(
                            &out_tx,
                            ServerMessage::WorkflowNodeUpdate {
                                run_id,
                                node_id: String::new(),
                                status: NodeStatus::Skipped,
                                output: Some("Workflow paused by user".to_string()),
                            },
                        );
                    }

                    ClientMessage::ResumeWorkflow { run_id } => {
                        tracing::info!("Resuming workflow run={run_id}");
                        send_direct_server_message(
                            &out_tx,
                            ServerMessage::Error {
                                message: "Workflow resume not yet implemented".to_string(),
                            },
                        );
                    }

                    ClientMessage::SetChatMode { mode } => {
                        chat_mode = mode;
                        send_direct_server_message(
                            &out_tx,
                            ServerMessage::ChatModeChanged { mode },
                        );
                    }

                    ClientMessage::ApprovePlan { plan_id, action } => {
                        match action {
                            PlanAction::ApproveAll | PlanAction::ApproveStep => {
                                let out_tx_plan = out_tx.clone();
                                let config_path = match state.runtime_snapshot().await {
                                    RuntimeState::Ready(ref hub) => hub.config_path.clone(),
                                    _ => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: "Runtime not ready".to_string(),
                                            },
                                        );
                                        continue;
                                    }
                                };
                                let llm_config = match load_config_from_file(&config_path) {
                                    Ok(cfg) => sa_core::plan_engine::PlanLlmConfig {
                                        base_url: cfg.llm.base_url.clone(),
                                        api_key: cfg.llm.api_key.clone(),
                                        model: cfg.llm.model.clone(),
                                    },
                                    Err(err) => {
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::Error {
                                                message: format!("Failed to load config: {err}"),
                                            },
                                        );
                                        continue;
                                    }
                                };

                                tokio::spawn(async move {
                                    let workspace_root = std::path::PathBuf::from(".");
                                    let plan = match sa_core::plan_engine::load_plan(&workspace_root, &plan_id).await {
                                        Ok(p) => p,
                                        Err(err) => {
                                            let _ = out_tx_plan.send(ServerMessage::Error {
                                                message: format!("Failed to load plan: {err}"),
                                            });
                                            return;
                                        }
                                    };

                                    let mut accumulated = String::new();
                                    let total_steps = plan.steps.len();
                                    for (i, step) in plan.steps.iter().enumerate() {
                                        // Send running status
                                        let _ = out_tx_plan.send(ServerMessage::PlanStepUpdate {
                                            plan_id,
                                            step_id: step.step_id.clone(),
                                            status: PlanStepStatus::Running,
                                        });

                                        let step_action = action;
                                        if step_action == PlanAction::ApproveStep && i > 0 {
                                            // Step mode: only execute one step
                                            // Mark the rest as still pending
                                            break;
                                        }

                                        match sa_core::plan_engine::execute_plan_step(
                                            &plan, i, &accumulated, &llm_config,
                                        ).await {
                                            Ok(output) => {
                                                let step_output = format!(
                                                    "--- Step {}: {} ---\n{}\n",
                                                    step.step_id, step.description, output
                                                );
                                                accumulated.push_str(&step_output);
                                                let _ = out_tx_plan.send(ServerMessage::PlanStepUpdate {
                                                    plan_id,
                                                    step_id: step.step_id.clone(),
                                                    status: PlanStepStatus::Completed,
                                                });
                                            }
                                            Err(err) => {
                                                let _ = out_tx_plan.send(ServerMessage::PlanStepUpdate {
                                                    plan_id,
                                                    step_id: step.step_id.clone(),
                                                    status: PlanStepStatus::Failed,
                                                });
                                                let _ = out_tx_plan.send(ServerMessage::Error {
                                                    message: format!("Step {} failed: {err}", step.step_id),
                                                });
                                                return;
                                            }
                                        }
                                    }
                                });
                            }
                            PlanAction::Modify => {
                                send_direct_server_message(
                                    &out_tx,
                                    ServerMessage::Error {
                                        message: "Plan modification not yet implemented".to_string(),
                                    },
                                );
                            }
                        }
                    }

                    ClientMessage::RespondApproval { run_id, node_id, approved, feedback } => {
                        let mut map = approval_txs.lock().await;
                        if let Some(sender) = map.remove(&(run_id, node_id.clone())) {
                            let _ = sender.send(ApprovalResponse {
                                run_id,
                                node_id: node_id.clone(),
                                approved,
                                feedback,
                            });
                            tracing::info!("Approval response sent for run={run_id} node={node_id} approved={approved}");
                        } else {
                            tracing::warn!("No pending approval found for run={run_id} node={node_id}");
                            send_direct_server_message(
                                &out_tx,
                                ServerMessage::Error {
                                    message: format!("No pending approval for run {run_id} node {node_id}"),
                                },
                            );
                        }
                    }

                    ClientMessage::GetSkillsList => {
                        if let RuntimeState::Ready(hub) = state.runtime_snapshot().await {
                            let preload_ctx = hub.preload_ctx.read().unwrap();
                            let skills: Vec<sa_core::ws_protocol::SkillListItem> = preload_ctx.skills.list_full()
                                .into_iter()
                                .map(|(_name, spec)| {
                                    let source = match spec.source {
                                        sa_core::commands::CommandSource::LocalSkill => sa_core::ws_protocol::SkillSource::Local,
                                        sa_core::commands::CommandSource::Bundled => sa_core::ws_protocol::SkillSource::Builtin,
                                        sa_core::commands::CommandSource::McpPrompt => sa_core::ws_protocol::SkillSource::Mcp,
                                    };
                                    // Mark core skills (毛选, 求是)
                                    let core = spec.name.contains("毛选") || spec.name.contains("求是");
                                    sa_core::ws_protocol::SkillListItem {
                                        name: spec.name.clone(),
                                        description: spec.description.clone(),
                                        source,
                                        triggers: spec.triggers.clone(),
                                        usage_count: 0,
                                        core,
                                    }
                                })
                                .collect();
                            send_direct_server_message(
                                &out_tx,
                                ServerMessage::SkillsList { skills },
                            );
                        }
                    }
                    ClientMessage::TestApiKey { base_url, api_key, model } => {
                        let out_tx = out_tx.clone();
                        tokio::spawn(async move {
                            let start = std::time::Instant::now();
                            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
                            let body = serde_json::json!({
                                "model": model,
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 1,
                                "stream": false
                            });
                            let client = reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(30))
                                .build()
                                .unwrap_or_default();
                            let result = client
                                .post(&url)
                                .header("Authorization", format!("Bearer {}", api_key))
                                .header("Content-Type", "application/json")
                                .json(&body)
                                .send()
                                .await;
                            let latency_ms = start.elapsed().as_millis() as u64;
                            match result {
                                Ok(resp) => {
                                    let status = resp.status();
                                    if status.is_success() {
                                        let provider = base_url.split("://").nth(1)
                                            .and_then(|s| s.split('/').next())
                                            .unwrap_or("unknown")
                                            .to_string();
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ApiKeyTestResult {
                                                ok: true,
                                                model: model.clone(),
                                                provider,
                                                latency_ms,
                                                error: None,
                                            },
                                        );
                                    } else {
                                        let err_text = resp.text().await.unwrap_or_else(|_| format!("HTTP {status}"));
                                        let provider = base_url.split("://").nth(1)
                                            .and_then(|s| s.split('/').next())
                                            .unwrap_or("unknown")
                                            .to_string();
                                        send_direct_server_message(
                                            &out_tx,
                                            ServerMessage::ApiKeyTestResult {
                                                ok: false,
                                                model: model.clone(),
                                                provider,
                                                latency_ms,
                                                error: Some(format!("HTTP {}: {}", status, err_text.chars().take(200).collect::<String>())),
                                            },
                                        );
                                    }
                                }
                                Err(err) => {
                                    send_direct_server_message(
                                        &out_tx,
                                        ServerMessage::ApiKeyTestResult {
                                            ok: false,
                                            model: model.clone(),
                                            provider: "unknown".to_string(),
                                            latency_ms,
                                            error: Some(format!("连接失败: {}", err)),
                                        },
                                    );
                                }
                            }
                        });
                    }
                }
            }
            Message::Close(_) => break,
            // Ignore binary/ping/pong for this minimal protocol.
            _ => {}
        }
    }


    // Drop outbound channel to stop writer task.
    drop(out_tx);

    // Best-effort: stop background tasks for this session.
    if let Some(forwarder) = forwarder {
        forwarder.abort();
    }
    writer.abort();
}

/// Ctrl+C handler for graceful shutdown.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Ctrl+C received; shutting down.");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure logging from `RUST_LOG`, defaulting to `info`.
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Parse CLI args.
    let args = Args::parse();

    let ws_identity = load_local_identity()?;
    tracing::info!("WS identity machine hint: {}", ws_identity.machine_hint);
    let (bind, ws_path, runtime_state) = if args.config.exists() {
        let cfg = load_config_from_file(&args.config)?;
        let runtime = build_runtime_from_config(cfg, &args.config).await?;
        (
            runtime.bind,
            runtime.ws_path,
            RuntimeState::Ready(runtime.hub),
        )
    } else {
        let bootstrap = build_bootstrap_state(&args.config)?;
        tracing::warn!(
            "Config file {} does not exist; starting in bootstrap initialization mode",
            bootstrap.config_path.display()
        );
        tracing::info!(
            "Bootstrap workspace root: {}",
            bootstrap.workspace_root.display()
        );
        tracing::info!(
            "Bootstrap Agents.md path: {}",
            bootstrap.agents_md_path.display()
        );
        (
            bootstrap.bind.clone(),
            bootstrap.ws_path.clone(),
            RuntimeState::Bootstrap(bootstrap),
        )
    };
    let server_state = Arc::new(ServerState {
        ws_identity,
        runtime: AsyncMutex::new(runtime_state),
    });

    // Build HTTP router (WS only).
    let app = Router::new()
        .route(&ws_path, get(ws_route))
        .with_state(server_state);

    // Bind listener.
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("Failed to bind {bind}"))?;

    tracing::info!("WebSocket listening on {}", bind);

    // Serve with graceful shutdown.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server crashed")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex::encode as hex_encode;
    use sa_core::interaction_history::{InteractionEntry, InteractionStore};
    use sa_core::openai::{ToolCall, ToolFunctionCall};
    use sa_core::ws_identity::{
        EXPECTED_CLIENT_NAME, WS_ALLOWED_SKEW_BUCKETS, WS_HASH_ALGO, WS_PROTOCOL_ID,
        WS_TIME_STEP_SECS, build_client_hello, current_time_bucket,
    };
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode as TungsteniteCloseCode;

    /// Test-only copy of the client proof label so the integration tests can build
    /// boundary-case handshakes against the live WS server.
    const TEST_CLIENT_PROOF_LABEL: &str = "sa-frontend-proof/v1";

    /// One running ephemeral WS server used by handshake integration tests.
    struct TestServer {
        addr: std::net::SocketAddr,
        _workspace: TempDir,
        task: tokio::task::JoinHandle<()>,
    }

    /// Load all durable interaction entries from the temp workspace.
    fn load_interaction_entries(workspace: &TempDir) -> Vec<InteractionEntry> {
        let store = InteractionStore::new(workspace.path().to_path_buf())
            .expect("interaction store should build for test workspace");
        let log_path = store.log_path().to_path_buf();
        if !log_path.is_file() {
            return Vec::new();
        }
        std::fs::read_to_string(&log_path)
            .expect("interaction log should read")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str::<InteractionEntry>(line).expect("valid interaction line")
            })
            .collect()
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    /// Minimal `Hub` factory for WS handshake tests.
    fn build_test_hub(workspace: &TempDir) -> Arc<Hub> {
        build_test_hub_with_skill_dirs(workspace, &[])
    }

    /// Variant of the test hub that scans explicit skill directories.
    fn build_test_hub_with_skill_dirs(workspace: &TempDir, skill_dirs: &[PathBuf]) -> Arc<Hub> {
        let skills = Arc::new(
            SkillRegistry::scan(skill_dirs).expect("skill registry should build for test hub"),
        );
        let tool_ctx = ToolContext::new(workspace.path().to_path_buf(), Arc::clone(&skills))
            .expect("tool context should build for temp workspace");
        let preload_ctx = tool_ctx.clone();
        let session_store = Arc::new(
            SessionStore::new(preload_ctx.workspace_root.clone())
                .expect("session store should build for temp workspace"),
        );
        let interaction_store = Arc::new(
            InteractionStore::new(preload_ctx.workspace_root.clone())
                .expect("interaction store should build for temp workspace"),
        );
        let task_audit_store = Arc::new(
            TaskAuditStore::new(preload_ctx.workspace_root.clone())
                .expect("task audit store should build for temp workspace"),
        );
        let tools = ToolExecutor::new(tool_ctx, None);
        let llm = OpenAiClient::new("http://127.0.0.1:1".to_string(), "test-key".to_string())
            .expect("test OpenAI client should build");
        let runner = AgentRunner::new(
            llm,
            tools,
            skills,
            AgentRunnerConfig {
                model: "test-model".to_string(),
                system_role_name: "developer".to_string(),
                reasoning_effort: None,
                compaction: sa_core::compact::CompactionConfig::default(),
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                fallback_model: None,
                max_consecutive_failures: 2,
                max_retries: 12,
                model_routing: None,
            },
        );
        let runtime_store =
            RuntimeStore::new(workspace.path().to_path_buf()).expect("runtime store should build");
        let root_agent_id = Uuid::new_v4();
        runtime_store
            .save_root_marker(&RootMarker { root_agent_id })
            .expect("root marker should persist");
        let team_state = TeamState::new(root_agent_id);
        runtime_store
            .save_team_state(&team_state)
            .expect("team state should persist");
        let root_session_store = SessionStore::new_in_relative_dir(
            workspace.path().to_path_buf(),
            Path::new(&format!("sessions/agents/{root_agent_id}")),
        )
        .expect("root agent session store should build");
        runtime_store
            .save_agent_state(&AgentState::new_root(
                root_agent_id,
                root_session_store.current_session_path(),
            ))
            .expect("root agent state should persist");

        Hub::new(
            workspace.path().join("sa.toml"),
            "127.0.0.1:8765".to_string(),
            "/ws".to_string(),
            preload_ctx.workspace_root.clone(),
            RuntimeReloadState {
                runner,
                preload_ctx,
                agents_md_path: workspace.path().join("AGENTS.md"),
                team_cfg: TeamConfig::default(),
                permissions: PermissionsConfig::default(),
                max_concurrent_model_calls: TeamConfig::default().max_concurrent_model_calls,
            },
            session_store,
            interaction_store,
            task_audit_store,
            runtime_store,
            team_state,
        )
    }

    /// Build either a bootstrap or ready server state for WS tests.
    fn build_test_server_state(workspace: &TempDir, bootstrap: bool) -> Arc<ServerState> {
        let runtime = if bootstrap {
            RuntimeState::Bootstrap(
                build_bootstrap_state(&workspace.path().join("sa.toml"))
                    .expect("bootstrap state should build for temp workspace"),
            )
        } else {
            RuntimeState::Ready(build_test_hub(workspace))
        };

        Arc::new(ServerState {
            ws_identity: load_local_identity().expect("local WS identity should load"),
            runtime: AsyncMutex::new(runtime),
        })
    }

    /// Spawn one live WS server bound to an ephemeral local port.
    async fn spawn_test_server(
        handshake_timeout: std::time::Duration,
        bootstrap: bool,
    ) -> TestServer {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let state = build_test_server_state(&workspace, bootstrap);
        let app = Router::new()
            .route(
                "/ws",
                get(
                    move |ws: WebSocketUpgrade, State(state): State<Arc<ServerState>>| async move {
                        ws.on_upgrade(move |socket| {
                            ws_session_with_timeout(socket, state, handshake_timeout)
                        })
                    },
                ),
            )
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener should bind");
        let addr = listener
            .local_addr()
            .expect("test listener should expose local addr");
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test WS server should keep serving until aborted");
        });

        TestServer {
            addr,
            _workspace: workspace,
            task,
        }
    }

    /// Return whether the current machine exposes a usable Git-Bash style shell
    /// for background-task integration tests. Some CI environments may not have
    /// Bash available, so those tests degrade to a no-op instead of producing a
    /// misleading failure unrelated to the streaming logic under test.
    async fn has_background_bash_for_tests() -> bool {
        for program in candidate_bash_programs() {
            match tokio::process::Command::new(&program)
                .arg("-lc")
                .arg("printf ok")
                .output()
                .await
            {
                Ok(output) if output.status.success() && output.stdout == b"ok" => return true,
                Ok(_) => continue,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => continue,
            }
        }

        false
    }

    #[tokio::test]
    async fn background_bash_writes_output_incrementally_before_process_exit() {
        if !has_background_bash_for_tests().await {
            return;
        }

        let workspace = TempDir::new().expect("temp workspace should build");
        let output_path = workspace.path().join("background-task.log");
        let request = StartTerminalTaskRequest {
            command: "printf start; sleep 1; printf end".to_string(),
            workdir: workspace.path().display().to_string(),
            timeout_seconds: Some(10),
            safety_warning: None,
        };

        let request_for_task = request.clone();
        let output_for_task = output_path.clone();
        let task = tokio::spawn(async move {
            run_background_bash_command(&request_for_task, &output_for_task).await
        });

        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let partial = String::from_utf8_lossy(
            &tokio::fs::read(&output_path)
                .await
                .expect("partial log should exist before command exit"),
        )
        .into_owned();
        assert!(
            partial.contains("start"),
            "expected partial log to contain the early stdout chunk, got: {partial:?}"
        );
        assert!(
            !partial.contains("end"),
            "partial log should not already contain the trailing chunk: {partial:?}"
        );

        let exit_code = task
            .await
            .expect("background command join should succeed")
            .expect("background command should exit successfully");
        assert_eq!(exit_code, 0);

        let final_log = String::from_utf8_lossy(
            &tokio::fs::read(&output_path)
                .await
                .expect("final log should read"),
        )
        .into_owned();
        assert!(final_log.contains("start"));
        assert!(final_log.contains("end"));
    }

    #[tokio::test]
    async fn background_bash_timeout_keeps_partial_output_without_trailing_tail() {
        if !has_background_bash_for_tests().await {
            return;
        }

        let workspace = TempDir::new().expect("temp workspace should build");
        let output_path = workspace.path().join("background-timeout.log");
        let request = StartTerminalTaskRequest {
            command: "printf start; sleep 3; printf end".to_string(),
            workdir: workspace.path().display().to_string(),
            timeout_seconds: Some(1),
            safety_warning: None,
        };

        let err = run_background_bash_command(&request, &output_path)
            .await
            .expect_err("command should time out");
        let err_text = format!("{err:#}");
        assert!(
            err_text.contains("timed out"),
            "unexpected error: {err_text}"
        );

        let partial = String::from_utf8_lossy(
            &tokio::fs::read(&output_path)
                .await
                .expect("timeout should still leave a streamed log"),
        )
        .into_owned();
        assert!(partial.contains("start"));
        assert!(
            !partial.contains("end"),
            "timed-out command should not flush trailing output after kill: {partial:?}"
        );
    }

    #[tokio::test]
    async fn durable_subagent_persists_prompt_configuration_and_persona_skill() {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let skill_dir = workspace.path().join("skills").join("role").join("teacher");
        std::fs::create_dir_all(&skill_dir).expect("create skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: teacher
description: Teaches patiently
allowed-tools:
  - Read
model: gpt-5.4
effort: high
---

你是一个耐心、清晰、循序渐进的老师。
"#,
        )
        .expect("write teacher skill");
        std::fs::write(
            workspace.path().join("DESIGNER.md"),
            "你是一位注重细节与版式的设计师。",
        )
        .expect("write prompt file");

        let hub = build_test_hub_with_skill_dirs(&workspace, &[workspace.path().join("skills")]);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;

        let handle = hub
            .run_durable_subagent(
                root_agent_id,
                Uuid::new_v4(),
                SubAgentRequest {
                    label: Some("designer-teacher".to_string()),
                    task: "完成界面设计".to_string(),
                    context: "聚焦布局与教学表达".to_string(),
                    prompt: Some("你必须始终先给出结构化思路，再给出细节。".to_string()),
                    prompt_file: Some("DESIGNER.md".to_string()),
                    prompt_skill: Some("role:teacher".to_string()),
                    allow_user_send: false,
                    allow_user_show: false,
                    allow_user_ask: false,
                    allow_input_transfer_target: false,
                    existing_agent_id: None,
                },
                sa_core::cancel::cancel_pair().1,
            )
            .await
            .expect("subagent should start");

        let state = hub
            .runtime_store
            .load_agent_state(handle.agent_id)
            .expect("agent state load should succeed")
            .expect("child agent state should exist");

        assert_eq!(
            state.subagent_prompt.as_deref(),
            Some("你必须始终先给出结构化思路，再给出细节。")
        );
        assert_eq!(state.subagent_prompt_file.as_deref(), Some("DESIGNER.md"));
        assert_eq!(state.subagent_prompt_skill.as_deref(), Some("role:teacher"));
        assert!(
            state
                .active_command_invocations
                .iter()
                .any(|invocation| invocation.name == "role:teacher")
        );
    }

    #[tokio::test]
    async fn worker_extra_prompt_includes_prompt_file_and_prompt_skill_content() {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let skill_dir = workspace.path().join("skills").join("role").join("teacher");
        std::fs::create_dir_all(&skill_dir).expect("create skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: teacher
description: Teaches patiently
---

你是一位会先讲概念、再讲例子的老师。
"#,
        )
        .expect("write teacher skill");
        std::fs::write(
            workspace.path().join("DESIGNER.md"),
            "你是一位会先做版式骨架再补视觉细节的设计师。",
        )
        .expect("write prompt file");

        let hub = build_test_hub_with_skill_dirs(&workspace, &[workspace.path().join("skills")]);
        let state = AgentState {
            subagent_prompt: Some("始终先输出提纲。".to_string()),
            subagent_prompt_file: Some("DESIGNER.md".to_string()),
            subagent_prompt_skill: Some("role:teacher".to_string()),
            ..AgentState::new_child(
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                "worker".to_string(),
                false,
                false,
                false,
                false,
                None,
                None,
                None,
                "sessions/agents/test.jsonl".to_string(),
            )
        };
        let agents_md = sa_core::agents_md::AgentsMd {
            path: workspace.path().join("AGENTS.md"),
            content: "主项目约束".to_string(),
            found: true,
        };

        let extra = hub
            .build_agent_extra_prompt(&state, Uuid::new_v4(), &agents_md)
            .await;

        assert!(extra.contains("始终先输出提纲"));
        assert!(extra.contains("版式骨架"));
        assert!(extra.contains("先讲概念、再讲例子"));
    }

    /// Connect one tungstenite client to the ephemeral test server.
    async fn connect_test_client(
        server: &TestServer,
    ) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    {
        let (stream, _) = connect_async(format!("ws://{}/ws", server.addr))
            .await
            .expect("test client should connect to local WS server");
        stream
    }

    /// Read one WS frame with a small timeout so failed tests do not hang.
    async fn next_ws_frame(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> TungsteniteMessage {
        tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for WS frame")
            .expect("WS stream ended unexpectedly")
            .expect("failed to read WS frame")
    }

    /// Parse the next JSON text frame as one backend server message.
    async fn next_server_message(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> ServerMessage {
        match next_ws_frame(ws).await {
            TungsteniteMessage::Text(text) => {
                serde_json::from_str(&text).expect("server text frame should contain valid JSON")
            }
            other => panic!("expected a text server frame, got {other:?}"),
        }
    }

    /// Assert that the next frame is a close frame emitted by the backend.
    async fn assert_next_frame_is_policy_close(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) {
        match next_ws_frame(ws).await {
            TungsteniteMessage::Close(Some(frame)) => {
                assert_eq!(frame.code, TungsteniteCloseCode::Policy);
                assert_eq!(frame.reason, WS_HANDSHAKE_REJECT_CLOSE_REASON);
            }
            other => panic!("expected a close frame after handshake rejection, got {other:?}"),
        }
    }

    /// Build the exact client proof used by the WS handshake.
    fn compute_client_proof(client_version: &str, time_bucket: i64, client_nonce: &str) -> String {
        let material = format!(
            "{TEST_CLIENT_PROOF_LABEL}|{WS_PROTOCOL_ID}|{EXPECTED_CLIENT_NAME}|{client_version}|{time_bucket}|{client_nonce}"
        );
        let mut hasher = Sha256::new();
        hasher.update(material.as_bytes());
        hex_encode(hasher.finalize())
    }

    /// Build one custom hello so tests can isolate bucket skew from proof mismatch.
    fn build_client_hello_with_bucket(
        client_version: &str,
        time_bucket: i64,
        client_nonce: &str,
    ) -> sa_core::ws_protocol::ClientHello {
        sa_core::ws_protocol::ClientHello {
            protocol: WS_PROTOCOL_ID.to_string(),
            hash_algo: WS_HASH_ALGO.to_string(),
            time_step_secs: WS_TIME_STEP_SECS,
            allowed_skew_buckets: WS_ALLOWED_SKEW_BUCKETS,
            client_name: EXPECTED_CLIENT_NAME.to_string(),
            client_version: client_version.to_string(),
            time_bucket,
            client_nonce: client_nonce.to_string(),
            proof: compute_client_proof(client_version, time_bucket, client_nonce),
        }
    }

    /// Serialize and send one client handshake message.
    async fn send_client_hello(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        hello: sa_core::ws_protocol::ClientHello,
    ) {
        let text = serde_json::to_string(&ClientMessage::ClientHello { hello })
            .expect("client hello should serialize");
        ws.send(TungsteniteMessage::Text(text.into()))
            .await
            .expect("client hello should be sent");
    }

    #[tokio::test]
    async fn handshake_rejects_wrong_first_message_and_closes() {
        let server = spawn_test_server(std::time::Duration::from_millis(250), false).await;
        let mut ws = connect_test_client(&server).await;

        let submit = serde_json::to_string(&ClientMessage::Submit {
            task_id: None,
            task: "hello".to_string(),
        })
        .expect("submit should serialize");
        ws.send(TungsteniteMessage::Text(submit.into()))
            .await
            .expect("submit frame should be sent");

        match next_server_message(&mut ws).await {
            ServerMessage::HelloReject { reject } => {
                assert!(
                    reject
                        .reason
                        .contains("first client message must be `client_hello`")
                );
            }
            other => panic!("expected hello_reject, got {other:?}"),
        }

        assert_next_frame_is_policy_close(&mut ws).await;
    }

    #[tokio::test]
    async fn handshake_timeout_rejects_and_closes() {
        let server = spawn_test_server(std::time::Duration::from_millis(75), false).await;
        let mut ws = connect_test_client(&server).await;

        match next_server_message(&mut ws).await {
            ServerMessage::HelloReject { reject } => {
                assert!(reject.reason.contains("Handshake timed out after 75 ms"));
            }
            other => panic!("expected hello_reject after timeout, got {other:?}"),
        }

        assert_next_frame_is_policy_close(&mut ws).await;
    }

    #[tokio::test]
    async fn handshake_rejects_time_bucket_skew_and_closes() {
        let server = spawn_test_server(std::time::Duration::from_millis(250), false).await;
        let mut ws = connect_test_client(&server).await;
        let now_bucket = current_time_bucket().expect("current bucket should resolve");
        let hello =
            build_client_hello_with_bucket("test-client", now_bucket + 2, "bucket-skew-nonce");

        send_client_hello(&mut ws, hello).await;

        match next_server_message(&mut ws).await {
            ServerMessage::HelloReject { reject } => {
                assert!(reject.reason.contains("outside the accepted window"));
            }
            other => panic!("expected hello_reject for bucket skew, got {other:?}"),
        }

        assert_next_frame_is_policy_close(&mut ws).await;
    }

    #[tokio::test]
    async fn handshake_rejects_proof_mismatch_and_closes() {
        let server = spawn_test_server(std::time::Duration::from_millis(250), false).await;
        let mut ws = connect_test_client(&server).await;
        let mut hello = build_client_hello("test-client").expect("valid client hello should build");
        hello.proof =
            "0000000000000000000000000000000000000000000000000000000000000000".to_string();

        send_client_hello(&mut ws, hello).await;

        match next_server_message(&mut ws).await {
            ServerMessage::HelloReject { reject } => {
                assert!(reject.reason.contains("proof mismatch"));
            }
            other => panic!("expected hello_reject for proof mismatch, got {other:?}"),
        }

        assert_next_frame_is_policy_close(&mut ws).await;
    }

    #[tokio::test]
    async fn handshake_accepts_valid_client_hello_before_normal_messages() {
        let server = spawn_test_server(std::time::Duration::from_millis(250), false).await;
        let mut ws = connect_test_client(&server).await;
        let hello = build_client_hello("test-client").expect("valid client hello should build");

        send_client_hello(&mut ws, hello).await;

        match next_server_message(&mut ws).await {
            ServerMessage::ServerHello { hello } => {
                assert_eq!(hello.server_name, "sa");
                assert_eq!(hello.protocol, WS_PROTOCOL_ID);
            }
            other => panic!("expected server_hello, got {other:?}"),
        }

        match next_server_message(&mut ws).await {
            ServerMessage::PendingQuestions { questions } => assert!(questions.is_empty()),
            other => panic!("expected pending_questions after handshake, got {other:?}"),
        }

        match next_server_message(&mut ws).await {
            ServerMessage::RecentShows { files } => assert!(files.is_empty()),
            other => panic!("expected recent_shows after handshake, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bootstrap_mode_sends_init_required_after_handshake() {
        let server = spawn_test_server(std::time::Duration::from_millis(250), true).await;
        let mut ws = connect_test_client(&server).await;
        let hello = build_client_hello("test-client").expect("valid client hello should build");

        send_client_hello(&mut ws, hello).await;

        match next_server_message(&mut ws).await {
            ServerMessage::ServerHello { hello } => {
                assert_eq!(hello.server_name, "sa");
            }
            other => panic!("expected server_hello, got {other:?}"),
        }

        match next_server_message(&mut ws).await {
            ServerMessage::InitRequired { request } => {
                assert!(request.config_path.ends_with("sa.toml"));
                assert_eq!(request.recommended_method, InitMethod::OpenAiCompatible);
                assert!(
                    request
                        .methods
                        .iter()
                        .any(|method| method.id == InitMethod::Custom)
                );
            }
            other => panic!("expected init_required after bootstrap handshake, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recover_runtime_restores_persisted_pending_questions() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let mut root_state = hub
            .runtime_store
            .load_agent_state(root_agent_id)
            .expect("root agent state should load")
            .expect("root agent state should exist");
        let work_id = Uuid::new_v4();
        let created_at = chrono::Utc::now();
        root_state.active_work_id = Some(work_id);
        root_state.status = AgentStatus::Running;
        hub.runtime_store
            .save_agent_state(&root_state)
            .expect("root state should persist");
        hub.runtime_store
            .save_pending_question(&PendingQuestionState {
                agent_id: root_agent_id,
                work_id,
                question_id: Uuid::new_v4(),
                tool_call_id: "call_restore".to_string(),
                prompt: "请选择".to_string(),
                mode: serde_json::to_string(&QuestionMode::SingleChoice)
                    .expect("mode should serialize"),
                options_json: serde_json::to_string(&vec![sa_core::ws_protocol::QuestionOption {
                    id: "a".to_string(),
                    label: "选项A".to_string(),
                    description: None,
                }])
                .expect("options should serialize"),
                allow_free_text: false,
                created_at,
            })
            .expect("pending question should persist");

        hub.recover_runtime().await.expect("runtime should recover");

        let questions = hub.pending_questions_snapshot();
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].task_id, work_id);
        assert_eq!(questions[0].prompt, "请选择");
        assert_eq!(questions[0].created_at, created_at);
        let root_state = hub
            .runtime_store
            .load_agent_state(root_agent_id)
            .expect("root agent state should reload")
            .expect("root agent state should still exist");
        assert_eq!(root_state.status, AgentStatus::WaitingUser);
    }

    #[tokio::test]
    async fn answer_question_persists_tool_result_into_agent_session() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let work_id = Uuid::new_v4();
        let question_id = Uuid::new_v4();
        hub.runtime_store
            .save_pending_question(&PendingQuestionState {
                agent_id: root_agent_id,
                work_id,
                question_id,
                tool_call_id: "call_answer".to_string(),
                prompt: "请选择".to_string(),
                mode: serde_json::to_string(&QuestionMode::SingleChoice)
                    .expect("mode should serialize"),
                options_json: serde_json::to_string(&vec![sa_core::ws_protocol::QuestionOption {
                    id: "a".to_string(),
                    label: "选项A".to_string(),
                    description: None,
                }])
                .expect("options should serialize"),
                allow_free_text: false,
                created_at: chrono::Utc::now(),
            })
            .expect("pending question should persist");
        {
            let mut pending = hub
                .pending_questions
                .lock()
                .expect("pending questions should lock");
            pending.push(PendingQuestionEntry {
                agent_id: root_agent_id,
                work_id,
                tool_call_id: "call_answer".to_string(),
                question: UserQuestion {
                    question_id,
                    task_id: work_id,
                    created_at: chrono::Utc::now(),
                    agent: None,
                    prompt: "请选择".to_string(),
                    mode: QuestionMode::SingleChoice,
                    options: vec![sa_core::ws_protocol::QuestionOption {
                        id: "a".to_string(),
                        label: "选项A".to_string(),
                        description: None,
                    }],
                    allow_free_text: false,
                },
                answer_tx: None,
            });
        }
        let session_store = hub
            .session_store_for_agent(root_agent_id)
            .await
            .expect("root session store should build");
        session_store
            .append_message(&ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_answer".to_string(),
                    kind: "function".to_string(),
                    function: ToolFunctionCall {
                        name: "Ask".to_string(),
                        arguments: "{\"prompt\":\"请选择\"}".to_string(),
                    },
                }]),
                tool_call_id: None,
                request_usage: None,
                responses_input_items: None,
            })
            .expect("assistant Ask turn should persist before the answer arrives");

        hub.answer_question(UserQuestionAnswer {
            question_id,
            selected_option_ids: vec!["a".to_string()],
            free_text: None,
        })
        .await
        .expect("answer should persist");

        assert!(
            hub.runtime_store
                .load_pending_question(root_agent_id)
                .expect("pending question should load")
                .is_none()
        );
        let snapshot = session_store
            .load_snapshot()
            .expect("session snapshot should load");
        assert_eq!(snapshot.messages.len(), 2);
        assert_eq!(snapshot.messages[0].role, "assistant");
        assert_eq!(
            snapshot.messages[0]
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.first())
                .map(|call| call.id.as_str()),
            Some("call_answer")
        );
        let last = snapshot.messages.last().expect("tool result should exist");
        assert_eq!(last.role, "tool");
        assert_eq!(last.tool_call_id.as_deref(), Some("call_answer"));
        let content = last
            .content
            .as_deref()
            .expect("tool result should contain text");
        assert!(content.contains("\"selected_option_ids\":[\"a\"]"));
        assert!(content.contains("\"selected_labels\":[\"选项A\"]"));
    }

    #[tokio::test]
    async fn recovered_pending_question_can_still_be_answered() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let question_id = Uuid::new_v4();
        let work_id = Uuid::new_v4();

        let mut root_state = hub
            .runtime_store
            .load_agent_state(root_agent_id)
            .expect("root state should load")
            .expect("root state should exist");
        root_state.active_work_id = Some(work_id);
        hub.runtime_store
            .save_agent_state(&root_state)
            .expect("root state should save");
        hub.runtime_store
            .save_pending_question(&PendingQuestionState {
                agent_id: root_agent_id,
                work_id,
                question_id,
                tool_call_id: "call_recovered".to_string(),
                prompt: "恢复后的问题".to_string(),
                mode: serde_json::to_string(&QuestionMode::SingleChoice)
                    .expect("mode should serialize"),
                options_json: serde_json::to_string(&vec![sa_core::ws_protocol::QuestionOption {
                    id: "ok".to_string(),
                    label: "继续".to_string(),
                    description: None,
                }])
                .expect("options should serialize"),
                allow_free_text: false,
                created_at: chrono::Utc::now(),
            })
            .expect("pending question should persist");

        hub.recover_runtime().await.expect("runtime should recover");
        assert_eq!(hub.pending_questions_snapshot().len(), 1);

        hub.answer_question(UserQuestionAnswer {
            question_id,
            selected_option_ids: vec!["ok".to_string()],
            free_text: None,
        })
        .await
        .expect("recovered question should be answerable");

        assert!(
            hub.runtime_store
                .load_pending_question(root_agent_id)
                .expect("pending question should load")
                .is_none()
        );
    }

    #[tokio::test]
    async fn submit_task_returns_stable_effective_work_id_for_retries_and_followups() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;

        let first_submit_id = Uuid::new_v4();
        let second_submit_id = Uuid::new_v4();

        let work_id = hub
            .submit_task(first_submit_id, "first".to_string())
            .await
            .expect("first submit should succeed");
        let followup_work_id = hub
            .submit_task(second_submit_id, "followup".to_string())
            .await
            .expect("follow-up submit should succeed");
        let retried_followup_work_id = hub
            .submit_task(second_submit_id, "followup retry".to_string())
            .await
            .expect("retried follow-up submit should be idempotent");

        assert_eq!(followup_work_id, work_id);
        assert_eq!(retried_followup_work_id, work_id);

        let mailbox = hub
            .runtime_store
            .read_mailbox(root_agent_id)
            .expect("root mailbox should load");
        assert_eq!(mailbox.len(), 2);
        assert_eq!(mailbox[0].work_id, Some(work_id));
        assert_eq!(mailbox[1].work_id, Some(work_id));
    }

    #[tokio::test]
    async fn submit_task_records_user_message_interaction_history() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let submit_id = Uuid::new_v4();
        let work_id = hub
            .submit_task(submit_id, "请帮我整理今天的学习任务".to_string())
            .await
            .expect("submit should succeed");

        let entries = load_interaction_entries(&workspace);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].work_id, Some(work_id));
        match &entries[0].payload {
            InteractionPayload::UserMessage {
                submit_id: recorded_submit_id,
                message,
                ..
            } => {
                assert_eq!(*recorded_submit_id, submit_id);
                assert_eq!(message, "请帮我整理今天的学习任务");
            }
            other => panic!("expected user_message interaction, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn persist_pending_question_records_ask_interaction_history() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let mut root_state = hub
            .runtime_store
            .load_agent_state(root_agent_id)
            .expect("root state should load")
            .expect("root state should exist");
        let work_id = Uuid::new_v4();
        root_state.active_work_id = Some(work_id);
        hub.runtime_store
            .save_agent_state(&root_state)
            .expect("root state should save");

        hub.persist_pending_question(
            &mut root_state,
            work_id,
            "call_ask".to_string(),
            AskRequest {
                prompt: "请选择接下来优先做什么".to_string(),
                mode: QuestionMode::SingleChoice,
                options: vec![sa_core::ws_protocol::QuestionOption {
                    id: "math".to_string(),
                    label: "数学".to_string(),
                    description: None,
                }],
                allow_free_text: true,
            },
        )
        .await
        .expect("ask should persist");

        let entries = load_interaction_entries(&workspace);
        assert_eq!(entries.len(), 1);
        match &entries[0].payload {
            InteractionPayload::Ask {
                prompt,
                mode,
                options,
                allow_free_text,
                ..
            } => {
                assert_eq!(prompt, "请选择接下来优先做什么");
                assert_eq!(*mode, QuestionMode::SingleChoice);
                assert_eq!(options.len(), 1);
                assert!(*allow_free_text);
            }
            other => panic!("expected ask interaction, got {other:?}"),
        }
    }

    #[test]
    fn root_frontend_identity_uses_study_administrator_display_name() {
        let root_agent_id = Uuid::new_v4();
        let state = AgentState::new_root(root_agent_id, "sessions/agents/root.jsonl".to_string());
        let identity = Hub::frontend_identity_from_state(&state);

        assert_eq!(identity.agent_id, root_agent_id);
        assert_eq!(identity.agent_label, "root");
        assert_eq!(identity.display_name, "学习委员");
        assert!(identity.is_root);
    }

    #[test]
    fn summarize_work_text_handles_multibyte_utf8_without_panicking() {
        let input = format!("{}中文摘要边界测试", "a".repeat(119));

        let result = std::panic::catch_unwind(|| Hub::summarize_work_text(&input));

        let summary = result.expect("multibyte summary truncation must not panic");
        assert!(summary.ends_with("..."));
        assert!(summary.len() <= 123);
    }

    #[tokio::test]
    async fn child_finish_does_not_reclaim_transferred_input_ownership() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let child_agent_id = Uuid::new_v4();
        let child_work_id = Uuid::new_v4();
        let parent_work_id = Uuid::new_v4();
        let started_at = chrono::Utc::now();
        let mut child_state = AgentState::new_child(
            child_agent_id,
            root_agent_id,
            root_agent_id,
            "聊天助手".to_string(),
            true,
            true,
            true,
            true,
            None,
            None,
            None,
            "sessions/agents/child.jsonl".to_string(),
        );
        child_state.active_work_id = Some(child_work_id);
        child_state.active_work_summary = Some("处理和同学的聊天".to_string());
        child_state.active_started_at = Some(started_at);
        child_state.parent_work_id = Some(parent_work_id);
        child_state.work_has_parent_message = true;
        hub.runtime_store
            .save_agent_state(&child_state)
            .expect("child state should save");

        hub.set_input_owner(child_agent_id, Some(child_work_id), "test transfer")
            .await
            .expect("input owner should transfer to child");
        hub.handle_finish_outcome(
            &mut child_state,
            "当前轮次完成".to_string(),
            "已完成当前工作".to_string(),
            false,
        )
        .await
        .expect("finish should succeed");

        assert_eq!(
            hub.team_state.lock().await.input_owner_agent_id,
            Some(child_agent_id)
        );
        let persisted = hub
            .runtime_store
            .load_team_state()
            .expect("team state should load")
            .expect("team state should exist");
        assert_eq!(persisted.input_owner_agent_id, Some(child_agent_id));
    }

    #[tokio::test]
    async fn transfer_input_notifies_both_previous_and_new_owners() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let child_agent_id = Uuid::new_v4();
        let root_work_id = Uuid::new_v4();
        let child_work_id = Uuid::new_v4();

        let mut root_state = hub
            .runtime_store
            .load_agent_state(root_agent_id)
            .expect("root state should load")
            .expect("root state should exist");
        root_state.active_work_id = Some(root_work_id);
        root_state.active_work_summary = Some("主代理继续工作".to_string());
        root_state.active_started_at = Some(chrono::Utc::now());
        hub.runtime_store
            .save_agent_state(&root_state)
            .expect("root state should save");

        let mut child_state = AgentState::new_child(
            child_agent_id,
            root_agent_id,
            root_agent_id,
            "聊天助手".to_string(),
            true,
            true,
            true,
            true,
            None,
            None,
            None,
            "sessions/agents/child.jsonl".to_string(),
        );
        child_state.active_work_id = Some(child_work_id);
        child_state.active_work_summary = Some("子代理接待同学".to_string());
        child_state.active_started_at = Some(chrono::Utc::now());
        hub.runtime_store
            .save_agent_state(&child_state)
            .expect("child state should save");

        hub.set_input_owner(child_agent_id, Some(root_work_id), "test transfer to child")
            .await
            .expect("input owner should transfer to child");

        let root_mailbox = hub
            .runtime_store
            .read_mailbox(root_agent_id)
            .expect("root mailbox should load");
        let child_mailbox = hub
            .runtime_store
            .read_mailbox(child_agent_id)
            .expect("child mailbox should load");
        assert!(
            root_mailbox
                .iter()
                .any(|entry| entry.kind == MailboxEntryKind::SystemNotice
                    && entry.message.contains("已从你这里移走")),
            "root should receive a losing-input notice",
        );
        assert!(
            child_mailbox
                .iter()
                .any(|entry| entry.kind == MailboxEntryKind::SystemNotice
                    && entry.message.contains("已交给你")),
            "child should receive a gaining-input notice",
        );

        hub.set_input_owner(root_agent_id, Some(root_work_id), "test reclaim to root")
            .await
            .expect("input owner should transfer back to root");

        let root_mailbox = hub
            .runtime_store
            .read_mailbox(root_agent_id)
            .expect("root mailbox should reload");
        let child_mailbox = hub
            .runtime_store
            .read_mailbox(child_agent_id)
            .expect("child mailbox should reload");
        assert!(
            root_mailbox
                .iter()
                .filter(|entry| entry.kind == MailboxEntryKind::SystemNotice)
                .any(|entry| entry.message.contains("已交给你")),
            "root should receive a gaining-input notice when reclaimed",
        );
        assert!(
            child_mailbox
                .iter()
                .filter(|entry| entry.kind == MailboxEntryKind::SystemNotice)
                .any(|entry| entry.message.contains("已从你这里移走")),
            "child should receive a losing-input notice when root reclaims input",
        );
    }

    #[tokio::test]
    async fn child_user_outputs_carry_identity_metadata_without_text_prefixes() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let child_agent_id = Uuid::new_v4();
        let work_id = Uuid::new_v4();
        let mut child_state = AgentState::new_child(
            child_agent_id,
            root_agent_id,
            root_agent_id,
            "数学老师".to_string(),
            true,
            true,
            true,
            true,
            None,
            None,
            None,
            "sessions/agents/child.jsonl".to_string(),
        );
        child_state.active_work_id = Some(work_id);
        child_state.active_work_summary = Some("和同学沟通".to_string());
        child_state.active_started_at = Some(chrono::Utc::now());
        hub.runtime_store
            .save_agent_state(&child_state)
            .expect("child state should save");

        let runtime = hub.build_agent_tool_runtime(&child_state, work_id, true);
        let expected_identity = AgentIdentity {
            agent_id: child_agent_id,
            agent_label: "数学老师".to_string(),
            display_name: "数学老师".to_string(),
            is_root: false,
        };
        let mut events_rx = hub.events_tx.subscribe();

        runtime
            .send_message("我来继续接手和同学对话。".to_string())
            .await
            .expect("send should succeed");
        match events_rx.recv().await.expect("send event should arrive") {
            ServerMessage::Event { event } => {
                assert_eq!(event.kind, EventKind::Message);
                assert_eq!(event.message, "我来继续接手和同学对话。");
                assert_eq!(event.agent, Some(expected_identity.clone()));
            }
            other => panic!("expected message event, got {other:?}"),
        }

        runtime
            .show_file(UserVisibleFile {
                show_id: Uuid::new_v4(),
                task_id: Uuid::nil(),
                agent: None,
                path: "plan.md".to_string(),
                title: Some("学习计划".to_string()),
                prompt: "展示整理好的学习计划".to_string(),
                media_type: "text/markdown".to_string(),
                encoding: UserVisibleFileEncoding::Utf8,
                content: "# Plan".to_string(),
                bytes: 6,
            })
            .await
            .expect("show should succeed");
        match events_rx.recv().await.expect("show event should arrive") {
            ServerMessage::Show { file } => {
                assert_eq!(file.title.as_deref(), Some("学习计划"));
                assert_eq!(file.agent, Some(expected_identity.clone()));
            }
            other => panic!("expected show event, got {other:?}"),
        }

        hub.persist_pending_question(
            &mut child_state,
            work_id,
            "call_child_ask".to_string(),
            AskRequest {
                prompt: "接下来先讲哪一科？".to_string(),
                mode: QuestionMode::SingleChoice,
                options: vec![sa_core::ws_protocol::QuestionOption {
                    id: "math".to_string(),
                    label: "数学".to_string(),
                    description: None,
                }],
                allow_free_text: false,
            },
        )
        .await
        .expect("ask should persist");
        match events_rx
            .recv()
            .await
            .expect("question event should arrive")
        {
            ServerMessage::Question { question } => {
                assert_eq!(question.prompt, "接下来先讲哪一科？");
                assert_eq!(question.agent, Some(expected_identity.clone()));
            }
            other => panic!("expected question event, got {other:?}"),
        }
        assert_eq!(
            hub.pending_questions_snapshot()[0].agent,
            Some(expected_identity)
        );
    }

    #[tokio::test]
    async fn wait_kind_work_recognizes_older_completed_work_records() {
        let workspace = TempDir::new().expect("temp workspace should build");
        let hub = build_test_hub(&workspace);
        let root_agent_id = hub
            .runtime_store
            .load_root_marker()
            .expect("root marker should load")
            .expect("root marker should exist")
            .root_agent_id;
        let older_work_id = Uuid::new_v4();
        let newer_work_id = Uuid::new_v4();

        hub.runtime_store
            .save_work_state(&sa_core::runtime::state::RuntimeWorkState {
                work_id: older_work_id,
                owner_agent_id: root_agent_id,
                root_agent_id,
                summary: "older".to_string(),
                status: sa_core::runtime::state::RuntimeWorkStatus::Finished,
                started_at: chrono::Utc::now(),
                finished_at: Some(chrono::Utc::now()),
                result_summary: Some("older finished".to_string()),
            })
            .expect("older work should persist");
        hub.runtime_store
            .save_work_state(&sa_core::runtime::state::RuntimeWorkState {
                work_id: newer_work_id,
                owner_agent_id: root_agent_id,
                root_agent_id,
                summary: "newer".to_string(),
                status: sa_core::runtime::state::RuntimeWorkStatus::Finished,
                started_at: chrono::Utc::now(),
                finished_at: Some(chrono::Utc::now()),
                result_summary: Some("newer finished".to_string()),
            })
            .expect("newer work should persist");

        let mut root_state = hub
            .runtime_store
            .load_agent_state(root_agent_id)
            .expect("root agent state should load")
            .expect("root agent state should exist");
        root_state.last_finished_work_id = Some(newer_work_id);
        hub.runtime_store
            .save_agent_state(&root_state)
            .expect("root agent state should save");

        let notice = hub
            .wait_satisfied_notice(&WaitingDependency {
                kind: WaitKind::Work,
                id: older_work_id,
                until: WaitUntil::Finished,
                timeout_at: None,
            })
            .expect("wait notice should compute");

        assert!(notice.is_some());
        assert!(
            notice
                .expect("older work should be recognized as finished")
                .contains(&older_work_id.to_string())
        );
    }

    #[test]
    fn generated_bootstrap_config_includes_dream_defaults() {
        let config_path = PathBuf::from("G:/AgentProjects/Claw/sa/sa.toml");
        let bootstrap = build_bootstrap_state(&config_path).expect("bootstrap should build");
        let request = InitializeConfigRequest {
            method: InitMethod::OpenAiCompatible,
            base_url: "https://example.com/v1".to_string(),
            api_key: "sk-test".to_string(),
            model: "gpt-5.2".to_string(),
            wire_api: None,
            auth_style: None,
            system_role_name: Some("developer".to_string()),
            reasoning_effort: Some("high".to_string()),
        };

        let rendered =
            render_initial_config_toml(&bootstrap, &request).expect("config should render");
        assert!(rendered.contains("[dream]"));
        assert!(rendered.contains("enabled = true"));
        assert!(rendered.contains("daily_note_lookback_days = 3"));
        assert!(rendered.contains("recent_session_segments = 6"));
        assert!(rendered.contains("recent_topic_files = 24"));
    }

    /// Write one minimal ready-runtime config used by hot-reload tests.
    fn write_test_ready_config(
        path: &Path,
        model: &str,
        skill_dirs: &[&str],
        max_concurrent_model_calls: usize,
        workspace_root: &str,
    ) {
        let skills = if skill_dirs.is_empty() {
            "dirs = []".to_string()
        } else {
            format!(
                "dirs = [{}]",
                skill_dirs
                    .iter()
                    .map(|dir| format!("{dir:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let raw = format!(
            r#"[llm]
base_url = "http://127.0.0.1:1/v1"
api_key = "test-key"
model = "{model}"
system_role_name = "developer"

[server]
bind = "127.0.0.1:8765"
ws_path = "/ws"

[workspace]
root_dir = {workspace_root:?}
agents_md = "AGENTS.md"

[skills]
{skills}

[dream]
enabled = false

[team]
max_concurrent_model_calls = {max_concurrent_model_calls}
"#
        );
        std::fs::write(path, raw).expect("test config should write");
    }

    #[tokio::test]
    async fn hot_reload_refreshes_runner_and_skills_in_place() {
        let workspace = TempDir::new().expect("temp workspace should build");
        std::fs::write(workspace.path().join("AGENTS.md"), "root instructions")
            .expect("AGENTS.md should write");
        let skills_dir = workspace.path().join("skills");
        std::fs::create_dir_all(&skills_dir).expect("skills dir should exist");
        let config_path = workspace.path().join("sa.toml");
        write_test_ready_config(&config_path, "model-before", &["./skills"], 2, ".");

        let cfg = load_config_from_file(&config_path).expect("config should load");
        let runtime = build_runtime_from_config(cfg, &config_path)
            .await
            .expect("runtime should build");
        let hub = runtime.hub;

        assert_eq!(hub.current_runner_config().model, "model-before");
        assert!(
            hub.current_preload_ctx()
                .skills
                .get("superpowers:brainstorming")
                .is_none()
        );
        assert_eq!(hub.current_team_cfg().max_concurrent_model_calls, 2);

        let skill_path = skills_dir.join("superpowers").join("brainstorming");
        std::fs::create_dir_all(&skill_path).expect("nested skill dir should exist");
        std::fs::write(
            skill_path.join("SKILL.md"),
            r#"---
name: brainstorming
description: test skill
---

test body
"#,
        )
        .expect("skill should write");
        write_test_ready_config(&config_path, "model-after", &["./skills"], 5, ".");

        let summary = hub
            .reload_runtime_from_disk()
            .await
            .expect("reload should succeed");

        assert!(summary.contains("model-after"));
        assert_eq!(hub.current_runner_config().model, "model-after");
        assert!(
            hub.current_preload_ctx()
                .skills
                .get("superpowers:brainstorming")
                .is_some()
        );
        assert_eq!(hub.current_team_cfg().max_concurrent_model_calls, 5);
    }

    #[tokio::test]
    async fn hot_reload_rejects_workspace_root_change() {
        let workspace = TempDir::new().expect("temp workspace should build");
        std::fs::write(workspace.path().join("AGENTS.md"), "root instructions")
            .expect("AGENTS.md should write");
        let config_path = workspace.path().join("sa.toml");
        write_test_ready_config(&config_path, "model-before", &[], 2, ".");

        let cfg = load_config_from_file(&config_path).expect("config should load");
        let runtime = build_runtime_from_config(cfg, &config_path)
            .await
            .expect("runtime should build");
        let hub = runtime.hub;

        let other_root = workspace.path().join("other-workspace");
        std::fs::create_dir_all(&other_root).expect("other workspace should exist");
        write_test_ready_config(
            &config_path,
            "model-after",
            &[],
            2,
            other_root.to_string_lossy().as_ref(),
        );

        let err = hub
            .reload_runtime_from_disk()
            .await
            .expect_err("workspace-root hot reload must fail");

        assert!(err.to_string().contains("workspace root"));
    }
}
