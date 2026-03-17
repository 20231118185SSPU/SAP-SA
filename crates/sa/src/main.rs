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
use sa_core::agent::{AgentRunner, AgentRunnerConfig, DrainQueuedUserMessagesFn, EmitEventFn};
use sa_core::agents_md::{extract_markdown_file_references, load_agents_md};
use sa_core::cancel::{CancelHandle, cancel_pair};
use sa_core::compact::CompactionConfig;
use sa_core::config::load_config_from_file;
use sa_core::mcp_client::McpRegistry;
use sa_core::memory::{build_prompt_block as build_memory_prompt_block, is_memory_reference};
use sa_core::openai::OpenAiClient;
use sa_core::session::SessionStore;
use sa_core::skills::SkillRegistry;
use sa_core::tools::{
    AskQuestionFn, AskRequest, MAX_SUBAGENT_DEPTH, RunSubAgentFn, SendMessageFn, ShowFileFn,
    SubAgentRequest, ToolContext, ToolExecutor, ToolRuntime,
};
use sa_core::ws_identity::{
    LocalIdentity, WS_HANDSHAKE_TIMEOUT_SECS, build_hello_reject, build_server_hello,
    load_local_identity, verify_client_hello,
};
use sa_core::ws_protocol::{
    ClientMessage, Event, EventKind, QuestionMode, ServerMessage, UserQuestion, UserQuestionAnswer,
    UserVisibleFile, UserVisibleFileEncoding,
};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{broadcast, mpsc, oneshot};
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

/// CLI arguments.
#[derive(clap::Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to `sa.toml`.
    #[arg(long, default_value = "sa.toml")]
    config: PathBuf,
}

/// A single queued task.
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
    /// Structured question visible to clients.
    question: UserQuestion,
    /// One-shot channel used to resume the blocked `Ask` tool.
    answer_tx: oneshot::Sender<UserQuestionAnswer>,
}

/// Shared daemon state.
///
/// This object is designed so:
/// - The agent worker loop can run without any WS connection.
/// - WS connections can subscribe to events and submit tasks.
#[derive(Debug)]
struct Hub {
    /// Task queue sender (worker loop receives from this).
    task_tx: mpsc::Sender<TaskRequest>,

    /// Broadcast channel for live server messages (events + questions).
    events_tx: broadcast::Sender<ServerMessage>,

    /// In-memory event buffer for reconnect/history.
    events_buf: Mutex<VecDeque<Event>>,

    /// Monotonic event id generator.
    next_event_id: AtomicU64,

    /// Set of submit ids already accepted (idempotency for retries).
    ///
    /// This now covers both:
    /// - top-level tasks
    /// - queued follow-up messages sent while a task is still running
    seen_tasks: Mutex<HashSet<Uuid>>,

    /// Currently running task (if any), so we can cancel it.
    current_task: Mutex<Option<CurrentTask>>,

    /// Questions waiting for a user answer.
    pending_questions: Mutex<Vec<PendingQuestionEntry>>,

    /// Recent files explicitly shown to the user.
    recent_shows: Mutex<VecDeque<UserVisibleFile>>,

    /// Context used for safe path resolution when preloading files.
    preload_ctx: ToolContext,

    /// The agent runner (OpenAI + tools loop).
    runner: AgentRunner,

    /// Path to Agents.md (reloaded for every task).
    ///
    /// Why reload each task?
    /// - Users often edit `Agents.md` while the daemon is running.
    /// - Users may create `Agents.md` after the daemon starts.
    /// Reloading makes this behavior verifiable and fixes "Agents.md not loaded" confusion.
    agents_md_path: PathBuf,

    /// Durable top-level conversation store rooted at `workspace/sessions/`.
    session_store: Arc<SessionStore>,

    /// Stable machine identity reused by every WS handshake.
    ws_identity: LocalIdentity,
}

impl Hub {
    /// Create a new hub and spawn the background worker.
    fn new(
        runner: AgentRunner,
        agents_md_path: PathBuf,
        preload_ctx: ToolContext,
        session_store: Arc<SessionStore>,
        ws_identity: LocalIdentity,
    ) -> Arc<Self> {
        // Task queue capacity (small but adequate for minimal agent).
        let (task_tx, task_rx) = mpsc::channel::<TaskRequest>(128);

        // Broadcast event channel. Capacity controls how many events a slow
        // subscriber can lag behind before it gets a `Lagged` error.
        let (events_tx, _) = broadcast::channel::<ServerMessage>(1024);

        // Build hub.
        let hub = Arc::new(Self {
            task_tx,
            events_tx,
            events_buf: Mutex::new(VecDeque::new()),
            next_event_id: AtomicU64::new(0),
            seen_tasks: Mutex::new(HashSet::new()),
            current_task: Mutex::new(None),
            pending_questions: Mutex::new(Vec::new()),
            recent_shows: Mutex::new(VecDeque::new()),
            preload_ctx,
            runner,
            agents_md_path,
            session_store,
            ws_identity,
        });

        // Spawn worker loop.
        let hub_clone = Arc::clone(&hub);
        tokio::spawn(async move {
            hub_clone.worker_loop(task_rx).await;
        });

        hub
    }

    /// Publish an event:
    /// - assign `event_id`
    /// - store in buffer
    /// - broadcast to subscribers
    fn publish(&self, kind: EventKind, task_id: Uuid, message: String) {
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
        };

        self.mirror_server_message(&ServerMessage::Event {
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
        self.mirror_server_message(&msg);
        let _ = self.events_tx.send(msg);
    }

    /// Mirror one outbound frontend message to the backend terminal.
    ///
    /// This is intentionally verbose for debugging because the current CLI does
    /// not render the whole stream.
    fn mirror_server_message(&self, msg: &ServerMessage) {
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
                eprintln!(
                    "[frontend][ask][task={}][id={}] {}",
                    question.task_id, question.question_id, question.prompt
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
            ServerMessage::Error { message } => {
                eprintln!("[frontend][error] {message}");
            }
        }
    }

    /// Request interruption (cancellation) of a running task.
    ///
    /// This is called from the WS handler when a client sends:
    /// `{"type":"interrupt","task_id":"..."}`
    fn request_interrupt(&self, task_id: Uuid) {
        let current = self
            .current_task
            .lock()
            .expect("current_task mutex poisoned");

        let Some(info) = current.as_ref() else {
            self.publish(
                EventKind::Error,
                task_id,
                "Interrupt requested, but no task is currently running.".to_string(),
            );
            return;
        };

        if info.task_id != task_id {
            self.publish(
                EventKind::Error,
                task_id,
                format!(
                    "Interrupt requested for task {task_id}, but currently running task is {}.",
                    info.task_id
                ),
            );
            return;
        }

        self.publish(
            EventKind::Log,
            task_id,
            "Interrupt requested; cancelling current task.".to_string(),
        );

        info.cancel.cancel();
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

    /// Build the OpenClaw-style root memory prompt block.
    ///
    /// We only inject stable root memory files (`MEMORY.md` / `memory.md`) for
    /// top-level tasks. Daily memory remains on-demand via `MemorySearch` /
    /// `MemoryGet`.
    async fn memory_prompt_block(&self, task_id: Uuid) -> String {
        match build_memory_prompt_block(&self.preload_ctx.workspace_root).await {
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
    fn answer_question(&self, answer: UserQuestionAnswer) -> anyhow::Result<()> {
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

        let PendingQuestionEntry {
            question,
            answer_tx,
        } = pending.remove(index);

        if answer_tx.send(answer).is_err() {
            anyhow::bail!("Question waiter dropped before receiving the answer");
        }

        self.broadcast_server_message(ServerMessage::QuestionResolved {
            question_id: question.question_id,
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
    async fn submit_task(&self, task_id: Uuid, task: String) -> anyhow::Result<()> {
        // Ensure idempotency: if we have already seen this submit id, do not process it again.
        {
            let mut seen = self.seen_tasks.lock().expect("seen_tasks mutex poisoned");
            if !seen.insert(task_id) {
                return Ok(());
            }
        }

        let current = self
            .current_task
            .lock()
            .expect("current_task mutex poisoned")
            .clone();

        if let Some(info) = current.filter(|info| !info.cancel.is_cancelled()) {
            {
                let mut queued = info
                    .follow_up_messages
                    .lock()
                    .expect("follow_up_messages mutex poisoned");
                queued.push_back(task);
            }

            self.publish(
                EventKind::Log,
                info.task_id,
                format!("Queued a follow-up message for the current task (submit_id={task_id})."),
            );
            return Ok(());
        }

        // Emit an event immediately so clients see that the task is queued.
        self.publish(
            EventKind::Log,
            task_id,
            "Task accepted and queued.".to_string(),
        );

        // Enqueue.
        self.task_tx
            .send(TaskRequest { task_id, task })
            .await
            .context("Failed to enqueue task")?;

        Ok(())
    }

    /// Ask the user a structured question and wait until an answer arrives.
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
                question: question.clone(),
                answer_tx,
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
    fn build_tool_runtime(self: &Arc<Self>, task_id: Uuid, depth: u32) -> ToolRuntime {
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

        ToolRuntime::new(send_message, ask_question, show_file, run_subagent)
    }

    /// Run a nested sub-agent while keeping all output attached to the
    /// top-level task event stream.
    async fn run_subagent(
        self: &Arc<Self>,
        task_id: Uuid,
        parent_depth: u32,
        request: SubAgentRequest,
        cancel: sa_core::cancel::CancelToken,
    ) -> anyhow::Result<String> {
        let depth = parent_depth.saturating_add(1);
        if depth > MAX_SUBAGENT_DEPTH {
            anyhow::bail!(
                "SubAgent depth limit exceeded (requested depth={}, max={})",
                depth,
                MAX_SUBAGENT_DEPTH
            );
        }

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

        let agents_md = match load_agents_md(self.agents_md_path.clone()).await {
            Ok(a) => a,
            Err(err) => {
                self.publish(
                    EventKind::Error,
                    task_id,
                    format!("Failed to read Agents.md for subagent: {err}"),
                );
                sa_core::agents_md::AgentsMd {
                    path: self.agents_md_path.clone(),
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
            let kind = if matches!(kind, EventKind::Final) {
                EventKind::Log
            } else {
                kind
            };
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
        let result = self
            .runner
            .run_task(
                task_id,
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

        result
    }

    /// Background worker loop that executes tasks sequentially.
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
            let agents_md = match load_agents_md(self.agents_md_path.clone()).await {
                Ok(a) => a,
                Err(err) => {
                    self.publish(
                        EventKind::Error,
                        req.task_id,
                        format!("Failed to read Agents.md: {err}"),
                    );
                    sa_core::agents_md::AgentsMd {
                        path: self.agents_md_path.clone(),
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
            let extra_prompt = format!("{memory_block}\n\n{preload_block}");

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
            match self
                .runner
                .run_task(
                    req.task_id,
                    req.task.clone(),
                    &agents_md,
                    Some(extra_prompt.as_str()),
                    Some(Arc::clone(&self.session_store)),
                    runtime,
                    &cancel_token,
                    Some(drain_queued_user_messages),
                    emit,
                )
                .await
            {
                Ok(final_text) => {
                    // The agent runner already emits `Final`, but we keep an explicit
                    // completion log for clarity.
                    self.publish(EventKind::Log, req.task_id, "Task finished.".to_string());
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
            let abs = match self.preload_ctx.resolve_under_workspace(&r) {
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

/// Decorate nested agent text so user-visible messages stay traceable.
fn decorate_nested_text(depth: u32, text: &str) -> String {
    if depth == 0 {
        return text.to_string();
    }

    format!("[subagent depth={depth}] {text}")
}

/// Decorate nested agent prompts for `Ask`.
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

    if !question.allow_free_text && !free_text.is_empty() {
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

/// Mirror one event in a stable multi-line format.
fn mirror_event_line(prefix: &str, event: &Event) {
    let header = format!(
        "{prefix}[kind={:?}][task={}][event_id={}][ts={}] ",
        event.kind, event.task_id, event.event_id, event.ts
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
    eprintln!(
        "{prefix}[task={}][title={}] path={} media_type={} bytes={}",
        file.task_id, title, file.path, file.media_type, file.bytes
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

/// Send one direct per-connection message and mirror it to the backend
/// terminal.
fn send_direct_server_message(
    hub: &Arc<Hub>,
    out_tx: &mpsc::UnboundedSender<ServerMessage>,
    msg: ServerMessage,
) {
    hub.mirror_server_message(&msg);
    let _ = out_tx.send(msg);
}

/// Send one handshake message before the per-connection writer task exists.
async fn send_handshake_server_message(
    hub: &Arc<Hub>,
    ws_tx: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    msg: ServerMessage,
) -> anyhow::Result<()> {
    hub.mirror_server_message(&msg);

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
    hub: &Arc<Hub>,
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
        hub,
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
async fn ws_route(ws: WebSocketUpgrade, State(hub): State<Arc<Hub>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_session(socket, hub))
}

/// Handle a single WS connection with the production handshake timeout.
async fn ws_session(socket: WebSocket, hub: Arc<Hub>) {
    ws_session_with_timeout(
        socket,
        hub,
        std::time::Duration::from_secs(WS_HANDSHAKE_TIMEOUT_SECS),
    )
    .await;
}

/// Handle a single WS connection with an explicit handshake timeout.
///
/// Tests use this variant directly so timeout behavior can be verified without
/// sleeping for the full production timeout.
async fn ws_session_with_timeout(
    socket: WebSocket,
    hub: Arc<Hub>,
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
                &hub,
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
                &hub,
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
            &hub,
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
                &hub,
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
                &hub,
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
            &hub,
            connection_id,
            "client_hello_verification_failed",
            server_version,
            &mut ws_tx,
            format!("Client hello verification failed: {err}"),
        )
        .await;
        return;
    }

    let server_hello =
        match build_server_hello(&hub.ws_identity, server_version, &client_hello.client_nonce) {
            Ok(hello) => hello,
            Err(err) => {
                reject_handshake(
                    &hub,
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
        &hub,
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

    // Immediately send any questions that were already pending before this
    // client connected. This makes reconnecting clients able to continue an
    // interrupted `Ask` interaction.
    send_direct_server_message(
        &hub,
        &out_tx,
        ServerMessage::PendingQuestions {
            questions: hub.pending_questions_snapshot(),
        },
    );
    send_direct_server_message(
        &hub,
        &out_tx,
        ServerMessage::RecentShows {
            files: hub.recent_shows_snapshot(),
        },
    );

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

    // Event forwarder task: broadcast -> out_tx.
    let mut events_rx = hub.events_tx.subscribe();
    let out_tx_events = out_tx.clone();
    let hub_for_forwarder = Arc::clone(&hub);
    let forwarder = tokio::spawn(async move {
        loop {
            match events_rx.recv().await {
                Ok(msg) => {
                    let _ = out_tx_events.send(msg);
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    send_direct_server_message(
                        &hub_for_forwarder,
                        &out_tx_events,
                        ServerMessage::Error {
                            message: format!("Lagged in event stream; skipped {skipped} events"),
                        },
                    );
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Reader loop: handle client requests.
    while let Some(Ok(frame)) = ws_rx.next().await {
        match frame {
            Message::Text(text) => {
                let parsed = serde_json::from_str::<ClientMessage>(&text);
                let msg = match parsed {
                    Ok(msg) => msg,
                    Err(err) => {
                        send_direct_server_message(
                            &hub,
                            &out_tx,
                            ServerMessage::Error {
                                message: format!("Invalid JSON: {err}"),
                            },
                        );
                        continue;
                    }
                };

                match msg {
                    ClientMessage::ClientHello { .. } => {
                        send_direct_server_message(
                            &hub,
                            &out_tx,
                            ServerMessage::Error {
                                message: "`client_hello` is only allowed as the first message on a connection".to_string(),
                            },
                        );
                    }
                    ClientMessage::Submit { task_id, task } => {
                        // If the client didn't provide an id, we generate one.
                        let task_id = task_id.unwrap_or_else(Uuid::new_v4);

                        // Acknowledge immediately.
                        send_direct_server_message(
                            &hub,
                            &out_tx,
                            ServerMessage::Accepted { task_id },
                        );

                        // Enqueue (idempotent).
                        if let Err(err) = hub.submit_task(task_id, task).await {
                            send_direct_server_message(
                                &hub,
                                &out_tx,
                                ServerMessage::Error {
                                    message: format!("Failed to submit task: {err}"),
                                },
                            );
                        }
                    }
                    ClientMessage::GetHistory { from_event_id } => {
                        let events = hub.history_since(from_event_id).await;
                        send_direct_server_message(
                            &hub,
                            &out_tx,
                            ServerMessage::History { events },
                        );
                    }
                    ClientMessage::Interrupt { task_id } => {
                        hub.request_interrupt(task_id);
                    }
                    ClientMessage::AnswerQuestion { answer } => {
                        if let Err(err) = hub.answer_question(answer) {
                            send_direct_server_message(
                                &hub,
                                &out_tx,
                                ServerMessage::Error {
                                    message: format!("Failed to answer question: {err}"),
                                },
                            );
                        }
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
    forwarder.abort();
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

    // Load config.
    let cfg = load_config_from_file(&args.config)?;
    let config_dir = args
        .config
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    // Resolve workspace paths.
    let workspace_root = cfg.workspace.root_dir_path(&config_dir);
    let agents_md_path = cfg.workspace.agents_md_path(&workspace_root);

    tracing::info!("Workspace root: {}", workspace_root.display());
    tracing::info!("Agents.md path: {}", agents_md_path.display());

    // Best-effort load `Agents.md` once for startup logging.
    //
    // The worker loop reloads `Agents.md` **for every task**, so changes take
    // effect without restarting the daemon.
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
    let skills = Arc::new(SkillRegistry::scan(&skill_dirs)?);
    tracing::info!("Discovered {} skill(s).", skills.list().len());

    // Connect external MCP servers before freezing the tool registry.
    let mcp_registry = if cfg.mcp.enabled && !cfg.mcp.servers.is_empty() {
        tracing::info!(
            "Initializing MCP client 闁?{} server(s) configured",
            cfg.mcp.servers.len()
        );
        match McpRegistry::connect_all(&cfg.mcp.servers).await {
            Ok(registry) if !registry.is_empty() => {
                tracing::info!(
                    "MCP: {} tool(s) registered from {} server(s)",
                    registry.tool_count(),
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

    // Build tools.
    let tool_ctx = ToolContext::new(workspace_root, Arc::clone(&skills))?;
    let preload_ctx = tool_ctx.clone();
    let session_store = Arc::new(SessionStore::new(preload_ctx.workspace_root.clone())?);

    let tools = ToolExecutor::new(tool_ctx, mcp_registry);

    // Build LLM client.
    //
    // IMPORTANT: compute derived string options before we move strings out of `cfg.llm`.
    let system_role_name = cfg.llm.effective_system_role_name().to_string();
    let reasoning_effort = cfg.llm.effective_reasoning_effort().map(str::to_string);
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
        model: cfg.llm.model,
        system_role_name,
        reasoning_effort,
        max_steps: cfg.llm.max_steps,
        compaction: cfg.compaction,
    };
    let runner = AgentRunner::new(llm, tools, Arc::clone(&skills), runner_cfg);

    // Hub (spawns worker loop).
    let ws_identity = load_local_identity()?;
    tracing::info!("WS identity machine hint: {}", ws_identity.machine_hint);
    let hub = Hub::new(
        runner,
        agents_md_path,
        preload_ctx,
        session_store,
        ws_identity,
    );

    // Build HTTP router (WS only).
    let app = Router::new()
        .route(&cfg.server.ws_path, get(ws_route))
        .with_state(hub);

    // Bind listener.
    let listener = tokio::net::TcpListener::bind(&cfg.server.bind)
        .await
        .with_context(|| format!("Failed to bind {}", cfg.server.bind))?;

    tracing::info!("WebSocket listening on {}", cfg.server.bind);

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

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    /// Minimal `Hub` factory for WS handshake tests.
    fn build_test_hub(workspace: &TempDir) -> Arc<Hub> {
        let skills = Arc::new(SkillRegistry::scan(&[]).expect("empty skill registry should build"));
        let tool_ctx = ToolContext::new(workspace.path().to_path_buf(), Arc::clone(&skills))
            .expect("tool context should build for temp workspace");
        let preload_ctx = tool_ctx.clone();
        let session_store = Arc::new(
            SessionStore::new(preload_ctx.workspace_root.clone())
                .expect("session store should build for temp workspace"),
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
                max_steps: 1,
                compaction: CompactionConfig::default(),
            },
        );

        Hub::new(
            runner,
            workspace.path().join("AGENTS.md"),
            preload_ctx,
            session_store,
            load_local_identity().expect("local WS identity should load"),
        )
    }

    /// Spawn one live WS server bound to an ephemeral local port.
    async fn spawn_test_server(handshake_timeout: std::time::Duration) -> TestServer {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let hub = build_test_hub(&workspace);
        let app = Router::new()
            .route(
                "/ws",
                get(
                    move |ws: WebSocketUpgrade, State(hub): State<Arc<Hub>>| async move {
                        ws.on_upgrade(move |socket| {
                            ws_session_with_timeout(socket, hub, handshake_timeout)
                        })
                    },
                ),
            )
            .with_state(hub);

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
        let server = spawn_test_server(std::time::Duration::from_millis(250)).await;
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
        let server = spawn_test_server(std::time::Duration::from_millis(75)).await;
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
        let server = spawn_test_server(std::time::Duration::from_millis(250)).await;
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
        let server = spawn_test_server(std::time::Duration::from_millis(250)).await;
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
        let server = spawn_test_server(std::time::Duration::from_millis(250)).await;
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
}
