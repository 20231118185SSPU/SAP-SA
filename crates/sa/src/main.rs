//! `sa` — the StudyAdministrator (SA) backend agent daemon.
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
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use clap::Parser;
use futures_util::{SinkExt as _, StreamExt as _};
use sa_core::agent::{AgentRunner, AgentRunnerConfig, EmitEventFn};
use sa_core::agents_md::{extract_markdown_file_references, load_agents_md};
use sa_core::cancel::{CancelHandle, cancel_pair};
use sa_core::config::load_config_from_file;
use sa_core::memory::{MemoryEntry, MemoryStore, default_memory_path};
use sa_core::openai::OpenAiClient;
use sa_core::skills::SkillRegistry;
use sa_core::tools::{
    AskQuestionFn, AskRequest, MAX_SUBAGENT_DEPTH, RunSubAgentFn, SendMessageFn, SubAgentRequest,
    ToolContext, ToolExecutor, ToolRuntime,
};
use sa_core::ws_protocol::{
    ClientMessage, Event, EventKind, QuestionMode, ServerMessage, UserQuestion, UserQuestionAnswer,
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

    /// Set of task ids already accepted (idempotency for retries).
    seen_tasks: Mutex<HashSet<Uuid>>,

    /// Currently running task (if any), so we can cancel it.
    current_task: Mutex<Option<CurrentTask>>,

    /// Questions waiting for a user answer.
    pending_questions: Mutex<Vec<PendingQuestionEntry>>,

    /// Context used for safe path resolution when preloading files.
    preload_ctx: ToolContext,

    /// Persistent long-term memory (JSONL on disk).
    memory: Mutex<MemoryStore>,

    /// The agent runner (OpenAI + tools loop).
    runner: AgentRunner,

    /// Path to Agents.md (reloaded for every task).
    ///
    /// Why reload each task?
    /// - Users often edit `Agents.md` while the daemon is running.
    /// - Users may create `Agents.md` after the daemon starts.
    /// Reloading makes this behavior verifiable and fixes "Agents.md not loaded" confusion.
    agents_md_path: PathBuf,
}

impl Hub {
    /// Create a new hub and spawn the background worker.
    fn new(
        runner: AgentRunner,
        agents_md_path: PathBuf,
        preload_ctx: ToolContext,
        memory: MemoryStore,
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
            preload_ctx,
            memory: Mutex::new(memory),
            runner,
            agents_md_path,
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
        let _ = self.events_tx.send(msg);
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

    /// Submit a task to the queue (idempotent).
    async fn submit_task(&self, task_id: Uuid, task: String) -> anyhow::Result<()> {
        // Ensure idempotency: if we have already seen this task id, do not enqueue again.
        {
            let mut seen = self.seen_tasks.lock().expect("seen_tasks mutex poisoned");
            if !seen.insert(task_id) {
                return Ok(());
            }
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

        let hub_for_subagent = Arc::clone(self);
        let run_subagent: RunSubAgentFn = Arc::new(move |request: SubAgentRequest, cancel| {
            let hub = Arc::clone(&hub_for_subagent);
            Box::pin(async move { hub.run_subagent(task_id, depth, request, cancel).await })
        });

        ToolRuntime::new(send_message, ask_question, run_subagent)
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

        let memory_block = {
            let mem = self.memory.lock().expect("memory mutex poisoned");
            mem.prompt_block()
        };
        let preload_block = self.preload_agents_md_references(&agents_md).await;
        let extra_prompt = format!(
            "{memory_block}\n\n{preload_block}\n\n## Parent-provided SubAgent Context\n\n- depth: {depth}\n- label: {label}\n\n```text\n{}\n```",
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
                runtime,
                &cancel,
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
            {
                let mut current = self
                    .current_task
                    .lock()
                    .expect("current_task mutex poisoned");
                *current = Some(CurrentTask {
                    task_id: req.task_id,
                    cancel: cancel_handle,
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
            // - persistent memory snapshot
            // - proactive preloading of files referenced in Agents.md
            let memory_block = {
                let mem = self.memory.lock().expect("memory mutex poisoned");
                mem.prompt_block()
            };
            let preload_block = self.preload_agents_md_references(&agents_md).await;
            let extra_prompt = format!("{memory_block}\n\n{preload_block}");

            // Build the emit callback for the agent runner.
            let hub_for_emit = Arc::clone(&self);
            let emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
                hub_for_emit.publish(kind, task_id, message);
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
                    runtime,
                    &cancel_token,
                    emit,
                )
                .await
            {
                Ok(final_text) => {
                    // The agent runner already emits `Final`, but we keep an explicit
                    // completion log for clarity.
                    self.publish(EventKind::Log, req.task_id, "Task finished.".to_string());

                    // Persist memory entry (best-effort).
                    //
                    // We skip persisting on cancellation to keep memory signal high.
                    if !cancel_token.is_cancelled()
                        && !final_text.starts_with("Task cancelled by user interrupt")
                    {
                        let entry = MemoryEntry {
                            ts: chrono::Utc::now(),
                            task_id: req.task_id,
                            user_task: req.task.trim().to_string(),
                            final_answer: final_text.trim().to_string(),
                        };

                        let mut mem = self.memory.lock().expect("memory mutex poisoned");
                        if let Err(err) = mem.append(entry) {
                            self.publish(
                                EventKind::Error,
                                req.task_id,
                                format!(
                                    "Failed to persist memory to {}: {err}",
                                    mem.path().display()
                                ),
                            );
                        }
                    }
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
/// - `YYYY-MM-DD` → replaced with today's date, and also yesterday's date
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

/// WS upgrade handler.
async fn ws_route(ws: WebSocketUpgrade, State(hub): State<Arc<Hub>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_session(socket, hub))
}

/// Handle a single WS connection.
async fn ws_session(socket: WebSocket, hub: Arc<Hub>) {
    // We split the socket so we can read and write concurrently.
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Outbound message channel (single writer task owns `ws_tx`).
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<ServerMessage>();

    // Immediately send any questions that were already pending before this
    // client connected. This makes reconnecting clients able to continue an
    // interrupted `Ask` interaction.
    let _ = out_tx.send(ServerMessage::PendingQuestions {
        questions: hub.pending_questions_snapshot(),
    });

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
    let forwarder = tokio::spawn(async move {
        loop {
            match events_rx.recv().await {
                Ok(msg) => {
                    let _ = out_tx_events.send(msg);
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    let _ = out_tx_events.send(ServerMessage::Error {
                        message: format!("Lagged in event stream; skipped {skipped} events"),
                    });
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
                        let _ = out_tx.send(ServerMessage::Error {
                            message: format!("Invalid JSON: {err}"),
                        });
                        continue;
                    }
                };

                match msg {
                    ClientMessage::Submit { task_id, task } => {
                        // If the client didn't provide an id, we generate one.
                        let task_id = task_id.unwrap_or_else(Uuid::new_v4);

                        // Acknowledge immediately.
                        let _ = out_tx.send(ServerMessage::Accepted { task_id });

                        // Enqueue (idempotent).
                        if let Err(err) = hub.submit_task(task_id, task).await {
                            let _ = out_tx.send(ServerMessage::Error {
                                message: format!("Failed to submit task: {err}"),
                            });
                        }
                    }
                    ClientMessage::GetHistory { from_event_id } => {
                        let events = hub.history_since(from_event_id).await;
                        let _ = out_tx.send(ServerMessage::History { events });
                    }
                    ClientMessage::Interrupt { task_id } => {
                        // Cancellation is implemented in the worker loop. Here we only
                        // forward the request into the hub.
                        //
                        // NOTE: If the task is not currently running, the hub will ignore it.
                        hub.request_interrupt(task_id);
                    }
                    ClientMessage::AnswerQuestion { answer } => {
                        if let Err(err) = hub.answer_question(answer) {
                            let _ = out_tx.send(ServerMessage::Error {
                                message: format!("Failed to answer question: {err}"),
                            });
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

    // Build tools.
    let tool_ctx = ToolContext::new(workspace_root, Arc::clone(&skills))?;
    let preload_ctx = tool_ctx.clone();

    // Load persistent memory store (JSONL).
    //
    // This provides "long-term memory" across tasks and daemon restarts.
    let memory_path = default_memory_path(&tool_ctx.workspace_root);
    let memory_store = MemoryStore::load_or_new(memory_path)?;
    tracing::info!("Memory file: {}", memory_store.path().display());

    let tools = ToolExecutor::new(tool_ctx);

    // Build LLM client.
    //
    // IMPORTANT: compute `system_role_name` before we move strings out of `cfg.llm`.
    let system_role_name = cfg.llm.effective_system_role_name().to_string();
    let llm = OpenAiClient::new(cfg.llm.base_url, cfg.llm.api_key)?;

    // Build agent runner.
    let runner_cfg = AgentRunnerConfig {
        model: cfg.llm.model,
        system_role_name,
        max_steps: cfg.llm.max_steps,
    };
    let runner = AgentRunner::new(llm, tools, Arc::clone(&skills), runner_cfg);

    // Hub (spawns worker loop).
    let hub = Hub::new(runner, agents_md_path, preload_ctx, memory_store);

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
