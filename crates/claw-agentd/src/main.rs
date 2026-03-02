//! `claw-agentd` — the minimal Claw agent daemon.
//!
//! Responsibilities:
//! - Load `claw.toml` (TOML config).
//! - Read `Agents.md` and discover skills (`SKILL.md`).
//! - Run an autonomous agent loop in the background (OpenAI tool calling).
//! - Expose a **WebSocket** interface for a CLI frontend.
//! - Ensure that **client disconnects do not stop the agent**:
//!   - tasks are queued and processed independently of WS connections
//!   - events are buffered and can be replayed on reconnect

use anyhow::Context as _;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use clap::Parser;
use claw_core::agent::{AgentRunner, AgentRunnerConfig, EmitEventFn};
use claw_core::agents_md::load_agents_md;
use claw_core::config::load_config_from_file;
use claw_core::openai::OpenAiClient;
use claw_core::skills::SkillRegistry;
use claw_core::tools::{ToolContext, ToolExecutor};
use claw_core::ws_protocol::{ClientMessage, Event, EventKind, ServerMessage};
use futures_util::{SinkExt as _, StreamExt as _};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::{broadcast, mpsc};
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
    /// Path to `claw.toml`.
    #[arg(long, default_value = "claw.toml")]
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

/// Shared daemon state.
///
/// This object is designed so:
/// - The agent worker loop can run without any WS connection.
/// - WS connections can subscribe to events and submit tasks.
#[derive(Debug)]
struct Hub {
    /// Task queue sender (worker loop receives from this).
    task_tx: mpsc::Sender<TaskRequest>,

    /// Broadcast channel for events (WS sessions subscribe).
    events_tx: broadcast::Sender<Event>,

    /// In-memory event buffer for reconnect/history.
    events_buf: Mutex<VecDeque<Event>>,

    /// Monotonic event id generator.
    next_event_id: AtomicU64,

    /// Set of task ids already accepted (idempotency for retries).
    seen_tasks: Mutex<HashSet<Uuid>>,

    /// The agent runner (OpenAI + tools loop).
    runner: AgentRunner,

    /// Loaded Agents.md (instructions).
    agents_md: claw_core::agents_md::AgentsMd,
}

impl Hub {
    /// Create a new hub and spawn the background worker.
    fn new(runner: AgentRunner, agents_md: claw_core::agents_md::AgentsMd) -> Arc<Self> {
        // Task queue capacity (small but adequate for minimal agent).
        let (task_tx, task_rx) = mpsc::channel::<TaskRequest>(128);

        // Broadcast event channel. Capacity controls how many events a slow
        // subscriber can lag behind before it gets a `Lagged` error.
        let (events_tx, _) = broadcast::channel::<Event>(1024);

        // Build hub.
        let hub = Arc::new(Self {
            task_tx,
            events_tx,
            events_buf: Mutex::new(VecDeque::new()),
            next_event_id: AtomicU64::new(0),
            seen_tasks: Mutex::new(HashSet::new()),
            runner,
            agents_md,
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
            let mut buf = self
                .events_buf
                .lock()
                .expect("events_buf mutex poisoned");
            buf.push_back(event.clone());
            while buf.len() > MAX_BUFFERED_EVENTS {
                buf.pop_front();
            }
        }

        // Broadcast (ignore errors if no receivers).
        let _ = self.events_tx.send(event);
    }

    /// Return all buffered events with `event_id > from_event_id`.
    async fn history_since(&self, from_event_id: u64) -> Vec<Event> {
        let buf = self
            .events_buf
            .lock()
            .expect("events_buf mutex poisoned");
        buf.iter()
            .filter(|e| e.event_id > from_event_id)
            .cloned()
            .collect()
    }

    /// Submit a task to the queue (idempotent).
    async fn submit_task(&self, task_id: Uuid, task: String) -> anyhow::Result<()> {
        // Ensure idempotency: if we have already seen this task id, do not enqueue again.
        {
            let mut seen = self
                .seen_tasks
                .lock()
                .expect("seen_tasks mutex poisoned");
            if !seen.insert(task_id) {
                return Ok(());
            }
        }

        // Emit an event immediately so clients see that the task is queued.
        self.publish(EventKind::Log, task_id, "Task accepted and queued.".to_string());

        // Enqueue.
        self.task_tx
            .send(TaskRequest { task_id, task })
            .await
            .context("Failed to enqueue task")?;

        Ok(())
    }

    /// Background worker loop that executes tasks sequentially.
    async fn worker_loop(self: Arc<Self>, mut task_rx: mpsc::Receiver<TaskRequest>) {
        while let Some(req) = task_rx.recv().await {
            // Emit task start.
            self.publish(
                EventKind::Log,
                req.task_id,
                format!("Task started: {}", req.task),
            );

            // Build the emit callback for the agent runner.
            let hub_for_emit = Arc::clone(&self);
            let emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
                hub_for_emit.publish(kind, task_id, message);
            });

            // Run the agent loop.
            match self
                .runner
                .run_task(req.task_id, req.task, &self.agents_md, emit)
                .await
            {
                Ok(_final_text) => {
                    // The agent runner already emits `Final`, but we keep an explicit
                    // completion log for clarity.
                    self.publish(EventKind::Log, req.task_id, "Task finished.".to_string());
                }
                Err(err) => {
                    self.publish(
                        EventKind::Error,
                        req.task_id,
                        format!("Task crashed: {err}"),
                    );
                }
            }
        }
    }
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
                Ok(event) => {
                    let _ = out_tx_events.send(ServerMessage::Event { event });
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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
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

    // Load Agents.md.
    let agents_md = load_agents_md(agents_md_path).await?;

    // Discover skills.
    let skill_dirs = cfg.skills.dirs_as_paths();
    let skills = Arc::new(SkillRegistry::scan(&skill_dirs)?);
    tracing::info!("Discovered {} skill(s).", skills.list().len());

    // Build tools.
    let tool_ctx = ToolContext::new(workspace_root, Arc::clone(&skills))?;
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
    let hub = Hub::new(runner, agents_md);

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
