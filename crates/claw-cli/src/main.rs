//! `claw-cli` — command line frontend for `claw-agentd`.
//!
//! Design goals (from the request):
//! - Communicate with the daemon via WebSocket.
//! - If communication is interrupted, the agent keeps running.
//! - The CLI can reconnect and continue following events (history replay).
//!
//! This CLI is intentionally minimal:
//! - `run "<task>"` submits a single task and prints events until `final`.

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use claw_core::config::load_config_from_file;
use claw_core::ws_protocol::{ClientMessage, EventKind, ServerMessage};
use futures_util::{SinkExt as _, StreamExt as _};
use std::path::PathBuf;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

/// CLI arguments.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Path to `claw.toml`.
    #[arg(long, default_value = "claw.toml")]
    config: PathBuf,

    /// Override WebSocket URL (example: `ws://127.0.0.1:8765/ws`).
    #[arg(long)]
    ws: Option<String>,

    /// Command to execute.
    #[command(subcommand)]
    cmd: Command,
}

/// Subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Submit a task and follow its events until completion.
    Run { task: String },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load config so we can derive the default WS URL.
    let cfg = load_config_from_file(&args.config)?;

    // Determine WS URL.
    let ws_url = args.ws.unwrap_or_else(|| default_ws_url(&cfg.server.bind, &cfg.server.ws_path));
    // Validate URL early so users get a good error message.
    let _ = url::Url::parse(&ws_url).context("Invalid ws URL")?;

    match args.cmd {
        Command::Run { task } => run_task(ws_url, task).await?,
    }

    Ok(())
}

/// Build the default WebSocket URL from the daemon bind + path.
fn default_ws_url(bind: &str, ws_path: &str) -> String {
    // Most users bind the daemon to `0.0.0.0:PORT` so other devices can connect.
    // For a local CLI, `127.0.0.1` is usually correct. We rewrite it here so
    // `cargo run -p claw-cli` works out of the box.
    let host = if bind.starts_with("0.0.0.0:") {
        bind.replacen("0.0.0.0", "127.0.0.1", 1)
    } else {
        bind.to_string()
    };

    // Ensure the WS path starts with `/`.
    let path = if ws_path.starts_with('/') {
        ws_path.to_string()
    } else {
        format!("/{ws_path}")
    };

    format!("ws://{host}{path}")
}

/// Submit a task and follow until `final`.
async fn run_task(ws_url: String, task: String) -> anyhow::Result<()> {
    // Client-generated id so reconnect retries are idempotent.
    let task_id = Uuid::new_v4();

    // Last seen event id (global). Used for history replay on reconnect.
    let mut last_event_id: u64 = 0;

    // Completion flag.
    let mut done = false;

    eprintln!("Connecting to {ws_url} ...");
    eprintln!("Task id: {task_id}");

    // Reconnect loop.
    while !done {
        let connect_res = tokio_tungstenite::connect_async(ws_url.clone()).await;
        let (ws_stream, _resp) = match connect_res {
            Ok(ok) => ok,
            Err(err) => {
                eprintln!("Connect failed: {err}; retrying in 1s");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // Always (re)submit the task; the daemon deduplicates by task_id.
        let submit = ClientMessage::Submit {
            task_id: Some(task_id),
            task: task.clone(),
        };
        ws_tx
            .send(Message::Text(serde_json::to_string(&submit)?.into()))
            .await
            .context("Failed to send submit")?;

        // On reconnect, request history since last_event_id.
        if last_event_id > 0 {
            let hist = ClientMessage::GetHistory {
                from_event_id: last_event_id,
            };
            ws_tx
                .send(Message::Text(serde_json::to_string(&hist)?.into()))
                .await
                .context("Failed to request history")?;
        }

        // Read loop.
        while let Some(msg) = ws_rx.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(err) => {
                    eprintln!("WebSocket error: {err}");
                    break;
                }
            };

            let Message::Text(text) = msg else {
                // Ignore non-text frames in this minimal client.
                continue;
            };

            let parsed = serde_json::from_str::<ServerMessage>(&text);
            let msg = match parsed {
                Ok(m) => m,
                Err(err) => {
                    eprintln!("Protocol error (invalid JSON): {err}");
                    continue;
                }
            };

            match msg {
                ServerMessage::Accepted { task_id: accepted } => {
                    // The daemon echoes the task id; useful for debugging.
                    if accepted == task_id {
                        eprintln!("Task accepted.");
                    } else {
                        eprintln!("Server accepted a different task id: {accepted}");
                    }
                }
                ServerMessage::History { events } => {
                    // Replay history in order.
                    for e in events {
                        last_event_id = last_event_id.max(e.event_id);
                        if e.task_id != task_id {
                            continue;
                        }
                        print_event(&e);
                        if matches!(e.kind, EventKind::Final) {
                            done = true;
                        }
                    }
                    if done {
                        break;
                    }
                }
                ServerMessage::Event { event } => {
                    last_event_id = last_event_id.max(event.event_id);
                    if event.task_id != task_id {
                        continue;
                    }
                    print_event(&event);
                    if matches!(event.kind, EventKind::Final) {
                        done = true;
                        break;
                    }
                }
                ServerMessage::Error { message } => {
                    eprintln!("Server error: {message}");
                }
            }
        }

        // If not done, reconnect.
        if !done {
            eprintln!("Disconnected; reconnecting in 1s...");
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    Ok(())
}

/// Render one event to stdout.
fn print_event(e: &claw_core::ws_protocol::Event) {
    // Minimal formatting:
    // - timestamp
    // - kind
    // - message
    println!("[{}][{:?}] {}", e.ts, e.kind, e.message);
}
