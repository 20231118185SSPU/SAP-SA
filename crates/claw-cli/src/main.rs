//! `claw-cli` — a terminal UI (TUI) frontend for `claw-agentd`.
//!
//! User requirements (summarized):
//! - The CLI should have a bottom input box (like a chat).
//! - Users can send messages at any time to interrupt the running task.
//! - Disconnections must not stop the agent; the CLI should reconnect forever.
//!
//! Practical design:
//! - The agent daemon is the source of truth: it buffers events and continues
//!   running even if the CLI disconnects.
//! - The CLI is a "view + input" layer:
//!   - input → WS (`submit`, `interrupt`)
//!   - output ← WS events (with history replay on reconnect)

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use claw_core::config::load_config_from_file;
use claw_core::retry::retry_delay;
use claw_core::ws_protocol::{ClientMessage, Event, EventKind, ServerMessage};
use crossterm::cursor;
use crossterm::event::{Event as CtEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::{SinkExt as _, StreamExt as _};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
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
    ///
    /// If omitted, we start the interactive UI.
    #[command(subcommand)]
    cmd: Option<Command>,
}

/// Subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Interactive terminal UI (bottom input box).
    Ui,

    /// One-shot: submit a task and print events until `final`.
    Run { task: String },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load config so we can derive the default WS URL.
    let cfg = load_config_from_file(&args.config)?;

    // Determine WS URL.
    let ws_url = args
        .ws
        .unwrap_or_else(|| default_ws_url(&cfg.server.bind, &cfg.server.ws_path));

    // Validate URL early so users get a good error message.
    let _ = url::Url::parse(&ws_url).context("Invalid ws URL")?;

    match args.cmd.unwrap_or(Command::Ui) {
        Command::Ui => run_ui(ws_url).await?,
        Command::Run { task } => run_one_shot(ws_url, task).await?,
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

/// Interactive TUI:
/// - logs in the top panel
/// - input at the bottom
/// - Enter submits a task (interrupting the previous one, if any)
/// - `/stop` interrupts without submitting a new task
/// - `/exit` quits
async fn run_ui(ws_url: String) -> anyhow::Result<()> {
    // ── Terminal setup ────────────────────────────────────────────────────────
    enable_raw_mode().context("Failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, cursor::Hide).context("Failed to init terminal")?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("Failed to create terminal backend")?;
    terminal.clear().ok();

    // When this flag becomes false, the input thread exits.
    let running = Arc::new(AtomicBool::new(true));

    // ── WebSocket worker channels ─────────────────────────────────────────────
    // Outbound: UI → WS
    let (ws_out_tx, ws_out_rx) = mpsc::unbounded_channel::<ClientMessage>();
    // Inbound: WS → UI
    let (ws_in_tx, mut ws_in_rx) = mpsc::unbounded_channel::<ServerMessage>();
    // Status: WS worker → UI (human-readable)
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<String>();

    // Start background WS worker (reconnects forever with requested backoff).
    tokio::spawn(ws_worker(ws_url.clone(), ws_out_rx, ws_in_tx, status_tx));

    // ── Input reader (blocking) ───────────────────────────────────────────────
    // We read crossterm events in a blocking thread and forward them to the async loop.
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<CtEvent>();
    let running_for_thread = Arc::clone(&running);
    let input_thread = tokio::task::spawn_blocking(move || {
        while running_for_thread.load(Ordering::Relaxed) {
            // Use poll with timeout so we can exit quickly when `running=false`.
            if crossterm::event::poll(Duration::from_millis(200)).unwrap_or(false) {
                if let Ok(ev) = crossterm::event::read() {
                    let _ = input_tx.send(ev);
                }
            }
        }
    });

    // ── UI state ──────────────────────────────────────────────────────────────
    // Collected log lines displayed in the upper panel.
    let mut log_lines: VecDeque<String> = VecDeque::new();
    let max_log_lines: usize = 3_000;

    // Current input line (bottom panel).
    let mut input = String::new();

    // Human-readable connection status (updated by the WS worker).
    let mut status = "disconnected".to_string();

    // Track the "current running task id" so we can interrupt it on new input.
    let mut current_task_id: Option<Uuid> = None;

    // UI tick (controls redraw rate).
    let mut tick = tokio::time::interval(Duration::from_millis(50));

    // ── Main loop ─────────────────────────────────────────────────────────────
    let mut should_exit = false;
    while !should_exit {
        tokio::select! {
            _ = tick.tick() => {
                draw_ui(&mut terminal, &log_lines, &input, &status, current_task_id)?;
            }
            Some(s) = status_rx.recv() => {
                status = s;
                push_log_line(&mut log_lines, max_log_lines, format!("[status] {status}"));
            }
            Some(msg) = ws_in_rx.recv() => {
                handle_server_message(&mut log_lines, max_log_lines, &mut current_task_id, msg);
            }
            Some(ev) = input_rx.recv() => {
                if let Some(action) = handle_input_event(&mut input, ev) {
                    match action {
                        UiAction::Exit => {
                            should_exit = true;
                        }
                        UiAction::Interrupt => {
                            if let Some(task_id) = current_task_id {
                                let _ = ws_out_tx.send(ClientMessage::Interrupt { task_id });
                                push_log_line(&mut log_lines, max_log_lines, format!("[local] interrupt task={task_id}"));
                            } else {
                                push_log_line(&mut log_lines, max_log_lines, "[local] no running task to interrupt".to_string());
                            }
                        }
                        UiAction::Submit(text) => {
                            // New message semantics:
                            // - If a task is running, interrupt it first.
                            // - Then submit a new task.
                            if let Some(task_id) = current_task_id {
                                let _ = ws_out_tx.send(ClientMessage::Interrupt { task_id });
                                push_log_line(&mut log_lines, max_log_lines, format!("[local] interrupt task={task_id}"));
                            }

                            let new_task_id = Uuid::new_v4();
                            current_task_id = Some(new_task_id);
                            let _ = ws_out_tx.send(ClientMessage::Submit {
                                task_id: Some(new_task_id),
                                task: text.clone(),
                            });
                            push_log_line(&mut log_lines, max_log_lines, format!("[you] {text}"));
                            push_log_line(&mut log_lines, max_log_lines, format!("[local] submitted task={new_task_id}"));
                        }
                    }
                }
            }
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    running.store(false, Ordering::Relaxed);
    input_thread.abort();

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen, cursor::Show).ok();
    terminal.show_cursor().ok();

    Ok(())
}

/// One-shot mode: submit one task and exit after `final`.
async fn run_one_shot(ws_url: String, task: String) -> anyhow::Result<()> {
    // Client-generated id so reconnect retries are idempotent.
    let task_id = Uuid::new_v4();

    // Last seen event id (global). Used for history replay on reconnect.
    let mut last_event_id: u64 = 0;

    // Completion flag.
    let mut done = false;

    eprintln!("Connecting to {ws_url} ...");
    eprintln!("Task id: {task_id}");

    // Infinite reconnect loop (requested backoff schedule).
    let mut connect_errors: u32 = 0;
    while !done {
        let connect_res = tokio_tungstenite::connect_async(ws_url.clone()).await;
        let (ws_stream, _resp) = match connect_res {
            Ok(ok) => {
                connect_errors = 0;
                ok
            }
            Err(err) => {
                connect_errors = connect_errors.saturating_add(1);
                let delay = retry_delay(connect_errors);
                eprintln!("Connect failed: {err}; retrying in {delay:?}");
                tokio::time::sleep(delay).await;
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

        // Always request history since `last_event_id`.
        //
        // Why do this even when `last_event_id == 0`?
        // - If the connection drops right after we submit the task but before we
        //   see the first event, `last_event_id` is still 0.
        // - Without requesting history, we could miss the `final` event and
        //   hang forever (the agent will finish, but the CLI never observes it).
        //
        // Requesting history from 0 is safe because:
        // - The daemon's buffer is bounded (`MAX_BUFFERED_EVENTS`).
        // - We filter by `task_id` before printing.
        let hist = ClientMessage::GetHistory {
            from_event_id: last_event_id,
        };
        ws_tx
            .send(Message::Text(serde_json::to_string(&hist)?.into()))
            .await
            .context("Failed to request history")?;

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
    }

    Ok(())
}

/// Actions produced by interpreting keyboard input.
#[derive(Debug)]
enum UiAction {
    /// Exit the UI.
    Exit,
    /// Interrupt the current task.
    Interrupt,
    /// Submit a new task.
    Submit(String),
}

/// Convert a single crossterm event into UI actions and/or input mutations.
///
/// This function is intentionally small and deterministic so it is easy to test
/// mentally and extend safely.
fn handle_input_event(input: &mut String, ev: CtEvent) -> Option<UiAction> {
    let CtEvent::Key(KeyEvent {
        code,
        modifiers,
        kind,
        ..
    }) = ev
    else {
        return None;
    };

    // IMPORTANT: only react to key presses (and repeats), not key releases.
    //
    // Why?
    // - On some terminals / platforms (notably Windows), crossterm can emit
    //   both `Press` and `Release` events.
    // - If we treat `Release` like `Press`, every typed character appears twice
    //   in the input box (and Enter can double-trigger).
    //
    // Filtering here keeps behavior deterministic and fixes the user's report:
    // "TUI 输入时会输出两次重复" (input shows duplicated characters).
    if !matches!(kind, KeyEventKind::Press | KeyEventKind::Repeat) {
        return None;
    }

    // Exit shortcuts.
    if modifiers.contains(KeyModifiers::CONTROL) {
        match code {
            KeyCode::Char('c') => return Some(UiAction::Exit),
            KeyCode::Char('d') => return Some(UiAction::Exit),
            _ => {}
        }
    }

    match code {
        KeyCode::Esc => Some(UiAction::Exit),
        KeyCode::Backspace => {
            input.pop();
            None
        }
        KeyCode::Enter => {
            let text = input.trim().to_string();
            input.clear();

            if text.is_empty() {
                return None;
            }

            match text.as_str() {
                "/exit" | "/quit" => Some(UiAction::Exit),
                "/stop" | "/interrupt" => Some(UiAction::Interrupt),
                _ => Some(UiAction::Submit(text)),
            }
        }
        KeyCode::Char(c) => {
            // Basic text input (no fancy cursor movement for the minimal UI).
            input.push(c);
            None
        }
        _ => None,
    }
}

/// Draw the TUI.
fn draw_ui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    log_lines: &VecDeque<String>,
    input: &str,
    status: &str,
    current_task_id: Option<Uuid>,
) -> anyhow::Result<()> {
    terminal
        .draw(|f| {
            let size = f.area();

            // Layout:
            // - top: logs
            // - bottom: input box
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(size);

            // Determine how many log lines fit.
            let available_lines = chunks[0].height.saturating_sub(2) as usize;
            let start = log_lines.len().saturating_sub(available_lines);
            let log_text = log_lines
                .iter()
                .skip(start)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");

            let title = match current_task_id {
                Some(id) => format!("Logs | status={status} | task={id}"),
                None => format!("Logs | status={status} | task=<none>"),
            };

            let log_widget = Paragraph::new(log_text)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(title));
            f.render_widget(log_widget, chunks[0]);

            // Input widget.
            let input_widget = Paragraph::new(format!("> {input}"))
                .wrap(Wrap { trim: false })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Input | Enter=send | /stop | /exit"),
                );
            f.render_widget(input_widget, chunks[1]);

            // Place the cursor at the end of the input.
            let cursor_x = chunks[1].x.saturating_add(2 + input.len() as u16);
            let cursor_y = chunks[1].y + 1;
            f.set_cursor_position((cursor_x, cursor_y));
        })
        .context("Failed to draw UI")?;

    Ok(())
}

/// Handle a server message and update UI state/logs.
fn handle_server_message(
    log_lines: &mut VecDeque<String>,
    max_log_lines: usize,
    current_task_id: &mut Option<Uuid>,
    msg: ServerMessage,
) {
    match msg {
        ServerMessage::Accepted { task_id } => {
            push_log_line(
                log_lines,
                max_log_lines,
                format!("[server] accepted task={task_id}"),
            );
        }
        ServerMessage::History { events } => {
            push_log_line(
                log_lines,
                max_log_lines,
                format!("[server] history events={}", events.len()),
            );
            for e in events {
                for line in format_event_lines(&e) {
                    push_log_line(log_lines, max_log_lines, line);
                }
                if matches!(e.kind, EventKind::Final) && Some(e.task_id) == *current_task_id {
                    *current_task_id = None;
                }
            }
        }
        ServerMessage::Event { event } => {
            for line in format_event_lines(&event) {
                push_log_line(log_lines, max_log_lines, line);
            }
            if matches!(event.kind, EventKind::Final) && Some(event.task_id) == *current_task_id {
                *current_task_id = None;
            }
        }
        ServerMessage::Error { message } => {
            push_log_line(
                log_lines,
                max_log_lines,
                format!("[server][error] {message}"),
            );
        }
    }
}

/// Format an agent event as one or more log lines.
fn format_event_lines(e: &Event) -> Vec<String> {
    // We keep the prefix stable so reconnect replay looks the same as live events.
    let prefix = format!("[{}][{:?}][task={}] ", e.ts, e.kind, e.task_id);

    // Some messages are multi-line (tool output, etc.). Split them so the UI
    // doesn't wrap a giant paragraph unpredictably.
    let mut out = Vec::new();
    for (idx, line) in e.message.lines().enumerate() {
        if idx == 0 {
            out.push(format!("{prefix}{line}"));
        } else {
            out.push(format!("{}{}", " ".repeat(prefix.len()), line));
        }
    }

    if out.is_empty() {
        out.push(prefix);
    }

    out
}

/// Append a log line, keeping the buffer bounded.
fn push_log_line(buf: &mut VecDeque<String>, max: usize, line: String) {
    buf.push_back(line);
    while buf.len() > max {
        buf.pop_front();
    }
}

/// WebSocket worker:
/// - reconnects forever with the requested backoff schedule
/// - requests event history on reconnect
/// - forwards events to the UI
async fn ws_worker(
    ws_url: String,
    mut outbound_rx: mpsc::UnboundedReceiver<ClientMessage>,
    inbound_tx: mpsc::UnboundedSender<ServerMessage>,
    status_tx: mpsc::UnboundedSender<String>,
) {
    // Monotonic event cursor for history replay.
    let mut last_event_id: u64 = 0;

    // Connection failure counter (drives backoff).
    let mut connect_errors: u32 = 0;

    loop {
        let _ = status_tx.send(format!("connecting to {ws_url}"));

        let connect_res = tokio_tungstenite::connect_async(ws_url.clone()).await;
        let (ws_stream, _resp) = match connect_res {
            Ok(ok) => {
                connect_errors = 0;
                let _ = status_tx.send("connected".to_string());
                ok
            }
            Err(err) => {
                connect_errors = connect_errors.saturating_add(1);
                let delay = retry_delay(connect_errors);
                let _ = status_tx.send(format!(
                    "connect failed (count={connect_errors}); retry in {delay:?}: {err}"
                ));
                tokio::time::sleep(delay).await;
                continue;
            }
        };

        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // Request history first so we catch up on events while we were offline.
        //
        // We do this even for `last_event_id == 0` so a fresh UI session can
        // immediately display recent buffered events.
        let hist = ClientMessage::GetHistory {
            from_event_id: last_event_id,
        };
        let _ = ws_tx
            .send(Message::Text(serde_json::to_string(&hist).unwrap().into()))
            .await;

        // Connected loop:
        // - forward outbound messages to server
        // - forward inbound messages to UI
        let mut disconnected = false;
        while !disconnected {
            tokio::select! {
                outbound = outbound_rx.recv() => {
                    let Some(outbound) = outbound else {
                        // UI dropped sender; nothing more to do.
                        return;
                    };
                    let text = match serde_json::to_string(&outbound) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    if ws_tx.send(Message::Text(text.into())).await.is_err() {
                        disconnected = true;
                    }
                }
                inbound = ws_rx.next() => {
                    let Some(inbound) = inbound else {
                        disconnected = true;
                        continue;
                    };

                    let inbound = match inbound {
                        Ok(m) => m,
                        Err(_) => {
                            disconnected = true;
                            continue;
                        }
                    };

                    let Message::Text(text) = inbound else {
                        continue;
                    };

                    let parsed = match serde_json::from_str::<ServerMessage>(&text) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };

                    // Update cursor for history replay.
                    match &parsed {
                        ServerMessage::History { events } => {
                            for e in events {
                                last_event_id = last_event_id.max(e.event_id);
                            }
                        }
                        ServerMessage::Event { event } => {
                            last_event_id = last_event_id.max(event.event_id);
                        }
                        _ => {}
                    }

                    let _ = inbound_tx.send(parsed);
                }
            }
        }

        let _ = status_tx.send("disconnected".to_string());
        // Loop continues to reconnect forever.
    }
}

/// Render one event to stdout (used by one-shot mode).
fn print_event(e: &Event) {
    println!("[{}][{:?}] {}", e.ts, e.kind, e.message);
}
