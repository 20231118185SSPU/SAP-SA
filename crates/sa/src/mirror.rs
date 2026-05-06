//! Event mirroring to backend terminal for diagnostics.

use sa_core::ws_protocol::{Event, ServerMessage, UserVisibleFile, UserVisibleFileEncoding};

/// Mirror one event in a stable multi-line format.
pub(crate) fn mirror_event_line(prefix: &str, event: &Event) {
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
pub(crate) fn mirror_shown_file(prefix: &str, file: &UserVisibleFile) {
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
pub(crate) fn mirror_server_message(msg: &ServerMessage) {
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
        ServerMessage::MemoryFacts { total, offset, .. } => {
            eprintln!("[frontend][memory_facts] total={total} offset={offset}");
        }
        ServerMessage::MemoryGraph { subject, hops, .. } => {
            eprintln!("[frontend][memory_graph] subject={subject} hops={hops}");
        }
        ServerMessage::MemoryStats { total_facts, .. } => {
            eprintln!("[frontend][memory_stats] total={total_facts}");
        }
        ServerMessage::MemoryOperationResult { operation, affected, summary } => {
            eprintln!("[frontend][memory_{operation}] affected={affected} {summary}");
        }
        // ── Memory protocol stubs (frontend → backend) ───────────────────────
        ServerMessage::MemoryStats { total_facts, .. } => {
            eprintln!("[frontend][memory_stats] total={total_facts}");
        }
        ServerMessage::MemoryFacts { total, offset, .. } => {
            eprintln!("[frontend][memory_facts] total={total} offset={offset}");
        }
        ServerMessage::MemoryGraph { subject, hops, .. } => {
            eprintln!("[frontend][memory_graph] subject={subject} hops={hops}");
        }
        ServerMessage::ContextInfo { used_tokens, max_tokens, message_count } => {
            eprintln!(
                "[frontend][context_info] used={}K max={}K messages={}",
                used_tokens / 1000, max_tokens / 1000, message_count
            );
        }
    }
}        ServerMessage::ContextInfo { used_tokens, max_tokens, message_count } => {
            eprintln!(
                "[frontend][context_info] used={}K max={}K messages={}",
                used_tokens / 1000, max_tokens / 1000, message_count
            );
        }
        ServerMessage::ApiKeyTestResult { ok, model, provider, latency_ms, error } => {
            eprintln!("[frontend][api_key_test_result] ok={ok} model={model} provider={provider} latency={latency_ms}ms error={:?}", error);
        }
    }
}

/// Send one direct per-connection message and mirror it to the backend
/// terminal.
pub(crate) fn send_direct_server_message(out_tx: &tokio::sync::mpsc::UnboundedSender<ServerMessage>, msg: ServerMessage) {
    mirror_server_message(&msg);
    let _ = out_tx.send(msg);
}
