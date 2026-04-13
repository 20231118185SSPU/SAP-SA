//! Minimal autonomous agent loop (tool-calling).
//!
//! This is the "heart" of the StudyAdministrator (SA) agent extracted from the large
//! `zeroclaw` codebase:
//! - We call an OpenAI-compatible Chat Completions endpoint.
//! - We provide tool definitions so the model can request actions.
//! - We execute those tools and feed results back to the model.
//! - We repeat until the model produces a final answer (no tool calls) or
//!   until `max_steps` is reached.
//!
//! The backend daemon (`sa`) owns task queues, event IDs, and WS connections.
//! This module is deliberately "pure core": it only needs an event callback.

use crate::agents_md::{AgentsMd, format_agents_md_block};
use crate::cancel::CancelToken;
use crate::compact::{
    CompactionConfig, CompactionState, build_request_messages, maybe_compact_history,
};
use crate::openai::{ChatCompletionsRequest, ChatMessage, OpenAiClient, ToolCall};
use crate::retry::retry_delay;
use crate::session::SessionStore;
use crate::skills::SkillRegistry;
use crate::tools::{ToolExecutor, ToolRuntime, ToolSession};
use crate::ws_protocol::EventKind;
use anyhow::Context as _;
use std::sync::Arc;
use uuid::Uuid;

/// Static prompt source-of-truth loaded from the repository-level `prompt.md`.
///
/// Keeping the large instruction block in a dedicated Markdown file prevents
/// prompt drift between runtime behavior and the human-editable prompt
/// document.
const STATIC_PROMPT_TEMPLATE: &str = include_str!("../../../prompt.md");

/// Callback used by the agent to emit progress events.
///
/// The daemon will wrap these events with event IDs and broadcast them to clients.
pub type EmitEventFn = Arc<dyn Fn(EventKind, Uuid, String) + Send + Sync>;

/// Callback used by the backend to drain follow-up user messages that arrived
/// while the current task was still running.
///
/// Why a callback instead of directly sharing backend state here?
/// - `sa-core` stays independent from the daemon implementation details.
/// - The daemon decides where queued input lives.
/// - The agent loop only needs a simple "give me any pending user turns now"
///   primitive before it issues the next model request.
pub type DrainQueuedUserMessagesFn = Arc<dyn Fn() -> Vec<String> + Send + Sync>;

/// Configuration for a single `AgentRunner`.
#[derive(Debug, Clone)]
pub struct AgentRunnerConfig {
    /// Model name.
    pub model: String,
    /// The role name used for "system instructions" (`system` or `developer`).
    pub system_role_name: String,
    /// Optional reasoning depth / effort forwarded to compatible GPT models.
    pub reasoning_effort: Option<String>,
    /// Maximum tool-call steps per task.
    pub max_steps: u32,
    /// History compaction behavior for long-running tasks.
    pub compaction: CompactionConfig,
}

/// The runnable agent.
#[derive(Debug, Clone)]
pub struct AgentRunner {
    /// OpenAI-compatible HTTP client.
    llm: OpenAiClient,
    /// Tool executor.
    tools: ToolExecutor,
    /// Skills registry (for metadata injection).
    skills: Arc<SkillRegistry>,
    /// Runner config.
    cfg: AgentRunnerConfig,
}

impl AgentRunner {
    /// Create a new runner.
    pub fn new(
        llm: OpenAiClient,
        tools: ToolExecutor,
        skills: Arc<SkillRegistry>,
        cfg: AgentRunnerConfig,
    ) -> Self {
        Self {
            llm,
            tools,
            skills,
            cfg,
        }
    }

    /// Run one task until completion.
    ///
    /// Returns the final assistant message content (best-effort).
    pub async fn run_task(
        &self,
        task_id: Uuid,
        task: String,
        agents_md: &AgentsMd,
        extra_system_prompt: Option<&str>,
        persistent_session: Option<Arc<SessionStore>>,
        runtime: ToolRuntime,
        cancel: &CancelToken,
        drain_queued_user_messages: Option<DrainQueuedUserMessagesFn>,
        emit: EmitEventFn,
    ) -> anyhow::Result<String> {
        self.run_task_inner(
            task_id,
            task,
            agents_md,
            extra_system_prompt,
            persistent_session,
            runtime,
            cancel,
            drain_queued_user_messages,
            emit,
            true,
        )
        .await
    }

    /// Internal implementation shared by the normal task runner and the
    /// compact-triggered memory refresh sub-agent.
    async fn run_task_inner(
        &self,
        task_id: Uuid,
        task: String,
        agents_md: &AgentsMd,
        extra_system_prompt: Option<&str>,
        persistent_session: Option<Arc<SessionStore>>,
        runtime: ToolRuntime,
        cancel: &CancelToken,
        drain_queued_user_messages: Option<DrainQueuedUserMessagesFn>,
        emit: EmitEventFn,
        enable_compaction_memory_refresh: bool,
    ) -> anyhow::Result<String> {
        // Make it visible in the event stream whether `Agents.md` was found.
        //
        // This directly addresses the user's report:
        // "Agents.md seems not successfully read / persona not loaded".
        if agents_md.found {
            (emit)(
                EventKind::Log,
                task_id,
                format!(
                    "Agents.md loaded ({} bytes) from {}",
                    agents_md.content.len(),
                    agents_md.path.display()
                ),
            );
        } else {
            (emit)(
                EventKind::Error,
                task_id,
                format!("Agents.md NOT FOUND at {}", agents_md.path.display()),
            );
        }

        // Restore the top-level persisted session if this run is attached to
        // the durable workspace conversation. Sub-agents pass `None` here and
        // therefore stay ephemeral.
        let mut session_compaction_context = None::<String>;
        let mut messages = Vec::<ChatMessage>::new();
        let mut compaction_state = CompactionState::default();

        if let Some(session_store) = persistent_session.as_ref() {
            let snapshot = session_store
                .load_snapshot()
                .context("Failed to load persisted session snapshot")?;
            compaction_state.restore_summary(snapshot.compaction_summary.clone());
            session_compaction_context = Some(build_session_compaction_context_block(&snapshot));
            messages = snapshot.messages;

            (emit)(
                EventKind::Log,
                task_id,
                format!(
                    "Loaded persisted session {} (conversation_id={}, messages={}, compacted={}, truncated_incomplete={}).",
                    snapshot.descriptor.current_session_path,
                    snapshot.descriptor.conversation_id,
                    messages.len(),
                    snapshot.compaction_summary.is_some(),
                    snapshot.truncated_incomplete_messages,
                ),
            );
        }

        // Build the "system prompt" (or "developer prompt") that stays
        // constant, except when compaction rotates the backing session file and
        // we therefore need to refresh the injected compaction metadata block.
        let mut system_message = ChatMessage::text(
            self.cfg.system_role_name.clone(),
            self.build_system_prompt(
                agents_md,
                join_extra_system_prompt(
                    extra_system_prompt,
                    session_compaction_context.as_deref(),
                )
                .as_deref(),
            ),
        );

        // Store only the real conversation here. Synthetic compaction summaries
        // are injected later when we build the provider request.
        append_message_and_persist(
            &mut messages,
            ChatMessage::text("user", task.clone()),
            persistent_session.as_deref(),
        )?;

        // We pre-compute tool definitions once. This keeps requests stable.
        let tool_definitions = self.tools.tool_definitions();

        // Each run gets its own session state.
        //
        // This is important for the `Edit` guardrail: "must `Read` before
        // `Edit`" is enforced per agent/sub-agent session.
        let mut tool_session = ToolSession::default();

        // Consecutive model-call failures. This drives the infinite retry
        // backoff schedule requested by the user.
        let mut model_error_count: u32 = 0;

        // Step loop.
        for step in 1..=self.cfg.max_steps {
            // Cooperative cancellation check before starting the next step.
            if cancel.is_cancelled() {
                let msg = "Task cancelled by user interrupt.".to_string();
                (emit)(EventKind::Error, task_id, msg.clone());
                (emit)(EventKind::Final, task_id, msg.clone());
                return Ok(msg);
            }

            drain_follow_up_messages(
                task_id,
                &mut messages,
                persistent_session.as_deref(),
                drain_queued_user_messages.as_ref(),
                &emit,
            )?;

            match maybe_compact_history(
                &self.llm,
                &self.cfg.model,
                &system_message.role,
                &system_message,
                self.cfg.reasoning_effort.as_deref(),
                &mut compaction_state,
                &mut messages,
                &tool_definitions,
                &self.cfg.compaction,
                cancel,
            )
            .await
            {
                Ok(Some(report)) => {
                    if let Some(session_store) = persistent_session.as_ref() {
                        let summary = compaction_state.summary().ok_or_else(|| {
                            anyhow::anyhow!(
                                "Compaction succeeded but no checkpoint summary remained in state"
                            )
                        })?;
                        let descriptor = session_store
                            .rollover_after_compaction(summary, &messages)
                            .context("Failed to rotate persisted session after compaction")?;
                        session_compaction_context = Some(descriptor.compaction_prompt_block());
                        system_message = ChatMessage::text(
                            self.cfg.system_role_name.clone(),
                            self.build_system_prompt(
                                agents_md,
                                join_extra_system_prompt(
                                    extra_system_prompt,
                                    session_compaction_context.as_deref(),
                                )
                                .as_deref(),
                            ),
                        );

                        (emit)(
                            EventKind::Log,
                            task_id,
                            format!(
                                "Rotated persisted session after compaction: current={}, previous={}",
                                descriptor.current_session_path,
                                descriptor
                                    .previous_session_path
                                    .as_deref()
                                    .unwrap_or("(none)")
                            ),
                        );
                    }

                    if enable_compaction_memory_refresh {
                        match run_compaction_memory_refresh(
                            self,
                                task_id,
                                agents_md,
                                extra_system_prompt,
                                persistent_session.as_ref(),
                                cancel,
                                &emit,
                                compaction_state.summary().unwrap_or_default(),
                            )
                            .await
                        {
                            Ok(Some(internal_notice)) => {
                                append_message_and_persist(
                                    &mut messages,
                                    internal_notice,
                                    persistent_session.as_deref(),
                                )?;
                            }
                            Ok(None) => {}
                            Err(err) => {
                                (emit)(
                                    EventKind::Error,
                                    task_id,
                                    format!(
                                        "Compaction-triggered memory refresh failed before step {step}: {err:#}"
                                    ),
                                );
                            }
                        }
                    }

                    (emit)(
                        EventKind::Log,
                        task_id,
                        format!(
                            "Compacted history before step {step}: tokens {} -> {}, summarized {} message(s), kept {}, split_turn={}",
                            report.tokens_before,
                            report.tokens_after,
                            report.summarized_messages,
                            report.kept_messages,
                            report.split_turn,
                        ),
                    );
                }
                Ok(None) => {}
                Err(err) => {
                    let msg = format!("History compaction failed before step {step}: {err}");
                    (emit)(EventKind::Error, task_id, msg.clone());
                    (emit)(EventKind::Final, task_id, msg.clone());
                    return Err(anyhow::anyhow!(msg));
                }
            }

            (emit)(
                EventKind::Log,
                task_id,
                format!("Step {step}/{}: calling model...", self.cfg.max_steps),
            );

            // Build the request.
            let req = ChatCompletionsRequest {
                model: self.cfg.model.clone(),
                messages: build_request_messages(&system_message, &compaction_state, &messages),
                max_tokens: None,
                reasoning_effort: self.cfg.reasoning_effort.clone(),
                tools: Some(tool_definitions.clone()),
                tool_choice: Some(serde_json::json!("auto")),
                // Stream provider output by default for the main agent loop.
                //
                // Rationale:
                // - Some gateways time out long non-streaming requests and return
                //   503 even though streaming succeeds.
                // - `OpenAiClient` still aggregates the final response back into
                //   the canonical SA shape, so the rest of the loop stays
                //   unchanged.
                // - If a provider rejects streaming outright, the client
                //   transparently falls back to one non-streaming retry.
                stream: Some(true),
            };

            // Call provider with **infinite retry** + backoff.
            let resp = loop {
                // Allow cancelling even while we are retrying.
                if cancel.is_cancelled() {
                    let msg = "Task cancelled by user interrupt.".to_string();
                    (emit)(EventKind::Error, task_id, msg.clone());
                    (emit)(EventKind::Final, task_id, msg.clone());
                    return Ok(msg);
                }

                let call = self.llm.chat_completions(&req);
                let result = tokio::select! {
                    _ = cancel.cancelled() => {
                        let msg = "Task cancelled by user interrupt.".to_string();
                        (emit)(EventKind::Error, task_id, msg.clone());
                        (emit)(EventKind::Final, task_id, msg.clone());
                        return Ok(msg);
                    }
                    r = call => r,
                };

                match result {
                    Ok(resp) => {
                        // Reset the counter on success.
                        model_error_count = 0;
                        break resp;
                    }
                    Err(err) if err.is_retriable() => {
                        model_error_count = model_error_count.saturating_add(1);
                        let delay = retry_delay(model_error_count);

                        (emit)(
                            EventKind::Error,
                            task_id,
                            format!(
                                "Model call failed (retryable; count={model_error_count}; next_retry_in={:?}): {err}",
                                delay
                            ),
                        );

                        // Wait before retrying, but stay cancellable.
                        tokio::select! {
                            _ = cancel.cancelled() => {
                                let msg = "Task cancelled by user interrupt.".to_string();
                                (emit)(EventKind::Error, task_id, msg.clone());
                                (emit)(EventKind::Final, task_id, msg.clone());
                                return Ok(msg);
                            }
                            _ = tokio::time::sleep(delay) => {}
                        }
                    }
                    Err(err) => {
                        // Non-retryable error: fail the task.
                        return Err(anyhow::Error::new(err))
                            .context("Non-retryable model call error");
                    }
                }
            };
            let response_usage = resp.usage.clone();
            let choice = resp.first_choice()?;

            // Copy assistant message for our history.
            //
            // Important: the top-level response `usage` is request-scoped, not
            // message-scoped. We anchor that snapshot to this assistant turn so
            // the next compaction pass can reuse it as "last known real usage".
            let mut assistant = choice.message.clone();
            assistant.request_usage = response_usage;

            // Emit assistant content (if present).
            if let Some(content) = assistant.content.as_deref() {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    (emit)(EventKind::Log, task_id, trimmed.to_string());
                }
            }

            // Extract tool calls (if any).
            let tool_calls: Vec<ToolCall> = assistant.tool_calls.clone().unwrap_or_default();

            // Persist assistant message in history.
            append_message_and_persist(&mut messages, assistant, persistent_session.as_deref())?;

            // If no tool calls => done.
            if tool_calls.is_empty() {
                if drain_follow_up_messages(
                    task_id,
                    &mut messages,
                    persistent_session.as_deref(),
                    drain_queued_user_messages.as_ref(),
                    &emit,
                )? {
                    continue;
                }

                let final_text = messages
                    .last()
                    .and_then(|m| m.content.clone())
                    .unwrap_or_default();
                (emit)(EventKind::Final, task_id, final_text.clone());
                return Ok(final_text);
            }

            // Execute tools sequentially.
            for call in tool_calls {
                if cancel.is_cancelled() {
                    let msg = "Task cancelled by user interrupt.".to_string();
                    (emit)(EventKind::Error, task_id, msg.clone());
                    (emit)(EventKind::Final, task_id, msg.clone());
                    return Ok(msg);
                }

                (emit)(
                    EventKind::Tool,
                    task_id,
                    format!(
                        "Tool call: {}({})",
                        call.function.name, call.function.arguments
                    ),
                );

                // Parse tool arguments (OpenAI provides them as a JSON string).
                let args_json: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .with_context(|| {
                    format!(
                        "Failed to parse tool arguments JSON for {}",
                        call.function.name
                    )
                })?;

                // Execute.
                let tool_result = match self
                    .tools
                    .execute(
                        &mut tool_session,
                        &runtime,
                        &call.function.name,
                        args_json,
                        cancel,
                    )
                    .await
                {
                    Ok(output) => output,
                    Err(err) => {
                        // If the tool failed because the user interrupted the task, treat it as
                        // a task cancellation rather than a "normal" tool failure.
                        if cancel.is_cancelled() {
                            let msg = "Task cancelled by user interrupt.".to_string();
                            (emit)(EventKind::Error, task_id, msg.clone());
                            (emit)(EventKind::Final, task_id, msg.clone());
                            return Ok(msg);
                        }

                        let msg = format!("Tool `{}` failed: {err}", call.function.name);
                        (emit)(EventKind::Error, task_id, msg.clone());
                        // Still return a tool result message so the model can react.
                        format!("ERROR: {msg}")
                    }
                };

                // Emit tool output (can be JSON).
                (emit)(EventKind::Tool, task_id, tool_result.clone());

                // Append tool result message.
                append_message_and_persist(
                    &mut messages,
                    ChatMessage::tool_result(call.id, tool_result),
                    persistent_session.as_deref(),
                )?;
            }
        }

        // If we exit the loop we hit the step cap.
        let msg = format!(
            "Reached max_steps={} without a final answer.",
            self.cfg.max_steps
        );
        (emit)(EventKind::Error, task_id, msg.clone());
        // Emit `Final` so clients can stop waiting even on failure.
        (emit)(EventKind::Final, task_id, msg.clone());
        Ok(msg)
    }

    /// Build the stable instructions block injected into the prompt.
    fn build_system_prompt(
        &self,
        agents_md: &AgentsMd,
        extra_system_prompt: Option<&str>,
    ) -> String {
        use std::fmt::Write as _;

        let now = chrono::Local::now();

        let mut out = String::from(STATIC_PROMPT_TEMPLATE.trim_end());
        out.push_str("\n\n");

        if !self.skills.list().is_empty() {
            out.push_str("## 技能授权\n\n");
            out.push_str("所有已注册技能都已经过授权，可以按需使用。同学的任务如果明显需要某项技能，就直接用 `Skill` 读取它，不要凭空编造\"策略限制\"来回避。\n\n");

            out.push_str("## 可用技能\n\n");
            out.push_str("技能是保存在本地目录中的说明包，每个技能目录至少包含一个 `SKILL.md`。\n");
            out.push_str("当某项技能与你的任务相关时，先用 `Skill` 读取该技能的 `SKILL.md`，再按其中引用的相对路径继续读取技能内文件。\n");
            out.push_str(
                "你不会看到技能在宿主机上的真实安装目录；只能通过技能名和技能内相对路径访问。\n\n",
            );

            for item in self.skills.list() {
                let _ = writeln!(
                    out,
                    "- `{}`：{}",
                    item.name,
                    normalize_skill_description(&item.description)
                );
            }
            out.push('\n');
        }

        let _ = writeln!(
            out,
            "## 工作区\n\n当前工作目录：`{}`\n",
            self.tools.ctx.workspace_root.display()
        );

        out.push_str("## 项目上下文\n\n");
        out.push_str(&format_agents_md_block(agents_md));
        out.push('\n');

        let _ = writeln!(
            out,
            "## 当前日期与时间\n\n{} ({})\n",
            now.format("%Y-%m-%d %H:%M:%S"),
            now.format("%Z")
        );

        let _ = writeln!(out, "## 运行时\n\n模型：`{}`\n", self.cfg.model);

        // Additional context injected by the daemon (memory, preloaded files, etc.).
        if let Some(extra) = extra_system_prompt {
            let extra = extra.trim();
            if !extra.is_empty() {
                out.push_str("## 附加运行时上下文\n\n");
                out.push_str(extra);
                out.push('\n');
            }
        }

        out
    }
}

/// Build one runtime context block that describes the currently active
/// compaction/session state restored from disk.
fn build_session_compaction_context_block(snapshot: &crate::session::SessionSnapshot) -> String {
    let mut out = snapshot.descriptor.compaction_prompt_block();

    if snapshot.compaction_summary.is_some() {
        out.push_str(
            "\n- 当前已有已恢复的压缩摘要检查点；摘要正文会作为单独的上下文消息自动注入，本块不重复展开。\n",
        );
    } else {
        out.push_str("\n- 当前还没有已生效的压缩摘要。\n");
    }

    out
}

/// Join daemon-provided runtime context with session/compaction context.
fn join_extra_system_prompt(
    daemon_extra_system_prompt: Option<&str>,
    session_compaction_context: Option<&str>,
) -> Option<String> {
    let mut parts = Vec::<String>::new();

    if let Some(extra) = daemon_extra_system_prompt {
        let trimmed = extra.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    if let Some(compaction_context) = session_compaction_context {
        let trimmed = compaction_context.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Run one isolated memory-refresh sub-agent immediately after a successful
/// compaction pass.
///
/// Why do this here instead of waiting for nightly dream?
/// - The just-compacted history contains fresh signal that may deserve
///   promotion into long-term memory.
/// - Running the refresh immediately prevents that signal from sitting only in
///   raw session logs until midnight.
/// - We then inject an internal notice into the parent conversation so the
///   main agent knows memory may have changed and can re-read it if needed.
async fn run_compaction_memory_refresh(
    runner: &AgentRunner,
    task_id: Uuid,
    agents_md: &AgentsMd,
    extra_system_prompt: Option<&str>,
    persistent_session: Option<&Arc<SessionStore>>,
    cancel: &CancelToken,
    emit: &EmitEventFn,
    summary: &str,
) -> anyhow::Result<Option<ChatMessage>> {
    let Some(session_store) = persistent_session else {
        return Ok(None);
    };

    let snapshot = session_store
        .load_snapshot()
        .context("Failed to load persisted session snapshot for memory refresh")?;
    let descriptor = snapshot.descriptor;
    let refresh_extra_context =
        build_compaction_memory_refresh_context(summary, &descriptor, &snapshot.messages);
    let merged_extra_prompt = join_extra_system_prompt(
        extra_system_prompt,
        Some(refresh_extra_context.as_str()),
    );

    let runtime = build_internal_memory_refresh_runtime(
        runner,
        task_id,
        agents_md,
        merged_extra_prompt.as_deref(),
        emit,
        0,
    );
    let emit_for_refresh = Arc::clone(emit);
    let refresh_emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
        let kind = if matches!(kind, EventKind::Final) {
            EventKind::Log
        } else {
            kind
        };
        (emit_for_refresh)(kind, task_id, format!("[memory-refresh] {message}"));
    });

    (emit)(
        EventKind::Log,
        task_id,
        "Starting compaction-triggered memory refresh sub-agent.".to_string(),
    );

    let final_text = Box::pin(runner.run_task_inner(
            task_id,
            build_compaction_memory_refresh_task(&descriptor),
            agents_md,
            merged_extra_prompt.as_deref(),
            None,
            runtime,
            cancel,
            None,
            refresh_emit,
            false,
        ))
        .await
        .context("Compaction memory refresh sub-agent failed")?;

    Ok(Some(build_memory_refresh_completion_notice(
        &descriptor, &final_text,
    )))
}

/// Build the compact-triggered memory-refresh task text.
fn build_compaction_memory_refresh_task(
    descriptor: &crate::session::SessionDescriptor,
) -> String {
    format!(
        "执行一次 compact 后的记忆整理。目标不是总结，而是把刚压缩掉的历史信号提炼进长期记忆。\
\n\n工作要求：\
\n- 优先检查 `MEMORY.md`、`memory/topics/*.md` 与最近原始记录，避免重复。\
\n- 必要时阅读最近的原始会话段，尤其是刚被 compact 掉的历史。\
\n- 只提升稳定、可复用、高价值的信息。\
\n- 如果更新了长期记忆，请同步整理相关 topic 文件。\
\n- 不要使用 `Ask`、`Send` 或 `Show`。\
\n- 必要时可以使用 `SubAgent`，但所有子代理与后代子代理同样禁止使用交互工具。\
\n- 最后给出一段简短摘要，说明你整理了哪些记忆。\
\n\n当前压缩后会话：`{}`",
        descriptor.current_session_path
    )
}

/// Build the extra runtime context used by the compact-triggered memory
/// refresh sub-agent.
fn build_compaction_memory_refresh_context(
    summary: &str,
    descriptor: &crate::session::SessionDescriptor,
    kept_messages: &[ChatMessage],
) -> String {
    use std::fmt::Write as _;

    let mut out = String::from("## Compact-triggered Memory Refresh Context\n\n");
    out.push_str("这是一次在 compact 成功后立即触发的后台记忆整理任务。");
    out.push_str("目标是把刚压缩掉的高价值信息提炼进长期记忆，并减少它只停留在原始会话文件中的时间。\n\n");
    let _ = writeln!(out, "- 当前压缩后会话文件：`{}`", descriptor.current_session_path);
    let _ = writeln!(
        out,
        "- 上一段原始会话文件：`{}`",
        descriptor
            .previous_session_path
            .as_deref()
            .unwrap_or("(none)")
    );
    let _ = writeln!(out, "- compact 后当前会话中保留的真实消息数：{}", kept_messages.len());
    out.push_str("- 这是内部运行时任务，不是同学的新请求。\n");
    out.push_str("- 你完成后，主 Agent 会收到一条内部提示消息，提醒它长期记忆可能已更新。\n\n");
    out.push_str("### 当前 compact 摘要\n\n<summary>\n");
    out.push_str(summary.trim());
    out.push_str("\n</summary>\n");
    out
}

/// Build one internal user-role message that informs the main agent that a
/// memory refresh finished after compaction.
fn build_memory_refresh_completion_notice(
    descriptor: &crate::session::SessionDescriptor,
    final_text: &str,
) -> ChatMessage {
    let summary = trim_internal_notice_text(final_text, 800);
    ChatMessage::text(
        "user",
        format!(
            "[系统记忆更新通知]\n\
compact 后已完成一次后台记忆整理。长期记忆文件可能已经更新（如 `MEMORY.md`、`memory/topics/*.md`）。\
\n这不是同学的新请求，也不是需要对外汇报的内容；它只是提醒你：如果后续推理依赖长期偏好、历史决定或稳定约束，请优先重新检查相关记忆文件。\
\n最近一次压缩后的当前会话文件：`{}`\
\n最近一次被压缩掉的原始会话文件：`{}`\
\n整理摘要：\n{summary}",
            descriptor.current_session_path,
            descriptor
                .previous_session_path
                .as_deref()
                .unwrap_or("(none)")
        ),
    )
}

/// Create a restricted runtime for the compact-triggered memory refresh agent.
fn build_internal_memory_refresh_subagent_context(
    base_extra_prompt: &str,
    request: &crate::tools::SubAgentRequest,
    depth: u32,
) -> String {
    let mut parts = Vec::<String>::new();
    let trimmed = base_extra_prompt.trim();
    if !trimmed.is_empty() {
        parts.push(trimmed.to_string());
    }

    parts.push(format!(
        "## Parent-provided Memory Refresh SubAgent Context\n\n- depth: {depth}\n- label: {}\n- 所有后代子代理都不得使用 `Ask`、`Send` 或 `Show`；如果需要拆分任务，只能继续使用同样受限的 `SubAgent`。\n\n```text\n{}\n```",
        request.label.as_deref().unwrap_or("(none)"),
        request.context.trim()
    ));

    parts.join("\n\n")
}

fn build_internal_memory_refresh_runtime(
    runner: &AgentRunner,
    task_id: Uuid,
    agents_md: &AgentsMd,
    extra_system_prompt: Option<&str>,
    emit: &EmitEventFn,
    depth: u32,
) -> ToolRuntime {
    use crate::tools::{
        AskQuestionFn, RunSubAgentFn, SendMessageFn, ShowFileFn, SubAgentRequest,
        MAX_SUBAGENT_DEPTH,
    };

    let runner_for_subagent = runner.clone();
    let agents_md_for_subagent = agents_md.clone();
    let extra_prompt_for_subagent = extra_system_prompt.map(str::to_string);

    let emit_for_send = Arc::clone(emit);
    let send_message: SendMessageFn = Arc::new(move |message: String| {
        let emit = Arc::clone(&emit_for_send);
        Box::pin(async move {
            (emit)(
                EventKind::Log,
                task_id,
                format!("[memory-refresh] suppressed Send: {message}"),
            );
            Ok(())
        })
    });

    let ask_question: AskQuestionFn = Arc::new(move |_request, _cancel| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use Ask") })
    });

    let emit_for_show = Arc::clone(emit);
    let show_file: ShowFileFn = Arc::new(move |file| {
        let emit = Arc::clone(&emit_for_show);
        Box::pin(async move {
            (emit)(
                EventKind::Log,
                task_id,
                format!("[memory-refresh] suppressed Show: {}", file.path),
            );
            Ok(())
        })
    });

    let emit_for_subagent = Arc::clone(emit);
    let run_subagent: RunSubAgentFn = Arc::new(move |request: SubAgentRequest, child_cancel| {
        let runner = runner_for_subagent.clone();
        let agents_md = agents_md_for_subagent.clone();
        let extra_prompt = extra_prompt_for_subagent.clone();
        let emit = Arc::clone(&emit_for_subagent);

        Box::pin(async move {
            let next_depth = depth.saturating_add(1);
            if next_depth > MAX_SUBAGENT_DEPTH {
                anyhow::bail!(
                    "memory refresh subagent depth limit exceeded (requested depth={}, max={})",
                    next_depth,
                    MAX_SUBAGENT_DEPTH
                );
            }

            let merged_extra_prompt = build_internal_memory_refresh_subagent_context(
                extra_prompt.as_deref().unwrap_or_default(),
                &request,
                next_depth,
            );
            let nested_runtime = build_internal_memory_refresh_runtime(
                &runner,
                task_id,
                &agents_md,
                Some(merged_extra_prompt.as_str()),
                &emit,
                next_depth,
            );
            let request_task = request.task.clone();
            let label = request
                .label
                .clone()
                .unwrap_or_else(|| format!("memory-refresh-depth-{next_depth}"));
            let emit_for_nested = Arc::clone(&emit);
            let nested_emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
                let kind = if matches!(kind, EventKind::Final) {
                    EventKind::Log
                } else {
                    kind
                };
                (emit_for_nested)(
                    kind,
                    task_id,
                    format!(
                        "[memory-refresh-subagent depth={} label={}] {}",
                        next_depth, label, message
                    ),
                );
            });

            let final_answer = Box::pin(runner.run_task_inner(
                task_id,
                request_task,
                &agents_md,
                Some(merged_extra_prompt.as_str()),
                None,
                nested_runtime,
                &child_cancel,
                None,
                nested_emit,
                false,
            ))
            .await?;

            Ok(final_answer)
        })
    });

    ToolRuntime::new(send_message, ask_question, show_file, run_subagent)
}

/// Trim a long internal runtime note so it stays useful without bloating the
/// next request.
fn trim_internal_notice_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "（无）".to_string();
    }
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(max_chars).collect();
    out.push('…');
    out
}

/// Append one real message to the in-memory conversation and, when enabled,
/// mirror it into the durable JSONL session segment.
fn append_message_and_persist(
    messages: &mut Vec<ChatMessage>,
    message: ChatMessage,
    persistent_session: Option<&SessionStore>,
) -> anyhow::Result<()> {
    if let Some(session_store) = persistent_session {
        session_store
            .append_message(&message)
            .context("Failed to append message to persisted session")?;
    }
    messages.push(message);
    Ok(())
}

/// Drain any queued follow-up user messages and append them to the current
/// conversation history as fresh user turns.
///
/// This is the core of the "don't interrupt on normal send" behavior:
/// - while the model/tool work is busy, new user messages are buffered outside
///   the agent loop,
/// - once we reach a safe point before the next model request, we splice those
///   messages into the same conversation,
/// - the model then sees them as ordinary subsequent user turns.
fn drain_follow_up_messages(
    task_id: Uuid,
    messages: &mut Vec<ChatMessage>,
    persistent_session: Option<&SessionStore>,
    drain_queued_user_messages: Option<&DrainQueuedUserMessagesFn>,
    emit: &EmitEventFn,
) -> anyhow::Result<bool> {
    let Some(drain) = drain_queued_user_messages else {
        return Ok(false);
    };

    let queued: Vec<String> = (drain)()
        .into_iter()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .collect();

    if queued.is_empty() {
        return Ok(false);
    }

    (emit)(
        EventKind::Log,
        task_id,
        format!(
            "Injecting {} queued follow-up message(s) into the current session.",
            queued.len()
        ),
    );

    for text in queued {
        append_message_and_persist(
            messages,
            ChatMessage::text("user", text),
            persistent_session,
        )?;
    }

    Ok(true)
}

/// Normalize third-party skill descriptions so prompt wording stays consistent
/// with the SA branding even when upstream skills still mention other agents.
fn normalize_skill_description(raw: &str) -> String {
    let mut out = raw.to_string();

    for (from, to) in [
        ("extends Codex's capabilities", "extends SA's capabilities"),
        ("extends Claude's capabilities", "extends SA's capabilities"),
        (
            "Install Codex skills into $CODEX_HOME/skills",
            "Install skills into $SA_HOME/skills",
        ),
        (
            "Install Claude skills into $CLAUDE_HOME/skills",
            "Install skills into $SA_HOME/skills",
        ),
        ("$CODEX_HOME", "$SA_HOME"),
        ("$CLAUDE_HOME", "$SA_HOME"),
    ] {
        out = out.replace(from, to);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents_md::AgentsMd;
    use crate::openai::OpenAiClient;
    use crate::skills::SkillRegistry;
    use crate::session::{SessionDescriptor, SessionSnapshot};
    use crate::tools::{ToolContext, ToolExecutor};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Restored sessions should describe the compaction checkpoint once, but the
    /// actual summary body must stay out of the system prompt so request
    /// building can inject it exactly once in synthetic message form.
    #[test]
    fn restored_compaction_context_omits_summary_body() {
        let summary_body = "## 目标\n- 保持唯一摘要注入";
        let snapshot = SessionSnapshot {
            descriptor: SessionDescriptor {
                conversation_id: Uuid::nil(),
                current_session_path: "sessions/current.jsonl".to_string(),
                previous_session_path: Some("sessions/previous.jsonl".to_string()),
            },
            compaction_summary: Some(summary_body.to_string()),
            messages: Vec::new(),
            truncated_incomplete_messages: 0,
        };

        let context = build_session_compaction_context_block(&snapshot);
        assert!(context.contains("sessions/current.jsonl"));
        assert!(context.contains("sessions/previous.jsonl"));
        assert!(context.contains("摘要正文会作为单独的上下文消息自动注入"));
        assert!(!context.contains(summary_body));
    }

    /// Create a unique temporary workspace for prompt-related tests.
    fn unique_workspace() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sa-agent-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Build a minimal runner so unit tests can exercise prompt assembly.
    fn test_runner() -> AgentRunner {
        let skills = Arc::new(SkillRegistry::default());
        let tool_context =
            ToolContext::new(unique_workspace(), skills.clone()).expect("tool context");
        let tool_executor = ToolExecutor::new(tool_context, None);
        let llm = OpenAiClient::new("http://127.0.0.1:11434/v1".to_string(), "test-key".into())
            .expect("openai client");

        AgentRunner::new(
            llm,
            tool_executor,
            skills,
            AgentRunnerConfig {
                model: "test-model".to_string(),
                system_role_name: "developer".to_string(),
                reasoning_effort: None,
                max_steps: 4,
                compaction: CompactionConfig::default(),
            },
        )
    }

    /// Create a small `Agents.md` payload so prompt composition stays traceable.
    fn test_agents_md() -> AgentsMd {
        AgentsMd {
            path: PathBuf::from("Agents.md"),
            content: "请保持可追溯与可验证。".to_string(),
            found: true,
        }
    }

    /// Resolve the repository-level static prompt source file.
    fn prompt_md_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("prompt.md")
    }

    /// The static prompt must come from `sa/prompt.md` so that prompt edits
    /// have one canonical source of truth.
    #[test]
    fn system_prompt_starts_with_prompt_md_static_source() {
        let static_prompt =
            fs::read_to_string(prompt_md_path()).expect("read repository prompt.md");

        assert!(static_prompt.contains("## 记忆分层"));
        assert!(static_prompt.contains("### 原始层"));
        assert!(static_prompt.contains("### 长期层"));
        assert!(static_prompt.contains("### dream 提炼层"));
        assert!(static_prompt.contains("memory/topics/"));
        assert!(static_prompt.contains("memory/dreams/"));

        let runner = test_runner();
        let prompt = runner.build_system_prompt(&test_agents_md(), None);
        assert!(prompt.starts_with(static_prompt.trim_end()));
    }

    /// Dynamic sections still need to be appended after the static prompt.
    #[test]
    fn system_prompt_appends_runtime_sections_after_static_prompt() {
        let runner = test_runner();
        let prompt = runner.build_system_prompt(
            &test_agents_md(),
            Some("## 测试运行时上下文\n\n- dream 状态：idle"),
        );

        assert!(prompt.contains("## Agents.md"));
        assert!(prompt.contains("请保持可追溯与可验证。"));
        assert!(prompt.contains("## 当前日期与时间"));
        assert!(prompt.contains("## 运行时"));
        assert!(prompt.contains("## 附加运行时上下文"));
        assert!(prompt.contains("dream 状态：idle"));
    }

    /// Compact-triggered memory refresh context should include the summary and
    /// the rotated session paths so the refresh sub-agent knows where to look.
    #[test]
    fn compaction_memory_refresh_context_includes_summary_and_paths() {
        let descriptor = SessionDescriptor {
            conversation_id: Uuid::nil(),
            current_session_path: "sessions/current.jsonl".to_string(),
            previous_session_path: Some("sessions/previous.jsonl".to_string()),
        };
        let context = build_compaction_memory_refresh_context(
            "## 关键决策\n- 已选择新的记忆结构",
            &descriptor,
            &[ChatMessage::text("assistant", "保留尾部消息")],
        );

        assert!(context.contains("Compact-triggered Memory Refresh Context"));
        assert!(context.contains("sessions/current.jsonl"));
        assert!(context.contains("sessions/previous.jsonl"));
        assert!(context.contains("已选择新的记忆结构"));
        assert!(context.contains("不是同学的新请求"));
    }

    /// The completion notice injected back into the parent session must stay an
    /// internal runtime reminder rather than looking like a new user request.
    #[test]
    fn memory_refresh_completion_notice_is_internal_runtime_note() {
        let descriptor = SessionDescriptor {
            conversation_id: Uuid::nil(),
            current_session_path: "sessions/current.jsonl".to_string(),
            previous_session_path: Some("sessions/previous.jsonl".to_string()),
        };
        let notice = build_memory_refresh_completion_notice(
            &descriptor,
            "已更新 MEMORY.md，并合并了 testing 偏好专题。",
        );

        assert_eq!(notice.role, "user");
        let text = notice.content.as_deref().unwrap_or_default();
        assert!(text.contains("[系统记忆更新通知]"));
        assert!(text.contains("这不是同学的新请求"));
        assert!(text.contains("MEMORY.md"));
        assert!(text.contains("memory/topics/*.md"));
    }

    /// The compact-triggered memory-refresh task should allow `SubAgent`, but
    /// still forbid direct user interaction tools.
    #[test]
    fn memory_refresh_task_allows_subagent_but_forbids_interaction_tools() {
        let descriptor = SessionDescriptor {
            conversation_id: Uuid::nil(),
            current_session_path: "sessions/current.jsonl".to_string(),
            previous_session_path: Some("sessions/previous.jsonl".to_string()),
        };
        let task = build_compaction_memory_refresh_task(&descriptor);

        assert!(task.contains("不要使用 `Ask`、`Send` 或 `Show`"));
        assert!(!task.contains("不要使用 `Ask`、`Send`、`Show` 或 `SubAgent`"));
        assert!(task.contains("必要时可以使用 `SubAgent`"));
    }

    /// Descendant sub-agents spawned by the memory-refresh task must inherit
    /// the "no direct interaction" rule.
    #[test]
    fn memory_refresh_subagent_context_inherits_no_interaction_rule() {
        let request = crate::tools::SubAgentRequest {
            label: Some("memory-topics".to_string()),
            task: "整理 testing 主题记忆".to_string(),
            context: "聚焦 testing 偏好".to_string(),
        };
        let context =
            build_internal_memory_refresh_subagent_context("## Base\n\n- compact 后上下文", &request, 2);

        assert!(context.contains("Parent-provided Memory Refresh SubAgent Context"));
        assert!(context.contains("所有后代子代理都不得使用 `Ask`、`Send` 或 `Show`"));
        assert!(context.contains("聚焦 testing 偏好"));
        assert!(context.contains("depth: 2"));
    }
}
