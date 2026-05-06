//! Minimal autonomous agent loop (tool-calling).
//!
//! This is the "heart" of the StudyAdministrator (SA) agent extracted from the large
//! `zeroclaw` codebase:
//! - We call an OpenAI-compatible Chat Completions endpoint.
//! - We provide tool definitions so the model can request actions.
//! - We execute those tools and feed results back to the model.
//! - We repeat until the current work explicitly finishes or is cancelled.
//!
//! The backend daemon (`sa`) owns task queues, event IDs, and WS connections.
//! This module is deliberately "pure core": it only needs an event callback.

use crate::agents_md::{AgentsMd, format_agents_md_block};
use crate::cancel::CancelToken;
use crate::compact::{
    CompactionConfig, CompactionState, build_request_messages, maybe_compact_history,
};
use crate::openai::{ChatCompletionsRequest, ChatMessage, ContentPart, ImageUrl, MessageContent, OpenAiClient, ToolCall};
use crate::retry::retry_delay;
use crate::session::SessionStore;
use crate::skills::SkillRegistry;
use crate::tools::{
    AskRequest, PromptProfile, ReloadRuntimeFn, ToolControl, ToolExecutionResult, ToolExecutor,
    ToolRuntime, ToolSession,
};
use std::collections::HashMap;
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

/// Static prompt source-of-truth for ordinary worker sub-agents.
const STATIC_SUBAGENT_PROMPT_TEMPLATE: &str = include_str!("../../../SubAgents.md");

/// Expand `{{file:path:start:end}}` references in user/system messages by
/// reading actual file content from the workspace.
///
/// This is a zero-tool-call mechanism inspired by GenericAgent's prompt
/// pre-processing: the system can inject file content directly into prompts
/// without the model needing to call a file-read tool first. Maximum 1000 lines
/// per expansion to prevent context blow-up.
fn expand_file_refs(
    messages: &mut [ChatMessage],
    workspace_root: &std::path::Path,
) {
    let re = regex::Regex::new(r"\{\{file:([^:}]+)(?::(\d+))?(?::(\d+))?\}\}").unwrap();
    for msg in messages.iter_mut() {
        let Some(MessageContent::Text(text)) = &msg.content else {
            continue;
        };
        if !re.is_match(text) {
            continue;
        }
        let mut result = text.clone();
        let mut modified = false;
        for caps in re.captures_iter(text) {
            let file_path = caps.get(1).unwrap().as_str();
            let start: usize = caps
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(1);
            let end: Option<usize> = caps.get(3).and_then(|m| m.as_str().parse().ok());
            let full_path = workspace_root.join(file_path);
            if let Ok(file_content) = std::fs::read_to_string(&full_path) {
                let lines: Vec<&str> = file_content.lines().collect();
                let end = end.unwrap_or(lines.len()).min(lines.len());
                let start = start.min(end).max(1);
                let excerpt: String = lines[start - 1..end]
                    .iter()
                    .enumerate()
                    .map(|(i, l)| format!("{:>6}| {}", start + i, l))
                    .collect::<Vec<_>>()
                    .join("\n");
                let limit = 1000.min(lines.len());
                let truncated = if end - start + 1 > limit {
                    format!(
                        "… (showing first {limit} of {} lines) …\n",
                        end - start + 1
                    )
                } else {
                    String::new()
                };
                let replacement = format!("[expanded {{file:{file_path}}}:]\n{truncated}{excerpt}\n[/expanded]");
                let full_match = caps.get(0).unwrap().as_str();
                result = result.replace(full_match, &replacement);
                modified = true;
            }
        }
        if modified {
            msg.content = Some(MessageContent::Text(result));
        }
    }
}

/// Callback used by the agent to emit progress events.
///
/// The daemon will wrap these events with event IDs and broadcast them to clients.
pub type EmitEventFn = Arc<dyn Fn(EventKind, Uuid, String) + Send + Sync>;

/// Callback used by the agent to emit real-time context window usage info.
///
/// Called after each successful LLM response so the frontend can display
/// a context usage progress bar.
pub type EmitContextInfoFn = Arc<dyn Fn(usize, usize, usize) + Send + Sync>;

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
    /// History compaction behavior for long-running tasks.
    pub compaction: CompactionConfig,
    /// Sampling temperature (0.0 – 2.0).
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter (0.0 – 1.0).
    pub top_p: Option<f64>,
    /// Maximum output tokens per request.
    pub max_output_tokens: Option<u32>,
    /// Fallback model name for automatic failover when the primary model
    /// produces consecutive retryable errors.
    /// `None` (default) disables failover.
    pub fallback_model: Option<String>,
    /// Number of consecutive retryable failures before switching to the
    /// fallback model. Defaults to 2.
    pub max_consecutive_failures: u32,
    /// Maximum total retry attempts across all retryable errors before
    /// giving up. Defaults to 12. When reached, the task is terminated
    /// with an error message.
    pub max_retries: u32,
    /// Model routing table (category → model_name), None = no routing.
    pub model_routing: Option<HashMap<String, String>>,
}

/// The runnable agent.
#[derive(Clone)]
pub struct AgentRunner {
    /// OpenAI-compatible HTTP client.
    llm: OpenAiClient,
    /// Tool executor.
    tools: ToolExecutor,
    /// Skills registry (for metadata injection).
    skills: Arc<SkillRegistry>,
    /// Runner config.
    cfg: AgentRunnerConfig,
    /// Optional callback for emitting context usage info.
    context_info_emit: Option<EmitContextInfoFn>,
}

impl std::fmt::Debug for AgentRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRunner")
            .field("cfg", &self.cfg)
            .field("has_context_info_emit", &self.context_info_emit.is_some())
            .finish()
    }
}

/// One single-turn execution result produced by `run_quantum`.
#[derive(Debug)]
pub struct AgentQuantumResult {
    /// Updated per-agent tool session state.
    pub tool_session: ToolSession,
    /// Outcome of the turn.
    pub outcome: AgentQuantumOutcome,
}

/// Outcome of one model/tool quantum.
#[derive(Debug)]
pub enum AgentQuantumOutcome {
    /// The work asked the runtime to persist a structured user question and
    /// suspend until the answer is available.
    Ask {
        /// Assistant tool-call message that triggered this ask.
        assistant_message: ChatMessage,
        /// Assistant tool-call id that should receive the eventual tool result.
        tool_call_id: String,
        /// Structured question definition that should be shown to the frontend.
        request: AskRequest,
    },
    /// The work should continue later.
    Continue {
        /// Whether the turn ended without tools and therefore needs the
        /// temporary "remember to Finish" runtime reminder.
        needs_finish_reminder: bool,
        /// Best-effort assistant text emitted during this turn.
        assistant_text: Option<String>,
    },
    /// The current work explicitly finished.
    Finish {
        /// Assistant control-tool message that triggered this finish.
        assistant_message: ChatMessage,
        /// Human-readable finish reason.
        reason: String,
        /// Human-readable finish result summary.
        result: String,
        /// Whether this finish came from the temporary
        /// `FinishWithoutOutput()` tool.
        without_output_confirmation: bool,
    },
    /// The work should wait on another dependency.
    Wait {
        /// Assistant control-tool message that triggered this wait.
        assistant_message: ChatMessage,
        /// Requested dependency wait.
        request: crate::tools::WaitRequest,
    },
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
            context_info_emit: None,
        }
    }

    /// Set the callback used to emit context window usage info.
    pub fn set_context_info_emit(&mut self, emit: EmitContextInfoFn) {
        self.context_info_emit = Some(emit);
    }

    /// Return the effective static runner configuration.
    pub fn config(&self) -> &AgentRunnerConfig {
        &self.cfg
    }

    /// Return the model name configured for this runner.
    pub fn model_name(&self) -> &str {
        &self.cfg.model
    }

    /// Emit context window usage info if the callback is set.
    fn emit_context_info(
        &self,
        system_message: &ChatMessage,
        compaction_state: &CompactionState,
        messages: &[ChatMessage],
        runtime: &ToolRuntime,
        task_id: Uuid,
    ) {
        if let Some(ref emit) = self.context_info_emit {
            let used_tokens = crate::compact::estimate_request_tokens(
                system_message,
                compaction_state,
                messages,
                &self.tools.tool_definitions(runtime),
            );
            let max_tokens = self.cfg.compaction.trigger_tokens;
            let message_count = messages.len();
            (emit)(used_tokens, max_tokens, message_count);
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

    /// Run a task with compaction-triggered memory refresh disabled.
    ///
    /// Use this for internal sub-tasks (e.g. post-task memory collection) that
    /// should not trigger their own compaction memory refresh cycle.
    pub async fn run_task_without_memory_refresh(
        &self,
        task_id: Uuid,
        task: String,
        agents_md: &AgentsMd,
        extra_system_prompt: Option<&str>,
        persistent_session: Option<Arc<SessionStore>>,
        runtime: ToolRuntime,
        cancel: &CancelToken,
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
            None,
            emit,
            false,
        )
        .await
    }

    /// Run exactly one model/tool quantum for one already-active work item.
    ///
    /// This is the primitive used by the durable multi-agent supervisor:
    /// - append any new mailbox-derived messages
    /// - optionally compact before the request
    /// - call the model once
    /// - execute the tool calls from that one assistant turn
    /// - return a structured outcome instead of looping forever in-core
    #[allow(clippy::too_many_arguments)]
    pub async fn run_quantum(
        &self,
        task_id: Uuid,
        incoming_messages: Vec<ChatMessage>,
        agents_md: &AgentsMd,
        extra_system_prompt: Option<&str>,
        persistent_session: Option<Arc<SessionStore>>,
        runtime: ToolRuntime,
        cancel: &CancelToken,
        emit: EmitEventFn,
        mut tool_session: ToolSession,
    ) -> anyhow::Result<AgentQuantumResult> {
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

        let mut system_message = ChatMessage::text(
            self.cfg.system_role_name.clone(),
            self.build_system_prompt(
                &runtime,
                agents_md,
                join_extra_system_prompt(
                    extra_system_prompt,
                    session_compaction_context.as_deref(),
                )
                .as_deref(),
            ),
        );

        for message in incoming_messages {
            append_message_and_persist(&mut messages, message, persistent_session.as_deref())?;
        }

        match maybe_compact_history(
            &self.llm,
            runtime.model_override.as_deref().unwrap_or(&self.cfg.model),
            &system_message.role,
            &system_message,
            runtime
                .effort_override
                .as_deref()
                .or(self.cfg.reasoning_effort.as_deref()),
            &mut compaction_state,
            &mut messages,
            &self.tools.tool_definitions(&runtime),
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
                            &runtime,
                            agents_md,
                            join_extra_system_prompt(
                                extra_system_prompt,
                                session_compaction_context.as_deref(),
                            )
                            .as_deref(),
                        ),
                    );
                }

                (emit)(
                    EventKind::Log,
                    task_id,
                    format!(
                        "Compacted history before quantum: tokens {} -> {}, summarized {} message(s), kept {}, split_turn={}",
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
                let msg = format!("History compaction failed before quantum: {err}");
                (emit)(EventKind::Error, task_id, msg.clone());
                return Err(anyhow::anyhow!(msg));
            }
        }

        let mut req = ChatCompletionsRequest {
            model: runtime
                .model_override
                .clone()
                .unwrap_or_else(|| self.cfg.model.clone()),
            messages: build_request_messages(&system_message, &compaction_state, &messages),
            max_tokens: self.cfg.max_output_tokens,
            reasoning_effort: runtime
                .effort_override
                .clone()
                .or_else(|| self.cfg.reasoning_effort.clone()),
            tools: Some(self.tools.tool_definitions(&runtime)),
            tool_choice: Some(serde_json::json!("auto")),
            stream: Some(true),
            temperature: self.cfg.temperature,
            top_p: self.cfg.top_p,
        };

        let mut model_error_count: u32 = 0;
        let mut total_retries: u32 = 0;
        let resp = loop {
            if cancel.is_cancelled() {
                anyhow::bail!("Task cancelled by user interrupt.");
            }

            let call = self.llm.chat_completions(&req);
            let result = tokio::select! {
                _ = cancel.cancelled() => {
                    anyhow::bail!("Task cancelled by user interrupt.");
                }
                r = call => r,
            };

            match result {
                Ok(resp) => break resp,
                Err(err) if err.is_retriable() => {
                    model_error_count = model_error_count.saturating_add(1);
                    total_retries = total_retries.saturating_add(1);

                    // Quota exhausted: skip retry delay and either failover
                    // immediately or terminate the task.
                    if err.is_quota_exhausted() {
                        if let Some(ref fallback) = self.cfg.fallback_model {
                            if req.model != *fallback {
                                (emit)(
                                    EventKind::Log,
                                    task_id,
                                    format!(
                                        "配额耗尽，自动切换至 fallback 模型: {fallback}"
                                    ),
                                );
                                req.model = fallback.clone();
                                model_error_count = 0;
                                continue;
                            }
                        }
                        // No fallback or already on fallback: terminate.
                        return Err(anyhow::Error::new(err))
                            .context("配额耗尽，请切换模型（当前模型配额已用完）");
                    }

                    // Total retry cap: prevent infinite retries.
                    if total_retries >= self.cfg.max_retries.max(1) {
                        return Err(anyhow::Error::new(err))
                            .context(format!(
                                "达到最大重试次数上限（{}/{}），任务终止",
                                total_retries, self.cfg.max_retries
                            ));
                    }

                    let delay = retry_delay(model_error_count);
                    (emit)(
                        EventKind::Error,
                        task_id,
                        format!(
                            "Model call failed (retryable; count={model_error_count}; total_retries={total_retries}; next_retry_in={:?}): {err}",
                            delay
                        ),
                    );
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            anyhow::bail!("Task cancelled by user interrupt.");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)).context("Non-retryable model call error");
                }
            }
        };

        let response_usage = resp.usage.clone();
        let choice = resp.first_choice()?;
        let mut assistant = choice.message.clone();
        assistant.request_usage = response_usage;

        // Emit context usage info after successful LLM response.
        self.emit_context_info(&system_message, &compaction_state, &messages, &runtime, task_id);

        let assistant_text = assistant.text_content();
        if let Some(text) = assistant_text.as_deref() {
            (emit)(EventKind::Log, task_id, text.to_string());
        }

        let tool_calls: Vec<ToolCall> = assistant.tool_calls.clone().unwrap_or_default();
        if tool_calls.is_empty() {
            // Only persist assistant message if it has actual content.
            // Empty assistant messages (no content and no tool calls) violate
            // OpenAI API requirements and will cause 400 Bad Request.
            if assistant.text_content().is_some() {
                append_message_and_persist(&mut messages, assistant, persistent_session.as_deref())?;
            }
            return Ok(AgentQuantumResult {
                tool_session,
                outcome: AgentQuantumOutcome::Continue {
                    needs_finish_reminder: true,
                    assistant_text,
                },
            });
        }

        let all_control_calls = tool_calls.iter().all(|call| {
            matches!(
                call.function.name.as_str(),
                "Ask" | "Finish" | "FinishWithoutOutput" | "Wait"
            )
        });
        if all_control_calls {
            if tool_calls.len() != 1 {
                anyhow::bail!("Control tools must not be batched in one assistant turn");
            }
            let call = &tool_calls[0];
            let args_json: serde_json::Value = serde_json::from_str(&call.function.arguments)
                .with_context(|| {
                    format!(
                        "Failed to parse tool arguments JSON for {}",
                        call.function.name
                    )
                })?;
            let control = match self
                .tools
                .execute(
                    &mut tool_session,
                    &runtime,
                    &call.function.name,
                    args_json,
                    cancel,
                )
                .await?
            {
                ToolExecutionResult::Control(control) => control,
                ToolExecutionResult::Observation(_) => {
                    anyhow::bail!(
                        "Control tool {} unexpectedly returned a normal observation",
                        call.function.name
                    );
                }
                ToolExecutionResult::ImagePayload { .. } => {
                    anyhow::bail!(
                        "Control tool {} unexpectedly returned an image payload",
                        call.function.name
                    );
                }
            };

            let outcome = match control {
                ToolControl::Ask(request) => AgentQuantumOutcome::Ask {
                    assistant_message: assistant,
                    tool_call_id: call.id.clone(),
                    request,
                },
                ToolControl::Finish(request) => AgentQuantumOutcome::Finish {
                    assistant_message: assistant,
                    reason: request.reason,
                    result: request.result,
                    without_output_confirmation: false,
                },
                ToolControl::FinishWithoutOutput => AgentQuantumOutcome::Finish {
                    assistant_message: assistant,
                    reason: "finish_without_output".to_string(),
                    result: String::new(),
                    without_output_confirmation: true,
                },
                ToolControl::Wait(request) => AgentQuantumOutcome::Wait {
                    assistant_message: assistant,
                    request,
                },
            };
            return Ok(AgentQuantumResult {
                tool_session,
                outcome,
            });
        }

        // Persist assistant message (should have tool calls at this point, but guard anyway).
        if assistant.text_content().is_some() || assistant.tool_calls.is_some() {
            append_message_and_persist(&mut messages, assistant, persistent_session.as_deref())?;
        }
        for call in tool_calls {
            let args_json: serde_json::Value = serde_json::from_str(&call.function.arguments)
                .with_context(|| {
                    format!(
                        "Failed to parse tool arguments JSON for {}",
                        call.function.name
                    )
                })?;
            let exec_result = self
                .tools
                .execute(
                    &mut tool_session,
                    &runtime,
                    &call.function.name,
                    args_json,
                    cancel,
                )
                .await;

            match exec_result {
                Ok(ToolExecutionResult::Observation(output)) => {
                    (emit)(EventKind::Tool, task_id, output.clone());
                    append_message_and_persist(
                        &mut messages,
                        ChatMessage::tool_result(call.id, output),
                        persistent_session.as_deref(),
                    )?;
                }
                Ok(ToolExecutionResult::ImagePayload { .. }) => {
                    // Image payload: inject as a multimodal user message so the
                    // LLM can "see" the image on the next request.
                    let (tool_msg, user_msg) =
                        build_image_payload_messages(&call.id, exec_result.as_ref().unwrap());
                    let b64_len = match exec_result.as_ref().unwrap() {
                        ToolExecutionResult::ImagePayload { data_url, .. } => data_url.len(),
                        _ => 0,
                    };
                    (emit)(
                        EventKind::Tool,
                        task_id,
                        format!(
                            "ImageAnalyze: image loaded (base64 {b64_len} bytes), injecting multimodal user message"
                        ),
                    );
                    append_message_and_persist(
                        &mut messages,
                        tool_msg,
                        persistent_session.as_deref(),
                    )?;
                    append_message_and_persist(
                        &mut messages,
                        user_msg,
                        persistent_session.as_deref(),
                    )?;
                }
                Ok(ToolExecutionResult::Control(_)) => {
                    anyhow::bail!("Control tools must be emitted in their own assistant turn");
                }
                Err(err) => {
                    let msg = format!("Tool `{}` failed: {err}", call.function.name);
                    (emit)(EventKind::Error, task_id, msg.clone());
                    append_message_and_persist(
                        &mut messages,
                        ChatMessage::tool_result(call.id, format!("ERROR: {msg}")),
                        persistent_session.as_deref(),
                    )?;
                }
            }
        }

        Ok(AgentQuantumResult {
            tool_session,
            outcome: AgentQuantumOutcome::Continue {
                needs_finish_reminder: false,
                assistant_text,
            },
        })
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
                &runtime,
                agents_md,
                join_extra_system_prompt(
                    extra_system_prompt,
                    session_compaction_context.as_deref(),
                )
                .as_deref(),
            ),
        );

        // 5.3 Read and inject checkpoint from previous run (if exists).
        // The checkpoint file is consumed (deleted) after reading so it only
        // applies to the first run after a crash/interrupt.
        {
            let checkpoint_path = self.tools.ctx.workspace_root.join("runtime").join("checkpoint.md");
            if checkpoint_path.exists() {
                match tokio::fs::read_to_string(&checkpoint_path).await {
                    Ok(content) => {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            let sys_content = system_message.content.as_mut();
                            if let Some(MessageContent::Text(sys_text)) = sys_content {
                                sys_text.push_str("\n\n## 恢复检查点\n\n以下是上次任务中断前保存的检查点信息，请据此继续任务：\n\n");
                                sys_text.push_str(trimmed);
                            }
                        }
                        let _ = tokio::fs::remove_file(&checkpoint_path).await;
                        (emit)(
                            EventKind::Log,
                            task_id,
                            "已加载并清除上次运行的检查点".to_string(),
                        );
                    }
                    Err(e) => {
                        (emit)(
                            EventKind::Error,
                            task_id,
                            format!("读取检查点失败: {e}"),
                        );
                    }
                }
            }
        }

        // Store only the real conversation here. Synthetic compaction summaries
        // are injected later when we build the provider request.
        append_message_and_persist(
            &mut messages,
            ChatMessage::text("user", task.clone()),
            persistent_session.as_deref(),
        )?;

        // Each run gets its own session state.
        //
        // This is important for the `Edit` guardrail: "must `Read` before
        // `Edit`" is enforced per agent/sub-agent session.
        let mut tool_session = ToolSession::default();

        // Consecutive model-call failures for the current model.
        // This drives the infinite retry backoff schedule.
        let mut model_error_count: u32 = 0;
        // Total retries across all models in this task. Used for the
        // global retry cap (max_retries).
        let mut total_retries: u32 = 0;

        // Step loop.
        let mut step: u64 = 0;
        let mut last_tool_step: u64 = 0;
        loop {
            step = step.saturating_add(1);
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

            // 5.2 External intervention file injection: check
            //    workspace/runtime/intervene.md and workspace/runtime/stop
            //    at the start of each turn. These files are consumed (deleted)
            //    after reading so they act as one-shot signals.
            {
                let rt_dir = self.tools.ctx.workspace_root.join("runtime");
                let stop_path = rt_dir.join("stop");
                let intervene_path = rt_dir.join("intervene.md");

                if stop_path.exists() {
                    let _ = tokio::fs::remove_file(&stop_path).await;
                    let msg = "[外部 STOP] 收到停止信号，任务中止。".to_string();
                    (emit)(EventKind::Error, task_id, msg.clone());
                    (emit)(EventKind::Final, task_id, msg.clone());
                    return Ok(msg);
                }

                if intervene_path.exists() {
                    match tokio::fs::read_to_string(&intervene_path).await {
                        Ok(content) => {
                            let _ = tokio::fs::remove_file(&intervene_path).await;
                            let inject =
                                format!("[外部干预] {content}");
                            append_message_and_persist(
                                &mut messages,
                                ChatMessage::text("user", &inject),
                                persistent_session.as_deref(),
                            )?;
                        }
                        Err(e) => {
                            (emit)(
                                EventKind::Error,
                                task_id,
                                format!("读取 intervene.md 失败: {e}"),
                            );
                        }
                    }
                }
            }

            // Periodic injection for long-running tasks (Phase 3.4).
            if step > 0 && step % 7 == 0 {
                let warning = ChatMessage::text(
                    "user",
                    "[System] 禁止无效重试，必须切换策略。现在已执行 {step} 轮，如果当前方法不奏效，请尝试不同方法。",
                );
                append_message_and_persist(
                    &mut messages,
                    warning,
                    persistent_session.as_deref(),
                )?;
            }
            if step > 0 && step % 10 == 0 {
                let reminder = ChatMessage::text(
                    "user",
                    "[System] 已执行 {step} 轮。请回顾任务目标，确认当前方向正确，必要时使用 memory 工具重新上下文化。",
                );
                append_message_and_persist(
                    &mut messages,
                    reminder,
                    persistent_session.as_deref(),
                )?;
            }

            match maybe_compact_history(
                &self.llm,
                runtime.model_override.as_deref().unwrap_or(&self.cfg.model),
                &system_message.role,
                &system_message,
                runtime
                    .effort_override
                    .as_deref()
                    .or(self.cfg.reasoning_effort.as_deref()),
                &mut compaction_state,
                &mut messages,
                &self.tools.tool_definitions(&runtime),
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
                                &runtime,
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
                format!("Step {step}: calling model..."),
            );

            // 5.8 Expand {{file:...}} references in messages before building request.
            let workspace_root = self.tools.ctx.workspace_root.clone();
            expand_file_refs(std::slice::from_mut(&mut system_message), &workspace_root);
            expand_file_refs(&mut messages, &workspace_root);

            // Build the request.
            let mut req = ChatCompletionsRequest {
                model: runtime
                    .model_override
                    .clone()
                    .unwrap_or_else(|| self.cfg.model.clone()),
                messages: build_request_messages(&system_message, &compaction_state, &messages),
                max_tokens: self.cfg.max_output_tokens,
                reasoning_effort: runtime
                    .effort_override
                    .clone()
                    .or_else(|| self.cfg.reasoning_effort.clone()),
                tools: Some(self.tools.tool_definitions(&runtime)),
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
                temperature: self.cfg.temperature,
                top_p: self.cfg.top_p,
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
                        total_retries = total_retries.saturating_add(1);

                        // 5.7 Quota exhausted: skip retry delay and either
                        //     failover immediately or terminate the task.
                        if err.is_quota_exhausted() {
                            let fallback_available = self.cfg.fallback_model.is_some()
                                && req.model != *self.cfg.fallback_model.as_ref().unwrap();
                            if fallback_available {
                                let fallback = self.cfg.fallback_model.as_ref().unwrap().clone();
                                (emit)(
                                    EventKind::Log,
                                    task_id,
                                    format!(
                                        "配额耗尽，自动切换至 fallback 模型: {fallback}"
                                    ),
                                );
                                req.model = fallback;
                                model_error_count = 0;
                                continue;
                            }
                            // No fallback or already on fallback: terminate.
                            let msg = format!(
                                "配额耗尽，请切换模型：{err}"
                            );
                            (emit)(EventKind::Error, task_id, msg.clone());
                            (emit)(EventKind::Final, task_id, msg.clone());
                            return Ok(msg);
                        }

                        // 5.8 Total retry cap: prevent infinite retries.
                        if total_retries >= self.cfg.max_retries.max(1) {
                            let msg = format!(
                                "达到最大重试次数上限（{}/{}），任务终止：{err}",
                                total_retries, self.cfg.max_retries
                            );
                            (emit)(EventKind::Error, task_id, msg.clone());
                            (emit)(EventKind::Final, task_id, msg.clone());
                            return Ok(msg);
                        }

                        // 5.6 Multi-model failover: switch to fallback model after
                        // consecutive failures exceed the configured threshold.
                        if model_error_count >= self.cfg.max_consecutive_failures.max(1)
                            && self.cfg.fallback_model.is_some()
                            && req.model != *self.cfg.fallback_model.as_ref().unwrap()
                        {
                            let fallback = self.cfg.fallback_model.as_ref().unwrap().clone();
                            (emit)(
                                EventKind::Log,
                                task_id,
                                format!(
                                    "模型 {model_error_count} 次连续失败，自动切换至 fallback 模型: {fallback}"
                                ),
                            );
                            req.model = fallback;
                            model_error_count = 0;
                            // Retry immediately with the fallback model (no delay).
                            continue;
                        }

                        let delay = retry_delay(model_error_count);

                        // Show a friendly message on the first retry so the user
                        // sees "正在重试" instead of multiple raw error logs.
                        if model_error_count == 1 {
                            (emit)(
                                EventKind::Log,
                                task_id,
                                "⏳ 模型调用遇到限流，正在自动重试...".to_string(),
                            );
                        }

                        (emit)(
                            EventKind::Error,
                            task_id,
                            format!(
                                "Model call failed (retryable; count={model_error_count}; total_retries={total_retries}; next_retry_in={:?}): {err}",
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

            // Emit context usage info after successful LLM response.
            self.emit_context_info(&system_message, &compaction_state, &messages, &runtime, task_id);

            // Emit assistant content (if present).
            let assistant_text = assistant.text_content();
            if let Some(ref content) = assistant_text {
                // If LLM returned tool calls, the text is intermediate reasoning (→ Log);
                // if no tool calls, the text is the final reply (→ Message, rendered in chat).
                let has_tools = !assistant.tool_calls.as_ref().map_or(true, |v| v.is_empty());
                let kind = if has_tools { EventKind::Log } else { EventKind::Message };
                (emit)(kind, task_id, content.clone());
            }

            // Extract tool calls (if any).
            let tool_calls: Vec<ToolCall> = assistant.tool_calls.clone().unwrap_or_default();

            // Persist assistant message in history (only if it has content or tool calls).
            let has_content = assistant_text.is_some();
            if has_content || !tool_calls.is_empty() {
                append_message_and_persist(&mut messages, assistant, persistent_session.as_deref())?;
            }

            // 5.1 <summary> protocol: detect missing summary tag and inject warning.
            // Runs every turn that has non-empty assistant text, regardless of tool calls.
            if let Some(ref text) = assistant_text {
                if !text.trim().is_empty() {
                    let summary_re =
                        regex::Regex::new(r"<summary>\s*(.*?)\s*</summary>").unwrap();
                    if !summary_re.is_match(text) {
                        let warning = ChatMessage::text(
                            "user",
                            "[DANGER] 上一轮回复遗漏了 <summary>做什么/为什么</summary> 标签。请在后续回复中严格遵守格式要求。",
                        );
                        append_message_and_persist(
                            &mut messages,
                            warning,
                            persistent_session.as_deref(),
                        )?;
                    }
                }
            }

            // no_tool fallback: if LLM returns no tool calls, inject retry prompts.
            if tool_calls.is_empty() {
                let no_tool_count = step.saturating_sub(last_tool_step);

                // Determine the appropriate injection.
                let injection = if no_tool_count >= 3 {
                    // 3 consecutive no_tool — report and pause.
                    let msg = format!(
                        "[System] 连续 {no_tool_count} 轮未调用工具，系统已暂停。请检查任务进度。"
                    );
                    (emit)(EventKind::Error, task_id, msg.clone());
                    return Ok(msg);
                } else {
                    match assistant_text.as_deref() {
                        Some(text) if text.trim().is_empty() => {
                            // Empty response — prompt to use tools.
                            "[System] 请调用工具来完成任务。不要只输出文字，需要执行实际操作（如 read_file、bash 等）。".to_string()
                        }
                        Some(text) if text.len() < 50 && !text.contains("完成") => {
                            // Short response that doesn't claim completion.
                            "[System] 请继续执行任务。使用工具完成实际操作，不要只描述计划。".to_string()
                        }
                        _ => {
                            // Had content but no tools — inject verification prompt.
                            "[System] 上一轮的输出未包含工具调用。如果任务已完成，请使用 Finish 工具确认；如果未完成，请使用工具继续执行。".to_string()
                        }
                    }
                };

                let retry_msg = ChatMessage::text("user", &injection);
                append_message_and_persist(&mut messages, retry_msg, persistent_session.as_deref())?;
                continue;
            }

            // Turn counter: track the last step that had tool calls.
            last_tool_step = step;

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
                let exec_result = self
                    .tools
                    .execute(
                        &mut tool_session,
                        &runtime,
                        &call.function.name,
                        args_json,
                        cancel,
                    )
                    .await;

                match exec_result {
                    Ok(ToolExecutionResult::Observation(output)) => {
                        // Emit tool output (can be JSON).
                        (emit)(EventKind::Tool, task_id, output.clone());
                        // Append tool result message.
                        append_message_and_persist(
                            &mut messages,
                            ChatMessage::tool_result(call.id, output),
                            persistent_session.as_deref(),
                        )?;
                    }
                    Ok(ToolExecutionResult::ImagePayload { .. }) => {
                        // Image payload: inject as a multimodal user message.
                        let (tool_msg, user_msg) =
                            build_image_payload_messages(&call.id, exec_result.as_ref().unwrap());
                        let b64_len = match exec_result.as_ref().unwrap() {
                            ToolExecutionResult::ImagePayload { data_url, .. } => data_url.len(),
                            _ => 0,
                        };
                        (emit)(
                            EventKind::Tool,
                            task_id,
                            format!(
                                "ImageAnalyze: image loaded (base64 {b64_len} bytes), injecting multimodal user message"
                            ),
                        );
                        append_message_and_persist(
                            &mut messages,
                            tool_msg,
                            persistent_session.as_deref(),
                        )?;
                        append_message_and_persist(
                            &mut messages,
                            user_msg,
                            persistent_session.as_deref(),
                        )?;
                    }
                    Ok(ToolExecutionResult::Control(control)) => match control {
                        ToolControl::Ask(_request) => {
                            let msg = "Ask requested in the legacy runner, but only the durable runtime may suspend for structured questions.".to_string();
                            (emit)(EventKind::Error, task_id, msg.clone());
                            append_message_and_persist(
                                &mut messages,
                                ChatMessage::tool_result(call.id, format!("ERROR: {msg}")),
                                persistent_session.as_deref(),
                            )?;
                        }
                        ToolControl::Finish(request) => {
                            // 5.4 Adversarial quality review before finalizing.
                            if !messages.is_empty() {
                                let recent: Vec<_> = messages
                                    .iter()
                                    .rev()
                                    .take(6)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .into_iter()
                                    .rev()
                                    .collect();
                                let model = runtime
                                    .model_override
                                    .as_deref()
                                    .unwrap_or(&self.cfg.model);
                                match crate::adversary::run_adversary_review(
                                    &self.llm,
                                    model,
                                    &recent,
                                )
                                .await
                                {
                                    Ok(review) if review.score < 60 => {
                                        let issues = review.issues.join("; ");
                                        (emit)(
                                            EventKind::Error,
                                            task_id,
                                            format!(
                                                "[质量警告] 低质量输出 (score={}): {issues}",
                                                review.score
                                            ),
                                        );
                                    }
                                    Ok(review) => {
                                        (emit)(
                                            EventKind::Log,
                                            task_id,
                                            format!(
                                                "[质量审查] 通过 (score={})",
                                                review.score
                                            ),
                                        );
                                    }
                                    Err(e) => {
                                        (emit)(
                                            EventKind::Error,
                                            task_id,
                                            format!("[质量审查] 审查失败: {e}"),
                                        );
                                    }
                                }
                            }

                            let final_text = format!(
                                "WORK_FINISH\nreason: {}\nresult: {}",
                                request.reason.trim(),
                                request.result.trim()
                            );
                            (emit)(EventKind::Final, task_id, final_text.clone());
                            return Ok(final_text);
                        }
                        ToolControl::FinishWithoutOutput => {
                            // 5.4 Adversarial quality review before finalizing.
                            if !messages.is_empty() {
                                let recent: Vec<_> = messages
                                    .iter()
                                    .rev()
                                    .take(6)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .into_iter()
                                    .rev()
                                    .collect();
                                let model = runtime
                                    .model_override
                                    .as_deref()
                                    .unwrap_or(&self.cfg.model);
                                match crate::adversary::run_adversary_review(
                                    &self.llm,
                                    model,
                                    &recent,
                                )
                                .await
                                {
                                    Ok(review) if review.score < 60 => {
                                        let issues = review.issues.join("; ");
                                        (emit)(
                                            EventKind::Error,
                                            task_id,
                                            format!(
                                                "[质量警告] 低质量输出 (score={}): {issues}",
                                                review.score
                                            ),
                                        );
                                    }
                                    Ok(review) => {
                                        (emit)(
                                            EventKind::Log,
                                            task_id,
                                            format!(
                                                "[质量审查] 通过 (score={})",
                                                review.score
                                            ),
                                        );
                                    }
                                    Err(e) => {
                                        (emit)(
                                            EventKind::Error,
                                            task_id,
                                            format!("[质量审查] 审查失败: {e}"),
                                        );
                                    }
                                }
                            }

                            let final_text = "WORK_FINISH_WITHOUT_OUTPUT".to_string();
                            (emit)(EventKind::Final, task_id, final_text.clone());
                            return Ok(final_text);
                        }
                        ToolControl::Wait(request) => {
                            let msg = format!(
                                "Wait requested for {:?} {} until {:?}, but the legacy runner has not been upgraded to durable waiting yet.",
                                request.kind, request.id, request.until
                            );
                            (emit)(EventKind::Error, task_id, msg.clone());
                            append_message_and_persist(
                                &mut messages,
                                ChatMessage::tool_result(call.id, format!("ERROR: {msg}")),
                                persistent_session.as_deref(),
                            )?;
                        }
                    },
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
                        append_message_and_persist(
                            &mut messages,
                            ChatMessage::tool_result(call.id, format!("ERROR: {msg}")),
                            persistent_session.as_deref(),
                        )?;
                    }
                }
            }

            // Check if current active skill has verify loop enabled.
            // This is a simplified implementation - in a full implementation,
            // we would need to track the active skill and its orchestration config.
            // For now, we'll just emit a log message indicating verify loop support.
            if step > 0 && step % 5 == 0 {
                (emit)(
                    EventKind::Log,
                    task_id,
                    format!("Step {step}: verify loop check point (feature placeholder)"),
                );
            }
        }
    }

    /// Build the stable instructions block injected into the prompt.
    fn build_system_prompt(
        &self,
        runtime: &ToolRuntime,
        agents_md: &AgentsMd,
        extra_system_prompt: Option<&str>,
    ) -> String {
        use std::fmt::Write as _;

        let now = chrono::Local::now();

        let base_template = match runtime.prompt_profile {
            PromptProfile::Root => STATIC_PROMPT_TEMPLATE,
            PromptProfile::SubAgent => STATIC_SUBAGENT_PROMPT_TEMPLATE,
            PromptProfile::Background => STATIC_PROMPT_TEMPLATE,
        };

        let mut out = String::from(base_template.trim_end());
        out.push_str("\n\n");
        out.push_str(&self.build_tool_prompt_block(runtime));
        out.push('\n');

        let visible_commands = self.skills.list_model_invocable(
            runtime
                .activated_conditional_commands
                .iter()
                .map(String::as_str),
            &runtime.denied_commands,
        );

        if !visible_commands.is_empty() {
            out.push_str("## 技能授权\n\n");
            out.push_str("所有已注册技能都已经过授权，可以按需使用。同学的任务如果明显需要某项技能，就直接用 `Skill` 读取它，不要凭空编造\"策略限制\"来回避。\n\n");

            out.push_str("## 可用技能\n\n");
            out.push_str("技能是保存在本地目录中的说明包，每个技能目录至少包含一个 `SKILL.md`。\n");
            out.push_str("当某项技能与你的任务相关时，先用 `Skill` 读取该技能的 `SKILL.md`，再按其中引用的相对路径继续读取技能内文件。\n");
            out.push_str(
                "你不会看到技能在宿主机上的真实安装目录；只能通过技能名和技能内相对路径访问。\n\n",
            );

            for item in visible_commands {
                let _ = write!(
                    out,
                    "- `{}`：{}",
                    item.name,
                    normalize_skill_description(&item.description)
                );
                if let Some(when_to_use) = item.when_to_use.as_deref() {
                    let _ = write!(out, "；适用时机：{}", when_to_use.trim());
                }
                out.push('\n');
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

        let _ = writeln!(
            out,
            "## 运行时\n\n模型：`{}`\n",
            runtime.model_override.as_deref().unwrap_or(&self.cfg.model)
        );

        // 5.5 Environment info injection: OS, shell, available runtimes.
        out.push_str("## 环境信息\n\n");
        out.push_str(&crate::skill_search::detect_environment_info());
        out.push('\n');
        out.push('\n');

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

    /// Build the runtime-accurate tool guidance block.
    ///
    /// This block is generated from the same tool schema that is sent to the
    /// model, so prompt-visible capabilities stay aligned with actual
    /// executable capabilities.
    fn build_tool_prompt_block(&self, runtime: &ToolRuntime) -> String {
        use std::fmt::Write as _;

        let definitions = self.tools.tool_definitions(runtime);
        let mut out = String::from(
            "## 工具\n\n以下是你当前这一轮真实可用的工具。只能依赖这里列出的工具，不要假设隐藏工具依然可用。\n\n",
        );

        for definition in &definitions {
            let _ = writeln!(
                out,
                "- `{}`：{}",
                definition.function.name,
                definition.function.description.trim()
            );
        }

        let tool_names = definitions
            .iter()
            .map(|definition| definition.function.name.as_str())
            .collect::<Vec<_>>();

        if tool_names
            .iter()
            .any(|name| matches!(*name, "Send" | "Ask" | "Show"))
        {
            out.push_str(
                "\n### 交互最佳实践\n\n- 简短进度、简短回答，用发送型工具。\n- 需要同学选择或补充关键信息时，再使用提问型工具。\n- 高信息密度内容优先通过展示文件的方式给同学看。\n",
            );
        } else if runtime.parent_agent_id.is_some() {
            out.push_str(
                "\n### 交互限制\n\n- 你当前没有直接面向同学的交互权限。\n- 你的普通文本不会直接显示给同学。\n- 如需汇报、同步或交接，请优先使用父代理/代理间协作工具。\n",
            );
        }

        if tool_names.iter().any(|name| *name == "SubAgent") {
            out.push_str(
                "\n### 子代理协作\n\n- 只有当子任务边界清晰、值得隔离上下文时，才继续派生子代理。\n- 传入子代理的上下文必须具体、自包含、可验证。\n",
            );
        }

        if tool_names.iter().any(|name| name.starts_with("mcp__")) {
            out.push_str(
                "\n### MCP 工具\n\n- 名称形如 `mcp__server__tool` 的工具来自外部 MCP server。\n- 它们和内置工具一样可直接调用，但参数必须严格遵守对应 schema。\n",
            );
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
    let merged_extra_prompt =
        join_extra_system_prompt(extra_system_prompt, Some(refresh_extra_context.as_str()));

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
        // 压缩刷新子代理 Final 冗余，各步骤已作为 Log emit
        if matches!(kind, EventKind::Final) {
            return;
        }
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
        &descriptor,
        &final_text,
    )))
}

/// 轻量记忆提取 — 任务完成后调用。
///
/// 仅在本次任务未触发 compaction 时使用（compaction 已有自己的记忆刷新）。
/// 通过 `run_task_without_memory_refresh` 执行，避免嵌套触发 compaction。
pub async fn post_task_memory_collect(
    runner: &AgentRunner,
    task_id: Uuid,
    agents_md: &AgentsMd,
    extra_system_prompt: Option<&str>,
    session_store: &Arc<SessionStore>,
    cancel: &CancelToken,
    emit: &EmitEventFn,
    task_summary: &str,
) -> anyhow::Result<()> {
    let snapshot = session_store
        .load_snapshot()
        .context("Failed to load session snapshot for post-task memory collect")?;
    let descriptor = snapshot.descriptor;

    let extract_task = format!(
        r#"请从以下对话中提取关键信息，写入语义记忆系统。

任务摘要：{task_summary}

请使用 MemoryWrite 工具记录：
1. 本次会话的主题（subject: "session_topic", predicate: "is_about"）
2. 解决的问题或完成的任务（如有）
3. 遇到的错误和解决方案（如有）
4. 可复用的工作模式（如有）

**严格排除以下内容（不得写入记忆）：**
- agent 对自身能力的评价（如"我的缓存机制很好""我的记忆功能完善"等自我评估）
- agent 的元评论或自我反思（如"我认为我的表现..."）
- 主观判断而非客观事实（如"用户满意度高"）
- 无法在后续会话中验证或复用的主观描述

只记录客观、可验证、可复用的事实信息。每条记忆的 confidence 设为 0.8。
如果没有值得记忆的内容，直接回复"无需记录"。

当前会话文件：`{}`"#,
        descriptor.current_session_path
    );

    let runtime = build_internal_memory_refresh_runtime(
        runner,
        task_id,
        agents_md,
        extra_system_prompt,
        emit,
        0,
    );
    let emit_for_collect = Arc::clone(emit);
    let collect_emit: EmitEventFn = Arc::new(move |kind, task_id, message| {
        if matches!(kind, EventKind::Final) {
            return;
        }
        (emit_for_collect)(kind, task_id, format!("[memory-collect] {message}"));
    });

    (emit)(
        EventKind::Log,
        task_id,
        "Starting post-task memory collection...".to_string(),
    );

    let _final_text = Box::pin(runner.run_task_without_memory_refresh(
        task_id,
        extract_task,
        agents_md,
        extra_system_prompt,
        None,
        runtime,
        cancel,
        collect_emit,
    ))
    .await
    .context("Post-task memory collection sub-agent failed")?;

    (emit)(
        EventKind::Log,
        task_id,
        "Post-task memory collection completed.".to_string(),
    );

    Ok(())
}

/// Build the compact-triggered memory-refresh task text.
fn build_compaction_memory_refresh_task(descriptor: &crate::session::SessionDescriptor) -> String {
    format!(
        "执行一次 compact 后的记忆整理。目标不是总结，而是把刚压缩掉的历史信号提炼进长期记忆。\
\n\n工作要求：\
\n- 优先检查 `MEMORY.md`、`memory/topics/*.md` 与最近原始记录，避免重复。\
\n- 必要时阅读最近的原始会话段，尤其是刚被 compact 掉的历史。\
\n- 只提升稳定、可复用、高价值的信息。\
\n- **严格排除**：agent 对自身能力的评价、自我反思、主观判断。只记录客观事实和可复用模式。\
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
    out.push_str(
        "目标是把刚压缩掉的高价值信息提炼进长期记忆，并减少它只停留在原始会话文件中的时间。\n\n",
    );
    let _ = writeln!(
        out,
        "- 当前压缩后会话文件：`{}`",
        descriptor.current_session_path
    );
    let _ = writeln!(
        out,
        "- 上一段原始会话文件：`{}`",
        descriptor
            .previous_session_path
            .as_deref()
            .unwrap_or("(none)")
    );
    let _ = writeln!(
        out,
        "- compact 后当前会话中保留的真实消息数：{}",
        kept_messages.len()
    );
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
    use crate::runtime::state::AgentStatus;
    use crate::tools::{
        AgentInfo, AskQuestionFn, BroadcastAgentsFn, GetAgentFn, GetTaskFn, ListAgentsFn,
        MAX_SUBAGENT_DEPTH, MessageAgentFn, NotifyParentFn, PromptProfile, RunSubAgentFn,
        SendMessageFn, ShowFileFn, StartTerminalTaskFn, SubAgentHandle, SubAgentRequest,
        ToolRuntime, TransferInputFn,
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

            Box::pin(runner.run_task_inner(
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

            Ok(SubAgentHandle {
                agent_id: Uuid::new_v4(),
                work_id: Uuid::new_v4(),
                label: request
                    .label
                    .unwrap_or_else(|| format!("memory-refresh-depth-{next_depth}")),
                status: AgentStatus::Idle,
            })
        })
    });
    let notify_parent: NotifyParentFn = Arc::new(move |_message| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use NotifyParent") })
    });
    let message_agent: MessageAgentFn = Arc::new(move |_request| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use MessageAgent") })
    });
    let broadcast_agents: BroadcastAgentsFn = Arc::new(move |_request| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use BroadcastAgents") })
    });
    let list_agents: ListAgentsFn =
        Arc::new(move |_request| Box::pin(async move { Ok(Vec::<AgentInfo>::new()) }));
    let get_agent: GetAgentFn = Arc::new(move |_agent_id| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use GetAgent") })
    });
    let transfer_input: TransferInputFn = Arc::new(move |_request| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use TransferInput") })
    });
    let start_terminal_task: StartTerminalTaskFn = Arc::new(move |_request, _cancel| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use background Bash") })
    });
    let get_task: GetTaskFn = Arc::new(move |_task_id| {
        Box::pin(async move { anyhow::bail!("memory refresh must not use GetTask") })
    });
    let reload_runtime: ReloadRuntimeFn =
        Arc::new(|| Box::pin(async { anyhow::bail!("memory refresh must not use Reload") }));

    ToolRuntime::new(
        task_id,
        None,
        task_id,
        format!("memory-refresh-depth-{depth}"),
        PromptProfile::Background,
        false,
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

/// Given an `ImagePayload` tool result, build a multimodal **user** message that
/// carries the image so the LLM can analyse it on the next turn.
///
/// The function returns two messages:
/// 1. A `tool` role message (so the model sees the tool completed).
/// 2. A `user` role message with `ContentPart::Image` (so the model sees the image).
fn build_image_payload_messages(
    tool_call_id: &str,
    payload: &ToolExecutionResult,
) -> (ChatMessage, ChatMessage) {
    let text_summary = if let ToolExecutionResult::ImagePayload {
        data_url,
        media_type,
        prompt,
    } = payload
    {
        let analysis_prompt = prompt
            .as_deref()
            .unwrap_or("Please analyse this image in detail.");
        format!(
            "Image loaded successfully.\n\
             - Media type: {media_type}\n\
             - Size: {} bytes (base64)\n\
             - The image has been injected into the conversation for visual analysis.\n\
             - Analysis request: {analysis_prompt}",
            data_url.len()
        )
    } else {
        unreachable!("build_image_payload_messages called with non-ImagePayload")
    };

    let tool_msg = ChatMessage::tool_result(tool_call_id, &text_summary);

    // Build a multimodal user message carrying the image.
    if let ToolExecutionResult::ImagePayload {
        data_url,
        prompt,
        ..
    } = payload
    {
        let analysis_prompt = prompt
            .as_deref()
            .unwrap_or("Please analyse this image in detail.");
        let parts = vec![
            ContentPart::Text {
                text: format!("[Image injected for analysis] {analysis_prompt}"),
            },
            ContentPart::Image {
                image_url: ImageUrl {
                    url: data_url.clone(),
                    detail: Some("auto".to_string()),
                },
            },
        ];
        let user_msg = ChatMessage::multimodal("user", parts);
        (tool_msg, user_msg)
    } else {
        unreachable!()
    }
}

/// Append one real message to the in-memory conversation and, when enabled,
/// mirror it into the durable JSONL session segment.
fn append_message_and_persist(
    messages: &mut Vec<ChatMessage>,
    message: ChatMessage,
    persistent_session: Option<&SessionStore>,
) -> anyhow::Result<()> {
    // Guard: providers reject assistant messages that carry none of
    // `content` or `tool_calls`.  This can happen when the model returns a
    // "stop" finish reason without producing any actual content.  Rather than
    // let an unserialisable message enter the history (which would poison
    // every subsequent request), we give it a minimal placeholder content so
    // the conversation structure stays valid.
    if message.role == "assistant" {
        let has_content = message.content.as_ref().is_some_and(|c| match c {
            MessageContent::Text(s) => !s.is_empty(),
            MessageContent::Parts(p) => !p.is_empty(),
        });
        let has_tool_calls = message.tool_calls.as_ref().is_some_and(|t| !t.is_empty());

        if !has_content && !has_tool_calls {
            let mut fixed = message;
            fixed.content = Some(MessageContent::Text(" ".to_string()));
            if let Some(session_store) = persistent_session {
                session_store
                    .append_message(&fixed)
                    .context("Failed to append message to persisted session")?;
            }
            messages.push(fixed);
            return Ok(());
        }
    }

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
    use crate::session::{SessionDescriptor, SessionSnapshot};
    use crate::skills::SkillRegistry;
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
                compaction: CompactionConfig::default(),
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                fallback_model: None,
                max_consecutive_failures: 2,
                max_retries: 12,
                model_routing: None,
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

    /// Resolve the repository-level sub-agent prompt source file.
    fn subagent_prompt_md_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("SubAgents.md")
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
        let prompt = runner.build_system_prompt(&ToolRuntime::detached(), &test_agents_md(), None);
        assert!(prompt.starts_with(static_prompt.trim_end()));
    }

    /// Dynamic sections still need to be appended after the static prompt.
    #[test]
    fn system_prompt_appends_runtime_sections_after_static_prompt() {
        let runner = test_runner();
        let prompt = runner.build_system_prompt(
            &ToolRuntime::detached(),
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

    #[test]
    fn system_prompt_hides_interaction_tools_when_runtime_lacks_permissions() {
        let runner = test_runner();
        let mut runtime = ToolRuntime::detached();
        runtime.allow_user_send = false;
        runtime.allow_user_show = false;
        runtime.allow_user_ask = false;

        let prompt = runner.build_system_prompt(&runtime, &test_agents_md(), None);

        assert!(!prompt.contains("`Send`"));
        assert!(!prompt.contains("`Show`"));
        assert!(!prompt.contains("`Ask`"));
    }

    #[test]
    fn subagent_prompt_starts_with_subagent_md_static_source() {
        let static_prompt =
            fs::read_to_string(subagent_prompt_md_path()).expect("read repository SubAgents.md");

        let runner = test_runner();
        let mut runtime = ToolRuntime::detached();
        runtime.prompt_profile = crate::tools::PromptProfile::SubAgent;
        runtime.allow_user_send = false;
        runtime.allow_user_show = false;
        runtime.allow_user_ask = false;

        let prompt = runner.build_system_prompt(&runtime, &test_agents_md(), None);
        assert!(prompt.starts_with(static_prompt.trim_end()));
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
        let text = notice.text_content().unwrap_or_default();
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
            prompt: None,
            prompt_file: None,
            prompt_skill: None,
            allow_user_send: false,
            allow_user_show: false,
            allow_user_ask: false,
            allow_input_transfer_target: false,
            existing_agent_id: None,
        };
        let context = build_internal_memory_refresh_subagent_context(
            "## Base\n\n- compact 后上下文",
            &request,
            2,
        );

        assert!(context.contains("Parent-provided Memory Refresh SubAgent Context"));
        assert!(context.contains("所有后代子代理都不得使用 `Ask`、`Send` 或 `Show`"));
        assert!(context.contains("聚焦 testing 偏好"));
        assert!(context.contains("depth: 2"));
    }
}
