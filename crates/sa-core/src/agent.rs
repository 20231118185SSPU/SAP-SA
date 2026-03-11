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
use crate::skills::SkillRegistry;
use crate::tools::{ToolExecutor, ToolRuntime, ToolSession};
use crate::ws_protocol::EventKind;
use anyhow::Context as _;
use std::sync::Arc;
use uuid::Uuid;

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
        runtime: ToolRuntime,
        cancel: &CancelToken,
        drain_queued_user_messages: Option<DrainQueuedUserMessagesFn>,
        emit: EmitEventFn,
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

        // Build the "system prompt" (or "developer prompt") that stays constant.
        let system_prompt = self.build_system_prompt(agents_md, extra_system_prompt);

        // Keep the stable system/developer instruction block separate from the
        // mutable conversation history so compaction only touches the real
        // dialogue and never rewrites the base instructions.
        let system_message = ChatMessage::text(self.cfg.system_role_name.clone(), system_prompt);

        // Store only the real conversation here. Synthetic compaction summaries
        // are injected later when we build the provider request.
        let mut messages = vec![ChatMessage::text("user", task.clone())];

        // One task keeps one evolving checkpoint summary.
        let mut compaction_state = CompactionState::default();

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
                drain_queued_user_messages.as_ref(),
                &emit,
            );

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
            messages.push(assistant);

            // If no tool calls => done.
            if tool_calls.is_empty() {
                if drain_follow_up_messages(
                    task_id,
                    &mut messages,
                    drain_queued_user_messages.as_ref(),
                    &emit,
                ) {
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
                messages.push(ChatMessage::tool_result(call.id, tool_result));
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

        // Section 1: translated/adapted ZeroClaw-style framework prompt.
        let mut out = String::new();
        out.push_str("# StudyAdministrator (SA) 运行提示\n\n");
        out.push_str("## 工具\n\n");
        out.push_str(
            "你可以使用以下内置工具来完成任务：\n\
- `Read`：读取工作区内的 UTF-8 文本文件。\n\
- `Write`：创建一个全新的 UTF-8 文本文件；如果目标已存在则必须拒绝。\n\
- `Edit`：编辑已存在的 UTF-8 文本文件；同一轮代理会话里必须先 `Read`，才能 `Edit`。\n\
- `Bash`：通过 Git Bash 执行命令（`bash -lc`）。\n\
- `Search`：在不知道具体页面时先做网络搜索，拿到候选标题、摘要和 URL。\n\
- `Fetch`：对已知 URL 发起直接 HTTP 请求，获取正文或接口响应。\n\
- `MemorySearch`：按需搜索 `MEMORY.md`、`memory.md` 和 `memory/*.md`。\n\
- `MemoryGet`：读取某个记忆 Markdown 文件的具体片段。\n\
- `Send`：向同学发送简短消息，不阻塞等待回复。\n\
- `Show`：把一个已存在的文件直接展示给同学；适合高信息密度内容。\n\
- `Ask`：向同学发起结构化提问，并等待同学选择或输入。\n\
- `Skill`：按技能名读取 `SKILL.md` 或技能目录中的其他相对文件；真实宿主目录不会暴露给你。\n\
- `SubAgent`：启动子代理，传入父代理整理好的上下文，让子代理独立完成聚焦子任务。\n\n\
### Send / Ask / Show 最佳实践\n\n\
这三个工具是你和同学交流的主要方式。用好它们的关键是——**像一个真人同学会怎么发消息，你就怎么用**。\n\n\
**`Send` —— 随手发一条消息**\n\n\
Send 是最轻量的交流方式，相当于微信里发一条消息。遵循以下原则：\n\
- 一次只说一件事，一两句话就够。不要把长篇大论塞进一条 Send。\n\
- 该发就发，不用憋着攒到最后一起说。比如刚开始处理时说\"我看看\"，找到关键信息时说\"找到了，是这个原因\"，做完了说\"搞定了\"。\n\
- 不要用 Send 发送大段内容（代码、表格、长列表）——那些用 Show。\n\
- 不要用 Send 代替 Ask——如果你需要同学回答才能继续，用 Ask。\n\
- 语气自然、简短、口语化。不要用\"尊敬的同学\"这种生硬措辞。\n\n\
**`Ask` —— 需要同学回答才能继续**\n\n\
Ask 会阻塞等待回复，所以只在真正需要对方输入时才用：\n\
- 缺少关键信息无法继续时（\"这个作业是要求用递归还是迭代？\"）\n\
- 需要同学做选择时（提供明确选项）\n\
- 需要确认才能执行有风险的操作时\n\
- 不要用 Ask 来展示结果或汇报进度——那些用 Send 或 Show。\n\
- 不要把多个不相关的问题塞进一个 Ask——拆开问，或者只问最关键的那个。\n\
- 选项要简洁明了，不要让同学读半天才知道在问什么。\n\n\
**`Show` —— 把文件直接摆出来**\n\n\
Show 适合信息密度高、同学需要仔细看的内容：\n\
- 代码文件、文档、解题过程、生成的报告、长表格。\n\
- 使用模式：先用 Send 简短说明（\"这是改好的代码\"），然后 Show 文件。\n\
- 不要用 Show 发送一句话——那用 Send。\n\
- 不要在 Show 之前或之后再用 Send 把文件内容复述一遍。\n\n\
**组合使用的节奏**\n\n\
像发微信一样自然地组合：\n\
1. 同学问了个问题 → Send \"我查一下\" → （做调查）→ Send \"找到了\" → Show 结果文件\n\
2. 同学要你写代码 → Send \"好的\" → （写代码）→ Send \"写好了，你看看\" → Show 代码文件\n\
3. 同学的问题不够清楚 → Ask 具体问题（带选项）→ 拿到回答后继续\n\
4. 长任务进行中 → 中途 Send 进度更新 → 完成后 Send 总结 + Show 成果\n\n\
### SubAgent 最佳实践\n\n\
SubAgent 是保护主上下文窗口的利器。用不用子代理的判断标准很简单：**这个子任务的过程信息会不会把主上下文撑爆或弄脏？**\n\n\
**该用 SubAgent 的情况：**\n\
- 需要阅读大量文件来获得一个简短结论（如\"帮我看看这 10 个源文件里哪个定义了 X\"）\n\
- 需要做大量搜索和筛选（如\"在网上找到这个概念的权威解释\"）\n\
- 独立的、边界清晰的子任务（如\"把这段代码翻译成 Python\"）\n\
- 多个互不依赖的子任务需要并行快速完成（如同时搜索三个不同概念的定义、同时检查多个文件的状态）\n\n\
**不该用 SubAgent 的情况：**\n\
- 一次简单的文件读取或搜索——直接做就行\n\
- 任务上下文已经在主会话里，传给子代理反而要重新组装\n\n\
**传入子代理的上下文必须：**\n\
- 具体：明确说清楚要做什么、在哪里找、结果格式是什么\n\
- 自包含：子代理不应该需要再回头问主代理要信息\n\
- 可验证：主代理拿到结果后能判断子代理做得对不对\n\n\
如果工具列表里额外出现形如 `<server>__<tool>` 的工具名，那些是外部 MCP server 提供的工具。它们和内置工具一样可直接调用，但参数必须严格遵守对应工具的 schema。\n\n",
        );

        out.push_str("## 你的任务\n\n");
        out.push_str(
            "当同学发送消息时，直接理解需求并行动。需要执行命令、读写文件、联网获取资料、展示结果、向同学提问或委派子任务时，使用对应工具。\n\
对普通问题、追问、澄清或基于上下文可以直接回答的内容，用 `Send` 直接答复，不要要求同学重复已提供的信息。\n\
不要总结这份配置，不要复述你的能力清单，不要输出空泛的元评论，也不要把本应执行的动作退化成\"步骤建议\"。\n\
你的结论和行为必须满足：**可追溯（Traceable）**、**可验证（Verifiable）**、**可解释（Explainable）**。\n\
如果不确定，先调查再行动，禁止猜测。\n\n",
        );

        out.push_str("## 安全\n\n");
        out.push_str(
            "- 不要泄露私密数据、密钥、令牌、凭据或敏感配置。\n\
- 未经确认，不要执行破坏性命令，不要做不可逆的外部操作。\n\
- 不要绕过监督、审批或同学明确设置的限制。\n\
- 任何涉及修改文件、执行命令、联网取数的动作，都优先选择可验证、可恢复、可说明的方式。\n\
- 当外部动作存在明显风险或信息不足时，先 `Ask`，不要自作主张。\n\n",
        );

        out.push_str("## 记忆检索\n\n");
        out.push_str(
            "在回答与过去工作、历史决定、时间点、人物信息、同学偏好、约定事项或待办相关的问题前，优先检查工作区记忆。\n\
推荐流程：\n\
- 先用 `MemorySearch` 在 `MEMORY.md`、`memory.md`、`memory/*.md` 中搜索。\n\
- 如果搜索命中，再用 `MemoryGet` 只读取必要的文件片段，避免把整份记忆一次性塞进上下文。\n\
- 如果没有命中或证据不足，明确说明你查过但仍不确定，不要假装记得。\n\n",
        );

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

        out.push_str("## 交互与中断\n\n");
        out.push_str(
            "- 你运行在本地自治代理环境中，同学通过外部交互层向你发送任务、接收消息、查看文件和回答问题。\n\
- 你的普通文字回复默认视作不存在，不会直接显示给同学；只有 `Send`、`Ask` 和 `Show` 会进入对同学可见的交互层。需要让同学看到内容时，必须使用这些工具，通常优先用 `Send`。\n\
- `Show` 会把文件直接展示给同学，因此它比长篇普通文本更适合承载高密度信息。\n\
- 同学可能在你运行过程中继续发送新消息；这些消息通常会被排队，并在下一次模型请求前插入当前会话。只有同学显式中断时，你才会被取消。你的行为应该保持可中断、可恢复、可解释。\n\
- 如果工具输出包含敏感信息，也不要在面向同学的文本中重复它们。\n\n",
        );

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
    drain_queued_user_messages: Option<&DrainQueuedUserMessagesFn>,
    emit: &EmitEventFn,
) -> bool {
    let Some(drain) = drain_queued_user_messages else {
        return false;
    };

    let queued: Vec<String> = (drain)()
        .into_iter()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .collect();

    if queued.is_empty() {
        return false;
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
        messages.push(ChatMessage::text("user", text));
    }

    true
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
