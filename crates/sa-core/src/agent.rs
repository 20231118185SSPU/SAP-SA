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

/// Configuration for a single `AgentRunner`.
#[derive(Debug, Clone)]
pub struct AgentRunnerConfig {
    /// Model name.
    pub model: String,
    /// The role name used for "system instructions" (`system` or `developer`).
    pub system_role_name: String,
    /// Maximum tool-call steps per task.
    pub max_steps: u32,
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

        // Initialize conversation.
        let mut messages = Vec::<ChatMessage>::new();

        // NOTE: The user requested configurability for the role name here.
        messages.push(ChatMessage::text(
            self.cfg.system_role_name.clone(),
            system_prompt,
        ));

        // User task.
        messages.push(ChatMessage::text("user", task.clone()));

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

            (emit)(
                EventKind::Log,
                task_id,
                format!("Step {step}/{}: calling model...", self.cfg.max_steps),
            );

            // Build the request.
            let req = ChatCompletionsRequest {
                model: self.cfg.model.clone(),
                messages: messages.clone(),
                tools: Some(tool_definitions.clone()),
                tool_choice: Some(serde_json::json!("auto")),
                stream: Some(false),
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
            let choice = resp.first_choice()?;

            // Copy assistant message for our history.
            let assistant = choice.message.clone();

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
- `Send`：向用户发送简短消息，不阻塞等待回复。\n\
- `Show`：把一个已存在的文件直接展示给用户；适合高信息密度内容。\n\
- `Ask`：向用户发起结构化提问，并等待用户选择或输入。\n\
- `Skill`：按技能名读取 `SKILL.md` 或技能目录中的其他相对文件；真实宿主目录不会暴露给你。\n\
- `SubAgent`：启动子代理，传入父代理整理好的上下文，让子代理独立完成聚焦子任务。\n\n\
重要用法约定：\n\
- 当你不知道网址、文档入口或权威来源时，先用 `Search`，再用 `Fetch` 深入读取。\n\
- 当问题涉及过去做过什么、已有决定、日期、偏好、待办、长期约定时，先用 `MemorySearch`，再按需要用 `MemoryGet` 拉取精确片段。\n\
- `Send` 要言简意赅，只同步状态、结论、下一步或一个明确提醒，不要长篇铺陈。\n\
- `Show` 用于展示高密度信息，例如代码、文档、报告、表格、生成结果、长说明；先用一句简短 `Send` 告诉用户该看什么，再 `Show` 文件。\n\
- `Read` / `Edit` / `Write` 要遵守工作流：已存在文件先 `Read`，修改用 `Edit`，只有新文件才用 `Write`。\n\
- `Ask` 用于缺少关键信息、需要用户做选择、确认取舍，或需要结构化输入的情况。\n\
- `Skill` 的默认入口是 `SKILL.md`；如果 `SKILL.md` 引用了同技能目录下的其他相对文件，再继续用 `Skill` 读取这些相对路径。\n\
- `SubAgent` 只用于边界清晰、上下文可明确封装的子任务；传给子代理的上下文必须具体、可执行、可验证。\n\n",
        );

        out.push_str("## 你的任务\n\n");
        out.push_str(
            "当用户发送消息时，直接理解需求并行动。需要执行命令、读写文件、联网获取资料、展示结果、向用户提问或委派子任务时，使用对应工具。\n\
对普通问题、追问、澄清或基于上下文可以直接回答的内容，直接回答，不要要求用户重复已提供的信息。\n\
不要总结这份配置，不要复述你的能力清单，不要输出空泛的元评论，也不要把本应执行的动作退化成“步骤建议”。\n\
你的结论和行为必须满足：**可追溯（Traceable）**、**可验证（Verifiable）**、**可解释（Explainable）**。\n\
如果不确定，先调查再行动，禁止猜测。\n\n",
        );

        out.push_str("## 安全\n\n");
        out.push_str(
            "- 不要泄露私密数据、密钥、令牌、凭据或敏感配置。\n\
- 未经确认，不要执行破坏性命令，不要做不可逆的外部操作。\n\
- 不要绕过监督、审批或用户明确设置的限制。\n\
- 任何涉及修改文件、执行命令、联网取数的动作，都优先选择可验证、可恢复、可说明的方式。\n\
- 当外部动作存在明显风险或信息不足时，先 `Ask`，不要自作主张。\n\n",
        );

        out.push_str("## 记忆检索\n\n");
        out.push_str(
            "在回答与过去工作、历史决定、时间点、人物信息、用户偏好、约定事项或待办相关的问题前，优先检查工作区记忆。\n\
推荐流程：\n\
- 先用 `MemorySearch` 在 `MEMORY.md`、`memory.md`、`memory/*.md` 中搜索。\n\
- 如果搜索命中，再用 `MemoryGet` 只读取必要的文件片段，避免把整份记忆一次性塞进上下文。\n\
- 如果没有命中或证据不足，明确说明你查过但仍不确定，不要假装记得。\n\n",
        );

        if !self.skills.list().is_empty() {
            out.push_str("## 技能授权\n\n");
            out.push_str("所有已注册技能都已经过授权，可以按需使用。用户的任务如果明显需要某项技能，就直接用 `Skill` 读取它，不要凭空编造“策略限制”来回避。\n\n");

            out.push_str("## 可用技能\n\n");
            out.push_str("技能是保存在本地目录中的说明包，每个技能目录至少包含一个 `SKILL.md`。\n");
            out.push_str("当某项技能与你的任务相关时，先用 `Skill` 读取该技能的 `SKILL.md`，再按其中引用的相对路径继续读取技能内文件。\n");
            out.push_str(
                "你不会看到技能在宿主机上的真实安装目录；只能通过技能名和技能内相对路径访问。\n\n",
            );

            for item in self.skills.list() {
                let _ = writeln!(out, "- `{}`：{}", item.name, item.description);
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
            "- 你运行在本地自治代理环境中，用户通过外部交互层向你发送任务、接收消息、查看文件和回答问题。\n\
- 你的普通文字回复会作为最终答案返回；`Send` 则用于中途主动同步简短信息。\n\
- `Show` 会把文件直接展示给用户，因此它比长篇普通文本更适合承载高密度信息。\n\
- 用户可能随时发送新消息来打断当前任务并启动新的任务；你的行为应该保持可中断、可恢复、可解释。\n\
- 如果工具输出包含敏感信息，也不要在面向用户的文本中重复它们。\n\n",
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
