//! Minimal autonomous agent loop (tool-calling).
//!
//! This is the "heart" of the minimal Claw agent extracted from the large
//! `zeroclaw` codebase:
//! - We call an OpenAI-compatible Chat Completions endpoint.
//! - We provide tool definitions so the model can request actions.
//! - We execute those tools and feed results back to the model.
//! - We repeat until the model produces a final answer (no tool calls) or
//!   until `max_steps` is reached.
//!
//! The daemon (`claw-agentd`) owns task queues, event IDs, and WS connections.
//! This module is deliberately "pure core": it only needs an event callback.

use crate::agents_md::{format_agents_md_block, AgentsMd};
use crate::openai::{ChatCompletionsRequest, ChatMessage, OpenAiClient, ToolCall};
use crate::skills::SkillRegistry;
use crate::tools::ToolExecutor;
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
        emit: EmitEventFn,
    ) -> anyhow::Result<String> {
        // Build the "system prompt" (or "developer prompt") that stays constant.
        let system_prompt = self.build_system_prompt(agents_md);

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

        // Step loop.
        for step in 1..=self.cfg.max_steps {
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

            // Call provider.
            let resp = self.llm.chat_completions(&req).await?;
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
                let tool_result = match self.tools.execute(&call.function.name, args_json).await {
                    Ok(output) => output,
                    Err(err) => {
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
        Ok(msg)
    }

    /// Build the stable instructions block injected into the prompt.
    fn build_system_prompt(&self, agents_md: &AgentsMd) -> String {
        // Section 1: agents instructions.
        let mut out = String::new();
        out.push_str("# Claw minimal agent instructions\n\n");
        out.push_str(
            "You are an autonomous software agent running locally with tool access.\n\
Your goals are to be **traceable**, **verifiable**, and **explainable**:\n\
- Prefer concrete commands and file edits over vague descriptions.\n\
- When unsure, investigate using tools instead of guessing.\n\
- After actions, summarize what changed and how to verify.\n\n",
        );

        // Inject Agents.md.
        out.push_str(&format_agents_md_block(agents_md));
        out.push('\n');

        // Section 2: skills metadata.
        out.push_str("## Skills (metadata)\n\n");
        out.push_str(
            "Skills are optional instruction bundles stored as directories containing `SKILL.md`.\n\
Use `list_skills` to discover them and `load_skill` to load details on demand.\n\n",
        );

        for item in self.skills.list() {
            out.push_str(&format!(
                "- `{}`: {} (dir: `{}`)\n",
                item.name, item.description, item.dir
            ));
        }

        // Section 3: tool usage notes.
        out.push_str("\n## Tools\n\n");
        out.push_str(
            "Available tools:\n\
- `shell_command`: run PowerShell commands inside the workspace.\n\
- `read_file`, `write_file`, `list_dir`: basic file operations.\n\
- `list_skills`, `load_skill`: skill discovery/loading.\n\n\
Rules:\n\
- Only operate inside the configured workspace root.\n\
- Prefer small, incremental changes with verification steps.\n",
        );

        out
    }
}

