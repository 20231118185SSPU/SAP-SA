//! Miscellaneous tool implementations for ToolExecutor (send, show, ask, skill, subagent, etc.).

use crate::cancel::CancelToken;
use crate::path_guard::{PathOperation, validate_resolved_tool_path, validate_tool_path_input};
use super::{
    ToolExecutor, ToolRuntime, ToolSession, ToolExecutionResult, ToolControl,
    FinishRequest, AskRequest, WaitRequest,
};
use crate::ws_protocol::{UserVisibleFile, UserVisibleFileEncoding, QuestionMode, QuestionOption};
use crate::tools::{MAX_SHOW_FILE_BYTES, guess_media_type};
use anyhow::Context as _;
use serde::Deserialize;
use serde_json;
use uuid::Uuid;
use crate::interaction_history::{InteractionDisclosureMode, InteractionReadOptions, InteractionStore};
use super::matches_command_patterns;
use base64::Engine;

impl ToolExecutor {
    /// `Send`: forward a message to the user through the daemon/runtime layer.
    pub(crate) async fn send(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if !runtime.allow_user_send {
            anyhow::bail!("Send is not allowed for this agent");
        }
        if cancel.is_cancelled() {
            anyhow::bail!("Send cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            message: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Send")?;
        if args.message.trim().is_empty() {
            anyhow::bail!("Send message must not be empty");
        }

        runtime.send_message(args.message.clone()).await?;

        Ok(serde_json::json!({
            "sent": true,
            "message": args.message,
        })
        .to_string())
    }

    /// `Show`: transport a file payload to the user-facing frontend.

    /// `Show`: transport a file payload to the user-facing frontend.
    pub(crate) async fn show(
        &self,
        session: &mut ToolSession,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if !runtime.allow_user_show {
            anyhow::bail!("Show is not allowed for this agent");
        }
        if cancel.is_cancelled() {
            anyhow::bail!("Show cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            title: Option<String>,
            prompt: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Show")?;
        if args.prompt.trim().is_empty() {
            anyhow::bail!("Show prompt must not be empty");
        }
        validate_tool_path_input(&args.path, PathOperation::Read)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Read)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file for Show: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "Show requires a file path, not a directory: {}",
                path.display()
            );
        }
        if meta.len() as usize > MAX_SHOW_FILE_BYTES {
            anyhow::bail!(
                "Show refuses files larger than {} bytes: {}",
                MAX_SHOW_FILE_BYTES,
                path.display()
            );
        }

        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("Failed to read file for Show: {}", path.display()))?;

        let (encoding, content) = match std::str::from_utf8(&bytes) {
            Ok(text) => (UserVisibleFileEncoding::Utf8, text.to_string()),
            Err(_) => (
                UserVisibleFileEncoding::Base64,
                base64::engine::general_purpose::STANDARD.encode(&bytes),
            ),
        };

        let file = UserVisibleFile {
            show_id: uuid::Uuid::new_v4(),
            task_id: uuid::Uuid::nil(),
            agent: None,
            path: path.display().to_string(),
            title: args.title.filter(|title| !title.trim().is_empty()),
            prompt: args.prompt,
            media_type: guess_media_type(&path, &encoding),
            encoding,
            content,
            bytes: bytes.len(),
        };

        runtime.show_file(file.clone()).await?;
        session.note_touched_path(path.clone());

        Ok(serde_json::json!({
            "shown": true,
            "path": file.path,
            "title": file.title,
            "bytes": file.bytes,
            "media_type": file.media_type,
            "encoding": file.encoding,
            "prompt": file.prompt,
        })
        .to_string())
    }

    /// `GetInteractionEntry`: retrieve one durable interaction-log entry by id
    /// with progressive disclosure.

    /// `GetInteractionEntry`: retrieve one durable interaction-log entry by id
    /// with progressive disclosure.
    pub(crate) async fn get_interaction_entry(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("GetInteractionEntry cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            id: Uuid,
            mode: Option<String>,
            offset: Option<usize>,
            limit: Option<usize>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for GetInteractionEntry")?;
        let mode = match args.mode.as_deref().unwrap_or("summary") {
            "summary" => InteractionDisclosureMode::Summary,
            "full" => InteractionDisclosureMode::Full,
            "slice" => InteractionDisclosureMode::Slice,
            other => anyhow::bail!(
                "GetInteractionEntry mode must be one of `summary`, `full`, `slice`, got `{other}`"
            ),
        };
        let store = InteractionStore::new(self.ctx.workspace_root.clone())?;
        let Some(serialized) = store.serialize_entry_by_id(
            args.id,
            InteractionReadOptions {
                mode,
                offset: args.offset.unwrap_or(0),
                limit: args.limit.unwrap_or_default(),
            },
        )?
        else {
            anyhow::bail!("Interaction entry not found: {}", args.id);
        };

        Ok(serialized)
    }

    /// `Ask`: block until the user answers a structured question.

    /// `Ask`: block until the user answers a structured question.
    pub(crate) async fn ask(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if !runtime.allow_user_ask {
            anyhow::bail!("Ask is not allowed for this agent");
        }
        if cancel.is_cancelled() {
            anyhow::bail!("Ask cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            prompt: String,
            mode: QuestionMode,
            #[serde(default)]
            options: Vec<QuestionOption>,
            #[serde(default)]
            allow_free_text: bool,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Ask")?;
        let request = AskRequest {
            prompt: args.prompt,
            mode: args.mode,
            options: args.options,
            allow_free_text: args.allow_free_text,
        };
        request.validate()?;
        Ok(ToolExecutionResult::Control(ToolControl::Ask(request)))
    }

    /// `Skill`: invoke one registered command or read one local skill file.

    /// `Skill`: invoke one registered command or read one local skill file.
    pub(crate) async fn skill(
        &self,
        session: &mut ToolSession,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Skill cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            #[serde(default = "default_skill_action")]
            action: String,
            name: String,
            args: Option<String>,
            path: Option<String>,
            session_id: Option<String>,
        }

        pub(crate) fn default_skill_action() -> String {
            "invoke".to_string()
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Skill")?;
        if matches_command_patterns(&args.name, &runtime.denied_commands) {
            anyhow::bail!(
                "Command `{}` is denied by the current permission policy",
                args.name
            );
        }
        let Some(skill) = self.ctx.skills.get(&args.name) else {
            anyhow::bail!("Skill not found: {}", args.name);
        };

        match args.action.as_str() {
            "read" => {
                let (path, content) = self
                    .ctx
                    .skills
                    .load_skill_file(&args.name, args.path.as_deref())
                    .await?;

                Ok(serde_json::json!({
                    "name": skill.name,
                    "description": skill.description,
                    "path": path,
                    "content": content,
                })
                .to_string())
            }
            "invoke" => {
                if skill.disable_model_invocation {
                    anyhow::bail!(
                        "Command `{}` is not model-invocable and must not be invoked automatically",
                        skill.name
                    );
                }

                let instructions = if let Some(prompt) = skill.mcp_prompt() {
                    let Some(registry) = &self.mcp_registry else {
                        anyhow::bail!("No MCP registry is configured for MCP prompt invocation");
                    };
                    registry
                        .expand_prompt(
                            &prompt.server_name,
                            &prompt.prompt_name,
                            &prompt.arguments,
                            args.args.as_deref(),
                        )
                        .await?
                } else {
                    let session_id = args
                        .session_id
                        .clone()
                        .unwrap_or_else(|| runtime.agent_id.to_string());
                    skill
                        .expand_invocation(args.args.as_deref(), &session_id)?
                        .instructions
                };

                session.note_command_invocation(skill.reminder());

                Ok(instructions)
            }
            other => anyhow::bail!("Skill action must be `invoke` or `read`, got `{other}`"),
        }
    }
    /// `Finish`: explicitly mark the current work as complete.

    /// `Finish`: explicitly mark the current work as complete.
    pub(crate) async fn finish(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if cancel.is_cancelled() {
            anyhow::bail!("Finish cancelled");
        }

        let args: FinishRequest =
            serde_json::from_value(args).context("Invalid arguments for Finish")?;
        args.validate()?;

        Ok(ToolExecutionResult::Control(ToolControl::Finish(args)))
    }

    /// `FinishWithoutOutput`: explicit confirmation for a temporary
    /// no-extra-output finish path.

    /// `FinishWithoutOutput`: explicit confirmation for a temporary
    /// no-extra-output finish path.
    pub(crate) async fn finish_without_output(
        &self,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if cancel.is_cancelled() {
            anyhow::bail!("FinishWithoutOutput cancelled");
        }

        Ok(ToolExecutionResult::Control(
            ToolControl::FinishWithoutOutput,
        ))
    }

    /// `NotifyParent`: convenience one-way message to the current parent.

    /// `Wait`: suspend the current work until a dependency reaches the desired
    /// state.
    pub(crate) async fn wait(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ToolExecutionResult> {
        if cancel.is_cancelled() {
            anyhow::bail!("Wait cancelled");
        }

        let args: WaitRequest =
            serde_json::from_value(args).context("Invalid arguments for Wait")?;
        args.validate()?;
        Ok(ToolExecutionResult::Control(ToolControl::Wait(args)))
    }

    /// `GetTask`: inspect one runtime task.

    /// `GetTask`: inspect one runtime task.
    pub(crate) async fn get_task(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("GetTask cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            task_id: Uuid,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for GetTask")?;
        let task = runtime.get_task(args.task_id).await?;
        Ok(serde_json::to_string(&task)?)
    }

    /// `Reload`: ask the backend to hot-reload runtime state from disk.

    /// `Reload`: ask the backend to hot-reload runtime state from disk.
    pub(crate) async fn reload(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Reload cancelled");
        }
        if !runtime.allow_runtime_reload {
            anyhow::bail!("Reload is not available in the current runtime");
        }

        #[derive(Debug, Deserialize, Default)]
        struct Args {}

        let _: Args = serde_json::from_value(args).context("Invalid arguments for Reload")?;
        let receipt = runtime.reload_runtime().await?;
        Ok(serde_json::json!({
            "reloaded": true,
            "summary": receipt.summary,
        })
        .to_string())
    }
}
