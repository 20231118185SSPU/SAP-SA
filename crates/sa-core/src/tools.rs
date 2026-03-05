//! Built-in tools for the StudyAdministrator (SA) agent.
//!
//! The user requested a deliberate redesign of the built-in toolset. The
//! previous minimal set (`shell_command`, `read_file`, `write_file`, etc.) is
//! replaced by these exact tools:
//! - `Read`: read a UTF-8 text file.
//! - `Write`: create a new UTF-8 text file; refuse if the file already exists.
//! - `Edit`: edit an existing UTF-8 text file; the file must have been `Read`
//!   earlier in the same agent session.
//! - `Bash`: execute a shell command via Git Bash (`bash -lc`).
//! - `Send`: send a user-facing message without blocking for a reply.
//! - `Ask`: ask the user a structured question and wait for the answer.
//! - `Skill`: load a skill's `SKILL.md`.
//! - `SubAgent`: launch a nested sub-agent and return its final answer.
//!
//! Design goals:
//! - Keep the file/workspace boundary explicit and verifiable.
//! - Make the `Edit` safety rule enforceable in Rust rather than hoping the
//!   model follows it.
//! - Allow the daemon layer to inject user-interaction and sub-agent behavior
//!   without coupling this crate to any particular transport.

use crate::cancel::CancelToken;
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::skills::SkillRegistry;
use crate::ws_protocol::{QuestionMode, QuestionOption, UserQuestionAnswer};
use anyhow::Context as _;
use serde::Deserialize;
use std::collections::HashSet;
use std::ffi::OsString;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// Maximum file size accepted by `Read`.
///
/// This is a context-window guardrail. Large files should be explored through
/// more targeted reads or shell commands rather than dumping megabytes into the
/// model context.
const MAX_READ_FILE_BYTES: u64 = 200_000;

/// Default timeout for `Bash` if the caller does not specify one.
const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_secs(300);

/// Maximum timeout accepted by `Bash`.
///
/// The tool should still be practical for builds/tests, but we keep an upper
/// bound so one tool call cannot hang indefinitely.
const MAX_BASH_TIMEOUT: Duration = Duration::from_secs(1_800);

/// Safety limit for nested sub-agents.
///
/// The user explicitly asked for recursive sub-agents. We still need a hard
/// ceiling so a prompt bug cannot recurse forever.
pub const MAX_SUBAGENT_DEPTH: u32 = 6;

/// Boxed async return type used by runtime callbacks.
type ToolFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Callback used by `Send`.
pub type SendMessageFn =
    Arc<dyn Fn(String) -> ToolFuture<anyhow::Result<()>> + Send + Sync + 'static>;

/// Callback used by `Ask`.
pub type AskQuestionFn = Arc<
    dyn Fn(AskRequest, CancelToken) -> ToolFuture<anyhow::Result<UserQuestionAnswer>>
        + Send
        + Sync
        + 'static,
>;

/// Callback used by `SubAgent`.
pub type RunSubAgentFn = Arc<
    dyn Fn(SubAgentRequest, CancelToken) -> ToolFuture<anyhow::Result<String>>
        + Send
        + Sync
        + 'static,
>;

/// Structured request emitted by the `Ask` tool.
#[derive(Debug, Clone)]
pub struct AskRequest {
    /// Human-readable question prompt.
    pub prompt: String,
    /// How the answer should be collected.
    pub mode: QuestionMode,
    /// Selectable options for choice-based prompts.
    pub options: Vec<QuestionOption>,
    /// Whether optional free text is allowed in addition to selections.
    pub allow_free_text: bool,
}

impl AskRequest {
    /// Validate the request before it leaves the tool layer.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.prompt.trim().is_empty() {
            anyhow::bail!("Ask prompt must not be empty");
        }

        match self.mode {
            QuestionMode::SingleChoice | QuestionMode::MultiChoice => {
                if self.options.is_empty() {
                    anyhow::bail!("Choice-based Ask requests must provide at least one option");
                }
            }
            QuestionMode::Text => {
                if !self.options.is_empty() {
                    anyhow::bail!("Text Ask requests must not provide options");
                }
            }
        }

        let mut seen = HashSet::<String>::new();
        for option in &self.options {
            if option.id.trim().is_empty() {
                anyhow::bail!("Ask option ids must not be empty");
            }
            if option.label.trim().is_empty() {
                anyhow::bail!("Ask option labels must not be empty");
            }
            if !seen.insert(option.id.clone()) {
                anyhow::bail!("Ask option ids must be unique: {}", option.id);
            }
        }

        Ok(())
    }
}

/// Request emitted by the `SubAgent` tool.
#[derive(Debug, Clone)]
pub struct SubAgentRequest {
    /// Optional traceable label shown in logs.
    pub label: Option<String>,
    /// Concrete task the child agent should complete.
    pub task: String,
    /// Parent-provided context that will be injected into the child prompt.
    pub context: String,
}

impl SubAgentRequest {
    /// Validate that the request is well-formed before the daemon executes it.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.task.trim().is_empty() {
            anyhow::bail!("SubAgent task must not be empty");
        }

        if self.context.trim().is_empty() {
            anyhow::bail!("SubAgent context must not be empty");
        }

        Ok(())
    }
}

/// Shared callbacks injected by the daemon layer.
///
/// The core tool implementation does not know how to talk to the user or how
/// to create child agents. Those behaviors are supplied here by the backend.
#[derive(Clone)]
pub struct ToolRuntime {
    /// Non-blocking "tell the user" channel.
    send_message: SendMessageFn,
    /// Blocking "ask the user" channel.
    ask_question: AskQuestionFn,
    /// Nested-agent launcher.
    run_subagent: RunSubAgentFn,
}

impl ToolRuntime {
    /// Construct a runtime from concrete callbacks.
    pub fn new(
        send_message: SendMessageFn,
        ask_question: AskQuestionFn,
        run_subagent: RunSubAgentFn,
    ) -> Self {
        Self {
            send_message,
            ask_question,
            run_subagent,
        }
    }

    /// Detached runtime used in unit tests.
    ///
    /// Any attempt to use a runtime-only tool fails with a clear error.
    pub fn detached() -> Self {
        let send_message: SendMessageFn = Arc::new(|_message| {
            Box::pin(async { anyhow::bail!("Send runtime is not configured") })
        });
        let ask_question: AskQuestionFn = Arc::new(|_request, _cancel| {
            Box::pin(async { anyhow::bail!("Ask runtime is not configured") })
        });
        let run_subagent: RunSubAgentFn = Arc::new(|_request, _cancel| {
            Box::pin(async { anyhow::bail!("SubAgent runtime is not configured") })
        });

        Self::new(send_message, ask_question, run_subagent)
    }

    /// Invoke the `Send` callback.
    pub async fn send_message(&self, message: String) -> anyhow::Result<()> {
        (self.send_message)(message).await
    }

    /// Invoke the `Ask` callback.
    pub async fn ask_question(
        &self,
        request: AskRequest,
        cancel: CancelToken,
    ) -> anyhow::Result<UserQuestionAnswer> {
        (self.ask_question)(request, cancel).await
    }

    /// Invoke the `SubAgent` callback.
    pub async fn run_subagent(
        &self,
        request: SubAgentRequest,
        cancel: CancelToken,
    ) -> anyhow::Result<String> {
        (self.run_subagent)(request, cancel).await
    }
}

/// Shared context used by all tool invocations.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Workspace root directory; all file activity is constrained to this tree.
    pub workspace_root: PathBuf,
    /// Registry of discovered skills.
    pub skills: Arc<SkillRegistry>,
}

impl ToolContext {
    /// Create a new tool context.
    pub fn new(workspace_root: PathBuf, skills: Arc<SkillRegistry>) -> anyhow::Result<Self> {
        let workspace_root = std::fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root: {}",
                workspace_root.display()
            )
        })?;

        Ok(Self {
            workspace_root,
            skills,
        })
    }

    /// Resolve a user-provided path into an absolute path under `workspace_root`.
    ///
    /// This must work for both existing and not-yet-existing paths.
    pub fn resolve_under_workspace(&self, raw: &str) -> anyhow::Result<PathBuf> {
        let raw_path = PathBuf::from(raw);
        let candidate = if raw_path.is_absolute() {
            raw_path
        } else {
            self.workspace_root.join(raw_path)
        };

        let mut ancestor = candidate.as_path();
        let mut suffix: Vec<OsString> = Vec::new();
        while !ancestor.exists() {
            let Some(name) = ancestor.file_name() else {
                anyhow::bail!("Invalid path: {raw}");
            };
            suffix.push(name.to_os_string());
            let Some(parent) = ancestor.parent() else {
                anyhow::bail!("Invalid path: {raw}");
            };
            ancestor = parent;
        }

        let mut canonical = std::fs::canonicalize(ancestor).with_context(|| {
            format!(
                "Failed to canonicalize existing ancestor: {}",
                ancestor.display()
            )
        })?;
        for component in suffix.into_iter().rev() {
            canonical.push(component);
        }

        if !canonical.starts_with(&self.workspace_root) {
            anyhow::bail!(
                "Path escapes workspace root (root: {}, requested: {})",
                self.workspace_root.display(),
                canonical.display()
            );
        }

        Ok(canonical)
    }
}

/// Per-agent-session tool state.
///
/// The important rule here is the `Edit` precondition:
/// - a file must be `Read` before it is `Edit`ed
/// - after a successful `Edit`, the "fresh read" marker is cleared again
///   because the file content has changed
#[derive(Debug, Default)]
pub struct ToolSession {
    /// Canonical file paths that are currently eligible for `Edit`.
    readable_for_edit: HashSet<PathBuf>,
}

impl ToolSession {
    /// Mark that a file has just been read in this session.
    fn note_read(&mut self, path: PathBuf) {
        self.readable_for_edit.insert(path);
    }

    /// Enforce the "must read before edit" rule.
    fn require_fresh_read(&self, path: &Path) -> anyhow::Result<()> {
        if self.readable_for_edit.contains(path) {
            return Ok(());
        }

        anyhow::bail!(
            "Edit is not allowed until the file has been Read in this session: {}",
            path.display()
        );
    }

    /// After an edit, the caller must re-read the file before the next edit.
    fn invalidate_after_edit(&mut self, path: &Path) {
        self.readable_for_edit.remove(path);
    }
}

/// Concrete tool executor.
#[derive(Debug, Clone)]
pub struct ToolExecutor {
    /// Shared tool context.
    pub ctx: ToolContext,
}

impl ToolExecutor {
    /// Create a new executor.
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }

    /// Tool definitions advertised to the model.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Read".to_string(),
                    description: "Read a UTF-8 text file inside the workspace.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path, relative to the workspace root or absolute under it."
                            }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Write".to_string(),
                    description: "Create a new UTF-8 text file. Refuses to overwrite existing files."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "New file path, relative to the workspace root or absolute under it."
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete file content to create."
                            }
                        },
                        "required": ["path", "content"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Edit".to_string(),
                    description: "Edit an existing UTF-8 text file that has already been Read in this session."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Existing file path, relative to the workspace root or absolute under it."
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to replace. Must be non-empty."
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Replacement text."
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "If true, replace all matches. If false or omitted, exactly one match must exist."
                            }
                        },
                        "required": ["path", "old_text", "new_text"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Bash".to_string(),
                    description: "Run a command via Git Bash (`bash -lc`) inside the workspace."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command string passed to `bash -lc`."
                            },
                            "workdir": {
                                "type": "string",
                                "description": "Optional working directory, relative to the workspace root or absolute under it."
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Optional timeout in seconds. Defaults to 300 and is capped at 1800."
                            }
                        },
                        "required": ["command"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Send".to_string(),
                    description: "Send a user-facing message without waiting for a reply."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message content shown to the user."
                            }
                        },
                        "required": ["message"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Ask".to_string(),
                    description: "Ask the user a structured question and wait for the answer."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Human-readable question prompt."
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["single_choice", "multi_choice", "text"],
                                "description": "How the answer should be collected."
                            },
                            "options": {
                                "type": "array",
                                "description": "Selectable options for choice-based questions.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": { "type": "string" },
                                        "label": { "type": "string" },
                                        "description": { "type": "string" }
                                    },
                                    "required": ["id", "label"]
                                }
                            },
                            "allow_free_text": {
                                "type": "boolean",
                                "description": "Whether the user may additionally provide free text."
                            }
                        },
                        "required": ["prompt", "mode"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Skill".to_string(),
                    description: "Load the full SKILL.md for a named skill.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name from the prompt's skill metadata list."
                            }
                        },
                        "required": ["name"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "SubAgent".to_string(),
                    description: "Launch a nested sub-agent with explicit parent-provided context."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Optional short label used for tracing in logs."
                            },
                            "task": {
                                "type": "string",
                                "description": "Concrete task the child agent should complete."
                            },
                            "context": {
                                "type": "string",
                                "description": "Parent-provided context, constraints, and findings for the child agent."
                            }
                        },
                        "required": ["task", "context"]
                    }),
                },
            },
        ]
    }

    /// Execute one tool call.
    pub async fn execute(
        &self,
        session: &mut ToolSession,
        runtime: &ToolRuntime,
        name: &str,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        match name {
            "Read" => self.read(session, args, cancel).await,
            "Write" => self.write(args, cancel).await,
            "Edit" => self.edit(session, args, cancel).await,
            "Bash" => self.bash(args, cancel).await,
            "Send" => self.send(runtime, args, cancel).await,
            "Ask" => self.ask(runtime, args, cancel).await,
            "Skill" => self.skill(args, cancel).await,
            "SubAgent" => self.subagent(runtime, args, cancel).await,
            _ => anyhow::bail!("Unknown tool: {name}"),
        }
    }

    /// `Read`: read a UTF-8 text file and mark it as eligible for `Edit`.
    async fn read(
        &self,
        session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Read cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Read")?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;

        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "Read requires a file path, not a directory: {}",
                path.display()
            );
        }
        if meta.len() > MAX_READ_FILE_BYTES {
            anyhow::bail!(
                "File is too large to Read via tool ({} bytes): {}",
                meta.len(),
                path.display()
            );
        }

        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        session.note_read(path.clone());

        Ok(serde_json::json!({
            "path": path.display().to_string(),
            "bytes": content.len(),
            "content": content,
        })
        .to_string())
    }

    /// `Write`: create a new file and refuse to overwrite an existing one.
    async fn write(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Write cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            content: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Write")?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;

        if tokio::fs::try_exists(&path).await? {
            anyhow::bail!(
                "Write refuses to overwrite an existing path: {}",
                path.display()
            );
        }

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create parent directories: {}", parent.display())
            })?;
        }

        tokio::fs::write(&path, args.content.as_bytes())
            .await
            .with_context(|| format!("Failed to write file: {}", path.display()))?;

        Ok(serde_json::json!({
            "created": true,
            "path": path.display().to_string(),
            "bytes": args.content.len(),
        })
        .to_string())
    }

    /// `Edit`: replace text in an existing file that was previously `Read`.
    async fn edit(
        &self,
        session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Edit cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            old_text: String,
            new_text: String,
            replace_all: Option<bool>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Edit")?;
        let replace_all = args.replace_all.unwrap_or(false);
        let path = self.ctx.resolve_under_workspace(&args.path)?;

        session.require_fresh_read(&path)?;

        if args.old_text.is_empty() {
            anyhow::bail!("Edit old_text must be non-empty");
        }

        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file for edit: {}", path.display()))?;

        let match_count = content.matches(&args.old_text).count();
        if match_count == 0 {
            anyhow::bail!("Edit could not find the target text in {}", path.display());
        }

        let new_content = if replace_all {
            content.replace(&args.old_text, &args.new_text)
        } else {
            if match_count != 1 {
                anyhow::bail!(
                    "Edit expected exactly one match in {}, but found {}. Use replace_all=true or provide a more specific old_text.",
                    path.display(),
                    match_count
                );
            }
            content.replacen(&args.old_text, &args.new_text, 1)
        };

        tokio::fs::write(&path, new_content.as_bytes())
            .await
            .with_context(|| format!("Failed to write edited file: {}", path.display()))?;

        session.invalidate_after_edit(&path);

        Ok(serde_json::json!({
            "edited": true,
            "path": path.display().to_string(),
            "replace_all": replace_all,
            "matches_replaced": if replace_all { match_count } else { 1 },
            "new_bytes": new_content.len(),
        })
        .to_string())
    }

    /// `Bash`: run a command through Git Bash.
    async fn bash(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            command: String,
            workdir: Option<String>,
            timeout_seconds: Option<u64>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Bash")?;
        let workdir = match args.workdir.as_deref() {
            Some(raw) => self.ctx.resolve_under_workspace(raw)?,
            None => self.ctx.workspace_root.clone(),
        };

        let timeout = args
            .timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_BASH_TIMEOUT)
            .min(MAX_BASH_TIMEOUT);

        let programs = candidate_bash_programs();
        let mut last_not_found: Option<anyhow::Error> = None;

        for program in programs {
            if cancel.is_cancelled() {
                anyhow::bail!("Bash cancelled");
            }

            let mut cmd = tokio::process::Command::new(&program);
            cmd.arg("-lc");
            cmd.arg(&args.command);
            cmd.current_dir(&workdir);
            cmd.kill_on_drop(true);

            let result = tokio::select! {
                _ = cancel.cancelled() => {
                    anyhow::bail!("Bash cancelled");
                }
                output = tokio::time::timeout(timeout, cmd.output()) => {
                    output.context("Bash command timed out")?
                }
            };

            match result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let exit_code = output.status.code().unwrap_or(-1);

                    return Ok(serde_json::json!({
                        "program": program.display().to_string(),
                        "workdir": workdir.display().to_string(),
                        "exit_code": exit_code,
                        "stdout": stdout,
                        "stderr": stderr,
                    })
                    .to_string());
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    last_not_found = Some(anyhow::Error::new(err).context(format!(
                        "Bash executable not found at {}",
                        program.display()
                    )));
                    continue;
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!("Failed to execute Bash via {}", program.display())
                    });
                }
            }
        }

        Err(last_not_found.unwrap_or_else(|| {
            anyhow::anyhow!(
                "No usable bash executable was found. Install Git Bash or ensure `bash` is on PATH."
            )
        }))
    }

    /// `Send`: forward a message to the user through the daemon/runtime layer.
    async fn send(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
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

    /// `Ask`: block until the user answers a structured question.
    async fn ask(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
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

        let answer = runtime
            .ask_question(request.clone(), cancel.clone())
            .await?;

        let selected_labels: Vec<String> = answer
            .selected_option_ids
            .iter()
            .filter_map(|id| {
                request
                    .options
                    .iter()
                    .find(|option| option.id == *id)
                    .map(|option| option.label.clone())
            })
            .collect();

        Ok(serde_json::json!({
            "question_prompt": request.prompt,
            "mode": request.mode,
            "selected_option_ids": answer.selected_option_ids,
            "selected_labels": selected_labels,
            "free_text": answer.free_text,
        })
        .to_string())
    }

    /// `Skill`: load a full `SKILL.md` by name.
    async fn skill(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Skill cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            name: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Skill")?;
        let Some(skill) = self.ctx.skills.get(&args.name) else {
            anyhow::bail!("Skill not found: {}", args.name);
        };

        let content = self.ctx.skills.load_skill_md(&args.name).await?;

        Ok(serde_json::json!({
            "name": skill.name,
            "description": skill.description,
            "dir": skill.dir.display().to_string(),
            "content": content,
        })
        .to_string())
    }

    /// `SubAgent`: delegate a focused sub-task to a nested agent.
    async fn subagent(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("SubAgent cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            label: Option<String>,
            task: String,
            context: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for SubAgent")?;
        let request = SubAgentRequest {
            label: args.label,
            task: args.task,
            context: args.context,
        };
        request.validate()?;

        let final_answer = runtime.run_subagent(request, cancel.clone()).await?;

        Ok(serde_json::json!({
            "final_answer": final_answer,
        })
        .to_string())
    }
}

/// Candidate `bash` programs to try, in order.
///
/// Strategy:
/// - `bash` from PATH is the preferred option because it respects the user's
///   environment.
/// - Then we try common Git for Windows installation paths.
fn candidate_bash_programs() -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("bash")];

    for env_name in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(root) = std::env::var(env_name) {
            let root = PathBuf::from(root);
            out.push(root.join("Git").join("bin").join("bash.exe"));
            out.push(root.join("Git").join("usr").join("bin").join("bash.exe"));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::SkillRegistry;
    use std::fs;
    use uuid::Uuid;

    /// Create a unique temp directory for tests without adding extra deps.
    fn unique_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sa-tools-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Create a minimal tool context rooted at a fresh temp directory.
    fn test_context() -> ToolContext {
        ToolContext::new(unique_temp_dir(), Arc::new(SkillRegistry::default())).expect("context")
    }

    #[test]
    fn resolve_under_workspace_rejects_escape() {
        let ctx = test_context();
        let err = ctx
            .resolve_under_workspace("..\\outside.txt")
            .expect_err("path traversal must fail");
        assert!(err.to_string().contains("escapes workspace root"));
    }

    #[tokio::test]
    async fn write_refuses_existing_file() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone());
        let path = ctx.workspace_root.join("already.txt");
        fs::write(&path, "hello").expect("seed file");

        let err = executor
            .write(
                serde_json::json!({
                    "path": "already.txt",
                    "content": "new",
                }),
                &crate::cancel::cancel_pair().1,
            )
            .await
            .expect_err("overwrite must fail");

        assert!(err.to_string().contains("refuses to overwrite"));
    }

    #[tokio::test]
    async fn edit_requires_read_first() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone());
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("note.txt"), "alpha beta").expect("seed file");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect_err("edit without read must fail");

        assert!(err.to_string().contains("has been Read"));
    }

    #[tokio::test]
    async fn edit_invalidates_read_marker_after_success() {
        let ctx = test_context();
        let executor = ToolExecutor::new(ctx.clone());
        let mut session = ToolSession::default();
        let cancel = crate::cancel::cancel_pair().1;

        fs::write(ctx.workspace_root.join("note.txt"), "alpha beta").expect("seed file");

        executor
            .read(
                &mut session,
                serde_json::json!({ "path": "note.txt" }),
                &cancel,
            )
            .await
            .expect("read");

        executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "beta",
                    "new_text": "gamma",
                }),
                &cancel,
            )
            .await
            .expect("edit");

        let err = executor
            .edit(
                &mut session,
                serde_json::json!({
                    "path": "note.txt",
                    "old_text": "gamma",
                    "new_text": "delta",
                }),
                &cancel,
            )
            .await
            .expect_err("second edit without reread must fail");

        assert!(err.to_string().contains("has been Read"));
    }
}
