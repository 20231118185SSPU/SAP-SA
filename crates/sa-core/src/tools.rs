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
//! - `Fetch`: make a direct HTTP request to a known URL.
//! - `Search`: perform a web search to discover relevant URLs.
//! - `MemorySearch`: search Markdown memory files on demand.
//! - `MemoryGet`: read one bounded slice from a Markdown memory file.
//! - `Send`: send a user-facing message without blocking for a reply.
//! - `Ask`: ask the user a structured question and wait for the answer.
//! - `Show`: transmit a file's contents to the user over WebSocket.
//! - `Skill`: read `SKILL.md` or another skill-relative file from a named skill.
//! - `SubAgent`: launch a nested sub-agent and return its final answer.
//!
//! Design goals:
//! - Keep the file/workspace boundary explicit and verifiable.
//! - Make the `Edit` safety rule enforceable in Rust rather than hoping the
//!   model follows it.
//! - Allow the daemon layer to inject user-interaction and sub-agent behavior
//!   without coupling this crate to any particular transport.

use crate::cancel::CancelToken;
use crate::memory::{read_markdown_memory, search_markdown_memory};
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::skills::SkillRegistry;
use crate::ws_protocol::{
    QuestionMode, QuestionOption, UserQuestionAnswer, UserVisibleFile, UserVisibleFileEncoding,
};
use anyhow::Context as _;
use base64::Engine as _;
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

/// Maximum response bytes returned by `Fetch`.
const MAX_FETCH_RESPONSE_BYTES: usize = 200_000;

/// Maximum file size transported by `Show`.
const MAX_SHOW_FILE_BYTES: usize = 512 * 1024;

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

/// Callback used by `Show`.
pub type ShowFileFn =
    Arc<dyn Fn(UserVisibleFile) -> ToolFuture<anyhow::Result<()>> + Send + Sync + 'static>;

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
    /// Structured file display channel.
    show_file: ShowFileFn,
    /// Nested-agent launcher.
    run_subagent: RunSubAgentFn,
}

impl ToolRuntime {
    /// Construct a runtime from concrete callbacks.
    pub fn new(
        send_message: SendMessageFn,
        ask_question: AskQuestionFn,
        show_file: ShowFileFn,
        run_subagent: RunSubAgentFn,
    ) -> Self {
        Self {
            send_message,
            ask_question,
            show_file,
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
        let show_file: ShowFileFn =
            Arc::new(|_file| Box::pin(async { anyhow::bail!("Show runtime is not configured") }));
        let run_subagent: RunSubAgentFn = Arc::new(|_request, _cancel| {
            Box::pin(async { anyhow::bail!("SubAgent runtime is not configured") })
        });

        Self::new(send_message, ask_question, show_file, run_subagent)
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

    /// Invoke the `Show` callback.
    pub async fn show_file(&self, file: UserVisibleFile) -> anyhow::Result<()> {
        (self.show_file)(file).await
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
                    name: "Fetch".to_string(),
                    description: "Send a direct HTTP request to a known URL and return the response body."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "HTTP or HTTPS URL to request."
                            },
                            "method": {
                                "type": "string",
                                "description": "HTTP method. Defaults to GET."
                            },
                            "headers": {
                                "type": "object",
                                "description": "Optional request headers as string key/value pairs."
                            },
                            "body": {
                                "type": "string",
                                "description": "Optional UTF-8 request body."
                            },
                            "max_bytes": {
                                "type": "integer",
                                "description": "Optional response size cap in bytes. Defaults to 200000 and is capped at 200000."
                            }
                        },
                        "required": ["url"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Search".to_string(),
                    description: "Search the web for relevant pages, returning titles, snippets, and URLs."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return. Defaults to 5 and is capped at 10."
                            }
                        },
                        "required": ["query"]
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
                    name: "MemorySearch".to_string(),
                    description:
                        "Search `MEMORY.md`, `memory.md`, and `memory/*.md` for prior decisions, dates, preferences, or todos."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for memory recall."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Optional maximum number of results. Defaults to 5 and is capped at 10."
                            },
                            "min_score": {
                                "type": "number",
                                "description": "Optional minimum lexical score threshold."
                            }
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "MemoryGet".to_string(),
                    description:
                        "Read one allowed memory Markdown file (`MEMORY.md`, `memory.md`, or `memory/*.md`) with an optional line range."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Memory file path. Must be `MEMORY.md`, `memory.md`, or a file under `memory/`."
                            },
                            "from": {
                                "type": "integer",
                                "description": "Optional 1-based starting line."
                            },
                            "lines": {
                                "type": "integer",
                                "description": "Optional number of lines to read."
                            }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "Show".to_string(),
                    description: "Display an existing workspace file to the user through the frontend."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Existing file path, relative to the workspace root or absolute under it."
                            },
                            "title": {
                                "type": "string",
                                "description": "Optional user-facing title shown above the file content."
                            }
                        },
                        "required": ["path"]
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
                    description:
                        "Read `SKILL.md` or another skill-relative text file from a named installed skill without exposing the real host path."
                            .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name from the prompt's skill metadata list."
                            },
                            "path": {
                                "type": "string",
                                "description": "Optional skill-relative path. Defaults to `SKILL.md`."
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
            "Fetch" => self.fetch(args, cancel).await,
            "Search" => self.search(args, cancel).await,
            "MemorySearch" => self.memory_search(args, cancel).await,
            "MemoryGet" => self.memory_get(args, cancel).await,
            "Send" => self.send(runtime, args, cancel).await,
            "Show" => self.show(runtime, args, cancel).await,
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

    /// `Fetch`: make a direct HTTP request to a known URL.
    async fn fetch(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            url: String,
            method: Option<String>,
            #[serde(default)]
            headers: serde_json::Map<String, serde_json::Value>,
            body: Option<String>,
            max_bytes: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Fetch")?;
        if cancel.is_cancelled() {
            anyhow::bail!("Fetch cancelled");
        }

        let url = validate_network_url(&args.url)?;
        let method = args
            .method
            .as_deref()
            .unwrap_or("GET")
            .parse::<reqwest::Method>()
            .context("Fetch method must be a valid HTTP method")?;
        let max_bytes = args
            .max_bytes
            .unwrap_or(MAX_FETCH_RESPONSE_BYTES)
            .min(MAX_FETCH_RESPONSE_BYTES);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("StudyAdministrator/0.6 Fetch")
            .build()
            .context("Failed to build Fetch HTTP client")?;

        let mut request = client.request(method.clone(), url.clone());
        for (name, value) in args.headers {
            let Some(value) = value.as_str() else {
                anyhow::bail!("Fetch headers must be string key/value pairs");
            };
            request = request.header(&name, value);
        }
        if let Some(body) = args.body {
            request = request.body(body);
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("Fetch cancelled");
            }
            response = request.send() => {
                response.context("Fetch request failed")?
            }
        };

        let status = response.status();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        let headers = response.headers().clone();

        let body_bytes = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("Fetch cancelled");
            }
            body = response.bytes() => {
                body.context("Failed to read Fetch response body")?
            }
        };

        let truncated = body_bytes.len() > max_bytes;
        let clipped = if truncated {
            &body_bytes[..max_bytes]
        } else {
            body_bytes.as_ref()
        };

        let body_text = String::from_utf8_lossy(clipped).to_string();
        let response_headers = headers
            .iter()
            .filter_map(|(name, value)| {
                value.to_str().ok().map(|value| {
                    (
                        name.as_str().to_string(),
                        serde_json::Value::String(value.to_string()),
                    )
                })
            })
            .collect::<serde_json::Map<_, _>>();

        Ok(serde_json::json!({
            "url": final_url,
            "method": method.as_str(),
            "status": status.as_u16(),
            "content_type": content_type,
            "headers": response_headers,
            "bytes": body_bytes.len(),
            "truncated": truncated,
            "body": body_text,
        })
        .to_string())
    }

    /// `Search`: discover relevant URLs before a more targeted `Fetch`.
    async fn search(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            max_results: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Search")?;
        if cancel.is_cancelled() {
            anyhow::bail!("Search cancelled");
        }
        if args.query.trim().is_empty() {
            anyhow::bail!("Search query must not be empty");
        }

        let max_results = args.max_results.unwrap_or(5).clamp(1, 10);
        let url = reqwest::Url::parse_with_params(
            "https://html.duckduckgo.com/html/",
            &[("q", args.query.trim())],
        )
        .context("Failed to build Search URL")?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("StudyAdministrator/0.6 Search")
            .build()
            .context("Failed to build Search HTTP client")?;

        let response = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("Search cancelled");
            }
            response = client.get(url).send() => {
                response.context("Search request failed")?
            }
        };

        let html = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("Search cancelled");
            }
            body = response.text() => {
                body.context("Failed to read Search response body")?
            }
        };

        let results = extract_duckduckgo_results(&html, max_results);

        Ok(serde_json::json!({
            "query": args.query,
            "results": results,
        })
        .to_string())
    }

    /// `MemorySearch`: search Markdown memory files on demand.
    async fn memory_search(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            max_results: Option<usize>,
            min_score: Option<f64>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MemorySearch")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemorySearch cancelled");
        }

        let results = search_markdown_memory(
            &self.ctx.workspace_root,
            &args.query,
            args.max_results,
            args.min_score,
        )
        .await?;

        Ok(serde_json::json!({
            "query": args.query,
            "results": results,
            "engine": "markdown_lexical_v1",
        })
        .to_string())
    }

    /// `MemoryGet`: read a bounded slice from one allowed memory file.
    async fn memory_get(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            from: Option<usize>,
            lines: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for MemoryGet")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemoryGet cancelled");
        }

        let result =
            read_markdown_memory(&self.ctx.workspace_root, &args.path, args.from, args.lines)
                .await?;

        Ok(serde_json::to_string(&result)?)
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

    /// `Show`: transport a file payload to the user-facing frontend.
    async fn show(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Show cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            title: Option<String>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Show")?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
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
            path: path.display().to_string(),
            title: args.title.filter(|title| !title.trim().is_empty()),
            media_type: guess_media_type(&path, &encoding),
            encoding,
            content,
            bytes: bytes.len(),
        };

        runtime.show_file(file.clone()).await?;

        Ok(serde_json::json!({
            "shown": true,
            "path": file.path,
            "title": file.title,
            "bytes": file.bytes,
            "media_type": file.media_type,
            "encoding": file.encoding,
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

    /// `Skill`: read `SKILL.md` or another skill-relative file by skill name.
    async fn skill(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Skill cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            name: String,
            path: Option<String>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Skill")?;
        let Some(skill) = self.ctx.skills.get(&args.name) else {
            anyhow::bail!("Skill not found: {}", args.name);
        };

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

/// Validate that a network URL is HTTP(S) and therefore appropriate for
/// `Fetch`.
fn validate_network_url(raw: &str) -> anyhow::Result<reqwest::Url> {
    let url = reqwest::Url::parse(raw).context("Fetch URL must be a valid absolute URL")?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => anyhow::bail!("Fetch only allows http/https URLs, got scheme `{other}`"),
    }
}

/// Extract a bounded list of search results from DuckDuckGo's lightweight HTML
/// page.
///
/// The parser stays dependency-light on purpose: this crate is intentionally
/// small and we only need a few stable fields (title, URL, snippet).
fn extract_duckduckgo_results(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while out.len() < max_results {
        let Some(anchor_rel) = html[cursor..]
            .find("result__a")
            .or_else(|| html[cursor..].find("result-link"))
        else {
            break;
        };
        let anchor_idx = cursor + anchor_rel;
        let Some(tag_start) = html[..anchor_idx].rfind("<a") else {
            cursor = anchor_idx + 1;
            continue;
        };
        let Some(tag_end_rel) = html[anchor_idx..].find("</a>") else {
            break;
        };
        let tag_end = anchor_idx + tag_end_rel + "</a>".len();
        let anchor_html = &html[tag_start..tag_end];

        let href = extract_href(anchor_html)
            .map(|href| normalize_duckduckgo_result_url(&href))
            .unwrap_or_default();
        let title = extract_anchor_text(anchor_html);

        // Look at a small fragment after the anchor and try a few known snippet
        // markers used by DuckDuckGo HTML/Lite pages.
        let next_anchor = html[tag_end..]
            .find("result__a")
            .or_else(|| html[tag_end..].find("result-link"))
            .map(|idx| tag_end + idx)
            .unwrap_or_else(|| html.len());
        let snippet_fragment = &html[tag_end..next_anchor.min(tag_end.saturating_add(4_000))];
        let snippet = extract_html_fragment(snippet_fragment, "result__snippet")
            .or_else(|| extract_html_fragment(snippet_fragment, "result-snippet"))
            .or_else(|| extract_html_fragment(snippet_fragment, "snippet"))
            .unwrap_or_default();

        if !title.is_empty() && !href.is_empty() {
            out.push(serde_json::json!({
                "title": title,
                "url": href,
                "snippet": snippet,
            }));
        }

        cursor = tag_end;
    }

    out
}

/// Extract a best-effort href from an anchor tag.
fn extract_href(anchor_html: &str) -> Option<String> {
    let href_pos = anchor_html.find("href=")?;
    let quote = anchor_html[href_pos + 5..].chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }

    let value_start = href_pos + 6;
    let value_end_rel = anchor_html[value_start..].find(quote)?;
    let value = &anchor_html[value_start..value_start + value_end_rel];
    Some(decode_html_entities(value.trim()))
}

/// Extract visible text from an anchor by stripping simple HTML tags.
fn extract_anchor_text(anchor_html: &str) -> String {
    let content_start = match anchor_html.find('>') {
        Some(idx) => idx + 1,
        None => return String::new(),
    };
    let content_end = match anchor_html.rfind("</a>") {
        Some(idx) if idx >= content_start => idx,
        _ => anchor_html.len(),
    };

    strip_html_tags(&anchor_html[content_start..content_end])
}

/// Extract a text fragment from an HTML snippet using a class marker.
fn extract_html_fragment(fragment: &str, class_marker: &str) -> Option<String> {
    let marker_idx = fragment.find(class_marker)?;
    let tag_start = fragment[..marker_idx].rfind('<')?;
    let tag_name_end = fragment[tag_start + 1..]
        .find(|ch: char| ch == '>' || ch.is_whitespace())
        .map(|idx| tag_start + 1 + idx)?;
    let tag_name = &fragment[tag_start + 1..tag_name_end];
    let open_end_rel = fragment[marker_idx..].find('>')?;
    let content_start = marker_idx + open_end_rel + 1;
    let close_marker = format!("</{tag_name}>");
    let close_tag_rel = fragment[content_start..].find(&close_marker)?;
    let close_tag = content_start + close_tag_rel;

    if close_tag <= tag_start {
        return None;
    }

    let raw = &fragment[content_start..close_tag];
    let cleaned = strip_html_tags(raw);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Remove simple HTML tags and decode common entities.
fn strip_html_tags(raw: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;

    for ch in raw.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }

    decode_html_entities(out.trim())
}

/// Normalize DuckDuckGo redirect links into their destination URL.
fn normalize_duckduckgo_result_url(raw: &str) -> String {
    let raw = raw.trim();
    let candidate = if raw.starts_with("//") {
        format!("https:{raw}")
    } else {
        raw.to_string()
    };

    let Ok(url) = reqwest::Url::parse(&candidate) else {
        return candidate;
    };

    if url.domain() == Some("duckduckgo.com") || url.domain() == Some("html.duckduckgo.com") {
        if let Some((_, value)) = url.query_pairs().find(|(key, _)| key == "uddg") {
            return value.into_owned();
        }
    }

    url.to_string()
}

/// Decode a small set of HTML entities commonly returned by search result
/// pages.
fn decode_html_entities(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if bytes[idx] != b'&' {
            out.push(bytes[idx] as char);
            idx += 1;
            continue;
        }

        let Some(end_rel) = raw[idx..].find(';') else {
            out.push('&');
            idx += 1;
            continue;
        };
        let end = idx + end_rel;
        let entity = &raw[idx + 1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" | "#x27" => Some('\''),
            "nbsp" => Some(' '),
            "#47" | "#x2F" => Some('/'),
            _ => decode_numeric_entity(entity),
        };

        if let Some(ch) = decoded {
            out.push(ch);
        } else {
            out.push('&');
            out.push_str(entity);
            out.push(';');
        }
        idx = end + 1;
    }

    out
}

/// Decode `&#...;` or `&#x...;` entities.
fn decode_numeric_entity(entity: &str) -> Option<char> {
    if let Some(hex) = entity
        .strip_prefix("#x")
        .or_else(|| entity.strip_prefix("#X"))
    {
        let value = u32::from_str_radix(hex, 16).ok()?;
        return char::from_u32(value);
    }

    let dec = entity.strip_prefix('#')?;
    let value = dec.parse::<u32>().ok()?;
    char::from_u32(value)
}

/// Infer a reasonable media type for `Show`.
fn guess_media_type(path: &Path, encoding: &UserVisibleFileEncoding) -> String {
    match encoding {
        UserVisibleFileEncoding::Base64 => {
            let ext = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase());
            match ext.as_deref() {
                Some("png") => "image/png".to_string(),
                Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
                Some("gif") => "image/gif".to_string(),
                Some("webp") => "image/webp".to_string(),
                Some("pdf") => "application/pdf".to_string(),
                Some("zip") => "application/zip".to_string(),
                _ => "application/octet-stream".to_string(),
            }
        }
        UserVisibleFileEncoding::Utf8 => {
            let ext = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase());
            match ext.as_deref() {
                Some("md") => "text/markdown".to_string(),
                Some("json") => "application/json".to_string(),
                Some("toml") => "application/toml".to_string(),
                Some("yaml") | Some("yml") => "application/yaml".to_string(),
                Some("rs") => "text/x-rust".to_string(),
                Some("sh") => "text/x-shellscript".to_string(),
                Some("txt") | Some("log") => "text/plain".to_string(),
                _ => "text/plain; charset=utf-8".to_string(),
            }
        }
    }
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

    #[test]
    fn validate_network_url_rejects_non_http() {
        let err = validate_network_url("file:///tmp/secret.txt")
            .expect_err("non-http URL must be rejected");
        assert!(err.to_string().contains("http/https"));
    }

    #[test]
    fn normalize_duckduckgo_redirect_extracts_uddg() {
        let url = normalize_duckduckgo_result_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide",
        );
        assert_eq!(url, "https://example.com/guide");
    }

    #[test]
    fn extract_duckduckgo_results_parses_title_url_and_snippet() {
        let html = r#"
<div class="result">
  <a rel="nofollow" class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
    Example &amp; Guide
  </a>
  <a class="result__snippet">A <b>useful</b> summary.</a>
</div>
"#;

        let results = extract_duckduckgo_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["title"], "Example & Guide");
        assert_eq!(results[0]["url"], "https://example.com/page");
        assert_eq!(results[0]["snippet"], "A useful summary.");
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
