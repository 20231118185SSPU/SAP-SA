//! Tool implementations for the StudyAdministrator (SA) agent.
//!
//! The agent loop relies on **OpenAI tool calling**:
//! - The model emits `tool_calls` with `function.name` + JSON arguments string.
//! - The host (this crate) executes the tool and returns a `role: tool` message.
//!
//! We intentionally keep the toolset small but practical:
//! - `shell_command`: run PowerShell commands (for repo operations, builds, etc.).
//! - `read_file`: read a workspace file as UTF-8 text.
//! - `write_file`: write a workspace file.
//! - `list_dir`: list directory entries.
//! - `list_skills`: list discovered skills (name + description).
//! - `load_skill`: load a skill's `SKILL.md` (full content).
//!
//! NOTE: The user request requires that communication interruptions do not stop
//! the agent. Tools therefore should not depend on any client connection.

use crate::cancel::CancelToken;
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::skills::SkillRegistry;
use anyhow::Context as _;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Shared context used by tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Workspace root directory (the agent is restricted to this tree).
    pub workspace_root: PathBuf,
    /// Registry of discovered skills.
    pub skills: Arc<SkillRegistry>,
}

impl ToolContext {
    /// Create a new tool context.
    pub fn new(workspace_root: PathBuf, skills: Arc<SkillRegistry>) -> anyhow::Result<Self> {
        // We canonicalize the root once to make prefix checks robust.
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
    /// This is critical to prevent `..` path traversal from escaping the workspace.
    pub fn resolve_under_workspace(&self, raw: &str) -> anyhow::Result<PathBuf> {
        let raw_path = PathBuf::from(raw);

        // Step 1: interpret relative paths as relative to workspace root.
        let candidate = if raw_path.is_absolute() {
            raw_path
        } else {
            self.workspace_root.join(raw_path)
        };

        // Step 2: canonicalize the *nearest existing ancestor* and then append
        // the remaining path suffix.
        //
        // This approach has two key properties:
        // 1) It works even when the target path does not exist yet (write ops).
        // 2) It still resolves symlinks in the existing prefix, which helps
        //    prevent escaping the workspace through symlink tricks.
        let mut ancestor = candidate.as_path();
        let mut suffix: Vec<std::ffi::OsString> = Vec::new();
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

        // Step 3: enforce workspace boundary.
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

/// The minimal tool executor.
///
/// This converts between:
/// - OpenAI tool definitions (JSON schema)
/// - Local Rust implementations
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
                    name: "shell_command".to_string(),
                    description: "Run a PowerShell command inside the workspace.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "command": { "type": "string", "description": "PowerShell command to execute." },
                            "workdir": { "type": "string", "description": "Optional working directory (relative to workspace root or absolute)."}
                        },
                        "required": ["command"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "read_file".to_string(),
                    description: "Read a UTF-8 text file from the workspace.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Path to the file (relative to workspace root or absolute under it)." }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "write_file".to_string(),
                    description: "Write a UTF-8 text file into the workspace.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Path to write (relative to workspace root or absolute under it)." },
                            "content": { "type": "string", "description": "Full file content to write." },
                            "overwrite": { "type": "boolean", "description": "Whether to overwrite if file exists (default true)." }
                        },
                        "required": ["path", "content"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "list_dir".to_string(),
                    description: "List directory entries under the workspace.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Directory path (relative to workspace root or absolute under it)." }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "list_skills".to_string(),
                    description: "List discovered skills (name + description + directory)."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: ToolFunctionDefinition {
                    name: "load_skill".to_string(),
                    description: "Load the full SKILL.md for a skill by name.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "Skill name (as returned by list_skills)." }
                        },
                        "required": ["name"]
                    }),
                },
            },
        ]
    }

    /// Execute a tool by name with already-parsed JSON arguments.
    ///
    /// Cancellation:
    /// - Long-running tools (especially `shell_command`) must support cooperative
    ///   cancellation so the user can interrupt the agent.
    pub async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        match name {
            "shell_command" => self.shell_command(args, cancel).await,
            "read_file" => self.read_file(args, cancel).await,
            "write_file" => self.write_file(args, cancel).await,
            "list_dir" => self.list_dir(args, cancel).await,
            "list_skills" => self.list_skills(args, cancel).await,
            "load_skill" => self.load_skill(args, cancel).await,
            _ => anyhow::bail!("Unknown tool: {name}"),
        }
    }

    /// Tool: `shell_command`.
    async fn shell_command(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            command: String,
            workdir: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for shell_command")?;

        // Resolve working directory (if provided) under the workspace.
        let workdir = match args.workdir.as_deref() {
            Some(raw) => Some(self.ctx.resolve_under_workspace(raw)?),
            None => None,
        };

        // Spawn PowerShell.
        let mut cmd = tokio::process::Command::new("powershell");
        cmd.arg("-NoProfile");
        cmd.arg("-Command");
        cmd.arg(&args.command);

        // Ensure processes die if we drop them (best-effort).
        cmd.kill_on_drop(true);

        // Set workdir (if any).
        if let Some(workdir) = workdir {
            cmd.current_dir(workdir);
        } else {
            cmd.current_dir(&self.ctx.workspace_root);
        }

        // Collect output with a timeout so the agent cannot hang indefinitely.
        //
        // Cancellation note:
        // - We use `kill_on_drop(true)` above.
        // - If the user interrupts, we drop the output future; Tokio will drop
        //   the child process handle and attempt to kill it.
        let output = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("shell_command cancelled");
            }
            output = tokio::time::timeout(Duration::from_secs(300), cmd.output()) => {
                output
                    .context("shell_command timed out")?
                    .context("shell_command failed to spawn or wait")?
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        // Return as JSON string so the model can parse reliably.
        Ok(serde_json::json!({
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        })
        .to_string())
    }

    /// Tool: `read_file`.
    async fn read_file(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("read_file cancelled");
        }
        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for read_file")?;

        let path = self.ctx.resolve_under_workspace(&args.path)?;

        // Basic size guard (helps avoid dumping megabytes into the context window).
        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file: {}", path.display()))?;
        if meta.len() > 200_000 {
            anyhow::bail!(
                "File is too large to read via tool ({} bytes): {}",
                meta.len(),
                path.display()
            );
        }

        tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))
    }

    /// Tool: `write_file`.
    async fn write_file(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("write_file cancelled");
        }
        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            content: String,
            overwrite: Option<bool>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for write_file")?;

        let overwrite = args.overwrite.unwrap_or(true);

        let path = self.ctx.resolve_under_workspace(&args.path)?;

        // Create parent directories if needed.
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create parent directories: {}", parent.display())
            })?;
        }

        // Enforce overwrite behavior.
        if !overwrite && tokio::fs::try_exists(&path).await? {
            anyhow::bail!(
                "File already exists and overwrite=false: {}",
                path.display()
            );
        }

        tokio::fs::write(&path, args.content.as_bytes())
            .await
            .with_context(|| format!("Failed to write file: {}", path.display()))?;

        Ok(serde_json::json!({
            "written": true,
            "path": path.display().to_string(),
            "bytes": args.content.as_bytes().len(),
        })
        .to_string())
    }

    /// Tool: `list_dir`.
    async fn list_dir(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("list_dir cancelled");
        }
        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for list_dir")?;

        let path = self.ctx.resolve_under_workspace(&args.path)?;

        let mut rd = tokio::fs::read_dir(&path)
            .await
            .with_context(|| format!("Failed to read directory: {}", path.display()))?;

        let mut entries = Vec::new();
        while let Some(entry) = rd.next_entry().await? {
            let file_type = entry.file_type().await?;
            entries.push(serde_json::json!({
                "name": entry.file_name().to_string_lossy(),
                "is_dir": file_type.is_dir(),
                "is_file": file_type.is_file(),
            }));
        }

        Ok(serde_json::json!({
            "path": path.display().to_string(),
            "entries": entries,
        })
        .to_string())
    }

    /// Tool: `list_skills`.
    async fn list_skills(
        &self,
        _args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("list_skills cancelled");
        }
        let skills = self.ctx.skills.list();
        Ok(serde_json::json!({ "skills": skills }).to_string())
    }

    /// Tool: `load_skill`.
    async fn load_skill(
        &self,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("load_skill cancelled");
        }
        #[derive(Debug, Deserialize)]
        struct Args {
            name: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for load_skill")?;
        self.ctx.skills.load_skill_md(&args.name).await
    }
}
