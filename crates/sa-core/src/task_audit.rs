//! Durable audit log for background runtime tasks.
//!
//! Claude Code keeps background-task state, output files, and progress events
//! separate from the main conversation transcript. SA already had durable task
//! state, but no dedicated append-only audit stream. This module fills that
//! gap with a small JSONL log.

use crate::runtime::state::RuntimeTaskStatus;
use anyhow::Context as _;
use chrono::{DateTime, FixedOffset, Local};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Directory containing task-audit JSONL files under `workspace/runtime/`.
pub const TASK_AUDIT_DIR_RELATIVE: &str = "runtime/audit";

/// Default append-only task-audit file name.
pub const TASK_AUDIT_LOG_FILE_NAME: &str = "tasks.jsonl";

/// One audit phase for one background task.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskAuditPhase {
    /// Task accepted and launched.
    Started,
    /// Task reached a terminal state.
    Finished,
}

/// One append-only task-audit entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskAuditEntry {
    /// Stable audit entry id.
    pub id: Uuid,
    /// Local wall-clock timestamp with timezone offset.
    pub local_timestamp: DateTime<FixedOffset>,
    /// Related background task id.
    pub task_id: Uuid,
    /// Owning agent id.
    pub owner_agent_id: Uuid,
    /// Related work id when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_id: Option<Uuid>,
    /// Current audit phase.
    pub phase: TaskAuditPhase,
    /// Task kind string such as `terminal`.
    pub kind: String,
    /// Command summary.
    pub command: String,
    /// Working directory used by the task.
    pub workdir: String,
    /// Best-effort output path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    /// Runtime status when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<RuntimeTaskStatus>,
    /// Exit code when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Safety warning propagated from tool-time validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_warning: Option<String>,
    /// Human-readable outcome summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Filesystem-backed append-only task-audit store.
#[derive(Debug, Clone)]
pub struct TaskAuditStore {
    log_path: PathBuf,
    append_lock: Arc<Mutex<()>>,
}

impl TaskAuditStore {
    /// Create a new task-audit store under the workspace root.
    pub fn new(workspace_root: PathBuf) -> anyhow::Result<Self> {
        let workspace_root = fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root for task audit: {}",
                workspace_root.display()
            )
        })?;
        let log_path = workspace_root
            .join(TASK_AUDIT_DIR_RELATIVE)
            .join(TASK_AUDIT_LOG_FILE_NAME);
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create task audit directory: {}",
                    parent.display()
                )
            })?;
        }
        Ok(Self {
            log_path,
            append_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Return the append-only audit file path.
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Append one audit entry.
    pub fn append(&self, entry: &TaskAuditEntry) -> anyhow::Result<()> {
        let _guard = self
            .append_lock
            .lock()
            .expect("task audit append lock poisoned");

        if let Some(parent) = self.log_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create task audit directory: {}",
                    parent.display()
                )
            })?;
        }
        if let Ok(meta) = fs::symlink_metadata(&self.log_path)
            && meta.file_type().is_symlink()
        {
            anyhow::bail!(
                "Refusing to append task audit log through symlink path: {}",
                self.log_path.display()
            );
        }

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .with_context(|| {
                format!("Failed to open task audit log: {}", self.log_path.display())
            })?;
        let mut line =
            serde_json::to_string(entry).context("Failed to serialize task audit entry")?;
        line.push('\n');
        file.write_all(line.as_bytes()).with_context(|| {
            format!(
                "Failed to append task audit entry: {}",
                self.log_path.display()
            )
        })?;
        file.flush().with_context(|| {
            format!(
                "Failed to flush task audit log: {}",
                self.log_path.display()
            )
        })?;
        file.sync_all()
            .with_context(|| format!("Failed to sync task audit log: {}", self.log_path.display()))
    }

    /// Build and append one new audit entry.
    #[allow(clippy::too_many_arguments)]
    pub fn append_new(
        &self,
        task_id: Uuid,
        owner_agent_id: Uuid,
        work_id: Option<Uuid>,
        phase: TaskAuditPhase,
        kind: impl Into<String>,
        command: impl Into<String>,
        workdir: impl Into<String>,
        output_path: Option<String>,
        status: Option<RuntimeTaskStatus>,
        exit_code: Option<i32>,
        safety_warning: Option<String>,
        summary: Option<String>,
    ) -> anyhow::Result<TaskAuditEntry> {
        let entry = TaskAuditEntry {
            id: Uuid::new_v4(),
            local_timestamp: Local::now().fixed_offset(),
            task_id,
            owner_agent_id,
            work_id,
            phase,
            kind: kind.into(),
            command: command.into(),
            workdir: workdir.into(),
            output_path,
            status,
            exit_code,
            safety_warning,
            summary,
        };
        self.append(&entry)?;
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::{TASK_AUDIT_LOG_FILE_NAME, TaskAuditPhase, TaskAuditStore};
    use crate::runtime::state::RuntimeTaskStatus;
    use std::fs;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn task_audit_store_appends_jsonl_entries() {
        let workspace = tempdir().expect("workspace");
        let store = TaskAuditStore::new(workspace.path().to_path_buf()).expect("task audit store");
        let task_id = Uuid::new_v4();
        let owner_agent_id = Uuid::new_v4();

        store
            .append_new(
                task_id,
                owner_agent_id,
                None,
                TaskAuditPhase::Started,
                "terminal",
                "git status",
                workspace.path().display().to_string(),
                Some("runtime/tasks/test.log".to_string()),
                Some(RuntimeTaskStatus::Running),
                None,
                None,
                None,
            )
            .expect("append start entry");

        let raw = fs::read_to_string(store.log_path()).expect("read audit log");
        assert!(raw.contains(TASK_AUDIT_LOG_FILE_NAME).not());
        assert!(raw.contains("\"phase\":\"started\""));
        assert!(raw.contains("git status"));
    }

    #[cfg(unix)]
    #[test]
    fn task_audit_store_rejects_symlink_log_path() {
        use std::os::unix::fs::symlink;

        let workspace = tempdir().expect("workspace");
        let store = TaskAuditStore::new(workspace.path().to_path_buf()).expect("task audit store");
        let real_target = workspace.path().join("other.jsonl");
        symlink(&real_target, store.log_path()).expect("create symlink log path");

        let err = store
            .append_new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                None,
                TaskAuditPhase::Started,
                "terminal",
                "git status",
                workspace.path().display().to_string(),
                None,
                Some(RuntimeTaskStatus::Running),
                None,
                None,
                None,
            )
            .expect_err("symlink audit path must fail");
        assert!(err.to_string().contains("symlink path"));
    }

    /// Tiny helper to avoid another dependency just for `!bool`.
    trait BoolNot {
        fn not(self) -> bool;
    }

    impl BoolNot for bool {
        fn not(self) -> bool {
            !self
        }
    }
}
