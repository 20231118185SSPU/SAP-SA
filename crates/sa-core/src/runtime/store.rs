//! Durable filesystem-backed storage for the SA multi-agent runtime.
//!
//! This layer owns:
//! - `runtime/root.json`
//! - `runtime/team_state.json`
//! - `runtime/agents/<agent_id>/state.json`
//! - `runtime/agents/<agent_id>/mailbox.jsonl`
//! - `runtime/agents/<agent_id>/pending_question.json`
//! - `runtime/tasks/<task_id>.json`

use crate::runtime::state::{
    AgentState, MailboxEntry, PendingQuestionState, RuntimeTaskState, TeamState,
};
use anyhow::Context as _;
use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;
use uuid::Uuid;

/// Runtime directory under the workspace root.
pub const RUNTIME_DIR_NAME: &str = "runtime";
/// Agent state directory under `runtime/`.
pub const RUNTIME_AGENTS_DIR_NAME: &str = "agents";
/// Runtime task directory under `runtime/`.
pub const RUNTIME_TASKS_DIR_NAME: &str = "tasks";
/// Root marker file under `runtime/`.
pub const ROOT_MARKER_FILE_NAME: &str = "root.json";
/// Team state file under `runtime/`.
pub const TEAM_STATE_FILE_NAME: &str = "team_state.json";
/// Per-agent state file name.
pub const AGENT_STATE_FILE_NAME: &str = "state.json";
/// Per-agent append-only mailbox file name.
pub const AGENT_MAILBOX_FILE_NAME: &str = "mailbox.jsonl";
/// Per-agent pending question file name.
pub const AGENT_PENDING_QUESTION_FILE_NAME: &str = "pending_question.json";

/// Minimal root marker file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RootMarker {
    /// Stable root agent id for this workspace.
    pub root_agent_id: Uuid,
}

/// Filesystem-backed runtime store rooted at one canonical workspace.
#[derive(Debug, Clone)]
pub struct RuntimeStore {
    workspace_root: PathBuf,
    runtime_dir: PathBuf,
    agents_dir: PathBuf,
    tasks_dir: PathBuf,
    root_marker_path: PathBuf,
    team_state_path: PathBuf,
    mailbox_locks: Arc<Mutex<HashMap<Uuid, Arc<Mutex<()>>>>>,
}

impl RuntimeStore {
    /// Create one workspace-bound runtime store and ensure all directories
    /// exist.
    pub fn new(workspace_root: PathBuf) -> anyhow::Result<Self> {
        let workspace_root = fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root for runtime store: {}",
                workspace_root.display()
            )
        })?;
        let runtime_dir = workspace_root.join(RUNTIME_DIR_NAME);
        let agents_dir = runtime_dir.join(RUNTIME_AGENTS_DIR_NAME);
        let tasks_dir = runtime_dir.join(RUNTIME_TASKS_DIR_NAME);
        fs::create_dir_all(&agents_dir)
            .with_context(|| format!("Failed to create agents dir: {}", agents_dir.display()))?;
        fs::create_dir_all(&tasks_dir)
            .with_context(|| format!("Failed to create tasks dir: {}", tasks_dir.display()))?;

        Ok(Self {
            root_marker_path: runtime_dir.join(ROOT_MARKER_FILE_NAME),
            team_state_path: runtime_dir.join(TEAM_STATE_FILE_NAME),
            workspace_root,
            runtime_dir,
            agents_dir,
            tasks_dir,
            mailbox_locks: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Return the in-process append lock for one agent mailbox.
    fn mailbox_lock(&self, agent_id: Uuid) -> Arc<Mutex<()>> {
        let mut locks = self
            .mailbox_locks
            .lock()
            .expect("runtime mailbox_locks mutex poisoned");
        locks
            .entry(agent_id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Return the canonical workspace root.
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Return the runtime directory.
    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    /// Return the path to one agent directory.
    pub fn agent_dir(&self, agent_id: Uuid) -> PathBuf {
        self.agents_dir.join(agent_id.to_string())
    }

    /// Return the per-agent state file path.
    pub fn agent_state_path(&self, agent_id: Uuid) -> PathBuf {
        self.agent_dir(agent_id).join(AGENT_STATE_FILE_NAME)
    }

    /// Return the per-agent mailbox file path.
    pub fn agent_mailbox_path(&self, agent_id: Uuid) -> PathBuf {
        self.agent_dir(agent_id).join(AGENT_MAILBOX_FILE_NAME)
    }

    /// Return the per-agent pending question file path.
    pub fn pending_question_path(&self, agent_id: Uuid) -> PathBuf {
        self.agent_dir(agent_id)
            .join(AGENT_PENDING_QUESTION_FILE_NAME)
    }

    /// Return the runtime task file path.
    pub fn task_state_path(&self, task_id: Uuid) -> PathBuf {
        self.tasks_dir.join(format!("{task_id}.json"))
    }

    /// Persist the root marker.
    pub fn save_root_marker(&self, marker: &RootMarker) -> anyhow::Result<()> {
        write_json_pretty(&self.root_marker_path, marker)
    }

    /// Load the root marker if it already exists.
    pub fn load_root_marker(&self) -> anyhow::Result<Option<RootMarker>> {
        read_json_optional(&self.root_marker_path)
    }

    /// Persist the team state.
    pub fn save_team_state(&self, state: &TeamState) -> anyhow::Result<()> {
        write_json_pretty(&self.team_state_path, state)
    }

    /// Load the team state if it already exists.
    pub fn load_team_state(&self) -> anyhow::Result<Option<TeamState>> {
        read_json_optional(&self.team_state_path)
    }

    /// Persist one agent state.
    pub fn save_agent_state(&self, state: &AgentState) -> anyhow::Result<()> {
        let agent_dir = self.agent_dir(state.agent_id);
        fs::create_dir_all(&agent_dir)
            .with_context(|| format!("Failed to create agent dir: {}", agent_dir.display()))?;
        write_json_pretty(&self.agent_state_path(state.agent_id), state)
    }

    /// Load one agent state, if present.
    pub fn load_agent_state(&self, agent_id: Uuid) -> anyhow::Result<Option<AgentState>> {
        read_json_optional(&self.agent_state_path(agent_id))
    }

    /// Enumerate all persisted agent states.
    pub fn list_agent_states(&self) -> anyhow::Result<Vec<AgentState>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.agents_dir)
            .with_context(|| format!("Failed to read agents dir: {}", self.agents_dir.display()))?
        {
            let entry = entry?;
            let state_path = entry.path().join(AGENT_STATE_FILE_NAME);
            if let Some(state) = read_json_optional::<AgentState>(&state_path)? {
                out.push(state);
            }
        }
        out.sort_by_key(|state| state.agent_id);
        Ok(out)
    }

    /// Persist one runtime task state.
    pub fn save_task_state(&self, task: &RuntimeTaskState) -> anyhow::Result<()> {
        write_json_pretty(&self.task_state_path(task.task_id), task)
    }

    /// Load one runtime task state if it exists.
    pub fn load_task_state(&self, task_id: Uuid) -> anyhow::Result<Option<RuntimeTaskState>> {
        read_json_optional(&self.task_state_path(task_id))
    }

    /// Enumerate all persisted runtime tasks.
    pub fn list_task_states(&self) -> anyhow::Result<Vec<RuntimeTaskState>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.tasks_dir)
            .with_context(|| format!("Failed to read tasks dir: {}", self.tasks_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if let Some(task) = read_json_optional::<RuntimeTaskState>(&path)? {
                out.push(task);
            }
        }
        out.sort_by_key(|task| task.task_id);
        Ok(out)
    }

    /// Persist a pending question for one agent.
    pub fn save_pending_question(&self, question: &PendingQuestionState) -> anyhow::Result<()> {
        let agent_dir = self.agent_dir(question.agent_id);
        fs::create_dir_all(&agent_dir)
            .with_context(|| format!("Failed to create agent dir: {}", agent_dir.display()))?;
        write_json_pretty(&self.pending_question_path(question.agent_id), question)
    }

    /// Load a pending question for one agent if it exists.
    pub fn load_pending_question(
        &self,
        agent_id: Uuid,
    ) -> anyhow::Result<Option<PendingQuestionState>> {
        read_json_optional(&self.pending_question_path(agent_id))
    }

    /// Remove the pending-question file for one agent if it exists.
    pub fn clear_pending_question(&self, agent_id: Uuid) -> anyhow::Result<()> {
        let path = self.pending_question_path(agent_id);
        if !path.exists() {
            return Ok(());
        }
        fs::remove_file(&path)
            .with_context(|| format!("Failed to remove pending question: {}", path.display()))
    }

    /// Append one mailbox entry and assign the next monotonic offset.
    pub fn append_mailbox_entry(
        &self,
        agent_id: Uuid,
        mut entry: MailboxEntry,
    ) -> anyhow::Result<MailboxEntry> {
        let lock = self.mailbox_lock(agent_id);
        let _guard = lock.lock().expect("runtime mailbox append lock poisoned");
        let mailbox_path = self.agent_mailbox_path(agent_id);
        let agent_dir = self.agent_dir(agent_id);
        fs::create_dir_all(&agent_dir)
            .with_context(|| format!("Failed to create agent dir: {}", agent_dir.display()))?;

        let next_offset = self
            .read_mailbox(agent_id)?
            .last()
            .map(|item| item.offset.saturating_add(1))
            .unwrap_or(1);
        entry.offset = next_offset;

        let mut line = serde_json::to_string(&entry).context("Failed to serialize mailbox entry")?;
        line.push('\n');
        append_text_file(&mailbox_path, &line)
            .with_context(|| format!("Failed to append mailbox entry: {}", mailbox_path.display()))?;
        Ok(entry)
    }

    /// Read the full mailbox for one agent.
    pub fn read_mailbox(&self, agent_id: Uuid) -> anyhow::Result<Vec<MailboxEntry>> {
        let mailbox_path = self.agent_mailbox_path(agent_id);
        if !mailbox_path.is_file() {
            return Ok(Vec::new());
        }
        let raw = fs::read_to_string(&mailbox_path)
            .with_context(|| format!("Failed to read mailbox: {}", mailbox_path.display()))?;
        raw.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<MailboxEntry>(line).context("Invalid mailbox JSONL entry"))
            .collect()
    }

    /// Read mailbox entries whose offset is greater than `last_offset`.
    pub fn read_mailbox_after(
        &self,
        agent_id: Uuid,
        last_offset: u64,
    ) -> anyhow::Result<Vec<MailboxEntry>> {
        Ok(self
            .read_mailbox(agent_id)?
            .into_iter()
            .filter(|entry| entry.offset > last_offset)
            .collect())
    }
}

/// Write one JSON document atomically-ish by replacing the entire target file.
fn write_json_pretty<T: serde::Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let raw = serde_json::to_string_pretty(value).context("Failed to serialize JSON state")?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("JSON state path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create parent dir: {}", parent.display()))?;

    let mut temp = NamedTempFile::new_in(parent)
        .with_context(|| format!("Failed to create temp JSON file in {}", parent.display()))?;
    temp.write_all(raw.as_bytes())
        .with_context(|| format!("Failed to write temp JSON state for {}", path.display()))?;
    temp.flush()
        .with_context(|| format!("Failed to flush temp JSON state for {}", path.display()))?;
    temp.as_file()
        .sync_all()
        .with_context(|| format!("Failed to sync temp JSON state for {}", path.display()))?;

    if path.exists() {
        #[cfg(windows)]
        {
            fs::remove_file(path).with_context(|| {
                format!("Failed to replace existing JSON state at {}", path.display())
            })?;
        }
    }

    temp.persist(path).map_err(|err| {
        anyhow::anyhow!(
            "Failed to persist JSON state to {}: {}",
            path.display(),
            err.error
        )
    })?;

    if let Ok(dir) = fs::File::open(parent) {
        let _ = dir.sync_all();
    }

    Ok(())
}

/// Read one JSON file if it exists.
fn read_json_optional<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<Option<T>> {
    if !path.is_file() {
        return Ok(None);
    }
    let raw =
        fs::read_to_string(path).with_context(|| format!("Failed to read JSON: {}", path.display()))?;
    let parsed = serde_json::from_str::<T>(&raw)
        .with_context(|| format!("Failed to parse JSON: {}", path.display()))?;
    Ok(Some(parsed))
}

/// Append text to one file, creating it if needed.
fn append_text_file(path: &Path, text: &str) -> anyhow::Result<()> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create parent dir: {}", parent.display()))?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open append target: {}", path.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("Failed to append text: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::state::{
        AgentKind, AgentState, AgentStatus, MailboxEntryKind, RuntimeTaskKind, RuntimeTaskStatus,
    };
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_store() -> (TempDir, RuntimeStore) {
        let workspace = TempDir::new().expect("temp workspace should build");
        let store =
            RuntimeStore::new(workspace.path().to_path_buf()).expect("runtime store should build");
        (workspace, store)
    }

    #[test]
    fn runtime_store_persists_root_marker_and_team_state() {
        let (_workspace, store) = create_store();
        let root_id = Uuid::new_v4();
        store
            .save_root_marker(&RootMarker {
                root_agent_id: root_id,
            })
            .expect("root marker should persist");
        store
            .save_team_state(&TeamState::new(root_id))
            .expect("team state should persist");

        assert_eq!(
            store
                .load_root_marker()
                .expect("root marker should load")
                .expect("root marker should exist")
                .root_agent_id,
            root_id
        );
        assert_eq!(
            store
                .load_team_state()
                .expect("team state should load")
                .expect("team state should exist")
                .root_agent_id,
            root_id
        );
    }

    #[test]
    fn runtime_store_appends_mailbox_offsets_monotonically() {
        let (_workspace, store) = create_store();
        let agent_id = Uuid::new_v4();

        let first = store
            .append_mailbox_entry(
                agent_id,
                MailboxEntry {
                    offset: 0,
                    entry_id: Uuid::new_v4(),
                    created_at: Utc::now(),
                    kind: MailboxEntryKind::UserInput,
                    from_agent_id: None,
                    from_label: None,
                    message: "hello".to_string(),
                    work_id: None,
                    related_id: None,
                },
            )
            .expect("first mailbox entry should append");
        let second = store
            .append_mailbox_entry(
                agent_id,
                MailboxEntry {
                    offset: 0,
                    entry_id: Uuid::new_v4(),
                    created_at: Utc::now(),
                    kind: MailboxEntryKind::SystemNotice,
                    from_agent_id: None,
                    from_label: None,
                    message: "note".to_string(),
                    work_id: None,
                    related_id: None,
                },
            )
            .expect("second mailbox entry should append");

        assert_eq!(first.offset, 1);
        assert_eq!(second.offset, 2);
        assert_eq!(
            store
                .read_mailbox_after(agent_id, 1)
                .expect("mailbox slice should load")
                .len(),
            1
        );
    }

    #[test]
    fn runtime_store_serializes_concurrent_mailbox_offsets() {
        use std::sync::Arc;
        use std::thread;

        let (_workspace, store) = create_store();
        let store = Arc::new(store);
        let agent_id = Uuid::new_v4();

        let mut handles = Vec::new();
        for _ in 0..16 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                store
                    .append_mailbox_entry(
                        agent_id,
                        MailboxEntry {
                            offset: 0,
                            entry_id: Uuid::new_v4(),
                            created_at: Utc::now(),
                            kind: MailboxEntryKind::SystemNotice,
                            from_agent_id: None,
                            from_label: None,
                            message: "race".to_string(),
                            work_id: None,
                            related_id: None,
                        },
                    )
                    .expect("concurrent mailbox append should succeed");
            }));
        }

        for handle in handles {
            handle.join().expect("mailbox thread should finish");
        }

        let mut offsets = store
            .read_mailbox(agent_id)
            .expect("mailbox should load")
            .into_iter()
            .map(|entry| entry.offset)
            .collect::<Vec<_>>();
        offsets.sort_unstable();
        assert_eq!(offsets, (1..=16).collect::<Vec<_>>());
    }

    #[test]
    fn runtime_store_persists_agent_and_task_state() {
        let (_workspace, store) = create_store();
        let agent_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        let agent = AgentState {
            agent_id,
            parent_agent_id: None,
            root_agent_id: agent_id,
            kind: AgentKind::Root,
            label: "root".to_string(),
            status: AgentStatus::Idle,
            allow_user_send: true,
            allow_user_show: true,
            allow_user_ask: true,
            allow_input_transfer_target: false,
            current_session_path: "sessions/root.jsonl".to_string(),
            previous_session_path: None,
            tool_session_read_set: vec![],
            last_mailbox_offset: 0,
            active_work_id: None,
            active_work_summary: None,
            active_started_at: None,
            work_has_user_output: false,
            work_has_parent_message: false,
            parent_work_id: None,
            needs_finish_reminder: false,
            waiting_on: None,
            pending_finish_confirmation: None,
            pending_control: None,
            last_finish_reason: None,
            last_finish_result: None,
            last_finished_work_id: None,
            last_finished_at: None,
        };
        let task = RuntimeTaskState {
            task_id,
            owner_agent_id: agent_id,
            kind: RuntimeTaskKind::Terminal,
            status: RuntimeTaskStatus::Running,
            command: "cargo test".to_string(),
            workdir: "G:/repo".to_string(),
            started_at: Utc::now(),
            finished_at: None,
            exit_code: None,
            output_path: Some("runtime/tasks/log.txt".to_string()),
            summary: None,
            metadata: Default::default(),
        };

        store
            .save_agent_state(&agent)
            .expect("agent state should persist");
        store.save_task_state(&task).expect("task state should persist");

        assert_eq!(
            store
                .load_agent_state(agent_id)
                .expect("agent state should load")
                .expect("agent state should exist")
                .label,
            "root"
        );
        assert_eq!(
            store
                .load_task_state(task_id)
                .expect("task state should load")
                .expect("task state should exist")
                .status,
            RuntimeTaskStatus::Running
        );
    }
}
