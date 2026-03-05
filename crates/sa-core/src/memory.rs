//! Long-term memory storage for the StudyAdministrator (SA) agent.
//!
//! The user report includes:
//! - "没有长期记忆" ("no long-term memory")
//!
//! We implement a very small, **verifiable** persistence layer:
//! - Store memory entries as JSON Lines (`.jsonl`) on disk.
//! - Load them on daemon startup.
//! - Inject a compact "memory snapshot" into each task's system prompt.
//!
//! This approach keeps the implementation deterministic and easy to inspect:
//! - Users can open the memory file and see exactly what the agent "remembers".
//! - The agent doesn't need a second LLM call to summarize; we simply store
//!   the user's task + the agent's final output (trimmed).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Default directory (under the workspace root) where we store agent state.
pub const DEFAULT_STATE_DIR: &str = ".sa";

/// Legacy directory used by the previous `claw` branding.
///
/// We keep this constant so existing users do not silently lose memory after
/// the rename to SA.
pub const LEGACY_STATE_DIR: &str = ".claw";

/// Default memory filename (under `DEFAULT_STATE_DIR`).
pub const DEFAULT_MEMORY_FILE: &str = "memory.jsonl";

/// A single persisted memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// UTC timestamp when the entry was recorded.
    pub ts: DateTime<Utc>,
    /// Task id (UUID) that produced this memory.
    pub task_id: uuid::Uuid,
    /// User task text (trimmed).
    pub user_task: String,
    /// Agent final answer (trimmed).
    pub final_answer: String,
}

/// In-memory view of the persisted memory file.
#[derive(Debug)]
pub struct MemoryStore {
    /// Where entries are persisted (JSON Lines).
    path: PathBuf,
    /// Recent entries (bounded).
    entries: VecDeque<MemoryEntry>,
    /// Maximum number of entries kept in memory (not necessarily file size).
    max_entries_in_memory: usize,
}

impl MemoryStore {
    /// Create a new store and load existing entries from disk (if present).
    pub fn load_or_new(path: PathBuf) -> anyhow::Result<Self> {
        let mut store = Self {
            path,
            entries: VecDeque::new(),
            max_entries_in_memory: 200,
        };

        store.load_from_disk()?;
        Ok(store)
    }

    /// Return the configured path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Add a memory entry and append it to disk.
    pub fn append(&mut self, entry: MemoryEntry) -> anyhow::Result<()> {
        // Keep in-memory entries bounded.
        self.entries.push_back(entry.clone());
        while self.entries.len() > self.max_entries_in_memory {
            self.entries.pop_front();
        }

        // Ensure parent directory exists.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Append as JSONL.
        let line = serde_json::to_string(&entry)?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?
            .write_all(format!("{line}\n").as_bytes())?;

        Ok(())
    }

    /// Build a compact memory block for prompt injection.
    ///
    /// The goal is to provide *useful continuity* without exploding context.
    pub fn prompt_block(&self) -> String {
        // We only inject a limited number of recent entries.
        let max_entries = 20;
        let mut out = String::new();

        out.push_str("## Long-term memory (persisted)\n\n");
        out.push_str(
            "Below are recent past tasks and final results. Use them as continuity.\n\
If something important is missing, ask the user or inspect files with tools.\n\n",
        );

        let start = self.entries.len().saturating_sub(max_entries);
        for entry in self.entries.iter().skip(start) {
            out.push_str(&format!(
                "- [{}] task_id={} task={}\n  result={}\n",
                entry.ts.to_rfc3339(),
                entry.task_id,
                compact_one_line(&entry.user_task, 200),
                compact_one_line(&entry.final_answer, 400)
            ));
        }

        if self.entries.is_empty() {
            out.push_str("- (no memory entries yet)\n");
        }

        out
    }

    /// Load entries from disk into memory (best-effort).
    fn load_from_disk(&mut self) -> anyhow::Result<()> {
        if !self.path.exists() {
            return Ok(());
        }

        let raw = std::fs::read_to_string(&self.path)?;
        for line in raw.lines() {
            // Skip blank lines.
            if line.trim().is_empty() {
                continue;
            }

            // Best-effort parse: if one line is corrupted, we skip it.
            let parsed = match serde_json::from_str::<MemoryEntry>(line) {
                Ok(p) => p,
                Err(_) => continue,
            };

            self.entries.push_back(parsed);
            while self.entries.len() > self.max_entries_in_memory {
                self.entries.pop_front();
            }
        }

        Ok(())
    }
}

/// Build the default memory path under a workspace root.
pub fn default_memory_path(workspace_root: &Path) -> PathBuf {
    let preferred = workspace_root
        .join(DEFAULT_STATE_DIR)
        .join(DEFAULT_MEMORY_FILE);

    // Backward-compatibility: if the new SA path does not exist yet but the old
    // Claw-branded path exists, keep reading/writing the legacy file so users
    // retain their memory history across the rename.
    if preferred.exists() {
        return preferred;
    }

    let legacy = workspace_root
        .join(LEGACY_STATE_DIR)
        .join(DEFAULT_MEMORY_FILE);
    if legacy.exists() {
        return legacy;
    }

    preferred
}

/// Compact multi-line text into a single line (for prompt readability).
///
/// We also limit the character count to avoid very large injections.
fn compact_one_line(input: &str, max_chars: usize) -> String {
    let one_line = input
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if one_line.chars().count() <= max_chars {
        return one_line;
    }

    let truncated: String = one_line.chars().take(max_chars).collect();
    format!("{truncated}…")
}
