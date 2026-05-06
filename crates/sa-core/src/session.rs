//! Persistent JSONL session storage for the top-level SA conversation.
//!
//! Goals of this module:
//! - keep the main conversation durable across backend restarts
//! - store every real conversation item as JSONL under `workspace/sessions/`
//! - let the agent rebuild its next request from the stored session file
//! - rotate to a fresh session file after each history compaction
//! - expose the previous session-file path so the prompt can point the model to
//!   the exact raw transcript that was compacted away
//!
//! Design notes:
//! - Only the **top-level** conversation is persisted here. Sub-agents remain
//!   ephemeral because they are short-lived helper runs.
//! - The current active segment is tracked by `workspace/sessions/.current`.
//! - Each segment is a standalone JSONL file:
//!   - first line: `session_meta`
//!   - optional second line: `compaction_checkpoint`
//!   - remaining lines: `message`
//! - When compaction happens we do **not** mutate the old segment. Instead we
//!   create a brand-new segment that points back to the prior one.
//! - On resume we load only the current segment and reconstruct:
//!   - the real conversation messages
//!   - the active compaction summary, if any
//! - If the process crashed in the middle of a tool-calling turn, we truncate
//!   the incomplete suffix so the restored history is replayable.

use crate::openai::{ChatMessage, ChatUsage, MessageContent, ResponsesInputItem, ToolCall};
use anyhow::Context as _;
use chrono::{DateTime, Utc};
use fs4::fs_std::FileExt as _;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Directory name used for persistent session files under the workspace root.
pub const SESSIONS_DIR_NAME: &str = "sessions";

/// Pointer file that stores the workspace-relative path of the active session
/// segment.
const CURRENT_SESSION_POINTER_FILE_NAME: &str = ".current";

/// Lock file used to guard one workspace session directory from concurrent
/// backend ownership.
const SESSION_STORE_LOCK_FILE_NAME: &str = ".sa-session.lock";

/// JSONL schema version for session files.
const SESSION_SCHEMA_VERSION: u32 = 1;

/// Lightweight metadata about the currently active session segment.
#[derive(Debug, Clone)]
pub struct SessionDescriptor {
    /// Stable conversation identifier shared by all rotated segments.
    pub conversation_id: Uuid,
    /// Workspace-relative path of the active segment, using forward slashes.
    pub current_session_path: String,
    /// Workspace-relative path of the previous segment, if compaction created
    /// the current segment from an older one.
    pub previous_session_path: Option<String>,
}

impl SessionDescriptor {
    /// Build the runtime prompt block injected into the system instructions.
    ///
    /// This block is intentionally framed as **compaction context** so the
    /// model understands that the previous session file is the raw source of
    /// history that was compressed out of the current prompt window.
    pub fn compaction_prompt_block(&self) -> String {
        let mut out = String::from("## 压缩上下文\n\n");
        let _ = writeln!(out, "- 当前压缩后会话文件：`{}`", self.current_session_path);

        if let Some(previous) = self.previous_session_path.as_deref() {
            let _ = writeln!(out, "- 上一段原始会话文件：`{previous}`");
            out.push_str(
                "- 如果你需要被压缩掉的更早原始聊天记录，请优先使用 `Read` 精确读取上面给出的上一段原始会话文件。\n",
            );
        } else {
            out.push_str("- 当前还没有更早的会话分段。\n");
        }

        out
    }
}

/// Full in-memory reconstruction of the active persisted session.
#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    /// Active session-segment metadata.
    pub descriptor: SessionDescriptor,
    /// Active compaction summary restored from the segment, if any.
    pub compaction_summary: Option<String>,
    /// Real conversation messages that should be replayed verbatim.
    pub messages: Vec<ChatMessage>,
    /// Number of trailing messages dropped because a crash left a partial
    /// tool-calling turn that could not be replayed safely.
    pub truncated_incomplete_messages: usize,
}

/// Durable session store rooted at one workspace.
#[derive(Debug, Clone)]
pub struct SessionStore {
    /// Canonical workspace root.
    workspace_root: PathBuf,
    /// `<workspace>/sessions`.
    sessions_dir: PathBuf,
    /// `<workspace>/sessions/.current`.
    pointer_path: PathBuf,
    /// File handle that keeps the cross-process lock alive.
    _process_lock: Arc<File>,
    /// Mutable pointer to the currently active segment.
    state: Arc<Mutex<SessionState>>,
}

/// Internal mutable state describing the active segment.
#[derive(Debug, Clone)]
struct SessionState {
    /// Stable conversation identifier.
    conversation_id: Uuid,
    /// Absolute path of the active segment file.
    current_absolute_path: PathBuf,
    /// Workspace-relative path of the active segment file.
    current_relative_path: String,
    /// Workspace-relative path of the previous segment file, if any.
    previous_relative_path: Option<String>,
}

/// One parsed session file.
#[derive(Debug, Clone)]
struct ParsedSessionFile {
    /// Metadata line.
    meta: SessionMetaEntry,
    /// Optional compaction checkpoint line.
    compaction_summary: Option<String>,
    /// Real conversation messages stored in this segment.
    messages: Vec<ChatMessage>,
    /// Whether the parser discarded one truncated trailing JSONL fragment.
    truncated_partial_tail: bool,
}

/// One replay-safety repair result.
#[derive(Debug, Clone)]
struct SanitizedMessages {
    /// Repaired replayable message list.
    messages: Vec<ChatMessage>,
    /// Number of tail messages removed.
    truncated_count: usize,
}

/// All supported JSONL entry types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionEntry {
    /// Metadata for the current segment.
    SessionMeta(SessionMetaEntry),
    /// Checkpoint created after compaction.
    CompactionCheckpoint(CompactionCheckpointEntry),
    /// One persisted conversation message.
    Message(SessionMessageEntry),
}

/// First line of every session segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionMetaEntry {
    /// Schema version for forward/backward debugging.
    schema_version: u32,
    /// Stable conversation identifier shared by all rotated segments.
    conversation_id: Uuid,
    /// Unique segment identifier for this file only.
    segment_id: Uuid,
    /// Creation timestamp of the segment file.
    created_at: DateTime<Utc>,
    /// Workspace-relative path of this segment.
    current_session_path: String,
    /// Workspace-relative path of the immediately previous segment, if any.
    previous_session_path: Option<String>,
}

/// Optional checkpoint line written immediately after compaction rotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactionCheckpointEntry {
    /// Creation timestamp of the checkpoint line.
    created_at: DateTime<Utc>,
    /// The active compaction summary that replaces older raw history.
    summary: String,
}

/// One persisted conversation message line.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionMessageEntry {
    /// Write timestamp for traceability.
    created_at: DateTime<Utc>,
    /// Full message payload.
    message: StoredChatMessage,
}

/// Fully serializable local copy of `ChatMessage`.
///
/// We cannot serialize `ChatMessage` directly because some internal fields are
/// intentionally skipped for provider requests, but we still need them on disk
/// so resumed sessions can reconstruct the exact local state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredChatMessage {
    /// Message role (`user`, `assistant`, `tool`, ...).
    role: String,
    /// Optional text or multimodal content.
    ///
    /// Uses `#[serde(untagged)]` via `MessageContent` so that existing on-disk
    /// sessions containing a plain JSON string deserialise transparently.
    content: Option<MessageContent>,
    /// Optional assistant tool calls.
    tool_calls: Option<Vec<ToolCall>>,
    /// Tool-result correlation id.
    tool_call_id: Option<String>,
    /// Local request-usage snapshot anchored to this assistant turn.
    request_usage: Option<ChatUsage>,
    /// Normalized Responses replay items preserved across turns.
    responses_input_items: Option<Vec<ResponsesInputItem>>,
}

impl From<&ChatMessage> for StoredChatMessage {
    /// Convert one in-memory message into the on-disk representation.
    fn from(value: &ChatMessage) -> Self {
        Self {
            role: value.role.clone(),
            content: value.content.clone(),
            tool_calls: value.tool_calls.clone(),
            tool_call_id: value.tool_call_id.clone(),
            request_usage: value.request_usage.clone(),
            responses_input_items: value.responses_input_items.clone(),
        }
    }
}

impl From<StoredChatMessage> for ChatMessage {
    /// Restore one on-disk message into the in-memory canonical format.
    fn from(value: StoredChatMessage) -> Self {
        Self {
            role: value.role,
            content: value.content,
            tool_calls: value.tool_calls,
            tool_call_id: value.tool_call_id,
            request_usage: value.request_usage,
            responses_input_items: value.responses_input_items,
        }
    }
}

impl SessionStore {
    /// Create one workspace-bound session store.
    ///
    /// If no active segment exists yet, this eagerly creates a fresh one so the
    /// first user turn can be appended immediately.
    pub fn new(workspace_root: PathBuf) -> anyhow::Result<Self> {
        Self::new_in_relative_dir(workspace_root, Path::new(SESSIONS_DIR_NAME))
    }

    /// Create one workspace-bound session store rooted at a custom relative
    /// directory under the workspace.
    ///
    /// This is used by the durable multi-agent runtime so every agent can keep
    /// an isolated transcript under paths such as `sessions/agents/<agent_id>/`.
    pub fn new_in_relative_dir(
        workspace_root: PathBuf,
        sessions_relative_dir: &Path,
    ) -> anyhow::Result<Self> {
        let workspace_root = fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root for session storage: {}",
                workspace_root.display()
            )
        })?;
        let sessions_relative_dir = validate_session_relative_dir(sessions_relative_dir)
            .context("Invalid session directory override")?;
        let sessions_dir = workspace_root.join(&sessions_relative_dir);
        fs::create_dir_all(&sessions_dir).with_context(|| {
            format!(
                "Failed to create session directory: {}",
                sessions_dir.display()
            )
        })?;
        let sessions_dir = fs::canonicalize(&sessions_dir).with_context(|| {
            format!(
                "Failed to canonicalize session directory after creation: {}",
                sessions_dir.display()
            )
        })?;
        ensure_path_is_under_directory(&sessions_dir, &workspace_root).with_context(|| {
            format!(
                "Session directory resolved outside workspace root: {}",
                sessions_dir.display()
            )
        })?;
        let pointer_path = sessions_dir.join(CURRENT_SESSION_POINTER_FILE_NAME);
        let process_lock = acquire_session_store_lock(&sessions_dir)?;
        let state = resolve_or_create_current_state(&workspace_root, &sessions_dir, &pointer_path)?;

        Ok(Self {
            workspace_root,
            sessions_dir,
            pointer_path,
            _process_lock: Arc::new(process_lock),
            state: Arc::new(Mutex::new(state)),
        })
    }

    /// Load the active segment from disk and reconstruct the replay state.
    ///
    /// If a previous crash left a partial tool-calling suffix, this method
    /// repairs the active file in place so later restarts do not repeat the
    /// same truncation work.
    pub fn load_snapshot(&self) -> anyhow::Result<SessionSnapshot> {
        let state = self
            .state
            .lock()
            .expect("session state mutex poisoned")
            .clone();
        let parsed = parse_session_file(&state.current_absolute_path)?;
        let repaired = sanitize_replayable_messages(parsed.messages);

        if parsed.truncated_partial_tail || repaired.truncated_count > 0 {
            rewrite_existing_segment(
                &state.current_absolute_path,
                &parsed.meta,
                parsed.compaction_summary.as_deref(),
                &repaired.messages,
            )?;
        }

        Ok(SessionSnapshot {
            descriptor: SessionDescriptor {
                conversation_id: parsed.meta.conversation_id,
                current_session_path: state.current_relative_path,
                previous_session_path: parsed.meta.previous_session_path,
            },
            compaction_summary: parsed.compaction_summary,
            messages: repaired.messages,
            truncated_incomplete_messages: repaired.truncated_count,
        })
    }

    /// Append one real conversation message to the active segment.
    pub fn append_message(&self, message: &ChatMessage) -> anyhow::Result<()> {
        let state = self
            .state
            .lock()
            .expect("session state mutex poisoned")
            .clone();
        let entry = SessionEntry::Message(SessionMessageEntry {
            created_at: Utc::now(),
            message: StoredChatMessage::from(message),
        });
        append_jsonl_entry(&state.current_absolute_path, &entry)
    }

    /// Check whether the active segment already contains a concrete `tool`
    /// result line for one tool-call id, without triggering replay repair.
    ///
    /// Why this exists:
    /// - a pending `Ask` legitimately leaves an assistant tool-call turn
    ///   without its `tool` result until the classmate answers
    /// - `load_snapshot()` intentionally repairs such incomplete tails for
    ///   replay safety, which would be incorrect while that answer is still
    ///   pending
    /// - the durable runtime therefore needs one raw existence check when it
    ///   is about to append the eventual answer payload
    pub fn has_raw_tool_result(&self, tool_call_id: &str) -> anyhow::Result<bool> {
        let state = self
            .state
            .lock()
            .expect("session state mutex poisoned")
            .clone();
        let parsed = parse_session_file(&state.current_absolute_path)?;
        Ok(parsed.messages.iter().any(|message| {
            message.role == "tool" && message.tool_call_id.as_deref() == Some(tool_call_id)
        }))
    }

    /// Rotate to a fresh segment after compaction and carry forward the compact
    /// summary plus retained real messages.
    pub fn rollover_after_compaction(
        &self,
        summary: &str,
        kept_messages: &[ChatMessage],
    ) -> anyhow::Result<SessionDescriptor> {
        let current_state = self
            .state
            .lock()
            .expect("session state mutex poisoned")
            .clone();
        let next_state = create_new_segment(
            &self.workspace_root,
            &self.sessions_dir,
            &self.pointer_path,
            current_state.conversation_id,
            Some(current_state.current_relative_path.clone()),
            Some(summary),
            kept_messages,
        )?;

        let descriptor = SessionDescriptor {
            conversation_id: next_state.conversation_id,
            current_session_path: next_state.current_relative_path.clone(),
            previous_session_path: next_state.previous_relative_path.clone(),
        };

        let mut guard = self.state.lock().expect("session state mutex poisoned");
        *guard = next_state;
        Ok(descriptor)
    }

    /// Return the workspace-relative path of the active segment.
    pub fn current_session_path(&self) -> String {
        self.state
            .lock()
            .expect("session state mutex poisoned")
            .current_relative_path
            .clone()
    }

    /// Return the canonical workspace root used by this store.
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }
}

/// Validate that the custom session directory is a normalized relative path
/// that stays under the workspace root.
fn validate_session_relative_dir(raw: &Path) -> anyhow::Result<PathBuf> {
    if raw.as_os_str().is_empty() {
        anyhow::bail!("Session directory override must not be empty");
    }
    if raw.is_absolute() {
        anyhow::bail!("Session directory override must be workspace-relative");
    }

    let mut normalized = PathBuf::new();
    for component in raw.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                anyhow::bail!("Session directory override must not contain `..`");
            }
            Component::Prefix(_) | Component::RootDir => {
                anyhow::bail!("Session directory override must stay relative");
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        anyhow::bail!("Session directory override must not normalize to the workspace root");
    }

    Ok(normalized)
}

/// Resolve the current active segment, or create a brand-new one if this is
/// the first run for the workspace.
fn resolve_or_create_current_state(
    workspace_root: &Path,
    sessions_dir: &Path,
    pointer_path: &Path,
) -> anyhow::Result<SessionState> {
    if let Some(relative_path) = read_pointer_file(pointer_path)? {
        let absolute_path =
            resolve_session_segment_path(workspace_root, sessions_dir, &relative_path)?;
        if absolute_path.is_file() {
            let parsed = parse_session_file(&absolute_path)?;
            return Ok(SessionState {
                conversation_id: parsed.meta.conversation_id,
                current_absolute_path: absolute_path,
                current_relative_path: relative_path,
                previous_relative_path: parsed.meta.previous_session_path,
            });
        }
    }

    if let Some(absolute_path) = latest_session_file(workspace_root, sessions_dir)? {
        let parsed = parse_session_file(&absolute_path)?;
        let relative_path = workspace_relative_path(workspace_root, &absolute_path)?;
        write_pointer_file(pointer_path, &relative_path)?;
        return Ok(SessionState {
            conversation_id: parsed.meta.conversation_id,
            current_absolute_path: absolute_path,
            current_relative_path: relative_path,
            previous_relative_path: parsed.meta.previous_session_path,
        });
    }

    create_new_segment(
        workspace_root,
        sessions_dir,
        pointer_path,
        Uuid::new_v4(),
        None,
        None,
        &[],
    )
}

/// Create one brand-new segment file and make it the active pointer target.
fn create_new_segment(
    workspace_root: &Path,
    sessions_dir: &Path,
    pointer_path: &Path,
    conversation_id: Uuid,
    previous_relative_path: Option<String>,
    compaction_summary: Option<&str>,
    messages: &[ChatMessage],
) -> anyhow::Result<SessionState> {
    let segment_id = Uuid::new_v4();
    let file_name = format!(
        "session-{}-{}.jsonl",
        Utc::now().format("%Y-%m-%dT%H-%M-%SZ"),
        segment_id
    );
    let current_absolute_path = sessions_dir.join(&file_name);
    let current_relative_path = workspace_relative_path(workspace_root, &current_absolute_path)?;
    let meta = SessionMetaEntry {
        schema_version: SESSION_SCHEMA_VERSION,
        conversation_id,
        segment_id,
        created_at: Utc::now(),
        current_session_path: current_relative_path.clone(),
        previous_session_path: previous_relative_path.clone(),
    };
    write_new_segment_file(&current_absolute_path, &meta, compaction_summary, messages)?;
    write_pointer_file(pointer_path, &current_relative_path)?;

    Ok(SessionState {
        conversation_id,
        current_absolute_path,
        current_relative_path,
        previous_relative_path,
    })
}

/// Parse one full session segment file.
fn parse_session_file(path: &Path) -> anyhow::Result<ParsedSessionFile> {
    let raw = fs::read(path)
        .with_context(|| format!("Failed to read session file: {}", path.display()))?;
    let mut meta = None::<SessionMetaEntry>;
    let mut compaction_summary = None::<String>;
    let mut messages = Vec::<ChatMessage>::new();
    let mut truncated_partial_tail = false;
    let mut line_number = 0usize;
    let mut line_start = 0usize;

    for (index, byte) in raw.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }

        line_number += 1;
        if let Some(entry) = parse_session_line(&raw[line_start..index], path, line_number)? {
            apply_session_entry(
                entry,
                path,
                line_number,
                &mut meta,
                &mut compaction_summary,
                &mut messages,
            )?;
        }
        line_start = index + 1;
    }

    if line_start < raw.len() {
        line_number += 1;
        match parse_session_line(&raw[line_start..], path, line_number) {
            Ok(Some(entry)) => {
                apply_session_entry(
                    entry,
                    path,
                    line_number,
                    &mut meta,
                    &mut compaction_summary,
                    &mut messages,
                )?;
            }
            Ok(None) => {}
            Err(_) => {
                // A crash can leave one final JSONL fragment without a newline.
                // We discard only that trailing fragment; all earlier lines stay
                // strict and must still parse.
                truncated_partial_tail = true;
            }
        }
    }

    let meta = meta.ok_or_else(|| {
        anyhow::anyhow!(
            "Session file {} does not contain `session_meta`",
            path.display()
        )
    })?;

    Ok(ParsedSessionFile {
        meta,
        compaction_summary,
        messages,
        truncated_partial_tail,
    })
}

/// Repair a stored message list so it can be replayed safely.
///
/// Crash scenario handled here:
/// - assistant emitted tool calls
/// - the process died before all corresponding `tool` results were appended
/// - replaying such a suffix would leave the model mid-turn with unresolved
///   tool calls and no way to resume the original execution context
///
/// In that case we truncate the whole incomplete assistant-turn suffix.
fn sanitize_replayable_messages(messages: Vec<ChatMessage>) -> SanitizedMessages {
    #[derive(Debug)]
    struct PendingToolTurn {
        /// Index of the assistant message that started the tool-calling turn.
        start_index: usize,
        /// Remaining tool-call ids that still need a `tool` result message.
        pending_call_ids: HashSet<String>,
    }

    let mut active = None::<PendingToolTurn>;

    for (index, message) in messages.iter().enumerate() {
        if let Some(pending) = active.as_mut() {
            if message.role == "tool" {
                if let Some(tool_call_id) = message.tool_call_id.as_deref() {
                    pending.pending_call_ids.remove(tool_call_id);
                }
                continue;
            }

            if !pending.pending_call_ids.is_empty() {
                return SanitizedMessages {
                    messages: messages[..pending.start_index].to_vec(),
                    truncated_count: messages.len().saturating_sub(pending.start_index),
                };
            }

            active = None;
        }

        if message.role == "assistant" {
            let Some(tool_calls) = message.tool_calls.as_ref() else {
                continue;
            };
            if tool_calls.is_empty() {
                continue;
            }

            active = Some(PendingToolTurn {
                start_index: index,
                pending_call_ids: tool_calls.iter().map(|call| call.id.clone()).collect(),
            });
        }
    }

    if let Some(pending) = active {
        if !pending.pending_call_ids.is_empty() {
            return SanitizedMessages {
                messages: messages[..pending.start_index].to_vec(),
                truncated_count: messages.len().saturating_sub(pending.start_index),
            };
        }
    }

    SanitizedMessages {
        messages,
        truncated_count: 0,
    }
}

/// Rewrite an existing segment in place after replay-safety repair.
fn rewrite_existing_segment(
    path: &Path,
    meta: &SessionMetaEntry,
    compaction_summary: Option<&str>,
    messages: &[ChatMessage],
) -> anyhow::Result<()> {
    let entries = build_segment_entries(meta.clone(), compaction_summary, messages);
    write_existing_segment_file(path, &entries)
}

/// Build the full ordered JSONL entry list for one segment.
fn build_segment_entries(
    meta: SessionMetaEntry,
    compaction_summary: Option<&str>,
    messages: &[ChatMessage],
) -> Vec<SessionEntry> {
    let mut entries = Vec::<SessionEntry>::with_capacity(
        1 + usize::from(compaction_summary.is_some()) + messages.len(),
    );
    entries.push(SessionEntry::SessionMeta(meta));

    if let Some(summary) = compaction_summary {
        entries.push(SessionEntry::CompactionCheckpoint(
            CompactionCheckpointEntry {
                created_at: Utc::now(),
                summary: summary.to_string(),
            },
        ));
    }

    entries.extend(messages.iter().map(|message| {
        SessionEntry::Message(SessionMessageEntry {
            created_at: Utc::now(),
            message: StoredChatMessage::from(message),
        })
    }));

    entries
}

/// Write one new segment file. The target path must not exist yet.
fn write_new_segment_file(
    path: &Path,
    meta: &SessionMetaEntry,
    compaction_summary: Option<&str>,
    messages: &[ChatMessage],
) -> anyhow::Result<()> {
    let entries = build_segment_entries(meta.clone(), compaction_summary, messages);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("Failed to create new session file: {}", path.display()))?;

    write_entries(&mut file, &entries).with_context(|| {
        format!(
            "Failed to write new session file contents: {}",
            path.display()
        )
    })
}

/// Overwrite an existing segment file with freshly generated JSONL entries.
fn write_existing_segment_file(path: &Path, entries: &[SessionEntry]) -> anyhow::Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("Failed to rewrite session file: {}", path.display()))?;
    write_entries(&mut file, entries).with_context(|| {
        format!(
            "Failed to rewrite session file contents: {}",
            path.display()
        )
    })
}

/// Write a full JSONL file from ordered session entries.
fn write_entries(file: &mut File, entries: &[SessionEntry]) -> anyhow::Result<()> {
    for entry in entries {
        let line = serde_json::to_string(entry).context("Failed to serialize session entry")?;
        file.write_all(line.as_bytes())
            .context("Failed to write session JSONL entry")?;
        file.write_all(b"\n")
            .context("Failed to write session JSONL newline")?;
    }
    file.flush().context("Failed to flush session file")?;
    Ok(())
}

/// Append one JSONL entry to an existing active segment.
fn append_jsonl_entry(path: &Path, entry: &SessionEntry) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open session file for append: {}", path.display()))?;
    let line = serde_json::to_string(entry).context("Failed to serialize session entry")?;
    file.write_all(line.as_bytes())
        .context("Failed to append session JSONL entry")?;
    file.write_all(b"\n")
        .context("Failed to append session JSONL newline")?;
    file.flush()
        .context("Failed to flush appended session JSONL entry")?;
    Ok(())
}

/// Read the active-session pointer file, if it exists.
fn read_pointer_file(pointer_path: &Path) -> anyhow::Result<Option<String>> {
    if !pointer_path.is_file() {
        return Ok(None);
    }
    reject_symlink_path(pointer_path, "Current-session pointer file")?;

    let raw = fs::read_to_string(pointer_path).with_context(|| {
        format!(
            "Failed to read current-session pointer file: {}",
            pointer_path.display()
        )
    })?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    Ok(Some(trimmed.replace('\\', "/")))
}

/// Persist the active-session pointer file.
fn write_pointer_file(pointer_path: &Path, relative_path: &str) -> anyhow::Result<()> {
    reject_symlink_path(pointer_path, "Current-session pointer file")?;
    fs::write(pointer_path, format!("{relative_path}\n")).with_context(|| {
        format!(
            "Failed to write current-session pointer file: {}",
            pointer_path.display()
        )
    })
}

/// Locate the lexicographically newest `*.jsonl` session file in the sessions
/// directory.
fn latest_session_file(
    workspace_root: &Path,
    sessions_dir: &Path,
) -> anyhow::Result<Option<PathBuf>> {
    let mut candidates = fs::read_dir(sessions_dir)
        .with_context(|| {
            format!(
                "Failed to read session directory: {}",
                sessions_dir.display()
            )
        })?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let relative_path = workspace_relative_path(workspace_root, &path).ok()?;
            resolve_session_segment_path(workspace_root, sessions_dir, &relative_path).ok()
        })
        .collect::<Vec<_>>();

    candidates.sort();
    Ok(candidates.pop())
}

/// Resolve a workspace-relative path and reject any attempt to escape the
/// workspace root.
fn normalize_workspace_relative_path(relative_path: &str) -> anyhow::Result<PathBuf> {
    let raw = Path::new(relative_path);
    if raw.is_absolute() {
        anyhow::bail!("Session pointer path must be workspace-relative: {relative_path}");
    }

    let mut normalized = PathBuf::new();
    for component in raw.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                anyhow::bail!("Session pointer path escapes workspace root: {relative_path}");
            }
            Component::Prefix(_) | Component::RootDir => {
                anyhow::bail!("Unsupported session pointer path component: {relative_path}");
            }
        }
    }

    Ok(normalized)
}

/// Resolve one session-segment path from the pointer file and enforce the
/// `workspace/sessions/*.jsonl` boundary.
fn resolve_session_segment_path(
    workspace_root: &Path,
    sessions_dir: &Path,
    relative_path: &str,
) -> anyhow::Result<PathBuf> {
    let normalized = normalize_workspace_relative_path(relative_path)?;
    let Some(Component::Normal(first_component)) = normalized.components().next() else {
        anyhow::bail!("Session pointer path must target a file under `sessions/`: {relative_path}");
    };
    if first_component != std::ffi::OsStr::new(SESSIONS_DIR_NAME) {
        anyhow::bail!("Session pointer path must stay under `sessions/`: {relative_path}");
    }
    if normalized.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
        anyhow::bail!("Session pointer path must target a `.jsonl` file: {relative_path}");
    }

    let absolute_path = workspace_root.join(&normalized);
    if !absolute_path.exists() {
        return Ok(absolute_path);
    }

    reject_symlink_path(&absolute_path, "Session segment path")?;
    let canonical = fs::canonicalize(&absolute_path).with_context(|| {
        format!(
            "Failed to canonicalize session segment path: {}",
            absolute_path.display()
        )
    })?;
    ensure_path_is_under_directory(&canonical, sessions_dir).with_context(|| {
        format!(
            "Session segment path resolved outside `sessions/`: {}",
            canonical.display()
        )
    })?;

    Ok(canonical)
}

/// Acquire one exclusive lock for the whole persistent session directory.
///
/// If the lock is held by a process that is no longer running (stale lock),
/// the lock file is removed and acquisition is retried once.
fn acquire_session_store_lock(sessions_dir: &Path) -> anyhow::Result<File> {
    let lock_path = sessions_dir.join(SESSION_STORE_LOCK_FILE_NAME);

    // First attempt: try to acquire the lock normally.
    match try_lock_file(&lock_path) {
        Ok(file) => return Ok(file),
        Err(_) => {}
    }

    // Lock held — check if it's a stale lock from a dead process.
    if is_stale_lock(&lock_path) {
        eprintln!(
            "Session lock is stale (owner process dead). Removing lock file: {}",
            lock_path.display()
        );
        let _ = fs::remove_file(&lock_path);

        // Retry after removing stale lock.
        return try_lock_file(&lock_path).with_context(|| {
            format!(
                "Failed to acquire session storage lock for {} even after removing stale lock",
                sessions_dir.display()
            )
        });
    }

    anyhow::bail!(
        "Another SA backend instance is already using session storage under {}",
        sessions_dir.display()
    )
}

/// Try to open and exclusively lock the lock file, writing the current PID on success.
fn try_lock_file(lock_path: &Path) -> anyhow::Result<File> {
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("Failed to open session lock file: {}", lock_path.display()))?;

    let acquired = lock_file.try_lock_exclusive().with_context(|| {
        format!(
            "Failed to acquire session storage lock: {}",
            lock_path.display()
        )
    })?;
    if !acquired {
        anyhow::bail!("Lock not acquired");
    }

    // Write current PID so future instances can detect stale locks.
    {
        let _ = lock_file.set_len(0);
        let mut writer = std::io::BufWriter::new(&lock_file);
        let _ = write!(writer, "{}", std::process::id());
        let _ = writer.flush();
    }

    Ok(lock_file)
}

/// Check if the lock file belongs to a process that is no longer running.
fn is_stale_lock(lock_path: &Path) -> bool {
    let Ok(content) = fs::read_to_string(lock_path) else {
        // Can't read the lock file — treat as stale so we can try removing it.
        return true;
    };
    let pid_str = content.trim();
    let Ok(pid) = pid_str.parse::<u32>() else {
        // PID not parseable — treat as stale.
        return true;
    };

    is_process_dead(pid)
}

/// Check if a process with the given PID is no longer running.
/// Uses OS-specific commands to avoid adding new dependencies.
fn is_process_dead(pid: u32) -> bool {
    #[cfg(target_os = "windows")]
    {
        // `tasklist` returns the PID if it exists, empty otherwise.
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // If the process exists, the output contains the PID number.
                // If not, it says "No tasks are running" or similar.
                !stdout.contains(&pid.to_string())
            })
            .unwrap_or(true) // If we can't check, assume dead to allow recovery.
    }
    #[cfg(not(target_os = "windows"))]
    {
        // On Unix, use `kill -0` to check process existence.
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|output| !output.status.success()) // success means process exists
            .unwrap_or(true)
    }
}

/// Reject symbolic links for security-sensitive session metadata paths.
fn reject_symlink_path(path: &Path, label: &str) -> anyhow::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            anyhow::bail!("{label} must not be a symbolic link: {}", path.display());
        }
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => {
            Err(err).with_context(|| format!("Failed to inspect {label}: {}", path.display()))
        }
    }
}

/// Verify that a canonicalized path still stays under the expected directory.
fn ensure_path_is_under_directory(path: &Path, directory: &Path) -> anyhow::Result<()> {
    path.strip_prefix(directory).map(|_| ()).with_context(|| {
        format!(
            "Path {} is not under {}",
            path.display(),
            directory.display()
        )
    })
}

/// Parse one raw JSONL line into a session entry.
fn parse_session_line(
    line_bytes: &[u8],
    path: &Path,
    line_number: usize,
) -> anyhow::Result<Option<SessionEntry>> {
    let line_bytes = line_bytes.strip_suffix(b"\r").unwrap_or(line_bytes);
    let line = std::str::from_utf8(line_bytes).with_context(|| {
        format!(
            "Failed to decode UTF-8 session entry at {} line {}",
            path.display(),
            line_number
        )
    })?;
    if line.trim().is_empty() {
        return Ok(None);
    }

    let entry = serde_json::from_str::<SessionEntry>(line).with_context(|| {
        format!(
            "Failed to parse JSONL entry at {} line {}",
            path.display(),
            line_number
        )
    })?;
    Ok(Some(entry))
}

/// Apply one parsed entry while enforcing the session-file ordering rules.
fn apply_session_entry(
    entry: SessionEntry,
    path: &Path,
    line_number: usize,
    meta: &mut Option<SessionMetaEntry>,
    compaction_summary: &mut Option<String>,
    messages: &mut Vec<ChatMessage>,
) -> anyhow::Result<()> {
    match entry {
        SessionEntry::SessionMeta(item) => {
            if meta.is_some() {
                anyhow::bail!(
                    "Session file {} contains more than one `session_meta` entry (line {})",
                    path.display(),
                    line_number
                );
            }
            *meta = Some(item);
        }
        SessionEntry::CompactionCheckpoint(item) => {
            if meta.is_none() {
                anyhow::bail!(
                    "Session file {} contains `compaction_checkpoint` before `session_meta` (line {})",
                    path.display(),
                    line_number
                );
            }
            *compaction_summary = Some(item.summary);
        }
        SessionEntry::Message(item) => {
            if meta.is_none() {
                anyhow::bail!(
                    "Session file {} contains `message` before `session_meta` (line {})",
                    path.display(),
                    line_number
                );
            }
            messages.push(item.message.into());
        }
    }

    Ok(())
}

/// Convert an absolute path under the workspace root into a forward-slash
/// relative display path.
fn workspace_relative_path(workspace_root: &Path, absolute_path: &Path) -> anyhow::Result<String> {
    let relative = absolute_path
        .strip_prefix(workspace_root)
        .with_context(|| {
            format!(
                "Path {} is not under workspace root {}",
                absolute_path.display(),
                workspace_root.display()
            )
        })?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ToolFunctionCall;
    use tempfile::TempDir;

    /// Create one isolated temporary workspace and session store pair.
    fn create_store() -> (TempDir, SessionStore) {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let store = SessionStore::new(workspace.path().to_path_buf())
            .expect("session store should build for temp workspace");
        (workspace, store)
    }

    /// New stores should eagerly create the first empty segment.
    #[test]
    fn session_store_creates_initial_segment() {
        let (_workspace, store) = create_store();
        let snapshot = store.load_snapshot().expect("initial snapshot should load");

        assert!(snapshot.messages.is_empty());
        assert!(snapshot.compaction_summary.is_none());
        assert!(
            snapshot
                .descriptor
                .current_session_path
                .starts_with("sessions/")
        );
        assert!(snapshot.descriptor.previous_session_path.is_none());
    }

    /// Appended messages must survive a fresh store reload.
    #[test]
    fn session_store_persists_messages_across_reload() {
        let (workspace, store) = create_store();
        store
            .append_message(&ChatMessage::text("user", "hello"))
            .expect("user message should append");
        let mut assistant = ChatMessage::text("assistant", "done");
        assistant.request_usage = Some(ChatUsage {
            total_tokens: Some(42),
            ..ChatUsage::default()
        });
        store
            .append_message(&assistant)
            .expect("assistant message should append");

        drop(store);
        let reloaded = SessionStore::new(workspace.path().to_path_buf())
            .expect("reloaded session store should build");
        let snapshot = reloaded
            .load_snapshot()
            .expect("reloaded snapshot should load");

        assert_eq!(snapshot.messages.len(), 2);
        assert_eq!(snapshot.messages[0].role, "user");
        assert_eq!(snapshot.messages[1].role, "assistant");
        assert_eq!(
            snapshot.messages[1]
                .request_usage
                .as_ref()
                .and_then(|usage| usage.total_tokens),
            Some(42)
        );
    }

    /// Compaction rollover must create a new segment that remembers the
    /// previous file path and checkpoint summary.
    #[test]
    fn session_store_rollover_carries_forward_summary_and_previous_path() {
        let (_workspace, store) = create_store();
        store
            .append_message(&ChatMessage::text("user", "old turn"))
            .expect("old user message should append");
        let previous_path = store.current_session_path();

        let descriptor = store
            .rollover_after_compaction(
                "summary body",
                &[
                    ChatMessage::text("assistant", "kept assistant"),
                    ChatMessage::tool_result("call_1", "kept tool"),
                ],
            )
            .expect("rollover should succeed");

        assert_ne!(descriptor.current_session_path, previous_path);
        assert_eq!(
            descriptor.previous_session_path.as_deref(),
            Some(previous_path.as_str())
        );

        let snapshot = store
            .load_snapshot()
            .expect("snapshot after rollover should load");
        assert_eq!(snapshot.compaction_summary.as_deref(), Some("summary body"));
        assert_eq!(snapshot.messages.len(), 2);
        assert_eq!(
            snapshot.descriptor.previous_session_path.as_deref(),
            Some(previous_path.as_str())
        );
        assert!(
            snapshot
                .descriptor
                .compaction_prompt_block()
                .contains(previous_path.as_str())
        );
    }

    /// Partially persisted tool-calling turns must be truncated on resume so
    /// the history stays replayable.
    #[test]
    fn session_store_truncates_incomplete_tool_turn_suffix() {
        let (_workspace, store) = create_store();
        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: Some("need tool".to_string()),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                kind: "function".to_string(),
                function: ToolFunctionCall {
                    name: "Read".to_string(),
                    arguments: "{\"path\":\"a.txt\"}".to_string(),
                },
            }]),
            tool_call_id: None,
            request_usage: None,
            responses_input_items: None,
        };
        store
            .append_message(&ChatMessage::text("user", "start"))
            .expect("user message should append");
        store
            .append_message(&assistant)
            .expect("assistant tool-call turn should append");

        let snapshot = store
            .load_snapshot()
            .expect("snapshot should load even with incomplete tail");
        assert_eq!(snapshot.truncated_incomplete_messages, 1);
        assert_eq!(snapshot.messages.len(), 1);
        assert_eq!(snapshot.messages[0].role, "user");
    }

    /// A truncated final JSONL fragment should be discarded and repaired in
    /// place so the durable session remains readable after a crash.
    #[test]
    fn session_store_repairs_truncated_partial_jsonl_tail() {
        let (workspace, store) = create_store();
        store
            .append_message(&ChatMessage::text("user", "hello"))
            .expect("user message should append");

        let current_path = workspace.path().join(store.current_session_path());
        let mut file = OpenOptions::new()
            .append(true)
            .open(&current_path)
            .expect("session file should open for append");
        file.write_all(br#"{"type":"message","created_at":"#)
            .expect("partial JSONL fragment should append");
        file.flush().expect("partial JSONL fragment should flush");

        let snapshot = store
            .load_snapshot()
            .expect("snapshot should recover from truncated final line");
        assert_eq!(snapshot.messages.len(), 1);
        assert_eq!(snapshot.messages[0].content.as_deref(), Some("hello"));

        let repaired = fs::read_to_string(&current_path).expect("repaired session should read");
        assert_eq!(repaired.lines().count(), 2);
        assert!(repaired.ends_with('\n'));
        let reloaded = store
            .load_snapshot()
            .expect("repaired session should stay readable on later loads");
        assert_eq!(reloaded.messages.len(), 1);
    }

    /// Pointer files must not redirect recovery outside the dedicated
    /// `workspace/sessions/` directory.
    #[test]
    fn session_store_rejects_pointer_outside_sessions_dir() {
        let (workspace, store) = create_store();
        let pointer_path = workspace
            .path()
            .join(SESSIONS_DIR_NAME)
            .join(CURRENT_SESSION_POINTER_FILE_NAME);
        fs::write(&pointer_path, "notes.jsonl\n").expect("pointer file should be overwritten");
        drop(store);

        let err = SessionStore::new(workspace.path().to_path_buf())
            .expect_err("pointer outside sessions dir must be rejected");
        let message = format!("{err:#}");
        assert!(message.contains("must stay under `sessions/`"));
    }

    /// Only one live backend owner may hold the persistent session directory at
    /// a time.
    #[test]
    fn session_store_rejects_second_live_owner_for_same_workspace() {
        let (workspace, _store) = create_store();

        let err = SessionStore::new(workspace.path().to_path_buf())
            .expect_err("second live owner should be rejected");
        let message = format!("{err:#}");
        assert!(message.contains("already using session storage"));
    }

    /// Custom agent session roots should stay isolated under nested session
    /// directories such as `sessions/agents/<agent_id>/`.
    #[test]
    fn session_store_supports_nested_agent_session_roots() {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let store = SessionStore::new_in_relative_dir(
            workspace.path().to_path_buf(),
            Path::new("sessions/agents/agent-1"),
        )
        .expect("nested agent session store should build");

        store
            .append_message(&ChatMessage::text("user", "hello child"))
            .expect("message should append");
        let snapshot = store.load_snapshot().expect("snapshot should load");

        assert!(
            snapshot
                .descriptor
                .current_session_path
                .starts_with("sessions/agents/agent-1/")
        );
        assert_eq!(snapshot.messages.len(), 1);
    }

    /// Custom session roots must still remain workspace-relative.
    #[test]
    fn session_store_rejects_invalid_nested_session_root() {
        let workspace = TempDir::new().expect("temp workspace should be created");
        let err = SessionStore::new_in_relative_dir(
            workspace.path().to_path_buf(),
            Path::new("../escape"),
        )
        .expect_err("escaping custom session roots must be rejected");
        assert!(format!("{err:#}").contains("must not contain `..`"));
    }
}
