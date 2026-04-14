//! Durable append-only interaction history for user-visible IO.
//!
//! This store is intentionally separate from:
//! - model session history
//! - runtime state
//!
//! Why a dedicated interaction log?
//! - Users explicitly asked for a chronological audit log of all direct
//!   frontend interactions.
//! - Agent prompt/session history is optimized for model replay, not for human
//!   interaction forensics.
//! - `Show` snapshots may contain large payloads, so retrieval must support
//!   progressive disclosure instead of always dumping the full entry back into
//!   context.

use crate::ws_protocol::{QuestionMode, QuestionOption, UserVisibleFile, UserVisibleFileEncoding};
use anyhow::Context as _;
use chrono::{DateTime, FixedOffset, Local};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Directory containing the durable interaction log.
pub const INTERACTIONS_DIR_NAME: &str = "interactions";

/// Default interaction log file name.
pub const INTERACTIONS_LOG_FILE_NAME: &str = "history.jsonl";

/// Default preview size returned by summary-mode retrieval.
const DEFAULT_PREVIEW_CHARS: usize = 1024;

/// Upper bound for one full disclosure from the tool.
const MAX_FULL_DISCLOSURE_CHARS: usize = 64 * 1024;

/// Default slice length for progressive reads.
const DEFAULT_SLICE_CHARS: usize = 4096;

/// Durable interaction entry stored as one JSONL line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InteractionEntry {
    /// Stable unique id for this interaction line.
    pub id: Uuid,
    /// Local wall-clock timestamp with timezone offset and nanosecond
    /// precision when the platform provides it.
    pub local_timestamp: DateTime<FixedOffset>,
    /// Related top-level work id when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_id: Option<Uuid>,
    /// Agent that produced this interaction when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<Uuid>,
    /// Concrete interaction payload.
    #[serde(flatten)]
    pub payload: InteractionPayload,
}

/// Concrete interaction payload variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InteractionPayload {
    /// Free-form user message submitted to the current input owner.
    UserMessage {
        /// Client-provided submit id when available.
        submit_id: Uuid,
        /// Agent that owned free-form input when the message was accepted.
        input_owner_agent_id: Uuid,
        /// Raw user message content.
        message: String,
    },
    /// One `Send` payload delivered to the frontend.
    Send {
        /// Outbound user-visible text.
        message: String,
    },
    /// One `Ask` question shown to the frontend.
    Ask {
        /// Stable question id.
        question_id: Uuid,
        /// Prompt shown to the frontend.
        prompt: String,
        /// Expected answer shape.
        mode: QuestionMode,
        /// Selectable options when applicable.
        #[serde(default)]
        options: Vec<QuestionOption>,
        /// Whether extra free text is allowed.
        #[serde(default)]
        allow_free_text: bool,
    },
    /// One answer the frontend returned for a previous `Ask`.
    AskAnswer {
        /// Stable question id being answered.
        question_id: Uuid,
        /// Selected option ids.
        #[serde(default)]
        selected_option_ids: Vec<String>,
        /// Selected option labels for easier grep/inspection.
        #[serde(default)]
        selected_labels: Vec<String>,
        /// Optional free-text answer.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        free_text: Option<String>,
    },
    /// One `Show` snapshot delivered to the frontend.
    Show {
        /// Stable show id.
        show_id: Uuid,
        /// Original path string.
        path: String,
        /// File name extracted from `path`.
        file_name: String,
        /// Optional user-facing title.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Non-user-facing concise description of the shown content.
        prompt: String,
        /// Best-effort media type.
        media_type: String,
        /// Snapshot encoding.
        encoding: UserVisibleFileEncoding,
        /// Snapshot byte size before transport encoding.
        bytes: usize,
        /// Snapshot content itself.
        content: String,
    },
}

/// Disclosure mode for retrieving one interaction entry by id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionDisclosureMode {
    /// Return metadata plus a short preview for large textual payloads.
    Summary,
    /// Return the full serialized entry when it is small enough.
    Full,
    /// Return only a bounded slice of the large textual field.
    Slice,
}

/// Read options for `serialize_entry_by_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteractionReadOptions {
    /// Disclosure mode.
    pub mode: InteractionDisclosureMode,
    /// Character offset used by slice mode.
    pub offset: usize,
    /// Character count used by slice mode.
    pub limit: usize,
}

impl Default for InteractionReadOptions {
    fn default() -> Self {
        Self {
            mode: InteractionDisclosureMode::Summary,
            offset: 0,
            limit: DEFAULT_SLICE_CHARS,
        }
    }
}

/// Filesystem-backed append-only interaction log.
#[derive(Debug, Clone)]
pub struct InteractionStore {
    workspace_root: PathBuf,
    interactions_dir: PathBuf,
    log_path: PathBuf,
    append_lock: Arc<Mutex<()>>,
}

impl InteractionStore {
    /// Create a new workspace-bound interaction store.
    pub fn new(workspace_root: PathBuf) -> anyhow::Result<Self> {
        let workspace_root = fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root for interaction store: {}",
                workspace_root.display()
            )
        })?;
        let interactions_dir = workspace_root.join(INTERACTIONS_DIR_NAME);
        fs::create_dir_all(&interactions_dir).with_context(|| {
            format!(
                "Failed to create interaction history directory: {}",
                interactions_dir.display()
            )
        })?;
        Ok(Self {
            log_path: interactions_dir.join(INTERACTIONS_LOG_FILE_NAME),
            workspace_root,
            interactions_dir,
            append_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Return the canonical workspace root.
    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Return the interaction history directory.
    pub fn interactions_dir(&self) -> &Path {
        &self.interactions_dir
    }

    /// Return the append-only JSONL file path.
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Append one fully-formed interaction entry.
    pub fn append(&self, entry: &InteractionEntry) -> anyhow::Result<()> {
        let _guard = self
            .append_lock
            .lock()
            .expect("interaction append lock poisoned");
        if let Some(parent) = self.log_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create interaction dir: {}", parent.display())
            })?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .with_context(|| {
                format!(
                    "Failed to open interaction log: {}",
                    self.log_path.display()
                )
            })?;
        let mut line =
            serde_json::to_string(entry).context("Failed to serialize interaction entry")?;
        line.push('\n');
        file.write_all(line.as_bytes()).with_context(|| {
            format!(
                "Failed to append interaction entry: {}",
                self.log_path.display()
            )
        })?;
        file.flush().with_context(|| {
            format!(
                "Failed to flush interaction log: {}",
                self.log_path.display()
            )
        })?;
        file.sync_all().with_context(|| {
            format!(
                "Failed to sync interaction log: {}",
                self.log_path.display()
            )
        })
    }

    /// Build and append one new interaction entry.
    pub fn append_new(
        &self,
        work_id: Option<Uuid>,
        agent_id: Option<Uuid>,
        payload: InteractionPayload,
    ) -> anyhow::Result<InteractionEntry> {
        let entry = InteractionEntry {
            id: Uuid::new_v4(),
            local_timestamp: Local::now().fixed_offset(),
            work_id,
            agent_id,
            payload,
        };
        self.append(&entry)?;
        Ok(entry)
    }

    /// Load the interaction entry with the given id.
    pub fn get_by_id(&self, id: Uuid) -> anyhow::Result<Option<InteractionEntry>> {
        if !self.log_path.is_file() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&self.log_path).with_context(|| {
            format!(
                "Failed to read interaction log: {}",
                self.log_path.display()
            )
        })?;
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            let entry = serde_json::from_str::<InteractionEntry>(line)
                .context("Invalid interaction history JSONL entry")?;
            if entry.id == id {
                return Ok(Some(entry));
            }
        }
        Ok(None)
    }

    /// Return one serialized interaction entry with progressive disclosure.
    pub fn serialize_entry_by_id(
        &self,
        id: Uuid,
        options: InteractionReadOptions,
    ) -> anyhow::Result<Option<String>> {
        let Some(entry) = self.get_by_id(id)? else {
            return Ok(None);
        };
        let value = serialize_entry_with_options(&entry, options)?;
        Ok(Some(serde_json::to_string(&value)?))
    }
}

/// Serialize one entry according to the requested disclosure mode.
pub fn serialize_entry_with_options(
    entry: &InteractionEntry,
    options: InteractionReadOptions,
) -> anyhow::Result<Value> {
    match options.mode {
        InteractionDisclosureMode::Summary => serialize_summary(entry),
        InteractionDisclosureMode::Full => serialize_full(entry),
        InteractionDisclosureMode::Slice => serialize_slice(entry, options.offset, options.limit),
    }
}

/// Serialize one entry in summary mode.
fn serialize_summary(entry: &InteractionEntry) -> anyhow::Result<Value> {
    match &entry.payload {
        InteractionPayload::UserMessage {
            submit_id,
            input_owner_agent_id,
            message,
        } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "user_message",
            "submit_id": submit_id,
            "input_owner_agent_id": input_owner_agent_id,
            "message_chars": message.chars().count(),
            "message_preview": preview_chars(message, DEFAULT_PREVIEW_CHARS),
        })),
        InteractionPayload::Send { message } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "send",
            "message_chars": message.chars().count(),
            "message_preview": preview_chars(message, DEFAULT_PREVIEW_CHARS),
        })),
        InteractionPayload::Ask {
            question_id,
            prompt,
            mode,
            options,
            allow_free_text,
        } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "ask",
            "question_id": question_id,
            "prompt": prompt,
            "mode": mode,
            "options": options,
            "allow_free_text": allow_free_text,
        })),
        InteractionPayload::AskAnswer {
            question_id,
            selected_option_ids,
            selected_labels,
            free_text,
        } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "ask_answer",
            "question_id": question_id,
            "selected_option_ids": selected_option_ids,
            "selected_labels": selected_labels,
            "free_text": free_text,
        })),
        InteractionPayload::Show {
            show_id,
            path,
            file_name,
            title,
            prompt,
            media_type,
            encoding,
            bytes,
            content,
        } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "show",
            "show_id": show_id,
            "path": path,
            "file_name": file_name,
            "title": title,
            "prompt": prompt,
            "media_type": media_type,
            "encoding": encoding,
            "bytes": bytes,
            "content_chars": content.chars().count(),
            "content_preview": preview_chars(content, DEFAULT_PREVIEW_CHARS),
        })),
    }
}

/// Serialize one entry in full mode.
fn serialize_full(entry: &InteractionEntry) -> anyhow::Result<Value> {
    let full = serde_json::to_value(entry)?;
    let content_len = content_char_len(&entry.payload);
    if content_len > MAX_FULL_DISCLOSURE_CHARS {
        anyhow::bail!(
            "Interaction entry content is too large for full disclosure ({} chars). Use summary or slice mode instead.",
            content_len
        );
    }
    Ok(full)
}

/// Serialize one entry in slice mode.
fn serialize_slice(entry: &InteractionEntry, offset: usize, limit: usize) -> anyhow::Result<Value> {
    let limit = if limit == 0 {
        DEFAULT_SLICE_CHARS
    } else {
        limit
    };
    match &entry.payload {
        InteractionPayload::UserMessage {
            submit_id,
            input_owner_agent_id,
            message,
        } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "user_message",
            "submit_id": submit_id,
            "input_owner_agent_id": input_owner_agent_id,
            "message_chars": message.chars().count(),
            "slice_offset": offset,
            "slice_limit": limit,
            "message_slice": slice_chars(message, offset, limit),
        })),
        InteractionPayload::Send { message } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "send",
            "message_chars": message.chars().count(),
            "slice_offset": offset,
            "slice_limit": limit,
            "message_slice": slice_chars(message, offset, limit),
        })),
        InteractionPayload::Show {
            show_id,
            path,
            file_name,
            title,
            prompt,
            media_type,
            encoding,
            bytes,
            content,
        } => Ok(serde_json::json!({
            "id": entry.id,
            "local_timestamp": entry.local_timestamp,
            "work_id": entry.work_id,
            "agent_id": entry.agent_id,
            "kind": "show",
            "show_id": show_id,
            "path": path,
            "file_name": file_name,
            "title": title,
            "prompt": prompt,
            "media_type": media_type,
            "encoding": encoding,
            "bytes": bytes,
            "content_chars": content.chars().count(),
            "slice_offset": offset,
            "slice_limit": limit,
            "content_slice": slice_chars(content, offset, limit),
        })),
        _ => serialize_summary(entry),
    }
}

/// Return the size of the potentially large textual field for one payload.
fn content_char_len(payload: &InteractionPayload) -> usize {
    match payload {
        InteractionPayload::UserMessage { message, .. } => message.chars().count(),
        InteractionPayload::Send { message } => message.chars().count(),
        InteractionPayload::Show { content, .. } => content.chars().count(),
        InteractionPayload::Ask { .. } | InteractionPayload::AskAnswer { .. } => 0,
    }
}

/// Return a preview of at most `limit` characters.
fn preview_chars(text: &str, limit: usize) -> String {
    let mut preview = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        preview.push_str("...");
    }
    preview
}

/// Return a bounded character slice from the given text.
fn slice_chars(text: &str, offset: usize, limit: usize) -> String {
    text.chars().skip(offset).take(limit).collect()
}

/// Convert one `Show` payload from the ws protocol shape into the durable
/// interaction-history shape.
pub fn show_payload_from_visible_file(file: &UserVisibleFile) -> InteractionPayload {
    InteractionPayload::Show {
        show_id: file.show_id,
        path: file.path.clone(),
        file_name: Path::new(&file.path)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| file.path.clone()),
        title: file.title.clone(),
        prompt: file.prompt.clone(),
        media_type: file.media_type.clone(),
        encoding: file.encoding.clone(),
        bytes: file.bytes,
        content: file.content.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_store() -> (TempDir, InteractionStore) {
        let workspace = TempDir::new().expect("temp workspace should build");
        let store = InteractionStore::new(workspace.path().to_path_buf())
            .expect("interaction store should build");
        (workspace, store)
    }

    #[test]
    fn append_and_get_by_id_round_trip() {
        let (_workspace, store) = create_store();
        let entry = store
            .append_new(
                Some(Uuid::new_v4()),
                Some(Uuid::new_v4()),
                InteractionPayload::Send {
                    message: "hello".to_string(),
                },
            )
            .expect("entry should append");

        let loaded = store
            .get_by_id(entry.id)
            .expect("entry should load")
            .expect("entry should exist");
        assert_eq!(loaded, entry);
    }

    #[test]
    fn summary_mode_previews_show_content() {
        let (_workspace, store) = create_store();
        let entry = store
            .append_new(
                Some(Uuid::new_v4()),
                Some(Uuid::new_v4()),
                InteractionPayload::Show {
                    show_id: Uuid::new_v4(),
                    path: "notes/video.bin".to_string(),
                    file_name: "video.bin".to_string(),
                    title: Some("Video".to_string()),
                    prompt: "用于展示调试视频".to_string(),
                    media_type: "application/octet-stream".to_string(),
                    encoding: UserVisibleFileEncoding::Base64,
                    bytes: 1234,
                    content: "abcdefghijk".repeat(400),
                },
            )
            .expect("entry should append");

        let raw = store
            .serialize_entry_by_id(
                entry.id,
                InteractionReadOptions {
                    mode: InteractionDisclosureMode::Summary,
                    ..Default::default()
                },
            )
            .expect("entry should serialize")
            .expect("entry should exist");
        let value: Value = serde_json::from_str(&raw).expect("summary should be valid JSON");
        assert_eq!(value["kind"], "show");
        assert_eq!(value["prompt"], "用于展示调试视频");
        assert!(
            value["content_preview"]
                .as_str()
                .expect("preview should be string")
                .len()
                < value["content_chars"].as_u64().expect("chars should exist") as usize
        );
    }

    #[test]
    fn full_mode_refuses_large_show_content() {
        let (_workspace, store) = create_store();
        let entry = store
            .append_new(
                None,
                None,
                InteractionPayload::Show {
                    show_id: Uuid::new_v4(),
                    path: "artifact.bin".to_string(),
                    file_name: "artifact.bin".to_string(),
                    title: None,
                    prompt: "大型二进制快照".to_string(),
                    media_type: "application/octet-stream".to_string(),
                    encoding: UserVisibleFileEncoding::Base64,
                    bytes: 1,
                    content: "x".repeat(MAX_FULL_DISCLOSURE_CHARS + 1),
                },
            )
            .expect("entry should append");

        let err = store
            .serialize_entry_by_id(
                entry.id,
                InteractionReadOptions {
                    mode: InteractionDisclosureMode::Full,
                    ..Default::default()
                },
            )
            .expect_err("large full disclosure must fail");
        assert!(err.to_string().contains("too large"));
    }

    #[test]
    fn slice_mode_returns_bounded_content_slice() {
        let (_workspace, store) = create_store();
        let entry = store
            .append_new(
                None,
                None,
                InteractionPayload::Send {
                    message: "0123456789abcdef".to_string(),
                },
            )
            .expect("entry should append");

        let raw = store
            .serialize_entry_by_id(
                entry.id,
                InteractionReadOptions {
                    mode: InteractionDisclosureMode::Slice,
                    offset: 4,
                    limit: 6,
                },
            )
            .expect("entry should serialize")
            .expect("entry should exist");
        let value: Value = serde_json::from_str(&raw).expect("slice should be valid JSON");
        assert_eq!(value["message_slice"], "456789");
    }
}
