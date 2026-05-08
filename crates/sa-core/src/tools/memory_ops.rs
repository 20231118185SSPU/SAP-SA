//! Memory-related tool implementations for ToolExecutor.
//!
//! All operations go through `MemoryStore` (SQLite-backed).

use crate::cancel::CancelToken;
use crate::memory_store::{MemoryEntry, MemoryPatch, MemoryStore};
use super::ToolExecutor;
use anyhow::Context as _;
use serde::Deserialize;

/// Memory operations trait — method implementations extracted from tools.rs.
#[async_trait::async_trait]
pub trait MemoryOps {
    async fn memory_search(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
    async fn memory_get(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
    async fn forget_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
    async fn memory_set(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
    async fn edit_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
    async fn pin_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
    async fn unpin_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String>;
}

fn require_store(executor: &ToolExecutor) -> anyhow::Result<&MemoryStore> {
    executor
        .memory_store
        .as_ref()
        .map(|arc| arc.as_ref())
        .context("MemoryStore not configured — memory tools unavailable")
}

#[async_trait::async_trait]
impl MemoryOps for ToolExecutor {

    // ── memory_search ──────────────────────────────────────────────────
    async fn memory_search(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            max_results: Option<usize>,
            scope: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MemorySearch")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemorySearch cancelled");
        }

        let store = require_store(self)?;
        let limit = args.max_results.unwrap_or(10).min(50);
        let results = store.search_memories(&args.query, args.scope.as_deref(), limit)?;

        Ok(serde_json::json!({
            "query": args.query,
            "count": results.len(),
            "results": results,
        })
        .to_string())
    }

    // ── memory_get ─────────────────────────────────────────────────────
    async fn memory_get(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            id: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for MemoryGet")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemoryGet cancelled");
        }

        let store = require_store(self)?;
        match store.get_memory(&args.id)? {
            Some(entry) => {
                store.touch_memory(&args.id)?;
                Ok(serde_json::to_string(&entry)?)
            }
            None => Ok(serde_json::json!({
                "error": "not_found",
                "id": args.id,
            })
            .to_string()),
        }
    }

    // ── forget_memory ──────────────────────────────────────────────────
    async fn forget_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            query: String,
            max_results: Option<usize>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ForgetMemory")?;
        if cancel.is_cancelled() {
            anyhow::bail!("ForgetMemory cancelled");
        }

        let store = require_store(self)?;
        let limit = args.max_results.unwrap_or(10).min(50);
        let results = store.search_memories(&args.query, None, limit)?;

        let mut deleted = 0usize;
        for entry in &results {
            if store.delete_memory(&entry.id)? {
                deleted += 1;
            }
        }

        Ok(serde_json::json!({
            "deleted": deleted,
            "total_matches": results.len(),
        })
        .to_string())
    }

    // ── memory_set ─────────────────────────────────────────────────────
    async fn memory_set(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            content: String,
            title: Option<String>,
            tags: Option<Vec<String>>,
            importance: Option<f64>,
            scope: Option<String>,
            force: Option<bool>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MemorySet")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemorySet cancelled");
        }

        let store = require_store(self)?;

        // Dedup check: search for similar existing entries
        if !args.force.unwrap_or(false) {
            let similar = store.search_memories(&args.content[..args.content.len().min(200)], None, 5)?;
            for existing in &similar {
                let sim = crate::memory::compute_similarity(&existing.content, &args.content);
                if sim > 0.7 {
                    return Ok(serde_json::json!({
                        "action": "skipped",
                        "reason": "duplicate",
                        "similarity": sim,
                        "existing_id": existing.id,
                        "existing_snippet": existing.content.chars().take(120).collect::<String>(),
                    })
                    .to_string());
                }
            }
        }

        let now = chrono::Utc::now().timestamp();
        let title = args.title.unwrap_or_else(|| {
            args.content.chars().take(50).collect()
        });
        let importance = args.importance.unwrap_or_else(|| {
            crate::memory::score_content_importance(&args.content)
        });
        let (valence, _arousal) = crate::memory::detect_emotion(&args.content);

        let entry = MemoryEntry {
            id: MemoryStore::memory_id(&title, &args.content),
            title,
            content: args.content.clone(),
            tags: args.tags.unwrap_or_default(),
            scope: args.scope.unwrap_or_else(|| "user".into()),
            importance,
            emotion_valence: valence,
            source: "agent".into(),
            access_count: 0,
            created_at: now,
            updated_at: now,
            accessed_at: now,
        };

        let id = entry.id.clone();
        store.insert_memory(&entry)?;

        Ok(serde_json::json!({
            "action": "written",
            "id": id,
            "content_snippet": args.content.chars().take(120).collect::<String>(),
        })
        .to_string())
    }

    // ── edit_memory ────────────────────────────────────────────────────
    async fn edit_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            id: String,
            content: Option<String>,
            title: Option<String>,
            tags: Option<Vec<String>>,
            importance: Option<f64>,
            scope: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for EditMemory")?;
        if cancel.is_cancelled() {
            anyhow::bail!("EditMemory cancelled");
        }

        let store = require_store(self)?;
        let patch = MemoryPatch {
            title: args.title,
            content: args.content,
            tags: args.tags,
            scope: args.scope,
            importance: args.importance,
            emotion_valence: None,
        };

        let updated = store.update_memory(&args.id, &patch)?;
        Ok(serde_json::json!({
            "action": if updated { "edited" } else { "not_found" },
            "id": args.id,
        })
        .to_string())
    }

    // ── pin_memory ─────────────────────────────────────────────────────
    async fn pin_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            key: String,
            content: String,
            label: Option<String>,
            note: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for PinMemory")?;
        if cancel.is_cancelled() {
            anyhow::bail!("PinMemory cancelled");
        }

        let store = require_store(self)?;
        let label = args.label.as_deref().unwrap_or(&args.key);
        store.pin(&args.key, label, &args.content, args.note.as_deref())?;

        Ok(serde_json::json!({
            "action": "pinned",
            "key": args.key,
        })
        .to_string())
    }

    // ── unpin_memory ───────────────────────────────────────────────────
    async fn unpin_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            key: String,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for UnpinMemory")?;
        if cancel.is_cancelled() {
            anyhow::bail!("UnpinMemory cancelled");
        }

        let store = require_store(self)?;
        let removed = store.unpin(&args.key)?;

        Ok(serde_json::json!({
            "action": if removed { "unpinned" } else { "not_found" },
            "key": args.key,
        })
        .to_string())
    }
}
