//! Memory-related tool implementations for ToolExecutor.

use crate::cancel::CancelToken;
use crate::cold_store;
use crate::memory::{
    self, MemoryMetadata, extract_entries_from_daily,
    read_markdown_memory, search_markdown_memory, write_daily_memory,
};
use crate::working_memory::{PinnedSlot, WorkingMemory};
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

#[async_trait::async_trait]
impl MemoryOps for ToolExecutor {

    // ── memory_search ──────────────────────────────────────────────────
    async fn memory_search(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
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
            None,
            None,
        )
        .await?;

        Ok(serde_json::json!({
            "query": args.query,
            "results": results,
            "engine": "markdown_lexical_v1",
        })
        .to_string())
    }

    // ── memory_get ─────────────────────────────────────────────────────
    async fn memory_get(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
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

        // Search for matching entries
        let results = search_markdown_memory(
            &self.ctx.workspace_root,
            &args.query,
            args.max_results.or(Some(10)),
            None,
            None,
            None,
        )
        .await?;

        let mut deleted = 0usize;
        let mut errors: Vec<String> = Vec::new();

        for result in &results {
            // Read the file and find the entry index by matching snippet
            let file_path = self.ctx.workspace_root.join(&result.path);
            let content = match std::fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(e) => {
                    errors.push(format!("{}: read error: {}", result.path, e));
                    continue;
                }
            };
            let entries = extract_entries_from_daily(&content);

            // Find entry index whose body overlaps with the search result lines
            // (approximate: match by snippet content)
            let mut entry_idx = None;
            for (idx, (body, _meta)) in entries.iter().enumerate() {
                if body.contains(&result.snippet[..std::cmp::min(50, result.snippet.len())])
                    || result.snippet.contains(&body[..std::cmp::min(50, body.len())])
                {
                    entry_idx = Some(idx);
                    break;
                }
            }

            let Some(idx) = entry_idx else {
                errors.push(format!(
                    "{}: could not match snippet to entry",
                    result.path
                ));
                continue;
            };

            match cold_store::soft_delete_entry(
                &self.ctx.workspace_root,
                &result.path,
                idx,
                "deleted via ForgetMemory tool",
            ) {
                Ok(sr) => {
                    if sr.success {
                        deleted += 1;
                    } else {
                        errors.push(format!("{}: {}", result.path, sr.reason));
                    }
                }
                Err(e) => {
                    errors.push(format!("{}: {}", result.path, e));
                }
            }
        }

        Ok(serde_json::json!({
            "deleted": deleted,
            "total_matches": results.len(),
            "errors": errors,
        })
        .to_string())
    }

    // ── memory_set ─────────────────────────────────────────────────────
    async fn memory_set(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            content: String,
            date: Option<String>,
            tags: Option<Vec<String>>,
            importance: Option<f64>,
            force: Option<bool>,
            title: Option<String>,
            summary: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for MemorySet")?;
        if cancel.is_cancelled() {
            anyhow::bail!("MemorySet cancelled");
        }

        let date = args
            .date
            .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%d").to_string());
        let file_path = self
            .ctx
            .workspace_root
            .join(format!("memory/{}.md", date));

        // Dedup check: if file exists and force is not set, check for similar entries
        if file_path.exists() && !args.force.unwrap_or(false) {
            let existing = std::fs::read_to_string(&file_path).unwrap_or_default();
            let entries = extract_entries_from_daily(&existing);
            let new_tokens: Vec<&str> = args.content.split_whitespace().collect();
            for (body, _meta) in &entries {
                let existing_tokens: Vec<&str> = body.split_whitespace().collect();
                let intersection = new_tokens
                    .iter()
                    .filter(|t| existing_tokens.contains(t))
                    .count();
                let union = new_tokens.len() + existing_tokens.len() - intersection;
                if union == 0 {
                    continue;
                }
                let jaccard = intersection as f64 / union as f64;
                if jaccard > 0.7 {
                    return Ok(serde_json::json!({
                        "action": "skipped",
                        "reason": "duplicate",
                        "similarity": jaccard,
                        "existing_snippet": body.chars().take(120).collect::<String>(),
                        "file": format!("memory/{}.md", date),
                    })
                    .to_string());
                }
            }
        }

        // Build metadata
        let mut meta = MemoryMetadata::default();
        meta.tags = args.tags;
        meta.importance = args.importance;
        meta.title = args.title;
        meta.summary = args.summary;

        write_daily_memory(
            &self.ctx.workspace_root,
            &date,
            &args.content,
            &meta,
            None,
        )?;

        Ok(serde_json::json!({
            "action": "written",
            "file": format!("memory/{}.md", date),
            "content_snippet": args.content.chars().take(120).collect::<String>(),
        })
        .to_string())
    }

    // ── edit_memory ────────────────────────────────────────────────────
    async fn edit_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            from_line: usize,
            #[allow(dead_code)]
            #[allow(dead_code)]
            #[allow(dead_code)]
            #[allow(dead_code)]
            to_line: usize,
            new_content: String,
            tags: Option<Vec<String>>,
            importance: Option<f64>,
            title: Option<String>,
            summary: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for EditMemory")?;
        if cancel.is_cancelled() {
            anyhow::bail!("EditMemory cancelled");
        }

        let file_path = self.ctx.workspace_root.join(&args.path);
        let existing = std::fs::read_to_string(&file_path)
            .with_context(|| format!("Cannot read {}", args.path))?;

        let entries = extract_entries_from_daily(&existing);

        // Find the entry containing from_line
        let edit_idx = if args.from_line > 0 && args.from_line <= entries.len() {
            args.from_line - 1
        } else {
            0
        };

        // Build updated metadata
        let (_old_body, old_meta) = &entries[edit_idx];
        let mut updated_meta = old_meta.clone();
        if let Some(ref tags) = args.tags {
            updated_meta.tags = Some(tags.clone());
        }
        if let Some(imp) = args.importance {
            updated_meta.importance = Some(imp);
        }
        if let Some(ref title) = args.title {
            updated_meta.title = Some(title.clone());
        }
        if let Some(ref summary) = args.summary {
            updated_meta.summary = Some(summary.clone());
        }

        memory::rebuild_daily_with_edited_entry(
            &file_path,
            &entries,
            edit_idx,
            &updated_meta,
            Some(&args.new_content),
        )?;

        Ok(serde_json::json!({
            "action": "edited",
            "file": args.path,
            "entry_index": edit_idx,
        })
        .to_string())
    }

    // ── pin_memory ─────────────────────────────────────────────────────
    async fn pin_memory(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            key: String,
            label: String,
            content: String,
            note: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for PinMemory")?;
        if cancel.is_cancelled() {
            anyhow::bail!("PinMemory cancelled");
        }

        let mut cache = self.pinned_cache.lock().unwrap();
        let slot = PinnedSlot {
            label: args.label.clone(),
            content: args.content.clone(),
            pinned_at: chrono::Utc::now().timestamp_millis(),
            note: args.note.clone(),
        };
        cache.insert(args.key.clone(), slot);
        WorkingMemory::save_pinned_to_file(&self.ctx.workspace_root, &cache)?;

        Ok(serde_json::json!({
            "action": "pinned",
            "key": args.key,
            "label": args.label,
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

        let mut cache = self.pinned_cache.lock().unwrap();
        let removed = cache.remove(&args.key);
        WorkingMemory::save_pinned_to_file(&self.ctx.workspace_root, &cache)?;

        Ok(serde_json::json!({
            "action": if removed.is_some() { "unpinned" } else { "not_found" },
            "key": args.key,
        })
        .to_string())
    }
}
