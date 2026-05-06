//! File operation tool implementations for ToolExecutor.

use crate::cancel::CancelToken;
use crate::path_guard::{PathOperation, validate_resolved_tool_path, validate_tool_path_input};
use super::{
    ToolExecutor, ToolSession,
};
use crate::tools::{
    MAX_READ_FILE_BYTES, SMART_TRUNCATE_BYTES,
    FileReadVersion, ImageAnalyzeOutput,
};
use anyhow::Context as _;
use serde::Deserialize;
use serde_json;
use std::time::Duration;
use base64::Engine;

impl ToolExecutor {
    /// `Read`: read a UTF-8 text file and mark it as eligible for `Edit`.
    pub(crate) async fn read(
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
            offset: Option<u64>,
            limit: Option<u64>,
            encoding: Option<String>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Read")?;
        validate_tool_path_input(&args.path, PathOperation::Read)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Read)?;

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
                "File is too large to Read via tool ({} bytes, limit is {} bytes): {}. \
                 Consider using offset/limit parameters, or a tool like grep to search specific content.",
                meta.len(),
                MAX_READ_FILE_BYTES,
                path.display()
            );
        }
        let needs_truncation = meta.len() > SMART_TRUNCATE_BYTES;

        // Compute mtime for cache key.
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // ── PDF branch: delegate to MCP OCR ──
        let is_pdf = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"));

        if is_pdf {
            let abs_path = path.display().to_string();
            if let Some(registry) = &self.mcp_registry {
                // Try known MCP OCR tool names in order of preference.
                let ocr_candidates = ["ocr_pdf", "ocr/ocr_pdf", "fileanalyzer/analyze_file"];
                for tool_name in &ocr_candidates {
                    if registry.has_tool(tool_name) {
                        let mcp_args = serde_json::json!({
                            "file_path": abs_path,
                        });
                        match registry.call_tool(tool_name, mcp_args).await {
                            Ok(text) => {
                                // Return OCR result in the same shape as a normal Read.
                                return Ok(serde_json::json!({
                                    "path": abs_path,
                                    "bytes": text.len(),
                                    "lines": [1, text.lines().count() as u64],
                                    "content": text,
                                    "source": "ocr",
                                })
                                .to_string());
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Read PDF: MCP OCR tool `{}` failed: {e}",
                                    tool_name
                                );
                                continue;
                            }
                        }
                    }
                }
            }
            // No MCP OCR available — provide a helpful fallback message.
            anyhow::bail!(
                "PDF files require an MCP OCR server. \
                 No compatible OCR tool found for: {}\n\
                 Hint: configure an MCP server that provides `ocr_pdf` or `fileanalyzer/analyze_file`.",
                path.display()
            );
        }

        // ── Normal text file branch ──

        // Determine encoding; default to UTF-8.
        let encoding_label = args.encoding.as_deref().unwrap_or("utf-8");
        let is_utf8 = encoding_label.eq_ignore_ascii_case("utf-8");

        // Try file content cache (only for UTF-8 full reads — cached content
        // is already decoded; non-UTF-8 and partial reads bypass the cache).
        let is_partial = args.offset.is_some() || args.limit.is_some();
        let content = if is_utf8 && !is_partial {
            if let Some(cached) = self.file_cache.get(&path, mtime).await {
                tracing::debug!(path = %path.display(), "Read cache hit");
                cached
            } else {
                let raw = tokio::fs::read_to_string(&path)
                    .await
                    .with_context(|| format!("Failed to read file: {}", path.display()))?;
                tracing::debug!(path = %path.display(), "Read cache miss");
                self.file_cache.put(path.clone(), mtime, raw.clone()).await;
                raw
            }
        } else if is_utf8 {
            tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("Failed to read file: {}", path.display()))?
        } else {
            let raw_bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("Failed to read file: {}", path.display()))?;
            let encoding = encoding_rs::Encoding::for_label(encoding_label.as_bytes())
                .ok_or_else(|| anyhow::anyhow!("Unknown encoding: \"{encoding_label}\""))?;
            let (cow, _encoding_used, had_errors) = encoding.decode(&raw_bytes);
            if had_errors {
                tracing::warn!(
                    "Read {}: some bytes could not be decoded with encoding \"{}\"",
                    path.display(),
                    encoding_label
                );
            }
            cow.into_owned()
        };

        // Smart truncation for large files (only when reading full file).
        let (_truncated_notice, content) = if !is_partial && needs_truncation {
            let total_bytes = content.len();
            // Keep ~60% head, ~40% tail, joined with a truncation notice.
            let head_ratio = 0.6_f64;
            let max_keep = (SMART_TRUNCATE_BYTES as f64 * 0.95) as usize; // leave room for the notice
            let head_bytes = (max_keep as f64 * head_ratio) as usize;
            let tail_bytes = max_keep.saturating_sub(head_bytes);

            // Find line boundaries to avoid splitting mid-line.
            let head_end = content
                .get(..head_bytes)
                .and_then(|s| s.rfind('\n'))
                .map(|i| i + 1)
                .unwrap_or(head_bytes);
            let tail_start = content
                .len()
                .saturating_sub(tail_bytes)
                .min(content.len());
            let tail_start = content
                .get(tail_start..)
                .and_then(|s| s.find('\n'))
                .map(|i| tail_start + i + 1)
                .unwrap_or(tail_start);

            let total_lines = content.lines().count();
            let head_lines = content[..head_end].lines().count();
            let tail_lines = content[tail_start..].lines().count();

            let notice = format!(
                "\n\n--- ⚠️ File truncated: {total_bytes} bytes, {total_lines} lines total. \
                 Showing first {head_lines} and last {tail_lines} lines. ---\n\n"
            );

            let mut truncated = String::with_capacity(head_end + tail_bytes + notice.len() + 100);
            truncated.push_str(&content[..head_end]);
            truncated.push_str(&notice);
            truncated.push_str(&content[tail_start..]);
            (notice, truncated)
        } else {
            (String::new(), content)
        };

        let (content, line_range) = if is_partial {
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len() as u64;

            let offset = args.offset.unwrap_or(1).max(1);
            if offset > total_lines {
                anyhow::bail!(
                    "Read offset {} exceeds total line count {} for: {}",
                    offset,
                    total_lines,
                    path.display()
                );
            }

            let start = (offset - 1) as usize;
            let remaining = total_lines - offset + 1;
            let limit = args.limit.unwrap_or(remaining).min(remaining) as usize;
            let end = (start + limit).min(lines.len());

            let sliced: String = lines[start..end].join("\n");
            (sliced, (offset, end as u64))
        } else {
            // Full read: compute line range for the response.
            let total = content.lines().count() as u64;
            (content, (1, total))
        };

        // Only record a read version for full, non-truncated reads so that Edit
        // safety checks remain correct — partial or truncated content would produce
        // a mismatching SHA-256 hash.
        if !is_partial && !needs_truncation {
            session.note_read(path.clone(), FileReadVersion::from_content(&content));
        }

        let mut result = serde_json::json!({
            "path": path.display().to_string(),
            "bytes": content.len(),
            "lines": line_range,
            "content": content,
        });
        if !is_partial && needs_truncation {
            result["truncated"] = serde_json::json!(true);
            result["original_bytes"] = serde_json::json!(meta.len());
        }
        Ok(result.to_string())
    }

    /// `ImageAnalyze`: read an image file and return its base64-encoded data URL
    /// along with enough metadata for the agent runtime to inject it as a
    /// multimodal user message.

    /// `ImageAnalyze`: read an image file and return its base64-encoded data URL
    /// along with enough metadata for the agent runtime to inject it as a
    /// multimodal user message.
    pub(crate) async fn image_analyze(
        &self,
        _session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<ImageAnalyzeOutput> {
        if cancel.is_cancelled() {
            anyhow::bail!("ImageAnalyze cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            prompt: Option<String>,
        }

        let args: Args =
            serde_json::from_value(args).context("Invalid arguments for ImageAnalyze")?;
        validate_tool_path_input(&args.path, PathOperation::Read)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Read)?;

        let meta = tokio::fs::metadata(&path)
            .await
            .with_context(|| format!("Failed to stat file: {}", path.display()))?;
        if !meta.is_file() {
            anyhow::bail!(
                "ImageAnalyze requires a file path, not a directory: {}",
                path.display()
            );
        }

        let raw_bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        // Infer MIME type from extension.
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        let media_type: String = match ext.as_deref() {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        }
        .to_string();

        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw_bytes);
        let data_url = format!("data:{media_type};base64,{b64}");

        Ok(ImageAnalyzeOutput {
            data_url,
            media_type,
            prompt: args.prompt,
        })
    }

    /// `Write`: create a new file and refuse to overwrite an existing one.

    /// `Write`: create a new file and refuse to overwrite an existing one.
    pub(crate) async fn write(
        &self,
        session: &mut ToolSession,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        if cancel.is_cancelled() {
            anyhow::bail!("Write cancelled");
        }

        #[derive(Debug, Deserialize)]
        struct Args {
            path: String,
            content: String,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Write")?;
        validate_tool_path_input(&args.path, PathOperation::Write)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Write)?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create parent directories: {}", parent.display())
            })?;
        }

        {
            use tokio::io::AsyncWriteExt as _;

            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .await
                .with_context(|| {
                    format!(
                        "Write refuses to overwrite an existing path: {}",
                        path.display()
                    )
                })?;
            file.write_all(args.content.as_bytes())
                .await
                .with_context(|| format!("Failed to write file: {}", path.display()))?;
            file.flush()
                .await
                .with_context(|| format!("Failed to flush file: {}", path.display()))?;
        }
        session.note_touched_path(path.clone());

        // Invalidate file cache for the newly created file.
        self.file_cache.invalidate_path(&path);

        Ok(serde_json::json!({
            "created": true,
            "path": path.display().to_string(),
            "bytes": args.content.len(),
        })
        .to_string())
    }

    /// `Edit`: replace text in an existing file that was previously `Read`.

    /// `Edit`: replace text in an existing file that was previously `Read`.
    pub(crate) async fn edit(
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
        validate_tool_path_input(&args.path, PathOperation::Edit)?;
        let path = self.ctx.resolve_under_workspace(&args.path)?;
        validate_resolved_tool_path(&self.ctx.workspace_root, &path, PathOperation::Edit)?;

        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file for edit: {}", path.display()))?;
        session.require_fresh_read(&path, &content)?;

        if args.old_text.is_empty() {
            anyhow::bail!("Edit old_text must be non-empty");
        }

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
        session.note_touched_path(path.clone());

        // Invalidate file cache for the edited file.
        self.file_cache.invalidate_path(&path);

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

    /// `Search`: discover relevant URLs before a more targeted `Fetch`.
    pub(crate) async fn search(
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

        let max_results = args.max_results.unwrap_or(5).clamp(1, 10);

        let results = crate::search_backends::multi_backend_search(
            args.query.trim(),
            max_results,
            &self.ctx.search_config,
            cancel,
        )
        .await?;

        Ok(serde_json::json!({
            "query": args.query,
            "results": results,
        })
        .to_string())
    }
}
