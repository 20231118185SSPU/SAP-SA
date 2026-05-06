//! Memory file indexer for vector semantic search (D2).
//!
//! Scans workspace memory files, chunks them, generates embeddings,
//! and stores them in the VectorIndex for ANN search.

use std::path::Path;
use std::time::SystemTime;

use anyhow::Context as _;
use tokio::fs;

use crate::openai::{EmbeddingInput, OpenAiClient};
use crate::vector_store::VectorIndex;

/// Configuration for memory indexing.
pub struct IndexConfig {
    /// Maximum characters per chunk (default 512).
    pub chunk_size: usize,
    /// Overlap between chunks in characters (default 64).
    pub chunk_overlap: usize,
    /// Maximum number of vector chunks. 0 = unlimited (default).
    pub max_episodic_entries: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
            max_episodic_entries: 0,
        }
    }
}

/// Result of indexing a single memory file.
#[derive(Debug, Clone)]
pub struct IndexResult {
    /// Workspace-relative path of the indexed file.
    pub path: String,
    /// Number of chunks created.
    pub chunks_indexed: usize,
    /// Total embedding API calls made.
    pub api_calls: usize,
}

/// Index a single memory file into the vector store.
///
/// - Reads the file content
/// - Chunks it into overlapping segments
/// - Batch-embeds all chunks
/// - Upserts into the VectorIndex
pub async fn index_memory_file(
    workspace_root: &Path,
    file_path: &Path,
    vector_index: &VectorIndex,
    client: &OpenAiClient,
    model: &str,
    ollama_mode: bool,
    config: &IndexConfig,
) -> anyhow::Result<IndexResult> {
    let rel_path = file_path
        .strip_prefix(workspace_root)
        .unwrap_or(file_path)
        .to_string_lossy()
        .replace('\\', "/");

    let content = fs::read_to_string(file_path)
        .await
        .with_context(|| format!("Failed to read memory file: {}", file_path.display()))?;

    // Strip YAML front matter
    let (_, body, fm_lines) = crate::memory::parse_yaml_front_matter(&content);

    // Chunk the content
    let chunks = chunk_text(&body, config.chunk_size, config.chunk_overlap);

    if chunks.is_empty() {
        return Ok(IndexResult {
            path: rel_path,
            chunks_indexed: 0,
            api_calls: 0,
        });
    }

    // Delete old chunks for this file (full re-index)
    vector_index.delete_by_path(&rel_path)?;

    // Batch embed (send all texts at once when possible)
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let _batch_size = if ollama_mode { 1 } else { texts.len() };
    let mut api_calls = 0usize;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    if ollama_mode {
        // Ollama: send one at a time
        for (i, text) in texts.iter().enumerate() {
            let input = EmbeddingInput::Single(text.to_string());
            let embeddings = client
                .embeddings(model, input, true)
                .await
                .map_err(|e| anyhow::anyhow!("embedding API error: {e}"))?;
            api_calls += 1;

            if let Some(emb) = embeddings.first() {
                let chunk = &chunks[i];
                vector_index.upsert_chunk(
                    &rel_path,
                    chunk.start_line + fm_lines,
                    chunk.end_line + fm_lines,
                    &chunk.text,
                    emb,
                    now,
                )?;
            }
        }
    } else {
        // OpenAI-compatible: batch send
        let input = EmbeddingInput::Batch(texts.iter().map(|s| s.to_string()).collect());
        let embeddings = client
            .embeddings(model, input, false)
            .await
            .map_err(|e| anyhow::anyhow!("embedding API error: {e}"))?;
        api_calls += 1;

        for (i, emb) in embeddings.iter().enumerate() {
            let chunk = &chunks[i];
            vector_index.upsert_chunk(
                &rel_path,
                chunk.start_line + fm_lines,
                chunk.end_line + fm_lines,
                &chunk.text,
                emb,
                now,
            )?;
        }
    }

    Ok(IndexResult {
        path: rel_path,
        chunks_indexed: chunks.len(),
        api_calls,
    })
}

/// Index all memory files in a workspace.
///
/// Returns the number of files indexed and total chunks created.
pub async fn index_workspace_memory(
    workspace_root: &Path,
    vector_index: &VectorIndex,
    client: &OpenAiClient,
    model: &str,
    ollama_mode: bool,
    config: &IndexConfig,
) -> anyhow::Result<(usize, usize)> {
    // Capacity eviction: remove oldest entries if over limit.
    if config.max_episodic_entries > 0 {
        if let Ok(evicted) = vector_index.evict_oldest(config.max_episodic_entries) {
            if evicted > 0 {
                tracing::info!("evicted {} episodic entries (capacity limit {})", evicted, config.max_episodic_entries);
            }
        }
    }

    let files = crate::memory::indexed_memory_files(workspace_root)?;

    let mut total_files = 0usize;
    let mut total_chunks = 0usize;

    for file_path in &files {
        match index_memory_file(
            workspace_root,
            file_path,
            vector_index,
            client,
            model,
            ollama_mode,
            config,
        )
        .await
        {
            Ok(result) => {
                total_files += 1;
                total_chunks += result.chunks_indexed;
                tracing::debug!(
                    path = %result.path,
                    chunks = result.chunks_indexed,
                    api_calls = result.api_calls,
                    "indexed memory file"
                );
            }
            Err(e) => {
                tracing::warn!(
                    path = %file_path.display(),
                    error = %e,
                    "failed to index memory file"
                );
            }
        }
    }

    Ok((total_files, total_chunks))
}

/// Index workspace memory files using `Arc<Mutex<VectorIndex>>`.
///
/// Unlike [`index_workspace_memory`], this function acquires and releases
/// the mutex around each sync operation, avoiding holding a non-Send
/// MutexGuard across async embedding calls.  Suitable for use inside
/// `tokio::spawn`.
pub async fn index_workspace_memory_arc(
    workspace_root: &Path,
    vector_index: &std::sync::Arc<std::sync::Mutex<VectorIndex>>,
    client: &OpenAiClient,
    model: &str,
    ollama_mode: bool,
    config: &IndexConfig,
) -> anyhow::Result<(usize, usize)> {
    let files = crate::memory::indexed_memory_files(workspace_root)?;

    let mut total_files = 0usize;
    let mut total_chunks = 0usize;

    for file_path in &files {
        let rel_path = file_path
            .strip_prefix(workspace_root)
            .unwrap_or(file_path)
            .to_string_lossy()
            .replace('\\', "/");

        // Async read — no lock held.
        let content = match fs::read_to_string(file_path).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %file_path.display(), error = %e, "read failed");
                continue;
            }
        };

        let (_, body, fm_lines) = crate::memory::parse_yaml_front_matter(&content);
        let chunks = chunk_text(&body, config.chunk_size, config.chunk_overlap);
        if chunks.is_empty() {
            continue;
        }

        // Sync: delete old chunks — lock briefly.
        {
            let vi = vector_index.lock().unwrap();
            if let Err(e) = vi.delete_by_path(&rel_path) {
                tracing::warn!(path = %rel_path, error = %e, "delete old chunks failed");
            }
        }

        // Async: embed — no lock.
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut api_calls = 0usize;

        if ollama_mode {
            for (i, text) in texts.iter().enumerate() {
                let input = EmbeddingInput::Single(text.to_string());
                let emb = match client.embeddings(model, input, true).await {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!(error = %e, "embedding failed");
                        continue;
                    }
                };
                api_calls += 1;
                if let Some(v) = emb.first() {
                    let chunk = &chunks[i];
                    let vi = vector_index.lock().unwrap();
                    let _ = vi.upsert_chunk(
                        &rel_path,
                        chunk.start_line + fm_lines,
                        chunk.end_line + fm_lines,
                        &chunk.text,
                        v,
                        now,
                    );
                }
            }
        } else {
            let input = EmbeddingInput::Batch(texts.iter().map(|s| s.to_string()).collect());
            let embeddings = match client.embeddings(model, input, false).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(error = %e, "batch embedding failed");
                    continue;
                }
            };
            api_calls += 1;
            let vi = vector_index.lock().unwrap();
            for (i, emb) in embeddings.iter().enumerate() {
                let chunk = &chunks[i];
                let _ = vi.upsert_chunk(
                    &rel_path,
                    chunk.start_line + fm_lines,
                    chunk.end_line + fm_lines,
                    &chunk.text,
                    emb,
                    now,
                );
            }
        }

        total_files += 1;
        total_chunks += chunks.len();
        tracing::debug!(
            path = %rel_path,
            chunks = chunks.len(),
            api_calls,
            "indexed memory file (arc)"
        );
    }

    Ok((total_files, total_chunks))
}

/// A chunk of text with its line range (relative to file body, excluding front matter).
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// 1-based start line within the body (after front matter).
    pub start_line: usize,
    /// 1-based end line within the body.
    pub end_line: usize,
    /// The chunk text.
    pub text: String,
}

/// Split text into overlapping chunks.
///
/// Chunks by lines, respecting paragraph boundaries when possible.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<TextChunk> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        let mut end = start;
        let mut char_count = 0usize;

        while end < lines.len() && char_count + lines[end].len() + 1 <= chunk_size {
            char_count += lines[end].len() + 1; // +1 for newline
            end += 1;
        }

        // Ensure at least one line per chunk
        if end == start {
            end += 1;
        }

        let chunk_text = lines[start..end].join("\n");
        chunks.push(TextChunk {
            start_line: start + 1,
            end_line: end,
            text: chunk_text,
        });

        // Advance with overlap
        if end >= lines.len() {
            break;
        }

        let overlap_chars = overlap;
        let mut overlap_lines = 0usize;
        let mut overlap_count = 0usize;
        let mut idx = end - 1;
        while idx > start && overlap_count < overlap_chars {
            overlap_count += lines[idx].len() + 1;
            overlap_lines += 1;
            if idx == 0 {
                break;
            }
            idx -= 1;
        }

        start = end - overlap_lines;
        if start >= end {
            start = end;
        }
    }

    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_basic() {
        let text = "line1\nline2\nline3\nline4\nline5";
        let chunks = chunk_text(text, 20, 10);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn test_chunk_empty() {
        let chunks = chunk_text("", 512, 64);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_single_line() {
        let text = "hello world";
        let chunks = chunk_text(text, 512, 64);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }
}
