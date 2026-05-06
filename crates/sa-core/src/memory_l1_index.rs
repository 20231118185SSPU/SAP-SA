//! L1 Memory Index — lightweight insight index for fast context lookups.
//!
//! After each dream audit, a concise index file is generated (≤30 lines)
//! mapping keywords to memory file paths. This index is injected into the
//! agent's system prompt (<1K tokens) to enable efficient context retrieval.
//!
//! When the agent needs deeper information, it uses `file_read` on the
//! indexed path instead of loading everything into context.

use std::path::PathBuf;

/// Generate an L1 index from memory files in the given directory.
///
/// Returns a Markdown string suitable for injection into the system prompt.
pub async fn generate_l1_index(memory_dir: &PathBuf) -> Result<String, anyhow::Error> {
    let mut index_lines: Vec<String> = Vec::new();
    index_lines.push("## Memory Index (L1)".to_string());
    index_lines.push(String::new());

    if !memory_dir.exists() {
        index_lines.push("_(no memory directory found)_".to_string());
        return Ok(index_lines.join("\n"));
    }

    let mut entries = tokio::fs::read_dir(memory_dir).await?;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Extract keywords from filename.
        let keywords = extract_keywords(&filename);

        // Read first few lines to get context.
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            let first_line = content.lines().next().unwrap_or("").to_string();
            let entry = format!(
                "- **{kw}** → `{file}` | {desc}",
                kw = keywords,
                file = filename,
                desc = truncate_str(&first_line, 60)
            );
            index_lines.push(entry);
        }

        // Limit to 30 lines.
        if index_lines.len() >= 30 {
            index_lines.push("_(truncated)_".to_string());
            break;
        }
    }

    // Save index to file.
    let index_path = memory_dir.join("insight_index.md");
    let content = index_lines.join("\n");
    tokio::fs::write(&index_path, &content).await?;

    Ok(content)
}

/// Extract keywords from a filename (e.g., "2026-05-04.md" → "2026-05-04").
fn extract_keywords(filename: &str) -> String {
    let stem = filename.trim_end_matches(".md");
    // Replace separators with spaces for readability.
    stem.replace(['-', '_'], " ")
}

/// Truncate a string to a max length, appending "..." if needed.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Load the existing L1 index from the insight_index.md file.
pub async fn load_l1_index(memory_dir: &PathBuf) -> Option<String> {
    let path = memory_dir.join("insight_index.md");
    if path.exists() {
        tokio::fs::read_to_string(&path).await.ok()
    } else {
        None
    }
}

/// Format the L1 index as a prompt block for the system prompt.
pub fn format_l1_index_block(index_content: &str) -> String {
    if index_content.is_empty() {
        String::new()
    } else {
        format!(
            "\n\n---\n\n## Memory Index (L1)\n\n\
             Use this index to locate memory files. When investigating a topic, \
             use `file_read` on the referenced path instead of guessing.\n\n\
             {index_content}\n"
        )
    }
}
