//! File auto-analysis module for the SA agent.
//!
//! Provides the `AnalyzeFile` built-in tool that automatically detects a file's
//! type and extracts a structured summary (text preview, encoding, line count,
//! image metadata, etc.).  For PDFs it delegates to MCP OCR; for images it
//! returns dimension/mime metadata.

use crate::cancel::CancelToken;
use crate::mcp_client::McpRegistry;
use crate::openai::{ToolDefinition, ToolFunctionDefinition};
use crate::path_guard::{PathOperation, validate_resolved_tool_path, validate_tool_path_input};
use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Maximum number of preview lines for text files.
const PREVIEW_LINES: usize = 20;

/// Maximum file size accepted by `AnalyzeFile`.
const MAX_ANALYZE_FILE_BYTES: u64 = 10_000_000; // 10 MB

/// Tool definition for `AnalyzeFile`.
pub fn analyze_file_definition() -> ToolDefinition {
    ToolDefinition {
        kind: "function".to_string(),
        function: ToolFunctionDefinition {
            name: "AnalyzeFile".to_string(),
            description: "Analyze a workspace file: detect its type, extract metadata and a content preview. Works for text, JSON, CSV, PDF, images, and binary files."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the workspace root or absolute under it."
                    },
                    "preview_lines": {
                        "type": "integer",
                        "description": "Number of text preview lines to include (0-100). Defaults to 20."
                    }
                },
                "required": ["path"]
            }),
        },
    }
}

/// Analysis result for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FileAnalysis {
    /// Absolute path of the analyzed file.
    pub path: String,
    /// Detected file category.
    pub file_type: String,
    /// MIME type hint.
    pub mime_type: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// For text files: detected encoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// For text files: total line count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_count: Option<u64>,
    /// For text files: first N lines of content preview.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
    /// For PDFs: OCR text content (if MCP OCR available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_text: Option<String>,
    /// For images: detected image dimensions info.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_info: Option<String>,
}

/// Detect MIME type from file extension.
fn detect_mime_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("txt") => "text/plain",
        Some("md") => "text/markdown",
        Some("json") => "application/json",
        Some("jsonl") | Some("ndjson") => "application/jsonl",
        Some("yaml") | Some("yml") => "application/yaml",
        Some("toml") => "application/toml",
        Some("csv") => "text/csv",
        Some("tsv") => "text/tab-separated-values",
        Some("xml") | Some("html") | Some("htm") => "text/xml",
        Some("rs") => "text/x-rust",
        Some("py") => "text/x-python",
        Some("js") | Some("ts") | Some("jsx") | Some("tsx") => "text/javascript",
        Some("css") => "text/css",
        Some("sh") | Some("bash") | Some("zsh") => "text/x-shellscript",
        Some("pdf") => "application/pdf",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("svg") => "image/svg+xml",
        Some("zip") => "application/zip",
        Some("gz") | Some("tar") => "application/gzip",
        Some("exe") | Some("dll") | Some("so") => "application/octet-stream",
        Some("parquet") => "application/vnd.apache.parquet",
        Some("sqlite") | Some("db") | Some("sqlite3") => "application/x-sqlite3",
        Some("epub") => "application/epub+zip",
        Some("pptx") => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        Some("docx") => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        Some("xlsx") | Some("xls") => {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        _ => "application/octet-stream",
    }
}

/// Classify file type into a category string.
fn classify_file_type(path: &Path) -> &'static str {
    match detect_mime_type(path) {
        t if t.starts_with("text/") => "text",
        "application/json" | "application/jsonl" | "application/yaml" | "application/toml" => {
            "structured_text"
        }
        "application/pdf" => "pdf",
        "application/vnd.apache.parquet" => "data_format",
        "application/x-sqlite3" => "database",
        "application/epub+zip" => "ebook",
        t if t.starts_with("application/vnd.openxmlformats") => "office_document",
        t if t.starts_with("image/") => "image",
        _ => "binary",
    }
}

/// Execute the `AnalyzeFile` tool.
pub async fn execute_analyze_file(
    workspace_root: &Path,
    path_resolver: &dyn crate::file_search::PathResolver,
    mcp_registry: &Option<Arc<McpRegistry>>,
    args: serde_json::Value,
    cancel: &CancelToken,
) -> anyhow::Result<String> {
    if cancel.is_cancelled() {
        anyhow::bail!("AnalyzeFile cancelled");
    }

    #[derive(Debug, Deserialize)]
    struct Args {
        path: String,
        preview_lines: Option<usize>,
    }

    let args: Args = serde_json::from_value(args).context("Invalid arguments for AnalyzeFile")?;
    validate_tool_path_input(&args.path, PathOperation::Read)?;
    let path = path_resolver.resolve_under_workspace(&args.path)?;
    validate_resolved_tool_path(workspace_root, &path, PathOperation::Read)?;

    let meta = tokio::fs::metadata(&path)
        .await
        .with_context(|| format!("Failed to stat file: {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!(
            "AnalyzeFile requires a file path, not a directory: {}",
            path.display()
        );
    }
    if meta.len() > MAX_ANALYZE_FILE_BYTES {
        anyhow::bail!(
            "File is too large to AnalyzeFile ({} bytes > {}): {}",
            meta.len(),
            MAX_ANALYZE_FILE_BYTES,
            path.display()
        );
    }

    let file_type = classify_file_type(&path);
    let mime_type = detect_mime_type(&path).to_string();
    let abs_path = path.display().to_string();
    let preview_n = args.preview_lines.unwrap_or(PREVIEW_LINES).min(100);

    let mut analysis = FileAnalysis {
        path: abs_path.clone(),
        file_type: file_type.to_string(),
        mime_type: mime_type.clone(),
        size_bytes: meta.len(),
        encoding: None,
        line_count: None,
        preview: None,
        ocr_text: None,
        image_info: None,
    };

    match file_type {
        "text" | "structured_text" => {
            let raw_bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("Failed to read file: {}", path.display()))?;

            // Detect encoding via BOM or default to UTF-8.
            let (encoding_label, content) = detect_encoding_and_decode(&raw_bytes);

            analysis.encoding = Some(encoding_label);
            let lines: Vec<&str> = content.lines().collect();
            analysis.line_count = Some(lines.len() as u64);

            if preview_n > 0 {
                let preview_lines: String = lines
                    .iter()
                    .take(preview_n)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n");
                analysis.preview = Some(preview_lines);
            }
        }
        "pdf" => {
            // Try MCP OCR.
            if let Some(registry) = mcp_registry {
                let ocr_candidates = ["ocr_pdf", "ocr/ocr_pdf", "fileanalyzer/analyze_file"];
                for tool_name in &ocr_candidates {
                    if registry.has_tool(tool_name) {
                        let mcp_args = serde_json::json!({ "file_path": abs_path });
                        if let Ok(text) = registry.call_tool(tool_name, mcp_args).await {
                            analysis.ocr_text = Some(text);
                            break;
                        }
                    }
                }
            }
            if analysis.ocr_text.is_none() {
                analysis.ocr_text =
                    Some("[PDF content unavailable — no MCP OCR server configured]".to_string());
            }
        }
        "image" => {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .unwrap_or_default();
            analysis.image_info = Some(format!(
                "Format: {ext}, MIME: {mime_type}, Size: {} bytes",
                meta.len()
            ));
        }
        _ => {
            // Generic binary — just report size.
        }
    }

    Ok(serde_json::to_string_pretty(&analysis).context("Failed to serialize FileAnalysis")?)
}

/// Detect text encoding from raw bytes (BOM + heuristic) and decode.
fn detect_encoding_and_decode(raw_bytes: &[u8]) -> (String, String) {
    // Check for UTF-8 BOM.
    if raw_bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        let content = String::from_utf8_lossy(&raw_bytes[3..]).into_owned();
        return ("utf-8-bom".to_string(), content);
    }
    // Check for UTF-16 LE BOM.
    if raw_bytes.len() >= 2 && raw_bytes[0] == 0xFF && raw_bytes[1] == 0xFE {
        if let Ok(s) = String::from_utf16(&u16_le_from_bytes(&raw_bytes[2..])) {
            return ("utf-16-le".to_string(), s);
        }
    }
    // Check for UTF-16 BE BOM.
    if raw_bytes.len() >= 2 && raw_bytes[0] == 0xFE && raw_bytes[1] == 0xFF {
        if let Ok(s) = String::from_utf16(&u16_be_from_bytes(&raw_bytes[2..])) {
            return ("utf-16-be".to_string(), s);
        }
    }
    // Default: try UTF-8 first, fall back to GBK for CJK content.
    if std::str::from_utf8(raw_bytes).is_ok() {
        let content = String::from_utf8_lossy(raw_bytes).into_owned();
        return ("utf-8".to_string(), content);
    }
    // Try GBK for Chinese content.
    let encoding = encoding_rs::Encoding::for_label(b"gbk").unwrap_or(encoding_rs::UTF_8);
    let (cow, _used, had_errors) = encoding.decode(raw_bytes);
    let label = if had_errors { "gbk (lossy)" } else { "gbk" };
    (label.to_string(), cow.into_owned())
}

fn u16_le_from_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn u16_be_from_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect()
}
