//! Procedural memory --automated skill discovery from session tool-call sequences (D7).
//!
//! Scans workspace `sessions/*.jsonl` files, extracts ordered tool-call
//! sequences from assistant turns, and detects repeated patterns that meet
//! minimum occurrence and success-rate thresholds.  Discovered procedures
//! are written to `memory/procedures/` as lightweight Markdown skill files
//! with YAML front matter.

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::SystemTime;

use crate::session::SESSIONS_DIR_NAME;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// One step inside a discovered procedure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ToolStep {
    /// Tool name as called by the model.
    pub tool_name: String,
    /// First argument keys, comma-separated (for disambiguation).
    pub arg_keys: String,
}

/// A fully discovered procedural skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureSkill {
    /// Stable name derived from the tool-name sequence.
    pub name: String,
    /// Ordered list of tool steps.
    pub steps: Vec<ToolStep>,
    /// How many times this exact sequence appeared.
    pub occurrences: usize,
    /// Fraction of runs where every step in the sequence succeeded.
    pub success_rate: f64,
    /// Source session files that contained this pattern.
    pub source_sessions: Vec<String>,
    /// When the skill was first discovered.
    pub discovered_at: String,
    /// Skill version (bumped on re-discovery with new data).
    pub version: u32,
}

/// Raw observation of a single tool call during session scanning.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RawToolCall {
    tool_name: String,
    arg_keys: String,
    /// Whether the corresponding tool result indicated success.
    success: bool,
}

/// Aggregated stats for one tool-name sequence.
#[derive(Debug, Clone)]
struct SequenceStats {
    name: String,
    steps: Vec<ToolStep>,
    occurrences: usize,
    successes: usize,
    source_sessions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thresholds for skill discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureConfig {
    /// Minimum occurrences for a sequence to qualify (default 5).
    pub min_occurrences: usize,
    /// Minimum success rate for a sequence to qualify (default 0.8).
    pub min_success_rate: f64,
    /// Minimum sequence length (default 2).
    pub min_sequence_length: usize,
    /// Maximum sequence length (default 5).
    pub max_sequence_length: usize,
    /// Output directory name under workspace root (default "memory/procedures").
    pub output_dir: String,
}

impl Default for ProcedureConfig {
    fn default() -> Self {
        Self {
            min_occurrences: 5,
            min_success_rate: 0.8,
            min_sequence_length: 2,
            max_sequence_length: 5,
            output_dir: "memory/procedures".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Session scanning
// ---------------------------------------------------------------------------

/// Minimal stored message for extracting tool calls.
#[derive(Debug, Deserialize)]
struct StoredMsg {
    role: String,
    #[serde(default)]
    tool_calls: Option<Vec<StoredToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

/// Lightweight tool-call stub (avoids depending on full `ToolCall` fields).
#[derive(Debug, Deserialize)]
struct StoredToolCall {
    id: String,
    function: StoredFunc,
}

#[derive(Debug, Deserialize)]
struct StoredFunc {
    name: String,
    #[serde(default)]
    arguments: String,
}

/// Result of scanning all sessions.
pub struct ScanResult {
    pub procedures: Vec<ProcedureSkill>,
    pub files_scanned: usize,
    pub sequences_found: usize,
}

/// Scan all session JSONL files and discover repeated tool-call patterns.
pub fn scan_procedures(
    workspace_root: &Path,
    config: &ProcedureConfig,
) -> anyhow::Result<ScanResult> {
    let sessions_dir = workspace_root.join(SESSIONS_DIR_NAME);
    if !sessions_dir.is_dir() {
        return Ok(ScanResult {
            procedures: Vec::new(),
            files_scanned: 0,
            sequences_found: 0,
        });
    }

    let mut all_sequences: HashMap<String, SequenceStats> = HashMap::new();
    let mut files_scanned = 0usize;

    let entries = fs::read_dir(&sessions_dir)
        .with_context(|| format!("Failed to read sessions dir: {}", sessions_dir.display()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }

        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to read session");
                continue;
            }
        };

        let calls = extract_tool_calls(&content);
        if calls.is_empty() {
            continue;
        }

        files_scanned += 1;
        extract_sequences(&file_stem, &calls, config, &mut all_sequences);
    }

    // Filter and build skills
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let discovered_at = chrono::DateTime::from_timestamp(now as i64, 0)
        .unwrap_or_default()
        .to_rfc3339();

    let procedures: Vec<ProcedureSkill> = all_sequences
        .into_values()
        .filter(|s| {
            s.occurrences >= config.min_occurrences
                && (s.successes as f64 / s.occurrences as f64) >= config.min_success_rate
        })
        .map(|s| ProcedureSkill {
            name: s.name,
            steps: s.steps,
            occurrences: s.occurrences,
            success_rate: s.successes as f64 / s.occurrences as f64,
            source_sessions: s.source_sessions,
            discovered_at: discovered_at.clone(),
            version: 1,
        })
        .collect();

    let sequences_found = procedures.len();

    Ok(ScanResult {
        procedures,
        files_scanned,
        sequences_found,
    })
}

/// Extract tool calls from a single session JSONL string.
///
/// Returns a flat list of `RawToolCall` with success/failure determined by
/// matching tool result messages.
fn extract_tool_calls(content: &str) -> Vec<RawToolCall> {
    // First pass: collect all assistant tool calls and tool results.
    let mut pending_calls: HashMap<String, StoredToolCall> = HashMap::new();
    let mut tool_results: HashMap<String, bool> = HashMap::new();
    let mut call_order: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Try parsing as a message (skip meta/compaction lines).
        if let Ok(msg) = serde_json::from_str::<StoredMsg>(line) {
            if msg.role == "assistant" {
                if let Some(calls) = msg.tool_calls {
                    for call in calls {
                        call_order.push(call.id.clone());
                        pending_calls.insert(call.id.clone(), call);
                    }
                }
            } else if msg.role == "tool" {
                if let Some(id) = msg.tool_call_id {
                    // A tool result is successful unless it contains an error.
                    // Heuristic: check raw line for error indicators.
                    let success = !line.contains("\"is_error\":true")
                        && !line.contains("\"is_error\": true");
                    tool_results.insert(id, success);
                }
            }
        }
    }

    // Build ordered RawToolCall list.
    let mut result = Vec::with_capacity(call_order.len());
    for id in &call_order {
        if let Some(call) = pending_calls.get(id) {
            let success = tool_results.get(id).copied().unwrap_or(true);
            let arg_keys = extract_arg_keys(&call.function.arguments);
            result.push(RawToolCall {
                tool_name: call.function.name.clone(),
                arg_keys,
                success,
            });
        }
    }
    result
}

/// Extract comma-separated sorted argument keys from JSON arguments string.
fn extract_arg_keys(arguments: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    match parsed {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
            keys.sort();
            keys.join(",")
        }
        _ => String::new(),
    }
}

/// Extract subsequences from a flat tool-call list and accumulate stats.
fn extract_sequences(
    session_id: &str,
    calls: &[RawToolCall],
    config: &ProcedureConfig,
    acc: &mut HashMap<String, SequenceStats>,
) {
    let n = calls.len();
    for len in config.min_sequence_length..=config.max_sequence_length.min(n) {
        for start in 0..=(n - len) {
            let slice = &calls[start..start + len];
            let steps: Vec<ToolStep> = slice
                .iter()
                .map(|c| ToolStep {
                    tool_name: c.tool_name.clone(),
                    arg_keys: c.arg_keys.clone(),
                })
                .collect();
            let seq_key = sequence_key(&steps);
            let all_ok = slice.iter().all(|c| c.success);

            let entry = acc.entry(seq_key.clone()).or_insert_with(|| SequenceStats {
                name: sequence_name(&steps),
                steps: steps.clone(),
                occurrences: 0,
                successes: 0,
                source_sessions: Vec::new(),
            });
            entry.occurrences += 1;
            if all_ok {
                entry.successes += 1;
            }
            if !entry.source_sessions.contains(&session_id.to_string()) {
                entry.source_sessions.push(session_id.to_string());
            }
        }
    }
}

/// Deterministic key for a tool-step sequence (name + arg_keys per step).
fn sequence_key(steps: &[ToolStep]) -> String {
    steps
        .iter()
        .map(|s| format!("{}:{}", s.tool_name, s.arg_keys))
        .collect::<Vec<_>>()
        .join("|")
}

/// Human-readable name for a sequence: "Tool1 ->Tool2 ->Tool3".
fn sequence_name(steps: &[ToolStep]) -> String {
    steps
        .iter()
        .map(|s| s.tool_name.as_str())
        .collect::<Vec<_>>()
        .join(" ->")
}

// ---------------------------------------------------------------------------
// Skill file generation
// ---------------------------------------------------------------------------

/// Write discovered procedures to `memory/procedures/` as Markdown skill files.
///
/// Each file gets a YAML front matter header with metadata, followed by a
/// human-readable description of the steps and stats.
pub fn write_procedures(
    workspace_root: &Path,
    procedures: &[ProcedureSkill],
    config: &ProcedureConfig,
) -> anyhow::Result<usize> {
    let output_dir = workspace_root.join(&config.output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create procedures dir: {}", output_dir.display()))?;

    let mut written = 0usize;

    for skill in procedures {
        let filename = sanitize_filename(&skill.name) + ".md";
        let path = output_dir.join(&filename);

        // Bump version if the skill already exists.
        let version = if path.exists() {
            let old = std::fs::read_to_string(&path).unwrap_or_default();
            let prev = old
                .lines()
                .find_map(|line| {
                    let line = line.trim();
                    if let Some(v) = line.strip_prefix("version:") {
                        v.trim().parse::<u32>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            prev + 1
        } else {
            skill.version
        };

        let md = render_skill_markdown(skill, version);
        fs::write(&path, &md)
            .with_context(|| format!("Failed to write skill file: {}", path.display()))?;
        written += 1;

        tracing::info!(
            name = %skill.name,
            version,
            occurrences = skill.occurrences,
            success_rate = skill.success_rate,
            "wrote procedure skill"
        );
    }

    Ok(written)
}

/// Render a `ProcedureSkill` as a Markdown file with YAML front matter.
fn render_skill_markdown(skill: &ProcedureSkill, version: u32) -> String {
    let mut out = String::new();

    // YAML front matter
    out.push_str("---\n");
    let _ = writeln!(out, "name: {}", skill.name);
    let _ = writeln!(out, "occurrences: {}", skill.occurrences);
    let _ = writeln!(out, "success_rate: {:.2}", skill.success_rate);
    let _ = writeln!(out, "version: {}", version);
    let _ = writeln!(out, "discovered_at: {}", skill.discovered_at);
    let _ = writeln!(
        out,
        "source_sessions: [{}]",
        skill
            .source_sessions
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );
    out.push_str("steps:\n");
    for step in &skill.steps {
        let _ = writeln!(out, "  - tool: {}", step.tool_name);
        if !step.arg_keys.is_empty() {
            let _ = writeln!(out, "    args: [{}]", step.arg_keys);
        }
    }
    out.push_str("---\n\n");

    // Body
    let _ = writeln!(out, "# {}", skill.name);
    out.push('\n');
    out.push_str("从重复工具调用模式自动发现的程序记忆。\n\n");

    out.push_str("## Steps\n\n");
    for (i, step) in skill.steps.iter().enumerate() {
        if step.arg_keys.is_empty() {
            let _ = writeln!(out, "{}. **{}**", i + 1, step.tool_name);
        } else {
            let _ = writeln!(
                out,
                "{}. **{}** (args: {})",
                i + 1,
                step.tool_name,
                step.arg_keys
            );
        }
    }
    out.push('\n');

    out.push_str("## Stats\n\n");
    let _ = writeln!(out, "- Occurrences: {}", skill.occurrences);
    let _ = writeln!(out, "- Success rate: {:.0}%", skill.success_rate * 100.0);
    let _ = writeln!(out, "- Version: {}", version);
    let _ = writeln!(out, "- Discovered: {}", skill.discovered_at);
    out.push('\n');

    out
}

/// Sanitize a sequence name into a safe filename.
fn sanitize_filename(name: &str) -> String {
    // Collapse "->" arrow sequences into Unicode arrow for uniform handling.
    let name = name.replace("->", "\u{2192}");
    name.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            ' ' | '\u{2192}' => '_',
            _ => '_',
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

// ---------------------------------------------------------------------------
// Dream integration --inject procedural-memory block into runtime context.
// ---------------------------------------------------------------------------

/// Build a runtime-context text block summarizing discovered procedures.
///
/// Called from `dream.rs` when preparing the dream context, so the model
/// knows what skills have been automatically discovered.
pub fn procedures_context_block(workspace_root: &Path, config: &ProcedureConfig) -> String {
    let output_dir = workspace_root.join(&config.output_dir);
    if !output_dir.is_dir() {
        return String::new();
    }

    let mut entries = Vec::new();
    if let Ok(dir) = fs::read_dir(&output_dir) {
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    entries.push(name.to_string());
                }
            }
        }
    }

    if entries.is_empty() {
        return String::new();
    }

    entries.sort();
    let mut out = String::from("## 已发现的程序记忆\n\n");
    out.push_str("以下技能从重复工具调用模式自动发现：\n\n");
    for name in &entries {
        let _ = writeln!(out, "- {name}");
    }
    out.push('\n');
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_arg_keys() {
        assert_eq!(extract_arg_keys(r#"{"b": 1, "a": 2}"#), "a,b");
        assert_eq!(extract_arg_keys("invalid"), "");
        assert_eq!(extract_arg_keys("null"), "");
    }

    #[test]
    fn test_sequence_key() {
        let steps = vec![
            ToolStep {
                tool_name: "Read".into(),
                arg_keys: "path".into(),
            },
            ToolStep {
                tool_name: "Edit".into(),
                arg_keys: "oldText,path".into(),
            },
        ];
        let key = sequence_key(&steps);
        assert_eq!(key, "Read:path|Edit:oldText,path");
    }

    #[test]
    fn test_sequence_name() {
        let steps = vec![
            ToolStep {
                tool_name: "Read".into(),
                arg_keys: String::new(),
            },
            ToolStep {
                tool_name: "Edit".into(),
                arg_keys: String::new(),
            },
        ];
        assert_eq!(sequence_name(&steps), "Read ->Edit");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Read ->Edit ->Write"), "Read__Edit__Write");
        assert_eq!(sanitize_filename("foo/bar\\baz"), "foo_bar_baz");
    }

    #[test]
    fn test_extract_tool_calls_from_session() {
        // Minimal session with one assistant turn containing 2 tool calls
        // followed by 2 tool results (one success, one failure).
        let session = r#"{"kind":"session_meta","schema_version":1,"conversation_id":"00000000-0000-0000-0000-000000000000","created_at":"2026-04-30T00:00:00Z"}
{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"a.txt\"}"}},{"id":"call_2","type":"function","function":{"name":"Edit","arguments":"{\"path\":\"a.txt\",\"oldText\":\"x\",\"newText\":\"y\"}"}}]}
{"role":"tool","tool_call_id":"call_1","content":"ok"}
{"role":"tool","tool_call_id":"call_2","content":"error","is_error":true}"#;

        let calls = extract_tool_calls(session);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_name, "Read");
        assert!(calls[0].success);
        assert_eq!(calls[1].tool_name, "Edit");
        assert!(!calls[1].success);
    }

    #[test]
    fn test_extract_sequences() {
        let calls = vec![
            RawToolCall {
                tool_name: "Read".into(),
                arg_keys: "path".into(),
                success: true,
            },
            RawToolCall {
                tool_name: "Edit".into(),
                arg_keys: "path,oldText,newText".into(),
                success: true,
            },
            RawToolCall {
                tool_name: "Bash".into(),
                arg_keys: "command".into(),
                success: true,
            },
        ];

        let mut acc = HashMap::new();
        let config = ProcedureConfig::default();

        // Simulate 5 occurrences
        for i in 0..5 {
            extract_sequences(&format!("sess_{i}"), &calls, &config, &mut acc);
        }

        // "Read ->Edit" should have 5 occurrences (from windows at [0,1] and [1,2] --wait, [0,1] and [1,2] are different)
        // Actually with len=3 and min_seq_len=2:
        // [0,1]: Read, Edit ->1 occurrence per file = 5 total
        // [1,2]: Edit, Bash ->1 occurrence per file = 5 total
        // [0,1,2]: Read, Edit, Bash ->1 occurrence per file = 5 total
        let _read_edit_key = format!(
            "{}|{}",
            sequence_key(&[ToolStep {
                tool_name: "Read".into(),
                arg_keys: "path".into()
            }]),
            sequence_key(&[ToolStep {
                tool_name: "Edit".into(),
                arg_keys: "path,oldText,newText".into()
            }])
        )
        .replace('|', "|");

        let key_2 = format!(
            "{}|{}",
            "Read:path",
            "Edit:path,oldText,newText"
        );

        assert!(acc.contains_key(&key_2));
        assert_eq!(acc[&key_2].occurrences, 5);
        assert_eq!(acc[&key_2].successes, 5);
    }

    #[test]
    fn test_render_skill_markdown() {
        let skill = ProcedureSkill {
            name: "Read ->Edit".into(),
            steps: vec![
                ToolStep {
                    tool_name: "Read".into(),
                    arg_keys: "path".into(),
                },
                ToolStep {
                    tool_name: "Edit".into(),
                    arg_keys: "path,oldText,newText".into(),
                },
            ],
            occurrences: 7,
            success_rate: 0.857,
            source_sessions: vec!["sess_001".into(), "sess_002".into()],
            discovered_at: "2026-04-30T15:00:00+08:00".into(),
            version: 1,
        };

        let md = render_skill_markdown(&skill, skill.version);
        assert!(md.contains("name: Read ->Edit"));
        assert!(md.contains("occurrences: 7"));
        assert!(md.contains("success_rate: 0.86"));
        assert!(md.contains("# Read ->Edit"));
        assert!(md.contains("1. **Read** (args: path)"));
        assert!(md.contains("2. **Edit** (args: path,oldText,newText)"));
    }
}
