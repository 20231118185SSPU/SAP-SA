//! Workflow data structures for YAML DAG-based workflow engine.
//!
//! A workflow is defined in YAML, parsed into a `WorkflowDef`, and executed
//! as a directed acyclic graph (DAG) where each node runs as an independent
//! async task.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ts_rs::TS;
use uuid::Uuid;

// ── Workflow Definition (parsed from YAML) ──────────────────────────────

/// Top-level YAML workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct WorkflowDef {
    /// Unique workflow name (matches the YAML filename without extension).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Ordered list of node definitions.
    pub nodes: Vec<WorkflowNodeDef>,
    /// Default model for all nodes (can be overridden per-node).
    #[serde(default)]
    pub default_model: Option<String>,
}

/// One node in the workflow DAG.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct WorkflowNodeDef {
    /// Unique node id within this workflow.
    pub id: String,
    /// What kind of work this node does.
    pub node_type: NodeType,
    /// Node ids that must complete before this node can start.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Optional conditional expression (evaluated against accumulated outputs).
    /// If the expression evaluates to false, the node is skipped.
    #[serde(default)]
    pub when: Option<String>,
    /// How to decide readiness when multiple dependencies exist.
    #[serde(default)]
    pub trigger_rule: TriggerRule,
    /// Whether to share context with other nodes or start fresh.
    #[serde(default)]
    pub context: ContextMode,
    /// Prompt template for Prompt / Loop / Approval nodes.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Shell command for Bash nodes.
    #[serde(default)]
    pub command: Option<String>,
    /// Optional human-readable label (defaults to `id`).
    #[serde(default)]
    pub label: Option<String>,
    /// Model ID override for this node. If None, uses the workflow default.
    #[serde(default)]
    pub model: Option<String>,
    /// Retry policy. If None, uses default (max_attempts=1, no retry).
    #[serde(default)]
    pub retry: Option<RetryPolicy>,
    /// Loop condition (only meaningful for Loop nodes).
    #[serde(default)]
    pub loop_condition: Option<LoopCondition>,
}

/// What kind of work a node performs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    /// Send a prompt to the LLM and collect the response.
    Prompt,
    /// Execute a shell command.
    Bash,
    /// Repeat a prompt until a condition is met.
    Loop,
    /// Pause and wait for human approval via the frontend.
    Approval,
}

/// Retry policy for a workflow node.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    /// Backoff delay in milliseconds between retries (exponential).
    #[serde(default = "default_backoff_ms")]
    pub backoff_ms: u64,
}

fn default_max_attempts() -> u32 {
    1
}
fn default_backoff_ms() -> u64 {
    1000
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: default_max_attempts(),
            backoff_ms: default_backoff_ms(),
        }
    }
}

/// Loop condition for a loop node.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct LoopCondition {
    /// Expression to evaluate after each iteration.
    /// e.g. "output not contains ERROR", "output len > 100".
    pub condition: String,
    /// Maximum number of iterations to prevent infinite loops.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

fn default_max_iterations() -> usize {
    5
}

/// How to determine if a node is ready when multiple dependencies exist.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum TriggerRule {
    /// All upstream nodes must succeed (default).
    AllSuccess,
    /// At least one upstream node must succeed.
    OneSuccess,
    /// No upstream node has failed (skipped is OK).
    NoneFailed,
}

impl Default for TriggerRule {
    fn default() -> Self {
        Self::AllSuccess
    }
}

/// Whether a node shares accumulated context or starts fresh.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum ContextMode {
    /// Fresh context — only the node's own prompt.
    Fresh,
    /// Shared context — accumulated outputs from upstream nodes.
    Shared,
}

impl Default for ContextMode {
    fn default() -> Self {
        Self::Shared
    }
}

// ── Workflow Run Instance ───────────────────────────────────────────────

/// A running (or completed) workflow instance.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct WorkflowRun {
    /// Unique run id.
    pub run_id: Uuid,
    /// Which workflow definition this run uses.
    pub workflow_name: String,
    /// Overall run status.
    pub status: WorkflowStatus,
    /// Per-node run states.
    pub nodes: Vec<NodeRun>,
    /// Inputs provided by the user.
    #[serde(default)]
    pub inputs: HashMap<String, String>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Completion timestamp.
    #[serde(default)]
    pub completed_at: Option<DateTime<Utc>>,
}

/// Per-node runtime state within a workflow run.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct NodeRun {
    /// Node id (matches `WorkflowNodeDef.id`).
    pub node_id: String,
    /// Current status.
    pub status: NodeStatus,
    /// Collected output text.
    #[serde(default)]
    pub output: Option<String>,
    /// When execution started.
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    /// When execution completed.
    #[serde(default)]
    pub completed_at: Option<DateTime<Utc>>,
    /// Error message if failed.
    #[serde(default)]
    pub error: Option<String>,
}

/// Overall workflow status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStatus {
    /// Created but not yet started.
    Pending,
    /// At least one node is running.
    Running,
    /// All nodes completed successfully.
    Completed,
    /// A critical node failed.
    Failed,
    /// User cancelled the workflow.
    Cancelled,
}

/// Per-node status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    /// Waiting for dependencies.
    Pending,
    /// Dependencies met, queued for execution.
    Waiting,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Encountered an error.
    Failed,
    /// Condition not met, or upstream failure.
    Skipped,
}

// ── Template Engine ─────────────────────────────────────────────────────

/// Render a template string by replacing `{{key}}` placeholders with values
/// from the given map.
pub fn render_template(template: &str, vars: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in vars {
        result = result.replace(&format!("{{{{{key}}}}}"), value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_template() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "world".to_string());
        assert_eq!(render_template("hello {{name}}", &vars), "hello world");
    }

    #[test]
    fn test_parse_workflow_yaml() {
        let yaml = r#"
name: test
description: A test workflow
nodes:
  - id: step1
    node_type: prompt
    prompt: "Hello {{input}}"
  - id: step2
    node_type: bash
    depends_on: [step1]
    command: "echo done"
"#;
        let def: WorkflowDef = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(def.name, "test");
        assert_eq!(def.nodes.len(), 2);
        assert_eq!(def.nodes[0].node_type, NodeType::Prompt);
        assert_eq!(def.nodes[1].depends_on, vec!["step1"]);
    }
}
