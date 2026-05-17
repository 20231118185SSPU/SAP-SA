//! DAG-based workflow execution engine.
//!
//! Parses a YAML workflow definition, builds a DAG, and executes nodes
//! in topological order with parallel execution of independent nodes.
//!
//! Supports:
//! - Parallel execution via `tokio::JoinSet`
//! - Conditional branching (`when` expressions)
//! - Trigger rules (AllSuccess / OneSuccess / NoneFailed)
//! - Template variable interpolation
//! - Timeout control per node

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use crate::cancel::CancelToken;
use crate::openai::{
    ChatCompletionsRequest, ChatMessage, MessageContent, OpenAiClient, ToolCall, ToolDefinition,
};
use crate::tools::{ToolExecutionResult, ToolExecutor, ToolRuntime, ToolSession};
use crate::workflow::{
    ContextMode, NodeRun, NodeStatus, NodeType, RetryPolicy, TriggerRule, WorkflowDef,
    WorkflowNodeDef, WorkflowRun, WorkflowStatus, render_template,
};

/// Callback channel for reporting workflow progress to the caller (e.g. WebSocket).
pub type ProgressSender = mpsc::UnboundedSender<WorkflowEvent>;

/// Events emitted by the workflow engine during execution.
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// Workflow has started.
    Started { run_id: Uuid, workflow_name: String },
    /// A node's status changed.
    NodeUpdate {
        run_id: Uuid,
        node_id: String,
        status: NodeStatus,
        output: Option<String>,
    },
    /// An approval node is waiting for a response.
    ApprovalRequested {
        run_id: Uuid,
        node_id: String,
        prompt: String,
    },
    /// Workflow completed successfully.
    Completed { run_id: Uuid, summary: String },
    /// Workflow failed.
    Failed {
        run_id: Uuid,
        node_id: Option<String>,
        error: String,
    },
}

/// Execution context shared across all nodes in a workflow run.
struct WorkflowContext {
    /// Accumulated outputs from completed nodes (node_id -> output).
    outputs: RwLock<HashMap<String, String>>,
    /// User-provided inputs.
    inputs: HashMap<String, String>,
    /// Pending approval senders (node_id -> oneshot::Sender).
    approval_pending: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}

/// Approval response sent from the frontend.
#[derive(Debug, Clone)]
pub struct ApprovalResponse {
    pub run_id: Uuid,
    pub node_id: String,
    pub approved: bool,
    pub feedback: Option<String>,
}

/// Maximum time a single node can run before being cancelled.
const NODE_TIMEOUT_SECS: u64 = 300; // 5 minutes
/// LLM configuration for workflow prompt nodes.
#[derive(Debug, Clone)]
pub struct WorkflowLlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub default_model: String,
}

/// Run a workflow definition to completion.
///
/// `workspace_root` is used for resolving relative paths in Bash nodes.
/// `progress_tx` receives real-time status updates.
/// `llm_config` provides LLM API settings for prompt nodes.
/// `approval_rx` receives approval responses from the frontend.
pub async fn run_workflow(
    def: WorkflowDef,
    inputs: HashMap<String, String>,
    workspace_root: PathBuf,
    progress_tx: ProgressSender,
    llm_config: Option<WorkflowLlmConfig>,
    mut approval_rx: mpsc::UnboundedReceiver<ApprovalResponse>,
    tool_resources: Option<(ToolExecutor, ToolRuntime, CancelToken)>,
) -> WorkflowRun {
    // If workflow has no default model, inherit from llm_config.
    let mut def = def;
    if def.default_model.is_none() {
        if let Some(ref cfg) = llm_config {
            def.default_model = Some(cfg.default_model.clone());
        }
    }

    let run_id = Uuid::new_v4();
    let now = Utc::now();
    let node_runs: Vec<NodeRun> = def
        .nodes
        .iter()
        .map(|n| NodeRun {
            node_id: n.id.clone(),
            status: if n.depends_on.is_empty() {
                NodeStatus::Waiting
            } else {
                NodeStatus::Pending
            },
            output: None,
            started_at: None,
            completed_at: None,
            error: None,
        })
        .collect();

    let mut run = WorkflowRun {
        run_id,
        workflow_name: def.name.clone(),
        status: WorkflowStatus::Running,
        nodes: node_runs,
        inputs: inputs.clone(),
        created_at: now,
        completed_at: None,
    };

    let _ = progress_tx.send(WorkflowEvent::Started {
        run_id,
        workflow_name: def.name.clone(),
    });

    let ctx = Arc::new(WorkflowContext {
        outputs: RwLock::new(HashMap::new()),
        inputs,
        approval_pending: Mutex::new(HashMap::new()),
    });

    // Spawn a task to forward approval responses to waiting nodes.
    let ctx_for_approval = Arc::clone(&ctx);
    let progress_tx_for_approval = progress_tx.clone();
    let approval_forwarder = tokio::spawn(async move {
        while let Some(resp) = approval_rx.recv().await {
            let mut pending = ctx_for_approval.approval_pending.lock().await;
            if let Some(sender) = pending.remove(&resp.node_id) {
                let _ = sender.send(resp.approved);
            }
        }
    });

    let run_state = Arc::new(Mutex::new(run.clone()));
    let node_map: HashMap<String, &WorkflowNodeDef> =
        def.nodes.iter().map(|n| (n.id.clone(), n)).collect();

    // Topological execution loop
    loop {
        // Find nodes that are ready to execute
        let ready_ids = {
            let state = run_state.lock().await;
            find_ready_nodes(&state, &node_map, &ctx).await
        };

        if ready_ids.is_empty() {
            // Check if all nodes are done
            let state = run_state.lock().await;
            let all_done = state.nodes.iter().all(|n| {
                matches!(
                    n.status,
                    NodeStatus::Completed | NodeStatus::Failed | NodeStatus::Skipped
                )
            });
            if all_done {
                break;
            }
            // If no nodes are ready but not all are done, we have a deadlock or
            // all remaining nodes have unmet dependencies due to failures.
            // Mark them as skipped.
            let mut state = run_state.lock().await;
            for node in state.nodes.iter_mut() {
                if node.status == NodeStatus::Pending || node.status == NodeStatus::Waiting {
                    node.status = NodeStatus::Skipped;
                    node.completed_at = Some(Utc::now());
                    let _ = progress_tx.send(WorkflowEvent::NodeUpdate {
                        run_id,
                        node_id: node.node_id.clone(),
                        status: NodeStatus::Skipped,
                        output: None,
                    });
                }
            }
            break;
        }

        // Execute ready nodes in parallel
        let mut join_set = tokio::task::JoinSet::new();
        for node_id in &ready_ids {
            let node_def = node_map[node_id].clone();
            let ctx = Arc::clone(&ctx);
            let ws_root = workspace_root.clone();
            let node_id_clone = node_id.clone();

            // Mark as running
            {
                let mut state = run_state.lock().await;
                if let Some(nr) = state.nodes.iter_mut().find(|n| n.node_id == *node_id) {
                    nr.status = NodeStatus::Running;
                    nr.started_at = Some(Utc::now());
                }
            }
            let _ = progress_tx.send(WorkflowEvent::NodeUpdate {
                run_id,
                node_id: node_id_clone.clone(),
                status: NodeStatus::Running,
                output: None,
            });

            let llm = llm_config.clone();
            let default_model = def.default_model.clone();
            let progress_tx_clone = progress_tx.clone();
            let tools = tool_resources.clone();
            join_set.spawn(execute_node(
                node_def,
                ctx,
                ws_root,
                llm,
                default_model,
                run_id,
                progress_tx_clone,
                tools,
            ));
        }

        // Wait for all nodes in this batch to complete
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((node_id, status, output, error)) => {
                    let mut state = run_state.lock().await;
                    if let Some(nr) = state.nodes.iter_mut().find(|n| n.node_id == node_id) {
                        nr.status = status;
                        nr.completed_at = Some(Utc::now());
                        nr.output = output.clone();
                        nr.error = error.clone();
                    }
                    // Store output for downstream nodes
                    if let Some(ref out) = output {
                        ctx.outputs
                            .write()
                            .await
                            .insert(node_id.clone(), out.clone());
                    }
                    drop(state);

                    let _ = progress_tx.send(WorkflowEvent::NodeUpdate {
                        run_id,
                        node_id: node_id.clone(),
                        status,
                        output,
                    });
                }
                Err(join_err) => {
                    tracing::error!("Node task panicked: {join_err}");
                }
            }
        }

        // Persist workflow state after each batch.
        {
            let state = run_state.lock().await;
            let _ = save_workflow_state(&workspace_root, &state).await;
        }
    }

    // Determine final status
    let mut final_run = run_state.lock().await;
    let has_failure = final_run
        .nodes
        .iter()
        .any(|n| n.status == NodeStatus::Failed);
    let all_completed = final_run
        .nodes
        .iter()
        .all(|n| matches!(n.status, NodeStatus::Completed | NodeStatus::Skipped));

    final_run.status = if has_failure {
        WorkflowStatus::Failed
    } else if all_completed {
        WorkflowStatus::Completed
    } else {
        WorkflowStatus::Cancelled
    };
    final_run.completed_at = Some(Utc::now());

    let result = final_run.clone();

    match result.status {
        WorkflowStatus::Completed => {
            // Collect all node outputs into the summary.
            let outputs = ctx.outputs.read().await;
            let mut parts: Vec<String> = Vec::new();
            parts.push(format!(
                "✅ 工作流 '{}' 执行完成，共 {} 个节点。",
                result.workflow_name,
                result.nodes.len()
            ));
            parts.push(String::new());
            for node in &result.nodes {
                if let Some(ref out) = outputs.get(&node.node_id) {
                    parts.push(format!("### {}", node.node_id));
                    parts.push(out.trim().to_string());
                    parts.push(String::new());
                }
            }
            let summary = parts.join("\n");
            drop(outputs);
            let _ = progress_tx.send(WorkflowEvent::Completed { run_id, summary });
        }
        WorkflowStatus::Failed => {
            let failed_node = result.nodes.iter().find(|n| n.status == NodeStatus::Failed);
            let _ = progress_tx.send(WorkflowEvent::Failed {
                run_id,
                node_id: failed_node.map(|n| n.node_id.clone()),
                error: failed_node
                    .and_then(|n| n.error.clone())
                    .unwrap_or_else(|| "Unknown error".to_string()),
            });
        }
        _ => {}
    }

    // Clean up the approval forwarder task.
    approval_forwarder.abort();

    result
}

/// Cancel a running workflow by marking all pending/waiting/running nodes as skipped.
pub async fn cancel_workflow(run: &mut WorkflowRun) {
    run.status = WorkflowStatus::Cancelled;
    run.completed_at = Some(Utc::now());
    for node in run.nodes.iter_mut() {
        if matches!(
            node.status,
            NodeStatus::Pending | NodeStatus::Waiting | NodeStatus::Running
        ) {
            node.status = NodeStatus::Skipped;
            node.completed_at = Some(Utc::now());
        }
    }
}

/// Find node IDs that are ready to execute.
async fn find_ready_nodes(
    run: &WorkflowRun,
    node_map: &HashMap<String, &WorkflowNodeDef>,
    ctx: &WorkflowContext,
) -> Vec<String> {
    let outputs = ctx.outputs.read().await;
    let mut ready = Vec::new();

    for nr in &run.nodes {
        if nr.status != NodeStatus::Pending && nr.status != NodeStatus::Waiting {
            continue;
        }

        let def = match node_map.get(&nr.node_id) {
            Some(d) => d,
            None => continue,
        };

        // Check trigger rule
        let deps_met = check_trigger_rule(def, run, &outputs);
        if !deps_met {
            continue;
        }

        // Check `when` condition
        if let Some(ref when_expr) = def.when {
            if !evaluate_when(when_expr, &outputs) {
                // Skip this node
                continue;
            }
        }

        ready.push(nr.node_id.clone());
    }

    ready
}

/// Check if a node's trigger rule is satisfied.
fn check_trigger_rule(
    def: &WorkflowNodeDef,
    run: &WorkflowRun,
    outputs: &HashMap<String, String>,
) -> bool {
    if def.depends_on.is_empty() {
        return true; // No dependencies
    }

    let dep_statuses: Vec<NodeStatus> = def
        .depends_on
        .iter()
        .filter_map(|dep_id| run.nodes.iter().find(|n| &n.node_id == dep_id))
        .map(|n| n.status)
        .collect();

    // All dependencies must be in a terminal state
    let all_terminal = dep_statuses.iter().all(|s| {
        matches!(
            s,
            NodeStatus::Completed | NodeStatus::Failed | NodeStatus::Skipped
        )
    });

    if !all_terminal {
        return false;
    }

    match def.trigger_rule {
        TriggerRule::AllSuccess => dep_statuses.iter().all(|s| *s == NodeStatus::Completed),
        TriggerRule::OneSuccess => dep_statuses.iter().any(|s| *s == NodeStatus::Completed),
        TriggerRule::NoneFailed => dep_statuses.iter().all(|s| *s != NodeStatus::Failed),
    }
}

/// When-expression evaluator with support for composite conditions.
///
/// Supported syntax:
/// - `"true"` / `"false"` — literal booleans
/// - `"output(node_id)"` — check non-empty output
/// - `"output(node_id) == 'value'"` / `"!= 'value'"` — equality comparison
/// - `"output(node_id) contains 'value'"` — substring check
/// - `"output(node_id) not contains 'value'"` — negated substring check
/// - `"output(node_id) len > N"` — length comparison
/// - `"expr1 and expr2"` — logical AND
/// - `"expr1 or expr2"` — logical OR
fn evaluate_when(expr: &str, outputs: &HashMap<String, String>) -> bool {
    let expr = expr.trim();

    // Handle top-level `and` / `or` (split on last occurrence to handle nesting).
    // We use a simple approach: split on ` and ` / ` or ` at the top level.
    if let Some((left, right)) = split_top_level_binary(expr, " and ") {
        return evaluate_when(left, outputs) && evaluate_when(right, outputs);
    }
    if let Some((left, right)) = split_top_level_binary(expr, " or ") {
        return evaluate_when(left, outputs) || evaluate_when(right, outputs);
    }

    if expr == "true" {
        return true;
    }
    if expr == "false" {
        return false;
    }

    // Parse comparison expressions: `output(id) <op> value`
    for op in &["!=", "=="] {
        if let Some(idx) = expr.find(op) {
            let left = expr[..idx].trim();
            let right = expr[idx + op.len()..].trim();

            // Ensure left side looks like an output() reference to avoid false matches.
            if left.starts_with("output(") || left.starts_with('"') || left.starts_with('\'') {
                let left_val = resolve_expr_value(left, outputs);
                let right_val = resolve_expr_value(right, outputs);
                return match *op {
                    "==" => left_val == right_val,
                    "!=" => left_val != right_val,
                    _ => false,
                };
            }
        }
    }

    // Check `contains` / `not contains`
    if let Some(rest) = expr
        .strip_prefix("output(")
        .and_then(|s| s.strip_suffix(')'))
    {
        // Direct output existence check: `output(node_id)`.
        // Also handle `output(node_id) contains 'value'` etc.
        // But the simple `output(node_id)` case — just check non-empty.
        if !rest.contains(" contains ") && !rest.contains(" not contains ") {
            return outputs.contains_key(rest) && !outputs[rest].is_empty();
        }
    }

    // Handle `output(node_id) contains 'value'`
    if let Some(caps) = parse_output_comparison(expr) {
        let (node_id, op, value) = caps;
        let actual = outputs.get(&node_id).cloned().unwrap_or_default();
        return match op.as_str() {
            "contains" => actual.contains(&value),
            "not contains" => !actual.contains(&value),
            _ => false,
        };
    }

    // Handle `output(node_id) len > N` etc.
    if expr.contains(" len ") {
        for cmp_op in &[">=", "<=", ">", "<"] {
            if let Some(idx) = expr.find(&format!(" len {cmp_op} ")) {
                let left_part = expr[..idx].trim();
                let right_part = expr[idx + &format!(" len {cmp_op} ").len()..].trim();
                if let Some(node_id) = left_part
                    .strip_prefix("output(")
                    .and_then(|s| s.strip_suffix(')'))
                {
                    let actual_len = outputs.get(node_id).map(|s| s.len()).unwrap_or(0);
                    if let Ok(n) = right_part.parse::<usize>() {
                        return match *cmp_op {
                            ">" => actual_len > n,
                            "<" => actual_len < n,
                            ">=" => actual_len >= n,
                            "<=" => actual_len <= n,
                            _ => false,
                        };
                    }
                }
            }
        }
    }

    // Unknown expression — default to true (permissive).
    true
}

/// Split an expression on a top-level binary operator (not nested inside quotes).
fn split_top_level_binary<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    // Find the last occurrence of `op` at depth 0 (outside quotes).
    let mut in_quote = false;
    let mut quote_char = '\0';
    let chars: Vec<(usize, char)> = expr.char_indices().collect();

    for (char_index, (byte_index, c)) in chars.iter().enumerate().rev() {
        if in_quote {
            if *c == quote_char {
                in_quote = false;
            }
            continue;
        }
        if *c == '\'' || *c == '"' {
            in_quote = true;
            quote_char = *c;
            continue;
        }

        let remaining_chars = chars.len().saturating_sub(char_index);
        if remaining_chars < op.chars().count() {
            continue;
        }
        let tail = &expr[*byte_index..];
        if tail.starts_with(op) {
            let left = expr[..*byte_index].trim();
            let right = expr[*byte_index + op.len()..].trim();
            if !left.is_empty() && !right.is_empty() {
                return Some((left, right));
            }
        }
    }
    None
}

/// Parse `output(node_id) contains 'value'` or `not contains`.
fn parse_output_comparison(expr: &str) -> Option<(String, String, String)> {
    for op in &[" not contains ", " contains "] {
        if let Some(idx) = expr.find(*op) {
            let left = expr[..idx].trim();
            let right = expr[idx + op.len()..].trim();
            if let Some(node_id) = left
                .strip_prefix("output(")
                .and_then(|s| s.strip_suffix(')'))
            {
                let value = right.trim_matches('\'').trim_matches('"').to_string();
                let op_str = op.trim().to_string();
                return Some((node_id.to_string(), op_str, value));
            }
        }
    }
    None
}

/// Resolve a value from an expression (either a quoted literal or `output(id)`).
fn resolve_expr_value(expr: &str, outputs: &HashMap<String, String>) -> String {
    let expr = expr.trim();

    // Quoted string literal
    if (expr.starts_with('\'') && expr.ends_with('\''))
        || (expr.starts_with('"') && expr.ends_with('"'))
    {
        return expr[1..expr.len() - 1].to_string();
    }

    // output(node_id) reference
    if let Some(id) = expr
        .strip_prefix("output(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return outputs.get(id).cloned().unwrap_or_default();
    }

    expr.to_string()
}

/// Execute a single workflow node.
///
/// Returns `(node_id, status, output, error)`.
async fn execute_node(
    def: WorkflowNodeDef,
    ctx: Arc<WorkflowContext>,
    workspace_root: PathBuf,
    llm_config: Option<WorkflowLlmConfig>,
    default_model: Option<String>,
    run_id: Uuid,
    progress_tx: ProgressSender,
    tool_resources: Option<(ToolExecutor, ToolRuntime, CancelToken)>,
) -> (String, NodeStatus, Option<String>, Option<String>) {
    let node_id = def.id.clone();

    // Build context string if shared mode
    let context_str = if def.context == ContextMode::Shared {
        let outputs = ctx.outputs.read().await;
        let mut parts: Vec<String> = Vec::new();
        for dep_id in &def.depends_on {
            if let Some(out) = outputs.get(dep_id) {
                parts.push(format!("--- Output from {dep_id} ---\n{out}"));
            }
        }
        if parts.is_empty() {
            String::new()
        } else {
            parts.join("\n\n")
        }
    } else {
        String::new()
    };

    // Build variable map for template rendering
    let mut vars = ctx.inputs.clone();
    {
        let outputs = ctx.outputs.read().await;
        for (k, v) in outputs.iter() {
            vars.insert(format!("output_{k}"), v.clone());
        }
    }

    // Retry wrapper: execute the node body, retry on failure if policy allows.
    let retry_policy = def.retry.clone().unwrap_or(RetryPolicy {
        max_attempts: 1,
        backoff_ms: 1000,
    });

    let mut last_result: Option<(NodeStatus, Option<String>, Option<String>)> = None;

    for attempt in 1..=retry_policy.max_attempts {
        tracing::info!(
            "Node '{}' attempt {}/{} — type={:?}",
            node_id,
            attempt,
            retry_policy.max_attempts,
            def.node_type
        );

        let result = execute_node_body(
            &def,
            &ctx,
            &workspace_root,
            &llm_config,
            &default_model,
            &context_str,
            &vars,
            run_id,
            &progress_tx,
            &tool_resources,
        )
        .await;

        match result {
            Ok(triple @ (NodeStatus::Completed, _, _)) => {
                last_result = Some(triple);
                break;
            }
            Ok((NodeStatus::Failed, output, error)) => {
                tracing::warn!("Node '{}' failed: {:?}", node_id, error);
                if attempt < retry_policy.max_attempts {
                    let delay = retry_policy.backoff_ms * 2u64.pow(attempt - 1);
                    let _ = progress_tx.send(WorkflowEvent::NodeUpdate {
                        run_id,
                        node_id: node_id.clone(),
                        status: NodeStatus::Running,
                        output: Some(format!(
                            "Attempt {attempt}/{} failed, retrying in {delay}ms...",
                            retry_policy.max_attempts
                        )),
                    });
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    last_result = Some((NodeStatus::Failed, output, error));
                    continue;
                }
                last_result = Some((NodeStatus::Failed, output, error));
                break;
            }
            Ok(triple) => {
                // Skipped or other status
                last_result = Some(triple);
                break;
            }
            Err(e) => {
                // Node execution error (e.g. program not found, timeout)
                tracing::error!("Node '{}' execution error: {e}", node_id);
                last_result = Some((NodeStatus::Failed, None, Some(format!("{e}"))));
                break;
            }
        }
    }

    let (status, output, error) = last_result.unwrap_or((
        NodeStatus::Failed,
        None,
        Some("No execution attempt made".to_string()),
    ));
    (node_id, status, output, error)
}

/// Inner node execution logic (single attempt).
async fn execute_node_body(
    def: &WorkflowNodeDef,
    ctx: &Arc<WorkflowContext>,
    workspace_root: &PathBuf,
    llm_config: &Option<WorkflowLlmConfig>,
    default_model: &Option<String>,
    context_str: &str,
    vars: &HashMap<String, String>,
    run_id: Uuid,
    progress_tx: &ProgressSender,
    tool_resources: &Option<(ToolExecutor, ToolRuntime, CancelToken)>,
) -> Result<(NodeStatus, Option<String>, Option<String>), anyhow::Error> {
    match def.node_type {
        NodeType::Prompt => {
            let prompt = def.prompt.as_deref().unwrap_or("No prompt specified");
            let rendered = render_template(prompt, vars);
            let full_prompt = if context_str.is_empty() {
                rendered
            } else {
                format!("{context_str}\n\n{rendered}")
            };

            let model = def
                .model
                .as_deref()
                .or(default_model.as_deref())
                .unwrap_or("default");
            tracing::info!("Prompt node '{}' using model={}", def.id, model);
            let tool_ref = tool_resources.as_ref().map(|(e, r, c)| (e, r, c));
            execute_prompt_node(&full_prompt, model, llm_config.as_ref(), tool_ref).await
        }

        NodeType::Loop => {
            let prompt = def.prompt.as_deref().unwrap_or("No prompt specified");
            let rendered = render_template(prompt, vars);
            let full_prompt = if context_str.is_empty() {
                rendered
            } else {
                format!("{context_str}\n\n{rendered}")
            };

            let model = def
                .model
                .as_deref()
                .or(default_model.as_deref())
                .unwrap_or("sensenova-6.7-flash-lite");

            let max_iter = def
                .loop_condition
                .as_ref()
                .map(|lc| lc.max_iterations)
                .unwrap_or(5);
            let condition = def.loop_condition.as_ref().map(|lc| lc.condition.clone());

            execute_loop_node(
                &full_prompt,
                max_iter,
                condition.as_deref(),
                model,
                llm_config.as_ref(),
            )
            .await
        }

        NodeType::Bash => {
            let command = def
                .command
                .as_deref()
                .unwrap_or("echo 'No command specified'");
            let rendered = render_template(command, vars);
            tracing::info!("Bash node '{}' executing: {}", def.id, rendered);
            execute_bash_node(&rendered, workspace_root).await
        }

        NodeType::Approval => {
            let prompt = def.prompt.as_deref().unwrap_or("Please review and approve");
            let rendered = render_template(prompt, vars);

            // Create a oneshot channel for this approval.
            let (tx, rx) = oneshot::channel::<bool>();

            // Register the sender so the approval response forwarder can find it.
            {
                let mut pending = ctx.approval_pending.lock().await;
                pending.insert(def.id.clone(), tx);
            }

            // Notify frontend that approval is needed.
            let _ = progress_tx.send(WorkflowEvent::ApprovalRequested {
                run_id,
                node_id: def.id.clone(),
                prompt: rendered.clone(),
            });

            // Wait for the approval response (with timeout).
            let approval_timeout = Duration::from_secs(600); // 10 minutes
            match tokio::time::timeout(approval_timeout, rx).await {
                Ok(Ok(approved)) => {
                    // Clean up the pending entry.
                    let mut pending = ctx.approval_pending.lock().await;
                    pending.remove(&def.id);

                    if approved {
                        Ok((
                            NodeStatus::Completed,
                            Some(format!("Approved: {rendered}")),
                            None,
                        ))
                    } else {
                        Ok((
                            NodeStatus::Failed,
                            Some(format!("Rejected: {rendered}")),
                            Some("Approval rejected by user".to_string()),
                        ))
                    }
                }
                Ok(Err(_)) => {
                    // Channel dropped (sender was dropped).
                    let mut pending = ctx.approval_pending.lock().await;
                    pending.remove(&def.id);
                    Ok((
                        NodeStatus::Failed,
                        None,
                        Some("Approval channel closed unexpectedly".to_string()),
                    ))
                }
                Err(_) => {
                    // Timeout.
                    let mut pending = ctx.approval_pending.lock().await;
                    pending.remove(&def.id);
                    Ok((
                        NodeStatus::Failed,
                        None,
                        Some("Approval timed out after 10 minutes".to_string()),
                    ))
                }
            }
        }
    }
}

/// Execute a prompt node via LLM API call, with optional tool-calling loop.
/// Execute a prompt node via LLM API call, with optional tool-calling loop.
async fn execute_prompt_node(
    prompt: &str,
    model: &str,
    llm_config: Option<&WorkflowLlmConfig>,
    tool_resources: Option<(&ToolExecutor, &ToolRuntime, &CancelToken)>,
) -> Result<(NodeStatus, Option<String>, Option<String>), anyhow::Error> {
    let config = match llm_config {
        Some(c) => c,
        None => {
            let output = format!("[No LLM config — prompt: {}]", truncate(prompt, 200));
            return Ok((NodeStatus::Completed, Some(output), None));
        }
    };

    let client = OpenAiClient::new(config.base_url.clone(), config.api_key.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create LLM client: {e}"))?;

    // Get tool definitions if available.
    let tool_defs: Option<Vec<ToolDefinition>> = tool_resources
        .as_ref()
        .map(|(executor, runtime, _)| executor.tool_definitions(runtime));

    let system_prompt = "你是一个专业的 AI 助手，具备文件读取、命令执行等工具能力。\
        请根据用户的指令完成任务。善用工具获取实际信息，然后基于真实数据给出结果。\
        完成后直接输出最终结果。";

    let mut messages = vec![
        ChatMessage::text("system", system_prompt),
        ChatMessage::text("user", prompt),
    ];

    // Tool-calling loop (max 10 iterations to prevent runaway).
    const MAX_TOOL_ROUNDS: usize = 10;
    let mut tool_session = ToolSession::default();

    for round in 0..MAX_TOOL_ROUNDS {
        let req = ChatCompletionsRequest {
            model: model.to_string(),
            messages: messages.clone(),
            max_tokens: Some(4096),
            reasoning_effort: None,
            tools: tool_defs.clone(),
            tool_choice: if tool_defs.is_some() {
                Some(serde_json::json!("auto"))
            } else {
                None
            },
            stream: Some(false),
            temperature: Some(0.3),
            top_p: None,
        };

        let result = timeout(
            Duration::from_secs(NODE_TIMEOUT_SECS),
            client.chat_completions(&req),
        )
        .await;

        let resp = match result {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => return Err(anyhow::anyhow!("LLM API error: {e}")),
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "LLM request timed out after {NODE_TIMEOUT_SECS}s"
                ));
            }
        };

        let choice = match resp.choices.first() {
            Some(c) => c,
            None => return Err(anyhow::anyhow!("LLM returned no choices")),
        };

        let assistant_msg = choice.message.clone();
        let tool_calls: Vec<ToolCall> = assistant_msg.tool_calls.clone().unwrap_or_default();

        // If no tool calls, we're done — return the text response.
        if tool_calls.is_empty() {
            let output = assistant_msg
                .content
                .as_ref()
                .and_then(|c| match c {
                    MessageContent::Text(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "(empty response)".to_string());
            return Ok((NodeStatus::Completed, Some(output), None));
        }

        // Has tool calls — execute them.
        tracing::info!(
            "Prompt node tool round {}: {} tool call(s)",
            round + 1,
            tool_calls.len()
        );
        messages.push(assistant_msg);

        let (executor, runtime, cancel) = match tool_resources {
            Some(r) => r,
            None => {
                // No tool executor but LLM requested tools — return what we have.
                let text = messages
                    .last()
                    .and_then(|m| m.content.as_ref())
                    .and_then(|c| match c {
                        MessageContent::Text(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                return Ok((NodeStatus::Completed, Some(text), None));
            }
        };

        for call in &tool_calls {
            if cancel.is_cancelled() {
                return Ok((NodeStatus::Failed, None, Some("Cancelled".to_string())));
            }

            let args_json: serde_json::Value = serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));

            tracing::info!(
                "  Tool: {}({})",
                call.function.name,
                truncate(&call.function.arguments, 100)
            );

            let exec_result = executor
                .execute(
                    &mut tool_session,
                    runtime,
                    &call.function.name,
                    args_json,
                    cancel,
                )
                .await;

            let output = match exec_result {
                Ok(ToolExecutionResult::Observation(text)) => text,
                Ok(ToolExecutionResult::ImagePayload { .. }) => "[Image result]".to_string(),
                Ok(ToolExecutionResult::Control(_)) => "[Control signal — skipped]".to_string(),
                Err(e) => format!("Tool error: {e}"),
            };

            tracing::info!("  Result: {}", truncate(&output, 200));
            messages.push(ChatMessage::tool_result(call.id.clone(), output));
        }
    }

    // Exhausted all rounds — extract whatever text we have.
    let final_text = messages
        .iter()
        .rev()
        .find_map(|m| {
            if m.role == "assistant" {
                m.content.as_ref().and_then(|c| match c {
                    MessageContent::Text(s) if !s.is_empty() => Some(s.clone()),
                    _ => None,
                })
            } else {
                None
            }
        })
        .unwrap_or_else(|| "(max tool rounds exceeded)".to_string());
    Ok((NodeStatus::Completed, Some(final_text), None))
}

/// Execute a bash node with timeout.
/// Decode command output bytes to UTF-8 string.
/// On Windows, cmd.exe outputs in the system codepage (e.g. GBK/CP936 for Chinese).
/// We first try UTF-8; if it contains replacement characters, fall back to GBK.
fn decode_output(bytes: &[u8]) -> String {
    // Try UTF-8 first.
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            // UTF-8 failed, try GBK (common on Chinese Windows).
            let (decoded, _, had_errors) = encoding_rs::GBK.decode(bytes);
            if had_errors {
                // GBK also had issues, fall back to lossy UTF-8.
                String::from_utf8_lossy(bytes).to_string()
            } else {
                decoded.into_owned()
            }
        }
    }
}

async fn execute_bash_node(
    command: &str,
    workspace_root: &PathBuf,
) -> Result<(NodeStatus, Option<String>, Option<String>), anyhow::Error> {
    // Platform-aware shell selection: use cmd on Windows, sh on Unix.
    let result = if cfg!(target_os = "windows") {
        timeout(
            Duration::from_secs(NODE_TIMEOUT_SECS),
            tokio::process::Command::new("cmd")
                .arg("/C")
                .arg(command)
                .current_dir(workspace_root)
                .output(),
        )
        .await
    } else {
        timeout(
            Duration::from_secs(NODE_TIMEOUT_SECS),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(workspace_root)
                .output(),
        )
        .await
    };

    match result {
        Ok(Ok(output)) => {
            let stdout = decode_output(&output.stdout);
            let stderr = decode_output(&output.stderr);

            if output.status.success() {
                Ok((NodeStatus::Completed, Some(stdout), None))
            } else {
                Ok((NodeStatus::Failed, Some(stdout), Some(stderr)))
            }
        }
        Ok(Err(e)) => Err(anyhow::anyhow!("Failed to execute command: {e}")),
        Err(_) => Err(anyhow::anyhow!(
            "Command timed out after {NODE_TIMEOUT_SECS}s"
        )),
    }
}

/// Execute a loop node — repeat prompt until condition or max iterations.
///
/// If a condition expression is provided, it is evaluated after each iteration.
/// The loop exits when the condition becomes true or max_iterations is reached.
async fn execute_loop_node(
    prompt: &str,
    max_iterations: usize,
    condition: Option<&str>,
    model: &str,
    llm_config: Option<&WorkflowLlmConfig>,
) -> Result<(NodeStatus, Option<String>, Option<String>), anyhow::Error> {
    let mut last_output = String::new();
    for i in 0..max_iterations {
        let iteration_prompt = format!(
            "(Iteration {}/{}) {}\n\nPrevious output:\n{}",
            i + 1,
            max_iterations,
            prompt,
            last_output
        );
        let result = execute_prompt_node(&iteration_prompt, model, llm_config, None).await?;
        match result {
            (NodeStatus::Completed, Some(output), _) => {
                last_output = output;

                // Evaluate the loop condition if provided.
                if let Some(cond) = condition {
                    let mut output_map = HashMap::new();
                    output_map.insert("loop_output".to_string(), last_output.clone());
                    if evaluate_loop_condition(cond, &last_output) {
                        break;
                    }
                    // Continue looping if condition not met.
                } else {
                    // No condition — run once and exit.
                    break;
                }
            }
            (NodeStatus::Failed, _, Some(err)) => {
                return Ok((NodeStatus::Failed, Some(last_output), Some(err)));
            }
            _ => {}
        }
    }
    Ok((NodeStatus::Completed, Some(last_output), None))
}

/// Evaluate a loop condition against the current output.
///
/// Supported expressions:
/// - `"output not contains ERROR"` — output does not contain a substring
/// - `"output contains DONE"` — output contains a substring
/// - `"output len > 100"` — output length comparison
/// - `"true"` / `"false"` — literal booleans
fn evaluate_loop_condition(condition: &str, output: &str) -> bool {
    let cond = condition.trim();

    if cond == "true" {
        return true;
    }
    if cond == "false" {
        return false;
    }

    // "output contains <substring>"
    if let Some(sub) = cond.strip_prefix("output contains ") {
        let sub = sub.trim().trim_matches('\'').trim_matches('"');
        return output.contains(sub);
    }

    // "output not contains <substring>"
    if let Some(sub) = cond.strip_prefix("output not contains ") {
        let sub = sub.trim().trim_matches('\'').trim_matches('"');
        return !output.contains(sub);
    }

    // "output len > N" / "output len < N" / "output len >= N" / "output len <= N"
    for op in &[">=", "<=", ">", "<"] {
        if let Some(rest) = cond.strip_prefix(&format!("output len {op} ")) {
            if let Ok(n) = rest.trim().parse::<usize>() {
                return match *op {
                    ">" => output.len() > n,
                    "<" => output.len() < n,
                    ">=" => output.len() >= n,
                    "<=" => output.len() <= n,
                    _ => false,
                };
            }
        }
    }

    // Unknown condition — default to false (stop looping for safety).
    false
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len { s } else { &s[..max_len] }
}

/// Load a workflow definition from a YAML file.
pub async fn load_workflow_def(path: &PathBuf) -> Result<WorkflowDef, anyhow::Error> {
    let content = tokio::fs::read_to_string(path).await?;
    let def: WorkflowDef = serde_yaml::from_str(&content)?;
    Ok(def)
}

/// Discover all workflow YAML files in a directory.
pub async fn discover_workflows(dir: &PathBuf) -> Vec<(String, PathBuf)> {
    let mut workflows = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                || path.extension().and_then(|e| e.to_str()) == Some("yml")
            {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    workflows.push((stem.to_string(), path));
                }
            }
        }
    }
    workflows
}

/// Save workflow run state to disk for crash recovery.
async fn save_workflow_state(workspace_root: &PathBuf, run: &WorkflowRun) -> std::io::Result<()> {
    let dir = workspace_root.join("runtime").join("workflows");
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{}.json", run.run_id));
    let json = serde_json::to_string_pretty(run)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    tokio::fs::write(&path, json).await
}

/// Scan for incomplete workflow runs and return their paths for recovery.
pub async fn discover_incomplete_workflows(
    workspace_root: &PathBuf,
) -> Result<Vec<(WorkflowRun, PathBuf)>, anyhow::Error> {
    let dir = workspace_root.join("runtime").join("workflows");
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut incomplete = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    if let Ok(run) = serde_json::from_str::<WorkflowRun>(&content) {
                        if matches!(
                            run.status,
                            WorkflowStatus::Running | WorkflowStatus::Pending
                        ) {
                            incomplete.push((run, path));
                        }
                    }
                }
            }
        }
    }
    Ok(incomplete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_evaluate_when_literal_true() {
        assert!(evaluate_when("true", &HashMap::new()));
    }

    #[test]
    fn test_evaluate_when_literal_false() {
        assert!(!evaluate_when("false", &HashMap::new()));
    }

    #[test]
    fn test_evaluate_when_output_exists() {
        let mut outputs = HashMap::new();
        outputs.insert("step1".to_string(), "result".to_string());
        assert!(evaluate_when("output(step1)", &outputs));
    }

    #[test]
    fn test_evaluate_when_output_missing() {
        assert!(!evaluate_when("output(step1)", &HashMap::new()));
    }

    #[test]
    fn test_evaluate_when_comparison() {
        let mut outputs = HashMap::new();
        outputs.insert("step1".to_string(), "hello".to_string());
        assert!(evaluate_when("output(step1) == 'hello'", &outputs));
        assert!(!evaluate_when("output(step1) == 'world'", &outputs));
        assert!(evaluate_when("output(step1) != 'world'", &outputs));
    }

    #[test]
    fn test_evaluate_when_contains() {
        let mut outputs = HashMap::new();
        outputs.insert(
            "step1".to_string(),
            "ERROR: something went wrong".to_string(),
        );
        assert!(evaluate_when("output(step1) contains 'ERROR'", &outputs));
        assert!(!evaluate_when("output(step1) contains 'SUCCESS'", &outputs));
        assert!(evaluate_when(
            "output(step1) not contains 'SUCCESS'",
            &outputs
        ));
    }

    #[test]
    fn test_evaluate_when_len() {
        let mut outputs = HashMap::new();
        outputs.insert("step1".to_string(), "hello world".to_string()); // len = 11
        assert!(evaluate_when("output(step1) len > 5", &outputs));
        assert!(!evaluate_when("output(step1) len > 20", &outputs));
        assert!(evaluate_when("output(step1) len <= 11", &outputs));
    }

    #[test]
    fn test_evaluate_when_and() {
        let mut outputs = HashMap::new();
        outputs.insert("a".to_string(), "hello".to_string());
        outputs.insert("b".to_string(), "world".to_string());
        assert!(evaluate_when(
            "output(a) == 'hello' and output(b) == 'world'",
            &outputs
        ));
        assert!(!evaluate_when(
            "output(a) == 'hello' and output(b) == 'nope'",
            &outputs
        ));
    }

    #[test]
    fn test_evaluate_when_or() {
        let mut outputs = HashMap::new();
        outputs.insert("a".to_string(), "hello".to_string());
        assert!(evaluate_when(
            "output(a) == 'hello' or output(b) == 'world'",
            &outputs
        ));
        assert!(evaluate_when("output(a) == 'nope' or output(b) == 'world'", &outputs) == false);
    }

    #[test]
    fn test_evaluate_loop_condition() {
        assert!(evaluate_loop_condition("true", "anything"));
        assert!(!evaluate_loop_condition("false", "anything"));
        assert!(evaluate_loop_condition(
            "output contains DONE",
            "Step completed DONE"
        ));
        assert!(!evaluate_loop_condition(
            "output contains DONE",
            "Still running"
        ));
        assert!(evaluate_loop_condition(
            "output not contains ERROR",
            "Success"
        ));
        assert!(!evaluate_loop_condition(
            "output not contains ERROR",
            "ERROR: failed"
        ));
        assert!(evaluate_loop_condition("output len > 5", "hello world"));
        assert!(!evaluate_loop_condition("output len > 100", "short"));
    }

    #[test]
    fn test_trigger_rule_all_success() {
        // This test verifies the logic without a full WorkflowRun
        let def = WorkflowNodeDef {
            id: "test".to_string(),
            node_type: NodeType::Prompt,
            depends_on: vec!["a".to_string(), "b".to_string()],
            when: None,
            trigger_rule: TriggerRule::AllSuccess,
            context: ContextMode::default(),
            prompt: None,
            command: None,
            label: None,
            model: None,
            retry: None,
            loop_condition: None,
        };

        let run = WorkflowRun {
            run_id: Uuid::new_v4(),
            workflow_name: "test".to_string(),
            status: WorkflowStatus::Running,
            nodes: vec![
                NodeRun {
                    node_id: "a".to_string(),
                    status: NodeStatus::Completed,
                    output: None,
                    started_at: None,
                    completed_at: None,
                    error: None,
                },
                NodeRun {
                    node_id: "b".to_string(),
                    status: NodeStatus::Failed,
                    output: None,
                    started_at: None,
                    completed_at: None,
                    error: None,
                },
            ],
            inputs: HashMap::new(),
            created_at: Utc::now(),
            completed_at: None,
        };

        assert!(!check_trigger_rule(&def, &run, &HashMap::new()));
    }
}
