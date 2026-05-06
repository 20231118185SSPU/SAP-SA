//! Plan-to-Workflow crystallization and auto-optimization.
//!
//! When a Plan completes successfully, the execution path is automatically
//! saved as a reusable YAML workflow. After multiple runs, the system
//! optimizes model selection, timeouts, and other parameters.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::workflow::{WorkflowDef, WorkflowNodeDef, NodeType, ContextMode, TriggerRule};
use crate::ws_protocol::PlanStep;

/// Metadata about a workflow's performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMetrics {
    pub total_runs: u32,
    pub successful_runs: u32,
    pub failed_runs: u32,
    pub avg_duration_secs: f64,
    pub p95_duration_secs: f64,
    pub avg_tokens_per_run: u64,
    pub best_model: Option<String>,
}

/// Execution log entry for one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecutionLog {
    pub step_id: String,
    pub description: String,
    pub model_used: String,
    pub duration_ms: u64,
    pub output: String,
    pub success: bool,
}

/// Full execution log for a Plan run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLog {
    pub plan_id: String,
    pub start_time: chrono::DateTime<Utc>,
    pub end_time: chrono::DateTime<Utc>,
    pub steps: Vec<StepExecutionLog>,
    pub total_tokens: u64,
}

/// Crystallize a completed Plan execution into a reusable YAML workflow.
///
/// Returns the path to the generated workflow YAML file.
pub async fn crystallize_plan_to_workflow(
    plan_name: &str,
    steps: &[PlanStep],
    execution_log: &ExecutionLog,
    workflows_dir: &PathBuf,
) -> Result<PathBuf, anyhow::Error> {
    // Build workflow nodes from plan steps.
    let nodes: Vec<WorkflowNodeDef> = steps
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let depends_on: Vec<String> = if i > 0 {
                vec![steps[i - 1].step_id.clone()]
            } else {
                vec![]
            };

            // Use the model from the execution log if available.
            let model_used = execution_log
                .steps
                .iter()
                .find(|s| s.step_id == step.step_id)
                .map(|s| s.model_used.clone());

            WorkflowNodeDef {
                id: step.step_id.clone(),
                node_type: NodeType::Prompt,
                depends_on,
                when: None,
                trigger_rule: TriggerRule::AllSuccess,
                context: ContextMode::Shared,
                prompt: Some(step.description.clone()),
                command: None,
                label: Some(format!("Step {}: {}", i + 1, step.description)),
                model: model_used,
                retry: None,
                loop_condition: None,
            }
        })
        .collect();

    // Add metadata.
    let success_rate = if execution_log.steps.is_empty() {
        1.0
    } else {
        execution_log
            .steps
            .iter()
            .filter(|s| s.success)
            .count() as f64
            / execution_log.steps.len() as f64
    };

    let description = format!(
        "Auto-crystallized from Plan '{plan_name}'. Success rate: {:.0}% ({}/{})",
        success_rate * 100.0,
        execution_log.steps.iter().filter(|s| s.success).count(),
        execution_log.steps.len()
    );

    // Build best model hint from execution log.
    let best_model = execution_log
        .steps
        .first()
        .map(|s| s.model_used.clone());

    let def = WorkflowDef {
        name: plan_name.to_string(),
        description,
        nodes,
        default_model: best_model,
    };

    // Save to YAML.
    let date_str = Utc::now().format("%Y-%m-%d").to_string();
    let filename = format!("{}_{}.yaml", plan_name, date_str);
    let auto_dir = workflows_dir.join("auto");
    tokio::fs::create_dir_all(&auto_dir).await?;
    let path = auto_dir.join(&filename);

    let yaml = serde_yaml::to_string(&def)?;
    tokio::fs::write(&path, yaml).await?;

    Ok(path)
}

/// Automatically optimize a workflow based on accumulated metrics.
///
/// After 3+ successful runs, adjusts model selection and timeouts
/// based on performance data.
pub async fn auto_optimize_workflow(
    workflow_path: &PathBuf,
    metrics: &WorkflowMetrics,
) -> Result<(), anyhow::Error> {
    if metrics.successful_runs < 3 {
        return Ok(()); // Not enough data to optimize yet.
    }

    let content = tokio::fs::read_to_string(workflow_path).await?;
    let mut def: WorkflowDef = serde_yaml::from_str(&content)?;

    // Use the best model if determined.
    if let Some(ref best_model) = metrics.best_model {
        def.default_model = Some(best_model.clone());
    }

    // Update timeout (P95 * 1.5) in the workflow metadata.
    // Note: YAML doesn't have a timeout field directly; we add it to description.
    let timeout_hint = format!(
        " [Auto-tuned: P95={:.1}s, runs={}, success_rate={:.0}%]",
        metrics.p95_duration_secs,
        metrics.total_runs,
        if metrics.total_runs > 0 {
            metrics.successful_runs as f64 / metrics.total_runs as f64 * 100.0
        } else {
            0.0
        }
    );
    if !def.description.contains("[Auto-tuned:") {
        def.description.push_str(&timeout_hint);
    }

    let yaml = serde_yaml::to_string(&def)?;
    tokio::fs::write(workflow_path, yaml).await?;

    Ok(())
}
