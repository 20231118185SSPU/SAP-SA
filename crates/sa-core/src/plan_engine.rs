//! Plan generation and execution engine.
//!
//! Handles the Plan chat mode: LLM generates a step-by-step plan,
//! user approves/rejects/modifies, then the engine executes step by step.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::openai::{ChatCompletionsRequest, ChatMessage, MessageContent, OpenAiClient};
use crate::ws_protocol::{PlanAction, PlanStep, PlanStepStatus};

/// A generated plan with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub plan_id: Uuid,
    pub title: Option<String>,
    pub steps: Vec<PlanStep>,
    pub created_at: chrono::DateTime<Utc>,
}

/// Plan execution progress events.
#[derive(Debug, Clone)]
pub enum PlanEvent {
    PlanCreated {
        plan_id: Uuid,
        steps: Vec<PlanStep>,
    },
    StepUpdate {
        plan_id: Uuid,
        step_id: String,
        status: PlanStepStatus,
        output: Option<String>,
    },
    PlanCompleted {
        plan_id: Uuid,
        summary: String,
    },
}

/// Progress sender type.
pub type PlanProgressSender = mpsc::UnboundedSender<PlanEvent>;

/// LLM configuration for plan engine.
#[derive(Debug, Clone)]
pub struct PlanLlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

/// Generate a plan from a user request using LLM.
pub async fn generate_plan(
    user_request: &str,
    context: &str,
    llm_config: &PlanLlmConfig,
) -> Result<Plan, anyhow::Error> {
    let client = OpenAiClient::new(llm_config.base_url.clone(), llm_config.api_key.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create LLM client: {e}"))?;

    let system_prompt = format!(
        "{}\n\nYou are a planning assistant. Given the user's request and context, \
         generate a step-by-step execution plan. Each step should be actionable and \
         self-contained. Return ONLY valid JSON, no other text.",
        context
    );

    let user_prompt = format!(
        "Generate an execution plan for the following request. \
         Respond with a JSON array of steps, where each step has: \
         'step_id' (unique identifier string), \
         'description' (what to do in this step), \
         'depends_on' (array of step_ids this step depends on, or empty array).\n\n\
         User request: {user_request}\n\n\
         Example format: [{{\"step_id\":\"1\",\"description\":\"Read the config file\",\"depends_on\":[]}},\
         {{\"step_id\":\"2\",\"description\":\"Analyze the config\",\"depends_on\":[\"1\"]}}]"
    );

    let req = ChatCompletionsRequest {
        model: llm_config.model.clone(),
        messages: vec![
            ChatMessage::text("system", &system_prompt),
            ChatMessage::text("user", &user_prompt),
        ],
        max_tokens: Some(2048),
        reasoning_effort: None,
        tools: None,
        tool_choice: None,
        stream: Some(false),
        temperature: Some(0.3),
        top_p: None,
    };

    let resp = client.chat_completions(&req).await?;
    let content = resp
        .choices
        .first()
        .and_then(|c| match &c.message.content {
            Some(MessageContent::Text(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default();

    // Parse the JSON response into steps.
    let steps = parse_plan_steps(&content)?;

    Ok(Plan {
        plan_id: Uuid::new_v4(),
        title: None,
        steps,
        created_at: Utc::now(),
    })
}

/// Parse LLM output into plan steps.
fn parse_plan_steps(content: &str) -> Result<Vec<PlanStep>, anyhow::Error> {
    // Try to extract JSON array from the response.
    let content = content.trim();

    // Find the first '[' and last ']'
    let start = content
        .find('[')
        .ok_or_else(|| anyhow::anyhow!("No JSON array found in response"))?;
    let end = content
        .rfind(']')
        .ok_or_else(|| anyhow::anyhow!("No closing bracket found"))?;

    let json_str = &content[start..=end];

    #[derive(Deserialize)]
    struct RawStep {
        step_id: String,
        description: String,
        #[serde(default)]
        depends_on: Vec<String>,
    }

    let raw_steps: Vec<RawStep> = serde_json::from_str(json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse plan steps: {e}"))?;

    let steps: Vec<PlanStep> = raw_steps
        .into_iter()
        .enumerate()
        .map(|(i, s)| PlanStep {
            step_id: if s.step_id.is_empty() {
                format!("step_{}", i + 1)
            } else {
                s.step_id
            },
            description: s.description,
            status: PlanStepStatus::Pending,
        })
        .collect();

    if steps.is_empty() {
        anyhow::bail!("No steps generated");
    }

    Ok(steps)
}

/// Execute a plan step-by-step using an LLM.
///
/// Each step gets the accumulated outputs from previous steps and produces
/// a result (text output). The step's output is stored and used in subsequent steps.
pub async fn execute_plan_step(
    plan: &Plan,
    step_index: usize,
    accumulated_context: &str,
    llm_config: &PlanLlmConfig,
) -> Result<String, anyhow::Error> {
    if step_index >= plan.steps.len() {
        anyhow::bail!("Step index out of bounds");
    }

    let step = &plan.steps[step_index];
    let client = OpenAiClient::new(llm_config.base_url.clone(), llm_config.api_key.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create LLM client: {e}"))?;

    let prompt = format!(
        "You are executing a plan step by step.\n\n\
         Plan context: {}\n\n\
         Previous steps output:\n{}\n\n\
         Current step:\n\
         Step ID: {}\n\
         Description: {}\n\n\
         Execute this step now. Be thorough and produce a clear result.\n\
         If you need to create files, describe what you would create.\n\
         Output only the execution result, no extra commentary.",
        plan.title.as_deref().unwrap_or("Untitled Plan"),
        if accumulated_context.is_empty() {
            "(no previous steps)"
        } else {
            accumulated_context
        },
        step.step_id,
        step.description,
    );

    let req = ChatCompletionsRequest {
        model: llm_config.model.clone(),
        messages: vec![
            ChatMessage::text("system", "You are a plan execution assistant. Execute the given step and output only the result."),
            ChatMessage::text("user", &prompt),
        ],
        max_tokens: Some(4096),
        reasoning_effort: None,
        tools: None,
        tool_choice: None,
        stream: Some(false),
        temperature: Some(0.5),
        top_p: None,
    };

    let resp = client.chat_completions(&req).await?;
    let content = resp
        .choices
        .first()
        .and_then(|c| match &c.message.content {
            Some(MessageContent::Text(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default();

    Ok(content)
}

/// Save a plan to disk for persistence.
pub async fn save_plan(workspace_root: &PathBuf, plan: &Plan) -> std::io::Result<()> {
    let dir = workspace_root.join("runtime").join("plans");
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{}.json", plan.plan_id));
    let json = serde_json::to_string_pretty(plan)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    tokio::fs::write(&path, json).await
}

/// Load a plan from disk.
pub async fn load_plan(workspace_root: &PathBuf, plan_id: &Uuid) -> Result<Plan, anyhow::Error> {
    let path = workspace_root
        .join("runtime")
        .join("plans")
        .join(format!("{plan_id}.json"));
    let content = tokio::fs::read_to_string(&path).await?;
    let plan: Plan = serde_json::from_str(&content)?;
    Ok(plan)
}

// ── Workflow Generation via LLM ─────────────────────────────────────────

use crate::workflow::{WorkflowDef, WorkflowNodeDef, NodeType};

/// Generate a workflow DAG from a user request using LLM.
///
/// The LLM decomposes the task into interdependent nodes that can be
/// executed as a DAG by the workflow engine.
pub async fn generate_workflow(
    user_request: &str,
    context: &str,
    llm_config: &PlanLlmConfig,
) -> Result<WorkflowDef, anyhow::Error> {
    let client = OpenAiClient::new(llm_config.base_url.clone(), llm_config.api_key.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create LLM client: {e}"))?;

    let system_prompt = format!(
        "{}\n\n\
         You are a workflow decomposition assistant. Given the user's request, \
         break it down into a directed acyclic graph (DAG) of executable nodes. \
         Each node should be a self-contained unit of work. \
         Return ONLY valid JSON, no other text.\n\n\
         Each node has access to tools (file reading, bash commands, search) via an LLM agent. \
         You can generate nodes that read files, analyze code, run commands, etc. \
         The agent will handle tool execution automatically.",
        context
    );

    let user_prompt = format!(
        "Decompose the following task into a workflow DAG.\n\n\
         Respond with a JSON object with this exact structure:\n\
         {{\n\
           \"name\": \"<short workflow name>\",\n\
           \"description\": \"<brief description>\",\n\
           \"nodes\": [\n\
             {{\n\
               \"id\": \"<unique_node_id>\",\n\
               \"node_type\": \"prompt\",\n\
               \"label\": \"<human-readable label>\",\n\
               \"prompt\": \"<detailed instruction for this LLM node>\",\n\
               \"depends_on\": [\"<node_ids this depends on>\"]\n\
             }}\n\
           ]\n\
         }}\n\n\
         Rules:\n\
         - Use snake_case for node ids.\n\
         - node_type must always be \"prompt\".\n\
         - Each prompt should be a detailed instruction for an AI agent with tool access.\n\
         - Nodes can read files, run commands, search code, etc.\n\
         - Keep nodes focused — one clear responsibility per node.\n\
         - Use depends_on to express ordering constraints.\n\
         - Aim for 3-5 nodes.\n\
         - The last node should synthesize all prior outputs into a final report/answer.\n\n\
         User request: {user_request}"
    );

    let req = ChatCompletionsRequest {
        model: llm_config.model.clone(),
        messages: vec![
            ChatMessage::text("system", &system_prompt),
            ChatMessage::text("user", &user_prompt),
        ],
        max_tokens: Some(4096),
        reasoning_effort: None,
        tools: None,
        tool_choice: None,
        stream: Some(false),
        temperature: Some(0.3),
        top_p: None,
    };

    let resp = client.chat_completions(&req).await?;
    let content = resp
        .choices
        .first()
        .and_then(|c| match &c.message.content {
            Some(MessageContent::Text(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let def = parse_workflow_def(&content)?;
    Ok(def)
}

/// Parse LLM output into a WorkflowDef.
fn parse_workflow_def(content: &str) -> Result<WorkflowDef, anyhow::Error> {
    let content = content.trim();

    // Extract JSON object from response (find first '{' to last '}')
    let start = content
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("No JSON object found in response"))?;
    let end = content
        .rfind('}')
        .ok_or_else(|| anyhow::anyhow!("No closing brace found"))?;

    let json_str = &content[start..=end];

    // Raw deserialization struct matching the LLM output schema.
    #[derive(Deserialize)]
    struct RawWorkflow {
        name: String,
        description: Option<String>,
        nodes: Vec<RawNode>,
    }

    #[derive(Deserialize)]
    struct RawNode {
        id: String,
        node_type: String,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        prompt: Option<String>,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        depends_on: Vec<String>,
    }

    let raw: RawWorkflow = serde_json::from_str(json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse workflow JSON: {e}"))?;

    if raw.nodes.is_empty() {
        anyhow::bail!("Workflow has no nodes");
    }

    let nodes: Vec<WorkflowNodeDef> = raw
        .nodes
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            // Force all LLM-generated nodes to Prompt type.
            // Bash/Loop/Approval are unreliable in auto-generated workflows.
            let node_type = NodeType::Prompt;

            // If the LLM generated a bash node with a command, convert it to a prompt.
            let prompt = if let Some(cmd) = n.command {
                if n.prompt.is_none() {
                    Some(format!("Execute the following task and report the result: {}", cmd))
                } else {
                    n.prompt
                }
            } else {
                n.prompt
            };

            WorkflowNodeDef {
                id: if n.id.is_empty() {
                    format!("node_{}", i + 1)
                } else {
                    n.id
                },
                node_type,
                depends_on: n.depends_on,
                when: None,
                trigger_rule: Default::default(),
                context: Default::default(),
                prompt,
                command: None,
                label: n.label,
                model: None,
                retry: None,
                loop_condition: None,
            }
        })
        .collect();

    Ok(WorkflowDef {
        name: raw.name,
        description: raw.description.unwrap_or_else(|| "LLM-generated workflow".to_string()),
        nodes,
        default_model: None,
    })
}
