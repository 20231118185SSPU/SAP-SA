//! WebSocket JSON protocol shared by the daemon and CLI.
//!
//! Handshake overview:
//! - The client must prove it is an expected SA frontend first.
//! - The server verifies that proof.
//! - Only after verification does the server return its own proof.
//! - Once both sides verify each other, the normal WS message stream begins.
//!
//! This is intentionally explicit JSON with tagged enums so reconnect logic,
//! manual debugging, and protocol evolution all stay traceable.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ts_rs::TS;
use uuid::Uuid;

use crate::openai::{AuthStyle, WireApi};use crate::workflow::{WorkflowNodeDef, NodeStatus};
use std::collections::HashMap;


/// Client -> server messages.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(tag = "type")]
#[ts(export, export_to = "../bindings/")]
pub enum ClientMessage {
    /// Mandatory first packet for mutual protocol identification.
    #[serde(rename = "client_hello")]
    ClientHello { hello: ClientHello },

    /// Submit a new task to the agent.
    #[serde(rename = "submit")]
    Submit {
        /// Client-generated task id for idempotent retries.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<Uuid>,
        /// The task text for the agent.
        task: String,
    },

    /// Request event history starting after `from_event_id`.
    #[serde(rename = "get_history")]
    GetHistory { from_event_id: u64 },

    /// Interrupt a running task.
    #[serde(rename = "interrupt")]
    Interrupt { task_id: Uuid },

    /// Answer a pending structured question.
    #[serde(rename = "answer_question")]
    AnswerQuestion { answer: UserQuestionAnswer },

    /// Initialize a missing `sa.toml` through a frontend onboarding flow.
    #[serde(rename = "initialize_config")]
    InitializeConfig { request: InitializeConfigRequest },

    /// Upload an image from the frontend for multimodal tasks.
    ///
    /// The backend saves the image under the workspace uploads directory and
    /// injects an image reference into the next agent message context.
    #[serde(rename = "image_upload")]
    ImageUpload {
        /// Client-generated upload id for correlation.
        upload_id: uuid::Uuid,
        /// Original filename provided by the user (for extension detection only).
        filename: String,
        /// MIME type hint, e.g. "image/png", "image/jpeg".
        mime_type: String,
        /// Base64-encoded image bytes.
        data: String,
        /// Optional task id to attach the image to. If None, the image will
        /// be associated with the next submitted task.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        task_id: Option<uuid::Uuid>,
    },

    /// Request a snapshot of the current backend configuration.
    ///
    /// The backend responds with `ConfigSnapshot` containing the full config
    /// with sensitive fields (e.g. api_key) masked.
    #[serde(rename = "get_config")]
    GetConfig,

    /// Request an update to the backend configuration.
    ///
    /// The client sends the full (modified) config as a TOML-compatible JSON
    /// object. The backend validates it, writes it to `sa.toml`, and triggers
    /// a hot reload of the runtime.
    #[serde(rename = "update_config")]
    UpdateConfig {
        /// The complete configuration as a JSON object matching the `sa.toml`
        /// schema. Sensitive fields (api_key) sent as masked values will be
        /// replaced by the current real values on the server side.
        #[ts(type = "unknown")]
        config: toml::Value,
    },


    /// Request the content of a skill's documentation file (SKILL.md).
    #[serde(rename = "get_skill_doc")]
    GetSkillDoc {
        /// The name of the skill to retrieve documentation for.
        name: String,
    },


    // ── Memory panel messages ────────────────────────────────────────

    /// Query semantic memory facts with optional search and pagination.
    #[serde(rename = "get_memory_facts")]
    GetMemoryFacts {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default = "default_memory_limit")]
        limit: usize,
        #[serde(default)]
        offset: usize,
    },

    /// Request memory system statistics.
    #[serde(rename = "get_memory_stats")]
    GetMemoryStats,

    /// Traverse the semantic memory graph from a starting subject.
    #[serde(rename = "traverse_memory_graph")]
    TraverseMemoryGraph {
        subject: String,
        #[serde(default = "default_memory_hops")]
        max_hops: usize,
        #[serde(default = "default_memory_limit")]
        max_results: usize,
    },

    /// Trigger memory decay (apply exponential confidence decay).
    #[serde(rename = "decay_memory_facts")]
    DecayMemoryFacts,

    /// Prune facts below a confidence threshold.
    #[serde(rename = "prune_memory_facts")]
    PruneMemoryFacts {
        min_confidence: f64,
    },
    /// Request the content of a memory report file (dream / post-task).
    #[serde(rename = "get_memory_report")]
    GetMemoryReport {
        /// Relative path under the memory directory (e.g. "dreams/report_20260505.md").
        report_path: String,
    },

    // ── Workflow + Plan + ChatMode messages ───────────────────────────

    /// Start a named workflow execution.
    #[serde(rename = "start_workflow")]
    StartWorkflow {
        workflow_name: String,
        #[serde(default)]
        inputs: HashMap<String, String>,
    },

    /// Cancel a running workflow.
    #[serde(rename = "cancel_workflow")]
    CancelWorkflow {
        run_id: Uuid,
    },

    /// Switch the chat mode (plan / chat / workflow).
    #[serde(rename = "set_chat_mode")]
    SetChatMode {
        mode: ChatMode,
    },

    /// Approve, step-execute, or request modification of a plan.
    #[serde(rename = "approve_plan")]
    ApprovePlan {
        plan_id: Uuid,
        action: PlanAction,
    },

    /// Respond to an approval request from a workflow node.
    #[serde(rename = "respond_approval")]
    RespondApproval {
        run_id: Uuid,
        node_id: String,
        approved: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        feedback: Option<String>,
    },

    /// Inject context into a running workflow node.
    #[serde(rename = "inject_workflow_context")]
    InjectWorkflowContext {
        run_id: Uuid,
        node_id: String,
        context: String,
    },

    /// Pause a running workflow.
    #[serde(rename = "pause_workflow")]
    PauseWorkflow {
        run_id: Uuid,
    },

    /// Resume a paused workflow.
    #[serde(rename = "resume_workflow")]
    ResumeWorkflow {
        run_id: Uuid,
    },

    /// Request the list of all available skills with metadata.
    #[serde(rename = "get_skills_list")]
    GetSkillsList,
    /// Test API key connectivity by sending a minimal chat completion request.
    #[serde(rename = "test_api_key")]
    TestApiKey {
        base_url: String,
        api_key: String,
        model: String,
    },

    /// Request the list of supported models with pricing info.
    #[serde(rename = "get_supported_models")]
    GetSupportedModels,
}


fn default_memory_limit() -> usize { 50 }
fn default_memory_hops() -> usize { 3 }

/// Server -> client messages.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(tag = "type")]
#[ts(export, export_to = "../bindings/")]
pub enum ServerMessage {
    /// Server proof returned only after the client's `client_hello` passed.
    #[serde(rename = "server_hello")]
    ServerHello { hello: ServerHello },

    /// Handshake rejection. The client should treat the connection as invalid.
    #[serde(rename = "hello_reject")]
    HelloReject { reject: HelloReject },

    /// Task accepted and queued.
    #[serde(rename = "accepted")]
    Accepted { task_id: Uuid },

    /// Buffered event history.
    #[serde(rename = "history")]
    History { events: Vec<Event> },

    /// One live event.
    #[serde(rename = "event")]
    Event { event: Event },

    /// A newly issued structured question.
    #[serde(rename = "question")]
    Question { question: UserQuestion },

    /// Snapshot of all pending questions.
    #[serde(rename = "pending_questions")]
    PendingQuestions { questions: Vec<UserQuestion> },

    /// A question has been resolved or cancelled.
    #[serde(rename = "question_resolved")]
    QuestionResolved { question_id: Uuid },

    /// A user-visible file payload emitted by the backend `Show` tool.
    #[serde(rename = "show")]
    Show { file: UserVisibleFile },

    /// Snapshot of the most recent shown files for reconnecting clients.
    #[serde(rename = "recent_shows")]
    RecentShows { files: Vec<UserVisibleFile> },

    /// The backend is running without `sa.toml` and needs first-run setup.
    #[serde(rename = "init_required")]
    InitRequired { request: InitRequired },

    /// The frontend-created configuration has been accepted and activated.
    #[serde(rename = "init_completed")]
    InitCompleted { info: InitCompleted },

    /// The submitted initialization request failed validation or activation.
    #[serde(rename = "init_failed")]
    InitFailed { error: InitFailed },

    /// Protocol or request error.
    #[serde(rename = "error")]
    Error { message: String },

    /// Acknowledgement for a successful image upload.
    #[serde(rename = "image_uploaded")]
    ImageUploaded {
        /// Echoed upload id from the client's `image_upload` message.
        upload_id: uuid::Uuid,
        /// Absolute workspace path where the image was saved.
        saved_path: String,
    },

    /// Snapshot of the current backend configuration with sensitive fields
    /// masked (e.g. api_key shown as `sk-...xxxx`).
    #[serde(rename = "config_snapshot")]
    ConfigSnapshot {
        /// The full config as a JSON value (TOML-compatible).
        #[ts(type = "unknown")]
        config: toml::Value,
        /// Absolute path to the `sa.toml` file on disk.
        config_path: String,
    },

    /// The configuration update requested by the client has been applied
    /// successfully and the runtime has been hot-reloaded.
    #[serde(rename = "config_updated")]
    ConfigUpdated {
        /// Human-readable summary of what changed.
        summary: String,
    },

    /// The configuration update failed validation or could not be applied.
    #[serde(rename = "config_update_failed")]
    ConfigUpdateFailed {
        /// Short user-facing error message.
        message: String,
        /// Optional detailed diagnostic text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },


    /// Skill documentation content response.
    #[serde(rename = "skill_doc")]
    SkillDoc {
        /// The name of the skill.
        name: String,
        /// The skill's description.
        description: String,
        /// The full content of the SKILL.md file.
        content: String,
    },


    // ── Memory panel responses ──────────────────────────────────────

    /// Semantic memory facts response.
    #[serde(rename = "memory_facts")]
    MemoryFacts {
        facts: Vec<MemoryFact>,
        total: usize,
        offset: usize,
    },

    /// Semantic memory graph traversal response.
    #[serde(rename = "memory_graph")]
    MemoryGraph {
        subject: String,
        facts: Vec<MemoryFact>,
        hops: usize,
    },

    /// Memory system statistics.
    #[serde(rename = "memory_stats")]
    MemoryStats {
        total_facts: usize,
        unique_subjects: usize,
        unique_predicates: usize,
        avg_confidence: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        oldest_fact: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        newest_fact: Option<String>,
    },

    /// Result of a memory write operation (decay/prune).
    #[serde(rename = "memory_operation_result")]
    MemoryOperationResult {
        operation: String,
        affected: usize,
        summary: String,
    },
    /// 记忆已变更（由 dream scheduler 或记忆操作触发），前端应重新拉取
    #[serde(rename = "memory_updated")]
    MemoryUpdated {
        /// 触发来源: "dream" | "decay" | "prune" | "manual"
        source: String,
        /// 关联的报告路径（可选）
        #[serde(default, skip_serializing_if = "Option::is_none")]
        report: Option<String>,
    },
    /// Content of a requested memory report file.
    #[serde(rename = "memory_report")]
    MemoryReport {
        /// The relative path that was requested.
        report_path: String,
        /// Full text content of the report file (Markdown).
        content: String,
    },

    /// Cache performance statistics snapshot.
    #[serde(rename = "cache_stats")]
    CacheStats {
        /// Number of requests tracked.
        total_requests: usize,
        /// Total cache hit tokens.
        total_cache_hits: u64,
        /// Total cache miss tokens.
        total_cache_misses: u64,
        /// Total output tokens.
        total_output_tokens: u64,
        /// Cache hit rate (0.0 - 1.0).
        hit_rate: f64,
        /// Estimated cost savings from cache hits.
        estimated_savings: f64,
        /// Per-model breakdown.
        model_breakdown: Vec<CacheModelStats>,
    },

    /// Real-time context window usage information.
    #[serde(rename = "context_info")]
    ContextInfo {
        /// Current estimated tokens used in the context window.
        used_tokens: usize,
        /// Compaction trigger threshold (max tokens before compact).
        max_tokens: usize,
        /// Number of messages in the current conversation.
        message_count: usize,
    },

    /// Workflow node execution update (legacy — used by ReactFlow visualizer).
    #[serde(rename = "workflow_update")]
    WorkflowUpdate {
        /// Workflow execution ID.
        workflow_id: String,
        /// Updated nodes.
        nodes: Vec<WorkflowNode>,
        /// Active connections between nodes.
        edges: Vec<WorkflowEdge>,
    },

    // ── Workflow engine responses ────────────────────────────────────

    /// A workflow has started execution.
    #[serde(rename = "workflow_started")]
    WorkflowStarted {
        run_id: Uuid,
        workflow_name: String,
        nodes: Vec<WorkflowNodeDef>,
    },

    /// A workflow node's status has changed.
    #[serde(rename = "workflow_node_update")]
    WorkflowNodeUpdate {
        run_id: Uuid,
        node_id: String,
        status: NodeStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
    },

    /// Workflow completed successfully.
    #[serde(rename = "workflow_completed")]
    WorkflowCompleted {
        run_id: Uuid,
        summary: String,
    },

    /// Workflow failed.
    #[serde(rename = "workflow_failed")]
    WorkflowFailed {
        run_id: Uuid,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
        error: String,
    },

    /// An approval node is waiting for user response.
    #[serde(rename = "approval_requested")]
    ApprovalRequested {
        run_id: Uuid,
        node_id: String,
        prompt: String,
    },

    // ── Plan mode responses ──────────────────────────────────────────

    /// A plan has been created (Plan mode).
    #[serde(rename = "plan_created")]
    PlanCreated {
        plan_id: Uuid,
        steps: Vec<PlanStep>,
    },

    /// A plan step's status has changed.
    #[serde(rename = "plan_step_update")]
    PlanStepUpdate {
        plan_id: Uuid,
        step_id: String,
        status: PlanStepStatus,
    },

    /// Chat mode has changed.
    #[serde(rename = "chat_mode_changed")]
    ChatModeChanged {
        mode: ChatMode,
    },

    /// Skill activated by trigger word matching.
    #[serde(rename = "skill_activated")]
    SkillActivated {
        skill_name: String,
        description: String,
        trigger: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        orchestration_mode: Option<String>,
    },

    /// Full list of available skills with metadata.
    #[serde(rename = "skills_list")]
    SkillsList {
        skills: Vec<SkillListItem>,
    },


    /// Verification step passed.
    #[serde(rename = "verify_pass")]
    VerifyPass {
        step: u32,
        message: String,
    },

    /// Verification step failed.
    #[serde(rename = "verify_fail")]
    VerifyFail {
        step: u32,
        error: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_count: Option<u32>,
    },

    /// Result of an API key connectivity test.
    #[serde(rename = "api_key_test_result")]
    ApiKeyTestResult {
        ok: bool,
        model: String,
        provider: String,
        latency_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    /// List of supported models with pricing information.
    #[serde(rename = "supported_models")]
    SupportedModels {
        models: Vec<SupportedModelInfo>,
    },
}

/// Cache statistics for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct CacheModelStats {
    pub model: String,
    pub requests: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub output_tokens: u64,
    pub hit_rate: f64,
}

/// Supported model information with pricing.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct SupportedModelInfo {
    /// Model identifier (e.g. "deepseek-v4-flash").
    pub id: String,
    /// User-facing display name (e.g. "DeepSeek V4 Flash").
    pub display_name: String,
    /// Provider identifier (e.g. "deepseek", "mimo", "openai").
    pub provider: String,
    /// Price per million tokens for cache hits (USD).
    pub cache_hit_price_per_million: f64,
    /// Price per million tokens for cache misses (USD).
    pub cache_miss_price_per_million: f64,
    /// Price per million tokens for output (USD).
    pub output_price_per_million: f64,
    /// Whether this model offers a cache discount.
    pub has_cache_discount: bool,
}

/// One skill entry in the skills list response.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct SkillListItem {
    /// Skill name (e.g. "华书·新闻评论").
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Where this skill comes from.
    pub source: SkillSource,
    /// Optional trigger words/phrases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<String>,
    /// Usage count (0 if unknown).
    #[serde(default)]
    pub usage_count: u32,
    /// Whether this is a core/preserved skill.
    #[serde(default)]
    pub core: bool,
}

/// Source of a skill in the skills list.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum SkillSource {
    /// Built into the binary.
    Builtin,
    /// Local SKILL.md file.
    Local,
    /// Exposed by an MCP server.
    Mcp,
}



/// Workflow node representation.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct WorkflowNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub status: String,
    pub x: f64,
    pub y: f64,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Workflow edge connection.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct WorkflowEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub active: bool,
}

/// A single semantic fact for wire transfer.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct MemoryFact {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// First-run initialization methods offered by the backend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum InitMethod {
    /// OpenAI-compatible gateways using Bearer auth.
    OpenAiCompatible,
    /// Anthropic-compatible `/v1/messages` providers.
    AnthropicCompatible,
    /// Advanced manual mode where the frontend exposes all wire settings.
    Custom,
}

/// One frontend-visible initialization option card.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct InitMethodOption {
    /// Stable option id.
    pub id: InitMethod,
    /// User-facing label.
    pub label: String,
    /// Explanatory help text.
    pub description: String,
    /// Recommended default wire protocol for this option.
    pub default_wire_api: WireApi,
    /// Recommended default auth style for this option.
    pub default_auth_style: AuthStyle,
    /// Recommended role name for the prompt prefix message.
    pub default_system_role_name: String,
    /// Whether this option should be highlighted as the recommended path.
    #[serde(default)]
    pub recommended: bool,
}

/// Message sent by the backend when `sa.toml` is missing.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct InitRequired {
    /// Absolute path where the backend expects `sa.toml`.
    pub config_path: String,
    /// Workspace root the generated config will point at.
    pub workspace_root: String,
    /// `Agents.md` path the generated config will use.
    pub agents_md_path: String,
    /// Supported initialization methods the frontend should render.
    pub methods: Vec<InitMethodOption>,
    /// Recommended method to preselect on the frontend.
    pub recommended_method: InitMethod,
}

/// Request sent by the frontend to create the first config file.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct InitializeConfigRequest {
    /// Which initialization path the user selected.
    pub method: InitMethod,
    /// Provider base URL.
    pub base_url: String,
    /// Provider API credential.
    pub api_key: String,
    /// Model name.
    pub model: String,
    /// Optional explicit wire protocol override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire_api: Option<WireApi>,
    /// Optional explicit auth style override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_style: Option<AuthStyle>,
    /// Optional role name override for the prompt prefix message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_role_name: Option<String>,
    /// Optional reasoning depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Positive confirmation returned after first-run initialization succeeds.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct InitCompleted {
    /// Absolute path where the backend wrote the new `sa.toml`.
    pub config_path: String,
    /// Workspace root activated by the new runtime.
    pub workspace_root: String,
    /// Human-readable summary for the frontend.
    pub message: String,
}

/// Structured initialization error returned to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct InitFailed {
    /// Short user-facing summary.
    pub message: String,
    /// Optional detailed diagnostic text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The client's proof-of-identity packet.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct ClientHello {
    /// Stable protocol identifier (example: `sa-ws/v1`).
    pub protocol: String,
    /// Hash algorithm identifier used by the proofs.
    pub hash_algo: String,
    /// Time bucket width in seconds.
    pub time_step_secs: u64,
    /// Allowed clock skew in time buckets.
    pub allowed_skew_buckets: u64,
    /// Expected logical frontend identity name (currently `sa-frontend`).
    pub client_name: String,
    /// Frontend binary version.
    pub client_version: String,
    /// UTC time bucket used to derive the proof.
    pub time_bucket: i64,
    /// Client-generated nonce to bind the server proof to this exact handshake.
    pub client_nonce: String,
    /// Hex-encoded SHA-256 proof.
    pub proof: String,
}

/// The server's proof-of-identity packet.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct ServerHello {
    /// Stable protocol identifier (example: `sa-ws/v1`).
    pub protocol: String,
    /// Hash algorithm identifier used by the proofs.
    pub hash_algo: String,
    /// Time bucket width in seconds.
    pub time_step_secs: u64,
    /// Allowed clock skew in time buckets.
    pub allowed_skew_buckets: u64,
    /// Expected backend name (currently `sa`).
    pub server_name: String,
    /// Backend binary version.
    pub server_version: String,
    /// UTC time bucket used to derive the proof.
    pub time_bucket: i64,
    /// Best-effort backend machine hint for audit/debugging. This is not part
    /// of the proof verification path.
    pub machine_hint: String,
    /// Echo of the client's nonce.
    pub client_nonce: String,
    /// Server-generated nonce.
    pub server_nonce: String,
    /// Hex-encoded SHA-256 proof.
    pub proof: String,
}

/// Rejection payload returned when the mandatory handshake fails.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct HelloReject {
    /// Stable protocol identifier.
    pub protocol: String,
    /// Server name emitting the rejection.
    pub server_name: String,
    /// Server binary version.
    pub server_version: String,
    /// Human-readable rejection reason.
    pub reason: String,
}

/// One event emitted by the backend.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct Event {
    /// Monotonic event id assigned by the server.
    pub event_id: u64,
    /// UTC timestamp of the event.
    pub ts: DateTime<Utc>,
    /// Task that produced the event.
    pub task_id: Uuid,
    /// Event classification used by the UI.
    pub kind: EventKind,
    /// Human-readable message.
    pub message: String,
    /// Optional structured sender identity for frontend rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentIdentity>,
}

/// Structured agent identity surfaced to frontends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
pub struct AgentIdentity {
    /// Stable agent UUID inside the current runtime tree.
    pub agent_id: Uuid,
    /// Internal durable label used by SA for this agent.
    pub agent_label: String,
    /// User-facing display name the frontend should render.
    pub display_name: String,
    /// Whether this identity belongs to the root "学习委员" agent.
    pub is_root: bool,
}

/// Structured question emitted by the backend.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct UserQuestion {
    /// Unique question id.
    pub question_id: Uuid,
    /// Top-level task id this question belongs to.
    pub task_id: Uuid,
    /// Backend-side creation time used by reconnecting frontends to restore
    /// the original ordering around blocking `Ask` messages.
    pub created_at: DateTime<Utc>,
    /// Agent identity that issued this question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentIdentity>,
    /// User-facing prompt.
    pub prompt: String,
    /// Expected answer shape.
    pub mode: QuestionMode,
    /// Choice options for non-text questions.
    #[serde(default)]
    pub options: Vec<QuestionOption>,
    /// Whether extra free text is allowed.
    #[serde(default)]
    pub allow_free_text: bool,
}

/// One selectable option inside a structured question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
pub struct QuestionOption {
    /// Stable option id used in answers.
    pub id: String,
    /// User-facing label.
    pub label: String,
    /// Optional descriptive help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// How the user should answer the question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum QuestionMode {
    /// Exactly one choice.
    SingleChoice,
    /// Zero or more choices.
    MultiChoice,
    /// Pure free text.
    Text,
}

/// User answer sent back to the backend.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct UserQuestionAnswer {
    /// Which question this answers.
    pub question_id: Uuid,
    /// Selected option ids, if any.
    #[serde(default)]
    pub selected_option_ids: Vec<String>,
    /// Optional free-text payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_text: Option<String>,
}

/// A file payload explicitly shown to the user by the backend.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct UserVisibleFile {
    /// Unique show id.
    pub show_id: Uuid,
    /// Top-level task that produced this file.
    pub task_id: Uuid,
    /// Agent identity that issued this file display.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentIdentity>,
    /// Original backend path string.
    pub path: String,
    /// Optional user-facing title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Concise non-user-facing description of why this file is being shown.
    pub prompt: String,
    /// Best-effort media type hint.
    pub media_type: String,
    /// How `content` should be interpreted.
    pub encoding: UserVisibleFileEncoding,
    /// Text body or base64-encoded bytes.
    pub content: String,
    /// Original byte size before transport encoding.
    pub bytes: usize,
}

/// Encoding used for `UserVisibleFile.content`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum UserVisibleFileEncoding {
    /// UTF-8 text sent directly.
    Utf8,
    /// Binary bytes encoded as base64 text.
    Base64,
}

/// Event kind used by the CLI for formatting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Normal logging/progress output.
    Log,
    /// Tool execution details.
    Tool,
    /// A direct user-facing message.
    Message,
    /// Final answer / completion.
    Final,
    /// Recoverable error output.
    Error,
}
// ── Workflow + Plan + ChatMode types ─────────────────────────────────

/// Chat mode selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum ChatMode {
    /// Agent generates a plan, user approves, then agent executes step-by-step.
    Plan,
    /// Free-form conversation (default).
    Chat,
    /// Run a named YAML workflow automatically.
    Workflow,
}

impl Default for ChatMode {
    fn default() -> Self {
        Self::Chat
    }
}

/// User action on a generated plan.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum PlanAction {
    /// Approve all steps for execution.
    ApproveAll,
    /// Approve only the next step.
    ApproveStep,
    /// Request plan modification.
    Modify,
}

/// One step in a generated plan.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct PlanStep {
    pub step_id: String,
    pub description: String,
    #[serde(default)]
    pub status: PlanStepStatus,
}

/// Status of a plan step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    /// Waiting to be executed.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Encountered an error.
    Failed,
}

impl Default for PlanStepStatus {
    fn default() -> Self {
        Self::Pending
    }
}

