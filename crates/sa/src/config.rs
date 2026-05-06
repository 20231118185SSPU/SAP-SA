//! Configuration loading and runtime building.

use anyhow::Context as _;
use sa_core::agents_md::load_agents_md;
use sa_core::config::{
    Config, TeamConfig, resolve_config_file_dir,
};
use sa_core::dream::DreamManager;
use sa_core::interaction_history::InteractionStore;
use sa_core::mcp_client::McpRegistry;
use sa_core::openai::OpenAiClient;
use sa_core::runtime::state::{AgentState, TeamState};
use sa_core::runtime::store::{RootMarker, RuntimeStore};
use sa_core::session::SessionStore;
use sa_core::skills::SkillRegistry;
use sa_core::task_audit::TaskAuditStore;
use sa_core::tools::{ToolContext, ToolExecutor};
use sa_core::ws_protocol::InitializeConfigRequest;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

use crate::utils::{effective_init_auth_style, effective_init_wire_api, normalize_optional_string, toml_string};

/// Default listen address used when `sa.toml` does not exist yet.
const DEFAULT_BOOTSTRAP_BIND: &str = "127.0.0.1:8765";

/// Default WS path used when `sa.toml` does not exist yet.
const DEFAULT_BOOTSTRAP_WS_PATH: &str = "/ws";

/// Bootstrap state used when `sa.toml` does not exist yet.
#[derive(Debug)]
pub(crate) struct BootstrapState {
    pub(crate) config_path: PathBuf,
    pub(crate) config_dir: PathBuf,
    pub(crate) workspace_root: PathBuf,
    pub(crate) agents_md_path: PathBuf,
    pub(crate) bind: String,
    pub(crate) ws_path: String,
}

/// Fully built runtime returned from one successfully loaded configuration.
pub(crate) struct LoadedRuntime {
    pub(crate) hub: Arc<crate::Hub>,
    pub(crate) bind: String,
    pub(crate) ws_path: String,
    pub(crate) workspace_root: PathBuf,
}

/// Prepared runtime components before Hub creation.
pub(crate) struct PreparedRuntime {
    pub(crate) bind: String,
    pub(crate) ws_path: String,
    pub(crate) workspace_root: PathBuf,
    pub(crate) reloadable: crate::RuntimeReloadState,
    pub(crate) dream_manager: DreamManager,
    pub(crate) reload_summary: String,
}

/// Build the bootstrap state used when `sa.toml` does not exist yet.
pub(crate) fn build_bootstrap_state(config_path: &Path) -> anyhow::Result<BootstrapState> {
    let config_path = std::path::absolute(config_path).with_context(|| {
        format!(
            "Failed to resolve bootstrap config path: {}",
            config_path.display()
        )
    })?;
    let config_dir = config_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let workspace_root = config_dir.clone();
    let agents_md_path = workspace_root.join("Agents.md");

    Ok(BootstrapState {
        config_path,
        config_dir,
        workspace_root,
        agents_md_path,
        bind: DEFAULT_BOOTSTRAP_BIND.to_string(),
        ws_path: DEFAULT_BOOTSTRAP_WS_PATH.to_string(),
    })
}

/// Build one fully initialized runtime from an already parsed config object.
pub(crate) async fn prepare_runtime_from_config(
    cfg: Config,
    config_path: &Path,
) -> anyhow::Result<PreparedRuntime> {
    let config_dir = resolve_config_file_dir(config_path)?;
    let bind = cfg.server.bind.clone();
    let ws_path = cfg.server.ws_path.clone();

    // Resolve workspace paths.
    let workspace_root = cfg.workspace.root_dir_path(&config_dir);
    let agents_md_path = cfg.workspace.agents_md_path(&workspace_root);

    tracing::info!("Workspace root: {}", workspace_root.display());
    tracing::info!("Agents.md path: {}", agents_md_path.display());

    // Best-effort load `Agents.md` once for startup logging.
    match load_agents_md(agents_md_path.clone()).await {
        Ok(a) if a.found => {
            tracing::info!("Agents.md detected ({} bytes).", a.content.len());
        }
        Ok(_) => {
            tracing::info!("Agents.md not found at startup (this is OK).");
        }
        Err(err) => {
            tracing::warn!("Failed to read Agents.md at startup: {err}");
        }
    }

    // Discover skills.
    let skill_dirs = cfg.skills.dirs_as_paths();
    let mut skills_registry = SkillRegistry::scan(&skill_dirs)?;
    let local_skill_count = skills_registry.list().len();
    tracing::info!("Discovered {} local skill(s).", local_skill_count);

    // Connect external MCP servers before freezing the tool registry.
    let mut connected_mcp_servers = 0usize;
    let mcp_registry = if cfg.mcp.enabled && !cfg.mcp.servers.is_empty() {
        tracing::info!(
            "Initializing MCP client — {} server(s) configured",
            cfg.mcp.servers.len()
        );
        match McpRegistry::connect_all(&cfg.mcp.servers).await {
            Ok(registry) if !registry.is_empty() => {
                connected_mcp_servers = registry.server_count();
                skills_registry.extend_mcp_prompts(registry.prompt_commands());
                tracing::info!(
                    "MCP: {} tool(s) and {} prompt command(s) registered from {} server(s)",
                    registry.tool_count(),
                    registry.prompt_commands().len(),
                    registry.server_count()
                );
                Some(Arc::new(registry))
            }
            Ok(_) => {
                tracing::warn!("MCP enabled, but no MCP servers connected successfully.");
                None
            }
            Err(error) => {
                tracing::error!("MCP registry failed to initialize: {error:#}");
                None
            }
        }
    } else {
        None
    };
    let skills = Arc::new(skills_registry);

    // Build tools.
    let tool_ctx = ToolContext::new(workspace_root.clone(), Arc::clone(&skills))?;
    let preload_ctx = tool_ctx.clone();
    let tools = ToolExecutor::new(tool_ctx, mcp_registry);

    // Build LLM client.
    let system_role_name = cfg.llm.effective_system_role_name().to_string();
    let reasoning_effort = cfg.llm.effective_reasoning_effort().map(str::to_string);
    let reasoning_effort_summary = reasoning_effort
        .clone()
        .unwrap_or_else(|| "none".to_string());
    let llm_cfg_for_dream = cfg.llm.clone();
    let wire_api = cfg.llm.effective_wire_api();
    let auth_style = cfg.llm.effective_auth_style(wire_api);
    let llm = OpenAiClient::with_wire_api_and_auth_style(
        cfg.llm.base_url,
        cfg.llm.api_key,
        wire_api,
        auth_style,
    )?;

    // Build agent runner.
    let runner_cfg = sa_core::agent::AgentRunnerConfig {
        model: cfg.llm.model.clone(),
        system_role_name: system_role_name.clone(),
        reasoning_effort,
        compaction: cfg.compaction,
        max_output_tokens: cfg.llm.max_output_tokens,
        temperature: cfg.llm.temperature,
        top_p: cfg.llm.top_p,
        fallback_model: cfg.llm.fallback_model.clone(),
        max_consecutive_failures: cfg.llm.max_consecutive_failures,
        max_retries: cfg.llm.max_retries,
        model_routing: if cfg.model_routing.enabled {
            Some(cfg.model_routing.categories.clone())
        } else {
            None
        },
    };
    let dream_manager = DreamManager::new(workspace_root.clone(), cfg.dream.clone(), cfg.semantic_memory.clone(), cfg.procedure.clone(), Some(llm_cfg_for_dream))?;
    let runner = sa_core::agent::AgentRunner::new(llm, tools, Arc::clone(&skills), runner_cfg, cfg.working_memory.clone(), dream_manager.wake_handle());
    let max_concurrent_model_calls = cfg.team.max_concurrent_model_calls;
    let reload_summary = format!(
        "reloaded model={} system_role={} reasoning_effort={} local_skills={} mcp_servers={} agents_md={}",
        cfg.llm.model,
        system_role_name,
        reasoning_effort_summary,
        local_skill_count,
        connected_mcp_servers,
        agents_md_path.display()
    );

    Ok(PreparedRuntime {
        bind,
        ws_path,
        workspace_root: preload_ctx.workspace_root.clone(),
        reloadable: crate::RuntimeReloadState {
            runner,
            preload_ctx,
            agents_md_path,
            team_cfg: cfg.team,
            permissions: cfg.permissions,
            max_concurrent_model_calls,
        },
        dream_manager,
        reload_summary,
    })
}

/// Build one fully initialized runtime from an already parsed config object.
pub(crate) async fn build_runtime_from_config(
    cfg: Config,
    config_path: &Path,
) -> anyhow::Result<LoadedRuntime> {
    let prepared = prepare_runtime_from_config(cfg, config_path).await?;
    let runtime_store = RuntimeStore::new(prepared.workspace_root.clone())?;

    let root_agent_id = match runtime_store.load_root_marker()? {
        Some(marker) => marker.root_agent_id,
        None => {
            let root_id = Uuid::new_v4();
            runtime_store.save_root_marker(&RootMarker {
                root_agent_id: root_id,
            })?;
            root_id
        }
    };
    let team_state = match runtime_store.load_team_state()? {
        Some(state) => state,
        None => {
            let state = TeamState::new(root_agent_id);
            runtime_store.save_team_state(&state)?;
            state
        }
    };

    if runtime_store.load_agent_state(root_agent_id)?.is_none() {
        let root_session_store = SessionStore::new_in_relative_dir(
            prepared.workspace_root.clone(),
            Path::new(&format!("sessions/agents/{root_agent_id}")),
        )?;
        runtime_store.save_agent_state(&AgentState::new_root(
            root_agent_id,
            root_session_store.current_session_path(),
        ))?;
    }

    let session_store = Arc::new(SessionStore::new(prepared.workspace_root.clone())?);
    let interaction_store = Arc::new(InteractionStore::new(prepared.workspace_root.clone())?);
    let task_audit_store = Arc::new(TaskAuditStore::new(prepared.workspace_root.clone())?);
    let config_path = std::path::absolute(config_path)
        .with_context(|| format!("Failed to resolve config path: {}", config_path.display()))?;

    let hub = crate::Hub::new(
        config_path,
        prepared.bind.clone(),
        prepared.ws_path.clone(),
        prepared.workspace_root.clone(),
        prepared.reloadable,
        session_store,
        interaction_store,
        task_audit_store,
        runtime_store,
        team_state,
    );
    hub.recover_runtime().await?;
    hub.replace_dream_scheduler(prepared.dream_manager);

    Ok(LoadedRuntime {
        hub,
        bind: prepared.bind,
        ws_path: prepared.ws_path,
        workspace_root: prepared.workspace_root,
    })
}

/// Render the initial `sa.toml` contents from one frontend onboarding request.
pub(crate) fn render_initial_config_toml(
    bootstrap: &BootstrapState,
    request: &InitializeConfigRequest,
) -> anyhow::Result<String> {
    use sa_core::openai::{AuthStyle, WireApi};
    use sa_core::ws_protocol::InitMethod;

    let base_url = request.base_url.trim();
    let api_key = request.api_key.trim();
    let model = request.model.trim();
    if base_url.is_empty() {
        anyhow::bail!("`base_url` 不能为空");
    }
    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        anyhow::bail!("`base_url` 必须以 http:// 或 https:// 开头");
    }
    if api_key.is_empty() {
        anyhow::bail!("`api_key` 不能为空");
    }
    if model.is_empty() {
        anyhow::bail!("`model` 不能为空");
    }

    let wire_api = effective_init_wire_api(request);
    let auth_style = effective_init_auth_style(request, wire_api);
    let system_role_name = normalize_optional_string(request.system_role_name.clone());
    let reasoning_effort = normalize_optional_string(request.reasoning_effort.clone());

    let mut out = String::new();
    out.push_str("# Generated by SA first-run initialization.\n");
    out.push_str("# You can edit this file later if you need more advanced settings.\n\n");
    out.push_str("[llm]\n");
    out.push_str(&format!("base_url = {}\n", toml_string(base_url)));
    out.push_str(&format!("api_key = {}\n", toml_string(api_key)));
    out.push_str(&format!("model = {}\n", toml_string(model)));
    out.push_str(&format!(
        "wire_api = {}\n",
        toml_string(match wire_api {
            WireApi::ChatCompletions => "chat_completions",
            WireApi::Responses => "responses",
            WireApi::AnthropicMessages => "anthropic_messages",
        })
    ));
    out.push_str(&format!(
        "auth_style = {}\n",
        toml_string(match auth_style {
            AuthStyle::Bearer => "bearer",
            AuthStyle::XApiKey => "x_api_key",
            AuthStyle::AnthropicAuto => "anthropic_auto",
        })
    ));
    if let Some(system_role_name) = system_role_name {
        out.push_str(&format!(
            "system_role_name = {}\n",
            toml_string(&system_role_name)
        ));
    }
    if let Some(reasoning_effort) = reasoning_effort {
        out.push_str(&format!(
            "reasoning_effort = {}\n",
            toml_string(&reasoning_effort)
        ));
    }
    out.push('\n');
    out.push_str("[server]\n");
    out.push_str(&format!("bind = {}\n", toml_string(&bootstrap.bind)));
    out.push_str(&format!(
        "ws_path = {}\n\n",
        toml_string(&bootstrap.ws_path)
    ));
    out.push_str("[workspace]\n");
    out.push_str(&format!(
        "root_dir = {}\n",
        toml_string(
            bootstrap
                .workspace_root
                .strip_prefix(&bootstrap.config_dir)
                .ok()
                .and_then(|path| if path.as_os_str().is_empty() {
                    Some(".")
                } else {
                    path.to_str()
                })
                .unwrap_or(".")
        )
    ));
    out.push_str(&format!(
        "agents_md = {}\n",
        toml_string(
            bootstrap
                .agents_md_path
                .strip_prefix(&bootstrap.workspace_root)
                .ok()
                .and_then(Path::to_str)
                .unwrap_or("Agents.md")
        )
    ));
    let dream = sa_core::dream::DreamConfig::default();
    out.push_str("\n[dream]\n");
    out.push_str(&format!("enabled = {}\n", dream.enabled));
    out.push_str(&format!(
        "daily_note_lookback_days = {}\n",
        dream.daily_note_lookback_days
    ));
    out.push_str(&format!(
        "recent_session_segments = {}\n",
        dream.recent_session_segments
    ));
    out.push_str(&format!(
        "recent_topic_files = {}\n",
        dream.recent_topic_files
    ));
    let team = TeamConfig::default();
    out.push_str("\n[team]\n");
    out.push_str(&format!("auto_resume = {}\n", team.auto_resume));
    out.push_str(&format!("max_active_agents = {}\n", team.max_active_agents));
    out.push_str(&format!(
        "max_concurrent_model_calls = {}\n",
        team.max_concurrent_model_calls
    ));

    Ok(out)
}
