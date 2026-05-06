//! `sa-core` is the shared implementation of the StudyAdministrator (SA) agent.
//!
//! Goals (from the user request):
//! - Call an **OpenAI-compatible** API endpoint.
//! - Read `Agents.md` and inject it as the "system instructions".
//! - Discover and load `SKILL.md` skills (Codex/Agents skill format).
//! - Run a minimal but **fully autonomous** agent loop via tool calling.
//! - Provide a stable JSON protocol used by the WebSocket daemon and CLI.

// We keep the crate as "small building blocks" so the daemon/CLI stay thin.

pub mod adversary;
pub mod agent;
pub mod agents_md;
pub mod cache;
pub mod cache_monitor;
pub mod bash_safety;
pub mod cancel;
pub mod cold_store;
pub mod commands;
pub mod compact;
pub mod config;
pub mod cost_budget;
pub mod dream;
pub mod fetch_safety;
pub mod field_encryption;
pub mod file_analyzer;
pub mod file_index;
pub mod file_search;
pub mod interaction_history;
pub mod mcp_client;
pub mod mcp_protocol;
pub mod mcp_transport;
pub mod memory;
pub mod memory_filter;
pub mod memory_metabolism;
pub mod memory_pointer;
pub mod memory_scope;
pub mod skill_metabolism;
pub mod timeline_retrieval;
pub mod noise_assessment;
pub mod mmr_rerank;
pub mod search_feedback;
pub mod semantic_memory;
pub mod procedural_memory;
pub mod memory_indexer;
pub mod memory_l1_index;
pub mod vector_store;
pub mod index;
pub mod openai;
pub mod path_guard;
pub mod pii_detector;
pub mod retry;
pub mod runtime;
pub mod search_backends;
pub mod session;


pub mod skills;
pub mod skill_search;
pub mod task_audit;
pub mod tool_cache;
pub mod tools;
pub mod working_memory;
pub mod ws_identity;
pub mod ws_protocol;pub mod workflow;
pub mod workflow_engine;
pub mod plan_engine;
pub mod crystallizer;

