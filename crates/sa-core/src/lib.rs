//! `sa-core` is the shared implementation of the StudyAdministrator (SA) agent.
//!
//! Goals (from the user request):
//! - Call an **OpenAI-compatible** API endpoint.
//! - Read `Agents.md` and inject it as the "system instructions".
//! - Discover and load `SKILL.md` skills (Codex/Agents skill format).
//! - Run a minimal but **fully autonomous** agent loop via tool calling.
//! - Provide a stable JSON protocol used by the WebSocket daemon and CLI.

// We keep the crate as "small building blocks" so the daemon/CLI stay thin.

pub mod agent;
pub mod agents_md;
pub mod cancel;
pub mod compact;
pub mod config;
pub mod dream;
pub mod mcp_client;
pub mod mcp_protocol;
pub mod mcp_transport;
pub mod memory;
pub mod openai;
pub mod retry;
pub mod session;
pub mod skills;
pub mod tools;
pub mod ws_identity;
pub mod ws_protocol;
