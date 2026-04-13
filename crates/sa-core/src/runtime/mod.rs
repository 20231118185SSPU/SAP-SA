//! Durable runtime types for the StudyAdministrator (SA) multi-agent runtime.
//!
//! This module intentionally starts with serializable state models first so the
//! daemon can evolve from a single in-memory worker into a restart-safe actor
//! system without coupling every caller to one concrete storage layout.

pub mod state;
pub mod store;
