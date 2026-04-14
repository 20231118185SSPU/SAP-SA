//! Backwards-compatible re-export layer for the legacy `skills` module name.
//!
//! The implementation moved to `commands.rs`, but much of the existing SA code
//! still imports `crate::skills::SkillRegistry`. Keeping this shim avoids a
//! large mechanical rename while the rest of the runtime is upgraded.

pub use crate::commands::*;
