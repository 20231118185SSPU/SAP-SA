//! L2 Cold Storage + Soft-Delete Recovery.
//!
//! Tiered memory:
//! - L1 (Hot):  `memory/YYYY-MM-DD.md` — daily, full search
//! - L2 (Cold): `memory/archive/YYYY-MM.md` — monthly archive, reduced search
//! - Deleted:   `memory/deleted/YYYY-MM-DD_HH-mm-ss_<hash>.md` — tombstones

use crate::memory::{
    extract_entries_from_daily, generate_yaml_front_matter,
    MemoryMetadata,
};
use crate::memory_scope::MemoryScope;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// L2 archive directory.
pub const MEMORY_ARCHIVE_DIR: &str = "memory/archive";

/// Soft-delete tombstone directory.
pub const MEMORY_DELETED_DIR: &str = "memory/deleted";

// ---- config ----------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ColdStorageConfig {
    pub archive_after_days: u32,
    pub archive_importance_threshold: f64,
    pub recovery_days: u32,
    pub enable_hard_delete: bool,
    pub max_entries_per_archive: usize,
}

impl Default for ColdStorageConfig {
    fn default() -> Self {
        Self {
            archive_after_days: 30,
            archive_importance_threshold: 0.3,
            recovery_days: 30,
            enable_hard_delete: true,
            max_entries_per_archive: 500,
        }
    }
}

// ---- result types ----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ArchiveResult {
    pub archived_count: usize,
    pub archived_files: Vec<String>,
    pub skipped_count: usize,
}

#[derive(Debug, Clone)]
pub struct RestoreResult {
    pub restored_count: usize,
    pub restored_to: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SoftDeleteResult {
    pub success: bool,
    pub tombstone_path: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub recovered_to: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletedEntry {
    pub original_path: String,
    pub deleted_at: String,
    pub reason: String,
    pub content_hash: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    pub archive_path: String,
    pub original_date: String,
    pub created_at: String,
    pub importance: Option<f64>,
    pub content_preview: String,
}

// =============================================================================
// L1 → L2 Archive
// =============================================================================

/// Move old low-importance L1 entries into monthly L2 archive files.
pub fn archive_old_entries(
    workspace_root: &Path,
    config: &ColdStorageConfig,
    now: &chrono::DateTime<chrono::Local>,
) -> anyhow::Result<ArchiveResult> {
    let memory_dir = workspace_root.join("memory");
    let archive_dir = workspace_root.join(MEMORY_ARCHIVE_DIR);
    if !archive_dir.exists() {
        std::fs::create_dir_all(&archive_dir)?;
    }

    let cutoff = now.date_naive() - chrono::Days::new(config.archive_after_days as u64);
    let mut total = 0usize;
    let mut archived = Vec::new();
    let mut skipped = 0usize;

    for e in std::fs::read_dir(&memory_dir)? {
        let e = e?;
        let p = e.path();
        let fname = match p.file_stem().and_then(|s| s.to_str()) {
            Some(f) => f,
            None => continue,
        };
        // Only YYYY-MM-DD daily files
        if fname.len() != 10 || !fname.chars().all(|c| c.is_ascii_digit() || c == '-') {
            continue;
        }
        let date = match chrono::NaiveDate::parse_from_str(fname, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => continue,
        };
        if date >= cutoff {
            continue;
        }

        let content = match std::fs::read_to_string(&p) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let entries = extract_entries_from_daily(&content);
        if entries.is_empty() {
            continue;
        }

        let (to_archive, to_keep): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .partition(|(_b, m)| {
                m.importance.map_or(true, |imp| imp < config.archive_importance_threshold)
                    && m.scope != Some(MemoryScope::Deleted)
            });

        if to_archive.is_empty() {
            skipped += 1;
            continue;
        }

        let month_key = date.format("%Y-%m").to_string();
        let ap = archive_dir.join(format!("{}.md", month_key));
        let mut ac = if ap.exists() {
            std::fs::read_to_string(&ap).unwrap_or_default()
        } else {
            String::new()
        };

        for (body, meta) in &to_archive {
            let ca = meta.created_at.clone().unwrap_or_else(|| {
                chrono::Utc::now()
                    .format("%Y-%m-%dT%H:%M:%S+08:00")
                    .to_string()
            });
            ac.push_str(&generate_yaml_front_matter(meta, &ca));
            ac.push_str(body);
            ac.push('\n');
            total += 1;
        }
        std::fs::write(&ap, ac)?;
        archived.push(format!("memory/{}.md → archive/{}.md", fname, month_key));

        // Rewrite daily
        if to_keep.is_empty() {
            std::fs::remove_file(&p)?;
        } else {
            let mut nc = String::new();
            for (body, meta) in &to_keep {
                let ca = meta.created_at.clone().unwrap_or_else(|| {
                    chrono::Utc::now()
                        .format("%Y-%m-%dT%H:%M:%S+08:00")
                        .to_string()
                });
                nc.push_str(&generate_yaml_front_matter(meta, &ca));
                nc.push_str(body);
                nc.push('\n');
            }
            std::fs::write(&p, nc)?;
        }
    }

    Ok(ArchiveResult {
        archived_count: total,
        archived_files: archived,
        skipped_count: skipped,
    })
}

/// Restore entries from a monthly L2 archive back to L1.
pub fn restore_from_archive(
    workspace_root: &Path,
    archive_file: &str,
    target_date: &str,
) -> anyhow::Result<RestoreResult> {
    let ap = workspace_root.join(archive_file);
    if !ap.exists() {
        return Ok(RestoreResult {
            restored_count: 0,
            restored_to: None,
            error: Some(format!("archive not found: {archive_file}")),
        });
    }

    let content = std::fs::read_to_string(&ap)?;
    let entries = extract_entries_from_daily(&content);

    let (to_restore, to_keep): (Vec<_>, Vec<_>) = entries.into_iter().partition(|(_b, m)| {
        m.created_at
            .as_ref()
            .and_then(|dt| dt.split('T').next())
            .map_or(false, |d| d == target_date)
    });

    if to_restore.is_empty() {
        return Ok(RestoreResult {
            restored_count: 0,
            restored_to: None,
            error: Some(format!("no entries for date: {target_date}")),
        });
    }

    let daily_path = workspace_root
        .join("memory")
        .join(format!("{target_date}.md"));
    let mut dc = if daily_path.exists() {
        std::fs::read_to_string(&daily_path).unwrap_or_default()
    } else {
        String::new()
    };

    for (body, meta) in &to_restore {
        let ca = meta.created_at.clone().unwrap_or_else(|| {
            chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S+08:00")
                .to_string()
        });
        dc.push_str(&generate_yaml_front_matter(meta, &ca));
        dc.push_str(body);
        dc.push('\n');
    }
    std::fs::write(&daily_path, dc)?;

    // Rewrite archive
    if to_keep.is_empty() {
        std::fs::remove_file(&ap)?;
    } else {
        let mut na = String::new();
        for (body, meta) in &to_keep {
            let ca = meta.created_at.clone().unwrap_or_else(|| {
                chrono::Utc::now()
                    .format("%Y-%m-%dT%H:%M:%S+08:00")
                    .to_string()
            });
            na.push_str(&generate_yaml_front_matter(meta, &ca));
            na.push_str(body);
            na.push('\n');
        }
        std::fs::write(&ap, na)?;
    }

    Ok(RestoreResult {
        restored_count: to_restore.len(),
        restored_to: Some(format!("memory/{target_date}.md")),
        error: None,
    })
}

/// List all L2 archive entries.
pub fn scan_archive_entries(workspace_root: &Path) -> anyhow::Result<Vec<ArchiveEntry>> {
    let ad = workspace_root.join(MEMORY_ARCHIVE_DIR);
    if !ad.exists() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for e in std::fs::read_dir(&ad)? {
        let e = e?;
        let fname = e
            .path()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let content = match std::fs::read_to_string(e.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for (_, meta) in extract_entries_from_daily(&content) {
            let od = meta
                .created_at
                .as_ref()
                .and_then(|dt| dt.split('T').next())
                .unwrap_or(&fname)
                .to_string();
            out.push(ArchiveEntry {
                archive_path: format!("memory/archive/{fname}.md"),
                original_date: od,
                created_at: meta.created_at.clone().unwrap_or_default(),
                importance: meta.importance,
                content_preview: meta.summary.unwrap_or_else(|| "(no summary)".into()),
            });
        }
    }
    Ok(out)
}

// =============================================================================
// Soft-Delete / Recovery
// =============================================================================

fn content_short_hash(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
        .chars()
        .take(8)
        .collect()
}

/// Soft-delete an entry: tombstone in `memory/deleted/`, remove from daily.
pub fn soft_delete_entry(
    workspace_root: &Path,
    daily_file: &str,
    entry_idx: usize,
    reason: &str,
) -> anyhow::Result<SoftDeleteResult> {
    let dp = workspace_root.join(daily_file);
    if !dp.exists() {
        return Ok(SoftDeleteResult {
            success: false,
            tombstone_path: None,
            reason: format!("file not found: {daily_file}"),
        });
    }

    let content = std::fs::read_to_string(&dp)?;
    let entries = extract_entries_from_daily(&content);
    if entry_idx >= entries.len() {
        return Ok(SoftDeleteResult {
            success: false,
            tombstone_path: None,
            reason: format!("index {entry_idx} out of range ({})", entries.len()),
        });
    }

    let (body, mut meta) = entries[entry_idx].clone();
    meta.scope = Some(MemoryScope::Deleted);

    // Write tombstone
    let dd = workspace_root.join(MEMORY_DELETED_DIR);
    std::fs::create_dir_all(&dd)?;

    let ts = chrono::Local::now()
        .format("%Y-%m-%d_%H-%M-%S")
        .to_string();
    let hash = content_short_hash(&body);
    let tn = format!("{ts}_{hash}.md");
    let tp = dd.join(&tn);

    let tomb = DeletedEntry {
        original_path: daily_file.to_string(),
        deleted_at: chrono::Local::now().to_rfc3339(),
        reason: reason.to_string(),
        content_hash: content_short_hash(&body),
        content: body.clone(),
    };
    let y = serde_yaml::to_string(&tomb)?;
    std::fs::write(&tp, format!("---\n{y}---\n"))?;

    // Rebuild daily without deleted entry
    let mut nc = String::new();
    for (i, (b, m)) in entries.iter().enumerate() {
        if i == entry_idx {
            continue;
        }
        let ca = m.created_at.clone().unwrap_or_else(|| {
            chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S+08:00")
                .to_string()
        });
        nc.push_str(&generate_yaml_front_matter(m, &ca));
        nc.push_str(b);
        nc.push('\n');
    }

    if nc.is_empty() {
        std::fs::remove_file(&dp)?;
    } else {
        std::fs::write(&dp, nc)?;
    }

    Ok(SoftDeleteResult {
        success: true,
        tombstone_path: Some(format!("memory/deleted/{tn}")),
        reason: reason.to_string(),
    })
}

/// Recover a soft-deleted entry from its tombstone.
pub fn recover_deleted_entry(
    workspace_root: &Path,
    tombstone_file: &str,
) -> anyhow::Result<RecoveryResult> {
    let tp = workspace_root.join(tombstone_file);
    if !tp.exists() {
        return Ok(RecoveryResult {
            success: false,
            recovered_to: None,
            error: Some(format!("tombstone not found: {tombstone_file}")),
        });
    }

    let raw = std::fs::read_to_string(&tp)?;
    let tomb: DeletedEntry = serde_yaml::from_str(
        raw.strip_prefix("---\n")
            .and_then(|s| s.strip_suffix("---\n"))
            .unwrap_or(&raw)
    ).unwrap_or_else(|_| serde_yaml::from_str(&raw).unwrap_or_else(|e| {
        panic!("bad tombstone {tombstone_file}: {e}")
    }));

    // Append back to original daily
    let dp = workspace_root.join(&tomb.original_path);
    let dc = if dp.exists() {
        std::fs::read_to_string(&dp).unwrap_or_default()
    } else {
        String::new()
    };

    let mut rm = MemoryMetadata::default();
    rm.created_at = Some(tomb.deleted_at.clone());
    rm.scope = Some(MemoryScope::User);

    std::fs::write(
        &dp,
        format!(
            "{dc}{fm}{body}\n",
            fm = generate_yaml_front_matter(&rm, &tomb.deleted_at),
            body = tomb.content,
        ),
    )?;
    std::fs::remove_file(&tp)?;

    Ok(RecoveryResult {
        success: true,
        recovered_to: Some(tomb.original_path),
        error: None,
    })
}

/// List all soft-deleted entries.
pub fn list_deleted_entries(workspace_root: &Path) -> anyhow::Result<Vec<DeletedEntry>> {
    let dd = workspace_root.join(MEMORY_DELETED_DIR);
    if !dd.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dd)? {
        let e = e?;
        let raw = match std::fs::read_to_string(e.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Ok(tomb) = serde_yaml::from_str::<DeletedEntry>(
            raw.strip_prefix("---\n")
                .and_then(|s| s.strip_suffix("---\n"))
                .unwrap_or(&raw),
        ) {
            out.push(tomb);
        }
    }
    Ok(out)
}

/// Permanently delete expired tombstones (beyond `recovery_days`).
pub fn hard_delete_expired(
    workspace_root: &Path,
    recovery_days: u32,
) -> anyhow::Result<usize> {
    let dd = workspace_root.join(MEMORY_DELETED_DIR);
    if !dd.exists() {
        return Ok(0);
    }

    let cutoff = chrono::Local::now().date_naive()
        - chrono::Days::new(recovery_days as u64);
    let mut removed = 0usize;

    for e in std::fs::read_dir(&dd)? {
        let e = e?;
        let raw = match std::fs::read_to_string(e.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let tomb: DeletedEntry = match serde_yaml::from_str(
            raw.strip_prefix("---\n")
                .and_then(|s| s.strip_suffix("---\n"))
                .unwrap_or(&raw),
        ) {
            Ok(t) => t,
            Err(_) => continue,
        };

        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&tomb.deleted_at) {
            if dt.date_naive() < cutoff {
                std::fs::remove_file(e.path())?;
                removed += 1;
            }
        }
    }

    Ok(removed)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_consistent() {
        let a = content_short_hash("hello");
        let b = content_short_hash("hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn cold_store_config_defaults() {
        let c = ColdStorageConfig::default();
        assert_eq!(c.archive_after_days, 30);
        assert_eq!(c.recovery_days, 30);
    }

    #[test]
    fn archive_result_format() {
        let r = ArchiveResult {
            archived_count: 5,
            archived_files: vec!["a".into()],
            skipped_count: 2,
        };
        assert_eq!(r.archived_count, 5);
        assert_eq!(r.skipped_count, 2);
    }
}
