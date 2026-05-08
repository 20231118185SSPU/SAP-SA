// ── memory_store.rs ─────────────────────────────────────────────────────────
//! Unified SQLite memory store replacing the fragmented file-based system.
//!
//! Single database with three tables:
//! - `memories` — episodic memory entries (replaces memory/*.md files)
//! - `facts` — semantic knowledge graph triples (replaces semantic_memory.rs)
//! - `pinned_slots` — working memory pins (replaces in-memory Vec)
//!
//! Full-text search via FTS5 for BM25-ranked retrieval.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Data models ─────────────────────────────────────────────────────────────

/// Episodic memory entry stored in the `memories` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub scope: String,
    pub importance: f64,
    pub emotion_valence: f64,
    pub source: String,
    pub access_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
    pub accessed_at: i64,
}

/// Partial update for a memory entry.
#[derive(Debug, Clone, Default)]
pub struct MemoryPatch {
    pub title: Option<String>,
    pub content: Option<String>,
    pub tags: Option<Vec<String>>,
    pub scope: Option<String>,
    pub importance: Option<f64>,
    pub emotion_valence: Option<f64>,
}

/// Semantic knowledge graph triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Working memory pinned slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedSlot {
    pub key: String,
    pub label: String,
    pub content: String,
    pub pinned_at: i64,
    pub note: Option<String>,
}

/// Aggregate statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: i64,
    pub total_facts: i64,
    pub unique_subjects: i64,
    pub unique_predicates: i64,
    pub avg_confidence: f64,
    pub total_pins: i64,
}

// ── Store ───────────────────────────────────────────────────────────────────

/// Thread-safe handle to the unified SQLite memory database.
pub struct MemoryStore {
    conn: Mutex<Connection>,
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStore").finish()
    }
}

impl MemoryStore {
    /// Open (or create) the database at `db_path` and run migrations.
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create DB directory: {}", parent.display()))?;
        }

        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open memory DB: {}", db_path.display()))?;

        // Enable WAL mode for concurrent read performance.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let store = Self {
            conn: Mutex::new(conn),
        };
        store.run_migrations()?;
        Ok(store)
    }

    /// Create an in-memory database (for testing).
    pub fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.run_migrations()?;
        Ok(store)
    }

    fn run_migrations(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                content         TEXT NOT NULL,
                tags            TEXT DEFAULT '[]',
                scope           TEXT NOT NULL DEFAULT 'user',
                importance      REAL NOT NULL DEFAULT 0.5,
                emotion_valence REAL DEFAULT 0.0,
                source          TEXT DEFAULT '',
                access_count    INTEGER NOT NULL DEFAULT 0,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL,
                accessed_at     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope);
            CREATE INDEX IF NOT EXISTS idx_mem_imp ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);

            CREATE TABLE IF NOT EXISTS facts (
                id          TEXT PRIMARY KEY,
                subject     TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                object      TEXT NOT NULL,
                confidence  REAL NOT NULL DEFAULT 1.0,
                source      TEXT DEFAULT '',
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_sub ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_pred ON facts(predicate);
            CREATE INDEX IF NOT EXISTS idx_facts_obj ON facts(object);
            CREATE INDEX IF NOT EXISTS idx_facts_conf ON facts(confidence);

            CREATE TABLE IF NOT EXISTS pinned_slots (
                key       TEXT PRIMARY KEY,
                label     TEXT NOT NULL DEFAULT '',
                content   TEXT NOT NULL,
                pinned_at INTEGER NOT NULL,
                note      TEXT
            );
            ",
        )?;

        // FTS5 virtual table — created separately because it doesn't support IF NOT EXISTS
        // in all SQLite versions. Use a sentinel to avoid re-creation.
        let fts_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='memories_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(false);

        if !fts_exists {
            conn.execute_batch(
                "CREATE VIRTUAL TABLE memories_fts USING fts5(
                    title, content, tags,
                    content='memories',
                    content_rowid='rowid'
                );
                -- Triggers to keep FTS in sync with the content table
                CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, title, content, tags)
                    VALUES (new.rowid, new.title, new.content, new.tags);
                END;
                CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
                    VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                END;
                CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
                    VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, title, content, tags)
                    VALUES (new.rowid, new.title, new.content, new.tags);
                END;
                ",
            )?;
        }

        Ok(())
    }

    // ── Episodic memory CRUD ────────────────────────────────────────────────

    pub fn insert_memory(&self, entry: &MemoryEntry) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tags_json = serde_json::to_string(&entry.tags).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT OR REPLACE INTO memories
                (id, title, content, tags, scope, importance, emotion_valence,
                 source, access_count, created_at, updated_at, accessed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                entry.id,
                entry.title,
                entry.content,
                tags_json,
                entry.scope,
                entry.importance,
                entry.emotion_valence,
                entry.source,
                entry.access_count,
                entry.created_at,
                entry.updated_at,
                entry.accessed_at,
            ],
        )
        .context("Failed to insert memory")?;
        Ok(())
    }

    pub fn update_memory(&self, id: &str, patch: &MemoryPatch) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let now = now_ts();
        let mut sets = vec!["updated_at = ?1".to_string()];
        let mut param_index = 2u32;

        if patch.title.is_some() {
            sets.push(format!("title = ?{param_index}"));
            param_index += 1;
        }
        if patch.content.is_some() {
            sets.push(format!("content = ?{param_index}"));
            param_index += 1;
        }
        if patch.tags.is_some() {
            sets.push(format!("tags = ?{param_index}"));
            param_index += 1;
        }
        if patch.scope.is_some() {
            sets.push(format!("scope = ?{param_index}"));
            param_index += 1;
        }
        if patch.importance.is_some() {
            sets.push(format!("importance = ?{param_index}"));
            param_index += 1;
        }
        if patch.emotion_valence.is_some() {
            sets.push(format!("emotion_valence = ?{param_index}"));
            param_index += 1;
        }

        let sql = format!(
            "UPDATE memories SET {} WHERE id = ?{param_index}",
            sets.join(", ")
        );

        // Build params dynamically — rusqlite doesn't support dynamic param counts
        // cleanly, so we use a Value-based approach.
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(now)];
        if let Some(ref v) = patch.title {
            param_values.push(Box::new(v.clone()));
        }
        if let Some(ref v) = patch.content {
            param_values.push(Box::new(v.clone()));
        }
        if let Some(ref v) = patch.tags {
            let json = serde_json::to_string(v).unwrap_or_else(|_| "[]".into());
            param_values.push(Box::new(json));
        }
        if let Some(ref v) = patch.scope {
            param_values.push(Box::new(v.clone()));
        }
        if let Some(v) = patch.importance {
            param_values.push(Box::new(v));
        }
        if let Some(v) = patch.emotion_valence {
            param_values.push(Box::new(v));
        }
        param_values.push(Box::new(id.to_string()));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let changed = conn
            .execute(&sql, param_refs.as_slice())
            .context("Failed to update memory")?;
        Ok(changed > 0)
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryEntry>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, title, content, tags, scope, importance, emotion_valence,
                    source, access_count, created_at, updated_at, accessed_at
             FROM memories WHERE id = ?1",
            params![id],
            |row| row_to_entry(row),
        )
        .optional()
        .context("Failed to get memory")
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .context("Failed to delete memory")?;
        Ok(n > 0)
    }

    pub fn touch_memory(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = now_ts();
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, accessed_at = ?1 WHERE id = ?2",
            params![now, id],
        )
        .context("Failed to touch memory")?;
        Ok(())
    }

    pub fn list_memories(
        &self,
        scope: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let conn = self.conn.lock().unwrap();
        let (sql, param): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match scope {
            Some(s) => (
                "SELECT id, title, content, tags, scope, importance, emotion_valence,
                        source, access_count, created_at, updated_at, accessed_at
                 FROM memories WHERE scope = ?1 ORDER BY updated_at DESC LIMIT ?2 OFFSET ?3"
                    .into(),
                vec![
                    Box::new(s.to_string()),
                    Box::new(limit as i64),
                    Box::new(offset as i64),
                ],
            ),
            None => (
                "SELECT id, title, content, tags, scope, importance, emotion_valence,
                        source, access_count, created_at, updated_at, accessed_at
                 FROM memories ORDER BY updated_at DESC LIMIT ?1 OFFSET ?2"
                    .into(),
                vec![Box::new(limit as i64), Box::new(offset as i64)],
            ),
        };
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), |row| row_to_entry(row))?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }

    // ── FTS5 search ─────────────────────────────────────────────────────────

    /// Full-text search using FTS5 BM25 ranking.
    pub fn search_memories(
        &self,
        query: &str,
        scope: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let conn = self.conn.lock().unwrap();

        // FTS5 query — escape double quotes in user input
        let fts_query = query.replace('"', "\"\"");

        let (sql, param): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match scope {
            Some(s) => (
                "SELECT m.id, m.title, m.content, m.tags, m.scope, m.importance,
                        m.emotion_valence, m.source, m.access_count,
                        m.created_at, m.updated_at, m.accessed_at
                 FROM memories m
                 INNER JOIN memories_fts fts ON fts.rowid = m.rowid
                 WHERE memories_fts MATCH ?1 AND m.scope = ?2
                 ORDER BY rank
                 LIMIT ?3"
                    .into(),
                vec![
                    Box::new(fts_query),
                    Box::new(s.to_string()),
                    Box::new(limit as i64),
                ],
            ),
            None => (
                "SELECT m.id, m.title, m.content, m.tags, m.scope, m.importance,
                        m.emotion_valence, m.source, m.access_count,
                        m.created_at, m.updated_at, m.accessed_at
                 FROM memories m
                 INNER JOIN memories_fts fts ON fts.rowid = m.rowid
                 WHERE memories_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2"
                    .into(),
                vec![Box::new(fts_query), Box::new(limit as i64)],
            ),
        };

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), |row| row_to_entry(row))?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }

    // ── Semantic facts ──────────────────────────────────────────────────────

    pub fn upsert_fact(&self, fact: &Fact) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO facts (id, subject, predicate, object, confidence, source, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(id) DO UPDATE SET
                 confidence = excluded.confidence,
                 source = excluded.source,
                 updated_at = excluded.updated_at",
            params![
                fact.id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                fact.source,
                fact.created_at,
                fact.updated_at,
            ],
        )
        .context("Failed to upsert fact")?;
        Ok(())
    }

    pub fn query_facts(&self, query: &str, limit: usize) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{query}%");
        let mut stmt = conn.prepare(
            "SELECT id, subject, predicate, object, confidence, source, created_at, updated_at
             FROM facts
             WHERE subject LIKE ?1 OR predicate LIKE ?1 OR object LIKE ?1
             ORDER BY confidence DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![pattern, limit as i64], |row| row_to_fact(row))?;
        let mut facts = Vec::new();
        for row in rows {
            facts.push(row?);
        }
        Ok(facts)
    }

    /// BFS graph traversal starting from `subject`.
    pub fn traverse_graph(&self, subject: &str, max_hops: usize) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().unwrap();
        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![subject.to_string()];
        let mut result = Vec::new();
        let mut stmt = conn.prepare(
            "SELECT id, subject, predicate, object, confidence, source, created_at, updated_at
             FROM facts WHERE subject = ?1 ORDER BY confidence DESC",
        )?;

        for _ in 0..max_hops {
            if frontier.is_empty() {
                break;
            }
            let mut next_frontier = Vec::new();
            for subj in &frontier {
                if !visited.insert(subj.clone()) {
                    continue;
                }
                let rows = stmt.query_map(params![subj], |row| row_to_fact(row))?;
                for row in rows {
                    let fact = row?;
                    next_frontier.push(fact.object.clone());
                    result.push(fact);
                }
            }
            frontier = next_frontier;
        }
        Ok(result)
    }

    /// Remove facts with confidence below threshold.
    pub fn prune_facts(&self, min_confidence: f64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let n = conn
            .execute(
                "DELETE FROM facts WHERE confidence < ?1",
                params![min_confidence],
            )
            .context("Failed to prune facts")?;
        Ok(n)
    }

    /// Apply exponential decay to all fact confidences.
    pub fn decay_facts(&self, decay_rate: f64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let now = now_ts();
        // Decay: confidence *= e^(-rate * days_since_update)
        let n = conn.execute(
            "UPDATE facts SET confidence = confidence * EXP(?1 * (updated_at - ?2) / -86400.0)
             WHERE confidence > 0.01",
            params![decay_rate, now],
        )?;
        Ok(n)
    }

    // ── Working memory pins ─────────────────────────────────────────────────

    pub fn pin(&self, key: &str, label: &str, content: &str, note: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = now_ts();
        conn.execute(
            "INSERT OR REPLACE INTO pinned_slots (key, label, content, pinned_at, note) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![key, label, content, now, note],
        )
        .context("Failed to pin slot")?;
        Ok(())
    }

    pub fn unpin(&self, key: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn
            .execute("DELETE FROM pinned_slots WHERE key = ?1", params![key])
            .context("Failed to unpin slot")?;
        Ok(n > 0)
    }

    pub fn list_pins(&self) -> Result<Vec<PinnedSlot>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT key, label, content, pinned_at, note FROM pinned_slots ORDER BY pinned_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(PinnedSlot {
                key: row.get(0)?,
                label: row.get(1)?,
                content: row.get(2)?,
                pinned_at: row.get(3)?,
                note: row.get(4)?,
            })
        })?;
        let mut slots = Vec::new();
        for row in rows {
            slots.push(row?);
        }
        Ok(slots)
    }

    // ── Maintenance ─────────────────────────────────────────────────────────

    /// Delete memories with low importance not accessed within `max_age_days`.
    pub fn cleanup_expired(&self, max_age_days: u32, min_importance: f64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let cutoff = now_ts() - (max_age_days as i64 * 86400);
        let n = conn
            .execute(
                "DELETE FROM memories WHERE importance < ?1 AND accessed_at < ?2",
                params![min_importance, cutoff],
            )
            .context("Failed to cleanup expired memories")?;
        Ok(n)
    }

    /// Aggregate statistics.
    pub fn stats(&self) -> Result<MemoryStats> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT
                (SELECT COUNT(*) FROM memories),
                (SELECT COUNT(*) FROM facts),
                (SELECT COUNT(DISTINCT subject) FROM facts),
                (SELECT COUNT(DISTINCT predicate) FROM facts),
                (SELECT COALESCE(AVG(confidence), 0.0) FROM facts),
                (SELECT COUNT(*) FROM pinned_slots)",
            [],
            |r| {
                Ok(MemoryStats {
                    total_memories: r.get(0)?,
                    total_facts: r.get(1)?,
                    unique_subjects: r.get(2)?,
                    unique_predicates: r.get(3)?,
                    avg_confidence: r.get(4)?,
                    total_pins: r.get(5)?,
                })
            },
        )
        .context("Failed to get memory stats")
    }

    /// Generate a deterministic fact ID from (subject, predicate, object).
    pub fn fact_id(subject: &str, predicate: &str, object: &str) -> String {
        use std::io::Write;
        let mut hasher = blake3::Hasher::new();
        hasher.write_all(subject.as_bytes()).ok();
        hasher.write_all(b"|").ok();
        hasher.write_all(predicate.as_bytes()).ok();
        hasher.write_all(b"|").ok();
        hasher.write_all(object.as_bytes()).ok();
        let hash = hasher.finalize();
        hex::encode(&hash.as_bytes()[..8])
    }

    /// Generate a memory entry ID from title + content hash.
    pub fn memory_id(title: &str, content: &str) -> String {
        use std::io::Write;
        let mut hasher = blake3::Hasher::new();
        hasher.write_all(title.as_bytes()).ok();
        hasher.write_all(b"|").ok();
        hasher.write_all(content.as_bytes()).ok();
        let hash = hasher.finalize();
        hex::encode(&hash.as_bytes()[..8])
    }

    /// Access the raw connection for migration purposes.
    pub(crate) fn raw_conn(&self) -> &Mutex<Connection> {
        &self.conn
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn row_to_entry(row: &rusqlite::Row) -> rusqlite::Result<MemoryEntry> {
    let tags_json: String = row.get(3)?;
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    Ok(MemoryEntry {
        id: row.get(0)?,
        title: row.get(1)?,
        content: row.get(2)?,
        tags,
        scope: row.get(4)?,
        importance: row.get(5)?,
        emotion_valence: row.get(6)?,
        source: row.get(7)?,
        access_count: row.get(8)?,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
        accessed_at: row.get(11)?,
    })
}

fn row_to_fact(row: &rusqlite::Row) -> rusqlite::Result<Fact> {
    Ok(Fact {
        id: row.get(0)?,
        subject: row.get(1)?,
        predicate: row.get(2)?,
        object: row.get(3)?,
        confidence: row.get(4)?,
        source: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> MemoryStore {
        MemoryStore::new_in_memory().unwrap()
    }

    fn sample_entry(title: &str) -> MemoryEntry {
        let now = now_ts();
        MemoryEntry {
            id: MemoryStore::memory_id(title, "content"),
            title: title.into(),
            content: "test content".into(),
            tags: vec!["tag1".into()],
            scope: "user".into(),
            importance: 0.5,
            emotion_valence: 0.0,
            source: "test".into(),
            access_count: 0,
            created_at: now,
            updated_at: now,
            accessed_at: now,
        }
    }

    #[test]
    fn insert_and_get_memory() {
        let store = test_store();
        let entry = sample_entry("test memory");
        let id = entry.id.clone();
        store.insert_memory(&entry).unwrap();
        let got = store.get_memory(&id).unwrap().unwrap();
        assert_eq!(got.title, "test memory");
        assert_eq!(got.content, "test content");
    }

    #[test]
    fn update_memory_patch() {
        let store = test_store();
        let entry = sample_entry("original");
        let id = entry.id.clone();
        store.insert_memory(&entry).unwrap();

        let patch = MemoryPatch {
            title: Some("updated".into()),
            ..Default::default()
        };
        assert!(store.update_memory(&id, &patch).unwrap());
        let got = store.get_memory(&id).unwrap().unwrap();
        assert_eq!(got.title, "updated");
        assert_eq!(got.content, "test content"); // unchanged
    }

    #[test]
    fn delete_memory() {
        let store = test_store();
        let entry = sample_entry("to delete");
        let id = entry.id.clone();
        store.insert_memory(&entry).unwrap();
        assert!(store.delete_memory(&id).unwrap());
        assert!(store.get_memory(&id).unwrap().is_none());
    }

    #[test]
    fn list_memories_with_scope() {
        let store = test_store();
        store.insert_memory(&sample_entry("a")).unwrap();
        let mut entry_b = sample_entry("b");
        entry_b.scope = "session".into();
        store.insert_memory(&entry_b).unwrap();

        let all = store.list_memories(None, 0, 100).unwrap();
        assert_eq!(all.len(), 2);

        let user_only = store.list_memories(Some("user"), 0, 100).unwrap();
        assert_eq!(user_only.len(), 1);
        assert_eq!(user_only[0].title, "a");
    }

    #[test]
    fn search_memories_fts() {
        let store = test_store();
        let mut entry = sample_entry("rust programming");
        entry.content = "learning Rust for systems programming".into();
        store.insert_memory(&entry).unwrap();
        store.insert_memory(&sample_entry("cooking recipe")).unwrap();

        let results = store.search_memories("rust", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "rust programming");
    }

    #[test]
    fn upsert_and_query_facts() {
        let store = test_store();
        let now = now_ts();
        let fact = Fact {
            id: MemoryStore::fact_id("Python", "is_a", "language"),
            subject: "Python".into(),
            predicate: "is_a".into(),
            object: "language".into(),
            confidence: 0.9,
            source: "test".into(),
            created_at: now,
            updated_at: now,
        };
        store.upsert_fact(&fact).unwrap();

        let results = store.query_facts("Python", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "Python");
    }

    #[test]
    fn traverse_graph() {
        let store = test_store();
        let now = now_ts();
        for (s, p, o) in [
            ("A", "knows", "B"),
            ("B", "knows", "C"),
            ("C", "likes", "D"),
        ] {
            store
                .upsert_fact(&Fact {
                    id: MemoryStore::fact_id(s, p, o),
                    subject: s.into(),
                    predicate: p.into(),
                    object: o.into(),
                    confidence: 1.0,
                    source: "test".into(),
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }
        let result = store.traverse_graph("A", 2).unwrap();
        assert_eq!(result.len(), 2); // A->B, B->C
    }

    #[test]
    fn pin_unpin_list() {
        let store = test_store();
        store.pin("task", "Current Task", "build the feature", None).unwrap();
        store.pin("identity", "User Identity", "senior dev", None).unwrap();

        let pins = store.list_pins().unwrap();
        assert_eq!(pins.len(), 2);

        store.unpin("task").unwrap();
        let pins = store.list_pins().unwrap();
        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].key, "identity");
    }

    #[test]
    fn cleanup_expired() {
        let store = test_store();
        let mut old = sample_entry("old low importance");
        old.importance = 0.05;
        old.accessed_at = now_ts() - 100 * 86400; // 100 days ago
        store.insert_memory(&old).unwrap();

        let recent = sample_entry("recent");
        store.insert_memory(&recent).unwrap();

        let deleted = store.cleanup_expired(90, 0.1).unwrap();
        assert_eq!(deleted, 1);

        let remaining = store.list_memories(None, 0, 100).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].title, "recent");
    }

    #[test]
    fn stats_counts() {
        let store = test_store();
        store.insert_memory(&sample_entry("a")).unwrap();
        store.insert_memory(&sample_entry("b")).unwrap();

        let now = now_ts();
        store
            .upsert_fact(&Fact {
                id: "f1".into(),
                subject: "X".into(),
                predicate: "p".into(),
                object: "Y".into(),
                confidence: 0.8,
                source: "".into(),
                created_at: now,
                updated_at: now,
            })
            .unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.total_facts, 1);
        assert_eq!(stats.unique_subjects, 1);
    }
}
