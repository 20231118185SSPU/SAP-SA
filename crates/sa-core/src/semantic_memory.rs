
// ── semantic_memory.rs ──────────────────────────────────────────────────────
//! Semantic fact table — a lightweight knowledge graph built on SQLite.
//!
//! Each fact is a *(subject, predicate, object)* triple with confidence and
//! provenance metadata.  Facts live in an on-disk SQLite database so they
//! survive restarts and can be queried efficiently.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for the semantic memory module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryConfig {
    /// Enable the semantic fact table.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// SQLite database path (relative to workspace root).
    #[serde(default = "default_db_path")]
    pub db_path: String,
    /// Minimum confidence for a fact to be stored.
    #[serde(default = "default_min_conf")]
    pub min_confidence: f64,
    /// Exponential decay rate (λ in e^(-λ·days)).
    #[serde(default = "default_decay")]
    pub decay_rate: f64,
    /// Maximum number of facts before eviction kicks in (0 = unlimited).
    #[serde(default = "default_capacity")]
    pub capacity: usize,
    /// Interference forgetting weight (0.0 = disabled).
    #[serde(default = "default_interference")]
    pub interference_weight: f64,
    /// Snapshot export interval in days (0 = manual only).
    #[serde(default = "default_snapshot_days")]
    pub snapshot_interval_days: i64,
}

fn default_enabled() -> bool { true }
fn default_db_path() -> String { "memory/semantic.db".into() }
fn default_min_conf() -> f64 { 0.3 }
fn default_decay() -> f64 { 0.01 }
fn default_capacity() -> usize { 100_000 }
fn default_interference() -> f64 { 0.15 }
fn default_snapshot_days() -> i64 { 30 }

impl Default for SemanticMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            db_path: default_db_path(),
            min_confidence: 0.3,
            decay_rate: 0.01,
            capacity: 100_000,
            interference_weight: 0.15,
            snapshot_interval_days: 30,
        }
    }
}

/// A single semantic fact (knowledge triple).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Deterministic ID: hex(SHA-256(subject|predicate|object))[:16].
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Handle to the semantic memory SQLite database.
pub struct SemanticMemory {
    conn: Connection,
}

impl SemanticMemory {
    // ── lifecycle ─────────────────────────────────────────────────────────

    /// Open (or create) the semantic database at `workspace_root / db_path`.
    pub fn open(workspace_root: &Path, db_path: &str) -> Result<Self> {
        let full = workspace_root.join(db_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
        let conn = Connection::open(&full)
            .with_context(|| format!("open sqlite {}", full.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS facts (
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
            CREATE INDEX IF NOT EXISTS idx_facts_obj  ON facts(object);
            CREATE INDEX IF NOT EXISTS idx_facts_conf ON facts(confidence);",
        )?;
        Ok(Self { conn })
    }

    fn now() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    fn fact_id(subject: &str, predicate: &str, object: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(subject.as_bytes());
        h.update(b"|");
        h.update(predicate.as_bytes());
        h.update(b"|");
        h.update(object.as_bytes());
        hex::encode(h.finalize())[..16].to_string()
    }

    // ── CRUD ──────────────────────────────────────────────────────────────

    /// Insert or update a fact. Returns `true` if a *new* row was created.
    pub fn upsert_fact(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source: &str,
    ) -> Result<bool> {
        let id = Self::fact_id(subject, predicate, object);
        let now = Self::now();
        let existed: Option<i64> = self.conn
            .query_row("SELECT 1 FROM facts WHERE id = ?1", params![id], |r| r.get(0))
            .optional()?;
        if existed.is_some() {
            self.conn.execute(
                "UPDATE facts SET confidence = MAX(confidence, ?2), source = ?3, updated_at = ?4 WHERE id = ?1",
                params![id, confidence, source, now],
            )?;
            Ok(false)
        } else {
            self.conn.execute(
                "INSERT INTO facts (id,subject,predicate,object,confidence,source,created_at,updated_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?7)",
                params![id, subject, predicate, object, confidence, source, now],
            )?;
            Ok(true)
        }
    }

    /// Query all facts where `subject` is the subject.
    pub fn query_about(&self, subject: &str) -> Result<Vec<SemanticFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
             FROM facts WHERE subject = ?1 ORDER BY confidence DESC",
        )?;
        let rows = stmt.query_map(params![subject], row_to_fact)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Full-text-ish search across subject/predicate/object with LIKE.
    pub fn search_facts(&self, query: &str) -> Result<Vec<SemanticFact>> {
        let pattern = format!("%{query}%");
        let mut stmt = self.conn.prepare(
            "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
             FROM facts
             WHERE subject LIKE ?1 OR predicate LIKE ?1 OR object LIKE ?1
             ORDER BY confidence DESC",
        )?;
        let rows = stmt.query_map(params![pattern], row_to_fact)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Return every fact.
    pub fn all_facts(&self) -> Result<Vec<SemanticFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
             FROM facts ORDER BY confidence DESC",
        )?;
        let rows = stmt.query_map([], row_to_fact)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ── graph ─────────────────────────────────────────────────────────────

    /// BFS traversal: follow edges from `start` for up to `max_hops`.
    pub fn traverse_graph(
        &self,
        start: &str,
        max_hops: usize,
        max_results: usize,
    ) -> Result<Vec<SemanticFact>> {
        let mut visited = std::collections::HashSet::new();
        visited.insert(start.to_string());
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start.to_string(), 0usize));
        let mut results = Vec::new();

        while let Some((node, hops)) = queue.pop_front() {
            if hops >= max_hops {
                continue;
            }
            let mut stmt = self.conn.prepare(
                "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
                 FROM facts WHERE subject = ?1",
            )?;
            let rows = stmt.query_map(params![node], row_to_fact)?;
            for row in rows.flatten() {
                if results.len() >= max_results {
                    break;
                }
                results.push(row.clone());
                let next = row.object;
                if visited.insert(next.clone()) {
                    queue.push_back((next, hops + 1));
                }
            }
        }
        Ok(results)
    }

    /// Query facts related to an entity, optionally by relation type.
    pub fn query_related(
        &self,
        entity: &str,
        relation: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        let sql = match relation {
            Some(_) => {
                "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
                 FROM facts WHERE (subject = ?1 OR object = ?1) AND predicate = ?2
                 ORDER BY confidence DESC LIMIT ?3"
            }
            None => {
                "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
                 FROM facts WHERE subject = ?1 OR object = ?1
                 ORDER BY confidence DESC LIMIT ?2"
            }
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = match relation {
            Some(r) => stmt.query_map(params![entity, r, limit as i64], row_to_fact),
            None => stmt.query_map(params![entity, limit as i64], row_to_fact),
        }?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ── maintenance ───────────────────────────────────────────────────────

    /// Count facts.
    pub fn fact_count(&self) -> Result<usize> {
        let n: i64 = self.conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    /// Aggregate stats: (total, unique_subjects, unique_predicates, avg_conf, oldest, newest).
    pub fn stats(&self) -> Result<(usize, usize, usize, f64, Option<i64>, Option<i64>)> {
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))?;
        let subs: i64 = self.conn.query_row("SELECT COUNT(DISTINCT subject) FROM facts", [], |r| r.get(0))?;
        let preds: i64 = self.conn.query_row("SELECT COUNT(DISTINCT predicate) FROM facts", [], |r| r.get(0))?;
        let avg: f64 = self.conn.query_row("SELECT COALESCE(AVG(confidence),0.0) FROM facts", [], |r| r.get(0)).unwrap_or(0.0);
        let oldest: Option<i64> = self.conn.query_row("SELECT MIN(created_at) FROM facts", [], |r| r.get(0)).ok();
        let newest: Option<i64> = self.conn.query_row("SELECT MAX(created_at) FROM facts", [], |r| r.get(0)).ok();
        Ok((total as usize, subs as usize, preds as usize, avg, oldest, newest))
    }

    /// Exponential decay: multiply confidence by `e^(-rate*days)` for each fact.
    /// Returns (decayed_count, pruned_count).
    pub fn decay_facts(&self, rate: f64) -> Result<(usize, usize)> {
        let now = Self::now();
        let mut stmt = self.conn.prepare("SELECT id, created_at, confidence FROM facts")?;
        let rows: Vec<(String, i64, f64)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut decayed = 0usize;
        let mut pruned = 0usize;
        for (id, created, conf) in rows {
            let days = ((now - created) as f64 / 86400.0).max(0.0);
            let new_conf = conf * (-rate * days).exp();
            if new_conf < 0.05 {
                self.conn.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
                pruned += 1;
            } else {
                self.conn.execute(
                    "UPDATE facts SET confidence = ?2, updated_at = ?3 WHERE id = ?1",
                    params![id, new_conf, now],
                )?;
                decayed += 1;
            }
        }
        Ok((decayed, pruned))
    }

    /// Delete all facts with confidence below `threshold`.
    pub fn prune_below(&self, threshold: f64) -> Result<usize> {
        let n = self.conn.execute("DELETE FROM facts WHERE confidence < ?1", params![threshold])?;
        Ok(n)
    }

    // ── extraction ────────────────────────────────────────────────────────

    /// Extract `(subject, predicate, object)` triples from markdown text.
    /// Returns the number of new facts upserted.
    pub fn extract_from_markdown(&self, content: &str, source: &str) -> Result<usize> {
        let mut count = 0usize;
        // Pattern 1: "X is Y" / "X 是 Y" → (X, "is", Y)
        let re_is = regex::Regex::new(r"([A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff\-]*)\s+(?:is|是|为)\s+([^.\r\n]+)")
            .expect("valid regex");
        for cap in re_is.captures_iter(content) {
            let s = cap[1].trim();
            let o = cap[2].trim().trim_end_matches('.');
            let (s, o) = (s.to_string(), o.to_string());
            if o.len() > 2 && o.len() < 200 && !o.contains('\n') {
                if self.upsert_fact(&s, "is", &o, 0.7, source)? {
                    count += 1;
                }
            }
        }
        // Pattern 2: "X should/必须 Y" → (X, "should", Y)
        let re_should = regex::Regex::new(r"([A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff\-]*)\s+(?:should|必须|应当)\s+([^.\r\n]+)")
            .expect("valid regex");
        for cap in re_should.captures_iter(content) {
            let s = cap[1].trim();
            let o = cap[2].trim().trim_end_matches('.');
            let (s, o) = (s.to_string(), o.to_string());
            if o.len() > 2 && o.len() < 200 && !o.contains('\n') {
                if self.upsert_fact(&s, "should", &o, 0.65, source)? {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    // ── multi-path retrieval ──────────────────────────────────────────────

    /// Search facts with optional time range and predicate filter.
    /// `query` is a fuzzy text match; `predicate` is exact; `since`/`until` are timestamps.
    pub fn search_filtered(
        &self,
        query: Option<&str>,
        predicate: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        let mut wheres: Vec<String> = Vec::new();
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(q) = query {
            let p = format!("%{q}%");
            wheres.push("(subject LIKE ? OR predicate LIKE ? OR object LIKE ?)".to_string());
            params_vec.push(Box::new(p.clone()));
            params_vec.push(Box::new(p.clone()));
            params_vec.push(Box::new(p));
        }
        if let Some(pred) = predicate {
            wheres.push("predicate = ?".to_string());
            params_vec.push(Box::new(pred.to_string()));
        }
        if let Some(s) = since {
            wheres.push("created_at >= ?".to_string());
            params_vec.push(Box::new(s));
        }
        if let Some(u) = until {
            wheres.push("created_at <= ?".to_string());
            params_vec.push(Box::new(u));
        }

        let where_clause = if wheres.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", wheres.join(" AND "))
        };

        let sql = format!(
            "SELECT id,subject,predicate,object,confidence,source,created_at,updated_at
             FROM facts {where_clause} ORDER BY confidence DESC LIMIT ?"
        );
        params_vec.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(&param_refs[..], row_to_fact)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ── interference forgetting ───────────────────────────────────────────

    /// When multiple facts share the same (subject, predicate) but differ in
    /// object, each older fact's confidence is lowered by `weight * (n-1)/n`.
    /// Returns (total_conflicts, total_updated).
    pub fn interfere_forget(&self, weight: f64) -> Result<(usize, usize)> {
        let mut stmt = self.conn.prepare(
            "SELECT subject, predicate, COUNT(*) as cnt
             FROM facts GROUP BY subject, predicate HAVING cnt > 1",
        )?;
        let pairs: Vec<(String, String)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let now = Self::now();
        let mut conflicts = 0usize;
        let mut updated = 0usize;

        for (sub, pred) in pairs {
            conflicts += 1;
            // Get facts for this (sub, pred), ordered by confidence DESC.
            let mut stmt2 = self.conn.prepare(
                "SELECT id, confidence FROM facts
                 WHERE subject = ?1 AND predicate = ?2
                 ORDER BY confidence DESC",
            )?;
            let facts: Vec<(String, f64)> = stmt2.query_map(params![sub, pred], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if facts.len() <= 1 {
                continue;
            }
            let penalty = weight * (facts.len() as f64 - 1.0) / facts.len() as f64;
            // Penalize all but the top one.
            for (id, conf) in facts.iter().skip(1) {
                let new_conf = (conf * (1.0 - penalty)).max(0.0);
                self.conn.execute(
                    "UPDATE facts SET confidence = ?2, updated_at = ?3 WHERE id = ?1",
                    params![id, new_conf, now],
                )?;
                updated += 1;
            }
        }
        Ok((conflicts, updated))
    }

    // ── capacity eviction ─────────────────────────────────────────────────

    /// Evict the lowest-confidence facts when count exceeds `capacity`.
    /// Returns number of evicted facts.
    pub fn evict_below_capacity(&self, capacity: usize) -> Result<usize> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))?;
        if (count as usize) <= capacity {
            return Ok(0);
        }
        let excess = count as usize - capacity;
        let n = self.conn.execute(
            "DELETE FROM facts WHERE id IN (
                SELECT id FROM facts ORDER BY confidence ASC, updated_at ASC LIMIT ?1
            )",
            params![excess as i64],
        )?;
        Ok(n)
    }

    // ── snapshot ──────────────────────────────────────────────────────────

    /// Export all facts as a JSON string (for backup / migration).
    pub fn export_snapshot(&self) -> Result<String> {
        let facts = self.all_facts()?;
        serde_json::to_string_pretty(&facts).context("serialize snapshot")
    }

    /// Import facts from a JSON snapshot string.
    /// Returns (imported_count, skipped_count).
    pub fn import_snapshot(&self, json: &str) -> Result<(usize, usize)> {
        let facts: Vec<SemanticFact> =
            serde_json::from_str(json).context("deserialize snapshot")?;
        let mut imported = 0usize;
        let mut skipped = 0usize;
        for f in &facts {
            if self.upsert_fact(&f.subject, &f.predicate, &f.object, f.confidence, &f.source)? {
                imported += 1;
            } else {
                skipped += 1;
            }
        }
        Ok((imported, skipped))
    }
}


/// Extract structured facts from text using LLM.
///
/// Returns a vector of (subject, predicate, object, confidence) tuples.
/// Falls back to empty vec on parse failure.
pub async fn extract_facts_llm(
    client: &crate::openai::OpenAiClient,
    model: &str,
    text: &str,
    source: &str,
) -> anyhow::Result<Vec<(String, String, String, f64)>> {
    let _source = source; // used for logging
    let prompt = format!(
        r#"You are a fact extraction system. Read the following text and extract structured facts as a JSON array.

Each fact must have these fields:
- "subject": the entity or person (string)
- "predicate": the relationship or attribute (string, e.g. "is", "has", "likes", "works_at", "lives_in")
- "object": the value or related entity (string)
- "confidence": how confident you are (float, 0.0 to 1.0)

Rules:
- Extract ONLY explicit, clear facts. Do not infer or hallucinate.
- Use lowercase for predicates.
- If no facts are found, return an empty array [].
- Output ONLY valid JSON, no markdown fences or explanation.

Text:
{text}

JSON array of facts:"#
    );

    let req = crate::openai::ChatCompletionsRequest {
        model: model.to_string(),
        messages: vec![
            crate::openai::ChatMessage::text("user", prompt),
        ],
        max_tokens: Some(2048),
        reasoning_effort: None,
        tools: None,
        tool_choice: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let resp = client
        .chat_completions(&req)
        .await
        .map_err(|e| anyhow::anyhow!("LLM fact extraction failed: {e}"))?;

    let choice = resp.first_choice()?;
    let content = choice
        .message
        .content
        .as_ref()
        .and_then(|c| match c {
            crate::openai::MessageContent::Text(s) => Some(s.as_str()),
            crate::openai::MessageContent::Parts(parts) => {
                parts.iter().find_map(|p| match p {
                    crate::openai::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
            }
        })
        .unwrap_or("[]");

    #[derive(serde::Deserialize)]
    struct ExtractedFact {
        subject: String,
        predicate: String,
        object: String,
        confidence: Option<f64>,
    }

    let raw_facts: Vec<ExtractedFact> = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("Failed to parse LLM fact JSON from {}: {} Content: {}", _source, e, content);
            return Ok(vec![]);
        }
    };

    let facts: Vec<(String, String, String, f64)> = raw_facts
        .into_iter()
        .filter(|f| !f.subject.is_empty() && !f.predicate.is_empty() && !f.object.is_empty())
        .map(|f| {
            let conf = f.confidence.unwrap_or(0.7).clamp(0.0, 1.0);
            (f.subject, f.predicate, f.object, conf)
        })
        .collect();

    Ok(facts)
}

// ── helpers ────────────────────────────────────────────────────────────────

fn row_to_fact(row: &rusqlite::Row) -> rusqlite::Result<SemanticFact> {
    Ok(SemanticFact {
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

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn new_mem() -> SemanticMemory {
        let tmp = tempdir().unwrap();
        SemanticMemory::open(tmp.path(), "test.db").unwrap()
    }

    #[test]
    fn test_upsert_and_query() {
        let m = new_mem();
        assert!(m.upsert_fact("Alice", "likes", "Bob", 0.9, "test").unwrap());
        assert!(!m.upsert_fact("Alice", "likes", "Bob", 0.95, "test").unwrap());
        let facts = m.query_about("Alice").unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].object, "Bob");
    }

    #[test]
    fn test_search_facts() {
        let m = new_mem();
        m.upsert_fact("Rust", "is", "fast", 0.9, "test").unwrap();
        m.upsert_fact("Python", "is", "slow", 0.8, "test").unwrap();
        let r = m.search_facts("fast").unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].subject, "Rust");
    }

    #[test]
    fn test_traverse_graph() {
        let m = new_mem();
        m.upsert_fact("A", "refs", "B", 0.9, "test").unwrap();
        m.upsert_fact("B", "refs", "C", 0.9, "test").unwrap();
        m.upsert_fact("C", "refs", "D", 0.9, "test").unwrap();
        let r = m.traverse_graph("A", 2, 100).unwrap();
        let objects: Vec<&str> = r.iter().map(|f| f.object.as_str()).collect();
        assert!(objects.contains(&"B"));
        assert!(objects.contains(&"C"));
    }

    #[test]
    fn test_query_related() {
        let m = new_mem();
        m.upsert_fact("Alice", "likes", "Bob", 0.9, "test").unwrap();
        m.upsert_fact("Bob", "likes", "Alice", 0.8, "test").unwrap();
        let r = m.query_related("Alice", None, 10).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_extract_from_markdown() {
        let m = new_mem();
        let n = m.extract_from_markdown("Rust is fast. Python is slow.", "test.md").unwrap();
        assert!(n >= 2);
        let facts = m.query_about("Rust").unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_decay_facts() {
        let m = new_mem();
        m.upsert_fact("A", "is", "B", 0.8, "test").unwrap();
        let (d, p) = m.decay_facts(0.01).unwrap();
        assert_eq!(d + p, 1);
    }

    #[test]
    fn test_prune_below() {
        let m = new_mem();
        m.upsert_fact("A", "is", "B", 0.9, "test").unwrap();
        m.upsert_fact("C", "is", "D", 0.2, "test").unwrap();
        let pruned = m.prune_below(0.3).unwrap();
        assert_eq!(pruned, 1);
        assert_eq!(m.fact_count().unwrap(), 1);
    }

    #[test]
    fn test_search_filtered() {
        let m = new_mem();
        m.upsert_fact("A", "is", "B", 0.9, "test").unwrap();
        m.upsert_fact("A", "has", "C", 0.8, "test").unwrap();
        m.upsert_fact("D", "is", "E", 0.7, "test").unwrap();
        let r = m.search_filtered(None, Some("is"), None, None, 10).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_interfere_forget() {
        let m = new_mem();
        m.upsert_fact("A", "is", "B", 0.9, "test").unwrap();
        m.upsert_fact("A", "is", "C", 0.7, "test").unwrap();
        let (conflicts, updated) = m.interfere_forget(0.2).unwrap();
        assert_eq!(conflicts, 1);
        assert_eq!(updated, 1);
        // The lower-confidence fact should have decreased confidence.
        let facts = m.query_about("A").unwrap();
        let c_fact = facts.iter().find(|f| f.object == "C").unwrap();
        assert!(c_fact.confidence < 0.7);
    }

    #[test]
    fn test_evict_below_capacity() {
        let m = new_mem();
        for i in 0..10 {
            m.upsert_fact(&format!("S{i}"), "is", &format!("O{i}"), 0.5 + (i as f64) * 0.05, "test").unwrap();
        }
        assert_eq!(m.fact_count().unwrap(), 10);
        let evicted = m.evict_below_capacity(7).unwrap();
        assert_eq!(evicted, 3);
        assert_eq!(m.fact_count().unwrap(), 7);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let m = new_mem();
        m.upsert_fact("A", "is", "B", 0.9, "test").unwrap();
        let json = m.export_snapshot().unwrap();
        assert!(json.contains("A"));

        let m2 = new_mem();
        let (imp, skip) = m2.import_snapshot(&json).unwrap();
        assert_eq!(imp, 1);
        assert_eq!(skip, 0);
        let facts = m2.all_facts().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "A");
    }

    #[test]
    fn test_stats() {
        let m = new_mem();
        m.upsert_fact("A", "is", "B", 0.9, "test").unwrap();
        m.upsert_fact("A", "has", "C", 0.8, "test").unwrap();
        let (total, subs, preds, avg, _, _) = m.stats().unwrap();
        assert_eq!(total, 2);
        assert_eq!(subs, 1);
        assert_eq!(preds, 2);
        assert!((avg - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_fact_id_deterministic() {
        let id1 = SemanticMemory::fact_id("A", "is", "B");
        let id2 = SemanticMemory::fact_id("A", "is", "B");
        assert_eq!(id1, id2);
        let id3 = SemanticMemory::fact_id("A", "is", "C");
        assert_ne!(id1, id3);
    }
}
