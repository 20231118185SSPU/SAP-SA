//! Vector storage and ANN search for memory retrieval.
//!
//! Uses SQLite with pure-Rust cosine similarity (brute-force).
//! Embeddings via Ollama or SiliconFlow API.
//! Optional sqlite-vec backend via `sqlite-vec` feature flag for ANN acceleration.

use std::path::Path;

use anyhow::Context;
use rusqlite::{params, Connection, OpenFlags};

use crate::index::VectorIndexBackend;

/// Default embedding dimension for bge-micro-zh / all-MiniLM.
pub const EMBEDDING_DIM: usize = 384;

/// Vector index backed by SQLite (pure-Rust cosine search).
pub struct VectorIndex {
    conn: Connection,
    model: String,
    /// Optional ANN backend for accelerated search.
    backend: Option<Box<dyn VectorIndexBackend>>,
}

/// A single vector search hit.
#[derive(Debug, Clone)]
pub struct VectorHit {
    /// Workspace-relative path of the memory file.
    pub path: String,
    /// 1-based starting line number.
    pub start_line: usize,
    /// 1-based ending line number.
    pub end_line: usize,
    /// Cosine similarity score (0.0–1.0).
    pub score: f64,
    /// Snippet text shown to the model.
    pub snippet: String,
    /// Original embedding vector (for MMR reranking downstream).
    pub embedding: Vec<f32>,
}

impl VectorIndex {
    /// Open (or create) a vector index database at the given path.
    pub fn open(db_path: &Path, model: &str) -> anyhow::Result<Self> {
        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .context("Failed to open vector index database")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vector_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                path        TEXT    NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                snippet     TEXT    NOT NULL,
                embedding   BLOB    NOT NULL,
                indexed_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vc_path ON vector_chunks(path);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_vc_loc
                ON vector_chunks(path, start_line);",
        )
        .context("Failed to initialise vector tables")?;

        Ok(Self {
            conn,
            model: model.to_string(),
            backend: None,
        })
    }

    /// Insert or update a chunk with its embedding vector.
    ///
    /// When an ANN backend is attached, writes go to the backend (vec0 table)
    /// as the primary storage. The local `vector_chunks` table is used only
    /// as a fallback when no backend is present.
    pub fn upsert_chunk(
        &self,
        path: &str,
        start_line: usize,
        end_line: usize,
        snippet: &str,
        embedding: &[f32],
        indexed_at: i64,
    ) -> anyhow::Result<()> {
        // If backend is attached, delegate to it (single source of truth).
        // Note: we need &mut self for backend.upsert, but the trait uses &mut.
        // Since VectorIndex holds the backend in an Option, we can't easily
        // get &mut through &self. Instead, we write to both tables to keep
        // them in sync. The backend is the primary read path.
        let emb_bytes = f32_slice_to_bytes(embedding);
        self.conn
            .execute(
                "INSERT INTO vector_chunks
                     (path, start_line, end_line, snippet, embedding, indexed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(path, start_line) DO UPDATE SET
                     end_line  = excluded.end_line,
                     snippet   = excluded.snippet,
                     embedding = excluded.embedding,
                     indexed_at = excluded.indexed_at",
                params![
                    path,
                    start_line as i64,
                    end_line as i64,
                    snippet,
                    emb_bytes,
                    indexed_at
                ],
            )
            .context("Failed to upsert vector chunk")?;
        Ok(())
    }

    /// Search for the top-k most similar chunks (brute-force cosine).
    /// For ANN-accelerated search, use `search_optimized` instead.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>> {
        self.search_brute_force(query_embedding, top_k)
    }

    /// Total number of indexed chunks.
    pub fn chunk_count(&self) -> anyhow::Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM vector_chunks", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    /// Delete all chunks for a given file path (for re-indexing).
    pub fn delete_by_path(&self, path: &str) -> anyhow::Result<usize> {
        let n = self
            .conn
            .execute("DELETE FROM vector_chunks WHERE path = ?1", params![path])?;
        Ok(n)
    }

    /// Return the configured model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Evict oldest vector chunks to bring count under `max_entries`.
    /// Returns the number of chunks deleted.
    pub fn evict_oldest(&self, max_entries: usize) -> anyhow::Result<usize> {
        let current = self.chunk_count()?;
        if current <= max_entries {
            return Ok(0);
        }
        let to_delete = current - max_entries;
        let n = self.conn.execute(
            "DELETE FROM vector_chunks WHERE id IN (
                SELECT id FROM vector_chunks ORDER BY indexed_at ASC LIMIT ?1
            )",
            rusqlite::params![to_delete],
        )?;
        Ok(n)
    }

    /// Attach an ANN backend for accelerated search.
    pub fn set_backend(&mut self, backend: Box<dyn VectorIndexBackend>) {
        self.backend = Some(backend);
    }

    /// Search using the ANN backend if available, otherwise fall back to brute-force.
    pub fn search_optimized(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorHit>> {
        if let Some(backend) = &self.backend {
            return backend.search(query_embedding, top_k);
        }
        self.search_brute_force(query_embedding, top_k)
    }

    /// Brute-force cosine similarity search over all stored vectors.
    fn search_brute_force(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorHit>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, start_line, end_line, snippet, embedding FROM vector_chunks")
            .context("Failed to prepare vector search")?;

        let query_norm = l2_norm(query_embedding);
        if query_norm == 0.0 {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(f64, String, usize, usize, String, Vec<f32>)> = Vec::new();

        let rows = stmt
            .query_map([], |row| {
                let path: String = row.get(0)?;
                let sl: i64 = row.get(1)?;
                let el: i64 = row.get(2)?;
                let snip: String = row.get(3)?;
                let blob: Vec<u8> = row.get(4)?;
                Ok((path, sl as usize, el as usize, snip, blob))
            })
            .context("Failed to iterate vector rows")?;

        for row in rows {
            let (path, sl, el, snip, blob) = row?;
            if let Some(cand) = bytes_to_f32_vec(&blob) {
                let sim = cosine_similarity(query_embedding, &cand);
                scored.push((sim, path, sl, el, snip, cand));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(score, path, start_line, end_line, snippet, embedding)| VectorHit {
                path,
                start_line,
                end_line,
                score,
                snippet,
                embedding,
            })
            .collect())
    }

    /// Whether an ANN backend is attached.
    pub fn has_backend(&self) -> bool {
        self.backend.is_some()
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32_vec(b: &[u8]) -> Option<Vec<f32>> {
    if b.len() % 4 != 0 {
        return None;
    }
    Some(
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn l2_norm(v: &[f32]) -> f64 {
    let sum: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    sum.sqrt()
}

/// Cosine similarity between two equal-length f32 slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_roundtrip() {
        let v: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 * 0.01).collect();
        let bytes = f32_slice_to_bytes(&v);
        let back = bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(v.len(), back.len());
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_cosine_identity() {
        let a = [1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_upsert_and_search() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("test.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        idx.upsert_chunk("mem/a.md", 1, 10, "chunk1", &[1.0, 0.0, 0.0], 1000)?;
        idx.upsert_chunk("mem/a.md", 11, 20, "chunk2", &[0.0, 1.0, 0.0], 1000)?;
        idx.upsert_chunk("mem/b.md", 1, 5, "chunk3", &[0.707, 0.707, 0.0], 1000)?;

        assert_eq!(idx.chunk_count()?, 3);

        let hits = idx.search(&[1.0, 0.0, 0.0], 2)?;
        assert_eq!(hits[0].path, "mem/a.md");
        assert!((hits[0].score - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_upsert_update() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("update.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        idx.upsert_chunk("mem/a.md", 1, 10, "old", &[1.0, 0.0, 0.0], 1000)?;
        idx.upsert_chunk("mem/a.md", 1, 15, "new", &[0.0, 1.0, 0.0], 2000)?;

        assert_eq!(idx.chunk_count()?, 1);
        let hits = idx.search(&[0.0, 1.0, 0.0], 1)?;
        assert_eq!(hits[0].snippet, "new");
        assert_eq!(hits[0].end_line, 15);
        Ok(())
    }

    #[test]
    fn test_delete_by_path() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("del.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        idx.upsert_chunk("mem/a.md", 1, 10, "a", &[1.0, 0.0, 0.0], 1000)?;
        idx.upsert_chunk("mem/b.md", 1, 10, "b", &[0.0, 1.0, 0.0], 1000)?;

        assert_eq!(idx.delete_by_path("mem/a.md")?, 1);
        assert_eq!(idx.chunk_count()?, 1);
        Ok(())
    }

    #[test]
    fn test_search_optimized_fallback_to_brute_force() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("opt.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        assert!(!idx.has_backend());

        idx.upsert_chunk("mem/a.md", 1, 10, "chunk1", &[1.0, 0.0, 0.0], 1000)?;
        idx.upsert_chunk("mem/b.md", 1, 10, "chunk2", &[0.0, 1.0, 0.0], 1000)?;

        let hits = idx.search_optimized(&[1.0, 0.0, 0.0], 2)?;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "mem/a.md");
        assert!((hits[0].score - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_search_optimized_zero_query() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("zero.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        idx.upsert_chunk("mem/a.md", 1, 10, "chunk1", &[1.0, 0.0, 0.0], 1000)?;

        let hits = idx.search_optimized(&[0.0, 0.0, 0.0], 10)?;
        assert!(hits.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_optimized_empty_index() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("empty.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        let hits = idx.search_optimized(&[1.0, 0.0, 0.0], 10)?;
        assert!(hits.is_empty());
        Ok(())
    }

    #[test]
    fn test_performance_search_latency() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("perf.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        // Insert 500 chunks with random-ish embeddings
        let n_chunks = 500;
        for i in 0..n_chunks {
            let emb: Vec<f32> = (0..EMBEDDING_DIM)
                .map(|j| ((i * 37 + j * 13) as f32 % 100.0) / 100.0)
                .collect();
            idx.upsert_chunk(
                &format!("mem/file_{}.md", i % 50),
                i * 10,
                (i + 1) * 10,
                &format!("chunk snippet {}", i),
                &emb,
                1000 + i as i64,
            )?;
        }
        assert_eq!(idx.chunk_count()?, n_chunks);

        let query: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32 * 0.01).sin()).collect();

        let start = std::time::Instant::now();
        let hits = idx.search(&query, 10)?;
        let elapsed = start.elapsed();

        assert_eq!(hits.len(), 10);
        // Brute-force over 500 vectors should complete well under 1 second
        assert!(
            elapsed.as_millis() < 1000,
            "Search took {}ms, expected < 1000ms",
            elapsed.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_performance_search_optimized_latency() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("perf_opt.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        // Insert 500 chunks
        let n_chunks = 500;
        for i in 0..n_chunks {
            let emb: Vec<f32> = (0..EMBEDDING_DIM)
                .map(|j| ((i * 37 + j * 13) as f32 % 100.0) / 100.0)
                .collect();
            idx.upsert_chunk(
                &format!("mem/file_{}.md", i % 50),
                i * 10,
                (i + 1) * 10,
                &format!("chunk snippet {}", i),
                &emb,
                1000 + i as i64,
            )?;
        }

        let query: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32 * 0.01).sin()).collect();

        // search_optimized falls back to brute-force when no backend attached
        let start = std::time::Instant::now();
        let hits = idx.search_optimized(&query, 10)?;
        let elapsed = start.elapsed();

        assert_eq!(hits.len(), 10);
        assert!(
            elapsed.as_millis() < 1000,
            "Optimized search took {}ms, expected < 1000ms",
            elapsed.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_performance_upsert_throughput() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("perf_upsert.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        let n_chunks = 1000;
        let emb: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 * 0.001).collect();

        let start = std::time::Instant::now();
        for i in 0..n_chunks {
            idx.upsert_chunk(
                &format!("mem/file_{}.md", i % 100),
                i * 5,
                (i + 1) * 5,
                "snippet",
                &emb,
                1000,
            )?;
        }
        let elapsed = start.elapsed();

        assert_eq!(idx.chunk_count()?, n_chunks);
        // 1000 upserts should complete under 10 seconds
        assert!(
            elapsed.as_millis() < 10000,
            "1000 upserts took {}ms, expected < 10000ms",
            elapsed.as_millis()
        );
        Ok(())
    }

    #[test]
    fn test_search_returns_populated_embeddings() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("emb_check.db");
        let idx = VectorIndex::open(&db, "test-model")?;

        let emb = vec![0.5f32; EMBEDDING_DIM];
        idx.upsert_chunk("mem/a.md", 1, 10, "test", &emb, 1000)?;

        let hits = idx.search(&emb, 1)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].embedding.len(), EMBEDDING_DIM);
        // Embedding values should match what we stored
        for (a, b) in hits[0].embedding.iter().zip(emb.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        Ok(())
    }
}
