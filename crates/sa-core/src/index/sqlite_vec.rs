//! sqlite-vec backed vector index for O(log n) ANN search.

use std::path::Path;

use anyhow::Context;
use rusqlite::{params, Connection, OpenFlags};
use zerocopy::IntoBytes;

use super::{VectorChunkMeta, VectorIndexBackend};
use crate::vector_store::VectorHit;
use crate::vector_store::EMBEDDING_DIM;

/// Vector index backed by sqlite-vec virtual table.
///
/// This is the primary storage backend. All vector data (embeddings, metadata)
/// lives in the `vec_items` vec0 virtual table, with `vec_meta` storing
/// `indexed_at` timestamps that vec0 cannot represent natively.
pub struct SqliteVecIndex {
    conn: Connection,
}

impl SqliteVecIndex {
    /// Open or create a sqlite-vec backed vector index.
    pub fn open(db_path: &Path) -> anyhow::Result<Self> {
        // Register sqlite-vec as an auto-extension so it loads on every connection.
        unsafe {
            sqlite_vec::sqlite3_vec_init();
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .context("Failed to open sqlite-vec index database")?;

        // Verify sqlite-vec is loaded
        let version: String = conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .context("sqlite-vec extension not available")?;
        tracing::debug!("sqlite-vec version: {version}");

        // Create the vec0 virtual table for ANN search.
        // Auxiliary columns (prefixed with +) store metadata alongside embeddings.
        // The embedding column stores the raw float vector so it can be retrieved
        // for downstream MMR reranking.
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                embedding float[{dim}],
                +path TEXT,
                +start_line INTEGER,
                +end_line INTEGER,
                +snippet TEXT
            );",
            dim = EMBEDDING_DIM,
        ))
        .context("Failed to create vec0 virtual table")?;

        // Metadata table for indexed_at timestamps (vec0 doesn't support all column types).
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vec_meta (
                rowid   INTEGER PRIMARY KEY,
                indexed_at INTEGER NOT NULL
            );",
        )
        .context("Failed to create vec_meta table")?;

        Ok(Self { conn })
    }

    /// Count of vectors in the vec0 table.
    pub fn vec_count(&self) -> anyhow::Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM vec_items", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    /// Retrieve the raw embedding bytes for a given rowid.
    fn embedding_by_rowid(&self, rowid: i64) -> anyhow::Result<Option<Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT embedding FROM vec_items WHERE rowid = ?1")
            .context("Failed to prepare embedding lookup")?;
        let mut rows = stmt.query_map(params![rowid], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        })?;
        if let Some(row) = rows.next() {
            let blob = row?;
            Ok(bytes_to_f32_vec(&blob))
        } else {
            Ok(None)
        }
    }

    /// Delete all vectors for a given path and return count deleted.
    pub fn delete_by_path(&self, path: &str) -> anyhow::Result<usize> {
        // Collect rowids first so we can also clean up vec_meta.
        let mut stmt = self
            .conn
            .prepare("SELECT rowid FROM vec_items WHERE path = ?1")?;
        let rowids: Vec<i64> = stmt
            .query_map(params![path], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let n = self
            .conn
            .execute("DELETE FROM vec_items WHERE path = ?1", params![path])?;

        // Clean up vec_meta for deleted rows.
        for rid in &rowids {
            let _ = self
                .conn
                .execute("DELETE FROM vec_meta WHERE rowid = ?1", params![rid]);
        }

        Ok(n)
    }
}

impl VectorIndexBackend for SqliteVecIndex {
    fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>> {
        let query_bytes = query.as_bytes();

        let mut stmt = self
            .conn
            .prepare(
                "SELECT rowid, path, start_line, end_line, snippet, distance
                 FROM vec_items
                 WHERE embedding MATCH ?1
                 ORDER BY distance
                 LIMIT ?2",
            )
            .context("Failed to prepare sqlite-vec search")?;

        let rows = stmt
            .query_map(params![query_bytes, top_k as i64], |row| {
                let rowid: i64 = row.get(0)?;
                let path: String = row.get(1)?;
                let start_line: i64 = row.get(2)?;
                let end_line: i64 = row.get(3)?;
                let snippet: String = row.get(4)?;
                let distance: f64 = row.get(5)?;
                Ok((rowid, path, start_line, end_line, snippet, distance))
            })
            .context("Failed to execute sqlite-vec search")?;

        let mut results = Vec::new();
        for row in rows {
            let (rowid, path, sl, el, snippet, distance) = row?;
            // Convert L2 distance to similarity score (0.0–1.0 range, approximate).
            // For normalized vectors: cosine_similarity ≈ 1 - (distance^2 / 2)
            let score = (1.0 - (distance * distance) / 2.0).clamp(0.0, 1.0);
            // Retrieve the raw embedding for MMR reranking downstream.
            let embedding = self.embedding_by_rowid(rowid)?.unwrap_or_default();
            results.push(VectorHit {
                path,
                start_line: sl as usize,
                end_line: el as usize,
                score,
                snippet,
                embedding,
            });
        }

        Ok(results)
    }

    fn upsert(&mut self, embedding: &[f32], meta: &VectorChunkMeta) -> anyhow::Result<()> {
        let emb_bytes = embedding.as_bytes();

        // Collect existing rowid for this (path, start_line) so we can update vec_meta.
        let existing_rowid: Option<i64> = self.conn.query_row(
            "SELECT rowid FROM vec_items WHERE path = ?1 AND start_line = ?2",
            params![meta.path, meta.start_line as i64],
            |row| row.get(0),
        ).ok();

        // Delete existing entry if present (vec0 doesn't support UPSERT natively).
        self.conn
            .execute(
                "DELETE FROM vec_items WHERE path = ?1 AND start_line = ?2",
                params![meta.path, meta.start_line as i64],
            )
            .context("Failed to delete before upsert")?;

        self.conn
            .execute(
                "INSERT INTO vec_items(rowid, embedding, path, start_line, end_line, snippet)
                 VALUES (NULL, ?1, ?2, ?3, ?4, ?5)",
                params![
                    emb_bytes,
                    meta.path,
                    meta.start_line as i64,
                    meta.end_line as i64,
                    meta.snippet,
                ],
            )
            .context("Failed to insert into vec_items")?;

        // Get the new rowid.
        let new_rowid: i64 = self.conn.last_insert_rowid();

        // Upsert vec_meta (indexed_at timestamp).
        if let Some(rid) = existing_rowid {
            let _ = self.conn.execute(
                "DELETE FROM vec_meta WHERE rowid = ?1",
                params![rid],
            );
        }
        self.conn
            .execute(
                "INSERT INTO vec_meta(rowid, indexed_at) VALUES (?1, ?2)",
                params![new_rowid, meta.indexed_at],
            )
            .context("Failed to upsert vec_meta")?;

        Ok(())
    }

    fn delete_by_path(&mut self, path: &str) -> anyhow::Result<usize> {
        SqliteVecIndex::delete_by_path(self, path)
    }

    fn count(&self) -> anyhow::Result<usize> {
        self.vec_count()
    }
}

/// Convert a byte slice to a Vec<f32> (little-endian).
fn bytes_to_f32_vec(b: &[u8]) -> Option<Vec<f32>> {
    if b.len() % 4 != 0 || b.is_empty() {
        return None;
    }
    Some(
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_meta<'a>(path: &'a str, start_line: usize, end_line: usize, snippet: &'a str) -> VectorChunkMeta<'a> {
        VectorChunkMeta { path, start_line, end_line, snippet, indexed_at: 1000 }
    }

    #[test]
    fn test_sqlite_vec_open() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("vec_test.db");
        let idx = SqliteVecIndex::open(&db)?;
        assert_eq!(idx.vec_count()?, 0);
        Ok(())
    }

    #[test]
    fn test_sqlite_vec_upsert_and_count() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("vec_upsert.db");
        let mut idx = SqliteVecIndex::open(&db)?;

        let emb1: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        let emb2: Vec<f32> = (0..384).map(|i| (i + 100) as f32 * 0.001).collect();

        idx.upsert(&emb1, &make_meta("doc_1", 1, 10, "chunk1"))?;
        idx.upsert(&emb2, &make_meta("doc_2", 1, 10, "chunk2"))?;

        assert_eq!(idx.count()?, 2);
        Ok(())
    }

    #[test]
    fn test_sqlite_vec_search() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("vec_search.db");
        let mut idx = SqliteVecIndex::open(&db)?;

        // Create distinct vectors
        let mut emb1 = vec![0.0f32; 384];
        emb1[0] = 1.0;
        let mut emb2 = vec![0.0f32; 384];
        emb2[1] = 1.0;
        let mut emb3 = vec![0.0f32; 384];
        emb3[0] = 0.707;
        emb3[1] = 0.707;

        idx.upsert(&emb1, &make_meta("doc_a", 1, 5, "a"))?;
        idx.upsert(&emb2, &make_meta("doc_b", 1, 5, "b"))?;
        idx.upsert(&emb3, &make_meta("doc_c", 1, 5, "c"))?;

        // Query identical to emb1
        let hits = idx.search(&emb1, 3)?;
        assert_eq!(hits.len(), 3);
        // First hit should be doc_a (most similar)
        assert_eq!(hits[0].path, "doc_a");
        // Embedding must be populated (not empty) for MMR reranking
        assert!(!hits[0].embedding.is_empty());
        assert_eq!(hits[0].embedding.len(), 384);
        Ok(())
    }

    #[test]
    fn test_sqlite_vec_delete_by_path() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("vec_del.db");
        let mut idx = SqliteVecIndex::open(&db)?;

        let emb = vec![1.0f32; 384];
        idx.upsert(&emb, &make_meta("doc_1", 1, 10, "chunk"))?;
        assert_eq!(idx.count()?, 1);

        idx.delete_by_path("doc_1")?;
        assert_eq!(idx.count()?, 0);
        Ok(())
    }

    #[test]
    fn test_sqlite_vec_search_top_k() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("vec_topk.db");
        let mut idx = SqliteVecIndex::open(&db)?;

        for i in 0..10 {
            let emb: Vec<f32> = (0..384).map(|j| (i * 384 + j) as f32 * 0.001).collect();
            idx.upsert(&emb, &make_meta(&format!("doc_{}", i), i * 10, (i + 1) * 10, &format!("chunk {}", i)))?;
        }

        let query: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        let hits = idx.search(&query, 3)?;
        assert_eq!(hits.len(), 3);
        // Verify embeddings are returned
        for hit in &hits {
            assert!(!hit.embedding.is_empty());
        }
        Ok(())
    }

    #[test]
    fn test_sqlite_vec_metadata_preserved() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db = dir.path().join("vec_meta.db");
        let mut idx = SqliteVecIndex::open(&db)?;

        let emb = vec![1.0f32; 384];
        idx.upsert(&emb, &make_meta("mem/test.md", 10, 20, "test snippet"))?;

        let hits = idx.search(&emb, 1)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "mem/test.md");
        assert_eq!(hits[0].start_line, 10);
        assert_eq!(hits[0].end_line, 20);
        assert_eq!(hits[0].snippet, "test snippet");
        Ok(())
    }
}
