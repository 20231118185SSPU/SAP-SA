//! Vector index backends for ANN search.

#[cfg(feature = "sqlite-vec")]
pub mod sqlite_vec;

use crate::vector_store::VectorHit;

/// Metadata associated with a vector chunk.
pub struct VectorChunkMeta<'a> {
    pub path: &'a str,
    pub start_line: usize,
    pub end_line: usize,
    pub snippet: &'a str,
    pub indexed_at: i64,
}

/// Abstraction for vector similarity search backends.
pub trait VectorIndexBackend {
    /// Search for the top-k nearest neighbors to the query vector.
    fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>>;

    /// Insert or update a vector with full metadata.
    fn upsert(&mut self, embedding: &[f32], meta: &VectorChunkMeta) -> anyhow::Result<()>;

    /// Delete all vectors for a given path.
    fn delete_by_path(&mut self, path: &str) -> anyhow::Result<usize>;

    /// Number of vectors in the index.
    fn count(&self) -> anyhow::Result<usize>;
}
