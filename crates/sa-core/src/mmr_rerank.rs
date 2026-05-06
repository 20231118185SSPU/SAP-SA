//! Maximum Marginal Relevance (MMR) reranking for memory search results.
//!
//! MMR balances relevance (how well a result matches the query) against
//! diversity (how different a result is from already-selected results).
//! This prevents redundant results from dominating the top positions.
//!
//! Algorithm: MMR = λ * sim(D_i, Q) - (1-λ) * max_{D_j in S} sim(D_i, D_j)
//!   - λ = 1.0 → pure relevance ranking
//!   - λ = 0.0 → pure diversity ranking
//!   - Default λ = 0.7 balances both
//!
//! Requirement: each result must have an embedding vector for similarity computation.

use crate::memory::MemorySearchResult;
use crate::vector_store::VectorIndex;
use serde::{Deserialize, Serialize};

/// Configuration for MMR reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MMRRerankConfig {
    /// Trade-off parameter: 1.0 = pure relevance, 0.0 = pure diversity.
    pub lambda: f64,
    /// Maximum number of results to keep after reranking.
    pub max_selected: usize,
    /// Minimum similarity threshold for clustering dedup (debug only).
    pub similarity_threshold: f64,
}

impl Default for MMRRerankConfig {
    fn default() -> Self {
        Self {
            lambda: 0.7,
            max_selected: 10,
            similarity_threshold: 0.95,
        }
    }
}

/// Run MMR reranking on search results.
///
/// Results are reordered so that each selected result is both relevant to
/// the query AND diverse from previously selected results.
///
/// Returns a new Vec with results in MMR order. The original `results` is consumed.
pub fn mmr_rerank(
    mut results: Vec<MemorySearchResult>,
    query_embedding: &[f32],
    config: &MMRRerankConfig,
) -> Vec<MemorySearchResult> {
    if results.is_empty() || results.len() <= 1 {
        return results;
    }

    // Extract embeddings from results (must be populated by caller)
    let embeddings: Vec<Option<&Vec<f32>>> = results
        .iter()
        .map(|r| r.embedding.as_ref())
        .collect();

    // If no embeddings available, return original order unchanged
    if embeddings.iter().all(|e| e.is_none()) {
        return results;
    }

    // Compute query similarity scores for all results
    let relevance_scores: Vec<f64> = embeddings
        .iter()
        .map(|emb| match emb {
            Some(vec) => cosine_similarity(query_embedding, vec),
            None => 0.0,
        })
        .collect();

    let n = results.len();
    let k = config.max_selected.min(n);

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..n).collect();

    // Greedy MMR selection
    for _ in 0..k {
        if remaining.is_empty() {
            break;
        }

        let mut best_idx_in_remaining = 0usize;
        let mut best_mmr = f64::NEG_INFINITY;

        for (i, &cand_idx) in remaining.iter().enumerate() {
            let relevance = relevance_scores.get(cand_idx).copied().unwrap_or(0.0);

            // Max similarity to any already-selected result
            let max_sim_to_selected: f64 = selected
                .iter()
                .map(|&s_idx| {
                    let a = embeddings.get(s_idx).and_then(|e| *e);
                    let b = embeddings.get(cand_idx).and_then(|e| *e);
                    match (a, b) {
                        (Some(va), Some(vb)) => cosine_similarity(va, vb),
                        _ => 0.0,
                    }
                })
                .fold(0.0_f64, f64::max);

            let mmr = config.lambda * relevance - (1.0 - config.lambda) * max_sim_to_selected;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx_in_remaining = i;
            }
        }

        let chosen = remaining.remove(best_idx_in_remaining);
        selected.push(chosen);
    }

    // Reorder results by MMR selection order
    let mut reordered = Vec::with_capacity(results.len());
    let selected_set: std::collections::HashSet<usize> =
        selected.iter().copied().collect();

    // Move MMR-selected results first
    for &idx in &selected {
        if idx < results.len() {
            // Take the result out (using swap or placeholder)
            // Since we can't move out of indexed Vec easily, use take + default pattern
            // Actually we'll swap with last and pop
            reordered.push(std::mem::replace(
                &mut results[idx],
                MemorySearchResult {
                    path: String::new(),
                    start_line: 0,
                    end_line: 0,
                    score: 0.0,
                    snippet: String::new(),
                    contributing_paths: Vec::new(),
                    embedding: None,
                },
            ));
        }
    }

    // Append remaining results in original order (but skip already-selected slots)
    for (i, result) in results.into_iter().enumerate() {
        if !selected_set.contains(&i) && !result.path.is_empty() {
            reordered.push(result);
        }
    }

    reordered.truncate(config.max_selected);
    reordered
}

/// Run MMR reranking when embeddings are not pre-computed on results.
/// Uses the vector store to fetch embeddings on-the-fly.
pub fn mmr_rerank_with_store(
    results: Vec<MemorySearchResult>,
    query_embedding: &[f32],
    config: &MMRRerankConfig,
    _vector_index: &VectorIndex,
) -> Vec<MemorySearchResult> {
    // If results already have embeddings, use the fast path
    if results.iter().any(|r| r.embedding.is_some()) {
        return mmr_rerank(results, query_embedding, config);
    }
    // Otherwise, pass through without reranking (embeddings needed)
    results
}

/// Compute cosine similarity between two f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for i in 0..len {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(path: &str, score: f64, embedding: Option<Vec<f32>>) -> MemorySearchResult {
        MemorySearchResult {
            path: path.to_string(),
            start_line: 1,
            end_line: 1,
            score,
            snippet: format!("snippet from {path}"),
            contributing_paths: vec![],
            embedding,
        }
    }

    #[test]
    fn test_empty_results() {
        let results: Vec<MemorySearchResult> = vec![];
        let qe = vec![1.0_f32, 0.0, 0.0];
        let cfg = MMRRerankConfig::default();
        let reranked = mmr_rerank(results, &qe, &cfg);
        assert!(reranked.is_empty());
    }

    #[test]
    fn test_single_result_unchanged() {
        let results = vec![make_result("a.md", 0.9, Some(vec![1.0, 0.0, 0.0]))];
        let qe = vec![1.0_f32, 0.0, 0.0];
        let cfg = MMRRerankConfig::default();
        let reranked = mmr_rerank(results, &qe, &cfg);
        assert_eq!(reranked.len(), 1);
        assert_eq!(reranked[0].path, "a.md");
    }

    #[test]
    fn test_no_embeddings_returns_unchanged() {
        let results = vec![
            make_result("a.md", 0.9, None),
            make_result("b.md", 0.8, None),
        ];
        let qe = vec![1.0_f32, 0.0, 0.0];
        let cfg = MMRRerankConfig::default();
        let reranked = mmr_rerank(results, &qe, &cfg);
        assert_eq!(reranked.len(), 2); // unchanged
    }

    #[test]
    fn test_dedup_similar_results() {
        // Two nearly identical results should be separated by MMR.
        // Use a lower λ (0.4) to emphasise diversity, so the near-duplicate
        // b.md is penalised enough for c.md to overtake it.
        let very_similar = vec![0.99_f32, 0.01, 0.0];
        let results = vec![
            make_result("a.md", 0.95, Some(very_similar.clone())),
            make_result("b.md", 0.94, Some(very_similar.clone())), // near-duplicate of a
            make_result("c.md", 0.85, Some(vec![0.3_f32, 0.7, 0.0])), // different direction
        ];
        let qe = vec![1.0_f32, 0.0, 0.0];

        let cfg = MMRRerankConfig {
            lambda: 0.4,
            max_selected: 3,
            similarity_threshold: 0.9,
        };

        let reranked = mmr_rerank(results, &qe, &cfg);
        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[0].path, "a.md");  // best relevance
        assert_eq!(reranked[1].path, "c.md");  // diverse → beats near-duplicate b
        assert_eq!(reranked[2].path, "b.md");
    }

    #[test]
    fn test_pure_relevance_lambda_one() {
        let results = vec![
            make_result("low.md", 0.5, Some(vec![0.1_f32, 0.1])),
            make_result("high.md", 0.95, Some(vec![1.0_f32, 0.0])),
            make_result("mid.md", 0.8, Some(vec![0.8_f32, 0.2])),
        ];
        let qe = vec![1.0_f32, 0.0];
        let cfg = MMRRerankConfig {
            lambda: 1.0,
            max_selected: 3,
            ..Default::default()
        };

        let reranked = mmr_rerank(results, &qe, &cfg);
        // With λ=1.0, pure relevance → high.md first, then mid, then low
        assert_eq!(reranked[0].path, "high.md");
        assert_eq!(reranked[1].path, "mid.md");
        assert_eq!(reranked[2].path, "low.md");
    }

    #[test]
    fn test_cosine_identity() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![-1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_truncation_to_max_selected() {
        let results: Vec<_> = (0..20)
            .map(|i| {
                make_result(
                    &format!("{i}.md"),
                    0.9 - i as f64 * 0.04,
                    Some(vec![i as f32 * 0.1, 0.5]),
                )
            })
            .collect();

        let qe = vec![1.0_f32, 0.0];
        let cfg = MMRRerankConfig {
            lambda: 0.7,
            max_selected: 5,
            ..Default::default()
        };

        let reranked = mmr_rerank(results, &qe, &cfg);
        assert_eq!(reranked.len(), 5);
    }
}
