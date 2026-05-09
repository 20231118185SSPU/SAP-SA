//! Memory panel helpers — stats, query, traverse, decay, prune.

use sa_core::memory_store::MemoryStore;
use sa_core::ws_protocol::MemoryFact;

fn fact_to_proto(f: sa_core::memory_store::Fact) -> MemoryFact {
    MemoryFact {
        id: f.id,
        subject: f.subject,
        predicate: f.predicate,
        object: f.object,
        confidence: f.confidence,
        source: Some(f.source),
        created_at: chrono::DateTime::from_timestamp(f.created_at, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default(),
        updated_at: chrono::DateTime::from_timestamp(f.updated_at, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default(),
    }
}

/// Get aggregate memory statistics from the unified store.
pub fn compute_memory_stats_from_store(
    store: &MemoryStore,
) -> anyhow::Result<(usize, usize, usize, f64, Option<String>, Option<String>)> {
    let stats = store.stats()?;
    Ok((
        stats.total_facts as usize,
        stats.unique_subjects as usize,
        stats.unique_predicates as usize,
        stats.avg_confidence,
        None, // oldest_fact — not tracked in MemoryStats
        None, // newest_fact — not tracked in MemoryStats
    ))
}

/// Query memory facts from the unified store with optional search and pagination.
pub fn query_memory_facts_from_store(
    store: &MemoryStore,
    query: Option<&str>,
    limit: usize,
    offset: usize,
) -> anyhow::Result<(Vec<MemoryFact>, usize)> {
    let effective_limit = limit + offset;
    let raw_facts = store.query_facts(query.unwrap_or(""), effective_limit)?;
    let total = store.stats()?.total_facts as usize;
    let page: Vec<MemoryFact> = raw_facts
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(fact_to_proto)
        .collect();
    Ok((page, total))
}

/// Traverse memory graph from the unified store.
pub fn traverse_memory_from_store(
    store: &MemoryStore,
    subject: &str,
    max_hops: usize,
    max_results: usize,
) -> anyhow::Result<Vec<MemoryFact>> {
    let raw_facts = store.traverse_graph(subject, max_hops)?;
    let facts: Vec<MemoryFact> = raw_facts
        .into_iter()
        .take(max_results)
        .map(fact_to_proto)
        .collect();
    Ok(facts)
}

/// Apply confidence decay to facts in the unified store.
pub fn decay_memory_in_store(store: &MemoryStore) -> anyhow::Result<usize> {
    store.decay_facts(0.01)
}

/// Prune facts below a confidence threshold in the unified store.
pub fn prune_memory_in_store(store: &MemoryStore, min_confidence: f64) -> anyhow::Result<usize> {
    store.prune_facts(min_confidence)
}
