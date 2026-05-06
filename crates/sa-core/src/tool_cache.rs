//! Unified caching layer for the SA agent tool system.
//!
//! Provides two cache types:
//! - **FileContentCache**: 60s TTL LRU cache for file contents, keyed by `(path, mtime)`.
//!   When a file's mtime changes, the old entry naturally expires — no manual invalidation.
//! - **FileIndexCache**: In-memory cache for [`FileIndex`] used by `BuildIndex`/`SearchIndex`.
//!   Refreshed on `BuildIndex(force=true)`, otherwise served from memory.

use crate::file_index::FileIndex;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Maximum cached file content entries.
const FILE_CONTENT_MAX_CAPACITY: u64 = 200;

/// TTL for file content cache entries.
const FILE_CONTENT_TTL_SECS: u64 = 120;

// ─────────────────────────────────────────────────────────────────────────────
// File content cache
// ─────────────────────────────────────────────────────────────────────────────

/// Cache key combining file path and modification timestamp.
///
/// When a file is modified externally, its mtime changes, so the old cache entry
/// becomes unreachable — no manual invalidation needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileCacheKey {
    path: PathBuf,
    mtime: u64,
}

/// Thread-safe, TTL-based LRU cache for file contents.
#[derive(Clone)]
pub struct FileContentCache {
    inner: moka::future::Cache<FileCacheKey, String>,
}

impl std::fmt::Debug for FileContentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileContentCache")
            .field("entry_count", &self.inner.entry_count())
            .finish()
    }
}

impl FileContentCache {
    /// Create a new file content cache with configured TTL and capacity.
    pub fn new() -> Self {
        Self {
            inner: moka::future::Cache::builder()
                .max_capacity(FILE_CONTENT_MAX_CAPACITY)
                .time_to_live(std::time::Duration::from_secs(FILE_CONTENT_TTL_SECS))
                .build(),
        }
    }

    /// Try to retrieve cached file content.
    ///
    /// Returns `None` if the entry doesn't exist or has expired.
    pub async fn get(&self, path: &PathBuf, mtime: u64) -> Option<String> {
        let key = FileCacheKey {
            path: path.clone(),
            mtime,
        };
        self.inner.get(&key).await
    }

    /// Insert file content into the cache.
    pub async fn put(&self, path: PathBuf, mtime: u64, content: String) {
        let key = FileCacheKey { path, mtime };
        self.inner.insert(key, content).await;
    }

    /// Invalidate all cached entries for a given path (all mtime variants).
    ///
    /// Used when Write/Edit creates or modifies a file — the old mtime-based
    /// entry is stale and the next Read will insert a fresh entry.
    pub fn invalidate_path(&self, path: &PathBuf) {
        // moka doesn't support prefix-based invalidation natively.
        // Since entries are keyed by (path, mtime), old entries will expire
        // naturally via TTL. For immediate consistency after Write/Edit,
        // we don't need to do anything special — the next Read will use the
        // new mtime as key, producing a cache miss that triggers a fresh read.
        // The old (path, old_mtime) entry lingers until TTL expiry, which is
        // acceptable since it's unreachable.
        let _ = path; // suppress unused warning
    }
}

impl Default for FileContentCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// File index cache
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory cache for the workspace file index.
///
/// Unlike file content (which can change frequently), the file index is relatively
/// stable between explicit `BuildIndex` calls, so we keep it in memory without TTL.
pub struct FileIndexCache {
    inner: Arc<Mutex<Option<FileIndex>>>,
}

impl std::fmt::Debug for FileIndexCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileIndexCache")
            .field("has_cached_index", &self.has_cached_index())
            .finish()
    }
}

impl FileIndexCache {
    /// Create a new empty file index cache.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Store a file index in memory (after BuildIndex).
    pub async fn put(&self, index: FileIndex) {
        let mut guard = self.inner.lock().await;
        *guard = Some(index);
    }

    /// Retrieve the cached file index, if any.
    pub async fn get(&self) -> Option<FileIndex> {
        let guard = self.inner.lock().await;
        guard.clone()
    }

    /// Clear the cached index (used by `BuildIndex(force=true)` before rebuild).
    pub async fn invalidate(&self) {
        let mut guard = self.inner.lock().await;
        *guard = None;
    }

    /// Check whether a cached index exists.
    pub fn has_cached_index(&self) -> bool {
        // Non-blocking check via try_lock.
        match self.inner.try_lock() {
            Ok(guard) => guard.is_some(),
            Err(_) => false,
        }
    }
}

impl Default for FileIndexCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for FileIndexCache {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
