use crate::memory::MemorySearchResult;
use moka::sync::Cache;
use std::time::Duration;

pub struct MemoryCache {
    search_cache: Cache<String, Vec<MemorySearchResult>>,
    file_cache: Cache<String, String>,
}

impl MemoryCache {
    pub fn new() -> Self {
        Self {
            search_cache: Cache::builder()
                .max_capacity(1000)
                .time_to_idle(Duration::from_secs(300))
                .build(),
            file_cache: Cache::builder()
                .max_capacity(500)
                .time_to_idle(Duration::from_secs(600))
                .build(),
        }
    }

    pub fn get_search_result(&self, query: &str) -> Option<Vec<MemorySearchResult>> {
        self.search_cache.get(query)
    }

    pub fn cache_search_result(&self, query: String, results: Vec<MemorySearchResult>) {
        self.search_cache.insert(query, results);
    }

    pub fn get_file_content(&self, path: &str) -> Option<String> {
        self.file_cache.get(path)
    }

    pub fn cache_file_content(&self, path: String, content: String) {
        self.file_cache.insert(path, content);
    }

    pub fn invalidate_path(&self, path: &str) {
        self.file_cache.remove(path);
        self.search_cache.invalidate_all();
    }

    pub fn invalidate_all(&self) {
        self.search_cache.invalidate_all();
        self.file_cache.invalidate_all();
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new()
    }
}
