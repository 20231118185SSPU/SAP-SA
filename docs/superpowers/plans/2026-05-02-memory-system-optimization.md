# SA 记忆系统优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化SA项目记忆系统的性能、可靠性和可维护性，解决向量搜索瓶颈、内存管理、并发控制等核心问题。

**Architecture:** 分三阶段实施：Phase 1（性能优化）→ Phase 2（可靠性增强）→ Phase 3（可维护性提升）。每阶段独立可交付，向下兼容。

**Tech Stack:** Rust, SQLite, tokio, serde, chrono

---

## 文件结构映射

### 核心修改文件
```
sa/crates/sa-core/src/
├── vector_store.rs          # 向量存储优化（添加ANN索引）
├── memory.rs                # 记忆搜索优化（添加缓存层）
├── working_memory.rs        # 工作记忆优化（内存限制）
├── memory_indexer.rs        # 索引器优化（增量更新）
├── config.rs                # 配置简化（预设模板）
├── semantic_memory.rs       # 语义记忆优化（并发控制）
└── memory_filter.rs         # 写入过滤优化（批量处理）

sa/crates/sa/src/
├── main.rs                  # 主服务（集成优化组件）
└── ws_protocol.rs           # 协议优化（批量消息）

sa/crates/sa-core/src/cache/
├── mod.rs                   # 新增：缓存模块
├── lru_cache.rs             # 新增：LRU缓存实现
└── memory_cache.rs          # 新增：记忆缓存适配器

sa/crates/sa-core/src/index/
├── mod.rs                   # 新增：索引模块
├── vector_index.rs          # 新增：向量索引抽象
└── sqlite_vec.rs            # 新增：sqlite-vec集成

sa/crates/sa-core/src/concurrency/
├── mod.rs                   # 新增：并发控制模块
├── file_lock.rs             # 新增：文件锁实现
└── rw_lock.rs               # 新增：读写锁实现
```

### 测试文件
```
sa/crates/sa-core/src/
├── vector_store_test.rs     # 向量存储测试
├── memory_cache_test.rs     # 缓存测试
├── concurrency_test.rs      # 并发测试
└── integration_test.rs      # 集成测试

sa/crates/sa/tests/
├── memory_performance.rs    # 性能基准测试
└── memory_stress.rs         # 压力测试
```

---

## Phase 1: 性能优化（第1-2周）

### Task 1: 添加LRU缓存层

**Files:**
- Create: `sa/crates/sa-core/src/cache/mod.rs`
- Create: `sa/crates/sa-core/src/cache/lru_cache.rs`
- Create: `sa/crates/sa-core/src/cache/memory_cache.rs`
- Modify: `sa/crates/sa-core/src/memory.rs`

- [ ] **Step 1: 创建缓存模块结构**

```rust
// sa/crates/sa-core/src/cache/mod.rs
pub mod lru_cache;
pub mod memory_cache;

pub use lru_cache::LruCache;
pub use memory_cache::MemoryCache;
```

- [ ] **Step 2: 实现LRU缓存**

```rust
// sa/crates/sa-core/src/cache/lru_cache.rs
use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

pub struct LruCache<K, V> {
    map: HashMap<K, CacheEntry<V>>,
    max_size: usize,
    default_ttl: Duration,
}

struct CacheEntry<V> {
    value: V,
    last_access: Instant,
    access_count: u64,
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            map: HashMap::with_capacity(max_size),
            max_size,
            default_ttl,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.map.get_mut(key) {
            if entry.last_access.elapsed() < self.default_ttl {
                entry.last_access = Instant::now();
                entry.access_count += 1;
                return Some(&entry.value);
            } else {
                self.map.remove(key);
            }
        }
        None
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.map.len() >= self.max_size {
            self.evict_oldest();
        }
        self.map.insert(key, CacheEntry {
            value,
            last_access: Instant::now(),
            access_count: 1,
        });
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self.map
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(k, _)| k.clone())
        {
            self.map.remove(&oldest_key);
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}
```

- [ ] **Step 3: 实现记忆缓存适配器**

```rust
// sa/crates/sa-core/src/cache/memory_cache.rs
use super::LruCache;
use crate::memory::MemorySearchResult;
use std::time::Duration;

pub struct MemoryCache {
    search_cache: LruCache<String, Vec<MemorySearchResult>>,
    file_cache: LruCache<String, String>,
}

impl MemoryCache {
    pub fn new() -> Self {
        Self {
            search_cache: LruCache::new(1000, Duration::from_secs(300)), // 5分钟TTL
            file_cache: LruCache::new(500, Duration::from_secs(600)),    // 10分钟TTL
        }
    }

    pub fn get_search_result(&mut self, query: &str) -> Option<Vec<MemorySearchResult>> {
        self.search_cache.get(&query.to_string())
    }

    pub fn cache_search_result(&mut self, query: String, results: Vec<MemorySearchResult>) {
        self.search_cache.insert(query, results);
    }

    pub fn get_file_content(&mut self, path: &str) -> Option<String> {
        self.file_cache.get(&path.to_string())
    }

    pub fn cache_file_content(&mut self, path: String, content: String) {
        self.file_cache.insert(path, content);
    }

    pub fn invalidate(&mut self, path: &str) {
        // 清除包含该路径的所有缓存
        self.search_cache.clear();
        self.file_cache.clear();
    }
}
```

- [ ] **Step 4: 集成缓存到memory.rs**

```rust
// sa/crates/sa-core/src/memory.rs 添加缓存支持
use crate::cache::MemoryCache;
use std::sync::Mutex;

// 在MemorySearchResult结构体后添加
pub struct MemorySearcher {
    cache: Mutex<MemoryCache>,
}

impl MemorySearcher {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(MemoryCache::new()),
        }
    }

    pub async fn search_with_cache(
        &self,
        workspace_root: &Path,
        query: &str,
        max_results: Option<usize>,
        min_score: Option<f64>,
        weights: Option<[f64; 3]>,
        filter_tags: Option<&[String]>,
    ) -> anyhow::Result<Vec<MemorySearchResult>> {
        // 检查缓存
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get_search_result(query) {
                return Ok(cached);
            }
        }

        // 执行搜索
        let results = search_markdown_memory(
            workspace_root,
            query,
            max_results,
            min_score,
            weights,
            filter_tags,
        ).await?;

        // 缓存结果
        {
            let mut cache = self.cache.lock().unwrap();
            cache.cache_search_result(query.to_string(), results.clone());
        }

        Ok(results)
    }
}
```

- [ ] **Step 5: 添加缓存测试**

```rust
// sa/crates/sa-core/src/cache/lru_cache.rs 添加测试
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_operations() {
        let mut cache = LruCache::new(3, Duration::from_secs(1));
        
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_eviction() {
        let mut cache = LruCache::new(2, Duration::from_secs(1));
        
        cache.insert("a", 1);
        thread::sleep(Duration::from_millis(10));
        cache.insert("b", 2);
        thread::sleep(Duration::from_millis(10));
        cache.insert("c", 3); // 应该淘汰"a"
        
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_ttl_expiration() {
        let mut cache = LruCache::new(3, Duration::from_millis(100));
        
        cache.insert("a", 1);
        assert_eq!(cache.get(&"a"), Some(&1));
        
        thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(&"a"), None); // 已过期
    }
}
```

- [ ] **Step 6: 运行测试验证**

```bash
cd sa/crates/sa-core
cargo test cache::tests --verbose
```

- [ ] **Step 7: 提交代码**

```bash
git add sa/crates/sa-core/src/cache/
git commit -m "feat(cache): add LRU cache layer for memory search"
```

---

### Task 2: 优化向量搜索性能

**Files:**
- Modify: `sa/crates/sa-core/src/vector_store.rs`
- Create: `sa/crates/sa-core/src/index/mod.rs`
- Create: `sa/crates/sa-core/src/index/sqlite_vec.rs`

- [ ] **Step 1: 添加sqlite-vec依赖**

```toml
# sa/crates/sa-core/Cargo.toml
[dependencies]
sqlite-vec = { version = "0.1", optional = true }

[features]
default = []
sqlite-vec = ["dep:sqlite-vec"]
```

- [ ] **Step 2: 实现向量索引抽象**

```rust
// sa/crates/sa-core/src/index/mod.rs
pub mod sqlite_vec;

pub trait VectorIndex {
    fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>>;
    fn upsert(&mut self, id: &str, embedding: &[f32]) -> anyhow::Result<()>;
    fn delete(&mut self, id: &str) -> anyhow::Result<()>;
}
```

- [ ] **Step 3: 实现sqlite-vec后端**

```rust
// sa/crates/sa-core/src/index/sqlite_vec.rs
use super::VectorIndex;
use crate::vector_store::VectorHit;
use rusqlite::{Connection, params};

pub struct SqliteVecIndex {
    conn: Connection,
}

impl SqliteVecIndex {
    pub fn new(db_path: &Path) -> anyhow::Result<Self> {
        let conn = Connection::open(db_path)?;
        
        // 启用sqlite-vec扩展
        rusqlite::vtab::array::load_module(&conn)?;
        
        // 创建向量表
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vec_items (
                id TEXT PRIMARY KEY,
                embedding FLOAT[384]
            );
            CREATE INDEX IF NOT EXISTS idx_vec_items_id ON vec_items(id);"
        )?;
        
        Ok(Self { conn })
    }
}

impl VectorIndex for SqliteVecIndex {
    fn search(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, vec_distance_cosine(embedding, ?1) as distance
             FROM vec_items
             ORDER BY distance ASC
             LIMIT ?2"
        )?;
        
        let rows = stmt.query_map(params![query, top_k], |row| {
            let id: String = row.get(0)?;
            let distance: f64 = row.get(1)?;
            Ok((id, 1.0 - distance)) // 转换为相似度分数
        })?;
        
        let mut results = Vec::new();
        for row in rows {
            let (id, score) = row?;
            results.push(VectorHit {
                path: id.clone(),
                start_line: 0,
                end_line: 0,
                score,
                snippet: String::new(),
                embedding: Vec::new(),
            });
        }
        
        Ok(results)
    }

    fn upsert(&mut self, id: &str, embedding: &[f32]) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_items (id, embedding) VALUES (?1, ?2)",
            params![id, embedding],
        )?;
        Ok(())
    }

    fn delete(&mut self, id: &str) -> anyhow::Result<()> {
        self.conn.execute("DELETE FROM vec_items WHERE id = ?1", params![id])?;
        Ok(())
    }
}
```

- [ ] **Step 4: 添加暴力搜索回退**

```rust
// sa/crates/sa-core/src/vector_store.rs 修改search方法
impl VectorIndex {
    pub fn search_optimized(&self, query_embedding: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>> {
        // 尝试使用sqlite-vec（如果可用且启用）
        #[cfg(feature = "sqlite-vec")]
        if let Some(vec_index) = &self.vec_index {
            return vec_index.search(query_embedding, top_k);
        }
        
        // 回退到暴力搜索
        self.search_brute_force(query_embedding, top_k)
    }
    
    fn search_brute_force(&self, query_embedding: &[f32], top_k: usize) -> anyhow::Result<Vec<VectorHit>> {
        // 现有实现...
        Ok(Vec::new())
    }
}
```

- [ ] **Step 5: 添加性能测试**

```rust
// sa/crates/sa-core/src/vector_store.rs 添加测试
#[cfg(test)]
mod performance_tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_brute_force_search(b: &mut Bencher) {
        let dir = tempdir().unwrap();
        let db = dir.path().join("bench.db");
        let idx = VectorIndex::open(&db, "test-model").unwrap();
        
        // 插入1000个向量
        for i in 0..1000 {
            let embedding: Vec<f32> = (0..384).map(|j| (i * 384 + j) as f32 * 0.001).collect();
            idx.upsert_chunk(&format!("doc_{}", i), 1, 10, "snippet", &embedding, 1000).unwrap();
        }
        
        let query: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        
        b.iter(|| {
            idx.search(&query, 10).unwrap();
        });
    }
}
```

- [ ] **Step 6: 运行性能测试**

```bash
cd sa/crates/sa-core
cargo bench --bench memory_performance
```

- [ ] **Step 7: 提交代码**

```bash
git add sa/crates/sa-core/src/index/ sa/crates sa/crates/sa-core/Cargo.toml
git commit -m "perf(vector): add sqlite-vec backend for faster ANN search"
```

---

### Task 3: 优化内存使用

**Files:**
- Modify: `sa/crates sa-core/src/working_memory.rs`
- Modify: `sa/crates/sa-core/src/config.rs`

- [ ] **Step 1: 添加内存限制配置**

```rust
// sa/crates/sa-core/src/working_memory.rs 修改WorkingMemoryConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkingMemoryConfig {
    /// Maximum number of messages held in the hot buffer.
    pub hot_buffer_max_messages: usize,
    /// Maximum total characters across all hot buffer messages.
    pub hot_buffer_max_chars: usize,
    /// Maximum memory usage in bytes (new field)
    pub max_memory_bytes: usize,
    /// Importance score threshold that marks a message as "consolidation
    /// candidate" for the dream pipeline.
    pub importance_threshold: f64,
    /// Fraction of the oldest hot buffer messages to compress when overflow
    /// is detected (e.g. 0.33 = compress the oldest 1/3).
    pub overflow_compress_fraction: f64,
    /// Whether overflow summarization is enabled.
    pub enable_summarization: bool,
    /// D8: Decay configuration.
    #[serde(default)]
    pub decay: Option<DecayConfig>,
}

impl Default for WorkingMemoryConfig {
    fn default() -> Self {
        Self {
            hot_buffer_max_messages: 12,
            hot_buffer_max_chars: 8_000,
            max_memory_bytes: 10 * 1024 * 1024, // 10MB默认限制
            importance_threshold: 0.6,
            overflow_compress_fraction: 0.33,
            enable_summarization: true,
            decay: None,
        }
    }
}
```

- [ ] **Step 2: 实现内存监控**

```rust
// sa/crates/sa-core/src/working_memory.rs 添加内存监控
impl WorkingMemory {
    /// 检查内存使用是否超限
    fn check_memory_limit(&self) -> bool {
        let estimated_size = self.estimate_memory_usage();
        estimated_size < self.config.max_memory_bytes
    }
    
    /// 估算当前内存使用量
    fn estimate_memory_usage(&self) -> usize {
        let mut total = 0;
        
        // hot_buffer内存估算
        for entry in &self.hot_buffer {
            total += entry.char_count * 4; // UTF-8字符
            total += std::mem::size_of::<WorkingMemoryEntry>();
        }
        
        // pinned_slots内存估算
        for (_, slot) in &self.pinned_slots {
            total += slot.content.len();
            total += slot.label.len();
            total += std::mem::size_of::<PinnedSlot>();
        }
        
        // scratchpad内存估算
        total += self.scratchpad.len();
        
        total
    }
    
    /// 强制内存清理
    fn force_cleanup(&mut self) {
        while !self.check_memory_limit() && !self.hot_buffer.is_empty() {
            if let Some(entry) = self.hot_buffer.pop_front() {
                self.total_chars = self.total_chars.saturating_sub(entry.char_count);
                self.message_count = self.message_count.saturating_sub(1);
            }
        }
    }
}
```

- [ ] **Step 3: 修改push_message添加内存检查**

```rust
// sa/crates/sa-core/src/working_memory.rs 修改push_message
pub fn push_message(&mut self, message: ChatMessage) -> OverflowAction {
    let entry = WorkingMemoryEntry::new(message, &self.config);
    let char_count = entry.char_count;

    // Pre-check: detect if buffer is at/over capacity before insert
    let needs_eviction = self.message_count >= self.config.hot_buffer_max_messages
        || self.total_chars + char_count > self.config.hot_buffer_max_chars
        || !self.check_memory_limit(); // 新增内存检查

    if needs_eviction {
        if self.config.enable_summarization {
            // Evict oldest to make room; return NeedsSummarization so caller
            // runs summarize_and_compress() after this insert.
            while self.message_count >= self.config.hot_buffer_max_messages
                || self.total_chars + char_count > self.config.hot_buffer_max_chars
                || !self.check_memory_limit()
            {
                if let Some(entry) = self.hot_buffer.pop_front() {
                    self.total_chars = self.total_chars.saturating_sub(entry.char_count);
                    self.message_count = self.message_count.saturating_sub(1);
                } else {
                    break;
                }
            }
            self.hot_buffer.push_back(entry);
            self.total_chars += char_count;
            self.message_count += 1;
            return OverflowAction::NeedsSummarization;
        } else {
            self.evict_oldest();
        }
    }

    self.hot_buffer.push_back(entry);
    self.total_chars += char_count;
    self.message_count += 1;

    OverflowAction::None
}
```

- [ ] **Step 4: 添加内存统计方法**

```rust
// sa/crates/sa-core/src/working_memory.rs 添加统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub hot_buffer_count: usize,
    pub hot_buffer_chars: usize,
    pub pinned_count: usize,
    pub compressions: u32,
    pub estimated_memory_bytes: usize,
    pub memory_limit_bytes: usize,
    pub memory_usage_percent: f64,
}

impl WorkingMemory {
    pub fn get_stats(&self) -> MemoryStats {
        let estimated = self.estimate_memory_usage();
        MemoryStats {
            hot_buffer_count: self.message_count,
            hot_buffer_chars: self.total_chars,
            pinned_count: self.pinned_slots.len(),
            compressions: self.compressions,
            estimated_memory_bytes: estimated,
            memory_limit_bytes: self.config.max_memory_bytes,
            memory_usage_percent: (estimated as f64 / self.config.max_memory_bytes as f64) * 100.0,
        }
    }
}
```

- [ ] **Step 5: 添加内存限制测试**

```rust
// sa/crates/sa-core/src/working_memory.rs 添加测试
#[test]
fn test_memory_limit_enforcement() {
    let config = WorkingMemoryConfig {
        max_memory_bytes: 1024, // 1KB限制
        hot_buffer_max_messages: 100,
        hot_buffer_max_chars: 100_000,
        ..Default::default()
    };
    let mut wm = WorkingMemory::from_config(config);
    
    // 添加大量小消息
    for i in 0..1000 {
        wm.push_message(make_message("user", format!("message {}", i)));
    }
    
    let stats = wm.get_stats();
    assert!(stats.estimated_memory_bytes <= 1024);
    assert!(stats.memory_usage_percent <= 100.0);
}
```

- [ ] **Step 6: 运行测试**

```bash
cd sa/crates/sa-core
cargo test working_memory::tests --verbose
```

- [ ] **Step 7: 提交代码**

```bash
git add sa/crates/sa-core/src/working_memory.rs
git commit -m "perf(memory): add memory limits and monitoring"
```

---

## Phase 2: 可靠性增强（第3-4周）

### Task 4: 实现并发控制

**Files:**
- Create: `sa/crates/sa-core/src/concurrency/mod.rs`
- Create: `sa/crates/sa-core/src/concurrency/file_lock.rs`
- Modify: `sa/crates/sa-core/src/memory.rs`

- [ ] **Step 1: 创建并发控制模块**

```rust
// sa/crates/sa-core/src/concurrency/mod.rs
pub mod file_lock;

pub use file_lock::FileLock;
```

- [ ] **Step 2: 实现文件锁**

```rust
// sa/crates/sa-core/src/concurrency/file_lock.rs
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use fs2::FileExt;

pub struct FileLock {
    file: File,
    path: PathBuf,
}

impl FileLock {
    pub fn acquire(lock_path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(lock_path)?;
        
        file.lock_exclusive()?;
        
        Ok(Self {
            file,
            path: lock_path.to_path_buf(),
        })
    }

    pub fn try_acquire(lock_path: &Path) -> io::Result<Option<Self>> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(lock_path)?;
        
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(Self {
                file,
                path: lock_path.to_path_buf(),
            })),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
        let _ = std::fs::remove_file(&self.path);
    }
}
```

- [ ] **Step 3: 为memory.rs添加锁保护**

```rust
// sa/crates/sa-core/src/memory.rs 添加锁支持
use crate::concurrency::FileLock;
use std::sync::Mutex;

pub struct MemoryManager {
    workspace_root: PathBuf,
    lock_dir: PathBuf,
    cache: Mutex<MemoryCache>,
}

impl MemoryManager {
    pub fn new(workspace_root: PathBuf) -> Self {
        let lock_dir = workspace_root.join(".locks");
        std::fs::create_dir_all(&lock_dir).ok();
        
        Self {
            workspace_root,
            lock_dir,
            cache: Mutex::new(MemoryCache::new()),
        }
    }

    pub fn write_daily_memory_with_lock(
        &self,
        date: &str,
        content: &str,
        metadata: &MemoryMetadata,
        filter: Option<&WriteFilterResult>,
    ) -> anyhow::Result<()> {
        let lock_path = self.lock_dir.join(format!("memory_{}.lock", date));
        let _lock = FileLock::acquire(&lock_path)
            .map_err(|e| anyhow::anyhow!("Failed to acquire lock: {}", e))?;
        
        write_daily_memory(&self.workspace_root, date, content, metadata, filter)?;
        
        // 清除相关缓存
        {
            let mut cache = self.cache.lock().unwrap();
            cache.invalidate(&format!("memory/{}.md", date));
        }
        
        Ok(())
    }

    pub fn read_daily_memory_with_lock(
        &self,
        date: &str,
    ) -> anyhow::Result<String> {
        let lock_path = self.lock_dir.join(format!("memory_{}.lock", date));
        let _lock = FileLock::acquire(&lock_path)
            .map_err(|e| anyhow::anyhow!("Failed to acquire lock: {}", e))?;
        
        let path = self.workspace_root.join(format!("memory/{}.md", date));
        let content = std::fs::read_to_string(&path)?;
        
        Ok(content)
    }
}
```

- [ ] **Step 4: 添加并发测试**

```rust
// sa/crates/sa-core/src/concurrency/file_lock.rs 添加测试
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use tempfile::tempdir;

    #[test]
    fn test_exclusive_lock() {
        let dir = tempdir().unwrap();
        let lock_path = dir.path().join("test.lock");
        
        let lock1 = FileLock::acquire(&lock_path).unwrap();
        
        // 第二个锁应该失败
        let result = FileLock::try_acquire(&lock_path).unwrap();
        assert!(result.is_none());
        
        drop(lock1);
        
        // 现在应该成功
        let lock2 = FileLock::try_acquire(&lock_path).unwrap();
        assert!(lock2.is_some());
    }

    #[test]
    fn test_concurrent_access() {
        let dir = tempdir().unwrap();
        let lock_path = dir.path().join("test.lock");
        let counter = Arc::new(Mutex::new(0));
        
        let mut handles = vec![];
        
        for _ in 0..10 {
            let lock_path = lock_path.clone();
            let counter = counter.clone();
            
            handles.push(thread::spawn(move || {
                let _lock = FileLock::acquire(&lock_path).unwrap();
                let mut count = counter.lock().unwrap();
                *count += 1;
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(*counter.lock().unwrap(), 10);
    }
}
```

- [ ] **Step 5: 运行并发测试**

```bash
cd sa/crates/sa-core
cargo test concurrency::tests --verbose
```

- [ ] **Step 6: 提交代码**

```bash
git add sa/crates/sa-core/src/concurrency/
git commit -m "feat(concurrency): add file locking for memory operations"
```

---

### Task 5: 增强错误处理和恢复

**Files:**
- Modify: `sa/crates/sa-core/src/memory.rs`
- Modify: `sa/crates/sa-core/src/vector_store.rs`
- Create: `sa/crates/sa-core/src/error.rs`

- [ ] **Step 1: 创建错误类型模块**

```rust
// sa/crates/sa-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    
    #[error("Lock error: {0}")]
    Lock(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

impl MemoryError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            MemoryError::Io(_) => true,
            MemoryError::Database(_) => true,
            MemoryError::Lock(_) => true,
            MemoryError::Cache(_) => true,
            MemoryError::Validation(_) => false,
            MemoryError::NotFound(_) => false,
            MemoryError::PermissionDenied(_) => false,
        }
    }
}
```

- [ ] **Step 2: 添加重试机制**

```rust
// sa/crates/sa-core/src/memory.rs 添加重试支持
use crate::error::MemoryError;
use std::time::Duration;

pub async fn with_retry<F, T, E>(
    operation: F,
    max_retries: u32,
    delay: Duration,
) -> Result<T, E>
where
    F: Fn() -> Result<T, E>,
    E: std::fmt::Display,
{
    let mut last_error = None;
    
    for attempt in 0..=max_retries {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt < max_retries {
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = max_retries,
                        error = %e,
                        "Operation failed, retrying..."
                    );
                    tokio::time::sleep(delay).await;
                }
                last_error = Some(e);
            }
        }
    }
    
    Err(last_error.unwrap())
}

// 使用示例
pub async fn search_with_retry(
    workspace_root: &Path,
    query: &str,
    max_results: Option<usize>,
) -> Result<Vec<MemorySearchResult>, MemoryError> {
    with_retry(
        || {
            // 实际搜索逻辑
            Ok(Vec::new())
        },
        3,
        Duration::from_millis(100),
    ).await
}
```

- [ ] **Step 3: 添加数据完整性检查**

```rust
// sa/crates/sa-core/src/memory.rs 添加完整性检查
pub fn verify_memory_integrity(workspace_root: &Path) -> anyhow::Result<IntegrityReport> {
    let mut report = IntegrityReport::new();
    
    // 检查记忆目录结构
    let memory_dir = workspace_root.join("memory");
    if !memory_dir.exists() {
        report.add_issue("memory directory missing");
    }
    
    // 检查每日记忆文件格式
    for entry in std::fs::read_dir(&memory_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            match verify_daily_file(&path) {
                Ok(_) => report.add_valid(path.display().to_string()),
                Err(e) => report.add_issue(format!("{}: {}", path.display(), e)),
            }
        }
    }
    
    // 检查语义数据库
    let db_path = workspace_root.join("memory/semantic.db");
    if db_path.exists() {
        match verify_semantic_db(&db_path) {
            Ok(_) => report.add_valid("semantic.db".to_string()),
            Err(e) => report.add_issue(format!("semantic.db: {}", e)),
        }
    }
    
    Ok(report)
}

fn verify_daily_file(path: &Path) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    
    // 检查YAML front matter格式
    if !content.starts_with("---") {
        return Err(anyhow::anyhow!("Missing YAML front matter"));
    }
    
    // 检查条目分隔符
    let entries = content.split("---").count();
    if entries < 2 {
        return Err(anyhow::anyhow!("No entries found"));
    }
    
    Ok(())
}

#[derive(Debug)]
pub struct IntegrityReport {
    pub valid_files: Vec<String>,
    pub issues: Vec<String>,
}

impl IntegrityReport {
    fn new() -> Self {
        Self {
            valid_files: Vec::new(),
            issues: Vec::new(),
        }
    }
    
    fn add_valid(&mut self, file: String) {
        self.valid_files.push(file);
    }
    
    fn add_issue(&mut self, issue: String) {
        self.issues.push(issue);
    }
    
    pub fn is_healthy(&self) -> bool {
        self.issues.is_empty()
    }
}
```

- [ ] **Step 4: 添加错误恢复测试**

```rust
// sa/crates/sa-core/src/error.rs 添加测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recoverability() {
        let io_error = MemoryError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_error.is_recoverable());
        
        let validation_error = MemoryError::Validation("invalid format".to_string());
        assert!(!validation_error.is_recoverable());
    }
}
```

- [ ] **Step 5: 运行测试**

```bash
cd sa/crates/sa-core
cargo test error::tests --verbose
```

- [ ] **Step 6: 提交代码**

```bash
git add sa/crates/sa-core/src/error.rs sa/crates/sa-core/src/memory.rs
git commit -m "feat(error): add error types and recovery mechanisms"
```

---

## Phase 3: 可维护性提升（第5-6周）

### Task 6: 简化配置系统

**Files:**
- Modify: `sa/crates/sa-core/src/config.rs`
- Create: `sa/crates/sa-core/src/config/presets.rs`

- [ ] **Step 1: 创建预设配置模块**

```rust
// sa/crates/sa-core/src/config/presets.rs
use super::Config;

pub struct PresetConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub config: Config,
}

pub fn get_preset(name: &str) -> Option<PresetConfig> {
    match name {
        "minimal" => Some(minimal_preset()),
        "balanced" => Some(balanced_preset()),
        "performance" => Some(performance_preset()),
        "memory-optimized" => Some(memory_optimized_preset()),
        _ => None,
    }
}

fn minimal_preset() -> PresetConfig {
    PresetConfig {
        name: "minimal",
        description: "Minimal configuration for basic usage",
        config: Config {
            working_memory: WorkingMemoryConfig {
                hot_buffer_max_messages: 5,
                hot_buffer_max_chars: 4_000,
                max_memory_bytes: 5 * 1024 * 1024, // 5MB
                importance_threshold: 0.7,
                overflow_compress_fraction: 0.5,
                enable_summarization: false,
                decay: None,
            },
            // ... 其他配置
        },
    }
}

fn balanced_preset() -> PresetConfig {
    PresetConfig {
        name: "balanced",
        description: "Balanced configuration for most users",
        config: Config {
            working_memory: WorkingMemoryConfig {
                hot_buffer_max_messages: 12,
                hot_buffer_max_chars: 8_000,
                max_memory_bytes: 10 * 1024 * 1024, // 10MB
                importance_threshold: 0.6,
                overflow_compress_fraction: 0.33,
                enable_summarization: true,
                decay: Some(DecayConfig::default()),
            },
            // ... 其他配置
        },
    }
}

fn performance_preset() -> PresetConfig {
    PresetConfig {
        name: "performance",
        description: "High-performance configuration for heavy workloads",
        config: Config {
            working_memory: WorkingMemoryConfig {
                hot_buffer_max_messages: 20,
                hot_buffer_max_chars: 16_000,
                max_memory_bytes: 50 * 1024 * 1024, // 50MB
                importance_threshold: 0.5,
                overflow_compress_fraction: 0.25,
                enable_summarization: true,
                decay: Some(DecayConfig {
                    decay_after_days: 7,
                    decay_base: 0.95,
                    access_boost: 0.05,
                    min_importance: 0.1,
                    alpha: 0.15,
                }),
            },
            // ... 其他配置
        },
    }
}

fn memory_optimized_preset() -> PresetConfig {
    PresetConfig {
        name: "memory-optimized",
        description: "Memory-optimized configuration for constrained environments",
        config: Config {
            working_memory: WorkingMemoryConfig {
                hot_buffer_max_messages: 8,
                hot_buffer_max_chars: 4_000,
                max_memory_bytes: 2 * 1024 * 1024, // 2MB
                importance_threshold: 0.8,
                overflow_compress_fraction: 0.5,
                enable_summarization: true,
                decay: Some(DecayConfig {
                    decay_after_days: 1,
                    decay_base: 0.8,
                    access_boost: 0.2,
                    min_importance: 0.2,
                    alpha: 0.05,
                }),
            },
            // ... 其他配置
        },
    }
}
```

- [ ] **Step 2: 添加配置验证**

```rust
// sa/crates/sa-core/src/config.rs 添加验证
impl Config {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        
        // 验证工作记忆配置
        if self.working_memory.hot_buffer_max_messages == 0 {
            errors.push("hot_buffer_max_messages must be greater than 0".to_string());
        }
        
        if self.working_memory.hot_buffer_max_chars == 0 {
            errors.push("hot_buffer_max_chars must be greater than 0".to_string());
        }
        
        if self.working_memory.max_memory_bytes < 1024 * 1024 {
            errors.push("max_memory_bytes must be at least 1MB".to_string());
        }
        
        if self.working_memory.importance_threshold < 0.0 || self.working_memory.importance_threshold > 1.0 {
            errors.push("importance_threshold must be between 0.0 and 1.0".to_string());
        }
        
        // 验证LLM配置
        if self.llm.base_url.is_empty() {
            errors.push("llm.base_url is required".to_string());
        }
        
        if self.llm.api_key.is_empty() {
            errors.push("llm.api_key is required".to_string());
        }
        
        if self.llm.model.is_empty() {
            errors.push("llm.model is required".to_string());
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
```

- [ ] **Step 3: 添加配置测试**

```rust
// sa/crates/sa-core/src/config.rs 添加测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = Config::default();
        invalid_config.llm.base_url = String::new();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_preset_configs() {
        let presets = ["minimal", "balanced", "performance", "memory-optimized"];
        
        for preset_name in &presets {
            let preset = get_preset(preset_name).unwrap();
            assert_eq!(preset.name, *preset_name);
            assert!(preset.config.validate().is_ok());
        }
    }
}
```

- [ ] **Step 4: 运行测试**

```bash
cd sa/crates/sa-core
cargo test config::tests --verbose
```

- [ ] **Step 5: 提交代码**

```bash
git add sa/crates/sa-core/src/config/
git commit -m "feat(config): add preset configurations and validation"
```

---

### Task 7: 添加监控和指标

**Files:**
- Create: `sa/crates/sa-core/src/metrics.rs`
- Modify: `sa/crates/sa-core/src/memory.rs`

- [ ] **Step 1: 创建指标模块**

```rust
// sa/crates/sa-core/src/metrics.rs
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug)]
pub struct MemoryMetrics {
    // 搜索指标
    pub search_count: AtomicU64,
    pub search_duration_ms: AtomicU64,
    pub search_cache_hits: AtomicU64,
    pub search_cache_misses: AtomicU64,
    
    // 写入指标
    pub write_count: AtomicU64,
    pub write_duration_ms: AtomicU64,
    pub write_errors: AtomicU64,
    
    // 缓存指标
    pub cache_size: AtomicU64,
    pub cache_memory_bytes: AtomicU64,
    
    // 内存指标
    pub working_memory_count: AtomicU64,
    pub working_memory_chars: AtomicU64,
}

impl MemoryMetrics {
    pub fn new() -> Self {
        Self {
            search_count: AtomicU64::new(0),
            search_duration_ms: AtomicU64::new(0),
            search_cache_hits: AtomicU64::new(0),
            search_cache_misses: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            write_duration_ms: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
            cache_size: AtomicU64::new(0),
            cache_memory_bytes: AtomicU64::new(0),
            working_memory_count: AtomicU64::new(0),
            working_memory_chars: AtomicU64::new(0),
        }
    }

    pub fn record_search(&self, duration_ms: u64, cache_hit: bool) {
        self.search_count.fetch_add(1, Ordering::Relaxed);
        self.search_duration_ms.fetch_add(duration_ms, Ordering::Relaxed);
        
        if cache_hit {
            self.search_cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.search_cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_write(&self, duration_ms: u64, success: bool) {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        self.write_duration_ms.fetch_add(duration_ms, Ordering::Relaxed);
        
        if !success {
            self.write_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn update_cache_stats(&self, size: u64, memory_bytes: u64) {
        self.cache_size.store(size, Ordering::Relaxed);
        self.cache_memory_bytes.store(memory_bytes, Ordering::Relaxed);
    }

    pub fn update_working_memory_stats(&self, count: u64, chars: u64) {
        self.working_memory_count.store(count, Ordering::Relaxed);
        self.working_memory_chars.store(chars, Ordering::Relaxed);
    }

    pub fn get_summary(&self) -> MetricsSummary {
        let search_count = self.search_count.load(Ordering::Relaxed);
        let search_duration = self.search_duration_ms.load(Ordering::Relaxed);
        let cache_hits = self.search_cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.search_cache_misses.load(Ordering::Relaxed);
        
        MetricsSummary {
            total_searches: search_count,
            avg_search_duration_ms: if search_count > 0 { search_duration / search_count } else { 0 },
            cache_hit_rate: if cache_hits + cache_misses > 0 {
                cache_hits as f64 / (cache_hits + cache_misses) as f64
            } else {
                0.0
            },
            total_writes: self.write_count.load(Ordering::Relaxed),
            write_error_rate: {
                let writes = self.write_count.load(Ordering::Relaxed);
                let errors = self.write_errors.load(Ordering::Relaxed);
                if writes > 0 { errors as f64 / writes as f64 } else { 0.0 }
            },
            cache_size: self.cache_size.load(Ordering::Relaxed),
            cache_memory_mb: self.cache_memory_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            working_memory_count: self.working_memory_count.load(Ordering::Relaxed),
            working_memory_chars: self.working_memory_chars.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MetricsSummary {
    pub total_searches: u64,
    pub avg_search_duration_ms: u64,
    pub cache_hit_rate: f64,
    pub total_writes: u64,
    pub write_error_rate: f64,
    pub cache_size: u64,
    pub cache_memory_mb: f64,
    pub working_memory_count: u64,
    pub working_memory_chars: u64,
}
```

- [ ] **Step 2: 集成指标到memory.rs**

```rust
// sa/crates/sa-core/src/memory.rs 添加指标支持
use crate::metrics::MemoryMetrics;
use std::sync::Arc;

pub struct InstrumentedMemoryManager {
    inner: MemoryManager,
    metrics: Arc<MemoryMetrics>,
}

impl InstrumentedMemoryManager {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            inner: MemoryManager::new(workspace_root),
            metrics: Arc::new(MemoryMetrics::new()),
        }
    }

    pub async fn search_with_metrics(
        &self,
        query: &str,
        max_results: Option<usize>,
    ) -> anyhow::Result<Vec<MemorySearchResult>> {
        let start = Instant::now();
        let cache_hit = false; // 实际实现中检查缓存
        
        let result = self.inner.search_with_cache(query, max_results).await;
        
        let duration = start.elapsed().as_millis() as u64;
        self.metrics.record_search(duration, cache_hit);
        
        result
    }

    pub fn get_metrics(&self) -> MetricsSummary {
        self.metrics.get_summary()
    }
}
```

- [ ] **Step 3: 添加指标测试**

```rust
// sa/crates/sa-core/src/metrics.rs 添加测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = MemoryMetrics::new();
        
        metrics.record_search(100, true);
        metrics.record_search(200, false);
        metrics.record_write(50, true);
        
        let summary = metrics.get_summary();
        assert_eq!(summary.total_searches, 2);
        assert_eq!(summary.avg_search_duration_ms, 150);
        assert_eq!(summary.cache_hit_rate, 0.5);
        assert_eq!(summary.total_writes, 1);
    }
}
```

- [ ] **Step 4: 运行测试**

```bash
cd sa/crates/sa-core
cargo test metrics::tests --verbose
```

- [ ] **Step 5: 提交代码**

```bash
git add sa/crates/sa-core/src/metrics.rs sa/crates/sa-core/src/memory.rs
git commit -m "feat(metrics): add performance monitoring and metrics"
```

---

### Task 8: 完善文档和示例

**Files:**
- Create: `sa/docs/memory-system-architecture.md`
- Create: `sa/docs/memory-system-usage.md`
- Create: `sa/examples/memory_optimization.rs`

- [ ] **Step 1: 创建架构文档**

```markdown
# sa/docs/memory-system-architecture.md

# SA 记忆系统架构

## 概述

SA记忆系统采用分层架构，从短期工作记忆到长期持久化记忆，支持多种检索模式。

## 架构图

┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│  Working Memory    │  Semantic Memory  │  Procedural Memory  │
│  (Hot Buffer)      │  (SQLite KG)      │  (Auto-discover)    │
├─────────────────────────────────────────────────────────────┤
│  Markdown Memory   │  Vector Memory    │  Cold Storage       │
│  (Files)           │  (Embeddings)     │  (Archive)          │
├─────────────────────────────────────────────────────────────┤
│  Cache Layer       │  Concurrency      │  Metrics            │
│  (LRU)             │  (File Locks)     │  (Monitoring)       │
└─────────────────────────────────────────────────────────────┘

## 核心组件

### 1. 工作记忆 (Working Memory)
- 位置: `working_memory.rs`
- 功能: 实时会话记忆，三段式结构
- 配置: `[working_memory]` in sa.toml

### 2. 语义记忆 (Semantic Memory)
- 位置: `semantic_memory.rs`
- 功能: 知识图谱三元组存储
- 配置: `[semantic_memory]` in sa.toml

### 3. Markdown记忆
- 位置: `memory.rs`
- 功能: 长期记忆文件管理
- 存储: `memory/YYYY-MM-DD.md`

### 4. 向量记忆 (Vector Memory)
- 位置: `vector_store.rs`, `memory_indexer.rs`
- 功能: 语义向量搜索
- 配置: `[vector]` in sa.toml

### 5. 程序记忆 (Procedural Memory)
- 位置: `procedural_memory.rs`
- 功能: 自动发现工具调用模式
- 存储: `memory/procedures/`

### 6. Dream蒸馏
- 位置: `dream.rs`
- 功能: 夜间记忆整合和优化
- 配置: `[dream]` in sa.toml

## 数据流

1. 用户输入 → 工作记忆 (hot_buffer)
2. 重要信息 → 长期记忆 (memory/YYYY-MM-DD.md)
3. 知识提取 → 语义记忆 (semantic.db)
4. 向量嵌入 → 向量记忆 (vector.db)
5. 模式发现 → 程序记忆 (memory/procedures/)
6. 夜间整合 → Dream蒸馏 (memory/dreams/)

## 配置示例

```toml
[working_memory]
hot_buffer_max_messages = 12
hot_buffer_max_chars = 8000
max_memory_bytes = 10485760  # 10MB
importance_threshold = 0.6

[semantic_memory]
enabled = true
db_path = "memory/semantic.db"
decay_rate = 0.01

[dream]
enabled = true
daily_note_lookback_days = 3
recent_session_segments = 6
```
```

- [ ] **Step 2: 创建使用指南**

```markdown
# sa/docs/memory-system-usage.md

# SA 记忆系统使用指南

## 快速开始

### 1. 选择配置预设

SA提供4种配置预设：

- `minimal`: 最小配置，适合资源受限环境
- `balanced`: 平衡配置，适合大多数用户
- `performance`: 高性能配置，适合重度使用
- `memory-optimized`: 内存优化配置，适合嵌入式环境

在 `sa.toml` 中使用：

```toml
[working_memory]
preset = "balanced"  # 使用预设配置
```

### 2. 手动配置

```toml
[working_memory]
hot_buffer_max_messages = 12
hot_buffer_max_chars = 8000
max_memory_bytes = 10485760  # 10MB
importance_threshold = 0.6
overflow_compress_fraction = 0.33
enable_summarization = true

[working_memory.decay]
decay_after_days = 3
decay_base = 0.90
access_boost = 0.10
min_importance = 0.15
alpha = 0.10
```

## 常见用例

### 1. 高频对话场景

```toml
[working_memory]
hot_buffer_max_messages = 20
hot_buffer_max_chars = 16000
max_memory_bytes = 52428800  # 50MB
importance_threshold = 0.5
```

### 2. 长期运行场景

```toml
[working_memory]
hot_buffer_max_messages = 8
hot_buffer_max_chars = 4000
max_memory_bytes = 2097152  # 2MB
importance_threshold = 0.8

[working_memory.decay]
decay_after_days = 1
decay_base = 0.8
```

### 3. 多用户共享场景

```toml
[semantic_memory]
enabled = true
scope = "shared"

[privacy]
pii_detection = true
sanitize_policy = "hash"
```

## 监控和调试

### 1. 查看内存统计

```bash
# 通过WebSocket获取统计
curl -X POST http://localhost:8765/ws \
  -H "Content-Type: application/json" \
  -d '{"type": "get_memory_stats"}'
```

### 2. 查看性能指标

```bash
# 获取性能指标
curl -X POST http://localhost:8765/ws \
  -H "Content-Type: application/json" \
  -d '{"type": "get_metrics"}'
```

### 3. 验证数据完整性

```bash
# 验证记忆文件完整性
cargo run --bin memory-integrity-check
```

## 故障排除

### 问题1: 内存使用过高

**症状**: 进程内存持续增长

**解决方案**:
1. 降低 `max_memory_bytes`
2. 增加 `overflow_compress_fraction`
3. 启用 `enable_summarization`

### 问题2: 搜索速度慢

**症状**: 搜索响应时间 > 1秒

**解决方案**:
1. 启用缓存层
2. 减少 `max_results`
3. 使用sqlite-vec后端

### 问题3: 写入冲突

**症状**: 并发写入时数据丢失

**解决方案**:
1. 确保文件锁正常工作
2. 减少并发写入频率
3. 使用事务性写入
```

- [ ] **Step 3: 创建优化示例**

```rust
// sa/examples/memory_optimization.rs
use sa_core::memory::MemoryManager;
use sa_core::working_memory::{WorkingMemory, WorkingMemoryConfig};
use sa_core::metrics::MemoryMetrics;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化日志
    tracing_subscriber::fmt::init();
    
    let workspace_root = PathBuf::from("./workspace");
    
    // 1. 创建内存优化配置
    let config = WorkingMemoryConfig {
        hot_buffer_max_messages: 8,
        hot_buffer_max_chars: 4_000,
        max_memory_bytes: 2 * 1024 * 1024, // 2MB限制
        importance_threshold: 0.8,
        overflow_compress_fraction: 0.5,
        enable_summarization: true,
        ..Default::default()
    };
    
    // 2. 创建带监控的记忆管理器
    let metrics = Arc::new(MemoryMetrics::new());
    let memory_manager = MemoryManager::new(workspace_root.clone());
    
    // 3. 创建工作记忆
    let mut working_memory = WorkingMemory::from_config(config);
    
    // 4. 模拟对话
    for i in 0..100 {
        let message = format!("用户消息 {}: 这是一条测试消息", i);
        working_memory.push_message(sa_core::openai::ChatMessage::text("user", &message));
        
        // 每10条消息检查一次统计
        if i % 10 == 0 {
            let stats = working_memory.get_stats();
            println!("消息 {}: 内存使用 {:.2}%", i, stats.memory_usage_percent);
            
            if stats.memory_usage_percent > 90.0 {
                println!("警告: 内存使用超过90%，触发清理");
                working_memory.force_cleanup();
            }
        }
    }
    
    // 5. 输出最终统计
    let final_stats = working_memory.get_stats();
    println!("最终统计:");
    println!("  消息数量: {}", final_stats.hot_buffer_count);
    println!("  内存使用: {:.2}%", final_stats.memory_usage_percent);
    println!("  压缩次数: {}", final_stats.compressions);
    
    Ok(())
}
```

- [ ] **Step 4: 运行示例**

```bash
cd sa
cargo run --example memory_optimization
```

- [ ] **Step 5: 提交文档**

```bash
git add sa/docs/ sa/examples/
git commit -m "docs: add memory system architecture and usage guides"
```

---

## 里程碑和时间表

| 阶段 | 任务 | 预计时间 | 交付物 |
|------|------|----------|--------|
| Phase 1 | Task 1-3 | 第1-2周 | 性能优化完成，基准测试通过 |
| Phase 2 | Task 4-5 | 第3-4周 | 并发控制和错误处理完成 |
| Phase 3 | Task 6-8 | 第5-6周 | 配置简化和文档完善 |

## 验收标准

### Phase 1 验收
- [ ] LRU缓存命中率 > 80%
- [ ] 向量搜索性能提升 > 50%
- [ ] 内存使用限制生效

### Phase 2 验收
- [ ] 并发写入无数据丢失
- [ ] 错误恢复机制正常工作
- [ ] 数据完整性检查通过

### Phase 3 验收
- [ ] 配置预设可用
- [ ] 监控指标正常输出
- [ ] 文档完整且准确

## 风险和缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| sqlite-vec兼容性问题 | 高 | 保留暴力搜索回退 |
| 性能回归 | 中 | 添加性能基准测试 |
| 配置迁移困难 | 低 | 提供迁移脚本 |

## 后续优化方向

1. **分布式支持**: 跨设备记忆同步
2. **可视化工具**: 记忆图谱可视化
3. **自动调优**: 基于使用模式的自动配置优化
4. **语义搜索增强**: 支持更复杂的查询语言
```

- [ ] **Step 6: 最终提交**

```bash
git add sa/docs/superpowers/plans/
git commit -m "docs: add complete memory system optimization plan"
```

---

## 执行选项

**计划已保存到 `sa/docs/superpowers/plans/2026-05-02-memory-system-optimization.md`**

两种执行方式：

**1. Subagent-Driven（推荐）** - 每个任务分派新子代理，任务间审查，快速迭代

**2. Inline Execution** - 在当前会话中执行任务，批量执行带检查点

**选择哪种方式？**