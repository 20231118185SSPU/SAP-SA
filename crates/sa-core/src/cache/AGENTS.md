<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-08 | Updated: 2026-05-08 -->

# sa-core/src/cache/

## Purpose

缓存模块。提供基于 moka 的内存缓存，用于加速记忆搜索和文件内容读取。

## Key Files

| File | Description |
|------|-------------|
| `mod.rs` | 模块导出入口，re-export `MemoryCache` |
| `memory_cache.rs` | MemoryCache 实现 — 双层缓存（搜索结果 + 文件内容） |

## Module Details

### memory_cache.rs
`MemoryCache` 使用 [moka](https://docs.rs/moka) 实现两个独立缓存：

| Cache | Capacity | TTL | Purpose |
|-------|----------|-----|---------|
| `search_cache` | 1000 entries | 300s (5 min) | 记忆搜索结果缓存 |
| `file_cache` | 500 entries | 600s (10 min) | 文件内容缓存 |

关键方法：
- `get_search_result(query)` / `cache_search_result(query, results)` — 搜索结果缓存
- `get_file_content(path)` / `cache_file_content(path, content)` — 文件内容缓存
- `invalidate_path(path)` — 指定路径失效（同时清空搜索缓存）
- `invalidate_all()` — 全部失效

## For AI Agents

### Working In This Directory
- 缓存容量和 TTL 在 `MemoryCache::new()` 中配置
- 修改缓存结构需同步检查 `memory.rs` 中的 `MemoryManager::search_with_cache`
- 编译验证：`cargo build -p sa-core`

### Common Patterns
- moka `Cache` 自动处理并发和过期
- 缓存 key 为 String（query 或 path）
- 失效策略：文件变更时清空对应 path + 全量搜索缓存

<!-- MANUAL: -->
