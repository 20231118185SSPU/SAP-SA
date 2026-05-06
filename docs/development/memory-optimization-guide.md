# 记忆系统优化 - 编译和测试指南

## 编译验证

```bash
# 进入 SA 目录
cd E:\SA\SAP-6.2\sa

# 检查 sa-core 编译
cargo check -p sa-core

# 检查整个项目编译
cargo check
```

## 单元测试

```bash
# 运行 sa-core 所有测试
cargo test -p sa-core

# 运行特定模块测试
cargo test -p sa-core noise_assessment
cargo test -p sa-core memory_pointer
cargo test -p sa-core memory_metabolism
cargo test -p sa-core timeline_retrieval
cargo test -p sa-core skill_metabolism

# 运行所有测试并显示输出
cargo test -p sa-core -- --nocapture
```

## 集成测试

```bash
# 复制示例配置
cp sa.example.toml sa.toml

# 编辑 sa.toml，添加以下配置段：

# [working_memory.noise]
# enabled = true
# high_noise_threshold = 0.4

# [memory_pointer]
# enabled = true
# auto_create_structure = true

# [memory_metabolism]
# enabled = true
# archive_after_days = 30

# [timeline_retrieval]
# enabled = true
# max_results = 50

# [skill_metabolism]
# enabled = true
# min_usage_threshold = 5

# 启动 SA 后端
cargo run -p sa --release
```

## 验证清单

### P0: 噪音控制
- [ ] `noise_assessment.rs` 编译通过
- [ ] `NoiseConfig` 可以从 TOML 反序列化
- [ ] `NoiseAssessor::assess_working_memory()` 返回正确结果
- [ ] `format_noise_report()` 生成可读报告

### P1: 记忆指针
- [ ] `memory_pointer.rs` 编译通过
- [ ] `MemoryPointerCollection::parse_from_content()` 正确解析指针
- [ ] `build_prompt_block()` 支持指针模式
- [ ] 指针目录结构可自动创建

### P2: 记忆新陈代谢
- [ ] `memory_metabolism.rs` 编译通过
- [ ] `MetabolismConfig` 可以从 TOML 反序列化
- [ ] `MemoryMetabolism::process_lifecycle()` 正确处理归档/晋升/删除
- [ ] `generate_report()` 生成可读报告

### P3: 时间线检索
- [ ] `timeline_retrieval.rs` 编译通过
- [ ] `TimelineConfig` 可以从 TOML 反序列化
- [ ] `TimelineRetrieval::search_by_range()` 返回正确结果
- [ ] `generate_timeline_summary()` 生成可读摘要

### P4: Skills 淘汰
- [ ] `skill_metabolism.rs` 编译通过
- [ ] `SkillMetabolismConfig` 可以从 TOML 反序列化
- [ ] `SkillMetabolism::process_lifecycle()` 正确处理淘汰/合并/晋升
- [ ] `generate_report()` 生成可读报告

## 常见问题

### Q: 编译错误 "cannot find type"
A: 确保所有新模块都已在 `lib.rs` 中注册：
```rust
pub mod noise_assessment;
pub mod memory_pointer;
pub mod memory_metabolism;
pub mod timeline_retrieval;
pub mod skill_metabolism;
```

### Q: 配置不生效
A: 检查 `sa.toml` 中的配置段名称是否正确，注意下划线：
- `[working_memory.noise]` 不是 `[working_memory.noise_assessment]`
- `[memory_pointer]` 不是 `[memory.pointers]`

### Q: 测试失败
A: 确保所有依赖都已正确导入：
```bash
cargo clean
cargo test -p sa-core
```
