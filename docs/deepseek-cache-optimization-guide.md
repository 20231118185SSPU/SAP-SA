# DeepSeek V4 缓存优化 - 最终执行指南

## 一、验证步骤

### 步骤 1：编译检查

```bash
cd E:\SA\SAP-6.2\sa
cargo check -p sa-core
```

**预期结果**：
- 无编译错误
- 可能有警告（如未使用的导入），但不影响功能

**如果遇到错误**：
- 检查 Rust 工具链：`rustc --version`
- 清理并重新编译：`cargo clean -p sa-core && cargo check -p sa-core`

---

### 步骤 2：运行测试

```bash
cd E:\SA\SAP-6.2\sa
cargo test -p sa-core
```

**预期结果**：
- 所有测试通过
- 包含新增的 DeepSeek V4 缓存相关测试

**测试内容**：
- `openai::tests::deepseek_cache_hit_returns_explicit_field_first`
- `openai::tests::deepseek_cache_hit_falls_back_to_generic_fields`
- `openai::tests::deepseek_cache_miss_calculated_when_not_explicit`
- `openai::tests::deepseek_v4_flash_cost_calculation`
- `openai::tests::cache_stats_summary_format`
- `cost_budget::tests::test_deepseek_pricing_v4_flash`
- `cost_budget::tests::test_deepseek_pricing_v4_pro`
- `cost_budget::tests::test_deepseek_cost_calculation`
- `cost_budget::tests::test_deepseek_savings_calculation`
- `cost_budget::tests::test_record_deepseek_usage`
- `cost_budget::tests::test_cache_hit_rate_today`
- `cost_budget::tests::test_deepseek_budget_status_line`
- `prompt_optimizer::tests::test_analyze_stable_prompt`
- `prompt_optimizer::tests::test_analyze_prompt_with_timestamp`
- `prompt_optimizer::tests::test_analyze_prompt_with_request_id`
- `prompt_optimizer::tests::test_analyze_short_prompt`
- `prompt_optimizer::tests::test_build_optimized_messages`
- `prompt_optimizer::tests::test_generate_optimized_template`
- `cache_monitor::tests::test_cache_monitor_new`
- `cache_monitor::tests::test_record_request`
- `cache_monitor::tests::test_generate_report_empty`
- `cache_monitor::tests::test_generate_report_with_data`
- `cache_monitor::tests::test_hit_rate_trend_insufficient_data`
- `cache_monitor::tests::test_hit_rate_trend_stable`
- `cache_monitor::tests::test_recommendations_low_hit_rate`
- `cache_monitor::tests::test_recommendations_high_hit_rate`
- `cache_monitor::tests::test_summary_string`
- `cache_monitor::tests::test_clear`

---

## 二、配置文件

### 2.1 复制配置文件

```bash
cd E:\SA\SAP-6.2\sa
cp sa.deepseek.toml sa.toml
```

### 2.2 编辑配置文件

打开 `sa.toml`，修改以下配置：

```toml
[llm]
# DeepSeek V4 API 端点
base_url = "https://api.deepseek.com"
# 你的 DeepSeek API Key（替换为真实 key）
api_key = "sk-your-api-key-here"
# 使用 DeepSeek V4 Flash 模型
model = "deepseek-v4-flash"
# 使用 OpenAI 兼容的 chat_completions 协议
wire_api = "chat_completions"
# 使用 Bearer 认证方式
auth_style = "bearer"

[cache]
# 启用缓存监控
enabled = true
# 使用 V4 Flash 模型进行成本计算
model = "v4_flash"
# 记录每次 API 调用的缓存统计
log_stats = true
# 追踪成本
cost_tracking = true
```

---

## 三、代码集成示例

### 3.1 创建带缓存监控的 AgentRunner

```rust
use sa_core::agent::{AgentRunner, AgentRunnerConfig};
use sa_core::cost_budget::DeepSeekPricing;
use sa_core::openai::OpenAiClient;
use sa_core::tools::ToolExecutor;
use sa_core::skills::SkillRegistry;
use std::sync::Arc;

// 创建 DeepSeek V4 Flash 定价配置
let pricing = DeepSeekPricing::v4_flash();

// 创建 OpenAI 客户端
let llm = OpenAiClient::new(
    "https://api.deepseek.com".to_string(),
    "sk-your-api-key".to_string(),
)?;

// 创建工具执行器和技能注册表
let tools = ToolExecutor::new();
let skills = Arc::new(SkillRegistry::new());

// 创建代理配置
let cfg = AgentRunnerConfig {
    model: "deepseek-v4-flash".to_string(),
    system_role_name: "system".to_string(),
    reasoning_effort: None,
    compaction: Default::default(),
};

// 创建带缓存监控的 runner
let runner = AgentRunner::with_cache_monitoring(
    llm,
    tools,
    skills,
    cfg,
    pricing,
);
```

### 3.2 动态启用缓存监控

```rust
use sa_core::cost_budget::DeepSeekPricing;

// 创建 runner
let mut runner = AgentRunner::new(llm, tools, skills, cfg);

// 动态启用缓存监控
let pricing = DeepSeekPricing::v4_flash();
runner.enable_cache_monitoring(pricing);
```

### 3.3 查看缓存统计

```rust
// 获取缓存监控器
if let Some(monitor) = runner.cache_monitor() {
    let report = monitor.generate_report();
    
    println!("📊 缓存性能报告");
    println!("  总请求数: {}", report.total_requests);
    println!("  平均命中率: {:.1}%", report.average_hit_rate * 100.0);
    println!("  总成本: ${:.4}", report.total_estimated_cost_usd);
    println!("  总节省: ${:.4}", report.total_estimated_savings_usd);
    println!("  命中率趋势: {:?}", report.hit_rate_trend);
    
    println!("\n💡 优化建议:");
    for rec in &report.recommendations {
        println!("  - {}", rec);
    }
}

// 获取成本追踪器
if let Some(tracker) = runner.cost_tracker() {
    let config = CostBudgetConfig::default();
    println!("\n📈 今日统计");
    println!("  {}", tracker.deepseek_budget_status_line(&config));
}
```

---

## 四、API 响应处理

### 4.1 解析 DeepSeek V4 响应

```rust
use sa_core::openai::ChatUsage;

// 假设你从 API 响应中获取了 usage
let usage: ChatUsage = response.usage.unwrap();

// 获取缓存统计
let cache_hit = usage.deepseek_cache_hit();
let cache_miss = usage.deepseek_cache_miss();
let hit_rate = usage.cache_hit_rate();

println!("缓存命中: {}/{} ({:.1}%)", 
    cache_hit, 
    cache_hit + cache_miss, 
    hit_rate * 100.0
);

// 计算成本
let flash_cost = usage.deepseek_v4_flash_cost();
let pro_cost = usage.deepseek_v4_pro_cost();

println!("V4 Flash 成本: ${:.6}", flash_cost);
println!("V4 Pro 成本: ${:.6}", pro_cost);

// 获取摘要
println!("{}", usage.cache_stats_summary());
```

### 4.2 使用成本追踪器

```rust
use sa_core::cost_budget::{CostBudgetTracker, DeepSeekPricing};
use chrono::Local;

let pricing = DeepSeekPricing::v4_flash();
let mut tracker = CostBudgetTracker::default();
let now = Local::now();

// 记录使用情况
tracker.record_deepseek_usage(
    &now,
    usage.deepseek_cache_hit(),
    usage.deepseek_cache_miss(),
    usage.output_tokens.unwrap_or(0),
    &pricing,
);

// 查看统计
println!("今日缓存命中率: {:.1}%", tracker.cache_hit_rate_today() * 100.0);
println!("今日节省: ${:.4}", tracker.estimated_savings_today(&pricing));
```

---

## 五、Prompt 优化

### 5.1 分析 System Prompt 稳定性

```rust
use sa_core::prompt_optimizer::PromptOptimizer;

let optimizer = PromptOptimizer::default();
let system_prompt = "你是一个有帮助的助手。请用中文回答问题。";

let report = optimizer.analyze_system_prompt(system_prompt);

if report.is_stable {
    println!("✅ System prompt 稳定，适合缓存");
} else {
    println!("⚠️ System prompt 存在问题:");
    for issue in &report.issues {
        println!("  - {}: {}", issue.issue_type, issue.suggestion);
    }
}

println!("📊 预估缓存命中率: {:.0}%", report.estimated_cache_hit_rate * 100.0);
```

### 5.2 构建优化的 Messages

```rust
use sa_core::prompt_optimizer::PromptOptimizer;
use sa_core::openai::ChatMessage;

let optimizer = PromptOptimizer::default();

let system_prompt = "你是一个有帮助的助手。请用中文回答问题。";
let few_shot = vec![
    ChatMessage::text("user", "什么是 Rust？"),
    ChatMessage::text("assistant", "Rust 是一种系统编程语言..."),
];
let history = vec![
    ChatMessage::text("user", "你好"),
    ChatMessage::text("assistant", "你好！有什么可以帮助你的吗？"),
];
let current_input = "请解释一下所有权的概念";

let messages = optimizer.build_optimized_messages(
    system_prompt,
    &few_shot,
    &history,
    current_input,
    &[],
);

// messages 结构：
// 1. system (稳定，可缓存)
// 2. few-shot[0] (稳定，可缓存)
// 3. few-shot[1] (稳定，可缓存)
// 4. history[0] (半稳定)
// 5. history[1] (半稳定)
// 6. current_input (变化，最低优先级缓存)
```

---

## 六、缓存监控

### 6.1 创建缓存监控器

```rust
use sa_core::cache_monitor::CacheMonitor;
use sa_core::cost_budget::DeepSeekPricing;

// 使用默认配置
let mut monitor = CacheMonitor::new();

// 使用自定义定价
let pricing = DeepSeekPricing::v4_pro();
let mut monitor = CacheMonitor::with_pricing(pricing);

// 使用自定义历史大小
let mut monitor = CacheMonitor::with_max_history(500);
```

### 6.2 记录请求

```rust
// 记录请求
monitor.record_request(
    "req_123".to_string(),
    "deepseek-v4-flash".to_string(),
    800,  // cache_hit_tokens
    200,  // cache_miss_tokens
    300,  // output_tokens
);

// 从 ChatUsage 记录
monitor.record_from_usage(
    "req_456".to_string(),
    "deepseek-v4-flash".to_string(),
    &usage,
);
```

### 6.3 生成报告

```rust
let report = monitor.generate_report();

println!("📊 缓存性能报告");
println!("  总请求数: {}", report.total_requests);
println!("  总输入 tokens: {}", report.total_input_tokens);
println!("  缓存命中 tokens: {}", report.total_cache_hit_tokens);
println!("  缓存未命中 tokens: {}", report.total_cache_miss_tokens);
println!("  平均命中率: {:.1}%", report.average_hit_rate * 100.0);
println!("  总成本: ${:.4}", report.total_estimated_cost_usd);
println!("  总节省: ${:.4}", report.total_estimated_savings_usd);
println!("  命中率趋势: {:?}", report.hit_rate_trend);

println!("\n💡 优化建议:");
for rec in &report.recommendations {
    println!("  - {}", rec);
}

println!("\n⏰ 时间范围");
println!("  开始: {}", report.time_period.start);
println!("  结束: {}", report.time_period.end);
println!("  持续: {} 秒", report.time_period.duration_secs);
```

---

## 七、预期收益

### 7.1 成本节省

| 场景 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 100 万次调用，80% 缓存命中 | $140 | $28 + $2.24 = $30.24 | **78%** |
| 100 万次调用，60% 缓存命中 | $140 | $56 + $5.6 = $61.6 | **56%** |
| 100 万次调用，40% 缓存命中 | $140 | $84 + $11.2 = $95.2 | **32%** |

### 7.2 性能提升

- **首 token 延迟**：128K prompt 从 13s 降至 500ms（高缓存命中时）
- **吞吐量**：缓存命中时处理速度提升 10-20 倍

---

## 八、故障排除

### 8.1 编译错误

**问题**：找不到 `chrono` 模块
**解决**：在 `sa-core/Cargo.toml` 中添加依赖：
```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
```

**问题**：找不到 `log` 模块
**解决**：在 `sa-core/Cargo.toml` 中添加依赖：
```toml
[dependencies]
log = "0.4"
```

### 8.2 运行时问题

**问题**：缓存命中率始终为 0
**检查**：
1. 确认使用的是 DeepSeek V4 API
2. 确认 system prompt 没有动态内容
3. 确认多次请求使用相同的 system prompt

**问题**：成本计算不准确
**检查**：
1. 确认使用的 pricing 配置正确
2. 确认 API 响应中包含 usage 字段

---

## 九、下一步

1. **运行验证**：
   ```bash
   cd E:\SA\SAP-6.2\sa
   cargo check -p sa-core
   cargo test -p sa-core
   ```

2. **配置 API Key**：
   编辑 `sa.toml`，填入你的 DeepSeek API Key

3. **实际测试**：
   启动 SA 后端，进行实际 API 调用测试

4. **监控调优**：
   根据缓存命中率调整 system prompt

---

## 十、参考资源

- [DeepSeek V4 API 文档](https://api-docs.deepseek.com/)
- [DeepSeek V4 定价](https://api-docs.deepseek.com/quick_start/pricing)
- [DeepSeek 缓存机制](https://api-docs.deepseek.com/guides/kv_cache)

---

**最后更新**: 2026-05-03
**版本**: v1.0
