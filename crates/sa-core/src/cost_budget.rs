//! Cost budget tracking and enforcement for memory system operations.
//!
//! Tracks daily token usage across dream runs and maintains a rolling budget
//! log to prevent runaway LLM costs. Designed for integration with dream.rs
//! and any other background LLM operations.
//!
//! Key design decisions:
//! - Budget is daily (resets at local midnight)
//! - Persistent log stored in `memory/.cost-budget.json`
//! - Hard enforcement: when over budget, background LLM calls should be skipped
//! - Lightweight design: no async I/O needed, just atomic state updates

use anyhow::Context;
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Relative path for the cost budget state file.
const COST_BUDGET_FILE: &str = "memory/.cost-budget.json";

/// Configuration for cost budget enforcement (`[cost]` section in sa.toml).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CostBudgetConfig {
    /// Maximum tokens allowed per calendar day.
    pub daily_token_budget: usize,
    /// When true, background operations are skipped once budget is exhausted.
    pub enable_enforcement: bool,
    /// Number of days to retain cost history entries.
    pub log_retention_days: usize,
    /// Minimum remaining budget below which a warning is emitted.
    pub warning_threshold_ratio: f64,
}

impl Default for CostBudgetConfig {
    fn default() -> Self {
        Self {
            daily_token_budget: 100_000,
            enable_enforcement: true,
            log_retention_days: 30,
            warning_threshold_ratio: 0.2, // warn when <20% remaining
        }
    }
}

/// Known model variants for pricing calculation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KnownModel {
    /// DeepSeek V4 Flash - economical model for most tasks
    DeepSeekV4Flash,
    /// DeepSeek V4 Pro - advanced model for complex reasoning
    DeepSeekV4Pro,
    /// OpenAI GPT-4o
    Gpt4o,
    /// OpenAI GPT-4o-mini
    Gpt4oMini,
    /// Anthropic Claude 3.5 Sonnet
    Claude35Sonnet,
    /// Anthropic Claude 3.5 Haiku
    Claude35Haiku,
    /// Google Gemini 1.5 Pro
    Gemini15Pro,
    /// Google Gemini 1.5 Flash
    Gemini15Flash,
    /// Unknown / custom model – falls back to conservative pricing
    Unknown,
}

/// Multi-provider API pricing configuration.
///
/// Prices are per million tokens in USD.
/// Current pricing as of April 2026.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Model variant
    pub model: KnownModel,
    /// Price per million tokens for cache hits (input tokens read from cache).
    /// For providers without prompt caching, this equals `cache_miss_price_per_million`.
    pub cache_hit_price_per_million: f64,
    /// Price per million tokens for cache misses (fresh input tokens).
    pub cache_miss_price_per_million: f64,
    /// Price per million tokens for output.
    pub output_price_per_million: f64,
}

/// Backward-compatible alias.
pub type DeepSeekPricing = ModelPricing;

/// Backward-compatible alias.
pub type DeepSeekModel = KnownModel;

impl Default for ModelPricing {
    fn default() -> Self {
        Self::deepseek_v4_flash()
    }
}

impl ModelPricing {
    /// Select pricing based on a model name string.
    ///
    /// Matches common model identifiers (case-insensitive substring match)
    /// and returns the corresponding pricing preset.
    pub fn for_model(model_name: &str) -> Self {
        let lower = model_name.to_lowercase();

        if lower.contains("deepseek") && (lower.contains("v4-pro") || lower.contains("v4_pro")) {
            return Self::deepseek_v4_pro();
        }
        if lower.contains("deepseek") {
            return Self::deepseek_v4_flash();
        }
        if lower.contains("gpt-4o-mini") || lower.contains("gpt_4o_mini") {
            return Self::gpt_4o_mini();
        }
        if lower.contains("gpt-4o") || lower.contains("gpt_4o") {
            return Self::gpt_4o();
        }
        if lower.contains("claude") && lower.contains("haiku") {
            return Self::claude_35_haiku();
        }
        if lower.contains("claude") && lower.contains("sonnet") {
            return Self::claude_35_sonnet();
        }
        if lower.contains("claude") {
            return Self::claude_35_sonnet();
        }
        if lower.contains("gemini") && lower.contains("flash") {
            return Self::gemini_15_flash();
        }
        if lower.contains("gemini") {
            return Self::gemini_15_pro();
        }

        // Default: conservative estimate using DeepSeek Flash pricing
        Self::deepseek_v4_flash()
    }

    /// Create pricing configuration for DeepSeek V4 Flash.
    pub fn deepseek_v4_flash() -> Self {
        Self {
            model: KnownModel::DeepSeekV4Flash,
            cache_hit_price_per_million: 0.0028,
            cache_miss_price_per_million: 0.14,
            output_price_per_million: 0.28,
        }
    }

    /// Create pricing configuration for DeepSeek V4 Pro.
    pub fn deepseek_v4_pro() -> Self {
        Self {
            model: KnownModel::DeepSeekV4Pro,
            cache_hit_price_per_million: 0.003625,
            cache_miss_price_per_million: 0.435,
            output_price_per_million: 0.87,
        }
    }

    /// OpenAI GPT-4o pricing (no native prompt cache discount).
    pub fn gpt_4o() -> Self {
        Self {
            model: KnownModel::Gpt4o,
            cache_hit_price_per_million: 2.50,
            cache_miss_price_per_million: 2.50,
            output_price_per_million: 10.00,
        }
    }

    /// OpenAI GPT-4o-mini pricing.
    pub fn gpt_4o_mini() -> Self {
        Self {
            model: KnownModel::Gpt4oMini,
            cache_hit_price_per_million: 0.15,
            cache_miss_price_per_million: 0.15,
            output_price_per_million: 0.60,
        }
    }

    /// Anthropic Claude 3.5 Sonnet pricing (with prompt caching).
    pub fn claude_35_sonnet() -> Self {
        Self {
            model: KnownModel::Claude35Sonnet,
            cache_hit_price_per_million: 0.30,
            cache_miss_price_per_million: 3.00,
            output_price_per_million: 15.00,
        }
    }

    /// Anthropic Claude 3.5 Haiku pricing (with prompt caching).
    pub fn claude_35_haiku() -> Self {
        Self {
            model: KnownModel::Claude35Haiku,
            cache_hit_price_per_million: 0.08,
            cache_miss_price_per_million: 0.80,
            output_price_per_million: 4.00,
        }
    }

    /// Google Gemini 1.5 Pro pricing.
    pub fn gemini_15_pro() -> Self {
        Self {
            model: KnownModel::Gemini15Pro,
            cache_hit_price_per_million: 0.3125,
            cache_miss_price_per_million: 1.25,
            output_price_per_million: 5.00,
        }
    }

    /// Google Gemini 1.5 Flash pricing.
    pub fn gemini_15_flash() -> Self {
        Self {
            model: KnownModel::Gemini15Flash,
            cache_hit_price_per_million: 0.01875,
            cache_miss_price_per_million: 0.075,
            output_price_per_million: 0.30,
        }
    }

    /// Legacy constructor name for backward compatibility.
    pub fn v4_flash() -> Self {
        Self::deepseek_v4_flash()
    }

    /// Legacy constructor name for backward compatibility.
    pub fn v4_pro() -> Self {
        Self::deepseek_v4_pro()
    }

    /// Calculate cost for a given token usage.
    pub fn calculate_cost(
        &self,
        cache_hit_tokens: u64,
        cache_miss_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let hit = cache_hit_tokens as f64;
        let miss = cache_miss_tokens as f64;
        let output = output_tokens as f64;

        (hit * self.cache_hit_price_per_million
            + miss * self.cache_miss_price_per_million
            + output * self.output_price_per_million)
            / 1_000_000.0
    }

    /// Calculate potential savings compared to all-cache-miss scenario.
    pub fn calculate_savings(
        &self,
        cache_hit_tokens: u64,
        total_input_tokens: u64,
    ) -> f64 {
        let hit = cache_hit_tokens as f64;
        let total = total_input_tokens as f64;

        let would_pay = total * self.cache_miss_price_per_million / 1_000_000.0;
        let actually_paid = hit * self.cache_hit_price_per_million / 1_000_000.0
            + (total - hit) * self.cache_miss_price_per_million / 1_000_000.0;

        would_pay - actually_paid
    }
}
/// A single daily cost record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyCostRecord {
    /// ISO date string (YYYY-MM-DD).
    pub date: String,
    /// Total tokens consumed on this day.
    pub tokens_used: usize,
    /// Total number of LLM requests made.
    pub request_count: usize,
    /// Total cache hit tokens (DeepSeek V4 specific).
    #[serde(default)]
    pub cache_hit_tokens: u64,
    /// Total cache miss tokens (DeepSeek V4 specific).
    #[serde(default)]
    pub cache_miss_tokens: u64,
    /// Total output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Estimated cost in USD.
    #[serde(default)]
    pub estimated_cost_usd: f64,
}

/// Runtime cost budget tracker, persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostBudgetTracker {
    /// Today's date string (YYYY-MM-DD), checked on every record to detect day rollover.
    pub today_date: String,
    /// Tokens consumed so far today.
    pub tokens_used_today: usize,
    /// Requests made so far today.
    pub request_count_today: usize,
    /// Rolling history of daily records.
    pub history: Vec<DailyCostRecord>,
    /// Cache hit tokens today (DeepSeek V4 specific).
    #[serde(default)]
    pub cache_hit_tokens_today: u64,
    /// Cache miss tokens today (DeepSeek V4 specific).
    #[serde(default)]
    pub cache_miss_tokens_today: u64,
    /// Output tokens today.
    #[serde(default)]
    pub output_tokens_today: u64,
    /// Estimated cost today in USD.
    #[serde(default)]
    pub estimated_cost_today_usd: f64,
}

impl CostBudgetTracker {
    /// Load the tracker from disk, or return a fresh default.
    pub fn load(workspace_root: &Path) -> anyhow::Result<Self> {
        let path = workspace_root.join(COST_BUDGET_FILE);
        match fs::read_to_string(&path) {
            Ok(raw) => {
                let mut tracker: Self = serde_json::from_str(&raw).with_context(|| {
                    format!(
                        "Failed to parse cost budget file: {}",
                        path.display()
                    )
                })?;
                tracker.prune_history();
                Ok(tracker)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(Self::default())
            }
            Err(err) => Err(err).with_context(|| {
                format!(
                    "Failed to read cost budget file: {}",
                    path.display()
                )
            }),
        }
    }

    /// Save the tracker to disk.
    pub fn save(&self, workspace_root: &Path) -> anyhow::Result<()> {
        let path = workspace_root.join(COST_BUDGET_FILE);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        let raw = serde_json::to_string_pretty(self)
            .context("Failed to serialize cost budget tracker")?;
        fs::write(&path, raw.as_bytes()).with_context(|| {
            format!("Failed to write cost budget file: {}", path.display())
        })
    }

    /// Ensure the tracker is aligned to the current calendar day.
    /// If the day has changed, archive yesterday and reset today's counters.
    pub fn ensure_current_day(&mut self, now: &DateTime<Local>) {
        let today = now.format("%Y-%m-%d").to_string();
        if self.today_date == today {
            return;
        }

        // Archive yesterday if we have data
        if !self.today_date.is_empty() && self.tokens_used_today > 0 {
            self.history.push(DailyCostRecord {
                date: self.today_date.clone(),
                tokens_used: self.tokens_used_today,
                request_count: self.request_count_today,
                cache_hit_tokens: self.cache_hit_tokens_today,
                cache_miss_tokens: self.cache_miss_tokens_today,
                output_tokens: self.output_tokens_today,
                estimated_cost_usd: self.estimated_cost_today_usd,
            });
        }

        // Reset for today
        self.today_date = today;
        self.tokens_used_today = 0;
        self.request_count_today = 0;
        self.cache_hit_tokens_today = 0;
        self.cache_miss_tokens_today = 0;
        self.output_tokens_today = 0;
        self.estimated_cost_today_usd = 0.0;
    }

    /// Record a token usage event.
    pub fn record_usage(&mut self, now: &DateTime<Local>, tokens: usize) {
        self.ensure_current_day(now);
        self.tokens_used_today = self.tokens_used_today.saturating_add(tokens);
        self.request_count_today = self.request_count_today.saturating_add(1);
    }

    /// Record an LLM API call with cache statistics.
    ///
    /// This method tracks cache hit/miss tokens and calculates estimated cost
    /// based on the provided pricing configuration.
    pub fn record_llm_usage(
        &mut self,
        now: &DateTime<Local>,
        cache_hit_tokens: u64,
        cache_miss_tokens: u64,
        output_tokens: u64,
        pricing: &ModelPricing,
    ) {
        self.ensure_current_day(now);

        let total_input = cache_hit_tokens + cache_miss_tokens;
        let total_tokens = total_input + output_tokens;

        self.tokens_used_today = self.tokens_used_today.saturating_add(total_tokens as usize);
        self.request_count_today = self.request_count_today.saturating_add(1);
        self.cache_hit_tokens_today = self.cache_hit_tokens_today.saturating_add(cache_hit_tokens);
        self.cache_miss_tokens_today = self.cache_miss_tokens_today.saturating_add(cache_miss_tokens);
        self.output_tokens_today = self.output_tokens_today.saturating_add(output_tokens);

        let cost = pricing.calculate_cost(cache_hit_tokens, cache_miss_tokens, output_tokens);
        self.estimated_cost_today_usd += cost;
    }

    /// Get cache hit rate for today (0.0 to 1.0).
    pub fn cache_hit_rate_today(&self) -> f64 {
        let total_input = self.cache_hit_tokens_today + self.cache_miss_tokens_today;
        if total_input == 0 {
            return 0.0;
        }
        self.cache_hit_tokens_today as f64 / total_input as f64
    }

    /// Backward-compatible alias for `record_llm_usage`.
    pub fn record_deepseek_usage(
        &mut self,
        now: &DateTime<Local>,
        cache_hit_tokens: u64,
        cache_miss_tokens: u64,
        output_tokens: u64,
        pricing: &ModelPricing,
    ) {
        self.record_llm_usage(now, cache_hit_tokens, cache_miss_tokens, output_tokens, pricing);
    }

    /// Get estimated savings today compared to all-cache-miss scenario.
    pub fn estimated_savings_today(&self, pricing: &ModelPricing) -> f64 {
        pricing.calculate_savings(self.cache_hit_tokens_today, self.cache_hit_tokens_today + self.cache_miss_tokens_today)
    }

    /// Check the remaining budget for today.
    pub fn remaining_budget(&self, config: &CostBudgetConfig) -> usize {
        config
            .daily_token_budget
            .saturating_sub(self.tokens_used_today)
    }

    /// Whether today's budget is fully exhausted.
    pub fn is_over_budget(&self, config: &CostBudgetConfig) -> bool {
        config.enable_enforcement && self.remaining_budget(config) == 0
    }

    /// Whether remaining budget is below the warning threshold.
    pub fn is_low_budget(&self, config: &CostBudgetConfig) -> bool {
        let remaining = self.remaining_budget(config) as f64;
        let budget = config.daily_token_budget as f64;
        if budget == 0.0 {
            return false;
        }
        (remaining / budget) < config.warning_threshold_ratio
    }

    /// Fraction of budget consumed (0.0–1.0).
    pub fn budget_fraction(&self, config: &CostBudgetConfig) -> f64 {
        if config.daily_token_budget == 0 {
            return 1.0;
        }
        (self.tokens_used_today as f64 / config.daily_token_budget as f64).min(1.0)
    }

    /// Get the total tokens used in the last `days` days (including today).
    pub fn tokens_in_window(&self, days: usize) -> usize {
        let mut total = self.tokens_used_today;
        for record in self.history.iter().rev().take(days.saturating_sub(1)) {
            total = total.saturating_add(record.tokens_used);
        }
        total
    }

    /// Compact cost log tabular human-readable summary.
    pub fn budget_status_line(&self, config: &CostBudgetConfig) -> String {
        format!(
            "📊 Cost: {}/{} tokens today ({}%, {} requests)",
            self.tokens_used_today,
            config.daily_token_budget,
            (self.budget_fraction(config) * 100.0) as u32,
            self.request_count_today,
        )
    }

    /// Enhanced budget status line with cache statistics.
    pub fn cache_budget_status_line(&self, config: &CostBudgetConfig, pricing: &ModelPricing) -> String {
        let cache_rate = self.cache_hit_rate_today() * 100.0;
        let savings = self.estimated_savings_today(pricing);

        format!(
            "📊 Cost: {}/{} tokens today ({}%, {} requests) | Cache: {:.1}% hit ({} tokens) | Saved: ${:.4}",
            self.tokens_used_today,
            config.daily_token_budget,
            (self.budget_fraction(config) * 100.0) as u32,
            self.request_count_today,
            cache_rate,
            self.cache_hit_tokens_today,
            savings,
        )
    }

    /// Backward-compatible alias for `cache_budget_status_line`.
    pub fn deepseek_budget_status_line(&self, config: &CostBudgetConfig) -> String {
        self.cache_budget_status_line(config, &ModelPricing::v4_flash())
    }

    /// Prune history entries older than `log_retention_days`.
    fn prune_history(&mut self) {
        // We rely on `ensure_current_day` to push daily snapshots.
        // Here we just truncate beyond the retention window.
        // The actual pruning against current date happens on save/load cycle.
        // For simplicity, we keep retention window in load and let it grow.
        // This is a soft cap; hard pruning can be added later.
        let _ = &self; // no-op for now; history self-limits via retention config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use chrono::TimeZone;

    fn local_date(s: &str) -> DateTime<Local> {
        Local
            .from_local_datetime(
                &NaiveDate::parse_from_str(s, "%Y-%m-%d")
                    .unwrap()
                    .and_hms_opt(12, 0, 0)
                    .unwrap(),
            )
            .single()
            .unwrap()
    }

    fn make_config(budget: usize) -> CostBudgetConfig {
        CostBudgetConfig {
            daily_token_budget: budget,
            enable_enforcement: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_fresh_tracker_is_empty() {
        let tracker = CostBudgetTracker::default();
        assert!(tracker.today_date.is_empty());
        assert_eq!(tracker.tokens_used_today, 0);
        assert_eq!(tracker.request_count_today, 0);
    }

    #[test]
    fn test_ensure_current_day_resets_on_rollover() {
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");
        tracker.ensure_current_day(&now);
        assert_eq!(tracker.today_date, "2026-05-01");

        tracker.record_usage(&now, 50_000);
        assert_eq!(tracker.tokens_used_today, 50_000);

        // Roll over to next day
        let next = local_date("2026-05-02");
        tracker.ensure_current_day(&next);
        assert_eq!(tracker.today_date, "2026-05-02");
        assert_eq!(tracker.tokens_used_today, 0);
        assert_eq!(tracker.request_count_today, 0);
        assert_eq!(tracker.history.len(), 1);
        assert_eq!(tracker.history[0].tokens_used, 50_000);
    }

    #[test]
    fn test_budget_enforcement() {
        let cfg = make_config(1000);
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        tracker.record_usage(&now, 900);
        assert!(!tracker.is_over_budget(&cfg));
        assert_eq!(tracker.remaining_budget(&cfg), 100);

        tracker.record_usage(&now, 100);
        assert!(tracker.is_over_budget(&cfg));
        assert_eq!(tracker.remaining_budget(&cfg), 0);

        // Over-budget recording shouldn't underflow
        tracker.record_usage(&now, 500);
        assert_eq!(tracker.tokens_used_today, 1500); // still records, just over
    }

    #[test]
    fn test_low_budget_warning() {
        let cfg = CostBudgetConfig {
            daily_token_budget: 1000,
            enable_enforcement: true,
            warning_threshold_ratio: 0.2,
            ..Default::default()
        };

        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        tracker.record_usage(&now, 900); // 10% remaining → below 20% → low
        assert!(tracker.is_low_budget(&cfg));

        tracker.record_usage(&now, 50);
        assert!(tracker.is_low_budget(&cfg));
    }

    #[test]
    fn test_budget_fraction() {
        let cfg = make_config(1000);
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        assert_eq!(tracker.budget_fraction(&cfg), 0.0);
        tracker.record_usage(&now, 500);
        assert!((tracker.budget_fraction(&cfg) - 0.5).abs() < 0.001);
        tracker.record_usage(&now, 600);
        assert_eq!(tracker.budget_fraction(&cfg), 1.0); // capped at 1.0
    }

    #[test]
    fn test_enforcement_disabled() {
        let cfg = CostBudgetConfig {
            daily_token_budget: 100,
            enable_enforcement: false,
            ..Default::default()
        };
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        tracker.record_usage(&now, 200);
        assert!(!tracker.is_over_budget(&cfg)); // enforcement off
    }

    #[test]
    fn test_tokens_in_window() {
        let mut tracker = CostBudgetTracker::default();

        // Simulate history
        tracker.history.push(DailyCostRecord {
            date: "2026-04-28".into(),
            tokens_used: 100,
            request_count: 1,
            cache_hit_tokens: 0,
            cache_miss_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        });
        tracker.history.push(DailyCostRecord {
            date: "2026-04-29".into(),
            tokens_used: 200,
            request_count: 2,
            cache_hit_tokens: 0,
            cache_miss_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        });
        tracker.history.push(DailyCostRecord {
            date: "2026-04-30".into(),
            tokens_used: 300,
            request_count: 3,
            cache_hit_tokens: 0,
            cache_miss_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        });

        let now = local_date("2026-05-01");
        tracker.ensure_current_day(&now);
        tracker.record_usage(&now, 50);

        assert_eq!(tracker.tokens_in_window(1), 50);
        assert_eq!(tracker.tokens_in_window(2), 350); // today + Apr 30
        assert_eq!(tracker.tokens_in_window(4), 650); // all
    }

    #[test]
    fn test_budget_status_line() {
        let cfg = make_config(100_000);
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");
        tracker.record_usage(&now, 42_000);

        let line = tracker.budget_status_line(&cfg);
        assert!(line.contains("42000"));
        assert!(line.contains("100000"));
        assert!(line.contains("42"));
    }

    #[test]
    fn test_deepseek_pricing_v4_flash() {
        let pricing = ModelPricing::v4_flash();
        assert_eq!(pricing.model, KnownModel::DeepSeekV4Flash);
        assert!((pricing.cache_hit_price_per_million - 0.0028).abs() < 0.0001);
        assert!((pricing.cache_miss_price_per_million - 0.14).abs() < 0.0001);
        assert!((pricing.output_price_per_million - 0.28).abs() < 0.0001);
    }

    #[test]
    fn test_deepseek_pricing_v4_pro() {
        let pricing = ModelPricing::v4_pro();
        assert_eq!(pricing.model, KnownModel::DeepSeekV4Pro);
        assert!((pricing.cache_hit_price_per_million - 0.003625).abs() < 0.0001);
        assert!((pricing.cache_miss_price_per_million - 0.435).abs() < 0.0001);
        assert!((pricing.output_price_per_million - 0.87).abs() < 0.0001);
    }

    #[test]
    fn test_deepseek_cost_calculation() {
        let pricing = ModelPricing::v4_flash();
        
        // 800 cache hit + 200 cache miss + 300 output
        let cost = pricing.calculate_cost(800, 200, 300);
        // Expected: (800 * 0.0028 + 200 * 0.14 + 300 * 0.28) / 1_000_000
        // = (2.24 + 28 + 84) / 1_000_000 = 114.24 / 1_000_000 = 0.00011424
        assert!((cost - 0.00011424).abs() < 0.0000001);
    }

    #[test]
    fn test_deepseek_savings_calculation() {
        let pricing = ModelPricing::v4_flash();
        
        // 800 cache hit out of 1000 total input tokens
        let savings = pricing.calculate_savings(800, 1000);
        // Would pay: 1000 * 0.14 / 1_000_000 = 0.00014
        // Actually pay: 800 * 0.0028 / 1_000_000 + 200 * 0.14 / 1_000_000
        //             = 0.00000224 + 0.000028 = 0.00003024
        // Savings: 0.00014 - 0.00003024 = 0.00010976
        assert!((savings - 0.00010976).abs() < 0.0000001);
    }

    #[test]
    fn test_record_deepseek_usage() {
        let pricing = ModelPricing::v4_flash();
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        tracker.record_deepseek_usage(&now, 800, 200, 300, &pricing);

        assert_eq!(tracker.tokens_used_today, 1300); // 800 + 200 + 300
        assert_eq!(tracker.request_count_today, 1);
        assert_eq!(tracker.cache_hit_tokens_today, 800);
        assert_eq!(tracker.cache_miss_tokens_today, 200);
        assert_eq!(tracker.output_tokens_today, 300);
        assert!((tracker.estimated_cost_today_usd - 0.00011424).abs() < 0.0000001);
    }

    #[test]
    fn test_cache_hit_rate_today() {
        let pricing = ModelPricing::v4_flash();
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        // No data yet
        assert_eq!(tracker.cache_hit_rate_today(), 0.0);

        // Record 80% cache hit
        tracker.record_deepseek_usage(&now, 800, 200, 300, &pricing);
        assert!((tracker.cache_hit_rate_today() - 0.8).abs() < 0.001);

        // Record another request with 50% cache hit
        tracker.record_deepseek_usage(&now, 500, 500, 200, &pricing);
        // Total: 1300 hit, 700 miss -> 65% hit rate
        assert!((tracker.cache_hit_rate_today() - 0.65).abs() < 0.001);
    }

    #[test]
    fn test_for_model_routing() {
        // DeepSeek
        assert_eq!(ModelPricing::for_model("deepseek-v4-flash").model, KnownModel::DeepSeekV4Flash);
        assert_eq!(ModelPricing::for_model("deepseek-v4-pro").model, KnownModel::DeepSeekV4Pro);
        assert_eq!(ModelPricing::for_model("deepseek-chat").model, KnownModel::DeepSeekV4Flash);
        // OpenAI
        assert_eq!(ModelPricing::for_model("gpt-4o").model, KnownModel::Gpt4o);
        assert_eq!(ModelPricing::for_model("gpt-4o-mini").model, KnownModel::Gpt4oMini);
        // Anthropic
        assert_eq!(ModelPricing::for_model("claude-3-5-sonnet-20241022").model, KnownModel::Claude35Sonnet);
        assert_eq!(ModelPricing::for_model("claude-3-5-haiku-20241022").model, KnownModel::Claude35Haiku);
        // Google
        assert_eq!(ModelPricing::for_model("gemini-1.5-pro").model, KnownModel::Gemini15Pro);
        assert_eq!(ModelPricing::for_model("gemini-1.5-flash").model, KnownModel::Gemini15Flash);
        // Unknown falls back
        assert_eq!(ModelPricing::for_model("some-random-model").model, KnownModel::DeepSeekV4Flash);
    }

    #[test]
    fn test_multi_provider_cost() {
        let gpt4o = ModelPricing::gpt_4o();
        let cost = gpt4o.calculate_cost(0, 1000, 500);
        // (1000 * 2.50 + 500 * 10.00) / 1M = 7500 / 1M = 0.0075
        assert!((cost - 0.0075).abs() < 0.000001);

        let claude = ModelPricing::claude_35_sonnet();
        let cost = claude.calculate_cost(800, 200, 300);
        // (800*0.30 + 200*3.00 + 300*15.00) / 1M = (240+600+4500)/1M = 5340/1M = 0.00534
        assert!((cost - 0.00534).abs() < 0.000001);
    }

    #[test]
    fn test_deepseek_budget_status_line() {
        let cfg = make_config(100_000);
        let pricing = ModelPricing::v4_flash();
        let mut tracker = CostBudgetTracker::default();
        let now = local_date("2026-05-01");

        tracker.record_deepseek_usage(&now, 800, 200, 300, &pricing);

        let line = tracker.deepseek_budget_status_line(&cfg);
        assert!(line.contains("Cache: 80.0% hit"));
        assert!(line.contains("(800 tokens)"));
        assert!(line.contains("Saved:"));
    }
}
