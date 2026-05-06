//! Cache performance monitoring and reporting for LLM API calls.
//!
//! This module provides real-time monitoring of cache hit rates,
//! cost savings, and performance metrics for LLM API calls.
//!
//! Supports multiple providers: DeepSeek, OpenAI, Anthropic, Google Gemini.
//!
//! Features:
//! - Track cache hit/miss rates per request
//! - Calculate cost savings from caching
//! - Generate performance reports
//! - Provide optimization recommendations

use crate::cost_budget::ModelPricing;
use crate::openai::ChatUsage;
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};

/// Cache performance monitor for tracking LLM API usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMonitor {
    /// History of cache hit snapshots.
    hit_rate_history: VecDeque<CacheHitSnapshot>,
    /// Maximum history size to prevent unbounded memory growth.
    max_history_size: usize,
    /// Pricing configuration for cost calculations.
    ///
    /// Transient: reconstructed from defaults on load if missing.
    #[serde(default)]
    pricing: ModelPricing,
}

/// A single cache hit snapshot for a specific request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHitSnapshot {
    /// Timestamp of the request.
    pub timestamp: DateTime<Local>,
    /// Request identifier (if available).
    pub request_id: String,
    /// Model used for the request.
    pub model: String,
    /// Total input tokens.
    pub total_input_tokens: u64,
    /// Tokens served from cache.
    pub cache_hit_tokens: u64,
    /// Tokens processed fresh.
    pub cache_miss_tokens: u64,
    /// Output tokens generated.
    pub output_tokens: u64,
    /// Cache hit rate (0.0 to 1.0).
    pub hit_rate: f64,
    /// Estimated cost for this request in USD.
    pub estimated_cost_usd: f64,
    /// Estimated savings from caching in USD.
    pub estimated_savings_usd: f64,
}

/// Per-model cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheStats {
    /// Number of requests for this model.
    pub requests: usize,
    /// Total cache hit tokens.
    pub cache_hits: u64,
    /// Total cache miss tokens.
    pub cache_misses: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Cache hit rate (0.0 to 1.0).
    pub hit_rate: f64,
}

/// Comprehensive cache performance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceReport {
    /// Total number of requests analyzed.
    pub total_requests: usize,
    /// Total input tokens across all requests.
    pub total_input_tokens: u64,
    /// Total cache hit tokens.
    pub total_cache_hit_tokens: u64,
    /// Total cache miss tokens.
    pub total_cache_miss_tokens: u64,
    /// Total output tokens.
    pub total_output_tokens: u64,
    /// Average cache hit rate (0.0 to 1.0).
    pub average_hit_rate: f64,
    /// Total estimated cost in USD.
    pub total_estimated_cost_usd: f64,
    /// Total estimated savings from caching in USD.
    pub total_estimated_savings_usd: f64,
    /// Cache hit rate trend (increasing, decreasing, or stable).
    pub hit_rate_trend: HitRateTrend,
    /// Optimization recommendations.
    pub recommendations: Vec<String>,
    /// Time period covered by the report.
    pub time_period: TimePeriod,
}

/// Cache hit rate trend indicator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HitRateTrend {
    /// Cache hit rate is improving.
    Increasing,
    /// Cache hit rate is declining.
    Decreasing,
    /// Cache hit rate is stable.
    Stable,
    /// Not enough data to determine trend.
    InsufficientData,
}

/// Time period for a performance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePeriod {
    /// Start of the period.
    pub start: DateTime<Local>,
    /// End of the period.
    pub end: DateTime<Local>,
    /// Duration in seconds.
    pub duration_secs: i64,
}

impl CacheMonitor {
    /// Create a new cache monitor with default settings.
    pub fn new() -> Self {
        Self {
            hit_rate_history: VecDeque::with_capacity(1000),
            max_history_size: 1000,
            pricing: ModelPricing::v4_flash(),
        }
    }

    /// Create a new cache monitor with custom pricing.
    pub fn with_pricing(pricing: ModelPricing) -> Self {
        Self {
            hit_rate_history: VecDeque::with_capacity(1000),
            max_history_size: 1000,
            pricing,
        }
    }

    /// Create a new cache monitor for a specific model name.
    pub fn for_model(model_name: &str) -> Self {
        Self::with_pricing(ModelPricing::for_model(model_name))
    }

    /// Update the pricing configuration (e.g. after config change).
    pub fn set_pricing(&mut self, pricing: ModelPricing) {
        self.pricing = pricing;
    }

    /// Create a new cache monitor with custom history size.
    pub fn with_max_history(max_history_size: usize) -> Self {
        Self {
            hit_rate_history: VecDeque::with_capacity(max_history_size),
            max_history_size,
            pricing: ModelPricing::v4_flash(),
        }
    }

    /// Record a request with explicit cache statistics.
    pub fn record_request(
        &mut self,
        request_id: String,
        model: String,
        cache_hit_tokens: u64,
        cache_miss_tokens: u64,
        output_tokens: u64,
    ) {
        let total_input = cache_hit_tokens + cache_miss_tokens;
        let hit_rate = if total_input > 0 {
            cache_hit_tokens as f64 / total_input as f64
        } else {
            0.0
        };

        let cost = self
            .pricing
            .calculate_cost(cache_hit_tokens, cache_miss_tokens, output_tokens);
        let savings = self
            .pricing
            .calculate_savings(cache_hit_tokens, total_input);

        let snapshot = CacheHitSnapshot {
            timestamp: Local::now(),
            request_id,
            model,
            total_input_tokens: total_input,
            cache_hit_tokens,
            cache_miss_tokens,
            output_tokens,
            hit_rate,
            estimated_cost_usd: cost,
            estimated_savings_usd: savings,
        };

        // Log the cache statistics
        tracing::info!(
            "Cache stats: {} | hit={}/{} ({:.1}%) | miss={} | cost=${:.6} | saved=${:.6}",
            snapshot.request_id,
            snapshot.cache_hit_tokens,
            snapshot.total_input_tokens,
            snapshot.hit_rate * 100.0,
            snapshot.cache_miss_tokens,
            snapshot.estimated_cost_usd,
            snapshot.estimated_savings_usd
        );

        // Add to history
        if self.hit_rate_history.len() >= self.max_history_size {
            self.hit_rate_history.pop_front();
        }
        self.hit_rate_history.push_back(snapshot);
    }

    /// Record a request using ChatUsage from the API response.
    /// 
    /// Uses `cache_read_tokens()` for cache hits and calculates misses
    /// from total input tokens minus cache hits.
    pub fn record_from_usage(
        &mut self,
        request_id: String,
        model: String,
        usage: &ChatUsage,
    ) {
        let cache_hit = usage.cache_read_tokens();
        let total_input = usage.input_tokens.unwrap_or(0);
        let cache_miss = total_input.saturating_sub(cache_hit);
        let output = usage.output_tokens.unwrap_or(0);

        self.record_request(request_id, model, cache_hit, cache_miss, output);
    }

    /// Generate a comprehensive performance report.
    pub fn generate_report(&self) -> CachePerformanceReport {
        if self.hit_rate_history.is_empty() {
            return CachePerformanceReport {
                total_requests: 0,
                total_input_tokens: 0,
                total_cache_hit_tokens: 0,
                total_cache_miss_tokens: 0,
                total_output_tokens: 0,
                average_hit_rate: 0.0,
                total_estimated_cost_usd: 0.0,
                total_estimated_savings_usd: 0.0,
                hit_rate_trend: HitRateTrend::InsufficientData,
                recommendations: vec!["No data available yet. Start making API calls to see statistics.".to_string()],
                time_period: TimePeriod {
                    start: Local::now(),
                    end: Local::now(),
                    duration_secs: 0,
                },
            };
        }

        let total_requests = self.hit_rate_history.len();
        let total_input: u64 = self.hit_rate_history.iter().map(|s| s.total_input_tokens).sum();
        let total_hit: u64 = self.hit_rate_history.iter().map(|s| s.cache_hit_tokens).sum();
        let total_miss: u64 = self.hit_rate_history.iter().map(|s| s.cache_miss_tokens).sum();
        let total_output: u64 = self.hit_rate_history.iter().map(|s| s.output_tokens).sum();

        let average_hit_rate = if total_input > 0 {
            total_hit as f64 / total_input as f64
        } else {
            0.0
        };

        let total_cost: f64 = self.hit_rate_history.iter().map(|s| s.estimated_cost_usd).sum();
        let total_savings: f64 = self
            .hit_rate_history
            .iter()
            .map(|s| s.estimated_savings_usd)
            .sum();

        let hit_rate_trend = self.calculate_hit_rate_trend();
        let recommendations = self.generate_recommendations(average_hit_rate, &hit_rate_trend);

        let start = self
            .hit_rate_history
            .front()
            .map(|s| s.timestamp)
            .unwrap_or_else(Local::now);
        let end = self
            .hit_rate_history
            .back()
            .map(|s| s.timestamp)
            .unwrap_or_else(Local::now);
        let duration_secs = (end - start).num_seconds();

        CachePerformanceReport {
            total_requests,
            total_input_tokens: total_input,
            total_cache_hit_tokens: total_hit,
            total_cache_miss_tokens: total_miss,
            total_output_tokens: total_output,
            average_hit_rate,
            total_estimated_cost_usd: total_cost,
            total_estimated_savings_usd: total_savings,
            hit_rate_trend,
            recommendations,
            time_period: TimePeriod {
                start,
                end,
                duration_secs,
            },
        }
    }

    /// Calculate the trend in cache hit rate.
    fn calculate_hit_rate_trend(&self) -> HitRateTrend {
        if self.hit_rate_history.len() < 10 {
            return HitRateTrend::InsufficientData;
        }

        // Compare recent 25% vs older 25%
        let len = self.hit_rate_history.len();
        let quarter = len / 4;

        let recent_avg: f64 = self
            .hit_rate_history
            .iter()
            .skip(len - quarter)
            .map(|s| s.hit_rate)
            .sum::<f64>()
            / quarter as f64;

        let older_avg: f64 = self
            .hit_rate_history
            .iter()
            .take(quarter)
            .map(|s| s.hit_rate)
            .sum::<f64>()
            / quarter as f64;

        let diff = recent_avg - older_avg;

        if diff > 0.05 {
            HitRateTrend::Increasing
        } else if diff < -0.05 {
            HitRateTrend::Decreasing
        } else {
            HitRateTrend::Stable
        }
    }

    /// Generate optimization recommendations based on performance.
    fn generate_recommendations(
        &self,
        average_hit_rate: f64,
        trend: &HitRateTrend,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Overall hit rate recommendations
        if average_hit_rate < 0.3 {
            recommendations.push(
                "Cache hit rate is very low (<30%). Review system prompt for dynamic content."
                    .to_string(),
            );
            recommendations.push(
                "Ensure system prompt and few-shot examples are at the beginning of messages."
                    .to_string(),
            );
            recommendations.push(
                "Remove timestamps, request IDs, and other variable content from system prompt."
                    .to_string(),
            );
        } else if average_hit_rate < 0.6 {
            recommendations.push(
                "Cache hit rate is moderate (30-60%). Consider further optimization.".to_string(),
            );
            recommendations.push(
                "Move more stable content to the beginning of messages.".to_string(),
            );
        } else if average_hit_rate < 0.8 {
            recommendations.push(
                "Cache hit rate is good (60-80%). Minor optimizations may be possible.".to_string(),
            );
        } else {
            recommendations.push(
                "Cache hit rate is excellent (>80%). Current prompt structure is well optimized."
                    .to_string(),
            );
        }

        // Trend-based recommendations
        match trend {
            HitRateTrend::Decreasing => {
                recommendations.push(
                    "Cache hit rate is declining. Check for recent changes to system prompt."
                        .to_string(),
                );
            }
            HitRateTrend::Increasing => {
                recommendations.push(
                    "Cache hit rate is improving. Current optimizations are working.".to_string(),
                );
            }
            HitRateTrend::Stable => {}
            HitRateTrend::InsufficientData => {
                recommendations.push(
                    "Not enough data to determine trend. Continue monitoring.".to_string(),
                );
            }
        }

        recommendations
    }

    /// Get the number of recorded requests.
    pub fn request_count(&self) -> usize {
        self.hit_rate_history.len()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.hit_rate_history.clear();
    }

    /// Get a summary string for logging.
    pub fn summary_string(&self) -> String {
        let report = self.generate_report();
        format!(
            "Cache Monitor: {} requests | {:.1}% hit rate | ${:.4} cost | ${:.4} saved",
            report.total_requests,
            report.average_hit_rate * 100.0,
            report.total_estimated_cost_usd,
            report.total_estimated_savings_usd
        )
    }

    /// Get per-model breakdown of cache statistics.
    pub fn get_model_breakdown(&self) -> Vec<(String, ModelCacheStats)> {
        use std::collections::HashMap;
        let mut model_stats: HashMap<String, ModelCacheStats> = HashMap::new();

        for snapshot in &self.hit_rate_history {
            let entry = model_stats.entry(snapshot.model.clone()).or_insert_with(|| ModelCacheStats {
                requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                output_tokens: 0,
                hit_rate: 0.0,
            });
            entry.requests += 1;
            entry.cache_hits += snapshot.cache_hit_tokens;
            entry.cache_misses += snapshot.cache_miss_tokens;
            entry.output_tokens += snapshot.output_tokens;
        }

        // Calculate hit rates
        for stats in model_stats.values_mut() {
            let total = stats.cache_hits + stats.cache_misses;
            stats.hit_rate = if total > 0 {
                stats.cache_hits as f64 / total as f64
            } else {
                0.0
            };
        }

        model_stats.into_iter().collect()
    }

    /// Persist cache monitor state to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, &json)
            .map_err(|e| anyhow::anyhow!("Failed to write cache monitor to {}: {e}", path.display()))?;
        tracing::info!("Cache monitor saved ({} snapshots) to {}", self.hit_rate_history.len(), path.display());
        Ok(())
    }

    /// Restore cache monitor state from a JSON file.
    /// Falls back to a fresh monitor if the file does not exist or is corrupt.
    pub fn load_from_file(path: &std::path::Path, pricing: ModelPricing) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str::<CacheMonitor>(&json) {
                Ok(mut monitor) => {
                    monitor.pricing = pricing;
                    tracing::info!("Cache monitor loaded ({} snapshots) from {}", monitor.hit_rate_history.len(), path.display());
                    monitor
                }
                Err(e) => {
                    tracing::warn!("Cache monitor corrupt at {}, starting fresh: {e}", path.display());
                    Self::with_pricing(pricing)
                }
            },
            Err(_) => {
                tracing::info!("No cache monitor file at {}, starting fresh", path.display());
                Self::with_pricing(pricing)
            }
        }
    }
}

impl Default for CacheMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_monitor_new() {
        let monitor = CacheMonitor::new();
        assert_eq!(monitor.request_count(), 0);
    }

    #[test]
    fn test_record_request() {
        let mut monitor = CacheMonitor::new();
        monitor.record_request(
            "req_1".to_string(),
            "deepseek-v4-flash".to_string(),
            800,
            200,
            300,
        );

        assert_eq!(monitor.request_count(), 1);
    }

    #[test]
    fn test_generate_report_empty() {
        let monitor = CacheMonitor::new();
        let report = monitor.generate_report();

        assert_eq!(report.total_requests, 0);
        assert_eq!(report.average_hit_rate, 0.0);
        assert_eq!(report.hit_rate_trend, HitRateTrend::InsufficientData);
    }

    #[test]
    fn test_generate_report_with_data() {
        let mut monitor = CacheMonitor::new();

        // Record several requests
        for i in 0..5 {
            monitor.record_request(
                format!("req_{}", i),
                "deepseek-v4-flash".to_string(),
                800,
                200,
                300,
            );
        }

        let report = monitor.generate_report();

        assert_eq!(report.total_requests, 5);
        assert_eq!(report.total_cache_hit_tokens, 4000); // 800 * 5
        assert_eq!(report.total_cache_miss_tokens, 1000); // 200 * 5
        assert_eq!(report.total_output_tokens, 1500); // 300 * 5
        assert!((report.average_hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_hit_rate_trend_insufficient_data() {
        let mut monitor = CacheMonitor::new();

        // Less than 10 requests
        for i in 0..5 {
            monitor.record_request(
                format!("req_{}", i),
                "deepseek-v4-flash".to_string(),
                800,
                200,
                300,
            );
        }

        let report = monitor.generate_report();
        assert_eq!(report.hit_rate_trend, HitRateTrend::InsufficientData);
    }

    #[test]
    fn test_hit_rate_trend_stable() {
        let mut monitor = CacheMonitor::new();

        // 20 requests with consistent 80% hit rate
        for i in 0..20 {
            monitor.record_request(
                format!("req_{}", i),
                "deepseek-v4-flash".to_string(),
                800,
                200,
                300,
            );
        }

        let report = monitor.generate_report();
        assert_eq!(report.hit_rate_trend, HitRateTrend::Stable);
    }

    #[test]
    fn test_recommendations_low_hit_rate() {
        let mut monitor = CacheMonitor::new();

        // 10 requests with low 20% hit rate
        for i in 0..10 {
            monitor.record_request(
                format!("req_{}", i),
                "deepseek-v4-flash".to_string(),
                200,
                800,
                300,
            );
        }

        let report = monitor.generate_report();
        assert!(report.recommendations.iter().any(|r| r.contains("very low")));
    }

    #[test]
    fn test_recommendations_high_hit_rate() {
        let mut monitor = CacheMonitor::new();

        // 10 requests with high 90% hit rate
        for i in 0..10 {
            monitor.record_request(
                format!("req_{}", i),
                "deepseek-v4-flash".to_string(),
                900,
                100,
                300,
            );
        }

        let report = monitor.generate_report();
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.contains("excellent")));
    }

    #[test]
    fn test_summary_string() {
        let mut monitor = CacheMonitor::new();
        monitor.record_request(
            "req_1".to_string(),
            "deepseek-v4-flash".to_string(),
            800,
            200,
            300,
        );

        let summary = monitor.summary_string();
        assert!(summary.contains("1 requests"));
        assert!(summary.contains("80.0% hit rate"));
    }

    #[test]
    fn test_clear() {
        let mut monitor = CacheMonitor::new();
        monitor.record_request(
            "req_1".to_string(),
            "deepseek-v4-flash".to_string(),
            800,
            200,
            300,
        );

        assert_eq!(monitor.request_count(), 1);
        monitor.clear();
        assert_eq!(monitor.request_count(), 0);
    }
}
