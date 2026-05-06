//! Prompt structure optimizer for DeepSeek V4 cache optimization.
//!
//! This module provides tools to analyze and optimize prompt structure
//! for maximum cache hit rates with DeepSeek V4's context caching system.
//!
//! Key principles:
//! - Stable content (system prompt, few-shot examples) should be at the beginning
//! - Variable content (user input) should be at the end
//! - Avoid timestamps, request IDs, or random elements in system prompts
//! - Keep system prompts consistent across requests

use crate::openai::{ChatMessage, MessageContent, ToolDefinition};
use serde::{Deserialize, Serialize};

/// Prompt stability analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptStabilityReport {
    /// Whether the prompt is considered stable for caching.
    pub is_stable: bool,
    /// List of issues found that may reduce cache hit rate.
    pub issues: Vec<StabilityIssue>,
    /// Estimated cache hit rate (0.0 to 1.0).
    pub estimated_cache_hit_rate: f64,
    /// Recommendations for improving cache performance.
    pub recommendations: Vec<String>,
}

/// A specific stability issue found in the prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityIssue {
    /// Type of issue detected.
    pub issue_type: IssueType,
    /// Description of the issue.
    pub description: String,
    /// Severity level (1-5, where 5 is most critical).
    pub severity: u8,
    /// Suggested fix.
    pub suggestion: String,
}

/// Types of stability issues that can affect caching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueType {
    /// Contains timestamp or date references.
    Timestamp,
    /// Contains request ID or UUID.
    RequestId,
    /// Contains random or dynamic content.
    DynamicContent,
    /// System prompt is too short to be useful for caching.
    TooShort,
    /// System prompt changes between requests.
    Inconsistent,
    /// Contains user-specific data that varies.
    UserData,
}

/// Prompt structure optimizer for DeepSeek V4 cache optimization.
pub struct PromptOptimizer {
    /// Minimum system prompt length to be considered cacheable.
    min_system_prompt_length: usize,
    /// Patterns that indicate dynamic content.
    dynamic_patterns: Vec<String>,
}

impl Default for PromptOptimizer {
    fn default() -> Self {
        Self {
            min_system_prompt_length: 100,
            dynamic_patterns: vec![
                "{{timestamp}}".to_string(),
                "{{date}}".to_string(),
                "{{time}}".to_string(),
                "{{request_id}}".to_string(),
                "{{uuid}}".to_string(),
                "{{session_id}}".to_string(),
                "当前时间".to_string(),
                "Current time".to_string(),
                "Current date".to_string(),
                "Now is".to_string(),
            ],
        }
    }
}

impl PromptOptimizer {
    /// Create a new prompt optimizer with custom settings.
    pub fn new(min_system_prompt_length: usize, dynamic_patterns: Vec<String>) -> Self {
        Self {
            min_system_prompt_length,
            dynamic_patterns,
        }
    }

    /// Analyze a system prompt for stability issues.
    ///
    /// Returns a detailed report with issues and recommendations.
    pub fn analyze_system_prompt(&self, prompt: &str) -> PromptStabilityReport {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Check for timestamp patterns
        if self.contains_timestamp_pattern(prompt) {
            issues.push(StabilityIssue {
                issue_type: IssueType::Timestamp,
                description: "System prompt contains timestamp or date references".to_string(),
                severity: 5,
                suggestion: "Remove timestamp references from system prompt. Use a separate \
                           field for time-sensitive information.".to_string(),
            });
            recommendations.push(
                "Move timestamp references to user messages or a separate context field."
                    .to_string(),
            );
        }

        // Check for request ID patterns
        if self.contains_request_id_pattern(prompt) {
            issues.push(StabilityIssue {
                issue_type: IssueType::RequestId,
                description: "System prompt contains request ID or UUID".to_string(),
                severity: 5,
                suggestion: "Remove request IDs from system prompt. These should be generated \
                           per-request, not embedded in cached content.".to_string(),
            });
            recommendations.push(
                "Generate request IDs at runtime, not in the system prompt.".to_string(),
            );
        }

        // Check for dynamic content patterns
        let dynamic_matches = self.find_dynamic_patterns(prompt);
        if !dynamic_matches.is_empty() {
            issues.push(StabilityIssue {
                issue_type: IssueType::DynamicContent,
                description: format!(
                    "System prompt contains dynamic patterns: {}",
                    dynamic_matches.join(", ")
                ),
                severity: 4,
                suggestion: "Replace dynamic patterns with static content or move them to \
                           user messages.".to_string(),
            });
            recommendations.push(
                "Use static placeholders in system prompt and inject dynamic values in user messages."
                    .to_string(),
            );
        }

        // Check prompt length
        if prompt.len() < self.min_system_prompt_length {
            issues.push(StabilityIssue {
                issue_type: IssueType::TooShort,
                description: format!(
                    "System prompt is too short ({} chars, minimum {})",
                    prompt.len(),
                    self.min_system_prompt_length
                ),
                severity: 2,
                suggestion: "Consider adding more context to system prompt for better cache utilization."
                    .to_string(),
            });
            recommendations.push(
                "Longer system prompts provide more cacheable content and better cache hit rates."
                    .to_string(),
            );
        }

        // Calculate estimated cache hit rate
        let estimated_cache_hit_rate = self.estimate_cache_hit_rate(&issues);

        // Generate general recommendations if prompt is stable
        if issues.is_empty() {
            recommendations.push(
                "System prompt looks stable. Ensure it remains consistent across requests."
                    .to_string(),
            );
            recommendations.push(
                "Place few-shot examples after system prompt for additional cacheable content."
                    .to_string(),
            );
        }

        PromptStabilityReport {
            is_stable: issues.iter().all(|i| i.severity < 4),
            issues,
            estimated_cache_hit_rate,
            recommendations,
        }
    }

    /// Build an optimized message list for maximum cache hit rate.
    ///
    /// Reorders messages to ensure stable content comes first.
    pub fn build_optimized_messages(
        &self,
        system_prompt: &str,
        few_shot_examples: &[ChatMessage],
        conversation_history: &[ChatMessage],
        current_user_input: &str,
        tool_definitions: &[ToolDefinition],
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 1. System prompt (highest priority for caching)
        if !system_prompt.is_empty() {
            messages.push(ChatMessage::text("system", system_prompt));
        }

        // 2. Tool definitions (stable, cacheable)
        // Note: Tool definitions are typically sent in the request body, not as messages.
        // This is included for completeness but may not be applicable to all APIs.

        // 3. Few-shot examples (stable, cacheable)
        messages.extend(few_shot_examples.iter().cloned());

        // 4. Conversation history (semi-stable)
        // Keep recent history to avoid token waste while maintaining context
        let max_history_turns = 10;
        let history_to_keep = if conversation_history.len() > max_history_turns * 2 {
            &conversation_history[conversation_history.len() - max_history_turns * 2..]
        } else {
            conversation_history
        };
        messages.extend(history_to_keep.iter().cloned());

        // 5. Current user input (variable, least priority for caching)
        messages.push(ChatMessage::text("user", current_user_input));

        messages
    }

    /// Check if a prompt contains timestamp patterns.
    fn contains_timestamp_pattern(&self, prompt: &str) -> bool {
        let lower = prompt.to_lowercase();
        lower.contains("{{timestamp}}")
            || lower.contains("{{date}}")
            || lower.contains("{{time}}")
            || lower.contains("当前时间")
            || lower.contains("current time")
            || lower.contains("current date")
            || lower.contains("now is")
    }

    /// Check if a prompt contains request ID patterns.
    fn contains_request_id_pattern(&self, prompt: &str) -> bool {
        let lower = prompt.to_lowercase();
        lower.contains("{{request_id}}")
            || lower.contains("{{uuid}}")
            || lower.contains("{{session_id}}")
            || lower.contains("request id:")
            || lower.contains("request id：")
    }

    /// Find all dynamic patterns in the prompt.
    fn find_dynamic_patterns(&self, prompt: &str) -> Vec<String> {
        let lower = prompt.to_lowercase();
        self.dynamic_patterns
            .iter()
            .filter(|pattern| lower.contains(&pattern.to_lowercase()))
            .cloned()
            .collect()
    }

    /// Estimate cache hit rate based on detected issues.
    fn estimate_cache_hit_rate(&self, issues: &[StabilityIssue]) -> f64 {
        if issues.is_empty() {
            return 0.9; // High confidence for stable prompts
        }

        let max_severity = issues.iter().map(|i| i.severity).max().unwrap_or(0);

        match max_severity {
            0..=1 => 0.8,
            2..=3 => 0.6,
            4 => 0.3,
            5 => 0.1,
            _ => 0.0,
        }
    }
}

/// Generate a prompt template optimized for DeepSeek V4 caching.
///
/// This template places stable content at the beginning and variable
/// content at the end for maximum cache hit rates.
pub fn generate_optimized_template(
    system_instructions: &str,
    few_shot_examples: &[(String, String)],
) -> String {
    let mut template = String::new();

    // System instructions (stable, cacheable)
    template.push_str("# System Instructions\n");
    template.push_str(system_instructions);
    template.push_str("\n\n");

    // Few-shot examples (stable, cacheable)
    if !few_shot_examples.is_empty() {
        template.push_str("# Examples\n");
        for (i, (input, output)) in few_shot_examples.iter().enumerate() {
            template.push_str(&format!("## Example {}\n", i + 1));
            template.push_str(&format!("User: {}\n", input));
            template.push_str(&format!("Assistant: {}\n\n", output));
        }
    }

    // Variable content placeholder
    template.push_str("# Current Conversation\n");
    template.push_str("{{conversation_history}}\n\n");
    template.push_str("User: {{current_input}}\n");
    template.push_str("Assistant: ");

    template
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_stable_prompt() {
        let optimizer = PromptOptimizer::default();
        let prompt = "You are a helpful assistant. You answer questions clearly and concisely. \
                     Always be polite and professional.";

        let report = optimizer.analyze_system_prompt(prompt);

        assert!(report.is_stable);
        assert!(report.issues.is_empty());
        assert!(report.estimated_cache_hit_rate > 0.8);
    }

    #[test]
    fn test_analyze_prompt_with_timestamp() {
        let optimizer = PromptOptimizer::default();
        let prompt = "You are a helpful assistant. Current time: {{timestamp}}. Answer questions.";

        let report = optimizer.analyze_system_prompt(prompt);

        assert!(!report.is_stable);
        assert!(report.issues.iter().any(|i| i.issue_type == IssueType::Timestamp));
        assert!(report.estimated_cache_hit_rate < 0.5);
    }

    #[test]
    fn test_analyze_prompt_with_request_id() {
        let optimizer = PromptOptimizer::default();
        let prompt = "You are a helpful assistant. Request ID: {{uuid}}. Process this request.";

        let report = optimizer.analyze_system_prompt(prompt);

        assert!(!report.is_stable);
        assert!(report.issues.iter().any(|i| i.issue_type == IssueType::RequestId));
    }

    #[test]
    fn test_analyze_short_prompt() {
        let optimizer = PromptOptimizer::default();
        let prompt = "Answer questions.";

        let report = optimizer.analyze_system_prompt(prompt);

        assert!(report.issues.iter().any(|i| i.issue_type == IssueType::TooShort));
    }

    #[test]
    fn test_build_optimized_messages() {
        let optimizer = PromptOptimizer::default();
        let system_prompt = "You are a helpful assistant.";
        let few_shot = vec![
            ChatMessage::text("user", "What is 2+2?"),
            ChatMessage::text("assistant", "4"),
        ];
        let history = vec![
            ChatMessage::text("user", "Hello"),
            ChatMessage::text("assistant", "Hi there!"),
        ];
        let current_input = "How are you?";

        let messages =
            optimizer.build_optimized_messages(system_prompt, &few_shot, &history, current_input, &[]);

        // Should be: system, few-shot[0], few-shot[1], history[0], history[1], current
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[5].role, "user");
    }

    #[test]
    fn test_generate_optimized_template() {
        let template = generate_optimized_template(
            "You are a helpful assistant.",
            &[
                ("What is 2+2?".to_string(), "4".to_string()),
                ("What is 3+3?".to_string(), "6".to_string()),
            ],
        );

        assert!(template.contains("# System Instructions"));
        assert!(template.contains("# Examples"));
        assert!(template.contains("# Current Conversation"));
        assert!(template.contains("{{conversation_history}}"));
        assert!(template.contains("{{current_input}}"));
    }
}
