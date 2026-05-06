//! Utility functions for the SA backend.

use sa_core::config::Config;
use sa_core::openai::{AuthStyle, WireApi};
use sa_core::ws_protocol::{InitializeConfigRequest, QuestionMode, UserQuestion, UserQuestionAnswer};
use std::collections::HashSet;

/// Normalize optional free-form config values.
pub(crate) fn normalize_optional_string(value: Option<String>) -> Option<String> {
    let value = value?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

/// Decide the effective wire protocol for a first-run initialization request.
pub(crate) fn effective_init_wire_api(request: &InitializeConfigRequest) -> WireApi {
    use sa_core::ws_protocol::InitMethod;
    request.wire_api.unwrap_or(match request.method {
        InitMethod::OpenAiCompatible => WireApi::ChatCompletions,
        InitMethod::AnthropicCompatible => WireApi::AnthropicMessages,
        InitMethod::Custom => WireApi::Responses,
    })
}

/// Decide the effective authentication style for a first-run initialization request.
pub(crate) fn effective_init_auth_style(request: &InitializeConfigRequest, wire_api: WireApi) -> AuthStyle {
    use sa_core::ws_protocol::InitMethod;
    request.auth_style.unwrap_or(match request.method {
        InitMethod::OpenAiCompatible => AuthStyle::Bearer,
        InitMethod::AnthropicCompatible => AuthStyle::AnthropicAuto,
        InitMethod::Custom => match wire_api {
            WireApi::AnthropicMessages => AuthStyle::AnthropicAuto,
            WireApi::ChatCompletions | WireApi::Responses => AuthStyle::Bearer,
        },
    })
}

/// Serialize one plain string as a TOML double-quoted string literal.
pub(crate) fn toml_string(value: &str) -> String {
    format!("{value:?}")
}

/// Mask API key in config before sending to frontend.
///
/// Replaces the middle portion of the key with `****` while preserving
/// the first 4 and last 4 characters for identification purposes.
pub(crate) fn mask_config_api_key(config: &mut Config) {
    if !config.llm.api_key.is_empty() {
        config.llm.api_key = mask_secret(&config.llm.api_key);
    }
}

/// Mask a secret string, showing only first 4 and last 4 characters.
pub(crate) fn mask_secret(secret: &str) -> String {
    if secret.len() <= 8 {
        "****".to_string()
    } else {
        format!("{}****{}", &secret[..4], &secret[secret.len() - 4..])
    }
}

/// Expand special placeholders in referenced paths.
///
/// Currently supported:
/// - `YYYY-MM-DD` — replaced with today's date, and also yesterday's date
///   (to match common "load today + yesterday" instructions).
pub(crate) fn expand_date_placeholders(mut refs: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();

    let today = chrono::Local::now().date_naive();
    let yesterday = today - chrono::Duration::days(1);

    for r in refs.drain(..) {
        if r.contains("YYYY-MM-DD") {
            out.push(r.replace("YYYY-MM-DD", &today.to_string()));
            out.push(r.replace("YYYY-MM-DD", &yesterday.to_string()));
        } else {
            out.push(r);
        }
    }

    // De-duplicate while preserving order.
    let mut seen = HashSet::<String>::new();
    out.into_iter().filter(|r| seen.insert(r.clone())).collect()
}

/// Decorate nested agent text so user-visible messages stay traceable.
#[allow(dead_code)]
pub(crate) fn decorate_nested_text(depth: u32, text: &str) -> String {
    if depth == 0 {
        return text.to_string();
    }

    format!("[subagent depth={depth}] {text}")
}

/// Decorate nested agent prompts for `Ask`.
#[allow(dead_code)]
pub(crate) fn decorate_nested_prompt(depth: u32, prompt: &str) -> String {
    if depth == 0 {
        return prompt.to_string();
    }

    format!("[subagent depth={depth}] {prompt}")
}

/// Validate a user answer against the question schema that produced it.
pub(crate) fn validate_user_answer(
    question: &UserQuestion,
    answer: &UserQuestionAnswer,
) -> anyhow::Result<()> {
    let free_text = answer.free_text.as_deref().map(str::trim).unwrap_or("");

    let allows_free_text = question.allow_free_text || question.mode == QuestionMode::Text;
    if !allows_free_text && !free_text.is_empty() {
        anyhow::bail!("This question does not accept free-text input");
    }

    let mut seen_ids = HashSet::<String>::new();
    for id in &answer.selected_option_ids {
        if !seen_ids.insert(id.clone()) {
            anyhow::bail!("Duplicate option id in answer: {id}");
        }
        if !question.options.iter().any(|option| option.id == *id) {
            anyhow::bail!("Unknown option id in answer: {id}");
        }
    }

    match question.mode {
        QuestionMode::Text => {
            if !answer.selected_option_ids.is_empty() {
                anyhow::bail!("Text questions do not accept selected options");
            }
            if free_text.is_empty() {
                anyhow::bail!("Text questions require a free-text answer");
            }
        }
        QuestionMode::SingleChoice => {
            if answer.selected_option_ids.len() > 1 {
                anyhow::bail!("Single-choice questions accept at most one selected option");
            }
            if answer.selected_option_ids.is_empty() && free_text.is_empty() {
                anyhow::bail!("Single-choice questions require one selection or free text");
            }
        }
        QuestionMode::MultiChoice => {
            if answer.selected_option_ids.is_empty() && free_text.is_empty() {
                anyhow::bail!("Multi-choice questions require at least one selection or free text");
            }
        }
    }

    Ok(())
}
