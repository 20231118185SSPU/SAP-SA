//! Working Memory Layer — a structured hot buffer that sits between the
//! agent's in-memory `messages` vector and the long-term Markdown files.
//!
//! Design goals:
//! - **Three-segment structure**: hot_buffer (recent messages), pinned_slots
//!   (user identity, task, never-evicted), scratchpad (internal reasoning).
//! - **Overflow summarization**: when hot_buffer exceeds capacity (message
//!   count OR token budget), the oldest 1/3 of non-pinned messages are
//!   LLM-compressed into a summary that stays in the buffer head.
//! - **Importance scoring**: lightweight rule-based scoring (no LLM call)
//!   on every inbound message; score > threshold triggers "consolidation
//!   candidate" flag for the dream pipeline.
//! - **Pinned slots**: model-controlled via PinMemory/UnpinMemory tools.
//!   Pinned entries are invisible to overflow eviction.
//!
//! This module is intentionally stateless across calls — each agent quantum
//! gets a fresh WorkingMemory instance. Long-term state lives in the
//! Markdown files (memory.rs / dream.rs).

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;

use crate::noise_assessment::{NoiseAssessment, NoiseAssessor, NoiseConfig};
use crate::openai::ChatMessage;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Runtime configuration for the working memory layer (`[working_memory]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkingMemoryConfig {
    /// Maximum number of messages held in the hot buffer.
    pub hot_buffer_max_messages: usize,
    /// Maximum total characters across all hot buffer messages.
    pub hot_buffer_max_chars: usize,
    /// Maximum memory usage in bytes (0 = unlimited).
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
    /// Noise assessment configuration.
    #[serde(default)]
    pub noise: NoiseConfig,
}

/// D8: Memory decay configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecayConfig {
    /// Days after last access before entry starts decaying.
    pub decay_after_days: u32,
    /// Exponential base — strength = decay_base ^ (-days_since_access).
    pub decay_base: f64,
    /// Boost added per access (strengthen feedback).
    pub access_boost: f64,
    /// Minimum effective importance after full decay.
    pub min_importance: f64,
    /// Access frequency weight — added as alpha * ln(1 + access_count).
    pub alpha: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            decay_after_days: 3,
            decay_base: 0.90,
            access_boost: 0.10,
            min_importance: 0.15,
            alpha: 0.10,
        }
    }
}

impl Default for WorkingMemoryConfig {
    fn default() -> Self {
        Self {
            hot_buffer_max_messages: 12,
            hot_buffer_max_chars: 8_000,
            max_memory_bytes: 10 * 1024 * 1024, // 10MB default
            importance_threshold: 0.6,
            overflow_compress_fraction: 0.33,
            enable_summarization: true,
            decay: None,
            noise: NoiseConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Importance scoring
// ---------------------------------------------------------------------------

/// Lightweight importance scoring using rule-based signals.
/// No LLM call — runs synchronously on every inbound message.
///
/// Score range: 0.0 – 1.0
///
/// Components:
///   - entity_indicator: contains proper nouns, dates, numbers (0.3)
///   - preference_keyword: contains preference/always/never/like/dislike (0.3)
///   - emotion_keyword: contains emotional language (0.2)
///   - explicit_marker: user prefixes with "remember" / "note that" (0.2)
fn score_importance(message: &ChatMessage) -> f64 {
    let text = message.text_content().unwrap_or("".to_string());
    let lower = text.to_ascii_lowercase();

    let mut score = 0.0;

    // Entity indicator: dates, numbers, proper noun patterns
    let has_entity = contains_entity_signal(&lower);
    if has_entity {
        score += 0.3;
    }

    // Preference / behavioral keywords
    let has_preference = contains_preference_signal(&lower);
    if has_preference {
        score += 0.6;
    }

    // Emotional intensity (high emotion=0.6, medium=0.4, low=0.1)
    let emotion = contains_emotion_signal(&lower);
    score += 0.6 * emotion;

    // Explicit importance markers
    let has_explicit = contains_explicit_marker(&lower);
    if has_explicit {
        score += 0.2;
    }

    score.min(1.0)
}

fn contains_entity_signal(text: &str) -> bool {
    // Dates, numbers, quoted terms, slash-separated identifiers
    let patterns = [
        r"\d{4}-\d{2}-\d{2}",  // ISO dates
        r"\d{1,2}[月日年]",     // Chinese date patterns
        r#""[^"]{3,}"#,       // Quoted strings
        r"/[\w-]+/[\w-]+",    // Slash paths
        r"\d[\d,.]+[万元人公里个次]", // Quantities with Chinese units
    ];
    patterns.iter().any(|p| regex_lite_match(p, text))
}

fn contains_preference_signal(text: &str) -> bool {
    let keywords = [
        "i prefer", "i like", "i dislike", "i hate", "always", "never",
        "我喜欢", "我讨厌", "我偏好", "我一般", "习惯了", "不要",
        "please", "don't", "avoid", "better", "worse", "最好", "不要",
        "remember", "note that", "keep in mind", "重要的是", "切记",
    ];
    keywords.iter().any(|kw| text.contains(kw))
}

fn contains_emotion_signal(text: &str) -> f64 {
    let high = ["angry", "frustrated", "terrible", "hate", "disappointed",
                "愤怒", "失望", "糟糕", "讨厌", "生气", "郁闷"];
    let medium = ["unhappy", "annoyed", "concerned", "worry", "hope",
                  "不太满意", "担心", "希望", "烦恼", "犹豫"];
    let low = ["okay", "fine", "alright", "ok", "还好", "一般"];

    if high.iter().any(|kw| text.contains(kw)) {
        1.0
    } else if medium.iter().any(|kw| text.contains(kw)) {
        0.6
    } else if low.iter().any(|kw| text.contains(kw)) {
        0.2
    } else {
        0.0
    }
}

fn contains_explicit_marker(text: &str) -> bool {
    let markers = [
        "remember", "note that", "keep in mind", "important",
        "切记", "重要的是", "请记住", "提醒", "别忘了",
    ];
    markers.iter().any(|m| text.contains(m))
}

/// Simple regex-lite pattern matching (no external crate dependency).
/// Supports: \d{4}-\d{2}-\d{2} (ISO dates), \d{4} (4-digit years), basic text contains.
fn regex_lite_match(pattern: &str, text: &str) -> bool {
    match pattern {
        r"\d{4}-\d{2}-\d{2}" => {
            // Check for YYYY-MM-DD pattern (including YYYY/MM/DD and YYYY.MM.DD)
            for window in text.as_bytes().windows(10) {
                if window[0].is_ascii_digit()
                    && window[1].is_ascii_digit()
                    && window[2].is_ascii_digit()
                    && window[3].is_ascii_digit()
                    && (window[4] == b'-' || window[4] == b'/' || window[4] == b'.')
                    && window[5].is_ascii_digit()
                    && window[6].is_ascii_digit()
                    && (window[7] == b'-' || window[7] == b'/' || window[7] == b'.')
                    && window[8].is_ascii_digit()
                    && window[9].is_ascii_digit()
                {
                    return true;
                }
            }
            false
        }
        _ => text.contains(pattern),
    }
}

// ---------------------------------------------------------------------------
// Hot message entry
// ---------------------------------------------------------------------------

/// One entry in the working memory hot buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryEntry {
    /// The message stored.
    pub message: ChatMessage,
    /// Importance score computed at insert time (0.0 – 1.0).
    pub importance: f64,
    /// Whether this entry is marked as a consolidation candidate
    /// (importance > threshold at insert time).
    pub is_consolidation_candidate: bool,
    /// Character count at insert time.
    pub char_count: usize,
    /// D8: Last access timestamp (ms, UTC). Used for decay calculation.
    #[serde(default)]
    pub last_access: Option<i64>,
    /// Number of times this entry has been accessed. Used for strength formula.
    #[serde(default)]
    pub access_count: u32,
}

impl WorkingMemoryEntry {
    fn new(message: ChatMessage, config: &WorkingMemoryConfig) -> Self {
        let importance = score_importance(&message);
        let char_count = message.text_content().unwrap_or("".to_string()).chars().count();
        Self {
            message,
            importance,
            is_consolidation_candidate: importance >= config.importance_threshold,
            char_count,
            last_access: Some(chrono::Utc::now().timestamp_millis()),
            access_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pinned slot entry
// ---------------------------------------------------------------------------

/// One pinned slot entry. Pinned entries are never evicted by overflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedSlot {
    /// Human-readable label for this pinned entry.
    pub label: String,
    /// The content stored in this slot.
    pub content: String,
    /// When this slot was created (Unix timestamp ms).
    pub pinned_at: i64,
    /// Optional free-form note about why this was pinned.
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// Working memory struct
// ---------------------------------------------------------------------------

/// Three-segment working memory.
///
/// - **hot_buffer**: ring buffer of recent messages with importance scores.
///   Evicted on overflow (oldest non-pinned messages compressed first).
/// - **pinned_slots**: model-controlled permanent entries (user identity,
///   current task, etc.). Never evicted automatically.
/// - **scratchpad**: agent's internal reasoning, can be discarded at any time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// Config reference (not stored, passed in at construction).
    #[serde(skip)]
    config: WorkingMemoryConfig,

    /// Ring buffer of recent conversation messages.
    hot_buffer: VecDeque<WorkingMemoryEntry>,

    /// Model-controlled pinned slots.
    pinned_slots: HashMap<String, PinnedSlot>,

    /// Internal reasoning scratchpad.
    scratchpad: String,

    /// Running character count of hot_buffer contents.
    total_chars: usize,

    /// Total message count in hot_buffer.
    message_count: usize,

    /// Number of overflow compressions performed in this session.
    compressions: u32,
}

impl WorkingMemory {
    /// Construct a new working memory with default config.
    pub fn new() -> Self {
        Self::from_config(WorkingMemoryConfig::default())
    }

    /// Construct a new working memory with custom config.
    pub fn from_config(config: WorkingMemoryConfig) -> Self {
        Self {
            config,
            hot_buffer: VecDeque::new(),
            pinned_slots: HashMap::new(),
            scratchpad: String::new(),
            total_chars: 0,
            message_count: 0,
            compressions: 0,
        }
    }

    /// Add a user or assistant message to the hot buffer.
    /// Triggers overflow check after insertion.
    pub fn push_message(&mut self, message: ChatMessage) -> OverflowAction {
        let entry = WorkingMemoryEntry::new(message, &self.config);
        let char_count = entry.char_count;

        // Pre-check: detect if buffer is at/over capacity before insert
        let needs_eviction = self.message_count >= self.config.hot_buffer_max_messages
            || self.total_chars + char_count > self.config.hot_buffer_max_chars;

        if needs_eviction {
            if self.config.enable_summarization {
                // Evict oldest to make room; return NeedsSummarization so caller
                // runs summarize_and_compress() after this insert.
                while self.message_count >= self.config.hot_buffer_max_messages
                    || self.total_chars + char_count > self.config.hot_buffer_max_chars
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

    /// Check if overflow has occurred and return the action to take.
    #[allow(dead_code)]
    fn check_overflow(&mut self) -> OverflowAction {
        let exceeds_count = self.message_count >= self.config.hot_buffer_max_messages;
        let exceeds_chars = self.total_chars >= self.config.hot_buffer_max_chars;

        if !exceeds_count && !exceeds_chars {
            return OverflowAction::None;
        }

        if !self.config.enable_summarization {
            // Without summarization, just evict the oldest non-pinned entries
            self.evict_oldest();
            return OverflowAction::Evicted;
        }

        OverflowAction::NeedsSummarization
    }

    /// Evict the oldest entries from the hot buffer until within limits.
    /// Pinned entries (if any were somehow added — they shouldn't be) are skipped.
    fn evict_oldest(&mut self) {
        while self.message_count > 0
            && (self.message_count > self.config.hot_buffer_max_messages
                || self.total_chars > self.config.hot_buffer_max_chars)
        {
            if let Some(entry) = self.hot_buffer.pop_front() {
                self.total_chars = self.total_chars.saturating_sub(entry.char_count);
                self.message_count = self.message_count.saturating_sub(1);
            } else {
                break;
            }
        }
    }

    /// Perform overflow summarization: compress the oldest 1/3 of the buffer
    /// into a single summary entry at the head.
    ///
    /// Returns the text of the compressed block so callers can optionally
    /// persist it to a daily memory file.
    pub fn summarize_and_compress(&mut self) -> String {
        let raw_count = (self.hot_buffer.len() as f64)
            * self.config.overflow_compress_fraction;
        let count_to_compress = raw_count.ceil() as usize;
        let count_to_compress = count_to_compress.max(1);

        let entries_to_compress: Vec<_> = self.hot_buffer.iter().take(count_to_compress).collect();
        let combined_text: String = entries_to_compress
            .iter()
            .map(|e| e.message.text_content().unwrap_or("".to_string()))
            .collect::<Vec<_>>()
            .join("\n");

        // Remove the compressed entries
        for _ in 0..count_to_compress {
            if let Some(entry) = self.hot_buffer.pop_front() {
                self.total_chars = self.total_chars.saturating_sub(entry.char_count);
                self.message_count = self.message_count.saturating_sub(1);
            }
        }

        // Insert summary at head
        let summary_text = format!(
            "[记忆压缩摘要 - {} 条消息]: {}",
            count_to_compress,
            combined_text.chars().take(500).collect::<String>()
        );
        let summary_message = ChatMessage::text("system", &summary_text);
        let summary_entry = WorkingMemoryEntry::new(summary_message, &self.config);

        let summary_char_count = summary_entry.char_count;
        self.hot_buffer.push_front(summary_entry);
        self.total_chars += summary_char_count;
        self.message_count += 1;
        self.compressions += 1;

        summary_text
    }

    /// Pin a slot. Overwrites any existing slot with the same key.
    pub fn pin(&mut self, key: String, label: String, content: String, note: Option<String>) {
        let slot = PinnedSlot {
            label,
            content,
            pinned_at: chrono::Utc::now().timestamp_millis(),
            note,
        };
        self.pinned_slots.insert(key, slot);
    }

    /// Unpin (remove) a slot by key.
    /// Returns the removed slot if it existed.
    pub fn unpin(&mut self, key: &str) -> Option<PinnedSlot> {
        self.pinned_slots.remove(key)
    }

    /// Check if a slot is currently pinned.
    pub fn is_pinned(&self, key: &str) -> bool {
        self.pinned_slots.contains_key(key)
    }

    /// Update the scratchpad content.
    pub fn set_scratchpad(&mut self, text: String) {
        self.scratchpad = text;
    }

    /// Get all pinned slot keys and labels.
    pub fn pinned_keys(&self) -> Vec<(String, String)> {
        self.pinned_slots
            .iter()
            .map(|(k, v)| (k.clone(), v.label.clone()))
            .collect()
    }

    /// Get all consolidation-candidate entries (for dream pipeline).
    /// Includes both: entries with high initial importance (is_consolidation_candidate),
    /// AND entries whose effective_importance has decayed below threshold.
    pub fn consolidation_candidates(&self) -> Vec<&WorkingMemoryEntry> {
        self.hot_buffer
            .iter()
            .filter(|e| {
                e.is_consolidation_candidate || self.effective_importance(e) < self.config.importance_threshold
            })
            .collect()
    }

    /// D8: Decay scanner — call this periodically (e.g. per quantum or heartbeat).
    /// - Computes effective_importance for each entry
    /// - Applies weaken() to entries below threshold (decline their importance score)
    /// - Returns how many entries were weakened (for logging/audit)
    /// - Does NOT remove entries — dream pipeline handles consolidation
    pub fn scan_and_decay(&mut self) -> usize {
        let Some(ref cfg) = self.config.decay else { return 0 };
        let now_ms = chrono::Utc::now().timestamp_millis();
        let threshold = self.config.importance_threshold;
        let mut weakened = 0;
        for i in 0..self.hot_buffer.len() {
            let last_access = self.hot_buffer[i].last_access.unwrap_or(0);
            if last_access == 0 { continue; }
            let elapsed_days = ((now_ms - last_access) / 86_400_000) as f64;
            if elapsed_days <= cfg.decay_after_days as f64 { continue; }
            // Compute effective_importance inlined to avoid borrow conflict
            let decay_days = elapsed_days - cfg.decay_after_days as f64;
            let strength = cfg.decay_base.powf(-decay_days);
            let freq_boost = cfg.alpha * (1.0 + self.hot_buffer[i].access_count as f64).ln();
            let effective = (strength * self.hot_buffer[i].importance + cfg.min_importance + freq_boost)
                .min(1.0)
                .max(cfg.min_importance);
            if effective < threshold {
                // Decrement importance by access_boost, clamped to min_importance
                let new_imp = (self.hot_buffer[i].importance - cfg.access_boost).max(cfg.min_importance);
                self.hot_buffer[i].importance = new_imp;
                weakened += 1;
            }
        }
        weakened
    }

    /// Get the current hot buffer as a vector of ChatMessages (for injection
    /// into the agent's messages list).
    pub fn to_messages(&self) -> Vec<ChatMessage> {
        self.hot_buffer.iter().map(|e| e.message.clone()).collect()
    }

    // ===================================================================
    // D8: Decay / Strengthen — forgettability formula
    // ===================================================================

    /// Compute effective importance of an entry after applying decay.
    /// strength = decay_base ^ (-days_since_last_access) * initial_importance
    /// clamped to [decay.min_importance, 1.0].
    pub fn effective_importance(&self, entry: &WorkingMemoryEntry) -> f64 {
        let Some(ref cfg) = self.config.decay else {
            return entry.importance;
        };
        let last_access = entry.last_access.unwrap_or(0);
        if last_access == 0 {
            return entry.importance;
        }
        let now_ms = chrono::Utc::now().timestamp_millis();
        let elapsed_days = ((now_ms - last_access) / 86_400_000) as f64;
        if elapsed_days <= cfg.decay_after_days as f64 {
            return entry.importance;
        }
        let decay_days = elapsed_days - cfg.decay_after_days as f64;
        let strength = cfg.decay_base.powf(-decay_days);
        let freq_boost = cfg.alpha * (1.0 + entry.access_count as f64).ln();
        let effective = (strength * entry.importance + cfg.min_importance + freq_boost).min(1.0);
        effective.max(cfg.min_importance)
    }

    /// Apply a weaken action to an entry (reduce importance after decay).
    /// Clamps importance to [decay.min_importance, 1.0].
    pub fn weaken(&mut self, index: usize) {
        let Some(ref cfg) = self.config.decay else { return };
        if index < self.hot_buffer.len() {
            let entry = &mut self.hot_buffer[index];
            let new_imp = (entry.importance - cfg.access_boost).max(cfg.min_importance);
            entry.importance = new_imp;
        }
    }

    /// Record an access (read/cite) of a hot buffer entry, preventing decay.
    pub fn record_access(&mut self, index: usize) {
        if index < self.hot_buffer.len() {
            let entry = &mut self.hot_buffer[index];
            entry.last_access = Some(chrono::Utc::now().timestamp_millis());
            entry.access_count = entry.access_count.saturating_add(1);
        }
    }

    /// Apply a strengthen boost to an entry after positive feedback.
    /// Clamps importance to [0.0, 1.0].
    pub fn strengthen(&mut self, index: usize) {
        let Some(ref cfg) = self.config.decay else { return };
        if index < self.hot_buffer.len() {
            let entry = &mut self.hot_buffer[index];
            let new_imp = (entry.importance + cfg.access_boost).min(1.0);
            entry.importance = new_imp;
            entry.last_access = Some(chrono::Utc::now().timestamp_millis());
        }
    }

    /// Return entries whose effective_importance has fallen below the
    /// consolidation threshold — candidates for the dream pipeline.
    /// Includes both static consolidation markers AND dynamic decayed entries.
    pub fn decayed_candidates(&self) -> Vec<(usize, WorkingMemoryEntry)> {
        self.hot_buffer
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.is_consolidation_candidate
                    || self.effective_importance(e) < self.config.importance_threshold
            })
            .map(|(i, e)| (i, e.clone()))
            .collect()
    }

    /// Build a human-readable decay candidates summary for dream prompt injection.
    pub fn decay_candidates_summary(&self) -> String {
        let candidates = self.decayed_candidates();
        if candidates.is_empty() {
            return String::from("（无待整合条目）");
        }
        let mut out = String::new();
        for (i, entry) in &candidates {
            let eff = self.effective_importance(entry);
            let last = entry.last_access.map(|ts| {
                let dt = chrono::DateTime::from_timestamp_millis(ts)
                    .unwrap_or_default();
                dt.format("%Y-%m-%d").to_string()
            }).unwrap_or_else(|| "从未访问".to_string());
            let content = entry.message.text_content()
                .unwrap_or_default()
                .chars()
                .take(80)
                .collect::<String>();
            out.push_str(&format!(
                "- [{i}] importance={:.2} effective={:.2} last_access={} content={}\n",
                entry.importance,
                eff,
                last,
                content
            ));
        }
        out
    }

    /// Get the pinned slots as a formatted string for prompt injection.
    pub fn pinned_context(&self) -> String {
        if self.pinned_slots.is_empty() {
            return String::new();
        }
        let mut out = String::from("\n## Pinned Memory\n\n");
        for (key, slot) in &self.pinned_slots {
            out.push_str(&format!("**{}** ({})\n{}\n\n", slot.label, key, slot.content));
        }
        out
    }

    /// Get scratchpad content.
    pub fn scratchpad(&self) -> &str {
        &self.scratchpad
    }

    /// Current stats for logging/debugging.
    pub fn stats(&self) -> WorkingMemoryStats {
        WorkingMemoryStats {
            hot_buffer_count: self.message_count,
            hot_buffer_chars: self.total_chars,
            pinned_count: self.pinned_slots.len(),
            compressions: self.compressions,
            consolidation_candidates: self
                .hot_buffer
                .iter()
                .filter(|e| e.is_consolidation_candidate)
                .count(),
        }
    }

    /// Load pinned slots from a JSON file on disk (called at session restore).
    pub fn load_pinned_from_file(
        workspace_root: &Path,
    ) -> anyhow::Result<HashMap<String, PinnedSlot>> {
        let path = workspace_root.join("memory/.working_memory_pinned.json");
        if !path.is_file() {
            return Ok(HashMap::new());
        }
        let raw = std::fs::read_to_string(&path)?;
        let slots: HashMap<String, PinnedSlot> = serde_json::from_str(&raw)
            .context("Failed to parse pinned slots file")?;
        Ok(slots)
    }

    /// Persist pinned slots to disk.
    pub fn save_pinned_to_file(
        workspace_root: &Path,
        slots: &HashMap<String, PinnedSlot>,
    ) -> anyhow::Result<()> {
        let path = workspace_root.join("memory/.working_memory_pinned.json");
        std::fs::create_dir_all(path.parent().unwrap())?;
        let raw = serde_json::to_string_pretty(slots)
            .context("Failed to serialize pinned slots")?;
        std::fs::write(&path, raw).context("Failed to write pinned slots file")?;
        Ok(())
    }

    /// Assess noise in the current hot buffer.
    pub fn assess_noise(&self, config: NoiseConfig) -> NoiseAssessment {
        let assessor = NoiseAssessor::new(config);
        let entries: Vec<WorkingMemoryEntry> = self.hot_buffer.iter().cloned().collect();
        assessor.assess_working_memory(&entries)
    }

    /// Get a noise report string for prompt injection.
    pub fn noise_report(&self, config: NoiseConfig) -> String {
        let assessment = self.assess_noise(config.clone());
        let assessor = NoiseAssessor::new(config);
        assessor.format_noise_report(&assessment)
    }

    /// Check if noise level is high (exceeds threshold).
    pub fn is_high_noise(&self, config: &NoiseConfig) -> bool {
        if !config.enabled {
            return false;
        }
        let assessment = self.assess_noise(config.clone());
        assessment.noise_ratio > config.high_noise_threshold
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Result of an overflow check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowAction {
    /// No overflow; buffer is within limits.
    None,
    /// Overflow detected but summarization is disabled; oldest entries evicted.
    Evicted,
    /// Overflow detected; summarization is needed.
    NeedsSummarization,
}

/// Snapshot stats for logging and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryStats {
    pub hot_buffer_count: usize,
    pub hot_buffer_chars: usize,
    pub pinned_count: usize,
    pub compressions: u32,
    pub consolidation_candidates: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: impl Into<String>, content: impl Into<String>) -> ChatMessage {
        ChatMessage::text(role, content)
    }

    #[test]
    fn importance_preference_keywords() {
        let msg = make_message("user", "I prefer working in the morning, please remember that");
        let score = score_importance(&msg);
        assert!(score >= 0.5, "preference + explicit marker should score >= 0.5, got {score}");
    }

    #[test]
    fn importance_entity_signals() {
        let msg = make_message("user", "On 2026-04-29 we decided to use Qdrant");
        let score = score_importance(&msg);
        assert!(score >= 0.3, "date entity should add 0.3, got {score}");
    }

    #[test]
    fn importance_emotion_high() {
        let msg = make_message("user", "I'm really frustrated and angry about this issue");
        let score = score_importance(&msg);
        assert!(score >= 0.4, "high emotion should add 0.4+, got {score}");
    }

    #[test]
    fn working_memory_respects_max_messages() {
        let config = WorkingMemoryConfig {
            hot_buffer_max_messages: 3,
            hot_buffer_max_chars: 50_000,
            ..Default::default()
        };
        let mut wm = WorkingMemory::from_config(config);

        for i in 0..5 {
            wm.push_message(make_message("user", format!("message {i}")));
        }

        // Should evict to max_messages
        assert_eq!(wm.message_count, 3);
    }

    #[test]
    fn working_memory_summarization() {
        let config = WorkingMemoryConfig {
            hot_buffer_max_messages: 3,
            hot_buffer_max_chars: 50_000,
            enable_summarization: true,
            overflow_compress_fraction: 0.33,
            ..Default::default()
        };
        let mut wm = WorkingMemory::from_config(config);

        for i in 0..5 {
            wm.push_message(make_message("user", format!("message {i}")));
        }

        let action = wm.check_overflow();
        assert_eq!(action, OverflowAction::NeedsSummarization);

        let summary = wm.summarize_and_compress();
        assert!(summary.contains("记忆压缩摘要"));
        assert!(wm.compressions >= 1);
        assert!(wm.message_count <= 3); // summary replaces compressed entries
    }

    #[test]
    fn pin_and_unpin() {
        let mut wm = WorkingMemory::new();
        wm.pin(
            "user_name".into(),
            "User Name".into(),
            "小明".into(),
            None,
        );

        assert!(wm.is_pinned("user_name"));
        assert!(!wm.is_pinned("nonexistent"));

        let removed = wm.unpin("user_name");
        assert!(removed.is_some());
        assert!(!wm.is_pinned("user_name"));
    }

    #[test]
    fn consolidation_candidates() {
        let config = WorkingMemoryConfig {
            importance_threshold: 0.6,
            ..Default::default()
        };
        let mut wm = WorkingMemory::from_config(config);

        // Low importance
        wm.push_message(make_message("user", "hello"));
        // High importance (preference keyword)
        wm.push_message(make_message("user", "I prefer to work in the morning (9am start)"));

        let candidates = wm.consolidation_candidates();
        assert!(!candidates.is_empty(), "preference message should be a candidate");
    }
}
