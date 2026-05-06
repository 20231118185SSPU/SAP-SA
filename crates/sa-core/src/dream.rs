//! Background "dream" support for StudyAdministrator (SA).
//!
//! `dream` is not a generic nightly summary job. Its purpose is to make the
//! agent's long-term memory system improve over time:
//! - keep raw experience in durable logs (`sessions/*.jsonl`,
//!   `memory/YYYY-MM-DD.md`)
//! - distill only high-value, reusable knowledge into long-term memory
//!   (`MEMORY.md`, `memory/topics/*.md`)
//! - record the consolidation/audit trail in `memory/dreams/*.md`
//!
//! The daemon owns scheduling. This module focuses on:
//! - config defaults
//! - state persistence
//! - per-workspace locking
//! - deciding when a dream run is due
//! - collecting the most relevant input files
//! - constructing the dream-specific prompt/context block

use anyhow::Context as _;
use chrono::{DateTime, Duration, Local, LocalResult, NaiveDate, TimeZone};
use fs4::fs_std::FileExt as _;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead as _, BufReader, Write as _};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Relative directory that stores curated topic memory.
pub const DREAM_TOPIC_MEMORY_DIR: &str = "memory/topics";

/// Relative directory that stores dream audit Markdown files.
pub const DREAM_AUDIT_DIR: &str = "memory/dreams";

/// Relative directory that stores raw main-session history.
pub const DREAM_SESSIONS_DIR: &str = "sessions";

/// State file persisted under the workspace memory directory.
const DREAM_STATE_FILE: &str = "memory/.dream-state.json";

/// Cross-process lock file used to prevent overlapping dream runs.
const DREAM_LOCK_FILE: &str = "memory/.dream.lock";

/// Nightly dream configuration (`[dream]` in `sa.toml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DreamConfig {
    /// Master kill-switch.
    pub enabled: bool,
    /// Number of flat daily notes (`memory/YYYY-MM-DD.md`) inspected per run.
    pub daily_note_lookback_days: usize,
    /// Number of recent `sessions/*.jsonl` segments offered to the dream task.
    pub recent_session_segments: usize,
    /// Maximum number of topic memory files listed in the prompt.
    pub recent_topic_files: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            daily_note_lookback_days: 3,
            recent_session_segments: 6,
            recent_topic_files: 24,
        }
    }
}

/// Durable per-workspace dream state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamState {
    /// Local calendar date of the last successful dream run.
    pub last_success_local_date: Option<String>,
    /// Local timestamp when the current/most recent run started.
    pub last_started_at: Option<String>,
    /// Local timestamp when the most recent successful run finished.
    pub last_completed_at: Option<String>,
    /// Relative path of the latest audit Markdown file.
    pub last_report_path: Option<String>,
}

/// Files selected for one dream run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DreamSources {
    /// Stable long-term memory index, if present.
    pub memory_index: Option<String>,
    /// Curated topic memories.
    pub topic_files: Vec<String>,
    /// Recent raw daily notes.
    pub daily_notes: Vec<String>,
    /// Recent raw session segments.
    pub session_segments: Vec<String>,
}

/// Fully prepared dream run description.
#[derive(Debug, Clone)]
pub struct PreparedDreamRun {
    /// User-task string passed into the isolated agent run.
    pub task: String,
    /// Extra runtime block injected into the system prompt.
    pub extra_system_prompt: String,
    /// Audit file that the dream task must create/update.
    pub report_relative_path: String,
    /// Source file set used to build the prompt.
    pub sources: DreamSources,
}

/// Workspace-bound dream manager.
#[derive(Debug, Clone)]
pub struct DreamManager {
    /// Canonical workspace root.
    workspace_root: PathBuf,
    /// Effective dream config.
    cfg: DreamConfig,
    /// State file path.
    state_path: PathBuf,
    /// Lock file path.
    lock_path: PathBuf,
}

/// RAII guard for an acquired dream lock.
pub struct DreamRunLock {
    /// Keeping the file handle alive keeps the lock alive.
    _file: File,
}

impl DreamManager {
    /// Create one workspace-bound dream manager.
    pub fn new(workspace_root: PathBuf, cfg: DreamConfig) -> anyhow::Result<Self> {
        let workspace_root = fs::canonicalize(&workspace_root).with_context(|| {
            format!(
                "Failed to canonicalize workspace root for dream manager: {}",
                workspace_root.display()
            )
        })?;

        Ok(Self {
            state_path: workspace_root.join(DREAM_STATE_FILE),
            lock_path: workspace_root.join(DREAM_LOCK_FILE),
            workspace_root,
            cfg,
        })
    }

    /// Return the effective config.
    pub fn config(&self) -> &DreamConfig {
        &self.cfg
    }

    /// Return `true` when a dream run should happen now.
    ///
    /// Current policy:
    /// - if disabled, never run
    /// - otherwise run at most once per local calendar date
    /// - a workspace that has already completed at least one dream keeps the
    ///   old "startup catch-up" behavior
    /// - a never-processed workspace only catches up when substantive memory or
    ///   session history already exists; an auto-created empty bootstrap
    ///   session must not trigger dream by itself
    pub fn should_run_now(&self, now: DateTime<Local>) -> anyhow::Result<bool> {
        if !self.cfg.enabled {
            return Ok(false);
        }

        let state = self.load_state()?;
        let today = now.date_naive().to_string();
        if state.last_success_local_date.as_deref() == Some(today.as_str()) {
            return Ok(false);
        }

        if state.last_success_local_date.is_some() {
            return Ok(true);
        }

        let sources = self.collect_sources(now)?;
        Ok(sources.memory_index.is_some()
            || !sources.topic_files.is_empty()
            || !sources.daily_notes.is_empty()
            || !sources.session_segments.is_empty())
    }

    /// Compute the next local midnight strictly after `now`.
    pub fn next_run_after(&self, now: DateTime<Local>) -> DateTime<Local> {
        let tomorrow = now.date_naive() + Duration::days(1);
        resolve_local_time(tomorrow)
    }

    /// Acquire the per-workspace dream lock.
    ///
    /// Returns `Ok(None)` when another live process/task already holds it.
    pub fn try_acquire_lock(&self) -> anyhow::Result<Option<DreamRunLock>> {
        self.ensure_layout()?;

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&self.lock_path)
            .with_context(|| {
                format!(
                    "Failed to open dream lock file: {}",
                    self.lock_path.display()
                )
            })?;

        let acquired = file.try_lock_exclusive().with_context(|| {
            format!("Failed to acquire dream lock: {}", self.lock_path.display())
        })?;
        if !acquired {
            return Ok(None);
        }

        Ok(Some(DreamRunLock { _file: file }))
    }

    /// Persist "run started" metadata before the isolated agent begins.
    pub fn mark_started(&self, now: DateTime<Local>) -> anyhow::Result<()> {
        let mut state = self.load_state()?;
        state.last_started_at = Some(now.to_rfc3339());
        self.save_state(&state)
    }

    /// Persist "run completed" metadata after a successful dream.
    pub fn mark_completed(
        &self,
        run_date: NaiveDate,
        completed_at: DateTime<Local>,
        report_relative_path: &str,
    ) -> anyhow::Result<()> {
        let mut state = self.load_state()?;
        state.last_success_local_date = Some(run_date.to_string());
        state.last_completed_at = Some(completed_at.to_rfc3339());
        state.last_report_path = Some(report_relative_path.replace('\\', "/"));
        self.save_state(&state)
    }

    /// Read the current dream state, returning defaults when the file does not
    /// exist yet.
    pub fn load_state(&self) -> anyhow::Result<DreamState> {
        match fs::read_to_string(&self.state_path) {
            Ok(raw) => serde_json::from_str(&raw).with_context(|| {
                format!(
                    "Failed to parse dream state file: {}",
                    self.state_path.display()
                )
            }),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(DreamState::default()),
            Err(err) => Err(err).with_context(|| {
                format!(
                    "Failed to read dream state file: {}",
                    self.state_path.display()
                )
            }),
        }
    }

    /// Build one fully prepared dream run for the current local date.
    pub fn prepare_run(&self, now: DateTime<Local>) -> anyhow::Result<PreparedDreamRun> {
        self.ensure_layout()?;

        let sources = self.collect_sources(now)?;
        let report_relative_path = format!("{DREAM_AUDIT_DIR}/{}.md", now.format("%Y-%m-%d"));
        let extra_system_prompt =
            self.build_runtime_context_block(now, &sources, &report_relative_path);
        let task = format!(
            "执行一次长期记忆 dream 提炼。目标不是生成普通摘要，而是净化长期记忆、去重、去噪、修正冲突，并把审计结果写入 `{report_relative_path}`。"
        );

        Ok(PreparedDreamRun {
            task,
            extra_system_prompt,
            report_relative_path,
            sources,
        })
    }

    /// Ensure the directories required by the dream workflow exist.
    fn ensure_layout(&self) -> anyhow::Result<()> {
        fs::create_dir_all(self.workspace_root.join("memory")).with_context(|| {
            format!(
                "Failed to create memory directory under {}",
                self.workspace_root.display()
            )
        })?;
        fs::create_dir_all(self.workspace_root.join(DREAM_TOPIC_MEMORY_DIR)).with_context(
            || {
                format!(
                    "Failed to create topic memory directory under {}",
                    self.workspace_root.display()
                )
            },
        )?;
        fs::create_dir_all(self.workspace_root.join(DREAM_AUDIT_DIR)).with_context(|| {
            format!(
                "Failed to create dream audit directory under {}",
                self.workspace_root.display()
            )
        })?;
        Ok(())
    }

    /// Persist the full state file.
    fn save_state(&self, state: &DreamState) -> anyhow::Result<()> {
        self.ensure_layout()?;
        let raw = serde_json::to_string_pretty(state).context("Failed to serialize dream state")?;
        let mut file = File::create(&self.state_path).with_context(|| {
            format!(
                "Failed to write dream state file: {}",
                self.state_path.display()
            )
        })?;
        file.write_all(raw.as_bytes())
            .context("Failed to write dream state contents")?;
        file.write_all(b"\n")
            .context("Failed to terminate dream state file")?;
        file.flush().context("Failed to flush dream state file")?;
        Ok(())
    }

    /// Collect the most relevant input files for one dream run.
    fn collect_sources(&self, now: DateTime<Local>) -> anyhow::Result<DreamSources> {
        let memory_index = self
            .workspace_root
            .join("MEMORY.md")
            .is_file()
            .then(|| "MEMORY.md".to_string())
            .or_else(|| {
                self.workspace_root
                    .join("memory.md")
                    .is_file()
                    .then(|| "memory.md".to_string())
            });

        let topic_files = list_recent_markdown_files(
            &self.workspace_root.join(DREAM_TOPIC_MEMORY_DIR),
            self.cfg.recent_topic_files,
            &self.workspace_root,
        )?;

        let mut daily_notes = Vec::<String>::new();
        for offset in 0..self.cfg.daily_note_lookback_days {
            let date = now.date_naive() - Duration::days(offset as i64);
            let relative = format!("memory/{}.md", date.format("%Y-%m-%d"));
            if self.workspace_root.join(&relative).is_file() {
                daily_notes.push(relative);
            }
        }

        let session_segments = list_recent_session_segments(
            &self.workspace_root.join(DREAM_SESSIONS_DIR),
            self.cfg.recent_session_segments,
            &self.workspace_root,
        )?;

        Ok(DreamSources {
            memory_index,
            topic_files,
            daily_notes,
            session_segments,
        })
    }

    /// Build the dream-specific runtime block injected into the system prompt.
    fn build_runtime_context_block(
        &self,
        now: DateTime<Local>,
        sources: &DreamSources,
        report_relative_path: &str,
    ) -> String {
        let mut out = String::new();
        out.push_str("## Dream Runtime\n\n");
        out.push_str("你当前执行的是后台 dream 记忆提炼任务。\n");
        out.push_str(
            "dream 的目标不是普通摘要，而是长期记忆治理：去噪、去重、修正冲突、抽象经验。\n",
        );
        out.push_str("这是后台任务：不要使用 `Send`、`Ask`、`Show` 或 `SubAgent`。如需查看文件，请使用 `Read`；如需局部搜索原始会话，请优先用 `Bash` 做只读 grep/find/cat，再用 `Read` 精确读取。\n\n");

        out.push_str("### 写入目标\n\n");
        out.push_str("- 更新 `MEMORY.md` 作为长期记忆总纲（如有必要）\n");
        out.push_str("- 更新或新建 `memory/topics/*.md` 作为专题长期记忆（如有必要）\n");
        out.push_str(&format!(
            "- 把本次提炼审计写入 `{report_relative_path}`\n\n"
        ));

        out.push_str("### 优先输入源\n\n");
        push_optional_path(&mut out, "长期总纲", sources.memory_index.as_deref());
        push_path_list(&mut out, "专题长期记忆", &sources.topic_files);
        push_path_list(&mut out, "最近原始日记", &sources.daily_notes);
        push_path_list(&mut out, "最近原始会话段", &sources.session_segments);

        out.push_str("### 工作流程\n\n");
        out.push_str("1. 先理解现有长期记忆结构，避免写重复内容。\n");
        out.push_str("2. 从最近原始经历中找出值得长期保留的稳定信息。\n");
        out.push_str("3. 删除一次性噪声、过时信息与互相冲突的旧记忆。\n");
        out.push_str("4. 把真正稳定、可复用的知识提炼到 `MEMORY.md` 或 `memory/topics/*.md`。\n");
        out.push_str("5. 在审计文件中记录：看了哪些来源、提升了哪些记忆、合并了哪些重复项、删除了哪些噪声、修正了哪些冲突。\n");
        out.push_str(
            "6. 如果没有需要调整的内容，也要在审计文件中写明“本次未发现值得更新的长期记忆”。\n\n",
        );

        out.push_str("### 质量要求\n\n");
        out.push_str("- 不要把一次性任务细节、临时错误日志或短期中间状态直接塞进长期记忆。\n");
        out.push_str("- **严格排除 agent 自评价**：不要把 agent 对自身能力的评价、自我反思、主观判断写入长期记忆。只记录客观事实和可复用模式。\n");
        out.push_str("- 相对时间要尽量转成绝对日期，避免过几天后无法理解。\n");
        out.push_str("- 长期记忆追求高密度、高稳定性，而不是数量。\n");
        out.push_str(&format!("- 当前本地日期：`{}`\n", now.format("%Y-%m-%d")));

        out
    }
}

/// Resolve one local-date midnight, tolerating DST edge cases conservatively.
fn resolve_local_time(date: NaiveDate) -> DateTime<Local> {
    let midnight = date
        .and_hms_opt(0, 0, 0)
        .expect("00:00:00 should be a valid NaiveDateTime");
    match Local.from_local_datetime(&midnight) {
        LocalResult::Single(dt) => dt,
        LocalResult::Ambiguous(first, _) => first,
        LocalResult::None => {
            for minute in 1..=180 {
                let shifted = midnight + Duration::minutes(minute);
                match Local.from_local_datetime(&shifted) {
                    LocalResult::Single(dt) => return dt,
                    LocalResult::Ambiguous(first, _) => return first,
                    LocalResult::None => continue,
                }
            }
            panic!("Failed to resolve a valid local datetime near {midnight}");
        }
    }
}

/// Append one optional path item to the prompt block.
fn push_optional_path(out: &mut String, title: &str, value: Option<&str>) {
    match value {
        Some(path) => {
            out.push_str(&format!("- {title}：`{path}`\n"));
        }
        None => {
            out.push_str(&format!("- {title}：（无）\n"));
        }
    }
}

/// Append one titled path list to the prompt block.
fn push_path_list(out: &mut String, title: &str, values: &[String]) {
    if values.is_empty() {
        out.push_str(&format!("- {title}：（无）\n"));
        return;
    }
    out.push_str(&format!("- {title}：\n"));
    for value in values {
        out.push_str(&format!("  - `{value}`\n"));
    }
}

/// List recent Markdown files under one directory, newest first by modified
/// time, breaking ties by path.
fn list_recent_markdown_files(
    dir: &Path,
    limit: usize,
    workspace_root: &Path,
) -> anyhow::Result<Vec<String>> {
    if !dir.is_dir() || limit == 0 {
        return Ok(Vec::new());
    }

    let mut files = Vec::<(std::time::SystemTime, String)>::new();
    for entry in walkdir::WalkDir::new(dir).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        if path
            .extension()
            .and_then(|value| value.to_str())
            .is_none_or(|value| !value.eq_ignore_ascii_case("md"))
        {
            continue;
        }

        let metadata = fs::metadata(path)
            .with_context(|| format!("Failed to stat dream source file: {}", path.display()))?;
        let relative = path
            .strip_prefix(workspace_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((
            metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
            relative,
        ));
    }

    files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    files.truncate(limit);
    Ok(files.into_iter().map(|(_, path)| path).collect())
}

/// List recent session segments, newest first.
fn list_recent_session_segments(
    sessions_dir: &Path,
    limit: usize,
    workspace_root: &Path,
) -> anyhow::Result<Vec<String>> {
    if !sessions_dir.is_dir() || limit == 0 {
        return Ok(Vec::new());
    }

    let mut entries = Vec::<(std::time::SystemTime, String)>::new();
    for entry in WalkDir::new(sessions_dir) {
        let entry = entry
            .with_context(|| format!("Failed to walk sessions dir: {}", sessions_dir.display()))?;
        let path = entry.path().to_path_buf();
        let is_jsonl = path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("jsonl"));
        if !is_jsonl || !path.is_file() {
            continue;
        }

        if !session_segment_contains_history(&path)? {
            continue;
        }
        let metadata = entry
            .metadata()
            .with_context(|| format!("Failed to stat session segment: {}", path.display()))?;
        let relative = path
            .strip_prefix(workspace_root)
            .with_context(|| {
                format!(
                    "Session segment resolved outside workspace root: {}",
                    path.display()
                )
            })?
            .to_string_lossy()
            .replace('\\', "/");
        entries.push((
            metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
            relative,
        ));
    }

    entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    entries.truncate(limit);
    Ok(entries.into_iter().map(|(_, path)| path).collect())
}

/// Return `true` when a session segment contains something beyond the mandatory
/// bootstrap metadata line.
fn session_segment_contains_history(path: &Path) -> anyhow::Result<bool> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open session segment: {}", path.display()))?;
    let mut non_empty_lines = 0usize;
    for line in BufReader::new(file).lines() {
        let line =
            line.with_context(|| format!("Failed to read session segment: {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }

        non_empty_lines += 1;
        if non_empty_lines >= 2 {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionStore;
    use chrono::Timelike as _;
    use tempfile::TempDir;

    /// Build one isolated temporary workspace for dream tests.
    fn temp_workspace() -> TempDir {
        TempDir::new().expect("temp workspace should be created")
    }

    /// Create a dream manager rooted at a temporary workspace.
    fn manager(workspace: &TempDir) -> DreamManager {
        DreamManager::new(workspace.path().to_path_buf(), DreamConfig::default())
            .expect("dream manager should build")
    }

    #[test]
    fn next_run_after_advances_to_next_local_midnight() {
        let workspace = temp_workspace();
        let manager = manager(&workspace);
        let now = resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 13).expect("valid date"))
            + Duration::hours(14);

        let next = manager.next_run_after(now);

        assert_eq!(
            next.date_naive(),
            NaiveDate::from_ymd_opt(2026, 4, 14).unwrap()
        );
        assert_eq!(next.time().hour(), 0);
        assert_eq!(next.time().minute(), 0);
    }

    #[test]
    fn should_run_now_is_false_after_same_day_completion() {
        let workspace = temp_workspace();
        let manager = manager(&workspace);
        let now =
            resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 13).unwrap()) + Duration::hours(9);

        manager
            .mark_completed(now.date_naive(), now, "memory/dreams/2026-04-13.md")
            .unwrap();

        assert!(!manager.should_run_now(now).unwrap());
    }

    #[test]
    fn should_run_now_is_false_for_first_start_with_only_bootstrap_session() {
        let workspace = temp_workspace();
        let manager = manager(&workspace);
        let now =
            resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 13).unwrap()) + Duration::hours(9);

        let store = SessionStore::new(workspace.path().to_path_buf())
            .expect("bootstrap session store should be created");
        let bootstrap_session = workspace.path().join(store.current_session_path());
        assert!(bootstrap_session.is_file());

        assert!(!manager.should_run_now(now).unwrap());
    }

    #[test]
    fn should_run_now_is_true_when_prior_session_history_exists() {
        let workspace = temp_workspace();
        fs::create_dir_all(workspace.path().join(DREAM_SESSIONS_DIR)).unwrap();
        fs::write(
            workspace
                .path()
                .join(DREAM_SESSIONS_DIR)
                .join("session-2026-04-12-history.jsonl"),
            "{\"type\":\"session_meta\"}\n{\"type\":\"message\",\"role\":\"user\",\"content\":\"history\"}\n",
        )
        .unwrap();
        let manager = manager(&workspace);
        let now =
            resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 13).unwrap()) + Duration::hours(9);

        assert!(manager.should_run_now(now).unwrap());
    }

    #[test]
    fn should_run_now_is_true_the_day_after_a_successful_run() {
        let workspace = temp_workspace();
        let manager = manager(&workspace);
        let yesterday =
            resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 12).unwrap()) + Duration::hours(23);
        let today =
            resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 13).unwrap()) + Duration::hours(1);

        manager
            .mark_completed(
                yesterday.date_naive(),
                yesterday,
                "memory/dreams/2026-04-12.md",
            )
            .unwrap();

        assert!(manager.should_run_now(today).unwrap());
    }

    #[test]
    fn dream_lock_allows_only_one_live_holder() {
        let workspace = temp_workspace();
        let manager = manager(&workspace);

        let first = manager
            .try_acquire_lock()
            .expect("lock attempt should succeed");
        assert!(first.is_some());
        let second = manager
            .try_acquire_lock()
            .expect("second lock attempt should not error");
        assert!(second.is_none());
    }

    #[test]
    fn prepare_run_collects_expected_sources_and_report_path() {
        let workspace = temp_workspace();
        fs::create_dir_all(workspace.path().join(DREAM_TOPIC_MEMORY_DIR)).unwrap();
        fs::create_dir_all(workspace.path().join("memory")).unwrap();
        fs::create_dir_all(workspace.path().join(DREAM_SESSIONS_DIR)).unwrap();
        fs::write(workspace.path().join("MEMORY.md"), "long-term memory").unwrap();
        fs::write(
            workspace
                .path()
                .join(DREAM_TOPIC_MEMORY_DIR)
                .join("preferences.md"),
            "topic memory",
        )
        .unwrap();
        fs::write(
            workspace.path().join("memory").join("2026-04-13.md"),
            "daily note",
        )
        .unwrap();
        fs::write(
            workspace
                .path()
                .join(DREAM_SESSIONS_DIR)
                .join("session-2026-04-13-a.jsonl"),
            "{\"type\":\"session_meta\"}\n{\"type\":\"message\",\"role\":\"user\",\"content\":\"hello\"}\n",
        )
        .unwrap();
        fs::create_dir_all(
            workspace
                .path()
                .join(DREAM_SESSIONS_DIR)
                .join("agents")
                .join("child-1"),
        )
        .unwrap();
        fs::write(
            workspace
                .path()
                .join(DREAM_SESSIONS_DIR)
                .join("agents")
                .join("child-1")
                .join("session-2026-04-13-b.jsonl"),
            "{\"type\":\"session_meta\"}\n{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"child work\"}\n",
        )
        .unwrap();

        let manager = manager(&workspace);
        let now =
            resolve_local_time(NaiveDate::from_ymd_opt(2026, 4, 13).unwrap()) + Duration::hours(8);
        let prepared = manager.prepare_run(now).expect("dream run should prepare");

        assert_eq!(prepared.report_relative_path, "memory/dreams/2026-04-13.md");
        assert_eq!(prepared.sources.memory_index.as_deref(), Some("MEMORY.md"));
        assert!(
            prepared
                .sources
                .topic_files
                .iter()
                .any(|path| path == "memory/topics/preferences.md")
        );
        assert!(
            prepared
                .sources
                .daily_notes
                .iter()
                .any(|path| path == "memory/2026-04-13.md")
        );
        assert!(
            prepared
                .sources
                .session_segments
                .iter()
                .any(|path| path == "sessions/session-2026-04-13-a.jsonl")
        );
        assert!(
            prepared
                .sources
                .session_segments
                .iter()
                .any(|path| path == "sessions/agents/child-1/session-2026-04-13-b.jsonl")
        );
        assert!(
            prepared
                .extra_system_prompt
                .contains("去噪、去重、修正冲突")
        );
        assert!(
            prepared
                .extra_system_prompt
                .contains("memory/dreams/2026-04-13.md")
        );
    }
}
