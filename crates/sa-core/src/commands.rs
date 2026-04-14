//! Unified command/skill registry for SA.
//!
//! This module upgrades SA's historical local-skill index into a richer
//! command layer that can hold:
//! - local `SKILL.md` commands
//! - bundled commands compiled into the binary
//! - MCP prompts exposed as command-like entries

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Maximum local skill file size loaded into memory.
const MAX_COMMAND_FILE_BYTES: u64 = 200_000;

/// One command source category.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandSource {
    /// Local `SKILL.md` discovered from configured skill directories.
    LocalSkill,
    /// Bundled command compiled into the binary.
    Bundled,
    /// Prompt advertised by an MCP server.
    McpPrompt,
}

/// How one command should execute when invoked.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CommandExecutionContext {
    /// Inject the expanded command content back into the current conversation.
    #[default]
    Inline,
    /// Run the command in a forked/nested durable sub-agent.
    Fork,
}

/// Minimal serializable metadata exposed to prompts and tool output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommandListItem {
    /// Stable command name.
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Optional "when to use" guidance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub when_to_use: Option<String>,
    /// Where this command came from.
    pub source: CommandSource,
}

/// One expanded command invocation ready for runtime use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedCommand {
    /// Stable command name.
    pub name: String,
    /// Source category for auditing.
    pub source: CommandSource,
    /// Whether the command should run inline or in a fork/sub-agent.
    pub execution_context: CommandExecutionContext,
    /// Expanded instruction body for the runtime.
    pub instructions: String,
    /// Optional tool allowlist patterns active while the command remains in
    /// scope.
    pub allowed_tools: Vec<String>,
    /// Optional model override for subsequent turns.
    pub model_override: Option<String>,
    /// Optional reasoning effort override for subsequent turns.
    pub effort_override: Option<String>,
}

/// Runtime reminder record persisted in durable agent state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveCommandInvocation {
    /// Stable command name.
    pub name: String,
    /// One-line description copied at invocation time.
    pub description: String,
    /// Optional "when to use" guidance copied at invocation time.
    pub when_to_use: Option<String>,
    /// Source category.
    pub source: CommandSource,
    /// Execution mode used for this invocation.
    pub execution_context: CommandExecutionContext,
    /// Allowed-tool patterns associated with this invocation.
    pub allowed_tools: Vec<String>,
    /// Optional model override associated with this invocation.
    pub model_override: Option<String>,
    /// Optional reasoning-effort override associated with this invocation.
    pub effort_override: Option<String>,
}

/// One MCP prompt command descriptor injected into the registry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpPromptCommand {
    /// Model-visible command name, for example `mcp__docs__summarize`.
    pub name: String,
    /// Original MCP server name.
    pub server_name: String,
    /// Original prompt name understood by the MCP server.
    pub prompt_name: String,
    /// Optional description advertised by the server.
    pub description: String,
    /// Declared argument names, in order.
    pub arguments: Vec<String>,
}

/// YAML frontmatter wrapper for local/bundled command markdown.
#[derive(Debug, Clone, Deserialize, Default)]
struct CommandFrontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(rename = "allowed-tools")]
    allowed_tools: Option<StringOrVec>,
    when_to_use: Option<String>,
    #[serde(rename = "argument-hint")]
    argument_hint: Option<String>,
    arguments: Option<StringOrVec>,
    #[serde(rename = "disable-model-invocation")]
    disable_model_invocation: Option<bool>,
    context: Option<String>,
    model: Option<String>,
    effort: Option<String>,
    paths: Option<StringOrVec>,
    hooks: Option<Value>,
    shell: Option<String>,
}

/// Helper that accepts either one string or a list of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    /// One scalar string field.
    One(String),
    /// Explicit YAML array.
    Many(Vec<String>),
}

impl StringOrVec {
    /// Flatten the field into a normalized string vector.
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(raw) => split_field_list(&raw),
            Self::Many(items) => items
                .into_iter()
                .flat_map(|item| split_field_list(&item))
                .collect(),
        }
    }
}

/// Content backing for one command.
#[derive(Debug, Clone)]
enum CommandContent {
    /// Local or bundled markdown with optional filesystem root.
    Markdown {
        /// Root directory used for relative file reads and `${SA_SKILL_DIR}`.
        root_dir: Option<PathBuf>,
        /// Main command markdown path, when it exists on disk.
        main_file_path: Option<PathBuf>,
        /// Raw markdown body without frontmatter.
        body: String,
    },
    /// MCP prompt loaded dynamically from a remote server.
    McpPrompt(McpPromptCommand),
}

/// Rich local command/skill definition.
#[derive(Debug, Clone)]
pub struct CommandSpec {
    /// Stable command name.
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Optional "when to use" guidance for model selection.
    pub when_to_use: Option<String>,
    /// Optional argument hint shown to human authors and future UIs.
    pub argument_hint: Option<String>,
    /// Ordered argument names used for substitution.
    pub arguments: Vec<String>,
    /// Optional tool allowlist patterns.
    pub allowed_tools: Vec<String>,
    /// Whether the model may invoke this command automatically.
    pub disable_model_invocation: bool,
    /// Inline vs fork execution.
    pub execution_context: CommandExecutionContext,
    /// Optional model override.
    pub model: Option<String>,
    /// Optional effort override.
    pub effort: Option<String>,
    /// Optional conditional activation path patterns.
    pub paths: Vec<String>,
    /// Optional hooks payload preserved for future policy integrations.
    pub hooks: Option<Value>,
    /// Optional shell hint preserved for future prompt-shell execution.
    pub shell: Option<String>,
    /// Backing source category.
    pub source: CommandSource,
    /// Backing content loader.
    content: CommandContent,
}

impl CommandSpec {
    /// Convert this command into lightweight prompt metadata.
    pub fn list_item(&self) -> CommandListItem {
        CommandListItem {
            name: self.name.clone(),
            description: self.description.clone(),
            when_to_use: self.when_to_use.clone(),
            source: self.source,
        }
    }

    /// Whether this command is local/bundled and therefore supports safe file
    /// reads through the `Skill(action="read")` mode.
    pub fn supports_local_file_read(&self) -> bool {
        matches!(self.content, CommandContent::Markdown { .. })
    }

    /// Expand one inline/fork invocation into concrete instruction text.
    pub fn expand_invocation(
        &self,
        raw_args: Option<&str>,
        session_id: &str,
    ) -> anyhow::Result<ExpandedCommand> {
        let instructions = match &self.content {
            CommandContent::Markdown { root_dir, body, .. } => expand_command_body(
                body,
                &self.arguments,
                raw_args.unwrap_or_default(),
                root_dir.as_deref(),
                session_id,
            )?,
            CommandContent::McpPrompt(prompt) => {
                anyhow::bail!(
                    "MCP prompt command `{}` must be expanded by the MCP runtime",
                    prompt.name
                );
            }
        };

        Ok(ExpandedCommand {
            name: self.name.clone(),
            source: self.source,
            execution_context: self.execution_context,
            instructions,
            allowed_tools: self.allowed_tools.clone(),
            model_override: self.model.clone(),
            effort_override: self.effort.clone(),
        })
    }

    /// Build one durable reminder payload stored in agent state.
    pub fn reminder(&self) -> ActiveCommandInvocation {
        ActiveCommandInvocation {
            name: self.name.clone(),
            description: self.description.clone(),
            when_to_use: self.when_to_use.clone(),
            source: self.source,
            execution_context: self.execution_context,
            allowed_tools: self.allowed_tools.clone(),
            model_override: self.model.clone(),
            effort_override: self.effort.clone(),
        }
    }

    /// Return the stored MCP prompt descriptor when this command came from an
    /// MCP prompt.
    pub fn mcp_prompt(&self) -> Option<&McpPromptCommand> {
        match &self.content {
            CommandContent::McpPrompt(prompt) => Some(prompt),
            CommandContent::Markdown { .. } => None,
        }
    }
}

/// Unified registry of model-visible commands.
#[derive(Debug, Clone, Default)]
pub struct CommandRegistry {
    /// Stable map keyed by command name.
    by_name: HashMap<String, CommandSpec>,
}

impl CommandRegistry {
    /// Scan the provided skill directories recursively and load local commands.
    pub fn scan(dirs: &[PathBuf]) -> anyhow::Result<Self> {
        let mut registry = Self::default();

        for dir in dirs {
            if !dir.exists() {
                continue;
            }

            let scan_root = std::fs::canonicalize(dir).with_context(|| {
                format!("Failed to canonicalize skill scan dir {}", dir.display())
            })?;

            for entry in walkdir::WalkDir::new(&scan_root).follow_links(true) {
                let entry = entry?;
                if !entry.file_type().is_file() || entry.file_name() != "SKILL.md" {
                    continue;
                }

                let logical_skill_dir = entry
                    .path()
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("SKILL.md has no parent directory"))?
                    .strip_prefix(&scan_root)
                    .ok()
                    .map(Path::to_path_buf);
                let skill_md_path = std::fs::canonicalize(entry.path()).with_context(|| {
                    format!("Failed to canonicalize {}", entry.path().display())
                })?;
                let root_dir = skill_md_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));
                let command = load_markdown_command(
                    &skill_md_path,
                    Some(root_dir),
                    CommandSource::LocalSkill,
                    logical_skill_dir.as_deref(),
                )
                .with_context(|| format!("Failed to load {}", skill_md_path.display()))?;
                registry
                    .by_name
                    .entry(command.name.clone())
                    .or_insert(command);
            }
        }

        Ok(registry)
    }

    /// Merge bundled markdown commands into this registry.
    pub fn extend_bundled(&mut self, commands: impl IntoIterator<Item = CommandSpec>) {
        for command in commands {
            self.by_name.entry(command.name.clone()).or_insert(command);
        }
    }

    /// Merge MCP prompt commands into this registry.
    pub fn extend_mcp_prompts(&mut self, prompts: impl IntoIterator<Item = McpPromptCommand>) {
        for prompt in prompts {
            let name = prompt.name.clone();
            self.by_name.entry(name.clone()).or_insert(CommandSpec {
                name,
                description: prompt.description.clone(),
                when_to_use: None,
                argument_hint: None,
                arguments: prompt.arguments.clone(),
                allowed_tools: Vec::new(),
                disable_model_invocation: false,
                execution_context: CommandExecutionContext::Inline,
                model: None,
                effort: None,
                paths: Vec::new(),
                hooks: None,
                shell: None,
                source: CommandSource::McpPrompt,
                content: CommandContent::McpPrompt(prompt),
            });
        }
    }

    /// Return all registered commands in stable order.
    pub fn list(&self) -> Vec<CommandListItem> {
        let mut items: Vec<_> = self.by_name.values().map(CommandSpec::list_item).collect();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }

    /// Return model-visible commands in stable order.
    ///
    /// Commands gated by `paths` only appear once their names are present in
    /// `activated_conditional`.
    pub fn list_model_invocable<'a>(
        &self,
        activated_conditional: impl IntoIterator<Item = &'a str>,
        denied_commands: &[String],
    ) -> Vec<CommandListItem> {
        let activated: BTreeSet<&str> = activated_conditional.into_iter().collect();
        let mut items: Vec<_> = self
            .by_name
            .values()
            .filter(|command| !command.disable_model_invocation)
            .filter(|command| command.paths.is_empty() || activated.contains(command.name.as_str()))
            .filter(|command| !matches_command_patterns(&command.name, denied_commands))
            .map(CommandSpec::list_item)
            .collect();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }

    /// Lookup one command by name.
    pub fn get(&self, name: &str) -> Option<&CommandSpec> {
        self.by_name.get(name)
    }

    /// Safely read one local skill-relative file.
    ///
    /// This powers `Skill(action="read")`.
    pub async fn load_skill_file(
        &self,
        name: &str,
        path: Option<&str>,
    ) -> anyhow::Result<(String, String)> {
        let Some(command) = self.by_name.get(name) else {
            anyhow::bail!("Skill not found: {name}");
        };
        if !command.supports_local_file_read() {
            anyhow::bail!("Command `{name}` does not expose local files");
        }

        let requested = normalize_relative_command_path(path.unwrap_or("SKILL.md"))?;
        let (root_dir, main_file_path) = match &command.content {
            CommandContent::Markdown {
                root_dir,
                main_file_path,
                ..
            } => (root_dir.clone(), main_file_path.clone()),
            CommandContent::McpPrompt(_) => unreachable!("checked above"),
        };

        let Some(root_dir) = root_dir else {
            anyhow::bail!("Command `{name}` has no local root directory");
        };

        let resolved = if requested == "SKILL.md" {
            if let Some(main_file_path) = main_file_path {
                main_file_path
            } else {
                anyhow::bail!("Command `{name}` has no on-disk SKILL.md");
            }
        } else {
            let candidate = root_dir.join(&requested);
            tokio::fs::canonicalize(&candidate)
                .await
                .with_context(|| format!("Skill file not found: {requested}"))?
        };

        if !resolved.starts_with(&root_dir) {
            anyhow::bail!("Skill path escapes the skill root: {requested}");
        }

        let meta = tokio::fs::metadata(&resolved)
            .await
            .with_context(|| format!("Failed to stat skill file: {requested}"))?;
        if !meta.is_file() {
            anyhow::bail!("Skill path must point to a file: {requested}");
        }
        if meta.len() > MAX_COMMAND_FILE_BYTES {
            anyhow::bail!(
                "Skill file is too large to load ({} bytes): {}",
                meta.len(),
                requested
            );
        }

        let content = tokio::fs::read_to_string(&resolved)
            .await
            .with_context(|| format!("Failed to read skill file: {requested}"))?;
        Ok((requested, content))
    }

    /// Return newly matching conditional commands for the touched paths.
    pub fn conditional_matches_for_paths(
        &self,
        workspace_root: &Path,
        touched_paths: &[PathBuf],
        already_activated: &BTreeSet<String>,
    ) -> Vec<String> {
        let mut matches = Vec::new();

        for command in self.by_name.values() {
            if command.paths.is_empty() || already_activated.contains(&command.name) {
                continue;
            }
            if touched_paths
                .iter()
                .any(|path| command_matches_path_patterns(command, workspace_root, path))
            {
                matches.push(command.name.clone());
            }
        }

        matches.sort();
        matches
    }
}

/// Load one local or bundled markdown command.
fn load_markdown_command(
    markdown_path: &Path,
    root_dir: Option<PathBuf>,
    source: CommandSource,
    logical_skill_dir: Option<&Path>,
) -> anyhow::Result<CommandSpec> {
    let raw = std::fs::read_to_string(markdown_path)?;
    let Some(frontmatter_raw) = extract_yaml_frontmatter(&raw) else {
        anyhow::bail!("SKILL.md missing YAML frontmatter (expected leading `---` block)");
    };
    let frontmatter: CommandFrontmatter =
        serde_yaml::from_str(&frontmatter_raw).context("Failed to parse YAML frontmatter")?;
    let body = strip_yaml_frontmatter(&raw)
        .ok_or_else(|| anyhow::anyhow!("SKILL.md missing markdown body"))?;

    let raw_name = frontmatter
        .name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Command frontmatter must define `name`"))?;
    let name = derive_command_name(raw_name, logical_skill_dir)?;
    let description = frontmatter
        .description
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Command `{name}` must define `description`"))?
        .to_string();
    let execution_context = match frontmatter
        .context
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some("fork") => CommandExecutionContext::Fork,
        _ => CommandExecutionContext::Inline,
    };

    Ok(CommandSpec {
        name,
        description,
        when_to_use: normalize_optional(frontmatter.when_to_use),
        argument_hint: normalize_optional(frontmatter.argument_hint),
        arguments: frontmatter
            .arguments
            .map(StringOrVec::into_vec)
            .unwrap_or_default(),
        allowed_tools: frontmatter
            .allowed_tools
            .map(StringOrVec::into_vec)
            .unwrap_or_default(),
        disable_model_invocation: frontmatter.disable_model_invocation.unwrap_or(false),
        execution_context,
        model: normalize_optional(frontmatter.model),
        effort: normalize_optional(frontmatter.effort),
        paths: frontmatter
            .paths
            .map(StringOrVec::into_vec)
            .unwrap_or_default(),
        hooks: frontmatter.hooks,
        shell: normalize_optional(frontmatter.shell),
        source,
        content: CommandContent::Markdown {
            root_dir,
            main_file_path: Some(markdown_path.to_path_buf()),
            body,
        },
    })
}

/// Expand one markdown command body with argument and runtime substitutions.
fn expand_command_body(
    body: &str,
    argument_names: &[String],
    raw_args: &str,
    root_dir: Option<&Path>,
    session_id: &str,
) -> anyhow::Result<String> {
    let raw_args = raw_args.trim();
    let parsed_args = shlex::split(raw_args)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse command arguments"))?;
    let mut out = body.to_string();

    // Replace the most specific placeholders first.
    for (index, value) in parsed_args.iter().enumerate() {
        out = out.replace(&format!("$ARGUMENTS[{index}]"), value);
        out = out.replace(&format!("${index}"), value);
    }
    for (index, argument_name) in argument_names.iter().enumerate() {
        let value = parsed_args.get(index).cloned().unwrap_or_default();
        out = out.replace(&format!("${argument_name}"), &value);
    }
    out = out.replace("$ARGUMENTS", raw_args);
    out = out.replace("${SA_SESSION_ID}", session_id);
    if let Some(root_dir) = root_dir {
        out = out.replace("${SA_SKILL_DIR}", &root_dir.display().to_string());
    }

    Ok(out)
}

/// Derive the canonical command name used inside SA.
///
/// Naming rules:
/// - flat skills keep their existing leaf name, for example `brainstorming`
/// - nested skill directories become colon-qualified, for example
///   `superpowers/brainstorming/SKILL.md` => `superpowers:brainstorming`
/// - if frontmatter already provides an explicit colon-qualified name, keep it
fn derive_command_name(raw_name: &str, logical_skill_dir: Option<&Path>) -> anyhow::Result<String> {
    let raw_name = raw_name.trim();
    if raw_name.is_empty() {
        anyhow::bail!("Command name must not be empty");
    }

    if raw_name.contains(':') {
        let explicit = raw_name
            .split(':')
            .map(normalize_command_name_segment)
            .collect::<anyhow::Result<Vec<_>>>()?;
        return Ok(explicit.join(":"));
    }

    let mut parts = logical_skill_dir
        .and_then(Path::parent)
        .map(command_namespace_parts_from_path)
        .transpose()?
        .unwrap_or_default();
    parts.push(normalize_command_name_segment(raw_name)?);
    Ok(parts.join(":"))
}

/// Convert a relative skill parent directory into colon namespace segments.
fn command_namespace_parts_from_path(path: &Path) -> anyhow::Result<Vec<String>> {
    path.components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part.to_string_lossy().to_string()),
            _ => None,
        })
        .map(|segment| normalize_command_name_segment(&segment))
        .collect()
}

/// Extract YAML frontmatter from the top of a markdown file.
fn extract_yaml_frontmatter(raw: &str) -> Option<String> {
    let mut lines = raw.lines();
    if lines.next()? != "---" {
        return None;
    }

    let mut yaml = String::new();
    for line in lines {
        if line == "---" {
            return Some(yaml);
        }
        yaml.push_str(line);
        yaml.push('\n');
    }
    None
}

/// Strip the leading YAML frontmatter fence and return the markdown body.
fn strip_yaml_frontmatter(raw: &str) -> Option<String> {
    let mut lines = raw.lines();
    if lines.next()? != "---" {
        return None;
    }

    let mut found_end = false;
    let mut body = String::new();
    for line in lines {
        if !found_end {
            if line == "---" {
                found_end = true;
            }
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }

    found_end.then_some(body.trim_start_matches('\n').to_string())
}

/// Normalize one optional string field.
fn normalize_optional(value: Option<String>) -> Option<String> {
    value
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
}

/// Normalize one command-name segment used inside a colon-qualified command
/// name such as `superpowers:brainstorming`.
fn normalize_command_name_segment(raw: &str) -> anyhow::Result<String> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        anyhow::bail!("Command name segment must not be empty");
    }
    if normalized.contains('/') || normalized.contains('\\') {
        anyhow::bail!("Command name segment must not contain path separators");
    }
    if normalized.contains(':') {
        anyhow::bail!("Command name segment must not contain `:`");
    }

    Ok(normalized.to_string())
}

/// Normalize one skill-relative path and reject traversal.
fn normalize_relative_command_path(raw: &str) -> anyhow::Result<String> {
    let normalized = raw
        .trim()
        .trim_start_matches("./")
        .trim_start_matches(".\\")
        .replace('\\', "/");

    if normalized.is_empty() {
        anyhow::bail!("Skill path must not be empty");
    }
    if Path::new(&normalized).is_absolute() {
        anyhow::bail!("Skill path must be relative, not absolute");
    }
    if normalized.split('/').any(|part| part == "..") {
        anyhow::bail!("Skill path must not contain `..`");
    }

    Ok(normalized)
}

/// Split one scalar frontmatter field into normalized items.
fn split_field_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .flat_map(|part| part.lines())
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// Match command-name patterns used by deny lists.
pub fn matches_command_patterns(name: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.is_empty())
        .any(|pattern| wildcard_match(name, pattern))
}

/// Match a touched workspace-relative path against the command's `paths`
/// patterns.
fn command_matches_path_patterns(
    command: &CommandSpec,
    workspace_root: &Path,
    touched_path: &Path,
) -> bool {
    let Ok(relative) = touched_path.strip_prefix(workspace_root) else {
        return false;
    };
    let relative = relative
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");

    command.paths.iter().any(|pattern| {
        let normalized = pattern
            .trim()
            .trim_start_matches("./")
            .trim_start_matches(".\\")
            .replace('\\', "/");
        if normalized == "**" {
            return true;
        }
        let prefix = normalized.strip_suffix("/**").unwrap_or(&normalized);
        relative == prefix || relative.starts_with(&format!("{prefix}/"))
    })
}

/// Simple wildcard matcher used for deny rules.
///
/// This intentionally supports only the pattern shapes SA currently needs:
/// - exact matches
/// - `*` suffix/prefix/infix wildcards
fn wildcard_match(value: &str, pattern: &str) -> bool {
    if !pattern.contains('*') {
        return value == pattern;
    }

    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return true;
    }

    let mut remaining = value;
    let mut first = true;
    for (index, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if first && !pattern.starts_with('*') {
            if !remaining.starts_with(part) {
                return false;
            }
            remaining = &remaining[part.len()..];
            first = false;
            continue;
        }

        if index == parts.len() - 1 && !pattern.ends_with('*') {
            return remaining.ends_with(part);
        }

        if let Some(position) = remaining.find(part) {
            remaining = &remaining[position + part.len()..];
        } else {
            return false;
        }
        first = false;
    }

    true
}

/// Backwards-compatible alias while the rest of SA still imports
/// `crate::skills::SkillRegistry`.
pub type SkillRegistry = CommandRegistry;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;

    /// Create one unique temp directory without introducing extra dev deps.
    fn unique_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sa-commands-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn local_skill_scan_preserves_rich_frontmatter() {
        let root = unique_temp_dir();
        let skill_dir = root.join("writer");
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: writer
description: Writes polished reports
allowed-tools:
  - Read
  - Bash(git:*)
when_to_use: Use when the user asks for a report or polished write-up.
argument-hint: "[topic]"
arguments:
  - topic
disable-model-invocation: false
context: fork
model: gpt-5.4
effort: high
paths:
  - docs/**
shell: bash
---

# Writer

Write about $topic from ${SA_SKILL_DIR} in session ${SA_SESSION_ID}.
"#,
        )
        .expect("write skill");

        let registry = CommandRegistry::scan(&[root]).expect("scan commands");
        let command = registry.get("writer").expect("command should exist");

        assert_eq!(command.description, "Writes polished reports");
        assert_eq!(
            command.when_to_use.as_deref(),
            Some("Use when the user asks for a report or polished write-up.")
        );
        assert_eq!(command.allowed_tools, vec!["Read", "Bash(git:*)"]);
        assert_eq!(command.argument_hint.as_deref(), Some("[topic]"));
        assert_eq!(command.arguments, vec!["topic"]);
        assert_eq!(command.execution_context, CommandExecutionContext::Fork);
        assert_eq!(command.model.as_deref(), Some("gpt-5.4"));
        assert_eq!(command.effort.as_deref(), Some("high"));
        assert_eq!(command.paths, vec!["docs/**"]);
        assert_eq!(command.shell.as_deref(), Some("bash"));
    }

    #[test]
    fn nested_skill_scan_uses_colon_namespaces_from_path() {
        let root = unique_temp_dir();
        let skill_dir = root.join("superpowers").join("brainstorming");
        fs::create_dir_all(&skill_dir).expect("create nested skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: brainstorming
description: Explores task design before implementation
---

# Brainstorming
"#,
        )
        .expect("write nested skill");

        let registry = CommandRegistry::scan(&[root]).expect("scan commands");
        assert!(registry.get("brainstorming").is_none());
        let command = registry
            .get("superpowers:brainstorming")
            .expect("nested skill should be namespaced");
        assert_eq!(
            command.description,
            "Explores task design before implementation"
        );
    }

    #[test]
    fn explicit_colon_name_in_frontmatter_is_preserved() {
        let root = unique_temp_dir();
        let skill_dir = root.join("superpowers").join("brainstorming");
        fs::create_dir_all(&skill_dir).expect("create nested skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: superpowers:brainstorming
description: Explicit namespaced skill
---

# Brainstorming
"#,
        )
        .expect("write nested skill");

        let registry = CommandRegistry::scan(&[root]).expect("scan commands");
        let command = registry
            .get("superpowers:brainstorming")
            .expect("explicit namespaced skill should load");
        assert_eq!(command.description, "Explicit namespaced skill");
    }

    #[test]
    fn model_invocable_listing_hides_disabled_and_unactivated_conditional_commands() {
        let mut registry = CommandRegistry::default();
        registry.by_name.insert(
            "always".to_string(),
            CommandSpec {
                name: "always".to_string(),
                description: "always visible".to_string(),
                when_to_use: None,
                argument_hint: None,
                arguments: Vec::new(),
                allowed_tools: Vec::new(),
                disable_model_invocation: false,
                execution_context: CommandExecutionContext::Inline,
                model: None,
                effort: None,
                paths: Vec::new(),
                hooks: None,
                shell: None,
                source: CommandSource::Bundled,
                content: CommandContent::Markdown {
                    root_dir: None,
                    main_file_path: None,
                    body: "body".to_string(),
                },
            },
        );
        registry.by_name.insert(
            "hidden".to_string(),
            CommandSpec {
                name: "hidden".to_string(),
                description: "hidden".to_string(),
                when_to_use: None,
                argument_hint: None,
                arguments: Vec::new(),
                allowed_tools: Vec::new(),
                disable_model_invocation: true,
                execution_context: CommandExecutionContext::Inline,
                model: None,
                effort: None,
                paths: Vec::new(),
                hooks: None,
                shell: None,
                source: CommandSource::Bundled,
                content: CommandContent::Markdown {
                    root_dir: None,
                    main_file_path: None,
                    body: "body".to_string(),
                },
            },
        );
        registry.by_name.insert(
            "conditional".to_string(),
            CommandSpec {
                name: "conditional".to_string(),
                description: "conditional".to_string(),
                when_to_use: None,
                argument_hint: None,
                arguments: Vec::new(),
                allowed_tools: Vec::new(),
                disable_model_invocation: false,
                execution_context: CommandExecutionContext::Inline,
                model: None,
                effort: None,
                paths: vec!["src/**".to_string()],
                hooks: None,
                shell: None,
                source: CommandSource::Bundled,
                content: CommandContent::Markdown {
                    root_dir: None,
                    main_file_path: None,
                    body: "body".to_string(),
                },
            },
        );

        let names: Vec<String> = registry
            .list_model_invocable(std::iter::empty::<&str>(), &[])
            .into_iter()
            .map(|item| item.name)
            .collect();
        assert_eq!(names, vec!["always".to_string()]);

        let names_after_activation: Vec<String> = registry
            .list_model_invocable(["conditional"].iter().copied(), &[])
            .into_iter()
            .map(|item| item.name)
            .collect();
        assert_eq!(
            names_after_activation,
            vec!["always".to_string(), "conditional".to_string()]
        );
    }

    #[test]
    fn expand_invocation_substitutes_arguments_and_runtime_variables() {
        let root = unique_temp_dir();
        let command = CommandSpec {
            name: "writer".to_string(),
            description: "writer".to_string(),
            when_to_use: None,
            argument_hint: Some("[topic]".to_string()),
            arguments: vec!["topic".to_string()],
            allowed_tools: vec!["Read".to_string()],
            disable_model_invocation: false,
            execution_context: CommandExecutionContext::Inline,
            model: Some("gpt-5.4".to_string()),
            effort: Some("high".to_string()),
            paths: Vec::new(),
            hooks: None,
            shell: None,
            source: CommandSource::Bundled,
            content: CommandContent::Markdown {
                root_dir: Some(root.clone()),
                main_file_path: None,
                body:
                    "Topic=$topic raw=$ARGUMENTS first=$0 dir=${SA_SKILL_DIR} sid=${SA_SESSION_ID}"
                        .to_string(),
            },
        };

        let expanded = command
            .expand_invocation(Some("study-notes"), "session-123")
            .expect("expand invocation");
        assert_eq!(expanded.execution_context, CommandExecutionContext::Inline);
        assert!(expanded.instructions.contains("Topic=study-notes"));
        assert!(expanded.instructions.contains("raw=study-notes"));
        assert!(expanded.instructions.contains("first=study-notes"));
        assert!(expanded.instructions.contains("sid=session-123"));
        assert!(
            expanded
                .instructions
                .contains(&format!("dir={}", root.display()))
        );
    }

    #[test]
    fn conditional_matches_detects_touched_workspace_paths() {
        let workspace_root = unique_temp_dir();
        let mut registry = CommandRegistry::default();
        registry.by_name.insert(
            "docs-helper".to_string(),
            CommandSpec {
                name: "docs-helper".to_string(),
                description: "doc helper".to_string(),
                when_to_use: None,
                argument_hint: None,
                arguments: Vec::new(),
                allowed_tools: Vec::new(),
                disable_model_invocation: false,
                execution_context: CommandExecutionContext::Inline,
                model: None,
                effort: None,
                paths: vec!["docs/**".to_string()],
                hooks: None,
                shell: None,
                source: CommandSource::Bundled,
                content: CommandContent::Markdown {
                    root_dir: None,
                    main_file_path: None,
                    body: "body".to_string(),
                },
            },
        );

        let touched = vec![workspace_root.join("docs/guide.md")];
        let matches =
            registry.conditional_matches_for_paths(&workspace_root, &touched, &BTreeSet::new());
        assert_eq!(matches, vec!["docs-helper".to_string()]);
    }

    #[test]
    fn mcp_prompts_are_registered_as_commands() {
        let mut registry = CommandRegistry::default();
        registry.extend_mcp_prompts([McpPromptCommand {
            name: "mcp__docs__summarize".to_string(),
            server_name: "docs".to_string(),
            prompt_name: "summarize".to_string(),
            description: "Summarize docs".to_string(),
            arguments: vec!["topic".to_string()],
        }]);

        let command = registry
            .get("mcp__docs__summarize")
            .expect("MCP prompt command should exist");
        assert_eq!(command.source, CommandSource::McpPrompt);
        assert_eq!(command.arguments, vec!["topic"]);
    }
}
