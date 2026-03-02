//! Skills discovery and loading (`SKILL.md`).
//!
//! The user request asks that the minimal agent can:
//! - Load "skills" (each skill is a directory containing a `SKILL.md`)
//! - Provide the skill instructions to the agent when needed
//!
//! We intentionally implement the same *shape* as Codex/Agents skills:
//! - `SKILL.md` starts with YAML frontmatter:
//!   - `name: ...`
//!   - `description: ...`
//! - The Markdown body contains the actual instructions.
//!
//! This module:
//! - Scans configured directories recursively for `SKILL.md`.
//! - Parses YAML frontmatter for metadata (name + description).
//! - Keeps an in-memory registry keyed by skill name.

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// YAML frontmatter at the top of `SKILL.md`.
///
/// We only care about `name` and `description` because they are the minimum
/// metadata needed for skill discovery and selection.
#[derive(Debug, Clone, Deserialize)]
struct SkillFrontmatter {
    /// Unique skill identifier (example: `skill-creator`).
    name: String,
    /// Human-readable description used for selection / triggering.
    description: String,
}

/// A single discovered skill.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Skill name (from frontmatter).
    pub name: String,
    /// Skill description (from frontmatter).
    pub description: String,
    /// Directory that contains the `SKILL.md`.
    pub dir: PathBuf,
    /// Full path to `SKILL.md`.
    pub skill_md_path: PathBuf,
}

/// Serializable subset of `Skill` used for tool output / prompt injection.
#[derive(Debug, Clone, Serialize)]
pub struct SkillListItem {
    /// Skill name.
    pub name: String,
    /// Skill description.
    pub description: String,
    /// Skill directory (string form for JSON).
    pub dir: String,
}

/// Registry of all discovered skills.
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    /// Map name -> skill.
    by_name: HashMap<String, Skill>,
}

impl SkillRegistry {
    /// Scan the provided directories recursively and build a skill registry.
    pub fn scan(dirs: &[PathBuf]) -> anyhow::Result<Self> {
        let mut registry = SkillRegistry::default();

        for dir in dirs {
            // If a directory does not exist, we skip it silently; this makes
            // configs portable between machines.
            if !dir.exists() {
                continue;
            }

            // Walk the directory tree looking for `SKILL.md`.
            for entry in walkdir::WalkDir::new(dir).follow_links(true) {
                let entry = entry?;

                // We only care about files.
                if !entry.file_type().is_file() {
                    continue;
                }

                // The canonical skill marker file.
                if entry.file_name() != "SKILL.md" {
                    continue;
                }

                let skill_md_path = entry.path().to_path_buf();
                let dir = skill_md_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));

                // Parse metadata from the file.
                let (name, description) = parse_skill_metadata(&skill_md_path)
                    .with_context(|| format!("Failed to parse {}", skill_md_path.display()))?;

                // Keep the first instance if duplicates exist; duplicates are
                // common when users have multiple skill folders.
                registry.by_name.entry(name.clone()).or_insert(Skill {
                    name,
                    description,
                    dir,
                    skill_md_path,
                });
            }
        }

        Ok(registry)
    }

    /// Return a stable, sorted list of skill metadata.
    pub fn list(&self) -> Vec<SkillListItem> {
        let mut items: Vec<_> = self
            .by_name
            .values()
            .map(|s| SkillListItem {
                name: s.name.clone(),
                description: s.description.clone(),
                dir: s.dir.display().to_string(),
            })
            .collect();

        // Stable order helps debugging and keeps prompts deterministic.
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }

    /// Load the full `SKILL.md` file for a skill by name.
    pub async fn load_skill_md(&self, name: &str) -> anyhow::Result<String> {
        let Some(skill) = self.by_name.get(name) else {
            anyhow::bail!("Skill not found: {name}");
        };

        tokio::fs::read_to_string(&skill.skill_md_path)
            .await
            .with_context(|| format!("Failed to read {}", skill.skill_md_path.display()))
    }

    /// Get a reference to a `Skill` by name (if present).
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.by_name.get(name)
    }
}

/// Parse `name` and `description` from a `SKILL.md`.
fn parse_skill_metadata(skill_md_path: &Path) -> anyhow::Result<(String, String)> {
    let raw = std::fs::read_to_string(skill_md_path)?;

    // We expect YAML frontmatter like:
    // ---
    // name: foo
    // description: bar
    // ---
    // (markdown body...)
    let Some(frontmatter) = extract_yaml_frontmatter(&raw) else {
        anyhow::bail!("SKILL.md missing YAML frontmatter (expected leading '---' block)");
    };

    let parsed: SkillFrontmatter =
        serde_yaml::from_str(frontmatter).context("Failed to parse YAML frontmatter")?;

    Ok((parsed.name, parsed.description))
}

/// Extract YAML frontmatter from the top of a Markdown file.
///
/// Returns the YAML string inside the `---` fence.
fn extract_yaml_frontmatter(raw: &str) -> Option<&str> {
    // Must start with `---` (optionally preceded by UTF-8 BOM, but we ignore BOM handling here).
    let raw = raw.strip_prefix("---\n")?;

    // Find the closing `---` fence.
    let end = raw.find("\n---\n")?;

    // Return YAML section (without fences).
    Some(&raw[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_yaml_frontmatter_happy_path() {
        let md = "---\nname: a\ndescription: b\n---\n\n# Body\n";
        let yaml = extract_yaml_frontmatter(md).expect("frontmatter");
        assert!(yaml.contains("name: a"));
        assert!(yaml.contains("description: b"));
    }

    #[test]
    fn extract_yaml_frontmatter_missing() {
        let md = "# No frontmatter\n";
        assert!(extract_yaml_frontmatter(md).is_none());
    }
}

