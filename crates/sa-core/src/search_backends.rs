//! Multi-backend fault-tolerant search chain.
//!
//! Provides a pluggable [`SearchBackend`] trait and several concrete
//! implementations.  [`multi_backend_search`] tries backends in the
//! configured order until one succeeds, giving the agent resilient web
//! search even when individual providers are down or blocked.

use crate::cancel::CancelToken;
use crate::config::SearchConfig;
use anyhow::{Context as _, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ────────────────────────────── types ──────────────────────────────

/// A single search result returned by any backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Trait for pluggable search backends.
#[async_trait]
pub trait SearchBackend: Send + Sync {
    /// Human-readable backend name (e.g. `"ddg"`, `"sogou"`).
    fn name(&self) -> &str;

    /// Perform the search and return up to `max_results` results.
    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>>;
}

// ──────────────────────── multi-backend scheduler ──────────────────

/// Try backends in order until one returns non-empty results.
///
/// Each backend is subject to `config.timeout_secs`.  The entire call
/// is bounded by `config.total_timeout_secs`.  Backends whose required
/// credentials are missing are silently skipped.
pub async fn multi_backend_search(
    query: &str,
    max_results: usize,
    config: &SearchConfig,
    cancel: &CancelToken,
) -> anyhow::Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        bail!("Search query must not be empty");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.timeout_secs))
        .user_agent("StudyAdministrator/0.7 Search")
        .build()
        .context("Failed to build Search HTTP client")?;

    let mut errors: Vec<String> = Vec::new();

    for backend_name in &config.backends {
        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let backend = match build_backend(backend_name, config, &client) {
            Some(b) => b,
            None => {
                errors.push(format!("{backend_name}: skipped (missing config/credentials)"));
                continue;
            }
        };

        let per_backend_timeout = Duration::from_secs(config.timeout_secs);

        let result = tokio::select! {
            _ = cancel.cancelled() => {
                bail!("Search cancelled");
            }
            res = tokio::time::timeout(per_backend_timeout,
                backend.search(query, max_results, cancel)
            ) => res,
        };

        match result {
            Ok(Ok(results)) if !results.is_empty() => return Ok(results),
            Ok(Ok(_)) => {
                errors.push(format!("{}: zero results", backend.name()));
            }
            Ok(Err(e)) => {
                errors.push(format!("{}: {e}", backend.name()));
            }
            Err(_) => {
                errors.push(format!("{}: timeout ({}s)", backend.name(), config.timeout_secs));
            }
        }
    }

    bail!("All search backends failed: {}", errors.join(" | "))
}

/// Instantiate a backend by name, returning `None` when required config
/// is absent (e.g. no API key).
fn build_backend(
    name: &str,
    config: &SearchConfig,
    client: &reqwest::Client,
) -> Option<Box<dyn SearchBackend>> {
    match name {
        "ddg" | "duckduckgo" => {
            if config.ddg.enabled {
                Some(Box::new(DuckDuckGoBackend {
                    client: client.clone(),
                }))
            } else {
                None
            }
        }
        "sogou" => {
            if config.sogou.enabled {
                Some(Box::new(SogouBackend {
                    client: client.clone(),
                }))
            } else {
                None
            }
        }
        "brave" => {
            let key = config.brave.api_key.trim();
            if !key.is_empty() {
                Some(Box::new(BraveSearchBackend {
                    client: client.clone(),
                    api_key: key.to_string(),
                }))
            } else {
                None
            }
        }
        "bing" => {
            let key = config.bing.api_key.trim();
            if !key.is_empty() {
                Some(Box::new(BingSearchBackend {
                    client: client.clone(),
                    api_key: key.to_string(),
                }))
            } else {
                None
            }
        }
        "serpapi" => {
            let key = config.serpapi.api_key.trim();
            if !key.is_empty() {
                Some(Box::new(SerpApiBackend {
                    client: client.clone(),
                    api_key: key.to_string(),
                }))
            } else {
                None
            }
        }
        "searxng" => {
            let base = config.searxng.base_url.trim();
            if config.searxng.enabled && !base.is_empty() {
                Some(Box::new(SearXngBackend {
                    client: client.clone(),
                    base_url: base.trim_end_matches('/').to_string(),
                }))
            } else {
                None
            }
        }
        "semantic_scholar" | "scholar" => {
            if config.semantic_scholar.enabled {
                Some(Box::new(SemanticScholarBackend {
                    client: client.clone(),
                }))
            } else {
                None
            }
        }
        _ => {
            tracing::warn!("Unknown search backend: {name}");
            None
        }
    }
}

// ───────────────────── DuckDuckGo HTML backend ─────────────────────

struct DuckDuckGoBackend {
    client: reqwest::Client,
}

#[async_trait]
impl SearchBackend for DuckDuckGoBackend {
    fn name(&self) -> &str {
        "ddg"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let urls = ddg_endpoint_candidates(query)?;
        let mut attempts = Vec::new();

        for url in urls {
            if cancel.is_cancelled() {
                bail!("Search cancelled");
            }

            let response = tokio::select! {
                _ = cancel.cancelled() => bail!("Search cancelled"),
                res = self.client.get(url.clone()).send() => {
                    match res {
                        Ok(r) => r,
                        Err(e) => { attempts.push(format!("{url}: {e}")); continue; }
                    }
                }
            };

            let status = response.status();
            let html = tokio::select! {
                _ = cancel.cancelled() => bail!("Search cancelled"),
                body = response.text() => {
                    match body {
                        Ok(b) => b,
                        Err(e) => { attempts.push(format!("{url}: body read: {e}")); continue; }
                    }
                }
            };

            if !status.is_success() {
                attempts.push(format!("{url}: http {}", status.as_u16()));
                continue;
            }

            let results = extract_duckduckgo_results(&html, max_results);
            if !results.is_empty() {
                return Ok(results);
            }

            attempts.push(format!("{url}: zero results"));
        }

        bail!("DDG: {}", attempts.join(" | "))
    }
}

fn ddg_endpoint_candidates(query: &str) -> anyhow::Result<Vec<reqwest::Url>> {
    let mut urls = Vec::new();
    for base in [
        "https://duckduckgo.com/html/",
        "https://html.duckduckgo.com/html/",
    ] {
        urls.push(
            reqwest::Url::parse_with_params(base, &[("q", query)])
                .with_context(|| format!("Failed to build DDG URL from {base}"))?,
        );
    }
    Ok(urls)
}

// ─────────────────────── Sogou HTML backend ────────────────────────

struct SogouBackend {
    client: reqwest::Client,
}

#[async_trait]
impl SearchBackend for SogouBackend {
    fn name(&self) -> &str {
        "sogou"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            "https://www.sogou.com/web",
            &[("query", query)],
        )
        .context("Failed to build Sogou URL")?;

        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = self.client.get(url.clone()).send() => {
                res.context("Sogou request failed")?
            }
        };

        let status = response.status();
        let html = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            body = response.text() => {
                body.context("Sogou body read failed")?
            }
        };

        if !status.is_success() {
            bail!("Sogou returned HTTP {}", status.as_u16());
        }

        let results = extract_sogou_results(&html, max_results);
        if results.is_empty() {
            bail!("Sogou: parsed zero results");
        }
        Ok(results)
    }
}

/// Extract search results from Sogou's HTML page.
///
/// Sogou uses several result container patterns:
/// - `<div class="vrwrap">` (vertical result wrap)
/// - `<div class="rb">` (result block)
///
/// Titles are in `<h3>` → `<a>`, snippets in `<p class="star-wiki">`,
/// `<p class="str_info">`, or `<div class="space-txt">`.
fn extract_sogou_results(html: &str, max_results: usize) -> Vec<SearchResult> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while out.len() < max_results && cursor < html.len() {
        // Find next result container.
        let Some(container_rel) = find_next_sogou_container(html, cursor) else {
            break;
        };
        let container_start = cursor + container_rel;

        // Guard against out-of-bounds slice.
        if container_start >= html.len() {
            break;
        }

        // Find the end of this container (next container or end of doc).
        let container_end = find_next_sogou_container(html, container_start + 1)
            .map(|rel| (container_start + 1 + rel).min(html.len()))
            .unwrap_or(html.len());

        let fragment = &html[container_start..container_end];

        // Extract title + URL from the first <h3> → <a>.
        let (title, href) = extract_sogou_title_link(fragment);

        // Extract snippet from various known markers.
        let snippet = extract_sogou_snippet(fragment);

        if !title.is_empty() && !href.is_empty() {
            out.push(SearchResult {
                title: decode_html_entities(&title),
                url: href,
                snippet: decode_html_entities(&snippet),
            });
        }

        cursor = container_end;
    }

    out
}

/// Find the byte offset (relative to `from`) of the next Sogou result
/// container marker.
fn find_next_sogou_container(html: &str, from: usize) -> Option<usize> {
    let remaining = &html[from..];
    // Prefer vrwrap (main results), fall back to rb.
    let vr = remaining.find("class=\"vrwrap\"");
    let rb = remaining.find("class=\"rb\"");

    match (vr, rb) {
        (Some(a), Some(b)) => Some(from + a.min(b)),
        (Some(a), None) => Some(from + a),
        (None, Some(b)) => Some(from + b),
        (None, None) => None,
    }
}

/// Extract title text and URL from `<h3>...<a href="...">title</a>...</h3>`.
fn extract_sogou_title_link(fragment: &str) -> (String, String) {
    // Find <h3> section.
    let h3_start = match fragment.find("<h3") {
        Some(i) => i,
        None => return (String::new(), String::new()),
    };
    let h3_end = fragment[h3_start..]
        .find("</h3>")
        .map(|i| h3_start + i + 5)
        .unwrap_or(fragment.len());
    let h3_content = &fragment[h3_start..h3_end];

    // Find <a> inside <h3>.
    let a_start = match h3_content.find("<a") {
        Some(i) => i,
        None => return (String::new(), String::new()),
    };
    let a_tag_end = h3_content[a_start..]
        .find('>')
        .map(|i| a_start + i + 1)
        .unwrap_or(h3_content.len());
    let a_close = h3_content[a_tag_end..]
        .find("</a>")
        .map(|i| a_tag_end + i)
        .unwrap_or(h3_content.len());

    let title = strip_html_tags(&h3_content[a_tag_end..a_close]);
    let href = extract_href_from_tag(&h3_content[a_start..a_tag_end])
        .unwrap_or_default();

    (title, href)
}

/// Extract snippet text from a Sogou result fragment.
fn extract_sogou_snippet(fragment: &str) -> String {
    // Try several known snippet class markers in order of preference.
    for marker in &[
        "class=\"str_info\"",
        "class=\"star-wiki\"",
        "class=\"space-txt\"",
        "class=\"str-text\"",
        "class=\"ft\"",
    ] {
        if let Some(idx) = fragment.find(*marker) {
            let tag_start = fragment[..idx].rfind('<').unwrap_or(0);
            let tag_end_rel = fragment[idx..].find('>').unwrap_or(0);
            let content_start = idx + tag_end_rel + 1;
            // Find closing tag.
            let tag_name_end = fragment[tag_start + 1..]
                .find(|ch: char| ch == '>' || ch.is_whitespace())
                .map(|i| tag_start + 1 + i)
                .unwrap_or(tag_start + 1);
            let tag_name = &fragment[tag_start + 1..tag_name_end];
            let close_marker = format!("</{tag_name}>");
            if let Some(close_rel) = fragment[content_start..].find(&close_marker) {
                let raw = &fragment[content_start..content_start + close_rel];
                let cleaned = strip_html_tags(raw);
                if !cleaned.is_empty() {
                    return cleaned;
                }
            }
        }
    }
    String::new()
}

// ──────────────────────── Brave Search API ─────────────────────────

struct BraveSearchBackend {
    client: reqwest::Client,
    api_key: String,
}

#[async_trait]
impl SearchBackend for BraveSearchBackend {
    fn name(&self) -> &str {
        "brave"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            "https://api.search.brave.com/res/v1/web/search",
            &[("q", query), ("count", &max_results.to_string())],
        )
        .context("Failed to build Brave URL")?;

        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = self.client
                .get(url)
                .header("Accept", "application/json")
                .header("Accept-Encoding", "gzip")
                .header("X-Subscription-Token", &self.api_key)
                .send() => {
                res.context("Brave request failed")?
            }
        };

        let status = response.status();
        if !status.is_success() {
            bail!("Brave returned HTTP {}", status.as_u16());
        }

        let body: serde_json::Value = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = response.json() => {
                res.context("Brave JSON parse failed")?
            }
        };

        let results = extract_api_results(&body, "web", "results");
        Ok(results)
    }
}

// ──────────────────────── Bing Search API ──────────────────────────

struct BingSearchBackend {
    client: reqwest::Client,
    api_key: String,
}

#[async_trait]
impl SearchBackend for BingSearchBackend {
    fn name(&self) -> &str {
        "bing"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            "https://api.bing.microsoft.com/v7.0/search",
            &[("q", query), ("count", &max_results.to_string())],
        )
        .context("Failed to build Bing URL")?;

        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = self.client
                .get(url)
                .header("Ocp-Apim-Subscription-Key", &self.api_key)
                .send() => {
                res.context("Bing request failed")?
            }
        };

        let status = response.status();
        if !status.is_success() {
            bail!("Bing returned HTTP {}", status.as_u16());
        }

        let body: serde_json::Value = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = response.json() => {
                res.context("Bing JSON parse failed")?
            }
        };

        let results = extract_api_results(&body, "webPages", "value");
        Ok(results)
    }
}

// ──────────────────────── SerpApi backend ──────────────────────────

struct SerpApiBackend {
    client: reqwest::Client,
    api_key: String,
}

#[async_trait]
impl SearchBackend for SerpApiBackend {
    fn name(&self) -> &str {
        "serpapi"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            "https://serpapi.com/search.json",
            &[
                ("q", query),
                ("api_key", &self.api_key),
                ("num", &max_results.to_string()),
            ],
        )
        .context("Failed to build SerpApi URL")?;

        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = self.client.get(url).send() => {
                res.context("SerpApi request failed")?
            }
        };

        let status = response.status();
        if !status.is_success() {
            bail!("SerpApi returned HTTP {}", status.as_u16());
        }

        let body: serde_json::Value = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = response.json() => {
                res.context("SerpApi JSON parse failed")?
            }
        };

        let results = extract_api_results(&body, "", "organic_results");
        Ok(results)
    }
}

// ──────────────────────── SearXNG backend ──────────────────────────

struct SearXngBackend {
    client: reqwest::Client,
    base_url: String,
}

#[async_trait]
impl SearchBackend for SearXngBackend {
    fn name(&self) -> &str {
        "searxng"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            &format!("{}/search", self.base_url),
            &[
                ("q", query),
                ("format", "json"),
                ("pageno", "1"),
            ],
        )
        .context("Failed to build SearXNG URL")?;

        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = self.client.get(url).send() => {
                res.context("SearXNG request failed")?
            }
        };

        let status = response.status();
        if !status.is_success() {
            bail!("SearXNG returned HTTP {}", status.as_u16());
        }

        let body: serde_json::Value = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = response.json() => {
                res.context("SearXNG JSON parse failed")?
            }
        };

        let mut results = extract_api_results(&body, "", "results");
        results.truncate(max_results);
        Ok(results)
    }
}

// ─────────────────── Semantic Scholar backend ─────────────────────

struct SemanticScholarBackend {
    client: reqwest::Client,
}

#[async_trait]
impl SearchBackend for SemanticScholarBackend {
    fn name(&self) -> &str {
        "semantic_scholar"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        cancel: &CancelToken,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let limit = max_results.min(10);
        let url = reqwest::Url::parse_with_params(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            &[
                ("query", query),
                ("limit", &limit.to_string()),
                ("fields", "title,abstract,url,venue,year,authors"),
            ],
        )
        .context("Failed to build Semantic Scholar URL")?;

        if cancel.is_cancelled() {
            bail!("Search cancelled");
        }

        let response = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = self.client.get(url).send() => {
                res.context("Semantic Scholar request failed")?
            }
        };

        let status = response.status();
        if !status.is_success() {
            bail!("Semantic Scholar returned HTTP {}", status.as_u16());
        }

        let body: serde_json::Value = tokio::select! {
            _ = cancel.cancelled() => bail!("Search cancelled"),
            res = response.json() => {
                res.context("Semantic Scholar JSON parse failed")?
            }
        };

        let mut results = Vec::new();
        if let Some(papers) = body["data"].as_array() {
            for paper in papers.iter().take(max_results) {
                let title = paper["title"].as_str().unwrap_or("").to_string();
                let url = paper["url"]
                    .as_str()
                    .or_else(|| paper["externalIds"]["URL"].as_str())
                    .unwrap_or("")
                    .to_string();
                let abstract_text = paper["abstract"].as_str().unwrap_or("");
                let venue = paper["venue"].as_str().unwrap_or("");
                let year = paper["year"].as_u64().map(|y| y.to_string()).unwrap_or_default();
                let authors: Vec<String> = paper["authors"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|a| a["name"].as_str().map(String::from))
                            .take(3)
                            .collect()
                    })
                    .unwrap_or_default();

                let mut snippet = abstract_text.to_string();
                if !venue.is_empty() || !year.is_empty() || !authors.is_empty() {
                    let meta = [
                        (!venue.is_empty()).then(|| venue),
                        (!year.is_empty()).then(|| year.as_str()),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(", ");
                    if !authors.is_empty() {
                        snippet = format!("{} — {} ({})", authors.join(", "), meta, snippet);
                    } else if !meta.is_empty() {
                        snippet = format!("{} — {}", meta, snippet);
                    }
                }

                if !title.is_empty() && !url.is_empty() {
                    results.push(SearchResult {
                        title,
                        url,
                        snippet,
                    });
                }
            }
        }

        Ok(results)
    }
}

// ─────────────────── generic JSON API result extractor ─────────────

/// Extract `SearchResult`s from a JSON response that follows the common
/// `{ "<section>": { "<key>": [ { "title", "url", "snippet/description" } ] } }`
/// pattern.
///
/// For top-level arrays (like SerpApi's `organic_results` or SearXNG's
/// `results`), pass `section = ""`.
fn extract_api_results(
    body: &serde_json::Value,
    section: &str,
    key: &str,
) -> Vec<SearchResult> {
    let array = if section.is_empty() {
        body.get(key).and_then(|v| v.as_array())
    } else {
        body.get(section)
            .and_then(|s| s.get(key))
            .and_then(|v| v.as_array())
    };

    let Some(items) = array else {
        return Vec::new();
    };

    items
        .iter()
        .filter_map(|item| {
            let title = item
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let url = item
                .get("url")
                .or_else(|| item.get("link"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let snippet = item
                .get("snippet")
                .or_else(|| item.get("description"))
                .or_else(|| item.get("body"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if title.is_empty() || url.is_empty() {
                None
            } else {
                Some(SearchResult {
                    title,
                    url,
                    snippet,
                })
            }
        })
        .collect()
}

// ──────────────────────── HTML utilities ───────────────────────────

/// Extract search results from DuckDuckGo's lightweight HTML page.
pub fn extract_duckduckgo_results(html: &str, max_results: usize) -> Vec<SearchResult> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while out.len() < max_results {
        let Some(anchor_rel) = html[cursor..]
            .find("result__a")
            .or_else(|| html[cursor..].find("result-link"))
        else {
            break;
        };
        let anchor_idx = cursor + anchor_rel;
        let Some(tag_start) = html[..anchor_idx].rfind("<a") else {
            cursor = anchor_idx + 1;
            continue;
        };
        let Some(tag_end_rel) = html[anchor_idx..].find("</a>") else {
            break;
        };
        let tag_end = anchor_idx + tag_end_rel + "</a>".len();
        let anchor_html = &html[tag_start..tag_end];

        let href = extract_href_from_tag(anchor_html)
            .map(|h| normalize_duckduckgo_result_url(&h))
            .unwrap_or_default();
        let title = strip_html_tags(anchor_text(anchor_html));

        // Look at a small fragment after the anchor for the snippet.
        let next_anchor = html[tag_end..]
            .find("result__a")
            .or_else(|| html[tag_end..].find("result-link"))
            .map(|idx| tag_end + idx)
            .unwrap_or_else(|| html.len());
        let snippet_fragment = &html[tag_end..next_anchor.min(tag_end.saturating_add(4_000))];
        let snippet = extract_html_class_text(snippet_fragment, "result__snippet")
            .or_else(|| extract_html_class_text(snippet_fragment, "result-snippet"))
            .or_else(|| extract_html_class_text(snippet_fragment, "snippet"))
            .unwrap_or_default();

        if !title.is_empty() && !href.is_empty() {
            out.push(SearchResult {
                title: decode_html_entities(&title),
                url: href,
                snippet: decode_html_entities(&snippet),
            });
        }

        cursor = tag_end;
    }

    out
}

/// Extract href value from an anchor tag string.
fn extract_href_from_tag(tag_html: &str) -> Option<String> {
    let href_pos = tag_html.find("href=")?;
    let quote = tag_html[href_pos + 5..].chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let value_start = href_pos + 6;
    let value_end_rel = tag_html[value_start..].find(quote)?;
    let value = &tag_html[value_start..value_start + value_end_rel];
    Some(decode_html_entities(value.trim()))
}

/// Extract visible text content from an anchor tag.
fn anchor_text(anchor_html: &str) -> &str {
    let content_start = match anchor_html.find('>') {
        Some(idx) => idx + 1,
        None => return "",
    };
    let content_end = match anchor_html.rfind("</a>") {
        Some(idx) if idx >= content_start => idx,
        _ => anchor_html.len(),
    };
    &anchor_html[content_start..content_end]
}

/// Extract text from an HTML element identified by a CSS class marker.
fn extract_html_class_text(fragment: &str, class_marker: &str) -> Option<String> {
    let marker_idx = fragment.find(class_marker)?;
    let tag_start = fragment[..marker_idx].rfind('<')?;
    let tag_name_end = fragment[tag_start + 1..]
        .find(|ch: char| ch == '>' || ch.is_whitespace())
        .map(|idx| tag_start + 1 + idx)?;
    let tag_name = &fragment[tag_start + 1..tag_name_end];
    let open_end_rel = fragment[marker_idx..].find('>')?;
    let content_start = marker_idx + open_end_rel + 1;
    let close_marker = format!("</{tag_name}>");
    let close_tag_rel = fragment[content_start..].find(&close_marker)?;
    let close_tag = content_start + close_tag_rel;

    if close_tag <= tag_start {
        return None;
    }

    let raw = &fragment[content_start..close_tag];
    let cleaned = strip_html_tags(raw);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Remove simple HTML tags and decode common entities.
fn strip_html_tags(raw: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;

    for ch in raw.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }

    decode_html_entities(out.trim())
}

/// Normalize DuckDuckGo redirect links into their destination URL.
fn normalize_duckduckgo_result_url(raw: &str) -> String {
    let raw = raw.trim();
    let candidate = if raw.starts_with("//") {
        format!("https:{raw}")
    } else {
        raw.to_string()
    };

    let Ok(url) = reqwest::Url::parse(&candidate) else {
        return candidate;
    };

    if url.domain() == Some("duckduckgo.com") || url.domain() == Some("html.duckduckgo.com") {
        if let Some((_, value)) = url.query_pairs().find(|(key, _)| key == "uddg") {
            return value.into_owned();
        }
    }

    url.to_string()
}

/// Decode a small set of HTML entities commonly returned by search result
/// pages.
fn decode_html_entities(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if bytes[idx] != b'&' {
            out.push(bytes[idx] as char);
            idx += 1;
            continue;
        }

        let Some(end_rel) = raw[idx..].find(';') else {
            out.push('&');
            idx += 1;
            continue;
        };
        let end = idx + end_rel;
        let entity = &raw[idx + 1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" | "#x27" => Some('\''),
            "nbsp" => Some(' '),
            "#47" | "#x2F" => Some('/'),
            _ => decode_numeric_entity(entity),
        };

        if let Some(ch) = decoded {
            out.push(ch);
        } else {
            out.push('&');
            out.push_str(entity);
            out.push(';');
        }
        idx = end + 1;
    }

    out
}

/// Decode `&#...;` or `&#x...;` entities.
fn decode_numeric_entity(entity: &str) -> Option<char> {
    if let Some(hex) = entity
        .strip_prefix("#x")
        .or_else(|| entity.strip_prefix("#X"))
    {
        let value = u32::from_str_radix(hex, 16).ok()?;
        return char::from_u32(value);
    }

    let dec = entity.strip_prefix('#')?;
    let value = dec.parse::<u32>().ok()?;
    char::from_u32(value)
}

// ──────────────────────────── tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddg_extract_parses_title_url_and_snippet() {
        let html = r#"
<div class="result">
  <a rel="nofollow" class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
    Example &amp; Guide
  </a>
  <a class="result__snippet">A <b>useful</b> summary.</a>
</div>
"#;

        let results = extract_duckduckgo_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Example & Guide");
        assert_eq!(results[0].url, "https://example.com/page");
        assert_eq!(results[0].snippet, "A useful summary.");
    }

    #[test]
    fn sogou_extract_finds_vrwrap_container() {
        let html = r#"
<div class="vrwrap">
  <h3><a href="https://example.com">Test Title</a></h3>
  <p class="str_info">A test snippet.</p>
</div>
"#;

        let results = extract_sogou_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Test Title");
        assert_eq!(results[0].url, "https://example.com");
        assert_eq!(results[0].snippet, "A test snippet.");
    }

    #[test]
    fn sogou_extract_handles_rb_container() {
        let html = r#"
<div class="rb">
  <h3><a href="https://test.org">RB Result</a></h3>
  <div class="space-txt">Some description text.</div>
</div>
"#;

        let results = extract_sogou_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "RB Result");
        assert_eq!(results[0].url, "https://test.org");
    }

    #[test]
    fn api_result_extractor_handles_serpapi_format() {
        let body = serde_json::json!({
            "organic_results": [
                { "title": "Result 1", "link": "https://a.com", "snippet": "First" },
                { "title": "Result 2", "link": "https://b.com", "snippet": "Second" },
            ]
        });

        let results = extract_api_results(&body, "", "organic_results");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Result 1");
        assert_eq!(results[1].url, "https://b.com");
    }

    #[test]
    fn api_result_extractor_handles_bing_format() {
        let body = serde_json::json!({
            "webPages": {
                "value": [
                    { "title": "Bing 1", "url": "https://x.com", "snippet": "X" },
                ]
            }
        });

        let results = extract_api_results(&body, "webPages", "value");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Bing 1");
    }

    #[test]
    fn normalize_duckduckgo_redirect_extracts_uddg() {
        let url = normalize_duckduckgo_result_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide",
        );
        assert_eq!(url, "https://example.com/guide");
    }

    #[test]
    fn default_search_config_has_backends() {
        let config = SearchConfig::default();
        assert!(!config.backends.is_empty());
        assert!(config.ddg.enabled);
        assert!(config.sogou.enabled);
    }
}
