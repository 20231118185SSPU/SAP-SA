//! Safety checks for the `Fetch` tool.
//!
//! Claude Code's `WebFetch` is much narrower than SA's `Fetch`: it is a
//! read-only webpage fetcher with strong URL validation and redirect
//! constraints. SA keeps the more general tool, but ports the core safety
//! ideas:
//! - reject obviously unsafe URL shapes;
//! - refuse localhost / private-address access;
//! - reject sensitive request headers;
//! - auto-upgrade `http` to `https`;
//! - only auto-follow redirects that stay on the same host (allowing `www.`
//!   add/remove).

use anyhow::Context as _;
use reqwest::{Method, Url};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Claude Code uses a 2000-character URL ceiling; SA mirrors that bound.
pub const MAX_FETCH_URL_LENGTH: usize = 2_000;

/// Hard transport ceiling inspired by Claude Code's `WebFetch`.
pub const MAX_FETCH_TRANSFER_BYTES: usize = 10 * 1024 * 1024;

/// Redirect hop ceiling inspired by Claude Code's `WebFetch`.
pub const MAX_FETCH_REDIRECTS: usize = 10;

/// Request timeout budget in seconds.
pub const FETCH_TIMEOUT_SECS: u64 = 60;

/// Prepared request after safety validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedFetchRequest {
    /// Sanitized URL that should actually be requested.
    pub url: Url,
    /// Best-effort warning that should be surfaced back to the agent.
    pub safety_warning: Option<String>,
}

/// Validate a raw Fetch request and return the sanitized request URL.
pub async fn validate_fetch_request(
    raw_url: &str,
    method: &Method,
    headers: &serde_json::Map<String, serde_json::Value>,
    body: Option<&str>,
) -> anyhow::Result<ValidatedFetchRequest> {
    let mut validated = validate_fetch_url_shape(raw_url)?;
    validate_fetch_method(method, body)?;
    validate_fetch_headers(headers)?;
    validate_fetch_host_safety(&validated.url).await?;

    if validated.url.scheme() == "http" {
        validated
            .url
            .set_scheme("https")
            .map_err(|_| anyhow::anyhow!("Failed to upgrade Fetch URL to https"))?;
        validated.safety_warning =
            Some("Fetch upgraded `http` to `https` before sending the request.".to_string());
    }

    Ok(validated)
}

/// Validate the static shape of a Fetch URL before DNS/network checks.
pub fn validate_fetch_url_shape(raw_url: &str) -> anyhow::Result<ValidatedFetchRequest> {
    if raw_url.len() > MAX_FETCH_URL_LENGTH {
        anyhow::bail!(
            "Fetch URL exceeds SA's {}-character safety limit",
            MAX_FETCH_URL_LENGTH
        );
    }

    let url = crate::tools::validate_network_url(raw_url)?;
    if !url.username().is_empty() || url.password().is_some() {
        anyhow::bail!("Fetch URLs must not include username/password credentials");
    }

    let Some(host) = url.host_str() else {
        anyhow::bail!("Fetch URL must include a hostname");
    };

    if is_localish_hostname(host) {
        anyhow::bail!("Fetch blocks localhost and local-network hostnames: {host}");
    }

    if !host.contains('.') && host.parse::<IpAddr>().is_err() {
        anyhow::bail!("Fetch blocks single-label hostnames that do not look public: {host}");
    }

    Ok(ValidatedFetchRequest {
        url,
        safety_warning: None,
    })
}

/// Validate Fetch method semantics.
fn validate_fetch_method(method: &Method, body: Option<&str>) -> anyhow::Result<()> {
    if method.as_str().trim().is_empty() {
        anyhow::bail!("Fetch HTTP method must not be empty");
    }
    if !matches!(*method, Method::GET | Method::HEAD) {
        anyhow::bail!(
            "Fetch only supports read-only GET/HEAD requests in SA's current safety model"
        );
    }
    if body.is_some() {
        anyhow::bail!("Fetch does not support request bodies in SA's current safety model");
    }
    Ok(())
}

/// Reject request headers that commonly carry authentication or proxy routing.
fn validate_fetch_headers(
    headers: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<()> {
    for (name, value) in headers {
        if !value.is_string() {
            anyhow::bail!("Fetch headers must be string key/value pairs");
        }

        if is_forbidden_header_name(name) {
            anyhow::bail!("Fetch blocks sensitive request header `{name}`");
        }
    }

    Ok(())
}

/// Perform host/IP safety checks, including best-effort DNS resolution.
async fn validate_fetch_host_safety(url: &Url) -> anyhow::Result<()> {
    let Some(host) = url.host_str() else {
        anyhow::bail!("Fetch URL must include a hostname");
    };

    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_forbidden_ip(ip) {
            anyhow::bail!("Fetch blocks private or local IP targets: {host}");
        }
        return Ok(());
    }

    let port = url.port_or_known_default().unwrap_or(443);
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("Fetch failed to resolve host `{host}` for safety validation"))?;
    let mut saw_any = false;
    for address in addresses {
        saw_any = true;
        if is_forbidden_ip(address.ip()) {
            anyhow::bail!(
                "Fetch blocks host `{host}` because it resolves to private/local address `{}`",
                address.ip()
            );
        }
    }
    if !saw_any {
        anyhow::bail!("Fetch could not resolve any addresses for host `{host}`");
    }

    Ok(())
}

/// Return whether a redirect may be auto-followed.
pub fn is_permitted_redirect(original: &Url, redirect: &Url) -> bool {
    if !redirect.username().is_empty() || redirect.password().is_some() {
        return false;
    }

    if original.scheme() != redirect.scheme() {
        return false;
    }
    if original.port_or_known_default() != redirect.port_or_known_default() {
        return false;
    }

    match (original.host_str(), redirect.host_str()) {
        (Some(original_host), Some(redirect_host)) => {
            strip_www(original_host) == strip_www(redirect_host)
        }
        _ => false,
    }
}

/// Return whether the hostname is obviously local-only.
fn is_localish_hostname(host: &str) -> bool {
    let normalized = host.trim_end_matches('.').to_ascii_lowercase();
    normalized == "localhost"
        || normalized.ends_with(".localhost")
        || normalized.ends_with(".local")
}

/// Return whether the request header name is too sensitive for generic Fetch.
fn is_forbidden_header_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization"
            | "cookie"
            | "proxy-authorization"
            | "host"
            | "forwarded"
            | "x-forwarded-for"
            | "x-forwarded-host"
            | "x-real-ip"
    )
}

/// Return whether the IP address is local/private or otherwise unsuitable for
/// generic network fetches.
fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_forbidden_ipv4(ip),
        IpAddr::V6(ip) => is_forbidden_ipv6(ip),
    }
}

/// IPv4-specific blocklist.
fn is_forbidden_ipv4(ip: Ipv4Addr) -> bool {
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || ip.is_multicast()
        || is_documentation_ipv4(ip)
}

/// IPv6-specific blocklist.
fn is_forbidden_ipv6(ip: Ipv6Addr) -> bool {
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.is_unicast_link_local()
        || ip.is_unique_local()
        || is_documentation_ipv6(ip)
}

/// Strip a leading `www.` prefix for redirect comparison.
fn strip_www(host: &str) -> &str {
    host.strip_prefix("www.").unwrap_or(host)
}

/// Return whether the IPv4 address is from one of the documentation ranges.
fn is_documentation_ipv4(ip: Ipv4Addr) -> bool {
    matches!(
        ip.octets(),
        [192, 0, 2, _] | [198, 51, 100, _] | [203, 0, 113, _]
    )
}

/// Return whether the IPv6 address is from the `2001:db8::/32` documentation range.
fn is_documentation_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    segments[0] == 0x2001 && segments[1] == 0x0db8
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_FETCH_URL_LENGTH, is_permitted_redirect, validate_fetch_headers,
        validate_fetch_request, validate_fetch_url_shape,
    };
    use reqwest::Method;
    use reqwest::Url;

    #[test]
    fn url_shape_rejects_userinfo() {
        let err = validate_fetch_url_shape("https://user:pass@example.com")
            .expect_err("userinfo URL must fail");
        assert!(err.to_string().contains("username/password"));
    }

    #[test]
    fn url_shape_rejects_localhost() {
        let err =
            validate_fetch_url_shape("https://localhost/admin").expect_err("localhost must fail");
        assert!(err.to_string().contains("localhost"));
    }

    #[test]
    fn url_shape_rejects_single_label_host() {
        let err = validate_fetch_url_shape("https://internal/path")
            .expect_err("single-label host must fail");
        assert!(err.to_string().contains("single-label"));
    }

    #[test]
    fn url_shape_accepts_public_https() {
        let validated =
            validate_fetch_url_shape("https://example.com/docs").expect("public URL should parse");
        assert_eq!(validated.url.as_str(), "https://example.com/docs");
        assert_eq!(validated.safety_warning, None);
    }

    #[test]
    fn header_validation_rejects_authorization() {
        let mut headers = serde_json::Map::new();
        headers.insert(
            "Authorization".to_string(),
            serde_json::Value::String("Bearer secret".to_string()),
        );
        let err = validate_fetch_headers(&headers).expect_err("auth header must fail");
        assert!(err.to_string().contains("sensitive request header"));
    }

    #[tokio::test]
    async fn fetch_request_rejects_non_readonly_method() {
        let headers = serde_json::Map::new();
        let err = validate_fetch_request("https://example.com/docs", &Method::POST, &headers, None)
            .await
            .expect_err("POST must fail");
        assert!(err.to_string().contains("GET/HEAD"));
    }

    #[tokio::test]
    async fn fetch_request_rejects_request_body() {
        let headers = serde_json::Map::new();
        let err = validate_fetch_request(
            "https://example.com/docs",
            &Method::GET,
            &headers,
            Some("payload"),
        )
        .await
        .expect_err("request body must fail");
        assert!(err.to_string().contains("request bodies"));
    }

    #[test]
    fn redirect_policy_allows_same_host_and_www_variant() {
        let original = Url::parse("https://example.com/docs").expect("original URL");
        let redirect = Url::parse("https://www.example.com/docs/latest").expect("redirect URL");
        assert!(is_permitted_redirect(&original, &redirect));
    }

    #[test]
    fn redirect_policy_rejects_cross_host() {
        let original = Url::parse("https://example.com/docs").expect("original URL");
        let redirect = Url::parse("https://evil.example.net/docs").expect("redirect URL");
        assert!(!is_permitted_redirect(&original, &redirect));
    }

    #[test]
    fn url_limit_constant_matches_expected_safety_budget() {
        assert_eq!(MAX_FETCH_URL_LENGTH, 2_000);
    }
}
