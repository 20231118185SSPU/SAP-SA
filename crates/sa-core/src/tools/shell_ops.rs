//! Shell command and HTTP fetch tool implementations for ToolExecutor.

use crate::cancel::CancelToken;
use super::{
    ToolExecutor, ToolRuntime,
};
use crate::tools::{DEFAULT_BASH_TIMEOUT, MAX_BASH_TIMEOUT, MAX_FETCH_RESPONSE_BYTES};
use anyhow::Context as _;
use serde::Deserialize;
use serde_json;
use std::time::Duration;
use crate::bash_safety::{BashSafetyDecision, validate_bash_command_safety};
use crate::fetch_safety::{FETCH_TIMEOUT_SECS, MAX_FETCH_REDIRECTS, MAX_FETCH_TRANSFER_BYTES, is_permitted_redirect, validate_fetch_request};
use crate::tools::{candidate_bash_programs, read_fetch_body_limited, StartTerminalTaskRequest};

impl ToolExecutor {
    /// `Bash`: run a command through Git Bash.
    pub(crate) async fn bash(
        &self,
        runtime: &ToolRuntime,
        args: serde_json::Value,
        cancel: &CancelToken,
    ) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            command: String,
            workdir: Option<String>,
            timeout_seconds: Option<u64>,
            #[serde(default)]
            run_in_background: bool,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Bash")?;
        let workdir = match args.workdir.as_deref() {
            Some(raw) => self.ctx.resolve_under_workspace(raw)?,
            None => self.ctx.workspace_root.clone(),
        };

        let timeout = args
            .timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_BASH_TIMEOUT)
            .min(MAX_BASH_TIMEOUT);

        let safety_warning =
            match validate_bash_command_safety(&args.command, &self.ctx.workspace_root, &workdir) {
                BashSafetyDecision::Allow { warning } => warning,
                BashSafetyDecision::Block { reason } => {
                    anyhow::bail!("Bash blocked by SA safety checks: {reason}");
                }
            };

        if args.run_in_background {
            let handle = runtime
                .start_terminal_task(
                    StartTerminalTaskRequest {
                        command: args.command.clone(),
                        workdir: workdir.display().to_string(),
                        timeout_seconds: args.timeout_seconds,
                        safety_warning: safety_warning.clone(),
                    },
                    cancel.clone(),
                )
                .await?;

            return Ok(serde_json::json!({
                "started": true,
                "task_id": handle.task_id,
                "status": handle.status,
                "output_path": handle.output_path,
                "workdir": workdir.display().to_string(),
                "safety_warning": safety_warning,
            })
            .to_string());
        }

        let programs = candidate_bash_programs();
        let mut last_not_found: Option<anyhow::Error> = None;

        for program in programs {
            if cancel.is_cancelled() {
                anyhow::bail!("Bash cancelled");
            }

            let mut cmd = tokio::process::Command::new(&program);
            cmd.arg("-lc");
            cmd.arg(&args.command);
            cmd.current_dir(&workdir);
            cmd.kill_on_drop(true);

            let result = tokio::select! {
                _ = cancel.cancelled() => {
                    anyhow::bail!("Bash cancelled");
                }
                output = tokio::time::timeout(timeout, cmd.output()) => {
                    output.context("Bash command timed out")?
                }
            };

            match result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let exit_code = output.status.code().unwrap_or(-1);

                    return Ok(serde_json::json!({
                        "program": program.display().to_string(),
                        "workdir": workdir.display().to_string(),
                        "exit_code": exit_code,
                        "stdout": stdout,
                        "stderr": stderr,
                        "safety_warning": safety_warning,
                    })
                    .to_string());
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    last_not_found = Some(anyhow::Error::new(err).context(format!(
                        "Bash executable not found at {}",
                        program.display()
                    )));
                    continue;
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!("Failed to execute Bash via {}", program.display())
                    });
                }
            }
        }

        Err(last_not_found.unwrap_or_else(|| {
            anyhow::anyhow!(
                "No usable bash executable was found. Install Git Bash or ensure `bash` is on PATH."
            )
        }))
    }

    /// `Fetch`: make a direct HTTP request to a known URL.

    /// `Fetch`: make a direct HTTP request to a known URL.
    pub(crate) async fn fetch(&self, args: serde_json::Value, cancel: &CancelToken) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct Args {
            url: String,
            method: Option<String>,
            #[serde(default)]
            headers: serde_json::Map<String, serde_json::Value>,
            body: Option<String>,
            max_bytes: Option<usize>,
        }

        let args: Args = serde_json::from_value(args).context("Invalid arguments for Fetch")?;
        if cancel.is_cancelled() {
            anyhow::bail!("Fetch cancelled");
        }

        let method = args
            .method
            .as_deref()
            .unwrap_or("GET")
            .parse::<reqwest::Method>()
            .context("Fetch method must be a valid HTTP method")?;
        let validated =
            validate_fetch_request(&args.url, &method, &args.headers, args.body.as_deref()).await?;
        let max_bytes = args
            .max_bytes
            .unwrap_or(MAX_FETCH_RESPONSE_BYTES)
            .min(MAX_FETCH_RESPONSE_BYTES);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("StudyAdministrator/0.6 Fetch")
            .build()
            .context("Failed to build Fetch HTTP client")?;
        let mut current_url = validated.url.clone();
        let mut redirect_hops = 0usize;

        loop {
            let mut request = client.request(method.clone(), current_url.clone());
            for (name, value) in &args.headers {
                let Some(value) = value.as_str() else {
                    anyhow::bail!("Fetch headers must be string key/value pairs");
                };
                request = request.header(name, value);
            }
            let mut response = tokio::select! {
                _ = cancel.cancelled() => {
                    anyhow::bail!("Fetch cancelled");
                }
                response = request.send() => {
                    response.context("Fetch request failed")?
                }
            };

            let status = response.status();
            if matches!(status.as_u16(), 301 | 302 | 303 | 307 | 308) {
                let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
                    anyhow::bail!("Fetch redirect response is missing a Location header");
                };
                let location = location
                    .to_str()
                    .context("Fetch redirect Location header is not valid UTF-8")?;
                let redirect_url = current_url.join(location).with_context(|| {
                    format!("Failed to resolve Fetch redirect target `{location}`")
                })?;

                if redirect_hops >= MAX_FETCH_REDIRECTS {
                    anyhow::bail!(
                        "Fetch exceeded SA's redirect safety limit of {} hops",
                        MAX_FETCH_REDIRECTS
                    );
                }

                let allows_auto_follow =
                    matches!(method, reqwest::Method::GET | reqwest::Method::HEAD)
                        && is_permitted_redirect(&current_url, &redirect_url);
                if !allows_auto_follow {
                    return Ok(serde_json::json!({
                        "redirect": true,
                        "redirect_blocked": true,
                        "original_url": args.url,
                        "current_url": current_url.as_str(),
                        "redirect_url": redirect_url.as_str(),
                        "method": method.as_str(),
                        "status": status.as_u16(),
                        "message": "Fetch detected a redirect that SA will not follow automatically. Re-run Fetch explicitly with the redirected URL if you intend to trust it.",
                        "safety_warning": validated.safety_warning,
                    })
                    .to_string());
                }

                validate_fetch_request(redirect_url.as_str(), &method, &args.headers, None).await?;
                redirect_hops += 1;
                current_url = redirect_url;
                continue;
            }

            if let Some(content_length) = response.content_length()
                && content_length > MAX_FETCH_TRANSFER_BYTES as u64
            {
                anyhow::bail!(
                    "Fetch response exceeds SA's {}-byte transport safety limit",
                    MAX_FETCH_TRANSFER_BYTES
                );
            }

            let headers = response.headers().clone();
            let content_type = headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            let (body_bytes, total_bytes, truncated) =
                read_fetch_body_limited(&mut response, max_bytes, cancel).await?;
            let body_text = String::from_utf8_lossy(&body_bytes).to_string();
            let response_headers = headers
                .iter()
                .filter_map(|(name, value)| {
                    value.to_str().ok().map(|value| {
                        (
                            name.as_str().to_string(),
                            serde_json::Value::String(value.to_string()),
                        )
                    })
                })
                .collect::<serde_json::Map<_, _>>();

            return Ok(serde_json::json!({
                "url": current_url.as_str(),
                "original_url": args.url,
                "method": method.as_str(),
                "status": status.as_u16(),
                "content_type": content_type,
                "headers": response_headers,
                "bytes": total_bytes,
                "truncated": truncated,
                "body": body_text,
                "safety_warning": validated.safety_warning,
            })
            .to_string());
        }
    }
}
