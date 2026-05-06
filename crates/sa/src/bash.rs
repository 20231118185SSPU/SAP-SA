//! Background Bash command execution and streaming.

use anyhow::Context as _;
use sa_core::tools::StartTerminalTaskRequest;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

/// Candidate `bash` programs to try when starting a background task.
pub(crate) fn candidate_bash_programs() -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("bash")];

    for env_name in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(root) = std::env::var(env_name) {
            let root = PathBuf::from(root);
            out.push(root.join("Git").join("bin").join("bash.exe"));
            out.push(root.join("Git").join("usr").join("bin").join("bash.exe"));
        }
    }

    out
}

/// Pump one child-process pipe into the central log writer channel.
///
/// The background task runtime keeps a single on-disk log file. Stdout and
/// stderr are read concurrently in small chunks and forwarded to the main task,
/// which serializes them into that file. This keeps output visible while the
/// process is still running instead of buffering everything until exit.
pub(crate) async fn stream_background_pipe_to_channel<R>(
    mut reader: R,
    tx: mpsc::UnboundedSender<Vec<u8>>,
) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt as _;

    let mut buffer = vec![0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        if tx.send(buffer[..read].to_vec()).is_err() {
            return Ok(());
        }
    }
}

/// Run one background Bash command and tee stdout/stderr into one log file.
pub(crate) async fn run_background_bash_command(
    request: &StartTerminalTaskRequest,
    output_path: &Path,
    workspace_root: &Path,
) -> anyhow::Result<i32> {
    use tokio::io::AsyncWriteExt as _;
    use tokio::process::Command;

    // Validate workdir is within workspace root
    // Use path components check to avoid Windows UNC path issues with canonicalize()
    let workdir_path = std::path::Path::new(&request.workdir);
    let workspace_normalized = workspace_root.components().collect::<Vec<_>>();
    let workdir_normalized = workdir_path.components().collect::<Vec<_>>();

    // Check if workdir starts with workspace_root components
    if workdir_normalized.len() < workspace_normalized.len()
        || !workdir_normalized
            .iter()
            .zip(workspace_normalized.iter())
            .all(|(a, b)| a.as_os_str() == b.as_os_str())
    {
        anyhow::bail!(
            "工作目录必须在工作区内：{} 不在 {} 内",
            request.workdir,
            workspace_root.display()
        );
    }

    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!(
                "Failed to create runtime task output directory: {}",
                parent.display()
            )
        })?;
    }

    let timeout = request
        .timeout_seconds
        .map(std::time::Duration::from_secs)
        .unwrap_or(std::time::Duration::from_secs(300))
        .min(std::time::Duration::from_secs(1800));

    let mut last_not_found: Option<anyhow::Error> = None;
    for program in candidate_bash_programs() {
        let mut cmd = Command::new(&program);
        cmd.arg("-lc");
        cmd.arg(&request.command);
        cmd.current_dir(&request.workdir);
        cmd.kill_on_drop(true);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                last_not_found = Some(anyhow::Error::new(err).context(format!(
                    "Bash executable not found at {}",
                    program.display()
                )));
                continue;
            }
            Err(err) => {
                return Err(anyhow::Error::new(err))
                    .with_context(|| format!("Failed to execute Bash via {}", program.display()));
            }
        };

        let stdout = child.stdout.take().with_context(|| {
            format!(
                "Failed to capture stdout for background Bash via {}",
                program.display()
            )
        })?;
        let stderr = child.stderr.take().with_context(|| {
            format!(
                "Failed to capture stderr for background Bash via {}",
                program.display()
            )
        })?;

        let mut file = match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)
            .await
        {
            Ok(file) => file,
            Err(err) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err(anyhow::Error::new(err).context(format!(
                    "Failed to create background task log securely: {}",
                    output_path.display()
                )));
            }
        };

        let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let stdout_task = {
            let tx = chunk_tx.clone();
            tokio::spawn(async move { stream_background_pipe_to_channel(stdout, tx).await })
        };
        let stderr_task = {
            let tx = chunk_tx.clone();
            tokio::spawn(async move { stream_background_pipe_to_channel(stderr, tx).await })
        };
        drop(chunk_tx);

        let mut timed_out = false;
        let mut child_status: Option<std::process::ExitStatus> = None;
        let mut streams_closed = false;
        let timeout_sleep = tokio::time::sleep(timeout);
        tokio::pin!(timeout_sleep);
        let mut poll_interval = tokio::time::interval(std::time::Duration::from_millis(25));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            if child_status.is_some() && streams_closed {
                break;
            }

            tokio::select! {
                maybe_chunk = chunk_rx.recv(), if !streams_closed => {
                    match maybe_chunk {
                        Some(chunk) => {
                            file.write_all(&chunk).await.with_context(|| {
                                format!("Failed to append streamed log: {}", output_path.display())
                            })?;
                            file.flush().await.with_context(|| {
                                format!("Failed to flush background task log: {}", output_path.display())
                            })?;
                        }
                        None => {
                            streams_closed = true;
                        }
                    }
                }
                _ = poll_interval.tick(), if child_status.is_none() => {
                    child_status = child.try_wait().with_context(|| {
                        format!("Failed to poll background Bash via {}", program.display())
                    })?;
                }
                _ = &mut timeout_sleep, if !timed_out && child_status.is_none() => {
                    timed_out = true;
                    match child.kill().await {
                        Ok(()) => {}
                        Err(err) if err.kind() == std::io::ErrorKind::InvalidInput => {}
                        Err(err) => {
                            return Err(anyhow::Error::new(err)).with_context(|| {
                                format!(
                                    "Failed to terminate timed-out background Bash via {}",
                                    program.display()
                                )
                            });
                        }
                    }
                    child_status = Some(child.wait().await.with_context(|| {
                        format!(
                            "Failed to wait for timed-out background Bash via {}",
                            program.display()
                        )
                    })?);
                }
            }

            if child_status.is_none() {
                child_status = child.try_wait().with_context(|| {
                    format!("Failed to poll background Bash via {}", program.display())
                })?;
            }
        }

        for (label, task) in [("stdout", stdout_task), ("stderr", stderr_task)] {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!(
                            "Failed while streaming background {label} for Bash via {}",
                            program.display()
                        )
                    });
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)).with_context(|| {
                        format!(
                            "Background {label} stream task crashed for Bash via {}",
                            program.display()
                        )
                    });
                }
            }
        }

        file.flush().await.with_context(|| {
            format!(
                "Failed to flush background task log: {}",
                output_path.display()
            )
        })?;

        if timed_out {
            return Err(anyhow::anyhow!(
                "Background Bash command timed out after {:?}",
                timeout
            ));
        }

        let status = child_status.expect("background Bash loop must observe child exit");
        return Ok(status.code().unwrap_or(-1));
    }

    Err(last_not_found.unwrap_or_else(|| {
        anyhow::anyhow!("No usable bash executable was found for background task execution.")
    }))
}
