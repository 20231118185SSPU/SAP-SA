//! Cooperative cancellation primitives.
//!
//! Why we need this:
//! - The user asked for a CLI that can "send messages at any time to interrupt".
//! - That requires the backend agent loop to be cancellable **without** killing
//!   the daemon process.
//!
//! Design constraints:
//! - Must be clonable so many async tasks can observe the same cancellation.
//! - Must be cheap and not require polling.
//! - Must avoid races (a cancellation that happens "just before" waiting must
//!   still be observed).
//!
//! Implementation choice:
//! - We use `tokio::sync::watch` because it:
//!   - stores the latest value (no missed wakeups),
//!   - supports many receivers,
//!   - is lightweight.

use tokio::sync::watch;

/// A handle that can request cancellation.
///
/// This is held by the backend "hub" so that a WS message can cancel the
/// currently running task.
#[derive(Debug, Clone)]
pub struct CancelHandle {
    /// `true` means "cancel requested".
    tx: watch::Sender<bool>,
}

/// A token that can be observed by long-running async operations.
///
/// This is passed to the agent loop and tools so they can stop early when the
/// user requests an interrupt.
#[derive(Debug, Clone)]
pub struct CancelToken {
    /// `true` means "cancel requested".
    rx: watch::Receiver<bool>,
}

/// Create a new cancellation pair.
///
/// - The `CancelHandle` is used to request cancellation.
/// - The `CancelToken` is used to observe cancellation.
pub fn cancel_pair() -> (CancelHandle, CancelToken) {
    // Start in "not cancelled" state.
    let (tx, rx) = watch::channel(false);
    (CancelHandle { tx }, CancelToken { rx })
}

impl CancelHandle {
    /// Request cancellation.
    ///
    /// This is idempotent: requesting cancellation multiple times has the same effect.
    pub fn cancel(&self) {
        // We ignore the error because receivers may already be dropped.
        let _ = self.tx.send(true);
    }

    /// Return `true` if cancellation has already been requested.
    pub fn is_cancelled(&self) -> bool {
        *self.tx.borrow()
    }
}

impl CancelToken {
    /// Return `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        *self.rx.borrow()
    }

    /// Wait until cancellation is requested (or the sender is dropped).
    pub async fn cancelled(&self) {
        // Clone a receiver so we can await changes without needing `&mut self`.
        let mut rx = self.rx.clone();

        // If already cancelled, return immediately.
        if *rx.borrow() {
            return;
        }

        // Otherwise wait until the value becomes `true`.
        while rx.changed().await.is_ok() {
            if *rx.borrow() {
                return;
            }
        }
    }
}
