//! Retry/backoff helpers.
//!
//! The user requested an **infinite retry** strategy with a very specific
//! delay schedule:
//! - First 3 errors: retry every 3 seconds
//! - Next 3 errors: retry every 5 seconds
//! - After that: retry delay doubles with each error (infinite retries)
//!
//! We implement that as a pure function so it can be reused for:
//! - LLM API retries (backend)
//! - WebSocket reconnect retries (frontend CLI)

use std::time::Duration;

/// Compute the delay before the next retry.
///
/// `error_count` is **1-based**:
/// - `1` means "first failure"
/// - `2` means "second consecutive failure"
///
/// This function never returns an error and never caps retries.
pub fn retry_delay(error_count: u32) -> Duration {
    // Defensive: treat 0 as 1 (callers should pass 1-based counts).
    let error_count = error_count.max(1);

    // First three failures: 3 seconds.
    if error_count <= 3 {
        return Duration::from_secs(3);
    }

    // Next three failures: 5 seconds.
    if error_count <= 6 {
        return Duration::from_secs(5);
    }

    // After that: double forever.
    //
    // For `error_count == 7`, we want 10 seconds (5 * 2^1).
    let exponent = error_count - 6;

    // Compute `2^exponent` with overflow safety:
    // - if exponent is too large, we saturate at `u64::MAX`.
    let multiplier = 1_u64.checked_shl(exponent).unwrap_or(u64::MAX);

    // Base is 5 seconds (the last fixed interval).
    let secs = 5_u64.saturating_mul(multiplier);
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_delay_matches_requested_schedule() {
        assert_eq!(retry_delay(1), Duration::from_secs(3));
        assert_eq!(retry_delay(2), Duration::from_secs(3));
        assert_eq!(retry_delay(3), Duration::from_secs(3));

        assert_eq!(retry_delay(4), Duration::from_secs(5));
        assert_eq!(retry_delay(5), Duration::from_secs(5));
        assert_eq!(retry_delay(6), Duration::from_secs(5));

        assert_eq!(retry_delay(7), Duration::from_secs(10));
        assert_eq!(retry_delay(8), Duration::from_secs(20));
        assert_eq!(retry_delay(9), Duration::from_secs(40));
    }
}
