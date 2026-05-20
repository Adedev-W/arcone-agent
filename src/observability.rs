#[cfg(feature = "tracing")]
use std::time::Instant;

pub fn redact_secret(_value: &str) -> &'static str {
    "<redacted>"
}

#[cfg(feature = "tracing")]
pub(crate) fn elapsed_ms(started_at: Instant) -> u128 {
    started_at.elapsed().as_millis()
}
