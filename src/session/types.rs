use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::AgentId;

/// Unique identifier for a session.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random session id using timestamp and a simple counter.
    ///
    /// Format: `ses_<millis>_<random>` where random is a pseudo-random u32
    /// derived from the lower bits of the timestamp and memory address entropy.
    pub fn random() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let millis = now.as_nanos();
        // Use stack address as simple entropy source (no extra deps needed).
        let entropy = (&millis as *const _ as u64).wrapping_mul(6364136223846793005);
        let random_part = (millis as u64).wrapping_add(entropy);
        Self(format!(
            "ses_{}_{:x}",
            now.as_millis(),
            random_part & 0xFFFF_FFFF
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Lightweight metadata attached to a session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unix timestamp in milliseconds when the session was created.
    pub created_at: u64,
    /// Unix timestamp in milliseconds when the session was last updated.
    pub updated_at: u64,
    /// Optional agent id that owns or started this session.
    pub agent_id: Option<AgentId>,
    /// Extensible JSON blob for custom metadata.
    pub extra: Option<serde_json::Value>,
}

impl SessionMetadata {
    pub fn new() -> Self {
        let now = now_millis();
        Self {
            created_at: now,
            updated_at: now,
            agent_id: None,
            extra: None,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = now_millis();
    }

    pub fn with_agent_id(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = Some(extra);
        self
    }
}

impl Default for SessionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
