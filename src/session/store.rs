use std::future::Future;
use std::pin::Pin;

use crate::{ChatMessage, Result};

use super::types::{SessionId, SessionMetadata};

/// Boxed future type for memory store operations, following the same pattern
/// as `ToolFuture` in the codebase.
pub type MemoryFuture<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

/// Async trait for session persistence.
///
/// Implementors provide storage and retrieval of chat message history
/// keyed by `SessionId`.
pub trait MemoryStore: Send + Sync {
    /// Persist messages for the given session, replacing any existing history.
    fn save_messages(&self, id: &SessionId, messages: &[ChatMessage]) -> MemoryFuture<()>;

    /// Load the full ordered message history for the given session.
    /// Returns an empty `Vec` if the session does not exist.
    fn load_messages(&self, id: &SessionId) -> MemoryFuture<Vec<ChatMessage>>;

    /// Persist metadata for the given session.
    fn save_metadata(&self, id: &SessionId, metadata: &SessionMetadata) -> MemoryFuture<()>;

    /// Load metadata for the given session.
    /// Returns `None` if metadata for the session does not exist.
    fn load_metadata(&self, id: &SessionId) -> MemoryFuture<Option<SessionMetadata>>;

    /// Remove all messages for the given session.
    fn clear(&self, id: &SessionId) -> MemoryFuture<()>;

    /// Check whether a session has any stored messages.
    fn exists(&self, id: &SessionId) -> MemoryFuture<bool>;
}
