use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::ChatMessage;

use super::store::{MemoryFuture, MemoryStore};
use super::types::{SessionId, SessionMetadata};

/// Thread-safe in-memory implementation of `MemoryStore`.
///
/// Uses `Arc<RwLock<HashMap>>` internally so it can be shared across
/// multiple agents via cloning. All operations are synchronous under the
/// hood (wrapped in immediate futures) since HashMap access is fast.
#[derive(Clone)]
pub struct InMemorySessionStore {
    sessions: Arc<RwLock<HashMap<SessionId, Vec<ChatMessage>>>>,
    metadata: Arc<RwLock<HashMap<SessionId, SessionMetadata>>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore for InMemorySessionStore {
    fn save_messages(&self, id: &SessionId, messages: &[ChatMessage]) -> MemoryFuture<()> {
        let id = id.clone();
        let messages = messages.to_vec();
        let sessions = Arc::clone(&self.sessions);

        Box::pin(async move {
            let mut store = sessions
                .write()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            store.insert(id, messages);
            Ok(())
        })
    }

    fn load_messages(&self, id: &SessionId) -> MemoryFuture<Vec<ChatMessage>> {
        let id = id.clone();
        let sessions = Arc::clone(&self.sessions);

        Box::pin(async move {
            let store = sessions
                .read()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            Ok(store.get(&id).cloned().unwrap_or_default())
        })
    }

    fn save_metadata(&self, id: &SessionId, metadata: &SessionMetadata) -> MemoryFuture<()> {
        let id = id.clone();
        let metadata = metadata.clone();
        let metadata_store = Arc::clone(&self.metadata);

        Box::pin(async move {
            let mut store = metadata_store
                .write()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            store.insert(id, metadata);
            Ok(())
        })
    }

    fn load_metadata(&self, id: &SessionId) -> MemoryFuture<Option<SessionMetadata>> {
        let id = id.clone();
        let metadata_store = Arc::clone(&self.metadata);

        Box::pin(async move {
            let store = metadata_store
                .read()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            Ok(store.get(&id).cloned())
        })
    }

    fn clear(&self, id: &SessionId) -> MemoryFuture<()> {
        let id = id.clone();
        let sessions = Arc::clone(&self.sessions);
        let metadata = Arc::clone(&self.metadata);

        Box::pin(async move {
            let mut session_store = sessions
                .write()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            session_store.remove(&id);

            let mut metadata_store = metadata
                .write()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            metadata_store.remove(&id);
            Ok(())
        })
    }

    fn exists(&self, id: &SessionId) -> MemoryFuture<bool> {
        let id = id.clone();
        let sessions = Arc::clone(&self.sessions);
        let metadata = Arc::clone(&self.metadata);

        Box::pin(async move {
            let store = sessions
                .read()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            if store.contains_key(&id) {
                return Ok(true);
            }
            drop(store);

            let metadata_store = metadata
                .read()
                .map_err(|e| crate::Error::MemoryStore(format!("lock poisoned: {e}")))?;
            Ok(metadata_store.contains_key(&id))
        })
    }
}
