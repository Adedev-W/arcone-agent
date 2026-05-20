use std::sync::Arc;

use arcone_agent::{
    AgentId, ChatMessage, InMemorySessionStore, MemoryStore, SessionId, SessionMetadata,
};
use serde_json::json;

#[tokio::test]
async fn test_save_and_load() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("test-session-1");

    let messages = vec![
        ChatMessage::user("hello"),
        ChatMessage::assistant("hi there!"),
    ];

    store.save_messages(&session, &messages).await.unwrap();
    let loaded = store.load_messages(&session).await.unwrap();

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].content.as_deref(), Some("hello"));
    assert_eq!(loaded[1].content.as_deref(), Some("hi there!"));
}

#[tokio::test]
async fn test_clear_session() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("clear-test");

    let messages = vec![ChatMessage::user("to be cleared")];
    store.save_messages(&session, &messages).await.unwrap();

    assert!(store.exists(&session).await.unwrap());

    store.clear(&session).await.unwrap();

    let loaded = store.load_messages(&session).await.unwrap();
    assert!(loaded.is_empty());
    assert!(!store.exists(&session).await.unwrap());
}

#[tokio::test]
async fn test_save_and_load_metadata() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("metadata-test");
    let metadata = SessionMetadata::new()
        .with_agent_id(AgentId::new("researcher"))
        .with_extra(json!({"tenant": "acme"}));

    store.save_metadata(&session, &metadata).await.unwrap();
    let loaded = store.load_metadata(&session).await.unwrap();

    assert_eq!(loaded, Some(metadata));
    assert!(store.exists(&session).await.unwrap());
}

#[tokio::test]
async fn test_clear_session_removes_metadata() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("clear-metadata-test");
    let metadata = SessionMetadata::new().with_agent_id(AgentId::new("planner"));

    store.save_metadata(&session, &metadata).await.unwrap();
    store
        .save_messages(&session, &[ChatMessage::user("hello")])
        .await
        .unwrap();

    store.clear(&session).await.unwrap();

    assert!(store.load_messages(&session).await.unwrap().is_empty());
    assert!(store.load_metadata(&session).await.unwrap().is_none());
    assert!(!store.exists(&session).await.unwrap());
}

#[tokio::test]
async fn test_exists() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("exists-test");

    assert!(!store.exists(&session).await.unwrap());

    store
        .save_messages(&session, &[ChatMessage::user("hi")])
        .await
        .unwrap();

    assert!(store.exists(&session).await.unwrap());
}

#[tokio::test]
async fn test_multiple_sessions() {
    let store = InMemorySessionStore::new();
    let session_a = SessionId::new("session-a");
    let session_b = SessionId::new("session-b");

    store
        .save_messages(&session_a, &[ChatMessage::user("from A")])
        .await
        .unwrap();
    store
        .save_messages(&session_b, &[ChatMessage::user("from B")])
        .await
        .unwrap();

    let loaded_a = store.load_messages(&session_a).await.unwrap();
    let loaded_b = store.load_messages(&session_b).await.unwrap();

    assert_eq!(loaded_a.len(), 1);
    assert_eq!(loaded_a[0].content.as_deref(), Some("from A"));
    assert_eq!(loaded_b.len(), 1);
    assert_eq!(loaded_b[0].content.as_deref(), Some("from B"));
}

#[tokio::test]
async fn test_session_id_random() {
    let id_a = SessionId::random();
    let id_b = SessionId::random();
    assert_ne!(id_a, id_b, "two random session IDs should differ");
    assert!(id_a.as_str().starts_with("ses_"));
}

#[tokio::test]
async fn test_shared_store_across_clones() {
    let store = InMemorySessionStore::new();
    let store_clone = store.clone();
    let session = SessionId::new("shared-test");

    store
        .save_messages(&session, &[ChatMessage::user("original")])
        .await
        .unwrap();

    // The clone should see the same data (they share the underlying Arc).
    let loaded = store_clone.load_messages(&session).await.unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].content.as_deref(), Some("original"));
}

#[tokio::test]
async fn test_load_nonexistent_session_returns_empty() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("ghost");

    let loaded = store.load_messages(&session).await.unwrap();
    assert!(loaded.is_empty());
}

#[tokio::test]
async fn test_store_as_arc_dyn() {
    // Ensure InMemorySessionStore can be used as Arc<dyn MemoryStore>.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemorySessionStore::new());
    let session = SessionId::new("dyn-test");

    store
        .save_messages(&session, &[ChatMessage::user("dynamic dispatch")])
        .await
        .unwrap();

    let loaded = store.load_messages(&session).await.unwrap();
    assert_eq!(loaded.len(), 1);
}
