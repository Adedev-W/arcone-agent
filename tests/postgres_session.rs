use arcone_agent::{
    AgentId, ChatMessage, MemoryStore, PostgresSessionConfig, PostgresSessionStore, SessionId,
    SessionMetadata,
};
use serde_json::json;

#[tokio::test]
async fn postgres_session_store_persists_messages_and_metadata() {
    dotenvy::dotenv().ok();
    let Ok(database_url) = std::env::var("DATABASE_URL") else {
        return;
    };

    let store = PostgresSessionStore::connect(
        PostgresSessionConfig::new(database_url).with_max_pool_size(4),
    )
    .await
    .expect("postgres session store");
    store.migrate().await.expect("idempotent migration");

    let session = SessionId::new(format!("test-session-{}", unique_suffix()));
    let metadata = SessionMetadata::new()
        .with_agent_id(AgentId::new("postgres-agent"))
        .with_extra(json!({"kind": "integration"}));
    let messages = vec![
        ChatMessage::user("hello postgres"),
        ChatMessage::assistant("hello user"),
    ];

    store.save_metadata(&session, &metadata).await.unwrap();
    store.save_messages(&session, &messages).await.unwrap();

    let loaded_metadata = store.load_metadata(&session).await.unwrap();
    let loaded_messages = store.load_messages(&session).await.unwrap();

    assert_eq!(loaded_metadata, Some(metadata));
    assert_eq!(loaded_messages, messages);
    assert!(store.exists(&session).await.unwrap());

    store.clear(&session).await.unwrap();

    assert!(store.load_messages(&session).await.unwrap().is_empty());
    assert!(store.load_metadata(&session).await.unwrap().is_none());
    assert!(!store.exists(&session).await.unwrap());
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
