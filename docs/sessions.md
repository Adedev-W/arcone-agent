# Sessions

Sessions move chat history out of a single `Agent` instance and into a
`MemoryStore`. The crate includes in-memory and PostgreSQL-backed stores.

Related docs: [Agents](agents.md), [Operations](operations.md),
[API reference](api-reference.md), [Examples](examples.md).

## Core Types

- `SessionId`: stable session key.
- `SessionMetadata`: timestamps, optional `AgentId`, and optional custom JSON.
- `MemoryStore`: async trait for saving, loading, clearing, and checking
  session data.
- `InMemorySessionStore`: thread-safe in-process store.
- `PostgresSessionStore`: durable store backed by PostgreSQL.
- `PostgresSessionConfig`: PostgreSQL connection settings for sessions.

## In-Memory Sessions

Use `InMemorySessionStore` for tests, examples, local tools, and short-lived
processes.

```rust
use std::sync::Arc;
use arcone_agent::{Agent, InMemorySessionStore, Result, SessionId};

async fn run() -> Result<()> {
    let store = Arc::new(InMemorySessionStore::new());
    let mut agent = Agent::from_env()?
        .session(SessionId::new("demo-session"), store);

    agent.ask_text("Remember that responses should be short.").await?;
    agent.ask_text("What did I ask you to remember?").await?;
    Ok(())
}
```

The in-memory store can be cloned and shared because it uses shared internal
state.

## PostgreSQL Sessions

Use `PostgresSessionStore` when conversation history must survive process
restarts.

```rust
use std::sync::Arc;
use arcone_agent::{
    Agent, PostgresSessionConfig, PostgresSessionStore, Result, SessionId,
};

async fn run() -> Result<()> {
    let store = Arc::new(
        PostgresSessionStore::connect(PostgresSessionConfig::from_env()?).await?,
    );

    let mut agent = Agent::from_env()?
        .session(SessionId::new("customer-123"), store);

    let answer = agent.ask_text("Remember this session uses PostgreSQL.").await?;
    println!("{answer}");
    Ok(())
}
```

`PostgresSessionStore::connect` validates the connection and runs the session
migration. The migration creates:

- `agent_sessions`: session metadata and timestamps.
- `agent_messages`: ordered JSONB chat messages.

## Session Metadata

Use `MemoryStore` directly when you need metadata without going through
`Agent`.

```rust
use arcone_agent::{
    AgentId, InMemorySessionStore, MemoryStore, Result, SessionId, SessionMetadata,
};
use serde_json::json;

async fn save_metadata() -> Result<()> {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("tenant-a-chat-1");
    let metadata = SessionMetadata::new()
        .with_agent_id(AgentId::new("support"))
        .with_extra(json!({ "tenant": "tenant-a" }));

    store.save_metadata(&session, &metadata).await?;
    Ok(())
}
```

## MemoryStore Trait

Implement the trait when you need Redis, object storage, another database, or a
custom retention policy.

```rust
pub trait MemoryStore: Send + Sync {
    fn save_messages(&self, id: &SessionId, messages: &[ChatMessage]) -> MemoryFuture<()>;
    fn load_messages(&self, id: &SessionId) -> MemoryFuture<Vec<ChatMessage>>;
    fn save_metadata(&self, id: &SessionId, metadata: &SessionMetadata) -> MemoryFuture<()>;
    fn load_metadata(&self, id: &SessionId) -> MemoryFuture<Option<SessionMetadata>>;
    fn clear(&self, id: &SessionId) -> MemoryFuture<()>;
    fn exists(&self, id: &SessionId) -> MemoryFuture<bool>;
}
```

## Lifecycle Behavior

- `Agent::ask` loads stored history before building the request.
- `Agent::ask` saves the updated history after receiving a final answer.
- `AgentStream::finish` saves the final streamed answer into the attached
  session store.
- `Agent::clear_history` only clears local runtime history.
- `Agent::clear_session` clears local history and the attached store entry.

## Session Best Practices

- Use stable `SessionId` values derived from your application domain, such as a
  conversation ID or tenant-scoped chat ID.
- Do not put secrets in `SessionMetadata::extra`; it is persisted as JSON.
- Use PostgreSQL sessions for durable applications and in-memory sessions for
  tests or ephemeral workflows.
- Keep session history bounded at the application layer when long conversations
  can exceed model context limits.
- Use one store instance shared by `Arc` rather than creating a new PostgreSQL
  pool per request.
