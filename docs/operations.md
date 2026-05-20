# Operations

This guide covers runtime configuration, migrations, tracing, errors, and
production practices.

Related docs: [Getting started](getting-started.md), [Sessions](sessions.md),
[Knowledge and retrieval](knowledge-and-retrieval.md), [API reference](api-reference.md).

## Environment Variables

DeepSeek chat:

```dotenv
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
# DEEPSEEK_MODEL=deepseek-v4-flash
# DEEPSEEK_BASE_URL=https://api.deepseek.com
```

OpenAI embeddings:

```dotenv
OPENAI_API_KEY=sk-your-openai-api-key
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_BASE_URL=https://api.openai.com
```

PostgreSQL:

```dotenv
DATABASE_URL=postgres://postgres:postgres@localhost:5432/arcone
```

Examples and tracing:

```dotenv
# RUST_LOG=arcone_agent=info
# STREAM_SERVER_ADDR=127.0.0.1:3000
# MAX_TOKENS=256
```

## Provider Configuration

Use explicit config in services so timeouts and models are visible at startup.

```rust
use std::time::Duration;
use arcone_agent::{DeepSeekClient, DeepSeekConfig};

let client = DeepSeekClient::new(
    DeepSeekConfig::from_env()?
        .with_timeout(Duration::from_secs(120)),
)?;
```

Best practice: create clients during service startup and fail fast when required
environment variables are missing.

## PostgreSQL Migrations

Session storage:

- `PostgresSessionStore::connect(config).await` validates the connection and
  runs `migrate()`.
- `PostgresSessionStore::new(config)` builds the pool but does not validate the
  connection until used.

Knowledge storage:

- `PostgresKnowledgeBase::migrate().await` creates document and chunk tables.
- `PostgresKnowledgeBase::connect(config).await` connects and migrates.

pgvector retrieval:

- `PgVectorRetriever::migrate().await` creates embedding tables and vector
  indexes.
- The PostgreSQL database must have the `pgvector` extension available.
- `PgVectorRetrieverOptions::embedding_dimension` must match the embedder model.

Best practice: run migrations at application startup or deployment time before
serving traffic. Keep the database user permissions aligned with that choice.

## Tracing

Enable the optional `tracing` feature:

```bash
cargo run --features tracing --example multi_agent_basic
```

The crate emits structured events for:

- DeepSeek chat and streaming calls.
- Agent requests and tool calls.
- Team routing and handoff completion.
- Knowledge retrieval and embedding calls.
- PostgreSQL connection and query operations.
- Guardrail pipeline checks.
- Answer composition.

Logs intentionally avoid API keys, prompts, tool argument bodies, tool output
bodies, retrieved content, and database record IDs. The `Debug` output for
provider configs redacts API keys.

## Secret Redaction

`redact_secret` always returns a redacted placeholder:

```rust
use arcone_agent::redact_secret;

assert_eq!(redact_secret("sk-live-secret"), "<redacted>");
```

Best practice: never log raw environment values, prompts, retrieved chunks, or
tool outputs unless your application has an explicit sanitization and retention
policy.

## Error Handling

All public operations return `arcone_agent::Result<T>`. Match error variants
when the application can recover.

```rust
use arcone_agent::{Error, Result};

async fn run() -> Result<()> {
    match agent.ask_text("Use a missing tool.").await {
        Ok(text) => println!("{text}"),
        Err(Error::UnknownTool(name)) => eprintln!("tool not registered: {name}"),
        Err(error) => return Err(error),
    }

    Ok(())
}
```

Useful operational variants include:

- `MissingApiKey`, `MissingOpenAiApiKey`, `MissingDatabaseUrl`
- `Api`, `OpenAiApi`
- `Timeout`, `OpenAiTimeout`
- `InvalidToolArguments`, `ToolLoopExceeded`
- `RoutingFailure`, `HandoffLoopExceeded`
- `DuplicateDocument`, `EmbeddingFailure`, `RetrievalFailure`
- `DatabaseConnection`, `DatabasePool`, `DatabaseMigration`
- `GuardrailBlocked`

## Database Pooling

`PostgresStoreConfig` and `PostgresSessionConfig` default to a max pool size of
16 and a 5 second connect timeout.

```rust
use std::time::Duration;
use arcone_agent::PostgresStoreConfig;

let config = PostgresStoreConfig::from_env()?
    .with_max_pool_size(32)
    .with_connect_timeout(Some(Duration::from_secs(5)))
    .with_statement_timeout(Some(Duration::from_secs(10)));
```

Best practice: share one pool per database and workload class. Avoid creating a
new pool for every request.

## Production Best Practices

- Bound model calls with explicit timeouts.
- Centralize provider model and base URL configuration.
- Use `max_tokens`, `max_tool_rounds`, and `KnowledgeAgentOptions` to bound
  request cost and context size.
- Use typed tools and validate external input before side effects.
- Keep tools idempotent where practical.
- Use durable sessions only for conversations that need persistence.
- Use pgvector with explicit embedding dimensions and migration checks.
- Attach guardrails at the stage where the risk appears: input, retrieved
  context, or output.
- Keep debug metadata out of user-facing responses unless sanitized.
- Run `cargo test --no-run --all-targets` before release to compile tests and
  examples.
