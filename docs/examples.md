# Examples

The repository includes runnable Rust examples in `examples/` and Python
examples in `python/examples/`. Rust examples exercise the native crate
directly, while Python examples use the maturin-built `arcone_agent` package
backed by the same Rust core.

Related docs: [Getting started](getting-started.md), [Tools](tools.md),
[Knowledge and retrieval](knowledge-and-retrieval.md), [Python binding](python-binding.md),
and [Operations](operations.md).

## Rust Examples

Use `deepseek_live` for a minimal high-level agent call, and
`deepseek_stream_server` when you want to see `Agent::stream` exposed through a
small HTTP server. `tool_registry` demonstrates typed Rust tools and a shared
`ToolRegistry`; `multi_agent_basic` shows a simple team with `StaticRouter`;
and `portfolio_worker_agent` combines tools, guardrails, answer composition,
and team orchestration.

The knowledge examples show the retrieval stack at different levels.
`knowledge_keyword` implements a custom keyword retriever,
`knowledge_openai` uses OpenAI embeddings with in-memory vector retrieval, and
`knowledge_pgvector` uses PostgreSQL plus pgvector for durable semantic search.
`postgres_session` demonstrates durable conversation history with
`PostgresSessionStore`.

```bash
cargo run --example deepseek_live -- "Explain arcone-agent in one sentence"
cargo run --example tool_registry
cargo run --example multi_agent_basic
cargo run --example knowledge_keyword
```

Run an example with tracing enabled when you want structured operational logs:

```bash
cargo run --features tracing --example portfolio_worker_agent
```

Run the streaming HTTP server with:

```bash
STREAM_SERVER_ADDR=127.0.0.1:3000 cargo run --example deepseek_stream_server
curl -N -X POST http://127.0.0.1:3000/chat \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain arcone-agent briefly","max_tokens":256}'
```

## Python Examples

After `maturin develop`, the scripts in `python/examples/` can be run with the
active virtualenv. `basic_agent.py` performs a normal async agent call,
`session.py` shows session-backed memory, `rag.py` builds an in-memory
knowledge flow with OpenAI embeddings, `tool_calling.py` registers a Python
callable as a model tool, and `streaming.py` consumes streamed text deltas with
`async for`.

```bash
python python/examples/basic_agent.py
python python/examples/session.py
python python/examples/rag.py
python python/examples/tool_calling.py
python python/examples/streaming.py
```

The Python examples use the same environment variables as the Rust examples:
`DEEPSEEK_API_KEY` for agent calls, `OPENAI_API_KEY` for embeddings, and
`DATABASE_URL` when PostgreSQL-backed components are used.

## Representative Snippets

The Rust basic agent example uses `Agent::from_env`, configures a short system
prompt, and asks for text through `ask_text`.

```rust
use arcone_agent::{Agent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = Agent::from_env()?
        .system("Use concise language")
        .thinking_disabled()
        .max_tokens(256);

    let text = agent.ask_text("Explain arcone-agent.").await?;
    println!("{text}");

    Ok(())
}
```

The Python streaming example uses the binding's async iterator. Normal
iteration finalizes the stream; `finish()` retrieves the cached response object.

```python
stream = await agent.stream("Write a short status update.")
async for delta in stream:
    print(delta, end="", flush=True)
response = await stream.finish()
```

For durable vector search, create a shared PostgreSQL pool, migrate the
knowledge and pgvector tables, then index chunks with `PgVectorRetriever`.

```rust
let pool = PostgresPool::connect(PostgresStoreConfig::from_env()?).await?;
let retriever = PgVectorRetriever::new(
    pool,
    OpenAiEmbedder::from_env()?,
    PgVectorRetrieverOptions::default(),
);
retriever.migrate().await?;
retriever.index(chunks).await?;
```

## Testing Examples

Compile all Rust targets without making live provider calls:

```bash
cargo test --no-run --all-targets
```

Run the full Rust test suite with:

```bash
cargo test
```

Run Python provider-free tests after building the binding:

```bash
maturin develop
python -m pytest -q python/tests
```

Examples that call DeepSeek, OpenAI, PostgreSQL, or pgvector require matching
credentials and services. They may incur provider or database costs.
