# Python Binding

This guide documents the `arcone_agent` Python package backed by the Rust
`arcone-agent` core. The binding is implemented with PyO3 and maturin, while
agent, session, knowledge, embedding, and retrieval logic remain in Rust.

Related docs: [Getting started](getting-started.md), [Sessions](sessions.md),
[Knowledge and retrieval](knowledge-and-retrieval.md), [Operations](operations.md),
[API reference](api-reference.md).

## Python Package Surface

The Python package exposes the implemented production path for the Rust core:
async agent calls, response conversion, typed exceptions, in-memory and
PostgreSQL sessions, document/chunk wrappers, OpenAI embeddings, in-memory
retrieval, PostgreSQL knowledge storage, pgvector retrieval, Python-defined
tools, and streaming async iterators.

The public surface is intentionally smaller than the Rust crate. Python users
work with facade classes such as `Agent`, `KnowledgeAgent`,
`InMemoryKnowledgeBase`, `PostgresKnowledgeBase`, `InMemoryVectorRetriever`,
`PgVectorRetriever`, and `AgentStream` instead of Rust traits, lifetimes, or
generic types. Custom Python retriever callbacks are not exposed yet.

## Development Install

Create a virtualenv and install the editable native extension:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.13,<2" "pytest>=8" "pytest-asyncio>=0.24" "mypy>=1.10"
maturin develop
```

Smoke test the import:

```bash
python -c "import arcone_agent; print(arcone_agent.runtime_info())"
```

The package distribution name is `arcone-agent`; the import name is
`arcone_agent`. The same package can be built as a release wheel with
`maturin build --release`.

## Environment

The binding reads the same environment variables as the Rust core:

```dotenv
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
# DEEPSEEK_MODEL=deepseek-v4-flash
# DEEPSEEK_BASE_URL=https://api.deepseek.com

OPENAI_API_KEY=sk-your-openai-api-key
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_BASE_URL=https://api.openai.com

DATABASE_URL=postgres://postgres:postgres@localhost:5432/arcone
```

`DEEPSEEK_API_KEY` is required for `Agent.from_env` and
`KnowledgeAgent.from_env`. `OPENAI_API_KEY` is required for `OpenAiEmbedder`.
`DATABASE_URL` is required for `PostgresSessionStore.from_env` and
`PostgresPool.from_env`.

The Python binding reads the process environment through the Rust core. It does
not load `.env` files automatically. If you keep credentials in `.env`, export
them before running Python scripts:

```bash
set -a
. .env
set +a
```

As an application-level alternative, install `python-dotenv` and call
`load_dotenv()` before `Agent.from_env()` or any other `from_env` constructor:

```python
from dotenv import load_dotenv

load_dotenv()
```

DeepSeek requests default to thinking mode. Leave `thinking` unset, or pass
`thinking=True`, for normal reasoning requests. Use `thinking=False` only for
explicit non-thinking calls; in that mode the binding omits
`reasoning_effort`, and the response will not include `reasoning_content`.

## Basic Agent

```python
import asyncio

from arcone_agent import Agent


async def main() -> None:
    agent = Agent.from_env(
        system="Answer clearly and keep responses concise.",
        thinking=True,
        max_tokens=256,
    )

    text = await agent.ask_text("Explain arcone-agent in one paragraph.")
    print(text)


asyncio.run(main())
```

Use `ask_text` when you only need the assistant text. Use `ask` when you need
finish reason, usage, reasoning text, or conversation history.

```python
import asyncio

from arcone_agent import Agent


async def main() -> None:
    agent = Agent.from_env(thinking=True, max_tokens=256)
    response = await agent.ask("Explain session memory.")
    print(response.content)
    print(response.finish_reason)
    print(response.usage)


asyncio.run(main())
```

Best practice: `Agent` is stateful and stores history. Reuse one agent for a
single conversation, and create separate agents for independent conversations.
Calls on the same agent are serialized internally to protect mutable Rust state.

## Python Tools

Register Python callables as model tools with a JSON schema. Sync and async
callables are supported.

```python
import asyncio

from arcone_agent import Agent


async def lookup_price(args: dict) -> dict:
    return {
        "symbol": args["symbol"].upper(),
        "price": 128.40,
        "currency": "USD",
    }


async def main() -> None:
    agent = Agent.from_env(
        system="Use tools when they help answer market questions.",
        thinking=True,
        max_tokens=256,
    )
    agent.add_tool(
        name="lookup_price",
        description="Return a demo market quote for a ticker symbol.",
        schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        handler=lookup_price,
    )

    print(await agent.ask_text("What is the demo quote for ACME?"))


asyncio.run(main())
```

Tool schemas and return values must be JSON serializable. Exceptions raised by
the Python callable are reported as `ToolError`.

## Streaming

Use `stream_text` when you want an async iterator directly:

```python
import asyncio

from arcone_agent import Agent


async def main() -> None:
    agent = Agent.from_env(thinking=True, max_tokens=256)

    stream = agent.stream_text("Write a short status update.")
    async for delta in stream:
        print(delta, end="", flush=True)

    response = await stream.finish()
    print(f"\nfinish_reason={response.finish_reason}")


asyncio.run(main())
```

Use `stream` when you want the stream-open step to be awaited before iteration:

```python
import asyncio

from arcone_agent import Agent


async def main() -> None:
    agent = Agent.from_env(thinking=True, max_tokens=256)
    stream = await agent.stream("Write a short status update.")

    async for delta in stream:
        print(delta, end="", flush=True)

    response = await stream.finish()
    print(f"\nfinish_reason={response.finish_reason}")


asyncio.run(main())
```

Normal iteration automatically finalizes the stream when the provider sends
`[DONE]`, so `finish()` can be called afterward to retrieve the cached
`AgentResponse`. If you stop early and want to cancel, call `stream.close()`;
partial assistant text is not persisted to history or session storage.

Streaming is for plain text responses. If a streamed response contains tool
calls, the binding raises `StreamingUnsupportedError`, which is a subclass of
`ToolError`. Use non-streaming `ask` for tool-enabled agents.

## Sessions

In-memory session:

```python
import asyncio

from arcone_agent import Agent, InMemorySessionStore


async def main() -> None:
    store = InMemorySessionStore()
    agent = Agent.from_env(
        session_id="demo-user",
        session_store=store,
        thinking=True,
        max_tokens=128,
    )
    print(await agent.ask_text("Remember that this session is in memory."))


asyncio.run(main())
```

PostgreSQL session:

```python
import asyncio

from arcone_agent import Agent, PostgresSessionStore


async def main() -> None:
    store = await PostgresSessionStore.from_env(
        max_pool_size=16,
        connect_timeout_seconds=5.0,
    )

    agent = Agent.from_env(
        session_id="demo-user",
        session_store=store,
        thinking=True,
        max_tokens=128,
    )
    print(await agent.ask_text("Persist this message in PostgreSQL."))


asyncio.run(main())
```

If `session_store` is provided, `session_id` is required. If only `session_id`
is provided, the binding creates a fresh in-memory store.

## Knowledge And Retrieval

Build an in-memory knowledge base, chunk documents, index them with OpenAI
embeddings, and ask a RAG-backed agent:

```python
import asyncio

from arcone_agent import (
    Agent,
    Document,
    InMemoryKnowledgeBase,
    InMemoryVectorRetriever,
    KnowledgeAgent,
    OpenAiEmbedder,
)


async def main() -> None:
    knowledge = InMemoryKnowledgeBase(max_chars=1200, overlap_chars=120)
    chunks = await knowledge.add_document(
        Document.text(
            "docs-python-binding",
            "arcone-agent exposes Rust agent capabilities through Python.",
            title="Python Binding Notes",
            source="local",
            metadata={"team": "platform"},
        )
    )

    retriever = InMemoryVectorRetriever(OpenAiEmbedder.from_env())
    await retriever.index(chunks)

    base_agent = Agent.from_env(
        system="Answer only from provided context.",
        thinking=True,
        max_tokens=256,
    )
    agent = KnowledgeAgent.from_agent(base_agent, retriever, top_k=4)

    response = await agent.ask("What does the Python binding expose?")
    print(response.content)
    for source in response.sources:
        print(source.index, source.title, source.score)


asyncio.run(main())
```

Best practice: keep chunk sizes large enough to preserve meaning and small
enough to stay inside model context. Start with `max_chars=1200` and
`overlap_chars=120`, then tune from retrieval quality and token budget.

## PostgreSQL And pgvector

Use a shared pool for durable document/chunk storage and vector retrieval:

```python
import asyncio

from arcone_agent import (
    Document,
    OpenAiEmbedder,
    PgVectorRetriever,
    PostgresKnowledgeBase,
    PostgresPool,
)


async def main() -> None:
    pool = await PostgresPool.from_env(statement_timeout_seconds=10.0)

    knowledge = PostgresKnowledgeBase(pool, max_chars=1200, overlap_chars=120)
    await knowledge.migrate()
    chunks = await knowledge.add_document(
        Document.text(
            "pgvector-overview",
            "pgvector stores embeddings in PostgreSQL.",
            title="pgvector Overview",
            source="local",
        )
    )

    retriever = PgVectorRetriever(
        pool,
        OpenAiEmbedder.from_env(),
        embedding_dimension=1536,
        metric="cosine",
        index_mode="auto",
    )
    await retriever.migrate()
    await retriever.index(chunks)


asyncio.run(main())
```

`metric` accepts `"cosine"` or `"l2"`. `index_mode` accepts `"auto"`, `"hnsw"`,
or `"none"`. Live PostgreSQL tests are opt-in through `DATABASE_URL`.

## Error Handling

Rust errors are mapped into a small Python exception hierarchy. `ArconeError`
is the base class; configuration problems become `ConfigError`, provider
failures become `ApiError` or `TimeoutError`, tool-loop failures become
`ToolError`, streamed tool calls become `StreamingUnsupportedError`, session
state becomes `SessionError`, PostgreSQL failures become `DatabaseError`,
knowledge-store failures become `KnowledgeError`, and embedding/retrieval
failures become `RetrievalError`.

Example:

```python
from arcone_agent import Agent, ConfigError

try:
    agent = Agent.from_env()
except ConfigError as exc:
    print(f"Configuration problem: {exc}")
```

Best practice: catch specific exceptions at application boundaries and avoid
logging secrets. Provider keys are read from environment variables and are not
included in binding exception messages.

## Method Reference

### `Agent`

- `Agent.from_env(...) -> Agent`
- `await agent.ask_text(prompt: str) -> str`
- `await agent.ask(prompt: str) -> AgentResponse`
- `await agent.stream(prompt: str) -> AgentStream`
- `agent.stream_text(prompt: str) -> AgentStream`
- `agent.clear_history() -> None`
- `await agent.clear_session() -> None`
- `agent.add_tool(name, description, schema, handler) -> None`

`Agent.from_env` accepts `system`, `model`, `thinking`, `reasoning_effort`,
`max_tokens`, `max_tool_rounds`, `session_id`, and `session_store`.

### `AgentResponse`

Properties:

- `content: str | None`
- `reasoning_content: str | None`
- `finish_reason: str`
- `usage: dict | None`
- `history: list[dict]`

### `AgentStream`

- `async for delta in stream -> str`
- `await stream.finish() -> AgentResponse`
- `stream.close() -> None`

### `Document`

- `Document.text(id, content, title=None, source=None, path=None, metadata=None)`

Properties:

- `id`
- `content`
- `title`
- `source`
- `path`
- `metadata`

### `InMemoryKnowledgeBase`

- `InMemoryKnowledgeBase(max_chars=1200, overlap_chars=120)`
- `await add_document(document) -> list[KnowledgeChunk]`
- `await list_documents() -> list[Document]`
- `await remove_document(document_id) -> bool`
- `await chunk_document(document) -> list[KnowledgeChunk]`
- `await chunks_for_document(document_id) -> list[KnowledgeChunk]`
- `await chunks_for_source(source) -> list[KnowledgeChunk]`

### `PostgresPool`

- `await PostgresPool.from_env(max_pool_size=16, connect_timeout_seconds=5.0, statement_timeout_seconds=None)`

### `PostgresKnowledgeBase`

- `PostgresKnowledgeBase(pool, max_chars=1200, overlap_chars=120)`
- `await migrate() -> None`
- `await add_document(document) -> list[KnowledgeChunk]`
- `await list_documents() -> list[Document]`
- `await remove_document(document_id) -> bool`
- `await chunk_document(document) -> list[KnowledgeChunk]`
- `await chunks_for_document(document_id) -> list[KnowledgeChunk]`
- `await chunks_for_source(source) -> list[KnowledgeChunk]`

### `OpenAiEmbedder`

- `OpenAiEmbedder.from_env(model=None, base_url=None, timeout_seconds=60.0)`

### `InMemoryVectorRetriever`

- `InMemoryVectorRetriever(embedder)`
- `await index(chunks) -> None`
- `await retrieve(query, top_k) -> list[ScoredChunk]`
- `len() -> int`
- `is_empty() -> bool`

### `PgVectorRetriever`

- `PgVectorRetriever(pool, embedder, embedding_dimension=1536, metric="cosine", index_mode="auto")`
- `await migrate() -> None`
- `await index(chunks) -> None`
- `await retrieve(query, top_k) -> list[ScoredChunk]`

### `KnowledgeAgent`

- `KnowledgeAgent.from_env(retriever, ..., top_k=4, max_context_chars=6000)`
- `KnowledgeAgent.from_agent(agent, retriever, top_k=4, max_context_chars=6000)`
- `await ask(question) -> KnowledgeAgentResponse`

`from_agent` consumes the original `Agent`. After it is moved into a
`KnowledgeAgent`, do not call methods on the original object.

## Examples And Testing

Runnable Python examples live in `python/examples/`. They cover the basic
agent flow, in-memory sessions, RAG, Python-defined tools, and streaming text.
The examples use the same environment variables as the Rust core and can be run
after `maturin develop`.

Run Rust compile verification:

```bash
cargo test --workspace --all-targets --no-run
```

Run Python tests after `maturin develop`:

```bash
python -m pytest -q python/tests
python -m mypy python/examples python/tests
python python/benchmarks/wrapper_overhead.py
```

Live provider tests are gated by environment variables. Without API keys, the
provider-free tests still validate import, config error mapping, document
wrappers, async knowledge base methods, session construction, and local fake-SSE
streaming.

Build a release wheel:

```bash
maturin build --release
python -m venv /tmp/arcone-agent-wheel-smoke
/tmp/arcone-agent-wheel-smoke/bin/python -m pip install target/wheels/arcone_agent-*.whl
/tmp/arcone-agent-wheel-smoke/bin/python -c "import arcone_agent; print(arcone_agent.runtime_info())"
```

Example scripts live in `python/examples/`:

- `basic_agent.py`
- `session.py`
- `rag.py`
- `tool_calling.py`
- `streaming.py`
