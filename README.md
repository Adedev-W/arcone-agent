# arcone-agent

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](/docs/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

`arcone-agent` is a Rust-first agent toolkit for building DeepSeek-powered
applications with typed tools, durable sessions, retrieval, multi-agent routing,
guardrails, answer composition, PostgreSQL storage, pgvector retrieval, and a
Python package backed by the same Rust core.

The project is designed around a small, typed runtime rather than a large
framework surface. Rust applications import the crate as `arcone_agent`, while
Python applications import the native extension package as `arcone_agent` after
building it with maturin.

## Rust Quickstart

Create a `.env` file from `.env.example` and set `DEEPSEEK_API_KEY`. The
high-level `Agent` API keeps conversation history, applies the configured model
options, and returns typed errors through `arcone_agent::Result`.

```rust
use arcone_agent::{Agent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = Agent::from_env()?
        .system("Answer clearly and keep responses concise.")
        .thinking_disabled()
        .max_tokens(256);

    let answer = agent.ask_text("What can arcone-agent do?").await?;
    println!("{answer}");

    Ok(())
}
```

Run the live example with:

```bash
cargo run --example deepseek_live -- "Explain arcone-agent in one sentence"
```

## Python Quickstart

The Python binding exposes the same core capabilities through an async-native
facade. It supports agent calls, response objects, sessions, RAG wrappers,
Python-defined tools, PostgreSQL/pgvector helpers, and streaming async
iterators.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.13,<2"
maturin develop
```

```python
import asyncio

from arcone_agent import Agent


async def main() -> None:
    agent = Agent.from_env(
        system="Answer clearly and keep responses concise.",
        thinking=False,
        max_tokens=256,
    )

    async for delta in agent.stream_text("Write a short project update."):
        print(delta, end="", flush=True)


asyncio.run(main())
```

Python examples for basic agents, sessions, RAG, tool calling, and streaming
live in `python/examples/`.

## What It Provides

At the core is a typed `Agent` runtime for normal chat, JSON responses,
streaming text, bounded tool loops, and session-backed history. Tools can be
defined as typed Rust functions, shared through a `ToolRegistry`, or registered
from Python with JSON-compatible callables.

For knowledge workflows, the crate includes document and chunk types,
in-memory knowledge stores, OpenAI embeddings, in-memory vector retrieval,
PostgreSQL-backed knowledge storage, and pgvector search. `KnowledgeAgent`
combines retrieval with a base agent and returns both the final answer and the
source metadata used to produce it.

Production-oriented pieces are built into the public API: typed error variants,
secret redaction helpers, opt-in tracing, guardrail pipelines, final answer
composition, durable session stores, database migrations, and multi-agent teams
with static, LLM, or custom routing.

## Documentation

Start with [Getting started](docs/getting-started.md) for the Rust API or
[Python binding](docs/python-binding.md) for the maturin package. The full
documentation index is in [docs/index.md](docs/index.md), with focused guides
for [agents](docs/agents.md), [tools](docs/tools.md),
[sessions](docs/sessions.md), [knowledge and retrieval](docs/knowledge-and-retrieval.md),
[multi-agent teams](docs/multi-agent.md), [guardrails and composition](docs/guardrails-and-composer.md),
[examples](docs/examples.md), [operations](docs/operations.md), and the
[API reference](docs/api-reference.md).

## Environment

Only configure the providers and stores you use. DeepSeek is required for agent
calls, OpenAI is required for the built-in embedder, and PostgreSQL is required
for durable sessions, knowledge storage, and pgvector retrieval.

```dotenv
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
# DEEPSEEK_MODEL=deepseek-v4-flash
# DEEPSEEK_BASE_URL=https://api.deepseek.com

OPENAI_API_KEY=sk-your-openai-api-key
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_BASE_URL=https://api.openai.com

DATABASE_URL=postgres://postgres:postgres@localhost:5432/arcone
# RUST_LOG=arcone_agent=info
```

## Development

Use Cargo for the Rust workspace and maturin for the Python package. The
provider-free tests cover the core runtime, retrieval, sessions, error mapping,
and the Python facade; live provider/database tests run when the matching
environment variables are present.

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
maturin develop
python -m pytest -q python/tests
python -m mypy python/examples python/tests
python python/benchmarks/wrapper_overhead.py
```

Build a release wheel with:

```bash
maturin build --release
```

Examples that call DeepSeek, OpenAI, PostgreSQL, or pgvector may incur provider
or database costs. See [Operations](docs/operations.md) for runtime
configuration, migrations, tracing, redaction, and release checks.
