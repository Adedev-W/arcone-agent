# arcone-agent Documentation

This documentation covers the public API and operational model of
`arcone-agent`, a Rust agent toolkit for DeepSeek chat models with typed tools,
stateful sessions, retrieval, multi-agent routing, guardrails, answer
composition, PostgreSQL integrations, pgvector retrieval, and Python bindings.

Start with [Getting started](getting-started.md) if you are using the Rust
crate for the first time. It walks through environment setup, the first agent
call, JSON output, streaming, and lower-level chat requests. Python users should
start with [Python binding](python-binding.md), which explains the maturin
package, async APIs, sessions, RAG, Python-defined tools, streaming iterators,
tests, and wheel builds.

The core runtime is documented in [Agents](agents.md). Tool calling is covered
in [Tools](tools.md), conversation persistence in [Sessions](sessions.md), and
retrieval workflows in [Knowledge and retrieval](knowledge-and-retrieval.md).
For larger systems, [Multi-agent teams](multi-agent.md) covers routing and
handoffs, while [Guardrails and answer composition](guardrails-and-composer.md)
explains input/output checks and final answer composition. Production setup,
database migrations, tracing, redaction, and release checks live in
[Operations](operations.md).

Use [Examples](examples.md) when you want runnable entry points, and use the
[API reference](api-reference.md) when you need grouped method and type
references for the public re-exports from `src/lib.rs`.

## Crate Shape

The package name is `arcone-agent`, and the Rust import path is
`arcone_agent`. The Python distribution name is `arcone-agent`, and the Python
import path is also `arcone_agent`.

The crate currently exposes one optional Rust feature:

```toml
[features]
tracing = ["dep:tracing"]
```

Related docs: [README](../README.md), [Python binding](python-binding.md),
[API reference](api-reference.md), [Examples](examples.md).
