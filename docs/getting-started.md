# Getting Started

This guide shows the quickest path from environment setup to a working
`arcone-agent` call.

Related docs: [Agents](agents.md), [Tools](tools.md), [Operations](operations.md),
[API reference](api-reference.md).

## Requirements

- A Rust toolchain that supports edition 2024.
- `tokio` for async runtimes.
- `DEEPSEEK_API_KEY` for chat and agent calls.
- `OPENAI_API_KEY` only when using `OpenAiEmbedder`.
- `DATABASE_URL` only when using PostgreSQL sessions, durable knowledge storage,
  or pgvector retrieval.

For local examples, copy `.env.example` to `.env` and fill the values you need.

## Add The Crate

For a local project that depends on this repository:

```toml
[dependencies]
arcone-agent = { path = "../arcone_agent" }
tokio = { version = "1", features = ["full"] }
```

If your application loads `.env` files, also add:

```toml
dotenvy = "0.15"
```

## First Agent

`Agent::from_env()` reads `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL`, and
optional `DEEPSEEK_MODEL`.

```rust
use arcone_agent::{Agent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let mut agent = Agent::from_env()?
        .system("Answer clearly and keep responses concise.")
        .thinking_disabled()
        .max_tokens(256);

    let text = agent.ask_text("Explain arcone-agent in one paragraph.").await?;
    println!("{text}");

    Ok(())
}
```

Use `ask_text` when you only need the final assistant text. Use `ask` when you
need the full `AgentResponse`, including finish reason, usage, history,
guardrail events, and composed answer metadata.

## Explicit Client Configuration

Use `DeepSeekConfig` when you need custom timeout, base URL, model, or default
reasoning settings.

```rust
use std::time::Duration;
use arcone_agent::{
    Agent, DeepSeekClient, DeepSeekConfig, DeepSeekModel, ReasoningEffort,
    ThinkingConfig,
};

let config = DeepSeekConfig::from_env()?
    .with_model(DeepSeekModel::V4Flash)
    .with_timeout(Duration::from_secs(120))
    .with_default_thinking(Some(ThinkingConfig::enabled()))
    .with_default_reasoning_effort(Some(ReasoningEffort::High));

let client = DeepSeekClient::new(config)?;
let mut agent = Agent::new(client).system("Use concise technical language.");
```

Best practice: create one `DeepSeekClient` and clone it into agents that share
the same provider configuration. The client is cheap to clone and keeps provider
configuration centralized.

## JSON Responses

`ask_json` temporarily sets `ResponseFormat::json_object()` for the request and
deserializes the assistant response into your type.

```rust
use arcone_agent::{Agent, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Summary {
    title: String,
    risk: String,
}

async fn summarize(agent: &mut Agent) -> Result<Summary> {
    agent
        .ask_json("Return JSON with fields title and risk for pgvector retrieval.")
        .await
}
```

Best practice: keep prompts explicit about the expected JSON fields, and treat
model output as untrusted input. Deserialization can fail and should be handled
through `arcone_agent::Result`.

## Streaming Text

Use `Agent::stream` for token-by-token text. Finish the stream to persist the
assistant message into history and attached sessions.

```rust
use arcone_agent::{Agent, Result};

async fn stream_answer(agent: &mut Agent) -> Result<()> {
    let mut stream = agent.stream("Write a short status update.").await?;

    while let Some(delta) = stream.next_text().await? {
        print!("{delta}");
    }

    let response = stream.finish().await?;
    eprintln!("finish_reason={:?}", response.finish_reason);
    Ok(())
}
```

Streaming does not run the tool-calling loop. If a streamed response contains
tool calls, the API returns `Error::StreamingToolCallsUnsupported`. Use
non-streaming `ask` for tool-enabled agents.

## Low-Level Chat Requests

The high-level agent API is recommended for application code, but low-level
request types remain public for protocol-level use.

```rust
use arcone_agent::{
    ChatMessage, ChatRequest, DeepSeekClient, DeepSeekModel, ThinkingConfig,
};

let client = DeepSeekClient::from_env()?;
let request = ChatRequest::new(
    DeepSeekModel::default(),
    vec![ChatMessage::user("What is retrieval augmented generation?")],
)
.with_thinking(ThinkingConfig::disabled())
.with_max_tokens(256);

let response = client.chat(request).await?;
let content = response.first_message().and_then(|message| message.content.as_deref());
```

Best practice: prefer `Agent` unless you need exact request-shape control,
manual message construction, or protocol-level testing.
