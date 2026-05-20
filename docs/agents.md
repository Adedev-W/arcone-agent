# Agents

`Agent` is the main runtime abstraction. It owns provider access, runtime
history, private tools, optional shared tools, optional session storage,
optional guardrails, and optional answer composition.

Related docs: [Getting started](getting-started.md), [Tools](tools.md),
[Sessions](sessions.md), [Guardrails and composer](guardrails-and-composer.md),
[API reference](api-reference.md).

## Core Types

- `Agent`: mutable runtime for asking, streaming, tool calls, and history.
- `AgentConfig`: reusable configuration for model, prompt, role, and response
  controls.
- `AgentProfile`: identity metadata used by single agents and team routers.
- `AgentId`: stable identifier for a role or runtime agent.
- `AgentOptions`: legacy ergonomic config converted into `AgentConfig`.
- `AgentResponse`: full response object returned by `Agent::ask`.
- `AgentStream`: high-level streaming wrapper returned by `Agent::stream`.

## Create An Agent From Environment

```rust
use arcone_agent::{Agent, ReasoningEffort, Result};

async fn run() -> Result<String> {
    let mut agent = Agent::from_env()?
        .system("Use precise language.")
        .thinking_enabled()
        .reasoning(ReasoningEffort::High)
        .max_tokens(512);

    agent.ask_text("What should I validate first?").await
}
```

The short chain aliases map to the longer builder methods:

- `system(...)` calls `with_system_prompt(...)`.
- `model(...)` calls `with_model(...)`.
- `reasoning(...)` calls `with_reasoning_effort(...)`.
- `max_tokens(...)` calls `with_max_tokens(...)`.
- `tool(...)` calls `with_tool(...)`.
- `session(...)` calls `with_session(...)`.

## Reusable Role Configuration

Use `AgentConfig` when agents need stable IDs and role metadata, especially in
multi-agent teams.

```rust
use arcone_agent::{Agent, AgentConfig, DeepSeekClient, ThinkingConfig};

let client = DeepSeekClient::from_env()?;
let config = AgentConfig::new("researcher")
    .with_name("Researcher")
    .with_role_description("Finds facts and constraints")
    .with_system_prompt("Answer with concise evidence.")
    .with_thinking(ThinkingConfig::disabled())
    .with_max_tokens(400);

let mut agent = Agent::with_config(client, config);
```

Best practice: keep role descriptions short and operational. Team routers use
`AgentProfile` metadata, so vague descriptions produce worse routing decisions.

## Asking

Use the response method that matches the data you need:

```rust
let full = agent.ask("Explain the migration path.").await?;
let text = agent.ask_text("Explain the migration path.").await?;
let json: serde_json::Value = agent.ask_json("Return JSON only.").await?;
```

`AgentResponse::content()` returns the composed answer text when an
`AnswerComposer` is attached, otherwise it returns the assistant message content.
`AgentResponse::reasoning_content()` returns reasoning content when the provider
returned it.

## History And Sessions

An agent stores local chat history in memory. Attach a `MemoryStore` when history
must survive across agent instances.

```rust
use std::sync::Arc;
use arcone_agent::{Agent, InMemorySessionStore, SessionId};

let store = Arc::new(InMemorySessionStore::new());
let mut agent = Agent::from_env()?
    .session(SessionId::new("customer-123"), store);

agent.ask_text("Remember that I prefer short answers.").await?;
agent.clear_session().await?;
```

`clear_history()` clears only local runtime history. `clear_session().await`
clears local history and the attached session store entry.

## Tools

Agents can own private tools and can also use a shared `ToolRegistry`.

```rust
let mut agent = Agent::from_env()?
    .thinking_disabled()
    .try_with_tool(my_tool)?;

agent.set_tool_registry(shared_registry)?;
```

Use `try_with_tool`, `try_add_tool`, `with_tool_registry`, or
`set_tool_registry` when duplicate tool names should be reported as
`Error::DuplicateTool`. The legacy `with_tool` and `add_tool` helpers keep
private tool insertion simple but do not validate duplicates.

## Streaming

```rust
let mut stream = agent.stream("Draft a short changelog entry.").await?;

while let Some(delta) = stream.next_text().await? {
    print!("{delta}");
}

let response = stream.finish().await?;
```

Call `finish()` even after all chunks are consumed. It appends the final
assistant message to history and saves it through the attached session store.

Best practice: use streaming for plain text responses and non-streaming `ask`
for tool-enabled agents, because streaming tool calls are intentionally rejected
with `Error::StreamingToolCallsUnsupported`.

## Agent Best Practices

- Prefer `Agent::from_env()` for simple apps and explicit `DeepSeekConfig` for
  services that need timeouts or custom base URLs.
- Keep one reusable `DeepSeekClient` per provider configuration and clone it
  into multiple agents.
- Use `AgentConfig` for named roles and multi-agent routing.
- Use `max_tool_rounds` to bound tool loops.
- Use `ask_json` only with clear schema prompts and typed deserialization.
- Attach sessions only when conversation history must persist outside the
  current `Agent` value.
