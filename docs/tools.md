# Tools

Tools let a model call Rust functions during a non-streaming `Agent::ask` loop.
`arcone-agent` supports low-level dynamic tools and typed tools that generate
JSON schema from Rust structs.

Related docs: [Agents](agents.md), [Examples](examples.md),
[API reference](api-reference.md), [Operations](operations.md).

## Recommended: TypedFunctionTool

Use `TypedFunctionTool` when arguments and output can be represented by Rust
types. Arguments implement `Deserialize` and `JsonSchema`; output implements
`Serialize`.

```rust
use std::sync::Arc;
use arcone_agent::{
    Agent, DeepSeekClient, Result, ThinkingConfig, ToolRegistry, TypedFunctionTool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct QuoteArgs {
    symbol: String,
}

#[derive(Serialize)]
struct Quote {
    symbol: String,
    price: f64,
    currency: &'static str,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let quote_tool = TypedFunctionTool::<QuoteArgs, Quote, _>::json(
        "quote",
        "Return a demo market quote for a ticker symbol.",
        |args: QuoteArgs| async move {
            Ok(Quote {
                symbol: args.symbol.to_uppercase(),
                price: 128.40,
                currency: "USD",
            })
        },
    )?;

    let mut registry = ToolRegistry::new();
    registry.add_tool(quote_tool)?;

    let mut agent = Agent::new(DeepSeekClient::from_env()?)
        .with_system_prompt("Use tools when exact portfolio facts are requested.")
        .with_thinking(ThinkingConfig::disabled())
        .with_tool_registry(Arc::new(registry))?;

    let response = agent
        .ask("Use the quote tool and explain the demo quote for AAPL.")
        .await?;

    println!("{}", response.content().unwrap_or(""));
    Ok(())
}
```

If the model provides arguments that cannot deserialize into `Args`, the call
returns `Error::InvalidToolArguments`.

## Shared ToolRegistry

`ToolRegistry` stores `Arc<dyn Tool>` values by function name. Reuse one
registry across several agents.

```rust
use std::sync::Arc;
use arcone_agent::{Agent, ToolRegistry};

let mut registry = ToolRegistry::new();
registry.add_tool(lookup_tool)?;
registry.add_tool(search_tool)?;

let shared = Arc::new(registry);
let mut analyst = analyst.with_tool_registry(Arc::clone(&shared))?;
let mut reviewer = reviewer.with_tool_registry(shared)?;
```

Best practice: use a shared registry for common infrastructure tools and private
agent tools for role-specific operations.

## Low-Level FunctionTool

Use `FunctionTool` when you need full control over the JSON argument shape.

```rust
use arcone_agent::{FunctionDefinition, FunctionTool, ToolDefinition};
use serde_json::{json, Value};

let definition = ToolDefinition::function(
    FunctionDefinition::new("lookup")
        .description("Look up a value by key.")
        .parameters(json!({
            "type": "object",
            "properties": {
                "key": { "type": "string" }
            },
            "required": ["key"]
        })),
);

let tool = FunctionTool::new(definition, |arguments: Value| async move {
    let key = arguments
        .get("key")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(format!("value for {key}"))
});
```

Best practice: prefer `TypedFunctionTool` unless a provider-specific schema or
dynamic argument payload is required.

## Custom Tool Implementations

Implement `Tool` directly when the tool owns state, clients, caches, or custom
call behavior.

```rust
use arcone_agent::{Result, Tool, ToolDefinition, ToolFuture};
use serde_json::Value;

struct MyTool {
    definition: ToolDefinition,
}

impl Tool for MyTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn call(&self, arguments: Value) -> ToolFuture {
        Box::pin(async move {
            Ok(format!("received arguments: {arguments}"))
        })
    }
}
```

## Tool Method Reference

- `Tool::definition() -> ToolDefinition`
- `Tool::call(arguments: serde_json::Value) -> ToolFuture`
- `ToolRegistry::new()`
- `ToolRegistry::add_tool(tool) -> Result<&mut Self>`
- `ToolRegistry::get(name) -> Option<Arc<dyn Tool>>`
- `ToolRegistry::definitions() -> Vec<ToolDefinition>`
- `FunctionTool::new(definition, handler)`
- `TypedFunctionTool::json(name, description, handler) -> Result<Self>`
- `Agent::try_with_tool(tool) -> Result<Self>`
- `Agent::try_add_tool(tool) -> Result<&mut Self>`
- `Agent::with_tool_registry(registry) -> Result<Self>`
- `Agent::set_tool_registry(registry) -> Result<&mut Self>`

## Tool Best Practices

- Use stable, descriptive tool names. Duplicate names return
  `Error::DuplicateTool` on checked paths.
- Keep tool descriptions short and precise; they are part of the model prompt.
- Validate and normalize tool arguments inside the tool, even when using typed
  deserialization.
- Return compact JSON or text. Large tool outputs increase context cost and can
  reduce answer quality.
- Keep tools idempotent when possible. Model retries and repeated calls can
  happen in real workflows.
- Bound external I/O with timeouts in stateful custom tools.
- Do not use high-level streaming for tool-calling agents; use `Agent::ask`.
