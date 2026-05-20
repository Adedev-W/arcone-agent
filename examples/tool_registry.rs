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
    let _ = dotenvy::dotenv();

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
        .ask("Use the quote tool and explain the current demo quote for AAPL.")
        .await?;

    println!("{}", response.content().unwrap_or(""));
    Ok(())
}
