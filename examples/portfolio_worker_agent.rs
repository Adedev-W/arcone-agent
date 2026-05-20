use std::sync::Arc;

use arcone_agent::{
    Agent, AgentConfig, AgentTeam, DeepSeekClient, DefaultAnswerComposer, EmptyAnswerGuardrail,
    GuardrailPipeline, PrivateInfoRedactionGuardrail, Result, StaticRouter, ThinkingConfig,
    ToolRegistry, TypedFunctionTool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct HoldingArgs {
    symbol: String,
}

#[derive(Serialize)]
struct Holding {
    symbol: String,
    shares: f64,
    average_cost: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let holding_tool = TypedFunctionTool::<HoldingArgs, Holding, _>::json(
        "holding",
        "Return a demo portfolio holding for a ticker symbol.",
        |args: HoldingArgs| async move {
            Ok(Holding {
                symbol: args.symbol.to_uppercase(),
                shares: 12.0,
                average_cost: 104.25,
            })
        },
    )?;

    let mut registry = ToolRegistry::new();
    registry.add_tool(holding_tool)?;
    let registry = Arc::new(registry);

    let guardrails = Arc::new(
        GuardrailPipeline::new()
            .with_guardrail(PrivateInfoRedactionGuardrail::new())
            .with_guardrail(EmptyAnswerGuardrail::new()),
    );

    let client = DeepSeekClient::from_env()?;
    let analyst = Agent::with_config(
        client.clone(),
        AgentConfig::new("portfolio_analyst")
            .with_name("Portfolio Analyst")
            .with_role_description("Reviews holdings and explains risk plainly")
            .with_system_prompt(
                "Use portfolio tools for holdings and avoid personalized financial advice.",
            )
            .with_thinking(ThinkingConfig::disabled()),
    )
    .with_tool_registry(Arc::clone(&registry))?
    .with_guardrails(Arc::clone(&guardrails))
    .with_answer_composer(DefaultAnswerComposer::new());

    let reviewer = Agent::with_config(
        client,
        AgentConfig::new("risk_reviewer")
            .with_name("Risk Reviewer")
            .with_role_description("Checks final answers for risk framing")
            .with_system_prompt("Review risk language and keep it concise.")
            .with_thinking(ThinkingConfig::disabled()),
    )
    .with_guardrails(Arc::clone(&guardrails));

    let mut team = AgentTeam::new()
        .with_agent(analyst)?
        .with_agent(reviewer)?
        .with_router(StaticRouter::new("portfolio_analyst"))
        .with_guardrails(guardrails)
        .with_answer_composer(DefaultAnswerComposer::new());

    let response = team
        .ask("Check my demo AAPL holding and summarize risk.")
        .await?;

    println!("{}", response.content().unwrap_or(""));
    Ok(())
}
