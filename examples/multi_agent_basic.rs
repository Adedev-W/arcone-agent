use arcone_agent::{
    Agent, AgentConfig, AgentTeam, DeepSeekClient, Result, StaticRouter, ThinkingConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let client = DeepSeekClient::from_env()?;
    let researcher = Agent::with_config(
        client.clone(),
        AgentConfig::new("researcher")
            .with_name("Researcher")
            .with_role_description("Finds facts and constraints")
            .with_system_prompt("Research carefully and answer with concise evidence.")
            .with_thinking(ThinkingConfig::enabled()),
    );
    let writer = Agent::with_config(
        client,
        AgentConfig::new("writer")
            .with_name("Writer")
            .with_role_description("Turns research into a short final answer")
            .with_system_prompt("Write plainly and keep the answer brief.")
            .with_thinking(ThinkingConfig::enabled()),
    );

    let mut team = AgentTeam::new()
        .with_agent(researcher)?
        .with_agent(writer)?
        .with_router(StaticRouter::new("researcher").with_reason("default research route"));

    let response = team.ask("apa itu ai?").await?;

    println!(
        "{}: {}",
        response.agent_id,
        response.content().unwrap_or("")
    );
    Ok(())
}
