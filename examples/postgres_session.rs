use std::sync::Arc;

use arcone_agent::{
    Agent, DeepSeekClient, PostgresSessionConfig, PostgresSessionStore, Result, SessionId,
};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let session_store =
        Arc::new(PostgresSessionStore::connect(PostgresSessionConfig::from_env()?).await?);
    let mut agent = Agent::new(DeepSeekClient::from_env()?)
        .with_session(SessionId::new("postgres-session-example"), session_store);

    let response = agent
        .ask_text("Remember that this example uses a PostgreSQL-backed session.")
        .await?;

    println!("{response}");
    Ok(())
}
