use std::time::Instant;

use arcone_agent::{Agent, ReasoningEffort};

#[tokio::main]
async fn main() -> arcone_agent::Result<()> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.trim().is_empty() {
        "Jelaskan dalam satu kalimat apa itu arcone-agent.".to_owned()
    } else {
        prompt
    };

    let mut agent = Agent::from_env()?
        .system("Gunakan bahasa indonesia yang baik dan sopan")
        .thinking_enabled()
        .reasoning(ReasoningEffort::High);

    let started_at = Instant::now();
    let text = agent.ask_text(prompt).await?;
    let elapsed = started_at.elapsed();

    println!("{text}");
    println!(
        "\ncompleted in {} ms ({:.3} s)",
        elapsed.as_millis(),
        elapsed.as_secs_f64()
    );

    Ok(())
}
