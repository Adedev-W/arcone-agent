use arcone_agent::{
    Agent, DeepSeekClient, Document, InMemoryKnowledgeBase, InMemoryVectorRetriever,
    KnowledgeAgent, KnowledgeAgentOptions, KnowledgeBase, OpenAiEmbedder, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let knowledge = InMemoryKnowledgeBase::new();
    let chunks = knowledge
        .add_document(
            Document::text(
                "arcone-overview",
                "Arcone combines typed tools, session memory, retrievers, and agent teams.",
            )
            .with_title("Arcone Overview")
            .with_source("example"),
        )
        .await?;

    let retriever = InMemoryVectorRetriever::new(OpenAiEmbedder::from_env()?);
    retriever.index(chunks).await?;

    let agent = Agent::new(DeepSeekClient::from_env()?);
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever)
        .with_options(KnowledgeAgentOptions::new().with_top_k(2));

    let response = knowledge_agent.ask("What does Arcone combine?").await?;
    println!("{}", response.content());

    Ok(())
}
