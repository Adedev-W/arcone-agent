use arcone_agent::{
    Agent, DeepSeekClient, Document, DocumentId, Error, KnowledgeAgent, KnowledgeAgentOptions,
    KnowledgeBase, OpenAiEmbedder, PgVectorRetriever, PgVectorRetrieverOptions,
    PostgresKnowledgeBase, PostgresPool, PostgresStoreConfig, Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let pool = PostgresPool::connect(PostgresStoreConfig::from_env()?).await?;
    let knowledge = PostgresKnowledgeBase::new(pool.clone());
    knowledge.migrate().await?;

    let document_id = DocumentId::new("arcone-pgvector-example");
    let document = Document::text(
        document_id.clone(),
        "pgvector stores embeddings in PostgreSQL for durable semantic retrieval.",
    )
    .with_title("pgvector Example")
    .with_source("example");

    let chunks = match knowledge.add_document(document).await {
        Ok(chunks) => chunks,
        Err(Error::DuplicateDocument(_)) => knowledge.chunks_for_document(&document_id).await?,
        Err(error) => return Err(error),
    };

    let retriever = PgVectorRetriever::new(
        pool,
        OpenAiEmbedder::from_env()?,
        PgVectorRetrieverOptions::default(),
    );
    retriever.migrate().await?;
    retriever.index(chunks).await?;

    let agent = Agent::new(DeepSeekClient::from_env()?);
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever)
        .with_options(KnowledgeAgentOptions::new().with_top_k(3));

    let response = knowledge_agent
        .ask("Where are pgvector embeddings stored?")
        .await?;

    println!("{}", response.content());
    Ok(())
}
