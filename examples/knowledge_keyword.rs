use std::sync::Arc;

use arcone_agent::{
    Agent, ChunkMetadata, DeepSeekClient, DocumentId, KnowledgeAgent, KnowledgeChunk, Result,
    RetrieveFuture, Retriever, ScoredChunk,
};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let chunks = vec![
        KnowledgeChunk::new(
            "chunk-rust-errors",
            DocumentId::new("guide-rust"),
            0,
            "Typed Rust errors make API failures explicit and easy to match in tests.",
            ChunkMetadata::new()
                .with_title("Rust Backend Guide")
                .with_source("local")
                .with_path("docs/rust-backend.md"),
        ),
        KnowledgeChunk::new(
            "chunk-agent-memory",
            DocumentId::new("guide-agent"),
            0,
            "Agent session stores keep conversation history behind a MemoryStore trait.",
            ChunkMetadata::new()
                .with_title("Agent Guide")
                .with_source("local")
                .with_path("docs/agent.md"),
        ),
    ];

    let retriever = KeywordRetriever::new(chunks);
    let agent = Agent::new(DeepSeekClient::from_env()?);
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever);

    let response = knowledge_agent
        .ask("Why are typed Rust errors useful?")
        .await?;

    println!("{}", response.content());
    for source in response.sources {
        println!("[{}] {:?}", source.index, source.title);
    }

    Ok(())
}

#[derive(Clone)]
struct KeywordRetriever {
    chunks: Arc<Vec<KnowledgeChunk>>,
}

impl KeywordRetriever {
    fn new(chunks: Vec<KnowledgeChunk>) -> Self {
        Self {
            chunks: Arc::new(chunks),
        }
    }
}

impl Retriever for KeywordRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> RetrieveFuture<Vec<ScoredChunk>> {
        let chunks = Arc::clone(&self.chunks);
        let terms = query
            .to_lowercase()
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        Box::pin(async move {
            let mut scored = chunks
                .iter()
                .filter_map(|chunk| {
                    let content = chunk.content.to_lowercase();
                    let hits = terms.iter().filter(|term| content.contains(*term)).count();
                    (hits > 0).then(|| ScoredChunk::new(chunk.clone(), hits as f32))
                })
                .collect::<Vec<_>>();

            scored.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(top_k);
            Ok(scored)
        })
    }
}
