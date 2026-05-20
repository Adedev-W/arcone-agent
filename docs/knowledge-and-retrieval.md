# Knowledge And Retrieval

The knowledge system turns documents into chunks, stores them, retrieves scored
chunks, and lets `KnowledgeAgent` answer using only retrieved context.

Related docs: [Agents](agents.md), [Guardrails and composer](guardrails-and-composer.md),
[Operations](operations.md), [Examples](examples.md), [API reference](api-reference.md).

## Core Types

- `Document`, `DocumentId`: source documents and stable document IDs.
- `ChunkOptions`: chunk size and overlap configuration.
- `KnowledgeChunk`, `ChunkId`, `ChunkMetadata`: chunked document content with
  metadata.
- `ScoredChunk`: retrieved chunk plus relevance score.
- `KnowledgeBase`: async trait for document and chunk storage.
- `InMemoryKnowledgeBase`: in-process document and chunk store.
- `PostgresKnowledgeBase`: durable PostgreSQL document and chunk store.
- `Embedder`, `Embedding`: embedding provider abstraction.
- `Retriever`: retrieval abstraction.
- `OpenAiEmbedder`, `OpenAiConfig`, `OpenAiEmbeddingModel`: OpenAI embedding
  adapter.
- `InMemoryVectorRetriever`: vector search in memory.
- `PgVectorRetriever`: vector search in PostgreSQL with pgvector.
- `KnowledgeAgent`: wraps an `Agent` and a `Retriever`.

## Add Documents

```rust
use arcone_agent::{Document, InMemoryKnowledgeBase, KnowledgeBase, Result};

async fn load_docs() -> Result<()> {
    let knowledge = InMemoryKnowledgeBase::new();
    let chunks = knowledge
        .add_document(
            Document::text(
                "overview",
                "Arcone combines typed tools, sessions, retrievers, and agent teams.",
            )
            .with_title("Overview")
            .with_source("local")
            .with_path("docs/overview.md"),
        )
        .await?;

    println!("indexed {} chunks", chunks.len());
    Ok(())
}
```

Default chunking uses `ChunkOptions { max_chars: 1200, overlap_chars: 120 }`.
Override it when documents have very short or very long sections:

```rust
use arcone_agent::{ChunkOptions, InMemoryKnowledgeBase};

let knowledge = InMemoryKnowledgeBase::new()
    .with_chunk_options(ChunkOptions::new(800, 80));
```

## KnowledgeAgent

`KnowledgeAgent` retrieves context first, then asks the inner `Agent` to answer
using only that context. If no context is available, it returns the configured
fallback message.

```rust
use arcone_agent::{
    Agent, DeepSeekClient, KnowledgeAgent, KnowledgeAgentOptions, Result,
};

async fn answer_with_knowledge(retriever: impl arcone_agent::Retriever + 'static) -> Result<()> {
    let agent = Agent::new(DeepSeekClient::from_env()?);
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever)
        .with_options(
            KnowledgeAgentOptions::new()
                .with_top_k(4)
                .with_max_context_chars(6_000),
        );

    let response = knowledge_agent
        .ask("What does the knowledge base say about sessions?")
        .await?;

    println!("{}", response.content());
    for source in response.sources {
        println!("[{}] {:?}", source.index, source.title);
    }

    Ok(())
}
```

Best practice: show or store `response.sources` for auditability whenever an
answer depends on retrieved context.

## OpenAI Embeddings And In-Memory Vector Search

Use `OpenAiEmbedder` for embedding calls and `InMemoryVectorRetriever` for local
semantic retrieval.

```rust
use arcone_agent::{
    Document, InMemoryKnowledgeBase, InMemoryVectorRetriever, KnowledgeBase,
    OpenAiEmbedder, Result,
};

async fn build_retriever() -> Result<InMemoryVectorRetriever> {
    let knowledge = InMemoryKnowledgeBase::new();
    let chunks = knowledge
        .add_document(Document::text("rag", "Retrievers return scored chunks."))
        .await?;

    let retriever = InMemoryVectorRetriever::new(OpenAiEmbedder::from_env()?);
    retriever.index(chunks).await?;
    Ok(retriever)
}
```

`OpenAiConfig::from_env()` reads `OPENAI_API_KEY`, optional
`OPENAI_BASE_URL`, and optional `OPENAI_EMBEDDING_MODEL`.

## PostgreSQL Knowledge Store

Use `PostgresKnowledgeBase` for durable documents and chunks.

```rust
use arcone_agent::{
    Document, KnowledgeBase, PostgresKnowledgeBase, PostgresPool,
    PostgresStoreConfig, Result,
};

async fn store_document() -> Result<()> {
    let pool = PostgresPool::connect(PostgresStoreConfig::from_env()?).await?;
    let knowledge = PostgresKnowledgeBase::new(pool);
    knowledge.migrate().await?;

    knowledge
        .add_document(Document::text("postgres", "Documents are stored in PostgreSQL."))
        .await?;

    Ok(())
}
```

`PostgresKnowledgeBase::migrate` creates the document and chunk tables.

## pgvector Retrieval

Use `PgVectorRetriever` for durable semantic retrieval in PostgreSQL.

```rust
use arcone_agent::{
    OpenAiEmbedder, PgVectorRetriever, PgVectorRetrieverOptions, PostgresPool,
    PostgresStoreConfig, Result,
};

async fn build_pgvector() -> Result<PgVectorRetriever> {
    let pool = PostgresPool::connect(PostgresStoreConfig::from_env()?).await?;
    let retriever = PgVectorRetriever::new(
        pool,
        OpenAiEmbedder::from_env()?,
        PgVectorRetrieverOptions::default(),
    );

    retriever.migrate().await?;
    Ok(retriever)
}
```

Default pgvector options use 1536 embedding dimensions, cosine distance, and
automatic HNSW index creation when available. Set the dimension to match your
embedding model.

```rust
use arcone_agent::{PgVectorIndexMode, PgVectorMetric, PgVectorRetrieverOptions};

let options = PgVectorRetrieverOptions::new(3_072)
    .with_metric(PgVectorMetric::Cosine)
    .with_index_mode(PgVectorIndexMode::Hnsw);
```

## Custom Retriever

Implement `Retriever` when you want keyword search, an external vector database,
hybrid search, or reranking.

```rust
use std::sync::Arc;
use arcone_agent::{KnowledgeChunk, RetrieveFuture, Retriever, ScoredChunk};

#[derive(Clone)]
struct StaticRetriever {
    chunks: Arc<Vec<KnowledgeChunk>>,
}

impl Retriever for StaticRetriever {
    fn retrieve(&self, _query: &str, top_k: usize) -> RetrieveFuture<Vec<ScoredChunk>> {
        let chunks = Arc::clone(&self.chunks);
        Box::pin(async move {
            Ok(chunks
                .iter()
                .take(top_k)
                .cloned()
                .map(|chunk| ScoredChunk::new(chunk, 1.0))
                .collect())
        })
    }
}
```

## Retrieval Best Practices

- Use stable document IDs and chunk metadata so answers can cite sources.
- Tune `ChunkOptions` with real documents; too-small chunks lose context and
  too-large chunks waste model context.
- Make indexing idempotent in your application workflow. Duplicate document IDs
  return `Error::DuplicateDocument`.
- Keep embedding dimensions consistent with `PgVectorRetrieverOptions`.
- Call `migrate()` during deployment or startup before indexing or retrieval.
- Set `KnowledgeAgentOptions::max_context_chars` to protect model context size.
- Use guardrails on the retrieved context stage when context can contain private
  or untrusted text.

 