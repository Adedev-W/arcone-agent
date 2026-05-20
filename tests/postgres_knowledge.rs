use std::{collections::HashMap, sync::Arc};

use arcone_agent::{
    ChunkOptions, Document, EmbedFuture, Embedder, Embedding, Error, KnowledgeBase,
    PgVectorIndexMode, PgVectorRetriever, PgVectorRetrieverOptions, PostgresKnowledgeBase,
    PostgresPool, PostgresStoreConfig, Retriever,
};

const EMBEDDING_DIMENSION: usize = 1_536;

#[tokio::test]
async fn postgres_knowledge_and_pgvector_retrieval_round_trip() {
    dotenvy::dotenv().ok();
    let Ok(database_url) = std::env::var("DATABASE_URL") else {
        return;
    };

    let pool = PostgresPool::connect(PostgresStoreConfig::new(database_url).with_max_pool_size(4))
        .await
        .expect("postgres pool");
    let knowledge =
        PostgresKnowledgeBase::new(pool.clone()).with_chunk_options(ChunkOptions::new(4_000, 0));

    if should_skip_pgvector(knowledge.migrate().await) {
        return;
    }

    let suffix = unique_suffix();
    let source = format!("postgres-knowledge-test-{suffix}");
    let alpha_doc_id = format!("pg-alpha-{suffix}");
    let beta_doc_id = format!("pg-beta-{suffix}");

    let alpha = Document::text(
        alpha_doc_id.clone(),
        format!("alpha arcone orchestration knowledge {suffix}"),
    )
    .with_title("Alpha")
    .with_source(source.clone())
    .with_path(format!("alpha-{suffix}.md"));
    let beta = Document::text(
        beta_doc_id.clone(),
        format!("beta unrelated billing knowledge {suffix}"),
    )
    .with_title("Beta")
    .with_source(source.clone())
    .with_path(format!("beta-{suffix}.md"));

    let mut chunks = knowledge
        .add_document(alpha.clone())
        .await
        .expect("add alpha");
    chunks.extend(
        knowledge
            .add_document(beta.clone())
            .await
            .expect("add beta"),
    );

    let documents = knowledge.list_documents().await.expect("list documents");
    assert!(documents.iter().any(|document| document.id == alpha.id));
    assert!(documents.iter().any(|document| document.id == beta.id));

    let source_chunks = knowledge
        .chunks_for_source(&source)
        .await
        .expect("chunks for source");
    assert_eq!(source_chunks.len(), 2);

    let embedder = StaticEmbedder::new([
        (
            format!("alpha arcone orchestration knowledge {suffix}"),
            vector(0.314_159, 0.271_828),
        ),
        (
            format!("beta unrelated billing knowledge {suffix}"),
            vector(-0.314_159, -0.271_828),
        ),
        (
            "arcone orchestration".to_owned(),
            vector(0.314_159, 0.271_828),
        ),
    ]);
    let retriever = PgVectorRetriever::new(
        pool,
        embedder,
        PgVectorRetrieverOptions::default()
            .with_embedding_dimension(EMBEDDING_DIMENSION)
            .with_index_mode(PgVectorIndexMode::None),
    );

    if should_skip_pgvector(retriever.migrate().await) {
        cleanup(&knowledge, &[alpha_doc_id.as_str(), beta_doc_id.as_str()]).await;
        return;
    }

    retriever.index(chunks).await.expect("index embeddings");
    let results = retriever
        .retrieve("arcone orchestration", 2)
        .await
        .expect("retrieve");

    assert!(!results.is_empty());
    assert_eq!(results[0].chunk.document_id, alpha.id);
    assert!(results[0].score > 0.99);

    assert!(
        knowledge
            .remove_document(&alpha.id)
            .await
            .expect("remove alpha")
    );
    assert!(
        knowledge
            .remove_document(&beta.id)
            .await
            .expect("remove beta")
    );
    assert!(
        knowledge
            .chunks_for_document(&alpha.id)
            .await
            .expect("alpha chunks after remove")
            .is_empty()
    );
}

#[derive(Clone)]
struct StaticEmbedder {
    vectors: Arc<HashMap<String, Vec<f32>>>,
}

impl StaticEmbedder {
    fn new<const N: usize>(vectors: [(String, Vec<f32>); N]) -> Self {
        Self {
            vectors: Arc::new(vectors.into_iter().collect()),
        }
    }
}

impl Embedder for StaticEmbedder {
    fn embed(&self, texts: Vec<String>) -> EmbedFuture<Vec<Embedding>> {
        let vectors = Arc::clone(&self.vectors);

        Box::pin(async move {
            texts
                .into_iter()
                .enumerate()
                .map(|(index, text)| {
                    let vector = vectors.get(&text).cloned().ok_or_else(|| {
                        Error::EmbeddingFailure(format!("missing vector for `{text}`"))
                    })?;
                    Ok(Embedding::new(index, vector))
                })
                .collect()
        })
    }
}

fn vector(first: f32, second: f32) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMENSION];
    vector[0] = first;
    vector[1] = second;
    vector
}

async fn cleanup(knowledge: &PostgresKnowledgeBase, ids: &[&str]) {
    for id in ids {
        let _ = knowledge.remove_document(&(*id).into()).await;
    }
}

fn should_skip_pgvector(result: arcone_agent::Result<()>) -> bool {
    match result {
        Ok(()) => false,
        Err(Error::DatabaseMigration(message))
            if message.contains("extension")
                || message.contains("vector")
                || message.contains("permission denied") =>
        {
            true
        }
        Err(error) => panic!("postgres migration failed: {error}"),
    }
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
