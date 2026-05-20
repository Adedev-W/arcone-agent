use std::sync::Arc;

use pgvector::Vector;
use tokio_postgres::Row;

use crate::knowledge::postgres::{migrate_knowledge_schema, row_to_chunk};
use crate::postgres::{PostgresPool, PostgresStoreConfig, now_millis};
use crate::{Embedder, Embedding, Error, KnowledgeChunk, Result, ScoredChunk};

use super::traits::{RetrieveFuture, Retriever};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PgVectorMetric {
    #[default]
    Cosine,
    L2,
}

impl PgVectorMetric {
    fn operator(&self) -> &'static str {
        match self {
            Self::Cosine => "<=>",
            Self::L2 => "<->",
        }
    }

    fn operator_class(&self) -> &'static str {
        match self {
            Self::Cosine => "vector_cosine_ops",
            Self::L2 => "vector_l2_ops",
        }
    }

    fn score(&self, distance: f64) -> f32 {
        match self {
            Self::Cosine => (1.0 - distance) as f32,
            Self::L2 => (1.0 / (1.0 + distance.max(0.0))) as f32,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PgVectorIndexMode {
    #[default]
    Auto,
    Hnsw,
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PgVectorRetrieverOptions {
    pub embedding_dimension: usize,
    pub metric: PgVectorMetric,
    pub index_mode: PgVectorIndexMode,
}

impl PgVectorRetrieverOptions {
    pub fn new(embedding_dimension: usize) -> Self {
        Self {
            embedding_dimension,
            ..Self::default()
        }
    }

    pub fn with_embedding_dimension(mut self, embedding_dimension: usize) -> Self {
        self.embedding_dimension = embedding_dimension;
        self
    }

    pub fn with_metric(mut self, metric: PgVectorMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_index_mode(mut self, index_mode: PgVectorIndexMode) -> Self {
        self.index_mode = index_mode;
        self
    }
}

impl Default for PgVectorRetrieverOptions {
    fn default() -> Self {
        Self {
            embedding_dimension: 1_536,
            metric: PgVectorMetric::default(),
            index_mode: PgVectorIndexMode::default(),
        }
    }
}

#[derive(Clone)]
pub struct PgVectorRetriever {
    pool: PostgresPool,
    embedder: Arc<dyn Embedder>,
    options: PgVectorRetrieverOptions,
}

impl PgVectorRetriever {
    pub fn new<E>(pool: PostgresPool, embedder: E, options: PgVectorRetrieverOptions) -> Self
    where
        E: Embedder + 'static,
    {
        Self::from_embedder(pool, Arc::new(embedder), options)
    }

    pub async fn connect<E>(
        config: PostgresStoreConfig,
        embedder: E,
        options: PgVectorRetrieverOptions,
    ) -> Result<Self>
    where
        E: Embedder + 'static,
    {
        let retriever = Self::new(PostgresPool::connect(config).await?, embedder, options);
        retriever.migrate().await?;
        Ok(retriever)
    }

    pub fn from_embedder(
        pool: PostgresPool,
        embedder: Arc<dyn Embedder>,
        options: PgVectorRetrieverOptions,
    ) -> Self {
        Self {
            pool,
            embedder,
            options,
        }
    }

    pub fn pool(&self) -> &PostgresPool {
        &self.pool
    }

    pub fn options(&self) -> &PgVectorRetrieverOptions {
        &self.options
    }

    pub async fn migrate(&self) -> Result<()> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        validate_dimension(self.options.embedding_dimension)?;
        migrate_knowledge_schema(&self.pool).await?;

        let client = self.pool.client().await?;
        let dimension = self.options.embedding_dimension;
        let create_table = format!(
            r#"
            CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                chunk_id TEXT PRIMARY KEY
                    REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE,
                document_id TEXT NOT NULL
                    REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                embedding vector({dimension}) NOT NULL,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_document
                ON knowledge_embeddings (document_id);
            "#
        );

        client
            .batch_execute(&create_table)
            .await
            .map_err(|error| Error::DatabaseMigration(error.to_string()))?;
        validate_table_dimension(&client, dimension).await?;
        self.create_vector_index(&client).await?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::retrieval",
            operation = "pgvector_migrate",
            embedding_dimension = dimension,
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "pgvector migration completed"
        );
        Ok(())
    }

    pub async fn index(&self, chunks: Vec<KnowledgeChunk>) -> Result<()> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "tracing")]
        let chunk_count = chunks.len();

        if chunks.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::retrieval",
                operation = "pgvector_index",
                chunk_count,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "pgvector index skipped empty batch"
            );
            return Ok(());
        }

        validate_dimension(self.options.embedding_dimension)?;

        let texts = chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<_>>();
        let embeddings = self.embedder.embed(texts).await?;
        let embeddings = self.align_embeddings(embeddings, chunks.len())?;

        let now = now_millis();
        let mut client = self.pool.client().await?;
        let transaction = client.transaction().await?;

        for (chunk, embedding) in chunks.iter().zip(embeddings) {
            let vector = Vector::from(embedding.vector);
            transaction
                .execute(
                    r#"
                    INSERT INTO knowledge_embeddings (
                        chunk_id, document_id, embedding, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET document_id = EXCLUDED.document_id,
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at
                    "#,
                    &[
                        &chunk.id.as_str(),
                        &chunk.document_id.as_str(),
                        &vector,
                        &now,
                        &now,
                    ],
                )
                .await?;
        }

        transaction.commit().await?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::retrieval",
            operation = "pgvector_index",
            chunk_count,
            embedding_dimension = self.options.embedding_dimension,
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "pgvector indexed chunks"
        );
        Ok(())
    }

    fn align_embeddings(
        &self,
        embeddings: Vec<Embedding>,
        chunk_count: usize,
    ) -> Result<Vec<Embedding>> {
        if embeddings.len() != chunk_count {
            return Err(Error::EmbeddingFailure(format!(
                "embedding response count {} did not match input count {}",
                embeddings.len(),
                chunk_count
            )));
        }

        let mut aligned = vec![None; chunk_count];
        for embedding in embeddings {
            if embedding.text_index >= chunk_count {
                return Err(Error::EmbeddingFailure(format!(
                    "embedding index {} is out of bounds for {} chunks",
                    embedding.text_index, chunk_count
                )));
            }

            if embedding.vector.len() != self.options.embedding_dimension {
                return Err(Error::EmbeddingFailure(format!(
                    "embedding dimension {} did not match configured dimension {}",
                    embedding.vector.len(),
                    self.options.embedding_dimension
                )));
            }

            let index = embedding.text_index;
            aligned[index] = Some(embedding);
        }

        aligned
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| {
                embedding.ok_or_else(|| {
                    Error::EmbeddingFailure(format!(
                        "embedding response did not include text index {index}"
                    ))
                })
            })
            .collect()
    }

    async fn create_vector_index(&self, client: &deadpool_postgres::Client) -> Result<()> {
        match self.options.index_mode {
            PgVectorIndexMode::None => Ok(()),
            PgVectorIndexMode::Auto | PgVectorIndexMode::Hnsw => {
                let sql = format!(
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_embedding_hnsw \
                     ON knowledge_embeddings USING hnsw (embedding {})",
                    self.options.metric.operator_class()
                );
                match client.batch_execute(&sql).await {
                    Ok(()) => Ok(()),
                    Err(error) if self.options.index_mode == PgVectorIndexMode::Auto => {
                        let _ = error;
                        Ok(())
                    }
                    Err(error) => Err(Error::DatabaseMigration(error.to_string())),
                }
            }
        }
    }
}

impl Retriever for PgVectorRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> RetrieveFuture<Vec<ScoredChunk>> {
        let query = query.to_owned();
        let top_k = top_k as i64;
        let embedder = Arc::clone(&self.embedder);
        let pool = self.pool.clone();
        let options = self.options.clone();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();

            if top_k <= 0 {
                #[cfg(feature = "tracing")]
                tracing::info!(
                    target: "arcone_agent::retrieval",
                    operation = "pgvector_retrieve",
                    top_k,
                    result_count = 0usize,
                    elapsed_ms = crate::observability::elapsed_ms(started_at),
                    "pgvector retrieval completed"
                );
                return Ok(Vec::new());
            }

            validate_dimension(options.embedding_dimension)?;
            let query_embeddings = embedder.embed(vec![query]).await?;
            let query_embedding = query_embeddings.into_iter().next().ok_or_else(|| {
                Error::EmbeddingFailure(
                    "embedding response did not contain query vector".to_owned(),
                )
            })?;

            if query_embedding.vector.len() != options.embedding_dimension {
                return Err(Error::EmbeddingFailure(format!(
                    "query embedding dimension {} did not match configured dimension {}",
                    query_embedding.vector.len(),
                    options.embedding_dimension
                )));
            }

            let vector = Vector::from(query_embedding.vector);
            let client = pool.client().await?;
            let operator = options.metric.operator();
            let sql = format!(
                r#"
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.content,
                    c.metadata,
                    e.embedding {operator} $1 AS distance
                FROM knowledge_embeddings e
                INNER JOIN knowledge_chunks c ON c.chunk_id = e.chunk_id
                ORDER BY e.embedding {operator} $1 ASC,
                         c.document_id ASC,
                         c.chunk_index ASC
                LIMIT $2
                "#
            );
            let rows = client.query(&sql, &[&vector, &top_k]).await?;
            #[cfg(feature = "tracing")]
            let result_count = rows.len();

            let results = rows
                .into_iter()
                .map(|row| row_to_scored_chunk(row, &options.metric))
                .collect::<Result<Vec<_>>>()?;
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::retrieval",
                operation = "pgvector_retrieve",
                top_k,
                result_count,
                embedding_dimension = options.embedding_dimension,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "pgvector retrieval completed"
            );
            Ok(results)
        })
    }
}

fn row_to_scored_chunk(row: Row, metric: &PgVectorMetric) -> Result<ScoredChunk> {
    let distance = row.try_get::<_, f64>("distance")?;
    let score = metric.score(distance);
    let chunk = row_to_chunk(row)?;
    Ok(ScoredChunk::new(chunk, score))
}

fn validate_dimension(dimension: usize) -> Result<()> {
    if dimension == 0 {
        return Err(Error::RetrievalFailure(
            "embedding dimension must be greater than zero".to_owned(),
        ));
    }

    Ok(())
}

async fn validate_table_dimension(
    client: &deadpool_postgres::Client,
    expected_dimension: usize,
) -> Result<()> {
    let row = client
        .query_one(
            r#"
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            INNER JOIN pg_class c ON c.oid = a.attrelid
            INNER JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'knowledge_embeddings'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
              AND n.nspname = current_schema()
            "#,
            &[],
        )
        .await
        .map_err(|error| Error::DatabaseMigration(error.to_string()))?;
    let actual_type = row.get::<_, String>(0);
    let expected_type = format!("vector({expected_dimension})");

    if actual_type != expected_type {
        return Err(Error::DatabaseMigration(format!(
            "knowledge_embeddings.embedding has type {actual_type}, expected {expected_type}"
        )));
    }

    Ok(())
}
