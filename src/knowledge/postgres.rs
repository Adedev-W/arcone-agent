use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio_postgres::Row;

use crate::postgres::{PostgresPool, PostgresStoreConfig, now_millis};
use crate::{Error, Result};

use super::chunking;
use super::store::{KnowledgeBase, KnowledgeFuture};
use super::types::{ChunkOptions, Document, DocumentId, KnowledgeChunk};

#[derive(Clone)]
pub struct PostgresKnowledgeBase {
    pool: PostgresPool,
    chunk_options: ChunkOptions,
}

impl PostgresKnowledgeBase {
    pub fn new(pool: PostgresPool) -> Self {
        Self {
            pool,
            chunk_options: ChunkOptions::default(),
        }
    }

    pub async fn connect(config: PostgresStoreConfig) -> Result<Self> {
        let knowledge = Self::new(PostgresPool::connect(config).await?);
        knowledge.migrate().await?;
        Ok(knowledge)
    }

    pub fn with_chunk_options(mut self, chunk_options: ChunkOptions) -> Self {
        self.chunk_options = chunk_options;
        self
    }

    pub fn chunk_options(&self) -> &ChunkOptions {
        &self.chunk_options
    }

    pub fn pool(&self) -> &PostgresPool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<()> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        migrate_knowledge_schema(&self.pool).await?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::knowledge",
            operation = "migrate",
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "postgres knowledge migration completed"
        );
        Ok(())
    }
}

impl KnowledgeBase for PostgresKnowledgeBase {
    fn add_document(&self, document: Document) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let pool = self.pool.clone();
        let chunk_options = self.chunk_options.clone();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let document_id = document.id.clone();
            let document_chunks = chunking::chunk_document(&document, &chunk_options)?;
            let content_hash = content_hash(&document.content);
            let now = now_millis();

            let mut client = pool.client().await?;
            let transaction = client.transaction().await?;
            let exists = transaction
                .query_one(
                    "SELECT EXISTS (SELECT 1 FROM knowledge_documents WHERE document_id = $1)",
                    &[&document_id.as_str()],
                )
                .await?
                .get::<_, bool>(0);

            if exists {
                return Err(Error::DuplicateDocument(document_id.into_inner()));
            }

            transaction
                .execute(
                    r#"
                    INSERT INTO knowledge_documents (
                        document_id, title, source, path, content, content_hash,
                        metadata, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    "#,
                    &[
                        &document.id.as_str(),
                        &document.title,
                        &document.source,
                        &document.path,
                        &document.content,
                        &content_hash,
                        &document.metadata,
                        &now,
                        &now,
                    ],
                )
                .await?;

            for chunk in &document_chunks {
                let metadata = serde_json::to_value(&chunk.metadata)?;
                let chunk_index = usize_to_i32(chunk.chunk_index)?;

                transaction
                    .execute(
                        r#"
                        INSERT INTO knowledge_chunks (
                            chunk_id, document_id, chunk_index, content, metadata,
                            title, source, path, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        "#,
                        &[
                            &chunk.id.as_str(),
                            &chunk.document_id.as_str(),
                            &chunk_index,
                            &chunk.content,
                            &metadata,
                            &chunk.metadata.title,
                            &chunk.metadata.source,
                            &chunk.metadata.path,
                            &now,
                        ],
                    )
                    .await?;
            }

            transaction.commit().await?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::knowledge",
                operation = "add_document",
                chunk_count = document_chunks.len(),
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres knowledge document inserted"
            );
            Ok(document_chunks)
        })
    }

    fn list_documents(&self) -> KnowledgeFuture<Vec<Document>> {
        let pool = self.pool.clone();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool.client().await?;
            let rows = client
                .query(
                    r#"
                    SELECT document_id, title, source, path, content, metadata
                    FROM knowledge_documents
                    ORDER BY document_id ASC
                    "#,
                    &[],
                )
                .await?;
            #[cfg(feature = "tracing")]
            let row_count = rows.len();

            let documents = rows
                .into_iter()
                .map(row_to_document)
                .collect::<Result<Vec<_>>>()?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::knowledge",
                operation = "list_documents",
                row_count,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres knowledge documents loaded"
            );
            Ok(documents)
        })
    }

    fn remove_document(&self, id: &DocumentId) -> KnowledgeFuture<bool> {
        let pool = self.pool.clone();
        let id = id.clone();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool.client().await?;
            let removed = client
                .execute(
                    "DELETE FROM knowledge_documents WHERE document_id = $1",
                    &[&id.as_str()],
                )
                .await?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::knowledge",
                operation = "remove_document",
                removed = removed > 0,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres knowledge document removed"
            );
            Ok(removed > 0)
        })
    }

    fn chunk_document(&self, document: &Document) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let document = document.clone();
        let chunk_options = self.chunk_options.clone();

        Box::pin(async move { chunking::chunk_document(&document, &chunk_options) })
    }

    fn chunks_for_document(&self, id: &DocumentId) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let pool = self.pool.clone();
        let id = id.clone();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool.client().await?;
            let rows = client
                .query(
                    r#"
                    SELECT chunk_id, document_id, chunk_index, content, metadata
                    FROM knowledge_chunks
                    WHERE document_id = $1
                    ORDER BY chunk_index ASC
                    "#,
                    &[&id.as_str()],
                )
                .await?;
            #[cfg(feature = "tracing")]
            let row_count = rows.len();

            let chunks = rows
                .into_iter()
                .map(row_to_chunk)
                .collect::<Result<Vec<_>>>()?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::knowledge",
                operation = "chunks_for_document",
                row_count,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres knowledge chunks loaded"
            );
            Ok(chunks)
        })
    }

    fn chunks_for_source(&self, source: &str) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let pool = self.pool.clone();
        let source = source.to_owned();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool.client().await?;
            let rows = client
                .query(
                    r#"
                    SELECT chunk_id, document_id, chunk_index, content, metadata
                    FROM knowledge_chunks
                    WHERE source = $1
                    ORDER BY document_id ASC, chunk_index ASC
                    "#,
                    &[&source],
                )
                .await?;
            #[cfg(feature = "tracing")]
            let row_count = rows.len();

            let chunks = rows
                .into_iter()
                .map(row_to_chunk)
                .collect::<Result<Vec<_>>>()?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::knowledge",
                operation = "chunks_for_source",
                row_count,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres knowledge chunks loaded"
            );
            Ok(chunks)
        })
    }
}

pub(crate) async fn migrate_knowledge_schema(pool: &PostgresPool) -> Result<()> {
    let client = pool.client().await?;
    client
        .batch_execute(
            r#"
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS knowledge_documents (
                document_id TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                path TEXT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata JSONB,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_documents_source_path_hash
                ON knowledge_documents (
                    COALESCE(source, ''),
                    COALESCE(path, ''),
                    content_hash
                );

            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL
                    REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL,
                title TEXT,
                source TEXT,
                path TEXT,
                created_at BIGINT NOT NULL,
                UNIQUE (document_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_position
                ON knowledge_chunks (document_id, chunk_index);

            CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source_position
                ON knowledge_chunks (source, document_id, chunk_index);
            "#,
        )
        .await
        .map_err(|error| Error::DatabaseMigration(error.to_string()))?;

    Ok(())
}

fn row_to_document(row: Row) -> Result<Document> {
    Ok(Document {
        id: DocumentId::new(row.get::<_, String>("document_id")),
        content: row.get("content"),
        title: row.get("title"),
        source: row.get("source"),
        path: row.get("path"),
        metadata: row.get::<_, Option<Value>>("metadata"),
    })
}

pub(crate) fn row_to_chunk(row: Row) -> Result<KnowledgeChunk> {
    let metadata: Value = row.get("metadata");
    let metadata = serde_json::from_value(metadata)?;
    let chunk_index = row.get::<_, i32>("chunk_index");

    if chunk_index < 0 {
        return Err(Error::KnowledgeStore(format!(
            "chunk index cannot be negative: {chunk_index}"
        )));
    }

    Ok(KnowledgeChunk::new(
        row.get::<_, String>("chunk_id"),
        row.get::<_, String>("document_id"),
        chunk_index as usize,
        row.get::<_, String>("content"),
        metadata,
    ))
}

fn content_hash(content: &str) -> String {
    let digest = Sha256::digest(content.as_bytes());
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = std::fmt::Write::write_fmt(&mut hash, format_args!("{byte:02x}"));
    }
    hash
}

fn usize_to_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::KnowledgeIndexing(format!(
            "chunk index {value} exceeds PostgreSQL integer range"
        ))
    })
}
