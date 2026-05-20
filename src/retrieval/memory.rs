use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

use crate::{Error, KnowledgeChunk, Result, ScoredChunk};

use super::traits::{Embedder, RetrieveFuture, Retriever};

#[derive(Clone)]
pub struct InMemoryVectorRetriever {
    embedder: Arc<dyn Embedder>,
    entries: Arc<RwLock<Vec<VectorEntry>>>,
}

impl InMemoryVectorRetriever {
    pub fn new<E>(embedder: E) -> Self
    where
        E: Embedder + 'static,
    {
        Self::from_embedder(Arc::new(embedder))
    }

    pub fn from_embedder(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn index(&self, chunks: Vec<KnowledgeChunk>) -> Result<()> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "tracing")]
        let chunk_count = chunks.len();

        if chunks.is_empty() {
            let mut entries = self
                .entries
                .write()
                .map_err(|error| Error::RetrievalFailure(format!("lock poisoned: {error}")))?;
            entries.clear();
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::retrieval",
                operation = "memory_index",
                chunk_count,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "in-memory retriever index cleared"
            );
            return Ok(());
        }

        let texts = chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<_>>();
        let embeddings = self.embedder.embed(texts).await?;

        if embeddings.len() != chunks.len() {
            return Err(Error::EmbeddingFailure(format!(
                "embedding response count {} did not match input count {}",
                embeddings.len(),
                chunks.len()
            )));
        }

        let mut entries = Vec::with_capacity(chunks.len());
        for embedding in embeddings {
            let chunk = chunks.get(embedding.text_index).ok_or_else(|| {
                Error::EmbeddingFailure(format!(
                    "embedding index {} is out of bounds for {} chunks",
                    embedding.text_index,
                    chunks.len()
                ))
            })?;
            entries.push(VectorEntry {
                chunk: chunk.clone(),
                vector: embedding.vector,
            });
        }

        entries.sort_by(|left, right| {
            left.chunk
                .document_id
                .cmp(&right.chunk.document_id)
                .then(left.chunk.chunk_index.cmp(&right.chunk.chunk_index))
        });

        let mut store = self
            .entries
            .write()
            .map_err(|error| Error::RetrievalFailure(format!("lock poisoned: {error}")))?;
        *store = entries;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::retrieval",
            operation = "memory_index",
            chunk_count,
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "in-memory retriever indexed chunks"
        );
        Ok(())
    }

    pub fn len(&self) -> Result<usize> {
        let entries = self
            .entries
            .read()
            .map_err(|error| Error::RetrievalFailure(format!("lock poisoned: {error}")))?;
        Ok(entries.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

impl Retriever for InMemoryVectorRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> RetrieveFuture<Vec<ScoredChunk>> {
        let query = query.to_owned();
        let embedder = Arc::clone(&self.embedder);
        let entries = Arc::clone(&self.entries);

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();

            if top_k == 0 {
                #[cfg(feature = "tracing")]
                tracing::info!(
                    target: "arcone_agent::retrieval",
                    operation = "memory_retrieve",
                    top_k,
                    result_count = 0usize,
                    elapsed_ms = crate::observability::elapsed_ms(started_at),
                    "in-memory retrieval completed"
                );
                return Ok(Vec::new());
            }

            let entries = entries
                .read()
                .map_err(|error| Error::RetrievalFailure(format!("lock poisoned: {error}")))?
                .clone();
            if entries.is_empty() {
                #[cfg(feature = "tracing")]
                tracing::info!(
                    target: "arcone_agent::retrieval",
                    operation = "memory_retrieve",
                    top_k,
                    result_count = 0usize,
                    elapsed_ms = crate::observability::elapsed_ms(started_at),
                    "in-memory retrieval completed"
                );
                return Ok(Vec::new());
            }

            let query_embeddings = embedder.embed(vec![query]).await?;
            let query_embedding = query_embeddings.into_iter().next().ok_or_else(|| {
                Error::EmbeddingFailure(
                    "embedding response did not contain query vector".to_owned(),
                )
            })?;

            let mut scored = Vec::with_capacity(entries.len());
            for entry in entries {
                let score = cosine_similarity(&query_embedding.vector, &entry.vector)?;
                scored.push(ScoredChunk::new(entry.chunk, score));
            }

            scored.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(Ordering::Equal)
                    .then(left.chunk.document_id.cmp(&right.chunk.document_id))
                    .then(left.chunk.chunk_index.cmp(&right.chunk.chunk_index))
            });
            scored.truncate(top_k);
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::retrieval",
                operation = "memory_retrieve",
                top_k,
                result_count = scored.len(),
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "in-memory retrieval completed"
            );
            Ok(scored)
        })
    }
}

#[derive(Clone)]
struct VectorEntry {
    chunk: KnowledgeChunk,
    vector: Vec<f32>,
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(Error::RetrievalFailure(format!(
            "vector dimension mismatch: left={}, right={}",
            left.len(),
            right.len()
        )));
    }

    if left.is_empty() {
        return Err(Error::RetrievalFailure(
            "vector dimension must be greater than zero".to_owned(),
        ));
    }

    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;

    for (left_value, right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
}
