use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use crate::Error;

use super::chunking;
use super::store::{KnowledgeBase, KnowledgeFuture};
use super::types::{ChunkOptions, Document, DocumentId, KnowledgeChunk};

#[derive(Clone)]
pub struct InMemoryKnowledgeBase {
    documents: Arc<RwLock<BTreeMap<DocumentId, Document>>>,
    chunks: Arc<RwLock<BTreeMap<DocumentId, Vec<KnowledgeChunk>>>>,
    chunk_options: ChunkOptions,
}

impl InMemoryKnowledgeBase {
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(BTreeMap::new())),
            chunks: Arc::new(RwLock::new(BTreeMap::new())),
            chunk_options: ChunkOptions::default(),
        }
    }

    pub fn with_chunk_options(mut self, chunk_options: ChunkOptions) -> Self {
        self.chunk_options = chunk_options;
        self
    }

    pub fn chunk_options(&self) -> &ChunkOptions {
        &self.chunk_options
    }
}

impl Default for InMemoryKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeBase for InMemoryKnowledgeBase {
    fn add_document(&self, document: Document) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let documents = Arc::clone(&self.documents);
        let chunks = Arc::clone(&self.chunks);
        let chunk_options = self.chunk_options.clone();

        Box::pin(async move {
            let document_id = document.id.clone();
            let document_chunks = chunking::chunk_document(&document, &chunk_options)?;

            let mut document_store = documents
                .write()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;
            if document_store.contains_key(&document_id) {
                return Err(Error::DuplicateDocument(document_id.into_inner()));
            }

            let mut chunk_store = chunks
                .write()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;

            document_store.insert(document_id.clone(), document);
            chunk_store.insert(document_id, document_chunks.clone());

            Ok(document_chunks)
        })
    }

    fn list_documents(&self) -> KnowledgeFuture<Vec<Document>> {
        let documents = Arc::clone(&self.documents);

        Box::pin(async move {
            let document_store = documents
                .read()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;
            Ok(document_store.values().cloned().collect())
        })
    }

    fn remove_document(&self, id: &DocumentId) -> KnowledgeFuture<bool> {
        let id = id.clone();
        let documents = Arc::clone(&self.documents);
        let chunks = Arc::clone(&self.chunks);

        Box::pin(async move {
            let mut document_store = documents
                .write()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;
            let removed = document_store.remove(&id).is_some();

            let mut chunk_store = chunks
                .write()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;
            chunk_store.remove(&id);

            Ok(removed)
        })
    }

    fn chunk_document(&self, document: &Document) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let document = document.clone();
        let chunk_options = self.chunk_options.clone();

        Box::pin(async move { chunking::chunk_document(&document, &chunk_options) })
    }

    fn chunks_for_document(&self, id: &DocumentId) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let id = id.clone();
        let chunks = Arc::clone(&self.chunks);

        Box::pin(async move {
            let chunk_store = chunks
                .read()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;
            Ok(chunk_store.get(&id).cloned().unwrap_or_default())
        })
    }

    fn chunks_for_source(&self, source: &str) -> KnowledgeFuture<Vec<KnowledgeChunk>> {
        let source = source.to_owned();
        let chunks = Arc::clone(&self.chunks);

        Box::pin(async move {
            let chunk_store = chunks
                .read()
                .map_err(|error| Error::KnowledgeStore(format!("lock poisoned: {error}")))?;
            Ok(chunk_store
                .values()
                .flat_map(|document_chunks| document_chunks.iter())
                .filter(|chunk| chunk.metadata.source.as_deref() == Some(source.as_str()))
                .cloned()
                .collect())
        })
    }
}
