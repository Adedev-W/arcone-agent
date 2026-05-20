use std::future::Future;
use std::pin::Pin;

use crate::Result;

use super::types::{Document, DocumentId, KnowledgeChunk};

pub type KnowledgeFuture<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

pub trait KnowledgeBase: Send + Sync {
    fn add_document(&self, document: Document) -> KnowledgeFuture<Vec<KnowledgeChunk>>;

    fn list_documents(&self) -> KnowledgeFuture<Vec<Document>>;

    fn remove_document(&self, id: &DocumentId) -> KnowledgeFuture<bool>;

    fn chunk_document(&self, document: &Document) -> KnowledgeFuture<Vec<KnowledgeChunk>>;

    fn chunks_for_document(&self, id: &DocumentId) -> KnowledgeFuture<Vec<KnowledgeChunk>>;

    fn chunks_for_source(&self, source: &str) -> KnowledgeFuture<Vec<KnowledgeChunk>>;
}
