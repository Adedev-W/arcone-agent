use std::future::Future;
use std::pin::Pin;

use crate::{Result, ScoredChunk};

use super::types::Embedding;

pub type EmbedFuture<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;
pub type RetrieveFuture<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

pub trait Embedder: Send + Sync {
    fn embed(&self, texts: Vec<String>) -> EmbedFuture<Vec<Embedding>>;
}

pub trait Retriever: Send + Sync {
    fn retrieve(&self, query: &str, top_k: usize) -> RetrieveFuture<Vec<ScoredChunk>>;
}
