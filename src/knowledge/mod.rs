mod chunking;
mod memory;
pub(crate) mod postgres;
mod store;
mod types;

pub use memory::InMemoryKnowledgeBase;
pub use postgres::PostgresKnowledgeBase;
pub use store::{KnowledgeBase, KnowledgeFuture};
pub use types::{
    ChunkId, ChunkMetadata, ChunkOptions, Document, DocumentId, KnowledgeChunk, ScoredChunk,
};
