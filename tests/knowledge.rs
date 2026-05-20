use std::sync::Arc;

use arcone_agent::{
    ChunkOptions, Document, DocumentId, Error, InMemoryKnowledgeBase, KnowledgeBase,
};
use serde_json::json;

#[tokio::test]
async fn add_list_query_and_remove_document() {
    let knowledge = InMemoryKnowledgeBase::new();
    let document = Document::text(
        "doc-1",
        "# Arcone Agent\n\nArcone supports multi-agent orchestration.",
    )
    .with_title("Arcone Agent")
    .with_source("manual")
    .with_path("docs/arcone.md")
    .with_metadata(json!({"tenant": "acme"}));

    let chunks = knowledge
        .add_document(document.clone())
        .await
        .expect("add document");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].document_id, DocumentId::new("doc-1"));
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[0].metadata.title.as_deref(), Some("Arcone Agent"));
    assert_eq!(chunks[0].metadata.source.as_deref(), Some("manual"));
    assert_eq!(chunks[0].metadata.path.as_deref(), Some("docs/arcone.md"));
    assert_eq!(chunks[0].metadata.extra, Some(json!({"tenant": "acme"})));

    let documents = knowledge.list_documents().await.expect("list documents");
    assert_eq!(documents, vec![document]);

    let by_document = knowledge
        .chunks_for_document(&DocumentId::new("doc-1"))
        .await
        .expect("chunks by document");
    assert_eq!(by_document, chunks);

    let by_source = knowledge
        .chunks_for_source("manual")
        .await
        .expect("chunks by source");
    assert_eq!(by_source, chunks);

    assert!(
        knowledge
            .remove_document(&DocumentId::new("doc-1"))
            .await
            .expect("remove document")
    );
    assert!(
        !knowledge
            .remove_document(&DocumentId::new("doc-1"))
            .await
            .expect("remove missing document")
    );
    assert!(
        knowledge
            .chunks_for_document(&DocumentId::new("doc-1"))
            .await
            .expect("chunks after remove")
            .is_empty()
    );
    assert!(knowledge.list_documents().await.unwrap().is_empty());
}

#[tokio::test]
async fn duplicate_document_id_is_rejected() {
    let knowledge = InMemoryKnowledgeBase::new();
    let document = Document::text("doc-dup", "first");

    knowledge
        .add_document(document.clone())
        .await
        .expect("first add");
    let error = knowledge
        .add_document(document)
        .await
        .expect_err("duplicate document should fail");

    assert!(matches!(error, Error::DuplicateDocument(id) if id == "doc-dup"));
}

#[tokio::test]
async fn empty_document_content_is_rejected() {
    let knowledge = InMemoryKnowledgeBase::new();

    let error = knowledge
        .add_document(Document::text("empty-doc", " \n\t "))
        .await
        .expect_err("empty document should fail");

    assert!(matches!(error, Error::KnowledgeIndexing(message) if message.contains("empty-doc")));
}

#[tokio::test]
async fn custom_chunk_options_create_bounded_chunks() {
    let knowledge = InMemoryKnowledgeBase::new().with_chunk_options(ChunkOptions::new(32, 0));
    let document = Document::text(
        "doc-chunks",
        "alpha alpha alpha\n\nbeta beta beta beta\n\ngamma gamma gamma",
    );

    let chunks = knowledge
        .add_document(document)
        .await
        .expect("chunk document");

    assert!(chunks.len() >= 2);
    for (index, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_index, index);
        assert!(chunk.content.chars().count() <= 32);
    }
}

#[tokio::test]
async fn large_unicode_paragraph_splits_on_char_boundaries() {
    let knowledge = InMemoryKnowledgeBase::new().with_chunk_options(ChunkOptions::new(3, 0));
    let document = Document::text("unicode-doc", "ééééé");

    let chunks = knowledge
        .chunk_document(&document)
        .await
        .expect("chunk unicode document");

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].content, "ééé");
    assert_eq!(chunks[1].content, "éé");
}

#[tokio::test]
async fn in_memory_knowledge_base_can_be_used_as_dyn_trait() {
    let knowledge: Arc<dyn KnowledgeBase> = Arc::new(InMemoryKnowledgeBase::new());

    knowledge
        .add_document(Document::text("dyn-doc", "dynamic dispatch"))
        .await
        .expect("add document");

    let chunks = knowledge
        .chunks_for_document(&DocumentId::new("dyn-doc"))
        .await
        .expect("load chunks");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "dynamic dispatch");
}
