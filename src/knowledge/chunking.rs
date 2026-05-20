use super::types::{ChunkId, ChunkOptions, Document, KnowledgeChunk};
use crate::{Error, Result};

pub(super) fn chunk_document(
    document: &Document,
    options: &ChunkOptions,
) -> Result<Vec<KnowledgeChunk>> {
    validate_input(document, options)?;

    let paragraphs = paragraphs(&document.content);
    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in paragraphs {
        if char_len(&paragraph) > options.max_chars {
            flush_chunk(&mut chunks, &mut current);
            chunks.extend(split_large_text(&paragraph, options.max_chars));
            continue;
        }

        if current.is_empty() {
            current = paragraph;
            continue;
        }

        let joined_len = char_len(&current) + 2 + char_len(&paragraph);
        if joined_len <= options.max_chars {
            current.push_str("\n\n");
            current.push_str(&paragraph);
        } else {
            flush_chunk(&mut chunks, &mut current);
            current = paragraph;
        }
    }

    flush_chunk(&mut chunks, &mut current);

    let chunks = apply_overlap(chunks, options);
    let metadata = super::types::ChunkMetadata::from_document(document);

    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(chunk_index, content)| {
            KnowledgeChunk::new(
                ChunkId::new(format!("{}:{}", document.id, chunk_index)),
                document.id.clone(),
                chunk_index,
                content,
                metadata.clone(),
            )
        })
        .collect())
}

fn validate_input(document: &Document, options: &ChunkOptions) -> Result<()> {
    if document.content.trim().is_empty() {
        return Err(Error::KnowledgeIndexing(format!(
            "document `{}` content is empty",
            document.id
        )));
    }

    if options.max_chars == 0 {
        return Err(Error::KnowledgeIndexing(
            "chunk max_chars must be greater than zero".to_owned(),
        ));
    }

    Ok(())
}

fn paragraphs(content: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            flush_paragraph(&mut paragraphs, &mut current);
            continue;
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line.trim_end());
    }

    flush_paragraph(&mut paragraphs, &mut current);
    paragraphs
}

fn flush_paragraph(paragraphs: &mut Vec<String>, current: &mut String) {
    if !current.trim().is_empty() {
        paragraphs.push(current.trim().to_owned());
    }
    current.clear();
}

fn split_large_text(text: &str, max_chars: usize) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    chars
        .chunks(max_chars)
        .map(|chunk| chunk.iter().collect::<String>().trim().to_owned())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

fn flush_chunk(chunks: &mut Vec<String>, current: &mut String) {
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_owned());
    }
    current.clear();
}

fn apply_overlap(chunks: Vec<String>, options: &ChunkOptions) -> Vec<String> {
    if chunks.len() <= 1 || options.overlap_chars == 0 {
        return chunks;
    }

    let mut overlapped: Vec<String> = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        if let Some(previous) = overlapped.last() {
            let available = options.max_chars.saturating_sub(char_len(&chunk));
            if available > 2 {
                let overlap_len = options.overlap_chars.min(available - 2);
                let overlap = suffix_chars(previous, overlap_len);
                if !overlap.is_empty() {
                    overlapped.push(format!("{overlap}\n\n{chunk}"));
                    continue;
                }
            }
        }

        overlapped.push(chunk);
    }

    overlapped
}

fn suffix_chars(value: &str, count: usize) -> String {
    if count == 0 {
        return String::new();
    }

    let mut chars = value.chars().rev().take(count).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}

fn char_len(value: &str) -> usize {
    value.chars().count()
}
