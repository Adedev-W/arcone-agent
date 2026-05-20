use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    AgentId, ChatMessage, ChunkId, DocumentId, GuardrailEvent, Result, Role, ScoredChunk, Usage,
};

pub type AnswerComposerFuture = Pin<Box<dyn Future<Output = Result<ComposedAnswer>> + Send>>;

pub trait AnswerComposer: Send + Sync {
    fn compose(&self, input: AnswerCompositionInput) -> AnswerComposerFuture;
}

#[derive(Clone, Debug)]
pub struct AnswerCompositionInput {
    pub original_question: String,
    pub selected_agent_id: Option<AgentId>,
    pub tool_outputs: Vec<ToolOutput>,
    pub retrieved_chunks: Vec<ScoredChunk>,
    pub draft_answer: String,
    pub guardrail_events: Vec<GuardrailEvent>,
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub content: String,
}

impl ToolOutput {
    pub fn from_history(history: &[ChatMessage]) -> Vec<Self> {
        history
            .iter()
            .filter(|message| message.role == Role::Tool)
            .filter_map(|message| {
                Some(Self {
                    tool_call_id: message.tool_call_id.clone(),
                    content: message.content.clone()?,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComposedAnswer {
    pub text: String,
    pub sources: Vec<AnswerSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_metadata: Option<serde_json::Value>,
}

impl ComposedAnswer {
    pub fn content(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnswerSource {
    pub index: usize,
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub score: f32,
}

#[derive(Clone, Debug, Default)]
pub struct DefaultAnswerComposer {
    include_debug_metadata: bool,
}

impl DefaultAnswerComposer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_debug_metadata(mut self, include_debug_metadata: bool) -> Self {
        self.include_debug_metadata = include_debug_metadata;
        self
    }
}

impl AnswerComposer for DefaultAnswerComposer {
    fn compose(&self, input: AnswerCompositionInput) -> AnswerComposerFuture {
        let include_debug_metadata = self.include_debug_metadata;

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            #[cfg(feature = "tracing")]
            let selected_agent_id = input
                .selected_agent_id
                .as_ref()
                .map(|agent_id| agent_id.as_str().to_owned());
            #[cfg(feature = "tracing")]
            let source_count = input.retrieved_chunks.len();
            #[cfg(feature = "tracing")]
            let tool_output_count = input.tool_outputs.len();
            #[cfg(feature = "tracing")]
            let guardrail_event_count = input.guardrail_events.len();
            let text = input.draft_answer.trim().to_owned();
            let sources = answer_sources(&input.retrieved_chunks);
            let debug_metadata = if include_debug_metadata {
                Some(json!({
                    "original_question": input.original_question,
                    "selected_agent_id": input.selected_agent_id.as_ref().map(AgentId::as_str),
                    "tool_outputs": input.tool_outputs,
                    "guardrail_events": input.guardrail_events,
                }))
            } else {
                None
            };

            let answer = ComposedAnswer {
                text,
                sources,
                usage: input.usage,
                debug_metadata,
            };
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::composer",
                operation = "compose",
                selected_agent_id = selected_agent_id.as_deref().unwrap_or(""),
                source_count,
                tool_output_count,
                guardrail_event_count,
                include_debug_metadata,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "answer composition completed"
            );
            Ok(answer)
        })
    }
}

pub fn answer_sources(chunks: &[ScoredChunk]) -> Vec<AnswerSource> {
    chunks
        .iter()
        .enumerate()
        .map(|(index, scored)| AnswerSource {
            index: index + 1,
            chunk_id: scored.chunk.id.clone(),
            document_id: scored.chunk.document_id.clone(),
            title: scored.chunk.metadata.title.clone(),
            source: scored.chunk.metadata.source.clone(),
            path: scored.chunk.metadata.path.clone(),
            score: scored.score,
        })
        .collect()
}
