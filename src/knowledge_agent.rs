use std::sync::Arc;

use crate::Result;
use crate::{
    Agent, AgentResponse, AnswerComposer, AnswerCompositionInput, ChunkId, ComposedAnswer,
    DocumentId, Error, GuardrailEvent, GuardrailPipeline, GuardrailPipelineResult,
    GuardrailRequest, GuardrailStage, Retriever, ScoredChunk, ToolOutput,
};

pub struct KnowledgeAgent {
    agent: Agent,
    retriever: Arc<dyn Retriever>,
    options: KnowledgeAgentOptions,
    guardrails: Option<Arc<GuardrailPipeline>>,
    answer_composer: Option<Arc<dyn AnswerComposer>>,
}

impl KnowledgeAgent {
    pub fn new<R>(agent: Agent, retriever: R) -> Self
    where
        R: Retriever + 'static,
    {
        Self::from_retriever(agent, Arc::new(retriever))
    }

    pub fn from_retriever(agent: Agent, retriever: Arc<dyn Retriever>) -> Self {
        Self {
            agent,
            retriever,
            options: KnowledgeAgentOptions::default(),
            guardrails: None,
            answer_composer: None,
        }
    }

    pub fn with_options(mut self, options: KnowledgeAgentOptions) -> Self {
        self.options = options;
        self
    }

    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    pub fn options(&self) -> &KnowledgeAgentOptions {
        &self.options
    }

    pub fn with_guardrails(mut self, guardrails: impl Into<Arc<GuardrailPipeline>>) -> Self {
        self.guardrails = Some(guardrails.into());
        self
    }

    pub fn set_guardrails(&mut self, guardrails: impl Into<Arc<GuardrailPipeline>>) -> &mut Self {
        self.guardrails = Some(guardrails.into());
        self
    }

    pub fn with_answer_composer<C>(mut self, composer: C) -> Self
    where
        C: AnswerComposer + 'static,
    {
        self.answer_composer = Some(Arc::new(composer));
        self
    }

    pub fn with_shared_answer_composer(mut self, composer: Arc<dyn AnswerComposer>) -> Self {
        self.answer_composer = Some(composer);
        self
    }

    pub async fn ask(&mut self, question: impl Into<String>) -> Result<KnowledgeAgentResponse> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let original_question = question.into();
        let input_guardrails = self
            .apply_guardrails(
                GuardrailStage::Input,
                original_question.clone(),
                Some(original_question.clone()),
                Vec::new(),
            )
            .await?;
        let mut guardrail_events = input_guardrails.events;
        if let Some(block) = input_guardrails.blocked {
            return Err(block.into_error());
        }
        let question = input_guardrails.text;

        let retrieved_chunks = self
            .retriever
            .retrieve(&question, self.options.top_k)
            .await?;
        let included_chunks =
            limit_context_chunks(retrieved_chunks, self.options.max_context_chars);
        let context = context_block(&included_chunks);
        let context_guardrails = self
            .apply_guardrails(
                GuardrailStage::RetrievedContext,
                context,
                Some(question.clone()),
                included_chunks.clone(),
            )
            .await?;
        guardrail_events.extend(context_guardrails.events);

        if let Some(block) = context_guardrails.blocked {
            let fallback = block
                .fallback_message
                .unwrap_or_else(|| self.options.fallback_message.clone());
            let response = self
                .fallback_response(question, fallback, included_chunks, guardrail_events)
                .await?;
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::knowledge_agent",
                operation = "ask",
                agent_id = self.agent.id().as_str(),
                retrieved_count = response.retrieved_chunks.len(),
                guardrail_event_count = response.guardrail_events.len(),
                used_fallback = true,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "knowledge agent request completed"
            );
            return Ok(response);
        }

        let context = context_guardrails.text;
        if included_chunks.is_empty() || context.trim().is_empty() {
            let response = self
                .fallback_response(
                    question,
                    self.options.fallback_message.clone(),
                    included_chunks,
                    guardrail_events,
                )
                .await?;
            #[cfg(feature = "tracing")]
            tracing::info!(
                target: "arcone_agent::knowledge_agent",
                operation = "ask",
                agent_id = self.agent.id().as_str(),
                retrieved_count = response.retrieved_chunks.len(),
                guardrail_event_count = response.guardrail_events.len(),
                used_fallback = true,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "knowledge agent request completed"
            );
            return Ok(response);
        }

        let prompt = compose_prompt(&question, &context);
        let mut agent_response = self.agent.ask(prompt).await?;
        guardrail_events.extend(agent_response.guardrail_events.clone());
        let mut answer = agent_response
            .content()
            .map(str::to_owned)
            .ok_or(Error::NoAssistantMessage)?;

        let output_guardrails = self
            .apply_guardrails(
                GuardrailStage::Output,
                answer,
                Some(question.clone()),
                included_chunks.clone(),
            )
            .await?;
        guardrail_events.extend(output_guardrails.events);

        if let Some(block) = output_guardrails.blocked {
            if let Some(fallback) = block.fallback_message.clone() {
                let response = self
                    .fallback_response(question, fallback, included_chunks, guardrail_events)
                    .await?;
                #[cfg(feature = "tracing")]
                tracing::info!(
                    target: "arcone_agent::knowledge_agent",
                    operation = "ask",
                    agent_id = self.agent.id().as_str(),
                    retrieved_count = response.retrieved_chunks.len(),
                    guardrail_event_count = response.guardrail_events.len(),
                    used_fallback = true,
                    elapsed_ms = crate::observability::elapsed_ms(started_at),
                    "knowledge agent request completed"
                );
                return Ok(response);
            }
            return Err(block.into_error());
        }

        answer = output_guardrails.text;
        if let Some(composed) = agent_response.composed_answer.as_mut() {
            composed.text = answer.clone();
        } else {
            agent_response.set_message_content(answer.clone());
        }

        let mut response = KnowledgeAgentResponse {
            answer,
            agent_response: Some(agent_response),
            retrieved_chunks: included_chunks.clone(),
            sources: sources_for_chunks(&included_chunks),
            used_fallback: false,
            guardrail_events,
            composed_answer: None,
        };
        self.compose_response(&question, &mut response).await?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::knowledge_agent",
            operation = "ask",
            agent_id = self.agent.id().as_str(),
            retrieved_count = response.retrieved_chunks.len(),
            guardrail_event_count = response.guardrail_events.len(),
            used_fallback = false,
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "knowledge agent request completed"
        );
        Ok(response)
    }

    async fn fallback_response(
        &self,
        question: String,
        answer: String,
        retrieved_chunks: Vec<ScoredChunk>,
        guardrail_events: Vec<GuardrailEvent>,
    ) -> Result<KnowledgeAgentResponse> {
        let mut response = KnowledgeAgentResponse::fallback(answer, retrieved_chunks);
        response.guardrail_events = guardrail_events;
        self.compose_response(&question, &mut response).await?;
        Ok(response)
    }

    async fn compose_response(
        &self,
        original_question: &str,
        response: &mut KnowledgeAgentResponse,
    ) -> Result<()> {
        let Some(composer) = &self.answer_composer else {
            return Ok(());
        };

        let input = AnswerCompositionInput {
            original_question: original_question.to_owned(),
            selected_agent_id: Some(self.agent.id().clone()),
            tool_outputs: response
                .agent_response
                .as_ref()
                .map(|agent_response| ToolOutput::from_history(&agent_response.history))
                .unwrap_or_default(),
            retrieved_chunks: response.retrieved_chunks.clone(),
            draft_answer: response.answer.clone(),
            guardrail_events: response.guardrail_events.clone(),
            usage: response
                .agent_response
                .as_ref()
                .and_then(|agent_response| agent_response.usage.clone()),
        };
        let composed = composer.compose(input).await?;
        response.answer = composed.text.clone();
        response.composed_answer = Some(composed);
        Ok(())
    }

    async fn apply_guardrails(
        &self,
        stage: GuardrailStage,
        text: String,
        original_question: Option<String>,
        retrieved_chunks: Vec<ScoredChunk>,
    ) -> Result<GuardrailPipelineResult> {
        let Some(guardrails) = &self.guardrails else {
            return Ok(GuardrailPipelineResult {
                text,
                events: Vec::new(),
                blocked: None,
            });
        };

        let mut request =
            GuardrailRequest::new(stage, text).with_retrieved_chunks(retrieved_chunks);
        if let Some(original_question) = original_question {
            request = request.with_original_question(original_question);
        }
        guardrails.check(request).await
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnowledgeAgentOptions {
    pub top_k: usize,
    pub max_context_chars: usize,
    pub fallback_message: String,
}

impl KnowledgeAgentOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_max_context_chars(mut self, max_context_chars: usize) -> Self {
        self.max_context_chars = max_context_chars;
        self
    }

    pub fn with_fallback_message(mut self, fallback_message: impl Into<String>) -> Self {
        self.fallback_message = fallback_message.into();
        self
    }
}

impl Default for KnowledgeAgentOptions {
    fn default() -> Self {
        Self {
            top_k: 4,
            max_context_chars: 6_000,
            fallback_message:
                "The knowledge base does not contain enough information to answer this question."
                    .to_owned(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KnowledgeAgentResponse {
    pub answer: String,
    pub agent_response: Option<AgentResponse>,
    pub retrieved_chunks: Vec<ScoredChunk>,
    pub sources: Vec<KnowledgeSource>,
    pub used_fallback: bool,
    pub guardrail_events: Vec<GuardrailEvent>,
    pub composed_answer: Option<ComposedAnswer>,
}

impl KnowledgeAgentResponse {
    pub fn content(&self) -> &str {
        self.composed_answer
            .as_ref()
            .map(|answer| answer.text.as_str())
            .unwrap_or(&self.answer)
    }

    fn fallback(answer: String, retrieved_chunks: Vec<ScoredChunk>) -> Self {
        Self {
            sources: sources_for_chunks(&retrieved_chunks),
            answer,
            agent_response: None,
            retrieved_chunks,
            used_fallback: true,
            guardrail_events: Vec::new(),
            composed_answer: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeSource {
    pub index: usize,
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub title: Option<String>,
    pub source: Option<String>,
    pub path: Option<String>,
    pub score: f32,
}

fn limit_context_chunks(chunks: Vec<ScoredChunk>, max_context_chars: usize) -> Vec<ScoredChunk> {
    if max_context_chars == 0 {
        return Vec::new();
    }

    let mut included = Vec::new();
    let mut used_chars = 0;

    for mut scored in chunks {
        let source_line = source_line(included.len() + 1, &scored);
        let overhead = char_len(&source_line) + 2;
        if used_chars + overhead >= max_context_chars {
            break;
        }

        let available_content = max_context_chars - used_chars - overhead;
        let content_len = char_len(&scored.chunk.content);
        if content_len > available_content {
            scored.chunk.content = take_chars(&scored.chunk.content, available_content);
        }

        used_chars += overhead + char_len(&scored.chunk.content);
        included.push(scored);

        if used_chars >= max_context_chars {
            break;
        }
    }

    included
}

fn context_block(chunks: &[ScoredChunk]) -> String {
    let mut context = String::new();
    for (index, chunk) in chunks.iter().enumerate() {
        if !context.is_empty() {
            context.push_str("\n\n");
        }

        context.push_str(&source_line(index + 1, chunk));
        context.push('\n');
        context.push_str(&chunk.chunk.content);
    }

    context
}

fn compose_prompt(question: &str, context: &str) -> String {
    format!(
        "Use only the knowledge context below to answer the question. \
If the context is insufficient, say that the data is not available in the knowledge base. \
Do not use outside knowledge. Cite relevant sources with bracket numbers like [1].\n\n\
Knowledge context:\n{context}\n\nQuestion:\n{question}"
    )
}

fn sources_for_chunks(chunks: &[ScoredChunk]) -> Vec<KnowledgeSource> {
    chunks
        .iter()
        .enumerate()
        .map(|(index, scored)| KnowledgeSource {
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

fn source_line(index: usize, scored: &ScoredChunk) -> String {
    let mut parts = Vec::new();
    if let Some(title) = &scored.chunk.metadata.title {
        parts.push(format!("title={title}"));
    }
    if let Some(source) = &scored.chunk.metadata.source {
        parts.push(format!("source={source}"));
    }
    if let Some(path) = &scored.chunk.metadata.path {
        parts.push(format!("path={path}"));
    }

    if parts.is_empty() {
        format!(
            "[{index}] document={} chunk={}",
            scored.chunk.document_id, scored.chunk.id
        )
    } else {
        format!("[{index}] {}", parts.join(" "))
    }
}

fn take_chars(value: &str, count: usize) -> String {
    value.chars().take(count).collect()
}

fn char_len(value: &str) -> usize {
    value.chars().count()
}
