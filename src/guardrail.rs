use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{Error, Result, ScoredChunk};

pub type GuardrailFuture = Pin<Box<dyn Future<Output = Result<GuardrailDecision>> + Send>>;

pub trait Guardrail: Send + Sync {
    fn name(&self) -> &str;

    fn check(&self, request: GuardrailRequest) -> GuardrailFuture;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailStage {
    Input,
    RetrievedContext,
    Output,
}

impl GuardrailStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::RetrievedContext => "retrieved_context",
            Self::Output => "output",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuardrailRequest {
    pub stage: GuardrailStage,
    pub text: String,
    pub original_question: Option<String>,
    pub retrieved_chunks: Vec<ScoredChunk>,
}

impl GuardrailRequest {
    pub fn new(stage: GuardrailStage, text: impl Into<String>) -> Self {
        Self {
            stage,
            text: text.into(),
            original_question: None,
            retrieved_chunks: Vec::new(),
        }
    }

    pub fn with_original_question(mut self, original_question: impl Into<String>) -> Self {
        self.original_question = Some(original_question.into());
        self
    }

    pub fn with_retrieved_chunks(mut self, retrieved_chunks: Vec<ScoredChunk>) -> Self {
        self.retrieved_chunks = retrieved_chunks;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardrailDecision {
    Allow,
    Modify(String),
    Block {
        reason: String,
        fallback_message: Option<String>,
    },
}

impl GuardrailDecision {
    pub fn allow() -> Self {
        Self::Allow
    }

    pub fn modify(text: impl Into<String>) -> Self {
        Self::Modify(text.into())
    }

    pub fn block(reason: impl Into<String>) -> Self {
        Self::Block {
            reason: reason.into(),
            fallback_message: None,
        }
    }

    pub fn block_with_fallback(
        reason: impl Into<String>,
        fallback_message: impl Into<String>,
    ) -> Self {
        Self::Block {
            reason: reason.into(),
            fallback_message: Some(fallback_message.into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailAction {
    Allow,
    Modify,
    Block,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardrailEvent {
    pub guardrail_name: String,
    pub stage: GuardrailStage,
    pub action: GuardrailAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardrailBlock {
    pub guardrail_name: String,
    pub stage: GuardrailStage,
    pub reason: String,
    pub fallback_message: Option<String>,
}

impl GuardrailBlock {
    pub fn into_error(self) -> Error {
        Error::GuardrailBlocked {
            stage: self.stage.as_str().to_owned(),
            reason: self.reason,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuardrailPipelineResult {
    pub text: String,
    pub events: Vec<GuardrailEvent>,
    pub blocked: Option<GuardrailBlock>,
}

#[derive(Clone, Default)]
pub struct GuardrailPipeline {
    guardrails: Vec<Arc<dyn Guardrail>>,
}

impl GuardrailPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_guardrail<G>(&mut self, guardrail: G) -> &mut Self
    where
        G: Guardrail + 'static,
    {
        self.guardrails.push(Arc::new(guardrail));
        self
    }

    pub fn with_guardrail<G>(mut self, guardrail: G) -> Self
    where
        G: Guardrail + 'static,
    {
        self.add_guardrail(guardrail);
        self
    }

    pub fn from_guardrails(guardrails: Vec<Arc<dyn Guardrail>>) -> Self {
        Self { guardrails }
    }

    pub fn len(&self) -> usize {
        self.guardrails.len()
    }

    pub fn is_empty(&self) -> bool {
        self.guardrails.is_empty()
    }

    pub async fn check(&self, request: GuardrailRequest) -> Result<GuardrailPipelineResult> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let mut text = request.text;
        let mut events = Vec::with_capacity(self.guardrails.len());

        for guardrail in &self.guardrails {
            let guardrail_request = GuardrailRequest {
                stage: request.stage,
                text: text.clone(),
                original_question: request.original_question.clone(),
                retrieved_chunks: request.retrieved_chunks.clone(),
            };

            match guardrail.check(guardrail_request).await? {
                GuardrailDecision::Allow => {
                    events.push(GuardrailEvent {
                        guardrail_name: guardrail.name().to_owned(),
                        stage: request.stage,
                        action: GuardrailAction::Allow,
                        reason: None,
                    });
                }
                GuardrailDecision::Modify(modified) => {
                    text = modified;
                    events.push(GuardrailEvent {
                        guardrail_name: guardrail.name().to_owned(),
                        stage: request.stage,
                        action: GuardrailAction::Modify,
                        reason: None,
                    });
                }
                GuardrailDecision::Block {
                    reason,
                    fallback_message,
                } => {
                    let guardrail_name = guardrail.name().to_owned();
                    events.push(GuardrailEvent {
                        guardrail_name: guardrail_name.clone(),
                        stage: request.stage,
                        action: GuardrailAction::Block,
                        reason: Some(reason.clone()),
                    });

                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "arcone_agent::guardrail",
                        operation = "check",
                        stage = request.stage.as_str(),
                        guardrail_count = self.guardrails.len(),
                        event_count = events.len(),
                        blocked = true,
                        blocked_by = guardrail_name.as_str(),
                        elapsed_ms = crate::observability::elapsed_ms(started_at),
                        "guardrail pipeline completed"
                    );
                    return Ok(GuardrailPipelineResult {
                        text,
                        events,
                        blocked: Some(GuardrailBlock {
                            guardrail_name,
                            stage: request.stage,
                            reason,
                            fallback_message,
                        }),
                    });
                }
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::guardrail",
            operation = "check",
            stage = request.stage.as_str(),
            guardrail_count = self.guardrails.len(),
            event_count = events.len(),
            blocked = false,
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "guardrail pipeline completed"
        );
        Ok(GuardrailPipelineResult {
            text,
            events,
            blocked: None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PrivateInfoRedactionGuardrail {
    name: String,
    email_replacement: String,
    phone_replacement: String,
}

impl PrivateInfoRedactionGuardrail {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_email_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.email_replacement = replacement.into();
        self
    }

    pub fn with_phone_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.phone_replacement = replacement.into();
        self
    }

    fn redact(&self, text: &str) -> String {
        let redacted = EMAIL_RE
            .replace_all(text, self.email_replacement.as_str())
            .into_owned();
        PHONE_RE
            .replace_all(&redacted, self.phone_replacement.as_str())
            .into_owned()
    }
}

impl Default for PrivateInfoRedactionGuardrail {
    fn default() -> Self {
        Self {
            name: "private_info_redaction".to_owned(),
            email_replacement: "[REDACTED_EMAIL]".to_owned(),
            phone_replacement: "[REDACTED_PHONE]".to_owned(),
        }
    }
}

impl Guardrail for PrivateInfoRedactionGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self, request: GuardrailRequest) -> GuardrailFuture {
        let redacted = self.redact(&request.text);
        Box::pin(async move {
            if redacted == request.text {
                Ok(GuardrailDecision::Allow)
            } else {
                Ok(GuardrailDecision::Modify(redacted))
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct EmptyAnswerGuardrail {
    name: String,
    reason: String,
}

impl EmptyAnswerGuardrail {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for EmptyAnswerGuardrail {
    fn default() -> Self {
        Self {
            name: "empty_answer_rejection".to_owned(),
            reason: "answer is empty".to_owned(),
        }
    }
}

impl Guardrail for EmptyAnswerGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self, request: GuardrailRequest) -> GuardrailFuture {
        let reason = self.reason.clone();
        Box::pin(async move {
            if request.stage == GuardrailStage::Output && request.text.trim().is_empty() {
                Ok(GuardrailDecision::block(reason))
            } else {
                Ok(GuardrailDecision::Allow)
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct NoHallucinationFallbackGuardrail {
    name: String,
    reason: String,
    fallback_message: String,
}

impl NoHallucinationFallbackGuardrail {
    pub fn new(fallback_message: impl Into<String>) -> Self {
        Self {
            fallback_message: fallback_message.into(),
            ..Self::default()
        }
    }
}

impl Default for NoHallucinationFallbackGuardrail {
    fn default() -> Self {
        Self {
            name: "no_hallucination_fallback".to_owned(),
            reason: "retrieved context is empty".to_owned(),
            fallback_message:
                "The knowledge base does not contain enough information to answer this question."
                    .to_owned(),
        }
    }
}

impl Guardrail for NoHallucinationFallbackGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self, request: GuardrailRequest) -> GuardrailFuture {
        let reason = self.reason.clone();
        let fallback_message = self.fallback_message.clone();
        Box::pin(async move {
            if request.stage == GuardrailStage::RetrievedContext
                && (request.text.trim().is_empty() || request.retrieved_chunks.is_empty())
            {
                Ok(GuardrailDecision::block_with_fallback(
                    reason,
                    fallback_message,
                ))
            } else {
                Ok(GuardrailDecision::Allow)
            }
        })
    }
}

static EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b").expect("valid email regex")
});

static PHONE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\+?\d[\d .()\-]{7,}\d)\b").expect("valid phone regex"));
