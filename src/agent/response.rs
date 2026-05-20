use crate::{ChatMessage, ComposedAnswer, FinishReason, GuardrailEvent, Usage};

#[derive(Clone, Debug)]
pub struct AgentResponse {
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
    pub history: Vec<ChatMessage>,
    pub guardrail_events: Vec<GuardrailEvent>,
    pub composed_answer: Option<ComposedAnswer>,
}

impl AgentResponse {
    pub fn content(&self) -> Option<&str> {
        self.composed_answer
            .as_ref()
            .map(|answer| answer.text.as_str())
            .or(self.message.content.as_deref())
    }

    pub fn reasoning_content(&self) -> Option<&str> {
        self.message.reasoning_content.as_deref()
    }

    pub(crate) fn set_message_content(&mut self, content: String) {
        self.message.content = Some(content.clone());
        if let Some(last_message) = self.history.last_mut()
            && last_message.role == self.message.role
        {
            last_message.content = Some(content);
        }
    }
}
