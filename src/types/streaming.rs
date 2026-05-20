use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{chat::FinishReason, message::Role, tool::ToolCall, usage::Usage};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatStreamChunk {
    pub id: String,
    pub choices: Vec<StreamChoice>,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StreamChoice {
    pub delta: ChatDelta,
    pub finish_reason: Option<FinishReason>,
    pub index: u32,
    #[serde(default)]
    pub logprobs: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub role: Option<Role>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}
