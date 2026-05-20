mod chat;
mod message;
mod model;
mod request;
mod streaming;
mod tool;
mod usage;

pub use chat::{ChatRequest, ChatResponse, FinishReason};
pub use message::{ChatMessage, Role};
pub use model::DeepSeekModel;
pub use request::{
    ReasoningEffort, ResponseFormat, ResponseFormatType, StopSequences, StreamOptions,
    ThinkingConfig, ThinkingMode,
};
pub use streaming::{ChatDelta, ChatStreamChunk, StreamChoice};
pub use tool::{
    FunctionCall, FunctionDefinition, NamedToolChoice, ToolCall, ToolChoice, ToolChoiceMode,
    ToolDefinition,
};
pub use usage::Usage;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn serializes_current_deepseek_v4_request_shape() {
        let request = ChatRequest::new(
            DeepSeekModel::V4Pro,
            vec![
                ChatMessage::system("Return json."),
                ChatMessage::user("Ping"),
            ],
        )
        .with_thinking(ThinkingConfig::enabled())
        .with_reasoning_effort(ReasoningEffort::High)
        .with_response_format(ResponseFormat::json_object())
        .with_tools(vec![ToolDefinition::function(
            FunctionDefinition::new("lookup")
                .description("Lookup a value")
                .parameters(json!({
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"}
                    },
                    "required": ["key"],
                    "additionalProperties": false
                })),
        )])
        .with_tool_choice(ToolChoice::auto());

        let value = serde_json::to_value(request).expect("request serializes");

        assert_eq!(value["model"], "deepseek-v4-pro");
        assert_eq!(value["thinking"]["type"], "enabled");
        assert_eq!(value["reasoning_effort"], "high");
        assert_eq!(value["response_format"]["type"], "json_object");
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tool_choice"], "auto");
    }

    #[test]
    fn serializes_named_tool_choice_as_deepseek_shape() {
        let value = serde_json::to_value(ToolChoice::function("lookup")).expect("serializes");

        assert_eq!(
            value,
            json!({
                "type": "function",
                "function": {"name": "lookup"}
            })
        );
    }
}
