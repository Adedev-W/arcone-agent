use std::{env, ffi::OsString, sync::Arc, sync::Mutex, time::Duration};

use arcone_agent::{
    Agent, AgentConfig, AgentOptions, AgentProfile, ChatMessage, ChatRequest, DeepSeekClient,
    DeepSeekConfig, DeepSeekModel, Error, FunctionDefinition, FunctionTool, InMemorySessionStore,
    MemoryStore, ReasoningEffort, ResponseFormat, SessionId, StreamEvent, ThinkingConfig, Tool,
    ToolCall, ToolDefinition, ToolFuture, ToolRegistry, TypedFunctionTool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn client_chat_posts_deepseek_request_and_parses_response() {
    let body = json!({
        "id": "chat-1",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "Hi", "reasoning_content": "brief"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let client = test_client(base_url);

    let response = client
        .chat(ChatRequest::new(
            DeepSeekModel::V4Flash,
            vec![ChatMessage::user("Hello")],
        ))
        .await
        .expect("chat response");

    assert_eq!(
        response
            .first_message()
            .and_then(|msg| msg.content.as_deref()),
        Some("Hi")
    );
    assert_eq!(
        response
            .first_message()
            .and_then(|msg| msg.reasoning_content.as_deref()),
        Some("brief")
    );

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let lower = request.to_ascii_lowercase();

    assert!(lower.contains("post /chat/completions http/1.1"));
    assert!(lower.contains("authorization: bearer test-key"));
    assert!(request.contains("\"model\":\"deepseek-v4-flash\""));
    assert!(request.contains("\"content\":\"Hello\""));
}

#[tokio::test]
async fn stream_chat_parses_sse_events() {
    let stream_body = concat!(
        ": keep-alive\n\n",
        "data: {\"id\":\"stream-1\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think\"},\"finish_reason\":null,\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-pro\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n",
        "data: {\"id\":\"stream-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\",\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-pro\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, server) = spawn_http_sequence(vec![stream_response(stream_body)]).await;
    let client = test_client(base_url);

    let mut stream = client
        .stream_chat(ChatRequest::new(
            DeepSeekModel::V4Pro,
            vec![ChatMessage::user("Stream")],
        ))
        .await
        .expect("stream");

    let first = stream.next_event().await.expect("first event");
    let second = stream.next_event().await.expect("second event");
    let done = stream.next_event().await.expect("done event");
    let after_done = stream.next_event().await.expect("after done");

    let Some(StreamEvent::Chunk(first)) = first else {
        panic!("expected first chunk");
    };
    assert_eq!(
        first.choices[0].delta.reasoning_content.as_deref(),
        Some("think")
    );

    let Some(StreamEvent::Chunk(second)) = second else {
        panic!("expected second chunk");
    };
    assert_eq!(second.choices[0].delta.content.as_deref(), Some("Hello"));
    assert!(matches!(done, Some(StreamEvent::Done)));
    assert!(after_done.is_none());

    let requests = server.await.expect("server task");
    assert!(requests[0].contains("\"stream\":true"));
}

#[test]
fn agent_from_env_uses_optional_deepseek_env_config() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let _api_key = EnvVarGuard::set("DEEPSEEK_API_KEY", "env-key");
    let _base_url = EnvVarGuard::set("DEEPSEEK_BASE_URL", "http://127.0.0.1:9");
    let _model = EnvVarGuard::set("DEEPSEEK_MODEL", "deepseek-v4-pro");

    let config = DeepSeekConfig::from_env().expect("config from env");
    let agent = Agent::from_env().expect("agent from env");

    assert_eq!(config.base_url(), "http://127.0.0.1:9");
    assert_eq!(config.model(), &DeepSeekModel::V4Pro);
    assert_eq!(agent.config().model, None);
}

#[tokio::test]
async fn agent_ask_text_returns_assistant_content() {
    let body = json!({
        "id": "chat-text",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "plain answer"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let mut agent = Agent::new(test_client(base_url))
        .system("Answer briefly.")
        .thinking_disabled()
        .reasoning(ReasoningEffort::High)
        .max_tokens(64);

    let text = agent.ask_text("Short answer").await.expect("text response");

    assert_eq!(text, "plain answer");
    let requests = server.await.expect("server task");
    let request = json_request_body(&requests[0]);
    assert_eq!(request["messages"][0]["content"], "Answer briefly.");
    assert_eq!(request["thinking"]["type"], "disabled");
    assert!(request.get("reasoning_effort").is_none());
    assert_eq!(request["max_tokens"], 64);
}

#[tokio::test]
async fn thinking_disabled_request_does_not_receive_default_reasoning_effort() {
    let body = json!({
        "id": "chat-default-thinking-off",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "plain answer"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let client = DeepSeekClient::new(
        DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_timeout(Duration::from_secs(5)),
    )
    .expect("client");
    let mut agent = Agent::new(client).thinking_disabled();

    let text = agent.ask_text("Short answer").await.expect("text response");

    assert_eq!(text, "plain answer");
    let requests = server.await.expect("server task");
    let request = json_request_body(&requests[0]);
    assert_eq!(request["thinking"]["type"], "disabled");
    assert!(request.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn agent_stream_accumulates_text_and_final_response() {
    let stream_body = concat!(
        "data: {\"id\":\"stream-agent\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"thinking\"},\"finish_reason\":null,\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n",
        "data: {\"id\":\"stream-agent\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null,\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n",
        "data: {\"id\":\"stream-agent\",\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\",\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":1,\"prompt_tokens\":1,\"total_tokens\":2}}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, server) = spawn_http_sequence(vec![stream_response(stream_body)]).await;
    let mut agent = Agent::new(test_client(base_url)).system("Stream briefly.");
    let mut stream = agent.stream("Say hello").await.expect("agent stream");

    assert_eq!(
        stream.next_text().await.expect("first text"),
        Some("Hello".to_owned())
    );
    assert_eq!(
        stream.next_text().await.expect("second text"),
        Some(" world".to_owned())
    );
    assert_eq!(stream.next_text().await.expect("done"), None);
    let response = stream.finish().await.expect("stream response");

    assert_eq!(response.content(), Some("Hello world"));
    assert_eq!(response.reasoning_content(), Some("thinking"));
    assert_eq!(agent.history().len(), 2);
    let requests = server.await.expect("server task");
    assert!(requests[0].contains("\"stream\":true"));
}

#[tokio::test]
async fn agent_stream_drop_rolls_back_unfinished_history() {
    let stream_body = concat!(
        "data: {\"id\":\"stream-agent\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null,\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, server) = spawn_http_sequence(vec![stream_response(stream_body)]).await;
    let mut agent = Agent::new(test_client(base_url));

    {
        let mut stream = agent.stream("Say hello").await.expect("agent stream");
        assert_eq!(
            stream.next_text().await.expect("first text"),
            Some("partial".to_owned())
        );
    }

    assert!(agent.history().is_empty());
    let requests = server.await.expect("server task");
    assert!(requests[0].contains("\"stream\":true"));
}

#[tokio::test]
async fn agent_stream_open_failure_rolls_back_user_message() {
    let (base_url, server) = spawn_http_sequence(vec![status_response(
        "500 Internal Server Error",
        "application/json",
        "{\"error\":\"boom\"}",
    )])
    .await;
    let mut agent = Agent::new(test_client(base_url));

    let error = match agent.stream("Say hello").await {
        Ok(_) => panic!("stream open should fail"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::Api { .. }));
    assert!(agent.history().is_empty());
    let requests = server.await.expect("server task");
    assert!(requests[0].contains("\"stream\":true"));
}

#[tokio::test]
async fn agent_stream_errors_for_tool_calls() {
    let stream_body = concat!(
        "data: {\"id\":\"stream-tool\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"key\\\":\\\"answer\\\"}\"}}]},\"finish_reason\":null,\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-flash\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, server) = spawn_http_sequence(vec![stream_response(stream_body)]).await;
    let mut agent = Agent::new(test_client(base_url));
    let mut stream = agent.stream("Use a tool").await.expect("agent stream");

    let error = stream.next_text().await.expect_err("stream tool call");

    assert!(matches!(error, Error::StreamingToolCallsUnsupported));
    let requests = server.await.expect("server task");
    assert!(requests[0].contains("\"stream\":true"));
}

#[tokio::test]
async fn agent_runs_tool_loop_and_preserves_reasoning_content() {
    let tool_response = json!({
        "id": "chat-1",
        "choices": [{
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "reasoning_content": "need lookup",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"key\":\"answer\"}"}
                }]
            },
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let final_response = json!({
        "id": "chat-2",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "value is 42"},
            "logprobs": null
        }],
        "created": 2,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![
        json_response(&tool_response),
        json_response(&final_response),
    ])
    .await;
    let mut agent = Agent::new(test_client(base_url))
        .with_system_prompt("Use tools when useful.")
        .with_tool(lookup_tool());

    let response = agent.ask("Find the answer").await.expect("agent response");

    assert_eq!(response.content(), Some("value is 42"));
    assert_eq!(agent.history().len(), 4);
    assert_eq!(
        agent.history()[1].reasoning_content.as_deref(),
        Some("need lookup")
    );
    assert_eq!(
        agent.history()[2].content.as_deref(),
        Some("{\"value\":42}")
    );

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains("\"role\":\"tool\""));
    assert!(requests[1].contains("\"tool_call_id\":\"call_1\""));
}

#[test]
fn tool_registry_stores_definitions_and_rejects_duplicates() {
    let mut registry = ToolRegistry::new();
    registry.add_tool(lookup_tool()).expect("add lookup");

    assert!(registry.get("lookup").is_some());
    let definitions = registry.definitions();
    assert_eq!(definitions.len(), 1);
    assert_eq!(definitions[0].function.name, "lookup");

    let error = match registry.add_tool(lookup_tool()) {
        Ok(_) => panic!("duplicate tool should fail"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::DuplicateTool(name) if name == "lookup"));
}

#[tokio::test]
async fn agents_can_share_tool_registry() {
    let first_tool_response = tool_call_response("lookup", "{\"key\":\"answer\"}");
    let first_final_response = json!({
        "id": "chat-shared-1-final",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "first used shared tool"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let second_tool_response = tool_call_response("lookup", "{\"key\":\"answer\"}");
    let second_final_response = json!({
        "id": "chat-shared-2-final",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "second used shared tool"},
            "logprobs": null
        }],
        "created": 2,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![
        json_response(&first_tool_response),
        json_response(&first_final_response),
        json_response(&second_tool_response),
        json_response(&second_final_response),
    ])
    .await;

    let mut registry = ToolRegistry::new();
    registry.add_tool(lookup_tool()).expect("add shared tool");
    let registry = Arc::new(registry);
    let client = test_client(base_url);
    let mut first_agent = Agent::new(client.clone())
        .with_tool_registry(registry.clone())
        .expect("first registry");
    let mut second_agent = Agent::new(client)
        .tool_registry(registry)
        .expect("second registry");

    let first = first_agent
        .ask("first question")
        .await
        .expect("first agent");
    let second = second_agent
        .ask("second question")
        .await
        .expect("second agent");

    assert_eq!(first.content(), Some("first used shared tool"));
    assert_eq!(second.content(), Some("second used shared tool"));
    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 4);
    assert!(requests[0].contains("\"name\":\"lookup\""));
    assert!(requests[2].contains("\"name\":\"lookup\""));
}

#[tokio::test]
async fn agent_combines_private_and_shared_tools() {
    let body = json!({
        "id": "chat-combined-tools",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "combined"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let mut registry = ToolRegistry::new();
    registry.add_tool(lookup_tool()).expect("add shared tool");
    let mut agent = Agent::new(test_client(base_url))
        .with_tool_registry(Arc::new(registry))
        .expect("registry")
        .with_tool(private_lookup_tool());

    let response = agent.ask("show tools").await.expect("agent response");

    assert_eq!(response.content(), Some("combined"));
    let requests = server.await.expect("server task");
    let request = json_request_body(&requests[0]);
    let tool_names = request["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| tool["function"]["name"].as_str().expect("tool name"))
        .collect::<Vec<_>>();
    assert!(tool_names.contains(&"lookup"));
    assert!(tool_names.contains(&"private_lookup"));
}

#[test]
fn agent_registry_rejects_duplicate_private_tool_names() {
    let mut registry = ToolRegistry::new();
    registry.add_tool(lookup_tool()).expect("add shared tool");

    let error = match Agent::new(test_client("http://127.0.0.1:9".to_owned()))
        .with_tool(lookup_tool())
        .with_tool_registry(Arc::new(registry))
    {
        Ok(_) => panic!("duplicate private/shared tool should fail"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::DuplicateTool(name) if name == "lookup"));
}

#[test]
fn agent_checked_private_tools_reject_duplicate_names() {
    let mut agent = Agent::new(test_client("http://127.0.0.1:9".to_owned()));
    agent.try_add_tool(lookup_tool()).expect("first tool");

    let error = match agent.try_add_tool(lookup_tool()) {
        Ok(_) => panic!("duplicate private tool should fail"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::DuplicateTool(name) if name == "lookup"));
}

#[tokio::test]
async fn agent_config_builds_profile_request() {
    let body = json!({
        "id": "chat-config",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "{\"ok\":true}"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-pro",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let config = AgentConfig::new("planner")
        .with_name("Planner")
        .with_role_description("Breaks down tasks")
        .with_system_prompt("Plan carefully.")
        .with_model(DeepSeekModel::V4Pro)
        .with_thinking(ThinkingConfig::disabled())
        .with_reasoning_effort(ReasoningEffort::High)
        .with_max_tokens(123)
        .with_max_tool_rounds(3)
        .with_response_format(ResponseFormat::json_object());
    let mut agent = Agent::with_config(test_client(base_url), config);

    let response = agent.ask("Return JSON").await.expect("agent response");

    assert_eq!(response.content(), Some("{\"ok\":true}"));
    assert_eq!(agent.id().as_str(), "planner");
    assert_eq!(agent.profile().name, "Planner");
    assert_eq!(agent.config().max_tool_rounds, 3);

    let requests = server.await.expect("server task");
    let request = json_request_body(&requests[0]);

    assert_eq!(request["model"], "deepseek-v4-pro");
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][0]["content"], "Plan carefully.");
    assert_eq!(request["messages"][1]["role"], "user");
    assert_eq!(request["messages"][1]["content"], "Return JSON");
    assert_eq!(request["thinking"]["type"], "disabled");
    assert!(request.get("reasoning_effort").is_none());
    assert_eq!(request["max_tokens"], 123);
    assert_eq!(request["response_format"]["type"], "json_object");
}

#[tokio::test]
async fn cloned_agent_config_does_not_share_history() {
    let body = json!({
        "id": "chat-one",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "first"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let client = test_client(base_url);
    let config = AgentConfig::new("researcher").with_system_prompt("Research carefully.");
    let mut first = Agent::with_config(client.clone(), config.clone());
    let second = Agent::with_config(client, config);

    let response = first.ask("First question").await.expect("first response");

    assert_eq!(response.content(), Some("first"));
    assert_eq!(first.history().len(), 2);
    assert!(second.history().is_empty());
    assert_eq!(first.config(), second.config());

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn agents_share_history_through_session_store() {
    let first_body = json!({
        "id": "chat-session-1",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "first answer"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let second_body = json!({
        "id": "chat-session-2",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "second answer"},
            "logprobs": null
        }],
        "created": 2,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![
        json_response(&first_body),
        json_response(&second_body),
    ])
    .await;
    let client = test_client(base_url);
    let store: Arc<dyn MemoryStore> = Arc::new(InMemorySessionStore::new());
    let session = SessionId::new("agent-shared-session");
    let mut first_agent = Agent::new(client.clone()).with_session(session.clone(), store.clone());
    let mut second_agent = Agent::new(client).with_session(session, store);

    let first_response = first_agent.ask("first question").await.unwrap();
    let second_response = second_agent.ask("second question").await.unwrap();

    assert_eq!(first_response.content(), Some("first answer"));
    assert_eq!(second_response.content(), Some("second answer"));

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 2);

    let second_request = json_request_body(&requests[1]);
    let messages = second_request["messages"]
        .as_array()
        .expect("messages array");

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "first question");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "first answer");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"], "second question");
}

#[test]
fn legacy_agent_options_convert_to_agent_config() {
    let options = AgentOptions {
        model: Some(DeepSeekModel::V4Pro),
        system_prompt: Some("Legacy prompt".to_owned()),
        thinking: Some(ThinkingConfig::disabled()),
        reasoning_effort: Some(ReasoningEffort::High),
        response_format: Some(ResponseFormat::text()),
        max_tool_rounds: 2,
    };
    let agent = Agent::with_options(test_client("http://127.0.0.1:9".to_owned()), options);

    assert_eq!(
        agent.profile().system_prompt.as_deref(),
        Some("Legacy prompt")
    );
    assert_eq!(agent.config().model, Some(DeepSeekModel::V4Pro));
    assert_eq!(agent.config().thinking, Some(ThinkingConfig::disabled()));
    assert_eq!(agent.config().reasoning_effort, Some(ReasoningEffort::High));
    assert_eq!(agent.config().response_format, Some(ResponseFormat::text()));
    assert_eq!(agent.config().max_tool_rounds, 2);
}

#[test]
fn agent_config_can_be_built_from_profile() {
    let profile = AgentProfile::new("composer")
        .with_name("Composer")
        .with_role_description("Composes final answers")
        .with_system_prompt("Compose a concise answer.");
    let config = AgentConfig::new("placeholder").with_profile(profile);

    assert_eq!(config.profile.id.as_str(), "composer");
    assert_eq!(config.profile.name, "Composer");
    assert_eq!(
        config.profile.role_description.as_deref(),
        Some("Composes final answers")
    );
    assert_eq!(
        config.profile.system_prompt.as_deref(),
        Some("Compose a concise answer.")
    );
}

#[tokio::test]
async fn agent_errors_for_unknown_tool() {
    let body = tool_call_response("missing", "{\"key\":\"answer\"}");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let mut agent = Agent::new(test_client(base_url));

    let error = agent
        .ask("Find the answer")
        .await
        .expect_err("missing tool");

    assert!(matches!(error, Error::UnknownTool(name) if name == "missing"));
    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn agent_errors_for_invalid_tool_arguments() {
    let body = tool_call_response("lookup", "not json");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let mut agent = Agent::new(test_client(base_url)).with_tool(lookup_tool());

    let error = agent
        .ask("Find the answer")
        .await
        .expect_err("invalid args");

    assert!(matches!(error, Error::InvalidToolArguments { name, .. } if name == "lookup"));
    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn agent_enforces_max_tool_rounds() {
    let body = tool_call_response("lookup", "{\"key\":\"answer\"}");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let mut agent = Agent::new(test_client(base_url))
        .with_tool(lookup_tool())
        .with_max_tool_rounds(0);

    let error = agent.ask("Find the answer").await.expect_err("max rounds");

    assert!(matches!(error, Error::ToolLoopExceeded { max_rounds: 0 }));
    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn agent_ask_json_sets_response_format_and_parses_content() {
    let body = json!({
        "id": "chat-json",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "{\"ok\":true}"},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let mut agent = Agent::new(test_client(base_url));

    let parsed: Value = agent.ask_json("Return json").await.expect("json response");

    assert_eq!(parsed, json!({"ok": true}));
    let requests = server.await.expect("server task");
    assert!(requests[0].contains("\"response_format\":{\"type\":\"json_object\"}"));
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LookupArgs {
    key: String,
}

#[derive(Debug, Serialize)]
struct LookupOutput {
    value: u32,
}

#[tokio::test]
async fn typed_function_tool_parses_args_and_serializes_output() {
    let tool = TypedFunctionTool::json("lookup", "Lookup a value", |args: LookupArgs| async move {
        let value = if args.key == "answer" { 42 } else { 0 };
        Ok(LookupOutput { value })
    })
    .expect("typed tool");

    let definition = tool.definition();
    assert_eq!(definition.function.name, "lookup");
    assert!(definition.function.parameters.is_some());

    let output = tool
        .call(json!({"key": "answer"}))
        .await
        .expect("tool output");
    assert_eq!(output, "{\"value\":42}");

    let error = tool
        .call(json!({"key": 1}))
        .await
        .expect_err("invalid args");
    assert!(matches!(error, Error::InvalidToolArguments { name, .. } if name == "lookup"));
}

fn lookup_tool() -> FunctionTool<fn(Value) -> ToolFuture> {
    fn handler(_arguments: Value) -> ToolFuture {
        Box::pin(async { Ok("{\"value\":42}".to_owned()) })
    }

    FunctionTool::new(
        ToolDefinition::function(
            FunctionDefinition::new("lookup")
                .description("Lookup a value")
                .parameters(json!({
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": false
                })),
        ),
        handler,
    )
}

fn private_lookup_tool() -> FunctionTool<fn(Value) -> ToolFuture> {
    fn handler(_arguments: Value) -> ToolFuture {
        Box::pin(async { Ok("{\"private\":true}".to_owned()) })
    }

    FunctionTool::new(
        ToolDefinition::function(
            FunctionDefinition::new("private_lookup")
                .description("Lookup a private value")
                .parameters(json!({
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": false
                })),
        ),
        handler,
    )
}

fn test_client(base_url: String) -> DeepSeekClient {
    DeepSeekClient::new(
        DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_timeout(Duration::from_secs(5))
            .with_default_thinking(None)
            .with_default_reasoning_effort(None),
    )
    .expect("client")
}

fn tool_call_response(name: &str, arguments: &str) -> String {
    json!({
        "id": "chat-tool",
        "choices": [{
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [ToolCall::function("call_1", name, arguments)]
            },
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string()
}

fn usage() -> Value {
    json!({
        "completion_tokens": 1,
        "prompt_tokens": 1,
        "total_tokens": 2
    })
}

async fn spawn_http_sequence(responses: Vec<String>) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        let mut requests = Vec::new();

        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("accept connection");
            requests.push(read_http_request(&mut socket).await);
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        }

        requests
    });

    (format!("http://{address}"), handle)
}

async fn read_http_request(socket: &mut TcpStream) -> String {
    let mut data = Vec::new();
    let mut buffer = [0; 1024];

    loop {
        let bytes_read = socket.read(&mut buffer).await.expect("read request");
        if bytes_read == 0 {
            break;
        }

        data.extend_from_slice(&buffer[..bytes_read]);

        if let Some(header_end) = header_end(&data) {
            let content_length = content_length(&data[..header_end + 4]);
            if data.len() >= header_end + 4 + content_length {
                break;
            }
        }
    }

    String::from_utf8_lossy(&data).into_owned()
}

fn header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(headers: &[u8]) -> usize {
    let headers = String::from_utf8_lossy(headers);

    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("content-length"))
        })
        .unwrap_or(0)
}

fn json_response(body: &str) -> String {
    response("application/json", body)
}

fn stream_response(body: &str) -> String {
    response("text/event-stream", body)
}

fn response(content_type: &str, body: &str) -> String {
    status_response("200 OK", content_type, body)
}

fn status_response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn json_request_body(request: &str) -> Value {
    let (_, body) = request.split_once("\r\n\r\n").expect("request body");

    serde_json::from_str(body).expect("json request body")
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = env::var_os(key);
        // SAFETY: tests that mutate process env hold ENV_LOCK for the full guard lifetime.
        unsafe {
            env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: tests that mutate process env hold ENV_LOCK for the full guard lifetime.
        unsafe {
            if let Some(previous) = &self.previous {
                env::set_var(self.key, previous);
            } else {
                env::remove_var(self.key);
            }
        }
    }
}
