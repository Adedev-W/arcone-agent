use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use arcone_agent::{
    Agent, AgentId, AgentTeam, ChunkMetadata, DeepSeekClient, DeepSeekConfig,
    DefaultAnswerComposer, DocumentId, EmptyAnswerGuardrail, Error, FunctionDefinition,
    FunctionTool, GuardrailAction, GuardrailPipeline, KnowledgeAgent, KnowledgeChunk,
    NoHallucinationFallbackGuardrail, PrivateInfoRedactionGuardrail, RetrieveFuture, Retriever,
    RouteDecision, RouteFuture, RouteRequest, ScoredChunk, StaticRouter, TeamRouter, ToolCall,
    ToolDefinition, ToolFuture,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

#[tokio::test]
async fn agent_input_guardrail_redacts_private_info_before_model_request() {
    let body = chat_response("redacted ok");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let guardrails = GuardrailPipeline::new().with_guardrail(PrivateInfoRedactionGuardrail::new());
    let mut agent = Agent::new(test_client(base_url)).with_guardrails(guardrails);

    let response = agent
        .ask("Email alice@example.com or call +1 555 123 4567")
        .await
        .expect("agent response");

    assert_eq!(response.content(), Some("redacted ok"));
    assert!(
        response
            .guardrail_events
            .iter()
            .any(|event| event.action == GuardrailAction::Modify)
    );

    let requests = server.await.expect("server task");
    let request = json_request_body(&requests[0]);
    let content = request["messages"][0]["content"]
        .as_str()
        .expect("user content");
    assert!(content.contains("[REDACTED_EMAIL]"));
    assert!(content.contains("[REDACTED_PHONE]"));
    assert!(!content.contains("alice@example.com"));
}

#[tokio::test]
async fn agent_output_guardrail_blocks_empty_answer() {
    let body = chat_response("   ");
    let (base_url, _server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let guardrails = GuardrailPipeline::new().with_guardrail(EmptyAnswerGuardrail::new());
    let mut agent = Agent::new(test_client(base_url)).with_guardrails(guardrails);

    let error = agent
        .ask("Return nothing")
        .await
        .expect_err("empty output should be blocked");

    assert!(
        matches!(error, Error::GuardrailBlocked { stage, reason } if stage == "output" && reason.contains("empty"))
    );
}

#[tokio::test]
async fn knowledge_context_guardrail_returns_fallback_without_model_call() {
    let guardrails =
        GuardrailPipeline::new().with_guardrail(NoHallucinationFallbackGuardrail::new("No data."));
    let agent = Agent::new(test_client("http://127.0.0.1:9".to_owned()));
    let retriever = StaticRetriever::new(Vec::new());
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever).with_guardrails(guardrails);

    let response = knowledge_agent
        .ask("What is missing?")
        .await
        .expect("fallback response");

    assert_eq!(response.content(), "No data.");
    assert!(response.used_fallback);
    assert!(response.agent_response.is_none());
    assert!(
        response
            .guardrail_events
            .iter()
            .any(|event| event.action == GuardrailAction::Block)
    );
}

#[tokio::test]
async fn team_input_guardrail_runs_before_router_and_agent_call() {
    let body = chat_response("team ok");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let seen_input = Arc::new(Mutex::new(None));
    let router = CaptureRouter::new("worker", seen_input.clone());
    let guardrails = GuardrailPipeline::new().with_guardrail(PrivateInfoRedactionGuardrail::new());
    let agent = Agent::with_config(
        test_client(base_url),
        arcone_agent::AgentConfig::new("worker"),
    );
    let mut team = AgentTeam::new()
        .with_agent(agent)
        .expect("add agent")
        .with_router(router)
        .with_guardrails(guardrails);

    let response = team
        .ask("Route email bob@example.com")
        .await
        .expect("team response");

    assert_eq!(response.content(), Some("team ok"));
    let routed_input = seen_input
        .lock()
        .expect("seen input lock")
        .clone()
        .expect("router saw input");
    assert!(routed_input.contains("[REDACTED_EMAIL]"));
    assert!(!routed_input.contains("bob@example.com"));

    let requests = server.await.expect("server task");
    let request = json_request_body(&requests[0]);
    let agent_content = request["messages"][0]["content"]
        .as_str()
        .expect("agent content");
    assert!(agent_content.contains("[REDACTED_EMAIL]"));
    assert!(!agent_content.contains("bob@example.com"));
}

#[tokio::test]
async fn default_composer_normalizes_answer_and_exposes_debug_tool_outputs() {
    let tool_response = tool_call_response("lookup", "{\"key\":\"answer\"}");
    let final_response = chat_response("  final answer  ");
    let (base_url, _server) = spawn_http_sequence(vec![
        json_response(&tool_response),
        json_response(&final_response),
    ])
    .await;
    let mut agent = Agent::new(test_client(base_url))
        .with_tool(lookup_tool())
        .with_answer_composer(DefaultAnswerComposer::new().with_debug_metadata(true));

    let response = agent.ask("Use the tool").await.expect("agent response");

    assert_eq!(response.content(), Some("final answer"));
    let composed = response.composed_answer.expect("composed answer");
    assert_eq!(composed.text, "final answer");
    let debug = composed.debug_metadata.expect("debug metadata");
    assert_eq!(debug["tool_outputs"][0]["content"], "{\"value\":42}");
}

#[tokio::test]
async fn knowledge_answer_composer_includes_source_metadata() {
    let body = chat_response("  arcone answer [1]  ");
    let (base_url, _server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let chunk = KnowledgeChunk::new(
        "chunk-1",
        DocumentId::new("doc-1"),
        0,
        "Arcone has knowledge agents.",
        ChunkMetadata::new()
            .with_title("Manual")
            .with_source("manual")
            .with_path("manual.md"),
    );
    let retriever = StaticRetriever::new(vec![ScoredChunk::new(chunk, 0.9)]);
    let agent = Agent::new(test_client(base_url));
    let mut knowledge_agent =
        KnowledgeAgent::new(agent, retriever).with_answer_composer(DefaultAnswerComposer::new());

    let response = knowledge_agent
        .ask("What has knowledge agents?")
        .await
        .expect("knowledge response");

    assert_eq!(response.content(), "arcone answer [1]");
    let composed = response.composed_answer.expect("composed answer");
    assert_eq!(composed.sources.len(), 1);
    assert_eq!(composed.sources[0].source.as_deref(), Some("manual"));
    assert!(composed.debug_metadata.is_none());
}

#[tokio::test]
async fn team_composer_does_not_include_debug_metadata_by_default() {
    let body = chat_response("  team final  ");
    let (base_url, _server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let agent = Agent::with_config(
        test_client(base_url),
        arcone_agent::AgentConfig::new("composer-worker"),
    );
    let mut team = AgentTeam::new()
        .with_agent(agent)
        .expect("add agent")
        .with_router(StaticRouter::new("composer-worker"))
        .with_answer_composer(DefaultAnswerComposer::new());

    let response = team.ask("Compose").await.expect("team response");

    assert_eq!(response.content(), Some("team final"));
    let composed = response.response.composed_answer.expect("composed answer");
    assert!(composed.debug_metadata.is_none());
}

#[derive(Clone)]
struct StaticRetriever {
    results: Arc<Vec<ScoredChunk>>,
}

impl StaticRetriever {
    fn new(results: Vec<ScoredChunk>) -> Self {
        Self {
            results: Arc::new(results),
        }
    }
}

impl Retriever for StaticRetriever {
    fn retrieve(&self, _query: &str, top_k: usize) -> RetrieveFuture<Vec<ScoredChunk>> {
        let results = Arc::clone(&self.results);

        Box::pin(async move {
            let mut results = (*results).clone();
            results.truncate(top_k);
            Ok(results)
        })
    }
}

#[derive(Clone)]
struct CaptureRouter {
    agent_id: AgentId,
    seen_input: Arc<Mutex<Option<String>>>,
}

impl CaptureRouter {
    fn new(agent_id: impl Into<AgentId>, seen_input: Arc<Mutex<Option<String>>>) -> Self {
        Self {
            agent_id: agent_id.into(),
            seen_input,
        }
    }
}

impl TeamRouter for CaptureRouter {
    fn route(&self, request: RouteRequest) -> RouteFuture {
        let seen_input = Arc::clone(&self.seen_input);
        let agent_id = self.agent_id.clone();

        Box::pin(async move {
            *seen_input.lock().expect("seen input lock") = Some(request.input);
            Ok(RouteDecision::agent(agent_id))
        })
    }
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

fn chat_response(content: &str) -> String {
    json!({
        "id": "chat",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string()
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
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn json_request_body(request: &str) -> Value {
    let (_, body) = request.split_once("\r\n\r\n").expect("request body");

    serde_json::from_str(body).expect("json request body")
}
