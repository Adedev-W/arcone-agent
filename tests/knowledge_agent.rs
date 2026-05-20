use std::{sync::Arc, time::Duration};

use arcone_agent::{
    Agent, ChunkMetadata, DeepSeekClient, DeepSeekConfig, DocumentId, KnowledgeAgent,
    KnowledgeAgentOptions, KnowledgeChunk, RetrieveFuture, Retriever, ScoredChunk,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

#[tokio::test]
async fn knowledge_agent_injects_context_and_returns_sources() {
    let body = json!({
        "id": "knowledge-chat",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "Arcone supports orchestration [1]."},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let chunk = KnowledgeChunk::new(
        "chunk-1",
        DocumentId::new("doc-1"),
        0,
        "Arcone supports multi-agent orchestration.",
        ChunkMetadata::new()
            .with_title("Arcone Manual")
            .with_source("manual")
            .with_path("docs/arcone.md"),
    );
    let retriever = StaticRetriever::new(vec![ScoredChunk::new(chunk, 0.91)]);
    let agent = Agent::new(test_client(base_url));
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever)
        .with_options(KnowledgeAgentOptions::new().with_top_k(1));

    let response = knowledge_agent
        .ask("What does Arcone support?")
        .await
        .expect("knowledge response");

    assert_eq!(response.content(), "Arcone supports orchestration [1].");
    assert!(!response.used_fallback);
    assert!(response.agent_response.is_some());
    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].source.as_deref(), Some("manual"));
    assert_eq!(response.sources[0].path.as_deref(), Some("docs/arcone.md"));

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
    let request = json_request_body(&requests[0]);
    let prompt = request["messages"][0]["content"]
        .as_str()
        .expect("prompt content");

    assert!(prompt.contains("Use only the knowledge context"));
    assert!(prompt.contains("[1] title=Arcone Manual source=manual path=docs/arcone.md"));
    assert!(prompt.contains("Arcone supports multi-agent orchestration."));
    assert!(prompt.contains("What does Arcone support?"));
}

#[tokio::test]
async fn knowledge_agent_returns_fallback_without_model_call_when_no_context() {
    let retriever = StaticRetriever::new(Vec::new());
    let agent = Agent::new(test_client("http://127.0.0.1:9".to_owned()));
    let mut knowledge_agent = KnowledgeAgent::new(agent, retriever)
        .with_options(KnowledgeAgentOptions::new().with_fallback_message("No matching knowledge."));

    let response = knowledge_agent
        .ask("Unknown question")
        .await
        .expect("fallback response");

    assert_eq!(response.content(), "No matching knowledge.");
    assert!(response.used_fallback);
    assert!(response.agent_response.is_none());
    assert!(response.sources.is_empty());
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
