use std::{
    collections::HashMap,
    env,
    ffi::OsString,
    sync::{Arc, Mutex},
    time::Duration,
};

use arcone_agent::{
    ChunkMetadata, DocumentId, EmbedFuture, Embedder, Embedding, Error, InMemoryVectorRetriever,
    KnowledgeChunk, OpenAiConfig, OpenAiEmbedder, OpenAiEmbeddingModel, Retriever,
    cosine_similarity,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn cosine_similarity_handles_common_cases() {
    assert_close(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap(), 1.0);
    assert_close(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0);
    assert_close(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]).unwrap(), 0.0);

    let error = cosine_similarity(&[1.0], &[1.0, 0.0]).expect_err("dimension mismatch");

    assert!(matches!(error, Error::RetrievalFailure(message) if message.contains("dimension")));
}

#[tokio::test]
async fn in_memory_vector_retriever_returns_top_k_by_cosine_score() {
    let embedder = StaticEmbedder::new([
        ("query", vec![1.0, 0.0]),
        ("alpha", vec![1.0, 0.0]),
        ("beta", vec![0.0, 1.0]),
        ("gamma", vec![0.5, 0.5]),
    ]);
    let retriever = InMemoryVectorRetriever::new(embedder);
    retriever
        .index(vec![chunk(0, "alpha"), chunk(1, "beta"), chunk(2, "gamma")])
        .await
        .expect("index chunks");

    let results = retriever.retrieve("query", 2).await.expect("retrieve");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].chunk.content, "alpha");
    assert_close(results[0].score, 1.0);
    assert_eq!(results[1].chunk.content, "gamma");
    assert!(results[1].score > 0.7);
}

#[tokio::test]
async fn in_memory_vector_retriever_handles_empty_index_and_zero_top_k() {
    let embedder = StaticEmbedder::new([("query", vec![1.0, 0.0])]);
    let retriever = InMemoryVectorRetriever::new(embedder);

    assert!(retriever.retrieve("query", 3).await.unwrap().is_empty());

    retriever
        .index(vec![chunk(0, "query")])
        .await
        .expect("index one chunk");

    assert!(retriever.retrieve("query", 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_vector_retriever_propagates_embedder_errors() {
    let retriever = InMemoryVectorRetriever::new(StaticEmbedder::failing("provider down"));

    let error = retriever
        .index(vec![chunk(0, "alpha")])
        .await
        .expect_err("embedder failure");

    assert!(matches!(error, Error::EmbeddingFailure(message) if message == "provider down"));
}

#[tokio::test]
async fn in_memory_vector_retriever_rejects_dimension_mismatch() {
    let embedder = StaticEmbedder::new([("query", vec![1.0, 0.0]), ("alpha", vec![1.0])]);
    let retriever = InMemoryVectorRetriever::new(embedder);
    retriever
        .index(vec![chunk(0, "alpha")])
        .await
        .expect("index chunk");

    let error = retriever
        .retrieve("query", 1)
        .await
        .expect_err("dimension mismatch");

    assert!(matches!(error, Error::RetrievalFailure(message) if message.contains("dimension")));
}

#[tokio::test]
async fn openai_embedder_posts_request_and_parses_embeddings() {
    let body = json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 4, "total_tokens": 4}
    })
    .to_string();
    let (base_url, server) = spawn_http_sequence(vec![json_response(200, &body)]).await;
    let embedder = OpenAiEmbedder::new(
        OpenAiConfig::new("test-openai-key")
            .with_base_url(base_url)
            .with_timeout(Duration::from_secs(5)),
    )
    .expect("openai embedder");

    let embeddings = embedder
        .embed_texts(vec!["first".to_owned(), "second".to_owned()])
        .await
        .expect("embeddings");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0], Embedding::new(0, vec![1.0, 0.0]));
    assert_eq!(embeddings[1], Embedding::new(1, vec![0.0, 1.0]));

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let lower = request.to_ascii_lowercase();
    assert!(lower.contains("post /v1/embeddings http/1.1"));
    assert!(lower.contains("authorization: bearer test-openai-key"));

    let body = json_request_body(request);
    assert_eq!(body["model"], "text-embedding-3-small");
    assert_eq!(body["input"], json!(["first", "second"]));
}

#[tokio::test]
async fn openai_embedder_maps_api_failure_to_typed_error() {
    let (base_url, server) =
        spawn_http_sequence(vec![json_response(500, "{\"error\":\"temporary\"}")]).await;
    let embedder = OpenAiEmbedder::new(
        OpenAiConfig::new("test-openai-key")
            .with_base_url(base_url)
            .with_timeout(Duration::from_secs(5)),
    )
    .expect("openai embedder");

    let error = embedder
        .embed_texts(vec!["hello".to_owned()])
        .await
        .expect_err("api error");

    assert!(
        matches!(error, Error::OpenAiApi { status, body } if status.as_u16() == 500 && body.contains("temporary"))
    );
    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn openai_embedder_rejects_empty_input_before_http_call() {
    let embedder = OpenAiEmbedder::new(
        OpenAiConfig::new("test-openai-key")
            .with_base_url("http://127.0.0.1:9")
            .with_timeout(Duration::from_millis(10)),
    )
    .expect("openai embedder");

    let error = embedder
        .embed_texts(Vec::new())
        .await
        .expect_err("empty input");

    assert!(matches!(error, Error::EmptyEmbeddingInput));
}

#[test]
fn openai_config_from_env_reads_optional_values_and_redacts_debug() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let _api_key = EnvVarGuard::set("OPENAI_API_KEY", "env-openai-key");
    let _base_url = EnvVarGuard::set("OPENAI_BASE_URL", "http://127.0.0.1:9");
    let _model = EnvVarGuard::set("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large");

    let config = OpenAiConfig::from_env().expect("config from env");
    let debug = format!("{config:?}");

    assert_eq!(config.base_url(), "http://127.0.0.1:9");
    assert_eq!(config.model(), &OpenAiEmbeddingModel::TextEmbedding3Large);
    assert!(debug.contains("<redacted>"));
    assert!(!debug.contains("env-openai-key"));
}

#[derive(Clone)]
struct StaticEmbedder {
    vectors: Arc<HashMap<String, Vec<f32>>>,
    failure: Option<String>,
}

impl StaticEmbedder {
    fn new<const N: usize>(vectors: [(&str, Vec<f32>); N]) -> Self {
        Self {
            vectors: Arc::new(
                vectors
                    .into_iter()
                    .map(|(text, vector)| (text.to_owned(), vector))
                    .collect(),
            ),
            failure: None,
        }
    }

    fn failing(message: impl Into<String>) -> Self {
        Self {
            vectors: Arc::new(HashMap::new()),
            failure: Some(message.into()),
        }
    }
}

impl Embedder for StaticEmbedder {
    fn embed(&self, texts: Vec<String>) -> EmbedFuture<Vec<Embedding>> {
        let vectors = Arc::clone(&self.vectors);
        let failure = self.failure.clone();

        Box::pin(async move {
            if let Some(message) = failure {
                return Err(Error::EmbeddingFailure(message));
            }

            texts
                .into_iter()
                .enumerate()
                .map(|(index, text)| {
                    let vector = vectors.get(&text).cloned().ok_or_else(|| {
                        Error::EmbeddingFailure(format!("missing vector for `{text}`"))
                    })?;
                    Ok(Embedding::new(index, vector))
                })
                .collect()
        })
    }
}

fn chunk(index: usize, content: &str) -> KnowledgeChunk {
    KnowledgeChunk::new(
        format!("chunk-{index}"),
        DocumentId::new("doc"),
        index,
        content,
        ChunkMetadata::new(),
    )
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 0.0001,
        "expected {actual} to be close to {expected}"
    );
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

fn json_response(status: u16, body: &str) -> String {
    let reason = match status {
        200 => "OK",
        500 => "Internal Server Error",
        _ => "Status",
    };
    format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
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
