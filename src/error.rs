use std::time::Duration;

use hyper::StatusCode;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("DEEPSEEK_API_KEY is not set")]
    MissingApiKey,

    #[error("DeepSeek API key cannot be empty")]
    EmptyApiKey,

    #[error("OPENAI_API_KEY is not set")]
    MissingOpenAiApiKey,

    #[error("OpenAI API key cannot be empty")]
    EmptyOpenAiApiKey,

    #[error("chat request must contain at least one message")]
    EmptyMessages,

    #[error("strict tool calling requires a beta base URL such as https://api.deepseek.com/beta")]
    StrictToolsRequireBetaBaseUrl,

    #[error("invalid URI")]
    InvalidUri(#[from] hyper::http::uri::InvalidUri),

    #[error("invalid HTTP header value")]
    InvalidHeaderValue(#[from] hyper::http::header::InvalidHeaderValue),

    #[error("failed to build HTTP request")]
    HttpBuild(#[from] hyper::http::Error),

    #[error("HTTP client error")]
    Client(#[from] hyper_util::client::legacy::Error),

    #[error("HTTP body error")]
    Body(#[from] hyper::Error),

    #[error("JSON error")]
    Json(#[from] serde_json::Error),

    #[error("DeepSeek API returned {status}: {body}")]
    Api { status: StatusCode, body: String },

    #[error("OpenAI API returned {status}: {body}")]
    OpenAiApi { status: StatusCode, body: String },

    #[error("DeepSeek request timed out after {timeout:?}")]
    Timeout { timeout: Duration },

    #[error("OpenAI request timed out after {timeout:?}")]
    OpenAiTimeout { timeout: Duration },

    #[error("SSE stream ended with a partial event")]
    PartialSseEvent,

    #[error("invalid SSE event: {0}")]
    InvalidSse(String),

    #[error("DeepSeek response did not contain any choices")]
    NoChoices,

    #[error("DeepSeek response did not contain an assistant message")]
    NoAssistantMessage,

    #[error("streaming agent API does not support tool calls yet")]
    StreamingToolCallsUnsupported,

    #[error("tool `{0}` is not registered")]
    UnknownTool(String),

    #[error("tool `{0}` is already registered")]
    DuplicateTool(String),

    #[error("agent `{0}` is already registered")]
    DuplicateAgent(String),

    #[error("document `{0}` is already registered")]
    DuplicateDocument(String),

    #[error("agent `{0}` is not registered")]
    UnknownAgent(String),

    #[error("routing failure: {0}")]
    RoutingFailure(String),

    #[error("handoff loop exceeded max_rounds={max_rounds}")]
    HandoffLoopExceeded { max_rounds: usize },

    #[error("tool `{name}` returned invalid arguments: {source}")]
    InvalidToolArguments {
        name: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("tool loop exceeded max_tool_rounds={max_rounds}")]
    ToolLoopExceeded { max_rounds: usize },

    #[error("tool `{name}` failed: {message}")]
    ToolExecution { name: String, message: String },

    #[error("session `{0}` not found")]
    SessionNotFound(String),

    #[error("memory store error: {0}")]
    MemoryStore(String),

    #[error("knowledge store error: {0}")]
    KnowledgeStore(String),

    #[error("knowledge indexing error: {0}")]
    KnowledgeIndexing(String),

    #[error("embedding input must contain at least one text")]
    EmptyEmbeddingInput,

    #[error("embedding failure: {0}")]
    EmbeddingFailure(String),

    #[error("retrieval failure: {0}")]
    RetrievalFailure(String),

    #[error("DATABASE_URL is not set")]
    MissingDatabaseUrl,

    #[error("invalid database URL: {0}")]
    InvalidDatabaseUrl(String),

    #[error("database pool error: {0}")]
    DatabasePool(String),

    #[error("database connection error: {0}")]
    DatabaseConnection(String),

    #[error("database query error")]
    DatabaseQuery(#[from] tokio_postgres::Error),

    #[error("database migration error: {0}")]
    DatabaseMigration(String),

    #[error("guardrail blocked {stage}: {reason}")]
    GuardrailBlocked { stage: String, reason: String },

    #[error("composer failure: {0}")]
    ComposerFailure(String),
}
