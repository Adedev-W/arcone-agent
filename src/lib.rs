mod agent;
mod client;
mod composer;
mod error;
mod guardrail;
mod knowledge;
mod knowledge_agent;
mod observability;
mod postgres;
mod retrieval;
mod session;
mod stream;
mod team;
mod types;

pub use agent::{
    Agent, AgentConfig, AgentId, AgentOptions, AgentProfile, AgentResponse, AgentStream,
    FunctionTool, Tool, ToolFuture, ToolRegistry, TypedFunctionTool,
};
pub use client::{DeepSeekClient, DeepSeekConfig};
pub use composer::{
    AnswerComposer, AnswerComposerFuture, AnswerCompositionInput, AnswerSource, ComposedAnswer,
    DefaultAnswerComposer, ToolOutput,
};
pub use error::{Error, Result};
pub use guardrail::{
    EmptyAnswerGuardrail, Guardrail, GuardrailAction, GuardrailBlock, GuardrailDecision,
    GuardrailEvent, GuardrailFuture, GuardrailPipeline, GuardrailPipelineResult, GuardrailRequest,
    GuardrailStage, NoHallucinationFallbackGuardrail, PrivateInfoRedactionGuardrail,
};
pub use knowledge::{
    ChunkId, ChunkMetadata, ChunkOptions, Document, DocumentId, InMemoryKnowledgeBase,
    KnowledgeBase, KnowledgeChunk, KnowledgeFuture, PostgresKnowledgeBase, ScoredChunk,
};
pub use knowledge_agent::{
    KnowledgeAgent, KnowledgeAgentOptions, KnowledgeAgentResponse, KnowledgeSource,
};
pub use observability::redact_secret;
pub use postgres::{PostgresPool, PostgresStoreConfig};
pub use retrieval::{
    EmbedFuture, Embedder, Embedding, InMemoryVectorRetriever, OpenAiConfig, OpenAiEmbedder,
    OpenAiEmbeddingModel, PgVectorIndexMode, PgVectorMetric, PgVectorRetriever,
    PgVectorRetrieverOptions, RetrieveFuture, Retriever, cosine_similarity,
};
pub use session::{
    InMemorySessionStore, MemoryFuture, MemoryStore, PostgresSessionConfig, PostgresSessionStore,
    SessionId, SessionMetadata,
};
pub use stream::{ChatStream, StreamEvent};
pub use team::{
    AgentTeam, Handoff, LlmRouter, RouteDecision, RouteFuture, RouteRequest, StaticRouter,
    TeamResponse, TeamRouter,
};
pub use types::{
    ChatDelta, ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, DeepSeekModel,
    FinishReason, FunctionCall, FunctionDefinition, NamedToolChoice, ReasoningEffort,
    ResponseFormat, ResponseFormatType, Role, StopSequences, StreamChoice, StreamOptions,
    ThinkingConfig, ThinkingMode, ToolCall, ToolChoice, ToolChoiceMode, ToolDefinition, Usage,
};
