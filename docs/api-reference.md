# API Reference

This reference groups the public types re-exported from `src/lib.rs`. It is a
human reference, not generated Rustdoc.

Related docs: [Getting started](getting-started.md), [Agents](agents.md),
[Tools](tools.md), [Knowledge and retrieval](knowledge-and-retrieval.md).

## Result And Errors

- `Result<T>`: alias for `std::result::Result<T, Error>`.
- `Error`: typed error enum for provider config, HTTP, JSON, API failures,
  streaming, tools, sessions, routing, knowledge, embeddings, retrieval,
  PostgreSQL, guardrails, and composer failures.

Common variants include `MissingApiKey`, `EmptyMessages`, `Api`,
`Timeout`, `InvalidToolArguments`, `ToolLoopExceeded`, `DuplicateAgent`,
`RoutingFailure`, `DuplicateDocument`, `EmbeddingFailure`, `RetrievalFailure`,
`MissingDatabaseUrl`, `DatabaseMigration`, and `GuardrailBlocked`.

## DeepSeek Client

### `DeepSeekConfig`

- `new(api_key) -> Self`
- `from_env() -> Result<Self>`
- `with_base_url(base_url) -> Self`
- `with_model(model: DeepSeekModel) -> Self`
- `with_timeout(timeout: Duration) -> Self`
- `with_default_thinking(thinking: Option<ThinkingConfig>) -> Self`
- `with_default_reasoning_effort(effort: Option<ReasoningEffort>) -> Self`
- `base_url() -> &str`
- `model() -> &DeepSeekModel`
- `timeout() -> Duration`

`from_env` reads `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL`, and optional
`DEEPSEEK_MODEL`.

### `DeepSeekClient`

- `new(config: DeepSeekConfig) -> Result<Self>`
- `from_env() -> Result<Self>`
- `config() -> &DeepSeekConfig`
- `chat(request: ChatRequest) -> Result<ChatResponse>`
- `stream_chat(request: ChatRequest) -> Result<ChatStream>`

## Agent Runtime

### `AgentId`

- `new(id) -> Self`
- `as_str() -> &str`
- `into_inner() -> String`
- Implements `Default`, `Display`, `From<&str>`, `From<String>`.

### `AgentProfile`

- `new(id) -> Self`
- `with_name(name) -> Self`
- `with_role_description(description) -> Self`
- `with_system_prompt(prompt) -> Self`

Fields: `id`, `name`, `role_description`, `system_prompt`.

### `AgentConfig`

- `new(id) -> Self`
- `with_profile(profile) -> Self`
- `with_name(name) -> Self`
- `with_role_description(description) -> Self`
- `with_system_prompt(prompt) -> Self`
- `with_model(model) -> Self`
- `with_thinking(thinking) -> Self`
- `with_reasoning_effort(effort) -> Self`
- `with_max_tokens(max_tokens) -> Self`
- `with_max_tool_rounds(max_tool_rounds) -> Self`
- `with_response_format(format) -> Self`

Fields include profile, optional model, thinking, reasoning effort, max tokens,
max tool rounds, and response format.

### `Agent`

Constructors and configuration:

- `new(client) -> Self`
- `from_env() -> Result<Self>`
- `with_options(client, options) -> Self`
- `with_config(client, config) -> Self`
- `with_system_prompt(prompt) -> Self`
- `system(prompt) -> Self`
- `with_model(model) -> Self`
- `model(model) -> Self`
- `with_thinking(thinking) -> Self`
- `thinking_enabled() -> Self`
- `thinking_disabled() -> Self`
- `with_reasoning_effort(effort) -> Self`
- `reasoning(effort) -> Self`
- `with_response_format(format) -> Self`
- `with_max_tokens(max_tokens) -> Self`
- `max_tokens(max_tokens) -> Self`
- `with_max_tool_rounds(max_tool_rounds) -> Self`

Tools, sessions, guardrails, and composer:

- `with_tool(tool) -> Self`
- `tool(tool) -> Self`
- `add_tool(tool) -> &mut Self`
- `try_with_tool(tool) -> Result<Self>`
- `try_add_tool(tool) -> Result<&mut Self>`
- `with_tool_registry(registry) -> Result<Self>`
- `tool_registry(registry) -> Result<Self>`
- `set_tool_registry(registry) -> Result<&mut Self>`
- `with_session(session_id, store) -> Self`
- `session(session_id, store) -> Self`
- `set_session(session_id, store) -> &mut Self`
- `session_id() -> Option<&SessionId>`
- `with_guardrails(guardrails) -> Self`
- `set_guardrails(guardrails) -> &mut Self`
- `with_answer_composer(composer) -> Self`
- `with_shared_answer_composer(composer) -> Self`
- `set_answer_composer(composer) -> &mut Self`

Runtime methods:

- `history() -> &[ChatMessage]`
- `config() -> &AgentConfig`
- `profile() -> &AgentProfile`
- `id() -> &AgentId`
- `clear_history()`
- `clear_session() -> Result<()>`
- `ask(input) -> Result<AgentResponse>`
- `ask_json<T>(input) -> Result<T>`
- `ask_text(input) -> Result<String>`
- `stream(input) -> Result<AgentStream<'_>>`

### `AgentResponse`

Fields: `message`, `finish_reason`, `usage`, `history`,
`guardrail_events`, `composed_answer`.

- `content() -> Option<&str>`
- `reasoning_content() -> Option<&str>`

### `AgentStream`

- `next_text() -> Result<Option<String>>`
- `finish() -> Result<AgentResponse>`

## Tools

### `Tool`

- `definition() -> ToolDefinition`
- `call(arguments: serde_json::Value) -> ToolFuture`

### `ToolRegistry`

- `new() -> Self`
- `add_tool(tool) -> Result<&mut Self>`
- `get(name) -> Option<Arc<dyn Tool>>`
- `definitions() -> Vec<ToolDefinition>`

### `FunctionTool`

- `new(definition, handler) -> Self`

### `TypedFunctionTool`

- `json(name, description, handler) -> Result<Self>`

Related protocol types:

- `FunctionCall`
- `ToolCall`
- `ToolDefinition`
- `FunctionDefinition`
- `ToolChoiceMode`
- `NamedToolChoice`
- `ToolChoice`

Protocol helpers:

- `ToolCall::function(id, name, arguments) -> Self`
- `ToolDefinition::function(function) -> Self`
- `FunctionDefinition::new(name) -> Self`
- `FunctionDefinition::description(description) -> Self`
- `FunctionDefinition::parameters(parameters) -> Self`
- `FunctionDefinition::strict(strict) -> Self`
- `NamedToolChoice::function(name) -> Self`
- `ToolChoice::none() -> Self`
- `ToolChoice::auto() -> Self`
- `ToolChoice::required() -> Self`
- `ToolChoice::function(name) -> Self`

## Chat Types

### `ChatMessage`

- `system(content) -> Self`
- `user(content) -> Self`
- `assistant(content) -> Self`
- `assistant_with_tool_calls(content, tool_calls) -> Self`
- `tool(tool_call_id, content) -> Self`
- `with_name(name) -> Self`
- `with_reasoning_content(reasoning_content) -> Self`

### `ChatRequest`

- `new(model, messages) -> Self`
- `with_thinking(thinking) -> Self`
- `with_reasoning_effort(effort) -> Self`
- `with_response_format(format) -> Self`
- `with_tools(tools) -> Self`
- `with_tool_choice(tool_choice) -> Self`
- `with_max_tokens(max_tokens) -> Self`
- `streaming(include_usage) -> Self`

### Response And Model Types

- `ChatResponse::first_choice() -> Option<&ChatChoice>`
- `ChatResponse::first_message() -> Option<&ChatMessage>`
- `DeepSeekModel::as_str() -> &str`
- `ThinkingConfig::enabled() -> Self`
- `ThinkingConfig::disabled() -> Self`
- `ResponseFormat::text() -> Self`
- `ResponseFormat::json_object() -> Self`
- `Role`: `System`, `User`, `Assistant`, `Tool`
- `ThinkingMode`: `Enabled`, `Disabled`
- `ReasoningEffort`: `High`, `Max`
- `ResponseFormatType`: `Text`, `JsonObject`
- `StopSequences`: one string or many strings.
- `StreamOptions`: streaming options, including usage inclusion.
- `FinishReason`: `Stop`, `Length`, `ContentFilter`, `ToolCalls`,
  `InsufficientSystemResource`, `Unknown`
- `Usage`: token usage metadata.

## Streaming Protocol

- `ChatStream::next_event() -> Result<Option<StreamEvent>>`
- `StreamEvent::Chunk(ChatStreamChunk)`
- `StreamEvent::Done`
- `ChatStreamChunk`, `StreamChoice`, `ChatDelta`: streaming response payloads.

Most application code should use `Agent::stream` and `AgentStream::next_text`
instead of reading `ChatStream` directly.

## Sessions

### `SessionId`

- `new(id) -> Self`
- `random() -> Self`
- `as_str() -> &str`
- `into_inner() -> String`
- Implements `Display`, `From<&str>`, `From<String>`.

### `SessionMetadata`

- `new() -> Self`
- `touch()`
- `with_agent_id(agent_id) -> Self`
- `with_extra(extra) -> Self`

### `MemoryStore`

- `save_messages(id, messages) -> MemoryFuture<()>`
- `load_messages(id) -> MemoryFuture<Vec<ChatMessage>>`
- `save_metadata(id, metadata) -> MemoryFuture<()>`
- `load_metadata(id) -> MemoryFuture<Option<SessionMetadata>>`
- `clear(id) -> MemoryFuture<()>`
- `exists(id) -> MemoryFuture<bool>`

### Stores

- `InMemorySessionStore::new() -> Self`
- `PostgresSessionConfig::new(database_url) -> Self`
- `PostgresSessionConfig::from_env() -> Result<Self>`
- `PostgresSessionConfig::with_max_pool_size(size) -> Self`
- `PostgresSessionConfig::with_connect_timeout(timeout) -> Self`
- `PostgresSessionStore::new(config) -> Result<Self>`
- `PostgresSessionStore::connect(config) -> Result<Self>`
- `PostgresSessionStore::from_pool(pool) -> Self`
- `PostgresSessionStore::pool() -> &Pool`
- `PostgresSessionStore::migrate() -> Result<()>`

## Knowledge

### IDs And Data

- `DocumentId::new(id) -> Self`
- `DocumentId::as_str() -> &str`
- `DocumentId::into_inner() -> String`
- `ChunkId::new(id) -> Self`
- `ChunkId::as_str() -> &str`
- `ChunkId::into_inner() -> String`
- `Document::new(id, content) -> Self`
- `Document::text(id, content) -> Self`
- `Document::with_title(title) -> Self`
- `Document::with_source(source) -> Self`
- `Document::with_path(path) -> Self`
- `Document::with_metadata(metadata) -> Self`
- `ChunkMetadata::new() -> Self`
- `ChunkMetadata::from_document(document) -> Self`
- `ChunkMetadata::with_title(title) -> Self`
- `ChunkMetadata::with_source(source) -> Self`
- `ChunkMetadata::with_path(path) -> Self`
- `ChunkMetadata::with_extra(extra) -> Self`
- `KnowledgeChunk::new(id, document_id, chunk_index, content, metadata) -> Self`
- `ScoredChunk::new(chunk, score) -> Self`
- `ChunkOptions::new(max_chars, overlap_chars) -> Self`
- `ChunkOptions::with_max_chars(max_chars) -> Self`
- `ChunkOptions::with_overlap_chars(overlap_chars) -> Self`

### `KnowledgeBase`

- `add_document(document) -> KnowledgeFuture<Vec<KnowledgeChunk>>`
- `list_documents() -> KnowledgeFuture<Vec<Document>>`
- `remove_document(id) -> KnowledgeFuture<bool>`
- `chunk_document(document) -> KnowledgeFuture<Vec<KnowledgeChunk>>`
- `chunks_for_document(id) -> KnowledgeFuture<Vec<KnowledgeChunk>>`
- `chunks_for_source(source) -> KnowledgeFuture<Vec<KnowledgeChunk>>`

### Stores

- `InMemoryKnowledgeBase::new() -> Self`
- `InMemoryKnowledgeBase::with_chunk_options(options) -> Self`
- `InMemoryKnowledgeBase::chunk_options() -> &ChunkOptions`
- `PostgresKnowledgeBase::new(pool) -> Self`
- `PostgresKnowledgeBase::connect(config) -> Result<Self>`
- `PostgresKnowledgeBase::with_chunk_options(options) -> Self`
- `PostgresKnowledgeBase::chunk_options() -> &ChunkOptions`
- `PostgresKnowledgeBase::pool() -> &PostgresPool`
- `PostgresKnowledgeBase::migrate() -> Result<()>`

## Retrieval

### Traits And Helpers

- `Embedder::embed(texts) -> EmbedFuture<Vec<Embedding>>`
- `Retriever::retrieve(query, top_k) -> RetrieveFuture<Vec<ScoredChunk>>`
- `Embedding::new(text_index, vector) -> Self`
- `cosine_similarity(left, right) -> Result<f32>`

### OpenAI Embeddings

- `OpenAiConfig::new(api_key) -> Self`
- `OpenAiConfig::from_env() -> Result<Self>`
- `OpenAiConfig::with_base_url(base_url) -> Self`
- `OpenAiConfig::with_model(model) -> Self`
- `OpenAiConfig::with_timeout(timeout) -> Self`
- `OpenAiConfig::base_url() -> &str`
- `OpenAiConfig::model() -> &OpenAiEmbeddingModel`
- `OpenAiConfig::timeout() -> Duration`
- `OpenAiEmbedder::new(config) -> Result<Self>`
- `OpenAiEmbedder::from_env() -> Result<Self>`
- `OpenAiEmbedder::config() -> &OpenAiConfig`
- `OpenAiEmbedder::embed_texts(texts) -> Result<Vec<Embedding>>`
- `OpenAiEmbeddingModel::as_str() -> &str`
- `OpenAiEmbeddingModel`: `TextEmbedding3Small`, `TextEmbedding3Large`,
  `TextEmbeddingAda002`, or `Other(String)`

### Vector Retrievers

- `InMemoryVectorRetriever::new(embedder) -> Self`
- `InMemoryVectorRetriever::from_embedder(embedder) -> Self`
- `InMemoryVectorRetriever::index(chunks) -> Result<()>`
- `InMemoryVectorRetriever::len() -> Result<usize>`
- `InMemoryVectorRetriever::is_empty() -> Result<bool>`
- `PgVectorRetrieverOptions::new(embedding_dimension) -> Self`
- `PgVectorRetrieverOptions::with_embedding_dimension(dimension) -> Self`
- `PgVectorRetrieverOptions::with_metric(metric) -> Self`
- `PgVectorRetrieverOptions::with_index_mode(mode) -> Self`
- `PgVectorMetric`: `Cosine`, `L2`
- `PgVectorIndexMode`: `Auto`, `Hnsw`, `None`
- `PgVectorRetriever::new(pool, embedder, options) -> Self`
- `PgVectorRetriever::connect(config, embedder, options) -> Result<Self>`
- `PgVectorRetriever::from_embedder(pool, embedder, options) -> Self`
- `PgVectorRetriever::pool() -> &PostgresPool`
- `PgVectorRetriever::options() -> &PgVectorRetrieverOptions`
- `PgVectorRetriever::migrate() -> Result<()>`
- `PgVectorRetriever::index(chunks) -> Result<()>`

## KnowledgeAgent

- `KnowledgeAgent::new(agent, retriever) -> Self`
- `KnowledgeAgent::from_retriever(agent, retriever) -> Self`
- `KnowledgeAgent::with_options(options) -> Self`
- `KnowledgeAgent::agent() -> &Agent`
- `KnowledgeAgent::agent_mut() -> &mut Agent`
- `KnowledgeAgent::options() -> &KnowledgeAgentOptions`
- `KnowledgeAgent::with_guardrails(guardrails) -> Self`
- `KnowledgeAgent::set_guardrails(guardrails) -> &mut Self`
- `KnowledgeAgent::with_answer_composer(composer) -> Self`
- `KnowledgeAgent::with_shared_answer_composer(composer) -> Self`
- `KnowledgeAgent::ask(question) -> Result<KnowledgeAgentResponse>`
- `KnowledgeAgentOptions::new() -> Self`
- `KnowledgeAgentOptions::with_top_k(top_k) -> Self`
- `KnowledgeAgentOptions::with_max_context_chars(chars) -> Self`
- `KnowledgeAgentOptions::with_fallback_message(message) -> Self`
- `KnowledgeAgentResponse::content() -> &str`

## Multi-Agent Teams

- `RouteRequest::new(input, agents, handoffs) -> Self`
- `Handoff::new(from, to) -> Self`
- `Handoff::with_reason(reason) -> Self`
- `RouteDecision::agent(agent_id) -> Self`
- `RouteDecision::agent_with_reason(agent_id, reason) -> Self`
- `RouteDecision::handoff(from, to) -> Self`
- `TeamRouter::route(request) -> RouteFuture`
- `TeamResponse::content() -> Option<&str>`
- `TeamResponse::reasoning_content() -> Option<&str>`
- `AgentTeam::new() -> Self`
- `AgentTeam::add_agent(agent) -> Result<&mut Self>`
- `AgentTeam::with_agent(agent) -> Result<Self>`
- `AgentTeam::set_router(router) -> &mut Self`
- `AgentTeam::with_router(router) -> Self`
- `AgentTeam::with_max_handoff_rounds(rounds) -> Self`
- `AgentTeam::with_guardrails(guardrails) -> Self`
- `AgentTeam::set_guardrails(guardrails) -> &mut Self`
- `AgentTeam::with_answer_composer(composer) -> Self`
- `AgentTeam::with_shared_answer_composer(composer) -> Self`
- `AgentTeam::ask(input) -> Result<TeamResponse>`
- `AgentTeam::ask_with_agent(agent_id, input) -> Result<TeamResponse>`
- `StaticRouter::new(agent_id) -> Self`
- `StaticRouter::with_reason(reason) -> Self`
- `LlmRouter::new(client) -> Self`
- `LlmRouter::with_max_tokens(max_tokens) -> Self`

## Guardrails

- `Guardrail::name() -> &str`
- `Guardrail::check(request) -> GuardrailFuture`
- `GuardrailStage::as_str() -> &'static str`
- `GuardrailRequest::new(stage, text) -> Self`
- `GuardrailRequest::with_original_question(question) -> Self`
- `GuardrailRequest::with_retrieved_chunks(chunks) -> Self`
- `GuardrailDecision::allow() -> Self`
- `GuardrailDecision::modify(text) -> Self`
- `GuardrailDecision::block(reason) -> Self`
- `GuardrailDecision::block_with_fallback(reason, fallback) -> Self`
- `GuardrailBlock::into_error() -> Error`
- `GuardrailPipeline::new() -> Self`
- `GuardrailPipeline::add_guardrail(guardrail) -> &mut Self`
- `GuardrailPipeline::with_guardrail(guardrail) -> Self`
- `GuardrailPipeline::from_guardrails(guardrails) -> Self`
- `GuardrailPipeline::len() -> usize`
- `GuardrailPipeline::is_empty() -> bool`
- `GuardrailPipeline::check(request) -> Result<GuardrailPipelineResult>`
- `PrivateInfoRedactionGuardrail::new() -> Self`
- `PrivateInfoRedactionGuardrail::with_email_replacement(value) -> Self`
- `PrivateInfoRedactionGuardrail::with_phone_replacement(value) -> Self`
- `EmptyAnswerGuardrail::new() -> Self`
- `NoHallucinationFallbackGuardrail::new(fallback_message) -> Self`

## Answer Composition

- `AnswerComposer::compose(input) -> AnswerComposerFuture`
- `ToolOutput::from_history(history) -> Vec<ToolOutput>`
- `ComposedAnswer::content() -> &str`
- `DefaultAnswerComposer::new() -> Self`
- `DefaultAnswerComposer::with_debug_metadata(include) -> Self`

Public data types:

- `AnswerCompositionInput`
- `ToolOutput`
- `ComposedAnswer`
- `AnswerSource`

## PostgreSQL Utilities

- `PostgresStoreConfig::new(database_url) -> Self`
- `PostgresStoreConfig::from_env() -> Result<Self>`
- `PostgresStoreConfig::with_max_pool_size(size) -> Self`
- `PostgresStoreConfig::with_connect_timeout(timeout) -> Self`
- `PostgresStoreConfig::with_statement_timeout(timeout) -> Self`
- `PostgresPool::connect(config) -> Result<Self>`
- `PostgresPool::new(config) -> Result<Self>`
- `PostgresPool::from_pool(pool) -> Self`
- `PostgresPool::pool() -> &Pool`
- `PostgresPool::client() -> Result<deadpool_postgres::Client>`

## Observability

- `redact_secret(value) -> &'static str`

When the `tracing` feature is enabled, the crate emits structured events for
model calls, streaming, tool calls, retrieval, guardrails, composition, teams,
sessions, PostgreSQL, and embeddings.
