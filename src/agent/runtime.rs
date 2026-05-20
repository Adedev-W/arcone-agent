use std::{collections::BTreeMap, sync::Arc};

use serde::de::DeserializeOwned;

use super::{
    config::{AgentConfig, AgentId, AgentOptions, AgentProfile},
    response::AgentResponse,
    tool::{Tool, ToolRegistry},
};
use crate::{
    AnswerComposer, AnswerCompositionInput, ChatMessage, ChatRequest, ChatStream, DeepSeekClient,
    Error, FinishReason, GuardrailPipeline, GuardrailPipelineResult, GuardrailRequest,
    GuardrailStage, MemoryStore, ReasoningEffort, ResponseFormat, Result, ScoredChunk, SessionId,
    StreamEvent, ThinkingConfig, ToolCall, ToolChoice, ToolOutput, Usage,
};

pub struct Agent {
    client: DeepSeekClient,
    config: AgentConfig,
    history: Vec<ChatMessage>,
    tools: BTreeMap<String, Arc<dyn Tool>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    session_id: Option<SessionId>,
    store: Option<Arc<dyn MemoryStore>>,
    guardrails: Option<Arc<GuardrailPipeline>>,
    answer_composer: Option<Arc<dyn AnswerComposer>>,
}

impl Agent {
    pub fn new(client: DeepSeekClient) -> Self {
        Self::with_config(client, AgentConfig::default())
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(DeepSeekClient::from_env()?))
    }

    pub fn with_options(client: DeepSeekClient, options: AgentOptions) -> Self {
        Self::with_config(client, options.into())
    }

    pub fn with_config(client: DeepSeekClient, config: AgentConfig) -> Self {
        Self {
            client,
            config,
            history: Vec::new(),
            tools: BTreeMap::new(),
            tool_registry: None,
            session_id: None,
            store: None,
            guardrails: None,
            answer_composer: None,
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.config.profile.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn system(self, system_prompt: impl Into<String>) -> Self {
        self.with_system_prompt(system_prompt)
    }

    pub fn with_model(mut self, model: crate::DeepSeekModel) -> Self {
        self.config.model = Some(model);
        self
    }

    pub fn model(self, model: crate::DeepSeekModel) -> Self {
        self.with_model(model)
    }

    pub fn with_thinking(mut self, thinking: crate::ThinkingConfig) -> Self {
        self.config.thinking = Some(thinking);
        self
    }

    pub fn thinking_enabled(self) -> Self {
        self.with_thinking(ThinkingConfig::enabled())
    }

    pub fn thinking_disabled(self) -> Self {
        self.with_thinking(ThinkingConfig::disabled())
    }

    pub fn with_reasoning_effort(mut self, effort: crate::ReasoningEffort) -> Self {
        self.config.reasoning_effort = Some(effort);
        self
    }

    pub fn reasoning(self, effort: ReasoningEffort) -> Self {
        self.with_reasoning_effort(effort)
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.config.response_format = Some(format);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    pub fn max_tokens(self, max_tokens: u32) -> Self {
        self.with_max_tokens(max_tokens)
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.config.max_tool_rounds = max_tool_rounds;
        self
    }

    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.add_tool_unchecked(Arc::new(tool));
        self
    }

    pub fn tool<T>(self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.with_tool(tool)
    }

    pub fn add_tool<T>(&mut self, tool: T) -> &mut Self
    where
        T: Tool + 'static,
    {
        self.add_tool_unchecked(Arc::new(tool));
        self
    }

    pub fn try_with_tool<T>(mut self, tool: T) -> Result<Self>
    where
        T: Tool + 'static,
    {
        self.try_add_tool(tool)?;
        Ok(self)
    }

    pub fn try_add_tool<T>(&mut self, tool: T) -> Result<&mut Self>
    where
        T: Tool + 'static,
    {
        let tool = Arc::new(tool);
        let name = tool.definition().function.name;
        self.ensure_tool_name_available(&name)?;
        self.tools.insert(name, tool);
        Ok(self)
    }

    pub fn with_tool_registry(mut self, registry: impl Into<Arc<ToolRegistry>>) -> Result<Self> {
        self.set_tool_registry(registry)?;
        Ok(self)
    }

    pub fn tool_registry(self, registry: impl Into<Arc<ToolRegistry>>) -> Result<Self> {
        self.with_tool_registry(registry)
    }

    pub fn set_tool_registry(
        &mut self,
        registry: impl Into<Arc<ToolRegistry>>,
    ) -> Result<&mut Self> {
        let registry = registry.into();
        self.ensure_registry_compatible(&registry)?;
        self.tool_registry = Some(registry);
        Ok(self)
    }

    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn profile(&self) -> &AgentProfile {
        &self.config.profile
    }

    pub fn id(&self) -> &AgentId {
        &self.config.profile.id
    }

    /// Attach a session and memory store to this agent.
    ///
    /// When a session is set, `ask()` will load history from the store
    /// before building the request and save it back after each response.
    pub fn with_session(
        mut self,
        session_id: impl Into<SessionId>,
        store: Arc<dyn MemoryStore>,
    ) -> Self {
        self.session_id = Some(session_id.into());
        self.store = Some(store);
        self
    }

    pub fn session(self, session_id: impl Into<SessionId>, store: Arc<dyn MemoryStore>) -> Self {
        self.with_session(session_id, store)
    }

    /// Set session and store on an existing agent reference.
    pub fn set_session(
        &mut self,
        session_id: impl Into<SessionId>,
        store: Arc<dyn MemoryStore>,
    ) -> &mut Self {
        self.session_id = Some(session_id.into());
        self.store = Some(store);
        self
    }

    /// Returns the current session id, if any.
    pub fn session_id(&self) -> Option<&SessionId> {
        self.session_id.as_ref()
    }

    pub fn with_guardrails(mut self, guardrails: impl Into<Arc<GuardrailPipeline>>) -> Self {
        self.guardrails = Some(guardrails.into());
        self
    }

    pub fn set_guardrails(&mut self, guardrails: impl Into<Arc<GuardrailPipeline>>) -> &mut Self {
        self.guardrails = Some(guardrails.into());
        self
    }

    pub fn with_answer_composer<C>(mut self, composer: C) -> Self
    where
        C: AnswerComposer + 'static,
    {
        self.answer_composer = Some(Arc::new(composer));
        self
    }

    pub fn with_shared_answer_composer(mut self, composer: Arc<dyn AnswerComposer>) -> Self {
        self.answer_composer = Some(composer);
        self
    }

    pub fn set_answer_composer<C>(&mut self, composer: C) -> &mut Self
    where
        C: AnswerComposer + 'static,
    {
        self.answer_composer = Some(Arc::new(composer));
        self
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub async fn clear_session(&mut self) -> Result<()> {
        self.history.clear();
        if let (Some(session_id), Some(store)) = (&self.session_id, &self.store) {
            store.clear(session_id).await?;
        }
        Ok(())
    }

    pub async fn ask(&mut self, user_input: impl Into<String>) -> Result<AgentResponse> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let original_input = user_input.into();
        let input_guardrails = self
            .apply_guardrails(
                GuardrailStage::Input,
                original_input.clone(),
                Some(original_input.clone()),
                Vec::new(),
            )
            .await?;
        let mut guardrail_events = input_guardrails.events;
        if let Some(block) = input_guardrails.blocked {
            return Err(block.into_error());
        }
        let user_input = input_guardrails.text;

        // If a session store is attached, load persisted history first.
        if let (Some(session_id), Some(store)) = (&self.session_id, &self.store) {
            self.history = store.load_messages(session_id).await?;
        }

        self.history.push(ChatMessage::user(user_input.clone()));

        let mut tool_rounds = 0;
        loop {
            let request = self.build_request();
            let response = self.client.chat(request).await?;
            let choice = response
                .choices
                .into_iter()
                .next()
                .ok_or(Error::NoChoices)?;
            let finish_reason = choice.finish_reason;
            let mut assistant_message = choice.message;
            let tool_calls = assistant_message.tool_calls.clone().unwrap_or_default();

            if tool_calls.is_empty() {
                let draft_answer = assistant_message.content.clone().unwrap_or_default();
                let output_guardrails = self
                    .apply_guardrails(
                        GuardrailStage::Output,
                        draft_answer,
                        Some(user_input.clone()),
                        Vec::new(),
                    )
                    .await?;
                guardrail_events.extend(output_guardrails.events);
                if let Some(block) = output_guardrails.blocked {
                    return Err(block.into_error());
                }
                assistant_message.content = Some(output_guardrails.text);
                self.history.push(assistant_message.clone());

                // Persist history to store before returning.
                if let (Some(session_id), Some(store)) = (&self.session_id, &self.store) {
                    store.save_messages(session_id, &self.history).await?;
                }

                let mut agent_response = AgentResponse {
                    message: assistant_message,
                    finish_reason,
                    usage: response.usage,
                    history: self.history.clone(),
                    guardrail_events,
                    composed_answer: None,
                };
                self.compose_response(&user_input, &mut agent_response, Vec::new())
                    .await?;
                #[cfg(feature = "tracing")]
                tracing::info!(
                    target: "arcone_agent::agent",
                    operation = "ask",
                    agent_id = self.id().as_str(),
                    tool_rounds,
                    history_len = self.history.len(),
                    guardrail_event_count = agent_response.guardrail_events.len(),
                    elapsed_ms = crate::observability::elapsed_ms(started_at),
                    "agent request completed"
                );
                return Ok(agent_response);
            }

            self.history.push(assistant_message.clone());

            if tool_rounds >= self.config.max_tool_rounds {
                return Err(Error::ToolLoopExceeded {
                    max_rounds: self.config.max_tool_rounds,
                });
            }
            tool_rounds += 1;

            for tool_call in tool_calls {
                let tool_message = self.run_tool_call(tool_call).await?;
                self.history.push(tool_message);
            }
        }
    }

    pub async fn ask_json<T>(&mut self, user_input: impl Into<String>) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let previous_format = self.config.response_format.clone();
        self.config.response_format = Some(ResponseFormat::json_object());
        let response = self.ask(user_input).await;
        self.config.response_format = previous_format;

        let response = response?;
        let content = response.content().ok_or(Error::NoAssistantMessage)?;

        Ok(serde_json::from_str(content)?)
    }

    pub async fn ask_text(&mut self, user_input: impl Into<String>) -> Result<String> {
        let response = self.ask(user_input).await?;
        response
            .content()
            .map(str::to_owned)
            .ok_or(Error::NoAssistantMessage)
    }

    pub async fn stream(&mut self, user_input: impl Into<String>) -> Result<AgentStream<'_>> {
        if let (Some(session_id), Some(store)) = (&self.session_id, &self.store) {
            self.history = store.load_messages(session_id).await?;
        }

        let initial_history_len = self.history.len();
        self.history.push(ChatMessage::user(user_input));
        let request = self.build_request().streaming(true);
        let stream = match self.client.stream_chat(request).await {
            Ok(stream) => stream,
            Err(error) => {
                self.history.truncate(initial_history_len);
                return Err(error);
            }
        };

        Ok(AgentStream {
            agent: self,
            stream,
            initial_history_len,
            content: String::new(),
            reasoning_content: String::new(),
            finish_reason: None,
            usage: None,
            done: false,
            finished: false,
        })
    }

    fn build_request(&self) -> ChatRequest {
        let model = self
            .config
            .model
            .clone()
            .unwrap_or_else(|| self.client.config().model().clone());
        let mut messages = Vec::with_capacity(self.history.len() + 1);

        if let Some(system_prompt) = &self.config.profile.system_prompt {
            messages.push(ChatMessage::system(system_prompt.clone()));
        }
        messages.extend(self.history.iter().cloned());

        let mut request = ChatRequest::new(model, messages);

        if let Some(thinking) = self.config.thinking.clone() {
            request.thinking = Some(thinking);
        }

        if let Some(effort) = self.config.reasoning_effort.clone() {
            request.reasoning_effort = Some(effort);
        }

        if let Some(max_tokens) = self.config.max_tokens {
            request.max_tokens = Some(max_tokens);
        }

        if let Some(format) = self.config.response_format.clone() {
            request.response_format = Some(format);
        }

        let tool_definitions = self.tool_definitions();
        if !tool_definitions.is_empty() {
            request.tools = Some(tool_definitions);
            request.tool_choice = Some(ToolChoice::auto());
        }

        request
    }

    pub(crate) async fn compose_response(
        &self,
        original_question: &str,
        response: &mut AgentResponse,
        retrieved_chunks: Vec<ScoredChunk>,
    ) -> Result<()> {
        let Some(composer) = &self.answer_composer else {
            return Ok(());
        };

        let input = AnswerCompositionInput {
            original_question: original_question.to_owned(),
            selected_agent_id: Some(self.id().clone()),
            tool_outputs: ToolOutput::from_history(&response.history),
            retrieved_chunks,
            draft_answer: response.message.content.clone().unwrap_or_default(),
            guardrail_events: response.guardrail_events.clone(),
            usage: response.usage.clone(),
        };
        response.composed_answer = Some(composer.compose(input).await?);
        Ok(())
    }

    async fn apply_guardrails(
        &self,
        stage: GuardrailStage,
        text: String,
        original_question: Option<String>,
        retrieved_chunks: Vec<ScoredChunk>,
    ) -> Result<GuardrailPipelineResult> {
        let Some(guardrails) = &self.guardrails else {
            return Ok(GuardrailPipelineResult {
                text,
                events: Vec::new(),
                blocked: None,
            });
        };

        let mut request =
            GuardrailRequest::new(stage, text).with_retrieved_chunks(retrieved_chunks);
        if let Some(original_question) = original_question {
            request = request.with_original_question(original_question);
        }
        guardrails.check(request).await
    }

    async fn run_tool_call(&self, tool_call: ToolCall) -> Result<ChatMessage> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let tool_name = tool_call.function.name.clone();
        let tool = self
            .find_tool(&tool_name)
            .ok_or_else(|| Error::UnknownTool(tool_name.clone()))?;
        let arguments = serde_json::from_str(&tool_call.function.arguments).map_err(|source| {
            Error::InvalidToolArguments {
                name: tool_name.clone(),
                source,
            }
        })?;
        let output = tool.call(arguments).await?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::agent",
            operation = "tool_call",
            agent_id = self.id().as_str(),
            tool_name = tool_name.as_str(),
            output_len = output.len(),
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "tool call completed"
        );

        Ok(ChatMessage::tool(tool_call.id, output))
    }

    fn add_tool_unchecked(&mut self, tool: Arc<dyn Tool>) {
        let definition = tool.definition();
        self.tools.insert(definition.function.name, tool);
    }

    fn ensure_tool_name_available(&self, name: &str) -> Result<()> {
        if self.tools.contains_key(name)
            || self
                .tool_registry
                .as_ref()
                .is_some_and(|registry| registry.contains(name))
        {
            return Err(Error::DuplicateTool(name.to_owned()));
        }

        Ok(())
    }

    fn ensure_registry_compatible(&self, registry: &ToolRegistry) -> Result<()> {
        for name in self.tools.keys() {
            if registry.contains(name) {
                return Err(Error::DuplicateTool(name.clone()));
            }
        }

        Ok(())
    }

    fn tool_definitions(&self) -> Vec<crate::ToolDefinition> {
        let mut definitions = BTreeMap::new();

        if let Some(registry) = &self.tool_registry {
            for definition in registry.definitions() {
                definitions.insert(definition.function.name.clone(), definition);
            }
        }

        for tool in self.tools.values() {
            let definition = tool.definition();
            definitions.insert(definition.function.name.clone(), definition);
        }

        definitions.into_values().collect()
    }

    fn find_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned().or_else(|| {
            self.tool_registry
                .as_ref()
                .and_then(|registry| registry.get(name))
        })
    }
}

pub struct AgentStream<'a> {
    agent: &'a mut Agent,
    stream: ChatStream,
    initial_history_len: usize,
    content: String,
    reasoning_content: String,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    done: bool,
    finished: bool,
}

impl AgentStream<'_> {
    pub async fn next_text(&mut self) -> Result<Option<String>> {
        if self.done {
            return Ok(None);
        }

        loop {
            let Some(event) = self.stream.next_event().await? else {
                self.done = true;
                return Ok(None);
            };

            match event {
                StreamEvent::Done => {
                    self.done = true;
                    return Ok(None);
                }
                StreamEvent::Chunk(chunk) => {
                    if let Some(usage) = chunk.usage {
                        self.usage = Some(usage);
                    }

                    let mut text = String::new();
                    for choice in chunk.choices {
                        if let Some(tool_calls) = choice.delta.tool_calls
                            && !tool_calls.is_empty()
                        {
                            return Err(Error::StreamingToolCallsUnsupported);
                        }

                        if let Some(reasoning_content) = choice.delta.reasoning_content {
                            self.reasoning_content.push_str(&reasoning_content);
                        }

                        if let Some(content) = choice.delta.content {
                            self.content.push_str(&content);
                            text.push_str(&content);
                        }

                        if let Some(finish_reason) = choice.finish_reason {
                            self.finish_reason = Some(finish_reason);
                        }
                    }

                    if !text.is_empty() {
                        return Ok(Some(text));
                    }
                }
            }
        }
    }

    pub async fn finish(mut self) -> Result<AgentResponse> {
        while self.next_text().await?.is_some() {}

        let mut message = ChatMessage::assistant(self.content.clone());
        if !self.reasoning_content.is_empty() {
            message = message.with_reasoning_content(self.reasoning_content.clone());
        }

        self.agent.history.push(message.clone());

        if let (Some(session_id), Some(store)) = (&self.agent.session_id, &self.agent.store) {
            store.save_messages(session_id, &self.agent.history).await?;
        }

        let finish_reason = self.finish_reason.take().unwrap_or(FinishReason::Unknown);
        let usage = self.usage.take();
        self.finished = true;

        Ok(AgentResponse {
            message,
            finish_reason,
            usage,
            history: self.agent.history.clone(),
            guardrail_events: Vec::new(),
            composed_answer: None,
        })
    }
}

impl Drop for AgentStream<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.agent.history.truncate(self.initial_history_len);
        }
    }
}
