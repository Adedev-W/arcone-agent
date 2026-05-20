use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};

use serde::Deserialize;

use crate::{
    Agent, AgentId, AgentProfile, AgentResponse, AnswerComposer, AnswerCompositionInput,
    DeepSeekClient, Error, GuardrailPipeline, GuardrailPipelineResult, GuardrailRequest,
    GuardrailStage, ResponseFormat, Result, ToolOutput,
};

pub type RouteFuture = Pin<Box<dyn Future<Output = Result<RouteDecision>> + Send>>;

pub trait TeamRouter: Send + Sync {
    fn route(&self, request: RouteRequest) -> RouteFuture;
}

#[derive(Clone, Debug)]
pub struct RouteRequest {
    pub input: String,
    pub agents: Vec<AgentProfile>,
    pub handoffs: Vec<Handoff>,
}

impl RouteRequest {
    pub fn new(
        input: impl Into<String>,
        agents: Vec<AgentProfile>,
        handoffs: Vec<Handoff>,
    ) -> Self {
        Self {
            input: input.into(),
            agents,
            handoffs,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Handoff {
    pub from: AgentId,
    pub to: AgentId,
    pub reason: Option<String>,
}

impl Handoff {
    pub fn new(from: impl Into<AgentId>, to: impl Into<AgentId>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteDecision {
    Agent {
        agent_id: AgentId,
        reason: Option<String>,
    },
    Handoff(Handoff),
}

impl RouteDecision {
    pub fn agent(agent_id: impl Into<AgentId>) -> Self {
        Self::Agent {
            agent_id: agent_id.into(),
            reason: None,
        }
    }

    pub fn agent_with_reason(agent_id: impl Into<AgentId>, reason: impl Into<String>) -> Self {
        Self::Agent {
            agent_id: agent_id.into(),
            reason: Some(reason.into()),
        }
    }

    pub fn handoff(from: impl Into<AgentId>, to: impl Into<AgentId>) -> Self {
        Self::Handoff(Handoff::new(from, to))
    }
}

#[derive(Debug)]
pub struct TeamResponse {
    pub agent_id: AgentId,
    pub route_reason: Option<String>,
    pub handoffs: Vec<Handoff>,
    pub response: AgentResponse,
}

impl TeamResponse {
    pub fn content(&self) -> Option<&str> {
        self.response.content()
    }

    pub fn reasoning_content(&self) -> Option<&str> {
        self.response.reasoning_content()
    }
}

pub struct AgentTeam {
    agents: BTreeMap<AgentId, Agent>,
    router: Option<Arc<dyn TeamRouter>>,
    max_handoff_rounds: usize,
    guardrails: Option<Arc<GuardrailPipeline>>,
    answer_composer: Option<Arc<dyn AnswerComposer>>,
}

impl Default for AgentTeam {
    fn default() -> Self {
        Self {
            agents: BTreeMap::new(),
            router: None,
            max_handoff_rounds: 4,
            guardrails: None,
            answer_composer: None,
        }
    }
}

impl AgentTeam {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_agent(&mut self, agent: Agent) -> Result<&mut Self> {
        let id = agent.id().clone();
        if self.agents.contains_key(&id) {
            return Err(Error::DuplicateAgent(id.into_inner()));
        }

        self.agents.insert(id, agent);
        Ok(self)
    }

    pub fn with_agent(mut self, agent: Agent) -> Result<Self> {
        self.add_agent(agent)?;
        Ok(self)
    }

    pub fn set_router<R>(&mut self, router: R) -> &mut Self
    where
        R: TeamRouter + 'static,
    {
        self.router = Some(Arc::new(router));
        self
    }

    pub fn with_router<R>(mut self, router: R) -> Self
    where
        R: TeamRouter + 'static,
    {
        self.set_router(router);
        self
    }

    pub fn with_max_handoff_rounds(mut self, max_handoff_rounds: usize) -> Self {
        self.max_handoff_rounds = max_handoff_rounds;
        self
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

    pub async fn ask(&mut self, input: impl Into<String>) -> Result<TeamResponse> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let original_input = input.into();
        let input_guardrails = self
            .apply_guardrails(
                GuardrailStage::Input,
                original_input.clone(),
                Some(original_input.clone()),
            )
            .await?;
        let guardrail_events = input_guardrails.events;
        if let Some(block) = input_guardrails.blocked {
            return Err(block.into_error());
        }
        let input = input_guardrails.text;

        if self.agents.is_empty() {
            return Err(Error::RoutingFailure(
                "agent team has no registered agents".to_owned(),
            ));
        }

        let router = self
            .router
            .clone()
            .ok_or_else(|| Error::RoutingFailure("agent team has no router".to_owned()))?;
        let mut handoffs = Vec::new();

        loop {
            let request = RouteRequest::new(input.clone(), self.agent_profiles(), handoffs.clone());
            match router.route(request).await? {
                RouteDecision::Agent { agent_id, reason } => {
                    let response = self
                        .ask_with_agent_internal(
                            agent_id,
                            input,
                            reason,
                            handoffs,
                            guardrail_events,
                        )
                        .await?;
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "arcone_agent::team",
                        operation = "ask",
                        selected_agent_id = response.agent_id.as_str(),
                        handoff_count = response.handoffs.len(),
                        guardrail_event_count = response.response.guardrail_events.len(),
                        elapsed_ms = crate::observability::elapsed_ms(started_at),
                        "team request completed"
                    );
                    return Ok(response);
                }
                RouteDecision::Handoff(handoff) => {
                    if handoffs.len() >= self.max_handoff_rounds {
                        return Err(Error::HandoffLoopExceeded {
                            max_rounds: self.max_handoff_rounds,
                        });
                    }

                    self.ensure_agent_exists(&handoff.from)?;
                    self.ensure_agent_exists(&handoff.to)?;
                    handoffs.push(handoff);
                }
            }
        }
    }

    pub async fn ask_with_agent(
        &mut self,
        agent_id: impl Into<AgentId>,
        input: impl Into<String>,
    ) -> Result<TeamResponse> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let agent_id = agent_id.into();
        let original_input = input.into();
        let input_guardrails = self
            .apply_guardrails(
                GuardrailStage::Input,
                original_input.clone(),
                Some(original_input),
            )
            .await?;
        let guardrail_events = input_guardrails.events;
        if let Some(block) = input_guardrails.blocked {
            return Err(block.into_error());
        }

        let response = self
            .ask_with_agent_internal(
                agent_id,
                input_guardrails.text,
                None,
                Vec::new(),
                guardrail_events,
            )
            .await?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::team",
            operation = "ask_with_agent",
            selected_agent_id = response.agent_id.as_str(),
            handoff_count = response.handoffs.len(),
            guardrail_event_count = response.response.guardrail_events.len(),
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "team direct agent request completed"
        );
        Ok(response)
    }

    fn agent_profiles(&self) -> Vec<AgentProfile> {
        self.agents
            .values()
            .map(|agent| agent.profile().clone())
            .collect()
    }

    fn ensure_agent_exists(&self, agent_id: &AgentId) -> Result<()> {
        if !self.agents.contains_key(agent_id) {
            return Err(Error::UnknownAgent(agent_id.to_string()));
        }

        Ok(())
    }

    async fn ask_with_agent_internal(
        &mut self,
        agent_id: AgentId,
        input: String,
        route_reason: Option<String>,
        handoffs: Vec<Handoff>,
        inherited_guardrail_events: Vec<crate::GuardrailEvent>,
    ) -> Result<TeamResponse> {
        let agent = self
            .agents
            .get_mut(&agent_id)
            .ok_or_else(|| Error::UnknownAgent(agent_id.to_string()))?;
        let mut response = agent.ask(input.clone()).await?;
        let mut guardrail_events = inherited_guardrail_events;
        guardrail_events.extend(response.guardrail_events.clone());

        let draft_answer = response.content().unwrap_or_default().to_owned();
        let output_guardrails = self
            .apply_guardrails(GuardrailStage::Output, draft_answer, Some(input.clone()))
            .await?;
        guardrail_events.extend(output_guardrails.events);
        if let Some(block) = output_guardrails.blocked {
            return Err(block.into_error());
        }

        if let Some(composed) = response.composed_answer.as_mut() {
            composed.text = output_guardrails.text.clone();
        } else {
            response.set_message_content(output_guardrails.text);
        }
        response.guardrail_events = guardrail_events;
        self.compose_response(&input, &agent_id, &mut response)
            .await?;

        Ok(TeamResponse {
            agent_id,
            route_reason,
            handoffs,
            response,
        })
    }

    async fn apply_guardrails(
        &self,
        stage: GuardrailStage,
        text: String,
        original_question: Option<String>,
    ) -> Result<GuardrailPipelineResult> {
        let Some(guardrails) = &self.guardrails else {
            return Ok(GuardrailPipelineResult {
                text,
                events: Vec::new(),
                blocked: None,
            });
        };

        let mut request = GuardrailRequest::new(stage, text);
        if let Some(original_question) = original_question {
            request = request.with_original_question(original_question);
        }
        guardrails.check(request).await
    }

    async fn compose_response(
        &self,
        original_question: &str,
        agent_id: &AgentId,
        response: &mut AgentResponse,
    ) -> Result<()> {
        let Some(composer) = &self.answer_composer else {
            return Ok(());
        };

        let input = AnswerCompositionInput {
            original_question: original_question.to_owned(),
            selected_agent_id: Some(agent_id.clone()),
            tool_outputs: ToolOutput::from_history(&response.history),
            retrieved_chunks: Vec::new(),
            draft_answer: response.content().unwrap_or_default().to_owned(),
            guardrail_events: response.guardrail_events.clone(),
            usage: response.usage.clone(),
        };
        response.composed_answer = Some(composer.compose(input).await?);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StaticRouter {
    agent_id: AgentId,
    reason: Option<String>,
}

impl StaticRouter {
    pub fn new(agent_id: impl Into<AgentId>) -> Self {
        Self {
            agent_id: agent_id.into(),
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

impl TeamRouter for StaticRouter {
    fn route(&self, _request: RouteRequest) -> RouteFuture {
        let agent_id = self.agent_id.clone();
        let reason = self.reason.clone();

        Box::pin(async move { Ok(RouteDecision::Agent { agent_id, reason }) })
    }
}

#[derive(Clone, Debug)]
pub struct LlmRouter {
    client: DeepSeekClient,
    max_tokens: u32,
}

impl LlmRouter {
    pub fn new(client: DeepSeekClient) -> Self {
        Self {
            client,
            max_tokens: 256,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

impl TeamRouter for LlmRouter {
    fn route(&self, request: RouteRequest) -> RouteFuture {
        let client = self.client.clone();
        let max_tokens = self.max_tokens;

        Box::pin(async move {
            if request.agents.is_empty() {
                return Err(Error::RoutingFailure(
                    "LLM router received no candidate agents".to_owned(),
                ));
            }

            let prompt = llm_router_prompt(&request);
            let mut router = Agent::new(client)
                .system(
                    "You route a user request to exactly one available agent. Return JSON only.",
                )
                .thinking_disabled()
                .with_response_format(ResponseFormat::json_object())
                .max_tokens(max_tokens);
            let text = router.ask_text(prompt).await?;
            let output: LlmRouteOutput = serde_json::from_str(&text).map_err(|source| {
                Error::RoutingFailure(format!("LLM router returned invalid JSON: {source}"))
            })?;

            if !request
                .agents
                .iter()
                .any(|profile| profile.id == output.agent_id)
            {
                return Err(Error::RoutingFailure(format!(
                    "LLM router selected unknown agent `{}`",
                    output.agent_id
                )));
            }

            Ok(RouteDecision::Agent {
                agent_id: output.agent_id,
                reason: output.reason,
            })
        })
    }
}

#[derive(Debug, Deserialize)]
struct LlmRouteOutput {
    agent_id: AgentId,
    #[serde(default)]
    reason: Option<String>,
}

fn llm_router_prompt(request: &RouteRequest) -> String {
    let mut prompt = String::from("User input:\n");
    prompt.push_str(&request.input);
    prompt.push_str("\n\nAvailable agents:\n");

    for profile in &request.agents {
        prompt.push_str("- id: ");
        prompt.push_str(profile.id.as_str());
        prompt.push_str("\n  name: ");
        prompt.push_str(&profile.name);

        if let Some(description) = &profile.role_description {
            prompt.push_str("\n  role: ");
            prompt.push_str(description);
        }

        prompt.push('\n');
    }

    if !request.handoffs.is_empty() {
        prompt.push_str("\nPrevious handoffs:\n");
        for handoff in &request.handoffs {
            prompt.push_str("- from: ");
            prompt.push_str(handoff.from.as_str());
            prompt.push_str(" to: ");
            prompt.push_str(handoff.to.as_str());
            if let Some(reason) = &handoff.reason {
                prompt.push_str(" reason: ");
                prompt.push_str(reason);
            }
            prompt.push('\n');
        }
    }

    prompt
        .push_str("\nReturn exactly one JSON object with fields: agent_id string, reason string.");
    prompt
}
