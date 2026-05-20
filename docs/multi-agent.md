# Multi-Agent Teams

`AgentTeam` coordinates multiple named agents behind a router. The router
chooses which agent receives a request, and the team returns a `TeamResponse`
with routing metadata and the selected agent response.

Related docs: [Agents](agents.md), [Tools](tools.md),
[Guardrails and composer](guardrails-and-composer.md), [API reference](api-reference.md).

## Core Types

- `AgentTeam`: registry and runtime for multiple agents.
- `TeamRouter`: trait for routing decisions.
- `StaticRouter`: always selects one agent.
- `LlmRouter`: asks DeepSeek to choose one candidate agent and return JSON.
- `RouteRequest`: user input, available agent profiles, and handoff history.
- `RouteDecision`: selected agent or handoff.
- `Handoff`: route transition from one agent to another.
- `TeamResponse`: selected agent ID, route reason, handoffs, and response.

## Static Routing

Use `StaticRouter` when the team should always start with the same role.

```rust
use arcone_agent::{
    Agent, AgentConfig, AgentTeam, DeepSeekClient, Result, StaticRouter, ThinkingConfig,
};

async fn run() -> Result<()> {
    let client = DeepSeekClient::from_env()?;

    let researcher = Agent::with_config(
        client.clone(),
        AgentConfig::new("researcher")
            .with_name("Researcher")
            .with_role_description("Finds facts and constraints")
            .with_system_prompt("Research carefully and answer with concise evidence.")
            .with_thinking(ThinkingConfig::enabled()),
    );

    let writer = Agent::with_config(
        client,
        AgentConfig::new("writer")
            .with_name("Writer")
            .with_role_description("Turns research into a short final answer")
            .with_system_prompt("Write plainly and keep the answer brief.")
            .with_thinking(ThinkingConfig::disabled()),
    );

    let mut team = AgentTeam::new()
        .with_agent(researcher)?
        .with_agent(writer)?
        .with_router(StaticRouter::new("researcher").with_reason("default research route"));

    let response = team.ask("Prepare a short implementation note.").await?;
    println!("{}: {}", response.agent_id, response.content().unwrap_or(""));

    Ok(())
}
```

## LLM Routing

Use `LlmRouter` when routing should depend on the request and role metadata.

```rust
use arcone_agent::{AgentTeam, DeepSeekClient, LlmRouter, Result};

async fn route_with_llm(mut team: AgentTeam) -> Result<AgentTeam> {
    let router = LlmRouter::new(DeepSeekClient::from_env()?)
        .with_max_tokens(256);

    team.set_router(router);
    Ok(team)
}
```

`LlmRouter` asks a temporary routing agent to return JSON. If it selects an
unknown agent or invalid JSON, routing fails with `Error::RoutingFailure`.

## Direct Agent Calls

Bypass the router when the application already knows which agent should answer.

```rust
let response = team
    .ask_with_agent("writer", "Rewrite this answer in a concise style.")
    .await?;
```

## Custom Router

Implement `TeamRouter` for deterministic business routing, classifiers, or
external routing services.

```rust
use arcone_agent::{AgentId, Result, RouteDecision, RouteFuture, RouteRequest, TeamRouter};

struct KeywordRouter;

impl TeamRouter for KeywordRouter {
    fn route(&self, request: RouteRequest) -> RouteFuture {
        Box::pin(async move {
            let selected = if request.input.to_lowercase().contains("risk") {
                AgentId::new("risk_reviewer")
            } else {
                AgentId::new("researcher")
            };

            Ok(RouteDecision::agent_with_reason(
                selected,
                "keyword routing",
            ))
        })
    }
}
```

## Handoffs

Routers can return `RouteDecision::handoff(from, to)` to record a handoff and
continue routing. `AgentTeam::with_max_handoff_rounds` bounds handoff loops.

```rust
let team = AgentTeam::new()
    .with_max_handoff_rounds(3)
    .with_router(my_router);
```

If the handoff limit is exceeded, the team returns
`Error::HandoffLoopExceeded`.

## Team Guardrails And Composer

Team-level guardrails run on team input and final output. Individual agents can
also have their own guardrails.

```rust
use arcone_agent::{
    AgentTeam, DefaultAnswerComposer, EmptyAnswerGuardrail, GuardrailPipeline,
};

let guardrails = GuardrailPipeline::new()
    .with_guardrail(EmptyAnswerGuardrail::new());

let mut team = AgentTeam::new()
    .with_guardrails(guardrails)
    .with_answer_composer(DefaultAnswerComposer::new());
```

## Multi-Agent Best Practices

- Assign every agent a stable `AgentId` and a useful `role_description`.
- Start with `StaticRouter` for predictable workflows and move to `LlmRouter`
  only when dynamic routing is needed.
- Keep router output bounded with `LlmRouter::with_max_tokens`.
- Use direct `ask_with_agent` for explicit application routing.
- Share common tools through `ToolRegistry`; keep role-specific tools private.
- Set `max_handoff_rounds` to prevent routing loops.
- Treat route reasons as operational metadata, not user-facing truth.
