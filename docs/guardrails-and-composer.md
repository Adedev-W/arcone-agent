# Guardrails And Answer Composition

Guardrails inspect or transform text at input, retrieved context, and output
stages. Answer composers normalize final responses and can attach source,
usage, and debug metadata.

Related docs: [Agents](agents.md), [Knowledge and retrieval](knowledge-and-retrieval.md),
[Multi-agent teams](multi-agent.md), [API reference](api-reference.md).

## Guardrail Stages

`GuardrailStage` has three values:

- `Input`: user input before an agent or team request.
- `RetrievedContext`: context returned by a retriever before a knowledge answer.
- `Output`: final draft answer before it is returned.

A guardrail returns one `GuardrailDecision`:

- `Allow`: keep text unchanged.
- `Modify(String)`: replace the text and continue the pipeline.
- `Block { reason, fallback_message }`: stop the pipeline. Knowledge agents can
  use the fallback message when one is provided.

## Built-In Guardrails

- `PrivateInfoRedactionGuardrail`: redacts email addresses and phone-like
  values.
- `EmptyAnswerGuardrail`: blocks empty output.
- `NoHallucinationFallbackGuardrail`: blocks empty retrieved context with a
  fallback message.

```rust
use arcone_agent::{
    EmptyAnswerGuardrail, GuardrailPipeline, NoHallucinationFallbackGuardrail,
    PrivateInfoRedactionGuardrail,
};

let guardrails = GuardrailPipeline::new()
    .with_guardrail(PrivateInfoRedactionGuardrail::new())
    .with_guardrail(EmptyAnswerGuardrail::new())
    .with_guardrail(NoHallucinationFallbackGuardrail::default());
```

Attach the pipeline to an agent, team, or knowledge agent:

```rust
let mut agent = agent.with_guardrails(guardrails);
```

Use `Arc<GuardrailPipeline>` when several agents or teams share the same
pipeline.

## Custom Guardrail

```rust
use arcone_agent::{
    Guardrail, GuardrailDecision, GuardrailFuture, GuardrailRequest, GuardrailStage,
};

struct BlockSecretWord;

impl Guardrail for BlockSecretWord {
    fn name(&self) -> &str {
        "block_secret_word"
    }

    fn check(&self, request: GuardrailRequest) -> GuardrailFuture {
        Box::pin(async move {
            if request.stage == GuardrailStage::Input
                && request.text.to_lowercase().contains("secret")
            {
                Ok(GuardrailDecision::block("input contained a blocked word"))
            } else {
                Ok(GuardrailDecision::allow())
            }
        })
    }
}
```

Best practice: return specific reasons for blocked requests. They become part
of `GuardrailEvent` and `Error::GuardrailBlocked`.

## Guardrail Events

Every pipeline check records `GuardrailEvent` values:

- `guardrail_name`
- `stage`
- `action`
- optional `reason`

Events are exposed on `AgentResponse`, `KnowledgeAgentResponse`, and the inner
agent response of `TeamResponse`.

## Default Answer Composer

`DefaultAnswerComposer` trims the final text, attaches source metadata from
retrieved chunks, carries usage metadata, and can optionally include debug
metadata.

```rust
use arcone_agent::{Agent, DefaultAnswerComposer};

let mut agent = Agent::from_env()?
    .with_answer_composer(DefaultAnswerComposer::new());

let response = agent.ask("Summarize the result.").await?;
println!("{}", response.content().unwrap_or(""));
```

Enable debug metadata when developing or testing:

```rust
use arcone_agent::DefaultAnswerComposer;

let composer = DefaultAnswerComposer::new()
    .with_debug_metadata(true);
```

Debug metadata can include the original question, selected agent ID, tool
outputs, and guardrail events. Do not expose debug metadata directly to end
users when tool outputs or prompts may contain sensitive data.

## Custom AnswerComposer

Implement `AnswerComposer` when you need a custom final response shape,
post-processing, citation formatting, or additional validation.

```rust
use arcone_agent::{
    AnswerComposer, AnswerComposerFuture, AnswerCompositionInput, ComposedAnswer,
};

struct PrefixComposer;

impl AnswerComposer for PrefixComposer {
    fn compose(&self, input: AnswerCompositionInput) -> AnswerComposerFuture {
        Box::pin(async move {
            Ok(ComposedAnswer {
                text: format!("Final: {}", input.draft_answer.trim()),
                sources: Vec::new(),
                usage: input.usage,
                debug_metadata: None,
            })
        })
    }
}
```

## Composition Best Practices

- Use guardrails for safety and filtering; use composers for final response
  normalization and metadata.
- Put redaction guardrails before guardrails that inspect exact text.
- Use `NoHallucinationFallbackGuardrail` with `KnowledgeAgent` when empty
  context should return a controlled fallback.
- Keep debug metadata disabled in production user responses unless it is
  explicitly sanitized.
- Use `response.content()` instead of reading `response.message.content`
  directly, because `content()` respects composed answers.
