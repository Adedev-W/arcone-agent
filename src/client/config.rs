use std::{fmt, time::Duration};

use crate::{ChatRequest, DeepSeekModel, Error, ReasoningEffort, Result, ThinkingConfig};

#[derive(Clone)]
pub struct DeepSeekConfig {
    pub(super) api_key: String,
    pub(super) base_url: String,
    model: DeepSeekModel,
    pub(super) timeout: Duration,
    default_thinking: Option<ThinkingConfig>,
    default_reasoning_effort: Option<ReasoningEffort>,
}

impl DeepSeekConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.deepseek.com".to_owned(),
            model: DeepSeekModel::default(),
            timeout: Duration::from_secs(600),
            default_thinking: Some(ThinkingConfig::enabled()),
            default_reasoning_effort: Some(ReasoningEffort::High),
        }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| Error::MissingApiKey)?;
        let mut config = Self::new(api_key);

        if let Ok(base_url) = std::env::var("DEEPSEEK_BASE_URL") {
            let base_url = base_url.trim();
            if !base_url.is_empty() {
                config = config.with_base_url(base_url.to_owned());
            }
        }

        if let Ok(model) = std::env::var("DEEPSEEK_MODEL") {
            let model = model.trim();
            if !model.is_empty() {
                let model = match model.parse() {
                    Ok(model) => model,
                    Err(error) => match error {},
                };
                config = config.with_model(model);
            }
        }

        Ok(config)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: DeepSeekModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_default_thinking(mut self, thinking: Option<ThinkingConfig>) -> Self {
        self.default_thinking = thinking;
        self
    }

    pub fn with_default_reasoning_effort(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.default_reasoning_effort = effort;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &DeepSeekModel {
        &self.model
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub(crate) fn apply_defaults(&self, request: &mut ChatRequest) {
        if request.thinking.is_none() {
            request.thinking = self.default_thinking.clone();
        }

        if request.reasoning_effort.is_none() {
            request.reasoning_effort = self.default_reasoning_effort.clone();
        }
    }
}

impl fmt::Debug for DeepSeekConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepSeekConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("default_thinking", &self.default_thinking)
            .field("default_reasoning_effort", &self.default_reasoning_effort)
            .finish()
    }
}
