use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{DeepSeekModel, ReasoningEffort, ResponseFormat, ThinkingConfig};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new("agent")
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for AgentId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for AgentId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentProfile {
    pub id: AgentId,
    pub name: String,
    pub role_description: Option<String>,
    pub system_prompt: Option<String>,
}

impl AgentProfile {
    pub fn new(id: impl Into<AgentId>) -> Self {
        let id = id.into();
        let name = id.as_str().to_owned();

        Self {
            id,
            name,
            role_description: None,
            system_prompt: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_role_description(mut self, role_description: impl Into<String>) -> Self {
        self.role_description = Some(role_description.into());
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }
}

impl Default for AgentProfile {
    fn default() -> Self {
        Self::new(AgentId::default())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentConfig {
    pub profile: AgentProfile,
    pub model: Option<DeepSeekModel>,
    pub thinking: Option<ThinkingConfig>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub max_tokens: Option<u32>,
    pub max_tool_rounds: usize,
    pub response_format: Option<ResponseFormat>,
}

impl AgentConfig {
    pub fn new(id: impl Into<AgentId>) -> Self {
        Self {
            profile: AgentProfile::new(id),
            ..Self::default()
        }
    }

    pub fn with_profile(mut self, profile: AgentProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.profile.name = name.into();
        self
    }

    pub fn with_role_description(mut self, role_description: impl Into<String>) -> Self {
        self.profile.role_description = Some(role_description.into());
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.profile.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn with_model(mut self, model: DeepSeekModel) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = max_tool_rounds;
        self
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            profile: AgentProfile::default(),
            model: None,
            thinking: None,
            reasoning_effort: None,
            max_tokens: None,
            max_tool_rounds: 8,
            response_format: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AgentOptions {
    pub model: Option<DeepSeekModel>,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub response_format: Option<ResponseFormat>,
    pub max_tool_rounds: usize,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            model: None,
            system_prompt: None,
            thinking: None,
            reasoning_effort: None,
            response_format: None,
            max_tool_rounds: 8,
        }
    }
}

impl From<AgentOptions> for AgentConfig {
    fn from(options: AgentOptions) -> Self {
        Self {
            profile: AgentProfile {
                system_prompt: options.system_prompt,
                ..AgentProfile::default()
            },
            model: options.model,
            thinking: options.thinking,
            reasoning_effort: options.reasoning_effort,
            max_tokens: None,
            max_tool_rounds: options.max_tool_rounds,
            response_format: options.response_format,
        }
    }
}
