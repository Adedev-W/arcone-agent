use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    pub text_index: usize,
    pub vector: Vec<f32>,
}

impl Embedding {
    pub fn new(text_index: usize, vector: Vec<f32>) -> Self {
        Self { text_index, vector }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiEmbeddingModel {
    #[serde(rename = "text-embedding-3-small")]
    #[default]
    TextEmbedding3Small,
    #[serde(rename = "text-embedding-3-large")]
    TextEmbedding3Large,
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
    #[serde(untagged)]
    Other(String),
}

impl OpenAiEmbeddingModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
            Self::Other(model) => model,
        }
    }
}

impl fmt::Display for OpenAiEmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OpenAiEmbeddingModel {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "text-embedding-3-small" => Self::TextEmbedding3Small,
            "text-embedding-3-large" => Self::TextEmbedding3Large,
            "text-embedding-ada-002" => Self::TextEmbeddingAda002,
            other => Self::Other(other.to_owned()),
        })
    }
}

impl From<&str> for OpenAiEmbeddingModel {
    fn from(value: &str) -> Self {
        value.parse().expect("infallible model parse")
    }
}

impl From<String> for OpenAiEmbeddingModel {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}
