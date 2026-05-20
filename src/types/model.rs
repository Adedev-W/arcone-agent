use std::{convert::Infallible, fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub enum DeepSeekModel {
    #[default]
    V4Flash,
    V4Pro,
    Custom(String),
}

impl DeepSeekModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::V4Flash => "deepseek-v4-flash",
            Self::V4Pro => "deepseek-v4-pro",
            Self::Custom(model) => model.as_str(),
        }
    }
}

impl fmt::Debug for DeepSeekModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for DeepSeekModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DeepSeekModel {
    type Err = Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "deepseek-v4-flash" => Self::V4Flash,
            "deepseek-v4-pro" => Self::V4Pro,
            _ => Self::Custom(value.to_owned()),
        })
    }
}

impl Serialize for DeepSeekModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DeepSeekModel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "deepseek-v4-flash" => Self::V4Flash,
            "deepseek-v4-pro" => Self::V4Pro,
            _ => Self::Custom(value),
        })
    }
}
