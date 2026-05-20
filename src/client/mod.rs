mod config;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{
    Method, Request, Uri,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue},
};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use tokio::time::timeout;

pub use config::DeepSeekConfig;

use crate::{
    Error, Result,
    stream::ChatStream,
    types::{ChatRequest, ChatResponse},
};

type HyperClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;

#[derive(Clone, Debug)]
pub struct DeepSeekClient {
    config: DeepSeekConfig,
    http: HyperClient,
}

impl DeepSeekClient {
    pub fn new(config: DeepSeekConfig) -> Result<Self> {
        if config.api_key.trim().is_empty() {
            return Err(Error::EmptyApiKey);
        }

        let connector = HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http1()
            .enable_http2()
            .build();
        let http = Client::builder(TokioExecutor::new()).build(connector);

        Ok(Self { config, http })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(DeepSeekConfig::from_env()?)
    }

    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse> {
        self.prepare_request(&mut request, false)?;
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "tracing")]
        let model = request.model.as_str().to_owned();
        #[cfg(feature = "tracing")]
        let message_count = request.messages.len();
        #[cfg(feature = "tracing")]
        let tool_count = request.tools.as_ref().map_or(0, Vec::len);
        let response = self.send(request).await?;
        let status = response.status();
        let body = response.into_body().collect().await?.to_bytes();

        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::client",
            operation = "chat",
            model = model.as_str(),
            message_count,
            tool_count,
            status = status.as_u16(),
            success = status.is_success(),
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "model call completed"
        );

        if !status.is_success() {
            return Err(Error::Api {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
            });
        }

        Ok(serde_json::from_slice(&body)?)
    }

    pub async fn stream_chat(&self, mut request: ChatRequest) -> Result<ChatStream> {
        self.prepare_request(&mut request, true)?;
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "tracing")]
        let model = request.model.as_str().to_owned();
        #[cfg(feature = "tracing")]
        let message_count = request.messages.len();
        #[cfg(feature = "tracing")]
        let tool_count = request.tools.as_ref().map_or(0, Vec::len);
        let response = self.send(request).await?;
        let status = response.status();

        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::client",
            operation = "stream_chat",
            model = model.as_str(),
            message_count,
            tool_count,
            status = status.as_u16(),
            success = status.is_success(),
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "streaming model call established"
        );

        if !status.is_success() {
            let body = response.into_body().collect().await?.to_bytes();
            return Err(Error::Api {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
            });
        }

        Ok(ChatStream::new(response.into_body()))
    }

    async fn send(&self, request: ChatRequest) -> Result<hyper::Response<hyper::body::Incoming>> {
        let http_request = self.build_http_request("/chat/completions", &request)?;
        match timeout(self.config.timeout, self.http.request(http_request)).await {
            Ok(response) => Ok(response?),
            Err(_) => Err(Error::Timeout {
                timeout: self.config.timeout,
            }),
        }
    }

    fn prepare_request(&self, request: &mut ChatRequest, stream: bool) -> Result<()> {
        if request.messages.is_empty() {
            return Err(Error::EmptyMessages);
        }

        if stream {
            request.stream = Some(true);
        }

        self.config.apply_defaults(request);
        self.validate_beta_requirements(request)
    }

    fn validate_beta_requirements(&self, request: &ChatRequest) -> Result<()> {
        let has_strict_tools = request.tools.as_ref().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| tool.function.strict.unwrap_or(false))
        });

        if has_strict_tools
            && !self
                .config
                .base_url
                .trim_end_matches('/')
                .ends_with("/beta")
        {
            return Err(Error::StrictToolsRequireBetaBaseUrl);
        }

        Ok(())
    }

    fn build_http_request<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<Request<Full<Bytes>>> {
        let uri: Uri = format!("{}{}", self.config.base_url.trim_end_matches('/'), path).parse()?;
        let auth = HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))?;
        let bytes = Bytes::from(serde_json::to_vec(body)?);

        Ok(Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header(AUTHORIZATION, auth)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .body(Full::new(bytes))?)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::{
        ToolDefinition,
        types::{ChatMessage, DeepSeekModel, FunctionDefinition},
    };

    #[test]
    fn config_debug_redacts_api_key() {
        let config = DeepSeekConfig::new("secret-key");
        let debug = format!("{config:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-key"));
    }

    #[tokio::test]
    async fn rejects_empty_message_list() {
        let client = DeepSeekClient::new(
            DeepSeekConfig::new("key")
                .with_base_url("http://127.0.0.1:9")
                .with_timeout(Duration::from_millis(10)),
        )
        .expect("client");
        let request = ChatRequest::new(DeepSeekModel::V4Flash, Vec::new());

        let error = client.chat(request).await.expect_err("empty messages fail");

        assert!(matches!(error, Error::EmptyMessages));
    }

    #[tokio::test]
    async fn strict_tools_require_beta_base_url() {
        let client = DeepSeekClient::new(DeepSeekConfig::new("key")).expect("client");
        let request = ChatRequest::new(DeepSeekModel::V4Flash, vec![ChatMessage::user("hello")])
            .with_tools(vec![ToolDefinition::function(
                FunctionDefinition::new("lookup")
                    .parameters(json!({"type": "object", "properties": {}}))
                    .strict(true),
            )]);

        let error = client.chat(request).await.expect_err("strict beta check");

        assert!(matches!(error, Error::StrictToolsRequireBetaBaseUrl));
    }
}
