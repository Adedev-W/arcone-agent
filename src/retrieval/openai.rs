use std::fmt;
use std::time::Duration;

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
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

use crate::{Error, Result};

use super::traits::{EmbedFuture, Embedder};
use super::types::{Embedding, OpenAiEmbeddingModel};

type HyperClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;

#[derive(Clone)]
pub struct OpenAiConfig {
    api_key: String,
    base_url: String,
    model: OpenAiEmbeddingModel,
    timeout: Duration,
}

impl OpenAiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com".to_owned(),
            model: OpenAiEmbeddingModel::default(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| Error::MissingOpenAiApiKey)?;
        let mut config = Self::new(api_key);

        if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            let base_url = base_url.trim();
            if !base_url.is_empty() {
                config = config.with_base_url(base_url.to_owned());
            }
        }

        if let Ok(model) = std::env::var("OPENAI_EMBEDDING_MODEL") {
            let model = model.trim();
            if !model.is_empty() {
                config = config.with_model(OpenAiEmbeddingModel::from(model));
            }
        }

        Ok(config)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<OpenAiEmbeddingModel>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &OpenAiEmbeddingModel {
        &self.model
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

impl fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct OpenAiEmbedder {
    config: OpenAiConfig,
    http: HyperClient,
}

impl OpenAiEmbedder {
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        if config.api_key.trim().is_empty() {
            return Err(Error::EmptyOpenAiApiKey);
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
        Self::new(OpenAiConfig::from_env()?)
    }

    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }

    pub async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        validate_texts(&texts)?;
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        #[cfg(feature = "tracing")]
        let input_count = texts.len();

        let request = OpenAiEmbeddingRequest {
            model: self.config.model.as_str().to_owned(),
            input: texts,
        };
        #[cfg(feature = "tracing")]
        let model = request.model.clone();
        let response = self.send(request).await?;
        let status = response.status();
        let body = response.into_body().collect().await?.to_bytes();

        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::retrieval",
            operation = "openai_embed",
            model = model.as_str(),
            input_count,
            status = status.as_u16(),
            success = status.is_success(),
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "embedding call completed"
        );

        if !status.is_success() {
            return Err(Error::OpenAiApi {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
            });
        }

        let response: OpenAiEmbeddingResponse = serde_json::from_slice(&body)?;
        if response.data.is_empty() {
            return Err(Error::EmbeddingFailure(
                "OpenAI embedding response did not contain data".to_owned(),
            ));
        }

        let mut embeddings = response
            .data
            .into_iter()
            .map(|item| {
                if item.embedding.is_empty() {
                    return Err(Error::EmbeddingFailure(format!(
                        "OpenAI embedding at index {} was empty",
                        item.index
                    )));
                }

                Ok(Embedding::new(item.index, item.embedding))
            })
            .collect::<Result<Vec<_>>>()?;
        embeddings.sort_by_key(|embedding| embedding.text_index);
        Ok(embeddings)
    }

    async fn send(
        &self,
        request: OpenAiEmbeddingRequest,
    ) -> Result<hyper::Response<hyper::body::Incoming>> {
        let http_request = self.build_http_request("/v1/embeddings", &request)?;
        match timeout(self.config.timeout, self.http.request(http_request)).await {
            Ok(response) => Ok(response?),
            Err(_) => Err(Error::OpenAiTimeout {
                timeout: self.config.timeout,
            }),
        }
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

impl Embedder for OpenAiEmbedder {
    fn embed(&self, texts: Vec<String>) -> EmbedFuture<Vec<Embedding>> {
        let embedder = self.clone();
        Box::pin(async move { embedder.embed_texts(texts).await })
    }
}

fn validate_texts(texts: &[String]) -> Result<()> {
    if texts.is_empty() {
        return Err(Error::EmptyEmbeddingInput);
    }

    if texts.iter().any(|text| text.trim().is_empty()) {
        return Err(Error::EmbeddingFailure(
            "embedding input cannot contain empty strings".to_owned(),
        ));
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    object: String,
    #[allow(dead_code)]
    usage: Option<OpenAiEmbeddingUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
    #[allow(dead_code)]
    object: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingUsage {
    #[allow(dead_code)]
    prompt_tokens: u64,
    #[allow(dead_code)]
    total_tokens: u64,
}
