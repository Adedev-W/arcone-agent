use std::time::Duration;

use arcone_agent::{
    DeepSeekConfig, Error, OpenAiConfig, PostgresPool, PostgresStoreConfig, redact_secret,
};

#[test]
fn redaction_helper_never_returns_secret_value() {
    assert_eq!(redact_secret("sk-live-secret"), "<redacted>");
}

#[test]
fn provider_config_debug_output_redacts_api_keys() {
    let deepseek = format!("{:?}", DeepSeekConfig::new("deepseek-secret"));
    let openai = format!("{:?}", OpenAiConfig::new("openai-secret"));

    assert!(deepseek.contains("<redacted>"));
    assert!(!deepseek.contains("deepseek-secret"));
    assert!(openai.contains("<redacted>"));
    assert!(!openai.contains("openai-secret"));
}

#[test]
fn typed_error_display_preserves_operational_context() {
    let blocked = Error::GuardrailBlocked {
        stage: "output".to_owned(),
        reason: "answer is empty".to_owned(),
    };
    let composer = Error::ComposerFailure("missing final text".to_owned());

    assert_eq!(
        blocked.to_string(),
        "guardrail blocked output: answer is empty"
    );
    assert_eq!(composer.to_string(), "composer failure: missing final text");
}

#[tokio::test]
async fn postgres_connect_maps_initial_pool_get_to_connection_error() {
    let config = PostgresStoreConfig::new("host=127.0.0.1 port=1 user=postgres dbname=postgres")
        .with_max_pool_size(1)
        .with_connect_timeout(Some(Duration::from_millis(50)));

    let error = match PostgresPool::connect(config).await {
        Ok(_) => panic!("port 1 should not accept a PostgreSQL connection"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::DatabaseConnection(_)));
}
