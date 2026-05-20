use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use tokio_postgres::{Config as TokioPostgresConfig, NoTls};

use crate::{ChatMessage, Error, Result};

use super::store::{MemoryFuture, MemoryStore};
use super::types::{SessionId, SessionMetadata};

#[derive(Clone, Debug)]
pub struct PostgresSessionConfig {
    pub database_url: String,
    pub max_pool_size: usize,
    pub connect_timeout: Option<Duration>,
}

impl PostgresSessionConfig {
    pub fn new(database_url: impl Into<String>) -> Self {
        Self {
            database_url: database_url.into(),
            max_pool_size: 16,
            connect_timeout: Some(Duration::from_secs(5)),
        }
    }

    pub fn from_env() -> Result<Self> {
        let database_url = std::env::var("DATABASE_URL").map_err(|_| Error::MissingDatabaseUrl)?;
        Ok(Self::new(database_url))
    }

    pub fn with_max_pool_size(mut self, max_pool_size: usize) -> Self {
        self.max_pool_size = max_pool_size;
        self
    }

    pub fn with_connect_timeout(mut self, connect_timeout: Option<Duration>) -> Self {
        self.connect_timeout = connect_timeout;
        self
    }

    fn postgres_config(&self) -> Result<TokioPostgresConfig> {
        let mut config = TokioPostgresConfig::from_str(&self.database_url)
            .map_err(|error| Error::InvalidDatabaseUrl(error.to_string()))?;

        if let Some(connect_timeout) = self.connect_timeout {
            config.connect_timeout(connect_timeout);
        }

        Ok(config)
    }
}

#[derive(Clone)]
pub struct PostgresSessionStore {
    pool: Pool,
}

impl PostgresSessionStore {
    pub fn new(config: PostgresSessionConfig) -> Result<Self> {
        let manager_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let manager = Manager::from_config(config.postgres_config()?, NoTls, manager_config);
        let pool = Pool::builder(manager)
            .max_size(config.max_pool_size)
            .build()
            .map_err(|error| Error::DatabasePool(error.to_string()))?;

        Ok(Self { pool })
    }

    pub async fn connect(config: PostgresSessionConfig) -> Result<Self> {
        let store = Self::new(config)?;
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let _client = store
            .pool
            .get()
            .await
            .map_err(|error| Error::DatabaseConnection(error.to_string()))?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::session",
            operation = "connect",
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "postgres session connection established"
        );
        store.migrate().await?;
        Ok(store)
    }

    pub fn from_pool(pool: Pool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<()> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let client = self.client().await?;
        client
            .batch_execute(
                r#"
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    metadata JSONB NOT NULL,
                    created_at BIGINT NOT NULL,
                    updated_at BIGINT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_messages (
                    session_id TEXT NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    message JSONB NOT NULL,
                    PRIMARY KEY (session_id, position)
                );

                CREATE INDEX IF NOT EXISTS idx_agent_messages_session_position
                    ON agent_messages (session_id, position);
                "#,
            )
            .await
            .map_err(|error| Error::DatabaseMigration(error.to_string()))?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::session",
            operation = "migrate",
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "postgres session migration completed"
        );
        Ok(())
    }

    async fn client(&self) -> Result<deadpool_postgres::Client> {
        self.pool
            .get()
            .await
            .map_err(|error| Error::DatabasePool(error.to_string()))
    }
}

impl MemoryStore for PostgresSessionStore {
    fn save_messages(&self, id: &SessionId, messages: &[ChatMessage]) -> MemoryFuture<()> {
        let pool = self.pool.clone();
        let session_id = id.as_str().to_owned();
        let messages = messages.to_vec();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let mut client = pool
                .get()
                .await
                .map_err(|error| Error::DatabasePool(error.to_string()))?;
            let transaction = client.transaction().await?;
            let metadata = SessionMetadata::new();
            let metadata_json = serde_json::to_value(&metadata)?;
            let created_at = millis_to_i64(metadata.created_at);
            let updated_at = millis_to_i64(now_millis());

            transaction
                .execute(
                    r#"
                    INSERT INTO agent_sessions (session_id, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (session_id) DO UPDATE
                    SET updated_at = EXCLUDED.updated_at
                    "#,
                    &[&session_id, &metadata_json, &created_at, &updated_at],
                )
                .await?;

            transaction
                .execute(
                    "DELETE FROM agent_messages WHERE session_id = $1",
                    &[&session_id],
                )
                .await?;

            for (position, message) in messages.iter().enumerate() {
                let message_json = serde_json::to_value(message)?;
                let position = position as i32;

                transaction
                    .execute(
                        r#"
                        INSERT INTO agent_messages (session_id, position, message)
                        VALUES ($1, $2, $3)
                        "#,
                        &[&session_id, &position, &message_json],
                    )
                    .await?;
            }

            transaction.commit().await?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::session",
                operation = "save_messages",
                message_count = messages.len(),
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres session messages saved"
            );
            Ok(())
        })
    }

    fn load_messages(&self, id: &SessionId) -> MemoryFuture<Vec<ChatMessage>> {
        let pool = self.pool.clone();
        let session_id = id.as_str().to_owned();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool
                .get()
                .await
                .map_err(|error| Error::DatabasePool(error.to_string()))?;
            let rows = client
                .query(
                    r#"
                    SELECT message
                    FROM agent_messages
                    WHERE session_id = $1
                    ORDER BY position ASC
                    "#,
                    &[&session_id],
                )
                .await?;
            let mut messages = Vec::with_capacity(rows.len());
            #[cfg(feature = "tracing")]
            let row_count = rows.len();

            for row in rows {
                let value: serde_json::Value = row.get("message");
                messages.push(serde_json::from_value(value)?);
            }

            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::session",
                operation = "load_messages",
                row_count,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres session messages loaded"
            );
            Ok(messages)
        })
    }

    fn save_metadata(&self, id: &SessionId, metadata: &SessionMetadata) -> MemoryFuture<()> {
        let pool = self.pool.clone();
        let session_id = id.as_str().to_owned();
        let metadata = metadata.clone();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool
                .get()
                .await
                .map_err(|error| Error::DatabasePool(error.to_string()))?;
            let metadata_json = serde_json::to_value(&metadata)?;
            let created_at = millis_to_i64(metadata.created_at);
            let updated_at = millis_to_i64(metadata.updated_at);

            client
                .execute(
                    r#"
                    INSERT INTO agent_sessions (session_id, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (session_id) DO UPDATE
                    SET metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    "#,
                    &[&session_id, &metadata_json, &created_at, &updated_at],
                )
                .await?;

            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::session",
                operation = "save_metadata",
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres session metadata saved"
            );
            Ok(())
        })
    }

    fn load_metadata(&self, id: &SessionId) -> MemoryFuture<Option<SessionMetadata>> {
        let pool = self.pool.clone();
        let session_id = id.as_str().to_owned();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool
                .get()
                .await
                .map_err(|error| Error::DatabasePool(error.to_string()))?;
            let row = client
                .query_opt(
                    "SELECT metadata FROM agent_sessions WHERE session_id = $1",
                    &[&session_id],
                )
                .await?;

            let metadata = match row {
                Some(row) => {
                    let value: serde_json::Value = row.get("metadata");
                    Some(serde_json::from_value(value)?)
                }
                None => None,
            };
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::session",
                operation = "load_metadata",
                found = metadata.is_some(),
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres session metadata loaded"
            );
            Ok(metadata)
        })
    }

    fn clear(&self, id: &SessionId) -> MemoryFuture<()> {
        let pool = self.pool.clone();
        let session_id = id.as_str().to_owned();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool
                .get()
                .await
                .map_err(|error| Error::DatabasePool(error.to_string()))?;
            client
                .execute(
                    "DELETE FROM agent_sessions WHERE session_id = $1",
                    &[&session_id],
                )
                .await?;
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::session",
                operation = "clear",
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres session cleared"
            );
            Ok(())
        })
    }

    fn exists(&self, id: &SessionId) -> MemoryFuture<bool> {
        let pool = self.pool.clone();
        let session_id = id.as_str().to_owned();

        Box::pin(async move {
            #[cfg(feature = "tracing")]
            let started_at = std::time::Instant::now();
            let client = pool
                .get()
                .await
                .map_err(|error| Error::DatabasePool(error.to_string()))?;
            let row = client
                .query_one(
                    "SELECT EXISTS (SELECT 1 FROM agent_sessions WHERE session_id = $1)",
                    &[&session_id],
                )
                .await?;
            let exists = row.get::<_, bool>(0);
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "arcone_agent::session",
                operation = "exists",
                exists,
                elapsed_ms = crate::observability::elapsed_ms(started_at),
                "postgres session existence checked"
            );
            Ok(exists)
        })
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn millis_to_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}
