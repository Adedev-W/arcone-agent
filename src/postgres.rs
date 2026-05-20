use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use tokio_postgres::{Config as TokioPostgresConfig, NoTls};

use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct PostgresStoreConfig {
    pub database_url: String,
    pub max_pool_size: usize,
    pub connect_timeout: Option<Duration>,
    pub statement_timeout: Option<Duration>,
}

impl PostgresStoreConfig {
    pub fn new(database_url: impl Into<String>) -> Self {
        Self {
            database_url: database_url.into(),
            max_pool_size: 16,
            connect_timeout: Some(Duration::from_secs(5)),
            statement_timeout: None,
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

    pub fn with_statement_timeout(mut self, statement_timeout: Option<Duration>) -> Self {
        self.statement_timeout = statement_timeout;
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
pub struct PostgresPool {
    pool: Pool,
    statement_timeout: Option<Duration>,
}

impl PostgresPool {
    pub async fn connect(config: PostgresStoreConfig) -> Result<Self> {
        let pool = Self::new(config)?;
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let _client = pool
            .pool
            .get()
            .await
            .map_err(|error| Error::DatabaseConnection(error.to_string()))?;
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "arcone_agent::postgres",
            operation = "connect",
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "postgres connection established"
        );
        Ok(pool)
    }

    pub fn new(config: PostgresStoreConfig) -> Result<Self> {
        let manager_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let manager = Manager::from_config(config.postgres_config()?, NoTls, manager_config);
        let pool = Pool::builder(manager)
            .max_size(config.max_pool_size)
            .build()
            .map_err(|error| Error::DatabasePool(error.to_string()))?;

        Ok(Self {
            pool,
            statement_timeout: config.statement_timeout,
        })
    }

    pub fn from_pool(pool: Pool) -> Self {
        Self {
            pool,
            statement_timeout: None,
        }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub async fn client(&self) -> Result<deadpool_postgres::Client> {
        #[cfg(feature = "tracing")]
        let started_at = std::time::Instant::now();
        let client = self
            .pool
            .get()
            .await
            .map_err(|error| Error::DatabasePool(error.to_string()))?;

        if let Some(timeout) = self.statement_timeout {
            let millis = duration_millis_i64(timeout).to_string();
            client
                .execute(
                    "SELECT set_config('statement_timeout', $1, false)",
                    &[&millis],
                )
                .await?;
        }

        #[cfg(feature = "tracing")]
        tracing::debug!(
            target: "arcone_agent::postgres",
            operation = "get_client",
            elapsed_ms = crate::observability::elapsed_ms(started_at),
            "postgres client acquired"
        );

        Ok(client)
    }
}

pub(crate) fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn duration_millis_i64(duration: Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}
