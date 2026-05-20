use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use arcone_agent as core;
use pyo3::IntoPyObjectExt;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyStopAsyncIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use serde_json::{Map, Number, Value};
use tokio::sync::{Mutex, mpsc, oneshot};

create_exception!(_arcone_agent, ArconeError, PyException);
create_exception!(_arcone_agent, ConfigError, ArconeError);
create_exception!(_arcone_agent, ApiError, ArconeError);
create_exception!(_arcone_agent, TimeoutError, ArconeError);
create_exception!(_arcone_agent, ToolError, ArconeError);
create_exception!(_arcone_agent, StreamingUnsupportedError, ToolError);
create_exception!(_arcone_agent, SessionError, ArconeError);
create_exception!(_arcone_agent, DatabaseError, ArconeError);
create_exception!(_arcone_agent, KnowledgeError, ArconeError);
create_exception!(_arcone_agent, RetrievalError, ArconeError);

type SharedAgent = Arc<Mutex<Option<core::Agent>>>;
type PyAwaitableFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

enum PythonToolCall {
    Ready(String),
    Pending(PyAwaitableFuture),
}

struct PythonTool {
    name: String,
    definition: core::ToolDefinition,
    handler: Py<PyAny>,
}

impl core::Tool for PythonTool {
    fn definition(&self) -> core::ToolDefinition {
        self.definition.clone()
    }

    fn call(&self, arguments: Value) -> core::ToolFuture {
        let name = self.name.clone();
        let handler = Python::attach(|py| self.handler.clone_ref(py));

        Box::pin(async move {
            match Python::attach(|py| start_python_tool_call(py, &name, &handler, &arguments))? {
                PythonToolCall::Ready(output) => Ok(output),
                PythonToolCall::Pending(future) => {
                    let output = future.await.map_err(|error| {
                        Python::attach(|py| map_python_tool_error(py, &name, error))
                    })?;
                    Python::attach(|py| py_tool_output_to_json_string(py, &name, output.bind(py)))
                }
            }
        })
    }
}

#[pyclass(
    name = "Agent",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyAgent {
    inner: SharedAgent,
}

#[pymethods]
impl PyAgent {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        system = None,
        model = None,
        thinking = None,
        reasoning_effort = None,
        max_tokens = None,
        max_tool_rounds = None,
        session_id = None,
        session_store = None
    ))]
    fn from_env(
        system: Option<String>,
        model: Option<String>,
        thinking: Option<bool>,
        reasoning_effort: Option<String>,
        max_tokens: Option<u32>,
        max_tool_rounds: Option<usize>,
        session_id: Option<String>,
        session_store: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let agent = build_agent_from_env(
            system,
            model,
            thinking,
            reasoning_effort,
            max_tokens,
            max_tool_rounds,
            session_id,
            session_store,
        )?;
        Ok(Self::new(agent))
    }

    fn ask_text<'py>(&self, py: Python<'py>, prompt: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let agent = guard.as_mut().ok_or_else(agent_moved_error)?;
            agent.ask_text(prompt).await.map_err(map_error)
        })
    }

    fn ask<'py>(&self, py: Python<'py>, prompt: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let agent = guard.as_mut().ok_or_else(agent_moved_error)?;
            let response = agent.ask(prompt).await.map_err(map_error)?;
            PyAgentResponse::from_core(response)
        })
    }

    fn stream<'py>(&self, py: Python<'py>, prompt: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = start_agent_stream_worker(inner, prompt);
            stream.await_started().await?;
            Ok(stream)
        })
    }

    fn stream_text(&self, prompt: String) -> PyAgentStream {
        start_agent_stream_worker(Arc::clone(&self.inner), prompt)
    }

    fn clear_history(&self) -> PyResult<()> {
        let mut guard = self
            .inner
            .try_lock()
            .map_err(|_| ArconeError::new_err("agent is busy"))?;
        let agent = guard.as_mut().ok_or_else(agent_moved_error)?;
        agent.clear_history();
        Ok(())
    }

    fn clear_session<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let agent = guard.as_mut().ok_or_else(agent_moved_error)?;
            agent.clear_session().await.map_err(map_error)
        })
    }

    #[pyo3(signature = (name, description, schema, handler))]
    fn add_tool(
        &self,
        py: Python<'_>,
        name: String,
        description: String,
        schema: &Bound<'_, PyAny>,
        handler: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if !handler.is_callable() {
            return Err(PyTypeError::new_err("handler must be callable"));
        }

        let parameters = py_to_value(py, schema).map_err(|error| {
            ToolError::new_err(format!(
                "tool `{name}` schema must be JSON serializable: {error}"
            ))
        })?;
        if !parameters.is_object() {
            return Err(ToolError::new_err(format!(
                "tool `{name}` schema must be a JSON object"
            )));
        }

        let definition = core::ToolDefinition::function(
            core::FunctionDefinition::new(name.clone())
                .description(description)
                .parameters(parameters),
        );
        let tool = PythonTool {
            name,
            definition,
            handler: handler.clone().unbind(),
        };

        let mut guard = self
            .inner
            .try_lock()
            .map_err(|_| ArconeError::new_err("agent is busy"))?;
        let agent = guard.as_mut().ok_or_else(agent_moved_error)?;
        agent.try_add_tool(tool).map_err(map_error)?;
        Ok(())
    }
}

impl PyAgent {
    fn new(agent: core::Agent) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(agent))),
        }
    }

    fn take(&self) -> PyResult<core::Agent> {
        let mut guard = self
            .inner
            .try_lock()
            .map_err(|_| ArconeError::new_err("agent is busy"))?;
        guard.take().ok_or_else(agent_moved_error)
    }
}

#[derive(Debug)]
enum StreamWorkerError {
    Core(core::Error),
    AgentMoved,
    Conversion(String),
}

enum StreamCommand {
    Next {
        respond_to: oneshot::Sender<Result<Option<String>, StreamWorkerError>>,
    },
    Finish {
        respond_to: oneshot::Sender<Result<PyAgentResponse, StreamWorkerError>>,
    },
    Close,
}

struct PyAgentStreamState {
    start: Option<oneshot::Receiver<Result<(), StreamWorkerError>>>,
    final_response: Option<PyAgentResponse>,
    closed: bool,
}

#[pyclass(
    name = "AgentStream",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyAgentStream {
    commands: mpsc::UnboundedSender<StreamCommand>,
    state: Arc<StdMutex<PyAgentStreamState>>,
}

#[pymethods]
impl PyAgentStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { stream.next_delta().await })
    }

    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { stream.finish_stream().await })
    }

    fn close(&self) -> PyResult<()> {
        self.close_stream()
    }
}

impl PyAgentStream {
    fn new(
        commands: mpsc::UnboundedSender<StreamCommand>,
        start: oneshot::Receiver<Result<(), StreamWorkerError>>,
    ) -> Self {
        Self {
            commands,
            state: Arc::new(StdMutex::new(PyAgentStreamState {
                start: Some(start),
                final_response: None,
                closed: false,
            })),
        }
    }

    async fn next_delta(self) -> PyResult<String> {
        if self.cached_response()?.is_some() {
            return Err(PyStopAsyncIteration::new_err("stream exhausted"));
        }

        self.await_started().await?;
        if self.is_closed()? {
            return Err(PyStopAsyncIteration::new_err("stream closed"));
        }

        let (respond_to, response) = oneshot::channel();
        self.commands
            .send(StreamCommand::Next { respond_to })
            .map_err(|_| ArconeError::new_err("stream is closed"))?;

        match response
            .await
            .map_err(|_| ArconeError::new_err("stream worker stopped"))?
        {
            Ok(Some(delta)) => Ok(delta),
            Ok(None) => {
                let response = self.request_finish().await?;
                self.store_response(response)?;
                Err(PyStopAsyncIteration::new_err("stream exhausted"))
            }
            Err(error) => {
                self.mark_closed()?;
                Err(map_stream_worker_error(error))
            }
        }
    }

    async fn finish_stream(self) -> PyResult<PyAgentResponse> {
        if let Some(response) = self.cached_response()? {
            return Ok(response);
        }

        self.await_started().await?;
        if self.is_closed()? {
            return Err(ArconeError::new_err("stream is closed"));
        }

        let response = self.request_finish().await?;
        self.store_response(response.clone())?;
        Ok(response)
    }

    async fn await_started(&self) -> PyResult<()> {
        let start = {
            let mut state = self.lock_state()?;
            if state.closed {
                return Err(ArconeError::new_err("stream is closed"));
            }
            state.start.take()
        };

        if let Some(start) = start {
            match start.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(error)) => {
                    self.mark_closed()?;
                    Err(map_stream_worker_error(error))
                }
                Err(_) => {
                    self.mark_closed()?;
                    Err(ArconeError::new_err("stream worker stopped before start"))
                }
            }
        } else {
            Ok(())
        }
    }

    async fn request_finish(&self) -> PyResult<PyAgentResponse> {
        let (respond_to, response) = oneshot::channel();
        self.commands
            .send(StreamCommand::Finish { respond_to })
            .map_err(|_| ArconeError::new_err("stream is closed"))?;

        match response
            .await
            .map_err(|_| ArconeError::new_err("stream worker stopped"))?
        {
            Ok(response) => {
                self.mark_closed()?;
                Ok(response)
            }
            Err(error) => {
                self.mark_closed()?;
                Err(map_stream_worker_error(error))
            }
        }
    }

    fn cached_response(&self) -> PyResult<Option<PyAgentResponse>> {
        Ok(self.lock_state()?.final_response.clone())
    }

    fn store_response(&self, response: PyAgentResponse) -> PyResult<()> {
        let mut state = self.lock_state()?;
        state.final_response = Some(response);
        state.closed = true;
        Ok(())
    }

    fn close_stream(&self) -> PyResult<()> {
        self.mark_closed()?;
        let _ = self.commands.send(StreamCommand::Close);
        Ok(())
    }

    fn mark_closed(&self) -> PyResult<()> {
        self.lock_state()?.closed = true;
        Ok(())
    }

    fn is_closed(&self) -> PyResult<bool> {
        Ok(self.lock_state()?.closed)
    }

    fn lock_state(&self) -> PyResult<std::sync::MutexGuard<'_, PyAgentStreamState>> {
        self.state
            .lock()
            .map_err(|_| ArconeError::new_err("stream state lock is poisoned"))
    }
}

fn start_agent_stream_worker(inner: SharedAgent, prompt: String) -> PyAgentStream {
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let (start_tx, start_rx) = oneshot::channel();

    pyo3_async_runtimes::tokio::get_runtime()
        .spawn(run_agent_stream_worker(inner, prompt, start_tx, command_rx));

    PyAgentStream::new(command_tx, start_rx)
}

async fn run_agent_stream_worker(
    inner: SharedAgent,
    prompt: String,
    started: oneshot::Sender<Result<(), StreamWorkerError>>,
    mut commands: mpsc::UnboundedReceiver<StreamCommand>,
) {
    let mut guard = inner.lock().await;
    let Some(agent) = guard.as_mut() else {
        let _ = started.send(Err(StreamWorkerError::AgentMoved));
        return;
    };

    let mut stream = match agent.stream(prompt).await {
        Ok(stream) => stream,
        Err(error) => {
            let _ = started.send(Err(StreamWorkerError::Core(error)));
            return;
        }
    };

    let _ = started.send(Ok(()));

    while let Some(command) = commands.recv().await {
        match command {
            StreamCommand::Next { respond_to } => match stream.next_text().await {
                Ok(delta) => {
                    let _ = respond_to.send(Ok(delta));
                }
                Err(error) => {
                    let _ = respond_to.send(Err(StreamWorkerError::Core(error)));
                    return;
                }
            },
            StreamCommand::Finish { respond_to } => {
                let response = match stream.finish().await {
                    Ok(response) => PyAgentResponse::from_core(response)
                        .map_err(|error| StreamWorkerError::Conversion(error.to_string())),
                    Err(error) => Err(StreamWorkerError::Core(error)),
                };
                let _ = respond_to.send(response);
                return;
            }
            StreamCommand::Close => return,
        }
    }
}

#[pyclass(
    name = "AgentResponse",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyAgentResponse {
    content: Option<String>,
    reasoning_content: Option<String>,
    finish_reason: String,
    usage: Option<Value>,
    history: Vec<Value>,
}

#[pymethods]
impl PyAgentResponse {
    #[getter]
    fn content(&self) -> Option<String> {
        self.content.clone()
    }

    #[getter]
    fn reasoning_content(&self) -> Option<String> {
        self.reasoning_content.clone()
    }

    #[getter]
    fn finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        option_value_to_py(py, self.usage.as_ref())
    }

    #[getter]
    fn history(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &Value::Array(self.history.clone()))
    }
}

impl PyAgentResponse {
    fn from_core(response: core::AgentResponse) -> PyResult<Self> {
        let content = response.content().map(str::to_owned);
        let reasoning_content = response.reasoning_content().map(str::to_owned);
        let finish_reason = finish_reason_string(&response.finish_reason);
        let usage = response.usage.map(value_from_serialize).transpose()?;
        let history = response
            .history
            .iter()
            .map(value_from_serialize)
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            content,
            reasoning_content,
            finish_reason,
            usage,
            history,
        })
    }
}

#[pyclass(
    name = "InMemorySessionStore",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyInMemorySessionStore {
    inner: Arc<core::InMemorySessionStore>,
}

#[pymethods]
impl PyInMemorySessionStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(core::InMemorySessionStore::new()),
        }
    }
}

#[pyclass(
    name = "PostgresSessionStore",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyPostgresSessionStore {
    inner: Arc<core::PostgresSessionStore>,
}

#[pymethods]
impl PyPostgresSessionStore {
    #[staticmethod]
    #[pyo3(signature = (max_pool_size = 16, connect_timeout_seconds = 5.0))]
    fn from_env<'py>(
        py: Python<'py>,
        max_pool_size: usize,
        connect_timeout_seconds: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let timeout = duration_from_seconds(connect_timeout_seconds, "connect_timeout_seconds")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let config = core::PostgresSessionConfig::from_env()
                .map_err(map_error)?
                .with_max_pool_size(max_pool_size)
                .with_connect_timeout(Some(timeout));
            let store = core::PostgresSessionStore::connect(config)
                .await
                .map_err(map_error)?;
            Ok(Self {
                inner: Arc::new(store),
            })
        })
    }
}

#[pyclass(
    name = "PostgresPool",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyPostgresPool {
    inner: core::PostgresPool,
}

#[pymethods]
impl PyPostgresPool {
    #[staticmethod]
    #[pyo3(signature = (
        max_pool_size = 16,
        connect_timeout_seconds = 5.0,
        statement_timeout_seconds = None
    ))]
    fn from_env<'py>(
        py: Python<'py>,
        max_pool_size: usize,
        connect_timeout_seconds: f64,
        statement_timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let connect_timeout =
            duration_from_seconds(connect_timeout_seconds, "connect_timeout_seconds")?;
        let statement_timeout = statement_timeout_seconds
            .map(|seconds| duration_from_seconds(seconds, "statement_timeout_seconds"))
            .transpose()?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let config = core::PostgresStoreConfig::from_env()
                .map_err(map_error)?
                .with_max_pool_size(max_pool_size)
                .with_connect_timeout(Some(connect_timeout))
                .with_statement_timeout(statement_timeout);
            let pool = core::PostgresPool::connect(config)
                .await
                .map_err(map_error)?;
            Ok(Self { inner: pool })
        })
    }
}

#[pyclass(
    name = "Document",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyDocument {
    inner: core::Document,
}

#[pymethods]
impl PyDocument {
    #[staticmethod]
    #[pyo3(signature = (id, content, title = None, source = None, path = None, metadata = None))]
    fn text(
        py: Python<'_>,
        id: String,
        content: String,
        title: Option<String>,
        source: Option<String>,
        path: Option<String>,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut document = core::Document::text(id, content);
        if let Some(title) = title {
            document = document.with_title(title);
        }
        if let Some(source) = source {
            document = document.with_source(source);
        }
        if let Some(path) = path {
            document = document.with_path(path);
        }
        if let Some(metadata) = metadata {
            document = document.with_metadata(py_to_value(py, metadata.bind(py))?);
        }
        Ok(Self { inner: document })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.as_str().to_owned()
    }

    #[getter]
    fn content(&self) -> String {
        self.inner.content.clone()
    }

    #[getter]
    fn title(&self) -> Option<String> {
        self.inner.title.clone()
    }

    #[getter]
    fn source(&self) -> Option<String> {
        self.inner.source.clone()
    }

    #[getter]
    fn path(&self) -> Option<String> {
        self.inner.path.clone()
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        option_value_to_py(py, self.inner.metadata.as_ref())
    }
}

#[pyclass(
    name = "KnowledgeChunk",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyKnowledgeChunk {
    inner: core::KnowledgeChunk,
}

#[pymethods]
impl PyKnowledgeChunk {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.as_str().to_owned()
    }

    #[getter]
    fn document_id(&self) -> String {
        self.inner.document_id.as_str().to_owned()
    }

    #[getter]
    fn chunk_index(&self) -> usize {
        self.inner.chunk_index
    }

    #[getter]
    fn content(&self) -> String {
        self.inner.content.clone()
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &chunk_metadata_value(&self.inner.metadata))
    }
}

impl From<core::KnowledgeChunk> for PyKnowledgeChunk {
    fn from(inner: core::KnowledgeChunk) -> Self {
        Self { inner }
    }
}

#[pyclass(
    name = "ScoredChunk",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyScoredChunk {
    inner: core::ScoredChunk,
}

#[pymethods]
impl PyScoredChunk {
    #[getter]
    fn chunk(&self) -> PyKnowledgeChunk {
        PyKnowledgeChunk::from(self.inner.chunk.clone())
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }
}

impl From<core::ScoredChunk> for PyScoredChunk {
    fn from(inner: core::ScoredChunk) -> Self {
        Self { inner }
    }
}

#[pyclass(
    name = "InMemoryKnowledgeBase",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyInMemoryKnowledgeBase {
    inner: Arc<core::InMemoryKnowledgeBase>,
}

#[pymethods]
impl PyInMemoryKnowledgeBase {
    #[new]
    #[pyo3(signature = (max_chars = 1200, overlap_chars = 120))]
    fn new(max_chars: usize, overlap_chars: usize) -> Self {
        let options = core::ChunkOptions::new(max_chars, overlap_chars);
        Self {
            inner: Arc::new(core::InMemoryKnowledgeBase::new().with_chunk_options(options)),
        }
    }

    fn add_document<'py>(
        &self,
        py: Python<'py>,
        document: PyRef<'_, PyDocument>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        let document = document.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::KnowledgeBase::add_document(kb.as_ref(), document)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn list_documents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let documents = core::KnowledgeBase::list_documents(kb.as_ref())
                .await
                .map_err(map_error)?;
            Ok(documents
                .into_iter()
                .map(|inner| PyDocument { inner })
                .collect::<Vec<_>>())
        })
    }

    fn remove_document<'py>(
        &self,
        py: Python<'py>,
        document_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = core::DocumentId::new(document_id);
            core::KnowledgeBase::remove_document(kb.as_ref(), &id)
                .await
                .map_err(map_error)
        })
    }

    fn chunk_document<'py>(
        &self,
        py: Python<'py>,
        document: PyRef<'_, PyDocument>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        let document = document.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::KnowledgeBase::chunk_document(kb.as_ref(), &document)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn chunks_for_document<'py>(
        &self,
        py: Python<'py>,
        document_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = core::DocumentId::new(document_id);
            let chunks = core::KnowledgeBase::chunks_for_document(kb.as_ref(), &id)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn chunks_for_source<'py>(
        &self,
        py: Python<'py>,
        source: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::KnowledgeBase::chunks_for_source(kb.as_ref(), &source)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }
}

#[pyclass(
    name = "PostgresKnowledgeBase",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyPostgresKnowledgeBase {
    inner: Arc<core::PostgresKnowledgeBase>,
}

#[pymethods]
impl PyPostgresKnowledgeBase {
    #[new]
    #[pyo3(signature = (pool, max_chars = 1200, overlap_chars = 120))]
    fn new(pool: PyRef<'_, PyPostgresPool>, max_chars: usize, overlap_chars: usize) -> Self {
        let options = core::ChunkOptions::new(max_chars, overlap_chars);
        Self {
            inner: Arc::new(
                core::PostgresKnowledgeBase::new(pool.inner.clone()).with_chunk_options(options),
            ),
        }
    }

    fn migrate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            kb.migrate().await.map_err(map_error)
        })
    }

    fn add_document<'py>(
        &self,
        py: Python<'py>,
        document: PyRef<'_, PyDocument>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        let document = document.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::KnowledgeBase::add_document(kb.as_ref(), document)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn list_documents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let documents = core::KnowledgeBase::list_documents(kb.as_ref())
                .await
                .map_err(map_error)?;
            Ok(documents
                .into_iter()
                .map(|inner| PyDocument { inner })
                .collect::<Vec<_>>())
        })
    }

    fn remove_document<'py>(
        &self,
        py: Python<'py>,
        document_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = core::DocumentId::new(document_id);
            core::KnowledgeBase::remove_document(kb.as_ref(), &id)
                .await
                .map_err(map_error)
        })
    }

    fn chunk_document<'py>(
        &self,
        py: Python<'py>,
        document: PyRef<'_, PyDocument>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        let document = document.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::KnowledgeBase::chunk_document(kb.as_ref(), &document)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn chunks_for_document<'py>(
        &self,
        py: Python<'py>,
        document_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = core::DocumentId::new(document_id);
            let chunks = core::KnowledgeBase::chunks_for_document(kb.as_ref(), &id)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn chunks_for_source<'py>(
        &self,
        py: Python<'py>,
        source: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kb = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::KnowledgeBase::chunks_for_source(kb.as_ref(), &source)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyKnowledgeChunk::from)
                .collect::<Vec<_>>())
        })
    }
}

#[pyclass(
    name = "OpenAiEmbedder",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyOpenAiEmbedder {
    inner: core::OpenAiEmbedder,
}

#[pymethods]
impl PyOpenAiEmbedder {
    #[staticmethod]
    #[pyo3(signature = (model = None, base_url = None, timeout_seconds = 60.0))]
    fn from_env(
        model: Option<String>,
        base_url: Option<String>,
        timeout_seconds: f64,
    ) -> PyResult<Self> {
        let mut config = core::OpenAiConfig::from_env().map_err(map_error)?;
        if let Some(model) = model {
            config = config.with_model(model);
        }
        if let Some(base_url) = base_url {
            config = config.with_base_url(base_url);
        }
        config = config.with_timeout(duration_from_seconds(timeout_seconds, "timeout_seconds")?);
        Ok(Self {
            inner: core::OpenAiEmbedder::new(config).map_err(map_error)?,
        })
    }
}

#[pyclass(
    name = "InMemoryVectorRetriever",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyInMemoryVectorRetriever {
    inner: Arc<core::InMemoryVectorRetriever>,
}

#[pymethods]
impl PyInMemoryVectorRetriever {
    #[new]
    fn new(embedder: PyRef<'_, PyOpenAiEmbedder>) -> Self {
        Self {
            inner: Arc::new(core::InMemoryVectorRetriever::new(embedder.inner.clone())),
        }
    }

    fn index<'py>(
        &self,
        py: Python<'py>,
        chunks: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let retriever = Arc::clone(&self.inner);
        let chunks = extract_chunks(chunks)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            retriever.index(chunks).await.map_err(map_error)
        })
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let retriever = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::Retriever::retrieve(retriever.as_ref(), &query, top_k)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyScoredChunk::from)
                .collect::<Vec<_>>())
        })
    }

    fn len(&self) -> PyResult<usize> {
        self.inner.len().map_err(map_error)
    }

    fn is_empty(&self) -> PyResult<bool> {
        self.inner.is_empty().map_err(map_error)
    }
}

#[pyclass(
    name = "PgVectorRetriever",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyPgVectorRetriever {
    inner: Arc<core::PgVectorRetriever>,
}

#[pymethods]
impl PyPgVectorRetriever {
    #[new]
    #[pyo3(signature = (
        pool,
        embedder,
        embedding_dimension = 1536,
        metric = "cosine",
        index_mode = "auto"
    ))]
    fn new(
        pool: PyRef<'_, PyPostgresPool>,
        embedder: PyRef<'_, PyOpenAiEmbedder>,
        embedding_dimension: usize,
        metric: &str,
        index_mode: &str,
    ) -> PyResult<Self> {
        let options = core::PgVectorRetrieverOptions::default()
            .with_embedding_dimension(embedding_dimension)
            .with_metric(parse_pgvector_metric(metric)?)
            .with_index_mode(parse_pgvector_index_mode(index_mode)?);
        Ok(Self {
            inner: Arc::new(core::PgVectorRetriever::new(
                pool.inner.clone(),
                embedder.inner.clone(),
                options,
            )),
        })
    }

    fn migrate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let retriever = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            retriever.migrate().await.map_err(map_error)
        })
    }

    fn index<'py>(
        &self,
        py: Python<'py>,
        chunks: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let retriever = Arc::clone(&self.inner);
        let chunks = extract_chunks(chunks)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            retriever.index(chunks).await.map_err(map_error)
        })
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let retriever = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunks = core::Retriever::retrieve(retriever.as_ref(), &query, top_k)
                .await
                .map_err(map_error)?;
            Ok(chunks
                .into_iter()
                .map(PyScoredChunk::from)
                .collect::<Vec<_>>())
        })
    }
}

#[pyclass(
    name = "KnowledgeAgent",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyKnowledgeAgent {
    inner: Arc<Mutex<core::KnowledgeAgent>>,
}

#[pymethods]
impl PyKnowledgeAgent {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        retriever,
        system = None,
        model = None,
        thinking = None,
        reasoning_effort = None,
        max_tokens = None,
        max_tool_rounds = None,
        top_k = 4,
        max_context_chars = 6000,
        fallback_message = None
    ))]
    fn from_env(
        retriever: &Bound<'_, PyAny>,
        system: Option<String>,
        model: Option<String>,
        thinking: Option<bool>,
        reasoning_effort: Option<String>,
        max_tokens: Option<u32>,
        max_tool_rounds: Option<usize>,
        top_k: usize,
        max_context_chars: usize,
        fallback_message: Option<String>,
    ) -> PyResult<Self> {
        let agent = build_agent_from_env(
            system,
            model,
            thinking,
            reasoning_effort,
            max_tokens,
            max_tool_rounds,
            None,
            None,
        )?;
        Self::from_core_agent(agent, retriever, top_k, max_context_chars, fallback_message)
    }

    #[staticmethod]
    #[pyo3(signature = (
        agent,
        retriever,
        top_k = 4,
        max_context_chars = 6000,
        fallback_message = None
    ))]
    fn from_agent(
        agent: PyRef<'_, PyAgent>,
        retriever: &Bound<'_, PyAny>,
        top_k: usize,
        max_context_chars: usize,
        fallback_message: Option<String>,
    ) -> PyResult<Self> {
        let agent = agent.take()?;
        Self::from_core_agent(agent, retriever, top_k, max_context_chars, fallback_message)
    }

    fn ask<'py>(&self, py: Python<'py>, question: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut agent = inner.lock().await;
            let response = agent.ask(question).await.map_err(map_error)?;
            PyKnowledgeAgentResponse::from_core(response)
        })
    }
}

impl PyKnowledgeAgent {
    fn from_core_agent(
        agent: core::Agent,
        retriever: &Bound<'_, PyAny>,
        top_k: usize,
        max_context_chars: usize,
        fallback_message: Option<String>,
    ) -> PyResult<Self> {
        let retriever = extract_retriever(retriever)?;
        let mut options = core::KnowledgeAgentOptions::new()
            .with_top_k(top_k)
            .with_max_context_chars(max_context_chars);
        if let Some(fallback_message) = fallback_message {
            options = options.with_fallback_message(fallback_message);
        }
        let agent = core::KnowledgeAgent::from_retriever(agent, retriever).with_options(options);
        Ok(Self {
            inner: Arc::new(Mutex::new(agent)),
        })
    }
}

#[pyclass(
    name = "KnowledgeAgentResponse",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyKnowledgeAgentResponse {
    content: String,
    used_fallback: bool,
    sources: Vec<PyKnowledgeSource>,
    retrieved_chunks: Vec<PyScoredChunk>,
    agent_response: Option<PyAgentResponse>,
}

#[pymethods]
impl PyKnowledgeAgentResponse {
    #[getter]
    fn content(&self) -> String {
        self.content.clone()
    }

    #[getter]
    fn used_fallback(&self) -> bool {
        self.used_fallback
    }

    #[getter]
    fn sources(&self) -> Vec<PyKnowledgeSource> {
        self.sources.clone()
    }

    #[getter]
    fn retrieved_chunks(&self) -> Vec<PyScoredChunk> {
        self.retrieved_chunks.clone()
    }

    #[getter]
    fn agent_response(&self) -> Option<PyAgentResponse> {
        self.agent_response.clone()
    }
}

impl PyKnowledgeAgentResponse {
    fn from_core(response: core::KnowledgeAgentResponse) -> PyResult<Self> {
        let content = response.content().to_owned();
        let used_fallback = response.used_fallback;
        let sources = response
            .sources
            .into_iter()
            .map(PyKnowledgeSource::from)
            .collect();
        let retrieved_chunks = response
            .retrieved_chunks
            .into_iter()
            .map(PyScoredChunk::from)
            .collect();
        let agent_response = response
            .agent_response
            .map(PyAgentResponse::from_core)
            .transpose()?;

        Ok(Self {
            content,
            used_fallback,
            sources,
            retrieved_chunks,
            agent_response,
        })
    }
}

#[pyclass(
    name = "KnowledgeSource",
    module = "arcone_agent._arcone_agent",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyKnowledgeSource {
    index: usize,
    chunk_id: String,
    document_id: String,
    title: Option<String>,
    source: Option<String>,
    path: Option<String>,
    score: f32,
}

#[pymethods]
impl PyKnowledgeSource {
    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn chunk_id(&self) -> String {
        self.chunk_id.clone()
    }

    #[getter]
    fn document_id(&self) -> String {
        self.document_id.clone()
    }

    #[getter]
    fn title(&self) -> Option<String> {
        self.title.clone()
    }

    #[getter]
    fn source(&self) -> Option<String> {
        self.source.clone()
    }

    #[getter]
    fn path(&self) -> Option<String> {
        self.path.clone()
    }

    #[getter]
    fn score(&self) -> f32 {
        self.score
    }
}

impl From<core::KnowledgeSource> for PyKnowledgeSource {
    fn from(source: core::KnowledgeSource) -> Self {
        Self {
            index: source.index,
            chunk_id: source.chunk_id.as_str().to_owned(),
            document_id: source.document_id.as_str().to_owned(),
            title: source.title,
            source: source.source,
            path: source.path,
            score: source.score,
        }
    }
}

#[pyfunction]
fn runtime_info() -> String {
    format!(
        "arcone-agent-py {} (pyo3 0.28, pyo3-async-runtimes 0.28)",
        env!("CARGO_PKG_VERSION")
    )
}

#[pymodule]
fn _arcone_agent(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(runtime_info, m)?)?;

    m.add("ArconeError", py.get_type::<ArconeError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    m.add("ApiError", py.get_type::<ApiError>())?;
    m.add("TimeoutError", py.get_type::<TimeoutError>())?;
    m.add("ToolError", py.get_type::<ToolError>())?;
    m.add(
        "StreamingUnsupportedError",
        py.get_type::<StreamingUnsupportedError>(),
    )?;
    m.add("SessionError", py.get_type::<SessionError>())?;
    m.add("DatabaseError", py.get_type::<DatabaseError>())?;
    m.add("KnowledgeError", py.get_type::<KnowledgeError>())?;
    m.add("RetrievalError", py.get_type::<RetrievalError>())?;

    m.add_class::<PyAgent>()?;
    m.add_class::<PyAgentStream>()?;
    m.add_class::<PyAgentResponse>()?;
    m.add_class::<PyInMemorySessionStore>()?;
    m.add_class::<PyPostgresSessionStore>()?;
    m.add_class::<PyPostgresPool>()?;
    m.add_class::<PyDocument>()?;
    m.add_class::<PyKnowledgeChunk>()?;
    m.add_class::<PyScoredChunk>()?;
    m.add_class::<PyInMemoryKnowledgeBase>()?;
    m.add_class::<PyPostgresKnowledgeBase>()?;
    m.add_class::<PyOpenAiEmbedder>()?;
    m.add_class::<PyInMemoryVectorRetriever>()?;
    m.add_class::<PyPgVectorRetriever>()?;
    m.add_class::<PyKnowledgeAgent>()?;
    m.add_class::<PyKnowledgeAgentResponse>()?;
    m.add_class::<PyKnowledgeSource>()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_agent_from_env(
    system: Option<String>,
    model: Option<String>,
    thinking: Option<bool>,
    reasoning_effort: Option<String>,
    max_tokens: Option<u32>,
    max_tool_rounds: Option<usize>,
    session_id: Option<String>,
    session_store: Option<&Bound<'_, PyAny>>,
) -> PyResult<core::Agent> {
    let client = core::DeepSeekClient::from_env().map_err(map_error)?;
    let mut agent = core::Agent::new(client);

    if let Some(system) = system {
        agent = agent.system(system);
    }
    if let Some(model) = model {
        agent = agent.model(parse_model(&model));
    }
    if let Some(thinking) = thinking {
        agent = if thinking {
            agent.thinking_enabled()
        } else {
            agent.thinking_disabled()
        };
    }
    if let Some(reasoning_effort) = reasoning_effort {
        agent = agent.reasoning(parse_reasoning_effort(&reasoning_effort)?);
    }
    if let Some(max_tokens) = max_tokens {
        agent = agent.max_tokens(max_tokens);
    }
    if let Some(max_tool_rounds) = max_tool_rounds {
        agent = agent.with_max_tool_rounds(max_tool_rounds);
    }

    match (session_id, session_store) {
        (Some(session_id), Some(store)) => {
            agent = agent.session(session_id, extract_session_store(store)?);
        }
        (Some(session_id), None) => {
            let store: Arc<dyn core::MemoryStore> = Arc::new(core::InMemorySessionStore::new());
            agent = agent.session(session_id, store);
        }
        (None, Some(_)) => {
            return Err(ConfigError::new_err(
                "session_id is required when session_store is provided",
            ));
        }
        (None, None) => {}
    }

    Ok(agent)
}

fn parse_model(model: &str) -> core::DeepSeekModel {
    model.parse().expect("DeepSeekModel parse is infallible")
}

fn parse_reasoning_effort(value: &str) -> PyResult<core::ReasoningEffort> {
    match value.trim().to_ascii_lowercase().as_str() {
        "high" => Ok(core::ReasoningEffort::High),
        "max" => Ok(core::ReasoningEffort::Max),
        other => Err(PyValueError::new_err(format!(
            "reasoning_effort must be 'high' or 'max', got {other:?}"
        ))),
    }
}

fn parse_pgvector_metric(value: &str) -> PyResult<core::PgVectorMetric> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cosine" => Ok(core::PgVectorMetric::Cosine),
        "l2" => Ok(core::PgVectorMetric::L2),
        other => Err(PyValueError::new_err(format!(
            "metric must be 'cosine' or 'l2', got {other:?}"
        ))),
    }
}

fn parse_pgvector_index_mode(value: &str) -> PyResult<core::PgVectorIndexMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(core::PgVectorIndexMode::Auto),
        "hnsw" => Ok(core::PgVectorIndexMode::Hnsw),
        "none" => Ok(core::PgVectorIndexMode::None),
        other => Err(PyValueError::new_err(format!(
            "index_mode must be 'auto', 'hnsw', or 'none', got {other:?}"
        ))),
    }
}

fn duration_from_seconds(seconds: f64, name: &str) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a positive finite number"
        )));
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn extract_session_store(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn core::MemoryStore>> {
    if let Ok(store) = obj.extract::<PyRef<'_, PyInMemorySessionStore>>() {
        let store: Arc<dyn core::MemoryStore> = store.inner.clone();
        return Ok(store);
    }
    if let Ok(store) = obj.extract::<PyRef<'_, PyPostgresSessionStore>>() {
        let store: Arc<dyn core::MemoryStore> = store.inner.clone();
        return Ok(store);
    }
    Err(PyTypeError::new_err(
        "session_store must be InMemorySessionStore or PostgresSessionStore",
    ))
}

fn extract_retriever(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn core::Retriever>> {
    if let Ok(retriever) = obj.extract::<PyRef<'_, PyInMemoryVectorRetriever>>() {
        let retriever: Arc<dyn core::Retriever> = retriever.inner.clone();
        return Ok(retriever);
    }
    if let Ok(retriever) = obj.extract::<PyRef<'_, PyPgVectorRetriever>>() {
        let retriever: Arc<dyn core::Retriever> = retriever.inner.clone();
        return Ok(retriever);
    }
    Err(PyTypeError::new_err(
        "retriever must be an InMemoryVectorRetriever or PgVectorRetriever",
    ))
}

fn extract_chunks(obj: &Bound<'_, PyAny>) -> PyResult<Vec<core::KnowledgeChunk>> {
    let list = obj.cast::<PyList>()?;
    let mut chunks = Vec::with_capacity(list.len());
    for item in list.iter() {
        let chunk = item.extract::<PyRef<'_, PyKnowledgeChunk>>()?;
        chunks.push(chunk.inner.clone());
    }
    Ok(chunks)
}

fn agent_moved_error() -> PyErr {
    ArconeError::new_err("agent has been moved into a KnowledgeAgent")
}

fn map_error(error: core::Error) -> PyErr {
    let message = error.to_string();
    match error {
        core::Error::MissingApiKey
        | core::Error::EmptyApiKey
        | core::Error::MissingOpenAiApiKey
        | core::Error::EmptyOpenAiApiKey
        | core::Error::MissingDatabaseUrl
        | core::Error::InvalidDatabaseUrl(_)
        | core::Error::StrictToolsRequireBetaBaseUrl => ConfigError::new_err(message),
        core::Error::Api { .. } | core::Error::OpenAiApi { .. } => ApiError::new_err(message),
        core::Error::Timeout { .. } | core::Error::OpenAiTimeout { .. } => {
            TimeoutError::new_err(message)
        }
        core::Error::UnknownTool(_)
        | core::Error::DuplicateTool(_)
        | core::Error::InvalidToolArguments { .. }
        | core::Error::ToolLoopExceeded { .. }
        | core::Error::ToolExecution { .. } => ToolError::new_err(message),
        core::Error::StreamingToolCallsUnsupported => StreamingUnsupportedError::new_err(message),
        core::Error::SessionNotFound(_) | core::Error::MemoryStore(_) => {
            SessionError::new_err(message)
        }
        core::Error::DatabasePool(_)
        | core::Error::DatabaseConnection(_)
        | core::Error::DatabaseQuery(_)
        | core::Error::DatabaseMigration(_) => DatabaseError::new_err(message),
        core::Error::DuplicateDocument(_)
        | core::Error::KnowledgeStore(_)
        | core::Error::KnowledgeIndexing(_) => KnowledgeError::new_err(message),
        core::Error::EmptyEmbeddingInput
        | core::Error::EmbeddingFailure(_)
        | core::Error::RetrievalFailure(_) => RetrievalError::new_err(message),
        _ => ArconeError::new_err(message),
    }
}

fn map_stream_worker_error(error: StreamWorkerError) -> PyErr {
    match error {
        StreamWorkerError::Core(error) => map_error(error),
        StreamWorkerError::AgentMoved => agent_moved_error(),
        StreamWorkerError::Conversion(message) => ArconeError::new_err(message),
    }
}

fn start_python_tool_call(
    py: Python<'_>,
    name: &str,
    handler: &Py<PyAny>,
    arguments: &Value,
) -> core::Result<PythonToolCall> {
    let arguments =
        value_to_py(py, arguments).map_err(|error| map_python_tool_error(py, name, error))?;
    let result = handler
        .bind(py)
        .call1((arguments,))
        .map_err(|error| map_python_tool_error(py, name, error))?;
    let is_awaitable = py
        .import("inspect")
        .and_then(|inspect| inspect.call_method1("isawaitable", (&result,)))
        .and_then(|value| value.extract::<bool>())
        .map_err(|error| map_python_tool_error(py, name, error))?;

    if is_awaitable {
        let future = pyo3_async_runtimes::tokio::into_future(result)
            .map_err(|error| map_python_tool_error(py, name, error))?;
        Ok(PythonToolCall::Pending(Box::pin(future)))
    } else {
        py_tool_output_to_json_string(py, name, &result).map(PythonToolCall::Ready)
    }
}

fn py_tool_output_to_json_string(
    py: Python<'_>,
    name: &str,
    output: &Bound<'_, PyAny>,
) -> core::Result<String> {
    let value = py_to_value(py, output).map_err(|error| map_python_tool_error(py, name, error))?;
    serde_json::to_string(&value).map_err(|error| core::Error::ToolExecution {
        name: name.to_owned(),
        message: error.to_string(),
    })
}

fn map_python_tool_error(py: Python<'_>, name: &str, error: PyErr) -> core::Error {
    core::Error::ToolExecution {
        name: name.to_owned(),
        message: format_python_error(py, &error),
    }
}

fn format_python_error(py: Python<'_>, error: &PyErr) -> String {
    let formatted = py.import("traceback").and_then(|traceback| {
        let traceback_value = error
            .traceback(py)
            .map(|traceback| traceback.into_any().unbind())
            .unwrap_or_else(|| py.None());
        let lines = traceback.call_method1(
            "format_exception",
            (error.get_type(py), error.value(py), traceback_value),
        )?;
        let lines = lines.extract::<Vec<String>>()?;
        Ok(lines.concat())
    });

    formatted
        .map(|message| message.trim().to_owned())
        .unwrap_or_else(|_| error.to_string())
}

fn finish_reason_string(reason: &core::FinishReason) -> String {
    match reason {
        core::FinishReason::Stop => "stop",
        core::FinishReason::Length => "length",
        core::FinishReason::ContentFilter => "content_filter",
        core::FinishReason::ToolCalls => "tool_calls",
        core::FinishReason::InsufficientSystemResource => "insufficient_system_resource",
        core::FinishReason::Unknown => "unknown",
    }
    .to_owned()
}

fn value_from_serialize<T: serde::Serialize>(value: T) -> PyResult<Value> {
    serde_json::to_value(value).map_err(|error| ArconeError::new_err(error.to_string()))
}

fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let text: String = json.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&text).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn option_value_to_py(py: Python<'_>, value: Option<&Value>) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) => value_to_py(py, value),
        None => Ok(py.None()),
    }
}

fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => value.into_py_any(py),
        Value::Number(value) => number_to_py(py, value),
        Value::String(value) => value.clone().into_py_any(py),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(value_to_py(py, value)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, value_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn number_to_py(py: Python<'_>, value: &Number) -> PyResult<Py<PyAny>> {
    if let Some(value) = value.as_i64() {
        return value.into_py_any(py);
    }
    if let Some(value) = value.as_u64() {
        return value.into_py_any(py);
    }
    if let Some(value) = value.as_f64() {
        return value.into_py_any(py);
    }
    Ok(py.None())
}

fn chunk_metadata_value(metadata: &core::ChunkMetadata) -> Value {
    let mut value = Map::new();
    insert_optional_string(&mut value, "title", metadata.title.clone());
    insert_optional_string(&mut value, "source", metadata.source.clone());
    insert_optional_string(&mut value, "path", metadata.path.clone());
    if let Some(extra) = metadata.extra.clone() {
        value.insert("extra".to_owned(), extra);
    }
    Value::Object(value)
}

fn insert_optional_string(map: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        map.insert(key.to_owned(), Value::String(value));
    }
}
