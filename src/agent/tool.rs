use std::{collections::BTreeMap, future::Future, marker::PhantomData, pin::Pin, sync::Arc};

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Error, FunctionDefinition, Result, ToolDefinition};

pub type ToolFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn call(&self, arguments: Value) -> ToolFuture;
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tool<T>(&mut self, tool: T) -> Result<&mut Self>
    where
        T: Tool + 'static,
    {
        self.add_shared_tool(Arc::new(tool))
    }

    pub fn get(&self, name: impl AsRef<str>) -> Option<Arc<dyn Tool>> {
        self.tools.get(name.as_ref()).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition())
            .collect::<Vec<_>>()
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    fn add_shared_tool(&mut self, tool: Arc<dyn Tool>) -> Result<&mut Self> {
        let definition = tool.definition();
        let name = definition.function.name;

        if self.tools.contains_key(&name) {
            return Err(Error::DuplicateTool(name));
        }

        self.tools.insert(name, tool);
        Ok(self)
    }
}

pub struct FunctionTool<F> {
    definition: ToolDefinition,
    handler: F,
}

impl<F> FunctionTool<F> {
    pub fn new(definition: ToolDefinition, handler: F) -> Self {
        Self {
            definition,
            handler,
        }
    }
}

impl<F, Fut> Tool for FunctionTool<F>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String>> + Send + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn call(&self, arguments: Value) -> ToolFuture {
        Box::pin((self.handler)(arguments))
    }
}

pub struct TypedFunctionTool<Args, Output, F> {
    definition: ToolDefinition,
    handler: F,
    marker: PhantomData<fn(Args) -> Output>,
}

impl<Args, Output, F> TypedFunctionTool<Args, Output, F> {
    pub fn json(name: impl Into<String>, description: impl Into<String>, handler: F) -> Result<Self>
    where
        Args: JsonSchema,
    {
        let parameters = serde_json::to_value(schemars::schema_for!(Args))?;
        let definition = ToolDefinition::function(
            FunctionDefinition::new(name)
                .description(description)
                .parameters(parameters),
        );

        Ok(Self {
            definition,
            handler,
            marker: PhantomData,
        })
    }
}

impl<Args, Output, F, Fut> Tool for TypedFunctionTool<Args, Output, F>
where
    Args: DeserializeOwned + JsonSchema + Send + 'static,
    Output: Serialize + Send + 'static,
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Output>> + Send + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn call(&self, arguments: Value) -> ToolFuture {
        let name = self.definition.function.name.clone();
        let arguments = match serde_json::from_value(arguments) {
            Ok(arguments) => arguments,
            Err(source) => {
                return Box::pin(async move { Err(Error::InvalidToolArguments { name, source }) });
            }
        };
        let future = (self.handler)(arguments);

        Box::pin(async move {
            let output = future.await?;
            Ok(serde_json::to_string(&output)?)
        })
    }
}
