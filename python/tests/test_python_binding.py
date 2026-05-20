import os
import asyncio

import pytest

from arcone_agent import (
    Agent,
    ConfigError,
    Document,
    InMemoryKnowledgeBase,
    InMemorySessionStore,
    PostgresPool,
    StreamingUnsupportedError,
    ToolError,
    runtime_info,
)


def test_import_and_runtime_info():
    assert "arcone-agent-py" in runtime_info()


def test_missing_deepseek_key_maps_to_config_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        Agent.from_env()


def test_document_properties():
    document = Document.text(
        "doc-1",
        "hello",
        title="Title",
        source="unit",
        path="docs/doc-1.md",
        metadata={"tenant": "test"},
    )
    assert document.id == "doc-1"
    assert document.content == "hello"
    assert document.title == "Title"
    assert document.metadata == {"tenant": "test"}


@pytest.mark.asyncio
async def test_in_memory_knowledge_base_without_providers():
    knowledge = InMemoryKnowledgeBase(max_chars=20, overlap_chars=5)
    chunks = await knowledge.add_document(Document.text("doc-1", "hello world from arcone"))
    assert chunks
    assert chunks[0].document_id == "doc-1"

    documents = await knowledge.list_documents()
    assert [document.id for document in documents] == ["doc-1"]

    by_document = await knowledge.chunks_for_document("doc-1")
    assert len(by_document) == len(chunks)

    assert await knowledge.remove_document("doc-1") is True


def test_in_memory_session_store_constructs():
    assert InMemorySessionStore() is not None


def test_python_tool_registration_and_duplicate_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    agent = Agent.from_env()
    schema = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    }

    agent.add_tool("lookup_price", "Return a demo quote.", schema, lambda args: {"ok": args})

    with pytest.raises(ToolError):
        agent.add_tool("lookup_price", "Duplicate.", schema, lambda args: args)


def test_python_tool_invalid_schema_maps_to_tool_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    agent = Agent.from_env()

    with pytest.raises(ToolError):
        agent.add_tool("bad_schema", "Bad schema.", object(), lambda args: args)


@pytest.mark.asyncio
async def test_postgres_pool_missing_database_url_maps_to_config_error(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ConfigError):
        await PostgresPool.from_env()


@pytest.mark.asyncio
async def test_stream_text_yields_deltas_and_finish(monkeypatch):
    stream_body = (
        'data: {"id":"stream-python","choices":[{"delta":{"role":"assistant",'
        '"reasoning_content":"thinking"},"finish_reason":null,"index":0,'
        '"logprobs":null}],"created":1,"model":"deepseek-v4-flash",'
        '"object":"chat.completion.chunk","usage":null}\n\n'
        'data: {"id":"stream-python","choices":[{"delta":{"content":"Hello"},'
        '"finish_reason":null,"index":0,"logprobs":null}],"created":1,'
        '"model":"deepseek-v4-flash","object":"chat.completion.chunk",'
        '"usage":null}\n\n'
        'data: {"id":"stream-python","choices":[{"delta":{"content":" world"},'
        '"finish_reason":"stop","index":0,"logprobs":null}],"created":1,'
        '"model":"deepseek-v4-flash","object":"chat.completion.chunk",'
        '"usage":{"completion_tokens":1,"prompt_tokens":1,"total_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    server, requests, base_url = await _start_http_sequence([_stream_response(stream_body)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", base_url)

    try:
        agent = Agent.from_env(thinking=False, max_tokens=64)
        stream = agent.stream_text("Say hello")

        deltas = []
        async for delta in stream:
            deltas.append(delta)

        response = await stream.finish()

        assert deltas == ["Hello", " world"]
        assert response.content == "Hello world"
        assert response.reasoning_content == "thinking"
        assert response.finish_reason == "stop"
        assert response.usage["completion_tokens"] == 1
        assert response.usage["prompt_tokens"] == 1
        assert response.usage["total_tokens"] == 2
        assert '"stream":true' in requests[0]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_stream_advanced_finish_drains_response(monkeypatch):
    stream_body = (
        'data: {"id":"stream-python","choices":[{"delta":{"content":"Done"},'
        '"finish_reason":"stop","index":0,"logprobs":null}],"created":1,'
        '"model":"deepseek-v4-flash","object":"chat.completion.chunk",'
        '"usage":null}\n\n'
        "data: [DONE]\n\n"
    )
    server, requests, base_url = await _start_http_sequence([_stream_response(stream_body)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", base_url)

    try:
        agent = Agent.from_env(thinking=False)
        stream = await agent.stream("Finish without manual iteration")
        response = await stream.finish()

        assert response.content == "Done"
        assert await stream.finish() is not None
        assert '"stream":true' in requests[0]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_streaming_tool_call_maps_to_specific_error(monkeypatch):
    stream_body = (
        'data: {"id":"stream-tool","choices":[{"delta":{"tool_calls":[{"id":"call_1",'
        '"type":"function","function":{"name":"lookup","arguments":"{}"}}]},'
        '"finish_reason":null,"index":0,"logprobs":null}],"created":1,'
        '"model":"deepseek-v4-flash","object":"chat.completion.chunk",'
        '"usage":null}\n\n'
        "data: [DONE]\n\n"
    )
    server, requests, base_url = await _start_http_sequence([_stream_response(stream_body)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", base_url)

    try:
        agent = Agent.from_env(thinking=False)
        stream = await agent.stream("Use a tool")

        with pytest.raises(StreamingUnsupportedError):
            await stream.__anext__()

        assert '"stream":true' in requests[0]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_live_agent_ask_text_when_configured():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is not configured")

    agent = Agent.from_env(system="Answer in one short sentence.", thinking=False, max_tokens=64)
    text = await agent.ask_text("What is arcone-agent?")
    assert isinstance(text, str)
    assert text


async def _start_http_sequence(responses):
    responses = list(responses)
    requests = []

    async def handle(reader, writer):
        request = await _read_http_request(reader)
        requests.append(request)
        response = responses.pop(0)
        writer.write(response.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()
    return server, requests, f"http://{host}:{port}"


async def _read_http_request(reader):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await reader.read(1024)
        if not chunk:
            break
        data += chunk

    headers, _, body = data.partition(b"\r\n\r\n")
    content_length = 0
    for line in headers.decode(errors="replace").splitlines():
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            content_length = int(value.strip())
            break

    while len(body) < content_length:
        body += await reader.read(content_length - len(body))

    return (headers + b"\r\n\r\n" + body).decode(errors="replace")


def _stream_response(body):
    return (
        "HTTP/1.1 200 OK\r\n"
        "content-type: text/event-stream\r\n"
        f"content-length: {len(body)}\r\n"
        "connection: close\r\n"
        "\r\n"
        f"{body}"
    )
