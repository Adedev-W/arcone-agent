use std::{convert::Infallible, net::SocketAddr, time::Instant};

use arcone_agent::{Agent, DeepSeekClient, DeepSeekConfig, Usage};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Frame, Incoming},
    header::{CONTENT_TYPE, HeaderValue},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;

type HttpBody = BoxBody<Bytes, Infallible>;
type HttpResponse = Response<HttpBody>;

#[derive(Clone)]
struct AppState {
    client: DeepSeekClient,
    default_max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatPayload {
    prompt: String,
    max_tokens: Option<u32>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    let address = std::env::var("STREAM_SERVER_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_owned())
        .parse::<SocketAddr>()?;
    let default_max_tokens = read_env_u32("MAX_TOKENS").unwrap_or(256);

    let config = DeepSeekConfig::from_env()?
        .with_default_thinking(None)
        .with_default_reasoning_effort(None);

    let state = AppState {
        client: DeepSeekClient::new(config)?,
        default_max_tokens,
    };
    let listener = TcpListener::bind(address).await?;

    eprintln!("deepseek stream server listening on http://{address}");
    eprintln!("client is initialized once and reused for every /chat request");
    eprintln!(
        "try: curl -N -X POST http://{address}/chat -H 'content-type: application/json' -d '{{\"prompt\":\"Jelaskan arcone-agent singkat\"}}'"
    );

    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();

        tokio::spawn(async move {
            let service = service_fn(move |request| handle_request(request, state.clone()));

            if let Err(error) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                eprintln!("connection error: {error}");
            }
        });
    }
}

async fn handle_request(
    request: Request<Incoming>,
    state: AppState,
) -> Result<HttpResponse, Infallible> {
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/") => Ok(text_response(
            StatusCode::OK,
            "POST /chat with JSON body: {\"prompt\":\"...\",\"max_tokens\":256}\n",
        )),
        (&Method::GET, "/health") => Ok(text_response(StatusCode::OK, "ok\n")),
        (&Method::POST, "/chat") => handle_chat(request, state).await,
        _ => Ok(text_response(StatusCode::NOT_FOUND, "not found\n")),
    }
}

async fn handle_chat(
    request: Request<Incoming>,
    state: AppState,
) -> Result<HttpResponse, Infallible> {
    let body = match request.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(error) => {
            return Ok(text_response(
                StatusCode::BAD_REQUEST,
                format!("failed to read request body: {error}\n"),
            ));
        }
    };
    let payload = match serde_json::from_slice::<ChatPayload>(&body) {
        Ok(payload) if !payload.prompt.trim().is_empty() => payload,
        Ok(_) => {
            return Ok(text_response(
                StatusCode::BAD_REQUEST,
                "prompt cannot be empty\n",
            ));
        }
        Err(error) => {
            return Ok(text_response(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON body: {error}\n"),
            ));
        }
    };

    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, Infallible>>(16);
    tokio::spawn(stream_deepseek_response(state, payload, tx));

    let mut response = Response::new(
        StreamBody::new(ReceiverStream::new(rx))
            .map_err(|never| match never {})
            .boxed(),
    );
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );

    Ok(response)
}

async fn stream_deepseek_response(
    state: AppState,
    payload: ChatPayload,
    tx: mpsc::Sender<Result<Frame<Bytes>, Infallible>>,
) {
    let started_at = Instant::now();
    let max_tokens = payload.max_tokens.unwrap_or(state.default_max_tokens);
    let mut agent = Agent::new(state.client.clone())
        .system("Gunakan bahasa indonesia yang baik, sopan, dan ringkas.")
        .thinking_disabled()
        .max_tokens(max_tokens);

    let mut stream = match agent.stream(payload.prompt).await {
        Ok(stream) => stream,
        Err(error) => {
            let _ = send_text(&tx, format!("DeepSeek request failed: {error}\n")).await;
            eprintln!(
                "deepseek_request_failed total_ms={} error={error}",
                started_at.elapsed().as_millis()
            );
            return;
        }
    };

    let mut first_token_at = None;

    loop {
        match stream.next_text().await {
            Ok(Some(content)) => {
                first_token_at.get_or_insert_with(Instant::now);

                if !send_text(&tx, content).await {
                    return;
                }
            }
            Ok(None) => break,
            Err(error) => {
                let _ = send_text(&tx, format!("\nstream error: {error}\n")).await;
                eprintln!(
                    "deepseek_stream_error total_ms={} error={error}",
                    started_at.elapsed().as_millis()
                );
                return;
            }
        }
    }

    let response = match stream.finish().await {
        Ok(response) => response,
        Err(error) => {
            let _ = send_text(&tx, format!("\nstream finish error: {error}\n")).await;
            eprintln!(
                "deepseek_stream_finish_error total_ms={} error={error}",
                started_at.elapsed().as_millis()
            );
            return;
        }
    };
    let total_elapsed = started_at.elapsed();
    let time_to_first_token = first_token_at.map(|instant| instant.duration_since(started_at));
    let metrics = format_metrics(
        max_tokens,
        time_to_first_token,
        total_elapsed,
        response.usage.as_ref(),
    );

    eprintln!("{metrics}");

    let _ = send_text(&tx, format!("\n\n{metrics}\n")).await;
}

async fn send_text(
    tx: &mpsc::Sender<Result<Frame<Bytes>, Infallible>>,
    text: impl Into<String>,
) -> bool {
    tx.send(Ok(Frame::data(Bytes::from(text.into()))))
        .await
        .is_ok()
}

fn format_metrics(
    max_tokens: u32,
    time_to_first_token: Option<std::time::Duration>,
    total_elapsed: std::time::Duration,
    usage: Option<&Usage>,
) -> String {
    let first_token = time_to_first_token
        .map(|elapsed| {
            format!(
                "{} ms ({:.3} s)",
                elapsed.as_millis(),
                elapsed.as_secs_f64()
            )
        })
        .unwrap_or_else(|| "n/a".to_owned());
    let usage = usage
        .map(|usage| {
            format!(
                "prompt_tokens={} completion_tokens={} total_tokens={}",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            )
        })
        .unwrap_or_else(|| "usage=n/a".to_owned());

    format!(
        "metrics max_tokens={} time_to_first_token={} total={} ms ({:.3} s) {}",
        max_tokens,
        first_token,
        total_elapsed.as_millis(),
        total_elapsed.as_secs_f64(),
        usage
    )
}

fn read_env_u32(key: &str) -> Option<u32> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
}

fn text_response(status: StatusCode, body: impl Into<String>) -> HttpResponse {
    let mut response = Response::new(
        Full::new(Bytes::from(body.into()))
            .map_err(|never| match never {})
            .boxed(),
    );
    *response.status_mut() = status;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}
