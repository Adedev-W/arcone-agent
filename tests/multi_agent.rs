use std::time::Duration;

use arcone_agent::{
    Agent, AgentConfig, AgentProfile, AgentTeam, DeepSeekClient, DeepSeekConfig, Error, Handoff,
    LlmRouter, RouteDecision, RouteFuture, RouteRequest, StaticRouter, TeamRouter,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

#[tokio::test]
async fn static_router_selects_agent_from_team() {
    let body = assistant_response("chat-static", "research answer");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let client = test_client(base_url);
    let researcher = test_agent(
        client.clone(),
        AgentProfile::new("researcher")
            .with_name("Researcher")
            .with_role_description("Finds facts")
            .with_system_prompt("Research carefully."),
    );
    let writer = test_agent(
        client,
        AgentProfile::new("writer")
            .with_name("Writer")
            .with_role_description("Writes final prose")
            .with_system_prompt("Write clearly."),
    );
    let mut team = AgentTeam::new()
        .with_agent(researcher)
        .expect("researcher")
        .with_agent(writer)
        .expect("writer")
        .with_router(StaticRouter::new("researcher").with_reason("default route"));

    let response = team.ask("Find facts").await.expect("team response");

    assert_eq!(response.agent_id.as_str(), "researcher");
    assert_eq!(response.route_reason.as_deref(), Some("default route"));
    assert_eq!(response.content(), Some("research answer"));
    assert!(response.handoffs.is_empty());

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
    let request = json_request_body(&requests[0]);
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][0]["content"], "Research carefully.");
    assert_eq!(request["messages"][1]["content"], "Find facts");
}

#[tokio::test]
async fn manual_route_bypasses_router() {
    let body = assistant_response("chat-manual", "draft answer");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let client = test_client(base_url);
    let mut team = AgentTeam::new()
        .with_agent(test_agent(
            client.clone(),
            AgentProfile::new("researcher").with_system_prompt("Research."),
        ))
        .expect("researcher")
        .with_agent(test_agent(
            client,
            AgentProfile::new("writer").with_system_prompt("Write."),
        ))
        .expect("writer");

    let response = team
        .ask_with_agent("writer", "Draft this")
        .await
        .expect("manual response");

    assert_eq!(response.agent_id.as_str(), "writer");
    assert_eq!(response.content(), Some("draft answer"));
    assert!(response.route_reason.is_none());

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
    let request = json_request_body(&requests[0]);
    assert_eq!(request["messages"][0]["content"], "Write.");
}

#[tokio::test]
async fn llm_router_selects_agent_from_structured_output() {
    let route_json = json!({
        "agent_id": "planner",
        "reason": "planning is required"
    })
    .to_string();
    let route_body = assistant_response("chat-route", &route_json);
    let agent_body = assistant_response("chat-planner", "planned answer");
    let (base_url, server) =
        spawn_http_sequence(vec![json_response(&route_body), json_response(&agent_body)]).await;
    let client = test_client(base_url);
    let planner = test_agent(
        client.clone(),
        AgentProfile::new("planner")
            .with_name("Planner")
            .with_role_description("Breaks work into steps")
            .with_system_prompt("Plan carefully."),
    );
    let writer = test_agent(
        client.clone(),
        AgentProfile::new("writer")
            .with_name("Writer")
            .with_role_description("Writes polished prose")
            .with_system_prompt("Write clearly."),
    );
    let mut team = AgentTeam::new()
        .with_agent(planner)
        .expect("planner")
        .with_agent(writer)
        .expect("writer")
        .with_router(LlmRouter::new(client));

    let response = team.ask("Build a roadmap").await.expect("team response");

    assert_eq!(response.agent_id.as_str(), "planner");
    assert_eq!(
        response.route_reason.as_deref(),
        Some("planning is required")
    );
    assert_eq!(response.content(), Some("planned answer"));

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 2);

    let router_request = json_request_body(&requests[0]);
    assert_eq!(
        router_request["response_format"]["type"],
        json!("json_object")
    );
    assert!(
        router_request["messages"][1]["content"]
            .as_str()
            .expect("router prompt")
            .contains("planner")
    );

    let agent_request = json_request_body(&requests[1]);
    assert_eq!(agent_request["messages"][0]["content"], "Plan carefully.");
}

#[tokio::test]
async fn router_handoff_is_recorded_before_final_agent_call() {
    let body = assistant_response("chat-handoff", "writer answer");
    let (base_url, server) = spawn_http_sequence(vec![json_response(&body)]).await;
    let client = test_client(base_url);
    let mut team = AgentTeam::new()
        .with_agent(test_agent(
            client.clone(),
            AgentProfile::new("researcher").with_system_prompt("Research."),
        ))
        .expect("researcher")
        .with_agent(test_agent(
            client,
            AgentProfile::new("writer").with_system_prompt("Write."),
        ))
        .expect("writer")
        .with_router(HandoffOnceRouter);

    let response = team.ask("Need a final draft").await.expect("team response");

    assert_eq!(response.agent_id.as_str(), "writer");
    assert_eq!(response.route_reason.as_deref(), Some("after handoff"));
    assert_eq!(response.handoffs.len(), 1);
    assert_eq!(response.handoffs[0].from.as_str(), "researcher");
    assert_eq!(response.handoffs[0].to.as_str(), "writer");
    assert_eq!(response.content(), Some("writer answer"));

    let requests = server.await.expect("server task");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn handoff_loop_is_limited() {
    let client = test_client("http://127.0.0.1:9".to_owned());
    let mut team = AgentTeam::new()
        .with_agent(test_agent(
            client.clone(),
            AgentProfile::new("researcher").with_system_prompt("Research."),
        ))
        .expect("researcher")
        .with_agent(test_agent(
            client,
            AgentProfile::new("writer").with_system_prompt("Write."),
        ))
        .expect("writer")
        .with_max_handoff_rounds(1)
        .with_router(AlwaysHandoffRouter);

    let error = team.ask("Loop").await.expect_err("handoff loop");

    assert!(matches!(
        error,
        Error::HandoffLoopExceeded { max_rounds: 1 }
    ));
}

#[tokio::test]
async fn missing_router_is_routing_failure() {
    let client = test_client("http://127.0.0.1:9".to_owned());
    let mut team = AgentTeam::new()
        .with_agent(test_agent(
            client,
            AgentProfile::new("researcher").with_system_prompt("Research."),
        ))
        .expect("researcher");

    let error = team.ask("No router").await.expect_err("missing router");

    assert!(matches!(error, Error::RoutingFailure(message) if message.contains("no router")));
}

#[tokio::test]
async fn unknown_route_target_is_typed_error() {
    let client = test_client("http://127.0.0.1:9".to_owned());
    let mut team = AgentTeam::new()
        .with_agent(test_agent(
            client,
            AgentProfile::new("researcher").with_system_prompt("Research."),
        ))
        .expect("researcher")
        .with_router(StaticRouter::new("missing"));

    let error = team.ask("Unknown").await.expect_err("unknown agent");

    assert!(matches!(error, Error::UnknownAgent(agent) if agent == "missing"));
}

#[test]
fn duplicate_agent_id_is_rejected() {
    let client = test_client("http://127.0.0.1:9".to_owned());
    let mut team = AgentTeam::new();
    team.add_agent(test_agent(
        client.clone(),
        AgentProfile::new("researcher").with_system_prompt("Research."),
    ))
    .expect("first agent");

    let error = match team.add_agent(test_agent(
        client,
        AgentProfile::new("researcher").with_system_prompt("Research again."),
    )) {
        Ok(_) => panic!("duplicate agent should fail"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::DuplicateAgent(agent) if agent == "researcher"));
}

struct HandoffOnceRouter;

impl TeamRouter for HandoffOnceRouter {
    fn route(&self, request: RouteRequest) -> RouteFuture {
        Box::pin(async move {
            if request.handoffs.is_empty() {
                return Ok(RouteDecision::Handoff(
                    Handoff::new("researcher", "writer").with_reason("writer should finish"),
                ));
            }

            Ok(RouteDecision::agent_with_reason("writer", "after handoff"))
        })
    }
}

struct AlwaysHandoffRouter;

impl TeamRouter for AlwaysHandoffRouter {
    fn route(&self, _request: RouteRequest) -> RouteFuture {
        Box::pin(async { Ok(RouteDecision::Handoff(Handoff::new("researcher", "writer"))) })
    }
}

fn test_agent(client: DeepSeekClient, profile: AgentProfile) -> Agent {
    Agent::with_config(
        client,
        AgentConfig::new("placeholder").with_profile(profile),
    )
}

fn test_client(base_url: String) -> DeepSeekClient {
    DeepSeekClient::new(
        DeepSeekConfig::new("test-key")
            .with_base_url(base_url)
            .with_timeout(Duration::from_secs(5))
            .with_default_thinking(None)
            .with_default_reasoning_effort(None),
    )
    .expect("client")
}

fn assistant_response(id: &str, content: &str) -> String {
    json!({
        "id": id,
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "logprobs": null
        }],
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion",
        "usage": usage()
    })
    .to_string()
}

fn usage() -> Value {
    json!({
        "completion_tokens": 1,
        "prompt_tokens": 1,
        "total_tokens": 2
    })
}

async fn spawn_http_sequence(responses: Vec<String>) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        let mut requests = Vec::new();

        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("accept connection");
            requests.push(read_http_request(&mut socket).await);
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        }

        requests
    });

    (format!("http://{address}"), handle)
}

async fn read_http_request(socket: &mut TcpStream) -> String {
    let mut data = Vec::new();
    let mut buffer = [0; 1024];

    loop {
        let bytes_read = socket.read(&mut buffer).await.expect("read request");
        if bytes_read == 0 {
            break;
        }

        data.extend_from_slice(&buffer[..bytes_read]);

        if let Some(header_end) = header_end(&data) {
            let content_length = content_length(&data[..header_end + 4]);
            if data.len() >= header_end + 4 + content_length {
                break;
            }
        }
    }

    String::from_utf8_lossy(&data).into_owned()
}

fn header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(headers: &[u8]) -> usize {
    let headers = String::from_utf8_lossy(headers);

    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().expect("content-length"))
        })
        .unwrap_or(0)
}

fn json_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn json_request_body(request: &str) -> Value {
    let (_, body) = request.split_once("\r\n\r\n").expect("request body");

    serde_json::from_str(body).expect("json request body")
}
