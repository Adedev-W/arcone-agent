use std::collections::VecDeque;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::Incoming;

use crate::{Error, Result, types::ChatStreamChunk};

#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    Chunk(ChatStreamChunk),
    Done,
}

#[derive(Debug)]
pub struct ChatStream {
    body: Incoming,
    decoder: SseDecoder,
    pending: VecDeque<StreamEvent>,
    done: bool,
}

impl ChatStream {
    pub(crate) fn new(body: Incoming) -> Self {
        Self {
            body,
            decoder: SseDecoder::new(),
            pending: VecDeque::new(),
            done: false,
        }
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>> {
        if let Some(event) = self.pending.pop_front() {
            return Ok(Some(self.mark_done(event)));
        }

        if self.done {
            return Ok(None);
        }

        while let Some(frame) = self.body.frame().await {
            let frame = frame?;

            let Ok(data) = frame.into_data() else {
                continue;
            };

            for event in self.decoder.push(data)? {
                self.pending.push_back(event);
            }

            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(self.mark_done(event)));
            }
        }

        if self.decoder.has_partial_event() {
            return Err(Error::PartialSseEvent);
        }

        self.done = true;
        Ok(None)
    }

    fn mark_done(&mut self, event: StreamEvent) -> StreamEvent {
        if matches!(event, StreamEvent::Done) {
            self.done = true;
        }

        event
    }
}

#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(&mut self, bytes: Bytes) -> Result<Vec<StreamEvent>> {
        self.buffer.extend_from_slice(&bytes);
        let mut events = Vec::new();

        while let Some((position, delimiter_len)) = find_event_delimiter(&self.buffer) {
            let event = self.buffer[..position].to_vec();
            self.buffer.drain(..position + delimiter_len);

            if let Some(parsed) = parse_event(&event)? {
                events.push(parsed);
            }
        }

        Ok(events)
    }

    pub(crate) fn has_partial_event(&self) -> bool {
        self.buffer.iter().any(|byte| !byte.is_ascii_whitespace())
    }
}

fn find_event_delimiter(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4));

    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn parse_event(event: &[u8]) -> Result<Option<StreamEvent>> {
    let text = std::str::from_utf8(event)
        .map_err(|error| Error::InvalidSse(format!("event is not valid UTF-8: {error}")))?;
    let mut data_lines = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim_end_matches('\r');

        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    if data_lines.is_empty() {
        return Ok(None);
    }

    let data = data_lines.join("\n");
    if data == "[DONE]" {
        return Ok(Some(StreamEvent::Done));
    }

    Ok(Some(StreamEvent::Chunk(serde_json::from_str(&data)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_sse_chunks_keep_alive_and_done() {
        let mut decoder = SseDecoder::new();

        let first = decoder
            .push(Bytes::from_static(b": keep-alive\n\ndata: {\"id\":\"1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"think\",\"role\":\"assistant\"},\"finish_reason\":null,\"index\":0,\"logprobs\":null}],\"created\":1,\"model\":\"deepseek-v4-pro\",\"object\":\"chat.completion.chunk\",\"usage\":null}\n"))
            .expect("first half");
        assert!(first.is_empty());

        let second = decoder
            .push(Bytes::from_static(b"\ndata: [DONE]\n\n"))
            .expect("second half");

        assert_eq!(second.len(), 2);
        let StreamEvent::Chunk(chunk) = &second[0] else {
            panic!("expected chunk");
        };
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("think")
        );
        assert!(matches!(second[1], StreamEvent::Done));
    }

    #[test]
    fn reports_partial_non_whitespace_event() {
        let mut decoder = SseDecoder::new();
        let events = decoder
            .push(Bytes::from_static(b"data: {\"not_done\": true}"))
            .expect("push");

        assert!(events.is_empty());
        assert!(decoder.has_partial_event());
    }

    #[test]
    fn malformed_json_is_error() {
        let mut decoder = SseDecoder::new();
        let error = decoder
            .push(Bytes::from_static(b"data: {bad json}\n\n"))
            .expect_err("malformed json");

        assert!(matches!(error, Error::Json(_)));
    }
}
