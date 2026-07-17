// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Media-URL correlation tagging.
//!
//! At dispatch every content-server media URL in an outgoing payload is tagged
//! with `?rid=<x_request_id>&mi=<ordinal>&td=<dispatch_wall_ns>`. The content
//! server records the query string verbatim, so a served transfer joins back to
//! the exact request and media slot that carried it, and carries its own
//! dispatch wall time for the `time_to_media_fetch` computation. Tagging walks
//! the whole JSON payload and rewrites any string that starts with this run's
//! content-server base, so it is agnostic to the endpoint dialect's media part
//! shape (Chat `image_url:{url}`, Responses `image_url:"<url>"`, Messages
//! `source:{url}`).

use serde_json::Value;

/// Tag every content-server media URL in `body` in place, assigning `mi` by
/// document walk order. Only strings beginning with `base` are rewritten, so
/// user-supplied external URLs are left untouched. Returns the number tagged.
pub fn tag_media_urls(body: &mut Value, base: &str, rid: &str, dispatch_wall_ns: u64) -> usize {
    let mut mi = 0u32;
    tag_walk(body, base, rid, dispatch_wall_ns, &mut mi);
    mi as usize
}

fn tag_walk(value: &mut Value, base: &str, rid: &str, td: u64, mi: &mut u32) {
    match value {
        Value::String(s) if s.starts_with(base) => {
            *s = append_tag(s, rid, *mi, td);
            *mi += 1;
        }
        Value::Array(items) => {
            for item in items {
                tag_walk(item, base, rid, td, mi);
            }
        }
        Value::Object(map) => {
            for value in map.values_mut() {
                tag_walk(value, base, rid, td, mi);
            }
        }
        _ => {}
    }
}

fn append_tag(url: &str, rid: &str, mi: u32, td: u64) -> String {
    let separator = if url.contains('?') { '&' } else { '?' };
    format!("{url}{separator}rid={rid}&mi={mi}&td={td}")
}

/// Correlation identity parsed back out of a served request's query string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaTag {
    pub rid: String,
    pub mi: u32,
    pub dispatch_wall_ns: u64,
}

/// Parse `rid`/`mi`/`td` out of a raw query string (no leading `?`). Returns
/// `None` if any of the three is absent or unparseable; unknown pairs are
/// ignored.
pub fn parse_media_tag(query_string: &str) -> Option<MediaTag> {
    let mut rid = None;
    let mut mi = None;
    let mut td = None;
    for pair in query_string.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        match key {
            "rid" => rid = Some(value.to_string()),
            "mi" => mi = value.parse().ok(),
            "td" => td = value.parse().ok(),
            _ => {}
        }
    }
    Some(MediaTag {
        rid: rid?,
        mi: mi?,
        dispatch_wall_ns: td?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BASE: &str = "http://127.0.0.1:8090";

    fn url_at(body: &Value, ptr: &str) -> String {
        body.pointer(ptr)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("no string at {ptr}"))
            .to_string()
    }

    #[test]
    fn tags_chat_nested_and_skips_external_and_data() {
        let mut body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8090/content/images/img_000001.png"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/external.jpg"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]}]
        });
        let n = tag_media_urls(&mut body, BASE, "req-1", 1000);
        assert_eq!(n, 1);
        assert_eq!(
            url_at(&body, "/messages/0/content/1/image_url/url"),
            "http://127.0.0.1:8090/content/images/img_000001.png?rid=req-1&mi=0&td=1000"
        );
        // External and data URIs untouched.
        assert_eq!(
            url_at(&body, "/messages/0/content/2/image_url/url"),
            "https://example.com/external.jpg"
        );
        assert_eq!(
            url_at(&body, "/messages/0/content/3/image_url/url"),
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn assigns_distinct_mi_across_multiple_media_in_one_payload() {
        let mut body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8090/content/images/a.png"}},
                {"type": "video_url", "video_url": {"url": "http://127.0.0.1:8090/content/video/b.mp4"}},
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8090/content/images/c.png"}},
            ]}]
        });
        let n = tag_media_urls(&mut body, BASE, "req-2", 42);
        assert_eq!(n, 3);
        assert!(
            url_at(&body, "/messages/0/content/0/image_url/url")
                .ends_with("a.png?rid=req-2&mi=0&td=42")
        );
        assert!(
            url_at(&body, "/messages/0/content/1/video_url/url")
                .ends_with("b.mp4?rid=req-2&mi=1&td=42")
        );
        assert!(
            url_at(&body, "/messages/0/content/2/image_url/url")
                .ends_with("c.png?rid=req-2&mi=2&td=42")
        );
    }

    #[test]
    fn tags_responses_string_and_messages_source_shapes() {
        // Responses: image_url is a bare string.
        let mut responses = json!({"input": [{"type": "input_image", "image_url": "http://127.0.0.1:8090/content/images/r.png"}]});
        assert_eq!(tag_media_urls(&mut responses, BASE, "r", 7), 1);
        assert!(url_at(&responses, "/input/0/image_url").ends_with("r.png?rid=r&mi=0&td=7"));

        // Messages (Anthropic): source.url.
        let mut messages = json!({"messages": [{"content": [{"type": "image", "source": {"type": "url", "url": "http://127.0.0.1:8090/content/images/m.png"}}]}]});
        assert_eq!(tag_media_urls(&mut messages, BASE, "m", 9), 1);
        assert!(
            url_at(&messages, "/messages/0/content/0/source/url")
                .ends_with("m.png?rid=m&mi=0&td=9")
        );
    }

    #[test]
    fn appends_with_ampersand_when_url_already_has_query() {
        let mut body = json!({"u": "http://127.0.0.1:8090/content/images/x.png?v=2"});
        tag_media_urls(&mut body, BASE, "req", 5);
        assert_eq!(
            url_at(&body, "/u"),
            "http://127.0.0.1:8090/content/images/x.png?v=2&rid=req&mi=0&td=5"
        );
    }

    #[test]
    fn parse_round_trips_and_requires_all_three() {
        let tag = parse_media_tag("rid=abc-123&mi=2&td=1752700000000000000").unwrap();
        assert_eq!(
            tag,
            MediaTag {
                rid: "abc-123".into(),
                mi: 2,
                dispatch_wall_ns: 1_752_700_000_000_000_000
            }
        );
        // Unknown pairs ignored.
        assert!(parse_media_tag("foo=bar&rid=a&mi=0&td=1&extra=z").is_some());
        // Missing td.
        assert!(parse_media_tag("rid=a&mi=0").is_none());
        // Unparseable mi.
        assert!(parse_media_tag("rid=a&mi=x&td=1").is_none());
    }
}
