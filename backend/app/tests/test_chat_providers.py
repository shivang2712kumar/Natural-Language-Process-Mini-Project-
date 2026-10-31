import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.core import config


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.headers.get("Authorization") == "Bearer bad":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Incorrect API key provided"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in ("Hel", "lo"):
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": w}}]}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def fake_server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    yield srv
    srv.shutdown()


def test_env_parser_strips_inline_comments():
    assert config._parse_value("https://x/v1   # comment") == "https://x/v1"
    assert config._parse_value('"a # b"  # c') == "a # b"
    assert config._parse_value("   # nothing") == ""


def test_real_provider_streams_and_saves(client, fake_server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "good")
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200 and r.text == "Hello"
    cid = r.headers["X-Conversation-Id"]
    msgs = client.get(f"/api/conversations/{cid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert client.post("/api/chat/test", json={"provider": "openai_compatible"}).json()["ok"] is True


def test_failed_provider_leaves_no_orphan_conversation(client, fake_server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "bad")
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 502 and "API key" in r.json()["error"]["message"]
    assert client.get("/api/conversations").json() == []
