"""Remote CrewAI recruiter desk exposed through the Exercise 3 A2A JSON-RPC shape."""
import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED_DIR))

import setup_env  # noqa: E402
from meter import Meter  # noqa: E402

print("⏳ waking up CrewAI (a few seconds)...", flush=True)
from crewai import Agent, Crew, Process, Task  # noqa: E402

CARD_PATH = "/.well-known/agent-card.json"


def agent_card(base_url: str) -> dict:
    return {
        "name": "Recruiter Desk",
        "description": "Screens job advertisements and candidate CVs for evidence, role fit, seniority, and unmet must-have qualifications.",
        "version": "1.0.0",
        "provider": {"organization": "Recruiting Team", "url": "https://example.com"},
        "supportedInterfaces": [{"url": base_url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{
            "id": "recruiter-cv-review",
            "name": "Recruiter CV review",
            "description": "Send a job advertisement and candidate CV as text. Returns concise findings; genuine unmet must-haves start with MISSING:.",
            "tags": ["recruiting", "cv-review"],
            "examples": ["Screen this CV against the job advertisement and distinguish must-haves from preferred qualifications."],
        }],
    }


def recruiter_review(text: str, meter: Meter) -> str:
    recruiter = Agent(
        role="Recruiter and CV Screener",
        goal="Assess the candidate's evidence and fit against the job advertisement, without inventing qualifications",
        backstory=(
            "You are a careful recruiter. You distinguish must-have requirements from preferred qualifications. "
            "A requirement written as A OR B is satisfied by either alternative. You only report a missing must-have "
            "when the supplied CV genuinely does not demonstrate it."
        ),
        llm=setup_env.crew_llm(),
        verbose=False,
        max_iter=2,
    )
    task = Task(
        description=(
            "Screen the following job advertisement and CV from a recruiter perspective:\n\n{text}\n\n"
            "List at most 5 concise findings, one per line. If a must-have is genuinely unmet, "
            "start that line with MISSING:. For positive evidence use MATCH:. For non-critical advice use NOTE:. "
            "Do not mark preferred qualifications as missing must-haves. For an 'A OR B' requirement, "
            "either alternative satisfies it. Never write MISSING: None. If there are no findings, reply exactly NONE."
        ),
        expected_output="Concise findings, one per line, using MISSING:, MATCH:, or NOTE:; or exactly NONE.",
        agent=recruiter,
    )
    result = Crew(agents=[recruiter], tasks=[task], process=Process.sequential,
                  verbose=False, tracing=False).kickoff(inputs={"text": text})
    meter.record_crew(result)
    return result.raw.strip()


def as_a2a_task(answer: str, context_id: str) -> dict:
    return {"task": {
        "id": str(uuid.uuid4()),
        "contextId": context_id,
        "status": {"state": "TASK_STATE_COMPLETED", "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "recruiter-review", "parts": [{"text": answer}]}],
    }}


def rpc_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


class RecruiterDesk(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _base_url(self) -> str:
        host = self.headers.get("Host", f"127.0.0.1:{self.server.server_port}")
        return f"http://{host}/"

    def do_GET(self):
        if self.path.split("?")[0] == CARD_PATH:
            print(f"📇 {now()} someone read our recruiter card", flush=True)
            self._send_json(200, agent_card(self._base_url()))
        else:
            self._send_json(404, {"error": f"The agent card is at {CARD_PATH}"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            call = json.loads(raw or b"null")
        except json.JSONDecodeError:
            self._send_json(200, rpc_error(None, -32700, "Invalid JSON payload"))
            return
        if not isinstance(call, dict) or call.get("jsonrpc") != "2.0" or "method" not in call:
            self._send_json(200, rpc_error(None, -32600, "Request payload validation error"))
            return
        req_id = call.get("id")
        if call["method"] != "SendMessage":
            self._send_json(200, rpc_error(req_id, -32601, f"Method not found: {call['method']}"))
            return
        message = (call.get("params") or {}).get("message") or {}
        text = "\n".join(part["text"] for part in message.get("parts", [])
                          if isinstance(part, dict) and "text" in part)
        if not text.strip():
            self._send_json(200, rpc_error(req_id, -32602, "Invalid parameters: no text part"))
            return
        print(f"📞 {now()} incoming A2A SendMessage · {len(text)} characters", flush=True)
        meter = Meter()
        try:
            print("   🤖 CrewAI recruiter is screening the CV...", flush=True)
            answer = recruiter_review(text, meter)
        except Exception as exc:  # return a protocol-shaped error; manager fails closed
            print(f"   💥 recruiter failed: {exc}", flush=True)
            self._send_json(200, rpc_error(req_id, -32603, "Internal error"))
            return
        context_id = message.get("contextId") or str(uuid.uuid4())
        self._send_json(200, {"jsonrpc": "2.0", "id": req_id, "result": as_a2a_task(answer, context_id)})
        cost = meter.dollars(setup_env.MODEL)
        print(f"✅ {now()} replied TASK_STATE_COMPLETED · {meter.calls} LLM call(s) · "
              f"{meter.tokens_in + meter.tokens_out} tokens · ${cost or 0:.5f} · {meter.seconds():.1f} s", flush=True)

    def log_message(self, *args):
        pass


def now() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recruiter Desk: CrewAI served over A2A")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    try:
        server = HTTPServer((args.host, args.port), RecruiterDesk)
    except OSError:
        print(f"Port {args.port} is busy. Try: python remote_desk.py --port 9998")
        return 1
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    print("\n📞 Recruiter Desk is open")
    print(f"   business card: http://{shown}:{args.port}{CARD_PATH}")
    print(f"   phone number:  http://{shown}:{args.port}/ (JSON-RPC 2.0, SendMessage)")
    print("   waiting for calls... (Ctrl+C to stop)\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n☎️ Recruiter Desk stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
