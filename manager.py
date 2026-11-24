"""Job Application Desk: LangGraph manager calling a remote CrewAI recruiter over A2A."""
import json
import operator
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED_DIR))

import setup_env  # noqa: E402
from meter import Meter  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

DEFAULT_DESK = "http://127.0.0.1:9999"
CARD_PATH = "/.well-known/agent-card.json"

INPUTS = {
    "match": """JOB AD — Junior Data Analyst
Must-have: basic SQL queries; Excel including pivot tables; clear written communication; Power BI OR Tableau; ability to explain findings to non-technical colleagues.
Preferred: Python and experience with cloud data platforms.

CV — Candidate
Education: BSc Computer Science, graduating this year.
Skills and projects: SQL SELECT/JOIN/GROUP BY queries in university projects; Excel pivot tables and charts; built a Power BI dashboard for a course project; presented project findings in written reports and to classmates. Basic Python coursework. No cloud platform experience yet.""",
    "stretch": """JOB AD — Senior Machine Learning Engineer
Must-have: at least 5 years of professional ML engineering experience; production deployment and monitoring of ML systems; leading technical design for an ML team; advanced experience training and optimizing large-scale models; Python and deep learning frameworks.
Preferred: publications and cloud certifications.

CV — Candidate
Education: recent BSc Computer Science graduate.
Experience: university capstone using Python and PyTorch to train a small image classifier; one academic internship lasting 8 weeks focused on data cleaning; no full-time professional ML engineering experience; no production ML deployment or monitoring; no team leadership experience; no large-scale model training.""",
}
EXPECTED = {"match": "APPLY", "stretch": "DON'T APPLY YET"}

LOCAL_SPECIALISTS = {
    "skills": (
      "Check ONLY which job-ad requirements are clearly demonstrated by the CV. "
    "Report evidence for matched must-haves. Do not invent evidence. "
    "Treat academic projects, coursework, written reports, and presentations as valid evidence when relevant. "
    "If the CV says the candidate presented findings to classmates, count that as evidence of explaining findings to others. "  
    "For a requirement written as A OR B, either alternative satisfies it. "
    "Preferred qualifications are not must-haves. Do not write MISSING: unless a must-have is genuinely unmet."
    ),
    "gaps": (
        "Check ONLY the job's must-have qualifications that the CV does not demonstrate, "
        "and briefly suggest how the candidate could close each gap. "
        "Do not report preferred qualifications as missing must-haves. For A OR B, either option meets the requirement. "
        "Write one line beginning MISSING: for each genuine unmet must-have. If none, reply exactly NONE."
    ),
}
REMOTE_RECRUITER = (
    "Screen the candidate from a recruiter perspective. Focus on evidence, seniority, relevance, "
    "and whether the CV supports the must-have qualifications. Distinguish must-haves from preferred items. "
    "For A OR B requirements, either option satisfies the requirement. Never invent experience. "
    "Only write a line beginning MISSING: when a must-have is genuinely unmet. Never write MISSING: None. "
    "Otherwise use MATCH: or NOTE: lines; if there are no findings, reply exactly NONE."
)
MENU = {**LOCAL_SPECIALISTS, "recruiter": REMOTE_RECRUITER}
MAX_SPECIALISTS = 2
STAY_IN_LANE = "Stay within your assigned responsibility; do not do another specialist's job."


def read_business_card(desk: str) -> dict:
    with urllib.request.urlopen(desk.rstrip("/") + CARD_PATH, timeout=5) as reply:
        return json.load(reply)


def offers(card: dict, tag: str) -> bool:
    return any(tag in skill.get("tags", []) for skill in card.get("skills", []))


def phone(card: dict, text: str) -> str:
    call = {
        "jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "SendMessage",
        "params": {"message": {"role": "ROLE_USER", "messageId": str(uuid.uuid4()),
                                "parts": [{"text": text}]}},
    }
    url = card["supportedInterfaces"][0]["url"]
    request = urllib.request.Request(url, data=json.dumps(call).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "A2A-Version": "1.0"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        answer = json.load(reply)
    if "error" in answer:
        raise RuntimeError(f"remote desk refused the call: {answer['error'].get('message', 'unknown error')}")
    task = (answer.get("result") or {}).get("task")
    if not task:
        raise RuntimeError("remote reply did not contain an A2A Task")
    state = (task.get("status") or {}).get("state")
    if state != "TASK_STATE_COMPLETED":
        raise RuntimeError(f"remote task did not complete: {state}")
    return "\n".join(part["text"] for artifact in task.get("artifacts", [])
                     for part in artifact.get("parts", []) if "text" in part)


def banner(title: str) -> str:
    line = "=" * (len(title) + 8)
    return f"{line}\n=== {title} ===\n{line}"
def indent(text: str) -> str:
    return "\n".join("    " + line for line in str(text).splitlines())

class Whiteboard(TypedDict):
    job: str
    team: list[str]
    findings: Annotated[list[str], operator.add]
    timings: Annotated[list[float], operator.add]
    fanout_started: float
    parallel_seconds: float
    decision: str
    summary: str
    remote_failed: bool


llm = setup_env.graph_llm()
meter = Meter()
desk_card: dict | None = None
why_no_desk = "the recruiter desk business card has not been read"


def ask(instructions: str, message: str) -> str:
    reply = llm.invoke([("system", instructions), ("user", message)])
    meter.record(reply)
    return reply.content.strip()


def lead_decides(board: Whiteboard) -> dict:
    prompt = (
        "You are only a router. Do NOT review the CV or job ad yourself. Choose exactly the specialists needed.\n"
        "Reply with specialist names only, separated by spaces; no explanation. Example: skills recruiter\n"
        "Available specialists:\n" + "\n".join(f"- {name}: {job}" for name, job in MENU.items()) +
        "\nChoose skills for identifying demonstrated requirements; choose gaps for missing must-haves; "
        "always choose recruiter for independent recruiter screening."
    )
    reply = ask(prompt, board["job"])
    words = re.findall(r"[a-z]+", reply.lower())
    requested = list(dict.fromkeys(word for word in words if word in MENU))
    # Validate against the allowlist and enforce the budget. Always require the remote recruiter.
    team = [name for name in requested if name in MENU][:MAX_SPECIALISTS]
    if "recruiter" not in team:
        team = [name for name in team if name != "recruiter"][:MAX_SPECIALISTS - 1] + ["recruiter"]
    local_names = [name for name in team if name in LOCAL_SPECIALISTS]
    if not local_names:
        # Simple fallback: route based on explicit seniority / missing-experience clues.
        local_names = ["gaps"] if re.search(r"senior|5 years|production|leadership|large-scale", board["job"], re.I) else ["skills"]
        team = local_names + ["recruiter"]
    print(f"🧭 Lead chose: {', '.join(team)}")
    return {"team": team, "fanout_started": time.perf_counter()}


def send_to_team(board: Whiteboard) -> list[str]:
    return board["team"]


def only_relevant_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and line.strip().upper() != "NONE"]


def make_local_specialist(name: str, job: str):
    def specialist(board: Whiteboard) -> dict:
        started = time.perf_counter()
        output = ask(
            f"You are the {name} specialist. {job} {STAY_IN_LANE} "
            "Return at most 5 concise lines. Prefix genuine unmet must-have requirements with MISSING:. "
            "Use MATCH: or NOTE: for other findings. Never write MISSING: None. If no findings, reply exactly NONE.",
            board["job"],
        )
        lines = only_relevant_lines(output)
        took = time.perf_counter() - started
        print(f"   ✍️ {name:<10} wrote {len(lines)} line(s) ({took:.1f} s, local)")
        return {"findings": [f"[{name}] {line}" for line in lines], "timings": [took]}
    specialist.__name__ = name
    return specialist


def recruiter_desk(board: Whiteboard) -> dict:
    started = time.perf_counter()
    try:
        if desk_card is None:
            raise ConnectionError(why_no_desk)
        lines = only_relevant_lines(phone(desk_card, board["job"]))
        failed = False
        print(f"   📞 recruiter answered with {len(lines)} line(s) (A2A)")
    except Exception as problem:  # fail closed if the remote specialist is unavailable
        lines = [f"MISSING: recruiter review could not be completed ({problem})."]
        failed = True
        print(f"   ☎️ recruiter no answer ({problem}) -> fail closed")
    return {"findings": [f"[recruiter·A2A] {line}" for line in lines],
            "timings": [time.perf_counter() - started], "remote_failed": failed}


def lead_summarizes(board: Whiteboard) -> dict:
    findings = board["findings"]
    has_missing = any(re.match(r"^\[[^]]+\]\s*MISSING\s*:", line, re.I) for line in findings)
    # Deterministic decision: Python, not the LLM, has the final say.
    decision = "DON'T APPLY YET" if has_missing or board.get("remote_failed", False) else "APPLY"
    try:
        summary = ask("Summarize these candidate-screening findings in 3 short bullets. Do not change the decision.",
                      "\n".join(findings) or "No findings.")
    except Exception:
        summary = "\n".join(findings) or "No findings were returned."
    return {"decision": decision, "summary": summary,
            "parallel_seconds": time.perf_counter() - board["fanout_started"]}


def build_graph():
    graph = StateGraph(Whiteboard)
    graph.add_node("lead_decides", lead_decides)
    for name, job in LOCAL_SPECIALISTS.items():
        graph.add_node(name, make_local_specialist(name, job))
        graph.add_edge(name, "lead_summarizes")
    graph.add_node("recruiter", recruiter_desk)
    graph.add_edge("recruiter", "lead_summarizes")
    graph.add_node("lead_summarizes", lead_summarizes)
    graph.add_edge(START, "lead_decides")
    graph.add_conditional_edges("lead_decides", send_to_team, list(MENU))
    graph.add_edge("lead_summarizes", END)
    return graph.compile()


def main() -> int:
    global desk_card, why_no_desk
    args = sys.argv[1:]
    if "--draw" in args:
        print("```mermaid\nflowchart TD\n    A[Job ad + CV] --> B[LangGraph lead routes]\n    B --> C[skills]\n    B --> D[gaps]\n    B --> E[Remote recruiter over A2A]\n    E --> F[CrewAI Recruiter Desk]\n    C --> G[Shared whiteboard]\n    D --> G\n    E --> G\n    G --> H{Python decision}\n    H -->|MISSING or remote failure| I[DON'T APPLY YET]\n    H -->|No MISSING and remote answered| J[APPLY]\n``` ")
        return 0
    desk = DEFAULT_DESK
    if "--desk" in args:
        index = args.index("--desk")
        if index + 1 < len(args):
            desk = args[index + 1]
        args = args[:index] + args[index + 2:]
    name = next((arg for arg in args if not arg.startswith("--")), "match")
    if name not in INPUTS:
        print(f"Unknown input '{name}'. Choose one of: {', '.join(INPUTS)}")
        return 2
    job = INPUTS[name]
    print(banner(f"JOB APPLICATION DESK · '{name}'"))
    print(indent(job))
    try:
        card = read_business_card(desk)
        print(f"📇 Read recruiter card at {desk}{CARD_PATH}: {card.get('name', 'unnamed desk')}")
        if offers(card, "recruiting") and offers(card, "cv-review"):
            desk_card = card
            print("   ✅ recruiter card advertises recruiting and cv-review")
        else:
            why_no_desk = "the card does not advertise both recruiting and cv-review"
            print(f"   ⛔ {why_no_desk}")
    except (urllib.error.URLError, OSError, TimeoutError) as problem:
        desk_card = None
        why_no_desk = f"nobody answered at {desk}: {problem}"
        print(f"☎️ Remote recruiter unavailable: {why_no_desk}")
    board = build_graph().invoke(
        {"job": job, "findings": [], "timings": [], "remote_failed": False},
        config={"recursion_limit": 10},
    )
    print("\n📋 THE WHITEBOARD")
    print(indent("\n".join(board["findings"]) or "(nothing)"))
    print(f"\n🏁 DECISION: {board['decision']} (computed by Python: any MISSING: or remote failure means DON'T APPLY YET)")
    print(indent(board["summary"]))
    print(f"\n⏱ {len(board['team'])} specialist(s): {board['parallel_seconds']:.1f} s")
    print(f"Expected: {EXPECTED[name]!r}, got: {board['decision']!r} — {'PASS' if EXPECTED[name] == board['decision'] else 'CHECK'}")
    print("\n" + meter.card(agents=len(board["team"]) + 1, model=setup_env.MODEL,
                            note="Recruiter desk tokens are counted in its own process."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
