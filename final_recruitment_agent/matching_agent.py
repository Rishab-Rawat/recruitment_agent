"""LangGraph recruitment matching agent.

Graph: START -> Parse JD -> Extract Requirements -> Search Resumes ->
Rank Candidates -> Generate Report -> Human Feedback Loop -> END

The implementation intentionally keeps ranking deterministic and explainable:
Milestone-2's hybrid RAG matcher supplies evidence, while LangGraph manages
state, conversational refinement, and multi-step orchestration.
"""
from __future__ import annotations

import json, os, re
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from tools.fs_tools import list_files, read_file, search_in_file, write_file
from tools.resume_rag import ResumeRAG, KNOWN_SKILLS
from tools.job_matcher import JobMatcher, MustHaveRequirement

ROOT = os.path.dirname(os.path.abspath(__file__))
RESUME_DIR = os.path.join(ROOT, "resumes")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")


class AgentState(TypedDict, total=False):
    conversation_history: list[dict[str, str]]
    user_query: str
    job_description: str
    parsed_job_description: str
    requirements: dict[str, Any]
    candidate_shortlist: list[dict[str, Any]]
    candidate_reasoning: dict[str, str]
    comparison: dict[str, Any]
    interview_questions: dict[str, list[str]]
    report: dict[str, Any]
    feedback: str
    round_results: dict[str, Any]
    final_recommendation: dict[str, Any]


def extract_requirements(jd: str) -> dict[str, Any]:
    """Parse must-have vs nice-to-have requirements from a JD."""
    text = jd.strip()
    lower = text.lower()
    must_skills = []
    nice_skills = []
    for skill in KNOWN_SKILLS:
        if re.search(r"(?<![a-zA-Z])" + re.escape(skill.lower()) + r"(?![a-zA-Z])", lower):
            # Classify using nearby sentences/phrases.
            if re.search(r"(?:must|required|requirement|essential|need|minimum).{0,180}" + re.escape(skill.lower()), lower) or re.search(re.escape(skill.lower()) + r".{0,120}(?:must|required|essential)", lower):
                must_skills.append(skill)
            else:
                nice_skills.append(skill)
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.I)
    min_years = float(m.group(1)) if m else 0.0
    # If no explicit must-have skill was detected, use first two mentioned skills
    # as practical defaults, matching Milestone 2's heuristic.
    if not must_skills:
        must_skills = nice_skills[:2]
        nice_skills = nice_skills[2:]
    return {
        "must_have": {"skills": must_skills, "min_years": min_years},
        "nice_to_have": {"skills": nice_skills[:8]},
        "source": "heuristic requirement extraction over JD text",
    }


def _parse_query(query: str) -> str:
    """Turn conversational requests into a search-oriented JD."""
    q = query.strip()
    # Refinement requests inherit the prior JD; caller combines history.
    return q


def _apply_feedback(req: dict[str, Any], feedback: str) -> dict[str, Any]:
    if not feedback:
        return req
    out = json.loads(json.dumps(req))
    lower = feedback.lower()
    # Add explicit skills named in feedback.
    mentioned = [s for s in KNOWN_SKILLS if re.search(r"(?<![a-zA-Z])" + re.escape(s.lower()) + r"(?![a-zA-Z])", lower)]
    if any(x in lower for x in ["must", "require", "required", "prioritize"]):
        for s in mentioned:
            if s not in out["must_have"]["skills"]:
                out["must_have"]["skills"].append(s)
            if s in out["nice_to_have"]["skills"]:
                out["nice_to_have"]["skills"].remove(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", feedback, re.I)
    if m:
        out["must_have"]["min_years"] = float(m.group(1))
    return out


def parse_jd_node(state: AgentState) -> AgentState:
    query = state.get("user_query", "")
    previous = state.get("job_description", "")
    if previous and any(k in query.lower() for k in ["add ", "change", "remove", "instead", "also", "now require", "make"]):
        jd = previous + "\n\nRecruiter refinement: " + query
    else:
        jd = query or previous
    return {"parsed_job_description": _parse_query(jd), "job_description": jd}


def extract_requirements_node(state: AgentState) -> AgentState:
    req = extract_requirements(state["parsed_job_description"])
    req = _apply_feedback(req, state.get("feedback", ""))
    return {"requirements": req}


def _ensure_rag() -> ResumeRAG:
    rag = ResumeRAG(chroma_dir=CHROMA_DIR, embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto"))
    # If no index exists, build it from the included 40-resume dataset.
    if rag.collection.count() == 0:
        rag.build(resume_dir=RESUME_DIR)
    return rag


def search_resumes_node(state: AgentState) -> AgentState:
    rag = _ensure_rag()
    req = state["requirements"]["must_have"]
    matcher = JobMatcher(rag, top_k=10)
    result = matcher.match(state["job_description"], MustHaveRequirement(req["skills"], req["min_years"]), top_k=10)
    return {"candidate_shortlist": result["top_matches"], "candidate_reasoning": {x["candidate_name"]: x["reasoning"] for x in result["top_matches"]}}


def rank_candidates_node(state: AgentState) -> AgentState:
    candidates = list(state.get("candidate_shortlist", []))
    nice = set(state.get("requirements", {}).get("nice_to_have", {}).get("skills", []))
    for c in candidates:
        c["nice_to_have_matches"] = [s for s in c.get("matched_skills", []) if s in nice]
        c["overall_explanation"] = c["reasoning"]
    candidates.sort(key=lambda x: (not x["passed_must_have"], -x["match_score"]))
    for i, c in enumerate(candidates, 1):
        c["rank"] = i
    return {"candidate_shortlist": candidates}


def generate_report_node(state: AgentState) -> AgentState:
    candidates = state.get("candidate_shortlist", [])
    req = state.get("requirements", {})
    report = {
        "summary": f"Ranked {len(candidates)} candidates against the stated requirements.",
        "requirements": req,
        "candidates": candidates,
        "ranking_basis": "65% semantic similarity + 35% BM25 keyword relevance, with must-have qualification prioritized.",
    }
    return {"report": report}


def human_feedback_node(state: AgentState) -> AgentState:
    # In CLI mode feedback is supplied on the next invocation/turn. Keeping
    # this node explicit satisfies the human-in-the-loop graph requirement.
    return {"conversation_history": state.get("conversation_history", []) + ([{"role": "system", "content": "Report generated; awaiting optional recruiter refinement."}] if not state.get("feedback") else [])}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse_jd", parse_jd_node)
    g.add_node("extract_requirements", extract_requirements_node)
    g.add_node("search_resumes", search_resumes_node)
    g.add_node("rank_candidates", rank_candidates_node)
    g.add_node("generate_report", generate_report_node)
    g.add_node("human_feedback", human_feedback_node)
    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "extract_requirements")
    g.add_edge("extract_requirements", "search_resumes")
    g.add_edge("search_resumes", "rank_candidates")
    g.add_edge("rank_candidates", "generate_report")
    g.add_edge("generate_report", "human_feedback")
    g.add_edge("human_feedback", END)
    return g.compile()


def run_agent(query: str, *, previous_state: AgentState | None = None, feedback: str = "") -> AgentState:
    state: AgentState = dict(previous_state or {})
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": query})
    state.update({"user_query": query, "feedback": feedback, "conversation_history": history})
    result = build_graph().invoke(state)
    result["conversation_history"] = history + [{"role": "assistant", "content": json.dumps(result.get("report", {}), default=str)}]
    return result


def compare_candidates(candidate_ids: list[str], state: AgentState) -> dict[str, Any]:
    """Head-to-head comparison by candidate name or resume filename."""
    rows = []
    wanted = {x.lower() for x in candidate_ids}
    for c in state.get("candidate_shortlist", []):
        if c["candidate_name"].lower() in wanted or os.path.basename(c["resume_path"]).lower() in wanted:
            rows.append(c)
    rows.sort(key=lambda x: -x["match_score"])
    if len(rows) < 2:
        return {"error": "Provide at least two candidate names from the shortlist.", "matches": rows}
    return {"matches": rows, "winner": rows[0]["candidate_name"], "score_gap": round(rows[0]["match_score"] - rows[1]["match_score"], 1),
            "why": f"{rows[0]['candidate_name']} ranks higher due to a stronger combined semantic/keyword score and must-have alignment."}


def generate_interview_questions(candidate_id: str, state: AgentState) -> dict[str, Any]:
    for c in state.get("candidate_shortlist", []):
        if c["candidate_name"].lower() == candidate_id.lower() or os.path.basename(c["resume_path"]).lower() == candidate_id.lower():
            skills = c.get("matched_skills", [])[:5]
            gaps = c.get("missing_must_have_skills", [])
            qs = [f"Walk me through a project where you used {s}. What was your contribution and measurable outcome?" for s in skills]
            qs.append(f"How would you approach the responsibilities of this role given your {c.get('years_experience', 0):.0f} years of experience?")
            if gaps:
                qs.append(f"Your resume does not clearly show {', '.join(gaps)}. What is your hands-on experience with it?")
            return {"candidate": c["candidate_name"], "questions": qs[:7]}
    return {"error": f"Candidate '{candidate_id}' was not found in the current shortlist."}


def multi_round_screening(jd: str, *, top_n: int = 10) -> dict[str, Any]:
    """Three-stage screening: retrieval top-10, deep evidence review, final recommendation."""
    state = run_agent(jd)
    top10 = state.get("candidate_shortlist", [])[:top_n]
    deep = []
    for c in top10:
        evidence = len(c.get("relevant_excerpts", []))
        deep_score = c["match_score"] + (3 if c["passed_must_have"] else -5) + min(2, evidence)
        deep.append({**c, "deep_analysis_score": round(deep_score, 1), "screening_status": "advance" if c["passed_must_have"] else "review"})
    deep.sort(key=lambda x: (-x["deep_analysis_score"], x["rank"]))
    finalists = deep[:3]
    hire = finalists[0] if finalists else None
    final = {"recommendation": hire["candidate_name"] if hire else None, "decision": "HIRE" if hire and hire["passed_must_have"] and hire["match_score"] >= 70 else "HOLD", "reason": hire.get("reasoning") if hire else "No viable finalist."}
    return {"round_1_top_10": top10, "round_2_deep_analysis": deep, "round_3_finalists": finalists, "final_recommendation": final}


# Required assignment tool names exposed as simple callables.
def filesystem_tools():
    return {"read_file": read_file, "list_files": list_files, "write_file": write_file, "search_in_file": search_in_file}
