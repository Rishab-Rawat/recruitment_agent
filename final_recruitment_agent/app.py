"""CLI chat interface for the recruitment matching agent."""
from __future__ import annotations
import json, os, sys
from matching_agent import run_agent, compare_candidates, generate_interview_questions, multi_round_screening

HELP = """Commands:
  /compare NAME1,NAME2[,NAME3]  Compare shortlisted candidates
  /questions NAME               Generate screening questions
  /screen                       Run 3-round screening
  /feedback TEXT                Re-rank with recruiter feedback
  /report                       Show current report
  /help                         Show help
  /quit                         Exit

Otherwise type a natural-language recruitment request, e.g.
  Find me candidates with React and 3+ years experience
"""

def main():
    print("Recruitment Matching Agent (LangGraph)\nType /help for commands.\n")
    state = None
    while True:
        try: q = input("Recruiter> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not q: continue
        if q == "/quit": break
        if q == "/help": print(HELP); continue
        if q == "/report": print(json.dumps(state.get("report", {}), indent=2, default=str) if state else "No report yet."); continue
        if q.startswith("/compare "):
            names = [x.strip() for x in q[9:].split(",") if x.strip()]
            print(json.dumps(compare_candidates(names, state or {}), indent=2, default=str)); continue
        if q.startswith("/questions "):
            print(json.dumps(generate_interview_questions(q[11:].strip(), state or {}), indent=2, default=str)); continue
        if q == "/screen":
            jd = (state or {}).get("job_description", "")
            if not jd: print("Run a matching request first."); continue
            print(json.dumps(multi_round_screening(jd), indent=2, default=str)); continue
        if q.startswith("/feedback "):
            feedback = q[10:].strip()
            state = run_agent(state.get("job_description", ""), previous_state=state, feedback=feedback)
        else:
            state = run_agent(q, previous_state=state)
        print(json.dumps({"requirements": state.get("requirements"), "top_candidates": state.get("candidate_shortlist", [])[:5]}, indent=2, default=str))

if __name__ == "__main__": main()
