# 5–6 minute demo script

**0:00–0:40 — Architecture**
Open `diagrams/architecture.mmd`. Explain the LangGraph flow and that Milestone 1 handles document/file operations while Milestone 2 supplies hybrid RAG retrieval.

**0:40–1:50 — Natural language search**
Run `python app.py`. Ask: `Find me candidates with React and 3+ years experience`. Explain must-have parsing, semantic + BM25 scoring, and the shortlist.

**1:50–2:40 — Explainability**
Pick two candidates and run `/compare Name1,Name2`. Point to score components, matched/missing skills, experience, and excerpts.

**2:40–3:30 — Interview questions**
Run `/questions Name1`. Explain that questions are grounded in matched skills and gaps.

**3:30–4:30 — Human feedback loop**
Run `/feedback Must have TypeScript and 5+ years experience`. Show that requirements are updated and the shortlist is re-ranked.

**4:30–5:30 — Multi-round screening**
Run `/screen`. Show top 10, deep analysis, finalists, and HIRE/HOLD recommendation.

**5:30–6:00 — Close**
Mention the included Milestone 2 metrics, test suite, architecture diagram, five conversation flows, and limitations of synthetic data.
