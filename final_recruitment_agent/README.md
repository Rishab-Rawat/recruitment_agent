# LangGraph Recruitment Matching Agent

A submission-oriented integration of the Milestone 1 filesystem tools and Milestone 2 resume RAG/matching system, orchestrated with LangGraph.

## Assignment mapping

### Part A — Agent Architecture
- `matching_agent.py` implements the required LangGraph state machine.
- State tracks conversation history, job requirements, shortlist, and reasoning.
- Graph: `START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates → Generate Report → Human Feedback Loop → END`.
- Required tools are exposed: `extract_requirements`, `compare_candidates`, `generate_interview_questions`, plus Milestone 1 filesystem tools and Milestone 2 RAG/matcher.

### Part B — Interactive Features
The CLI accepts natural-language queries and supports iterative refinement with `/feedback`. It also supports candidate comparison and interview-question generation.

### Part C — Advanced Capabilities
`multi_round_screening()` implements retrieval top-10, deeper evidence scoring, finalist selection, and a final HIRE/HOLD recommendation. Reports include match scores, semantic/keyword components, matched/missing skills, experience, excerpts, and reasoning.

## Project layout

```text
matching_agent.py
app.py
tools/
  fs_tools.py              # Milestone 1
  resume_rag.py            # Milestone 2
  job_matcher.py           # Milestone 2
  embeddings.py            # Milestone 2
resumes/                   # 40-resume Milestone 2 dataset
job_descriptions/          # 6 sample JDs
chroma_db/                 # included Milestone 2 index/cache
metrics.json
build_stats.json
diagrams/architecture.mmd
tests/
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The Milestone 2 index is included. If you change resumes, rebuild it:

```bash
python tools/resume_rag.py --rebuild --resume-dir resumes --chroma-dir chroma_db
```

### Embeddings
Set `EMBEDDING_BACKEND=auto` for the original Milestone 2 behavior. `auto` attempts hosted/local backends and falls back to the cached TF-IDF/SVD backend. On a machine with HuggingFace access, `EMBEDDING_BACKEND=huggingface` uses `all-MiniLM-L6-v2`.

## Run the agent

```bash
python app.py
```

Example:

```text
Recruiter> Find me candidates with React and 3+ years experience
Recruiter> /compare John Lee,Jane Smith
Recruiter> /questions John Lee
Recruiter> /feedback Must have TypeScript and 5+ years experience
Recruiter> /screen
```

## Five demo conversation flows

1. **Basic search** — `Find me candidates with React and 3+ years experience`.
2. **Comparison** — run a search, then `/compare Candidate A,Candidate B,Candidate C`.
3. **Explainability** — inspect the returned `reasoning`, `matched_skills`, `missing_must_have_skills`, and score components.
4. **Iterative refinement** — `/feedback Must have TypeScript and 5+ years experience` and observe re-ranking.
5. **Multi-round screening** — `/screen` to show top-10 → deep analysis → finalists → final recommendation.

## Demo checklist

- Show the architecture diagram.
- Run one natural-language search.
- Point out semantic score + keyword score + must-have status.
- Compare two or three candidates.
- Change a requirement and show the ranking change.
- Run multi-round screening and explain the final recommendation.

## Important note
This is a student-project integration, not a production hiring system. The included dataset is synthetic. Review recruiter decisions for fairness and do not use the score as an automated employment decision.

### Milestone 1 sample documents
`sample_resumes_milestone1/` preserves the original PDF/DOCX/TXT examples so the filesystem tools can be demonstrated on multiple document formats. The larger `resumes/` directory is the Milestone 2 RAG corpus used by the agent.
