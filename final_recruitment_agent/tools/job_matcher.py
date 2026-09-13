"""
job_matcher.py
Part B: Job Matching Engine

Given a job description, this module:
  1. Embeds the JD and runs semantic search over the resume-chunk vector
     store built by resume_rag.py.
  2. Combines that with a keyword/BM25 check against a set of "critical
     skills" pulled out of the JD (hybrid search).
  3. Aggregates chunk-level hits back up to one row per candidate.
  4. Scores each candidate 0-100, filters out anyone missing a stated
     must-have requirement (e.g. "5+ years Python"), and produces a short
     natural-language reasoning string plus the resume excerpts that drove
     the match.

Run directly to try it against the sample job descriptions:
    python job_matcher.py --jd job_descriptions/jd_ml_engineer.txt --must-have-skills Python "Machine Learning" --min-years 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

try:
    from .resume_rag import ResumeRAG, KNOWN_SKILLS, BASE_DIR
except ImportError:
    from resume_rag import ResumeRAG, KNOWN_SKILLS, BASE_DIR

TOP_K = 10
# Hybrid search weighting: how much of the final ranking score comes from
# dense semantic similarity vs. sparse keyword/BM25 overlap on critical skills.
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


@dataclass
class MustHaveRequirement:
    skills: list = field(default_factory=list)
    min_years: float = 0.0


@dataclass
class CandidateMatch:
    candidate_name: str
    resume_file: str
    resume_path: str
    match_score: float
    semantic_score: float
    keyword_score: float
    matched_skills: list
    missing_must_have_skills: list
    years_experience: float
    education: str
    relevant_excerpts: list
    reasoning: str
    passed_must_have: bool


def extract_must_have(jd_text: str) -> MustHaveRequirement:
    """
    Pull an implicit must-have requirement straight out of the JD text, e.g.
    "5+ years of experience with Python" -> min_years=5, skills includes Python.
    This is a heuristic (regex + skill-vocabulary match); extract_must_have can
    also be overridden by explicit --must-have-skills/--min-years CLI args.
    """
    skills_found = []
    lower = jd_text.lower()
    for skill in KNOWN_SKILLS:
        pattern = r"(?<![a-zA-Z])" + re.escape(skill.lower()) + r"(?![a-zA-Z])"
        if re.search(pattern, lower):
            skills_found.append(skill)

    min_years = 0.0
    m = re.search(r"(\d+)\s*\+?\s*years?", jd_text, re.IGNORECASE)
    if m:
        min_years = float(m.group(1))

    # "Critical" skills = the ones mentioned in the same sentence as "require"/
    # "must have"/"requirement", falling back to the first few skills found.
    critical_sentence = ""
    for sent in re.split(r"(?<=[.!?])\s+", jd_text):
        if re.search(r"require|must have|must-have|need", sent, re.IGNORECASE):
            critical_sentence += " " + sent
    critical_skills = [s for s in skills_found
                       if re.search(re.escape(s.lower()), critical_sentence.lower())] or skills_found[:2]

    return MustHaveRequirement(skills=critical_skills, min_years=min_years)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9\+\#\.]+", text.lower())


class JobMatcher:
    def __init__(self, rag: ResumeRAG, top_k: int = TOP_K):
        self.rag = rag
        self.top_k = top_k

    def _keyword_scores(self, jd_text: str, candidate_docs: dict[str, str]) -> dict[str, float]:
        """
        BM25 keyword score of the JD against each candidate's full resume
        text (all their chunks concatenated). This is the "keyword for
        critical skills" half of the hybrid search -- it rewards candidates
        whose resume literally contains the JD's important terms, which
        catches exact-match cases (e.g. an exact tool/certification name)
        that a dense embedding can sometimes blur across near-synonyms.
        """
        file_order = list(candidate_docs.keys())
        corpus_tokens = [_tokenize(candidate_docs[f]) for f in file_order]
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(_tokenize(jd_text))
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        return {f: float(s) / max_score for f, s in zip(file_order, scores)}

    def match(self, jd_text: str, must_have: MustHaveRequirement | None = None,
              top_k: int | None = None) -> dict:
        top_k = top_k or self.top_k
        t0 = time.time()

        if must_have is None:
            must_have = extract_must_have(jd_text)

        # --- Semantic search over chunks (retrieve more than top_k chunks
        # since multiple chunks per candidate need to be aggregated back to
        # one row per candidate). ---
        raw = self.rag.query(jd_text, top_k=top_k * 6)
        all_meta = self.rag.get_all_resume_metadata()

        # Aggregate chunk-level cosine similarity to a per-candidate score:
        # take the best-matching chunk's similarity (so a strong Experience-
        # section match isn't diluted by a weak Education-section match).
        semantic_by_candidate: dict[str, float] = {}
        excerpts_by_candidate: dict[str, list[str]] = {}
        matched_sections_by_candidate: dict[str, set] = {}
        ids, distances, documents, metadatas = (
            raw["ids"][0], raw["distances"][0], raw["documents"][0], raw["metadatas"][0]
        )
        for _id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            similarity = 1 - dist  # chroma cosine distance -> similarity
            fname = meta["resume_file"]
            if fname not in semantic_by_candidate or similarity > semantic_by_candidate[fname]:
                semantic_by_candidate[fname] = similarity
            excerpts_by_candidate.setdefault(fname, [])
            if len(excerpts_by_candidate[fname]) < 2:
                snippet = doc if len(doc) < 220 else doc[:220].rsplit(" ", 1)[0] + "..."
                excerpts_by_candidate[fname].append(snippet)
            matched_sections_by_candidate.setdefault(fname, set()).add(meta["section"])

        # --- Keyword/BM25 half of the hybrid search, over full resume text. ---
        candidate_docs = {f: self.rag.collection.get(
            where={"resume_file": f}
        ) for f in semantic_by_candidate}
        candidate_fulltext = {
            f: " ".join(res["documents"]) for f, res in candidate_docs.items()
        }
        keyword_by_candidate = self._keyword_scores(jd_text, candidate_fulltext)

        # --- Combine, score, filter, and build reasoning per candidate. ---
        results: list[CandidateMatch] = []
        for fname, sem_score in semantic_by_candidate.items():
            meta = all_meta[fname]
            candidate_skills = meta["skills"].split(",") if meta["skills"] else []
            years = float(meta["years_experience"])
            kw_score = keyword_by_candidate.get(fname, 0.0)

            combined = SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score
            match_score = round(max(0.0, min(1.0, combined)) * 100, 1)

            matched_skills = [s for s in must_have.skills if s in candidate_skills] or \
                [s for s in KNOWN_SKILLS if s in candidate_skills][:5]
            missing_must_have = [s for s in must_have.skills if s not in candidate_skills]
            passed_years = years >= must_have.min_years if must_have.min_years else True
            passed_must_have = passed_years and not missing_must_have

            sections_hit = sorted(matched_sections_by_candidate.get(fname, []))
            reasoning_parts = []
            if matched_skills:
                reasoning_parts.append(f"matches on {', '.join(matched_skills[:4])}")
            if sections_hit:
                reasoning_parts.append(f"strongest alignment in the {'/'.join(sections_hit)} section(s)")
            reasoning_parts.append(f"{years:.0f} years of relevant experience")
            if missing_must_have:
                reasoning_parts.append(f"missing must-have skill(s): {', '.join(missing_must_have)}")
            reasoning = "Candidate " + "; ".join(reasoning_parts) + "."

            results.append(CandidateMatch(
                candidate_name=meta["candidate_name"],
                resume_file=fname,
                resume_path=meta["resume_path"],
                match_score=match_score,
                semantic_score=round(sem_score * 100, 1),
                keyword_score=round(kw_score * 100, 1),
                matched_skills=matched_skills,
                missing_must_have_skills=missing_must_have,
                years_experience=years,
                education=meta["education"],
                relevant_excerpts=excerpts_by_candidate.get(fname, []),
                reasoning=reasoning,
                passed_must_have=passed_must_have,
            ))

        # Passing candidates first (still sorted by score within each group),
        # so a must-have filter narrows the list without silently discarding
        # the rest -- callers can see who almost qualified and why.
        results.sort(key=lambda r: (not r.passed_must_have, -r.match_score))
        results = results[:top_k]

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "job_description": jd_text,
            "must_have": {"skills": must_have.skills, "min_years": must_have.min_years},
            "latency_ms": elapsed_ms,
            "top_matches": [
                {
                    "candidate_name": r.candidate_name,
                    "resume_path": r.resume_path,
                    "match_score": r.match_score,
                    "semantic_score": r.semantic_score,
                    "keyword_score": r.keyword_score,
                    "matched_skills": r.matched_skills,
                    "missing_must_have_skills": r.missing_must_have_skills,
                    "passed_must_have": r.passed_must_have,
                    "years_experience": r.years_experience,
                    "education": r.education,
                    "relevant_excerpts": r.relevant_excerpts,
                    "reasoning": r.reasoning,
                }
                for r in results
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Match resumes against a job description.")
    parser.add_argument("--jd", required=True, help="Path to a job description .txt file")
    parser.add_argument("--must-have-skills", nargs="*", default=None)
    parser.add_argument("--min-years", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--chroma-dir", default=os.path.join(BASE_DIR, "chroma_db"))
    args = parser.parse_args()

    with open(args.jd) as f:
        jd_text = f.read()

    rag = ResumeRAG(chroma_dir=args.chroma_dir)
    matcher = JobMatcher(rag, top_k=args.top_k)

    must_have = None
    if args.must_have_skills is not None or args.min_years is not None:
        must_have = MustHaveRequirement(
            skills=args.must_have_skills or [],
            min_years=args.min_years or 0.0,
        )

    result = matcher.match(jd_text, must_have=must_have)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
