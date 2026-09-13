"""
resume_rag.py
Part A: RAG System Setup

Pipeline:
  1. Load resumes from disk (plain-text files -- the output format that the
     Milestone-1 file-loading tools would hand off, whether the source was a
     .pdf, .docx, or .txt resume).
  2. Chunk each resume *by section* (Summary / Skills / Experience / Education /
     Certifications) rather than by a fixed character window, so a retrieved
     chunk is always a coherent, human-readable unit.
  3. Extract structured metadata (name, skills, years of experience, education)
     with a mix of regex and section-aware heuristics.
  4. Embed each chunk with a local HuggingFace sentence-transformer (no API key
     required -- this is the "HuggingFace models" option from the assignment).
  5. Store embeddings + metadata in a persistent ChromaDB collection.

Run directly to (re)build the vector store from resumes/*.txt:
    python resume_rag.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import chromadb

try:
    from .embeddings import get_embedding_backend
except ImportError:
    from embeddings import get_embedding_backend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_DIR = os.path.join(BASE_DIR, "resumes")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "resumes"

SECTION_HEADERS = ["SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS"]

# A reasonably broad skill vocabulary used for metadata extraction & keyword
# search. In a production system this would come from a maintained taxonomy
# (e.g. ESCO, LinkedIn Skills) rather than a hardcoded list.
KNOWN_SKILLS = [
    "Python", "Django", "PostgreSQL", "REST APIs", "Docker", "Kubernetes", "AWS",
    "Redis", "Kafka", "GraphQL", "Microservices", "CI/CD", "Go", "Java", "Celery",
    "gRPC", "MongoDB", "Machine Learning", "PyTorch", "Scikit-learn", "Pandas",
    "TensorFlow", "NLP", "Computer Vision", "MLOps", "AWS SageMaker",
    "Deep Learning", "Transformers", "LangChain", "Vector Databases",
    "Feature Engineering", "A/B Testing", "Spark", "SQL", "Statistics",
    "Data Visualization", "R", "Tableau", "Power BI", "Excel",
    "Experiment Design", "NumPy", "JavaScript", "React", "HTML", "CSS",
    "TypeScript", "Redux", "Next.js", "Webpack", "Tailwind CSS", "Vue.js",
    "Jest", "Accessibility", "Figma", "Node.js", "Terraform", "Jenkins",
    "Ansible", "Linux", "Bash", "Prometheus", "Grafana", "Azure", "GCP", "Helm",
    "Product Strategy", "Roadmapping", "Agile", "Stakeholder Management",
    "Data Analysis", "User Research", "Jira", "Go-to-Market",
    "Pricing Strategy", "Competitive Analysis",
]


@dataclass
class ResumeChunk:
    chunk_id: str
    resume_file: str
    candidate_name: str
    section: str
    text: str


@dataclass
class ResumeMetadata:
    candidate_name: str
    resume_file: str
    resume_path: str
    skills: list = field(default_factory=list)
    years_experience: float = 0.0
    education: str = ""
    title: str = ""


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_resumes(resume_dir: str = RESUME_DIR) -> dict[str, str]:
    """Load every resume .txt file in resume_dir. Returns {filename: raw_text}."""
    resumes = {}
    for fname in sorted(os.listdir(resume_dir)):
        if fname.endswith(".txt") and not fname.startswith("_"):
            with open(os.path.join(resume_dir, fname), "r") as f:
                resumes[fname] = f.read()
    return resumes


# ---------------------------------------------------------------------------
# 2. Section-aware chunking
# ---------------------------------------------------------------------------

def chunk_resume(fname: str, text: str) -> list[ResumeChunk]:
    """
    Split a resume into chunks aligned to its sections (Summary, Skills,
    Experience, Education, Certifications) instead of a fixed-size sliding
    window. This preserves semantic coherence -- an "Experience" chunk always
    contains full job entries, never half a bullet point cut off mid-sentence.

    The header block (name / title / contact line, before the first section
    header) is kept as its own "HEADER" chunk since it usually contains the
    candidate's name and current title.
    """
    lines = text.split("\n")
    name = lines[0].strip() if lines else fname

    # Find the line index of each section header.
    header_positions = []
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped in SECTION_HEADERS:
            header_positions.append((i, stripped))

    chunks: list[ResumeChunk] = []

    # Header/contact block: everything before the first recognized section.
    first_section_line = header_positions[0][0] if header_positions else len(lines)
    header_text = "\n".join(lines[:first_section_line]).strip()
    if header_text:
        chunks.append(ResumeChunk(
            chunk_id=f"{fname}::HEADER",
            resume_file=fname,
            candidate_name=name,
            section="HEADER",
            text=header_text,
        ))

    # One chunk per section, spanning from its header to the next header.
    for idx, (start_line, section_name) in enumerate(header_positions):
        end_line = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(lines)
        section_text = "\n".join(lines[start_line:end_line]).strip()
        if section_text:
            chunks.append(ResumeChunk(
                chunk_id=f"{fname}::{section_name}",
                resume_file=fname,
                candidate_name=name,
                section=section_name,
                text=section_text,
            ))

    return chunks


# ---------------------------------------------------------------------------
# 3. Metadata extraction
# ---------------------------------------------------------------------------

def extract_skills(full_text: str) -> list[str]:
    found = []
    lower = full_text.lower()
    for skill in KNOWN_SKILLS:
        # word-boundary-ish match so "Go" doesn't match inside "Google"
        pattern = r"(?<![a-zA-Z])" + re.escape(skill.lower()) + r"(?![a-zA-Z])"
        if re.search(pattern, lower):
            found.append(skill)
    return found


def extract_years_experience(full_text: str) -> float:
    """
    Two complementary strategies, combined by taking the max:
      (a) an explicit "N years of experience" phrase in the summary
      (b) the span of years covered by the EXPERIENCE section's date ranges
          (e.g. "2018-2026"), which catches resumes that don't state a figure.
    """
    years = 0.0

    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years? of experience", full_text, re.IGNORECASE)
    if m:
        years = max(years, float(m.group(1)))

    date_ranges = re.findall(r"\((\d{4})\s*-\s*(\d{4})\)", full_text)
    if date_ranges:
        earliest = min(int(r[0]) for r in date_ranges)
        latest = max(int(r[1]) for r in date_ranges)
        years = max(years, float(latest - earliest))

    return years


def extract_education(full_text: str) -> str:
    m = re.search(r"EDUCATION\s*\n(.+)", full_text)
    if m:
        return m.group(1).strip().split("\n")[0]
    return ""


def extract_name_and_title(full_text: str) -> tuple[str, str]:
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    name = lines[0] if lines else "Unknown"
    title = ""
    if len(lines) > 1 and "|" in lines[1]:
        title = lines[1].split("|")[0].strip()
    return name, title


def extract_metadata(fname: str, full_text: str, resume_dir: str = RESUME_DIR) -> ResumeMetadata:
    name, title = extract_name_and_title(full_text)
    return ResumeMetadata(
        candidate_name=name,
        resume_file=fname,
        resume_path=os.path.join(resume_dir, fname),
        skills=extract_skills(full_text),
        years_experience=extract_years_experience(full_text),
        education=extract_education(full_text),
        title=title,
    )


# ---------------------------------------------------------------------------
# 4 & 5. Embedding + vector store
# ---------------------------------------------------------------------------

class ResumeRAG:
    """
    Owns the embedding model and the persistent ChromaDB collection.
    Each stored vector corresponds to one section-chunk of one resume;
    resume-level metadata (skills/years/education) is duplicated onto every
    chunk of that resume so filtering works no matter which chunk matched.
    """

    def __init__(self, chroma_dir: str = CHROMA_DIR, embedding_backend: str = "auto",
                 fallback_corpus: Optional[list[str]] = None):
        # `embedding_backend`: "auto" | "huggingface" | "openai" | "cohere" | "tfidf".
        # See embeddings.py for what "auto" resolves to and why.
        self._backend_name = embedding_backend
        self._fallback_corpus = fallback_corpus
        self._tfidf_cache_path = os.path.join(chroma_dir, "tfidf_embedder.joblib")
        self.model = None  # lazily resolved in build()/query() via _ensure_model()
        os.makedirs(chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_model(self, corpus_for_fallback: Optional[list[str]] = None):
        if self.model is None:
            fallback_corpus = corpus_for_fallback or self._fallback_corpus
            if fallback_corpus is None:
                # Querying an already-built store (no corpus passed in): if we
                # end up on the local TF-IDF+SVD fallback, prefer the cached
                # fitted vectorizer/SVD (see tfidf_cache_path) so the query
                # vector lands in the exact same space used at build time,
                # without needing to refit on every new process. Only fall
                # back to refitting from Chroma's stored documents if no cache
                # exists yet.
                if not os.path.exists(self._tfidf_cache_path):
                    existing = self.collection.get(include=["documents"])
                    fallback_corpus = existing["documents"] or None
            self.model = get_embedding_backend(
                prefer=self._backend_name,
                corpus_for_fallback=fallback_corpus,
                tfidf_cache_path=self._tfidf_cache_path,
            )

    def embed(self, texts: list[str]):
        self._ensure_model(corpus_for_fallback=texts)
        return self.model.encode(texts)

    def build(self, resume_dir: str = RESUME_DIR, batch_size: int = 32) -> dict:
        """Full pipeline: load -> chunk -> extract metadata -> embed -> upsert."""
        t0 = time.time()
        resumes = load_resumes(resume_dir)

        # Reset the collection so re-runs don't duplicate entries, and drop any
        # stale cached fallback-embedder fit (it must be refit on the new corpus).
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        if os.path.exists(self._tfidf_cache_path):
            os.remove(self._tfidf_cache_path)
        self.model = None
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        all_chunks: list[ResumeChunk] = []
        all_metadata: dict[str, ResumeMetadata] = {}
        for fname, text in resumes.items():
            all_chunks.extend(chunk_resume(fname, text))
            all_metadata[fname] = extract_metadata(fname, text, resume_dir)

        # If we end up on the local TF-IDF+SVD fallback, fit it once on the
        # full chunk corpus up front so every batch embeds into the same space.
        self._ensure_model(corpus_for_fallback=[c.text for c in all_chunks])

        # Batch-embed and upsert.
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embed(texts)
            ids = [c.chunk_id for c in batch]
            metadatas = []
            for c in batch:
                meta = all_metadata[c.resume_file]
                metadatas.append({
                    "resume_file": c.resume_file,
                    "resume_path": meta.resume_path,
                    "candidate_name": meta.candidate_name,
                    "section": c.section,
                    "skills": ",".join(meta.skills),
                    "years_experience": meta.years_experience,
                    "education": meta.education,
                    "title": meta.title,
                })
            self.collection.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas,
            )

        elapsed = time.time() - t0
        stats = {
            "num_resumes": len(resumes),
            "num_chunks": len(all_chunks),
            "build_time_sec": round(elapsed, 2),
        }
        print(f"Indexed {stats['num_resumes']} resumes into {stats['num_chunks']} chunks "
              f"in {stats['build_time_sec']}s")
        return stats

    def query(self, text: str, top_k: int = 10, where: Optional[dict] = None):
        """Raw semantic search over chunks. Returns Chroma's query result dict."""
        self._ensure_model()
        embedding = self.model.encode([text])[0]
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )

    def get_all_resume_metadata(self) -> dict[str, dict]:
        """Convenience: one representative metadata row per resume_file."""
        got = self.collection.get(include=["metadatas"])
        by_file = {}
        for meta in got["metadatas"]:
            by_file.setdefault(meta["resume_file"], meta)
        return by_file


def main():
    parser = argparse.ArgumentParser(description="Build the resume RAG vector store.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the ChromaDB collection from resumes/")
    parser.add_argument("--resume-dir", default=RESUME_DIR)
    parser.add_argument("--chroma-dir", default=CHROMA_DIR)
    args = parser.parse_args()

    rag = ResumeRAG(chroma_dir=args.chroma_dir)
    if args.rebuild:
        stats = rag.build(resume_dir=args.resume_dir)
        with open(os.path.join(BASE_DIR, "build_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("Nothing to do. Pass --rebuild to (re)build the vector store.")


if __name__ == "__main__":
    main()
