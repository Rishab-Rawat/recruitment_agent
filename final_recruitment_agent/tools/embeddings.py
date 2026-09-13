"""
embeddings.py
A small adapter layer so resume_rag.py and job_matcher.py never call a specific
embedding provider directly -- they call `get_embedding_backend()` and get back
something with an `.encode(texts) -> list[list[float]]` method.

Backends:
  - SentenceTransformerBackend : local HuggingFace model (all-MiniLM-L6-v2).
        This is the intended default per the assignment spec. It requires
        downloading model weights from huggingface.co on first use.
  - OpenAIEmbeddingBackend     : OpenAI's text-embedding-3-small via API key.
  - CohereEmbeddingBackend     : Cohere's embed-english-v3.0 via API key.
  - TfidfSvdEmbeddingBackend   : a pure scikit-learn TF-IDF + SVD (LSA)
        embedding. No network access and no API key required. This is the
        fallback used automatically in this sandboxed environment, where
        egress to huggingface.co/api.openai.com/api.cohere.ai is blocked by
        network policy (confirmed: HTTP 403 host_not_allowed). It is NOT
        what the assignment asks for as the primary approach -- it exists so
        the pipeline is runnable end-to-end here. Swap it out in one line
        (see get_embedding_backend) when running with real network/API
        access, which is the expected way to run this in production.

Every backend exposes the same two things so the rest of the codebase is
backend-agnostic:
    backend.encode(texts: list[str]) -> list[list[float]]
    backend.dim  (int, embedding dimensionality)
"""
from __future__ import annotations

import os
from typing import Protocol


class EmbeddingBackend(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerBackend:
    """HuggingFace sentence-transformers, run locally. Requires model download."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()


class OpenAIEmbeddingBackend:
    """OpenAI text-embedding-3-small. Requires OPENAI_API_KEY."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name
        self.dim = 1536

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


class CohereEmbeddingBackend:
    """Cohere embed-english-v3.0. Requires COHERE_API_KEY."""

    def __init__(self, model_name: str = "embed-english-v3.0"):
        import cohere
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.model_name = model_name
        self.dim = 1024

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embed(
            texts=texts, model=self.model_name, input_type="search_document"
        )
        return resp.embeddings


class TfidfSvdEmbeddingBackend:
    """
    Local, network-free fallback: TF-IDF vectors reduced to a dense space
    with truncated SVD (i.e. classic LSA). Fit once on a representative
    corpus (all resume chunks + a generic English background vocabulary),
    then every subsequent `.encode()` call reuses that fitted vectorizer/SVD
    to produce embeddings in the same vector space. Cosine similarity on the
    resulting vectors behaves like a lexical-semantic search: it will not
    generalize to unseen synonyms the way a neural embedding does, but it is
    genuinely learned (not hardcoded rules) and needs no internet or API key.
    """

    def __init__(self, n_components: int = 128, corpus: list[str] | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self.dim = n_components
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000, sublinear_tf=True
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._normalizer = Normalizer(copy=False)
        self._fitted = False
        if corpus:
            self.fit(corpus)

    def fit(self, corpus: list[str]):
        tfidf = self._vectorizer.fit_transform(corpus)
        n_components = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        if n_components < self.dim:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=max(2, n_components), random_state=42)
            self.dim = self._svd.n_components
        reduced = self._svd.fit_transform(tfidf)
        self._normalizer.fit(reduced)
        self._fitted = True

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            # Nothing fitted yet (encode called before fit) -- fit on this batch.
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        reduced = self._svd.transform(tfidf)
        normalized = self._normalizer.transform(reduced)
        return normalized.tolist()

    def save(self, path: str):
        import joblib
        joblib.dump({
            "vectorizer": self._vectorizer, "svd": self._svd,
            "normalizer": self._normalizer, "dim": self.dim,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TfidfSvdEmbeddingBackend":
        import joblib
        state = joblib.load(path)
        obj = cls(n_components=state["dim"])
        obj._vectorizer = state["vectorizer"]
        obj._svd = state["svd"]
        obj._normalizer = state["normalizer"]
        obj._fitted = True
        return obj


def get_embedding_backend(prefer: str = "auto", corpus_for_fallback: list[str] | None = None,
                           tfidf_cache_path: str | None = None):
    """
    Resolve which backend to use.
      prefer="auto"       -> try sentence-transformers, then OpenAI, then Cohere,
                              then fall back to the local TF-IDF+SVD backend.
      prefer="huggingface"/"openai"/"cohere"/"tfidf" -> force that backend.

    In THIS sandboxed environment, huggingface/openai/cohere all fail because
    egress to their hosts is blocked by network policy, so "auto" resolves to
    the tfidf backend here. With normal internet access, change the call in
    resume_rag.py / job_matcher.py to prefer="huggingface" (or pass an
    OPENAI_API_KEY / COHERE_API_KEY and prefer="openai"/"cohere") to use a real
    embedding model -- no other code needs to change, since every backend
    implements the same .encode() interface.
    """
    if prefer == "tfidf":
        if tfidf_cache_path and os.path.exists(tfidf_cache_path):
            return TfidfSvdEmbeddingBackend.load(tfidf_cache_path)
        backend = TfidfSvdEmbeddingBackend(corpus=corpus_for_fallback)
        if tfidf_cache_path:
            backend.save(tfidf_cache_path)
        return backend
    if prefer == "huggingface":
        return SentenceTransformerBackend()
    if prefer == "openai":
        return OpenAIEmbeddingBackend()
    if prefer == "cohere":
        return CohereEmbeddingBackend()

    # auto
    for factory in (SentenceTransformerBackend, OpenAIEmbeddingBackend, CohereEmbeddingBackend):
        try:
            return factory()
        except Exception:
            continue
    print(
        "[embeddings] No hosted embedding provider was reachable/configured "
        "(huggingface.co / api.openai.com / api.cohere.ai). Falling back to a "
        "local TF-IDF+SVD embedding backend -- see embeddings.py docstring."
    )
    if tfidf_cache_path and os.path.exists(tfidf_cache_path):
        return TfidfSvdEmbeddingBackend.load(tfidf_cache_path)
    backend = TfidfSvdEmbeddingBackend(corpus=corpus_for_fallback)
    if tfidf_cache_path:
        backend.save(tfidf_cache_path)
    return backend
