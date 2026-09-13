"""
fs_tools.py
===========

Core file-system tools for the LLM Resume Assistant.

Each function in this module is designed to be exposed to an LLM as a
"tool" (a.k.a. function call). Every tool:

  * takes simple, JSON-serializable arguments (str, int, etc.)
  * returns a plain dict (JSON-serializable) so it can be sent straight
    back to the LLM as a tool result
  * never raises an exception to the caller -- all errors are caught and
    reported inside the returned dict as {"success": False, "error": ...}

Supported resume formats: .txt, .pdf, .docx

Dependencies (see requirements.txt):
  - PyPDF2      (PDF text extraction)
  - python-docx (DOCX text extraction)
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from docx import Document
except ImportError:  # pragma: no cover
    Document = None


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _extract_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf(filepath: str) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "PyPDF2 is not installed. Run: pip install PyPDF2"
        )
    reader = PdfReader(filepath)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def _extract_docx(filepath: str) -> str:
    if Document is None:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        )
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs]
    # Also pull text out of any tables, since resumes sometimes use them
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text:
                    paragraphs.append(cell.text)
    return "\n".join(paragraphs)


def _file_metadata(filepath: str) -> dict:
    stat = os.stat(filepath)
    return {
        "filename": os.path.basename(filepath),
        "path": os.path.abspath(filepath),
        "extension": os.path.splitext(filepath)[1].lower(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


# --------------------------------------------------------------------------
# Tool 1: read_file
# --------------------------------------------------------------------------

def read_file(filepath: str) -> dict:
    """
    Read a resume file (.txt, .pdf, or .docx) and return its text content
    plus metadata.

    Args:
        filepath: Path to the file to read.

    Returns:
        dict with keys:
            success (bool)
            content (str)            -- extracted text (only if success)
            metadata (dict)          -- filename, path, extension, size_bytes,
                                         modified (only if success)
            word_count (int)         -- word count of extracted text
            error (str)              -- present only if success is False
    """
    try:
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found: {filepath}"}

        if not os.path.isfile(filepath):
            return {"success": False, "error": f"Not a file: {filepath}"}

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".txt":
            content = _extract_txt(filepath)
        elif ext == ".pdf":
            content = _extract_pdf(filepath)
        elif ext == ".docx":
            content = _extract_docx(filepath)
        else:
            return {
                "success": False,
                "error": (
                    f"Unsupported file extension '{ext}'. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                ),
            }

        content = content.strip()

        return {
            "success": True,
            "content": content,
            "metadata": _file_metadata(filepath),
            "word_count": len(content.split()) if content else 0,
        }

    except Exception as exc:  # noqa: BLE001 - deliberately broad, tool must not raise
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# Tool 2: list_files
# --------------------------------------------------------------------------

def list_files(directory: str, extension: Optional[str] = None) -> list:
    """
    List files in a directory, optionally filtered by extension.

    Args:
        directory: Directory to list.
        extension: Optional extension filter, e.g. ".pdf" or "pdf".
                   Case-insensitive. If None, all files are returned.

    Returns:
        A list of dicts (one per file), each with:
            name, path, extension, size_bytes, modified
        On error, returns a list containing a single dict:
            [{"success": False, "error": "..."}]
    """
    try:
        if not os.path.isdir(directory):
            return [{"success": False, "error": f"Directory not found: {directory}"}]

        if extension is not None:
            extension = extension.lower()
            if not extension.startswith("."):
                extension = "." + extension

        results = []
        for name in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, name)
            if not os.path.isfile(full_path):
                continue

            file_ext = os.path.splitext(name)[1].lower()
            if extension is not None and file_ext != extension:
                continue

            meta = _file_metadata(full_path)
            results.append(meta)

        return results

    except Exception as exc:  # noqa: BLE001
        return [{"success": False, "error": f"{type(exc).__name__}: {exc}"}]


# --------------------------------------------------------------------------
# Tool 3: write_file
# --------------------------------------------------------------------------

def write_file(filepath: str, content: str) -> dict:
    """
    Write text content to a file, creating parent directories if needed.

    Args:
        filepath: Destination path.
        content: Text content to write.

    Returns:
        dict with keys:
            success (bool)
            path (str)          -- absolute path written (only if success)
            bytes_written (int) -- (only if success)
            error (str)         -- present only if success is False
    """
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "path": os.path.abspath(filepath),
            "bytes_written": len(content.encode("utf-8")),
        }

    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# Tool 4: search_in_file
# --------------------------------------------------------------------------

def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> dict:
    """
    Search for a keyword inside a file's content (case-insensitive) and
    return each match with surrounding context.

    Args:
        filepath: File to search (.txt, .pdf, or .docx).
        keyword: Keyword or phrase to search for.
        context_chars: Number of characters of context to include on each
                        side of a match (default 60).

    Returns:
        dict with keys:
            success (bool)
            filepath (str)
            keyword (str)
            match_count (int)
            matches (list[dict])   -- each: {"context": str, "position": int}
            error (str)            -- present only if success is False
    """
    try:
        read_result = read_file(filepath)
        if not read_result.get("success"):
            return {"success": False, "error": read_result.get("error", "Read failed")}

        content = read_result["content"]
        if not keyword:
            return {"success": False, "error": "keyword must be a non-empty string"}

        matches = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(content):
            start = max(0, m.start() - context_chars)
            end = min(len(content), m.end() + context_chars)
            snippet = content[start:end].replace("\n", " ").strip()
            matches.append({"context": snippet, "position": m.start()})

        return {
            "success": True,
            "filepath": os.path.abspath(filepath),
            "keyword": keyword,
            "match_count": len(matches),
            "matches": matches,
        }

    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# Manual smoke test (run: python fs_tools.py)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("== list_files('resumes') ==")
    print(json.dumps(list_files("resumes"), indent=2))

    print("\n== read_file('resumes/resume_michael_lee.txt') ==")
    print(json.dumps(read_file("resumes/resume_michael_lee.txt"), indent=2)[:600])

    print("\n== search_in_file(..., 'python') ==")
    print(json.dumps(search_in_file("resumes/resume_john_doe.pdf", "python"), indent=2))

    print("\n== write_file('output/test.txt', ...) ==")
    print(json.dumps(write_file("output/test.txt", "hello world"), indent=2))
