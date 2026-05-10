"""
Loads documents from backend/information/, chunks and embeds them,then retrieves relevant context for each user query.

file types: .txt  .md  .pdf  .docx
Embeddings:           sentence-transformers (all-MiniLM-L6-v2, runs fully local/CPU)
uses cosine similarity via numpy, negating external db requirements e.g. chromaDB
"""

import os
import re
import logging
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed — RAG disabled. "
                   "Run: pip install sentence-transformers")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False



@dataclass
class Chunk:
    text: str
    source: str          # filename
    chunk_index: int


@dataclass
class VectorStore:
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: np.ndarray = None   # shape (N, D)



_store: VectorStore = VectorStore()
_embed_model: "SentenceTransformer | None" = None
_information_dir: str = ""



def init_rag(information_dir: str) -> bool:
    """
    Loads the embedding model and index all documents in backend/information.
    Returns True if RAG is ready, False if it could not initialise.
    """
    global _embed_model, _information_dir

    if not ST_AVAILABLE:
        logger.error("RAG unavailable: sentence-transformers missing.")
        return False

    _information_dir = information_dir
    os.makedirs(information_dir, exist_ok=True)

    logger.info("Loading embedding model (all-MiniLM-L6-v2)…")
    try:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    except Exception as exc:
        logger.error(f"Failed to load embedding model: {exc}")
        return False

    return _index_directory(information_dir)


def reload_index() -> Tuple[bool, str]:
    if _embed_model is None:
        return False, "Embedding model not loaded — RAG was not initialised."
    if not _information_dir:
        return False, "Information directory not set."
    ok = _index_directory(_information_dir)
    msg = (f"Index rebuilt — {len(_store.chunks)} chunks from "
           f"{_information_dir}.") if ok else "Indexing failed — check logs."
    return ok, msg


def _index_directory(directory: str) -> bool:
    global _store

    files = _discover_files(directory)
    if not files:
        logger.warning(f"No supported documents found in {directory}. "
                       "Add .txt / .md / .pdf / .docx files and reload.")
        _store = VectorStore()
        return True  # not an error, no knowledge yet

    all_chunks: List[Chunk] = []
    for filepath in files:
        filename = os.path.basename(filepath)
        text = _read_file(filepath)
        if not text:
            continue
        chunks = _chunk_text(text, source=filename)
        all_chunks.extend(chunks)
        logger.info(f"  {filename} → {len(chunks)} chunks")

    if not all_chunks:
        logger.warning("All files were empty or unreadable.")
        _store = VectorStore()
        return True

    logger.info(f"Embedding {len(all_chunks)} chunks…")
    texts = [c.text for c in all_chunks]
    try:
        embeddings = _embed_model.encode(texts, batch_size=32,
                                         show_progress_bar=False,
                                         normalize_embeddings=True)
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        return False

    _store = VectorStore(chunks=all_chunks,
                         embeddings=np.array(embeddings, dtype=np.float32))
    logger.info(f"RAG index ready — {len(all_chunks)} chunks across "
                f"{len(files)} file(s).")
    return True


def _discover_files(directory: str) -> List[str]:
    supported = {".txt", ".md", ".pdf", ".docx"}
    found = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in supported:
                found.append(os.path.join(root, name))
    return sorted(found)



def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md"):
            return _read_text(path)
        elif ext == ".pdf":
            return _read_pdf(path)
        elif ext == ".docx":
            return _read_docx(path)
    except Exception as exc:
        logger.error(f"Could not read {path}: {exc}")
    return ""


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_pdf(path: str) -> str:
    if not PYPDF_AVAILABLE:
        logger.warning("pypdf not installed — skipping PDF. "
                       "Run: pip install pypdf")
        return ""
    reader = pypdf.PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(path: str) -> str:
    if not DOCX_AVAILABLE:
        logger.warning("python-docx not installed — skipping .docx. "
                       "Run: pip install python-docx")
        return ""
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())



CHUNK_SIZE   = 1800   # characters
CHUNK_OVERLAP = 200   # characters


def _chunk_text(text: str, source: str) -> List[Chunk]:
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, source=source,
                                chunk_index=idx))
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks



def retrieve(query: str, top_k: int = 4) -> List[Chunk]:
    """
    Return the top_K (most probable) most relevant chunks for query.
    Returns an empty list if RAG is not ready or the store is empty.
    """
    if _embed_model is None or _store.embeddings is None or len(_store.chunks) == 0:
        return []

    query_vec = _embed_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)           # (1, D)


    scores = (_store.embeddings @ query_vec.T).squeeze()


    top_indices = np.argsort(scores)[::-1]
    selected: List[Chunk] = []
    seen_sources: dict = {}
    for i in top_indices:
        chunk = _store.chunks[int(i)]
        count = seen_sources.get(chunk.source, 0)
        if count < 2:
            selected.append(chunk)
            seen_sources[chunk.source] = count + 1
        if len(selected) >= top_k:
            break

    return selected


def build_context_block(chunks: List[Chunk]) -> str:
    if not chunks:
        return ""
    parts = []
    for chunk in chunks:
        parts.append(f"[Source: {chunk.source}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def is_ready() -> bool:
    return _embed_model is not None and _store.embeddings is not None


def index_stats() -> dict:
    return {
        "ready": is_ready(),
        "chunks": len(_store.chunks),
        "files": len({c.source for c in _store.chunks}),
        "information_dir": _information_dir,
    }
