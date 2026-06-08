
from pathlib import Path
from typing import Dict, List

import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import KB_DIR


SUPPORTED_KB_EXTENSIONS = {
    ".txt",
    ".pdf",
}


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    document = fitz.open(path)
    texts = []

    for page in document:
        texts.append(page.get_text())

    document.close()

    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks


class LocalTfidfRetriever:
    def __init__(self, kb_dir: str | Path | None = None):
        self.kb_dir = Path(kb_dir) if kb_dir is not None else KB_DIR
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None

        self._load_kb()

    def _load_kb(self):
        if not self.kb_dir.exists():
            print(f"[RAG] KB folder does not exist: {self.kb_dir}")
            return

        for path in self.kb_dir.rglob("*"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            if suffix not in SUPPORTED_KB_EXTENSIONS:
                continue

            try:
                if suffix == ".txt":
                    text = read_txt(path)
                elif suffix == ".pdf":
                    text = read_pdf(path)
                else:
                    continue

                chunks = chunk_text(text)

                for index, chunk in enumerate(chunks):
                    self.documents.append(
                        {
                            "source": str(path),
                            "chunk_id": index,
                            "text": chunk,
                        }
                    )

            except Exception as error:
                print(f"[RAG] Could not read {path}: {error}")

        if self.documents:
            texts = [doc["text"] for doc in self.documents]
            self.matrix = self.vectorizer.fit_transform(texts)

        print(f"[RAG] Indexed KB chunks: {len(self.documents)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.documents or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            doc = self.documents[index]

            results.append(
                {
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
                    "score": score,
                    "text": doc["text"],
                }
            )

        return results
