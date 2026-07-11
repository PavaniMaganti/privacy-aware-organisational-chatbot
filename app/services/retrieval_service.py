
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalService:
    """Organisation-isolated document retrieval service."""

    def __init__(self, documents_root: Path):
        self.documents_root = documents_root
        self.organisation_data: Dict[str, dict] = {}

        self._load_documents()

    @staticmethod
    def _split_into_chunks(text: str) -> List[str]:
        """Split document text into paragraph-based chunks."""

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= 900:
                chunks.append(paragraph)
            else:
                sentences = paragraph.split(". ")
                current_chunk = ""

                for sentence in sentences:
                    proposed_chunk = (
                        current_chunk + " " + sentence
                    ).strip()

                    if len(proposed_chunk) <= 900:
                        current_chunk = proposed_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk)

        return chunks

    def _load_documents(self) -> None:
        """Load and index documents separately for each organisation."""

        if not self.documents_root.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_root}"
            )

        organisation_chunks: Dict[str, List[dict]] = {}

        for file_path in self.documents_root.rglob("*.txt"):
            relative_path = file_path.relative_to(
                self.documents_root
            )

            if len(relative_path.parts) < 2:
                continue

            organisation_id = relative_path.parts[0]

            document_text = file_path.read_text(
                encoding="utf-8"
            )

            chunks = self._split_into_chunks(document_text)

            organisation_chunks.setdefault(
                organisation_id,
                []
            )

            for chunk_number, chunk in enumerate(chunks):
                organisation_chunks[organisation_id].append({
                    "organisation_id": organisation_id,
                    "source": file_path.name,
                    "chunk_number": chunk_number,
                    "text": chunk
                })

        for organisation_id, chunks in organisation_chunks.items():
            chunk_texts = [
                chunk["text"]
                for chunk in chunks
            ]

            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            document_matrix = vectorizer.fit_transform(
                chunk_texts
            )

            self.organisation_data[organisation_id] = {
                "chunks": chunks,
                "vectorizer": vectorizer,
                "matrix": document_matrix
            }

    def search(
        self,
        organisation_id: str,
        question: str,
        top_k: int = 3,
        minimum_score: float = 0.08
    ) -> List[dict]:
        """
        Search only documents belonging to the specified organisation.
        """

        organisation = self.organisation_data.get(
            organisation_id
        )

        if organisation is None:
            return []

        query_vector = organisation[
            "vectorizer"
        ].transform([question])

        scores = cosine_similarity(
            query_vector,
            organisation["matrix"]
        )[0]

        ranked_indexes = scores.argsort()[::-1]

        results = []

        for index in ranked_indexes[:top_k]:
            score = float(scores[index])

            if score < minimum_score:
                continue

            chunk = organisation["chunks"][index].copy()
            chunk["score"] = round(score, 4)

            results.append(chunk)

        return results

    def available_organisations(self) -> List[str]:
        return sorted(self.organisation_data.keys())
