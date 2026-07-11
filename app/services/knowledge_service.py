
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeService:
    """
    Loads approved structured knowledge separately for each organisation.

    It:
    1. identifies the topic;
    2. identifies the course/product;
    3. identifies the customer category when required;
    4. asks clarification questions;
    5. returns one exact approved answer.
    """

    TOPIC_TERMS = {
        "tuition_fee": [
            "tuition fee",
            "tuition fees",
            "course fee",
            "course fees",
            "fee",
            "fees",
            "cost",
            "price"
        ],
        "course_duration": [
            "course duration",
            "programme duration",
            "how long",
            "course length",
            "programme length",
            "duration"
        ],
        "scholarship": [
            "scholarship",
            "financial support",
            "financial aid",
            "funding"
        ],
        "application_deadline": [
            "application deadline",
            "closing date",
            "last date",
            "when can i apply",
            "applications close",
            "deadline"
        ],
        "refund_policy": [
            "refund policy",
            "tuition refund",
            "withdrawal refund",
            "get a refund",
            "eligible for a refund",
            "refund"
        ],
        "warranty": [
            "warranty period",
            "product warranty",
            "manufacturer warranty",
            "warranty",
            "guarantee"
        ],
        "product_setup": [
            "set up",
            "setup",
            "install",
            "installation",
            "connect device",
            "configure"
        ],
        "return_policy": [
            "return policy",
            "return a product",
            "return the product",
            "send it back",
            "returns"
        ],
        "refund_time": [
            "refund time",
            "refund arrive",
            "refund take",
            "when will my refund",
            "working days",
            "refund processed"
        ],
        "support_hours": [
            "support hours",
            "opening hours",
            "technical support hours",
            "when is support available",
            "contact support"
        ]
    }

    REQUIRED_FIELDS = {
        "tuition_fee": ["subject", "audience"],
        "course_duration": ["subject"],
        "warranty": ["subject"],
        "product_setup": ["subject"]
    }

    SUBJECT_ALIASES = {
        "northbridge-university": {
            "MSc Artificial Intelligence": [
                "msc artificial intelligence",
                "artificial intelligence",
                "msc ai",
                "ai course",
                "ai programme",
                "ai"
            ],
            "MSc Data Science": [
                "msc data science",
                "data science",
                "data science course",
                "data science programme"
            ],
            "MSc Cyber Security": [
                "msc cyber security",
                "cyber security",
                "cybersecurity",
                "cyber security course"
            ],
            "MSc Business Analytics": [
                "msc business analytics",
                "business analytics",
                "business analytics course"
            ]
        },
        "novatech-solutions": {
            "NovaHub Mini": [
                "novahub mini",
                "nova hub mini",
                "novahub",
                "hub mini"
            ],
            "NovaCam Pro": [
                "novacam pro",
                "nova cam pro",
                "novacam",
                "camera pro"
            ]
        }
    }

    AUDIENCE_ALIASES = {
        "International": [
            "international",
            "overseas",
            "foreign student",
            "international student"
        ],
        "UK": [
            "uk student",
            "home student",
            "domestic student",
            "uk fee",
            "british student"
        ]
    }

    def __init__(self, knowledge_root: Path):
        self.knowledge_root = Path(knowledge_root)

        self.organisation_records: Dict[str, List[dict]] = {}
        self.organisation_indexes: Dict[str, dict] = {}

        self._load_knowledge()

    @staticmethod
    def _normalise(text: str) -> str:
        """Create a consistent lowercase representation of text."""

        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s-]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    @staticmethod
    def _contains_phrase(
        normalised_text: str,
        phrase: str
    ) -> bool:
        """Match complete words and phrases."""

        normalised_phrase = KnowledgeService._normalise(
            phrase
        )

        pattern = (
            r"(?<!\w)"
            + re.escape(normalised_phrase)
            + r"(?!\w)"
        )

        return bool(
            re.search(pattern, normalised_text)
        )

    def _load_knowledge(self) -> None:
        """Load each organisation's knowledge independently."""

        if not self.knowledge_root.exists():
            raise FileNotFoundError(
                f"Knowledge directory not found: "
                f"{self.knowledge_root}"
            )

        for knowledge_path in self.knowledge_root.rglob(
            "knowledge.json"
        ):
            organisation_id = knowledge_path.parent.name

            records = json.loads(
                knowledge_path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(records, list):
                raise ValueError(
                    f"Knowledge file must contain a list: "
                    f"{knowledge_path}"
                )

            validated_records = []

            for record in records:
                required_keys = {
                    "record_id",
                    "topic",
                    "answer",
                    "source",
                    "keywords"
                }

                missing_keys = (
                    required_keys - set(record.keys())
                )

                if missing_keys:
                    raise ValueError(
                        f"Record {record.get('record_id')} "
                        f"is missing: {missing_keys}"
                    )

                searchable_parts = [
                    record.get("topic", ""),
                    record.get("subject") or "",
                    record.get("audience") or "",
                    record.get("answer", ""),
                    " ".join(record.get("keywords", []))
                ]

                record_copy = record.copy()

                record_copy["_search_text"] = " ".join(
                    searchable_parts
                )

                validated_records.append(record_copy)

            self.organisation_records[
                organisation_id
            ] = validated_records

            search_texts = [
                record["_search_text"]
                for record in validated_records
            ]

            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            matrix = vectorizer.fit_transform(
                search_texts
            )

            self.organisation_indexes[
                organisation_id
            ] = {
                "vectorizer": vectorizer,
                "matrix": matrix
            }

    def available_organisations(self) -> List[str]:
        return sorted(
            self.organisation_records.keys()
        )

    def _detect_topic(
        self,
        organisation_id: str,
        message: str
    ) -> Optional[str]:
        """Detect only topics available to the organisation."""

        normalised_message = self._normalise(message)

        available_topics = {
            record["topic"]
            for record in self.organisation_records.get(
                organisation_id,
                []
            )
        }

        topic_scores = {}

        for topic in available_topics:
            terms = self.TOPIC_TERMS.get(
                topic,
                []
            )

            score = 0

            for term in terms:
                if self._contains_phrase(
                    normalised_message,
                    term
                ):
                    # Longer phrases are more informative.
                    score += max(
                        1,
                        len(term.split())
                    )

            if score > 0:
                topic_scores[topic] = score

        if not topic_scores:
            return None

        return max(
            topic_scores,
            key=topic_scores.get
        )

    def _detect_subject(
        self,
        organisation_id: str,
        message: str
    ) -> Optional[str]:
        """Detect a course or product name."""

        normalised_message = self._normalise(
            message
        )

        subject_aliases = self.SUBJECT_ALIASES.get(
            organisation_id,
            {}
        )

        matches = []

        for official_subject, aliases in (
            subject_aliases.items()
        ):
            for alias in aliases:
                if self._contains_phrase(
                    normalised_message,
                    alias
                ):
                    matches.append(
                        (
                            len(
                                self._normalise(
                                    alias
                                )
                            ),
                            official_subject
                        )
                    )

        if not matches:
            return None

        # Prefer the longest, most specific alias.
        matches.sort(reverse=True)

        return matches[0][1]

    def _detect_audience(
        self,
        message: str
    ) -> Optional[str]:
        """Detect UK or international fee category."""

        normalised_message = self._normalise(
            message
        )

        for audience, aliases in (
            self.AUDIENCE_ALIASES.items()
        ):
            for alias in aliases:
                if self._contains_phrase(
                    normalised_message,
                    alias
                ):
                    return audience

        return None

    def _available_subjects(
        self,
        organisation_id: str,
        topic: str
    ) -> List[str]:
        """Return valid subjects for a topic."""

        subjects = {
            record.get("subject")
            for record in self.organisation_records.get(
                organisation_id,
                []
            )
            if (
                record["topic"] == topic
                and record.get("subject")
            )
        }

        return sorted(subjects)

    def _clarification_for_subject(
        self,
        organisation_id: str,
        topic: str
    ) -> str:
        subjects = self._available_subjects(
            organisation_id,
            topic
        )

        if not subjects:
            return (
                "Which course or product are you asking about?"
            )

        options = ", ".join(subjects)

        if topic in {
            "tuition_fee",
            "course_duration"
        }:
            return (
                "Which course are you asking about? "
                f"Available courses are: {options}."
            )

        if topic in {
            "warranty",
            "product_setup"
        }:
            return (
                "Which product are you asking about? "
                f"Available products are: {options}."
            )

        return (
            "Which specific service or product "
            "are you asking about?"
        )

    @staticmethod
    def _clarification_for_audience() -> str:
        return (
            "Are you asking about the UK tuition fee "
            "or the international tuition fee?"
        )

    def _fallback_search(
        self,
        organisation_id: str,
        message: str,
        minimum_score: float = 0.32,
        minimum_margin: float = 0.04
    ) -> Optional[dict]:
        """
        Select one approved record only when the match is strong.

        It never returns raw document chunks.
        """

        records = self.organisation_records.get(
            organisation_id
        )

        index = self.organisation_indexes.get(
            organisation_id
        )

        if not records or not index:
            return None

        query_vector = index[
            "vectorizer"
        ].transform([message])

        scores = cosine_similarity(
            query_vector,
            index["matrix"]
        )[0]

        ranked_indexes = scores.argsort()[::-1]

        best_index = int(ranked_indexes[0])
        best_score = float(scores[best_index])

        second_score = 0.0

        if len(ranked_indexes) > 1:
            second_score = float(
                scores[int(ranked_indexes[1])]
            )

        margin = best_score - second_score

        if (
            best_score < minimum_score
            or margin < minimum_margin
        ):
            return None

        result = records[best_index].copy()
        result["confidence"] = round(
            best_score,
            4
        )

        return result

    def resolve(
        self,
        organisation_id: str,
        message: str,
        conversation_state: Optional[dict] = None
    ) -> dict:
        """
        Resolve one customer message.

        conversation_state carries information from a previous
        clarification question.
        """

        message = message.strip()

        if not message:
            return {
                "status": "error",
                "answer": "Please enter a question.",
                "source": None,
                "state": {}
            }

        if organisation_id not in (
            self.organisation_records
        ):
            return {
                "status": "unknown_organisation",
                "answer": (
                    "The requested organisation "
                    "could not be found."
                ),
                "source": None,
                "state": {}
            }

        state = dict(
            conversation_state or {}
        )

        topic = (
            self._detect_topic(
                organisation_id,
                message
            )
            or state.get("pending_topic")
        )

        subject = (
            self._detect_subject(
                organisation_id,
                message
            )
            or state.get("subject")
        )

        audience = (
            self._detect_audience(message)
            or state.get("audience")
        )

        # If no clear topic was detected, use a strict fallback.
        if topic is None:
            fallback_record = self._fallback_search(
                organisation_id,
                message
            )

            if fallback_record is None:
                return {
                    "status": "not_found",
                    "answer": (
                        "I could not find a reliable answer "
                        "in this organisation's approved "
                        "information. I can pass your question "
                        "to a member of staff."
                    ),
                    "source": None,
                    "record_id": None,
                    "confidence": None,
                    "state": {}
                }

            return {
                "status": "answered",
                "answer": fallback_record["answer"],
                "source": fallback_record["source"],
                "record_id": fallback_record["record_id"],
                "confidence": fallback_record[
                    "confidence"
                ],
                "state": {}
            }

        required_fields = self.REQUIRED_FIELDS.get(
            topic,
            []
        )

        if (
            "subject" in required_fields
            and subject is None
        ):
            return {
                "status": "needs_clarification",
                "answer": self._clarification_for_subject(
                    organisation_id,
                    topic
                ),
                "source": None,
                "record_id": None,
                "confidence": None,
                "state": {
                    "pending_topic": topic,
                    "subject": None,
                    "audience": audience
                }
            }

        if (
            "audience" in required_fields
            and audience is None
        ):
            return {
                "status": "needs_clarification",
                "answer": (
                    self._clarification_for_audience()
                ),
                "source": None,
                "record_id": None,
                "confidence": None,
                "state": {
                    "pending_topic": topic,
                    "subject": subject,
                    "audience": None
                }
            }

        matching_records = []

        for record in self.organisation_records[
            organisation_id
        ]:
            if record["topic"] != topic:
                continue

            if (
                "subject" in required_fields
                and record.get("subject") != subject
            ):
                continue

            if (
                "audience" in required_fields
                and record.get("audience") != audience
            ):
                continue

            matching_records.append(record)

        if len(matching_records) == 1:
            record = matching_records[0]

            return {
                "status": "answered",
                "answer": record["answer"],
                "source": record["source"],
                "record_id": record["record_id"],
                "confidence": 1.0,
                "state": {}
            }

        if len(matching_records) > 1:
            return {
                "status": "needs_clarification",
                "answer": (
                    "I found more than one possible answer. "
                    "Could you provide a little more detail?"
                ),
                "source": None,
                "record_id": None,
                "confidence": None,
                "state": {
                    "pending_topic": topic,
                    "subject": subject,
                    "audience": audience
                }
            }

        return {
            "status": "not_found",
            "answer": (
                "I could not find this information in the "
                "organisation's approved knowledge base. "
                "I can pass the question to a member of staff."
            ),
            "source": None,
            "record_id": None,
            "confidence": None,
            "state": {}
        }
