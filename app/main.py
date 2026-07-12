
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.services.intent_service import IntentService
from app.services.knowledge_service import KnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "trained_models"
    / "intent_classifier.joblib"
)

KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
STATIC_DIR = PROJECT_ROOT / "app" / "static"


intent_service = IntentService(MODEL_PATH)
knowledge_service = KnowledgeService(KNOWLEDGE_ROOT)


app = FastAPI(
    title="Privacy-Aware Organisational Support Assistant",
    description=(
        "A multi-organisation customer-support chatbot with "
        "intent classification, approved knowledge retrieval, "
        "clarification, security controls and human escalation."
    ),
    version="2.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


ORGANISATIONS = {
    "northbridge-university": {
        "id": "northbridge-university",
        "name": "Northbridge University",
        "description": (
            "Course, fee, scholarship, deadline and "
            "student-support information."
        ),
        "welcome_message": (
            "Hello. I can help with approved information about "
            "courses, tuition fees, scholarships and applications."
        ),
        "primary_colour": "#243b6b",
        "secondary_colour": "#e7ecf6"
    },
    "novatech-solutions": {
        "id": "novatech-solutions",
        "name": "NovaTech Solutions",
        "description": (
            "Product, warranty, return and "
            "technical-support information."
        ),
        "welcome_message": (
            "Hello. I can help with approved information about "
            "NovaTech products, warranties, returns and support."
        ),
        "primary_colour": "#185c55",
        "secondary_colour": "#e2f2ef"
    }
}


# Temporary prototype storage.
# This will later be replaced by a database.
conversation_states: Dict[Tuple[str, str], dict] = {}
escalations = []
security_events = []


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=8,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$"
    )

    message: str = Field(
        min_length=1,
        max_length=1000
    )


class ChatResponse(BaseModel):
    session_id: str
    status: str
    reply: str
    predicted_intent: Optional[str] = None
    action: str
    source: Optional[str] = None
    record_id: Optional[str] = None
    escalation_id: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: str = Field(
        min_length=8,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$"
    )


SUSPICIOUS_PATTERNS = [
    r"ignore (all|your|the|previous) instructions",
    r"reveal (the )?system prompt",
    r"show (me )?(the )?(api key|password|secret token)",
    r"bypass (the )?(security|access control)",
    r"give me (administrator|admin) access",
    r"show me (all )?(confidential|private) files",
    r"another (customer|student|user).*(details|information|records)",
    r"export (the )?(complete|entire) database",
    r"disable (the )?(privacy|security)",
    r"delete (the )?audit logs",
    r"pretend i am (an )?(administrator|admin)"
]


def looks_suspicious(message: str) -> bool:
    normalised_message = message.lower().strip()

    return any(
        re.search(pattern, normalised_message)
        for pattern in SUSPICIOUS_PATTERNS
    )


def redact_basic_pii(message: str) -> str:
    redacted = message

    redacted = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REDACTED]",
        redacted
    )

    redacted = re.sub(
        r"(?<!\d)(?:\+?44\s?|0)7\d{3}\s?\d{6}(?!\d)",
        "[PHONE REDACTED]",
        redacted
    )

    redacted = re.sub(
        r"\b(?:\d[ -]*?){13,19}\b",
        "[NUMBER REDACTED]",
        redacted
    )

    return redacted


def create_escalation(
    organisation_id: str,
    session_id: str,
    message: str,
    reason: str
) -> str:
    escalation_id = (
        "ESC-"
        + uuid4().hex[:8].upper()
    )

    escalations.append({
        "escalation_id": escalation_id,
        "organisation_id": organisation_id,
        "session_id": session_id,
        "message": redact_basic_pii(message),
        "reason": reason,
        "status": "OPEN",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    })

    return escalation_id


def record_security_event(
    organisation_id: str,
    session_id: str,
    message: str
) -> None:
    security_events.append({
        "organisation_id": organisation_id,
        "session_id": session_id,
        "message": redact_basic_pii(message),
        "event_type": "SUSPICIOUS_REQUEST_BLOCKED",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    })


def convert_knowledge_result(
    session_key: Tuple[str, str],
    session_id: str,
    predicted_intent: str,
    result: dict
) -> ChatResponse:
    status = result.get("status")

    if status == "needs_clarification":
        conversation_states[session_key] = (
            result.get("state") or {}
        )

        return ChatResponse(
            session_id=session_id,
            status="needs_clarification",
            reply=result["answer"],
            predicted_intent=predicted_intent,
            action="ASK_CLARIFICATION"
        )

    conversation_states.pop(
        session_key,
        None
    )

    if status == "answered":
        return ChatResponse(
            session_id=session_id,
            status="answered",
            reply=result["answer"],
            predicted_intent=predicted_intent,
            action="RETURN_APPROVED_ANSWER",
            source=result.get("source"),
            record_id=result.get("record_id")
        )

    return ChatResponse(
        session_id=session_id,
        status="not_found",
        reply=result.get(
            "answer",
            (
                "I could not find a reliable answer in the "
                "organisation's approved information."
            )
        ),
        predicted_intent=predicted_intent,
        action="OFFER_HUMAN_SUPPORT"
    )


@app.get(
    "/",
    response_class=HTMLResponse
)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "organisations": list(
                ORGANISATIONS.values()
            )
        }
    )


@app.get(
    "/org/{organisation_id}/chat",
    response_class=HTMLResponse
)
def organisation_chat(
    request: Request,
    organisation_id: str
):
    organisation = ORGANISATIONS.get(
        organisation_id
    )

    if organisation is None:
        raise HTTPException(
            status_code=404,
            detail="Organisation not found."
        )

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "organisation": organisation
        }
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "intent_model_loaded": True,
        "knowledge_loaded": True,
        "organisations": (
            knowledge_service.available_organisations()
        )
    }


@app.post(
    "/api/org/{organisation_id}/reset"
)
def reset_conversation(
    organisation_id: str,
    request: ResetRequest
):
    if organisation_id not in ORGANISATIONS:
        raise HTTPException(
            status_code=404,
            detail="Organisation not found."
        )

    session_key = (
        organisation_id,
        request.session_id
    )

    conversation_states.pop(
        session_key,
        None
    )

    return {
        "status": "reset",
        "session_id": request.session_id
    }


@app.post(
    "/api/org/{organisation_id}/chat",
    response_model=ChatResponse
)
def chat(
    organisation_id: str,
    request: ChatRequest
):
    if organisation_id not in ORGANISATIONS:
        raise HTTPException(
            status_code=404,
            detail="Organisation not found."
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    session_key = (
        organisation_id,
        request.session_id
    )

    # Rule-based security check runs before all other processing.
    if looks_suspicious(message):
        conversation_states.pop(
            session_key,
            None
        )

        record_security_event(
            organisation_id,
            request.session_id,
            message
        )

        return ChatResponse(
            session_id=request.session_id,
            status="blocked",
            reply=(
                "I cannot assist with requests to access private, "
                "restricted or confidential information. "
                "This request has been blocked."
            ),
            predicted_intent="SUSPICIOUS_REQUEST",
            action="BLOCK_AND_AUDIT"
        )

    # A clarification conversation must continue before
    # the message is sent back through general routing.
    existing_state = conversation_states.get(
        session_key
    )

    if existing_state:
        if message.lower() in {
            "cancel",
            "reset",
            "start again",
            "start over"
        }:
            conversation_states.pop(
                session_key,
                None
            )

            return ChatResponse(
                session_id=request.session_id,
                status="reset",
                reply=(
                    "The previous question has been cancelled. "
                    "What would you like help with?"
                ),
                predicted_intent=None,
                action="RESET_CONVERSATION"
            )

        result = knowledge_service.resolve(
            organisation_id=organisation_id,
            message=message,
            conversation_state=existing_state
        )

        return convert_knowledge_result(
            session_key=session_key,
            session_id=request.session_id,
            predicted_intent="PUBLIC_INFORMATION",
            result=result
        )

    try:
        predicted_intent = intent_service.predict(
            message
        )

    except Exception:
        predicted_intent = "UNKNOWN"

    if predicted_intent == "SUSPICIOUS_REQUEST":
        record_security_event(
            organisation_id,
            request.session_id,
            message
        )

        return ChatResponse(
            session_id=request.session_id,
            status="blocked",
            reply=(
                "I cannot assist with requests to access private, "
                "restricted or confidential information."
            ),
            predicted_intent=predicted_intent,
            action="BLOCK_AND_AUDIT"
        )

    if predicted_intent == "CUSTOMER_SPECIFIC":
        return ChatResponse(
            session_id=request.session_id,
            status="authentication_required",
            reply=(
                "This request appears to involve personal account "
                "information. For your security, customer-specific "
                "information requires verified authentication and "
                "cannot be accessed through this public demonstration."
            ),
            predicted_intent=predicted_intent,
            action="REQUIRE_AUTHENTICATION"
        )

    if predicted_intent in {
        "COMPLAINT",
        "HUMAN_ESCALATION"
    }:
        escalation_id = create_escalation(
            organisation_id=organisation_id,
            session_id=request.session_id,
            message=message,
            reason=predicted_intent
        )

        return ChatResponse(
            session_id=request.session_id,
            status="escalated",
            reply=(
                "I have prepared this request for human support. "
                f"Your reference number is {escalation_id}."
            ),
            predicted_intent=predicted_intent,
            action="CREATE_ESCALATION",
            escalation_id=escalation_id
        )

    # Try approved knowledge even if the classifier predicted
    # OUT_OF_SCOPE. This prevents a weak model prediction from
    # incorrectly blocking a valid organisation question.
    knowledge_result = knowledge_service.resolve(
        organisation_id=organisation_id,
        message=message
    )

    if knowledge_result.get("status") in {
        "answered",
        "needs_clarification"
    }:
        return convert_knowledge_result(
            session_key=session_key,
            session_id=request.session_id,
            predicted_intent=predicted_intent,
            result=knowledge_result
        )

    if predicted_intent == "OUT_OF_SCOPE":
        return ChatResponse(
            session_id=request.session_id,
            status="refused",
            reply=(
                "I can only help with approved information provided "
                "by this organisation. I cannot answer that question."
            ),
            predicted_intent=predicted_intent,
            action="SAFE_REFUSAL"
        )

    return ChatResponse(
        session_id=request.session_id,
        status="not_found",
        reply=(
            "I could not find a reliable answer in this "
            "organisation's approved information. "
            "Please ask in a different way or request human support."
        ),
        predicted_intent=predicted_intent,
        action="OFFER_HUMAN_SUPPORT"
    )
