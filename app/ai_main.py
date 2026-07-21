import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.services.gemini_service import GeminiService
from app.services.intent_service import IntentService
from app.services.knowledge_service import KnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "trained_models" / "intent_classifier.joblib"
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
STATIC_DIR = PROJECT_ROOT / "app" / "static"

intent_service = IntentService(MODEL_PATH)
knowledge_service = KnowledgeService(KNOWLEDGE_ROOT)
gemini_service = GeminiService()

app = FastAPI(
    title="Privacy-Aware Organisational Support Assistant",
    description=(
        "A multi-organisation conversational assistant with intent classification, "
        "approved knowledge retrieval, privacy controls, clarification and escalation."
    ),
    version="3.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ORGANISATIONS = {
    "demo-university-a": {
        "id": "demo-university-a",
        "knowledge_id": "northbridge-university",
        "name": "Demo University A",
        "description": (
            "Fictional demo organisation for course, fee, scholarship, deadline "
            "and student-support information."
        ),
        "welcome_message": (
            "Hello. This is a fictional university demonstration. I can help with "
            "approved information about courses, tuition fees, scholarships and applications."
        ),
        "primary_colour": "#243b6b",
        "secondary_colour": "#e7ecf6",
    },
    "demo-technology-company-b": {
        "id": "demo-technology-company-b",
        "knowledge_id": "novatech-solutions",
        "name": "Demo Technology Company B",
        "description": (
            "Fictional demo organisation for product, warranty, return and "
            "technical-support information."
        ),
        "welcome_message": (
            "Hello. This is a fictional technology-company demonstration. I can help "
            "with approved information about DemoHub A1, DemoCam B1, warranties, "
            "returns and support."
        ),
        "primary_colour": "#185c55",
        "secondary_colour": "#e2f2ef",
    },
}

LEGACY_ROUTE_REDIRECTS = {
    "northbridge-university": "demo-university-a",
    "novatech-solutions": "demo-technology-company-b",
}

PUBLIC_TEXT_REPLACEMENTS = {
    "Northbridge University": "Demo University A",
    "Northbridge": "Demo University A",
    "NovaTech Solutions": "Demo Technology Company B",
    "NovaTech": "Demo Technology Company B",
    "NovaHub Mini": "DemoHub A1",
    "NovaCam Pro": "DemoCam B1",
}

INTERNAL_TEXT_REPLACEMENTS = {
    "Demo University A": "Northbridge University",
    "Demo Technology Company B": "NovaTech Solutions",
    "DemoHub A1": "NovaHub Mini",
    "DemoCam B1": "NovaCam Pro",
}

# Prototype in-memory storage. It resets when Render restarts.
pending_states: Dict[Tuple[str, str], dict] = {}
conversation_contexts: Dict[Tuple[str, str], dict] = {}
conversation_histories: Dict[Tuple[str, str], List[dict]] = {}
escalations: List[dict] = []
security_events: List[dict] = []


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=8,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    message: str = Field(min_length=1, max_length=1000)


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
        pattern=r"^[a-zA-Z0-9_-]+$",
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
    r"pretend i am (an )?(administrator|admin)",
]


def looks_suspicious(message: str) -> bool:
    normalised_message = message.lower().strip()
    return any(re.search(pattern, normalised_message) for pattern in SUSPICIOUS_PATTERNS)


def knowledge_organisation_id(public_organisation_id: str) -> str:
    return ORGANISATIONS[public_organisation_id]["knowledge_id"]


def publicise_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    updated = value
    for internal_name, public_name in PUBLIC_TEXT_REPLACEMENTS.items():
        updated = updated.replace(internal_name, public_name)
    return updated


def internalise_message(message: str) -> str:
    updated = message
    for public_name, internal_name in INTERNAL_TEXT_REPLACEMENTS.items():
        updated = re.sub(
            re.escape(public_name),
            internal_name,
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def is_greeting(message: str) -> bool:
    """Recognise harmless greetings before the ML intent classifier runs."""

    normalised_message = re.sub(r"[^a-z\s]", "", message.lower()).strip()

    greetings = {
        "hi",
        "hello",
        "hey",
        "hiya",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }

    return normalised_message in greetings


def redact_basic_pii(message: str) -> str:
    redacted = message
    redacted = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?<!\d)(?:\+?44\s?|0)7\d{3}\s?\d{6}(?!\d)",
        "[PHONE REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\b(?:\d[ -]*?){13,19}\b",
        "[NUMBER REDACTED]",
        redacted,
    )
    return redacted


def add_history(session_key: Tuple[str, str], role: str, content: str) -> None:
    history = conversation_histories.setdefault(session_key, [])
    history.append({"role": role, "content": redact_basic_pii(content)})
    conversation_histories[session_key] = history[-10:]


def get_record(organisation_id: str, record_id: Optional[str]) -> Optional[dict]:
    if not record_id:
        return None
    internal_id = knowledge_organisation_id(organisation_id)
    for record in knowledge_service.organisation_records.get(internal_id, []):
        if record.get("record_id") == record_id:
            return record
    return None


def remember_record_context(
    session_key: Tuple[str, str],
    organisation_id: str,
    record_id: Optional[str],
) -> None:
    record = get_record(organisation_id, record_id)
    if not record:
        return
    conversation_contexts[session_key] = {
        "pending_topic": record.get("topic"),
        "subject": record.get("subject"),
        "audience": record.get("audience"),
    }


def create_escalation(
    organisation_id: str,
    session_id: str,
    message: str,
    reason: str,
) -> str:
    escalation_id = "ESC-" + uuid4().hex[:8].upper()
    escalations.append(
        {
            "escalation_id": escalation_id,
            "organisation_id": organisation_id,
            "session_id": session_id,
            "message": redact_basic_pii(message),
            "reason": reason,
            "status": "OPEN",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return escalation_id


def record_security_event(
    organisation_id: str,
    session_id: str,
    message: str,
) -> None:
    security_events.append(
        {
            "organisation_id": organisation_id,
            "session_id": session_id,
            "message": redact_basic_pii(message),
            "event_type": "SUSPICIOUS_REQUEST_BLOCKED",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def convert_knowledge_result(
    organisation_id: str,
    session_key: Tuple[str, str],
    session_id: str,
    user_message: str,
    predicted_intent: str,
    result: dict,
) -> ChatResponse:
    status = result.get("status")

    if status == "needs_clarification":
        pending_states[session_key] = result.get("state") or {}
        clarification_reply = publicise_text(result["answer"]) or result["answer"]
        add_history(session_key, "user", user_message)
        add_history(session_key, "assistant", clarification_reply)
        return ChatResponse(
            session_id=session_id,
            status="needs_clarification",
            reply=clarification_reply,
            predicted_intent=predicted_intent,
            action="ASK_CLARIFICATION",
        )

    pending_states.pop(session_key, None)

    if status == "answered":
        organisation_name = ORGANISATIONS[organisation_id]["name"]
        history = conversation_histories.get(session_key, [])
        safe_user_message = redact_basic_pii(user_message)
        source = publicise_text(result.get("source")) or "Approved organisation information"
        approved_answer = publicise_text(result["answer"]) or result["answer"]

        reply = gemini_service.create_grounded_reply(
            organisation_name=organisation_name,
            user_message=safe_user_message,
            approved_answer=approved_answer,
            source=source,
            history=history,
        )

        reply = publicise_text(reply) or reply
        add_history(session_key, "user", safe_user_message)
        add_history(session_key, "assistant", reply)
        remember_record_context(
            session_key,
            organisation_id,
            result.get("record_id"),
        )

        return ChatResponse(
            session_id=session_id,
            status="answered",
            reply=reply,
            predicted_intent=predicted_intent,
            action="RETURN_APPROVED_ANSWER",
            source=publicise_text(result.get("source")),
            record_id=result.get("record_id"),
        )

    reply = publicise_text(
        result.get(
            "answer",
            "I could not find a reliable answer in this organisation's approved information.",
        )
    ) or "I could not find a reliable answer in this organisation's approved information."
    add_history(session_key, "user", user_message)
    add_history(session_key, "assistant", reply)
    return ChatResponse(
        session_id=session_id,
        status="not_found",
        reply=reply,
        predicted_intent=predicted_intent,
        action="OFFER_HUMAN_SUPPORT",
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"organisations": list(ORGANISATIONS.values())},
    )


@app.get("/org/{organisation_id}/chat", response_class=HTMLResponse)
def organisation_chat(request: Request, organisation_id: str):
    if organisation_id in LEGACY_ROUTE_REDIRECTS:
        new_id = LEGACY_ROUTE_REDIRECTS[organisation_id]
        return RedirectResponse(url=f"/org/{new_id}/chat", status_code=307)

    organisation = ORGANISATIONS.get(organisation_id)
    if organisation is None:
        raise HTTPException(status_code=404, detail="Organisation not found.")
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"organisation": organisation},
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "intent_model_loaded": True,
        "knowledge_loaded": True,
        "gemini_available": gemini_service.available,
        "organisations": [organisation["id"] for organisation in ORGANISATIONS.values()],
        "fictional_demo": True,
    }


@app.post("/api/org/{organisation_id}/reset")
def reset_conversation(organisation_id: str, request: ResetRequest):
    organisation_id = LEGACY_ROUTE_REDIRECTS.get(organisation_id, organisation_id)
    if organisation_id not in ORGANISATIONS:
        raise HTTPException(status_code=404, detail="Organisation not found.")

    session_key = (organisation_id, request.session_id)
    pending_states.pop(session_key, None)
    conversation_contexts.pop(session_key, None)
    conversation_histories.pop(session_key, None)

    return {"status": "reset", "session_id": request.session_id}


@app.post(
    "/api/org/{organisation_id}/chat",
    response_model=ChatResponse,
)
def chat(organisation_id: str, request: ChatRequest):
    organisation_id = LEGACY_ROUTE_REDIRECTS.get(organisation_id, organisation_id)
    if organisation_id not in ORGANISATIONS:
        raise HTTPException(status_code=404, detail="Organisation not found.")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_key = (organisation_id, request.session_id)

    if looks_suspicious(message):
        pending_states.pop(session_key, None)
        record_security_event(organisation_id, request.session_id, message)
        reply = (
            "I cannot assist with requests to access private, restricted or "
            "confidential information. This request has been blocked."
        )
        add_history(session_key, "user", message)
        add_history(session_key, "assistant", reply)
        return ChatResponse(
            session_id=request.session_id,
            status="blocked",
            reply=reply,
            predicted_intent="SUSPICIOUS_REQUEST",
            action="BLOCK_AND_AUDIT",
        )

    # Handle greetings before the small intent model. The model was trained on
    # support/security intents and can misclassify short messages such as "hi".
    if is_greeting(message):
        organisation_name = ORGANISATIONS[organisation_id]["name"]
        reply = (
            f"Hello! I’m the {organisation_name} support assistant. "
            "What would you like help with today?"
        )
        add_history(session_key, "user", message)
        add_history(session_key, "assistant", reply)
        return ChatResponse(
            session_id=request.session_id,
            status="greeting",
            reply=reply,
            predicted_intent="GREETING",
            action="WELCOME",
        )

    existing_state = pending_states.get(session_key)
    if existing_state:
        if message.lower() in {"cancel", "reset", "start again", "start over"}:
            pending_states.pop(session_key, None)
            reply = "The previous question has been cancelled. What would you like help with?"
            add_history(session_key, "user", message)
            add_history(session_key, "assistant", reply)
            return ChatResponse(
                session_id=request.session_id,
                status="reset",
                reply=reply,
                predicted_intent=None,
                action="RESET_CONVERSATION",
            )

        result = knowledge_service.resolve(
            organisation_id=knowledge_organisation_id(organisation_id),
            message=internalise_message(message),
            conversation_state=existing_state,
        )
        return convert_knowledge_result(
            organisation_id=organisation_id,
            session_key=session_key,
            session_id=request.session_id,
            user_message=message,
            predicted_intent="PUBLIC_INFORMATION",
            result=result,
        )

    try:
        predicted_intent = intent_service.predict(message)
    except Exception:
        predicted_intent = "UNKNOWN"

    # The small ML classifier is advisory only for security. Hard blocking is
    # controlled by the explicit suspicious-pattern check above, which avoids
    # false positives from short or unfamiliar messages.
    if predicted_intent == "SUSPICIOUS_REQUEST":
        predicted_intent = "UNKNOWN"

    if predicted_intent == "CUSTOMER_SPECIFIC":
        reply = (
            "This request appears to involve personal account information. "
            "For your security, customer-specific information requires verified "
            "authentication and cannot be accessed through this public demonstration."
        )
        add_history(session_key, "user", message)
        add_history(session_key, "assistant", reply)
        return ChatResponse(
            session_id=request.session_id,
            status="authentication_required",
            reply=reply,
            predicted_intent=predicted_intent,
            action="REQUIRE_AUTHENTICATION",
        )

    if predicted_intent in {"COMPLAINT", "HUMAN_ESCALATION"}:
        escalation_id = create_escalation(
            organisation_id=organisation_id,
            session_id=request.session_id,
            message=message,
            reason=predicted_intent,
        )
        reply = (
            "I have prepared this request for human support. "
            f"Your reference number is {escalation_id}."
        )
        add_history(session_key, "user", message)
        add_history(session_key, "assistant", reply)
        return ChatResponse(
            session_id=request.session_id,
            status="escalated",
            reply=reply,
            predicted_intent=predicted_intent,
            action="CREATE_ESCALATION",
            escalation_id=escalation_id,
        )

    context = conversation_contexts.get(session_key)
    knowledge_result = knowledge_service.resolve(
        organisation_id=knowledge_organisation_id(organisation_id),
        message=internalise_message(message),
        conversation_state=context,
    )

    if knowledge_result.get("status") in {"answered", "needs_clarification"}:
        return convert_knowledge_result(
            organisation_id=organisation_id,
            session_key=session_key,
            session_id=request.session_id,
            user_message=message,
            predicted_intent=predicted_intent,
            result=knowledge_result,
        )

    if predicted_intent == "OUT_OF_SCOPE":
        reply = (
            "I can only help with approved information provided by this organisation. "
            "I cannot answer that question."
        )
        add_history(session_key, "user", message)
        add_history(session_key, "assistant", reply)
        return ChatResponse(
            session_id=request.session_id,
            status="refused",
            reply=reply,
            predicted_intent=predicted_intent,
            action="SAFE_REFUSAL",
        )

    reply = (
        "I could not find a reliable answer in this organisation's approved "
        "information. Please ask in a different way or request human support."
    )
    add_history(session_key, "user", message)
    add_history(session_key, "assistant", reply)
    return ChatResponse(
        session_id=request.session_id,
        status="not_found",
        reply=reply,
        predicted_intent=predicted_intent,
        action="OFFER_HUMAN_SUPPORT",
    )
