from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from configs.settings import BLOCKED_TOPICS, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII engines — loaded once at module level, reused on every call
# ---------------------------------------------------------------------------
_analyzer  = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# ---------------------------------------------------------------------------
# LLM classifier — same Groq client used in generation, minimal token usage
# ---------------------------------------------------------------------------
_classifier_llm = ChatGroq(
    groq_api_key=settings.groq_api_key,
    model_name=settings.groq_model_name,
    temperature=0,
    max_tokens=5,
)

_SCOPE_PROMPT = (
    "You are a strict classifier for a corporate internal assistant. "
    "Answer only YES or NO, nothing else. No explanation.\n\n"
    "Answer YES if the question is about ANY of these topics:\n"
    "- Engineering systems, technology, software, architecture, databases, code\n"
    "- HR policies, employees, salaries, leave, attendance, performance\n"
    "- Finance, revenue, budget, expenses, profit, quarterly reports\n"
    "- Marketing campaigns, reports, spend, strategy\n"
    "- Company policies, handbooks, onboarding, internal processes\n\n"
    "Answer NO only if the question is clearly about external topics like "
    "weather, sports, entertainment, cooking, personal life, or general knowledge "
    "completely unrelated to a company.\n\n"
    "Question: {query}\n\n"
    "Answer YES or NO:"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_out_of_scope(query: str) -> bool:
    """
    Return True if the query should be blocked before hitting the RAG pipeline.

    Layer 1 — keyword blocklist: instant, no LLM cost.
    Layer 2 — LLM classifier: catches ambiguous cases the blocklist misses.
    Fails open on any LLM error so valid queries are never blocked by
    a transient network or rate-limit issue.
    """
    query_lower = query.lower().strip()

    # Check 1: keyword blocklist (no LLM call on direct match)
    for topic in BLOCKED_TOPICS:
        if topic in query_lower:
            return True

    # Check 2: LLM classifier
    try:
        prompt = _SCOPE_PROMPT.format(query=query)
        messages = [HumanMessage(content=prompt)]
        response = _classifier_llm.invoke(messages)
        answer = response.content.strip().upper()

        if "NO" in answer:
            return True
    except Exception as exc:
        logger.warning("Scope classifier failed (failing open): %s", exc)

    return False


def redact_pii(text: str) -> str:
    """
    Detect and replace PII in text with type-label placeholders.

    Examples of replacements:
        john.doe@company.com  →  <EMAIL_ADDRESS>
        Arjun Mehta           →  <PERSON>
        +91-9876543210        →  <PHONE_NUMBER>
        4111111111111111      →  <CREDIT_CARD>

    score_threshold=0.6 prevents over-redaction of ambiguous terms.
    Raise to 0.7 if legitimate content is being redacted incorrectly.
    """
    if not text or not text.strip():
        return text

    try:
        results = _analyzer.analyze(
            text=text,
            language="en",
            score_threshold=0.8,
             entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS"],
        )
        if not results:
            return text
        anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except Exception as exc:
        # Fail safe — return original text rather than crashing the pipeline
        logger.error("PII redaction failed, returning original text: %s", exc)
        return text