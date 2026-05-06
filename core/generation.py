from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from configs.prompts import SYSTEM_PROMPT
from configs.settings import settings
from core.guardrails import is_out_of_scope, redact_pii
from core.retrieval import retrieve

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "I don't have that information in the provided context."
OUT_OF_SCOPE_ANSWER = "This question is outside the scope of the internal assistant."

llm = ChatGroq(
    groq_api_key=settings.groq_api_key,
    model_name=settings.groq_model_name,
    temperature=0.1,
    max_tokens=1024,
)


def generate(query: str, role: str) -> dict[str, object]:

    # 1. Scope check — block before any retrieval or LLM cost
    if is_out_of_scope(query):
        logger.warning("Out-of-scope query blocked: %s", query[:80])
        return {"answer": OUT_OF_SCOPE_ANSWER, "sources": []}

    # 2. Vector retrieval with RBAC filter
    top_k = 20 if role == "admin" else 10
    retrieval_results = retrieve(query, role, top_k=top_k)
    if not retrieval_results:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    # 3. Context assembly — no PII redaction on chunks
    # Redaction happens on the final answer only so the LLM
    # receives full context and can generate a grounded response
    context_parts = []
    for chunk in retrieval_results:
        raw_text = str(chunk.get("text", "")).strip()
        if raw_text:
            context_parts.append(raw_text)

    context = "\n---\n".join(context_parts)

    if not context.strip():
        return {"answer": FALLBACK_ANSWER, "sources": []}

    # 4. Prompt construction and LLM call
    formatted_prompt = SYSTEM_PROMPT.format(role=role, context=context)
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)

    # 5. PII redaction on final answer only — raw data never leaves the system
    answer = redact_pii(response.content.strip())

    # 6. Source metadata — department and file only, no raw text
    sources = [
        {
            "department": chunk.get("department", "unknown"),
            "source_file": chunk.get("source_file", "unknown"),
        }
        for chunk in retrieval_results
    ]

    return {"answer": answer, "sources": sources}