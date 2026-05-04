from __future__ import annotations

import logging
from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer

from core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "company_docs"
_embedding_model = None

# lazy — initialized on first retrieve() call, not at import time
_qdrant_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model initialized: all-MiniLM-L6-v2")
    return _embedding_model


def _get_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
        logger.info(
            "Qdrant client initialized: %s",
            type(_qdrant_client._client).__name__,
        )
    return _qdrant_client


ROLE_PERMISSIONS: dict[str, list[str]] = {
    "engineering": ["engineering", "shared"],
    "finance":     ["finance",     "shared"],
    "hr":          ["hr",          "shared"],
    "marketing":   ["marketing",   "shared"],
    "shared":      ["shared"],
    "admin":       ["engineering", "finance", "hr", "marketing", "shared"],
}

VALID_ROLES: set[str] = set(ROLE_PERMISSIONS.keys())


def get_allowed_departments(role: str) -> list[str]:
    """Return permitted departments for a role, or raise on unknown roles."""
    if role not in ROLE_PERMISSIONS:
        raise PermissionError(f"Unknown role '{role}' is not permitted.")
    return ROLE_PERMISSIONS[role]


def build_qdrant_filter(role: str) -> Filter:
    """Build a Qdrant payload filter that enforces department-level RBAC."""
    allowed_departments = get_allowed_departments(role)
    return Filter(
        must=[
            FieldCondition(
                key="department",
                match=MatchAny(any=allowed_departments),
            )
        ]
    )


def retrieve(query: str, role: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Embed the query, search Qdrant with RBAC filter, and return matching chunks.

    Each returned dict contains:
        text        — the chunk content
        score       — cosine similarity score
        department  — the department tag from metadata
        source_file — the original file name
        doc_type    — tabular / markdown / text
    """
    query_vector = _get_embedding_model().encode(query).tolist()
    role_filter = build_qdrant_filter(role)

    results = _get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=role_filter,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "text":        (result.payload or {}).get("text", ""),
            "score":       result.score,
            "department":  (result.payload or {}).get("department", ""),
            "source_file": (result.payload or {}).get("source_file", ""),
            "doc_type":    (result.payload or {}).get("doc_type", ""),
        }
        for result in results
    ]