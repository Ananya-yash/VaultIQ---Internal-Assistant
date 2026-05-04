from __future__ import annotations

import logging
from qdrant_client import QdrantClient
from configs.settings import settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.qdrant_mode == "local":
            logger.info("Qdrant mode: local storage at %s", settings.qdrant_local_path)
            _client = QdrantClient(path=settings.qdrant_local_path)
        else:
            logger.info("Qdrant mode: cloud at %s", settings.qdrant_url)
            _client = QdrantClient(
                timeout=120,
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
    return _client