from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import pandas as pd


from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, PointStruct, VectorParams, PayloadSchemaType
from configs.settings import settings
from core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

COLLECTION_NAME = "company_docs"

ALLOWED_DEPARTMENTS = {"hr", "engineering", "finance", "marketing", "shared"}
DEPARTMENT_FOLDERS = ["hr", "engineering", "finance", "marketing", "shared"]
qdrant_client = get_qdrant_client()

def _row_to_text(row: pd.Series) -> str:
    return " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))

def _load_file_to_rows(file_path: Path) -> list[pd.Series]:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        return [row for _, row in df.iterrows()]

    rows: list[pd.Series] = []
    excel_sheets = pd.read_excel(file_path, sheet_name=None)
    for _, sheet_df in excel_sheets.items():
        rows.extend([row for _, row in sheet_df.iterrows()])
    return rows

def _documents_from_single_path(file_path: Path, department: str) -> list[Document]:
    ext = file_path.suffix.lower()
    documents: list[Document] = []

    if ext in [".md", ".txt"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()

        doc_type = "markdown" if ext == ".md" else "text"
        documents.append(
            Document(
                page_content=text_content,
                metadata={
                    "department": department,
                    "source_file": file_path.name,
                    "doc_type": doc_type,
                    "created_at": datetime.now().isoformat(),
                    "row_index": 0,
                },
            )
        )
    else:
        rows = _load_file_to_rows(file_path)
        for row_index, row in enumerate(rows):
            row_text = _row_to_text(row)
            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "department": department,
                        "source_file": file_path.name,
                        "doc_type": "tabular",
                        "row_index": row_index,
                    },
                )
            )

    return documents


def load_department(folder_path: str | Path) -> list[Document]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder}")

    department = folder.name
    extensions = ["*.csv", "*.xlsx", "*.xls", "*.md", "*.txt"]
    supported_files = sorted(
        [f for ext in extensions for f in folder.glob(ext)],
        key=lambda p: p.name.lower(),
    )

    documents: list[Document] = []
    for file_path in supported_files:
        documents.extend(_documents_from_single_path(file_path, department))

    return documents


def ingest_single_file(file_path: str | Path, department: str) -> int:
    path = Path(file_path)
    if department not in ALLOWED_DEPARTMENTS:
        raise ValueError(
            f"Invalid department {department!r}; expected one of {sorted(ALLOWED_DEPARTMENTS)}."
        )

    documents = _documents_from_single_path(path, department)
    chunked_documents = chunk_documents(documents)
    chunked_documents, embeddings = embed_documents(chunked_documents)
    upsert_to_qdrant(chunked_documents, embeddings)
    return len(chunked_documents)

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents: list[Document] = []

    for document in documents:
        if len(document.page_content) <= 500:
            chunked_documents.append(document)
            continue

        chunks = splitter.split_documents([document])
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
        chunked_documents.extend(chunks)

    return chunked_documents

def embed_documents(documents: list[Document]) -> tuple[list[Document], list[list[float]]]:
    if not documents:
        return documents, []

    texts = [document.page_content for document in documents]
    embeddings = EMBEDDING_MODEL.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
    )
    embedding_vectors = embeddings.tolist()
    return documents, embedding_vectors

def init_qdrant_collection() -> None:
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        logger.info("Qdrant collection '%s' already exists; skipping creation.", COLLECTION_NAME)
    else:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="department",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("Created payload index for 'department'")

def upsert_to_qdrant(documents: list[Document], vectors: list[list[float]]) -> None:
    if len(documents) != len(vectors):
        raise ValueError("Documents and vectors must have the same length.")
    if not documents:
        return

    points: list[PointStruct] = []
    for document, vector in zip(documents, vectors):
        department = document.metadata.get("department")
        if department not in ALLOWED_DEPARTMENTS:
            raise ValueError(
                "Invalid department metadata. Expected one of "
                f"{sorted(ALLOWED_DEPARTMENTS)}, got: {department!r}"
            )

        payload = dict(document.metadata)
        payload["text"] = document.page_content
        payload.setdefault("department", department)
        payload.setdefault("source_file", "")
        payload.setdefault("doc_type", "")
        payload.setdefault("created_at", datetime.now().isoformat())
        payload.setdefault("chunk_index", 0)

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=[float(value) for value in vector],
                payload=payload,
            )
        )

    batch_size = 30
    for start in range(0, len(points), batch_size):
        batch_points = points[start : start + batch_size]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
        logger.info(
            "Upserted %d points into collection '%s'.",
            len(batch_points),
            COLLECTION_NAME,
        )

def ingest_all_departments() -> None:
    for department_name in DEPARTMENT_FOLDERS:
        department_path = Path("data") / department_name
        if not department_path.exists():
            logger.warning(
                "Skipping department '%s': folder not found at '%s'.",
                department_name,
                department_path,
            )
            continue

        documents = load_department(department_path)
        chunked_documents = chunk_documents(documents)
        chunked_documents, embeddings = embed_documents(chunked_documents)
        upsert_to_qdrant(chunked_documents, embeddings)
        logger.info(
            "Completed ingestion for department '%s' with %d chunks upserted.",
            department_name,
            len(chunked_documents),
        )

if __name__ == "__main__":
    init_qdrant_collection()
    ingest_all_departments()