from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from api.auth import USERS, create_access_token, get_current_role, require_admin_role
from configs.settings import settings
from core.generation import generate
from core.ingestion import ingest_single_file
from core.qdrant_client import get_qdrant_client

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

VALID_UPLOAD_DEPARTMENTS = frozenset(
    {"engineering", "finance", "hr", "marketing", "shared"}
)
ALLOWED_UPLOAD_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls", ".md", ".txt"})

app = FastAPI(title="VaultIQ — Internal Knowledge Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",
        "https://AnanyaYash-vaultiq-api.hf.space",
        "https://vault-iq.netlify.app",
    ],
    allow_methods=["POST", "GET", "OPTIONS", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    qdrant_ok = True
    groq_ok = True

    try:
        client = get_qdrant_client()
        client.get_collections()
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        qdrant_ok = False

    try:
        readiness_llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
            temperature=0,
            max_tokens=1,
        )
        readiness_llm.invoke([HumanMessage(content="ping")])
    except Exception as e:
        logger.error("Groq health check failed: %s", e)
        groq_ok = False

    status_text = "ready" if qdrant_ok and groq_ok else "degraded"
    return JSONResponse(
        status_code=200 if status_text == "ready" else 503,
        content={"status": status_text, "qdrant": qdrant_ok, "groq": groq_ok},
    )


@app.post("/admin/upload")
async def admin_upload(
    department: str = Form(...),
    file: UploadFile = File(...),
    _: str = Depends(require_admin_role),
) -> dict[str, str | int]:
    if department not in VALID_UPLOAD_DEPARTMENTS:
        raise HTTPException(status_code=400, detail="Invalid department.")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    base_dir = Path("data") / department
    base_dir.mkdir(parents=True, exist_ok=True)
    target_path = base_dir / filename

    if target_path.exists():
        stem = Path(filename).stem
        ext = Path(filename).suffix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = base_dir / f"{stem}_{ts}{ext}"

    content = await file.read()
    target_path.write_bytes(content)

    chunks_ingested = ingest_single_file(target_path, department)
    return {
        "message": "File uploaded and ingested successfully.",
        "filename": target_path.name,
        "department": department,
        "chunks_ingested": chunks_ingested,
    }


@app.get("/admin/documents")
def admin_documents(
    department: str | None = None,
    _: str = Depends(require_admin_role),
) -> dict[str, list[dict[str, str | int]]]:
    if department is not None and department not in VALID_UPLOAD_DEPARTMENTS:
        raise HTTPException(status_code=400, detail="Invalid department.")

    departments = [department] if department else list(VALID_UPLOAD_DEPARTMENTS)
    results: list[dict[str, str | int]] = []

    for dept in departments:
        dept_path = Path("data") / dept
        if not dept_path.exists():
            continue
        for file_path in dept_path.iterdir():
            if file_path.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
                continue
            stat = file_path.stat()
            results.append({
                "filename": file_path.name,
                "department": dept,
                "size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%S"),
            })

    return {"documents": results}


class DeleteDocumentRequest(BaseModel):
    department: str
    filename: str


@app.delete("/admin/documents")
def admin_delete_document(
    body: DeleteDocumentRequest,
    _: str = Depends(require_admin_role),
) -> dict[str, str]:
    if body.department not in VALID_UPLOAD_DEPARTMENTS:
        raise HTTPException(status_code=400, detail="Invalid department.")

    file_path = Path("data") / body.department / body.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    qdrant = get_qdrant_client()
    qdrant.delete(
        collection_name="company_docs",
        points_selector=Filter(
            must=[
                FieldCondition(key="department", match=MatchValue(value=body.department)),
                FieldCondition(key="source_file", match=MatchValue(value=body.filename)),
            ]
        ),
    )

    file_path.unlink()

    return {
        "message": "File deleted and removed from knowledge base.",
        "filename": body.filename,
        "department": body.department,
    }


@app.post("/auth/token")
def auth_token(form_data: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    user = USERS.get(form_data.username)
    if not user or user.get("password") != form_data.password:
        raise HTTPException(status_code=401, detail="Credentials are incorrect.")
    role = str(user["role"])
    token = create_access_token(role)
    return {"access_token": token, "token_type": "bearer", "role": role}


@app.post("/chat")
def chat(request: ChatRequest, role: str = Depends(get_current_role)) -> dict[str, object]:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        result = generate(query=request.query, role=role)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat generation failed for role '%s': %s", role, exc)
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
    return {
        "answer": str(result.get("answer", "")).strip(),
        "sources": result.get("sources", []),
        "role": role,
    }