from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from configs.settings import settings

USERS: dict[str, dict[str, str]] = {
    "eng_user": {"password": "eng123", "role": "engineering"},
    "fin_user": {"password": "fin123", "role": "finance"},
    "hr_user": {"password": "hr123", "role": "hr"},
    "mkt_user": {"password": "mkt123", "role": "marketing"},
    "shared_user": {"password": "shared123", "role": "shared"},
    "admin_user": {"password": "admin123", "role": "admin"},
}

security = HTTPBearer()


def create_access_token(role: str) -> str:
    payload = {
        "sub": role,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
    }
    jwt_secret = getattr(settings, "JWT_SECRET", settings.jwt_secret)
    algorithm = getattr(settings, "ALGORITHM", settings.algorithm)
    return jwt.encode(payload, jwt_secret, algorithm=algorithm)


def verify_token(token: str) -> str:
    jwt_secret = getattr(settings, "JWT_SECRET", settings.jwt_secret)
    algorithm = getattr(settings, "ALGORITHM", settings.algorithm)

    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[algorithm])
    except (JWTError, ExpiredSignatureError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired.",
        )

    role = payload.get("role")
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the role claim.",
        )

    return str(role)


def get_current_role(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    return verify_token(credentials.credentials)


def require_admin_role(role: str = Depends(get_current_role)) -> str:
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return role
