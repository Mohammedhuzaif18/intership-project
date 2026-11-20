"""Password hashing and signed JWT handling for patient/doctor accounts."""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from backend.patient_history import STORE_DIR

TOKEN_ISSUER = "pulmoscan"
TOKEN_LIFETIME_SECONDS = int(os.environ.get("AUTH_TOKEN_LIFETIME_SECONDS", "3600"))
PASSWORD_HASHER = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
SECRET_PATH = STORE_DIR / "jwt_signing_key"


class AuthenticationError(Exception):
    """Raised when account credentials or a bearer token are invalid."""


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _signing_key() -> bytes:
    configured = os.environ.get("AUTH_JWT_SECRET")
    if configured:
        if len(configured.encode("utf-8")) < 32:
            raise RuntimeError("AUTH_JWT_SECRET must contain at least 32 bytes.")
        return configured.encode("utf-8")

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return SECRET_PATH.read_bytes()
    except FileNotFoundError:
        key = secrets.token_bytes(48)
        try:
            fd = os.open(SECRET_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return SECRET_PATH.read_bytes()
        with os.fdopen(fd, "wb") as stream:
            stream.write(key)
        return key


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not 6 <= len(password) <= 256:
        raise ValueError("Password must be between 6 and 256 characters.")
    if len(password.encode("utf-8")) > 1024:
        raise ValueError("Password is too long.")
    return PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError, TypeError):
        return False


def create_access_token(user_id: str, role: str) -> tuple[str, int]:
    now = int(time.time())
    expires = now + TOKEN_LIFETIME_SECONDS
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "role": role,
        "iss": TOKEN_ISSUER,
        "iat": now,
        "exp": expires,
        "jti": secrets.token_hex(12),
    }
    signing_input = f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode())}."
    signing_input += _b64url_encode(json.dumps(payload, separators=(',', ':')).encode())
    signature = hmac.new(_signing_key(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}", expires


def decode_access_token(token: str) -> dict:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        signing_input = f"{encoded_header}.{encoded_payload}"
        expected = hmac.new(_signing_key(), signing_input.encode("ascii"), hashlib.sha256).digest()
        supplied = _b64url_decode(encoded_signature)
        if not hmac.compare_digest(expected, supplied):
            raise AuthenticationError("Invalid bearer token.")
        header = json.loads(_b64url_decode(encoded_header))
        payload = json.loads(_b64url_decode(encoded_payload))
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise AuthenticationError("Invalid bearer token.")
        if payload.get("iss") != TOKEN_ISSUER or not isinstance(payload.get("sub"), str):
            raise AuthenticationError("Invalid bearer token.")
        if payload.get("role") not in {"PATIENT", "DOCTOR"}:
            raise AuthenticationError("Invalid bearer token.")
        now = int(time.time())
        if not isinstance(payload.get("exp"), int) or payload["exp"] <= now:
            raise AuthenticationError("Bearer token has expired.")
        if not isinstance(payload.get("iat"), int) or payload["iat"] > now + 30:
            raise AuthenticationError("Invalid bearer token.")
        return payload
    except AuthenticationError:
        raise
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Invalid bearer token.") from exc
