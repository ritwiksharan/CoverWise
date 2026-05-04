from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, EmailStr
from typing import Optional
from .db import create_user, verify_user, create_session, get_user_by_token, delete_session, email_exists

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


def _require_token(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.removeprefix("Bearer ").strip()
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Token invalid or expired")
    return user


@router.post("/signup")
def signup(req: SignupRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if email_exists(req.email):
        raise HTTPException(status_code=409, detail="An account with that email already exists")
    user = create_user(req.email, req.password, req.name)
    token = create_session(user["user_id"])
    return {"token": token, "user_id": user["user_id"], "email": user["email"], "name": user["name"]}


@router.post("/login")
def login(req: LoginRequest):
    user = verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_session(user["user_id"])
    return {"token": token, "user_id": user["user_id"], "email": user["email"], "name": user["name"]}


@router.get("/me")
def me(authorization: Optional[str] = Header(None)):
    user = _require_token(authorization)
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"]}


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        delete_session(token)
    return {"ok": True}
