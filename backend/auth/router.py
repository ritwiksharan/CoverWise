from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from .db import create_user, verify_user, create_session, get_user_by_token, delete_session, username_exists

router = APIRouter(tags=["auth"])


class SignupRequest(BaseModel):
    username: str
    password: str
    name: str


class LoginRequest(BaseModel):
    username: str
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
    if len(req.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if username_exists(req.username):
        raise HTTPException(status_code=409, detail="That username is already taken")
    user = create_user(req.username, req.password, req.name)
    token = create_session(user["user_id"])
    return {"token": token, "user_id": user["user_id"], "username": user["username"], "name": user["name"]}


@router.post("/login")
def login(req: LoginRequest):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_session(user["user_id"])
    return {"token": token, "user_id": user["user_id"], "username": user["username"], "name": user["name"]}


@router.get("/me")
def me(authorization: Optional[str] = Header(None)):
    user = _require_token(authorization)
    return {"user_id": user["user_id"], "username": user["username"], "name": user["name"]}


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        delete_session(token)
    return {"ok": True}
