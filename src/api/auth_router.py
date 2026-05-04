"""FastAPI auth router — session-based, no JWT.

Endpoints:
    POST /auth/login          — authenticate, return session token
    POST /auth/logout         — invalidate session
    GET  /auth/me             — current user from token
    GET  /auth/users          — list all users (superAdmin)
    POST /auth/users          — create user (superAdmin)
    PUT  /auth/users/{username} — update user (superAdmin)
    DELETE /auth/users/{username} — delete user (superAdmin)
    POST /auth/users/{username}/reset-password — reset password (superAdmin)
    GET  /auth/activity       — activity log

Auth dependency:
    from api.auth_router import get_current_user, require_role
    user = Depends(get_current_user)        # 401 if not logged in
    user = Depends(require_role("superAdmin"))  # 403 if wrong role
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from auth.service import AuthService, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str
    services: dict
    expires_at: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "operator"
    services: Optional[dict] = None
    email: Optional[str] = None
    empid: Optional[str] = None
    name: Optional[str] = None


class UserUpdateRequest(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = None
    services: Optional[dict] = None
    email: Optional[str] = None
    empid: Optional[str] = None
    name: Optional[str] = None
    active: Optional[bool] = None


class ResetPasswordRequest(BaseModel):
    new_password: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    services: dict
    email: Optional[str] = None
    empid: Optional[str] = None
    name: Optional[str] = None
    active: bool = True


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Auth dependency — extracts token from Authorization header
# ---------------------------------------------------------------------------

def _get_auth_service(request: Request) -> AuthService:
    """Pull AuthService from app state (set during startup)."""
    return request.app.state.auth_service


def get_optional_user(
    authorization: Optional[str] = Header(None),
    auth: AuthService = Depends(_get_auth_service),
) -> Optional[User]:
    """Return current user if valid token provided, else None.

    Use this for endpoints that work for both authed and anonymous users.
    """
    if not authorization:
        return None
    return auth.validate_session(authorization)


def get_current_user(
    authorization: Optional[str] = Header(None),
    auth: AuthService = Depends(_get_auth_service),
) -> User:
    """Require valid session. Returns User or raises 401."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    user = auth.validate_session(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def require_role(*roles: str):
    """Dependency factory — require user to have one of the given roles."""
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {', '.join(roles)}")
        return user
    return dependency


def require_service(service_name: str):
    """Dependency factory — require user to have a specific service enabled."""
    def dependency(user: User = Depends(get_current_user)) -> User:
        if not user.services.get(service_name, False):
            raise HTTPException(status_code=403, detail=f"Service '{service_name}' not enabled for this user")
        return user
    return dependency


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, auth: AuthService = Depends(_get_auth_service)):
    try:
        session = auth.login(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return LoginResponse(
        token=session.token,
        username=session.username,
        role=session.role,
        services=session.services,
        expires_at=session.expires_at,
    )


@router.post("/logout", response_model=MessageResponse)
def logout(
    authorization: Optional[str] = Header(None),
    auth: AuthService = Depends(_get_auth_service),
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if auth.logout(authorization):
        return MessageResponse(message="Logged out")
    raise HTTPException(status_code=401, detail="Invalid session")


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        services=user.services,
        email=user.email,
        empid=user.empid,
        name=user.name,
        active=user.active,
    )


# -- User management (superAdmin only) -------------------------------------

@router.get("/users", response_model=list[UserResponse])
def list_users(
    _user: User = Depends(require_role("superAdmin")),
    auth: AuthService = Depends(_get_auth_service),
):
    return [
        UserResponse(
            id=u.id, username=u.username, role=u.role,
            services=u.services, email=u.email, empid=u.empid,
            name=u.name, active=u.active,
        )
        for u in auth.list_users()
    ]


@router.post("/users", response_model=UserResponse, status_code=201)
def create_user(
    body: UserCreateRequest,
    _user: User = Depends(require_role("superAdmin")),
    auth: AuthService = Depends(_get_auth_service),
):
    try:
        u = auth.create_user(
            username=body.username,
            password=body.password,
            role=body.role,
            services=body.services,
            email=body.email,
            empid=body.empid,
            name=body.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return UserResponse(
        id=u.id, username=u.username, role=u.role,
        services=u.services, email=u.email, empid=u.empid,
        name=u.name, active=u.active,
    )


@router.put("/users/{username}", response_model=UserResponse)
def update_user(
    username: str,
    body: UserUpdateRequest,
    _user: User = Depends(require_role("superAdmin")),
    auth: AuthService = Depends(_get_auth_service),
):
    try:
        u = auth.update_user(
            username,
            password=body.password,
            role=body.role,
            services=body.services,
            email=body.email,
            empid=body.empid,
            name=body.name,
            active=body.active,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return UserResponse(
        id=u.id, username=u.username, role=u.role,
        services=u.services, email=u.email, empid=u.empid,
        name=u.name, active=u.active,
    )


@router.delete("/users/{username}", response_model=MessageResponse)
def delete_user(
    username: str,
    current_user: User = Depends(require_role("superAdmin")),
    auth: AuthService = Depends(_get_auth_service),
):
    if username == current_user.username:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    if not auth.delete_user(username):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")

    return MessageResponse(message=f"User '{username}' deleted")


@router.post("/users/{username}/reset-password", response_model=MessageResponse)
def reset_password(
    username: str,
    body: ResetPasswordRequest,
    _user: User = Depends(require_role("superAdmin")),
    auth: AuthService = Depends(_get_auth_service),
):
    try:
        auth.update_user(username, password=body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return MessageResponse(message=f"Password reset for '{username}'")


# -- Activity log -----------------------------------------------------------

@router.get("/activity")
def activity_log(
    username: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    _user: User = Depends(require_role("superAdmin", "engineer")),
    auth: AuthService = Depends(_get_auth_service),
):
    return auth.get_activity_log(username=username, limit=limit, offset=offset)
