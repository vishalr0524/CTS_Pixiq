"""Core authentication service — login, logout, session management.

All state lives in SQLite (users + sessions tables).
Passwords hashed with bcrypt.  Tokens are plain uuid4 strings.

Usage:
    auth = AuthService(conn, default_session_hours=8.0)
    session = auth.login("operator1", "password123")
    user = auth.validate_session(session.token)
    auth.logout(session.token)
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class User:
    id: int
    username: str
    role: str
    services: dict
    email: Optional[str] = None
    empid: Optional[str] = None
    name: Optional[str] = None
    active: bool = True


@dataclass
class Session:
    token: str
    user_id: int
    username: str
    role: str
    services: dict
    created_at: str
    expires_at: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AuthService:
    """Session-based auth backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection, default_session_hours: float = 8.0):
        self._conn = conn
        self._session_hours = default_session_hours

    # -- password helpers ---------------------------------------------------

    @staticmethod
    def hash_password(plain: str) -> str:
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(plain.encode(), hashed.encode())

    # -- login / logout -----------------------------------------------------

    def login(self, username: str, password: str) -> Session:
        """Authenticate and create a session.

        Raises ValueError on bad credentials or inactive user.
        """
        row = self._conn.execute(
            "SELECT id, username, password, role, services, email, empid, name, active "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None:
            raise ValueError("Invalid username or password")

        (uid, uname, pw_hash, role, services_json, email, empid, name, active) = row

        if not active:
            raise ValueError("Account is disabled")

        if not self.verify_password(password, pw_hash):
            raise ValueError("Invalid username or password")

        services = json.loads(services_json) if services_json else {}
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=self._session_hours)
        token = uuid.uuid4().hex

        self._conn.execute(
            "INSERT INTO sessions (token, user_id, username, role, services, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                token,
                uid,
                uname,
                role,
                json.dumps(services),
                now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
        )

        # Log activity
        self._conn.execute(
            "INSERT INTO activity_log (username, action, details) VALUES (?, 'login', ?)",
            (uname, json.dumps({"role": role})),
        )
        self._conn.commit()

        logger.info("User '%s' logged in (role=%s, expires=%s)", uname, role, expires.isoformat())

        return Session(
            token=token,
            user_id=uid,
            username=uname,
            role=role,
            services=services,
            created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            expires_at=expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def logout(self, token: str) -> bool:
        """Delete session and log the event. Returns True if session existed."""
        row = self._conn.execute(
            "SELECT username FROM sessions WHERE token = ?", (token,)
        ).fetchone()

        if row is None:
            return False

        username = row[0]
        self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        self._conn.execute(
            "INSERT INTO activity_log (username, action) VALUES (?, 'logout')",
            (username,),
        )
        self._conn.commit()
        logger.info("User '%s' logged out", username)
        return True

    # -- session validation -------------------------------------------------

    def validate_session(self, token: str) -> Optional[User]:
        """Look up session token, return User if valid and not expired."""
        row = self._conn.execute(
            "SELECT s.user_id, s.username, s.role, s.services, s.expires_at, u.active "
            "FROM sessions s JOIN users u ON s.user_id = u.id "
            "WHERE s.token = ?",
            (token,),
        ).fetchone()

        if row is None:
            return None

        user_id, username, role, services_json, expires_at, active = row
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if expires_at < now:
            # Expired — clean up
            self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self._conn.commit()
            return None

        if not active:
            return None

        return User(
            id=user_id,
            username=username,
            role=role,
            services=json.loads(services_json) if services_json else {},
        )

    def cleanup_expired_sessions(self) -> int:
        """Delete all expired sessions. Returns count deleted."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = self._conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (now,)
        )
        self._conn.commit()
        return cur.rowcount

    # -- user CRUD ----------------------------------------------------------

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "operator",
        services: Optional[dict] = None,
        email: Optional[str] = None,
        empid: Optional[str] = None,
        name: Optional[str] = None,
    ) -> User:
        """Create a new user. Raises ValueError if username exists."""
        if services is None:
            services = {
                "live": True,
                "master": False,
                "settings": False,
                "report": True,
                "activityLog": True,
                "inspection": False,
                "email": False,
            }

        pw_hash = self.hash_password(password)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            cur = self._conn.execute(
                "INSERT INTO users (username, password, role, services, email, empid, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (username, pw_hash, role, json.dumps(services), email, empid, name or username, now, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")

        logger.info("User '%s' created (role=%s)", username, role)
        return User(
            id=cur.lastrowid,
            username=username,
            role=role,
            services=services,
            email=email,
            empid=empid,
            name=name or username,
            active=True,
        )

    def update_user(
        self,
        username: str,
        *,
        password: Optional[str] = None,
        role: Optional[str] = None,
        services: Optional[dict] = None,
        email: Optional[str] = None,
        empid: Optional[str] = None,
        name: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> User:
        """Update user fields. Only non-None fields are changed."""
        row = self._conn.execute(
            "SELECT id, username, role, services, email, empid, name, active "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None:
            raise ValueError(f"User '{username}' not found")

        uid, uname, cur_role, cur_services_json, cur_email, cur_empid, cur_name, cur_active = row

        new_role = role if role is not None else cur_role
        new_services = json.dumps(services) if services is not None else cur_services_json
        new_email = email if email is not None else cur_email
        new_empid = empid if empid is not None else cur_empid
        new_name = name if name is not None else cur_name
        new_active = (1 if active else 0) if active is not None else cur_active
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if password is not None:
            pw_hash = self.hash_password(password)
            self._conn.execute(
                "UPDATE users SET password=?, role=?, services=?, email=?, empid=?, name=?, active=?, updated_at=? "
                "WHERE id=?",
                (pw_hash, new_role, new_services, new_email, new_empid, new_name, new_active, now, uid),
            )
        else:
            self._conn.execute(
                "UPDATE users SET role=?, services=?, email=?, empid=?, name=?, active=?, updated_at=? "
                "WHERE id=?",
                (new_role, new_services, new_email, new_empid, new_name, new_active, now, uid),
            )

        # If user is deactivated, kill all their sessions
        if active is False:
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))

        self._conn.commit()
        logger.info("User '%s' updated", username)

        return User(
            id=uid,
            username=uname,
            role=new_role,
            services=json.loads(new_services) if isinstance(new_services, str) else new_services,
            email=new_email,
            empid=new_empid,
            name=new_name,
            active=bool(new_active),
        )

    def delete_user(self, username: str) -> bool:
        """Delete user and all their sessions. Returns True if user existed."""
        row = self._conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            return False

        uid = row[0]
        self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
        self._conn.execute("DELETE FROM users WHERE id = ?", (uid,))
        self._conn.commit()
        logger.info("User '%s' deleted", username)
        return True

    def get_user(self, username: str) -> Optional[User]:
        """Fetch a single user by username."""
        row = self._conn.execute(
            "SELECT id, username, role, services, email, empid, name, active "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None:
            return None

        uid, uname, role, services_json, email, empid, name, active = row
        return User(
            id=uid,
            username=uname,
            role=role,
            services=json.loads(services_json) if services_json else {},
            email=email,
            empid=empid,
            name=name,
            active=bool(active),
        )

    def list_users(self) -> list[User]:
        """Return all users (no passwords)."""
        rows = self._conn.execute(
            "SELECT id, username, role, services, email, empid, name, active "
            "FROM users ORDER BY username"
        ).fetchall()

        return [
            User(
                id=r[0], username=r[1], role=r[2],
                services=json.loads(r[3]) if r[3] else {},
                email=r[4], empid=r[5], name=r[6], active=bool(r[7]),
            )
            for r in rows
        ]

    # -- activity log -------------------------------------------------------

    def log_activity(self, username: str, action: str, details: Optional[dict] = None) -> None:
        """Write an entry to the activity log."""
        self._conn.execute(
            "INSERT INTO activity_log (username, action, details) VALUES (?, ?, ?)",
            (username, action, json.dumps(details) if details else None),
        )
        self._conn.commit()

    def get_activity_log(
        self,
        username: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch activity log entries, newest first."""
        if username:
            rows = self._conn.execute(
                "SELECT id, username, action, details, timestamp "
                "FROM activity_log WHERE username = ? "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (username, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, username, action, details, timestamp "
                "FROM activity_log "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        return [
            {
                "id": r[0],
                "username": r[1],
                "action": r[2],
                "details": json.loads(r[3]) if r[3] else None,
                "timestamp": r[4],
            }
            for r in rows
        ]

    # -- seed admin ---------------------------------------------------------

    def seed_admin(
        self,
        username: str = "admin",
        password: str = "admin",
        role: str = "superAdmin",
    ) -> None:
        """Create default admin if no users exist. Idempotent."""
        count = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return

        all_services = {
            "live": True,
            "master": True,
            "settings": True,
            "report": True,
            "activityLog": True,
            "inspection": True,
            "email": True,
        }
        self.create_user(
            username=username,
            password=password,
            role=role,
            services=all_services,
            name="Administrator",
        )
        logger.info("Seeded default admin user '%s'", username)
