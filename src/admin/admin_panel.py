"""
admin_panel.py
================
A password-protected control panel for system monitoring — stores
classification attempts (every model prediction, its confidence, and
timestamp) and shows ordinary (Tier-1) user information to the
developer/maintenance technician, **without ever** exposing the values of
sensitive (Tier-2) fields from this panel — even with admin privileges,
this is an intentional architectural separation, not a temporary
restriction.

Core security decision:
  - The admin password is stored as a salted hash (via
    hashlib.pbkdf2_hmac), never as plaintext, even in memory after first use.
  - The classification-attempt log is a table completely separate from
    profile_store, and never contains the actual value of any user profile
    field — only: the predicted letter/word, the confidence, the
    timestamp, and which model was used (Grayscale/Color/Fusion).
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# profile_store.py lives in a sibling package (src/profile/) after this project's
# repository reorganization; make it importable whether this file is run directly
# or imported as part of the src package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "profile"))
from profile_store import NormalProfileStore, SensitiveProfileStore


# ----------------------------------------------------------------- password

def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


class AdminAuth:
    def __init__(self, credentials_path: str = "admin_credentials.json"):
        self.credentials_path = credentials_path

    def is_configured(self) -> bool:
        return os.path.exists(self.credentials_path)

    def set_password(self, password: str):
        """Called once only, during the panel's initial setup."""
        salt = os.urandom(16)
        hashed = _hash_password(password, salt)
        with open(self.credentials_path, "w") as f:
            json.dump({"salt": salt.hex(), "hash": hashed.hex()}, f)
        os.chmod(self.credentials_path, 0o600)

    def verify_password(self, password: str) -> bool:
        if not self.is_configured():
            return False
        with open(self.credentials_path) as f:
            data = json.load(f)
        salt = bytes.fromhex(data["salt"])
        expected = bytes.fromhex(data["hash"])
        attempt = _hash_password(password, salt)
        return attempt == expected  # a plain comparison is fine here since pbkdf2 itself is already slow


class AuthenticationError(Exception):
    pass


# ---------------------------------------------------------- classification-attempt log

@dataclass
class ClassificationAttempt:
    timestamp: float
    model_used: str      # "grayscale" | "color" | "fusion"
    predicted_label: str
    confidence: float
    accepted: bool        # whether it cleared the required confidence threshold or not


class ClassificationLog:
    def __init__(self, db_path: str = "classification_log.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accepted INTEGER NOT NULL
                )
            """)

    def log_attempt(self, model_used: str, predicted_label: str,
                     confidence: float, accepted: bool):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO attempts (timestamp, model_used, predicted_label, confidence, accepted) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), model_used, predicted_label, confidence, int(accepted)),
            )

    def recent_attempts(self, limit: int = 50) -> List[ClassificationAttempt]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, model_used, predicted_label, confidence, accepted "
                "FROM attempts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ClassificationAttempt(ts, model, label, conf, bool(acc))
                for ts, model, label, conf, acc in rows]

    def accuracy_summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            accepted = conn.execute("SELECT COUNT(*) FROM attempts WHERE accepted=1").fetchone()[0]
            by_model = conn.execute(
                "SELECT model_used, COUNT(*), AVG(confidence) FROM attempts GROUP BY model_used"
            ).fetchall()
        return {
            "total_attempts": total,
            "accepted_attempts": accepted,
            "acceptance_rate": round(accepted / total, 3) if total else None,
            "by_model": {m: {"count": c, "avg_confidence": round(a, 3)} for m, c, a in by_model},
        }


# ---------------------------------------------------------------- admin panel

class AdminPanel:
    """
    The unified interface used by the developer/technician. Every call
    other than login() requires a previously authenticated session — there
    is no path that bypasses verify_password().
    """

    def __init__(self, auth: AdminAuth, log: ClassificationLog,
                 normal_store: NormalProfileStore, sensitive_store: SensitiveProfileStore):
        self.auth = auth
        self.log = log
        self.normal_store = normal_store
        self.sensitive_store = sensitive_store
        self._authenticated = False

    def login(self, password: str) -> bool:
        self._authenticated = self.auth.verify_password(password)
        return self._authenticated

    def _require_auth(self):
        if not self._authenticated:
            raise AuthenticationError("You must log in first via login() with a correct password.")

    def view_normal_profile(self) -> dict:
        self._require_auth()
        return self.normal_store.list_fields()

    def view_sensitive_field_names_only(self) -> list:
        """Sensitive field names only, never their values — not even for the admin."""
        self._require_auth()
        return self.sensitive_store.list_field_names()

    def view_recent_classification_attempts(self, limit: int = 50) -> List[ClassificationAttempt]:
        self._require_auth()
        return self.log.recent_attempts(limit)

    def view_accuracy_summary(self) -> dict:
        self._require_auth()
        return self.log.accuracy_summary()
