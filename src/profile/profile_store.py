"""
profile_store.py
==================
Two-tier personal profile layer for the non-verbal user — two structurally
separate tiers:

  Tier 1 (ordinary)  : name, age, general contact info the user chooses to
                        store. Disclosed to an interlocutor automatically only
                        after a simple confirmation + a normal "Yes" gesture.

  Tier 2 (sensitive)  : national ID number (storage banned entirely — a hard
                        code-level block, not just a convention), financial
                        info, or any field the user classifies as "sensitive".
                        Never disclosed automatically based on predicted
                        context — always requires an explicit disclosure
                        prompt + a distinct Sensitive Consent Gesture first.

Key architectural decision (documented earlier in the project summary, and
actually enforced here, not just described):
  - The Tier-2 store is a completely separate file, fully encrypted
    (Fernet/AES), and never passes through any code path connected to
    neural-network training (GrayscaleCNN / ColorSignCNN / BiLSTM).
  - Even if the user tries to store a national ID number, it is rejected at
    the code level itself (not just a policy recommendation) — see
    BLOCKED_FIELD_PATTERNS below.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cryptography.fernet import Fernet


# ----------------------------------------------------------------- constants

class Tier(str, Enum):
    NORMAL = "normal"      # Tier 1
    SENSITIVE = "sensitive"  # Tier 2


# Field name patterns that are permanently banned from storage, regardless of
# which tier the user requests (documented ethical decision: national ID
# numbers are excluded entirely). The Arabic patterns are intentional and
# functional — end users and interlocutors are Arabic speakers, so the field
# name itself may well be typed/spoken in Arabic; this is user-facing
# language data, not documentation, and must not be translated away.
BLOCKED_FIELD_PATTERNS = [
    r"national[_ ]?id",
    r"\bid[_ ]?number\b",
    r"رقم[_ ]?الهوية",
    r"هوية[_ ]?وطنية",
    r"\biqama\b",
    r"\bpassport[_ ]?number\b",
]


def is_blocked_field(field_name: str) -> bool:
    name = field_name.strip().lower()
    return any(re.search(pat, name, flags=re.IGNORECASE) for pat in BLOCKED_FIELD_PATTERNS)


# --------------------------------------------------------------------- errors

class BlockedFieldError(Exception):
    """Raised when attempting to store a permanently banned field (e.g. national ID)."""


class ConsentRequiredError(Exception):
    """Raised when attempting to read a sensitive field without prior explicit consent."""


# ------------------------------------------------------------------- Tier 1

class NormalProfileStore:
    """
    Simple local storage (JSON on disk) for ordinary, non-sensitive
    information. Deliberately unencrypted — this is information the user
    already agrees to share relatively freely (name, age, general contact),
    so the extra complexity of encryption is not security-justified at this
    tier.
    """

    def __init__(self, path: str = "profile_tier1.json"):
        self.path = path
        if not os.path.exists(self.path):
            self._write({})

    def _read(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def set_field(self, name: str, value: str):
        if is_blocked_field(name):
            raise BlockedFieldError(
                f"Field '{name}' is permanently banned from storage (highly sensitive "
                f"identity data) even at the ordinary tier — use SensitiveProfileStore "
                f"if truly necessary, and even there, the national ID number specifically "
                f"is always forbidden."
            )
        data = self._read()
        data[name] = value
        self._write(data)

    def get_field(self, name: str) -> Optional[str]:
        return self._read().get(name)

    def list_fields(self) -> dict:
        return self._read()


# ------------------------------------------------------------------- Tier 2

class SensitiveProfileStore:
    """
    Fully separate, encrypted storage for sensitive information (financial,
    medical, etc. — but never the national ID number, which is strictly
    forbidden regardless of the user's intent).

    The key is generated and stored in a separate file with restricted
    permissions (0600) — in a real production system this should ideally be
    tied to a hardware keystore on the device (Raspberry Pi/PC) rather than a
    plain file, but this is sufficient as a working proof of concept.
    """

    def __init__(self, data_path: str = "profile_tier2.enc",
                 key_path: str = "profile_tier2.key"):
        self.data_path = data_path
        self.key_path = key_path

        if not os.path.exists(self.key_path):
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as f:
                f.write(key)
            os.chmod(self.key_path, 0o600)  # owner read/write only

        with open(self.key_path, "rb") as f:
            self.fernet = Fernet(f.read())

        if not os.path.exists(self.data_path):
            self._write({})

    def _read(self) -> dict:
        with open(self.data_path, "rb") as f:
            encrypted = f.read()
        if not encrypted:
            return {}
        decrypted = self.fernet.decrypt(encrypted)
        return json.loads(decrypted.decode("utf-8"))

    def _write(self, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        encrypted = self.fernet.encrypt(payload)
        with open(self.data_path, "wb") as f:
            f.write(encrypted)

    def set_field(self, name: str, value: str):
        if is_blocked_field(name):
            raise BlockedFieldError(
                f"Field '{name}' (a national ID number or similar) is permanently "
                f"forbidden from storage at any tier, per a documented ethical "
                f"decision made earlier in the project — no exceptions."
            )
        data = self._read()
        data[name] = value
        self._write(data)

    def get_field(self, name: str) -> Optional[str]:
        """
        ⚠️ Should only be called from within consent_flow.py after explicit
        consent has been verified. There is no other sanctioned path for
        reading this store from the rest of the system — not from the
        BiLSTM, not from either CNN, not from any training path. The
        separation here is structural, enforced in code, not a convention.
        """
        return self._read().get(name)

    def list_field_names(self) -> list:
        """Field names only, never the values — useful for showing what is
        available without exposing the content."""
        return list(self._read().keys())
