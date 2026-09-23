"""
intent_classifier.py
======================
Analyzes the interlocutor's question (STT text) and determines: which
profile field is being asked about, and is it ordinary (Tier 1) or
sensitive (Tier 2)?

This is an initial rule/keyword-based version — sufficient to get started
and to prove the concept, and upgradeable later to a small language-model
intent classifier without changing the external interface (match_intent
always returns the same shape).
"""

import re
from dataclasses import dataclass
from typing import Optional

from profile_store import Tier


@dataclass
class IntentMatch:
    field_name: str
    tier: Tier
    confidence: float
    matched_pattern: str


# Each field is linked to question patterns the interlocutor might ask
# (Arabic + English — kept as-is since these are the actual languages
# interlocutors speak, not documentation) and its default tier.
# Note: tier classification here is an initial default only — the user is
# the one who ultimately decides when a field is first added (see
# consent_flow.register_field).
FIELD_PATTERNS = {
    "name": {
        "tier": Tier.NORMAL,
        "patterns": [
            r"what'?s your name", r"what is your name", r"who are you",
            r"شو اسمك", r"ما اسمك", r"وش اسمك", r"اسمك ايه",
        ],
    },
    "age": {
        "tier": Tier.NORMAL,
        "patterns": [
            r"how old are you", r"what'?s your age",
            r"كم عمرك", r"ما عمرك", r"عندك كم سنة",
        ],
    },
    "specialization": {
        "tier": Tier.NORMAL,
        "patterns": [
            r"what do you (study|do)", r"your (major|specialization|job)",
            r"شو تخصصك", r"ايش شغلتك", r"تخصصك ايه",
        ],
    },
    "nationality": {
        "tier": Tier.NORMAL,
        "patterns": [
            r"where are you from", r"what'?s your nationality",
            r"من وين انت", r"ايش جنسيتك", r"من أي بلد",
        ],
    },
    "phone_number": {
        "tier": Tier.SENSITIVE,   # left to the user to decide; defaults to sensitive out of caution
        "patterns": [
            r"what'?s your (phone|number)", r"can i (call|contact) you",
            r"رقم جوالك", r"كيف اتواصل معك",
        ],
    },
    "bank_account": {
        "tier": Tier.SENSITIVE,
        "patterns": [
            r"bank account", r"your iban", r"credit card",
            r"رقم حسابك", r"ايبان", r"بطاقتك البنكية",
        ],
    },
    "medical_condition": {
        "tier": Tier.SENSITIVE,
        "patterns": [
            r"medical condition", r"do you have (a disability|any condition)",
            r"حالتك الصحية", r"عندك مرض",
        ],
    },
    # Note: "national_id" deliberately has no entry here at all — this field
    # is banned entirely at the code level in profile_store.py, so there is
    # no need to even attempt to recognize it.
}


def match_intent(question_text: str) -> Optional[IntentMatch]:
    """
    Attempts to match the question text (post-STT) against a known profile
    field. Returns None if no match is found (i.e. the question is not
    about a known personal-info field).
    """
    text = question_text.strip().lower()
    for field_name, spec in FIELD_PATTERNS.items():
        for pattern in spec["patterns"]:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return IntentMatch(
                    field_name=field_name,
                    tier=spec["tier"],
                    confidence=0.9,  # fixed for now since this is rule-based; replaced by a real
                                     # confidence score once upgraded to a trained classifier
                    matched_pattern=pattern,
                )
    return None
