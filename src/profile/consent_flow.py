"""
consent_flow.py
=================
Connects: the interlocutor's question (via STT) -> intent_classifier ->
profile_store, and implements the radically different consent logic between
the two tiers:

  Tier 1 (ordinary):
      1. The non-verbal user's display/interface shows text such as:
         "The person is asking for your name: Husham. Share it?"
      2. The user agrees with a normal "Yes" gesture (simple, from the
         standard sign-language vocabulary).
      3. On agreement: the answer is spoken immediately via TTS.

  Tier 2 (sensitive):
      1. Never disclosed automatically based on predicted context, no
         matter how clear the question seems.
      2. An explicit consent request is shown, visually distinct from
         Tier 1 (a warning + clearer text).
      3. Requires a distinct "Sensitive Consent Gesture" (a different,
         harder-to-trigger-by-accident gesture) — not the same gesture as
         normal consent.
      4. The value itself is only read from the encrypted
         SensitiveProfileStore at this exact moment, and is never loaded
         into memory before this call.

This file can actually be run and tested without any hardware (camera/real
TTS) by simulating gestures as plain values/strings — see
demo_profile_system.py.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from profile_store import NormalProfileStore, SensitiveProfileStore, Tier, ConsentRequiredError
from intent_classifier import match_intent, IntentMatch


class GestureType(str, Enum):
    YES_NORMAL = "yes_normal"                  # normal consent gesture (Tier 1)
    SENSITIVE_CONSENT = "sensitive_consent"     # distinct sensitive-consent gesture (Tier 2)
    NO = "no"
    NONE = "none"


@dataclass
class DisclosureResult:
    disclosed: bool
    field_name: str
    tier: Tier
    prompt_shown: str
    value: Optional[str] = None
    reason: str = ""


class ConsentFlowController:
    """
    gesture_provider: a function this layer calls to wait for / obtain the
        user's actual gesture (in a real system: calls the live sign-language
        recognition module reading from a D435 camera). Here it is a
        callable so it can be easily tested without hardware, and later
        replaced with a real function reading from a live camera.
    tts_speaker: a function that speaks the text aloud (a stand-in here;
        replaced with a real TTS engine).
    """

    def __init__(self,
                 normal_store: NormalProfileStore,
                 sensitive_store: SensitiveProfileStore,
                 gesture_provider: Callable[[], GestureType],
                 tts_speaker: Callable[[str], None]):
        self.normal_store = normal_store
        self.sensitive_store = sensitive_store
        self.gesture_provider = gesture_provider
        self.tts_speaker = tts_speaker
        self.audit_log = []  # audit log: records the event and time, never the sensitive value itself

    def _log(self, event: str, field_name: str, tier: Tier):
        self.audit_log.append({
            "timestamp": time.time(),
            "event": event,
            "field": field_name,
            "tier": tier.value,
        })

    def handle_question(self, question_text: str) -> Optional[DisclosureResult]:
        intent = match_intent(question_text)
        if intent is None:
            return None  # the question doesn't match any known profile field

        if intent.tier == Tier.NORMAL:
            return self._handle_normal(intent)
        else:
            return self._handle_sensitive(intent)

    # ------------------------------------------------------------- Tier 1

    def _handle_normal(self, intent: IntentMatch) -> DisclosureResult:
        value = self.normal_store.get_field(intent.field_name)
        if value is None:
            return DisclosureResult(
                disclosed=False, field_name=intent.field_name, tier=Tier.NORMAL,
                prompt_shown="", reason="Field is not set in the user's profile.",
            )

        # Show the simple consent prompt in exactly the required wording
        field_label = self._friendly_label(intent.field_name)
        prompt = f"The person is asking for your {field_label}: {value}. Share it?"
        self._log("normal_prompt_shown", intent.field_name, Tier.NORMAL)
        print(prompt)  # in the real UI: shown on the user's screen/glasses

        gesture = self.gesture_provider()
        if gesture == GestureType.YES_NORMAL:
            self.tts_speaker(value)
            self._log("normal_disclosed", intent.field_name, Tier.NORMAL)
            return DisclosureResult(True, intent.field_name, Tier.NORMAL, prompt, value=value)

        self._log("normal_declined_or_ignored", intent.field_name, Tier.NORMAL)
        return DisclosureResult(False, intent.field_name, Tier.NORMAL, prompt,
                                 reason="No normal Yes gesture was given.")

    # ------------------------------------------------------------- Tier 2

    def _handle_sensitive(self, intent: IntentMatch) -> DisclosureResult:
        # ⚠️ We do not read the value from the encrypted store until the
        # sensitive gesture is specifically confirmed, so the value never
        # exists in memory before consent is verified.
        field_label = self._friendly_label(intent.field_name)
        prompt = (
            f"⚠️ Sensitive information requested: your {field_label}. "
            f"This requires your explicit Sensitive Consent Gesture to share it — "
            f"a regular Yes gesture is NOT enough."
        )
        self._log("sensitive_prompt_shown", intent.field_name, Tier.SENSITIVE)
        print(prompt)

        gesture = self.gesture_provider()
        if gesture != GestureType.SENSITIVE_CONSENT:
            self._log("sensitive_declined_or_wrong_gesture", intent.field_name, Tier.SENSITIVE)
            return DisclosureResult(
                False, intent.field_name, Tier.SENSITIVE, prompt,
                reason="The distinct Sensitive Consent Gesture was not given.",
            )

        try:
            value = self.sensitive_store.get_field(intent.field_name)
        except ConsentRequiredError:
            self._log("sensitive_read_blocked", intent.field_name, Tier.SENSITIVE)
            return DisclosureResult(False, intent.field_name, Tier.SENSITIVE, prompt,
                                     reason="Read was rejected at the store level itself.")

        if value is None:
            return DisclosureResult(False, intent.field_name, Tier.SENSITIVE, prompt,
                                     reason="Field is not set.")

        self.tts_speaker(value)
        self._log("sensitive_disclosed", intent.field_name, Tier.SENSITIVE)
        return DisclosureResult(True, intent.field_name, Tier.SENSITIVE, prompt, value=value)

    @staticmethod
    def _friendly_label(field_name: str) -> str:
        labels = {
            "name": "name", "age": "age", "specialization": "specialization",
            "nationality": "nationality", "phone_number": "phone number",
            "bank_account": "bank account", "medical_condition": "medical condition",
        }
        return labels.get(field_name, field_name)
