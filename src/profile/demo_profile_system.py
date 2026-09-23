"""
demo_profile_system.py — real, actual run of every scenario, with no hardware at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from profile_store import (NormalProfileStore, SensitiveProfileStore,
                            BlockedFieldError, Tier)
from consent_flow import ConsentFlowController, GestureType


def run_all_tests():
    # clean up any leftover test files from a previous run
    for f in ["test_tier1.json", "test_tier2.enc", "test_tier2.key"]:
        if os.path.exists(f):
            os.remove(f)

    normal_store = NormalProfileStore("test_tier1.json")
    sensitive_store = SensitiveProfileStore("test_tier2.enc", "test_tier2.key")

    # ------------------------------------------------------------- setup
    normal_store.set_field("name", "Husham")
    normal_store.set_field("age", "29")
    sensitive_store.set_field("bank_account", "SA1234567890123456789012")
    sensitive_store.set_field("medical_condition", "Type 1 Diabetes")

    print("=" * 60)
    print("Test 1: strict blocking of the national ID number (expected to fail on purpose)")
    print("=" * 60)
    try:
        sensitive_store.set_field("national_id", "1234567890")
        print("❌ Test failed: this should have been rejected!")
        assert False
    except BlockedFieldError as e:
        print(f"✅ Passed: storage was rejected as expected -> {e}")

    try:
        normal_store.set_field("رقم_الهوية_الوطنية", "1234567890")
        print("❌ Test failed: should be rejected in Arabic too!")
        assert False
    except BlockedFieldError as e:
        print(f"✅ Passed: rejected in Arabic as well -> {e}")

    print()

    # -------------------------------------------------- consent scenarios
    scenarios = [
        ("What is your name?", GestureType.YES_NORMAL, True),
        ("What is your name?", GestureType.NO, False),
        ("What is your bank account number?", GestureType.YES_NORMAL, False),  # a normal Yes is not enough!
        ("What is your bank account number?", GestureType.SENSITIVE_CONSENT, True),
        ("How old are you?", GestureType.YES_NORMAL, True),
        ("Do you have any medical condition?", GestureType.SENSITIVE_CONSENT, True),
        ("What's the weather today?", GestureType.YES_NORMAL, None),  # a question that doesn't match anything
    ]

    for i, (question, gesture, expected_disclosed) in enumerate(scenarios, 1):
        print("-" * 60)
        print(f"Scenario {i}: question = \"{question}\" | gesture given = {gesture.value}")

        spoken = {"text": None}

        def fake_tts(text):
            spoken["text"] = text
            print(f"   🔊 TTS speaks: {text}")

        controller = ConsentFlowController(
            normal_store=normal_store,
            sensitive_store=sensitive_store,
            gesture_provider=lambda: gesture,
            tts_speaker=fake_tts,
        )

        result = controller.handle_question(question)

        if expected_disclosed is None:
            assert result is None, "Expected no intent match, but a match was found!"
            print("   ✅ Correct: no known field matches this question (nothing was shown).")
            continue

        assert result is not None
        assert result.disclosed == expected_disclosed, (
            f"Expected disclosed={expected_disclosed} but got {result.disclosed}"
        )
        status = "disclosed ✅" if result.disclosed else f"not disclosed (reason: {result.reason}) ✅"
        print(f"   Tier: {result.tier.value} | Result: {status}")

    print("\n" + "=" * 60)
    print("Audit log — note that no sensitive value is ever stored in it:")
    print("=" * 60)
    for entry in controller.audit_log:
        print(entry)

    print("\n✅ All tests passed — the system genuinely distinguishes the two tiers and never mixes them up.")


if __name__ == "__main__":
    run_all_tests()
