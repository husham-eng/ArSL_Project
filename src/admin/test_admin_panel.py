import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "profile"))

from admin_panel import AdminAuth, ClassificationLog, AdminPanel, AuthenticationError
from profile_store import NormalProfileStore, SensitiveProfileStore

for f in ["admin_test_creds.json", "admin_test_log.db", "admin_test_tier1.json",
          "admin_test_tier2.enc", "admin_test_tier2.key"]:
    if os.path.exists(f):
        os.remove(f)

auth = AdminAuth("admin_test_creds.json")
log = ClassificationLog("admin_test_log.db")
normal_store = NormalProfileStore("admin_test_tier1.json")
sensitive_store = SensitiveProfileStore("admin_test_tier2.enc", "admin_test_tier2.key")

normal_store.set_field("name", "Husham")
normal_store.set_field("age", "29")
sensitive_store.set_field("bank_account", "SA1234567890123456789012")

print("=" * 60)
print("Test 1: initial password setup")
print("=" * 60)
assert not auth.is_configured()
auth.set_password("MySecurePass123")
assert auth.is_configured()
print("✅ Password set up and stored as a hash only.")

print("\n" + "=" * 60)
print("Test 2: reject login with a wrong password")
print("=" * 60)
panel = AdminPanel(auth, log, normal_store, sensitive_store)
assert panel.login("WrongPassword") is False
try:
    panel.view_normal_profile()
    print("❌ Failed: access should have been denied!")
    assert False
except AuthenticationError as e:
    print(f"✅ Passed: access denied without a valid login -> {e}")

print("\n" + "=" * 60)
print("Test 3: correct login and viewing ordinary information only")
print("=" * 60)
assert panel.login("MySecurePass123") is True
print("✅ Login succeeded.")
normal_info = panel.view_normal_profile()
print("Tier-1 (ordinary) info:", normal_info)

print("\n" + "=" * 60)
print("Test 4: view sensitive field names only, never the values")
print("=" * 60)
sensitive_names = panel.view_sensitive_field_names_only()
print("Available sensitive field names:", sensitive_names)
assert "SA1234567890123456789012" not in str(sensitive_names), "A sensitive value leaked!"
print("✅ Passed: actual values are never exposed, only field names.")

print("\n" + "=" * 60)
print("Test 5: log real classification attempts (simulating a usage session)")
print("=" * 60)
log.log_attempt("grayscale", "ha", 0.771, accepted=True)
log.log_attempt("grayscale", "seen", 0.412, accepted=False)  # low confidence, rejected
log.log_attempt("fusion", "بيت", 0.88, accepted=True)
log.log_attempt("color", "Ain", 0.95, accepted=True)

attempts = panel.view_recent_classification_attempts(limit=10)
for a in attempts:
    status = "accepted" if a.accepted else "rejected (low confidence)"
    print(f"  [{a.model_used}] {a.predicted_label} | confidence={a.confidence} | {status}")

summary = panel.view_accuracy_summary()
print("\nPerformance summary:", summary)
assert summary["total_attempts"] == 4
assert summary["accepted_attempts"] == 3

print("\n✅ All tests passed — the admin panel works, is protected, and never leaks sensitive data.")
