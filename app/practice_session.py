"""
practice_session.py
======================
A local desktop practice/testing application for the Arabic Sign Language
assistant, with a modern, chat-style interface (inspired by messaging apps:
the interlocutor's question appears as one bubble, and the signer's
recognized-and-corrected answer appears as a reply bubble on the other
side, spoken aloud via TTS).

This ties together every piece built and tested so far in this project:
  - The full dual-branch fusion pipeline (GrayscaleCNN + ColorSignResNet18 +
    ModalityFusionModel, average fusion) when color_sign_resnet18_best.pt is
    present at the repository root -- the same pipeline evaluated end-to-end
    on real images in evaluation/evaluate_real_fusion.py (97.90% color-alone,
    96.56% fused, docs/scientific_paper.docx Section 4.7), reproducing
    Figure 3's "Per-Frame Prediction Pipeline" exactly: the live, ROI-cropped
    color frame is preprocessed once, a grayscale-3ch copy is derived from
    that SAME crop (not a second capture), both branches run a forward pass,
    and their probabilities are fused. Falls back automatically to
    GrayscaleCNN alone (grayscale_simplecnn_best.pt) if the color checkpoint
    or src/color/color_sign_resnet18.py is not present, so the app still
    runs with only Track B complete.
  - Fixed-interval letter capture (every CAPTURE_INTERVAL_SEC seconds, a
    clear commit either way -- see "دورة التقاط ثابتة" below) feeding
    guess_word_with_context ("خوارزمية التكهن": dictionary + context-aware
    word inference, weighted by how much of the word was actually
    confidently detected -- correct_word/predict_words_from_prefix remain
    available too, for the simpler letters-only and autocomplete cases)
  - ClassificationLog (from the admin panel) for a real usage log

Run from the repository root:
    python app/practice_session.py

Requirements beyond the core repo (see app/requirements_app.txt):
    opencv-python, customtkinter, pyttsx3, arabic-reshaper, python-bidi, gTTS,
    sounddevice, SpeechRecognition
"""

import json
import os
import queue
import shutil
import sys
import tempfile
from collections import Counter
import threading
import time
import tkinter as tk

import cv2
import customtkinter as ctk
import torch
from PIL import Image, ImageTk
from torchvision import transforms
import arabic_reshaper
from bidi.algorithm import get_display

# ---------------------------------------------------------------- wiring up
# يميّز هذا القسم بين وضعين: التشغيل من المصدر (python app/practice_session.py،
# كما كان الحال طوال هذا المشروع)، والتشغيل كملف .exe مجمَّع عبر PyInstaller
# (راجع packaging/build_exe.bat). في الوضع المجمَّع، __file__ يشير إلى مجلد
# مؤقت (sys._MEIPASS) يُستخرج ويُحذف مع كل تشغيل، فلا يصلح لتخزين أي شيء
# يجب أن يبقى بين الجلسات (سجل التصنيف، لقطات التوثيق) — لذلك BASE_DIR
# (الموارد المرفقة للقراءة فقط) و WRITE_DIR (بيانات المستخدم القابلة للكتابة،
# تُحفظ بجانب ملف .exe نفسه) منفصلان عمدًا.
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    WRITE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WRITE_DIR = BASE_DIR

APP_DIR = os.path.join(BASE_DIR, "app")
REPO_ROOT = BASE_DIR

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "grayscale"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "color"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "fusion"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "letter_to_word"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "admin"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "profile"))

from image_cnn_model import SimpleCNN
from modality_fusion_model import ModalityFusionModel, rgb_to_grayscale_3ch
from class_mapping import build_class_mappings, ARASL_TO_ARABIC, aasl_folder_to_arasl_key
from spelling_correction import predict_words_from_prefix, guess_word_with_context
from admin_panel import ClassificationLog

try:
    from color_sign_resnet18 import ColorSignResNet18
except ImportError:
    # src/color/color_sign_resnet18.py not present in this checkout -- the
    # app still runs in grayscale-only fallback mode without it, see
    # FusionPredictor / _load_model_async below.
    ColorSignResNet18 = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as MPBaseOptions
except ImportError:
    # mediapipe not installed -- HandTracker below cannot be constructed,
    # and the app falls back to the fixed-center ROI box automatically.
    mp = None
    mp_vision = None
    MPBaseOptions = None

HAND_SIGNS_DIR = os.path.join(APP_DIR, "hand_signs")
# عكس خريطة الحرف->المجلد، للانتقال من حرف الكلمة المستهدفة إلى صورة الإشارة المناسبة
CHAR_TO_SIGN_KEY = {arabic: key for key, arabic in ARASL_TO_ARABIC.items()}
CHAR_TO_SIGN_KEY.setdefault("ا", "aleff")  # الألف العادية بدون همزة تُعرَض بنفس إشارة "أ"


def word_to_sign_keys(word: str):
    """يحوّل كلمة عربية إلى تسلسل مفاتيح صور الإشارة المطلوبة لتنفيذها،
    مع معاملة "لا" ككيان إشارة واحد (وليس حرفين منفصلين) لأنها إشارة مركّبة
    قائمة بذاتها في AASL/ArASL."""
    if word == "لا":
        return ["la"]
    keys = []
    for ch in word:
        key = CHAR_TO_SIGN_KEY.get(ch)
        if key:
            keys.append(key)
    return keys

CHECKPOINT_PATH = os.path.join(REPO_ROOT, "grayscale_simplecnn_best.pt")
COLOR_CHECKPOINT_PATH = os.path.join(REPO_ROOT, "color_sign_resnet18_best.pt")
SCENARIOS_PATH = os.path.join(APP_DIR, "conversation_scenarios.json")
# مجلد لقطات التوثيق (زر "📸 التقط لقطة للتوثيق") — صور حقيقية من الجلسة
# الحية تُستخدم لاحقًا في قسم الأسلوب بالورقة البحثية، وليست جزءًا من بيانات
# التدريب أو التقييم.
SCREENSHOTS_DIR = os.path.join(WRITE_DIR, "screenshots")
# --- التعرّف الصوتي الحقيقي (STT) لسؤال المحاور -- "الوضع الحر" ---
# في وضع التمرين (الافتراضي)، نص "سؤال المحاور" يأتي من سيناريو محفوظ مسبقًا
# في conversation_scenarios.json — مفيد للتمرين والاختبار المتكرر بنفس
# الظروف، لكنه محاكاة، وليس تعرّفًا صوتيًا فعليًا. "الوضع الحر" يستبدل هذا
# بتسجيل صوتي حقيقي من الميكروفون عبر Google Web Speech API (نفس فكرة
# الاعتماد على خدمة سحابية مجانية عبر الإنترنت المستخدمة أصلًا لنطق العربي
# عبر gTTS في هذا المشروع — راجع _speak_via_gtts)، ونستخدم نص السؤال
# المُتعرَّف عليه فعليًا كسياق لتصحيح تهجئة الكلمة المُشارة، بدل أي نص ثابت.
STT_LANGUAGE = "ar-SA"       # رمز اللغة العربية (السعودية) لخدمة Google للتعرّف الصوتي
STT_RECORD_SECONDS = 4.0     # مدة تسجيل السؤال الواحد؛ كافية لجملة قصيرة عادية
STT_SAMPLE_RATE = 16000      # هرتز — معدل قياسي مقبول لخدمات التعرّف الصوتي
# --- دورة التقاط ثابتة كل CAPTURE_INTERVAL_SEC ثانية ---
# المشكلة اللي حلّها هذا التصميم: بالآلية القديمة (segmenter يُصدر حرفًا
# فقط لو استقرت نافذة الإطارات على نفس التنبؤ)، لو ما استقرت النافذة أبدًا
# (إضاءة سيئة، إشارة غير مألوفة للنموذج...) فالمستخدم ما يحصل على أي إشارة
# إطلاقًا إنه يكمل للحرف التالي أو يعيد المحاولة — سكوت تام بلا توثيق.
# الحل: **كل CAPTURE_INTERVAL_SEC ثانية بالضبط، وبصرف النظر عن النتيجة**،
# النظام يوثّق دورة كاملة (حرف مكتشف بثقة كافية، أو "لا شيء" صراحة)، يعطي
# إشارة بصرية واضحة (لون مربع الدليل)، ثم يبدأ دورة جديدة فورًا — إيقاع
# ثابت يعرف معه المستخدم دائمًا "الدورة خلصت، انتقل للحرف التالي أو أعد
# المحاولة"، دون انتظار غامض. الحروف "الفارغة" (لا اكتشاف بثقة كافية) لا
# تُضاف لـ word_buffer (المعروض وشريط الدليل)، لكنها تُسجَّل في
# captured_slots وتُستخدم لاحقًا بخوارزمية التكهن (guess_word_with_context
# في spelling_correction.py) لحساب نسبة تغطية الاكتشاف بالكلمة كاملة.
CAPTURE_INTERVAL_SEC = 2.0
CAPTURE_STABILITY_RATIO = 0.6  # نسبة الإطارات داخل الدورة الواحدة اللي لازم تتفق على نفس الحرف لاعتماده
# --- نهاية الكلمة: مهلة صارمة، لا احتياطية ---
# بمجرد مرور WORD_RECOGNITION_TIMEOUT_SEC ثانية بالضبط من بداية اكتشاف
# الكلمة (لحظة أول دورة التقاط، لا أول حرف مؤكَّد -- راجع _tick_capture_cycle)،
# يتوقف الاكتشاف والتصنيف فورًا، ويلعب مربع الدليل حركة انكماش مرئية نحو
# منتصف الإطار (_start_end_of_word_animation) توضح للمستخدم بلا لبس إن
# الوقت انتهى، ثم تُنطق الكلمة المتراكمة (أو تُسقط بصمت لو ما تأكّد أي
# حرف). زر "أنهِ الكلمة الآن" اليدوي يبقى متاحًا لإنهاء الكلمة قبل
# الوصول لهذي المهلة، لا بديلًا عنها.
WORD_RECOGNITION_TIMEOUT_SEC = 10.0
CONFIDENCE_THRESHOLD = 0.6
MAX_WORD_PREDICTIONS = 5  # عدد الكلمات المقترحة المعروضة أثناء التهجئة الحية

# نموذج GrayscaleCNN اتدرّب على صور ArASL: يد مقصوصة فقط على خلفية بسيطة،
# وليس على الإطار الكامل من الكاميرا (وجه/كتفين/خلفية الغرفة). تمرير الإطار
# كاملًا للتصنيف يخلق فجوة كبيرة بين بيانات التدريب ومدخلات الاستخدام الفعلي،
# ويُنتج تصنيفات شبه عشوائية بغض النظر عن جودة الإضاءة.
#
# الحل الأساسي: كشف/تتبع يد حقيقي عبر MediaPipe Tasks (HandTracker أدناه) —
# مربع الدليل يتحرك مع اليد الفعلية بدل انتظار المستخدم إنه يقرّب يده لمربع
# ثابت بمنتصف الشاشة. يحتاج ملف نموذج محلي (HAND_LANDMARKER_MODEL_PATH)
# غير مُضمَّن بهذا المستودع (حجمه ~8MB) — راجع README قسم "تفعيل تتبع اليد
# الحقيقي" لرابط التحميل الرسمي ومكان وضعه. إن كان mediapipe غير مثبَّت، أو
# ملف النموذج غير موجود، أو لم تُكتشف يد بإطار معيّن، يرجع النظام تلقائيًا
# لمربع ثابت بمنتصف الصورة (HAND_ROI_FRACTION) كخط دفاع أخير يبقي التطبيق
# يعمل بلا أي اعتمادية إضافية.
HAND_LANDMARKER_MODEL_PATH = os.path.join(APP_DIR, "models", "hand_landmarker.task")
HAND_BOX_PADDING_FRACTION = 0.35  # يوسّع صندوق اليد المكتشف (الملاصق للأصابع) بهذي
                                   # النسبة على كل جهة قبل القص -- خُفِّضت لنصف قيمتها
                                   # الأصلية (0.8) تقليصًا لحجم المربع كما طُلب
HAND_ROI_FRACTION = 0.27  # نسبة طول ضلع المربع الثابت الاحتياطي من أصغر بُعد بالإطار --
                            # خُفِّضت لنصف قيمتها الأصلية (0.55) تقليصًا لحجم المربع
ROI_BOX_COLOR_DEFAULT = (0, 255, 0)      # أخضر: لسا ينتظر/يجرّب — استمر بنفس الإشارة أو عدّلها
ROI_BOX_COLOR_CONFIRMED = (0, 210, 255)  # سماوي: الحرف اتسجّل فعلاً — يمكنك الانتقال للحرف التالي
ROI_BOX_COLOR_CYCLE_EMPTY = (255, 165, 0)  # برتقالي: خلصت دورة كاملة (CAPTURE_INTERVAL_SEC) بلا اكتشاف واثق — أعد المحاولة
ROI_BOX_COLOR_NO_HAND = (120, 120, 120)  # رمادي: تتبع اليد مفعّل لكن لم تُكتشف يد بهذا الإطار
# مدة بقاء اللون السماوي بعد تأكيد حرف، بالثواني — قصيرة بما يكفي ما تعطّل
# الإيقاع، لكن كافية للعين إنها تلاحظها (فريم واحد ~80ms يمر بسرعة جدًا).
CONFIRMED_HIGHLIGHT_DURATION_SEC = 0.8

# ------------------------------------------------------------ camera setup
# جهاز الكاميرا المدمجة في هذا الجهاز معطوب على مستوى العتاد (يُرجع دائمًا إطارات
# سوداء رغم أن isOpened() تعيد True) — راجع قسم 4.6.1 في الورقة العلمية للتشخيص
# الكامل. كذلك الـ RealSense D435 يعرض عادة عدة "كاميرات" منفصلة على مستوى
# ويندوز/DirectShow لنفس الجهاز الفيزيائي الواحد: تيار اللون RGB، وتيار (أو
# تيارين) الأشعة تحت الحمراء (IR)، وربما تيار العمق (Depth) — وكلها تُرقّم
# كفهارس OpenCV منفصلة. الفحص الأول (مجرد "هل الإطار غير أسود؟") لا يكفي
# لتمييزها، لأن تيار IR تحت إضاءة غرفة عادية يعطي صورة رمادية غير سوداء لكنها
# تبدو "سوداء تقريبًا" للعين ولا تصلح لتصنيف الحروف بالألوان. لذلك نفحص أيضًا
# التشبع اللوني (saturation): الصورة الملوّنة الحقيقية لها تشبع واضح، بينما أي
# تيار IR/Depth تكون قنواته الثلاث شبه متطابقة (تشبع قريب من الصفر).
#
# لإجبار فهرس معيّن يدويًا (مثلًا بعد مراجعة سطر التشخيص أدناه وتحديد رقم
# التيار الملوّن للـ D435 بنفسك)، عيّن متغير البيئة ARSL_CAMERA_INDEX قبل
# التشغيل، مثال على Windows:
#     set ARSL_CAMERA_INDEX=1
#     python app/practice_session.py
CAMERA_INDEX_OVERRIDE = os.environ.get("ARSL_CAMERA_INDEX")
CAMERA_SCAN_RANGE = range(6)          # يفحص الفهارس 0..5 (كافٍ لتغطية كل تيارات D435 + الكاميرا المدمجة)
BLACK_FRAME_MEAN_THRESHOLD = 8.0      # متوسط شدة البكسل تحت هذا الرقم يُعتبر "إطار أسود"
COLOR_SATURATION_THRESHOLD = 15.0     # متوسط التشبع تحت هذا الرقم يُعتبر تيار رمادي/IR وليس تيار ألوان حقيقي
# بعض الكاميرات (خصوصًا D435 عبر UVC/DirectShow) تفتح بنجاح (isOpened=True)
# لكن ترفض القراءة على الصيغة/الدقة الافتراضية التي يطلبها OpenCV تلقائيًا،
# فنفرض دقة شائعة ومدعومة على أغلب الأجهزة، ونجرّب أكثر من backend، ونعطي
# الجهاز وقتًا أطول للتهيئة بدل خمس محاولات فورية فقط.
PROBE_BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]
PROBE_READ_ATTEMPTS = 15
PROBE_READ_DELAY_SEC = 0.15
# بعد أول إطار ناجح، بعض الكاميرات (وخصوصًا عبر UVC/DirectShow) يبدأ التعريض
# الضوئي التلقائي (auto-exposure) عندها شبه صفري ويحتاج عشرات الإطارات لكي
# يستقر على إضاءة الغرفة الفعلية — أول إطار ناجح لا يعني أن الصورة استقرت.
# لذلك نستهلك عدد إطارات "إحماء" إضافي بعد أول نجاح، ثم نقيس السطوع من
# أفضل إطار (الأعلى سطوعًا) بين آخر عدة إطارات، لا من أول إطار فقط.
PROBE_WARMUP_FRAMES = 40
PROBE_WARMUP_DELAY_SEC = 0.03
PROBE_FINAL_SAMPLE_FRAMES = 5


def _probe_camera(idx: int):
    """يفتح فهرس كاميرا مرشّح مؤقتًا (يجرّب عدة backends)، يفرض دقة شائعة،
    يقرأ عدة إطارات مع فاصل زمني بينها (تهيئة)، ثم يعطي الجهاز فترة إحماء
    إضافية لاستقرار التعريض الضوئي التلقائي قبل القياس. يرجع معلومات
    تشخيصية عنه (الـ backend الذي نجح، السطوع، والتشبع اللوني) أو None إذا
    تعذر فتحه أو قراءة أي إطار صالح منه بأي backend. يُغلق الجهاز دائمًا
    قبل الرجوع."""
    for backend_name, backend in PROBE_BACKENDS:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # نفرض دقة شائعة ومدعومة على أغلب الكاميرات (بما فيها D435) بدل ما
        # نترك OpenCV يطلب الدقة الافتراضية للجهاز، والتي قد تكون عالية جدًا
        # (مثلاً 1920x1080 لتيار اللون) وتفشل القراءة بسببها عبر USB2 أو
        # صيغة ضغط غير مدعومة من الـ backend.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ok, frame = False, None
        for _ in range(PROBE_READ_ATTEMPTS):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(PROBE_READ_DELAY_SEC)

        if not ok or frame is None:
            cap.release()
            continue  # هذا الـ backend فتح الجهاز لكن فشل بقراءة أي إطار — جرّب التالي

        # إحماء إضافي: أول إطار ناجح لا يعني أن التعريض الضوئي استقر بعد،
        # فنستهلك إطارات إضافية بفاصل زمني يقارب معدل الالتقاط الطبيعي.
        for _ in range(PROBE_WARMUP_FRAMES):
            cap.read()
            time.sleep(PROBE_WARMUP_DELAY_SEC)

        # نقيس من أفضل إطار (الأعلى سطوعًا) بين عدة إطارات أخيرة، لا إطار واحد
        # فقط، تفاديًا لالتقاط إطار عابر غامق أثناء تذبذب التعريض الضوئي.
        best_brightness, best_saturation = 0.0, 0.0
        for _ in range(PROBE_FINAL_SAMPLE_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None:
                brightness = float(frame.mean())
                if brightness > best_brightness:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    best_brightness = brightness
                    best_saturation = float(hsv[:, :, 1].mean())
            time.sleep(PROBE_WARMUP_DELAY_SEC)
        cap.release()

        if best_brightness < BLACK_FRAME_MEAN_THRESHOLD:
            kind = "black"
        elif best_saturation < COLOR_SATURATION_THRESHOLD:
            kind = "grayscale/IR-or-depth"
        else:
            kind = "color"
        return {
            "index": idx, "backend_name": backend_name, "backend": backend,
            "brightness": best_brightness, "saturation": best_saturation, "kind": kind,
        }
    return None


def scan_cameras(scan_range=CAMERA_SCAN_RANGE):
    """يفحص كل فهارس الكاميرا المرشّحة (بعدة backends لكل فهرس) ويرجع قائمة
    معلومات تشخيصية عن كل ما استجاب (بأي حالة: أسود، رمادي/IR، أو ملوّن)،
    ويطبع سطر تشخيص لكل واحد على الطرفية (terminal)."""
    candidates = []
    for idx in scan_range:
        info = _probe_camera(idx)
        if info is None:
            print(f"[camera scan] index {idx}: not available (all backends failed to read a frame)")
            continue
        print(
            f"[camera scan] index {idx} ({info['backend_name']}): "
            f"brightness={info['brightness']:.1f} saturation={info['saturation']:.1f} -> {info['kind']}"
        )
        candidates.append(info)
    return candidates


def find_working_camera_from(candidates):
    """يختار أفضل مرشّح من قائمة نتائج scan_cameras() بنفس منطق الأفضلية
    (ملوّن أولاً، ثم أي غير أسود)، ويفتحه فعليًا. يرجع (index, cap, kind)
    أو (None, None, None)."""
    color_candidates = [c for c in candidates if c["kind"] == "color"]
    nonblack_candidates = [c for c in candidates if c["kind"] != "black"]

    if color_candidates:
        chosen = max(color_candidates, key=lambda c: c["brightness"])
    elif nonblack_candidates:
        chosen = max(nonblack_candidates, key=lambda c: c["brightness"])
    else:
        return None, None, None

    cap = cv2.VideoCapture(chosen["index"], chosen["backend"])
    if not cap.isOpened():
        return None, None, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # نفس فترة الإحماء المستخدمة أثناء الفحص: الفتح الفعلي هنا جهاز جديد
    # (المؤقت المستخدم بالفحص أُغلق بالفعل)، فيحتاج فرصة مماثلة لاستقرار
    # التعريض الضوئي التلقائي قبل ما يبدأ العرض الحي، وإلا تظهر الصورة
    # غامقة لثانية أو ثانيتين قبل أن تتضح تلقائيًا.
    for _ in range(PROBE_WARMUP_FRAMES):
        cap.read()
        time.sleep(PROBE_WARMUP_DELAY_SEC)
    return chosen["index"], cap, chosen["kind"]


def find_working_camera(scan_range=CAMERA_SCAN_RANGE):
    """يفحص كل الفهارس المرشّحة ثم يرجع (index, cap, kind) لأفضل واحد، بنفس
    منطق الأفضلية في find_working_camera_from(). يرجع (None, None, None) إذا
    فشل كل الفهارس."""
    return find_working_camera_from(scan_cameras(scan_range))


def ar(text: str) -> str:
    """Reshape + reorder Arabic text so it renders correctly in Tkinter
    labels (Tkinter does not perform Arabic shaping/BiDi automatically)."""
    return get_display(arabic_reshaper.reshape(text))


def rgb_to_grayscale_3ch_pil(pil_img: Image.Image) -> Image.Image:
    """Converts a genuine color frame to a 3-channel replicated grayscale
    image, matching the domain GrayscaleCNN was trained on (see
    modality_fusion_model.py's rgb_to_grayscale_3ch for the tensor
    equivalent of this same idea)."""
    gray = pil_img.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


# --------------------------------------------------------------- hand ROI

class HandTracker:
    """Real hand detection via MediaPipe Tasks' HandLandmarker, replacing
    the earlier fixed-center ROI box: the guide box now follows the
    signer's actual hand instead of requiring them to hold it steady at a
    fixed screen position.

    Uses the modern MediaPipe Tasks API (mediapipe>=0.10), not the legacy
    `mp.solutions.hands` module, which newer mediapipe releases no longer
    ship at all -- code written against the old API silently fails to
    import on a fresh `pip install mediapipe`.

    Raises ImportError if mediapipe itself is not installed, or
    FileNotFoundError if HAND_LANDMARKER_MODEL_PATH does not exist -- both
    are caught by the caller (see _load_hand_tracker), which falls back to
    the fixed-center box so the app still runs without this dependency.

    ⚠️ Real bug found and worked around: MediaPipe's underlying C++ model
    loader can fail with "Unable to open file at <path>" on Windows when
    the full path contains non-ASCII characters (e.g. Arabic folder names)
    -- even though the file genuinely exists and Python itself opens it
    without any problem. This is a known category of Unicode-path bug in
    C/C++ libraries ported from Linux, where narrow (non-wide-char) file
    APIs are used internally. Confirmed in this project: a real user's
    project folder was named "...مشروع ترجمة لغة الاشارة بكاميرا العمق...",
    and MediaPipe reported that exact correct path as unopenable while
    Windows Explorer and Python's own os.path.getsize() on the same path
    worked fine. The fix below copies the model to a plain-ASCII temp path
    before handing it to MediaPipe, whenever the given path is not already
    ASCII-safe, rather than requiring the user to rename/relocate their
    entire project folder.
    """

    def __init__(self, model_path: str):
        if mp_vision is None:
            raise ImportError("mediapipe is not installed -- run: pip install mediapipe")

        safe_path = model_path
        try:
            model_path.encode("ascii")
        except UnicodeEncodeError:
            safe_path = os.path.join(tempfile.gettempdir(), "arsl_hand_landmarker.task")
            need_copy = (
                not os.path.exists(safe_path)
                or os.path.getsize(safe_path) != os.path.getsize(model_path)
            )
            if need_copy:
                shutil.copyfile(model_path, safe_path)
            print(f"[info] Model path contains non-ASCII characters (a known MediaPipe/Windows "
                  f"limitation) -- using a copy at {safe_path} instead.")

        options = mp_vision.HandLandmarkerOptions(
            base_options=MPBaseOptions(model_asset_path=safe_path),
            num_hands=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        # create_from_options raises FileNotFoundError itself if the (now
        # ASCII-safe) path does not exist -- no separate os.path.exists
        # check needed here.
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect_box(self, frame_rgb, padding_fraction: float = HAND_BOX_PADDING_FRACTION):
        """frame_rgb: an HxWx3 uint8 numpy array (a full live camera frame,
        NOT a crop). Returns (x1, y1, x2, y2) -- a padded, frame-clamped,
        square box around the first/only detected hand -- or None if no
        hand was detected in this particular frame."""
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]  # num_hands=1, so at most one hand
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Pad generously around the tight landmark bounding box (which
        # hugs just the finger/palm keypoints) so the crop keeps the whole
        # hand shape with margin, not just the fingertips -- and make it
        # square, since both LetterPredictor and FusionPredictor resize to
        # a square input regardless, so a square source crop avoids
        # unnecessary aspect-ratio distortion.
        box_w, box_h = x2 - x1, y2 - y1
        side = max(box_w, box_h) * (1 + padding_fraction)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = side / 2
        nx1, ny1 = cx - half, cy - half
        nx2, ny2 = cx + half, cy + half

        # Clamp to frame bounds. A simple clamp (rather than re-centering)
        # is enough here: the hand is virtually never at the extreme edge
        # of a reasonably-framed webcam shot.
        nx1 = max(0.0, nx1)
        ny1 = max(0.0, ny1)
        nx2 = min(float(w), nx2)
        ny2 = min(float(h), ny2)
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return (int(nx1), int(ny1), int(nx2), int(ny2))


# ------------------------------------------------------------------- model

class LetterPredictor:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.input_size = checkpoint["input_size"]
        self.model = SimpleCNN(num_classes=len(self.classes), input_size=self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ])

    def predict(self, pil_frame: Image.Image):
        img = rgb_to_grayscale_3ch_pil(pil_frame.convert("RGB"))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        folder_name = self.classes[idx.item()]
        arabic_letter = ARASL_TO_ARABIC.get(folder_name, "?")
        return arabic_letter, folder_name, conf.item()


class FusionPredictor:
    """
    Real-time predictor using the trained dual-branch fusion pipeline
    (GrayscaleCNN + ColorSignResNet18 + ModalityFusionModel, average
    fusion), reproducing Figure 3's "Per-Frame Prediction Pipeline" from
    docs/scientific_paper.docx exactly: the live, ROI-cropped color frame
    (see HAND_ROI_FRACTION / _camera_loop) is preprocessed once, a
    grayscale-3ch copy is derived from that SAME crop (not a second
    capture) via rgb_to_grayscale_3ch, both branches run a forward pass,
    and their probabilities are fused with the unweighted-average strategy
    actually evaluated end-to-end in Section 4.7 (97.90% color-alone /
    21.26% grayscale-alone / 96.56% fused on 1,571 real held-out AASL
    images).

    Uses AASL's 31-class space as the output space (the same convention as
    evaluation/evaluate_real_fusion.py), since ColorSignResNet18 is the
    stronger branch and AASL is the fused space's ground truth there too.
    """

    def __init__(self, grayscale_checkpoint_path: str, color_checkpoint_path: str):
        if ColorSignResNet18 is None:
            raise ImportError(
                "src/color/color_sign_resnet18.py is not present in this checkout -- "
                "FusionPredictor cannot be constructed. Add that file, or catch this "
                "and fall back to LetterPredictor (grayscale-only)."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gray_ckpt = torch.load(grayscale_checkpoint_path, map_location=self.device)
        color_ckpt = torch.load(color_checkpoint_path, map_location=self.device)

        self.grayscale_classes = gray_ckpt["classes"]
        self.gray_input_size = gray_ckpt["input_size"]
        self.aasl_classes = color_ckpt["classes"]
        self.color_input_size = color_ckpt["input_size"]

        grayscale_model = SimpleCNN(num_classes=len(self.grayscale_classes), input_size=self.gray_input_size)
        grayscale_model.load_state_dict(gray_ckpt["model_state_dict"])

        color_model = ColorSignResNet18(num_classes=len(self.aasl_classes), pretrained=False)
        color_model.load_state_dict(color_ckpt["model_state_dict"])

        gray_map, color_map = build_class_mappings(self.aasl_classes, self.grayscale_classes)

        self.fusion = ModalityFusionModel(
            grayscale_model, color_model, num_classes=len(self.aasl_classes),
            fusion_mode="average",  # the mode actually evaluated end-to-end, Section 4.7
            freeze_backbones=True,
            grayscale_class_to_fused_idx=gray_map,
            color_class_to_fused_idx=color_map,
        ).to(self.device)
        self.fusion.eval()

        # ImageNet normalization, matching ColorSignResNet18's training
        # transform exactly (train_color_resnet18.py / evaluate_real_fusion.py)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.color_transform = transforms.Compose([
            transforms.Resize((self.color_input_size, self.color_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, pil_frame: Image.Image):
        color_tensor = self.color_transform(pil_frame.convert("RGB")).unsqueeze(0).to(self.device)

        # Derive the grayscale branch's input from the SAME ROI-cropped
        # frame (Figure 3, step 2: "Preprocess: resize + derive
        # grayscale-3ch copy"), not from a second capture, so both branches
        # see identical content -- exactly as evaluated in
        # evaluation/evaluate_real_fusion.py.
        rgb01 = (color_tensor * self._std + self._mean).clamp(0, 1)
        gray_input = rgb_to_grayscale_3ch(rgb01)
        if gray_input.shape[-1] != self.gray_input_size:
            gray_input = torch.nn.functional.interpolate(
                gray_input, size=(self.gray_input_size, self.gray_input_size),
                mode="bilinear", align_corners=False,
            )

        with torch.no_grad():
            fused_logp = self.fusion(color_tensor, grayscale_input=gray_input)
            probs = torch.softmax(fused_logp, dim=1)  # recovers the fused probabilities (average mode returns log-probs)
            conf, idx = probs.max(dim=1)

        folder_name = self.aasl_classes[idx.item()]
        arabic_letter = ARASL_TO_ARABIC.get(aasl_folder_to_arasl_key(folder_name), "?")
        return arabic_letter, folder_name, conf.item()


# --------------------------------------------------------------------- app

class PracticeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Assistant — Practice Session")
        # الطول زاد بشكل كبير (720 -> 1200) بسبب إضافة شبكة الحروف المرجعية
        # الكاملة (400px تقريبًا) تحت شاشة الكاميرا (480px) -- بدون هذا
        # التكبير كانت الشبكة (وأحيانًا صف الأزرار السفلي) تُقتَطع بصمت خارج
        # حدود النافذة (تخطيط .pack() لا يُظهر أي تنبيه أو شريط تمرير عند
        # الفيضان، ببساطة يخفي الزائد). لو شاشتك أصغر من 1200px ارتفاعًا،
        # كبّر النافذة (Maximize) للحصول على أفضل عرض.
        self.geometry("1280x1200")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ---- state ----
        self.scenarios = json.load(open(SCENARIOS_PATH, encoding="utf-8"))["scenarios"]
        self.scenario_idx = 0
        self.word_buffer = []  # الحروف المؤكَّدة فقط (بترتيبها) -- يستثني الدورات الفارغة
        self.camera_running = False
        self.cap = None
        self._last_letter_confirmed_at = 0.0  # لتلوين إطار الدليل بعد تأكيد حرف
        self._last_cycle_empty_at = 0.0  # لتلوين إطار الدليل برتقالي بعد دورة بلا اكتشاف
        self._camera_candidates = []
        self._current_camera_index = None
        # آخر إطار/قصّة/تنبؤ فعليين من الكاميرا الحية — يقرأهما فقط زر التوثيق
        # (capture_documentation_screenshot)، ولا شيء يُنشئهما بشكل مصطنع.
        self._last_display_frame_rgb = None
        self._last_hand_crop_rgb = None
        self._last_prediction = None
        # وقت تأكيد أول حرف في الكلمة الحالية (wall-clock) — يبدأ منه عدّاد
        # مهلة الـ10 ثوانٍ الاحتياطية (WORD_RECOGNITION_TIMEOUT_SEC). None
        # يعني لا توجد كلمة قيد التهجئة حاليًا.
        self._word_start_time = None
        # --- دورة التقاط ثابتة (CAPTURE_INTERVAL_SEC) ---
        # captured_slots: سجل كامل لكل دورة انتهت لهذي الكلمة، سواء انتهت
        # بحرف مؤكَّد أو بلا شيء -- يُستخدم فقط لحساب نسبة تغطية الاكتشاف
        # لخوارزمية التكهن (guess_word_with_context)، وليس للعرض المباشر.
        # كل عنصر: dict بمفاتيح letter (str أو None) و confidence (float).
        self.captured_slots = []
        self._capture_window_start = None  # وقت بداية الدورة الحالية (wall-clock)
        self._capture_window_preds = []    # تنبؤات الإطارات المتجمّعة بالدورة الحالية
        self._cycle_count = 0  # رقم الدورة الحالية بهذي الكلمة -- يُعرض بسجل الحوار كل دورة
        # لحظة نهاية الكلمة الصارمة: بعد WORD_RECOGNITION_TIMEOUT_SEC ثانية
        # بالضبط من بداية الكلمة (لا من أول حرف مؤكَّد)، يتوقف الاكتشاف
        # تمامًا، ويلعب مربع الدليل حركة انكماش نحو المنتصف (راجع
        # _start_end_of_word_animation)، ثم تُنطق الكلمة. لا دورات التقاط
        # جديدة ولا تصنيف يحصل أثناء الحركة.
        self._animating_end = False
        # نص سؤال المحاور الفعلي المُتعرَّف عليه صوتيًا (الوضع الحر فقط) —
        # None يعني لم يُستمع لأي سؤال بعد في هذه المحادثة، وفي هذه الحالة
        # guess_word_with_context() تعمل بدون سياق بدل استخدام نص وهمي.
        self._current_question = None
        self.predictor = None
        self.model_used_label = None
        self.hand_tracker = None
        self.classification_log = ClassificationLog(os.path.join(REPO_ROOT, "classification_log.db"))
        self.tts_queue = queue.Queue()
        threading.Thread(target=self._tts_worker, daemon=True).start()

        self._build_layout()
        self._load_hand_sign_images()
        self._build_full_reference_grid()
        self._load_model_async()
        self._load_hand_tracker()
        self._render_scenario()

    # ---------------------------------------------------------------- UI

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ---- left: camera + controls ----
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # ---- hint strip: sequential hand-sign images guiding how to sign the target word ----
        hint_header = ctk.CTkFrame(left, fg_color="transparent")
        hint_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(hint_header, text=ar("دليل الإشارة المتسلسل:"), font=("Tahoma", 14, "bold")).pack(side="right")
        self.hint_toggle_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            hint_header, text=ar("إظهار"), variable=self.hint_toggle_var,
            command=self._render_hint_strip, width=20,
        ).pack(side="right", padx=10)
        # الوضع الحر: يستبدل السيناريوهات الجاهزة وشريط الدليل بمحادثة حقيقية
        # مفتوحة — المحاور يتحدث فعليًا عبر الميكروفون (راجع listen_for_question)،
        # والمُشير يعبّر عمّا يريده هو، لا عن إجابة معروفة مسبقًا للنظام.
        self.free_mode_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            hint_header, text=ar("🎙️ الوضع الحر (تعرّف صوتي حقيقي)"), variable=self.free_mode_var,
            command=self._on_free_mode_toggle, width=20,
        ).pack(side="right", padx=10)

        self.hint_strip = ctk.CTkFrame(left, fg_color="#101010", height=140)
        self.hint_strip.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 5))

        # حاوية تجمع شاشة الكاميرا (بالأعلى، حجم ثابت) والشبكة المرجعية
        # الكاملة لكل الحروف الـ32 (بالأسفل، دائمة الظهور) بنفس الخلية
        # المرنة بالتخطيط -- راجع _build_full_reference_grid.
        camera_container = ctk.CTkFrame(left, fg_color="#1a1a1a")
        camera_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        camera_container.grid_columnconfigure(0, weight=1)

        self.camera_label = tk.Label(camera_container, bg="#1a1a1a")
        self.camera_label.pack(pady=(6, 4))
        placeholder = Image.new("RGB", (640, 480), "#1a1a1a")
        self._placeholder_img = ImageTk.PhotoImage(placeholder)
        self.camera_label.configure(image=self._placeholder_img)

        ctk.CTkLabel(
            camera_container, text=ar("مرجع كل الحروف (32):"),
            font=("Tahoma", 11, "bold"), text_color="#888888",
        ).pack(pady=(4, 0))
        # أفقيًا-قابلة للتمرير: الصور الآن أكبر 3 أضعاف (راجع
        # _build_full_reference_grid)، فلا تتسع كل الـ16 عمود بعرض اللوحة
        # دفعة وحدة -- المستخدم يمرّر يمينًا/يسارًا ليشوف الباقي، مع بقاء
        # سطرين فقط كما طُلب.
        #
        # ⚠️ عيب حقيقي مكتشَف ومُصحَّح: CTkScrollableFrame(orientation=
        # "horizontal") بهذا الإصدار من customtkinter يُبلغ xview()=1.0
        # (يزعم إنه ممرَّر لأقصى اليمين) لكن يعرض فعليًا محتوى أقصى اليسار
        # فقط -- تناقض داخلي بين الحالة المُبلَّغة والعرض الفعلي، تحقّقتُ
        # منه بفحص الإحداثيات الفعلية على الشاشة مباشرة (winfo_rootx)، لا
        # بالاعتماد على واجهته البرمجية وحدها. الحل: Canvas + Scrollbar
        # قياسيان من tkinter نفسه، أبسط وأكثر قابلية للتنبؤ.
        grid_outer = ctk.CTkFrame(camera_container, fg_color="#101010")
        grid_outer.pack(fill="x", padx=6, pady=(2, 6))
        # عرض ابتدائي صريح (700) ضروري: بدونه الـCanvas يبقى بعرضه
        # الافتراضي شبه المعدوم (1px فعليًا) لحظة إنشائه هنا (قبل رسم
        # النافذة كاملة لأول مرة)، و.pack(fill="x") لا يُصحّح هذا فورًا --
        # هذا هو السبب الحقيقي وراء ظهور اللوحة فارغة تمامًا، تحقّقتُ منه
        # مباشرة (canvas.winfo_width() == 1 وقت البناء). fill="x" لاحقًا
        # يسمح للـCanvas يكبر لو النافذة اتوسّعت، لكن ما يُصلح البداية.
        self._reference_canvas = tk.Canvas(
            grid_outer, bg="#101010", height=380, width=700, highlightthickness=0,
        )
        h_scroll = tk.Scrollbar(grid_outer, orient="horizontal", command=self._reference_canvas.xview)
        self._reference_canvas.configure(xscrollcommand=h_scroll.set)
        self._reference_canvas.pack(side="top", fill="x")
        h_scroll.pack(side="top", fill="x")
        # الإطار الفعلي اللي تُبنى بداخله الصفوف (راجع _build_full_reference_grid)
        # مُضمَّن داخل الـCanvas عبر نافذة داخلية، لا بالتعبئة المباشرة.
        self.full_reference_frame = tk.Frame(self._reference_canvas, bg="#101010")
        self._reference_canvas.create_window((0, 0), window=self.full_reference_frame, anchor="nw")

        # تمرير أفقي بعجلة الماوس مباشرة (بدون حاجة لسحب شريط التمرير
        # يدويًا، اللي قد يكون غير واضح للمستخدم) -- المستخدم يمرّر
        # الماوس فوق الشبكة، فتتحرك يمينًا/يسارًا. مربوطة بالـCanvas وبكل
        # عناصره الفرعية (الصور والتسميات) عشان تشتغل بغض النظر أين بالضبط
        # يكون مؤشر الماوس داخل منطقة الشبكة.
        def _on_reference_wheel(event):
            delta = -1 if event.delta > 0 else 1  # اتجاه العجلة يختلف بين الأنظمة
            self._reference_canvas.xview_scroll(delta * 2, "units")
        self._reference_wheel_handler = _on_reference_wheel

        def _bind_wheel_recursive(widget):
            widget.bind("<MouseWheel>", _on_reference_wheel)   # ويندوز/ماك
            widget.bind("<Shift-MouseWheel>", _on_reference_wheel)
            widget.bind("<Button-4>", lambda e: self._reference_canvas.xview_scroll(-2, "units"))  # لينكس
            widget.bind("<Button-5>", lambda e: self._reference_canvas.xview_scroll(2, "units"))
            for child in widget.winfo_children():
                _bind_wheel_recursive(child)
        self._bind_reference_wheel_recursive = _bind_wheel_recursive
        _bind_wheel_recursive(self._reference_canvas)

        status_frame = ctk.CTkFrame(left, fg_color="transparent")
        status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.roi_hint_label = ctk.CTkLabel(
            status_frame, text=ar(
                "أخضر: استمر | سماوي: الحرف تسجّل | رمادي: لا يد مكتشفة | ✓ فوق الصورة: تأكد نهائيًا"
            ),
            font=("Tahoma", 12), text_color="#7fbf7f",
        )
        self.roi_hint_label.pack(side="left", padx=10)
        self.letter_status = ctk.CTkLabel(status_frame, text=ar("الحرف الحالي: —"), font=("Tahoma", 16, "bold"))
        self.letter_status.pack(side="left", padx=10)
        self.word_status = ctk.CTkLabel(status_frame, text=ar("الكلمة الحالية: —"), font=("Tahoma", 16, "bold"))
        self.word_status.pack(side="left", padx=10)
        # عدّاد تنازلي مرئي لموعد إغلاق دورة الالتقاط الحالية (كل
        # CAPTURE_INTERVAL_SEC ثانية) -- يحدَّث كل إطار كاميرا بغضّ النظر عن
        # اكتشاف يد أو لا، ليعرف المستخدم دائمًا متى ستُغلق الدورة القادمة.
        self.cycle_countdown_label = ctk.CTkLabel(
            status_frame, text="", font=("Tahoma", 13), text_color="#e0a030",
        )
        self.cycle_countdown_label.pack(side="left", padx=10)

        # صف اقتراحات الكلمات الحية (autocomplete): يظهر فقط أثناء وجود
        # بادئة حروف حالية غير فاضية، ويعرض أزرار الكلمات المرشّحة التي
        # تبدأ بنفس الحروف المهجّاة لحد الآن — يتيح للشخص اختيار الكلمة
        # مباشرة بدل إكمال تهجئتها حرفًا حرفًا بيده لو تعرّف عليها بدري.
        self.predictions_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.predictions_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.predictions_label = ctk.CTkLabel(
            self.predictions_frame, text="", font=("Tahoma", 13), text_color="#999999"
        )
        self.predictions_label.pack(side="left", padx=(10, 5))
        self._prediction_buttons = []

        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.camera_btn = ctk.CTkButton(controls, text=ar("ابدأ الكاميرا"), command=self.toggle_camera)
        self.camera_btn.pack(side="left", padx=5)
        self.next_camera_btn = ctk.CTkButton(
            controls, text=ar("كاميرا أخرى"), command=self.cycle_camera_source,
            fg_color="#444444", hover_color="#5a5a5a",
        )
        self.next_camera_btn.pack(side="left", padx=5)
        self.camera_source_label = ctk.CTkLabel(
            controls, text="", font=("Tahoma", 11), text_color="#999999"
        )
        self.camera_source_label.pack(side="left", padx=10)
        ctk.CTkButton(controls, text=ar("إعادة تعيين الكلمة"), command=self.reset_word).pack(side="left", padx=5)
        ctk.CTkButton(
            controls, text=ar("أنهِ الكلمة الآن"), command=self.finish_word_now,
            fg_color="#1F8B4C", hover_color="#27a85c",
        ).pack(side="left", padx=5)

        # صف ثانٍ منفصل (بدل إضافة كل الأزرار في صف واحد قد يفيض عن عرض
        # النافذة ويُخفي آخر الأزرار بصمت بلا أي رسالة خطأ) — يضمن ظهور زر
        # التوثيق دائمًا بغض النظر عن حجم النافذة أو دقة الشاشة.
        controls_row2 = ctk.CTkFrame(left, fg_color="transparent")
        controls_row2.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))
        # أزرار وضع التمرين (سيناريوهات جاهزة) — تُخفى في الوضع الحر لأنها
        # لا معنى لها بدون إجابة معروفة مسبقًا للنظام (راجع _on_free_mode_toggle).
        self.prev_scenario_btn = ctk.CTkButton(controls_row2, text=ar("السيناريو السابق"), command=self.prev_scenario)
        self.prev_scenario_btn.pack(side="left", padx=5)
        self.next_scenario_btn = ctk.CTkButton(controls_row2, text=ar("السيناريو التالي"), command=self.next_scenario)
        self.next_scenario_btn.pack(side="left", padx=5)
        # زر الميكروفون — يظهر فقط في الوضع الحر (راجع _on_free_mode_toggle).
        # يسجّل STT_RECORD_SECONDS ثانية من الميكروفون الفعلي، ويرسلها لخدمة
        # Google Web Speech (يحتاج إنترنت)، ويستخدم النص المُتعرَّف عليه فعليًا
        # كسؤال المحاور وكسياق تصحيح الإملاء — راجع listen_for_question أدناه.
        self.mic_btn = ctk.CTkButton(
            controls_row2, text=ar("🎤 استمع لسؤال المحاور"), command=self.listen_for_question,
            fg_color="#c0392b", hover_color="#e74c3c",
        )
        # زر توثيق: يحفظ لقطة حقيقية من الجلسة الحية (الإطار الكامل مع مربع
        # الدليل، القصّة الفعلية المرسلة للنموذج، والتنبؤ/الثقة الحاليين)
        # لاستخدامها كصور فعلية في قسم الأسلوب بالورقة البحثية، بدل أي
        # رسم توضيحي مصطنع. راجع capture_documentation_screenshot أدناه.
        ctk.CTkButton(
            controls_row2, text=ar("📸 التقط لقطة للتوثيق"), command=self.capture_documentation_screenshot,
            fg_color="#6a4fb3", hover_color="#7f5fd6",
        ).pack(side="left", padx=5)

        # ---- right: chat-style conversation ----
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.scenario_counter = ctk.CTkLabel(right, text="", font=("Tahoma", 14, "bold"))
        self.scenario_counter.grid(row=0, column=0, sticky="ew", pady=(10, 0))

        self.chat_frame = ctk.CTkScrollableFrame(right, label_text=ar("سجل الحوار"))
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)

    # -------------------------------------------------------- hand-sign hints

    def _load_hand_sign_images(self):
        self.hand_sign_photos = {}
        for fname in os.listdir(HAND_SIGNS_DIR):
            if fname.endswith(".png"):
                key = fname[:-4]
                img = Image.open(os.path.join(HAND_SIGNS_DIR, fname)).resize((72, 104))
                self.hand_sign_photos[key] = ImageTk.PhotoImage(img)

    def _build_full_reference_grid(self):
        """يبني شبكة مرجعية ثابتة ودائمة الظهور (سطران × 16 عمودًا = 32
        حرفًا) لكل أشكال اليد الحقيقية بأسفل شاشة الكاميرا مباشرة -- على
        عكس شريط الدليل بالأعلى (يقتصر على حروف الكلمة الحالية فقط، ويُخفى
        بالوضع الحر)، هذي الشبكة تعرض الأبجدية كاملة طوال الوقت، فيقدر
        المستخدم يتذكر شكل يد أي حرف بأي لحظة أثناء التدرّب أو الاستخدام
        الحر، لا حروف الكلمة الحالية بس. الصور مكبَّرة 108×156 (3× الحجم
        الأصلي 36×52)، فاللوحة (self.full_reference_frame) أفقيًا-قابلة
        للتمرير عشان الـ16 عمود ما يتسعوا دفعة وحدة بعرض معقول للنافذة."""
        keys = list(ARASL_TO_ARABIC.keys())  # 32 مفتاح، بنفس ترتيب class_mapping.py
        self._reference_photos = {}  # يحفظ المرجع؛ Tkinter يمسح الصور بلا مرجع فعلي إليها
        for key in keys:
            path = os.path.join(HAND_SIGNS_DIR, f"{key}.png")
            if os.path.exists(path):
                img = Image.open(path).resize((108, 156))  # 3× الحجم الأصلي (36×52) كما طُلب
                self._reference_photos[key] = ImageTk.PhotoImage(img)

        for row_keys in (keys[:16], keys[16:]):
            row_frame = ctk.CTkFrame(self.full_reference_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            for key in row_keys:
                cell = ctk.CTkFrame(row_frame, fg_color="transparent")
                cell.pack(side="right", padx=3)
                photo = self._reference_photos.get(key)
                if photo is not None:
                    tk.Label(cell, image=photo, bg="#101010").pack()
                ctk.CTkLabel(
                    cell, text=ar(ARASL_TO_ARABIC.get(key, "?")), font=("Tahoma", 13), text_color="#aaaaaa",
                ).pack()

        # يُحدَّث نطاق التمرير بعد بناء كل الصفوف، ثم يُمرَّر افتراضيًا لأقصى
        # اليمين (بما إن الخلايا معبّأة بـ side="right" لمحاكاة اتجاه القراءة
        # العربي -- الحرف الأول "ع" يظهر يمينًا فور فتح التطبيق بلا أي تمرير
        # يدوي مطلوب أول مرة؛ بقية الحروف تظهر بالتمرير يسارًا).
        self.full_reference_frame.update_idletasks()
        self._reference_canvas.configure(scrollregion=self._reference_canvas.bbox("all"))
        self._reference_canvas.xview_moveto(1.0)
        # يُعاد الربط الآن بعد بناء كل الخلايا (كانت فارغة وقت الربط الأول
        # بـ_build_layout) عشان التمرير بعجلة الماوس يشتغل حتى لو المؤشر
        # فوق صورة أو تسمية بالذات، لا فوق خلفية الـCanvas الفارغة فقط.
        self._bind_reference_wheel_recursive(self.full_reference_frame)

    def _render_hint_strip(self):
        for widget in self.hint_strip.winfo_children():
            widget.destroy()
        self._hint_check_labels = []  # يُعاد بناؤها بالكامل مع كل استدعاء

        if not self.hint_toggle_var.get():
            return

        scenario = self.scenarios[self.scenario_idx]
        keys = word_to_sign_keys(scenario["expected_answer"])

        if not keys:
            return

        for i, key in enumerate(keys):
            photo = self.hand_sign_photos.get(key)
            cell = ctk.CTkFrame(self.hint_strip, fg_color="transparent")
            cell.pack(side="left", padx=4, pady=6)
            if photo is not None:
                tk.Label(cell, image=photo, bg="#101010").pack()
            ctk.CTkLabel(cell, text=ar(ARASL_TO_ARABIC.get(key, "?")), font=("Tahoma", 13, "bold")).pack()
            # علامة صح خضراء دائمة تظهر بمجرد تأكيد هذا الحرف فعليًا (راجع
            # _update_hint_progress) — أثر بصري متراكم لا يختفي، على عكس
            # وميض لون مربع الدليل اللحظي الذي قد يفوت المستخدم ملاحظته.
            check_label = ctk.CTkLabel(cell, text="", font=("Tahoma", 16, "bold"), text_color="#00e676")
            check_label.pack()
            self._hint_check_labels.append(check_label)
            if i < len(keys) - 1:
                arrow = ctk.CTkLabel(self.hint_strip, text="→", font=("Tahoma", 18))
                arrow.pack(side="left", padx=2)

        self._update_hint_progress()

    def _update_hint_progress(self):
        """يعلّم كل صورة حرف في شريط الدليل بعلامة صح خضراء دائمة بمجرد
        تأكيد ذلك الحرف فعليًا (بناءً على عدد الحروف المؤكدة في
        word_buffer لحد الآن) — أثر بصري متراكم وواضح يبقى ظاهرًا طوال
        الكلمة، عكس وميض لون مربع الدليل اللحظي وحده الذي كان يمر بسرعة."""
        confirmed_count = len(self.word_buffer)
        for i, check_label in enumerate(getattr(self, "_hint_check_labels", [])):
            check_label.configure(text="✓" if i < confirmed_count else "")

    # ------------------------------------------------------------ model load

    def _load_hand_tracker(self):
        """Loads MediaPipe's HandLandmarker synchronously (fast -- it reads
        a small local .task file, no network access or heavy warm-up),
        unlike the ML checkpoints loaded asynchronously below. Leaves
        self.hand_tracker as None on any failure (mediapipe not installed,
        or HAND_LANDMARKER_MODEL_PATH not present), which _camera_loop
        checks to fall back to the fixed-center ROI box automatically."""
        try:
            self.hand_tracker = HandTracker(HAND_LANDMARKER_MODEL_PATH)
            print("[info] Real hand tracking active (MediaPipe HandLandmarker).")
        except (FileNotFoundError, ImportError) as e:
            print(f"[info] Hand tracking unavailable ({e}); using the fixed-center ROI box instead. "
                  f"See README 'Enabling real hand tracking' to add it.")
        except Exception as e:
            print(f"[warning] Hand tracking failed to load unexpectedly ({e}); "
                  f"using the fixed-center ROI box instead.")

    def _load_model_async(self):
        def load():
            try:
                self.predictor = FusionPredictor(CHECKPOINT_PATH, COLOR_CHECKPOINT_PATH)
                self.model_used_label = "fusion"
                self.letter_status.configure(text=ar("النموذج جاهز (رمادي + ملوّن مدمج) ✅"))
                return
            except (FileNotFoundError, ImportError) as e:
                # color_sign_resnet18_best.pt and/or src/color/color_sign_resnet18.py
                # not present in this checkout yet -- fall back to grayscale-only,
                # matching this app's earlier, still-fully-functional behavior.
                print(f"[info] Fusion pipeline unavailable ({e}); falling back to grayscale-only.")
            except Exception as e:
                print(f"[warning] Fusion pipeline failed to load unexpectedly ({e}); "
                      f"falling back to grayscale-only.")

            try:
                self.predictor = LetterPredictor(CHECKPOINT_PATH)
                self.model_used_label = "grayscale_only"
                self.letter_status.configure(text=ar("النموذج جاهز (رمادي فقط) ✅"))
            except FileNotFoundError:
                self.letter_status.configure(
                    text=ar(f"⚠️ لم يُعثر على النموذج المدرَّب في: {CHECKPOINT_PATH}")
                )
        threading.Thread(target=load, daemon=True).start()

    # ------------------------------------------------------------- scenarios

    def _render_scenario(self):
        if self.free_mode_var.get():
            return  # الوضع الحر له مسار عرض مختلف تمامًا — راجع _on_free_mode_toggle
        for widget in self.chat_frame.winfo_children():
            widget.destroy()

        scenario = self.scenarios[self.scenario_idx]
        self.scenario_counter.configure(
            text=ar(f"السيناريو {self.scenario_idx + 1} / {len(self.scenarios)}")
        )
        self._add_bubble(scenario["question"], sender="interlocutor")
        self._render_hint_strip()

    def _on_free_mode_toggle(self):
        """يبدّل بين وضع التمرين (سيناريوهات جاهزة + شريط دليل الحروف) و
        الوضع الحر (محادثة مفتوحة حقيقية: سؤال المحاور عبر ميكروفون فعلي،
        والمُشير يهجّئ ما يريده هو بلا أي إجابة معروفة مسبقًا للنظام). راجع
        القسم 3.7 من الورقة البحثية للنقاش الكامل حول الفرق بين الوضعين."""
        free = self.free_mode_var.get()
        self.reset_word()
        self._current_question = None

        if free:
            # إخفاء عناصر وضع التمرين التي لا معنى لها بدون إجابة معروفة مسبقًا
            self.hint_strip.grid_remove()
            self.prev_scenario_btn.pack_forget()
            self.next_scenario_btn.pack_forget()
            self.mic_btn.pack(side="left", padx=5)
            for widget in self.chat_frame.winfo_children():
                widget.destroy()
            self.scenario_counter.configure(text=ar("🎙️ الوضع الحر — اضغط زر الميكروفون لبدء محادثة جديدة"))
        else:
            # العودة لوضع التمرين: يعيد كل شيء بالضبط لحالته المُختبَرة والموثَّقة
            self.mic_btn.pack_forget()
            self.hint_strip.grid()
            self.prev_scenario_btn.pack(side="left", padx=5)
            self.next_scenario_btn.pack(side="left", padx=5)
            self._render_scenario()

    def listen_for_question(self):
        """يبدأ تسجيل سؤال المحاور فعليًا من الميكروفون الحقيقي (الوضع الحر
        فقط) — التسجيل والتعرّف الصوتي يعملان في thread منفصل حتى لا تتجمّد
        واجهة المستخدم أثناءهما، ونتائجهما تُطبَّق على الواجهة عبر self.after
        (القاعدة العامة في Tkinter: لا تُحدَّث عناصر الواجهة من أي thread
        غير الرئيسي مباشرة)."""
        if not self.free_mode_var.get():
            return
        self.mic_btn.configure(state="disabled", text=ar("🎙️ جارٍ الاستماع..."))
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _listen_worker(self):
        try:
            import sounddevice as sd
            import speech_recognition as sr
        except ImportError as e:
            self.after(0, lambda: self._on_stt_result(
                None, f"مكتبة ناقصة ({e}) — ثبّت: pip install SpeechRecognition sounddevice"
            ))
            return

        try:
            recording = sd.rec(
                int(STT_RECORD_SECONDS * STT_SAMPLE_RATE),
                samplerate=STT_SAMPLE_RATE, channels=1, dtype="int16",
            )
            sd.wait()
        except Exception as e:
            self.after(0, lambda: self._on_stt_result(None, f"تعذّر تسجيل الصوت (تحقق من الميكروفون): {e}"))
            return

        recognizer = sr.Recognizer()
        # sr.AudioData تُبنى مباشرة من بيانات PCM خام (int16 = عيّنتان بايت)
        # بدل استخدام sr.Microphone، لتفادي الاعتماد على PyAudio (صعبة
        # التثبيت على ويندوز خصوصًا مع إصدارات بايثون الحديثة) — sounddevice
        # يوفّر عجلات (wheels) جاهزة موثوقة بديلًا عنها.
        audio_data = sr.AudioData(recording.tobytes(), STT_SAMPLE_RATE, 2)
        try:
            text = recognizer.recognize_google(audio_data, language=STT_LANGUAGE)
            self.after(0, lambda: self._on_stt_result(text, None))
        except sr.UnknownValueError:
            self.after(0, lambda: self._on_stt_result(None, "لم يُلتقط كلام مفهوم — حاول مرة أخرى بصوت أوضح وأقرب للميكروفون"))
        except sr.RequestError as e:
            self.after(0, lambda: self._on_stt_result(None, f"تعذّر الاتصال بخدمة التعرّف الصوتي (تحقق من الإنترنت): {e}"))

    def _on_stt_result(self, text, error):
        """يعمل دائمًا على الـ thread الرئيسي (مُستدعى عبر self.after من
        _listen_worker)، فآمن هنا تحديث عناصر الواجهة مباشرة."""
        self.mic_btn.configure(state="normal", text=ar("🎤 استمع لسؤال المحاور"))
        if error:
            self.scenario_counter.configure(text=ar(f"⚠️ {error}"))
            return
        self._current_question = text
        self._add_bubble(text, sender="interlocutor")
        self.scenario_counter.configure(text=ar("🎙️ الوضع الحر — استمع لسؤال جديد في أي وقت"))

    def _add_bubble(self, text, sender="interlocutor", caption=None):
        # "interlocutor" bubble aligns right (they speak first, standard chat convention here),
        # "signer" (device-spoken answer) bubble aligns left, colored differently — mirroring
        # familiar messaging-app visual language (e.g. WhatsApp/Telegram) so the interface
        # feels immediately familiar rather than bespoke.
        is_interlocutor = sender == "interlocutor"
        bubble_color = "#2B5278" if is_interlocutor else "#1F8B4C"
        anchor_side = "e" if is_interlocutor else "w"

        row = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        row.pack(fill="x", pady=4)

        bubble = ctk.CTkLabel(
            row, text=ar(text), font=("Tahoma", 15), fg_color=bubble_color,
            corner_radius=12, padx=14, pady=8, justify="right", anchor="e",
        )
        bubble.pack(side="right" if is_interlocutor else "left", padx=10)

        if caption:
            cap_label = ctk.CTkLabel(row, text=ar(caption), font=("Tahoma", 11), text_color="#999999")
            cap_label.pack(side="right" if is_interlocutor else "left", padx=10)

    def _log_cycle_result(self, letter):
        """يضيف سطرًا صغيرًا لسجل الحوار (يمين) عند إغلاق كل دورة التقاط
        (كل CAPTURE_INTERVAL_SEC ثانية)، بالحرف المكتشف أو "0" صراحة لو
        الدورة انتهت بلا اكتشاف واثق -- إيقاع مرئي مستمر يعرف معه المستخدم
        بالضبط وش سُجِّل بكل دورة أثناء تهجئة الكلمة، لا بعد اكتمالها فقط."""
        self._cycle_count += 1
        text = letter if letter is not None else "0"
        row = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        row.pack(fill="x", pady=1)
        ctk.CTkLabel(
            row, text=ar(f"دورة {self._cycle_count}: {text}"),
            font=("Tahoma", 11), text_color="#777777",
        ).pack(side="right", padx=14)

    def next_scenario(self):
        if self.free_mode_var.get():
            return
        self.reset_word()
        self.scenario_idx = (self.scenario_idx + 1) % len(self.scenarios)
        self._render_scenario()

    def prev_scenario(self):
        if self.free_mode_var.get():
            return
        self.reset_word()
        self.scenario_idx = (self.scenario_idx - 1) % len(self.scenarios)
        self._render_scenario()

    def reset_word(self):
        self.word_buffer = []
        self.captured_slots = []
        self._capture_window_start = None  # يبدأ من جديد أول إطار كاميرا قادم
        self._capture_window_preds = []
        self._cycle_count = 0
        # يُعاد ضبطها هنا فقط — تُفعَّل من جديد لحظة بداية أول دورة التقاط
        # بالكلمة التالية (راجع _tick_capture_cycle)، بنفس الإعدادات.
        self._word_start_time = None
        self.word_status.configure(text=ar("الكلمة الحالية: —"))
        for btn in self._prediction_buttons:
            btn.destroy()
        self._prediction_buttons = []
        self.predictions_label.configure(text="")
        self._update_hint_progress()  # يمسح كل علامات الصح مع بداية كلمة جديدة

    # --------------------------------------------------------------- camera

    def toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.camera_btn.configure(text=ar("ابدأ الكاميرا"))
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_source_label.configure(text="")
        else:
            if CAMERA_INDEX_OVERRIDE is not None:
                # المستخدم حدد فهرسًا يدويًا عبر ARSL_CAMERA_INDEX. نجرّب نفس
                # الـ backends المستخدمة بالفحص التلقائي (وليس DSHOW فقط)، لأن
                # بعض الأجهزة (خصوصًا D435) تحتاج MSMF لتقرأ إطارات فعلية.
                idx = int(CAMERA_INDEX_OVERRIDE)
                cap, kind = None, "manual"
                for backend_name, backend in PROBE_BACKENDS:
                    trial_cap = cv2.VideoCapture(idx, backend)
                    if trial_cap.isOpened():
                        cap = trial_cap
                        break
                    trial_cap.release()
                if cap is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                else:
                    idx, kind = None, None
                self._camera_candidates = []
            else:
                # فحص تلقائي: يفضّل تيارًا ملوّنًا حقيقيًا، ويتجنب تيارات IR/Depth
                # الرمادية التي قد تخرجها D435 على فهارس أخرى، ويطبع تشخيصًا
                # كاملًا على الطرفية (terminal) لكل فهرس تمت تجربته.
                self._camera_candidates = scan_cameras()
                idx, cap, kind = find_working_camera_from(self._camera_candidates)

            if cap is None:
                self.letter_status.configure(
                    text=ar("⚠️ تعذّر العثور على كاميرا تعمل (تحقق من توصيل D435، أو راجع الطرفية لتفاصيل الفحص)")
                )
                return

            self.cap = cap
            self.camera_running = True
            self._current_camera_index = idx
            self.camera_btn.configure(text=ar("أوقف الكاميرا"))
            self._update_camera_source_label(idx, kind)
            self._camera_loop()

    def _update_camera_source_label(self, idx, kind):
        kind_ar = {
            "color": ar("تيار ملوّن"),
            "grayscale/IR-or-depth": ar("⚠️ تيار رمادي/IR — قد يكون خاطئًا"),
            "manual": ar("محدد يدويًا"),
        }.get(kind, kind or "")
        self.camera_source_label.configure(text=f"index {idx} · {kind_ar}")

    def cycle_camera_source(self):
        """يبدّل يدويًا إلى فهرس الكاميرا المرشّح التالي من آخر فحص — مفيد لو
        اختار الفحص التلقائي تيار IR/Depth بالخطأ بدل تيار اللون الحقيقي
        (كلاهما قد يمر فحص 'غير أسود' حسب الإضاءة)."""
        candidates = [c for c in getattr(self, "_camera_candidates", []) if c["kind"] != "black"]
        if not candidates:
            self.letter_status.configure(text=ar("⚠️ لا توجد كاميرات مرشّحة أخرى من آخر فحص"))
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        current_idx = getattr(self, "_current_camera_index", None)
        indices = [c["index"] for c in candidates]
        if current_idx in indices:
            next_pos = (indices.index(current_idx) + 1) % len(indices)
        else:
            next_pos = 0
        chosen = candidates[next_pos]
        cap = cv2.VideoCapture(chosen["index"], chosen["backend"])
        if not cap.isOpened():
            self.letter_status.configure(text=ar("⚠️ تعذّر فتح الكاميرا المرشّحة التالية"))
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap = cap
        self._current_camera_index = chosen["index"]
        self.camera_running = True
        self.camera_btn.configure(text=ar("أوقف الكاميرا"))
        self._update_camera_source_label(chosen["index"], chosen["kind"])
        self._camera_loop()

    def _camera_loop(self):
        if not self.camera_running:
            return
        if self._animating_end:
            # حركة نهاية الكلمة (انظر _start_end_of_word_animation) تدير
            # حلقة after() خاصة فيها؛ لا نقرأ إطارات جديدة ولا نصنّف أي شيء
            # حتى تنتهي، ثم هي نفسها تستأنف _camera_loop.
            return

        ok, frame_bgr = self.cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # يتبع اليد الفعلية عبر HandTracker (MediaPipe) إن كان متاحًا؛
            # وإلا -- أو لو لم تُكتشف يد بهذا الإطار بالذات -- يرجع لمربع
            # ثابت بمنتصف الإطار كخط دفاع أخير. راجع تعليق
            # HAND_LANDMARKER_MODEL_PATH أعلاه للتفاصيل الكاملة.
            box = self.hand_tracker.detect_box(frame_rgb) if self.hand_tracker is not None else None
            hand_missing = self.hand_tracker is not None and box is None

            if box is not None:
                x1, y1, x2, y2 = box
            else:
                box_side = int(min(h, w) * HAND_ROI_FRACTION)
                x1 = (w - box_side) // 2
                y1 = (h - box_side) // 2
                x2, y2 = x1 + box_side, y1 + box_side

            # المؤقّت الصارم: بمجرد وصول WORD_RECOGNITION_TIMEOUT_SEC ثانية
            # بالضبط منذ بداية اكتشاف هذي الكلمة (لا من أول حرف مؤكَّد --
            # راجع _tick_capture_cycle)، يتوقف الاكتشاف والتصنيف فورًا،
            # وتبدأ حركة الانكماش بدل الاستمرار بالتصنيف العادي لهذا الإطار.
            if (
                self._word_start_time is not None
                and (time.time() - self._word_start_time) >= WORD_RECOGNITION_TIMEOUT_SEC
            ):
                self._start_end_of_word_animation(frame_rgb, (x1, y1, x2, y2))
                return

            hand_crop_rgb = frame_rgb[y1:y2, x1:x2].copy()
            pil_hand = Image.fromarray(hand_crop_rgb)

            # يُستدعى دائمًا -- حتى لو hand_missing -- لأن دورة الالتقاط
            # الثابتة (CAPTURE_INTERVAL_SEC) يجب تستمر بالعدّ وتُغلق بموعدها
            # بصرف النظر عن اكتشاف يد أو لا (راجع _process_frame/_tick_capture_cycle).
            # التصنيف نفسه (استدعاء النموذج) يُتخطّى داخليًا لو hand_missing.
            self._process_frame(pil_hand, hand_missing)

            if hand_missing:
                box_color = ROI_BOX_COLOR_NO_HAND
            else:
                recently_confirmed = (
                    time.time() - self._last_letter_confirmed_at
                ) < CONFIRMED_HIGHLIGHT_DURATION_SEC
                recently_empty = (
                    time.time() - self._last_cycle_empty_at
                ) < CONFIRMED_HIGHLIGHT_DURATION_SEC
                if recently_confirmed:
                    box_color = ROI_BOX_COLOR_CONFIRMED
                elif recently_empty:
                    box_color = ROI_BOX_COLOR_CYCLE_EMPTY
                else:
                    box_color = ROI_BOX_COLOR_DEFAULT

            # النسخة المعروضة فقط تحمل مربع الدليل الملوّن — النسخة المقصوصة
            # المرسلة للتصنيف (hand_crop_rgb) نظيفة بدون أي خط مرسوم عليها.
            display_frame_rgb = frame_rgb.copy()
            cv2.rectangle(display_frame_rgb, (x1, y1), (x2, y2), box_color, 3)
            pil_display = Image.fromarray(display_frame_rgb)

            display_img = pil_display.resize((640, 480))
            tk_img = ImageTk.PhotoImage(display_img)
            self.camera_label.configure(image=tk_img, text="")
            self.camera_label.image = tk_img  # keep a reference, otherwise Tkinter garbage-collects it

            # نحتفظ بأحدث إطار كامل (مع مربع الدليل) وأحدث قصّة يد فعلية
            # (نظيفة بلا خط) لاستخدامهما لاحقًا في capture_documentation_screenshot
            # لو ضغط المستخدم زر التوثيق — بدون هذا نضطر لإعادة قراءة الكاميرا
            # وقتها، وقد تكون اللحظة قد فاتت (مثلاً بعد رفع اليد).
            self._last_display_frame_rgb = display_frame_rgb
            self._last_hand_crop_rgb = hand_crop_rgb

        self.after(80, self._camera_loop)  # ~12 fps, plenty for sign holding several frames

    def _start_end_of_word_animation(self, frame_rgb, start_box):
        """ينفَّذ بمجرد وصول العدّاد الصارم (WORD_RECOGNITION_TIMEOUT_SEC)
        لنهايته: يوقف أي تصنيف أو دورات التقاط جديدة فورًا، ويُشغّل حركة
        انكماش مرئية لمربع الدليل من مكانه ومقاسه الحاليين نحو نقطة صغيرة
        بمنتصف الإطار بالضبط -- إشارة بصرية واضحة لا لبس فيها إن الاكتشاف
        انتهى -- ثم يُنطق الكلمة المتراكمة (أو تُسقط بصمت لو ما تأكّد أي
        حرف إطلاقًا)، ثم يستأنف حلقة الكاميرا الطبيعية للكلمة التالية."""
        self._animating_end = True
        self.letter_status.configure(text=ar("⏹ انتهى وقت الاكتشاف (10 ث)"))

        h, w = frame_rgb.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        x1, y1, x2, y2 = start_box
        steps = 10
        target_half = 12  # نصف طول ضلع المربع الصغير جدًا بنهاية الحركة

        def step(i):
            if i > steps:
                self._animating_end = False
                if self.word_buffer:
                    self._finalize_word()
                else:
                    # لا حرف واحد اتأكَّد خلال الـ10 ثوانٍ كاملة -- إعلام
                    # صريح بدل إسقاط صامت، عشان المستخدم يعرف بوضوح إن
                    # الدورة انتهت (لا يظل يظن إن البرنامج ما زال ينتظر).
                    self._add_bubble(
                        "لم يتم التعرف", sender="signer",
                        caption="لا حروف مؤكَّدة خلال 10 ثوانٍ",
                    )
                    self._speak_async("لم يتم التعرف")
                    self.reset_word()
                self._camera_loop()  # يستأنف قراءة الكاميرا للكلمة التالية
                return
            t = i / steps
            nx1 = x1 + ((cx - target_half) - x1) * t
            ny1 = y1 + ((cy - target_half) - y1) * t
            nx2 = x2 + ((cx + target_half) - x2) * t
            ny2 = y2 + ((cy + target_half) - y2) * t
            disp = frame_rgb.copy()
            cv2.rectangle(disp, (int(nx1), int(ny1)), (int(nx2), int(ny2)), ROI_BOX_COLOR_CONFIRMED, 3)
            img = Image.fromarray(disp).resize((640, 480))
            tk_img = ImageTk.PhotoImage(img)
            self.camera_label.configure(image=tk_img)
            self.camera_label.image = tk_img
            self.after(35, lambda: step(i + 1))

        step(0)

    def _process_frame(self, pil_frame, hand_missing: bool):
        # المؤقّت الصارم فُحص بالفعل بـ_camera_loop قبل الوصول هنا (وحرّك
        # حركة الانكماش إن انتهى) -- هذي الدالة تفترض أن الوقت المتبقي لسا
        # كافٍ لمعالجة إطار عادي.
        #
        # التصنيف نفسه يُتخطّى فقط لو تتبع اليد مفعّل ولم يكتشف يدًا بهذا
        # الإطار بالذات (تصنيف خلفية فارغة ضوضاء بلا فائدة) — لكن دورة
        # الالتقاط الثابتة أدناه تستمر بالعدّ وتُغلَق بموعدها بصرف النظر،
        # تمامًا كما طُلب: "خلال ثانيتين سواء اكتشف البرنامج الحرف أم لا".
        if not hand_missing and self.predictor is not None:
            arabic_letter, folder_name, confidence = self.predictor.predict(pil_frame)
            accepted = confidence >= CONFIDENCE_THRESHOLD
            self.classification_log.log_attempt(
                model_used=self.model_used_label or "unknown", predicted_label=folder_name,
                confidence=confidence, accepted=accepted,
            )
            self._capture_window_preds.append((arabic_letter, confidence))
            self.letter_status.configure(
                text=ar(f"الحرف الحالي: {arabic_letter} (ثقة {confidence:.2f})")
            )
            # يُستخدم في تعليق الصورة عند التقاط لقطة توثيق (نفس القيم
            # المعروضة للمستخدم فعليًا في تلك اللحظة، لا قيم مُعاد حسابها).
            self._last_prediction = (arabic_letter, confidence, accepted)

        self._tick_capture_cycle()

    def _tick_capture_cycle(self):
        """يُستدعى كل إطار كاميرا (سواء اكتُشفت يد أم لا) — يبدأ دورة جديدة
        عند أول استدعاء (وهي نفس لحظة بداية العدّاد الصارم للكلمة كاملة،
        WORD_RECOGNITION_TIMEOUT_SEC)، ويُغلق الدورة الحالية ويوثّقها بمجرد
        مرور CAPTURE_INTERVAL_SEC ثانية بالضبط منذ بدايتها، بصرف النظر
        تمامًا عن وجود أي تنبؤات واثقة تجمّعت خلالها أو لا. هذا هو قلب
        "دورة التقاط ثابتة": المستخدم يعرف دائمًا، كل ثانيتين بالضبط، إما
        أن حرفًا اتسجّل (سماوي + سطر جديد بسجل الحوار) أو لا (برتقالي + "0"
        بسجل الحوار) — بدون أي انتظار غامض."""
        now = time.time()
        if self._capture_window_start is None:
            self._capture_window_start = now
            if self._word_start_time is None:
                # بداية اكتشاف هذي الكلمة بالكامل -- منها يبدأ العدّاد
                # الصارم أيضًا، لا من أول حرف يتأكَّد (راجع _camera_loop).
                self._word_start_time = now
            return

        elapsed = now - self._capture_window_start
        remaining = max(0.0, CAPTURE_INTERVAL_SEC - elapsed)
        self.cycle_countdown_label.configure(
            text=ar(f"⏱ الدورة التالية خلال: {remaining:.1f} ث")
        )
        if elapsed < CAPTURE_INTERVAL_SEC:
            return

        # --- إغلاق الدورة: تصويت أغلبية على تنبؤات هذي الدورة فقط ---
        confident = [(lbl, conf) for lbl, conf in self._capture_window_preds if conf >= CONFIDENCE_THRESHOLD]
        winner_label, winner_conf = None, 0.0
        if confident:
            counts = Counter(lbl for lbl, _ in confident)
            top_label, top_count = counts.most_common(1)[0]
            if top_count / len(confident) >= CAPTURE_STABILITY_RATIO:
                winner_label = top_label
                winner_conf = sum(c for l, c in confident if l == top_label) / top_count

        self.captured_slots.append({"letter": winner_label, "confidence": winner_conf})
        self._log_cycle_result(winner_label)  # يظهر بسجل الحوار فورًا: الحرف أو "0"

        if winner_label is not None:
            self.word_buffer.append(winner_label)
            self._last_letter_confirmed_at = time.time()  # يلوّن المربع سماويًا
            self._update_predictions()
            self._update_hint_progress()
        else:
            self._last_cycle_empty_at = time.time()  # يلوّن المربع برتقاليًا

        coverage = (
            sum(1 for s in self.captured_slots if s["letter"] is not None) / len(self.captured_slots)
            if self.captured_slots else 0.0
        )
        self.word_status.configure(
            text=ar(
                f"الكلمة الحالية: {''.join(self.word_buffer)} "
                f"(دورات: {len(self.captured_slots)}، تغطية: {coverage:.0%})"
            )
        )

        # دورة جديدة تبدأ فورًا من الصفر (العدّاد الصارم للكلمة لا يتأثر)
        self._capture_window_start = time.time()
        self._capture_window_preds = []

    def capture_documentation_screenshot(self):
        """يحفظ لقطة حقيقية واحدة من الجلسة الحية إلى screenshots/ لاستخدامها
        كصورة فعلية في قسم الأسلوب بالورقة البحثية (خطوات الأسلوب والشبكة
        العصبية في الاكتشاف). لا شيء هنا يُركَّب أو يُزيَّف: الصورة المحفوظة
        هي بالضبط ما كانت الكاميرا والنموذج ينتجانه في لحظة الضغط على الزر،
        مع تعليق نصي (caption) يذكر الحرف المتوقَّع والثقة الفعليين فقط إن
        كانا متوفرين لتلك اللقطة بالذات.
        """
        if self._last_display_frame_rgb is None:
            self.letter_status.configure(text=ar("⚠️ شغّل الكاميرا أولًا قبل التقاط لقطة توثيق"))
            return

        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        full_frame = Image.fromarray(self._last_display_frame_rgb)
        full_path = os.path.join(SCREENSHOTS_DIR, f"{timestamp}_full_frame.png")
        full_frame.save(full_path)

        crop_path = None
        if self._last_hand_crop_rgb is not None:
            hand_crop = Image.fromarray(self._last_hand_crop_rgb)
            crop_path = os.path.join(SCREENSHOTS_DIR, f"{timestamp}_hand_crop_model_input.png")
            hand_crop.save(crop_path)

        caption_lines = [f"captured_at: {timestamp}"]
        if self._last_prediction is not None:
            letter, confidence, accepted = self._last_prediction
            caption_lines.append(f"predicted_letter: {letter}")
            caption_lines.append(f"confidence: {confidence:.4f}")
            caption_lines.append(f"accepted (>= threshold): {accepted}")
        caption_lines.append(f"full_frame_file: {os.path.basename(full_path)}")
        if crop_path:
            caption_lines.append(f"model_input_crop_file: {os.path.basename(crop_path)}")
        caption_path = os.path.join(SCREENSHOTS_DIR, f"{timestamp}_caption.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write("\n".join(caption_lines) + "\n")

        self.letter_status.configure(text=ar(f"✅ حُفظت لقطة التوثيق في screenshots/{timestamp}_*"))

    def _update_predictions(self):
        """يحدّث صف الاقتراحات الحية بناءً على بادئة الحروف المهجّاة لحد
        الآن (self.word_buffer) — نداء خفيف (بحث بادئة على قاموس صغير)
        فيُستدعى بأمان بعد كل حرف جديد بدل انتظار نهاية الكلمة."""
        for btn in self._prediction_buttons:
            btn.destroy()
        self._prediction_buttons = []

        prefix = "".join(self.word_buffer)
        predictions = predict_words_from_prefix(prefix, max_results=MAX_WORD_PREDICTIONS)
        if not predictions:
            self.predictions_label.configure(text="")
            return

        self.predictions_label.configure(text=ar("اقتراحات:"))
        for word in predictions:
            btn = ctk.CTkButton(
                self.predictions_frame, text=ar(word), width=70,
                fg_color="#2f6f4f", hover_color="#3f8f66",
                command=lambda w=word: self.accept_predicted_word(w),
            )
            btn.pack(side="left", padx=3)
            self._prediction_buttons.append(btn)

    def accept_predicted_word(self, word: str):
        """يقبل اقتراحًا مباشرة بدل إكمال باقي حروف الكلمة يدويًا — يُغلق
        الكلمة الحالية فورًا بنفس مسار _finalize_word تقريبًا، لكن بدون
        تصحيح إملائي (الكلمة أصلاً من القاموس، لا حاجة لمسافة تحرير)."""
        raw_word = "".join(self.word_buffer)
        caption = f"من: {raw_word} | اقتراح مقبول قبل اكتمال التهجئة"
        self._add_bubble(word, sender="signer", caption=caption)
        self._speak_async(word)
        self.reset_word()

    def finish_word_now(self):
        """يُنهي الكلمة الحالية فورًا بضغطة يدوية، دون انتظار المهلة
        الصارمة (WORD_RECOGNITION_TIMEOUT_SEC) — يعطي المستخدم تحكمًا
        مباشرًا ومضمونًا لإنهاء كلمة قصيرة مبكرًا لو رغب بذلك."""
        if self._animating_end:
            return  # حركة نهاية الكلمة شغّالة بالفعل من المؤقّت الصارم
        if self.word_buffer:
            self._finalize_word()

    def _finalize_word(self):
        raw_word = "".join(self.word_buffer)
        if self.free_mode_var.get():
            # في الوضع الحر لا يوجد نص سؤال جاهز أصلًا — نستخدم آخر سؤال
            # حقيقي تم الاستماع إليه فعليًا عبر الميكروفون، أو None لو لم
            # يُستمع لأي سؤال بعد (guess_word_with_context تتعامل مع None
            # بأمان، بدون أي محاولة لتلفيق سياق وهمي).
            question = self._current_question
        else:
            question = self.scenarios[self.scenario_idx]["question"]

        # خوارزمية التكهن: نسبة تغطية الاكتشاف = كم من دورات الالتقاط
        # الثابتة (captured_slots، بما فيها الدورات الفارغة) انتهت فعلًا
        # بحرف مؤكَّد -- هذي النسبة هي اللي تُحدِّد كم نثق بالحروف المكتشفة
        # مقابل سياق المحادثة عند اختيار الكلمة النهائية (راجع التوثيق
        # الكامل بدالة guess_word_with_context في spelling_correction.py).
        coverage = (
            sum(1 for s in self.captured_slots if s["letter"] is not None) / len(self.captured_slots)
            if self.captured_slots else 0.0
        )
        result = guess_word_with_context(
            raw_word, letter_confidence_coverage=coverage, stt_context_text=question,
        )

        caption = (
            f"من: {raw_word} | تغطية الاكتشاف: {coverage:.0%} | "
            f"ثقة التخمين: {result.confidence_percent:.0f}% "
            f"(حروف {result.letter_weight:.0%} + سياق {result.context_weight:.0%})"
        )
        if result.used_context:
            caption += " | استُخدم السياق"
        self._add_bubble(result.guessed_word, sender="signer", caption=caption)

        self._speak_async(result.guessed_word)
        self.reset_word()

    def _speak_async(self, text):
        """يضيف النص لطابور النطق بدل فتح thread وتهيئة محرك pyttsx3 من
        الصفر في كل مرة — راجع _tts_worker للسبب (تهيئة COM/SAPI5 المتكررة
        من threads مختلفة بالتتابع السريع كانت تفشل بصمت أحيانًا على ويندوز،
        فتُصنَّف الكلمة وتُعرض صح لكن بدون أي صوت فعلي)."""
        self.tts_queue.put(text)

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        return any("\u0600" <= ch <= "\u06FF" for ch in text)

    def _speak_via_gtts(self, text: str) -> bool:
        """احتياطي لنطق العربي عبر خدمة Google TTS السحابية (يحتاج اتصال
        إنترنت لكل جملة). اعتمدنا هذا بدل محاولة الوصول لأصوات ويندوز
        OneCore محليًا (عبر PowerShell/WinRT) بعد أن واجهت تلك الطريقة
        عائقين منفصلين: (1) واجهة WinRT لا "ترى" الصوت العربي المثبّت إطلاقًا
        عند استدعائها من عملية بدون هوية تطبيق (identity) رسمية مثل
        PowerShell عادي، و(2) تلف ترميز النص العربي نفسه عند تمريره كوسيط
        سطر أوامر لـ PowerShell. gTTS أبسط بكثير: لا يعتمد على أي صوت مثبّت
        محليًا إطلاقًا، ولا على تمرير نص عبر سطر أوامر (يُمرَّر كوسيط دالة
        بايثون مباشرة، فلا خطر تلف ترميز). التشغيل نفسه عبر واجهة MCI
        المدمجة في ويندوز (ctypes فقط، بدون أي مشغل صوت خارجي إضافي).
        يرجع True لو نجح النطق فعليًا."""
        try:
            from gtts import gTTS
        except ImportError:
            print("[tts] gTTS not installed — run: pip install gTTS")
            return False

        mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(mp3_fd)
        try:
            gTTS(text=text, lang="ar").save(mp3_path)
        except Exception as e:
            print(f"[tts] gTTS synthesis failed (check internet connection): {e}")
            os.remove(mp3_path)
            return False

        try:
            import ctypes
            alias = "arsl_tts_clip"
            winmm = ctypes.windll.winmm
            winmm.mciSendStringW(f'open "{mp3_path}" type mpegvideo alias {alias}', None, 0, None)
            winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
            winmm.mciSendStringW(f"close {alias}", None, 0, None)
            return True
        except Exception as e:
            print(f"[tts] gTTS playback failed: {e}")
            return False
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass

    def _tts_worker(self):
        """Thread واحد دائم يعيش طوال عمر التطبيق: يهيّئ محرك pyttsx3 مرة
        وحدة فقط، ثم يستهلك طابور النطق تباعًا. هذا يتجنب إعادة تهيئة
        COM/SAPI5 لكل كلمة (المصدر الأرجح للفشل الصامت السابق)، ويقلل زمن
        الاستجابة كمان (لا حاجة لتهيئة محرك جديد بكل نطق)."""
        import pyttsx3
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass  # غير ويندوز — لا حاجة لتهيئة COM أصلاً

        try:
            engine = pyttsx3.init()
        except Exception as e:
            print(f"[tts] failed to initialize engine: {e}")
            return

        # نفضّل صراحة أي صوت عربي كلاسيكي (SAPI5) مثبّت على النظام، إن وُجد.
        # في الغالب لن يوجد (أصوات ويندوز العربية الحديثة، مثل "Naayf"،
        # تُسجَّل كأصوات OneCore لا تظهر هنا إطلاقًا — راجع
        # _speak_via_gtts للتفاصيل والحل البديل)، وهذا متوقع
        # وليس خطأ يستدعي القلق.
        arabic_voice = None
        try:
            voices = engine.getProperty("voices")
            arabic_voice = next(
                (v for v in voices if "arabic" in (v.name or "").lower()
                 or "ar" in [lang.lower() for lang in getattr(v, "languages", [])]
                 or "_ar-" in (v.id or "").lower()),
                None,
            )
            if arabic_voice is not None:
                engine.setProperty("voice", arabic_voice.id)
                print(f"[tts] using classic SAPI5 Arabic voice: {arabic_voice.name}")
            else:
                print("[tts] no classic SAPI5 Arabic voice found — Arabic text will use "
                      "the Windows OneCore voice (e.g. Naayf) via PowerShell instead.")
        except Exception as e:
            print(f"[tts] voice selection failed, using system default: {e}")

        while True:
            text = self.tts_queue.get()
            if text is None:  # إشارة إيقاف عند إغلاق التطبيق
                break
            try:
                if self._contains_arabic(text) and arabic_voice is None:
                    # لا يوجد صوت عربي كلاسيكي على هذا الجهاز — نستخدم
                    # gTTS (خدمة سحابية، تحتاج إنترنت) بدل pyttsx3 لهذا النص.
                    if not self._speak_via_gtts(text):
                        # كحل أخير لو فشل الاحتياطي نفسه لأي سبب (لا إنترنت
                        # مثلًا): نجرب pyttsx3 بالصوت الافتراضي بدل السكوت
                        # الكامل، رغم أنه غالبًا لن يُصدر نطقًا عربيًا مفهومًا.
                        engine.say(text)
                        engine.runAndWait()
                else:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print(f"[tts] speak failed for {text!r}: {e}")
            finally:
                self.tts_queue.task_done()

    def on_close(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
        self.tts_queue.put(None)  # يوقف _tts_worker بهدوء
        self.destroy()


if __name__ == "__main__":
    app = PracticeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
