"""
i18n.py
=======
Interface translations for app/practice_session.py: Arabic (default),
English and Russian.

Only the *interface* is translated. The signed content itself stays Arabic
by design: letters, dictionary words, the practice scenarios, recognized
words and the spoken output are all Arabic Sign Language / Arabic, and the
interlocutor's speech recognition stays set to Arabic (ar-SA).

The chosen language is saved in app_settings.json (next to the project,
or next to the .exe in a packaged build) and applied at startup; changing
it from the in-app menu restarts the application.
"""

import json
import os

LANGUAGES = {"ar": "العربية", "en": "English", "ru": "Русский"}
DEFAULT_LANGUAGE = "ar"

STRINGS = {
    "language": {"ar": "🌐 اللغة:", "en": "🌐 Language:", "ru": "🌐 Язык:"},
    "language_busy": {
        "ar": "⚠️ أنهِ الاختبار الميداني أولًا قبل تغيير اللغة",
        "en": "⚠️ Finish the field test before changing the language",
        "ru": "⚠️ Завершите полевой тест перед сменой языка",
    },
    "hint_title": {"ar": "دليل الإشارة المتسلسل:", "en": "Sign sequence guide:", "ru": "Подсказка жестов:"},
    "show": {"ar": "إظهار", "en": "Show", "ru": "Показать"},
    "free_mode": {
        "ar": "🎙️ الوضع الحر (تعرّف صوتي حقيقي)",
        "en": "🎙️ Free mode (live speech recognition)",
        "ru": "🎙️ Свободный режим (распознавание речи)",
    },
    "ref_grid": {"ar": "مرجع كل الحروف (32):", "en": "All letters reference (32):", "ru": "Все буквы (32):"},
    "roi_hint": {
        "ar": "أخضر: استمر | سماوي: الحرف تسجّل | رمادي: لا يد مكتشفة | ✓ فوق الصورة: تأكد نهائيًا",
        "en": "Green: keep signing | Cyan: letter recorded | Gray: no hand detected | ✓ above image: confirmed",
        "ru": "Зелёный: продолжайте | Голубой: буква записана | Серый: рука не найдена | ✓ над картинкой: подтверждено",
    },
    "current_letter_empty": {"ar": "الحرف الحالي: —", "en": "Current letter: —", "ru": "Текущая буква: —"},
    "current_word_empty": {"ar": "الكلمة الحالية: —", "en": "Current word: —", "ru": "Текущее слово: —"},
    "current_letter": {
        "ar": "الحرف الحالي: {letter} (ثقة {conf})",
        "en": "Current letter: {letter} (confidence {conf})",
        "ru": "Текущая буква: {letter} (уверенность {conf})",
    },
    "current_word": {
        "ar": "الكلمة الحالية: {word} (دورات: {cycles}، تغطية: {cov})",
        "en": "Current word: {word} (cycles: {cycles}, coverage: {cov})",
        "ru": "Текущее слово: {word} (циклы: {cycles}, покрытие: {cov})",
    },
    "next_cycle": {
        "ar": "⏱ الدورة التالية خلال: {sec} ث",
        "en": "⏱ Next cycle in: {sec} s",
        "ru": "⏱ Следующий цикл через: {sec} с",
    },
    "start_camera": {"ar": "ابدأ الكاميرا", "en": "Start camera", "ru": "Включить камеру"},
    "stop_camera": {"ar": "أوقف الكاميرا", "en": "Stop camera", "ru": "Выключить камеру"},
    "other_camera": {"ar": "كاميرا أخرى", "en": "Other camera", "ru": "Другая камера"},
    "reset_word": {"ar": "إعادة تعيين الكلمة", "en": "Reset word", "ru": "Сбросить слово"},
    "finish_word": {"ar": "أنهِ الكلمة الآن", "en": "Finish word now", "ru": "Завершить слово"},
    "prev_scenario": {"ar": "السيناريو السابق", "en": "Previous scenario", "ru": "Предыдущий сценарий"},
    "next_scenario": {"ar": "السيناريو التالي", "en": "Next scenario", "ru": "Следующий сценарий"},
    "listen_question": {
        "ar": "🎤 استمع لسؤال المحاور",
        "en": "🎤 Listen to the question",
        "ru": "🎤 Слушать вопрос",
    },
    "listening": {"ar": "🎙️ جارٍ الاستماع...", "en": "🎙️ Listening...", "ru": "🎙️ Слушаю..."},
    "screenshot": {
        "ar": "📸 التقط لقطة للتوثيق",
        "en": "📸 Documentation screenshot",
        "ru": "📸 Снимок для документации",
    },
    "field_test": {"ar": "🧪 اختبار ميداني", "en": "🧪 Field test", "ru": "🧪 Полевой тест"},
    "chat_log": {"ar": "سجل الحوار", "en": "Conversation log", "ru": "Журнал диалога"},
    "suggestions": {"ar": "اقتراحات:", "en": "Suggestions:", "ru": "Подсказки:"},
    "model_ready_fusion": {
        "ar": "النموذج جاهز (رمادي + ملوّن مدمج) ✅",
        "en": "Model ready (grayscale + color fusion) ✅",
        "ru": "Модель готова (слияние серого и цветного) ✅",
    },
    "model_ready_gray": {
        "ar": "النموذج جاهز (رمادي فقط) ✅",
        "en": "Model ready (grayscale only) ✅",
        "ru": "Модель готова (только серый) ✅",
    },
    "model_not_found": {
        "ar": "⚠️ لم يُعثر على النموذج المدرَّب في: {path}",
        "en": "⚠️ Trained model not found at: {path}",
        "ru": "⚠️ Обученная модель не найдена: {path}",
    },
    "model_not_ready": {
        "ar": "⚠️ النموذج لسه ما جهز، انتظر شوي وحاول مرة ثانية",
        "en": "⚠️ The model is still loading, please wait and try again",
        "ru": "⚠️ Модель ещё загружается, подождите и попробуйте снова",
    },
    "scenario_counter": {"ar": "السيناريو {i} / {n}", "en": "Scenario {i} / {n}", "ru": "Сценарий {i} / {n}"},
    "free_mode_start": {
        "ar": "🎙️ الوضع الحر — اضغط زر الميكروفون لبدء محادثة جديدة",
        "en": "🎙️ Free mode — press the microphone button to start a conversation",
        "ru": "🎙️ Свободный режим — нажмите кнопку микрофона, чтобы начать",
    },
    "free_mode_idle": {
        "ar": "🎙️ الوضع الحر — استمع لسؤال جديد في أي وقت",
        "en": "🎙️ Free mode — listen to a new question at any time",
        "ru": "🎙️ Свободный режим — можно слушать новый вопрос в любой момент",
    },
    "stt_missing_lib": {
        "ar": "مكتبة ناقصة ({e}) — ثبّت: pip install SpeechRecognition sounddevice",
        "en": "Missing library ({e}) — install: pip install SpeechRecognition sounddevice",
        "ru": "Нет библиотеки ({e}) — установите: pip install SpeechRecognition sounddevice",
    },
    "stt_record_fail": {
        "ar": "تعذّر تسجيل الصوت (تحقق من الميكروفون): {e}",
        "en": "Could not record audio (check the microphone): {e}",
        "ru": "Не удалось записать звук (проверьте микрофон): {e}",
    },
    "stt_unclear": {
        "ar": "لم يُلتقط كلام مفهوم — حاول مرة أخرى بصوت أوضح وأقرب للميكروفون",
        "en": "No clear speech captured — try again, louder and closer to the microphone",
        "ru": "Речь не распознана — повторите громче и ближе к микрофону",
    },
    "stt_service_fail": {
        "ar": "تعذّر الاتصال بخدمة التعرّف الصوتي (تحقق من الإنترنت): {e}",
        "en": "Could not reach the speech-recognition service (check the internet): {e}",
        "ru": "Нет связи с сервисом распознавания речи (проверьте интернет): {e}",
    },
    "cycle_log": {"ar": "دورة {n}: {text}", "en": "Cycle {n}: {text}", "ru": "Цикл {n}: {text}"},
    "need_camera_ft": {
        "ar": "⚠️ ابدأ الكاميرا أولًا قبل الاختبار الميداني",
        "en": "⚠️ Start the camera before the field test",
        "ru": "⚠️ Включите камеру перед полевым тестом",
    },
    "ft_title": {"ar": "اختبار ميداني", "en": "Field test", "ru": "Полевой тест"},
    "ft_prompt": {"ar": "رقم/اسم المتطوّع:", "en": "Volunteer code:", "ru": "Код участника:"},
    "ft_target": {
        "ar": "أشِر بحرف: {letter}  —  {i} / {n}",
        "en": "Sign the letter: {letter}  —  {i} / {n}",
        "ru": "Покажите букву: {letter}  —  {i} / {n}",
    },
    "ft_status": {
        "ar": "🧪 اختبار ميداني — المتطوّع: {vid}",
        "en": "🧪 Field test — volunteer: {vid}",
        "ru": "🧪 Полевой тест — участник: {vid}",
    },
    "ft_done": {
        "ar": "انتهى الاختبار الميداني — المتطوّع {vid}",
        "en": "Field test finished — volunteer {vid}",
        "ru": "Полевой тест завершён — участник {vid}",
    },
    "ft_done_caption": {
        "ar": "الدقة: {c}/{n} ({acc}) | حُفظت التفاصيل في: {path}",
        "en": "Accuracy: {c}/{n} ({acc}) | details saved to: {path}",
        "ru": "Точность: {c}/{n} ({acc}) | результаты сохранены: {path}",
    },
    "no_camera_found": {
        "ar": "⚠️ تعذّر العثور على كاميرا تعمل (تحقق من توصيل D435، أو راجع الطرفية لتفاصيل الفحص)",
        "en": "⚠️ No working camera found (check the D435 connection, or see the terminal for scan details)",
        "ru": "⚠️ Рабочая камера не найдена (проверьте подключение D435 или журнал в терминале)",
    },
    "cam_color": {"ar": "تيار ملوّن", "en": "color stream", "ru": "цветной поток"},
    "cam_ir": {
        "ar": "⚠️ تيار رمادي/IR — قد يكون خاطئًا",
        "en": "⚠️ gray/IR stream — may be wrong",
        "ru": "⚠️ серый/ИК поток — возможно, неверный",
    },
    "cam_manual": {"ar": "محدد يدويًا", "en": "set manually", "ru": "задан вручную"},
    "no_other_cams": {
        "ar": "⚠️ لا توجد كاميرات مرشّحة أخرى من آخر فحص",
        "en": "⚠️ No other candidate cameras from the last scan",
        "ru": "⚠️ Других камер при последнем поиске не найдено",
    },
    "cam_open_fail": {
        "ar": "⚠️ تعذّر فتح الكاميرا المرشّحة التالية",
        "en": "⚠️ Could not open the next candidate camera",
        "ru": "⚠️ Не удалось открыть следующую камеру",
    },
    "timeout_end": {
        "ar": "⏹ انتهى وقت الاكتشاف (10 ث)",
        "en": "⏹ Detection time is over (10 s)",
        "ru": "⏹ Время распознавания истекло (10 с)",
    },
    "not_recognized": {"ar": "لم يتم التعرف", "en": "Not recognized", "ru": "Не распознано"},
    "not_recognized_caption": {
        "ar": "لا حروف مؤكَّدة خلال 10 ثوانٍ",
        "en": "No confirmed letters within 10 seconds",
        "ru": "За 10 секунд не подтверждено ни одной буквы",
    },
    "need_camera_ss": {
        "ar": "⚠️ شغّل الكاميرا أولًا قبل التقاط لقطة توثيق",
        "en": "⚠️ Start the camera before taking a screenshot",
        "ru": "⚠️ Включите камеру перед снимком",
    },
    "screenshot_saved": {
        "ar": "✅ حُفظت لقطة التوثيق في screenshots/{ts}_*",
        "en": "✅ Screenshot saved to screenshots/{ts}_*",
        "ru": "✅ Снимок сохранён в screenshots/{ts}_*",
    },
    "caption_accepted": {
        "ar": "من: {raw} | اقتراح مقبول قبل اكتمال التهجئة",
        "en": "from: {raw} | suggestion accepted before spelling finished",
        "ru": "из: {raw} | подсказка принята до конца дактилирования",
    },
    "caption_guess": {
        "ar": "من: {raw} | تغطية الاكتشاف: {cov} | ثقة التخمين: {conf}% (حروف {lw} + سياق {cw})",
        "en": "from: {raw} | detection coverage: {cov} | guess confidence: {conf}% (letters {lw} + context {cw})",
        "ru": "из: {raw} | покрытие: {cov} | уверенность: {conf}% (буквы {lw} + контекст {cw})",
    },
    "caption_used_context": {
        "ar": " | استُخدم السياق",
        "en": " | context used",
        "ru": " | использован контекст",
    },
}


def settings_path(write_dir):
    return os.path.join(write_dir, "app_settings.json")


def load_language(write_dir):
    try:
        with open(settings_path(write_dir), encoding="utf-8") as f:
            lang = json.load(f).get("language", DEFAULT_LANGUAGE)
        return lang if lang in LANGUAGES else DEFAULT_LANGUAGE
    except (OSError, ValueError):
        return DEFAULT_LANGUAGE


def save_language(write_dir, lang):
    path = settings_path(write_dir)
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    data["language"] = lang
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Translator:
    def __init__(self, lang=DEFAULT_LANGUAGE):
        self.lang = lang if lang in LANGUAGES else DEFAULT_LANGUAGE

    def __call__(self, key, **kwargs):
        entry = STRINGS[key]
        text = entry.get(self.lang) or entry[DEFAULT_LANGUAGE]
        return text.format(**kwargs) if kwargs else text
