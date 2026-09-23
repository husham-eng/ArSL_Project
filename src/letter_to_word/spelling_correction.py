"""
spelling_correction.py
=========================
Dictionary/language-based spelling correction for the raw letter sequence
coming from letter_stream_segmenter (item 4-b), with an explicit point of
use for the spoken dialogue context (the interlocutor's speech via STT) —
item 4-c.

Approach: edit-distance (Levenshtein) correction against an Arabic
dictionary (here, a small list for proof-of-concept purposes — to be
replaced later with a full Arabic dictionary or a language model, without
changing the interface).

The specific spoken-context point of use: if the interlocutor said a
sentence containing certain words (e.g., asked "What's your name?"), and
the raw letters received are ambiguous between two roughly-equal-distance
candidates, we prefer the candidate that is more semantically/contextually
relevant to the topic of the question (here: preferring words from the
"names" dictionary if the question was about a name), instead of relying
only on general dictionary frequency.

Note on language: the dictionary and pattern lists below are genuine Arabic
vocabulary data — this is the actual language the system corrects spelling
in and the actual language interlocutors speak, not documentation, so it is
intentionally left in Arabic.
"""

from dataclasses import dataclass
from typing import List, Optional


# A small dictionary for proof-of-concept purposes only — in the real system
# this is replaced with a full Arabic dictionary (e.g. Tashkeela word lists
# or a Hunspell Arabic dictionary) loaded from an external file.
MINI_ARABIC_DICTIONARY = [
    "بيت", "باب", "بنت", "ولد", "اسم", "اسمي", "كتاب", "قلم", "طالب",
    "جامعة", "مستشفى", "دكتور", "ممرضة", "شكرا", "مرحبا", "نعم", "لا",
    "هشام", "حسام", "احمد", "محمد", "سارة", "فاطمة", "خالد", "عمر", "علي",
]

# Context dictionaries: keywords in the question -> a preferred candidate
# category to use when there is ambiguity
CONTEXT_CATEGORY_HINTS = {
    "name": ["هشام", "احمد", "محمد", "سارة", "فاطمة", "خالد", "عمر", "علي", "اسمي", "اسم"],
    "greeting": ["مرحبا", "شكرا", "نعم", "لا"],
}

QUESTION_KEYWORDS_TO_CATEGORY = {
    "name": ["اسمك", "شو اسمك", "who are you", "what's your name", "what is your name"],
    "greeting": ["كيفك", "شلونك", "هلا", "hello", "hi"],
}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,     # insertion
                prev_row[j] + 1,         # deletion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


@dataclass
class CorrectionResult:
    raw_word: str
    corrected_word: str
    edit_distance: int
    used_context: bool
    candidates_considered: List[str]


def detect_context_category(stt_context_text: Optional[str]) -> Optional[str]:
    """Determines the context category (e.g. 'name') from the interlocutor's
    speech, if a match is found."""
    if not stt_context_text:
        return None
    text = stt_context_text.strip().lower()
    for category, keywords in QUESTION_KEYWORDS_TO_CATEGORY.items():
        if any(kw in text for kw in keywords):
            return category
    return None


def predict_words_from_prefix(prefix: str,
                               dictionary: Optional[List[str]] = None,
                               max_results: int = 5) -> List[str]:
    """Live word-prediction (autocomplete) while the person is still
    signing, so they can accept a suggested word early instead of always
    finishing every letter by hand — a proof-of-concept for the same idea
    behind predictive text on a phone keyboard, adapted to sign input.

    Returns every dictionary word that starts with `prefix` (an exact match
    for the whole word is included too — useful the moment the prefix
    already equals a full short word, e.g. "لا" or "نعم"), shortest first
    as a simple stand-in for "more likely/common" until a real frequency-
    ranked dictionary replaces MINI_ARABIC_DICTIONARY (see the module
    docstring). Returns an empty list for an empty prefix, since every word
    would "match" and that is not a useful suggestion.

    This is intentionally a separate, simpler function from correct_word():
    correct_word() runs once, after the full word is finished, and tolerates
    classification mistakes via edit distance. predict_words_from_prefix()
    runs repeatedly, after every new confidently-classified letter, and only
    needs exact-prefix matching — swapping the dictionary out for a real one
    later does not require changing either function's interface.
    """
    if not prefix:
        return []
    dictionary = dictionary or MINI_ARABIC_DICTIONARY
    matches = [word for word in dictionary if word.startswith(prefix)]
    matches.sort(key=len)
    return matches[:max_results]


@dataclass
class GuessResult:
    """Output of guess_word_with_context() -- see that function's
    docstring. Every field is exposed (not just guessed_word) so the
    caller can show/log *why* a particular word was picked, e.g. in the
    chat bubble's caption or the classification log."""
    raw_word: str
    guessed_word: str
    confidence_percent: float   # 0-100, the blended score of the winning candidate
    letter_score: float         # 0-1, how well guessed_word matches the raw detected letters
    context_score: float        # 0-1, how well guessed_word matches the conversation context
    letter_weight: float        # 0-1, how much the blend trusted the letters this time
    context_weight: float       # 0-1 = 1 - letter_weight
    used_context: bool          # whether stt_context_text matched a known category at all


def guess_word_with_context(raw_word: str,
                             letter_confidence_coverage: float,
                             dictionary: Optional[List[str]] = None,
                             stt_context_text: Optional[str] = None,
                             min_letter_weight: float = 0.3,
                             max_letter_weight: float = 0.9) -> GuessResult:
    """
    "خوارزمية التكهن" (the guessing algorithm): picks the most likely
    intended word by blending two independent signals into one percentage
    score per dictionary candidate, then returning the best-scoring one --
    rather than treating the raw signed letters as ground truth the way
    correct_word() does (correct_word remains available for that simpler,
    letters-only use case; this function is for the fixed-interval capture
    flow in practice_session.py, where some capture cycles may have no
    confident detection at all).

    The two signals, per candidate dictionary word:
      - letter_score: normalized similarity between raw_word (the
        concatenation of only the CONFIDENTLY captured letters, in order --
        capture cycles with no confident detection are simply skipped, not
        treated as a wildcard character) and the candidate, via Levenshtein
        edit distance normalized by the longer string's length.
      - context_score: 1.0 if the candidate is in the preferred-word list
        for the conversation-context category detected from
        stt_context_text (see detect_context_category /
        CONTEXT_CATEGORY_HINTS), else a neutral 0.3 baseline -- neutral
        rather than 0.0, so an out-of-category word is not zeroed out
        entirely just because no context matched, or the context guess was
        wrong.

    The blend weight is NOT fixed: letter_weight is set directly from
    letter_confidence_coverage -- the fraction of this word's fixed-
    interval capture cycles (see CAPTURE_INTERVAL_SEC in
    practice_session.py) that actually produced a confident detection, out
    of ALL cycles that ran for this word, including the ones where nothing
    was confidently detected. Intuitively: a word signed cleanly with every
    cycle landing a confident letter should trust those letters almost
    entirely; a word where most cycles came up empty (poor lighting, an
    unfamiliar letter, camera angle) should lean much more on what the
    interlocutor's question already implies. Clamped to
    [min_letter_weight, max_letter_weight] so neither signal is ever
    completely ignored even at 0% or 100% coverage.

    Returns a GuessResult with confidence_percent (0-100) so the caller can
    show/log how confident this particular guess was -- this is the actual
    percentage requested: how much of the final decision came from context
    vs. from the detected letters, expressed as one blended confidence
    figure for the winning word, with letter_weight/context_weight showing
    the two contributions separately.
    """
    dictionary = dictionary or MINI_ARABIC_DICTIONARY
    letter_weight = max(min_letter_weight, min(max_letter_weight, letter_confidence_coverage))
    context_weight = 1.0 - letter_weight

    category = detect_context_category(stt_context_text)
    preferred = set(CONTEXT_CATEGORY_HINTS.get(category, [])) if category else set()

    best_word, best_score = None, -1.0
    best_letter_score, best_context_score = 0.0, 0.0
    for word in dictionary:
        if raw_word:
            dist = levenshtein(raw_word, word)
            max_len = max(len(raw_word), len(word), 1)
            letter_score = max(0.0, 1.0 - dist / max_len)
        else:
            # no confidently-detected letters at all this word -- letters
            # contribute nothing, so (given the weight clamp above) context
            # alone decides among candidates, at min_letter_weight/context_weight.
            letter_score = 0.0
        context_score = 1.0 if word in preferred else 0.3
        combined = letter_weight * letter_score + context_weight * context_score
        if combined > best_score:
            best_word, best_score = word, combined
            best_letter_score, best_context_score = letter_score, context_score

    if best_word is None:
        # empty dictionary -- degrade to the raw word unchanged rather than crash
        return GuessResult(raw_word, raw_word, 0.0, 0.0, 0.0, letter_weight, context_weight, False)

    return GuessResult(
        raw_word=raw_word,
        guessed_word=best_word,
        confidence_percent=round(best_score * 100, 1),
        letter_score=round(best_letter_score, 3),
        context_score=round(best_context_score, 3),
        letter_weight=round(letter_weight, 3),
        context_weight=round(context_weight, 3),
        used_context=category is not None,
    )


def correct_word(raw_word: str,
                  dictionary: Optional[List[str]] = None,
                  stt_context_text: Optional[str] = None,
                  max_edit_distance: int = 2) -> CorrectionResult:
    """
    raw_word: the raw string from letter aggregation (may contain simple
        classification errors).
    stt_context_text: the interlocutor's most recent spoken sentence (from
        STT) — used to break ties when more than one candidate is at
        roughly the same edit distance.
    """
    dictionary = dictionary or MINI_ARABIC_DICTIONARY

    if raw_word in dictionary:
        return CorrectionResult(raw_word, raw_word, 0, used_context=False, candidates_considered=[raw_word])

    scored = [(word, levenshtein(raw_word, word)) for word in dictionary]
    scored.sort(key=lambda x: x[1])
    best_distance = scored[0][1]

    if best_distance > max_edit_distance:
        # nothing in the dictionary is close enough — return the raw word as-is
        # (better than guessing wrong)
        return CorrectionResult(raw_word, raw_word, best_distance, used_context=False,
                                 candidates_considered=[w for w, d in scored[:3]])

    # all words tied at the same minimum edit distance = genuinely tied candidates
    tied_candidates = [w for w, d in scored if d == best_distance]

    if len(tied_candidates) == 1:
        return CorrectionResult(raw_word, tied_candidates[0], best_distance,
                                 used_context=False, candidates_considered=tied_candidates)

    # a genuine tie between more than one word -> this is exactly where we
    # use the spoken dialogue context to break the tie
    category = detect_context_category(stt_context_text)
    if category:
        preferred = CONTEXT_CATEGORY_HINTS.get(category, [])
        context_matches = [w for w in tied_candidates if w in preferred]
        if context_matches:
            return CorrectionResult(raw_word, context_matches[0], best_distance,
                                     used_context=True, candidates_considered=tied_candidates)

    # no context available or it didn't help -> fall back to the shortest
    # word (a simple default guess, open to future improvement)
    fallback = min(tied_candidates, key=len)
    return CorrectionResult(raw_word, fallback, best_distance, used_context=False,
                             candidates_considered=tied_candidates)
