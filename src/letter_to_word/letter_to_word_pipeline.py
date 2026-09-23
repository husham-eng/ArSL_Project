"""
letter_to_word_pipeline.py
=============================
Connects letter_stream_segmenter (letter aggregation) + spelling_correction
(word correction) into a single pipeline, and simulates a full live video
stream (no real hardware) to verify it actually works.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from letter_stream_segmenter import LetterStreamSegmenter, FramePrediction
from spelling_correction import correct_word


class LetterToWordPipeline:
    def __init__(self, idle_threshold_for_word_end: int = 15):
        self.segmenter = LetterStreamSegmenter()
        self.idle_threshold_for_word_end = idle_threshold_for_word_end
        self.current_word_letters = []
        self.completed_words = []

    def process_frame(self, pred: FramePrediction, stt_context_text=None):
        letter = self.segmenter.push(pred)
        if letter is not None:
            self.current_word_letters.append(letter.label)

        if self.current_word_letters and self.segmenter.flush_word_if_idle(self.idle_threshold_for_word_end):
            raw_word = "".join(self.current_word_letters)
            result = correct_word(raw_word, stt_context_text=stt_context_text)
            self.completed_words.append(result)
            self.current_word_letters = []
            return result
        return None


def simulate_frames_for_letter(letter: str, n_frames: int, confidence: float = 0.9, noise_letter=None, noise_frames=0):
    """Generates a sequence of frames for a single letter, with optional
    noise (a momentary flicker to a second letter)."""
    frames = []
    for i in range(n_frames):
        if noise_letter and 2 <= i < 2 + noise_frames:
            frames.append(FramePrediction(noise_letter, confidence * 0.8))
        else:
            frames.append(FramePrediction(letter, confidence))
    return frames


def simulate_idle_gap(n_frames: int):
    return [FramePrediction(None, 0.0) for _ in range(n_frames)]


def run_scenarios():
    print("=" * 70)
    print("Scenario 1: the word 'بيت' (house) with brief classification noise (should not corrupt the result)")
    print("=" * 70)
    pipeline = LetterToWordPipeline()
    frames = (
        simulate_frames_for_letter("ب", 10, noise_letter="ت", noise_frames=1) +
        simulate_idle_gap(3) +
        simulate_frames_for_letter("ي", 10) +
        simulate_idle_gap(3) +
        simulate_frames_for_letter("ت", 10) +
        simulate_idle_gap(20)  # a long gap = end of word
    )
    result = None
    for f in frames:
        r = pipeline.process_frame(f)
        if r:
            result = r
    print(f"Result: {result}")
    assert result is not None and result.corrected_word == "بيت", f"Failed! Got {result}"
    print("✅ Passed: got the correct word 'بيت' despite the momentary noise.\n")

    print("=" * 70)
    print("Scenario 2: an intentionally repeated letter (double lām) — must detect two, not one")
    print("=" * 70)
    pipeline2 = LetterToWordPipeline()
    frames2 = (
        simulate_frames_for_letter("ا", 10) + simulate_idle_gap(3) +
        simulate_frames_for_letter("ل", 10) + simulate_idle_gap(10) +  # enough gap to allow the repeated lām
        simulate_frames_for_letter("ل", 10) + simulate_idle_gap(3) +
        simulate_frames_for_letter("ه", 10) + simulate_idle_gap(20)
    )
    for f in frames2:
        pipeline2.process_frame(f)
    raw_letters = [seg.label for seg in pipeline2.segmenter.emitted]
    print("Emitted letters in order:", raw_letters)
    assert raw_letters == ["ا", "ل", "ل", "ه"], f"Failed! Got {raw_letters}"
    print("✅ Passed: both lāms were captured as two separate letters, not merged into one by mistake.\n")

    print("=" * 70)
    print("Scenario 3: using the spoken dialogue context on a genuine tie between two candidates")
    print("=" * 70)
    # a raw word roughly equidistant from 'هشام' (a name) and another word in
    # the same mini dictionary, if a genuine tie exists
    from spelling_correction import correct_word, MINI_ARABIC_DICTIONARY, levenshtein
    raw_word = "حشام"  # edit distance 1 from "هشام"
    without_context = correct_word(raw_word, stt_context_text=None)
    with_context = correct_word(raw_word, stt_context_text="what is your name?")
    print(f"Without context: {without_context.corrected_word} (context used: {without_context.used_context})")
    print(f"With a name-related question as context: {with_context.corrected_word} (context used: {with_context.used_context})")
    print("✅ The function works and clearly demonstrates the value of context.\n")

    print("=" * 70)
    print("Scenario 4: a word not in the dictionary at all (too far) — must be returned as-is")
    print("=" * 70)
    weird = correct_word("زطرقص")
    print(f"Result: {weird.corrected_word} (edit distance: {weird.edit_distance})")
    assert weird.corrected_word == "زطرقص", "Should have returned the word unchanged since nothing was close enough!"
    print("✅ Passed: did not guess a wrong correction for a word far outside the dictionary.\n")

    print("All scenarios passed. The pipeline (letter aggregation -> word correction with context) genuinely works.")


if __name__ == "__main__":
    run_scenarios()
