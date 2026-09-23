"""
letter_stream_segmenter.py
=============================
Detects the start/end of each letter in a live video stream (item 4-a of
the task list).

The problem: the model (letter CNN / word BiLSTM later) produces one
prediction per frame, but a hand stays displayed for several consecutive
frames showing the same letter, and the predictions themselves are noisy
(classification can flicker between two visually similar letters for an
instant). We need to turn a "noisy per-frame prediction stream" into a
"clean sequence of distinct letters" — without:
  1. Repeating the same letter multiple times while it is held steady
     (deduplication)
  2. Missing two consecutive, genuinely-intended identical letters (e.g.
     the double lām in "الله")
  3. Picking up momentary noise as a phantom letter (false positive)

The solution: a simple state machine with a sliding window, plus tracking
of "stability gaps" (hold gaps) to distinguish between "still the same
letter" and "the user lowered and re-raised their hand to intentionally
repeat the same letter."
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SegmenterState(str, Enum):
    IDLE = "idle"                  # no hand / confidence too low
    ACCUMULATING = "accumulating"  # accumulating frames for a candidate letter
    HELD = "held"                  # a letter has been confirmed and emitted; waiting for a change or gap


@dataclass
class FramePrediction:
    """A single model prediction for a single frame (comes straight from
    ColorSignCNN/GrayscaleCNN)."""
    label: Optional[str]   # the predicted letter name, or None if no hand / low confidence
    confidence: float      # 0.0 - 1.0


@dataclass
class SegmentedLetter:
    label: str
    frame_start: int
    frame_end: int
    mean_confidence: float


class LetterStreamSegmenter:
    def __init__(self,
                 window_size: int = 5,
                 min_confidence: float = 0.6,
                 stability_ratio: float = 0.6,
                 idle_gap_to_allow_repeat: int = 8):
        """
        window_size: sliding-window size for majority voting (frames)
        min_confidence: minimum confidence accepted for any single-frame
            prediction (otherwise treated as None/no hand)
        stability_ratio: fraction of frames in the window that must agree
            on the same letter for it to be accepted
        idle_gap_to_allow_repeat: number of "no hand / low confidence"
            frames required before we allow re-emitting the same letter
            that was already emitted (an intentional repeated letter, like
            the two lāms in "الله")
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.stability_ratio = stability_ratio
        self.idle_gap_to_allow_repeat = idle_gap_to_allow_repeat

        self.buffer: deque = deque(maxlen=window_size)
        self.state = SegmenterState.IDLE
        self.last_emitted_label: Optional[str] = None
        self.idle_run = 0
        self.prev_raw_label: Optional[str] = None
        self.last_gap_length = 0  # length of the last no-hand gap that ended right before the current run of frames
        self.current_segment_start = None
        self.current_segment_labels: List[str] = []
        self.current_segment_confidences: List[float] = []

        self.frame_idx = -1
        self.emitted: List[SegmentedLetter] = []

    def _effective_label(self, pred: FramePrediction) -> Optional[str]:
        if pred.label is None or pred.confidence < self.min_confidence:
            return None
        return pred.label

    def push(self, pred: FramePrediction) -> Optional[SegmentedLetter]:
        """
        Called once per new incoming frame. Returns a SegmentedLetter when a
        new confirmed letter is detected, or None if no decision has been
        reached yet.
        """
        self.frame_idx += 1
        label = self._effective_label(pred)

        # If we just transitioned from "no hand" to "hand present", record
        # the length of the gap that just ended (before resetting idle_run)
        # — this is the correct signal for deciding whether to allow a
        # repeated letter, unlike idle_run itself which resets immediately
        # on the first frame with a hand.
        if label is not None and self.prev_raw_label is None:
            self.last_gap_length = self.idle_run

        self.buffer.append((label, pred.confidence))

        if label is None:
            self.idle_run += 1
        else:
            self.idle_run = 0
        self.prev_raw_label = label

        if len(self.buffer) < self.window_size:
            return None  # window not full yet

        # majority vote over the current window (ignoring None when counting)
        labels_only = [l for l, c in self.buffer if l is not None]
        if not labels_only:
            self.state = SegmenterState.IDLE
            return None

        counts = Counter(labels_only)
        winner, win_count = counts.most_common(1)[0]
        ratio = win_count / len(self.buffer)

        if ratio < self.stability_ratio:
            return None  # window not yet stable (flickering between two letters)

        # the window has stabilized on "winner" — is this a new letter to emit?
        if winner == self.last_emitted_label:
            if self.last_gap_length < self.idle_gap_to_allow_repeat:
                return None  # same letter as before, and not enough gap preceded it -> ignore (prevents phantom repeats)
            # a sufficient gap preceded this run + the same letter appeared again = an intentional repeat, allow it

        mean_conf = sum(c for l, c in self.buffer if l == winner) / win_count
        segment = SegmentedLetter(
            label=winner,
            frame_start=self.frame_idx - len(self.buffer) + 1,
            frame_end=self.frame_idx,
            mean_confidence=round(mean_conf, 3),
        )
        self.emitted.append(segment)
        self.last_emitted_label = winner
        self.state = SegmenterState.HELD
        self.last_gap_length = 0  # the gap (if used) has now been consumed — prevent reusing it for the rest of this run
        self.buffer.clear()  # start a completely fresh window after every emission, to avoid window contamination
        return segment

    def flush_word_if_idle(self, idle_threshold: int) -> bool:
        """
        Used by the higher-level layer (letter_to_word_pipeline) to
        determine the end of a word: a gap longer than idle_threshold means
        "the user lowered their hand = the word is finished."
        """
        return self.idle_run >= idle_threshold
