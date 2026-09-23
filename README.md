# Arabic Sign Language Interpretation Assistant

A multimodal assistive communication system for Arabic Sign Language (ArSL): a
dual-branch grayscale/color CNN pipeline for letter classification, a
letter-to-word aggregation module with dialogue-context disambiguation, a
two-tier consent-gated user profile store, and a password-protected
monitoring panel — designed for real-time deployment on a laptop today, and
an embedded Raspberry Pi 4 or cloud endpoint later.

Full technical details, methodology, and results are in
[`docs/scientific_paper.docx`](docs/scientific_paper.docx). Short,
non-technical guides for the actual end user (a non-verbal Arabic Sign
Language signer) are in `docs/`:

- **`user_booklet_ar.docx`** — the primary, original Arabic guide. This is
  the version meant for the real end user and their interlocutors, since
  Arabic is the language they actually speak.
- **`user_booklet_en.docx`** — a parallel English translation of the same
  guide, provided for developers and reviewers browsing this repository who
  don't read Arabic. It is not meant to replace the Arabic version for an
  actual user.

Both include the same illustrations (the ArSL alphabet reference, correct
camera positioning, and the normal-vs-sensitive consent comparison).

> **Status note:** this repository reflects development through real,
> end-to-end training and evaluation of both classification branches:
> GrayscaleCNN (92.51% validation accuracy on ArASL) and ColorSignResNet18
> (97.90% validation accuracy on AASL, via ImageNet transfer learning,
> after an initial from-scratch architecture failed to learn the same
> data — see `src/color/color_sign_cnn.py`), fused end-to-end and
> evaluated against 1,571 real held-out AASL images (96.56% fusion
> accuracy). A first, single-trial proof-of-concept live-camera test on
> the target D435 sensor has also been completed. Systematic,
> multi-trial, multi-signer live-accuracy measurement remains future work
> (`docs/scientific_paper.docx` Section 6) and is the main item still
> gating a tagged, Zenodo-archived release — see "Publishing this
> repository" below.

## Real, reported results

- **GrayscaleCNN** (trained from scratch on the full ArASL corpus — 54,049
  images, 32 classes): **92.51% validation accuracy** after 15 epochs on a
  single GPU. See `docs/figures/training_curve.png` for the real per-epoch
  curve and `docs/scientific_paper.docx` Section 4.3 for the full table.
- **ColorSignResNet18** (ImageNet-pretrained ResNet-18, fully fine-tuned on
  the full AASL corpus — 7,856 images, 31 classes): **97.90% validation
  accuracy** after 15 epochs, reproduced at 97.96% on an independent run.
  An initial from-scratch architecture (`src/color/color_sign_cnn.py`) was
  tried first and failed to learn the data reliably (best 20.6%); see
  `docs/scientific_paper.docx` Section 3.2 for the full account.
- **Real end-to-end fusion evaluation** (1,571 held-out AASL images,
  `evaluation/evaluate_real_fusion.py`): ColorSignResNet18 alone 97.90%,
  GrayscaleCNN alone 21.26% (fed a grayscale derivative of the same real
  color images — evidence of a genuine domain gap, not a bug), and
  unweighted-average fusion 96.56% — measurably *below* the color branch
  alone. This is reported as measured rather than adjusted to match the
  original fusion-improves-accuracy hypothesis; see
  `docs/figures/confusion_matrix_{color,grayscale,fusion}.png` for the
  three confusion matrices and Section 4.7 of the paper for the full
  discussion, including confidence-weighted fusion as the identified next
  step.
- A full 32↔31 class remapping between the two datasets is verified and
  stored in [`src/fusion/grayscale_color_class_mapping.json`](src/fusion/grayscale_color_class_mapping.json).

## Repository structure

```
src/
  grayscale/        GrayscaleCNN (SimpleCNN) architecture + training script (PyTorch)
  color/            color_sign_resnet18.py + train_color_resnet18.py (current, 97.90% val. acc.);
                     color_sign_cnn.py + train_color_cnn.py (archived from-scratch attempt, failed —
                     kept for documentation, see docs/scientific_paper.docx Section 3.2)
  fusion/           ModalityFusionModel (grayscale/color decision fusion), class_mapping.py
                    (shared ArASL<->AASL class-space mapping, used by both evaluation/ and
                    app/practice_session.py's FusionPredictor)
  letter_to_word/   Per-frame letter segmentation, dictionary + context-aware spelling correction
  profile/          Two-tier (ordinary/sensitive) encrypted user profile store + consent flow
  admin/            Password-protected admin panel + classification-attempt log
evaluation/
  evaluate_real_fusion.py     Real end-to-end evaluation of GrayscaleCNN, ColorSignResNet18, and
                               their fusion against real held-out AASL images (Section 4.7)
  plot_confusion_matrices.py  Generates the three confusion-matrix figures from evaluate_real_fusion.py's output
experiments/
  numpy_color_cnn_train.py   Pure-NumPy proof-of-concept (no GPU/PyTorch required) validating
                             the training pipeline mechanics on a small AASL subset
app/
  practice_session.py          Chat-style desktop GUI: live camera → letters → word → TTS
  conversation_scenarios.json  20 verified practice/testing conversations
  requirements_app.txt         Extra dependencies for the GUI app (camera, TTS, Arabic text
                                shaping, and free-mode speech-to-text)
  hand_signs/                   32 real, photographed hand-shape images (one per ArASL letter,
                                 cropped from a reference photo sheet), used for the in-app
                                 sequential signing guide. Real photos rather than line-art
                                 sketches, to help a non-verbal signer recall the exact hand
                                 shape and help a new user physically learn each letter. 'yaa'
                                 (ى) reuses 'ya' (ي)'s photo since AASL/real capture doesn't
                                 distinguish them (Section 4.7).
  models/                       Not tracked in git (binary, ~8MB): place hand_landmarker.task
                                 here for real hand tracking -- see "Enabling real hand tracking"
data/
  aasl_upload_progress.json   Verified per-class image counts and integrity checks for AASL
docs/
  scientific_paper.docx       Full write-up: architecture, experiments, figures, limitations
  user_booklet_ar.docx        Short Arabic user guide (primary, for the actual end user)
  user_booklet_en.docx        Parallel English translation (for developers/reviewers)
  session_handoff_summary.md  Development log / session-to-session handoff notes
  figures/                    All diagrams used in the paper and booklets (architecture, training
                               curve, the three real confusion matrices, etc.)
```

## Setup (laptop / local machine)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick smoke tests (no GPU, no dataset required)

These validate the non-model logic end-to-end in seconds:

```bash
python -m src.letter_to_word.letter_to_word_pipeline   # letter segmentation + spelling correction
python -m src.profile.demo_profile_system              # two-tier consent-gated profile store
python -m src.admin.test_admin_panel                    # password-protected admin panel
```

## Training the models (requires the datasets and, ideally, a GPU)

```bash
# GrayscaleCNN on ArASL (32 classes)
python src/grayscale/train_grayscale_cnn.py --data_dir /path/to/ArASL --epochs 15

# ColorSignResNet18 on AASL (31 classes) — full fine-tuning of an
# ImageNet-pretrained ResNet-18; reached 97.90% validation accuracy.
# Requires src/color/train_color_resnet18.py (see the warning above).
python src/color/train_color_resnet18.py --data_dir /path/to/AASL --epochs 15 \
    --batch_size 32 --freeze_backbone false --out color_sign_resnet18_best.pt

# Real end-to-end evaluation of both branches + their fusion against real
# held-out AASL images (produces fusion_eval_summary.txt and 3 confusion
# matrices — see docs/scientific_paper.docx Section 4.7)
python evaluation/evaluate_real_fusion.py --aasl_dir /path/to/AASL \
    --grayscale_checkpoint grayscale_simplecnn_best.pt \
    --color_checkpoint color_sign_resnet18_best.pt
python evaluation/plot_confusion_matrices.py --prefix fusion_eval
```

Datasets are not included in this repository due to size and third-party
licensing (see Data Sources below); download them from their original
sources and point `--data_dir` at the extracted folder.

## Data sources

- **ArASL** (32-class grayscale alphabet, Latif et al. 2019 and follow-ups).
- **AASL** — [RGB Arabic Alphabets Sign Language Dataset](https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset),
  licensed CC BY-NC-SA 4.0 (non-commercial).

⚠️ Both dataset folder structures have a known case-sensitivity pitfall
(one class folder name differs in letter case from the rest, which silently
corrupts `torchvision.datasets.ImageFolder`'s class indexing). See
`docs/scientific_paper.docx` Section 4.2 before retraining on a fresh copy
of either dataset.

## Citation

If you use this work, please cite it — see [`CITATION.cff`](CITATION.cff).
A DOI (via Zenodo) will be added here once the first GitHub release is
archived.

## Practice / testing GUI application

`app/practice_session.py` is a local desktop application with a modern,
chat-style interface (inspired by messaging apps: the interlocutor's
question appears as one bubble, the signer's recognized-and-corrected
answer appears as a reply bubble on the other side, spoken aloud via TTS).
It comes with 20 built-in conversation scenarios
(`app/conversation_scenarios.json`) for practicing and testing the full
pipeline — live camera → letter recognition → word aggregation → dictionary
and context-aware spelling correction → text-to-speech — end to end on
real hardware, no cloud/Colab dependency for the core recognition path.

Beyond the base pipeline, the app includes:
- **Hand-tracked ROI crop** (with a fixed-center fallback, see "Enabling real
  hand tracking" below) crops just the hand region before it reaches the
  model, instead of feeding the model the full frame (face, shoulders,
  background) — closing the gap between AASL/ArASL's cropped training
  images and a real live camera feed, and improving confidence accordingly.
  The guide box is color-coded: green = still counting this capture cycle,
  cyan = a letter was just confirmed this cycle, orange = the cycle just
  closed with no confident detection, gray = hand-tracking is on but no
  hand is visible this frame. Both the tracked box's padding
  (`HAND_BOX_PADDING_FRACTION`) and the fixed-fallback box's size
  (`HAND_ROI_FRACTION`) are set to half their original values, for a
  visibly tighter box around the hand.
- **Fixed-interval letter capture, every 2 seconds** (`CAPTURE_INTERVAL_SEC`):
  rather than waiting indefinitely for a stable prediction (which gave no
  feedback at all when nothing ever stabilized), the app commits a decision
  every 2 seconds without exception — either a confidently majority-voted
  letter, or an explicit "no detection" — so the signer always has a clear,
  predictable rhythm for when to move to the next letter or retry. A live
  countdown ("⏱ الدورة التالية خلال: ...") shows time left in the current
  cycle, and each cycle's outcome (the letter, or "0" for none) is logged
  in real time as a small entry in the chat panel as the word is being
  signed — not just the final guessed word at the end.
- **Strict 10-second word cutoff with a visual end-of-detection animation**
  (`WORD_RECOGNITION_TIMEOUT_SEC`): exactly 10 seconds after a word's first
  capture cycle begins, detection stops outright and the guide box animates
  shrinking to a point at the frame's center (making the cutoff visually
  unambiguous). If at least one letter was confirmed, the accumulated word
  is guessed and spoken; if none were, the app explicitly announces
  "لم يتم التعرف" (not recognized) rather than silently resetting, so the
  person always gets clear feedback that the 10 seconds ended. The manual
  "أنهِ الكلمة الآن" button remains available to end a word earlier.
- **"خوارزمية التكهن" (the guessing algorithm)** — `guess_word_with_context`
  in `spelling_correction.py`: picks the final word by blending two scored
  signals per dictionary candidate — similarity to the confidently-detected
  letters, and relevance to the conversation's context category (e.g.
  names, greetings) — with the blend weight itself set by what fraction of
  this word's fixed-interval cycles actually produced a confident letter.
  A cleanly-signed word trusts the letters almost entirely; a word with
  many empty cycles (poor lighting, an unfamiliar letter) leans more on
  what the interlocutor's question already implies. The resulting
  confidence percentage and the letters/context split are shown in the
  chat bubble's caption for full transparency.
- **Full 32-letter reference grid**: a permanent, always-visible two-row
  grid of all 32 real hand-sign photos beneath the camera view (distinct
  from the dynamic hint strip above it, which only shows the current
  target word's letters) — lets the signer check any letter's hand shape
  at any time, not just the ones in the word currently being practiced.
  Images are 3× the hint strip's size (108×156), and the panel scrolls
  horizontally (a plain `tkinter.Canvas` + `Scrollbar`, not
  `CTkScrollableFrame` — see the note below on why) since all 16 letters
  per row do not fit at that size within a normal window width; it opens
  already scrolled to show the first letter (ع) on the right, matching
  Arabic reading order.
- **Live word-prediction/autocomplete** (`predict_words_from_prefix`):
  suggests dictionary words matching the letters signed so far, so the
  signer can accept a word early instead of finishing every letter by hand.
- **"الوضع الحر" (free mode)**: real speech-to-text on the interlocutor's
  spoken question (Google Web Speech via a real microphone recording, not
  a scripted scenario), used as genuine context for
  `guess_word_with_context`.
- **Documentation screenshots** (📸 button): saves the real full frame, the
  real model-input crop, and a caption with the actual predicted letter and
  confidence at that moment, to `screenshots/` — real artifacts, not staged
  illustrations, for the paper's methodology figures.
- **Robust camera discovery**: probes multiple indices and backends
  (DirectShow/MSMF/ANY) with warmup frames and a color-vs-IR/depth
  saturation check, since the D435 exposes several sub-streams as separate
  OpenCV indices.
- **PyInstaller-ready**: `BASE_DIR`/`WRITE_DIR` are split so a packaged
  `.exe` (see `packaging/`) reads bundled resources correctly while still
  writing the classification log and screenshots next to the executable.

> ⚠️ **Runtime note:** this application requires a webcam, a display, and
> a trained checkpoint at the repository root. **It tries the full fusion
> pipeline first**: if both `grayscale_simplecnn_best.pt` and
> `color_sign_resnet18_best.pt` are present (and
> `src/color/color_sign_resnet18.py` is in this checkout), it loads
> `FusionPredictor`, which reproduces Figure 3's pipeline exactly on the
> hand-tracked, cropped frame — GrayscaleCNN + ColorSignResNet18 +
> ModalityFusionModel (average fusion), the same configuration evaluated
> end-to-end at 96.56% in `docs/scientific_paper.docx` Section 4.7. If the
> color checkpoint or `color_sign_resnet18.py` is missing, it automatically
> falls back to `LetterPredictor` (grayscale-only), so the app still runs
> with only Track B complete. The status label under the camera view shows
> which mode loaded ("رمادي + ملوّن مدمج" = fusion, "رمادي فقط" =
> grayscale-only). The full GUI (fusion path, both real trained
> checkpoints, real hand-sign photos, real hand tracking with its
> fixed-box fallback, free-mode toggle, screenshot button) has been
> launched end-to-end on a virtual display and produced a valid prediction
> from a synthetic test frame — but this sandbox has no physical webcam,
> microphone, or access to MediaPipe's model-hosting domain (see "Enabling
> real hand tracking" below), so the live-camera capture loop, the
> free-mode STT path, real hand-tracking accuracy, and accuracy on genuine
> hand gestures still need testing on your own machine per the steps
> below. Report any runtime issue for a fix.

### Enabling real hand tracking

By default the ROI guide box that crops the classifier's input is a
**fixed box at the center of the frame** (you move your hand to it, not
the other way around) — this always works with no extra setup, since it
only needs `opencv-python`. For a box that **follows your hand and can be
made larger**, the app also supports real hand detection via MediaPipe:

1. Make sure `mediapipe` is installed (included in
   `app/requirements_app.txt`).
2. Download the hand-landmark model file (~8 MB) from Google's official,
   versioned MediaPipe model host:
   ```
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```
3. Save it to exactly this path in the repo:
   ```
   app/models/hand_landmarker.task
   ```
4. Run the app as usual. The console prints `[info] Real hand tracking
   active (MediaPipe HandLandmarker).` on success. The guide box now
   follows your detected hand (green = keep going, cyan = a letter was
   just confirmed, **gray = tracking is on but no hand is detected in
   this frame** — the app skips classification entirely on gray frames
   rather than classifying an empty background), padded generously around
   the detected hand (`HAND_BOX_PADDING_FRACTION` in `practice_session.py`,
   default 80% larger than the tight landmark bounding box — raise this if
   you want an even bigger box).
5. If `mediapipe` is not installed, or the model file is not at that exact
   path, the app prints a one-line `[info]` message and automatically
   falls back to the fixed-center box — nothing else changes, and the app
   is never blocked on this.

> ⚠️ This path could not be tested end-to-end in the environment these
> changes were written in: that sandbox's network is restricted to a
> handful of package registries (PyPI, npm, GitHub, apt) and cannot reach
> `storage.googleapis.com`, so the model file itself could not be
> downloaded there. `mediapipe` was confirmed to install and import
> correctly, and `HandLandmarker.create_from_options()` was confirmed to
> raise a clean, caught `FileNotFoundError` when the model file is absent
> (exercising the exact fallback path above) — but real detection accuracy
> on an actual camera frame has not been verified. Please test this on
> your own machine and report any issue.

> ✅ The fixed-interval capture cycle, the strict 10-second cutoff with its
> shrink animation, the live per-cycle chat log, and `guess_word_with_context`
> **were** verified end-to-end in that same sandbox, by driving `PracticeApp`
> under a virtual display with a scripted fake predictor and manually
> pumping Tkinter's event loop (so the `after()`-based animation actually
> ran, not just got scheduled): a word signed across multiple capture
> cycles correctly reached the strict cutoff, the shrink animation
> completed and returned the guide box to its normal state, `_finalize_word`
> fired with the right accumulated letters, and each cycle's result
> appeared as its own row in the chat panel in real time. This confirms
> the state machine itself is correct; only the genuinely physical parts
> (a real hand in front of a real camera) still need your own testing.

> 🐛 **A real bug was found and fixed** while adding the enlarged reference
> grid: the app's window height (720px) was left unchanged even after
> adding ~400px of new content (the grid, at 3× size, plus its scrollbar),
> so everything from the grid downward — the grid itself, and on some
> screens the bottom control buttons — was silently clipped by Tkinter's
> `.pack()` geometry manager, which does not warn or scroll when content
> overflows its allotted space; it simply hides whatever does not fit. A
> first attempt to fix the grid specifically by switching to
> `CTkScrollableFrame(orientation="horizontal")` looked correct in code but
> rendered as a fully empty panel; direct pixel-coordinate inspection
> (`winfo_rootx()` on individual grid cells) showed the scrollable frame's
> `xview()` reporting itself fully scrolled (`1.0`) while the actually
> *visible* pixels still showed the opposite, unscrolled end of the
> content — an internal inconsistency in that widget's horizontal mode in
> this customtkinter version. It was replaced with a plain
> `tkinter.Canvas` + `tkinter.Scrollbar`, which is what the grid now uses;
> the true root cause, though, was the window height, fixed by increasing
> `self.geometry(...)` from `"1180x720"` to `"1280x1200"`. If your screen
> is shorter than 1200px, maximize the window, or reduce
> `HAND_BOX_PADDING_FRACTION`'s neighboring reference-grid image size (108,
> 156) in `_build_full_reference_grid` to make the panel shorter.

> 🐛 **A second real bug, found via live user testing**: `HandLandmarker`
> reported `Unable to open file at <path>` for the *exact, correct* path to
> a genuinely present, correctly-sized (7.45 MB) `hand_landmarker.task` --
> Windows Explorer and Python's own `os.path.getsize()` on that identical
> path worked fine, only MediaPipe's loader failed. The cause: the user's
> project folder name contained Arabic text
> ("...مشروع ترجمة لغة الاشارة بكاميرا العمق..."), and MediaPipe's C++
> backend (ported from Linux, like many such libraries) does not reliably
> handle non-ASCII paths on Windows. `HandTracker.__init__` now detects a
> non-ASCII `model_path` and transparently copies the model to
> `%TEMP%\arsl_hand_landmarker.task` (an ASCII-safe location) before
> handing it to MediaPipe, rather than requiring you to rename or relocate
> your project folder. Verified with the user's real folder-name pattern
> and a dummy model file: the copy step worked, and the resulting error
> changed from a path/file error to a "not a valid model" error --
> confirming MediaPipe successfully *opened* the file from the new
> location. With a genuine `hand_landmarker.task`, this error does not
> occur at all.

### Setup

```bash
pip install -r requirements.txt
pip install -r app/requirements_app.txt
```

### Run

From the repository root, with `grayscale_simplecnn_best.pt` present there:

```bash
python app/practice_session.py
```

**Camera selection:** the app no longer assumes camera index `0`. On start,
"ابدأ الكاميرا" (Start Camera) automatically scans indices `0`–`5` (via the
`CAP_DSHOW` backend) and picks the best real **color** stream — not just
one that reports `isOpened() == True` or merely "not pitch black". This
matters specifically for the RealSense D435: it exposes its color, IR, and
depth sensors as *separate* OpenCV camera indices, and an IR/depth stream
under normal room light can look non-black in brightness terms while still
being a grayscale feed useless for classification. The scan checks color
saturation to tell them apart, and prints a full diagnostic line per index
to the terminal (brightness, saturation, and classification) so you can
see exactly what was found. The chosen index and stream type are shown
next to the Start/Stop button (a "⚠️ grayscale/IR" label means no real
color stream was found and it fell back to one).

If the wrong stream still gets picked, click **"كاميرا أخرى"** (Try
another camera) next to Start/Stop to cycle to the next non-black
candidate from the same scan, without leaving the app. To skip the scan
entirely and force a specific index every time:

```bash
# Windows (cmd)
set ARSL_CAMERA_INDEX=1
python app/practice_session.py

# Windows (PowerShell)
$env:ARSL_CAMERA_INDEX = "1"
python app/practice_session.py
```

To find the right index for a specific camera (e.g. the RealSense D435)
yourself, run:

```bash
python -c "import cv2; [print(i, cv2.VideoCapture(i, cv2.CAP_DSHOW).isOpened()) for i in range(4)]"
```

Click "ابدأ الكاميرا" (Start Camera), then sign the answer to the question
shown in the chat panel, letter by letter. A **sequential hand-sign guide**
above the camera view (toggle with the "إظهار" checkbox) shows, in order,
the exact hand shape for each letter of the target answer — useful when the
person testing the app is not themselves a fluent signer and needs a visual
reference for what to perform (the word "لا" is shown as its own single
combined sign, matching how it is actually signed, rather than as two
separate letters). Lower your hand briefly when the word is complete; the
app will aggregate the letters, correct the spelling (using the on-screen
question as context when needed), speak the answer aloud, and add it to
the conversation as a reply bubble. Use "السيناريو التالي" / "السيناريو
السابق" (Next/Previous scenario) to move through the 20 practice
conversations, and "إعادة تعيين الكلمة" (Reset word) to discard the current
in-progress word without waiting for the idle timeout.

Every recognition attempt (successful or not) is logged to
`classification_log.db` at the repository root — the same log the
password-protected admin panel (`src/admin/admin_panel.py`) reads from, so
a real practice session doubles as real monitoring data.



## Before publishing (GitHub release → Zenodo DOI)

Both classification branches are now trained and evaluated end-to-end on
real data (see "Real, reported results" above), which was the main
scientific blocker for a citable release. Before tagging a release:

1. ~~Add the two missing color-branch files~~ — done: `src/color/color_sign_resnet18.py`
   and `src/color/train_color_resnet18.py` are now present, verified to load real
   trained checkpoints and run the full fusion pipeline end-to-end.
2. **Fill in `CITATION.cff` and `LICENSE`** — both still contain
   `REPLACE_WITH_YOUR_...` placeholders (author name, affiliation, ORCID,
   GitHub repository URL, copyright name). Zenodo reads `CITATION.cff` for
   the archived record's author metadata, so this must be done *before*
   the first release, not after.
3. **(Optional but recommended) real laptop live-camera testing** across
   more than one trial/lighting condition remains open per
   `docs/scientific_paper.docx` Section 6; the current live-test evidence
   is a single proof-of-concept trial. This does not block a release, but
   is worth noting explicitly in the release notes if still pending.

### GitHub → Zenodo → DOI steps

1. Push this repository to GitHub (public).
2. On [zenodo.org](https://zenodo.org), sign in with GitHub, go to
   *GitHub* under your profile settings, and flip the toggle on for this
   repository.
3. On GitHub, create a **Release** (not just a tag) — e.g. `v1.0.0`. Zenodo
   automatically archives that release and mints a DOI within a few
   minutes.
4. Copy the DOI badge Zenodo gives you into this README, and add the DOI
   itself to `docs/scientific_paper.docx`'s "Ethics, Consent, and
   Data/Code Availability" section (currently a placeholder statement
   there says the scripts are "available from the corresponding author
   upon reasonable request" — replace that sentence with the DOI once
   minted).
5. Every subsequent GitHub Release gets its own DOI automatically, plus
   Zenodo maintains a "concept DOI" that always resolves to the latest
   version — cite the concept DOI in the paper if you expect to publish
   further releases later.

## License

See [`LICENSE`](LICENSE).

## Status

This is active, in-progress research software. See
`docs/session_handoff_summary.md` for exactly what is validated versus
pending at any given point, and `docs/scientific_paper.docx` Section 6 for
documented limitations.
