"""
analyze_field_test_results.py
=============================
Analyzes the per-volunteer CSV files written by the Field Test Mode of
app/practice_session.py (v1.0.4 format) and produces everything needed for
the paper's field-validation section:

  <out_prefix>.txt                    overall accuracy (+95% Wilson CI),
                                      per-volunteer accuracy (mean +/- SD),
                                      per-letter accuracy, failure-cause
                                      breakdown, confidence-threshold
                                      sensitivity, most frequent confusions
  <out_prefix>_confusion_matrix.png   32 x 33 confusion matrix (last column =
                                      no letter accepted in the window)
  <out_prefix>_threshold_sensitivity.csv
  <out_prefix>_per_letter.csv

Only files whose volunteer_id matches --id_pattern (default V01..V99) are
included, so pilot/rehearsal files (V00, v00, voo...) are excluded even if
they are accidentally left in the results folder.

Usage (from the repository root):
    python evaluation/analyze_field_test_results.py --results_dir field_test_results
    # optional: per-group results by ArSL experience
    python evaluation/analyze_field_test_results.py --results_dir field_test_results \
        --volunteer_info volunteer_info.csv
    # volunteer_info.csv columns: volunteer_id,arsl_experience,hand

Older CSVs (before v1.0.4) lack the diagnostic columns; they are still
analyzed, with failure causes reported as "unknown" and without threshold
sensitivity.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict

# Must match app/practice_session.py (v1.0.4)
STABILITY_RATIO = 0.6
NO_PRED = "∅"


def read_rows(results_dir, id_pattern):
    rows, skipped = [], Counter()
    pat = re.compile(id_pattern)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                vid = (r.get("volunteer_id") or "").strip()
                if not pat.fullmatch(vid):
                    skipped[vid] += 1
                    continue
                r["_file"] = os.path.basename(path)
                r["correct"] = str(r.get("correct")).strip().lower() == "true"
                r["confidence"] = float(r.get("confidence") or 0)
                if r.get("frame_predictions"):
                    r["_frames"] = [(l, float(c)) for l, c in json.loads(r["frame_predictions"])]
                else:
                    r["_frames"] = None
                rows.append(r)
    return rows, skipped


def decide(frames, threshold, ratio=STABILITY_RATIO):
    """Re-implements the app's per-window decision (practice_session.py,
    _tick_capture_cycle) so any threshold can be evaluated post hoc."""
    confident = [l for l, c in frames if c >= threshold]
    if not confident:
        return None
    label, count = Counter(confident).most_common(1)[0]
    return label if count / len(confident) >= ratio else None


def failure_cause(r):
    if r["correct"]:
        return "correct"
    if r.get("frames_total") in (None, ""):
        return "unknown (pre-v1.0.4 file)"
    ft, fc = int(r["frames_total"]), int(r["frames_confident"])
    if r.get("predicted_letter"):
        return "wrong letter accepted"
    if ft == 0:
        return "no hand detected"
    if fc == 0:
        return "all frames below threshold"
    return "no stable majority"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ar(text):
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def plot_confusion(rows, letters, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = letters + [NO_PRED]
    idx_r = {l: i for i, l in enumerate(letters)}
    idx_c = {l: i for i, l in enumerate(cols)}
    m = [[0] * len(cols) for _ in letters]
    for r in rows:
        p = r.get("predicted_letter") or NO_PRED
        if r["target_letter"] in idx_r and p in idx_c:
            m[idx_r[r["target_letter"]]][idx_c[p]] += 1

    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(letters)))
    ax.set_xticklabels([ar(c) for c in cols], fontsize=10)
    ax.set_yticklabels([ar(l) for l in letters], fontsize=10)
    ax.set_xlabel("Predicted letter (∅ = no letter accepted in the window)")
    ax.set_ylabel("Target letter")
    ax.set_title(f"Field-test confusion matrix (n = {len(rows)} attempts)")
    for i in range(len(letters)):
        for j in range(len(cols)):
            if m[i][j]:
                ax.text(j, i, m[i][j], ha="center", va="center", fontsize=7,
                        color="white" if m[i][j] > max(map(max, m)) / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="field_test_results")
    ap.add_argument("--out_prefix", default="field_test_summary")
    ap.add_argument("--id_pattern", default=r"V\d{2}",
                    help="regex (full match) for volunteer_id values to include; "
                         "default excludes pilot IDs such as V00 only if you use V01+")
    ap.add_argument("--exclude_ids", default="V00",
                    help="comma-separated IDs to exclude even if they match --id_pattern")
    ap.add_argument("--volunteer_info", default=None)
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    args = ap.parse_args()

    rows, skipped = read_rows(args.results_dir, args.id_pattern)
    excluded = {x.strip() for x in args.exclude_ids.split(",") if x.strip()}
    rows = [r for r in rows if r["volunteer_id"] not in excluded]
    if not rows:
        raise SystemExit(f"No matching rows in {args.results_dir} (skipped IDs: {dict(skipped)})")

    out = []
    w = out.append
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    vols = sorted({r["volunteer_id"] for r in rows})
    thr_used = sorted({r.get("confidence_threshold") or "n/a" for r in rows})

    w("FIELD TEST SUMMARY")
    w("=" * 60)
    w(f"Files: {len({r['_file'] for r in rows})} | Volunteers: {len(vols)} | Attempts: {n}")
    w(f"Confidence threshold recorded in files: {', '.join(thr_used)}")
    if skipped:
        w(f"Skipped rows (volunteer_id not matching {args.id_pattern}): {dict(skipped)}")
    w("")
    w(f"OVERALL ACCURACY: {k}/{n} = {k/n:.1%}  (95% Wilson CI {lo:.1%} - {hi:.1%})")

    per_vol = {}
    for v in vols:
        vr = [r for r in rows if r["volunteer_id"] == v]
        per_vol[v] = (sum(r["correct"] for r in vr), len(vr))
    accs = [c / t for c, t in per_vol.values()]
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    w(f"Per-volunteer mean accuracy: {statistics.mean(accs):.1%} +/- {sd:.1%} (SD), "
      f"range {min(accs):.1%} - {max(accs):.1%}")
    w("")
    w("PER-VOLUNTEER")
    for v, (c, t) in per_vol.items():
        w(f"  {v}: {c}/{t} = {c/t:.1%}")

    if args.volunteer_info and os.path.exists(args.volunteer_info):
        with open(args.volunteer_info, encoding="utf-8-sig", newline="") as f:
            info = {r["volunteer_id"].strip(): r for r in csv.DictReader(f)}
        groups = defaultdict(list)
        for v, (c, t) in per_vol.items():
            groups[(info.get(v) or {}).get("arsl_experience", "unrecorded")].append((c, t))
        w("")
        w("BY ArSL EXPERIENCE")
        for g, lst in sorted(groups.items()):
            c, t = sum(x for x, _ in lst), sum(y for _, y in lst)
            w(f"  {g}: {len(lst)} volunteers, {c}/{t} = {c/t:.1%}")

    w("")
    w("FAILURE CAUSES")
    causes = Counter(failure_cause(r) for r in rows)
    for cause, cnt in causes.most_common():
        w(f"  {cause}: {cnt} ({cnt/n:.1%})")

    letters = sorted({r["target_letter"] for r in rows})
    per_letter = []
    for l in letters:
        lr = [r for r in rows if r["target_letter"] == l]
        c = sum(r["correct"] for r in lr)
        top_wrong = Counter(r.get("predicted_letter") or NO_PRED for r in lr if not r["correct"]).most_common(1)
        per_letter.append((l, c, len(lr), top_wrong[0][0] if top_wrong else ""))
    per_letter.sort(key=lambda x: x[1] / x[2])
    w("")
    w("PER-LETTER ACCURACY (hardest first; most frequent wrong outcome)")
    for l, c, t, tw in per_letter:
        w(f"  {l}: {c}/{t} = {c/t:.0%}" + (f"   -> {tw}" if tw else ""))
    with open(f"{args.out_prefix}_per_letter.csv", "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["target_letter", "correct", "attempts", "accuracy", "most_frequent_wrong_outcome"])
        for l, c, t, tw in per_letter:
            cw.writerow([l, c, t, round(c / t, 4), tw])

    conf = Counter((r["target_letter"], r.get("predicted_letter")) for r in rows
                   if not r["correct"] and r.get("predicted_letter"))
    w("")
    w("MOST FREQUENT CONFUSIONS (target -> accepted wrong letter)")
    for (t, p), cnt in conf.most_common(10):
        w(f"  {t} -> {p}: {cnt}")

    with_frames = [r for r in rows if r["_frames"] is not None]
    if with_frames:
        w("")
        w(f"CONFIDENCE-THRESHOLD SENSITIVITY (recomputed from raw frames, "
          f"{len(with_frames)} attempts, stability ratio {STABILITY_RATIO})")
        w("  threshold  correct  wrong_accepted  no_letter  accuracy")
        sens = []
        for thr in [float(x) for x in args.thresholds.split(",")]:
            d = [(decide(r["_frames"], thr), r["target_letter"]) for r in with_frames]
            c = sum(p == t for p, t in d)
            wa = sum(p is not None and p != t for p, t in d)
            e = sum(p is None for p, _ in d)
            sens.append((thr, c, wa, e, c / len(d)))
            w(f"  {thr:9.2f}  {c:7d}  {wa:14d}  {e:9d}  {c/len(d):8.1%}")
        with open(f"{args.out_prefix}_threshold_sensitivity.csv", "w", encoding="utf-8-sig", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["threshold", "correct", "wrong_accepted", "no_letter", "accuracy"])
            cw.writerows([(t, c, wa, e, round(a, 4)) for t, c, wa, e, a in sens])
        raw_ok = sum(Counter(l for l, _ in r["_frames"]).most_common(1)[0][0] == r["target_letter"]
                     for r in with_frames if r["_frames"])
        w(f"  Raw frame-majority (no threshold, no stability rule): "
          f"{raw_ok}/{len(with_frames)} = {raw_ok/len(with_frames):.1%}")
        fps = [len(r["_frames"]) for r in with_frames]
        w(f"  Frames per window: mean {statistics.mean(fps):.1f}")

    report = "\n".join(out)
    with open(f"{args.out_prefix}.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    plot_confusion(rows, letters, f"{args.out_prefix}_confusion_matrix.png")
    print(report)
    print(f"\nWrote {args.out_prefix}.txt, {args.out_prefix}_confusion_matrix.png, "
          f"{args.out_prefix}_per_letter.csv"
          + (f", {args.out_prefix}_threshold_sensitivity.csv" if with_frames else ""))


if __name__ == "__main__":
    main()
