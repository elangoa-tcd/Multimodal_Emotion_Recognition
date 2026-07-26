#!/usr/bin/env python3
"""
Linear probe on the CMU-MOSEI audio / visual features.

Question this answers
---------------------
Your ablation shows the model ignores audio and video. There are two possible reasons:

  (A) The features carry signal, but the fusion architecture cannot use it.
  (B) The features carry no signal by the time the model sees them.

This script decides between them WITHOUT the model. It fits plain logistic regression
directly on the pooled audio/visual features from your cached .pkl files and compares
the result against a constant predictor.

  probe clearly beats the trivial baseline -> (A), the fusion is at fault
  probe lands at the trivial baseline      -> (B), preprocessing destroyed the signal

A text probe (TF-IDF on the transcripts) is included as a CONTROL. If the text probe
also fails, the probe itself is broken and no conclusion can be drawn.

Protocol matches tune_thresholds.py: fit on train, pick per-emotion cut-offs on dev,
report test. So the macro-F1 printed here is directly comparable to your model numbers.

Usage
-----
    source ~/tmbl_env/bin/activate
    python3 probe_av_features.py
    python3 probe_av_features.py --data_dir /home/elangoa/data/MOSEI
"""

import argparse
import os
import pickle
import sys

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import f1_score
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
SPLITS = ["train", "dev", "test"]


# ----------------------------------------------------------------------------- loading

def load_split(data_dir, name):
    """Each sample is ((words, visual, acoustic, actual_words), label(1,7), segment_id)."""
    path = os.path.join(data_dir, f"{name}.pkl")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — check --data_dir")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_labels(data):
    """[N, 6] binary: emotion present if intensity > 0."""
    y = np.array([np.array(s[1]).reshape(-1)[1:7] for s in data], dtype=float)
    return (y > 0).astype(int)


def segment_ids(data_dir):
    ids = set()
    for split in SPLITS:
        path = os.path.join(data_dir, f"{split}.pkl")
        if not os.path.exists(path):
            sys.exit(f"--common_with: {path} not found")
        with open(path, "rb") as f:
            for s in pickle.load(f):
                ids.add(s[2])
    return ids


def intersect_ids(dirs):
    """Segment IDs present in EVERY listed directory.

    Restricting every variant to this set is what makes macro-F1 comparable
    across variants. Filtering in one direction only is NOT enough: if FACET
    covers fewer segments than OpenFace2, a one-way filter still leaves the
    baseline scored on extra clips the variant never sees.

    And the clips FACET drops are not a random sample — face tracking fails on
    small, turned, occluded or poorly lit faces, which is precisely where visual
    features carry least information. Comparing across different clip sets would
    measure that selection effect rather than feature quality.
    """
    sets = [segment_ids(d) for d in dirs]
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out


# ----------------------------------------------------------------------------- pooling

def pool_clip(x):
    """Collapse one clip's [T, D] sequence to a fixed-length descriptor.

    Five statistics, chosen so that at least some of them survive the
    per-utterance z-normalisation applied in create_dataset.py:378-381:

        mean    <- becomes identically 0 if the clip was z-normalised
        std     <- becomes identically 1 if the clip was z-normalised
        max     <- survives (shape of the trajectory)
        min     <- survives
        motion  <- mean absolute frame-to-frame change; survives
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if x.ndim != 2 or x.shape[0] == 0:
        return None
    d = x.shape[1]
    motion = np.abs(np.diff(x, axis=0)).mean(0) if x.shape[0] > 1 else np.zeros(d)
    return np.concatenate([x.mean(0), x.std(0), x.max(0), x.min(0), motion])


def build_matrix(data, which):
    """which: 1 = visual, 2 = acoustic. Returns [N, 5*D] plus a keep-mask."""
    rows, keep = [], []
    dim = None
    for s in data:
        v = pool_clip(s[0][which])
        if v is None:
            keep.append(False)
            continue
        if dim is None:
            dim = v.shape[0]
        if v.shape[0] != dim:
            keep.append(False)
            continue
        rows.append(v)
        keep.append(True)
    return np.asarray(rows), np.asarray(keep)


def build_text(data):
    return [" ".join(s[0][3]) for s in data]


# ------------------------------------------------------------------- feature diagnostics

def report_feature_stats(data, name, which):
    """Directly tests the per-utterance z-normalisation hypothesis (diagnosis §8).

    If create_dataset.py subtracted each clip's own mean and divided by its own
    standard deviation, then EVERY clip has per-clip mean 0 and std 1 — meaning
    'average over time' carries literally no information between clips.
    """
    means, stds = [], []
    for s in data[:3000]:
        x = np.nan_to_num(np.asarray(s[0][which], dtype=np.float64))
        if x.ndim == 2 and x.shape[0] > 0:
            means.append(x.mean(0))
            stds.append(x.std(0))
    if not means:
        print(f"  {name}: no usable clips")
        return
    means, stds = np.asarray(means), np.asarray(stds)
    print(f"  {name}  ({means.shape[1]} dims, {means.shape[0]} clips sampled)")
    print(f"    per-clip MEAN over time : avg |mean| = {np.abs(means).mean():.6f}   "
          f"spread across clips (std) = {means.std(0).mean():.6f}")
    print(f"    per-clip STD  over time : avg  std   = {stds.mean():.6f}   "
          f"spread across clips (std) = {stds.std(0).mean():.6f}")
    if np.abs(means).mean() < 1e-3 and abs(stds.mean() - 1.0) < 0.2:
        print("    >>> CONFIRMED per-utterance z-normalisation: mean~0 and std~1 for every")
        print("        clip, so mean- and std-pooling carry NO between-clip information.")


# ----------------------------------------------------------------------------- evaluation

def trivial_baseline(y_test):
    """F1 of always predicting 'present'. F1 = 2p/(1+p) where p is prevalence."""
    p = y_test.mean(0)
    return 2 * p / (1 + p)


def fit_and_score(Xtr, ytr, Xdv, ydv, Xte, yte, label, grid=200):
    """Fit one classifier per emotion, tune the cut-off on dev, report test F1."""
    f1_default, f1_tuned = [], []
    for i in range(len(EMOTIONS)):
        if ytr[:, i].sum() == 0 or ydv[:, i].sum() == 0:
            f1_default.append(0.0)
            f1_tuned.append(0.0)
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, ytr[:, i])

        p_dv = clf.predict_proba(Xdv)[:, 1]
        p_te = clf.predict_proba(Xte)[:, 1]

        f1_default.append(f1_score(yte[:, i], (p_te > 0.5).astype(int), zero_division=0))

        cands = np.percentile(p_dv, np.linspace(0.5, 99.5, grid))
        scores = [f1_score(ydv[:, i], (p_dv > c).astype(int), zero_division=0) for c in cands]
        best_c = cands[int(np.argmax(scores))]
        f1_tuned.append(f1_score(yte[:, i], (p_te > best_c).astype(int), zero_division=0))

    return label, np.asarray(f1_default), np.asarray(f1_tuned)


def print_row(label, f1s, macro_only=False):
    if macro_only:
        print(f"  {label:28s} {np.mean(f1s):7.4f}")
    else:
        cells = "".join(f"{v:8.3f}" for v in f1s)
        print(f"  {label:28s}{cells} {np.mean(f1s):9.4f}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/elangoa/data/MOSEI")
    ap.add_argument("--grid", type=int, default=200, help="cut-off candidates per emotion")
    ap.add_argument("--common_with", nargs="+", default=None, metavar="DIR",
                    help="restrict to the segment IDs present in ALL of these variant "
                         "directories (and in --data_dir). Pass the SAME list to every "
                         "variant so all of them are scored on identical clips.")
    args = ap.parse_args()

    print("=" * 78)
    print("LINEAR PROBE ON MOSEI AUDIO / VISUAL FEATURES")
    print("=" * 78)

    data = {s: load_split(args.data_dir, s) for s in SPLITS}
    for s in SPLITS:
        print(f"  {s:6s}: {len(data[s]):6d} clips")

    if args.common_with:
        dirs = [args.data_dir] + list(args.common_with)
        allowed = intersect_ids(dirs)
        print(f"\n  restricting to the {len(allowed)} segment IDs common to all of:")
        for d in dirs:
            print(f"    {d}")
        before = {s: len(data[s]) for s in SPLITS}
        for s in SPLITS:
            data[s] = [d for d in data[s] if d[2] in allowed]
        print("  after restriction:")
        for s in SPLITS:
            print(f"    {s:6s}: {len(data[s]):6d} clips  "
                  f"(dropped {before[s] - len(data[s])})")
        if not all(data[s] for s in SPLITS):
            sys.exit("\nA split is empty after restriction — the directories share no "
                     "segment IDs. Check they were built from the same source.")

    y = {s: get_labels(data[s]) for s in SPLITS}

    # ---------------------------------------------------------------- 1. feature sanity
    print("\n" + "-" * 78)
    print("1. FEATURE STATISTICS  (train split) — tests the z-normalisation hypothesis")
    print("-" * 78)
    report_feature_stats(data["train"], "visual  ", 1)
    report_feature_stats(data["train"], "acoustic", 2)

    # ---------------------------------------------------------------- 2. trivial baseline
    print("\n" + "-" * 78)
    print("2. TRIVIAL BASELINE  (always predict 'present')")
    print("-" * 78)
    triv = trivial_baseline(y["test"])
    header = f"  {'':28s}" + "".join(f"{e[:6]:>8s}" for e in EMOTIONS) + f"{'MACRO':>10s}"
    print(header)
    print_row("say-yes-to-everything", triv)
    print(f"\n  >>> Any probe at or below {np.mean(triv):.4f} macro-F1 has learned nothing.")

    # ---------------------------------------------------------------- 3. build features
    print("\n" + "-" * 78)
    print("3. BUILDING FEATURES")
    print("-" * 78)
    V, A, keep, yy = {}, {}, {}, {}
    for s in SPLITS:
        Vs, kv = build_matrix(data[s], 1)
        As, ka = build_matrix(data[s], 2)
        k = kv & ka
        # re-index: build_matrix already dropped bad rows, so realign on the mask
        Vs, _ = build_matrix([d for d, ok in zip(data[s], k) if ok], 1)
        As, _ = build_matrix([d for d, ok in zip(data[s], k) if ok], 2)
        V[s], A[s], keep[s], yy[s] = Vs, As, k, y[s][k]
        print(f"  {s:6s}: visual {Vs.shape}   acoustic {As.shape}   labels {yy[s].shape}"
              f"   (dropped {int((~k).sum())})")

    def scaled(train_mat, mats):
        sc = StandardScaler().fit(train_mat)
        return [sc.transform(m) for m in mats]

    # ---------------------------------------------------------------- 4. run the probes
    print("\n" + "-" * 78)
    print("4. PROBE RESULTS  (fit on train, cut-offs chosen on dev, scored on test)")
    print("-" * 78)
    print(header)
    print_row("say-yes-to-everything", triv)

    results = []

    Vtr, Vdv, Vte = scaled(V["train"], [V["train"], V["dev"], V["test"]])
    results.append(fit_and_score(Vtr, yy["train"], Vdv, yy["dev"], Vte, yy["test"],
                                 "video only", args.grid))

    Atr, Adv, Ate = scaled(A["train"], [A["train"], A["dev"], A["test"]])
    results.append(fit_and_score(Atr, yy["train"], Adv, yy["dev"], Ate, yy["test"],
                                 "audio only", args.grid))

    AVtr = np.hstack([Vtr, Atr]); AVdv = np.hstack([Vdv, Adv]); AVte = np.hstack([Vte, Ate])
    results.append(fit_and_score(AVtr, yy["train"], AVdv, yy["dev"], AVte, yy["test"],
                                 "audio + video", args.grid))

    # text control — if this also fails, the probe is broken, not the features
    txt = {s: [t for t, ok in zip(build_text(data[s]), keep[s]) if ok] for s in SPLITS}
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Ttr = vec.fit_transform(txt["train"])
    Tdv, Tte = vec.transform(txt["dev"]), vec.transform(txt["test"])
    results.append(fit_and_score(Ttr, yy["train"], Tdv, yy["dev"], Tte, yy["test"],
                                 "text only (TF-IDF) [control]", args.grid))

    # THE CEILING TEST: does audio+video add anything ON TOP of text?
    # This is the quantity a multimodal model is trying to capture. If text+A/V
    # scores the same as text alone, the non-verbal channels are redundant with
    # the transcript and NO fusion architecture can extract value that is not there.
    #
    # TF-IDF rows are L2-normalised; the A/V block is standardised, so its rows have
    # much larger norm across 545 dimensions and would dominate the regularisation.
    # L2-normalising the A/V block puts the two on comparable footing.
    from scipy.sparse import hstack as sp_hstack, csr_matrix
    from sklearn.preprocessing import normalize as l2norm
    TAVtr = sp_hstack([Ttr, csr_matrix(l2norm(AVtr))]).tocsr()
    TAVdv = sp_hstack([Tdv, csr_matrix(l2norm(AVdv))]).tocsr()
    TAVte = sp_hstack([Tte, csr_matrix(l2norm(AVte))]).tocsr()
    results.append(fit_and_score(TAVtr, yy["train"], TAVdv, yy["dev"], TAVte, yy["test"],
                                 "text + audio + video", args.grid))

    for label, _, f1_tuned in results:
        print_row(label, f1_tuned)

    # ---------------------------------------------------------------- 5. verdict
    print("\n" + "-" * 78)
    print("5. VERDICT")
    print("-" * 78)
    tb = float(np.mean(triv))
    av = float(np.mean(results[2][2]))
    tx = float(np.mean(results[3][2]))
    vid = float(np.mean(results[0][2]))
    aud = float(np.mean(results[1][2]))

    tav = float(np.mean(results[4][2]))

    print(f"  trivial baseline      : {tb:.4f}")
    print(f"  video only            : {vid:.4f}   ({vid - tb:+.4f} vs trivial)")
    print(f"  audio only            : {aud:.4f}   ({aud - tb:+.4f} vs trivial)")
    print(f"  audio + video         : {av:.4f}   ({av - tb:+.4f} vs trivial)")
    print(f"  text control          : {tx:.4f}   ({tx - tb:+.4f} vs trivial)")
    print(f"  text + audio + video  : {tav:.4f}   ({tav - tx:+.4f} vs TEXT ALONE)")
    print()
    print("  The last line is the one that decides whether fusion is worth pursuing:")
    if tav - tx >= 0.015:
        print(f"  => Audio/video add {tav - tx:+.4f} ON TOP of text. There IS headroom for")
        print("     a multimodal model to beat a text-only one. Fixing the fusion is")
        print("     worth doing, and this number is roughly the ceiling to aim for.")
    else:
        print(f"  => Audio/video add only {tav - tx:+.4f} on top of text — they are largely")
        print("     REDUNDANT with the transcript on this dataset. No fusion architecture")
        print("     can extract value that is not present. This is a legitimate finding:")
        print("     report the ceiling, not a fusion fix.")
    print()

    if tx - tb < 0.02:
        print("  !! The TEXT control also failed. The probe itself is not working —")
        print("     do not draw conclusions about audio/video from this run.")
    elif av - tb < 0.02:
        print("  => The audio/visual features carry essentially NO usable signal.")
        print("     The text control works, so the probe is sound.")
        print("     CONCLUSION: the bottleneck is FEATURE PREPROCESSING, not fusion.")
        print("     Next: diagnosis items 9 (per-utterance z-normalisation,")
        print("     create_dataset.py:378-381) and 10 (OpenFace2 vs FACET, :282).")
    else:
        print("  => The audio/visual features DO carry signal that a linear model can find,")
        print("     but the full model ignores them.")
        print("     CONCLUSION: the bottleneck is the FUSION ARCHITECTURE.")
        print("     Next: proceed with Stage 2, then modality dropout (item 5).")
    print()


if __name__ == "__main__":
    main()
