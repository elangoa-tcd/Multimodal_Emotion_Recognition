#!/usr/bin/env python3
"""
Analyse the preprocessed CMU-MOSEI label distribution.

Reads the cached train/dev/test .pkl files and reports:
  - number of data points per split and overall
  - sentiment distribution (7-point scale + pos/neg/neutral)
  - per-emotion presence counts and intensity stats
  - emotion co-occurrence (how many samples have 0/1/2/... emotions)
  - most common emotion combinations
  - emotion-vs-emotion co-occurrence matrix

Each label is a (1, 7) vector:
  col 0 = sentiment in [-3, +3]
  cols 1-6 = emotion intensities in [0, 3]  -> [happy, sad, anger, surprise, disgust, fear]
An emotion counts as "present" when its intensity > 0 (standard MOSEI rule).

Results are printed to stdout AND saved to an Excel workbook (one sheet per report).

Usage:
    source ~/tmbl_env/bin/activate
    python3 ~/analyze_mosei_data.py
"""

import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd

DATA_DIR = "/home/elangoa/data/MOSEI"
SPLITS = ["train", "dev", "test"]
EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
OUTPUT_XLSX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mosei_analysis_results.xlsx")


def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    # each sample: ((words, visual, acoustic, text), label(1,7), segment_id)
    labels = np.array([np.array(s[1]).reshape(-1) for s in data])  # [N, 7]
    return labels


def sentiment_report(sent):
    print("\n--- SENTIMENT (column 0, range [-3, +3]) ---")
    print(f"  mean: {sent.mean():.3f}   std: {sent.std():.3f}   "
          f"min: {sent.min():.2f}   max: {sent.max():.2f}")
    neg = (sent < 0).sum()
    neu = (sent == 0).sum()
    pos = (sent > 0).sum()
    n = len(sent)
    print(f"  negative (<0) : {neg:6d}  ({neg/n*100:5.1f}%)")
    print(f"  neutral  (=0) : {neu:6d}  ({neu/n*100:5.1f}%)")
    print(f"  positive (>0) : {pos:6d}  ({pos/n*100:5.1f}%)")
    print("  7-point distribution (rounded sentiment score):")

    summary_df = pd.DataFrame([{
        "mean": sent.mean(), "std": sent.std(), "min": sent.min(), "max": sent.max(),
        "negative_count": neg, "negative_pct": neg / n * 100,
        "neutral_count": neu, "neutral_pct": neu / n * 100,
        "positive_count": pos, "positive_pct": pos / n * 100,
    }])

    rounded = np.clip(np.round(sent), -3, 3).astype(int)
    dist_rows = []
    for k in range(-3, 4):
        c = (rounded == k).sum()
        bar = "#" * int(c / n * 100)
        print(f"    {k:+d} : {c:6d}  ({c/n*100:5.1f}%) {bar}")
        dist_rows.append({"score": k, "count": c, "pct": c / n * 100})
    dist_df = pd.DataFrame(dist_rows)

    return summary_df, dist_df


def emotion_report(emo):
    print("\n--- EMOTIONS (columns 1-6, intensity [0, 3], present if > 0) ---")
    n = len(emo)
    present = emo > 0
    print(f"  {'emotion':10s} {'present':>8s} {'pct':>7s} "
          f"{'mean_int':>9s} {'max_int':>8s}  (mean over PRESENT rows)")
    rows = []
    for i, name in enumerate(EMOTIONS):
        p = present[:, i].sum()
        col = emo[:, i]
        mean_present = col[col > 0].mean() if p > 0 else 0.0
        print(f"  {name:10s} {p:8d} {p/n*100:6.1f}% "
              f"{col.mean():9.3f} {col.max():8.2f}  {mean_present:6.3f}")
        rows.append({
            "emotion": name, "present_count": p, "present_pct": p / n * 100,
            "mean_intensity_all": col.mean(), "max_intensity": col.max(),
            "mean_intensity_present_only": mean_present,
        })
    return pd.DataFrame(rows)


def cooccurrence_report(emo):
    print("\n--- EMOTION CO-OCCURRENCE (how many emotions per sample) ---")
    n = len(emo)
    counts = (emo > 0).sum(axis=1)  # number of emotions present per row
    print(f"  average emotions present per sample: {counts.mean():.2f}")
    rows = []
    for k in range(0, 7):
        c = (counts == k).sum()
        bar = "#" * int(c / n * 100)
        print(f"    {k} emotion(s) : {c:6d}  ({c/n*100:5.1f}%) {bar}")
        rows.append({"num_emotions": k, "count": c, "pct": c / n * 100})
    mixed = (counts >= 2).sum()
    print(f"  samples with a MIX (2+ emotions): {mixed}  ({mixed/n*100:.1f}%)")
    pure = (counts == 1).sum()
    print(f"  samples with exactly ONE emotion: {pure}  ({pure/n*100:.1f}%)")
    none = (counts == 0).sum()
    print(f"  samples with NO emotion at all  : {none}  ({none/n*100:.1f}%)")

    dist_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame([{
        "avg_emotions_per_sample": counts.mean(),
        "mixed_count": mixed, "mixed_pct": mixed / n * 100,
        "pure_count": pure, "pure_pct": pure / n * 100,
        "none_count": none, "none_pct": none / n * 100,
    }])
    return dist_df, summary_df


def top_combinations(emo, topn=10):
    print(f"\n--- TOP {topn} EMOTION COMBINATIONS ---")
    n = len(emo)
    present = emo > 0
    combos = Counter()
    for row in present:
        names = tuple(EMOTIONS[i] for i in range(len(EMOTIONS)) if row[i])
        combos[names if names else ("(none)",)] += 1
    rows = []
    for combo, c in combos.most_common(topn):
        label = " + ".join(combo)
        print(f"  {c:6d}  ({c/n*100:5.1f}%)  {label}")
        rows.append({"combination": label, "count": c, "pct": c / n * 100})
    return pd.DataFrame(rows)


def cooccurrence_matrix(emo):
    print("\n--- EMOTION PAIR CO-OCCURRENCE (count of samples where BOTH present) ---")
    present = (emo > 0).astype(int)
    mat = present.T @ present  # [6, 6]
    header = "            " + "".join(f"{e[:5]:>8s}" for e in EMOTIONS)
    print(header)
    for i, name in enumerate(EMOTIONS):
        row = "".join(f"{mat[i, j]:8d}" for j in range(len(EMOTIONS)))
        print(f"  {name:10s}{row}")
    return pd.DataFrame(mat, index=EMOTIONS, columns=EMOTIONS)


def main():
    all_labels = []
    print("=" * 70)
    print("CMU-MOSEI LABEL ANALYSIS")
    print("=" * 70)

    print("\n--- DATASET SIZE ---")
    size_rows = []
    for name in SPLITS:
        try:
            lab = load_split(name)
        except FileNotFoundError:
            print(f"  {name}: MISSING ({DATA_DIR}/{name}.pkl not found)")
            size_rows.append({"split": name, "samples": None, "status": "MISSING"})
            continue
        all_labels.append(lab)
        print(f"  {name:6s}: {len(lab):6d} samples")
        size_rows.append({"split": name, "samples": len(lab), "status": "OK"})

    if not all_labels:
        print("No data found. Check DATA_DIR.")
        return

    labels = np.concatenate(all_labels, axis=0)
    print(f"  {'TOTAL':6s}: {len(labels):6d} samples")
    size_rows.append({"split": "TOTAL", "samples": len(labels), "status": "OK"})
    size_df = pd.DataFrame(size_rows)

    sent = labels[:, 0]
    emo = labels[:, 1:7]

    # Overall (all splits combined)
    print("\n" + "=" * 70)
    print("OVERALL (train + dev + test combined)")
    print("=" * 70)
    sentiment_summary_df, sentiment_dist_df = sentiment_report(sent)
    emotion_df = emotion_report(emo)
    cooc_dist_df, cooc_summary_df = cooccurrence_report(emo)
    top_combos_df = top_combinations(emo)
    cooc_matrix_df = cooccurrence_matrix(emo)

    # Per-split emotion presence (quick view, for spotting train/test skew)
    print("\n" + "=" * 70)
    print("PER-SPLIT emotion presence rate (%)")
    print("=" * 70)
    print(f"  {'split':6s} " + "".join(f"{e[:5]:>8s}" for e in EMOTIONS))
    per_split_rows = []
    for name, lab in zip(SPLITS, all_labels):
        e = lab[:, 1:7]
        rates = (e > 0).mean(0) * 100
        print(f"  {name:6s} " + "".join(f"{r:7.1f}%" for r in rates))
        per_split_rows.append({"split": name, **{EMOTIONS[i]: rates[i] for i in range(len(EMOTIONS))}})
    per_split_df = pd.DataFrame(per_split_rows)

    # --- Save everything to Excel ---
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        size_df.to_excel(writer, sheet_name="Dataset Size", index=False)
        sentiment_summary_df.to_excel(writer, sheet_name="Sentiment Summary", index=False)
        sentiment_dist_df.to_excel(writer, sheet_name="Sentiment Distribution", index=False)
        emotion_df.to_excel(writer, sheet_name="Emotion Presence", index=False)
        cooc_dist_df.to_excel(writer, sheet_name="Co-occurrence Counts", index=False)
        cooc_summary_df.to_excel(writer, sheet_name="Co-occurrence Summary", index=False)
        top_combos_df.to_excel(writer, sheet_name="Top Combinations", index=False)
        cooc_matrix_df.to_excel(writer, sheet_name="Co-occurrence Matrix")
        per_split_df.to_excel(writer, sheet_name="Per-Split Emotion Rates", index=False)

    print(f"\nResults saved to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
