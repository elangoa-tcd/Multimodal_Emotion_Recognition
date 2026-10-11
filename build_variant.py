#!/usr/bin/env python3
"""
Build a CMU-MOSEI feature variant into its own directory, for probing.

Why this exists as a separate script
------------------------------------
create_dataset.py has three failure modes that would silently produce a wrong
answer to the question "does removing the per-utterance z-normalisation help?":

  1. It loads cached .pkl files inside a try/except (create_dataset.py:252-258).
     If the target directory already holds a train.pkl, your edit never runs and
     you get the OLD features while believing they are new.  A confident false
     negative on the exact question that matters.

  2. A bare `except: continue` in the segment loop (:342-343) swallows a wrong
     field name for every segment, yielding an empty or partial dataset that the
     probe will happily score.

  3. In a fresh directory it calls md.mmdataset(DATASET.highlevel/.raw, ...),
     which tries to DOWNLOAD all of MOSEI from CMU rather than using your copies.

This script removes all three by construction: it refuses to overwrite, it never
downloads, and it counts every skip by reason.  create_dataset.py is not modified,
so your existing data and training pipeline are untouched.

Output format is byte-identical in structure to create_dataset.py, so the result
works with both probe_av_features.py and train.py.

Examples
--------
  # A: no per-clip z-normalisation (raw features)
  python3 build_variant.py --out ~/data/MOSEI_A_nonorm      --norm none

  # A': z-normalise using TRAIN-SPLIT statistics instead of each clip's own
  python3 build_variant.py --out ~/data/MOSEI_A_corpus      --norm corpus

  # B: FACET instead of OpenFace2, keeping the original normalisation
  python3 build_variant.py --out ~/data/MOSEI_B_facet       --norm clip \
      --visual_field CMU_MOSEI_VisualFacet42

  # C: both changes
  python3 build_variant.py --out ~/data/MOSEI_C_facet_corpus --norm corpus \
      --visual_field CMU_MOSEI_VisualFacet42

  # then, to guarantee a like-for-like comparison:
  python3 build_variant.py --out ... --common_with ~/data/MOSEI
"""

import argparse
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

EPS = 1e-6
EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

# Baseline split sizes from your original run, used only for a sanity warning.
EXPECTED = {"train": 16318, "dev": 1868, "test": 4655}


def return_unk():
    """Module-level so word2id stays picklable (a lambda would not)."""
    return 0


def sanitize(x, drops=None, key=None):
    """Replace nan AND +/-inf with 0.

    np.nan_to_num's DEFAULT maps +inf to 1.797e308 and -inf to -1.797e308 —
    only nan becomes 0. COVAREP contains genuine infinities (log-scale features
    during silence), so create_dataset.py:378-381 has been carrying 1.8e308
    values all along.

    Per-clip z-normalisation hides this: it is self-scaling, so a clip holding
    1.8e308 gets a matching mean and std and the ratio comes out order-1.
    Corpus normalisation cannot hide it — the train-split statistics overflow
    and the values survive into the output as inf.

    Counting them here turns a silent corruption into a reported number.
    """
    x = np.asarray(x, dtype=np.float64)
    if drops is not None:
        bad = int((~np.isfinite(x)).sum())
        if bad:
            drops[key] += bad
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- guards

def guard_output_dir(out_dir, force):
    """Scenario 1: never let a cached .pkl silently win over this run."""
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in ("train.pkl", "dev.pkl", "test.pkl", "embedding_and_mapping.pt")
                if os.path.exists(os.path.join(out_dir, f))]
    if existing and not force:
        sys.exit(
            f"\nREFUSING TO RUN: {out_dir} already contains {existing}.\n"
            f"If these were left by an earlier or half-finished attempt, the probe would\n"
            f"silently read the OLD features. Delete them first:\n"
            f"    rm -f {out_dir}/*.pkl {out_dir}/embedding_and_mapping.pt\n"
            f"or pass --force to overwrite deliberately.\n")
    if existing:
        print(f"  --force given: overwriting {existing}")


def guard_csd_files(src_dir, fields):
    """Scenarios 2 and 3: fail loudly and early on a wrong/missing field name."""
    missing = []
    for f in fields:
        if not os.path.exists(os.path.join(src_dir, f + ".csd")):
            missing.append(f + ".csd")
    if missing:
        try:
            present = sorted(x for x in os.listdir(src_dir) if x.endswith(".csd"))
        except OSError:
            present = ["<could not list directory>"]
        sys.exit(
            f"\nMISSING .csd FILE(S) in {src_dir}:\n"
            + "".join(f"    {m}\n" for m in missing)
            + f"\n.csd files actually present:\n"
            + "".join(f"    {p}\n" for p in present)
            + "\nCopy the exact filename (without .csd) into --visual_field.\n"
              "Nothing has been downloaded or written.\n")


# --------------------------------------------------------------------------- loading

def load_aligned_dataset(src_dir, text_field, visual_field, acoustic_field, label_field):
    """Load and word-align the .csd files. No download calls anywhere in here."""
    try:
        from mmsdk import mmdatasdk as md
    except ImportError:
        sys.exit("mmsdk not importable. Activate the venv:  source ~/tmbl_env/bin/activate")

    recipe = {f: os.path.join(src_dir, f + ".csd")
              for f in (text_field, visual_field, acoustic_field)}
    print("  loading:")
    for k, v in recipe.items():
        print(f"    {k:38s} <- {v}")
    dataset = md.mmdataset(recipe)

    def avg(intervals, features):
        try:
            return np.average(features, axis=0)
        except Exception:
            return features

    print("  aligning to words (averaging within each word interval)...")
    dataset.align(text_field, collapse_functions=[avg])

    print("  adding and aligning labels...")
    dataset.add_computational_sequences(
        {label_field: os.path.join(src_dir, label_field + ".csd")}, destination=None)
    dataset.align(label_field)
    return dataset, md


# --------------------------------------------------------------------------- extraction

def iter_segments(dataset, fields, label_field, drops):
    """Yield (segment, vid, label, words, visual, acoustic).

    Every skip is counted by reason — no bare except (Scenario 2).
    """
    text_field, visual_field, acoustic_field = fields
    pattern = re.compile(r"(.*)\[.*\]")

    for segment in dataset[label_field].keys():
        m = re.search(pattern, segment)
        if m is None:
            drops["segment_id_unparseable"] += 1
            continue
        vid = m.group(1)

        try:
            label = dataset[label_field][segment]["features"]
        except KeyError:
            drops["missing_label"] += 1
            continue
        try:
            _words = dataset[text_field][segment]["features"]
        except KeyError:
            drops["missing_text"] += 1
            continue
        try:
            _visual = dataset[visual_field][segment]["features"]
        except KeyError:
            drops["missing_visual"] += 1
            continue
        try:
            _acoustic = dataset[acoustic_field][segment]["features"]
        except KeyError:
            drops["missing_acoustic"] += 1
            continue

        if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
            drops["length_mismatch"] += 1
            continue

        yield segment, vid, sanitize(label), _words, \
            sanitize(_visual, drops, "nonfinite_values_in_visual"), \
            sanitize(_acoustic, drops, "nonfinite_values_in_acoustic")


def strip_pauses(_words, _visual, _acoustic, word2id):
    """Remove 'sp' tokens, keeping the three streams aligned."""
    actual_words, words, visual, acoustic = [], [], [], []
    for i, word in enumerate(_words):
        w = word[0]
        if w == b"sp":
            continue
        w = w.decode("utf-8") if isinstance(w, bytes) else str(w)
        actual_words.append(w)
        words.append(word2id[w])
        visual.append(_visual[i, :])
        acoustic.append(_acoustic[i, :])
    if not actual_words:
        return None
    return (np.asarray(words), np.asarray(visual),
            np.asarray(acoustic), actual_words)


# --------------------------------------------------------------------------- normalisation

def znorm_clip(x):
    """The ORIGINAL behaviour: normalise each clip against its own statistics.
    Guarantees every clip has mean 0 and std 1 -> no between-clip information
    survives in the mean or the standard deviation."""
    return np.nan_to_num((x - x.mean(0, keepdims=True))
                         / (EPS + np.std(x, axis=0, keepdims=True)))


def znorm_corpus(x, mu, sd):
    """Normalise against TRAIN-SPLIT statistics, so a clip's own level survives."""
    return np.nan_to_num((x - mu) / (EPS + sd))


class RunningStats:
    """Streaming mean/std, so corpus statistics need no extra memory."""

    def __init__(self):
        self.n = 0
        self.s = None
        self.ss = None

    def update(self, x):
        if self.s is None:
            self.s = np.zeros(x.shape[1], dtype=np.float64)
            self.ss = np.zeros(x.shape[1], dtype=np.float64)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x.astype(np.float64) ** 2).sum(0)

    def result(self):
        if self.n == 0:
            raise RuntimeError("no data seen while computing corpus statistics")
        mu = self.s / self.n
        var = np.maximum(self.ss / self.n - mu ** 2, 0.0)
        sd = np.sqrt(var)
        bad = int((~np.isfinite(mu)).sum() + (~np.isfinite(sd)).sum())
        if bad:
            raise RuntimeError(
                f"corpus statistics are non-finite in {bad} dimension(s). The input "
                f"still contains inf/nan — check that sanitize() is being applied.")
        return mu, sd


# --------------------------------------------------------------------------- reporting

def report_stats(split_data, name, which, label):
    """Print per-clip mean/std. For --norm none/corpus these MUST be off 0/1,
    otherwise the intended change did not take effect."""
    means, stds = [], []
    for s in split_data[:3000]:
        x = np.asarray(s[0][which], dtype=np.float64)
        if x.ndim == 2 and x.shape[0] > 0:
            means.append(x.mean(0))
            stds.append(x.std(0))
    if not means:
        print(f"    {label}: no clips")
        return
    means, stds = np.asarray(means), np.asarray(stds)
    n_bad = int((~np.isfinite(means)).sum() + (~np.isfinite(stds)).sum())
    if n_bad:
        # Report on the finite part rather than printing inf/nan, which tells you
        # nothing about whether the intended change took effect.
        print(f"    {label:9s} dims={means.shape[1]:5d}  "
              f"*** {n_bad} NON-FINITE values — output is CORRUPT, do not use ***")
        return
    print(f"    {label:9s} dims={means.shape[1]:5d}  "
          f"avg|per-clip mean|={np.abs(means).mean():.6f}  "
          f"between-clip spread={means.std(0).mean():.6f}  "
          f"avg per-clip std={stds.mean():.4f}")


def load_segment_ids(other_dir):
    ids = set()
    for split in ("train", "dev", "test"):
        p = os.path.join(other_dir, f"{split}.pkl")
        if not os.path.exists(p):
            sys.exit(f"--common_with: {p} not found")
        with open(p, "rb") as f:
            for s in pickle.load(f):
                ids.add(s[2])
    return ids


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MOSEI"),
                    help="directory holding the original .csd files (read-only); default <repo>/data/MOSEI")
    ap.add_argument("--out", required=True, help="directory to write this variant into")
    ap.add_argument("--norm", choices=["clip", "none", "corpus"], default="clip",
                    help="clip = original per-utterance z-norm; none = raw; "
                         "corpus = z-norm using train-split statistics")
    ap.add_argument("--visual_field", default="CMU_MOSEI_VisualOpenFace2")
    ap.add_argument("--acoustic_field", default="CMU_MOSEI_COVAREP")
    ap.add_argument("--text_field", default="CMU_MOSEI_TimestampedWords")
    ap.add_argument("--label_field", default="CMU_MOSEI_Labels")
    ap.add_argument("--common_with", default=None,
                    help="restrict to the segment IDs present in another variant "
                         "directory, so macro-F1 is computed on identical clips")
    ap.add_argument("--force", action="store_true", help="overwrite existing .pkl files")
    args = ap.parse_args()
    # allow ~ in paths even when written as --out=~/data/...
    args.src_dir = os.path.expanduser(args.src_dir)
    args.out = os.path.expanduser(args.out)
    if args.common_with:
        args.common_with = os.path.expanduser(args.common_with)

    print("=" * 78)
    print("BUILD MOSEI VARIANT")
    print("=" * 78)
    print(f"  source        : {args.src_dir}   (read-only, nothing downloaded)")
    print(f"  output        : {args.out}")
    print(f"  normalisation : {args.norm}")
    print(f"  visual field  : {args.visual_field}")

    guard_output_dir(args.out, args.force)
    fields = (args.text_field, args.visual_field, args.acoustic_field)
    guard_csd_files(args.src_dir, list(fields) + [args.label_field])

    common = load_segment_ids(args.common_with) if args.common_with else None
    if common:
        print(f"  restricting to {len(common)} segment IDs from {args.common_with}")

    print("\n" + "-" * 78)
    print("LOADING AND ALIGNING")
    print("-" * 78)
    dataset, md = load_aligned_dataset(args.src_dir, *fields, args.label_field)

    DATASET = md.cmu_mosei
    train_split = set(DATASET.standard_folds.standard_train_fold)
    dev_split = set(DATASET.standard_folds.standard_valid_fold)
    test_split = set(DATASET.standard_folds.standard_test_fold)

    # ---- pass 1: corpus statistics over the TRAIN split only ----
    v_mu = v_sd = a_mu = a_sd = None
    if args.norm == "corpus":
        print("\n" + "-" * 78)
        print("PASS 1 — train-split statistics")
        print("-" * 78)
        vs, as_ = RunningStats(), RunningStats()
        drops = Counter()
        seen = 0
        for segment, vid, _, _, _visual, _acoustic in \
                iter_segments(dataset, fields, args.label_field, drops):
            if vid not in train_split:
                continue
            if common is not None and segment not in common:
                continue
            vs.update(_visual)
            as_.update(_acoustic)
            seen += 1
        v_mu, v_sd = vs.result()
        a_mu, a_sd = as_.result()
        print(f"  statistics from {seen} train clips")
        print(f"    visual   : mean of means {v_mu.mean():+.4f}   mean of stds {v_sd.mean():.4f}")
        print(f"    acoustic : mean of means {a_mu.mean():+.4f}   mean of stds {a_sd.mean():.4f}")

    # ---- pass 2: build the splits ----
    print("\n" + "-" * 78)
    print("BUILDING SPLITS")
    print("-" * 78)
    word2id = defaultdict(lambda: len(word2id))
    _ = word2id["<unk>"]
    _ = word2id["<pad>"]

    out = {"train": [], "dev": [], "test": []}
    drops = Counter()

    for segment, vid, label, _words, _visual, _acoustic in \
            iter_segments(dataset, fields, args.label_field, drops):

        if common is not None and segment not in common:
            drops["not_in_common_subset"] += 1
            continue

        if vid in train_split:
            split = "train"
        elif vid in dev_split:
            split = "dev"
        elif vid in test_split:
            split = "test"
        else:
            drops["not_in_any_split"] += 1
            continue

        stripped = strip_pauses(_words, _visual, _acoustic, word2id)
        if stripped is None:
            drops["empty_after_pause_removal"] += 1
            continue
        words, visual, acoustic, actual_words = stripped

        if args.norm == "clip":
            visual, acoustic = znorm_clip(visual), znorm_clip(acoustic)
        elif args.norm == "corpus":
            visual = znorm_corpus(visual, v_mu, v_sd)
            acoustic = znorm_corpus(acoustic, a_mu, a_sd)
        else:  # none
            visual, acoustic = np.nan_to_num(visual), np.nan_to_num(acoustic)

        out[split].append(((words, visual, acoustic, actual_words), label, segment))

    print("  clips kept:")
    ok = True
    for s in ("train", "dev", "test"):
        n, exp = len(out[s]), EXPECTED[s]
        delta = abs(n - exp) / exp * 100 if exp else 0
        flag = "" if delta <= 2 else f"   <-- {delta:.1f}% off baseline ({exp})"
        if delta > 2:
            ok = False
        print(f"    {s:6s}: {n:6d}{flag}")

    if drops:
        print("  clips dropped, by reason:")
        for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
            print(f"    {k:28s}: {v}")

    if not any(out.values()):
        sys.exit("\nEVERY segment was dropped. Check --visual_field against the listing above.\n"
                 "Nothing was written.")

    # --- reverse-direction check -------------------------------------------------
    # --common_with drops clips this variant has that the reference lacks. It does
    # NOT handle the opposite: clips the reference has that THIS variant dropped
    # (missing_visual / length_mismatch above). So the result is a proper subset of
    # the reference, not the intersection, and the reference would still be scored
    # on extra clips. Report that gap explicitly rather than implying it is handled.
    kept_ids = {s[2] for split in out.values() for s in split}
    if common is not None:
        shortfall = len(common - kept_ids)
        if shortfall:
            print(f"\n  NOTE: {shortfall} segment IDs from --common_with are absent here")
            print("  (this variant dropped them — see the drop table above). So this")
            print("  variant is a strict SUBSET of the reference, not the intersection.")

    if not ok:
        print("\n  WARNING: split sizes differ from the baseline by more than 2%.")
        print("  A macro-F1 computed on a different set of clips is NOT comparable to")
        print("  the baseline.")

    if (not ok) or (common is not None and len(common - kept_ids)):
        print("\n  Do NOT compare this variant's macro-F1 against a differently-sized one.")
        print("  Build every variant first, then make the probe intersect them at scoring")
        print("  time — that handles both directions, and needs no rebuilds:")
        print("      python3 probe_av_features.py --data_dir <this variant> \\")
        print("          --common_with <other variant dirs ...>")
        print("  Pass the SAME directory list to every variant you probe.")

    # ---- feature statistics: did the change actually take effect? ----
    print("\n" + "-" * 78)
    print("FEATURE CHECK  (train split)")
    print("-" * 78)
    report_stats(out["train"], "visual", 1, "visual")
    report_stats(out["train"], "acoustic", 2, "acoustic")
    if args.norm == "clip":
        print("    expected: avg|per-clip mean| ~ 0 and per-clip std ~ 1 (the original bug)")
    else:
        print("    expected: avg|per-clip mean| CLEARLY ABOVE 0 and a non-zero")
        print("    between-clip spread. If it is still ~0, the change did NOT take effect.")

    # ---- write ----
    print("\n" + "-" * 78)
    print("WRITING")
    print("-" * 78)
    word2id.default_factory = return_unk          # make it picklable
    for s in ("train", "dev", "test"):
        p = os.path.join(args.out, f"{s}.pkl")
        with open(p, "wb") as f:
            pickle.dump(out[s], f)
        print(f"    {p}  ({len(out[s])} clips)")

    import torch
    rng = np.random.RandomState(0)
    emb = torch.tensor(rng.randn(len(word2id), 300)).float()
    torch.save((emb, dict(word2id)), os.path.join(args.out, "embedding_and_mapping.pt"))
    print(f"    {os.path.join(args.out, 'embedding_and_mapping.pt')}  "
          f"(vocab {len(word2id)}; random, as the GloVe path is a dummy file)")

    print("\nNext:")
    print(f"    python3 probe_av_features.py --data_dir {args.out}")
    print()


if __name__ == "__main__":
    main()
