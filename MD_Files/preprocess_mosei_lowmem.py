#!/usr/bin/env python3
"""
Low-memory MOSEI preprocessing for the TMBL repo.

WHY THIS EXISTS
---------------
`python3 train.py` triggers create_dataset.MOSEI(), which calls the CMU
MultimodalSDK's `mmdataset.align()`. The SDK's private helper
`__get_relevant_entries()` loads EVERY video's raw visual (OpenFace2, ~38 GB
uncompressed) and acoustic (COVAREP, ~14 GB) features into RAM at once before
aligning. On this 91 GB machine that peaks at ~90 GB, exhausts swap, and the
Linux OOM killer sends SIGKILL (exit code 137) with no Python traceback -- which
is exactly why training "stopped in the middle without any error".

WHAT THIS DOES
--------------
It monkey-patches ONLY that one helper with a lazy, per-video equivalent that
streams one video's features from disk at a time (peak ~6 GB instead of ~90 GB).
The replacement returns the SAME concatenated+sorted arrays the original would,
so every downstream computation (intersect / collapse / alignment / the entire
create_dataset.py loop) is byte-for-byte identical. It then calls the repo's
own MOSEI() unchanged, which writes the cache files:

    train.pkl  dev.pkl  test.pkl  embedding_and_mapping.pt

Once those exist in ~/data/MOSEI/, `python3 train.py` loads them and skips
preprocessing entirely -- so training runs with no further memory pressure.

NOTHING IN THE TMBL REPO IS MODIFIED. This script lives outside the repo and
patches the SDK in memory for this process only.
"""

import os
import sys
import time
import numpy as np

# Make the TMBL repo importable (config.py, create_dataset.py live there).
TMBL_DIR = "/home/elangoa/TMBL"
sys.path.insert(0, TMBL_DIR)

# ---------------------------------------------------------------------------
# Fix #2: create_dataset.py uses tqdm.tqdm_notebook (the Jupyter-notebook
# progress bar) in load_emb(). In a plain terminal that raises
# "ImportError: IProgress not found" and kills the run right before the .pkl
# files are saved. Point tqdm_notebook at the normal terminal tqdm BEFORE
# create_dataset imports it. (Repo equivalent: change `tqdm_notebook` -> `tqdm`
# on line 54 of create_dataset.py -- not done, to keep the repo untouched.)
# ---------------------------------------------------------------------------
import tqdm as _tqdm_mod
_tqdm_mod.tqdm_notebook = _tqdm_mod.tqdm

from mmsdk.mmdatasdk.dataset.dataset import mmdataset


# ---------------------------------------------------------------------------
# Lazy, per-video replacement for mmdataset.__get_relevant_entries
# ---------------------------------------------------------------------------
class _LazySeq:
    """Provides relevant[otherseq][reference_vid] on demand, reading only the
    keys that belong to that single reference video and caching a handful."""

    def __init__(self, parent, otherseq):
        self.parent = parent
        self.otherseq = otherseq

    def __getitem__(self, stripped_vid):
        cache = self.parent._cache
        ck = (self.otherseq, stripped_vid)
        if ck in cache:
            return cache[ck]

        data = self.parent.ds.computational_sequences[self.otherseq].data
        full_keys = self.parent.index[self.otherseq][stripped_vid]

        intervals_list = []
        features_list = []
        for fk in full_keys:
            iv = np.asarray(data[fk]["intervals"])
            fv = np.asarray(data[fk]["features"])
            if iv.ndim < 2:
                iv = iv[None, :]
                fv = fv[None, :]
            intervals_list.append(iv)
            features_list.append(fv)

        intervals = np.concatenate(intervals_list, axis=0)
        features = np.concatenate(features_list, axis=0)

        # Identical sort to the SDK: ascending by interval start time.
        order = sorted(range(intervals.shape[0]), key=lambda x: intervals[x, 0])
        intervals = intervals[order, :]
        features = features[order, :]

        entry = {"intervals": intervals, "features": features}

        # Bound memory: keep only the most recent few videos.
        cache[ck] = entry
        self.parent._order.append(ck)
        if len(self.parent._order) > 8:
            cache.pop(self.parent._order.pop(0), None)
        return entry


class _LazyRelevant:
    def __init__(self, dataset, reference):
        self.ds = dataset
        self.reference = reference
        self._cache = {}
        self._order = []
        # Build {otherseq: {stripped_vid: [full_keys]}} -- strings only, cheap.
        self.index = {}
        for name in dataset.computational_sequences.keys():
            if name == reference:
                continue
            idx = {}
            for k in dataset.computational_sequences[name].data.keys():
                idx.setdefault(k.split("[")[0], []).append(k)
            self.index[name] = idx

    def __getitem__(self, otherseq):
        return _LazySeq(self, otherseq)


def _lazy_get_relevant_entries(self, reference):
    return _LazyRelevant(self, reference)


# Name-mangled private method on the class.
mmdataset._mmdataset__get_relevant_entries = _lazy_get_relevant_entries


# ---------------------------------------------------------------------------
# Run the repo's own preprocessing, unchanged, with the patch in effect.
# ---------------------------------------------------------------------------
def main():
    from config import get_config
    import create_dataset

    config = get_config(parse=False, mode="train")
    out_dir = str(config.dataset_dir)
    print(f"[lowmem] dataset_dir = {out_dir}", flush=True)
    print(f"[lowmem] sdk_dir     = {config.sdk_dir}", flush=True)
    print(f"[lowmem] word_emb    = {config.word_emb_path}", flush=True)

    t0 = time.time()
    print("[lowmem] Building MOSEI train/dev/test (streaming alignment)...", flush=True)
    dataset = create_dataset.MOSEI(config)
    dt = time.time() - t0

    train = dataset.get_data("train")[0]
    dev = dataset.get_data("dev")[0]
    test = dataset.get_data("test")[0]
    print(f"[lowmem] DONE in {dt/60:.1f} min  train={len(train)} dev={len(dev)} test={len(test)}",
          flush=True)

    print("[lowmem] Files in", out_dir, ":", flush=True)
    for f in ["train.pkl", "dev.pkl", "test.pkl", "embedding_and_mapping.pt"]:
        p = os.path.join(out_dir, f)
        ok = os.path.exists(p)
        sz = (os.path.getsize(p) / 1e6) if ok else 0
        print(f"   {'OK ' if ok else 'MISSING'} {f}  ({sz:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
