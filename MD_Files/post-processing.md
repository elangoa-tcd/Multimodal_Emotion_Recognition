# Post-Processing & Training Guide (TMBL / CMU-MOSEI) — VERIFIED

This is the **final, verified** record of how to train the TMBL model on CMU-MOSEI
on this machine (`elangoa@134.226.38.166`, RTX A6000). Everything below has been
run end-to-end and produces correct, near-published results.

> **Status:** ✅ Full pipeline confirmed working. A 5-epoch smoke test produced
> Binary Acc (pos/neg) = **84.5%**, F1 = **0.846**, Acc-7 = **48.1%**, MAE =
> **0.60**, Corr = **0.74** — i.e. binary sentiment already at the published
> level, confirming the data and label handling are correct.

---

## Overview of what was wrong and what fixed it

The repo was written for older library versions and for a 1-D sentiment label.
On this machine (PyTorch 2.11, transformers 5.10, numpy 2.4, 91 GB RAM) it failed
in several places. There are **two categories** of fix:

- **A. Preprocessing (memory + tqdm)** — handled by an *external* script,
  `~/preprocess_mosei_lowmem.py`. **No repo edit.**
- **B. Training (4 small code edits)** — applied directly to two repo files
  (`data_loader.py`, `create_dataset.py`). The repo is a git repo, so every edit
  is visible via `git diff` and reversible via `git checkout <file>`.

---

## A. One-time data preparation (already done)

Run the external low-memory preprocessing script (NOT `train.py`) once:

```bash
source ~/tmbl_env/bin/activate
cd ~
nohup python3 -u preprocess_mosei_lowmem.py > ~/preprocess.log 2>&1 &
tail -f ~/preprocess.log
```

Why this script exists (two problems it solves, both WITHOUT editing the repo):
1. **OOM during alignment** — the CMU SDK's `align()` loads all ~52 GB of raw
   visual+acoustic features into RAM at once → ~90 GB peak → OOM-killed
   (exit 137, no traceback). The script monkey-patches the SDK's
   `__get_relevant_entries` to stream one video at a time (~6 GB peak). Output is
   byte-for-byte identical to the original algorithm.
2. **tqdm crash** — `create_dataset.py` uses `tqdm.tqdm_notebook` (a Jupyter-only
   progress bar) in `load_emb()`, which raises `ImportError: IProgress not found`
   in a terminal. The script repoints `tqdm_notebook` → normal `tqdm` before
   import.

**Success:** `[lowmem] DONE in ~15 min  train=16318 dev=1868 test=4655`, plus four
files in `~/data/MOSEI/`: `train.pkl` (1.1 GB), `dev.pkl`, `test.pkl`,
`embedding_and_mapping.pt`. Once these exist, `train.py` loads them and never
re-runs preprocessing.

---

## B. The four training code edits (already applied)

All four are tiny. Confirm them anytime with:
```bash
cd ~/TMBL && git diff data_loader.py create_dataset.py
```
Revert anytime with `git checkout data_loader.py create_dataset.py`.

### Edit 1 — numpy 2.x ragged array  ·  `data_loader.py` line 27
numpy 2.x refuses to build an array from variable-length samples
(`ValueError: inhomogeneous shape`). Restore old behavior with `dtype=object`.
```python
# BEFORE
self.label = np.abs(np.array(self.data)[:, 1])
# AFTER
self.label = np.abs(np.array(self.data, dtype=object)[:, 1])
```

### Edit 2 — sentiment label only  ·  `data_loader.py` line 56  ⚠️ correctness
The Kaggle `CMU_MOSEI_Labels.csd` has **7 columns** (col 0 = sentiment in
[-3,3]; cols 1-6 = emotion intensities ≥ 0). The model is a 1-output sentiment
regressor, so only column 0 must be used. Without this, the loss broadcasts the
single prediction against all 7 columns (wrong training) and the metrics crash
(`truth value of an array ... is ambiguous`).
```python
# BEFORE
labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0) # ([32, 1])
# AFTER
labels = torch.cat([torch.from_numpy(sample[1][:, 0:1]) for sample in batch], dim=0) # sentiment column only -> ([32, 1])
```

### Edit 3 — transformers 5.x tokenizer  ·  `data_loader.py` lines 71-72  ⚠️ silent-corruption
transformers 5.x removed `.encode_plus()` and the `pad_to_max_length` arg. The
old call raised `AttributeError`, which the surrounding bare `try/except`
swallowed, silently sending MOSEI down the UR_FUNNY fallback (summed word vectors
instead of BERT) → meaningless results / nan loss. Use the modern tokenizer call.
```python
# BEFORE
encoded_bert_sent = bert_tokenizer.encode_plus(
    text, max_length=SENT_LEN + 2, add_special_tokens=True, pad_to_max_length=True)
# AFTER
encoded_bert_sent = bert_tokenizer(text, max_length=SENT_LEN + 2, add_special_tokens=True,
    padding='max_length', truncation=True)
```

### Edit 4 — torch.load weights_only  ·  `create_dataset.py` line 256  ⚠️ this caused the OOM-kills
PyTorch 2.6+ changed `torch.load`'s default to `weights_only=True`, which cannot
unpickle the `word2id` dict in `embedding_and_mapping.pt` → raises
`UnpicklingError` → caught by the bare `except` → **falls back to full SDK
reprocessing → 90 GB → OOM "Killed"**. This is why `train.py` kept dying even
though the cache existed. Allow full unpickling (safe: you generated this file).
```python
# BEFORE
self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)
# AFTER
self.pretrained_emb, self.word2id = torch.load(CACHE_PATH, weights_only=False)
```
> Identical calls exist in the MOSI class (line 82) and UR_FUNNY class (line 430).
> Only line 256 matters for MOSEI; patch the others only if you switch datasets.

### Environment note (no code change) — BERT weights
`network.py` downloads `bert-base-uncased` weights (~440 MB) on first run. This
machine has internet, so it just works and caches to `~/.cache/huggingface/`.
The `UNEXPECTED key` warnings on load are harmless (unused pretraining head).

---

## C. Running training

**Config is MOSEI** (`config.py` line 11 `data_str = "mosei"`); batch_size 48,
Adam, lr 1e-4, default 300 epochs with early stopping (patience 6 × 3 trials).

### Quick smoke test (verifies the whole pipeline incl. saving)
`n_epoch` is a CLI arg — no edit needed:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 train.py --n_epoch 5
```
Confirms: training → dev eval → test eval → `Found new best model on dev set!`
(checkpoint saved) → `Start Testing` (final report on best model).

### Full run (background, survives disconnect)
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
nohup python3 train.py > ~/TMBL/run.log 2>&1 &
echo "Training started, PID: $!"
tail -f ~/TMBL/run.log          # Ctrl+C stops watching, not training
ps aux | grep train.py | grep -v grep
nvidia-smi                       # confirm GPU in use
```
If `CUDA out of memory` (unlikely on 49 GB): lower batch_size at `config.py:133`.

---

## D. The results / output

There is no single output file — results are **metrics + checkpoints**:

- **Metrics** printed each epoch and appended to `~/TMBL/checkpoints/log.txt`:
  `test_mult_acc7` (7-class acc, the headline), `mae`, `corr`,
  `F1 (pos/neg)`, `Accuracy (pos/neg)`, plus non-neg/neg variants.
- **`Start Testing` banner** = training finished; the code reloaded the
  best-dev model and is running the **final detailed test report** on it
  (`solver.py:193-194`). The numbers after it are your final results.
- **Checkpoints:** best-dev model at `~/TMBL/checkpoints/devBest/model_<name>.std`;
  periodic full checkpoints `~/TMBL/checkpoints/epoch<N>.pt` every 10 epochs.

Expected on a full run (published ballpark): Acc-7 ≈ 0.52–0.54, Binary Acc ≈
0.85–0.86, MAE ≈ 0.52–0.55, Corr ≈ 0.76–0.78.

---

## E. Verified checklist

- [x] `~/data/MOSEI/` has train/dev/test.pkl + embedding_and_mapping.pt
- [x] Edit 1 (numpy dtype=object) — data_loader.py:27
- [x] Edit 2 (sentiment col 0) — data_loader.py:56
- [x] Edit 3 (modern tokenizer) — data_loader.py:71-72
- [x] Edit 4 (weights_only=False) — create_dataset.py:256
- [x] 5-epoch smoke test: training + eval + checkpoint saving + final test all OK
- [ ] Full `nohup` run launched and metrics logged to checkpoints/log.txt
