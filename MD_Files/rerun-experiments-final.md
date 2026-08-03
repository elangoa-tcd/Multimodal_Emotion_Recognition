# Re-run Experiments on the Corrected Dataset (variant C) — FINAL

All earlier experiments (baseline, pos_weight, sqrt_pos_weight, RoBERTa,
attention-pooling, threshold tuning) were run **before** the dataset faults in
`issues_fixes.md` were fixed. This doc re-runs the full set on the corrected
data so the conclusions are trustworthy. **No experiment result changes are
assumed — everything is re-measured.**

Results (console + final per-emotion test table) go to
**`~/TMBL/test_results_final/`**, one `.log` per experiment.

---

## 0. What is already in place (verified — do NOT redo)

| Item | State | Where |
|---|---|---|
| Corrected dataset (variant C) | **active** — `MOSEI_C_both` | `config.py:29` |
| Stage 1 structural fixes (binding, cls_token, share-encoder, loss weights) | **committed, always on** | git `d397349`… |
| `--ablate` flag | present | `config.py:125` |
| `AttentionPool` module | present | `network.py:262` |
| `tune_thresholds.py` | present | repo root |
| Output folder | exists | `test_results_final/` |
| Seed | fixed **55654** for every run (single-seed, per your choice) | `train.py:16` |

**Design (your choice): clean single-factor deltas.** Each experiment changes
**exactly one thing** vs the plain baseline, so every effect is attributable.

| experiment | encoder | pooling | loss |
|---|---|---|---|
| 1. baseline | BERT | mean | **plain BCE** |
| 2. pos_weight | BERT | mean | pos_weight |
| 3. sqrt_pos_weight | BERT | mean | sqrt pos_weight |
| 4. RoBERTa | **RoBERTa** | mean | plain BCE |
| 5. attention_pooling | BERT | **attention** | plain BCE |
| 6. threshold tuning | *(post-hoc on the best of 1–5 — no training)* | | |

---

## 1. The canonical BASELINE code state (return here between experiments)

Because the edits live in three files and would otherwise stack, **reset to
this exact state before starting each experiment**, then apply only that
experiment's delta.

**`config.py` (encoder = BERT):**
```python
# text_encoder_str = "roberta-large"
text_encoder_str = "bert-base-uncased"          # line 14 active
```
**`network.py` (pooling = mean):**
```python
# self.text_pool = AttentionPool(self.text_dim, attn_hidden=128, dropout=self.config.dropout)   # line 34 COMMENTED
...
# masked mean                                    # mean block ACTIVE
masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
t_feature = bert_output
# t_feature = self.text_pool(bert_output, bert_sent_mask)   # attention line COMMENTED
```
**`solver.py` (loss = plain BCE):**
```python
self.loss_task = nn.BCEWithLogitsLoss(reduction="mean")                       # line 38 ACTIVE
# self.loss_task = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=_pos_weight)   # line 47 COMMENTED
```
> The `_pos_weight` computation on lines 40–46 can stay uncommented — it just
> computes an unused tensor in baseline/RoBERTa/attention runs (harmless). Only
> the *active `loss_task` line* decides which loss is used.

> ⚠️ Current repo state is **not** baseline — right now line 47 (pos_weight) is
> active, not line 38. So the very first edit for Experiment 1 is to flip the
> loss to plain BCE.

**Naming rule (from `issues_fixes.md`):** every run gets a **unique `--name`**
so checkpoints never overwrite each other. Seed stays 55654 for all.

---

## 2. Per-experiment instructions

Prereq for every run:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"   # MUST be True
```

---

### Experiment 1 — baseline (BERT + mean + plain BCE)

**Edit — `solver.py`:** set the loss to plain BCE (flip from the current
pos_weight state).
```python
# comment line 47:
# self.loss_task = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=_pos_weight)
# uncomment line 38:
self.loss_task = nn.BCEWithLogitsLoss(reduction="mean")
```
Everything else = canonical baseline (§1).

**Run:**
```bash
nohup python3 -u train.py --name final_baseline --patience 15 --trials 5 \
  > ~/TMBL/test_results_final/baseline.log 2>&1 &
echo "PID: $!"
```

---

### Experiment 2 — pos_weight (BERT + mean + pos_weight)

**Edit — `solver.py`:** loss = pos_weight (linear `#absent/#present`).
```python
# comment line 38:
# self.loss_task = nn.BCEWithLogitsLoss(reduction="mean")
# uncomment line 47:
self.loss_task = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=_pos_weight)
```
Make sure lines 44–45 are the **linear** form (no `torch.sqrt`).

**Run:**
```bash
nohup python3 -u train.py --name final_posweight --patience 15 --trials 5 \
  > ~/TMBL/test_results_final/posweight.log 2>&1 &
echo "PID: $!"
```

---

### Experiment 3 — sqrt_pos_weight (BERT + mean + sqrt pos_weight)

Start from Experiment 2's state (pos_weight loss active), then **wrap the weight
in `torch.sqrt`** — `solver.py` lines 44–45:
```python
# OLD:
# _pos_weight = torch.tensor((_tl.shape[0] - _present) / np.maximum(_present, 1),
#                            dtype=torch.float32).cuda()
# NEW:
_pos_weight = torch.sqrt(torch.tensor((_tl.shape[0] - _present) / np.maximum(_present, 1),
                         dtype=torch.float32)).cuda()
```
Line 47 (pos_weight loss) stays active.

**Run:**
```bash
nohup python3 -u train.py --name final_sqrt --patience 15 --trials 5 \
  > ~/TMBL/test_results_final/sqrt_posweight.log 2>&1 &
echo "PID: $!"
```
**After this run, remove the `torch.sqrt` wrapper** so the repo is back to the
linear form before you reset to baseline.

---

### Experiment 4 — RoBERTa (RoBERTa + mean + plain BCE)

Reset to canonical baseline (§1) — loss must be **plain BCE** (only the encoder
changes vs baseline). Then **`config.py`**:
```python
text_encoder_str = "roberta-large"          # uncomment line 13
# text_encoder_str = "bert-base-uncased"    # comment line 14
```
That single line auto-sets `text_dim`→1024 and `text_freeze_upto`→19 (config
lines 151–153). No other edit.

**Run:**
```bash
nohup python3 -u train.py --name final_roberta --patience 15 --trials 5 \
  > ~/TMBL/test_results_final/roberta.log 2>&1 &
echo "PID: $!"
```
**After this run, flip the encoder back to `bert-base-uncased`.**

---

### Experiment 5 — attention_pooling (BERT + attention + plain BCE)

Reset to canonical baseline (§1) — encoder **BERT**, loss **plain BCE** (only
pooling changes vs baseline). Then **`network.py`**:

1. Uncomment the module (line 34):
```python
self.text_pool = AttentionPool(self.text_dim, attn_hidden=128, dropout=self.config.dropout)
```
2. In `alignment()`, comment the mean-pool block and uncomment the attention
   line:
```python
# masked mean
# masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
# mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
# bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
# t_feature = bert_output
t_feature = self.text_pool(bert_output, bert_sent_mask)   # attention pooling
```

**Run:**
```bash
nohup python3 -u train.py --name final_attnpool --patience 15 --trials 5 \
  > ~/TMBL/test_results_final/attnpool.log 2>&1 &
echo "PID: $!"
```
**After this run, reset `network.py` back to mean pooling (§1).**

---

### Experiment 6 — threshold tuning (post-hoc on the BEST config, no training)

**Step 1 — find the winner.** After Experiments 1–5 finish, read the final
`MACRO` F1 from each log:
```bash
cd ~/TMBL/test_results_final
for f in baseline posweight sqrt_posweight roberta attnpool; do
  echo -n "$f: "; grep "^MACRO" $f.log | tail -1
done
```
Pick the highest MACRO F1 — call its config **BEST** and its checkpoint
`checkpoints/devBest/model_final_<BEST>.std`.

**Step 2 — ⚠️ match the code to that checkpoint** (same rule as `issues_fixes.md`
§5.3 / the earlier threshold work): before loading, put `config.py` /
`network.py` back into the exact encoder+pooling state that produced BEST:
- if BEST = RoBERTa → `text_encoder_str = "roberta-large"`, mean pool
- if BEST = attnpool → BERT, attention pool (network.py §Exp 5 state)
- otherwise (baseline / posweight / sqrt) → BERT, mean pool
(The loss setting does **not** matter for threshold tuning — it's inference
only — so you don't need to touch `solver.py` here.)

**Step 3 — run it:**
```bash
python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_final_<BEST>.std \
  2>&1 | tee ~/TMBL/test_results_final/threshold_<BEST>.log
```
Reports the default-cut-off table, the dev-chosen per-emotion cut-offs, and the
tuned table with the `+delta` macro-F1. (Cut-offs chosen on **dev**, applied to
**test** — legitimate.)

---

## 3. Optional: structured secondary log

`solver.py:27` currently writes per-epoch metrics to
`checkpoints/log_emotion_meanpool_FIXED.txt`. If you want that stream inside the
results folder too, set it **once**:
```python
log_file = "/home/elangoa/TMBL/test_results_final/log_structured_FINAL.txt"
```
It's name-stamped per run (`Log File Time{name}…`), so a shared file across the
6 runs is fine. The per-run `.log` files above already capture the console
output + final test table, so this is optional.

---

## 4. Monitoring & completion

```bash
# is a run still going?
ps aux | grep train.py | grep -v grep
# watch live
tail -f ~/TMBL/test_results_final/<name>.log
# final table of a finished run
grep -A 9 "Per-emotion" ~/TMBL/test_results_final/<name>.log | tail -9
```
Run them **one at a time** (a single A6000) — launch the next only after
`ps` shows the previous finished, or they'll contend for GPU memory.

---

## 5. Reading the results (keep the noise floor in mind)

Build this table from the six logs (all single-seed, variant C):

| experiment | default MACRO-F1 | tuned MACRO-F1 (Exp 6, best only) |
|---|---|---|
| baseline | | — |
| pos_weight | | — |
| sqrt_pos_weight | | — |
| RoBERTa | | — |
| attention_pooling | | — |
| **best + threshold** | *(= best above)* | |

- **Noise floor ≈ 0.007** (`issues_fixes.md` §3.1). Single-seed, so **any gap
  smaller than ~0.01 is unresolved** — report it as "no difference", not a win.
- Expect (from the pre-fix runs + the ceiling analysis) the encoder/pooling
  deltas to stay within that floor, pos_weight/threshold to actually move the
  operating point. The point of this pass is to confirm those hold **on the
  corrected data**, not to assume they do.

---

## 6. Checklist

- [ ] cuda available = True
- [ ] **Exp 1** baseline — solver→plain BCE → `baseline.log`
- [ ] **Exp 2** pos_weight — solver→pos_weight (linear) → `posweight.log`
- [ ] **Exp 3** sqrt — solver→`torch.sqrt(...)` → `sqrt_posweight.log` → **remove sqrt after**
- [ ] **Exp 4** RoBERTa — reset baseline, config→roberta-large → `roberta.log` → **revert encoder after**
- [ ] **Exp 5** attnpool — reset baseline, network→attention → `attnpool.log` → **revert pooling after**
- [ ] Reset repo to canonical baseline (§1)
- [ ] **Exp 6** find BEST, match code to it, `tune_thresholds.py` → `threshold_<BEST>.log`
- [ ] Fill the §5 table; treat sub-0.01 gaps as noise
