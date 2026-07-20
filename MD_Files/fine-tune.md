# Fine-Tuning Guide: Threshold Tuning + Modality Ablation (TMBL / CMU-MOSEI)

Two experiments in one doc, in the order I'd run them:

- **Part A — Threshold tuning.** Fixes the over-prediction problem
  (fear/surprise firing on ~40% of segments when the true rate is ~8%).
  **No retraining**, no repo edits — one standalone script against a checkpoint
  you already have. Minutes.
- **Part B — Modality ablation.** Answers "which of text / audio / video is
  actually carrying the signal?" Two flavours: a **free** test-time version
  and a **rigorous** retrain version.

> **Why this order.** Three encoder/pooling experiments all landed at
> macro-F1 0.416–0.437 — a ~2-point band. The precision columns show why:
> every model predicts "fear present" 4–6× too often, so precision is pinned
> at 0.11–0.14 and F1 can't rise no matter what the text encoder does. The
> threshold is the binding constraint. Fix that first, then ask where the
> signal lives.

---

# PART A — Threshold Tuning (no retraining)

## A.1 What it does

Your `pos_weight = #absent / #present` makes the model fire whenever it thinks
the probability exceeds that emotion's **base rate** (~8% for fear). Reading
predictions at `logit > 0` therefore flags far too many segments.

The model's *ranking* of segments is fine — only the **cut-off** is wrong.
Sliding the cut-off needs no retraining, like re-grading a finished exam with a
different pass mark. This script sweeps the cut-off **per emotion on the dev
set**, picks whichever maximises F1, then applies those cut-offs to test.

## A.2 ⚠️ Match the code state to the checkpoint

`network.py` must be in the same configuration it was trained in, or
`load_state_dict` will fail (or silently mismatch):

| Checkpoint | macro-F1 | requires `config.py` `text_encoder_str` | requires `network.py` pooling |
|---|---|---|---|
| `model_emotion_run2_posweight.std` | **0.437** (best) | `"bert-base-uncased"` | **mean** pool (attnpool commented out) |
| `model_emotion_roberta_run1.std` | 0.430 | `"roberta-large"` | **mean** pool |
| `model_emotion_attnpool_run1.std` | 0.416 | `"roberta-large"` | **attention** pool ← *your current code state* |

**Recommended:** tune thresholds on your **best** model
(`model_emotion_run2_posweight.std`) — but that means temporarily setting
`text_encoder_str = "bert-base-uncased"` and re-commenting the attention-pool
swap back to masked mean (Change 3 of attention-pooling.md, reversed). If you'd
rather not touch anything, run it on `model_emotion_attnpool_run1.std`, which
matches your code right now — the threshold gain will still show up clearly.

## A.3 The script

Create **a new file** `~/TMBL/tune_thresholds.py` (new file = no repo edits):

```python
"""Post-hoc per-emotion threshold tuning. No retraining.
Sweeps the decision cut-off on DEV to maximise positive-class F1,
then reports TEST metrics at those cut-offs vs the default (logit > 0).
Also supports test-time modality ablation via --ablate (see Part B)."""
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from config import get_config
from data_loader import get_loader
import network

EMO = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def collect(model, loader):
    """Run the model over a loader; return (logits [N,6], raw labels [N,6])."""
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            v, a, y, l, bert_sent, bert_type, bert_mask = batch
            out = model(v.cuda(), a.cuda(), l,
                        bert_sent.cuda(), bert_type.cuda(), bert_mask.cuda())
            logits.append(out.detach().cpu().numpy())
            labels.append(y.numpy())
    return (np.concatenate(logits, 0).reshape(-1, 6),
            np.concatenate(labels, 0).reshape(-1, 6))


def report(true_bin, pred_bin, title):
    print(f"\n===== {title} =====")
    print(f"{'emotion':10s} {'support':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
    f1s, recs = [], []
    for i, e in enumerate(EMO):
        t, p = true_bin[:, i], pred_bin[:, i]
        acc = accuracy_score(t, p)
        pr = precision_score(t, p, zero_division=0)
        rc = recall_score(t, p, zero_division=0)
        f1 = f1_score(t, p, zero_division=0)
        f1s.append(f1); recs.append(rc)
        print(f"{e:10s} {int(t.sum()):8d} {acc:7.3f} {pr:7.3f} {rc:7.3f} {f1:7.3f}")
    print(f"{'MACRO':10s} {'':8s} {'':7s} {'':7s} {np.mean(recs):7.3f} {np.mean(f1s):7.3f}")
    return float(np.mean(f1s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a .std model file")
    ap.add_argument("--ablate", default="none", choices=["none", "text", "audio", "video"])
    ap.add_argument("--grid", type=int, default=200, help="threshold candidates per emotion")
    args = ap.parse_args()

    dev_cfg = get_config(parse=False, mode="dev")
    test_cfg = get_config(parse=False, mode="test")
    dev_loader = get_loader(dev_cfg, shuffle=False)     # also sets visual_size/acoustic_size
    test_loader = get_loader(test_cfg, shuffle=False)

    dev_cfg.ablate = args.ablate                        # read by network.alignment() (Part B)
    model = network.AOTN(dev_cfg).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(state)
    print(f"loaded {args.checkpoint}  (ablate={args.ablate})")

    dev_logit, dev_lab = collect(model, dev_loader)
    test_logit, test_lab = collect(model, test_loader)
    dev_true, test_true = (dev_lab > 0).astype(int), (test_lab > 0).astype(int)

    # ---- baseline: the current fixed cut-off ----
    base = report(test_true, (test_logit > 0).astype(int),
                  "TEST @ default cut-off (logit > 0)")

    # ---- sweep each emotion's cut-off on DEV ----
    thresholds = []
    for i in range(6):
        cands = np.percentile(dev_logit[:, i], np.linspace(0.5, 99.5, args.grid))
        scores = [f1_score(dev_true[:, i], (dev_logit[:, i] > c).astype(int),
                           zero_division=0) for c in cands]
        thresholds.append(float(cands[int(np.argmax(scores))]))
    print("\nchosen cut-offs (from DEV):")
    for e, t in zip(EMO, thresholds):
        print(f"  {e:10s} {t:+.4f}")

    tuned = report(test_true,
                   (test_logit > np.array(thresholds)).astype(int),
                   "TEST @ tuned cut-offs (chosen on DEV)")
    print(f"\nmacro-F1: {base:.4f} (default) -> {tuned:.4f} (tuned)   delta {tuned-base:+.4f}")
```

**What it does, step by step:** builds dev+test loaders (which also set
`visual_size`/`acoustic_size` on the config); builds `AOTN` **directly** rather
than through `Solver`, so it never loads the 1.1 GB train pickle or recomputes
`pos_weight`; loads your checkpoint; collects raw logits once for dev and test;
prints the baseline table at `logit > 0`; sweeps 200 candidate cut-offs per
emotion **on dev only** (never on test — that would be cheating); then reports
test metrics at those cut-offs.

## A.4 Run it

```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_emotion_attnpool_run1.std \
  2>&1 | tee ~/TMBL/threshold_tuning.log
```
Runtime: a few minutes (two inference passes, no training).

## A.5 How to read it

You'll get three blocks: the default-cut-off table (should reproduce your
current numbers), the chosen cut-offs, and the tuned table.

- **Expect the cut-offs to come out positive**, especially for fear/surprise —
  that's the script raising the bar the model must clear, correcting the
  over-firing.
- **Expect precision up, recall down, F1 up** on the rare emotions. Recall
  dropping is *not* a regression here; it's the point. You are trading away
  false alarms you were making 4–6× too often.
- **Honest limit:** this slides you along the existing precision/recall curve.
  It cannot add discriminative power the model lacks. A solid outcome is a
  meaningful macro-F1 gain on rare classes; it will not transform a model that
  fundamentally can't rank fear segments.
- **Report the tuned cut-offs alongside results.** They were chosen on dev, so
  they're legitimate — but they are part of the model definition now, not a
  free parameter you retune per test set.

---

# PART B — Modality Ablation

## B.1 The two flavours (do B2 only if B1 looks interesting)

| | **B2 — test-time ablation** | **B3 — retrain ablation** |
|---|---|---|
| What it measures | how much the *trained* model **relies on** a modality | the modality's true **marginal contribution** |
| Cost | ~minutes ×3 (inference only) | ~hours ×3 (full training runs) |
| Caveat | zeroing an input the model expects is out-of-distribution → can **overstate** the drop | none, this is the rigorous version |

Start with B2 because it's nearly free and usually shows the pattern
immediately. Escalate to B3 for the writeup, or if B2 gives a surprising
answer.

## B.2 Code changes (2 edits)

### Change B-1 — add the flag · `config.py`, near line 120
Right after `parser.add_argument('--data', ...)`:
```python
    # modality ablation: zero out one modality's pooled features ('none' = full model)
    parser.add_argument('--ablate', type=str, default='none', choices=['none', 'text', 'audio', 'video'])
```
**What it does:** adds `--ablate` as a normal CLI argument so `train.py` can
take it, defaulting to `none` (unchanged behaviour). The tuning script above
sets `cfg.ablate` directly, so it works either way.

### Change B-2 — apply it · `network.py`, immediately BEFORE line 196
Line 196 is currently `self.shared_modaties(t_feature, video_feature, audio_feature)`.
Insert above it:
```python
        # ---- modality ablation: zero the chosen modality's pooled feature ----
        _ab = getattr(self.config, 'ablate', 'none')
        if _ab == 'text':    t_feature     = torch.zeros_like(t_feature)
        elif _ab == 'video': video_feature = torch.zeros_like(video_feature)
        elif _ab == 'audio': audio_feature = torch.zeros_like(audio_feature)
```
**What it does:** all three modalities have, by this point, been reduced to
their pooled feature vectors (`t_feature` line 185, `video_feature` line 189,
`audio_feature` line 193). Zeroing one there removes its information from
**everything** downstream — the private transformers, the shared/bound
features, and the fusion — while keeping every tensor shape identical, so
nothing crashes.

**Why zero the pooled feature rather than the raw input:** zeroing `bert_sent`
would feed real `[PAD]` token ids (not "nothing"), and zeroing `bert_sent_mask`
makes the mean-pool divide by zero and the attention softmax return `NaN`.
Zeroing at the feature level is uniform across all three modalities and safe.
`getattr(..., 'none')` means the line is harmless if `ablate` isn't set.

## B.3 Run B2 — test-time ablation (free)

Uses the Part A script, which already accepts `--ablate`:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
CKPT=checkpoints/devBest/model_emotion_attnpool_run1.std
for M in none text audio video; do
  echo "########## ABLATE = $M ##########"
  python3 tune_thresholds.py --checkpoint $CKPT --ablate $M
done 2>&1 | tee ~/TMBL/ablation_testtime.log
```
Read the **default-cut-off macro-F1** across the four runs (compare like with
like; the tuned numbers are also printed and are the fairer comparison if you
report tuned results everywhere).

## B.4 Run B3 — retrain ablation (rigorous, 3 runs)

```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
for M in text audio video; do
  nohup python3 -u train.py --ablate $M --name emotion_ablate_$M \
    --patience 15 --trials 5 > ~/TMBL/run_ablate_$M.log 2>&1
done &
echo "PID: $!"
```
Runs sequentially in one background job (don't run three at once — they'd
contend for the GPU). Your existing full-model run is the `none` baseline.
Afterwards, tune thresholds on each ablated checkpoint before comparing, so
every arm is measured at its own best operating point:
```bash
python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_emotion_ablate_text.std
```

> Note: `train.py` hardcodes `random_seed = 55654`, so all arms share a seed —
> good for comparability, but it means each arm is still a **single** sample.

## B.5 How to read the ablation

Build this table (macro-F1, all at the same cut-off policy):

| run | macro-F1 | drop vs full |
|---|---|---|
| full model | *(baseline)* | — |
| −text | | |
| −audio | | |
| −video | | |

**The biggest drop = the most important modality.**

- **My prediction:** text dominates, audio and video contribute little. If
  removing text collapses performance while removing audio/video barely moves
  it, that explains the entire encoder saga — the model is essentially a text
  classifier with decorative extra branches, and the fusion isn't extracting
  much from A/V.
- ⚠️ **The noise floor rules.** Your three text-side experiments spanned ~2
  points of macro-F1 with no real change. **A drop smaller than ~2 points is
  not evidence of anything.** Only differences clearly bigger than that band
  are interpretable from single runs.
- **Optional follow-up — leave-one-*in*.** Your plan is leave-one-*out*
  (marginal contribution). The complement is keeping only one modality (zero
  the other two), which measures standalone power. They often disagree
  informatively: text can be near-total standalone yet show a small marginal
  drop, because audio/video partly duplicate what text already carries. To do
  this you'd extend the `--ablate` choices to e.g. `only_text` and zero the
  other two — say the word and I'll add it.

---

## Checklist

**Part A — threshold tuning**
- [ ] Decide which checkpoint (and match `config.py` / `network.py` to it — §A.2)
- [ ] Create `~/TMBL/tune_thresholds.py` (new file, no repo edits)
- [ ] Run it → `threshold_tuning.log`
- [ ] Record the chosen cut-offs and the macro-F1 delta

**Part B — ablation**
- [ ] Change B-1 — `config.py:120`: add `--ablate` argument
- [ ] Change B-2 — `network.py:196`: zero the chosen pooled feature
- [ ] B2 test-time sweep (4 runs, minutes) → `ablation_testtime.log`
- [ ] B3 retrain runs only if B2 warrants it (3 runs, hours)
- [ ] Compare drops against the ~2-point noise floor before concluding
