# Emotion Prediction Guide (TMBL / CMU-MOSEI)

Convert the model from **sentiment regression** (1 output) to **6-emotion
multi-label classification** (6 outputs), *without* disturbing the working
sentiment pipeline. Every change is "comment out the old line, add the new line"
so both versions stay visible and you can revert instantly.

> **Read this first — you do NOT need to re-run preprocessing.**
> Your `.pkl` files (`train.pkl`, `dev.pkl`, `test.pkl`) already contain the full
> **(1, 7)** label for every segment. The current code just *slices away* columns
> 1–6. So none of the data-preparation steps change — you only change how the
> label is read, the model head width, the loss, and the metrics.

---

## 1. Design decisions (what we are building and why)

| Decision | Choice | Reason |
|---|---|---|
| Task | **6 emotions only** (drop sentiment) | You asked for emotion prediction; keeps scope tight |
| Emotion order (cols 1–6) | happy, sad, anger, surprise, disgust, fear | From the `.csd` metadata `dimension names` |
| Framing | **Multi-label binary** (each emotion present/absent) | Matches how MOSEI was annotated (a row can have several emotions) |
| "Present" threshold | intensity **> 0** | Standard MOSEI benchmark; > 0 means ≥ 1 annotator saw it |
| Loss | **BCEWithLogitsLoss** | Optimises present/absent directly; handles the sparsity |
| Metrics | **per-emotion Accuracy + weighted-F1** | Weighted-F1 is honest under class imbalance (fear ≈ 9%) |

**Label mapping reminder** — for a raw row `[-0.667, 0.333, 0.667, 0, 0, 0.333, 0.333]`:
col 0 (sentiment, `-0.667`) is **dropped**; cols 1–6 become the binary target
`[happy=1, sad=1, anger=0, surprise=0, disgust=1, fear=1]`.

---

## 2. The code changes (6 edits across 4 files)

Make every change as: **comment out the old line(s), then add the new line(s)
right below.** File paths are all under `~/TMBL/`.

---

### Change 1 — read the emotion columns instead of sentiment
**File: `data_loader.py`, line 58** (inside `collate_fn`)

**OLD:**
```python
labels = torch.cat([torch.from_numpy(sample[1][:, 0:1]) for sample in batch], dim=0) # sentiment column only -> ([32, 1])
```
**NEW:**
```python
# labels = torch.cat([torch.from_numpy(sample[1][:, 0:1]) for sample in batch], dim=0) # sentiment column only -> ([32, 1])
labels = torch.cat([torch.from_numpy(sample[1][:, 1:7]) for sample in batch], dim=0) # 6 emotion columns -> ([32, 6])
```
**What the old code did:** `sample[1]` is the full `(1, 7)` label. `[:, 0:1]`
kept **only column 0 (sentiment)** and threw away the 6 emotions, producing a
`[batch, 1]` target.
**What the new code does:** `[:, 1:7]` keeps **columns 1–6 (the 6 emotions)** and
drops sentiment, producing a `[batch, 6]` target. The values are still the raw
intensities in `[0, 3]` — they get binarized later, inside the loss and metrics,
not here (so the intensity is still available if you ever want MAE too).

---

### Change 2 — tell the model to produce 6 outputs
**File: `config.py`, line 132** (inside the `elif kwargs.data == "mosei":` block)

**OLD:**
```python
        kwargs.num_classes = 1
```
**NEW:**
```python
        # kwargs.num_classes = 1
        kwargs.num_classes = 6   # 6 Ekman emotions: happy, sad, anger, surprise, disgust, fear
```
**What the old code did:** told the network to build a **single-output** head
(one sentiment score).
**What the new code does:** sets the head width to **6**, one logit per emotion.
This value flows into `network.py` (Change 3) to size the final layer.

---

### Change 3 — widen the model's output head from 1 to 6
**File: `network.py`, line 90** (inside the `self.fusion3 = nn.Sequential(...)` block)

**OLD:**
```python
            nn.Conv1d(self.config.hidden_size//2, 1, kernel_size=3, stride=1, padding=1),
```
**NEW:**
```python
            # nn.Conv1d(self.config.hidden_size//2, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(self.config.hidden_size//2, self.config.num_classes, kernel_size=3, stride=1, padding=1),
```
**What the old code did:** the final fusion convolution output **1 channel**, so
after `AdaptiveAvgPool1d(1)` the model emitted a single scalar `[batch, 1]`
(sentiment).
**What the new code does:** outputs **`num_classes` (= 6) channels**, so the model
emits `[batch, 6]` — one raw logit per emotion. No sigmoid here on purpose;
`BCEWithLogitsLoss` (Change 4) applies the sigmoid internally, and at eval a logit
`> 0` is exactly `sigmoid(logit) > 0.5` = "predicted present".
*(Nothing else in `network.py` needs changing — `second_o7` at line 206 already
does `.view(all_modal.size(0), -1)`, which reshapes `[batch, 6, 1]` → `[batch, 6]`.)*

---

### Change 4 — switch the task loss to BCE
**File: `solver.py`, line 33** (in `Solver.__init__`)

**OLD:**
```python
        self.loss_task = nn.MSELoss(reduction="mean") # MOSEI和MOSI情感分析属于回归任务,使用MSE损失
```
**NEW:**
```python
        # self.loss_task = nn.MSELoss(reduction="mean") # (sentiment regression)
        self.loss_task = nn.BCEWithLogitsLoss(reduction="mean")  # emotion multi-label classification
```
**What the old code did:** mean-squared-error loss for regressing the single
sentiment value.
**What the new code does:** binary cross-entropy (with built-in sigmoid) — the
correct loss for present/absent classification across the 6 emotions.

---

### Change 5 — binarize the target inside the loss
**File: `solver.py`, line 48** (inside `Solver.loss_function`)

**OLD:**
```python
        loss_task = self.loss_task(y_tilde, y)
```
**NEW:**
```python
        # loss_task = self.loss_task(y_tilde, y)   # sentiment: raw MSE target
        loss_task = self.loss_task(y_tilde, (y > 0).float())  # binarize emotion intensities -> present/absent
```
**What the old code did:** compared the prediction against the raw target directly
(fine for MSE on a continuous sentiment value).
**What the new code does:** converts the `[0, 3]` intensities into `{0, 1}`
present/absent labels (`y > 0`) before the BCE loss — this is the "threshold > 0"
rule turned into the training target. *(The two representation/similarity
regularizers `0.5*loss_diff + 0.5*loss_simi` on line 50 are architectural and stay
unchanged.)*

---

### Change 6a — send emotion metrics to a SEPARATE log file
**File: `solver.py`, line 22** (module level, near the top)

**OLD:**
```python
log_file = "/home/elangoa/TMBL/checkpoints/log.txt"
```
**NEW:**
```python
# log_file = "/home/elangoa/TMBL/checkpoints/log.txt"          # sentiment run log
log_file = "/home/elangoa/TMBL/checkpoints/log_emotion.txt"    # emotion run log (separate, preserves sentiment log.txt)
```
**What this does:** keeps your original sentiment `log.txt` untouched; all
per-epoch emotion metrics go into a brand-new `log_emotion.txt`.

---

### Change 6b — replace the metrics function with a per-emotion version
**File: `solver.py`** — the entire `calc_metrics` method (**lines 240–341**).

**Comment out the whole existing `calc_metrics` method** (the sentiment version:
Acc-7 / MAE / Corr / pos-neg), and paste this new one in its place. It keeps the
same signature and return type, so nothing that *calls* it needs to change.

**NEW method:**
```python
    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Per-emotion DETECTION metrics for the 6 Ekman emotions.
        y_true : raw intensities in [0,3], shape [N,6]
        y_pred : raw model logits,          shape [N,6]
        present  <=>  true intensity > 0  /  logit > 0 (== sigmoid > 0.5).

        Reports, per emotion, metrics on the POSITIVE ("present") class:
          Precision, Recall, F1  -- these expose the "always predict absent"
          shortcut that Accuracy hides on rare emotions (fear, surprise, ...).
          'support' = how many test clips are truly present for that emotion.
        The headline number is MACRO-F1 (every emotion weighted equally), not
        accuracy.
        """
        from sklearn.metrics import (precision_score, recall_score,
                                     f1_score, accuracy_score)
        emotions = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

        y_true = np.array(y_true).reshape(-1, len(emotions))
        y_pred = np.array(y_pred).reshape(-1, len(emotions))
        true_bin = (y_true > 0).astype(int)
        pred_bin = (y_pred > 0).astype(int)

        rows = []
        for i, emo in enumerate(emotions):
            t, p = true_bin[:, i], pred_bin[:, i]
            rows.append((
                emo,
                int(t.sum()),                                        # support (present count)
                accuracy_score(t, p),
                precision_score(t, p, pos_label=1, zero_division=0), # present-class precision
                recall_score(t, p, pos_label=1, zero_division=0),    # present-class recall
                f1_score(t, p, pos_label=1, zero_division=0),        # present-class F1
            ))
        macro_f1 = float(np.mean([r[5] for r in rows]))
        macro_rec = float(np.mean([r[4] for r in rows]))

        def emit(write):
            write(f"{'emotion':10s} {'support':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
            for emo, sup, acc, prec, rec, f1 in rows:
                write(f"{emo:10s} {sup:8d} {acc:7.3f} {prec:7.3f} {rec:7.3f} {f1:7.3f}")
            write(f"{'MACRO':10s} {'':>8s} {'':>7s} {'':>7s} {macro_rec:7.3f} {macro_f1:7.3f}")

        if mode == "dev":
            print(f"[dev] macro-F1: {macro_f1:.4f}  macro-Recall: {macro_rec:.4f}")
            with open(log_file, "a") as file:
                file.write(f"[dev] macro-F1: {macro_f1:.4f}  macro-Recall: {macro_rec:.4f}\n")

        if mode == "test" or to_print:
            title = "\n===== Per-emotion detection metrics (positive / 'present' class) ====="
            print(title)
            emit(print)
            with open(log_file, "a") as file:
                file.write(title + "\n")
                emit(lambda s: file.write(s + "\n"))

        return macro_f1
```
**What the old code did:** computed sentiment-only metrics (7-class accuracy, MAE,
correlation, and pos/neg binary accuracy/F1) on a single predicted scalar. It
crashed for anything that wasn't 1-D.
**What the new code does:** treats predictions and truth as `[N, 6]`, binarizes
both at `> 0`, and for **each emotion** reports **Precision, Recall, and F1 on the
"present" class** plus its **support** and (for reference) Accuracy. It also prints
**MACRO-F1 / MACRO-Recall** (the mean across the 6 emotions, every emotion counted
equally). Precision/Recall/F1 on the positive class are **not fooled by the
"always predict absent" shortcut** the way Accuracy is — a model that never detects
fear gets Recall = 0 and F1 = 0 for fear, no matter how high its Accuracy looks.
The returned `macro_f1` is only used for the per-epoch "val acc" log line; early
stopping still uses the validation **loss** (now BCE), so model selection keeps
working.

> Why not `average='weighted'`? Weighted-F1 is *also* dominated by the common
> "absent" class on rare emotions, so it looks deceptively high even when the model
> detects nothing. Positive-class **Recall** and **macro-F1** are the honest signals.

> The helper methods `multiclass_acc` (line 231) and the old sentiment code become
> unused after this — harmless to leave commented out.

---

## 3. What to expect in the output

### Where the outputs go (all separate from your sentiment run)

| Output | Sentiment run (existing) | Emotion run (new) |
|---|---|---|
| Structured metrics log | `checkpoints/log.txt` | `checkpoints/log_emotion.txt` (Change 6a) |
| Full console output | `run.log` / `run2.log` | `run_emotion.log` (set at launch, see §4) |
| Best model checkpoint | `checkpoints/devBest/model_2026-06-15_23:31:24.std` | `checkpoints/devBest/model_emotion_run1.std` (via `--name`) |

Your sentiment results are **not overwritten** — different log file, different
console file, different model name.

> Minor note: the periodic `checkpoints/epochN.pt` files (saved every 10 epochs)
> reuse fixed names and *will* be overwritten by the emotion run. They are
> throwaway resume-checkpoints, not results — ignore unless you specifically want
> the old ones, in which case move `checkpoints/epoch*.pt` aside first.

### What the final result looks like

After training finishes, at the end of `run_emotion.log` (and in
`log_emotion.txt`) you'll see a per-emotion table like:

```
===== Per-emotion detection metrics (positive / 'present' class) =====
emotion     support     Acc    Prec  Recall      F1
happy          2504   0.6xx   0.6xx   0.7xx   0.6xx
sad            1287   0.7xx   0.5xx   0.5xx   0.5xx
anger           838   0.7xx   0.5xx   0.4xx   0.4xx
surprise        507   0.8xx   0.4xx   0.2xx   0.3xx
disgust         698   0.8xx   0.5xx   0.3xx   0.4xx
fear            437   0.9xx   0.3xx   0.1xx   0.2xx
MACRO                                  0.3xx   0.4xx
```

### How to read it (important)

- **Recall is the metric that exposes lazy predictions.** For a rare emotion like
  fear (~9% present), a model that just says "absent" for everyone scores ~0.91
  **Accuracy** but **Recall ≈ 0** — it caught none of the real fear clips. Always
  read Recall (and F1) for the rare emotions; ignore their Accuracy.
- **Precision vs Recall tells you the failure type.** Low Recall = the model misses
  the emotion (too cautious). Low Precision = the model cries wolf (false alarms).
- **`support`** tells you how rare each emotion is — the smaller it is, the more you
  should distrust Accuracy and trust Recall/F1.
- **MACRO-F1 is your single headline number.** It averages the 6 per-emotion F1s
  equally, so a model that ignores the rare emotions can't hide behind the common
  ones. Expect happy/sad to carry it; surprise/fear will drag it down.
- **First-run expectation:** it is normal for the rare emotions to start with low
  Recall — the model defaults to "absent" because that minimises loss. You raise it
  later with a `pos_weight` in `BCEWithLogitsLoss` (weights the rare "present" class
  higher) or a longer run. A macro-F1 around **0.4–0.55** is a reasonable first
  result; happy/sad F1 in the 0.6s is the meaningful signal early on.

---

## 4. How to run the training

**Do NOT re-run preprocessing.** Go straight to training.

### Step A — (recommended) 5-epoch smoke test first
Confirms all 6 edits work together (loader → 6-output model → BCE loss →
per-emotion metrics → saving) before committing to a long run:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 train.py --name emotion_smoke --n_epoch 5
```
Success = it trains, prints `[dev] mean emotion Acc/F1` each epoch, and ends with
the per-emotion test table. If you see a shape error, re-check Changes 1–3
(they must all agree on 6).

### Step B — full run (background, survives disconnect)
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
nohup python3 -u train.py --name emotion_run1 --patience 15 --trials 5 > ~/TMBL/run_emotion.log 2>&1 &
echo "PID: $!"
```
- `--name emotion_run1` → best model saved as `model_emotion_run1.std` (won't clash
  with the sentiment checkpoint).
- `--patience 15 --trials 5` → same generous early-stopping budget we used for the
  improved sentiment run (drop these to use the defaults 6 / 3 for a quicker run).
- `-u` → unbuffered, so you can `tail -f ~/TMBL/run_emotion.log` and watch epochs
  live. (You can also `tail -f ~/TMBL/checkpoints/log_emotion.txt`.)

### Step C — verify it finished cleanly (after reconnecting)
```bash
ps aux | grep train.py | grep -v grep                  # empty = finished
grep "early stopping" ~/TMBL/run_emotion.log            # present = stopped normally
grep -A 8 "Per-emotion detection metrics" ~/TMBL/run_emotion.log | tail -8   # final table
```

---

## 5. Reverting to the sentiment model

Every edit is a comment-out + add. To go back to sentiment, just swap the
comments back (or, if `~/TMBL` is a git repo):
```bash
cd ~/TMBL && git checkout data_loader.py config.py network.py solver.py
```

---

## 6. Change checklist

- [ ] Change 1 — `data_loader.py:58` slice `[:, 1:7]`
- [ ] Change 2 — `config.py:132` `num_classes = 6`
- [ ] Change 3 — `network.py:90` Conv1d out-channels → `self.config.num_classes`
- [ ] Change 4 — `solver.py:33` `BCEWithLogitsLoss`
- [ ] Change 5 — `solver.py:48` binarize target `(y > 0).float()`
- [ ] Change 6a — `solver.py:22` log file → `log_emotion.txt`
- [ ] Change 6b — `solver.py:240–341` replace `calc_metrics` with per-emotion version
- [ ] Smoke test (`--n_epoch 5`) passes
- [ ] Full run launched → results in `run_emotion.log` / `log_emotion.txt`

---

## 7. Troubleshooting: training runs on CPU / "stuck" / GPU disappears

**Symptom:** Training launches, prints the config and `Start training from epoch -1`,
then appears **frozen** with no epoch output. It may run for hours without finishing
one epoch. Historically caused by an **automatic kernel upgrade + reboot** that left
the NVIDIA driver unable to load.

### How to recognise it (run these — no sudo needed)
```bash
nvidia-smi                                                     # "couldn't communicate with the NVIDIA driver" = GPU offline
python3 -c "import torch; print(torch.cuda.is_available())"    # prints False = training will run on CPU
ps aux | grep train.py | grep -v grep                          # CPU% in the 1000s = it's grinding on CPU, not GPU
uptime                                                          # small "up" time = machine recently rebooted
uname -r                                                        # the kernel you're now running
dpkg -l | grep linux-modules-nvidia                            # which kernels HAVE an nvidia module built
```
If `nvidia-smi` fails **and** `torch.cuda.is_available()` is `False`, the GPU is
offline and the model is running on CPU. It is **not** stuck — just crawling.

### Why it happens
The NVIDIA driver includes a **kernel module** compiled for one exact kernel
version. When Ubuntu auto-installs a newer kernel and reboots, if the matching
`linux-modules-nvidia-<ver>-<kernel>` package isn't present for the **new** kernel,
the module can't load → GPU vanishes → CUDA unavailable → PyTorch silently uses CPU.
Compare `uname -r` (running kernel) against the kernels listed by the `dpkg`
command above — if the running kernel isn't in that list, that's the mismatch.

### How to fix it (needs sudo)
**1. Kill the CPU-bound job first** (find its PID with the `ps` command above):
```bash
kill <PID>
```
**2a. Proper fix — install the module for the current kernel and load it:**
```bash
sudo apt update
sudo apt install linux-modules-nvidia-590-open-$(uname -r)
sudo modprobe nvidia
nvidia-smi                       # should now list the GPU
```
**2b. Guaranteed fix — if the package for the new kernel isn't available yet,
reboot into an older kernel that already has a working module** (e.g. `6.17.0-22`):
```bash
sudo reboot
# at GRUB: Advanced options for Ubuntu -> pick the kernel shown by the dpkg command
```
> `590-open` is this machine's driver series; adjust the number if `dpkg -l | grep
> nvidia-driver` shows a different one.

### Verify before rerunning
```bash
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"   # must print True
```
Only start training once this prints `True`. On the GPU an epoch takes minutes; on
CPU it takes hours — so if the first epoch hasn't printed within ~10 min, re-check
the GPU with the recognition commands above.

### Preventing recurrence
Install the "hwe" metapackage so the NVIDIA module is pulled in automatically for
each new HWE kernel, reducing the chance of this happening on the next upgrade:
```bash
sudo apt install linux-modules-nvidia-590-open-generic-hwe-24.04
```

---

## 8. Run B — pos_weight experiment (rescuing rare emotions)

**This is a second, separate experiment run *after* the baseline (Run A).** Keep
Run A's results — the whole point is the A-vs-B comparison for the writeup.

### The problem it fixes
The baseline leaves the two rarest emotions with **recall = 0.00**:

```
Baseline (Run A) final test  ->  MACRO-F1 = 0.273, MACRO-Recall = 0.296
  surprise (support 441): recall 0.000     fear (support 384): recall 0.000
```
The model ignores them because predicting "absent" for a ~9%-present emotion already
scores ~91% accuracy and low loss. (Note: disgust, despite being uncommon, works fine
at recall 0.573 — it is specifically the *two rarest* emotions that collapse.)

### The single change (solver.py `__init__`, the Change 4 line)
Replace the plain BCE loss with a class-weighted one:

**OLD:**
```python
self.loss_task = nn.BCEWithLogitsLoss(reduction="mean")
```
**NEW:**
```python
# pos_weight[i] = (#absent / #present) for emotion i, computed from the TRAIN set.
# NOTE: use the `train_data_loader` PARAMETER, not self.train_data_loader --
# self.train_data_loader isn't assigned until later in __init__ (would AttributeError).
_tl = np.array([np.array(s[1]).reshape(-1)[1:7]
                for s in train_data_loader.dataset.data])
_present = (_tl > 0).sum(0)
_pos_weight = torch.tensor((_tl.shape[0] - _present) / np.maximum(_present, 1),
                           dtype=torch.float32).cuda()
print("pos_weight [happy,sad,anger,surprise,disgust,fear]:", _pos_weight.tolist())
self.loss_task = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=_pos_weight)
```

**What `pos_weight` does:** BCE normally penalises a missed "present" and a missed
"absent" equally. `pos_weight` multiplies the penalty for missing a *present* case by
`(#absent / #present)` for that emotion — so for fear (~9% present) a miss costs ~10x
more, forcing the model to actually predict it. It is computed from the training set
so it is exact and reproducible (inverse class frequency). Everything else is
unchanged: the target binarization (Change 5), the model, and the metrics.

> `.cuda()` is hardcoded (this machine always has a GPU). If CUDA is ever unavailable
> this line errors loudly — which is *better* than silently training on CPU (Section 7).

### Run it (identical settings to Run A — only the loss differs)
```bash
cd ~/TMBL && source ~/tmbl_env/bin/activate
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"   # must be True
nohup python3 -u train.py --name emotion_run2_posweight --patience 15 --trials 5 \
      > ~/TMBL/run_emotion_posweight.log 2>&1 &
echo "PID: $!"
```
Same `--patience 15 --trials 5` as the baseline; new `--name` and log file keep the
two runs separate. On first launch it prints the computed `pos_weight` vector — expect
roughly `[0.9, 2.6, 4.6, 8.1, 5.7, 9.6]` (fear/surprise get the largest weights).

### Comparing A vs B (what to write about)
| | Run A (baseline) | Run B (pos_weight) |
|---|---|---|
| surprise / fear recall | 0.00 / 0.00 | should move **off zero** (the goal) |
| precision (rare emotions) | higher | **lower** (more false alarms) |
| MACRO-F1 | 0.273 | hopefully higher |

The precision/recall trade-off is expected. Even if MACRO-F1 is roughly flat, the
finding — *"a positive-class weight rescues rare-emotion recall at a precision cost,
showing the failure was an optimisation artefact of the class imbalance, not a lack of
signal"* — is a legitimate dissertation contribution.
