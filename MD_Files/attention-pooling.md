# Attention-Pooling Experiment (TMBL / CMU-MOSEI, 6-emotion mode)

Replace the **masked mean pooling** of the text encoder output with a small
**learned attention pooling** module. The encoder (currently `roberta-large`
via `text_encoder_str`) is **untouched** — this changes only *how* the
per-token sequence is collapsed into the single text vector `t_feature`.
Style matches the other guides: comment out the old line(s), add the new
line(s) below; revertible with `git checkout`.

> **The hypothesis being tested.** RoBERTa-large ≈ BERT-base on this task
> (macro-F1 0.430 vs 0.437 — a tie within single-seed noise). A clearly
> stronger encoder producing the same output points at a *downstream* step
> erasing the difference. The mean-pool
> (`network.py:177-181`) is the prime suspect: it's a crude **unlearned**
> average that weights every token `1/N`, diluting the few emotion-bearing
> words among the filler. Attention pooling learns *which* tokens matter.
> **If this helps → pooling was the bottleneck (and DeBERTa becomes worth
> trying). If it doesn't → the bottleneck is fusion/data, ruled out cheaply.**

> **Not a guaranteed win.** On ~16k training samples, attention pooling's
> extra parameters carry a small overfitting risk, and masked mean is a
> genuinely strong baseline. Expect a modest change either way; the value is
> as much diagnostic as it is a gain. Run **2-3 seeds** so the ~0.7-pt noise
> floor is visible before you conclude anything.

---

## 1. What mean vs attention pooling actually compute

The encoder emits one vector per token: `bert_output` of shape
`[batch, L, text_dim]` (`text_dim` = 1024 for roberta-large). Both methods
collapse the `L` token vectors into one `[batch, text_dim]` vector, using the
attention mask to ignore padding.

**Current — masked mean** (`network.py:177-181`): every real token gets weight
`1/N`.
```
pooled = ( Σ_i  mask_i · h_i ) / Σ_i mask_i
```

**New — attention pooling**: a tiny learned scorer gives each token a weight
`α_i` (softmax over real tokens, padding forced to 0), then a weighted sum.
```
score_i = v · tanh(W · h_i)          # W: [text_dim → attn_hidden], v: [attn_hidden]
α = softmax(score, over real tokens) # padding masked to -inf before softmax → α=0
pooled = Σ_i α_i · h_i
```
Added parameters: `W` (`text_dim × attn_hidden`) + `v` (`attn_hidden`). With
`attn_hidden = 128` and `text_dim = 1024` that's ≈ 131k params — negligible
vs the encoder, small enough to regularize with the existing dropout.

---

## 2. The code changes (3 edits, all in `network.py`)

Everything is in `~/TMBL/network.py`. No other file changes: the loader
already provides `bert_sent_mask`, and `alignment()` already receives it.

---

### Change 1 — add the `AttentionPool` module
**File: `network.py`** — paste this as a **new module**, at top level (not
indented), anywhere outside the `AOTN` class. Cleanest spot: right **after the
end of the `AOTN` class**, just before the `def l2_normalize` helper (currently
line ~228), so it sits with the other building-block modules.

```python
class AttentionPool(nn.Module):
    """Learned attention pooling over the token dimension.
    Collapses [B, L, dim] -> [B, dim] with mask-aware softmax weights,
    a drop-in replacement for masked mean pooling."""
    def __init__(self, dim, attn_hidden=128, dropout=0.5):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),   # one score per token
        )

    def forward(self, hidden, mask):
        # hidden: [B, L, dim]   mask: [B, L] (1 = real token, 0 = padding)
        scores = self.score(hidden).squeeze(-1)                  # [B, L]
        scores = scores.masked_fill(mask == 0, float('-inf'))    # padding cannot win weight
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)       # [B, L, 1], sums to 1 over real tokens
        pooled = torch.sum(alpha * hidden, dim=1)                # [B, dim]
        return pooled
```
**What it does:** the `score` MLP reads each token's `dim`-vector and outputs a
scalar relevance score; `masked_fill(-inf)` guarantees padding tokens get
exactly 0 weight after softmax; the softmax normalizes the real-token scores to
sum to 1; the weighted sum produces the pooled vector. Output shape and dtype
are identical to the mean-pool it replaces, so nothing downstream changes.
`torch` and `nn` are already imported at the top of the file.

---

### Change 2 — instantiate it in `__init__`
**File: `network.py`, after line 33** (`self.text_dim = bertconfig.hidden_size`)

**ADD:**
```python
        # attention pooling over encoder tokens (replaces masked mean); operates in text_dim space
        self.text_pool = AttentionPool(self.text_dim, attn_hidden=128, dropout=self.config.dropout)
```
**What it does:** creates the pooling layer sized to the encoder's hidden
dimension (`self.text_dim`, so it auto-adapts to bert-base 768 or roberta-large
1024). Placing it right after `self.text_dim` is defined guarantees the value
exists. Reuses `config.dropout` (0.5) so the extra capacity is regularized like
the rest of the model. `self.text_dim` is defined at line 33, well before this,
so ordering is fine.

---

### Change 3 — use it in `alignment()` instead of the mean-pool
**File: `network.py`, lines 177-181** (the `# masked mean` block)

**OLD:**
```python
        bert_output = bert_output[0] # bert_output[0].shape ([64, 41, 768]) len(bert_output)=3  bert_output[1].shape ([64, 768])
        # masked mean
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output) # ([64, 41])->([64, 41, 1]) @ ([64, 41, 768]) -> masked_output ([64, 41, 768])
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) # mask_len (64,1)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len # 'torch.cuda.FloatTensor' bert_output ([64, 768])
        t_feature = bert_output # ([64, 768])
```
**NEW:**
```python
        bert_output = bert_output[0] # [B, L, text_dim] full token sequence
        # ---- masked mean (original) ----
        # masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        # mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        # bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        # t_feature = bert_output
        # ---- attention pooling (new) ----
        t_feature = self.text_pool(bert_output, bert_sent_mask) # [B, L, text_dim] -> [B, text_dim]
```
**What it does:** feeds the full token sequence + the attention mask into the
learned pooler and takes its `[B, text_dim]` output as `t_feature`. Everything
after this line (`project_t`, the private transformers, `fc_t`, fusion) is
unchanged — `t_feature` has the exact same shape and role as before. Keep
`bert_output = bert_output[0]` (the new pooler needs the full sequence, not the
mean).

---

## 3. What does NOT change

- **The encoder** (`roberta-large` / whatever `text_encoder_str` selects) — this
  experiment isolates pooling from encoder choice on purpose.
- `data_loader.py` — `bert_sent_mask` is already produced and passed through.
- The 6-emotion head, the `pos_weight` BCE loss, `calc_metrics` — untouched.
- Downstream text path (`project_t` line 52, private transformers, `fc_t`
  line 89) — all still receive a `[B, text_dim]` `t_feature`.
- Learning rate / optimizer — keep Adam 1e-4. The new params are tiny and land
  in the `requires_grad` filter automatically (`solver.py` build), so they train
  with no optimizer edit.

---

## 4. Sanity check before a full run (optional, no code edit)

Confirms shapes line up and the pooler runs end-to-end on one batch:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 - <<'EOF'
import torch
from config import get_config
from data_loader import get_loader
import network
cfg = get_config(parse=False, mode="train")
loader = get_loader(cfg)
v, a, y, l, bs, bst, bsm = next(iter(loader))
cfg.visual_size = v.shape[2]; cfg.acoustic_size = a.shape[2]
m = network.AOTN(cfg)
assert hasattr(m, "text_pool"), "Change 2 missing: self.text_pool not created"
# check the pooler in isolation
h = torch.randn(4, 7, m.text_dim); mask = torch.ones(4, 7); mask[0, 5:] = 0
out = m.text_pool(h, mask)
print("pool out:", out.shape, "(expect [4,", m.text_dim, "])")
assert out.shape == (4, m.text_dim) and torch.isfinite(out).all()
print("OK: AttentionPool runs and shape is correct")
EOF
```
Expect `pool out: torch.Size([4, 1024])` and the OK line. A shape error or
`AttributeError: text_pool` means Change 2 or 3 was missed.

---

## 5. How to run

### Step 0 — prerequisites
```bash
source ~/tmbl_env/bin/activate
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"   # MUST print True
```

### Step 1 — 5-epoch smoke test
```bash
cd ~/TMBL
python3 train.py --name emotion_attnpool_smoke --n_epoch 5
```
Success = trains, prints `[dev] mean emotion Acc/F1` each epoch, saves a
checkpoint, ends with the per-emotion test table. Runtime per epoch is
essentially unchanged from the roberta run (the pooler is a rounding error on
top of a 24-layer encoder).

### Step 2 — full run (fresh `--name`, separate log)
Use a distinct name so it won't clash with the roberta/bert checkpoints, and
send the log somewhere separate. **First** set the log file in `solver.py`
line 23 (same one-line pattern as before):
```python
log_file = "/home/elangoa/TMBL/checkpoints/log_emotion_attnpool.txt"
```
Then:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
nohup python3 -u train.py --name emotion_attnpool_run1 --patience 15 --trials 5 > ~/TMBL/run_emotion_attnpool.log 2>&1 &
echo "PID: $!"
tail -f ~/TMBL/run_emotion_attnpool.log
```

### Step 3 — for a real verdict, run 2-3 seeds
The bert/roberta gap (0.7 pt) was inside single-seed noise, so one attnpool run
can't settle anything. Vary the seed and give each its own name:
```bash
# edit solver.py line 12: torch.manual_seed(123) -> 456, then 789
python3 -u train.py --name emotion_attnpool_seed456 --patience 15 --trials 5 > ~/TMBL/run_attnpool_seed456.log 2>&1 &
```
Compare the **mean ± spread** of macro-F1 across seeds against the roberta
mean-pool baseline, not single numbers.

### Step 4 — verify finished + read results
```bash
ps aux | grep train.py | grep -v grep                                          # empty = finished
grep -A 9 "Per-emotion" ~/TMBL/run_emotion_attnpool.log | tail -9              # final table
```

---

## 6. How to judge it

Compare the per-emotion table against your **roberta mean-pool** run
(`run_emotion_roberta.log`, macro-F1 0.430) — same encoder, same loss, same
data, only pooling differs.

- **Watch macro-F1 and the rare emotions (fear/surprise) most.** Attention
  pooling's whole thesis is picking out the few salient tokens, which should
  help precisely where the signal is sparse. If fear/surprise F1 moves up while
  the frequent classes hold, that's the mechanism working.
- **A real result is a consistent shift across seeds**, not one lucky run. A
  +1-2 pt macro-F1 that holds across 2-3 seeds = pooling mattered → DeBERTa is
  now worth trying (a better encoder finally has a channel to help). No movement
  across seeds = pooling wasn't the bottleneck; stop swapping text-side pieces
  and look at the fusion head / class imbalance instead.

---

## 7. Reverting

Every edit is comment-out + add:
- Change 3: uncomment the four masked-mean lines, comment the `self.text_pool`
  line.
- Changes 1 & 2 can be left in place (an unused module/attribute is harmless),
  or removed.
- Or `git checkout network.py` — but that also drops the RoBERTa encoder edits,
  returning to bert-base; prefer the comment-swap to keep the encoder.

---

## 8. Checklist

- [ ] Step 0: `torch.cuda.is_available()` → True
- [ ] Change 1 — `network.py`: add `AttentionPool` class (after `AOTN`)
- [ ] Change 2 — `network.py:33`: instantiate `self.text_pool`
- [ ] Change 3 — `network.py:177-181`: swap masked-mean → `self.text_pool(...)`
- [ ] Sanity check passes (`pool out: [4, 1024]`)
- [ ] Smoke test (`--n_epoch 5`) passes
- [ ] `solver.py:23` log → `log_emotion_attnpool.txt`
- [ ] Full run + 2-3 seeds launched → results in `run_emotion_attnpool*.log`
- [ ] Compared mean±spread vs roberta mean-pool baseline (macro-F1 0.430)
