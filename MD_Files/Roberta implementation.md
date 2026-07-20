# RoBERTa Implementation Guide (TMBL / CMU-MOSEI, 6-emotion mode)

Swap the text encoder `bert-base-uncased` → **`roberta-large`** in the current
**emotion** pipeline (6-output BCE, per emotion-prediction.md). Every change is
"comment out the old line(s), add the new line(s) below", so it stays visible
and revertible. **8 changes across 4 files**, then a 5-epoch smoke test, then
the full run.

> Why roberta-large is safe on this setup (verified 2026-07-07):
> non-gated (no HF token), supported by transformers 5.10.2, **no new pip
> installs needed**, hidden size **1024**, **24** layers (BERT-base: 768 / 12),
> 355 M params ≈ 1.4 GB download to `~/.cache/huggingface/` on first run.

> ⚠️ **The one dangerous trap:** the RoBERTa tokenizer does **not** return
> `token_type_ids` (verified on this machine). The current `collate_fn` reads
> `sample["token_type_ids"]` inside a bare `try/except`, so without Change 3
> the KeyError is silently swallowed and MOSEI falls into the UR_FUNNY
> summed-word-vector fallback → meaningless training, no error printed. This
> is the same silent-corruption mode as Edit 3 in post-processing.md.
> **Change 3 is not optional.**

---

## 1. The code changes (8 edits across 4 files)

All file paths are under `~/TMBL/`. Line numbers refer to the current files
(emotion-mode edits already applied).

---

### Change 1 — one switch point for the encoder
**File: `config.py`, near line 11** (right beside `data_str`)

**ADD (new lines, nothing to comment out):**
```python
# text encoder: "bert-base-uncased" (768d, 12 layers) or "roberta-large" (1024d, 24 layers)
text_encoder_str = "roberta-large"
```

**File: `config.py`, line ~143** (inside `get_config()`, just BEFORE
`kwargs = vars(kwargs)`)

**ADD:**
```python
    kwargs.text_encoder = text_encoder_str
    # freeze encoder layers 0..N, train the top ~25%: N=8 for 12-layer bert-base, N=19 for 24-layer roberta-large
    kwargs.text_freeze_upto = 19 if "large" in text_encoder_str else 8
```
**What it does:** makes the encoder name a single module-level constant (same
pattern as `data_str`) that both `data_loader.py` and `network.py` can use,
and computes the matching layer-freeze threshold. A module constant — not only
a CLI arg — is required because `data_loader.py` builds its tokenizer at
import time, before any `Config` object exists. Reverting to BERT later =
change this one string; every other edit auto-adapts.

---

### Change 2 — tokenizer
**File: `data_loader.py`, lines 5 and 8**

**OLD:**
```python
from transformers import BertTokenizer
...
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```
**NEW:**
```python
# from transformers import BertTokenizer
from transformers import AutoTokenizer
from config import text_encoder_str
...
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer = AutoTokenizer.from_pretrained(text_encoder_str)
```
**What it does:** loads the tokenizer that matches whatever encoder Change 1
selects (RoBERTa's byte-level BPE instead of BERT's WordPiece). The tokenizer
*call* at line 76 (`padding='max_length', truncation=True`) already uses the
modern transformers-5 API and needs **no** change; `max_length=SENT_LEN + 2`
is still correct because RoBERTa also adds exactly 2 special tokens
(`<s>` … `</s>`) around a single sequence.

---

### Change 3 — token_type_ids fallback  ⚠️ prevents silent corruption
**File: `data_loader.py`, line 82** (inside `collate_fn`)

**OLD:**
```python
bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details]) # bert_sentence_types.shape=([64, 41]) 堆叠(64): len(41): [0, 0 ...]
```
**NEW:**
```python
# bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details]) # bert_sentence_types.shape=([64, 41])
bert_sentence_types = torch.LongTensor([sample.get("token_type_ids", [0] * len(sample["input_ids"])) for sample in bert_details]) # roberta returns no token_type_ids -> substitute zeros
```
**What it does:** RoBERTa's tokenizer output has no `token_type_ids` key, so
the old line raises KeyError → caught by the bare `except:` at line 84 → the
loader silently switches to summed GloVe vectors instead of transformer input.
The new line substitutes an all-zeros list of the right length when the key is
missing — which is exactly what BERT's segment ids were anyway (single
sentence = all zeros). The tensor keeps flowing through the existing plumbing
so no function signatures change. (After Change 6 the model no longer consumes
it, but the collate/solver code still passes it around.)

---

### Change 4 — encoder instantiation
**File: `network.py`, lines 4 and 27-28**

**OLD:**
```python
from transformers import BertModel, BertConfig
...
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
```
**NEW:**
```python
# from transformers import BertModel, BertConfig
from transformers import AutoModel, AutoConfig
...
        # bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        bertconfig = AutoConfig.from_pretrained(config.text_encoder, output_hidden_states=True)
        self.bertmodel = AutoModel.from_pretrained(config.text_encoder, config=bertconfig)
        self.text_dim = bertconfig.hidden_size # 768 for bert-base, 1024 for roberta-large
```
**"But nothing here imports RoBERTa — how does it get pulled in?"**
`AutoModel` is not a model class, it is a **dispatcher**. When
`AutoModel.from_pretrained("roberta-large")` runs it:
1. treats `"roberta-large"` as a **Hub repo ID** and downloads that repo's
   `config.json` (cached in `~/.cache/huggingface/`);
2. reads the field `"model_type": "roberta"` out of it;
3. looks that string up in transformers' internal registry
   (`'roberta' -> RobertaModel`, `'bert' -> BertModel`, ...);
4. **lazily imports** `transformers.models.roberta.modeling_roberta.RobertaModel`
   internally;
5. instantiates it and loads the roberta-large weights.

So the import does happen — inside transformers, driven by the downloaded
config file, rather than by a static `import` line in this repo. After this
change `self.bertmodel` really is a `RobertaModel`; only the *attribute name*
still says "bert". Same for `AutoTokenizer`, which returns a real
`RobertaTokenizer` (byte-level BPE, not BERT WordPiece).

*Explicit alternative, if you prefer it:*
`from transformers import RobertaModel` + `RobertaModel.from_pretrained("roberta-large")`
behaves identically — the only cost is that switching encoders then means
editing imports in two files instead of flipping Change 1's single string.

**What it does:** `AutoModel`/`AutoConfig` resolve to `RobertaModel` for
`roberta-large` (and back to `BertModel` if Change 1 is reverted).
`self.text_dim` reads the true hidden size from the model config so nothing
downstream hardcodes 768 or 1024. **Keep the attribute name `self.bertmodel`**
— the layer-freezing logic in `solver.py` matches on that string (Change 7).
The masked-mean pooling at lines 165-168 and the `bert_output[0]` indexing are
encoder-agnostic — no change there. On first run it downloads ~1.4 GB of
weights; the `UNEXPECTED key` warnings on load are harmless (unused
pretraining head), same as with BERT.

---

### Change 5 — replace the four hardcoded 768s (text feature dimension)
**File: `network.py`, lines 46, 72, 73, 79** — four places, all must agree.

**5a — line 46 (`project_t`):**
```python
# nn.Linear(in_features=768, out_features=config.hidden_size))
nn.Linear(in_features=self.text_dim, out_features=config.hidden_size))
```
**5b — line 72 (`tf_encoder_private_v`, second input dim):**
```python
# self.tf_encoder_private_v = MyTransformer(d_emb_1=4*input_sizes[1], d_emb_2=768, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
self.tf_encoder_private_v = MyTransformer(d_emb_1=4*input_sizes[1], d_emb_2=self.text_dim, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
```
**5c — line 73 (`tf_encoder_private_t`, first input dim):**
```python
# self.tf_encoder_private_t = MyTransformer(d_emb_1=768, d_emb_2=4*input_sizes[2], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
self.tf_encoder_private_t = MyTransformer(d_emb_1=self.text_dim, d_emb_2=4*input_sizes[2], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
```
**5d — line 79 (`fc_t`):**
```python
# self.fc_t = nn.Linear(768, self.config.hidden_size)
self.fc_t = nn.Linear(self.text_dim, self.config.hidden_size)
```
**What they do:** these are the only four live places where the text feature
width (768) is baked in. 5a resizes the projection of the pooled text vector;
5b/5c resize the two "private" cross-modal transformers that consume raw text
features; 5d resizes the linear layer that maps `private_t` into the shared
256-d fusion space. All four must become `self.text_dim` (1024) or the model
crashes with a matmul shape error at the first forward pass. (The commented
lines 68-70 also contain 768s but are dead code — leave them.) These changes
must be **after** Change 4 in `__init__`, which they are (line 28 < line 46).

---

### Change 6 — drop token_type_ids from the encoder call
**File: `network.py`, lines 160-162** (inside `alignment()`)

**OLD:**
```python
        bert_output = self.bertmodel(input_ids=bert_sent, # bert_sent ([64, 41])
                                    attention_mask=bert_sent_mask, # bert_sent_mask ([64, 41])
                                    token_type_ids=bert_sent_type) # bert_sent_type ([64, 41])
```
**NEW:**
```python
        # bert_output = self.bertmodel(input_ids=bert_sent,
        #                             attention_mask=bert_sent_mask,
        #                             token_type_ids=bert_sent_type)
        bert_output = self.bertmodel(input_ids=bert_sent, # bert_sent ([48, L])
                                    attention_mask=bert_sent_mask) # token_type_ids dropped: roberta has no segment embeddings; bert default is the all-zeros we passed anyway
```
**What it does:** RoBERTa has no meaningful segment embeddings, so the ids are
useless to it; and for BERT, omitting the argument makes it default to
all-zeros — bit-identical to what was being passed. This makes the forward
call work for either encoder. The `bert_sent_type` tensor still arrives as a
function argument (unchanged signatures); it's simply unused now.

---

### Change 7 — scale the layer freezing from 12-layer to 24-layer
**File: `solver.py`, inside `build()` (starts line 453)** — the freeze
threshold appears **3 times**: mosei branch **line 463**, ur_funny branch
**line 468**, mosi branch **line 476**. Only the **mosei** branch runs for
you, but changing all three keeps them consistent. Each is:

**OLD:**
```python
                    if layer_num <= (8):
                        param.requires_grad = False
```
**NEW:**
```python
                    # if layer_num <= (8):
                    if layer_num <= self.train_config.text_freeze_upto: # 8 of 12 (bert-base) / 19 of 24 (roberta-large) -> top ~25% trainable
                        param.requires_grad = False
```
**What it does:** the working BERT recipe froze layers 0-8 of 12, training the
top 3 (25%). `roberta-large` has 24 layers, so the same recipe proportion is
freeze 0-19, train the top 4 (20-23). The threshold comes from Change 1's
`text_freeze_upto`, so it flips back automatically with the encoder. The name
match `"bertmodel.encoder.layer"` itself needs no edit — RoBERTa's layers are
also named `encoder.layer.N` under the `self.bertmodel` attribute. The
optimizer needs no edit either: it already filters on `requires_grad`
(solver.py line ~425).

**7b (RECOMMENDED) — also freeze the word-embedding table.**

*Background — why this is needed at all.* The loop
`for name, param in self.model.named_parameters()` walks **every** parameter
tensor together with its dotted name. The names look like this:
```
bertmodel.embeddings.word_embeddings.weight       <- the vocabulary lookup table
bertmodel.embeddings.position_embeddings.weight
bertmodel.encoder.layer.0.attention.self.query.weight
bertmodel.encoder.layer.1. ...
```
Change 7's filter matches only the string `"bertmodel.encoder.layer"`. The
**embedding** tensors do not contain that string, so they never match and stay
**trainable** — nobody chose to fine-tune them, they simply fall through the
filter. That was tolerable with bert-base; with roberta-large it is not:

| Trainable text params | Size |
|---|---|
| The 4 encoder layers you *intend* to train (layers 20-23) | 50.3 M |
| The embedding table that leaks through by accident | **51.5 M** (50265 vocab × 1024) |

The accidental part is **bigger than the deliberate part** — you would more
than double the trainable text parameters, on a lookup table. Cost: ~600 MB
extra GPU memory (Adam stores two state tensors per trainable param), and each
token's embedding row only receives a gradient when that token happens to
appear in a batch, so rare words get erratic sparse updates at lr 1e-4 and
drift away from their pretrained values.

**THE EDIT.** `solver.py` lines 476-479 currently read:
```python
                    if layer_num <= (8):
                        param.requires_grad = False       
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
```
Insert two lines in the middle (indent **12 spaces**, i.e. the same level as
`if 'weight_hh' in name:`) so it becomes:
```python
                    if layer_num <= (8):
                        param.requires_grad = False       
            if "bertmodel.embeddings" in name: # freeze the 51.5M-param embedding table (bert recipe left it trainable by accident)
                param.requires_grad = False
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
```
Placement is flexible — anywhere inside that `for` loop works, since it is an
independent check on the same `name`. Just before `weight_hh` keeps all the
freezing logic together.

**Only reason to skip it:** if you want a strictly-faithful port of the BERT
recipe (embeddings trainable in both runs) for the purest apples-to-apples
encoder comparison. Note which you chose when comparing results.

---

### Change 8 — separate log file for this experiment
**File: `solver.py`, lines 22-23**

**OLD:**
```python
# log_file = "/home/elangoa/TMBL/checkpoints/log.txt"
log_file = "/home/elangoa/TMBL/checkpoints/log_emotion.txt"
```
**NEW:**
```python
# log_file = "/home/elangoa/TMBL/checkpoints/log.txt"          # sentiment run log (bert-base)
# log_file = "/home/elangoa/TMBL/checkpoints/log_emotion.txt"  # emotion run log (bert-base baseline)
log_file = "/home/elangoa/TMBL/checkpoints/log_emotion_roberta.txt"  # emotion run log (roberta-large)
```
**What it does:** keeps the bert-base emotion baseline log untouched so the
comparison stays clean — same pattern you used to separate sentiment from
emotion logs. All per-epoch metrics and the final per-emotion table for the
RoBERTa runs go to `log_emotion_roberta.txt`.

---

## 2. What does NOT change

- **No new pip installs** (sentencepiece is only needed for DeBERTa).
- `create_dataset.py`, the `.pkl` files, preprocessing — untouched. **Do not
  re-run preprocessing**; tokenization happens live in `collate_fn`.
- The 6-emotion head, `calc_metrics`, and your **`pos_weight` BCE loss**
  (`solver.py:36-44`, the per-emotion `sqrt((#absent/#present))` weighting) —
  all orthogonal to the encoder, untouched. ⚠️ **But it affects your
  baseline:** compare the RoBERTa run against a bert-base run that used the
  **same** `pos_weight` setting (i.e. `run_emotion_posweight.log`, not the
  earlier un-weighted `run_emotion.log`). Mixing the two confounds the
  encoder effect with the class-weighting effect.
- Masked-mean pooling in `network.py:165-168` — encoder-agnostic, untouched.
- Learning rate / optimizer — keep Adam 1e-4 for run 1 (parity with the
  working recipe). Contingency only if training is clearly unstable vs the
  BERT baseline in the first ~5 epochs: split param groups (encoder 1e-5,
  rest 1e-4) — ask for this edit if needed.

---

## 3. How to run

### Step 0 — prerequisites
```bash
source ~/tmbl_env/bin/activate
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"   # MUST print True (GPU was offline; admin fixing kernel)
```
Do not start any run until this prints `True` — otherwise it will silently
crawl on CPU (see emotion-prediction.md §7).

### Step 1 — one-batch sanity check (OPTIONAL — you can skip straight to Step 2)

**Why it is safe to skip.** Verified by direct test: if Change 3 is missed, the
bare `except:` at `data_loader.py:84` swallows the KeyError and hands the model
a *float* tensor — but the embedding lookup rejects non-integer indices, and
the encoder call at `network.py:160` is **not** inside any try/except, so the
run **crashes within the first batch** (~1 min in, right after the weight
download). It fails loudly, not silently. A missed Change 5 likewise crashes
immediately with a 768-vs-1024 matmul shape error.

⚠️ **But the error message is misleading — memorise this mapping.** Because the
bare `except:` already ate the real cause, you will NOT see
`KeyError: token_type_ids`. You will see:
```
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following
scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
```
**That error means Change 3 was missed** — nothing else. Fix Change 3 and rerun.

**If you do want the check, it needs NO code edit.** (The earlier version of
this guide asked you to temporarily modify the `except:` — that was only for a
prettier error message. The `assert` below catches the fallback by itself.)
```bash
cd ~/TMBL
python3 - <<'EOF'
from config import get_config
from data_loader import get_loader
cfg = get_config(parse=False, mode="train")
v, a, y, l, bs, bst, bsm = next(iter(get_loader(cfg)))
print("input_ids:", bs.shape, bs.dtype, "| mask:", bsm.shape, "| labels:", y.shape)
assert not bs.dtype.is_floating_point, "loader fell into the glove fallback -> Change 3 missing!"
print("OK: integer token ids reached the model input — no silent fallback")
EOF
```
Expected: `input_ids: torch.Size([48, L]) torch.int64` and the OK line. This
runs in well under a minute and does not download the model weights (tokenizer
only), which is its one advantage over just letting Step 2 crash.

### Step 2 — 5-epoch smoke test (~30-60 min on GPU)
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 train.py --name emotion_roberta_smoke --n_epoch 5
```
Success = epochs train, `[dev] mean emotion Acc/F1` prints each epoch, a
checkpoint saves (`Found new best model on dev set!`), and it ends with the
per-emotion test table.

Failure → diagnosis (both crash in the first minute):

| Error you see | What it means |
|---|---|
| `RuntimeError: ...'indices' to have... Long, Int; but got torch.FloatTensor... (embedding)` | **Change 3 missed** — loader fell into the glove fallback |
| matmul / shape error mentioning **768** vs **1024** | one of the four **Change 5** edits missed |

Expect each epoch to take roughly
**2-2.5×** the bert-base epoch time. If the *first* epoch takes hours →
re-check Step 0 (CPU fallback), not the code.

### Step 3 — full run (background, survives disconnect)
Use a **fresh `--name`** — old checkpoints are shape-incompatible with the
1024-d model, and this also protects your bert-base best models:
```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
nohup python3 -u train.py --name emotion_roberta_run1 --patience 15 --trials 5 > ~/TMBL/run_emotion_roberta.log 2>&1 &
echo "PID: $!"
tail -f ~/TMBL/run_emotion_roberta.log     # Ctrl+C stops watching, not training
```
If `CUDA out of memory` (unlikely: expected footprint ~8-12 GB of 48 GB):
halve `batch_size` at `config.py` line ~134 (48 → 24).

### Step 4 — verify it finished and read the results
```bash
ps aux | grep train.py | grep -v grep                                        # empty = finished
grep "early stopping" ~/TMBL/run_emotion_roberta.log                          # present = stopped normally
grep -A 9 "Per-emotion test results" ~/TMBL/run_emotion_roberta.log | tail -9 # final table
```

### Where the outputs go (all separate from previous runs)

| Output | bert-base emotion runs (existing) | roberta-large emotion run (new) |
|---|---|---|
| Structured metrics log | `checkpoints/log_emotion.txt` | `checkpoints/log_emotion_roberta.txt` (Change 8) |
| Full console output | `run_emotion.log`, `run_emotion_posweight.log` | `run_emotion_roberta.log` (set at launch, Step 3) |
| Best model checkpoint | `checkpoints/devBest/model_emotion_run1.std` | `checkpoints/devBest/model_emotion_roberta_run1.std` (via `--name`) |

> Reminder: the periodic `checkpoints/epochN.pt` files reuse fixed names and
> WILL be overwritten by this run — they're throwaway resume checkpoints, not
> results. Move them aside first if you want to keep the old ones.

### How to judge the result
Compare the final per-emotion table (Acc + weighted-F1 per emotion, and MEAN)
against the bert-base numbers from the run that used the **same `pos_weight`
setting** (see §2) — same task, same data, same loss, only the encoder
differs. Watch **F1 on the rare emotions**
(fear, surprise): text is MOSEI's dominant modality, so a genuinely better
encoder should lift mean F1 consistently, but expect a modest gain (~1 point
level), not a step change — the fusion head and 16k training samples cap the
upside. If RoBERTa wins clearly, DeBERTa-v3-large (see
text-encoder-upgrade.md) is the natural next experiment; if it changes
nothing, the bottleneck is fusion/data, not the encoder.

---

## 4. Reverting

- Back to bert-base **without undoing the refactor**: set
  `text_encoder_str = "bert-base-uncased"` in `config.py` — dims, tokenizer,
  and freeze threshold all auto-adapt; Changes 3/6 are bit-identical no-ops
  for BERT. (Also swap Change 8's log line back if you want new bert runs in
  the old log.)
- Full git revert (`git checkout config.py data_loader.py network.py solver.py`)
  also wipes the **emotion-mode** edits and returns to sentiment — avoid
  unless that's what you want.

---

## 5. Checklist

- [ ] Step 0: `torch.cuda.is_available()` → True
- [ ] Change 1 — `config.py`: `text_encoder_str = "roberta-large"` + `text_encoder`/`text_freeze_upto` in `get_config()`
- [ ] Change 2 — `data_loader.py:5,8`: AutoTokenizer from `text_encoder_str`
- [ ] Change 3 — `data_loader.py:82`: `.get("token_type_ids", zeros)` ⚠️
- [ ] Change 4 — `network.py:4,27-28`: AutoModel/AutoConfig + `self.text_dim`
- [ ] Change 5 — `network.py:46,72,73,79`: all four 768 → `self.text_dim`
- [ ] Change 6 — `network.py:160-162`: drop `token_type_ids` from the call
- [ ] Change 7 — `solver.py` build(): threshold → `text_freeze_upto` (×3 branches)
- [ ] Change 7b (optional) — freeze `bertmodel.embeddings` (note the choice)
- [ ] Change 8 — `solver.py:23`: log → `log_emotion_roberta.txt`
- [ ] Step 1 one-batch sanity check — OPTIONAL, no code edit needed (skippable: Step 2 crashes loudly on the same faults)
- [ ] Step 2 smoke test (`--n_epoch 5`) passes
- [ ] Step 3 full run launched → results in `run_emotion_roberta.log` / `log_emotion_roberta.txt`
