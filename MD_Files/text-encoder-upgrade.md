# Text Encoder Upgrade Plan (TMBL / CMU-MOSEI) — PROPOSAL

Replace `bert-base-uncased` with a stronger text encoder to test whether text
representation quality is a bottleneck. **No code has been changed yet** — this
is the reviewable plan. Every edit below follows the house style: comment out
the old line, add the new line under it, revertible with
`git checkout <file>`.

> **Recommendation: try `microsoft/deberta-v3-large` first**, with
> `roberta-large` as the zero-new-dependency fallback. Defer the decoder-LLM
> (Gemma/Llama) route — see §7 for why.

---

## 0. Verified facts (checked on this machine, 2026-07-07)

- transformers **5.10.2**, torch 2.11.0+cu128, tokenizers 0.22.2, protobuf ✓.
- `roberta-large` and `microsoft/deberta-v3-large` both load in transformers
  5.10.2: **hidden 1024, 24 layers** (vs BERT-base 768 / 12). Both are
  **non-gated** — no HF token needed.
- ⚠️ The RoBERTa tokenizer returns only `input_ids` + `attention_mask` —
  **no `token_type_ids`**. The current `collate_fn` does
  `sample["token_type_ids"]` inside a bare `try/except`, so with RoBERTa the
  KeyError would be silently swallowed and MOSEI would fall into the UR_FUNNY
  glove-sum fallback — the exact silent-corruption failure mode of
  post-processing.md Edit 3. Change 3 below fixes this for all encoders.
- ⚠️ DeBERTa-v3's tokenizer **fails without `sentencepiece`** (verified:
  `ValueError` on load). One-time fix, no sudo:
  `source ~/tmbl_env/bin/activate && pip install sentencepiece`
- DeBERTa-v3 has `type_vocab_size = 0` (no segment embeddings); RoBERTa has 1.
  Passing `token_type_ids` is useless for both → Change 6 drops it from the
  model call (harmless for BERT too: our tensor is all zeros, which is BERT's
  default anyway).
- Gemma and Llama are **gated** on HF (license acceptance + token required) —
  another reason to defer the LLM route.
- The repo is currently in **emotion mode** (6-class BCE, per
  emotion-prediction.md). The encoder swap is orthogonal to the task head —
  it works identically for sentiment or emotion. Just compare like-for-like:
  new-encoder emotion run vs BERT-base emotion run.
- GPU is offline right now (kernel 6.17.0-35 has no nvidia module yet; admin
  is updating). Do not launch training until
  `python3 -c "import torch; print(torch.cuda.is_available())"` prints `True`.

---

## 1. Candidate comparison

| | RoBERTa-large | **DeBERTa-v3-large** (pick) | Gemma/Llama (LLM) |
|---|---|---|---|
| Type | drop-in encoder | drop-in encoder | decoder; frozen + LoRA |
| Hidden / layers | 1024 / 24 | 1024 / 24 | 2048–4096 / 28–32 |
| Params | 355 M | 435 M (131 M is embeddings) | 4–9 B |
| Gated? | no | no | **yes (token needed)** |
| New deps | none | `sentencepiece` | `peft`, bf16 path |
| Code delta | Changes 1–7 | Changes 1–7 | §7: much larger |
| Text quality | strong | **strongest** (best GLUE/SST-2 of the three encoder options; consistently wins in MSA encoder-swap studies) | unclear benefit here (see §7) |
| Per-epoch cost vs BERT-base | ~2–2.5× | ~2.5–4× (relative-position attention is slower) | ≫10× without care |

Why DeBERTa-v3-large first: it is the strongest sentence-level encoder that is
still a pure drop-in for this architecture, and the marginal risk over
RoBERTa-large is just one pip install and somewhat slower epochs. If it
misbehaves (instability, speed), switching to `roberta-large` is a **one-line
change** (Change 1) with no other edits.

---

## 2. Where the encoder is wired in today

| Location | What it hardcodes |
|---|---|
| `data_loader.py:5,8` | `BertTokenizer` + `'bert-base-uncased'` |
| `data_loader.py:81-83` | assumes tokenizer returns `token_type_ids` |
| `network.py:4,27-28` | `BertModel/BertConfig` + `'bert-base-uncased'` |
| `network.py:46` | `project_t` in_features = **768** |
| `network.py:72` | `tf_encoder_private_v` d_emb_2 = **768** |
| `network.py:73` | `tf_encoder_private_t` d_emb_1 = **768** |
| `network.py:79` | `fc_t` in_features = **768** |
| `network.py:160-162` | passes `token_type_ids` to the encoder |
| `solver.py:400-417` (build) | freezes `bertmodel.encoder.layer` ≤ 8 (of 12); embeddings left trainable |
| `config.py` | no encoder setting at all |

The masked-mean pooling in `network.py:165-168` is encoder-agnostic (it only
uses `last_hidden_state` + the attention mask) — **no change needed**. Same
for `bert_output[0]` indexing (still valid on transformers-5 ModelOutput).

---

## 3. The changes (7 edits across 4 files)

### Change 1 — single switch point · `config.py` (near line 11, beside `data_str`)
```python
# text encoder: "bert-base-uncased" (768d, 12L) / "roberta-large" / "microsoft/deberta-v3-large" (1024d, 24L)
text_encoder_str = "microsoft/deberta-v3-large"
```
And inside `get_config()`, right before `kwargs = vars(kwargs)` (line ~143):
```python
    kwargs.text_encoder = text_encoder_str
    # freeze encoder layers 0..N (train the top ~25%): 8 for 12-layer base, 19 for 24-layer large
    kwargs.text_freeze_upto = 19 if "large" in text_encoder_str else 8
```
A module-level constant (like `data_str`) rather than only a CLI arg because
`data_loader.py` builds its tokenizer at import time, before any config object
exists. Reverting to BERT later = edit this one string.

### Change 2 — tokenizer · `data_loader.py:5,8`
```python
# from transformers import BertTokenizer
from transformers import AutoTokenizer
from config import text_encoder_str
...
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer = AutoTokenizer.from_pretrained(text_encoder_str)
```
The tokenizer **call** (line 76, `padding='max_length', truncation=True`)
already uses the modern API and needs no change. `max_length=SENT_LEN+2` is
still right: RoBERTa (`<s>…</s>`) and DeBERTa (`[CLS]…[SEP]`) both add 2
special tokens for a single sequence.

### Change 3 — token_type_ids robustness · `data_loader.py:82` ⚠️ silent-corruption
```python
# bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
bert_sentence_types = torch.LongTensor([sample.get("token_type_ids", [0] * len(sample["input_ids"])) for sample in bert_details])
```
Without this, RoBERTa's missing `token_type_ids` raises KeyError → the bare
`except:` at line 84 silently switches to summed word vectors → garbage
training with no error message. (DeBERTa-v3 does return the key, but this
makes the loader safe for any encoder.)

### Change 4 — model instantiation · `network.py:4,27-28`
```python
# from transformers import BertModel, BertConfig
from transformers import AutoModel, AutoConfig
...
# bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
# self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
bertconfig = AutoConfig.from_pretrained(config.text_encoder, output_hidden_states=True)
self.bertmodel = AutoModel.from_pretrained(config.text_encoder, config=bertconfig)
self.text_dim = bertconfig.hidden_size  # 768 for bert-base, 1024 for roberta/deberta large
```
Keep the attribute name `self.bertmodel` — the freezing logic in
`solver.py` matches on that string (see Change 7).

### Change 5 — replace the four hardcoded 768s · `network.py:46,72,73,79`
All four are the text feature dimension; each becomes `self.text_dim`:
```python
# line 46:  nn.Linear(in_features=768, out_features=config.hidden_size)
            nn.Linear(in_features=self.text_dim, out_features=config.hidden_size)
# line 72:  ... d_emb_2=768 ...   (tf_encoder_private_v)
            ... d_emb_2=self.text_dim ...
# line 73:  ... d_emb_1=768 ...   (tf_encoder_private_t)
            ... d_emb_1=self.text_dim ...
# line 79:  self.fc_t = nn.Linear(768, self.config.hidden_size)
            self.fc_t = nn.Linear(self.text_dim, self.config.hidden_size)
```
(The commented-out lines 68-70 also contain 768s but are dead code — leave.)
With `text_dim = bertconfig.hidden_size` this stays correct if you flip back
to bert-base.

### Change 6 — encoder-agnostic forward call · `network.py:160-162`
```python
# bert_output = self.bertmodel(input_ids=bert_sent,
#                             attention_mask=bert_sent_mask,
#                             token_type_ids=bert_sent_type)
bert_output = self.bertmodel(input_ids=bert_sent,
                            attention_mask=bert_sent_mask)
```
RoBERTa/DeBERTa don't use segment embeddings; BERT's default is the all-zeros
tensor we were passing anyway — bit-identical output for BERT. The
`bert_sent_type` tensor keeps flowing through the loader/solver plumbing
(harmless), so no function signatures change.

### Change 7 — freezing · `solver.py` build(), lines ~400-417
In each of the three dataset branches (or collapse them — they're identical):
```python
# if layer_num <= (8):
if layer_num <= self.train_config.text_freeze_upto:  # 8 of 12 (base) / 19 of 24 (large)
    param.requires_grad = False
```
And add, in the same loop, embedding freezing:
```python
if "bertmodel.embeddings" in name:
    param.requires_grad = False
```
Rationale: `<=8` of 12 layers trains the top 3 (25%); `<=19` of 24 trains the
top 4 — same proportion, same recipe. **Note:** today the BERT *embeddings*
are trainable (the name match never catches them). For DeBERTa-v3 that would
put a 131 M-param, 128k-vocab embedding matrix under Adam at lr 1e-4 —
wasteful and destabilizing, so freeze it. If you want a strictly-faithful
recipe port, you can skip the embeddings freeze for the roberta fallback, but
I recommend keeping it for both.

---

## 4. Learning rate & optimizer (no edit initially)

Keep Adam lr 1e-4 for run 1 (parity with the working recipe: the optimizer
already only sees `requires_grad` params, `solver.py:425-427`). Contingency —
if training loss oscillates or dev loss is clearly worse than the BERT
baseline in the first ~5 epochs, split param groups: encoder at 1e-5, the
rest at 1e-4:
```python
enc = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("bertmodel")]
rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("bertmodel")]
self.optimizer = self.train_config.optimizer(
    [{"params": enc, "lr": 1e-5}, {"params": rest, "lr": self.train_config.learning_rate}])
```

---

## 5. Memory / runtime expectations (RTX A6000 48 GB)

- **Weights:** DeBERTa-v3-large ≈ 1.7 GB fp32 (one-time ~0.9 GB download to
  `~/.cache/huggingface/`); RoBERTa-large ≈ 1.4 GB.
- **Training footprint** at batch 48, typical MOSEI utterance lengths
  (~40-80 tokens): forward activations across 24 layers plus optimizer states
  for only the top 4 layers + fusion → expect **~8-14 GB**, far under 48 GB.
  If OOM anyway, halve batch_size at `config.py:134` (48 → 24).
- **Speed:** expect ~2.5-4× the BERT-base epoch time (bigger model; DeBERTa's
  disentangled attention is slower per token). If the first epoch takes
  *hours*, that's the CPU-fallback symptom — re-check the GPU
  (emotion-prediction.md §7), not the code.
- The `UNEXPECTED key` warnings on load stay harmless (unused pretraining
  heads), same as with BERT.

---

## 6. Verification plan (in order)

1. **Prereqs:** GPU back (`torch.cuda.is_available()` → `True`);
   `pip install sentencepiece` in `tmbl_env`.
2. **One-batch sanity check** (catches the silent-fallback trap before
   wasting a run). Temporarily make the loader's bare `except:` loud —
   `except Exception as ex: print("BERT PREP FAILED:", ex); raise` — then:
   ```bash
   cd ~/TMBL && python3 - <<'EOF'
   from config import get_config
   from data_loader import get_loader
   cfg = get_config(parse=False, mode="train")
   batch = next(iter(get_loader(cfg)))
   v, a, y, l, bs, bst, bsm = batch
   print("input_ids", bs.shape, "mask", bsm.shape, "labels", y.shape)
   assert bs.dtype.is_floating_point is False, "loader fell into glove fallback!"
   EOF
   ```
   Expect integer `input_ids` of shape `[48, L]`. Revert the except to quiet
   afterwards (or keep the loud version — MOSEI should never hit it).
3. **5-epoch smoke test:**
   `python3 train.py --name deberta_smoke --n_epoch 5`
   Success = epochs train at sane speed, dev metrics print, checkpoint saves,
   final test table appears. Shape errors here mean one of the four 768s in
   Change 5 was missed.
4. **Full run** (fresh `--name` — old checkpoints are shape-incompatible with
   the 1024-d model, and this also protects your BERT baselines):
   ```bash
   nohup python3 -u train.py --name emotion_deberta_run1 --patience 15 --trials 5 > ~/TMBL/run_emotion_deberta.log 2>&1 &
   ```
5. **Compare like-for-like** against the BERT-base run of the *same task*
   (emotion: mean/per-emotion Acc & F1 from `log_emotion.txt`; sentiment:
   Acc-7 / MAE / Corr / binary Acc from `log.txt`). Text is the dominant
   modality in MOSEI, so a real encoder win should be visible; a realistic
   good outcome is a modest but consistent gain (e.g. ~1 pt binary/F1-level),
   not a step change — the fusion head and 16k training samples bound how
   much a better encoder can add.
6. If unstable → §4 param-group LR. If still no good → flip Change 1 to
   `roberta-large` and repeat from step 3 (no other edits needed).

---

## 7. The decoder-LLM route (deferred — what it would take)

Not recommended as the first experiment, for concrete reasons:

- **Access:** Gemma and Llama are gated → HF license acceptance + token
  management. (Qwen would be the non-gated alternative if this route is ever
  taken.)
- **Code delta is much larger:** `peft` install + LoRA config; hidden size
  2048–4096 into Change 5; no `token_type_ids`; tokenizer has no pad token
  (must set `pad_token = eos_token`); causal attention means masked-*mean*
  pooling is suboptimal (last-token pooling preferred, which needs the
  padding side handled correctly); the freezing logic in `solver.py` matches
  `encoder.layer.N` names that don't exist in decoder models (`layers.N`);
  bf16 + possibly gradient checkpointing to keep speed sane.
- **Expected payoff is poor here:** TMBL immediately mean-pools the text
  encoder output into a *single vector* that feeds small fusion transformers
  trained on 16k samples. That bottleneck caps how much extra capacity can
  help; encoder-swap studies on MOSEI-style tasks get most of the gain from
  RoBERTa/DeBERTa-large, with LLM embeddings adding little at 10-20× cost.

Decision rule: run DeBERTa-v3-large first. If it beats BERT-base clearly,
text quality matters and an LLM-embedding experiment becomes worth its own
plan doc. If it doesn't, the bottleneck is fusion/data, and an LLM would
almost certainly not help either.

---

## 8. Reverting

- Back to BERT-base behavior *without* undoing the refactor: set
  `text_encoder_str = "bert-base-uncased"` in `config.py` (dims and freezing
  auto-adapt; Changes 3/6 are bit-identical no-ops for BERT).
- Full revert: `cd ~/TMBL && git checkout config.py data_loader.py network.py solver.py`
  — but note those files also carry the emotion-mode edits; reverting wholesale
  restores *sentiment* mode too. Prefer the one-line flip.

---

## 9. Checklist

- [ ] GPU back online (`torch.cuda.is_available()` → True)
- [ ] `pip install sentencepiece` (tmbl_env)
- [ ] Change 1 — `config.py` `text_encoder_str` + `text_freeze_upto`
- [ ] Change 2 — `data_loader.py` AutoTokenizer
- [ ] Change 3 — `data_loader.py:82` token_type_ids `.get()` fallback
- [ ] Change 4 — `network.py` AutoModel/AutoConfig + `self.text_dim`
- [ ] Change 5 — `network.py` 768 → `self.text_dim` ×4 (lines 46/72/73/79)
- [ ] Change 6 — `network.py` drop token_type_ids from encoder call
- [ ] Change 7 — `solver.py` freeze ≤ `text_freeze_upto` + freeze embeddings
- [ ] One-batch sanity check passes (integer input_ids, no fallback)
- [ ] 5-epoch smoke test passes
- [ ] Full run vs BERT-base baseline, same task, fresh `--name`
