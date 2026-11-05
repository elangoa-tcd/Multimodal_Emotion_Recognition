# Multimodal Emotion Recognition on CMU-MOSEI with TMBL

Code and results for the MSc dissertation
**"Transformer based Multimodal Binding Learning Model for Emotion Recognition"**
(Akhil Elango, MSc Computer Science – Intelligent Systems, Trinity College Dublin, 2026;
supervisor Dr Yvette Graham).

This repository adapts **TMBL** (Huang et al., 2024), a transformer model for multimodal *sentiment
regression*, to **six-way multi-label emotion detection** (happy, sad, anger, surprise, disgust,
fear) on CMU-MOSEI, using text, audio and video. It also includes diagnostic tools that measure
how much each modality actually contributes.

## Key findings

- **Handling class imbalance matters most.** Weighting rare emotions in the loss (`pos_weight`)
  raises macro-F1 from 0.385 to 0.438. A stronger text encoder (RoBERTa-large) adds +0.015, and
  per-emotion decision thresholds tuned on dev add +0.009. Final model: **0.462 macro-F1**.
- **Text dominates.** Removing text at test time costs 0.096 macro-F1; removing both audio and
  video costs only 0.016.
- **The model already uses almost all the audio/video signal there is.** A linear probe shows
  audio+video can add at most **+0.0167** macro-F1 on top of the model's own text embedding. The
  trained model realises **+0.0162** of it (about 97%), so the remaining limit lies in the
  features and dataset, not in the fusion architecture.

---

## Main results

Test set (4,654 clips). F1 is for the "present" class of each emotion. Macro-F1 is the unweighted
mean over the six emotions.

| Configuration | Happy | Sad | Anger | Surprise | Disgust | Fear | **Macro-F1** | Dissertation |
|---|---|---|---|---|---|---|---|---|
| Baseline (BERT, no class weighting) | 0.695 | 0.401 | 0.365 | 0.177 | 0.495 | 0.180 | **0.385** | Table 5.3 |
| BERT + pos_weight | 0.727 | 0.433 | 0.491 | 0.255 | 0.527 | 0.195 | **0.438** | Table 5.4 |
| BERT + sqrt pos_weight | 0.718 | 0.432 | 0.424 | 0.210 | 0.524 | 0.201 | **0.418** | Table 5.5 |
| BERT + pos_weight + attention pooling | 0.732 | 0.438 | 0.477 | 0.252 | 0.519 | 0.209 | **0.438** | Table 5.6 |
| RoBERTa-large + pos_weight | 0.727 | 0.457 | 0.494 | 0.266 | 0.551 | 0.225 | **0.453** | Table 5.7 |
| **RoBERTa-large + pos_weight + tuned thresholds** | 0.747 | 0.474 | 0.507 | 0.265 | 0.557 | 0.221 | **0.462** | Table 5.8 |

**Modality contribution** (final model, tuned thresholds; Tables 5.10–5.11):

| Measurement | Macro-F1 | Change |
|---|---|---|
| Full model | 0.462 | – |
| Audio removed | 0.455 | −0.007 |
| Video removed | 0.453 | −0.009 |
| Audio and video removed | 0.446 | −0.016 |
| Text removed | 0.366 | −0.096 |
| Probe: text only → text + audio + video | 0.4463 → 0.4629 | **+0.0167 available** |
| Model: text only → full model | 0.4457 → 0.4619 | **+0.0162 realised** |

---

## Quick start

| I want to… | Go to | GPU? | Time |
|---|---|---|---|
| Check the reported numbers | [Check the numbers without a GPU](#check-the-numbers-without-a-gpu) | No | 1 min |
| Train a model | [Train](#3-train) | Yes | 3–6 h per run |
| Evaluate a model you trained | [Evaluate a trained model](#4-evaluate-a-trained-model) | Yes | ~10 min |

### Check the numbers without a GPU

Every test-set prediction is saved in `results/predictions/`. This recomputes Table 5.8 from them:

```bash
pip install scikit-learn
python3 - <<'EOF'
import csv, json
from sklearn.metrics import f1_score
EMO = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
rows = list(csv.DictReader(open("results/predictions/final_roberta_test.csv")))
thr = json.load(open("results/thresholds/final_roberta.json"))["thresholds"]
f1s = []
for e in EMO:
    true = [float(r[f"label_{e}"]) > 0 for r in rows]        # present = intensity > 0
    pred = [float(r[f"logit_{e}"]) > thr[e] for r in rows]    # cut-off chosen on dev
    f1s.append(f1_score(true, pred)); print(f"{e:9s} {f1s[-1]:.3f}")
print(f"macro-F1  {sum(f1s)/6:.4f}")                         # expected 0.4619
EOF
```

---

## 1. Environment

Tested with **Python 3.13.7, CUDA 12.8, PyTorch 2.11.0 and transformers 5.10.2 on an NVIDIA RTX A6000**, on Linux.

```bash
git clone https://github.com/elangoa-tcd/Multimodal_Emotion_Recognition.git
cd Multimodal_Emotion_Recognition
python3 -m venv .venv && source .venv/bin/activate

# 1) PyTorch first, from the CUDA index (pins the CUDA build); version as in requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# 2) everything else, pinned
pip install -r requirements.txt
# 3) the CMU Multimodal SDK, bundled in this repo
pip install -e CMU-MultimodalSDK-main --no-build-isolation
```

Check:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
python3 -c "from mmsdk import mmdatasdk; print('sdk ok')"
```

`requirements.txt` lists the direct dependencies.
environment the reported results were produced in (`pip freeze`), for debugging differences.

---

## 2. Data

### 2.1 Download CMU-MOSEI

This project uses the computational sequences (`.csd`) of
[CMU-MOSEI](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) (Bagher Zadeh et al., 2018).
The CMU download servers are often unreliable, and community mirrors (e.g. on Kaggle) host the
same files. Put them in `data/MOSEI/` under **exactly** these names:

```
data/MOSEI/CMU_MOSEI_TimestampedWords.csd
data/MOSEI/CMU_MOSEI_Labels.csd
data/MOSEI/CMU_MOSEI_COVAREP.csd          # audio
data/MOSEI/CMU_MOSEI_VisualFacet42.csd    # video (used for all reported results)
```

CMU-MOSEI is distributed by its authors under their own terms; this repository does not
redistribute it.

### 2.2 Build the dataset variant

The original preprocessing (`create_dataset.py`) z-normalises every clip against **its own**
statistics. That forces each clip's audio and video features to mean 0 and standard deviation 1,
which deletes the absolute level of the signal (measured at about 70% of the extractable
non-verbal signal). `build_variant.py` rebuilds the dataset with normalisation against
**train-split** statistics instead. All reported results use this variant:

```bash
python3 build_variant.py --out data/MOSEI_C_both --norm corpus --visual_field CMU_MOSEI_VisualFacet42
touch data/MOSEI_C_both/glove_dummy.txt     # placeholder; the BERT/RoBERTa pipeline never reads it
```

`build_variant.py` reads `data/MOSEI/` (change with `--src_dir`), never downloads anything,
refuses to overwrite existing files (`--force` to override), and counts every dropped clip by
reason. **Read the `FEATURE CHECK` block it prints at the end:** with `--norm corpus`,
`avg|per-clip mean|` must be clearly above zero. If it reads ~0.000001, the normalisation change
did not take effect.

| Flag | Values | Meaning |
|---|---|---|
| `--norm` | `corpus` (used) / `clip` (original) / `none` | normalisation of audio and video features |
| `--visual_field` | `CMU_MOSEI_VisualFacet42` (used) / `CMU_MOSEI_VisualOpenFace2` | 35-d facial action units vs. 713-d OpenFace features (no better, much larger) |
| `--common_with DIR` | another variant | keep only clips present in both, for like-for-like comparisons |

Memory: about 13 GB RAM with FACET and about 30 GB with OpenFace2.

Resulting splits (standard CMU-MOSEI folds): **16,315 train / 1,871 dev / 4,654 test** clips.

---

## 3. Train

```bash
python3 train.py --data_dir data/MOSEI_C_both --name final_roberta
```

The defaults are the final configuration (RoBERTa-large, pos_weight, mean pooling, seed 55654), so
this command trains the model behind Tables 5.7–5.11. Training stops early (patience 15, at most
5 learning-rate halvings); the reported runs stopped after 85–138 epochs. On an **NVIDIA RTX A6000** this took about
**5.5 h** for RoBERTa-large and **2.5–4 h** for the BERT runs.

For long runs, detach from the terminal so the job survives a disconnected SSH session:

```bash
mkdir -p logs
nohup python3 -u train.py --data_dir data/MOSEI_C_both --name final_roberta \
  > logs/final_roberta.log 2>&1 &
```

Each run writes:

| File | Contents |
|---|---|
| `checkpoints/devBest/model_<name>.std` | weights of the best epoch on dev (macro-F1) |
| `checkpoints/devBest/run_<name>.json` | exact command, all settings, seed, git commit, library and GPU versions |
| `checkpoints/log_<name>.txt` | per-epoch dev/test metrics |

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--name` | timestamp | run name; used in every output file name, **use a unique one per run** |
| `--data_dir` | `data/MOSEI_C_both` | dataset variant built in §2.2 |
| `--text_encoder` | `roberta-large` | or `bert-base-uncased`. The top ~25% of encoder layers are fine-tuned |
| `--pooling` | `mean` | `attention` = learned attention pooling over tokens (Table 5.6) |
| `--pos_weight` | `full` | class weighting in the loss: `none` (Table 5.3), `full`, `sqrt` (Table 5.5) |
| `--seed` | `55654` | seed for all random number generators (value used for every reported run) |
| `--patience` / `--trials` | `15` / `5` | epochs without dev improvement before halving the LR / number of halvings before stopping |
| `--ablate` | `none` | zero a modality during training: `text`, `audio`, `video`, `only_text` |
| `--modality_dropout` | `0.0` | probability of dropping the text feature per sample during training (not used in reported runs) |
| `--learning_rate`, `--dropout`, `--n_epoch` | `1e-4`, `0.5`, `300` | standard settings |
| `--checkpoint_dir` | `checkpoints/` | where models and logs are written |

---

## 4. Evaluate a trained model

Trained models are not distributed with this repository. Train the model you want to evaluate
with §3 first; it is saved as `checkpoints/devBest/model_<name>.std`.

### 4.1 Checkpoints and the flags they need

| Checkpoint (from §3) | Dissertation | Train with | Evaluate with |
|---|---|---|---|
| `model_final_baseline.std` | Table 5.3 | `--text_encoder bert-base-uncased --pos_weight none --name final_baseline` | `--text_encoder bert-base-uncased` |
| `model_final_posweight.std` | Table 5.4 | `--text_encoder bert-base-uncased --name final_posweight` | `--text_encoder bert-base-uncased` |
| `model_final_sqrt.std` | Table 5.5 | `--text_encoder bert-base-uncased --pos_weight sqrt --name final_sqrt` | `--text_encoder bert-base-uncased` |
| `model_final_attnpool.std` | Table 5.6 | `--text_encoder bert-base-uncased --pooling attention --name final_attnpool` | `--text_encoder bert-base-uncased --pooling attention` |
| `model_final_roberta.std` | Tables 5.7–5.11 | `--name final_roberta` | none (defaults) |

All training commands also take `--data_dir data/MOSEI_C_both`. A checkpoint must be evaluated
with the same `--text_encoder` and `--pooling` it was trained with (they're also recorded in
`checkpoints/devBest/run_<name>.json`). A mismatch fails with a `load_state_dict` error rather
than giving wrong numbers. The `--pos_weight` setting only affects training, so it isn't needed
for evaluation.

### 4.2 Threshold tuning and test metrics

```bash
python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_final_roberta.std --data_dir data/MOSEI_C_both
```

This prints the test metrics at the default cut-off (logit > 0), picks one cut-off per emotion
**on the dev set only**, and prints the test metrics at those cut-offs. Expected last line:
`macro-F1: 0.4533 (default) -> 0.4619 (tuned)`. It writes to `results/`:

| File | Contents |
|---|---|
| `predictions/<run>_{dev,test}.csv` | per clip: `segment_id`, six logits, six label intensities |
| `metrics/<run>_test_default.csv` | per-emotion support, accuracy, precision, recall, F1, weighted accuracy, weighted F1 + MACRO row |
| `metrics/<run>_test_tuned.csv` | the same at the tuned cut-offs |
| `thresholds/<run>.json` | the six cut-offs and both macro-F1 values |

### 4.3 Modality ablation (Table 5.10)

Zeroes one modality at test time on the trained model (no retraining):

```bash
for M in none text audio video only_text; do
  python3 tune_thresholds.py --checkpoint checkpoints/devBest/model_final_roberta.std \
    --data_dir data/MOSEI_C_both --ablate $M
done
```

Outputs are tagged `final_roberta__ablate-<modality>`.

### 4.4 Ceiling analysis (Table 5.11)

```bash
python3 probe_bert_ceiling.py --checkpoint checkpoints/devBest/model_final_roberta.std --data_dir data/MOSEI_C_both
```

This extracts the model's own pooled text embedding and fits logistic-regression probes on text
only, audio+video only, and all three. The text-only → all-three gap is the most that audio and
video can add on top of this text. The result goes to `results/metrics/ceiling_final_roberta.json`.

---

## Paper ↔ repository map

| Dissertation | Data in this repository | Produced by |
|---|---|---|
| Table 3.1 | – (reproduced from Bagher Zadeh et al., 2018) | – |
| Table 5.2 (sentiment replication) | `test_results_final/Sentiment Results.log` | an earlier version of this code; see [Limitations](#limitations) |
| Table 5.3 | `results/metrics/final_baseline_test_default.csv` | `train.py --text_encoder bert-base-uncased --pos_weight none --name final_baseline` |
| Table 5.4 | `results/metrics/final_posweight_test_default.csv` | `train.py --text_encoder bert-base-uncased --name final_posweight` |
| Table 5.5 | `results/metrics/final_sqrt_test_default.csv` | `train.py --text_encoder bert-base-uncased --pos_weight sqrt --name final_sqrt` |
| Table 5.6 | `results/metrics/final_attnpool_test_default.csv` | `train.py --text_encoder bert-base-uncased --pooling attention --name final_attnpool` |
| Table 5.7 | `results/metrics/final_roberta_test_default.csv` | `train.py --name final_roberta` |
| Table 5.8 | `results/metrics/final_roberta_test_tuned.csv`, `results/thresholds/final_roberta.json` | `tune_thresholds.py` (§4.2) |
| Table 5.9 (this work's row) | `weighted_acc` / `weighted_f1` columns of `results/metrics/final_roberta_test_tuned.csv` | `tune_thresholds.py` (§4.2); Akhtar et al. (2019) rows quoted from their paper |
| Table 5.10, Fig 5.4 | `results/metrics/final_roberta__ablate-*_test_tuned.csv` | §4.3 |
| Table 5.11, Fig 5.5 | `results/metrics/ceiling_final_roberta.json` + the `only_text` ablation | §4.4 |
| Training curves, final test tables | `results/training/<run>_{history,final_test}.csv` | `python3 scripts/parse_training_logs.py test_results_final/*.log` |
| Raw training and evaluation logs | `test_results_final/*.log` | stdout of the runs above |

All `train.py` commands also take `--data_dir data/MOSEI_C_both`. Table 5.1 is a description of the
experiments and has no data behind it.

---

## Reproducibility

- **Seed:** every reported run used seed 55654 (`--seed`), with cuDNN in deterministic mode.
  Re-running on the **same** GPU, driver and library versions gives near-identical numbers.
  Different hardware or library versions can shift results slightly, so a retrained model may not
  match the reported numbers to the last decimal. The predictions of the reported models are
  published in `results/predictions/`, so the exact reported numbers can always be recomputed
  (see [Check the numbers without a GPU](#check-the-numbers-without-a-gpu)).
- **No test-set tuning:** model selection (best epoch) and threshold tuning use the dev set only.
  The test set is only used for reporting.
- **Each run records itself:** `run_<name>.json` stores the exact command, all settings, the git
  commit, and whether the code had uncommitted changes.
- **Single seed:** each configuration was trained once. Differences smaller than about 0.01
  macro-F1 between configurations (e.g. Table 5.4 vs 5.6) should not be over-interpreted.

## Limitations

- **Sentiment replication (Table 5.2)** was run with an earlier version of this code. The current
  code supports the emotion task only (`--task emotion`); the log is kept for reference.
- Results are reported on a single seed per configuration (see above).
- Trained checkpoints are not distributed. Re-running the evaluations in §4 requires training the
  model first (3–6 h per run on one GPU). The published predictions cover the reported numbers
  without retraining.

---

## Repository layout

```
├── train.py                  # train one configuration (§3)
├── tune_thresholds.py        # evaluate, tune per-emotion cut-offs on dev, test-time ablations (§4.2–4.3)
├── probe_bert_ceiling.py     # ceiling analysis with linear probes (§4.4)
├── build_variant.py          # build the corpus-normalised dataset variant (§2.2)
├── config.py                 # all command-line flags and defaults
├── network.py                # the TMBL model (AOTN class), adapted for 6 emotion outputs
├── jack_transformer_share.py # TMBL's shared transformer encoder
├── solver.py                 # training loop, losses, metrics
├── data_loader.py            # batching and tokenisation
├── create_dataset.py         # original TMBL dataset loading
├── utils/                    # losses (from TMBL), run recording, results writing
├── scripts/
│   └── parse_training_logs.py
├── results/                  # metrics, predictions, thresholds, training curves
├── test_results_final/       # raw logs of the reported runs
└── CMU-MultimodalSDK-main/   # bundled CMU Multimodal SDK (MIT licence)
```

---

## Acknowledgements and attribution

- The model and much of the training code come from **TMBL** by Huang et al. (2024),
  [github.com/JackAILab/TMBL](https://github.com/JackAILab/TMBL). `network.py`, `jack_transformer_share.py`,
  `create_dataset.py`, `data_loader.py`, `solver.py` and `utils/functions.py` are adapted from it.
  Model variants from the upstream repository that this work doesn't use were removed; see the
  upstream repository for them.
- `CMU-MultimodalSDK-main/` is the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK),
  included unchanged under its MIT licence.
- Supervision: Dr Yvette Graham, Trinity College Dublin.

## Citation

If you use this code, please cite the dissertation and the original TMBL paper:

```bibtex
@mastersthesis{elango2026emotion,
  author = {Elango, Akhil},
  title  = {Transformer based Multimodal Binding Learning Model for Emotion Recognition},
  school = {Trinity College Dublin},
  year   = {2026},
  type   = {{MSc} dissertation}
}

@article{huang2024tmbl,
  author  = {Huang, Jiehui and Zhou, Jun and Tang, Zhenchao and Lin, Jiaying and Chen, Calvin Yu-Chian},
  title   = {{TMBL}: Transformer-based multimodal binding learning model for multimodal sentiment analysis},
  journal = {Knowledge-Based Systems},
  volume  = {285},
  pages   = {111346},
  year    = {2024},
  doi     = {10.1016/j.knosys.2023.111346}
}
```

Please also cite CMU-MOSEI (Bagher Zadeh et al., 2018) if you use the dataset.


